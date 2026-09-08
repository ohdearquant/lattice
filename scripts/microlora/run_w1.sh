#!/bin/bash
# run_w1.sh — the pre-registered W1 training run for one micro-LoRA task.
#
# Three seeds, plus a shuffled-label control at every seed. The control is not optional and it is
# not run afterwards: a task whose completions can be permuted across rows and still show the same
# NLL drop is being learned as format, not as content, and the whole W1 claim is void. Running both
# arms in one script is what makes "we also ran the control" checkable rather than remembered.
#
# Correctness numbers only. Nothing here is a timing measurement: the wall clocks in the logs
# describe whatever else the host was doing, and are recorded for scheduling, never for a claim.
#
# Usage:
#   run_w1.sh --task NAME --data DIR --out DIR --bin PATH [--steps 400] [--seeds "1 2 3"]
#             [--first-layer 19] [--rank 8] [--alpha 16] [--lr 1e-3] [--seq-len 512]
#
# Every run writes:  <out>/<task>_seed<K>.log        real labels
#                    <out>/<task>_seed<K>_shuf.log   permuted labels (the control)
#                    <out>/<task>_seed<K>.safetensors
#                    <out>/SUMMARY.tsv               one row per arm, parsed from the logs
set -u

TASK="" DATA="" OUT="" BIN=""
STEPS=400 SEEDS="1 2 3" FIRST_LAYER=19 RANK=8 ALPHA=16 LR=1e-3 SEQ_LEN=512
while [ $# -gt 0 ]; do
  case "$1" in
    --task) TASK=$2; shift 2;;
    --data) DATA=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --bin) BIN=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --seeds) SEEDS=$2; shift 2;;
    --first-layer) FIRST_LAYER=$2; shift 2;;
    --rank) RANK=$2; shift 2;;
    --alpha) ALPHA=$2; shift 2;;
    --lr) LR=$2; shift 2;;
    --seq-len) SEQ_LEN=$2; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
for v in TASK DATA OUT BIN; do
  eval "[ -n \"\$$v\" ]" || { echo "missing --$(echo $v | tr 'A-Z_' 'a-z-')" >&2; exit 2; }
done
[ -x "$BIN" ] || { echo "not executable: $BIN" >&2; exit 2; }
[ -f "$DATA/train.jsonl" ] && [ -f "$DATA/valid.jsonl" ] || { echo "need train.jsonl and valid.jsonl under $DATA" >&2; exit 2; }

# REFUSE A TRAINER THAT CANNOT VARY ITS SEED. Without the flag every "seed" runs the same
# initialisation, three identical numbers read as agreement, and the seed column in the summary is
# a lie told by the script rather than by the model.
"$BIN" --help 2>&1 | grep -q -- "--seed" || {
  echo "REFUSED: $BIN has no --seed flag, so the three seeds would be one run repeated." >&2
  echo "         Build a trainer that carries the flag, or state a different variation source." >&2
  exit 3; }

mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.tsv"
printf 'task\tarm\tseed\tsteps\ttrain_first\ttrain_last\theldout_first\theldout_last\trc\tlog\n' > "$SUMMARY"

# The control set: completions permuted across rows with a fixed permutation seed, prompts untouched.
SHUF="$OUT/shuffled"
mkdir -p "$SHUF"
python3 - "$DATA/train.jsonl" "$SHUF/train.jsonl" "$DATA/valid.jsonl" "$SHUF/valid.jsonl" <<'PY' || exit 4
import json, random, sys
rng = random.Random(0xC0FFEE)
for src, dst in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    comps = [r["completion"] for r in rows]
    # A derangement is not required, but a permutation that leaves rows in place would weaken the
    # control, so re-draw until at most a tenth of the rows keep their own completion.
    for _ in range(64):
        rng.shuffle(comps)
        if sum(a == b["completion"] for a, b in zip(comps, rows)) <= max(1, len(rows) // 10):
            break
    with open(dst, "w", encoding="utf-8") as fh:
        for r, c in zip(rows, comps):
            fh.write(json.dumps({"prompt": r["prompt"], "completion": c}, ensure_ascii=False) + "\n")
    print(f"shuffled {len(rows)} rows -> {dst}", file=sys.stderr)
PY

parse_row () {  # task arm seed log rc
  python3 - "$1" "$2" "$3" "$4" "$5" "$STEPS" <<'PY'
import re, sys
task, arm, seed, log, rc, steps = sys.argv[1:7]
tr, ho = [], []
for line in open(log, encoding="utf-8", errors="replace"):
    m = re.search(r"train NLL:\s*([0-9.]+)\s+held-out NLL:\s*([0-9.]+)", line)
    if m:
        tr.append(float(m.group(1))); ho.append(float(m.group(2)))
f = lambda xs, i: (f"{xs[i]:.4f}" if xs else "NA")
print("\t".join([task, arm, seed, steps, f(tr,0), f(tr,-1), f(ho,0), f(ho,-1), rc, log]))
PY
}

for SEED in $SEEDS; do
  for ARM in real shuffled; do
    if [ "$ARM" = real ]; then D=$DATA; SUF=""; else D=$SHUF; SUF="_shuf"; fi
    LOG="$OUT/${TASK}_seed${SEED}${SUF}.log"
    echo "=== $TASK arm=$ARM seed=$SEED -> $LOG"
    SAVE=""
    [ "$ARM" = real ] && SAVE="--save $OUT/${TASK}_seed${SEED}.safetensors"
    "$BIN" --data-dir "$D" --seq-len "$SEQ_LEN" --steps "$STEPS" \
           --max-train 100000 --max-valid 200 --first-layer "$FIRST_LAYER" \
           --rank "$RANK" --alpha "$ALPHA" --lr "$LR" --seed "$SEED" \
           --log-every 10 $SAVE > "$LOG" 2>&1
    RC=$?
    parse_row "$TASK" "$ARM" "$SEED" "$LOG" "$RC" >> "$SUMMARY"
    [ $RC -eq 0 ] || echo "  arm=$ARM seed=$SEED EXITED $RC (row kept; a failed arm is data, not a gap)" >&2
  done
done

echo
echo "== $SUMMARY"
cat "$SUMMARY"
