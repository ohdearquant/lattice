#!/bin/bash
# run_w1.sh — the pre-registered W1 training run for one micro-LoRA task.
#
# SUPERSEDED. W1's absolute-drop metric was KILLED by this script's own control: permuted labels
# reached a 35.0% held-out drop against a 15% bar, so the metric cannot separate learning the
# repair from learning the patch format. The paired-metric runner run_w1_staged.sh supersedes it. This file is kept because it is what produced that result and it must stay
# runnable to reproduce it, not because it is the current path.
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
MAX_TRAIN=100000 MAX_VALID=200 LOG_EVERY=10
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
    --max-train) MAX_TRAIN=$2; shift 2;;
    --max-valid) MAX_VALID=$2; shift 2;;
    --log-every) LOG_EVERY=$2; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
for v in TASK DATA OUT BIN; do
  eval "[ -n \"\$$v\" ]" || { echo "missing --$(echo $v | tr 'A-Z_' 'a-z-')" >&2; exit 2; }
done
[ -x "$BIN" ] || { echo "not executable: $BIN" >&2; exit 2; }
TIMER="$(cd "$(dirname "$0")" && pwd)/time_run.sh"
# The trainer's own "in Xs" figure is its step loop only and omits the model load, the held-out
# cache build and both baseline scoring passes; measured on a representative run it covered 42%
# of the wall clock. Any duration this script leaves behind must therefore come from an external clock.
[ -x "$TIMER" ] || { echo "REFUSED: $TIMER missing; wall clocks would be loop-only" >&2; exit 2; }
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

# The control set: TRAIN completions permuted across rows with a fixed permutation seed, prompts
# untouched. THE HELD-OUT SET IS NOT PERMUTED, and that is the whole point of the control. The
# decision quantity is held-out NLL on the REAL held-out split; permuting that split too would
# measure how well a model predicts random targets, which is a different question and always
# answers "badly". Training on permuted labels while scoring on real ones is what separates
# learning the TASK from learning the output FORMAT: a model that only picked up the patch syntax
# improves on the real held-out set no matter what labels it trained on, and this arm is what makes
# that visible instead of letting it read as a pass.
SHUF="$OUT/shuffled"
mkdir -p "$SHUF"
cp "$DATA/valid.jsonl" "$SHUF/valid.jsonl"
python3 - "$DATA/train.jsonl" "$SHUF/train.jsonl" <<'PY' || exit 4
import json, random, sys
rng = random.Random(0xC0FFEE)
for src, dst in ((sys.argv[1], sys.argv[2]),):
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
    "$TIMER" --label "${TASK}_seed${SEED}${SUF}" --out "$OUT" -- \
      "$BIN" --data-dir "$D" --seq-len "$SEQ_LEN" --steps "$STEPS" \
           --max-train "$MAX_TRAIN" --max-valid "$MAX_VALID" --first-layer "$FIRST_LAYER" \
           --rank "$RANK" --alpha "$ALPHA" --lr "$LR" --seed "$SEED" \
           --log-every "$LOG_EVERY" $SAVE
    RC=$?
    parse_row "$TASK" "$ARM" "$SEED" "$LOG" "$RC" >> "$SUMMARY"
    [ $RC -eq 0 ] || echo "  arm=$ARM seed=$SEED EXITED $RC (row kept; a failed arm is data, not a gap)" >&2
  done
done

echo
echo "== $SUMMARY"
cat "$SUMMARY"
