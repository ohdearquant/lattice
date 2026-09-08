#!/bin/bash
# run_w1_staged.sh - the resized W1 run: 3 seeds x {real labels, permuted labels}.
#
# WHY THIS EXISTS BESIDE run_w1.sh. The exact-gradient CPU trainer takes the FIRST N rows of
# train.jsonl, so three seeds against one data directory vary the adapter initialisation and
# nothing else. Three runs over one fixed subset cannot bound subset variance, and subset variance
# is the larger term when the pool is small. This script therefore gives every seed its OWN
# subset, drawn with that seed, and pairs each permuted arm with the SAME subset as its real arm.
# A seed here varies both the initialisation and the draw; the arms stay paired.
#
# THE CONTROL. Completions are permuted WITHIN the seed's own subset: same prompts, same set of
# completions, wrong pairing. The held-out split is never touched, because the decision quantity is
# held-out NLL on the REAL split and permuting it would answer a different question.
#
# Correctness numbers only. Wall clocks in the logs are scheduling information and no timing claim
# is made from them.
#
# Usage:
#   run_w1_staged.sh --data DIR --out DIR --bin PATH [--pool 32] [--steps 48] [--seeds "1 2 3"]
set -u

DATA="" OUT="" BIN=""
POOL=32 STEPS=48 SEEDS="1 2 3" SEQ_LEN=512 FIRST_LAYER=19 RANK=8 ALPHA=16 LR=1e-3 MAX_VALID=22
# LOG_EVERY defaults to STEPS, which logs only the endpoints and is the cheapest setting. Longer
# runs pass 12 (five points per arm) so a non-monotonic or diverging curve is visible. That is a
# deliberate cost increase and it is chosen before the run, not after seeing the totals.
LOG_EVERY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --bin) BIN=$2; shift 2;;
    --pool) POOL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --seeds) SEEDS=$2; shift 2;;
    --seq-len) SEQ_LEN=$2; shift 2;;
    --first-layer) FIRST_LAYER=$2; shift 2;;
    --rank) RANK=$2; shift 2;;
    --alpha) ALPHA=$2; shift 2;;
    --lr) LR=$2; shift 2;;
    --max-valid) MAX_VALID=$2; shift 2;;
    --log-every) LOG_EVERY=$2; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ -n "$DATA" ] || { echo "missing --data" >&2; exit 2; }
[ -n "$OUT" ]  || { echo "missing --out"  >&2; exit 2; }
[ -n "$BIN" ]  || { echo "missing --bin"  >&2; exit 2; }
[ -x "$BIN" ] || { echo "not executable: $BIN" >&2; exit 2; }
[ -n "$LOG_EVERY" ] || LOG_EVERY=$STEPS
TIMER="$(cd "$(dirname "$0")" && pwd)/time_run.sh"
# Every timing number is an EXTERNAL wall clock around the whole process. The trainer's own
# "in Xs" figure is its step loop only and excludes the model load, the held-out cache build and
# both baseline scoring passes; on a representative run the two disagreed by ~2.3x. Refusing
# here rather than silently falling back keeps a run from producing loop-only numbers again.
[ -x "$TIMER" ] || { echo "REFUSED: $TIMER missing or not executable; timing would be loop-only" >&2; exit 2; }
[ -f "$DATA/train.jsonl" ] && [ -f "$DATA/valid.jsonl" ] || { echo "need train.jsonl and valid.jsonl under $DATA" >&2; exit 2; }

# A trainer that cannot vary its seed makes the seed column a lie told by this script.
"$BIN" --help 2>&1 | grep -q -- "--seed" || {
  echo "REFUSED: $BIN has no --seed flag, so the seeds would be one run repeated." >&2; exit 3; }

mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.tsv"
printf 'task\tarm\tseed\tpool\tsteps\theldout_base\theldout_final\theldout_drop_rel\ttrain_base\ttrain_final\twall_s\tself_reported_s\tstate\tlog\n' > "$SUMMARY"

for SEED in $SEEDS; do
  for ARM in real permuted; do
    D="$OUT/data_seed${SEED}_${ARM}"
    mkdir -p "$D"
    cp "$DATA/valid.jsonl" "$D/valid.jsonl"
    python3 - "$DATA/train.jsonl" "$D/train.jsonl" "$SEED" "$POOL" "$ARM" <<'PY' || exit 4
import json, random, sys
src, dst, seed, pool, arm = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
# The SUBSET is drawn from the seed, so the real and permuted arms of one seed see the same rows.
sub = random.Random(seed).sample(rows, min(pool, len(rows)))
if arm == "permuted":
    comps = [r["completion"] for r in sub]
    # Permute WITHIN the subset: same prompts, same completions, wrong pairing. Re-draw with a
    # stream derived from the seed until at most a tenth of rows keep their own completion.
    rng = random.Random(seed * 1000003 + 7)
    ok = False
    for _ in range(256):
        rng.shuffle(comps)
        if sum(a == b["completion"] for a, b in zip(comps, sub)) <= max(1, len(sub) // 10):
            ok = True
            break
    if not ok:
        sys.exit("could not find a permutation leaving at most a tenth of rows in place")
    sub = [{"prompt": r["prompt"], "completion": c} for r, c in zip(sub, comps)]
with open(dst, "w", encoding="utf-8") as fh:
    for r in sub:
        fh.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}, ensure_ascii=False) + "\n")
print(f"seed {seed} arm {arm}: {len(sub)} rows -> {dst}", file=sys.stderr)
PY

    # The paired-subset property is the reason this design is worth the extra files, so it is
    # asserted rather than assumed: the permuted arm must carry its real arm's prompts in order.
    if [ "$ARM" = permuted ]; then
      python3 - "$OUT/data_seed${SEED}_real/train.jsonl" "$D/train.jsonl" "$DATA/valid.jsonl" "$D/valid.jsonl" <<'PY' || exit 5
import json, sys, hashlib
a=[json.loads(l) for l in open(sys.argv[1])]; b=[json.loads(l) for l in open(sys.argv[2])]
assert len(a)==len(b), f"row counts differ: {len(a)} vs {len(b)}"
assert [x["prompt"] for x in a]==[y["prompt"] for y in b], "arms do not share prompts in order"
same=sum(1 for x,y in zip(a,b) if x["completion"]==y["completion"])
assert same <= max(1,len(a)//10), f"{same}/{len(a)} rows kept their own completion"
h=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
assert h(sys.argv[3])==h(sys.argv[4]), "held-out differs from source; the control is invalid"
print(f"paired-control check ok: {len(a)} rows, {same} self-kept, held-out untouched")
PY
    fi

    LABEL="seed${SEED}_${ARM}"
    LOG="$OUT/$LABEL.log"
    SAVE=""
    [ "$ARM" = real ] && SAVE="--save $OUT/seed${SEED}_real.safetensors"

    # PER-ARM CHECKPOINT. The host this runs on drops its network in short windows, and a
    # multi-hour run restarted from scratch after a flap never finishes. An arm counts as done only
    # if its log carries the trainer's own terminal line, so a truncated or killed arm is re-run
    # rather than silently accepted. Completion is read from the ARTIFACT, never from a marker file
    # this script writes about itself.
    if [ -f "$LOG" ] && grep -q "=== done" "$LOG"; then
      echo "=== seed=$SEED arm=$ARM ALREADY COMPLETE, skipping ($LOG)"
      continue
    fi
    [ -f "$LOG" ] && echo "  re-running seed=$SEED arm=$ARM: log exists but has no terminal line"

    echo "=== seed=$SEED arm=$ARM -> $LOG"
    "$TIMER" --label "$LABEL" --out "$OUT" -- \
      "$BIN" --data-dir "$D" --seq-len "$SEQ_LEN" --steps "$STEPS" \
           --max-train "$POOL" --max-valid "$MAX_VALID" --first-layer "$FIRST_LAYER" \
           --rank "$RANK" --alpha "$ALPHA" --lr "$LR" --seed "$SEED" \
           --log-every "$LOG_EVERY" $SAVE
    RC=$?
    [ $RC -eq 0 ] || echo "  seed=$SEED arm=$ARM EXITED $RC (row kept; a failed arm is data, not a gap)" >&2
  done
done

# SUMMARY IS BUILT AFTER THE LOOP, over every EXPECTED arm, from the artifacts on disk. It used to
# be appended inside the loop, which is wrong the moment arms can be skipped: a resumed run
# truncates this file at startup and would then emit rows only for the arms it happened to re-run,
# so a COMPLETE experiment would report as a partial one. Building it from the artifacts also lets
# an arm that never ran appear as an explicit MISSING row, because a gap in a table reads as 'not
# applicable' while a MISSING row reads as what it is.
for SEED in $SEEDS; do
  for ARM in real permuted; do
    LOG="$OUT/seed${SEED}_${ARM}.log"
    TSV="$OUT/seed${SEED}_${ARM}.timing.tsv"
    python3 - "$SEED" "$ARM" "$POOL" "$STEPS" "$LOG" "$TSV" >> "$SUMMARY" <<'PYROW'
import re, sys, os
seed, arm, pool, steps, log, tsv = sys.argv[1:7]
cols = ['frontmatter-repair', arm, seed, pool, steps]
if not os.path.exists(log):
    print('\t'.join(cols + ['MISSING'] * 5 + ['NA', 'NA', 'MISSING', log])); raise SystemExit(0)
tr, ho, done = [], [], False
for line in open(log, encoding='utf-8', errors='replace'):
    m = re.search(r'train NLL:\s*([0-9.]+)\s+held-out NLL:\s*([0-9.]+)', line)
    if m:
        tr.append(float(m.group(1))); ho.append(float(m.group(2)))
    if '=== done' in line:
        done = True
# Wall clock comes from the external timer file, never from the trainer's own figure.
wall = self_s = 'NA'
if os.path.exists(tsv):
    rows = [l.rstrip('\n').split('\t') for l in open(tsv) if l.strip()]
    if len(rows) > 1:
        wall, self_s = rows[1][1], rows[1][2]
if ho and done:
    cells = ['%.4f' % ho[0], '%.4f' % ho[-1], '%.4f' % ((ho[0]-ho[-1])/ho[0]), '%.4f' % tr[0], '%.4f' % tr[-1]]
elif ho:
    # Unfinished. The observed points are real, but a DROP computed across a run that was cut
    # short is a fabricated result, and 0.0000 in that column reads exactly like a measured
    # null. Report the endpoints and refuse the derived quantity.
    cells = ['%.4f' % ho[0], '%.4f' % ho[-1], 'NA', '%.4f' % tr[0], '%.4f' % tr[-1]]
else:
    cells = ['NA'] * 5
print('\t'.join(cols + cells + [wall, self_s, 'complete' if done else 'INCOMPLETE', log]))
PYROW
  done
done

echo
echo "== $SUMMARY"
cat "$SUMMARY"
BAD=$(grep -c "INCOMPLETE\|MISSING" "$SUMMARY" || true)
if [ "${BAD:-0}" -gt 0 ]; then
  echo >&2
  echo "$BAD arm(s) INCOMPLETE or MISSING. Re-run this exact command to resume: complete arms are" >&2
  echo "skipped by their own terminal line, so only the unfinished work repeats." >&2
  exit 6
fi
