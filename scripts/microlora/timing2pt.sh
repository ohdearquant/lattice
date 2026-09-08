#!/bin/bash
# Two-point timing measurement: isolate the per-sample SCORING cost from the per-step cost.
#
# One run cannot separate them. Two runs identical except for --log-every differ ONLY in how many
# scoring passes fire, so the difference in WALL-CLOCK totals divided by the difference in
# sample-forwards is the scoring rate, measured rather than assumed.
#
# The number of scoring passes is COUNTED FROM EACH LOG, never derived from steps/log-every: the
# trainer evaluates at step 0 as well, so a formula undercounts by one and the error rides silently
# into the rate.
#
# WHAT THIS VERSION FIXES, and it would have produced a confidently wrong number. The previous
# version reported `total=${t}s` where t was scraped from the trainer's own `in Xs ===` line. That
# figure is the STEP LOOP ONLY. Measured on a representative run: model load 4.3s, prefix cache
# 312.9s, step loop 1088.8s, total accounted 1406.0s against a wall span of about 2590s. The
# held-out cache build and both baseline scoring passes print NO duration at all, so roughly 1184s,
# 46% of the arm, was invisible, and the self-report covered 42% of the run.
#
# The failure would not have looked like a failure: both points would have been understated by their
# own prologues while the DIFFERENCE still isolated scoring passes correctly inside the timed
# region, so the derived rate would have passed its own sanity check. Every total here is therefore
# an external wall clock around the whole process, with the trainer's figure recorded beside it.
set -u

BIN=${BIN:-/Volumes/LaCie/lattice-target/release/train_grad_full}
DATA=${DATA:-/Volumes/LaCie/lattice-microlora/frontmatter-512}
OUT=${OUT:-/Volumes/LaCie/lattice-microlora/timing2pt}
TIMER="$(cd "$(dirname "$0")" && pwd)/time_run.sh"

[ -x "$BIN" ]   || { echo "REFUSED: trainer not executable: $BIN" >&2; exit 2; }
[ -x "$TIMER" ] || { echo "REFUSED: $TIMER missing; totals would be loop-only again" >&2; exit 2; }
[ -d "$DATA" ]  || { echo "REFUSED: data dir absent: $DATA" >&2; exit 2; }
mkdir -p "$OUT"

COMMON="--data-dir $DATA --seq-len 512 --steps 6 --max-train 8 --max-valid 4 --first-layer 19 --rank 8 --alpha 16 --lr 1e-3"

# A timing run states the machine it ran on. This is a disclosure, not a certification: a quiet
# reading here does not prove the window stayed quiet, which is why both ends are sampled.
echo "=== host state BEFORE ==="
uptime; /bin/ps -Ao pcpu=,comm= | sort -rn | head -3

for LE in 1 6; do
  echo "=== log-every $LE ==="
  "$TIMER" --label "le${LE}" --out "$OUT" -- \
    $BIN $COMMON --log-every $LE
  echo "rc=$?"
done

echo "=== host state AFTER ==="
uptime; /bin/ps -Ao pcpu=,comm= | sort -rn | head -3
echo "=== DONE_TIMING $(date -u +%FT%TZ) ==="

echo
printf 'log_every\tscoring_passes\twall_s\tself_reported_s\tprologue_s\n'
for LE in 1 6; do
  LOG="$OUT/le${LE}.log"
  TSV="$OUT/le${LE}.timing.tsv"
  n=$(grep -c "held-out NLL" "$LOG" 2>/dev/null || echo NA)
  if [ -f "$TSV" ]; then
    # Columns: label wall_s self_reported_s prologue_s rc start end
    awk -v le="$LE" -v n="$n" 'NR==2{printf "%s\t%s\t%s\t%s\t%s\n", le, n, $2, $3, $4}' "$TSV"
  else
    printf '%s\t%s\tNA\tNA\tNA\n' "$LE" "$n"
  fi
done

echo
echo "The scoring rate is (wall_s difference) / (scoring_passes difference), using WALL seconds."
echo "self_reported_s is the trainer's step-loop span and is shown for contrast only; it is not a"
echo "total and must never be used as one. A SUSPECT: prefix on it means the wall clock came out"
echo "shorter than the span it contains, which voids that pair."
