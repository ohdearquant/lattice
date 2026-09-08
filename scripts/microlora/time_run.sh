#!/bin/bash
# time_run.sh — put an EXTERNAL wall clock around a whole command, and report the program's own
# self-reported duration beside it rather than instead of it.
#
# WHY THIS EXISTS. The exact-gradient trainer prints `... in {secs}s ===` at the end of every run.
# That figure is its step loop only: it starts its clock after the model load, after the frozen-prefix
# cache build, after the held-out cache build, and after the two baseline scoring passes that produce
# the step-0 row. Measured on a representative run, the self-report read 1088.8s while the arm
# spanned about 2530s, a factor of ~2.3. A cost model built on the self-report is therefore a FLOOR,
# and it understates by a term that grows with the pool and the held-out size, which is exactly the
# term a scheduling decision depends on.
#
# The rule this enforces: a timing number is an external wall clock around the whole process. The
# program's own figure is still recorded, labelled as the loop-only span, because the DIFFERENCE
# between the two is the prologue cost and that is a quantity worth having.
#
# Usage:
#   time_run.sh --label NAME --out DIR -- <command> [args...]
#
# Writes:  <out>/<label>.log          the command's stdout+stderr
#          <out>/<label>.timing.tsv   label, wall_s, self_reported_s, prologue_s, rc, start, end
set -u

LABEL="" OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --) shift; break;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ -n "$LABEL" ] || { echo "missing --label" >&2; exit 2; }
[ -n "$OUT" ] || { echo "missing --out" >&2; exit 2; }
[ $# -gt 0 ] || { echo "no command given after --" >&2; exit 2; }

mkdir -p "$OUT"
LOG="$OUT/$LABEL.log"
TSV="$OUT/$LABEL.timing.tsv"

START_EPOCH=$(date +%s)
START_ISO=$(date -Iseconds)
"$@" > "$LOG" 2>&1
RC=$?
END_EPOCH=$(date +%s)
END_ISO=$(date -Iseconds)
WALL=$((END_EPOCH - START_EPOCH))

# The program's own figure, if it printed one. Absent is recorded as NA, never as zero: a missing
# self-report is not a run that took no time, and a downstream sum must not silently treat it as one.
SELF=$(sed -n 's/.*[[:space:]]in \([0-9][0-9]*\.[0-9]*\)s ===.*/\1/p' "$LOG" | tail -1)
[ -n "$SELF" ] || SELF=NA

if [ "$SELF" = NA ]; then
  PROLOGUE=NA
else
  PROLOGUE=$(awk -v w="$WALL" -v s="$SELF" 'BEGIN{printf "%.1f", w - s}')
  # An external wall clock CONTAINS the span the program timed itself, so the prologue can
  # never be negative. If it is, the self-report was mis-parsed (the wrong number matched) or
  # the log carries a figure from a different run, and either way the pair must not be used.
  case "$PROLOGUE" in
    -*) echo "WARNING: prologue ${PROLOGUE}s is negative for $LABEL. Wall clock cannot be shorter" >&2
        echo "         than the span it contains, so the self-reported figure is not this run's." >&2
        SELF="SUSPECT:$SELF"; PROLOGUE=NA;;
  esac
fi

printf 'label\twall_s\tself_reported_s\tprologue_s\trc\tstart\tend\n' > "$TSV"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$WALL" "$SELF" "$PROLOGUE" "$RC" "$START_ISO" "$END_ISO" >> "$TSV"

# "NA" must not be printed as "NAs": a unit suffix on a non-value reads like a measurement.
case "$PROLOGUE" in NA) PSHOW=NA;; *) PSHOW="${PROLOGUE}s";; esac
case "$SELF" in NA|SUSPECT:*) SSHOW="$SELF";; *) SSHOW="${SELF}s";; esac
echo "$LABEL: wall ${WALL}s | self-reported $SSHOW (loop only) | prologue $PSHOW | rc $RC"
exit $RC
