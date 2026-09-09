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
# EXCLUSIVITY, and why it is enforced here rather than trusted to the caller. A wall clock that
# shares its machine with a second copy of the same job is not a measurement of that job. On
# 2026-09-08 a run of this trainer took 3994s; the identical run, same binary and same inputs, on
# an idle machine took 615s. The driver had checked for competing processes and PRINTED the result:
#
#     === nothing of ours running: 1 ===
#
# The label asserts the answer while the value beside it contradicts it, and the run proceeded. A
# printed count is not a guard, it is a number waiting to be read past, so --exclusive REFUSES
# instead of reporting. The competing process was left over from an earlier run that had been
# discarded as invalid: discarding a MEASUREMENT says nothing about the PROCESS, which keeps
# running and keeps consuming exactly the resource the next measurement needs.
#
# The post-run count is recorded too, because contention can arrive after launch and a check that
# only runs at t=0 certifies a window it did not observe.
#
# Usage:
#   time_run.sh --label NAME --out DIR [--exclusive PATTERN] -- <command> [args...]
#
# Writes:  <out>/<label>.log          the command's stdout+stderr
#          <out>/<label>.timing.tsv   label, wall_s, self_reported_s, prologue_s, rc, start, end,
#                                     competing_at_start, competing_at_end
set -u

LABEL="" OUT="" EXCLUSIVE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --exclusive) EXCLUSIVE=$2; shift 2;;
    --) shift; break;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

# Counts processes whose command line contains PATTERN, excluding this script and its own shell.
#
# The process table is snapshotted into a variable BEFORE any grep runs, which is what keeps the
# search from matching itself: a naive `ps | grep "$PATTERN"` puts PATTERN in the grep's own argv
# and the grep appears in its own results. The bracket trick does not help when the pattern arrives
# as a variable. Prints the count on stdout and the pids on stderr.
competing_count () {
  snap=$(/bin/ps -Ao pid=,command=)
  # ARM: a number-bearing instrument asserts its input was non-empty in the same breath. An
  # unreadable process table returning "0 competitors" is the reassuring failure.
  if [ -z "$snap" ]; then
    echo "REFUSED: the process table read back empty, so the exclusivity check could not run." >&2
    echo "         An instrument that cannot see anything must not report that it saw nothing." >&2
    exit 5
  fi
  hits=$(printf '%s\n' "$snap" | /usr/bin/grep -F -- "$1" \
           | /usr/bin/grep -v "time_run.sh" \
           | /usr/bin/awk -v self="$$" '$1 != self { print }')
  [ -n "$hits" ] && printf '%s\n' "$hits" >&2
  printf '%s\n' "$hits" | /usr/bin/grep -c . 
}
[ -n "$LABEL" ] || { echo "missing --label" >&2; exit 2; }
[ -n "$OUT" ] || { echo "missing --out" >&2; exit 2; }
[ $# -gt 0 ] || { echo "no command given after --" >&2; exit 2; }

mkdir -p "$OUT"
LOG="$OUT/$LABEL.log"
TSV="$OUT/$LABEL.timing.tsv"

COMPETING_START=0
if [ -n "$EXCLUSIVE" ]; then
  COMPETING_START=$(competing_count "$EXCLUSIVE")
  if [ "$COMPETING_START" -ne 0 ]; then
    echo "REFUSED: $COMPETING_START process(es) matching '$EXCLUSIVE' are already running (pids above)." >&2
    echo "         A timing run that shares the machine with another copy of its own workload" >&2
    echo "         measures the contention, not the workload. Stop them and re-run." >&2
    exit 4
  fi
fi

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

# Measured AFTER the run, not assumed from the pre-check: a machine that was quiet at t=0 can
# acquire a competitor at minute 13, and contamination landing on one arm of a comparison biases
# that comparison with a sign rather than averaging out.
COMPETING_END=0
[ -n "$EXCLUSIVE" ] && COMPETING_END=$(competing_count "$EXCLUSIVE" 2>/dev/null)

printf 'label\twall_s\tself_reported_s\tprologue_s\trc\tstart\tend\tcompeting_at_start\tcompeting_at_end\n' > "$TSV"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$WALL" "$SELF" "$PROLOGUE" "$RC" "$START_ISO" "$END_ISO" "$COMPETING_START" "$COMPETING_END" >> "$TSV"
if [ -n "$EXCLUSIVE" ] && [ "$COMPETING_END" -ne 0 ]; then
  echo "WARNING: $COMPETING_END process(es) matching '$EXCLUSIVE' were running when this finished," >&2
  echo "         so the machine was not exclusive for the whole span and this number is SUSPECT." >&2
fi

# "NA" must not be printed as "NAs": a unit suffix on a non-value reads like a measurement.
case "$PROLOGUE" in NA) PSHOW=NA;; *) PSHOW="${PROLOGUE}s";; esac
case "$SELF" in NA|SUSPECT:*) SSHOW="$SELF";; *) SSHOW="${SELF}s";; esac
echo "$LABEL: wall ${WALL}s | self-reported $SSHOW (loop only) | prologue $PSHOW | rc $RC"
exit $RC
