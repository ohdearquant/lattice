#!/bin/bash
# Controls for time_run.sh --exclusive. A guard that fails open is worse than no guard, so the
# refusal arm asserts the COMMAND DID NOT RUN, not merely that the exit code was 4.
set -u
TR="$(cd "$(dirname "$0")" && pwd)/time_run.sh"
D=$(mktemp -d); fails=0; ran=0
MARK="zzexclusivitymarker$$"

note () { ran=$((ran+1)); if [ "$1" = ok ]; then :; else fails=$((fails+1)); echo "  FAIL $2"; fi; }

# --- A. clean machine, pattern matches nothing -> runs normally.
rm -f "$D/a.touched"
"$TR" --label a --out "$D" --exclusive "$MARK" -- /usr/bin/touch "$D/a.touched" >/dev/null 2>&1
rc=$?
[ $rc -eq 0 ] && [ -f "$D/a.touched" ] && note ok || note bad "A: clean run should proceed (rc=$rc, touched=$([ -f $D/a.touched ] && echo yes || echo no))"
grep -q "competing_at_start" "$D/a.timing.tsv" && note ok || note bad "A2: tsv missing the new columns"
awk 'NR==2{exit !($8==0)}' "$D/a.timing.tsv" && note ok || note bad "A3: competing_at_start should be 0, tsv=$(sed -n 2p $D/a.timing.tsv)"

# --- B. a competitor exists -> REFUSE, and the command must NOT run.
# The decoy must carry the property the search looks for, and that is ASSERTED, not assumed.
# `bash -c "sleep 120" MARK` does NOT work: a single simple command makes bash exec sleep
# directly, so the process argv is "sleep 120" and the marker is gone. The first version of this
# test used that form, the guard correctly found nothing, and the test reported the GUARD as
# failing open. A decoy that does not carry the property tests the test, not the subject.
bash -c "sleep 120; :" "$MARK" &
COMP=$!
sleep 1
if [ "$(/bin/ps -o command= -p $COMP | /usr/bin/grep -c "$MARK")" -ne 1 ]; then
  echo "  FAIL B0: the decoy does not carry the marker in its argv, so arms B..B4 would be vacuous"
  echo "     argv: $(/bin/ps -o command= -p $COMP)"
  fails=$((fails+1))
fi
ran=$((ran+1))
rm -f "$D/b.touched"
"$TR" --label b --out "$D" --exclusive "$MARK" -- /usr/bin/touch "$D/b.touched" >/dev/null 2>"$D/b.err"
rc=$?
[ $rc -eq 4 ] && note ok || note bad "B: expected exit 4, got $rc"
[ ! -f "$D/b.touched" ] && note ok || note bad "B2: THE GUARD FAILED OPEN -- the command ran anyway"
grep -q "REFUSED" "$D/b.err" && note ok || note bad "B3: refusal must say so on stderr; got: $(cat $D/b.err)"
[ ! -f "$D/b.timing.tsv" ] && note ok || note bad "B4: a refused run must not leave a timing row"

# --- B5. MUST-MATCH control: prove the counter can SEE that process at all. Without this, arm B
#         would pass identically if the counter were simply broken and returned 0... except it
#         returns 4, so instead prove the negative arm is a real negative: same competitor, a
#         pattern that should NOT match, must proceed.
rm -f "$D/c.touched"
"$TR" --label c --out "$D" --exclusive "definitelynotrunning${MARK}xyz" -- /usr/bin/touch "$D/c.touched" >/dev/null 2>&1
rc=$?
[ $rc -eq 0 ] && [ -f "$D/c.touched" ] && note ok || note bad "B5: a non-matching pattern must not refuse (rc=$rc)"

kill "$COMP" 2>/dev/null; wait "$COMP" 2>/dev/null

# --- C. no --exclusive -> unchanged behaviour, still runs.
rm -f "$D/d.touched"
"$TR" --label d --out "$D" -- /usr/bin/touch "$D/d.touched" >/dev/null 2>&1
rc=$?
[ $rc -eq 0 ] && [ -f "$D/d.touched" ] && note ok || note bad "C: without --exclusive the wrapper must behave as before (rc=$rc)"

# --- D. the wrapper still propagates the command's own exit code.
"$TR" --label e --out "$D" --exclusive "$MARK" -- /bin/sh -c 'exit 7' >/dev/null 2>&1
[ $? -eq 7 ] && note ok || note bad "D: the command's rc must survive the wrapper"

echo "exclusivity controls: $ran run, $fails failed"
rm -rf "$D"
exit $([ $fails -eq 0 ] && echo 0 || echo 1)
