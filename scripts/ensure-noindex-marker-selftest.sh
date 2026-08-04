#!/usr/bin/env bash
# Self-test for scripts/lib/ensure-noindex-marker.sh.
#
# The marker suppresses Spotlight indexing of benchmark worktrees and in-place
# target directories. Indexing churn can overlap a timing phase and read as a
# code delta, so a silently absent marker does not fail the run: it corrupts
# the numbers while they still look trustworthy. That makes the guard
# fail-closed infrastructure, and this proves each branch's exit code in a
# sandbox:
#   bash scripts/ensure-noindex-marker-selftest.sh
#
# The invariant under test, stated once: WHEN THE GUARD EXITS 0, A REGULAR
# MARKER FILE EXISTS. When one cannot be established, the guard exits 2 (the
# input/instrumentation-error class, not a regression verdict) and the caller
# never measures. Both halves are asserted on every case.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/scripts/lib/ensure-noindex-marker.sh"
# The measurement body, not the entry point: scripts/bench-compare.sh only
# takes the machine-wide locks and execs this. The slopefit driver is the build
# path used by the scheduled macOS measurement workflow.
COMPARE_CALLER="$REPO/scripts/lib/bench-compare-impl.sh"
SLOPEFIT_CALLER="$REPO/scripts/bench_decode_slopefit.sh"
for caller in "$COMPARE_CALLER" "$SLOPEFIT_CALLER"; do
  if [ ! -f "$caller" ]; then
    echo "FATAL: expected caller $caller does not exist — the call-site assertions" >&2
    echo "  below would vacuously pass against a moved or renamed file." >&2
    exit 1
  fi
done
SB="$(mktemp -d)"
trap 'chmod -R u+w "$SB" 2>/dev/null; rm -rf "$SB"' EXIT

pass=0; fail=0
check() {  # $1=desc $2=expected_exit $3=actual_exit
  if [ "$2" = "$3" ]; then
    echo "  PASS: $1 (exit $3)"; pass=$((pass+1))
  else
    echo "  FAIL: $1 — expected exit $2 got $3"
    echo "        output: $(tr '\n' '|' <<<"$OUT" | tail -c 300)"
    fail=$((fail+1))
  fi
}
check_marker() {  # $1=desc $2=dir $3=want ("file" or "absent")
  local m="$2/.metadata_never_index" got="absent"
  [ -f "$m" ] && got="file"
  [ -L "$m" ] && got="symlink"
  [ -d "$m" ] && got="dir"
  if [ "$got" = "$3" ]; then
    echo "  PASS: $1 (marker=$got)"; pass=$((pass+1))
  else
    echo "  FAIL: $1 — wanted marker=$3 got marker=$got"; fail=$((fail+1))
  fi
}
run() { OUT="$(bash "$SRC" "$1" 2>&1)"; return $?; }

echo "=== ensure-noindex-marker.sh fail-closed self-test ==="

# 1. Absent marker in a fresh dir: created.
D="$SB/fresh/.cache"
run "$D"; check "absent marker is created" 0 $?
check_marker "  -> marker is a regular file" "$D" file

# 2. Parent dir does not exist yet: mkdir -p path still lands protected.
D="$SB/deep/a/b/.cache"
run "$D"; check "missing parent dirs are created" 0 $?
check_marker "  -> marker is a regular file" "$D" file

# 3. Existing regular marker: left untouched, not re-truncated (idempotent).
D="$SB/existing/.cache"; mkdir -p "$D"; echo "sentinel" > "$D/.metadata_never_index"
run "$D"; check "existing regular marker is kept" 0 $?
if [ "$(cat "$D/.metadata_never_index")" = "sentinel" ]; then
  echo "  PASS:   -> existing content preserved"; pass=$((pass+1))
else
  echo "  FAIL:   -> existing marker was truncated"; fail=$((fail+1))
fi

# 4. Build-tree cleanup removes the marker with the guarded resource. A caller
#    that reapplies the helper before every build must recreate both.
D="$SB/recreated/target"
run "$D"; first_rc=$?
rm -rf "$D"
run "$D"; second_rc=$?
if [ "$first_rc" -eq 0 ] && [ "$second_rc" -eq 0 ]; then
  echo "  PASS: marker is recreated after target cleanup"; pass=$((pass+1))
else
  echo "  FAIL: marker recreation failed (first=$first_rc second=$second_rc)"; fail=$((fail+1))
fi
check_marker "  -> recreated marker is a regular file" "$D" file

# 5. THE REVIEWED DEFECT (#1089 round 1). A dangling marker symlink is the case
#    the fail-open form got wrong: `test -e` is FALSE for a dangling link, the
#    redirect then followed it to an uncreatable target and failed, and the
#    trailing `|| true` reported success anyway — so the A/B ran unprotected.
#    The guard now repairs it instead of merely refusing, which is the stronger
#    outcome: a stray symlink must not block every bench on the machine when a
#    real marker can be established in a writable directory.
D="$SB/dangling/.cache"; mkdir -p "$D"
ln -s "$SB/dangling/no-such-dir/target" "$D/.metadata_never_index"
run "$D"; check "dangling marker symlink is repaired" 0 $?
check_marker "  -> replaced by a regular file" "$D" file

# 6. Symlink pointing at an existing file elsewhere: still not a real marker in
#    this directory (Spotlight reads the directory), so it is replaced.
D="$SB/livelink/.cache"; mkdir -p "$D"; : > "$SB/livelink/elsewhere"
ln -s "$SB/livelink/elsewhere" "$D/.metadata_never_index"
run "$D"; check "live marker symlink is replaced" 0 $?
check_marker "  -> replaced by a regular file" "$D" file

# 7. Marker path occupied by a directory: cannot become a file, must not proceed.
D="$SB/isdir/.cache"; mkdir -p "$D/.metadata_never_index/occupied"
run "$D"; check "non-empty dir at marker path fails closed" 2 $?

# 8. FAIL-CLOSED PROOF. An unwritable .cache cannot hold a marker at all, so the
#    guard must refuse rather than let an unprotected A/B proceed. This is the
#    case that fails if the trailing `|| true` ever comes back.
if [ "$(id -u)" -eq 0 ]; then
  echo "  SKIP: unwritable-dir case (running as root bypasses permissions)"
else
  D="$SB/readonly/.cache"; mkdir -p "$D"; chmod a-w "$D"
  run "$D"; check "unwritable dir fails closed" 2 $?
  check_marker "  -> no marker was created" "$D" absent
  case "$OUT" in
    *FATAL*) echo "  PASS:   -> diagnostic names the failure"; pass=$((pass+1)) ;;
    *) echo "  FAIL:   -> no FATAL diagnostic in output"; fail=$((fail+1)) ;;
  esac
  chmod u+w "$D"
fi

# 9. NO ARGUMENT. The required-positional-argument expansion must not leak a
#    raw exit 1 (the same code the gate reserves for a confirmed regression).
OUT="$(bash "$SRC" 2>&1)"; rc=$?
check "no argument fails closed" 2 "$rc"
case "$OUT" in
  *FATAL*) echo "  PASS:   -> diagnostic names the failure"; pass=$((pass+1)) ;;
  *) echo "  FAIL:   -> no FATAL diagnostic in output"; fail=$((fail+1)) ;;
esac

# 10. UNWRITABLE PARENT (mkdir -p failure). A regular file occupying the
#     target path makes `mkdir -p` fail with its own raw exit 1; that must be
#     normalized too, not just the marker-creation failures below it.
F="$SB/occupied-by-file"; : > "$F"
OUT="$(bash "$SRC" "$F/sub" 2>&1)"; rc=$?
check "mkdir -p failure (path occupied by a file) fails closed" 2 "$rc"
case "$OUT" in
  *FATAL*) echo "  PASS:   -> diagnostic names the failure"; pass=$((pass+1)) ;;
  *) echo "  FAIL:   -> no FATAL diagnostic in output"; fail=$((fail+1)) ;;
esac

# 11. CALL-SITE ASSERTIONS. Testing the helper in isolation would still pass if
#    a measurement body stopped calling it, so pin the exact protected trees.
cache_call="\"\$REPO/scripts/lib/ensure-noindex-marker.sh\" \"\$REPO/.cache\""
target_call="\"\$REPO/scripts/lib/ensure-noindex-marker.sh\" \"\$REPO/target\""

if grep -qF "$cache_call" "$COMPARE_CALLER"; then
  echo "  PASS: bench-compare protects its worktree parent"; pass=$((pass+1))
else
  echo "  FAIL: bench-compare no longer protects its worktree parent"; fail=$((fail+1))
fi

if grep -qF "$target_call" "$COMPARE_CALLER"; then
  echo "  PASS: bench-compare protects its in-place target"; pass=$((pass+1))
else
  echo "  FAIL: bench-compare no longer protects its in-place target"; fail=$((fail+1))
fi

if grep -qF "$target_call" "$SLOPEFIT_CALLER"; then
  echo "  PASS: slopefit protects its in-place target"; pass=$((pass+1))
else
  echo "  FAIL: slopefit no longer protects its in-place target"; fail=$((fail+1))
fi

# 12. Each marker must exist before the operation that creates or builds the
#     protected resource. These ordering checks fail if a call drifts too late.
cache_guard_ln=$(grep -nF "$cache_call" "$COMPARE_CALLER" | head -1 | cut -d: -f1)
wt_ln=$(grep -n 'worktree add' "$COMPARE_CALLER" | head -1 | cut -d: -f1)
if [ -n "$cache_guard_ln" ] && [ -n "$wt_ln" ] && [ "$cache_guard_ln" -lt "$wt_ln" ]; then
  echo "  PASS: cache guard precedes worktree creation (line $cache_guard_ln < $wt_ln)"; pass=$((pass+1))
else
  echo "  FAIL: cache guard does not precede worktree creation (guard=$cache_guard_ln wt=$wt_ln)"; fail=$((fail+1))
fi

target_guard_ln=$(grep -nF "$target_call" "$COMPARE_CALLER" | head -1 | cut -d: -f1)
compare_build_ln=$(grep -nE '^[[:space:]]*(if[[:space:]]+)?cargo bench[[:space:]]' "$COMPARE_CALLER" | head -1 | cut -d: -f1)
if [ -n "$target_guard_ln" ] && [ -n "$compare_build_ln" ] && [ "$target_guard_ln" -lt "$compare_build_ln" ]; then
  echo "  PASS: target guard precedes bench-compare build (line $target_guard_ln < $compare_build_ln)"; pass=$((pass+1))
else
  echo "  FAIL: target guard does not precede bench-compare build (guard=$target_guard_ln build=$compare_build_ln)"; fail=$((fail+1))
fi

slopefit_guard_ln=$(grep -nF "$target_call" "$SLOPEFIT_CALLER" | head -1 | cut -d: -f1)
slopefit_build_ln=$(grep -n '^cargo build' "$SLOPEFIT_CALLER" | head -1 | cut -d: -f1)
if [ -n "$slopefit_guard_ln" ] && [ -n "$slopefit_build_ln" ] && [ "$slopefit_guard_ln" -lt "$slopefit_build_ln" ]; then
  echo "  PASS: target guard precedes slopefit build (line $slopefit_guard_ln < $slopefit_build_ln)"; pass=$((pass+1))
else
  echo "  FAIL: target guard does not precede slopefit build (guard=$slopefit_guard_ln build=$slopefit_build_ln)"; fail=$((fail+1))
fi

# 13. CLOSED STDERR. Every FATAL diagnostic writes to fd 2 under `set -e`
#     (originally `set -euo pipefail`, self-test drops -e above but the
#     guard script itself keeps it). If fd 2 is closed by the caller, an
#     unguarded `echo ... >&2` is itself a failing command and would abort
#     the guard with the shell's raw exit 1 -- the code this contract
#     reserves for a confirmed regression -- before the guard's own
#     explicit `exit 2` is reached. No argument is the cheapest way to
#     force a FATAL diagnostic.
OUT="(stderr closed; not captured)"
rc=$(/bin/bash -c "\"$SRC\" 2>&-; echo \$?" | tail -1)
check "no-argument FATAL survives closed stderr" 2 "$rc"

echo "=== $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]
