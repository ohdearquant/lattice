#!/usr/bin/env bash
# Self-test for scripts/lib/ensure-noindex-marker.sh.
#
# The marker suppresses Spotlight indexing of the bench worktrees. Indexing
# churn lands asymmetrically across a base-then-head A/B and reads as a code
# delta, so a silently absent marker does not fail the run: it corrupts the
# numbers while they still look trustworthy. That makes the guard fail-closed
# infrastructure, and this proves each branch's exit code in a sandbox:
#   bash scripts/ensure-noindex-marker-selftest.sh
#
# The invariant under test, stated once: WHEN THE GUARD EXITS 0, A REGULAR
# MARKER FILE EXISTS. When one cannot be established, the guard exits nonzero
# and the caller never measures. Both halves are asserted on every case.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/scripts/lib/ensure-noindex-marker.sh"
# The measurement body, not the entry point: scripts/bench-compare.sh only
# takes the machine-wide locks and execs this. Naming a file by path is how a
# check like this goes stale, so the existence assertion below is what turns a
# future move into a loud failure instead of a silently-skipped call-site test.
CALLER="$REPO/scripts/lib/bench-compare-impl.sh"
if [ ! -f "$CALLER" ]; then
  echo "FATAL: expected caller $CALLER does not exist — the call-site assertions" >&2
  echo "  below would vacuously pass against a moved or renamed file." >&2
  exit 1
fi
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

# 4. THE REVIEWED DEFECT (#1089 round 1). A dangling marker symlink is the case
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

# 5. Symlink pointing at an existing file elsewhere: still not a real marker in
#    this directory (Spotlight reads the directory), so it is replaced.
D="$SB/livelink/.cache"; mkdir -p "$D"; : > "$SB/livelink/elsewhere"
ln -s "$SB/livelink/elsewhere" "$D/.metadata_never_index"
run "$D"; check "live marker symlink is replaced" 0 $?
check_marker "  -> replaced by a regular file" "$D" file

# 6. Marker path occupied by a directory: cannot become a file, must not proceed.
D="$SB/isdir/.cache"; mkdir -p "$D/.metadata_never_index/occupied"
run "$D"; check "non-empty dir at marker path fails closed" 1 $?

# 7. FAIL-CLOSED PROOF. An unwritable .cache cannot hold a marker at all, so the
#    guard must refuse rather than let an unprotected A/B proceed. This is the
#    case that fails if the trailing `|| true` ever comes back.
if [ "$(id -u)" -eq 0 ]; then
  echo "  SKIP: unwritable-dir case (running as root bypasses permissions)"
else
  D="$SB/readonly/.cache"; mkdir -p "$D"; chmod a-w "$D"
  run "$D"; check "unwritable dir fails closed" 1 $?
  check_marker "  -> no marker was created" "$D" absent
  case "$OUT" in
    *FATAL*) echo "  PASS:   -> diagnostic names the failure"; pass=$((pass+1)) ;;
    *) echo "  FAIL:   -> no FATAL diagnostic in output"; fail=$((fail+1)) ;;
  esac
  chmod u+w "$D"
fi

# 8. CALL-SITE ASSERTION. Testing the helper in isolation would still pass if
#    the measurement body stopped calling it, so assert the wiring too.
if grep -q 'scripts/lib/ensure-noindex-marker.sh' "$CALLER"; then
  echo "  PASS: the measurement body invokes the guard"; pass=$((pass+1))
else
  echo "  FAIL: the measurement body no longer invokes the guard"; fail=$((fail+1))
fi

# 9. The guard must run BEFORE the worktrees it protects are created.
guard_ln=$(grep -n 'ensure-noindex-marker.sh' "$CALLER" | head -1 | cut -d: -f1)
wt_ln=$(grep -n 'worktree add' "$CALLER" | head -1 | cut -d: -f1)
if [ -n "$guard_ln" ] && [ -n "$wt_ln" ] && [ "$guard_ln" -lt "$wt_ln" ]; then
  echo "  PASS: guard precedes worktree creation (line $guard_ln < $wt_ln)"; pass=$((pass+1))
else
  echo "  FAIL: guard does not precede worktree creation (guard=$guard_ln wt=$wt_ln)"; fail=$((fail+1))
fi

echo "=== $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]
