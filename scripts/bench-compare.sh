#!/usr/bin/env bash
# bench-compare.sh — A/B benchmark comparison across two git refs.
#
# Usage is unchanged and documented in scripts/lib/bench-compare-impl.sh, which
# holds the measurement body. This file is the LOCKING ENTRY POINT and nothing
# else: it runs that body under scripts/lib/bench-locks.py, which holds the
# machine-wide bench-window and Metal GPU locks for the whole run.
#
# DO NOT WRAP THIS SCRIPT in a caller-side bench-window helper. It takes both
# locks itself now. Wrapping it makes the body wait on a lock its own ancestor
# holds; the wait is bounded and names the ancestor, but the run is lost.
#
# WHY A SEPARATE FILE rather than one script that re-execs itself under a guard
# flag or an environment marker. A marker is a claim about lock state supplied
# by the environment, and the thing being checked must never supply the data
# the check depends on: a marker left exported in a shell makes the body skip
# locking and produce an unlocked measurement that is indistinguishable, in the
# report and in the exit status, from a locked one. Two files cannot recurse
# and there is no state to go stale. The body additionally refuses to run
# unless the PID recorded in the lock status is one of its own ancestors, which
# catches a stale or copied status file and an accidental direct invocation. It
# does not make direct invocation impossible: the file supplies the PID, so a
# caller who deliberately records an ancestor's PID gets through. See the
# comment above verify_locks in the body for what that check does and does not
# establish.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO/.cache"
exec python3 "$REPO/scripts/lib/bench-locks.py" \
  --label "bench-compare" \
  --status-file "$REPO/.cache/bench-locks-status.txt" \
  -- "$REPO/scripts/lib/bench-compare-impl.sh" "$@"
