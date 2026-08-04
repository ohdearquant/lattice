#!/usr/bin/env bash
# bench-compare.sh — A/B benchmark comparison across two git refs.
#
# Usage is unchanged and documented in scripts/lib/bench-compare-impl.sh, which
# holds the measurement body. This file is the LOCKING ENTRY POINT and nothing
# else: it runs that body through scripts/lib/bench_supervision.py, whose
# dedicated supervisor holds the machine-wide bench-window and Metal GPU locks.
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
# and there is no state to go stale. The Python supervisor verifies the two
# inherited descriptor capabilities against the ordered canonical lock paths,
# keeps them private, and gives the shell body only a non-lock handoff pipe.
# Descriptor/path comparisons diagnose a mismatch present at either sampled
# boundary; they do not prove pathname continuity against rename-and-restore,
# so callers must cooperate by leaving lock names intact. A direct invocation
# with only a fabricated status file is refused at the handoff sample.
set -euo pipefail
if ! REPO="$(cd "$(dirname "$0")/.." && pwd)"; then
  echo "bench-compare: FATAL: cannot resolve the repository root (the" >&2 || :
  echo "script's parent directory was removed or is unreachable). Refusing" >&2 || :
  echo "to continue without a resolved root." >&2 || :
  exit 2
fi
if ! mkdir -p "$REPO/.cache" 2>/dev/null; then
  echo "bench-compare: FATAL: cannot create $REPO/.cache (unwritable parent," >&2 || :
  echo "or a non-directory already occupies that path). Refusing to continue" >&2 || :
  echo "rather than run the A/B without its lock-status directory." >&2 || :
  exit 2
fi
source "$REPO/scripts/lib/bench-python.sh"
PYTHON_BIN="$(bench_require_python3 "bench-compare.sh")" || exit 1
exec "$PYTHON_BIN" "$REPO/scripts/lib/bench_supervision.py" run \
  --label "bench-compare" \
  --entrypoint \
  -- "$REPO/scripts/lib/bench-compare-impl.sh" "$@"
