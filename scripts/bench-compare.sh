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
REPO="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$REPO/scripts/lib/bench_supervision.py" run \
  --label "bench-compare" \
  --entrypoint \
  -- "$REPO/scripts/lib/bench-compare-impl.sh" "$@"
