#!/usr/bin/env bash
# Runs an arbitrary COMMAND under scripts/lib/bench_supervision.py's
# machine-state supervision (GPU-lock/thermal/power checks), tagging the run
# with LABEL for its logs.
# Usage: scripts/bench-command.sh --label LABEL [--durable] [--gpu-handoff] -- COMMAND [ARG...]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/lib/bench-python.sh"
HELPER="$REPO/scripts/lib/bench_supervision.py"
LABEL=""
MODE="ordinary"
GPU_HANDOFF=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)
            LABEL="${2:-}"
            shift 2
            ;;
        --gpu-handoff)
            GPU_HANDOFF=(--gpu-handoff command)
            shift
            ;;
        --durable)
            MODE="durable"
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "usage: bench-command.sh --label LABEL [--durable] [--gpu-handoff] -- COMMAND [ARG...]" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$LABEL" || $# -eq 0 ]]; then
    echo "usage: bench-command.sh --label LABEL [--durable] [--gpu-handoff] -- COMMAND [ARG...]" >&2
    exit 2
fi

PYTHON_BIN="$(bench_require_python3 "bench-command.sh")" || exit 1
# Bash 3 treats an empty array as unset under nounset.
#
# --entrypoint is required here, not optional: it is what makes
# bench_supervision.py hand the wrapped command a liveness pipe
# (LATTICE_BENCH_SUPERVISOR_FD) instead of only a lock-status marker.
# scripts/lib/bench-supervision.sh's bench_supervise_entry and
# scripts/bench-compare.sh already pass it unconditionally for exactly
# this reason. Without it, a wrapped command that is itself a
# self-supervising Python entry point (one that calls
# ensure_python_entrypoint) sees the marker, concludes a supervisor is
# already present, and then refuses because the pipe it looks for was
# never created ("LATTICE_BENCH_SUPERVISOR_FD is not set"). An ordinary
# command (cargo, a plain binary) never reads that pipe or its file
# descriptor, so passing it unconditionally does not change behavior
# for the common case.
if [[ "$MODE" == "durable" ]]; then
    exec "$PYTHON_BIN" "$HELPER" run --label "$LABEL" --quiet --entrypoint ${GPU_HANDOFF[@]+"${GPU_HANDOFF[@]}"} -- "$@"
fi
exec "$PYTHON_BIN" "$HELPER" run --label "$LABEL" --entrypoint ${GPU_HANDOFF[@]+"${GPU_HANDOFF[@]}"} -- "$@"
