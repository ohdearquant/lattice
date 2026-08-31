#!/usr/bin/env bash
# Runs an arbitrary COMMAND under scripts/lib/bench_supervision.py's
# machine-state supervision (GPU-lock/thermal/power checks), tagging the run
# with LABEL for its logs.
# Usage: scripts/bench-command.sh --label LABEL [--durable] -- COMMAND [ARG...]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/lib/bench-python.sh"
HELPER="$REPO/scripts/lib/bench_supervision.py"
LABEL=""
MODE="ordinary"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)
            LABEL="${2:-}"
            shift 2
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
            echo "usage: bench-command.sh --label LABEL [--durable] -- COMMAND [ARG...]" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$LABEL" || $# -eq 0 ]]; then
    echo "usage: bench-command.sh --label LABEL [--durable] -- COMMAND [ARG...]" >&2
    exit 2
fi

PYTHON_BIN="$(bench_require_python3 "bench-command.sh")" || exit 1
if [[ "$MODE" == "durable" ]]; then
    exec "$PYTHON_BIN" "$HELPER" run --label "$LABEL" --quiet -- "$@"
fi
exec "$PYTHON_BIN" "$HELPER" run --label "$LABEL" -- "$@"
