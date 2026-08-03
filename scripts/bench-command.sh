#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
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

if [[ "$MODE" == "durable" ]]; then
    exec python3 "$HELPER" run --label "$LABEL" --quiet -- "$@"
fi
exec python3 "$HELPER" run --label "$LABEL" -- "$@"
