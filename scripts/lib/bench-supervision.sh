#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bench-python.sh"

bench_close_supervisor_witness() {
    local fd="${LATTICE_BENCH_SUPERVISOR_FD:-}"

    if [[ "$fd" =~ ^[0-9]+$ ]]; then
        eval "exec ${fd}>&-"
    fi
    unset LATTICE_BENCH_SUPERVISOR_FD
}

bench_supervise_entry() {
    local label="$1"
    local mode="$2"
    local measurement="$3"
    shift 3

    local repo helper python_bin
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    helper="$repo/scripts/lib/bench_supervision.py"
    python_bin="$(bench_require_python3 "${0##*/}")" || exit 1

    if [[ -z "${LATTICE_BENCH_LOCK_STATUS:-}" ]]; then
        if [[ "$mode" == "durable" ]]; then
            exec "$python_bin" "$helper" run --label "$label" --quiet --entrypoint -- "$0" "$@"
        fi
        exec "$python_bin" "$helper" run --label "$label" --entrypoint -- "$0" "$@"
    fi
    if [[ "$mode" == "durable" ]]; then
        if ! "$python_bin" "$helper" verify --require-quiet; then
            exit 2
        fi
    else
        if ! "$python_bin" "$helper" verify; then
            exit 2
        fi
    fi
    if [[ "$mode" == "handoff" ]]; then
        return 0
    fi

    local measurement_rc=0 restore_errexit=0
    if [[ "$-" == *e* ]]; then
        restore_errexit=1
        set +e
    fi
    (
        if [[ "$restore_errexit" -eq 1 ]]; then
            set -e
        fi
        bench_close_supervisor_witness
        "$measurement" "$@"
    )
    measurement_rc=$?
    if [[ "$restore_errexit" -eq 1 ]]; then
        set -e
    fi
    if ! "$python_bin" "$helper" verify; then
        return 2
    fi
    return "$measurement_rc"
}

bench_quiet_checkpoint() {
    local label="$1"
    local repo
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    if ! python3 "$repo/scripts/lib/quiet-probe.py" --label "$label"; then
        echo "bench-supervision: machine was not quiet at $label; refusing to continue" >&2
        exit 2
    fi
}
