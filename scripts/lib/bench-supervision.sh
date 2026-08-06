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

    local repo helper
    if ! repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; then
        printf 'bench-supervision: FATAL: cannot resolve the repository root from %s; ' "${BASH_SOURCE[0]}" >&2 || :
        printf 'refusing to continue.\n' >&2 || :
        exit 2
    fi
    helper="$repo/scripts/lib/bench_supervision.py"
    PYTHON_BIN="$(bench_require_python3 "${0##*/}")" || exit 1
    export PYTHON_BIN

    if [[ -z "${LATTICE_BENCH_LOCK_STATUS:-}" ]]; then
        if [[ "$mode" == "durable" ]]; then
            exec "$PYTHON_BIN" "$helper" run --label "$label" --quiet --entrypoint -- "$0" "$@"
        fi
        exec "$PYTHON_BIN" "$helper" run --label "$label" --entrypoint -- "$0" "$@"
    fi
    if [[ "$mode" == "durable" ]]; then
        if ! "$PYTHON_BIN" "$helper" verify --require-quiet; then
            exit 2
        fi
    else
        if ! "$PYTHON_BIN" "$helper" verify; then
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
    if ! "$PYTHON_BIN" "$helper" verify; then
        return 2
    fi
    return "$measurement_rc"
}

bench_quiet_checkpoint() {
    local label="$1"
    local repo
    if ! repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; then
        printf 'bench-supervision: FATAL: cannot resolve the repository root from %s; ' "${BASH_SOURCE[0]}" >&2 || :
        printf 'refusing to continue.\n' >&2 || :
        exit 2
    fi
    if ! "${PYTHON_BIN:?PYTHON_BIN not set - bench_quiet_checkpoint requires bench_supervise_entry to have run first}" "$repo/scripts/lib/quiet-probe.py" --label "$label"; then
        echo "bench-supervision: machine was not quiet at $label; refusing to continue" >&2 || :
        exit 2
    fi
}
