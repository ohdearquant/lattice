#!/usr/bin/env bash

bench_close_lock_fds() {
    local fd
    local -a bench_lock_fds

    IFS=',' read -r -a bench_lock_fds <<<"$LATTICE_BENCH_LOCK_FDS"
    for fd in "${bench_lock_fds[@]}"; do
        eval "exec ${fd}>&-"
    done
    unset LATTICE_BENCH_LOCK_FDS
}

bench_supervise_entry() {
    local label="$1"
    local mode="$2"
    local measurement="$3"
    shift 3

    local repo helper
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    helper="$repo/scripts/lib/bench_supervision.py"

    if [[ -z "${LATTICE_BENCH_LOCK_STATUS:-}" ]]; then
        if [[ "$mode" == "durable" ]]; then
            exec python3 "$helper" run --label "$label" --quiet --entrypoint -- "$0" "$@"
        fi
        exec python3 "$helper" run --label "$label" --entrypoint -- "$0" "$@"
    fi
    if [[ "$mode" == "durable" ]]; then
        if ! python3 "$helper" verify --require-quiet; then
            exit 2
        fi
    else
        if ! python3 "$helper" verify; then
            exit 2
        fi
    fi
    if [[ "$mode" == "handoff" ]]; then
        return 0
    fi

    local measurement_rc=0
    (
        bench_close_lock_fds
        "$measurement" "$@"
    ) || measurement_rc=$?
    if ! python3 "$helper" verify; then
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
