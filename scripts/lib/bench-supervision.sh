#!/usr/bin/env bash

bench_supervise_entry() {
    local label="$1"
    local mode="$2"
    shift 2

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
    if [[ -n "${LATTICE_BENCH_LOCK_FDS:-}" ]]; then
        local fd
        local -a bench_lock_fds
        IFS=',' read -r -a bench_lock_fds <<<"$LATTICE_BENCH_LOCK_FDS"
        for fd in "${bench_lock_fds[@]}"; do
            eval "exec ${fd}>&-"
        done
        unset LATTICE_BENCH_LOCK_FDS
    fi
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
