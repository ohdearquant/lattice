#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -ne 0 ]]; then
    echo "usage: scripts/e2e-parity-local.sh" >&2
    exit 2
fi
source "$REPO/scripts/lib/bench-supervision.sh"
bench_supervise_entry "e2e-parity-local" handoff - "$@"

(
    bench_close_lock_fds
    cd "$REPO"
    cargo build --release --bin qwen35_generate -p lattice-inference --features f16
)
cd "$REPO"
exec python3 scripts/e2e_parity_check.py
