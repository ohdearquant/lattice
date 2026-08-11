#!/usr/bin/env bash
# Builds the release qwen35_generate binary, then runs scripts/e2e_parity_check.py
# under scripts/lib/bench-supervision.sh's machine-state supervision to
# compare local generation output against the recorded parity reference.
# Usage: scripts/e2e-parity-local.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -ne 0 ]]; then
    echo "usage: scripts/e2e-parity-local.sh" >&2
    exit 2
fi
source "$REPO/scripts/lib/bench-supervision.sh"
bench_supervise_entry "e2e-parity-local" handoff - "$@"

(
    bench_close_supervisor_witness
    cd "$REPO"
    cargo build --release --bin qwen35_generate -p lattice-inference --features f16
)
cd "$REPO"
exec "${PYTHON_BIN:?PYTHON_BIN not set - bench_supervise_entry should have exported it}" scripts/e2e_parity_check.py
