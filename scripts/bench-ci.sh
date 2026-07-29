#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -ne 0 ]]; then
    echo "usage: scripts/bench-ci.sh" >&2
    exit 2
fi
source "$REPO/scripts/lib/bench-supervision.sh"
bench_supervise_entry "bench-ci" durable "$@"

cd "$REPO"
cargo bench -p lattice-inference --bench elementwise_cpu_bench -- \
    --save-baseline local --noplot
bench_quiet_checkpoint "bench-ci: between targets"
cargo bench -p lattice-embed --bench simd -- --save-baseline local --noplot
bench_quiet_checkpoint "bench-ci: after targets"
