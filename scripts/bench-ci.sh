#!/usr/bin/env bash
# Runs the lattice-inference and lattice-embed Criterion benches once each,
# saving a local Criterion baseline (no gate comparison, no provenance
# tracking). Supervised via scripts/lib/bench-supervision.sh.
# Usage: scripts/bench-ci.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -ne 0 ]]; then
    echo "usage: scripts/bench-ci.sh" >&2
    exit 2
fi
source "$REPO/scripts/lib/bench-supervision.sh"

bench_ci_measurement() {
cd "$REPO"
cargo bench -p lattice-inference --bench elementwise_cpu_bench -- \
    --save-baseline local --noplot || {
    echo "bench-ci: NOT MEASURABLE: lattice-inference benchmark failed" >&2
    return 2
}
bench_quiet_checkpoint "bench-ci: between targets"
cargo bench -p lattice-embed --bench simd -- --save-baseline local --noplot || {
    echo "bench-ci: NOT MEASURABLE: lattice-embed benchmark failed" >&2
    return 2
}
bench_quiet_checkpoint "bench-ci: after targets"
}

bench_supervise_entry "bench-ci" durable bench_ci_measurement "$@"
