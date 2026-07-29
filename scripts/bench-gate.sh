#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -ne 0 ]]; then
    echo "usage: scripts/bench-gate.sh" >&2
    exit 2
fi
source "$REPO/scripts/lib/bench-supervision.sh"
bench_supervise_entry "bench-gate" durable "$@"

cd "$REPO"
if [[ ! -d .cache/perf-baselines ]]; then
    git clone --depth=1 --branch=perf-baselines \
        "$(git remote get-url origin)" .cache/perf-baselines ||
        {
            echo "no perf-baselines branch yet — run bench-update.yml on main first"
            exit 1
        }
else
    git -C .cache/perf-baselines pull --ff-only
fi

arch="$(uname -m | sed 's/arm64/aarch64/')-$(uname -s | tr '[:upper:]' '[:lower:]')"
target_dir="${CARGO_TARGET_DIR:-$REPO/target}"
criterion_root="$target_dir/criterion"
echo "arch: $arch"
mkdir -p "$criterion_root"
cp -r ".cache/perf-baselines/$arch/." "$criterion_root/" 2>/dev/null ||
    {
        echo "no baseline for $arch"
        exit 1
    }

bench_quiet_checkpoint "bench-gate: before measurements"
cargo bench -p lattice-inference --bench elementwise_cpu_bench -- \
    --baseline base --noplot
bench_quiet_checkpoint "bench-gate: between targets"
cargo bench -p lattice-embed --bench simd -- --baseline base --noplot
bench_quiet_checkpoint "bench-gate: after measurements"
python3 scripts/perf-bench-gate.py "$criterion_root" "$arch-local"
