#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -ne 0 ]]; then
    echo "usage: scripts/bench-gate.sh" >&2
    exit 2
fi
source "$REPO/scripts/lib/bench-supervision.sh"

bench_gate_measurement() {
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
echo "arch: $arch"
root=".cache/bench-gate-criterion"
inference_root="$root/inference/criterion"
embed_root="$root/embed/criterion"
rm -rf "$root"
mkdir -p "$inference_root" "$embed_root"
baseline=".cache/perf-baselines/$arch"
[[ -d "$baseline" ]] ||
    {
        echo "no baseline for $arch"
        exit 1
    }

cargo bench -p lattice-inference --bench elementwise_cpu_bench -- --list \
    >"$root/inference-bench-list"
sed -n 's/: benchmark$//p' "$root/inference-bench-list" \
    >"$root/inference-bench-ids"
while IFS= read -r bench; do
    [[ -d "$baseline/$bench" ]] || continue
    mkdir -p "$inference_root/$(dirname "$bench")"
    cp -R "$baseline/$bench" "$inference_root/$bench"
done <"$root/inference-bench-ids"

cargo bench -p lattice-embed --bench simd -- --list \
    >"$root/embed-bench-list"
sed -n 's/: benchmark$//p' "$root/embed-bench-list" \
    >"$root/embed-bench-ids"
while IFS= read -r bench; do
    [[ -d "$baseline/$bench" ]] || continue
    mkdir -p "$embed_root/$(dirname "$bench")"
    cp -R "$baseline/$bench" "$embed_root/$bench"
done <"$root/embed-bench-ids"

bench_quiet_checkpoint "bench-gate: before measurements"
CRITERION_HOME="$inference_root" cargo bench \
    -p lattice-inference --bench elementwise_cpu_bench -- \
    --baseline base --noplot
bench_quiet_checkpoint "bench-gate: between targets"
CRITERION_HOME="$embed_root" cargo bench \
    -p lattice-embed --bench simd -- --baseline base --noplot
bench_quiet_checkpoint "bench-gate: after measurements"

echo "UNSUITABLE AS BENCHMARK EVIDENCE: local bench-gate has no run provenance"
rc=0
gate_rc=0
python3 scripts/perf-bench-gate.py \
    "$inference_root" "$arch-local/lattice-inference:elementwise_cpu_bench" \
    --target lattice-inference:elementwise_cpu_bench \
    --require-measurements || rc=$?
python3 scripts/perf-bench-gate.py \
    "$embed_root" "$arch-local/lattice-embed:simd" \
    --target lattice-embed:simd \
    --require-measurements || gate_rc=$?
if [[ "$rc" -eq 2 || "$gate_rc" -eq 2 ]]; then
    exit 2
fi
if [[ "$rc" -ne 0 || "$gate_rc" -ne 0 ]]; then
    exit 1
fi
}

bench_supervise_entry "bench-gate" durable bench_gate_measurement "$@"
