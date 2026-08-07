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
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -f "$REPO/scripts/lib/bench-python.sh" ]]; then
        source "$REPO/scripts/lib/bench-python.sh"
        PYTHON_BIN="$(bench_resolve_python3)" || PYTHON_BIN="python3"
    else
        # tests/test_bench_targets.py's bench-gate-policy test strips the
        # bench_supervise_entry call (which normally exports PYTHON_BIN) and
        # copies only Makefile + scripts/perf-bench-gate.py into its sandbox,
        # so scripts/lib/bench-python.sh is unreachable there. Bare python3
        # is the only interpreter this branch can resolve to; any caller
        # that skips both the supervisor and scripts/lib hits this fallback
        # and the original version-floor defect reappears for it.
        PYTHON_BIN="python3"
    fi
fi
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

# Build a full_id -> directory_name map from the stored baseline metadata.
# Criterion's directory_name differs from full_id for three-level IDs such as
# group/function/value (e.g. tier_prepared_query/int8_query_per_call/1000 is
# stored at tier_prepared_query/int8_query_per_call_1000). Reconstructing the
# path by string manipulation silently dropped those baselines; deriving the
# mapping from criterion's own benchmark.json avoids re-implementing its naming
# rule and makes the script resilient to future criterion changes.
baseline_id_map="$root/baseline-id-map.json"
find "$baseline" -type f -name benchmark.json -print0 |
    xargs -0 jq -s 'map({(.full_id): .directory_name}) | add' \
        >"$baseline_id_map"

copy_baseline_for_target() {
    local bench_list="$1"
    local target_root="$2"
    local skipped=0
    local bench
    while IFS= read -r bench; do
        [[ -n "$bench" ]] || continue
        local dir
        dir=$(jq -r --arg id "$bench" '.[$id] // empty' "$baseline_id_map")
        if [[ -z "$dir" ]]; then
            echo "bench-gate: skipping $bench: no stored baseline"
            skipped=$((skipped + 1))
            continue
        fi
        mkdir -p "$target_root/$(dirname "$dir")"
        cp -R "$baseline/$dir" "$target_root/$dir"
    done <"$bench_list"
    echo "bench-gate: $target_root skipped $skipped benchmarks"
}

cargo bench -p lattice-inference --bench elementwise_cpu_bench -- --list \
    >"$root/inference-bench-list"
sed -n 's/: benchmark$//p' "$root/inference-bench-list" \
    >"$root/inference-bench-ids"
copy_baseline_for_target "$root/inference-bench-ids" "$inference_root"

cargo bench -p lattice-embed --bench simd -- --list \
    >"$root/embed-bench-list"
sed -n 's/: benchmark$//p' "$root/embed-bench-list" \
    >"$root/embed-bench-ids"
copy_baseline_for_target "$root/embed-bench-ids" "$embed_root"

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
"$PYTHON_BIN" scripts/perf-bench-gate.py \
    "$inference_root" "$arch-local/lattice-inference:elementwise_cpu_bench" \
    --target lattice-inference:elementwise_cpu_bench \
    --require-measurements || rc=$?
"$PYTHON_BIN" scripts/perf-bench-gate.py \
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
