#!/usr/bin/env bash
set -euo pipefail

# Run all synthetic benchmarks (no model weights required).
# Usage: ./scripts/bench_all.sh
#
# Skips model-dependent benches: e2e_bench, metal_decode_bench, mtp_decode
# Skips feature-gated benches unless opt-in: compute_attention_bench,
#   kv_cache_layout_bench (--features bench-internals),
#   decode_attn_bench (--features f16)

TIMESTAMP=$(date -Iseconds)

echo "=== Lattice Benchmark Suite ==="
echo "Date: $TIMESTAMP"
echo "Platform: $(uname -m) / $(uname -s)"
echo ""

INFERENCE_BENCHES=(
    inference_bench
    inference_perf
    attention_bench
    differential_attention_bench
    gated_attention_bench
    native_sparse_attention_bench
    tokenizer_bench
    topk_readback
)

EMBED_BENCHES=(
    simd
    simd_bench
    embeddings
)

FAILED=()

echo "--- inference benches ---"
for bench in "${INFERENCE_BENCHES[@]}"; do
    echo ">> $bench"
    if ! RUSTC_WRAPPER= cargo bench -p lattice-inference --bench "$bench" 2>&1; then
        FAILED+=("$bench")
        echo "FAILED: $bench"
    fi
    echo ""
done

echo "--- embed benches ---"
for bench in "${EMBED_BENCHES[@]}"; do
    echo ">> $bench"
    if ! RUSTC_WRAPPER= cargo bench -p lattice-embed --bench "$bench" 2>&1; then
        FAILED+=("$bench")
        echo "FAILED: $bench"
    fi
    echo ""
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "=== FAILED (${#FAILED[@]}) ==="
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi

echo "=== Done (all passed) ==="
