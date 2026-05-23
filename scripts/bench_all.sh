#!/usr/bin/env bash
set -euo pipefail

# Run all synthetic benchmarks (no model weights required).
# Usage: ./scripts/bench_all.sh [--save DIR]
#
# Skips: e2e_bench, metal_decode_bench (require model weights)

SAVE_DIR="${1:-.}"
TIMESTAMP=$(date -Iseconds)

echo "=== Lattice Benchmark Suite ==="
echo "Date: $TIMESTAMP"
echo "Platform: $(uname -m) / $(uname -s)"
echo ""

INFERENCE_BENCHES=(
    inference_bench
    inference_perf
    attention_bench
    compute_attention_bench
    decode_attn_bench
    differential_attention_bench
    gated_attention_bench
    native_sparse_attention_bench
    kv_cache_layout_bench
    tokenizer_bench
    topk_readback
    mtp_decode
)

EMBED_BENCHES=(
    simd
    simd_bench
    embeddings
)

echo "--- inference benches ---"
for bench in "${INFERENCE_BENCHES[@]}"; do
    echo ">> $bench"
    RUSTC_WRAPPER= cargo bench -p lattice-inference --bench "$bench" 2>&1 || echo "FAILED: $bench"
    echo ""
done

echo "--- embed benches ---"
for bench in "${EMBED_BENCHES[@]}"; do
    echo ">> $bench"
    RUSTC_WRAPPER= cargo bench -p lattice-embed --bench "$bench" 2>&1 || echo "FAILED: $bench"
    echo ""
done

echo "=== Done ==="
