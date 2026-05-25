#!/usr/bin/env bash
set -euo pipefail

LI="/Users/lion/projects/.venv/bin/li"
WT="/Users/lion/projects/khive/lattice"

"$LI" agent -a reviewer --bypass --effort high --timeout 900 --cwd "$WT" "
Review PR #76: perf: SIMD optimization + Metal GPU fixes + regression tests

Branch: show/perf-opt/integration → main
Diff: 32 files changed, 4381 insertions, 347 deletions

## Scope
This PR delivers CPU SIMD optimizations (NEON/AVX2), Metal GPU compile fixes,
and regression tests for the lattice pure-Rust transformer inference engine.

Key changes:
1. **CPU SIMD kernels** (crates/inference/src/forward/cpu/):
   - elementwise.rs: NEON polynomial exp for silu (replaces scalar exp()), 4x unrolled silu/rms_norm/elementwise_mul
   - activation.rs: 4x unrolled GELU and add_bias_gelu NEON paths
   - softmax.rs: 4x unrolled softmax NEON path
   - norm.rs: fused mean+variance pass in layer_norm (3-pass → 2-pass)

2. **Decode attention NEON** (crates/inference/src/attention/decode.rs):
   - Added NEON-accelerated QK dot product, softmax, V accumulation
   - Schraudolph fast_exp for decode softmax (bias cancels in normalization)

3. **Embed SIMD** (crates/embed/src/simd/):
   - quantized.rs: assert! → debug_assert! for -128 invariant check (27x speedup)
   - cosine.rs: SIMD dot kernel for norms in batch_cosine_one_vs_many
   - normalize.rs: fast reciprocal sqrt via vrsqrteq_f32 + 2 Newton-Raphson steps

4. **Metal GPU** (crates/inference/src/forward/metal_qwen35.rs):
   - Fixed MetalCommonLayerWeights struct (gate/up/down → ffn enum)
   - Fixed break-outside-loop → proper GenerateOutput return
   - Fixed missing enable_mtp/grammar fields in bench binary
   - Zero-copy greedy argmax (skip 993KB logits readback per step)
   - Gate+up MLP weight fusion (24 fewer Metal dispatches)

5. **Tests** (crates/inference/src/forward/cpu/tests.rs):
   - 6 SIMD/scalar parity tests: silu, gelu, rms_norm, layer_norm, softmax, elementwise_mul
   - Each tests at hidden sizes 896, 2048, 4096

6. **Benchmarks** (crates/inference/benches/elementwise_bench.rs):
   - Added: gelu, softmax, add_bias_gelu, layer_norm, decode_attention benchmarks

## VERIFY blocks
- [ ] VERIFY: neon_exp_f32 polynomial accuracy — check that the degree-6 Horner polynomial
      matches libm exp() to within 1 ULP for the clamped range [-87.33, 88.72]
- [ ] VERIFY: Schraudolph fast_exp in decode softmax — does the systematic bias truly cancel
      after normalization? Check with adversarial inputs (e.g., all-equal scores, one-hot)
- [ ] VERIFY: debug_assert! change in quantized.rs — the original assert! caught -128 values
      that would overflow in signed dot product. Is debug_assert sufficient, or could -128
      slip through in production and cause silent numerical corruption?
- [ ] VERIFY: fused layer_norm variance formula (E[x²]-E[x]²) — is it numerically stable
      for typical hidden-state magnitudes, or can catastrophic cancellation occur?
- [ ] VERIFY: zero-copy greedy argmax — does scanning GPU shared buffer directly after
      wait_until_completed guarantee coherent reads on all Apple Silicon generations?
- [ ] VERIFY: gate+up weight fusion — is the concatenated Q8/Q4 buffer layout compatible
      with the existing gemv_q8_decode/gemv_q4_decode kernels (row_bytes calculation)?

## What I want
Adversarial review. Find real bugs, not style nits. Focus on:
- Numerical correctness (SIMD vs scalar divergence)
- Safety of unsafe blocks (pointer arithmetic, bounds)
- Metal GPU correctness (MSL shader changes, buffer sizing)
- Missing edge cases in the NEON paths (non-multiple-of-16 sizes, empty inputs)

Write findings to codex_review_pr76.md in the repo root.
Verdict: APPROVE / APPROVE-WITH-SUGGESTIONS / REQUEST CHANGES / REJECT
"
