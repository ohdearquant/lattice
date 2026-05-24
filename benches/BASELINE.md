# Lattice Performance Baseline

**Date**: 2026-05-23
**Commit**: ac38f20 (main, post ADR-057 LoRA lifecycle merge)
**Platform**: Apple Silicon (aarch64 / Darwin)
**Rust**: stable (release profile, LTO)

## Inference Benchmarks (`lattice-inference`)

### Attention Kernel (standard, synthetic)

| Seq Len | Time/op  | Throughput    |
| ------- | -------- | ------------- |
| 16      | 96.6 us  | 31.8 Melem/s  |
| 32      | 125.6 us | 97.8 Melem/s  |
| 64      | 230.6 us | 213.1 Melem/s |
| 128     | 506.1 us | 388.5 Melem/s |

### Batch Encoding

| Batch Size | Time/op  | Throughput   |
| ---------- | -------- | ------------ |
| 32         | 180.6 ms | 177.2 elem/s |

### Logits Projection (matmul_bt)

| Variant          | Time/op  | Throughput    |
| ---------------- | -------- | ------------- |
| scalar           | 443.5 ms | 1.15 Gelem/s  |
| matmul_bt (SIMD) | 42.4 ms  | 11.99 Gelem/s |

**matmul_bt speedup over scalar: 10.5x**

### Tokenizer BPE

| Input Size | Time/op  | Throughput   |
| ---------- | -------- | ------------ |
| 4096 chars | 111.7 us | 30.7 Melem/s |

## Embed Benchmarks (`lattice-embed`)

### SIMD Distance Operations (dim=768)

| Operation             | Time/op | GFLOPS |
| --------------------- | ------- | ------ |
| cosine (scalar)       | 797 ns  | 2.9    |
| cosine (SIMD, full)   | 31 ns   | 74.3   |
| dot_product (f32)     | 26 ns   | 29.5   |
| normalize (SIMD)      | 77 ns   | 15.0   |
| euclidean_dist (SIMD) | 28 ns   | 41.1   |
| dot_product (int8)    | 287 ns  | 2.7    |
| cosine (int8)         | 370 ns  | 6.2    |

**f32 cosine SIMD speedup: 25.7x over scalar**
**int8 dot product: 2.2x over scalar** (room for improvement)

## LoRA Training Benchmarks (`lattice-tune`)

### train_lora (SGD, rank=2, d_in=3, d_out=2)

| Samples | Time/op  |
| ------- | -------- |
| 10      | 69.7 us  |
| 50      | 369.8 us |
| 100     | 733.0 us |

## Key Observations

1. **matmul_bt** is already well-optimized on macOS (Accelerate/AMX), 10.5x
   over scalar
2. **f32 SIMD** distance ops are excellent (19.8-25.7x speedup)
3. **int8 SIMD** has significant room for improvement (only 2.2x)
4. **Attention kernel** throughput scales well with seq_len
5. **Elementwise ops** (rms_norm, silu, elementwise_mul) have NO SIMD paths —
   pure scalar
6. **LoRA training** linear scaling with sample count (~7 us/sample)

## Optimization Work (perf-opt show, 10 PRs)

### Merge Sequence

Apply PRs in this order to avoid conflicts:

1. **PR #66** — `perf(bench): baseline performance numbers + bench_all.sh`
2. **PR #73** — `perf(inference): optimize matmul_bt scalar fallback + NEON prefetch`
3. **PR #71** — `perf(inference): SIMD elementwise ops (rms_norm, silu, elementwise_mul)`
4. **PR #70** — `perf(embed): optimize i8 dot product and fuse f32 cosine similarity`
5. **PR #67** — `feat(tune): batch LoRA training loop with train_lora API`
6. **PR #72** — `feat(tune): Adam/AdamW optimizer for LoRA training`
7. **PR #68** — `feat(tune): end-to-end LoRA lifecycle integration tests`
8. **PR #69** — `test(perf): add performance regression tests for CI`
9. **PR #74** — `perf(inference): attention throughput optimizations` (pending)
10. **PR #75** — `docs(bench): performance optimization report` (this PR)

### Optimization Summary

| Area             | PR  | What Changed                                 |
| ---------------- | --- | -------------------------------------------- |
| matmul_bt scalar | #73 | 8x k-loop unrolled m=1 fast path             |
| matmul_bt NEON   | #73 | Software prefetch for next k-chunk B-rows    |
| rms_norm         | #71 | NEON/AVX2 SIMD dispatch (was pure scalar)    |
| silu_inplace     | #71 | NEON/AVX2 SIMD with Schraudolph fast_exp     |
| elementwise_mul  | #71 | NEON/AVX2 SIMD dispatch                      |
| int8 dot product | #70 | Optimized SIMD path for i8 distance ops      |
| f32 cosine       | #70 | Fused single-pass cosine similarity          |
| attention        | #74 | Decode-path throughput optimization          |
| LoRA train loop  | #67 | Batch training with per-epoch loss tracking  |
| LoRA optimizer   | #72 | Adam/AdamW with bias correction, lr schedule |
| LoRA lifecycle   | #68 | End-to-end integration tests                 |
| Perf regression  | #69 | CI-safe performance regression tests         |
| Bench infra      | #66 | bench_all.sh + baseline numbers              |
