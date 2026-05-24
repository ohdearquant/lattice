# Lattice Performance Baseline

**Date**: 2026-05-23
**Commit**: 8ac486d (main, `feat: implement ADR-057 LoRA full-lifecycle consumer API (D1-D5) (#65)`)
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

## Key Observations

1. **matmul_bt** is already well-optimized on macOS (Accelerate/AMX), 10.5x
   over scalar
2. **f32 SIMD** distance ops are excellent (25-40x speedup)
3. **int8 SIMD** has significant room for improvement (only 2.2x)
4. **Attention kernel** throughput scales well with seq_len
5. **Elementwise ops** (rms_norm, silu, elementwise_mul) have NO SIMD paths —
   pure scalar

## Optimization Targets (prioritized)

1. **Elementwise SIMD** — rms_norm, silu, elementwise_mul (0 SIMD, called
   4x/layer/token)
2. **Int8 dot product** — 2.2x vs target of ~8-10x
3. **Attention decode path** — single-token (seq_len=1) specialization
4. **matmul_bt non-macOS** — tiling improvements for Linux/aarch64
