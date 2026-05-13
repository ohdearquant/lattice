# lattice-embed Benchmarks

SIMD-accelerated vector operations benchmark suite for lattice-embed.

## Quick Start

```bash
# Quick SOTA validation (~1 min)
RUSTFLAGS="-C target-cpu=native" cargo bench -p lattice-embed --bench simd_bench

# Full benchmark suite (~5 min)
RUSTFLAGS="-C target-cpu=native" cargo bench -p lattice-embed

# SimSIMD comparison
RUSTFLAGS="-C target-cpu=native" cargo bench -p lattice-embed --bench simsimd_comparison

# Service benchmarks (requires model download)
cargo bench -p lattice-embed --features local -- --ignored
```

## Benchmark Files

| File                    | Purpose                  | Framework                 |
| ----------------------- | ------------------------ | ------------------------- |
| `simd_bench.rs`         | Quick SOTA validation    | Custom (Instant/Duration) |
| `simd.rs`               | Comprehensive SIMD tests | Criterion                 |
| `embeddings.rs`         | Service-level benchmarks | Criterion                 |
| `simsimd_comparison.rs` | Industry comparison      | Criterion                 |

## Important Clarifications

### Cosine Similarity: Full vs Pre-normalized

There are **two ways** to compute cosine similarity:

| Method                                | Time (384-dim) | Use Case                         |
| ------------------------------------- | -------------- | -------------------------------- |
| **Full cosine** (`cosine_similarity`) | ~85-90ns       | Arbitrary vectors                |
| **Pre-normalized** (`dot_product`)    | ~28-32ns       | Embeddings already L2-normalized |

**Full cosine** computes in a single pass:

```text
cosine(a, b) = (a · b) / (||a|| × ||b||)
```

This requires 6 FLOPs/element (3 FMAs for dot, norm_a, norm_b).

**Pre-normalized** vectors have `||a|| = ||b|| = 1`, so:

```text
cosine(a, b) = a · b  (just dot product)
```

This requires only 2 FLOPs/element.

Most embedding models (BGE, OpenAI, etc.) return normalized vectors, so `dot_product` is
the correct choice for similarity search.

### SimSIMD Comparison Methodology

The SimSIMD comparison includes these caveats:

1. **API overhead**: SimSIMD returns `Option<f64>`, adding ~2-5ns per call for Option
   handling. lattice-embed returns `f32` directly, assuming pre-validated slices.

2. **Distance vs Similarity**: SimSIMD returns cosine _distance_ (1 - similarity). We
   convert to similarity, adding a subtraction.

3. **Both approaches are valid**: SimSIMD is safer (defensive), lattice-embed is faster
   (validated at API boundary).

### GFLOPS Calculation

| Operation          | FLOPs/element | Notes                        |
| ------------------ | ------------- | ---------------------------- |
| dot_product        | 2             | multiply + add               |
| cosine_similarity  | 6             | 3 FMAs (dot, norm_a, norm_b) |
| normalize          | 3             | square + add + scale         |
| euclidean_distance | 3             | sub + square + add           |

## Performance Results (Apple Silicon M-series)

### SIMD Configuration

- **AVX2/FMA**: N/A (x86_64 only)
- **AVX-512 VNNI**: N/A (x86_64 only)
- **NEON**: Enabled (mandatory on aarch64)

### Core Operations (384-dim, BGE-small)

| Operation                    | Time  | GFLOPS | Target | Status   |
| ---------------------------- | ----- | ------ | ------ | -------- |
| cosine (scalar)              | 629ns | 3.7    | ~650ns | Baseline |
| cosine (SIMD, full)          | 27ns  | 85.3   | <90ns  | **PASS** |
| dot_product (pre-normalized) | 23ns  | 33.4   | <35ns  | **PASS** |
| normalize (SIMD)             | 80ns  | 14.4   | <60ns  | _Over_   |
| euclidean_distance           | 41ns  | 28.1   | ~90ns  | **PASS** |
| dot_product (int8)           | 13ns  | 59.1   | <30ns  | **PASS** |
| cosine (int8, pre-quant)     | 15ns  | 153.6  | <35ns  | **PASS** |

_Note: normalize is over target due to required two-pass algorithm (compute norm, then
scale)._

### Speedup Summary

| Comparison                          | Speedup   |
| ----------------------------------- | --------- |
| Scalar → SIMD float32 (full cosine) | **23.3x** |
| Scalar → SIMD int8                  | **41.9x** |
| Float32 → Int8                      | **1.8x**  |

## Industry Comparison (vs SimSIMD)

[SimSIMD](https://github.com/ashvardanian/simsimd) is the industry-standard SIMD
library.

### Cosine Similarity (float32)

| Dimension | lattice-embed | SimSIMD | lattice-embed Advantage |
| --------- | ------------- | ------- | ----------------------- |
| 384       | 32ns          | 82ns    | **2.56x faster**        |
| 768       | 62ns          | 182ns   | **2.94x faster**        |
| 1024      | 81ns          | 251ns   | **3.10x faster**        |
| 1536      | 121ns         | 440ns   | **3.64x faster**        |

_Note: Part of lattice-embed's advantage comes from API design (no Option wrapping). See
methodology notes above._

### Int8 Dot Product

| Dimension | lattice-embed | SimSIMD | lattice-embed Advantage |
| --------- | ------------- | ------- | ----------------------- |
| 384       | 9.8ns         | 15.8ns  | **1.61x faster**        |
| 768       | 19.3ns        | 31.6ns  | **1.64x faster**        |
| 1024      | 27.8ns        | 39.1ns  | **1.41x faster**        |
| 1536      | 40.2ns        | 64.3ns  | **1.60x faster**        |

### Batch Search (1000 vectors @ 384-dim)

| Method  | lattice-embed | SimSIMD | lattice-embed Advantage   |
| ------- | ------------- | ------- | ------------------------- |
| float32 | 31.8µs        | 85.2µs  | **2.68x faster**          |
| int8    | 10.3µs        | N/A     | 3.1x vs lattice-embed f32 |

## Memory Efficiency

Int8 quantization provides ~4x memory reduction:

| Dimension | float32 | int8    | Reduction |
| --------- | ------- | ------- | --------- |
| 384       | 1,536 B | 400 B   | 3.8x      |
| 768       | 3,072 B | 784 B   | 3.9x      |
| 1024      | 4,096 B | 1,040 B | 3.9x      |
| 1536      | 6,144 B | 1,552 B | 4.0x      |

## Benchmark Methodology

### Best Practices Applied

- **Warmup**: 1,000 iterations before measurement
- **Statistical analysis**: Criterion provides confidence intervals
- **Setup isolation**: `iter_batched` separates setup from measured code
- **No heap in hot loop**: Pre-allocated buffers, `copy_from_slice` for reset
- **Compiler optimization**: `black_box()` prevents dead code elimination
- **Native instructions**: `RUSTFLAGS="-C target-cpu=native"`

### Dimensions Tested

- **384**: BGE-small-en-v1.5
- **768**: BGE-base-en-v1.5
- **1024**: BGE-large-en-v1.5
- **1536**: OpenAI Ada-002

### Batch Sizes

- **10**: Small batch (API call)
- **100**: Medium batch (document processing)
- **1000**: Large batch (bulk indexing)

## Understanding the Results

### Why lattice-embed NEON is Fast

1. **Single-pass cosine**: Computes dot + both norms in one traversal
2. **4× unrolling**: Reduces loop overhead, improves ILP
3. **4 independent accumulators**: Hides FMA latency (~4 cycles)
4. **FMA everywhere**: `vfmaq_f32` for all multiply-accumulate
5. **Deferred horizontal reduction**: Only at the end

### Optional: 8-Accumulator Optimization

Apple M-series can benefit from 8 accumulators for ~15% more throughput. Current
4-accumulator design prioritizes portability and register pressure.

### Normalize Two-Pass Limitation

Normalize requires two passes (compute norm, then scale), making it inherently ~2x
slower than single-pass operations. The 80ns achieved is reasonable given this
algorithmic constraint (vs 27ns for single-pass cosine).

## Running on Different Platforms

### Apple Silicon (M1/M2/M3/M4)

```bash
# NEON is always enabled
RUSTFLAGS="-C target-cpu=native" cargo bench -p lattice-embed
```

### Intel/AMD x86_64

```bash
# AVX2 + FMA (most modern CPUs)
RUSTFLAGS="-C target-cpu=native" cargo bench -p lattice-embed

# Check SIMD capabilities in output
```

### AWS Graviton (ARM)

```bash
# Graviton 3 (SVE 256-bit) - best performance
# Graviton 4 (SVE 128-bit) - NEON fallback recommended
RUSTFLAGS="-C target-cpu=native" cargo bench -p lattice-embed
```

## Criterion Reports

After running benchmarks, HTML reports are generated at:

```text
target/criterion/report/index.html
```

## References

- [SimSIMD](https://github.com/ashvardanian/simsimd) - Industry-standard SIMD library
- [Criterion.rs](https://bheisler.github.io/criterion.rs/book/) - Rust benchmarking
  framework
- [Faiss](https://github.com/facebookresearch/faiss) - Meta's vector similarity library
