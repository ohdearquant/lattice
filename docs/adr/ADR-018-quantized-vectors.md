# ADR-018: Quantized Vector Tiers

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-embed

## Context

Storing and computing distances over millions of f32 embeddings at full precision is expensive:
a 384-dim BGE-small vector is 1536 bytes; 1M such vectors occupy 1.5 GB. Semantic search does
not require full f32 fidelity — approximate distances within a few percent of the exact value
are sufficient for top-K retrieval candidates that are later re-ranked.

The system must support multiple precision levels to allow trading memory/speed for accuracy.
The selection criterion for a stored vector should be data-dependent (access recency) rather
than requiring per-query configuration.

## Decision

Four quantization tiers are implemented, each with dedicated SIMD-accelerated distance kernels.
The tiers and their distance methods are unified under `QuantizedData` and `QuantizationTier` enums
in `src/simd/tier.rs`.

### Key Design Choices

**Four tiers with access-age heuristic**

| Tier   | Type              | Bytes/dim | Compression | Distance                | Age threshold |
| ------ | ----------------- | --------- | ----------- | ----------------------- | ------------- |
| Full   | f32               | 4.0       | 1x          | Exact cosine            | < 1 hour      |
| Int8   | i8, symmetric     | 1.0       | 4x          | SIMD i8 dot + scale     | < 1 day       |
| Int4   | u4, packed nibble | 0.5       | 8x          | Dequantize + accumulate | < 1 week      |
| Binary | 1-bit sign        | 0.125     | 32x         | Hamming distance        | ≥ 1 week      |

`QuantizationTier::from_age_seconds(age)` implements this heuristic with hard thresholds
at `HOUR=3600`, `DAY=86400`, `WEEK=604800`.

**INT8 symmetric quantization with VNNI invariant**

`QuantizedVector` uses symmetric quantization: maps `[-max_abs, max_abs]` to `[-127, 127]`.
The range is deliberately clamped to `[-127, 127]` rather than `[-128, 127]` because AVX-512 VNNI's
`_mm512_dpbusd_epi32` uses `vpabsb` internally, and `abs(-128)` saturates to `127` in signed 8-bit,
producing incorrect dot product results. This invariant is enforced in `from_f32` via `clamp(-127.0, 127.0)`
and validated at runtime in `dot_product_i8` with a hard `assert!` (not `debug_assert!`) — the `data`
field is `pub`, so callers can bypass the constructor.

Separate `dot_product_i8_trusted` / `cosine_similarity_i8_trusted` variants use `debug_assert!`
only and are used in the prepared-query hot path where the invariant is statically guaranteed.

**INT8 SIMD kernel dispatch hierarchy**

The dispatch happens once via `OnceLock<I8DotKernel>` (a function pointer):

1. aarch64 NEON: `dot_product_i8_neon_unrolled` — 4x unrolled `vmull/vpadal`, processes 64 i8s/iter
2. x86_64 + `avx512` feature: `dot_product_i8_avx512vnni_kernel` — `_mm512_dpbusd_epi32`, 256 i8s/iter, requires `--features avx512` (gated because it requires nightly intrinsics for AVX-512BW)
3. x86_64 AVX2: `dot_product_i8_avx2_unrolled` — `_mm256_maddubs_epi16`, 128 i8s/iter
4. Scalar fallback: i32 accumulation

The AVX-512F path for f32 (used in `dot_product.rs`) activates via runtime `is_x86_feature_detected!`
with no Cargo feature gate, because f32 AVX-512F intrinsics don't require nightly. The VNNI path
requires a compile-time feature gate because `_mm512_dpbusd_epi32` and `_mm512_cmplt_epi8_mask`
are extended ISA intrinsics that Rust gates behind `--features avx512`.

**INT4 packed nibble format**

`Int4Vector` uses unsigned symmetric quantization: `[-max_abs, max_abs]` → `[0, 15]` (scale = `15 / (2 * max_abs)`).
Two dimensions are packed per byte: high nibble = even index, low nibble = odd index.
Storage is `ceil(dims / 2)` bytes (8x vs f32). Distance computation dequantizes to signed values
before accumulation to handle the unsigned offset uniformly across targets.
NEON acceleration via `vld1_u8/vget_low_u8` processes nibble pairs. No x86 SIMD for Int4 currently.

**Binary sign-bit format with Hamming distance**

`BinaryVector` uses threshold 0.0 by default: positive values map to 1, negative to 0.
A custom threshold variant `from_f32_with_threshold(vector, threshold)` is provided.
Packing: bit 7 = first dimension of the byte, bit 6 = second, etc. (MSB = first).
Storage is `ceil(dims / 8)` bytes (32x vs f32).

Hamming distance uses `count_ones()` on XOR of packed bytes. NEON path uses `vcntq_u8`
(per-byte popcount) + `vpaddlq_*` widening, processing 16 bytes/iteration with u64 accumulators
(to avoid u8 overflow for large vectors). Approximate cosine similarity from Hamming:
`cos_approx = 1.0 - 2.0 * hamming / dims`.

**`PreparedQuery` for repeated distance computation**

`PreparedQuery` quantizes a query vector once and stores it at the target tier. `approximate_cosine_distance_prepared(query, stored)` then dispatches without re-quantizing the query per candidate. For HNSW search against a large candidate list, this eliminates O(N) `from_f32` calls, replacing them with one `from_f32` plus O(N) SIMD integer distance calls.

**`NormalizationHint::Unit` fast path**

When both query and stored vectors are unit-normalized (norm ≈ 1.0), cosine similarity equals
the dot product — the norm division can be skipped. `PreparedQueryWithMeta` carries a
`NormalizationHint` field; `approximate_cosine_distance_prepared_with_meta` activates the
`1.0 - dot_product` fast path for `Full` tier when both are `Unit`. This saves ~26% at 384d
by eliminating two norm-squared accumulations and two square roots.

### Alternatives Considered

| Alternative                | Pros                                             | Cons                                                            | Why Not                                                            |
| -------------------------- | ------------------------------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------ |
| Product Quantization (PQ)  | Higher recall at same bit budget                 | Complex codebook training; inference requires codebook lookup   | Training dependency; complexity not justified for current scale    |
| Single INT8 tier only      | Simpler; AVX-512 VNNI gives excellent throughput | No memory savings beyond 4x; binary needed for cold-data budget | Need 32x compression for cold data that rarely surfaces in queries |
| FAISS integration          | Battle-tested quantization + index               | C++ FFI; requires FAISS install; opaque memory model            | Pure Rust path is required; no C++ FFI allowed                     |
| ScaNN / HNSW with pure f32 | Simpler; no quantization error                   | 4-32x more memory; cache pressure increases                     | Memory budget and L3 fit are primary constraints                   |

## Consequences

### Positive

- A 384-dim BGE-small vector fits in: 1536 B (Full), 384 B (Int8), 192 B (Int4), 48 B (Binary). At 1M vectors, this is 1.5 GB / 366 MB / 183 MB / 46 MB respectively.
- SIMD parity tests (`test_i8_neon_scalar_parity`, `test_i8_avx2_scalar_parity`) verify that the SIMD and scalar paths agree within ±1.0 (due to floating-point accumulation order).
- The `QuantizedData` enum provides a single storage type for all tiers, enabling heterogeneous collections without an additional abstraction layer.

### Negative

- INT8 round-trip error is bounded by `max_abs / 254` per element. For a unit-norm 384-dim vector, element-wise absolute error ≤ 0.004; cosine similarity error ≤ ~0.5%. This is acceptable for HNSW candidate pre-filtering.
- Binary cosine approximation is rough (within ~0.35 of f32 cosine for random 384-dim vectors per the test). It should only be used for coarse pre-filtering, not for final ranking.
- The `avx512` Cargo feature must be explicitly enabled to activate the AVX-512 VNNI path. Omitting it falls back to AVX2, which is still ~3x faster than scalar.
- `promote()` / `demote()` between tiers loses information — Int4 promoted to Int8 fills new bits from the dequantized approximation, not the original f32 values. This is documented but could surprise callers.

### Risks

- The `-128` invariant on `QuantizedVector.data` is enforced in release builds by the public `dot_product_i8` function. The `data` field is `pub` (marked `Unstable`) — direct writes can bypass the invariant and cause silently incorrect VNNI results without panicking in release mode. The `trusted` variants skip the O(N) scan in release; callers must guarantee construction via `from_f32`.
- The `from_age_seconds` age thresholds (HOUR/DAY/WEEK) are not configurable at runtime. Workloads with different access patterns (e.g., archival data always queried with long gaps) will get suboptimal tier assignments.

## References

- [`crates/embed/src/simd/quantized.rs`](/Users/lion/projects/lattice/crates/embed/src/simd/quantized.rs) — INT8 implementation, NEON/AVX2/VNNI kernels
- [`crates/embed/src/simd/int4.rs`](/Users/lion/projects/lattice/crates/embed/src/simd/int4.rs) — INT4 nibble packing
- [`crates/embed/src/simd/binary.rs`](/Users/lion/projects/lattice/crates/embed/src/simd/binary.rs) — binary quantization, Hamming distance
- [`crates/embed/src/simd/tier.rs`](/Users/lion/projects/lattice/crates/embed/src/simd/tier.rs) — `QuantizationTier`, `QuantizedData`, `PreparedQuery`, distance dispatch
