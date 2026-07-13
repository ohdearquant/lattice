# SIMD kernels and quantized-vector paths

`lattice-embed` exposes vector operations through `lattice_embed::simd` and
selects the best supported implementation for the running target. The public
operations cover float dot products, cosine similarity, Euclidean distance,
in-place normalization, and approximate operations over INT8, INT4, and binary
vectors. Scalar implementations are always available as the fallback.

The SIMD-facing API is marked unstable. Use the higher-level `utils` wrappers
where a stable API is required; use this module when the lower-level return
semantics, prepared-query paths, or allocation behaviour matter to a search
implementation.

## Dispatch model

Feature detection is performed once per process and cached in `SimdConfig`.
Float dot-product and cosine dispatchers, batch-four dot-product dispatch, and
INT8 dot-product dispatch also cache their chosen function pointers. This keeps
feature checks and `OnceLock` initialization out of inner batch loops. Distance
and normalization reuse the cached configuration when they choose a kernel.

| Operation family | x86_64 priority | aarch64 priority | wasm32 | fallback |
| --- | --- | --- | --- | --- |
| Float dot, cosine, squared L2, normalize | AVX-512F, then AVX2 + FMA | NEON | SIMD128 when compiled in | scalar |
| Batch-four float dot | AVX2 + FMA | NEON | none | scalar |
| INT8 dot | AVX-512 VNNI + BW when the `avx512` feature is built, then AVX2 | NEON only with FEAT_DotProd | none | scalar |
| Binary Hamming | none | NEON `vcnt` | none | scalar |
| INT4 dot | none | NEON | none | scalar |

On `wasm32`, SIMD128 is a build property rather than a runtime capability. A
module compiled with `-C target-feature=+simd128` contains the SIMD kernels;
one built without it does not. `SimdConfig::simd128_enabled()` mirrors that
compile-time setting, so constructing a different `SimdConfig` cannot disable
SIMD inside an already-SIMD128-enabled artifact.

The dispatch order is deliberately per operation. For example, the float
batch-four kernel is implemented for AVX2 and NEON, while the single-vector
float kernel can use AVX-512F. INT8 on Arm additionally requires FEAT_DotProd:
baseline AArch64 NEON alone is not enough to execute `SDOT`.

## Float reductions

### Dot product

`dot_product(a, b)` computes `Σ a[i] * b[i]` for equal-length `f32` slices.
It returns `0.0` for a length mismatch. For unit-normalized vectors, this is
cosine similarity, and squared L2 distance is `2 * (1 - dot)`. That makes the
operation useful for inner-product and cosine-oriented ANN search.

The single-vector implementations load vector chunks, multiply them, keep
several partial sums, reduce those partial sums horizontally, and finish with a
vector tail followed by a scalar tail. AVX-512F uses 16-float registers and
four-way unrolling; AVX2 uses 8-float registers and eight independent
accumulators; NEON and wasm SIMD128 use 4-float registers and four
accumulators. AVX2 has a 384-dimension specialization: 384 is exactly 48
AVX2 registers, so its six 64-element chunks need no remainder path.

The batch-four API computes one query against four candidates at once:

```text
(q, c0, c1, c2, c3) -> [dot(q, c0), dot(q, c1), dot(q, c2), dot(q, c3)]
```

It returns four zeros if any candidate length differs from the query length.
The AVX2 and NEON implementations reuse each query load across all four
candidates, with two accumulators per candidate. `batch_dot_product` uses this
kernel only for consecutive groups of four whose query slices have the same
pointer and length; mixed groups and a final short group use the ordinary
per-pair kernel. This pointer check avoids treating merely equal-valued queries
as the same borrowed query.

### Cosine similarity

Cosine similarity is computed as:

```text
dot(a, b) / (sqrt(dot(a, a)) * sqrt(dot(b, b)))
```

The SIMD kernels calculate the dot product and both squared norms in one pass.
They maintain independent accumulators for all three reductions, combine them,
add the remainder, take two square roots, and return `0.0` if either norm is
zero. `cosine_similarity` returns `0.0` for empty input or a length mismatch.

`cosine_similarity_fused` makes the one-pass property explicit. The ordinary
SIMD cosine dispatcher is already fused, while the scalar reference named
`cosine_similarity_scalar` performs separate reductions. For normalized input,
prefer the cheaper dot-product operation instead of either cosine function.

`batch_cosine_similarity` resolves the cosine kernel once before iterating.
It deliberately does not scan each pair to discover unit normalization: an
`O(pair_count * dimensions)` pre-scan would cost as much as the computation.
Callers that know their vectors are normalized should use batch dot products or
the prepared-query APIs with explicit normalization metadata.

For one query and many candidates, `batch_cosine_one_vs_many` calculates the
query norm once. Each candidate still requires a dot product and its own norm;
dimension-mismatched candidates contribute `0.0` in their original order.

### Euclidean distance

Squared Euclidean distance is reduced in the same chunk/tail shape as a dot
product, using `Σ (a[i] - b[i])²`. `euclidean_distance` is the square root of
that result. Both APIs return `f32::MAX` for a length mismatch.

When only ranking is required, use `squared_euclidean_distance`. Squaring is
monotonic for non-negative distances, so it preserves the ordering of true L2
distance while avoiding a square root per graph comparison. Apply `.sqrt()` at
the output boundary only when callers need the actual distance.

### Normalization

`normalize(&mut vector)` performs two passes: it first reduces the squared L2
norm, then multiplies every element by its reciprocal. A zero-norm vector is
left unchanged. A vector that produces a NaN norm is also left unchanged so the
SIMD and scalar paths do not turn its contents into NaNs through scaling.

AVX-512F, AVX2, and wasm SIMD128 calculate the inverse norm from a scalar
square root. The NEON implementation uses `vrsqrteq_f32` followed by two
Newton-Raphson refinements, then falls back to the scalar reciprocal-square-root
calculation if the estimate is non-finite for a positive subnormal squared norm.
That fallback keeps the Arm result compatible with the scalar and x86 paths.

## Floating-point behaviour and kernel boundaries

SIMD reductions are mathematically equivalent to their scalar counterparts but
do not promise bit-identical output. Lanes and independent accumulators change
the summation order; FMA-capable targets may also round differently from a
separate multiply and add. Near-tied squared-L2 comparisons therefore must not
depend on reproducing an exact scalar ordering. The supported invariant is that
squared L2 and L2 have the same mathematical ordering, not that every backend
emits the same bit pattern.

The low-level kernel comments carry the exact safety requirements for lane
widths, unaligned loads, tails, target features, and numeric tolerances. In
particular, the kernels use unaligned load/store intrinsics and calculate chunk
counts with floor division before doing pointer arithmetic; the final tail is
handled without reading beyond a slice. Those source-local requirements are
part of the safety boundary for changes to the kernels.

## Quantization tiers and storage policy

`QuantizationTier` selects a storage/accuracy trade-off. `storage_bytes(dims)`
uses ceiling division for packed formats, so it is authoritative for odd
dimensions.

| Tier | Representation | Bytes per dimension | Compression vs. `f32` | Typical role |
| --- | --- | ---: | ---: | --- |
| `Full` | `Vec<f32>` | 4 | 1x | Hot data and exact search |
| `Int8` | signed byte plus parameters | 1 | 4x | Warm data and HNSW search |
| `Int4` | two unsigned nibbles per byte | 0.5 | 8x | Cool data and pre-filtering |
| `Binary` | one sign bit per dimension | 0.125 | 32x | Cold data and coarse filtering |

`QuantizationTier::from_age_seconds` is a simple recency heuristic, not a
measurement of vector quality: under one hour selects `Full`; one hour through
under one day selects `Int8`; one day through under one week selects `Int4`;
one week or older selects `Binary`. Applications can select tiers directly when
their retention or recall policy differs.

`QuantizedData` holds any tier behind one enum. Promoting or demoting it always
dequantizes to `f32` and quantizes into the destination tier. Promotion does not
recover information discarded by a previous lower-precision representation.

### Prepared queries and tier matching

Quantizing a query inside every candidate comparison repeats work and can
allocate. `PreparedQuery` quantizes once at the candidate tier, then is reused
against a homogeneous list. The prepared and stored tiers must match:

| Prepared query and stored data | Cosine-distance path | Dot-product path |
| --- | --- | --- |
| `Full` / `Full` | `1 - cosine_similarity` | float dot product |
| `Int8` / `Int8` | `1 - cosine_similarity_i8_trusted` | trusted INT8 dot product |
| `Int4` / `Int4` | INT4 cosine distance | INT4 dequantized dot product |
| `Binary` / `Binary` | Hamming-derived approximation | no meaningful prepared dot product |

The prepared distance and dot-product APIs return `EmbedError::TierMismatch`
for a mixed tier. A prepared binary dot product returns `EmbedError::Internal`;
binary vectors provide a cosine-distance approximation instead. The legacy
`try_approximate_*_prepared` names are aliases of the corresponding non-`try_`
functions and return the same `Result`.

The non-prepared `approximate_cosine_distance` takes an `f32` query and
quantizes it on each call for INT8, INT4, or binary stored data. It is convenient
but not the hot-loop choice. `approximate_dot_product` follows the same pattern;
for binary data it dequantizes to signed float values before using the float dot
product.

`approximate_cosine_distance` requires its float query to have the same
dimensionality as the stored value. This is a caller precondition, enforced with
a debug assertion rather than an error return; construct queries from the same
embedding model and tiered index schema as the stored vectors.

Batch prepared APIs exist both as allocating and buffer-reusing variants. The
`*_into` forms clear the caller buffer first and leave it cleared on an error,
so they never publish a partial result set. Tier-specific INT8 and INT4 batch
APIs similarly avoid per-candidate query quantization.

`PreparedQueryWithMeta` adds a `NormalizationHint`. With `Unit` hints for both
the query and stored `Full` vector, the metadata-aware cosine-distance function
uses `1 - clamp(dot(query, stored), -1, 1)` and skips norm division. The hints
are caller assertions; callers that need to establish the property can use
`is_unit_norm`, whose squared-norm tolerance is `1e-4`. This shortcut applies
only to the `Full` tier; all other combinations use the normal prepared
distance dispatch.

## INT8 vectors

### Encoding and error model

INT8 quantization is symmetric. The parameter builder examines only finite
input elements, finds `max_abs = max(abs(min), abs(max))`, and chooses:

```text
scale = 127 / max_abs, if max_abs > 1e-10
scale = 1,             otherwise
q[i] = clamp(round(v[i] * scale), -127, 127)
```

Non-finite input values quantize as zero. The stored L2 norm is calculated from
the finite source values, not from the integers. Dequantization is `q / scale`;
an invalid or zero stored scale falls back to one. The format has a zero point of
zero even though `QuantizationParams` retains the field for its current public
layout.

Mapping `[-max_abs, max_abs]` to `[-127, 127]` gives a step size of
`max_abs / 127` and a maximum round-trip error of half a step,
`max_abs / 254`. The error and the resulting approximate cosine similarity must
be acceptable for the tier selected by the caller; use `Full` or `Int8` instead
of lower-precision tiers when recall needs more fidelity.

### Dot product and dispatch

The raw integer reduction is dequantized by dividing by the product of the two
scales. INT8 cosine similarity divides that result by the product of the stored
source norms. The normal public API accepts `QuantizedVector` values, while
`dot_product_i8_raw` takes slices and returns the unscaled integer-domain dot
product as `f32`; its caller owns scale handling and avoids constructing vector
wrappers.

Every INT8 SIMD input must be in `[-127, 127]`. `i8::MIN` is excluded because
the x86 sign-handling technique negates an operand; two's-complement negation
of `-128` remains `-128` rather than producing `+128`, which yields a silent
wrong numerical result. Constructor-owned vectors clamp into the valid range.
The raw-slice entry point checks this only with `debug_assert!`, so external
quantizers must clamp `-128` to `-127` before release-mode hot-path calls.

On Arm, the INT8 kernel is dispatched only after runtime confirmation of
FEAT_DotProd and uses `SDOT` on 16-byte vectors. On x86, AVX-512 VNNI can use
`dpbusd` after rewriting signed-times-signed multiplication as absolute values
times a sign-adjusted operand; AVX2 uses the analogous `maddubs`/`madd`
sequence. Both implementations unroll their loops and issue a guarded software
prefetch for a future chunk. Scalar accumulation uses `i32` products and sums.

The trusted internal INT8 functions exist for prepared-query paths whose data
comes from the constructor. They skip a release-mode invariant scan while
retaining debug assertions; do not use that optimization for untrusted raw
bytes.

## INT4 vectors

### Packed format

INT4 uses unsigned symmetric quantization. For finite input values:

```text
max_abs = max(abs(v[i]))
scale   = 15 / (2 * max_abs), if max_abs > 1e-10; otherwise 1
q[i]    = clamp(round((v[i] + max_abs) * scale), 0, 15)
v[i]    = q[i] / scale - max_abs
```

Two values share one byte: the even-indexed dimension is bits 7 through 4 and
the odd-indexed dimension is bits 3 through 0. Storage is `ceil(dimensions / 2)`
bytes. For an odd-dimensional vector, only the final byte's high nibble is a
real dimension; the low nibble is padding and is ignored by dequantization and
dot-product accumulation. Non-finite source values are treated as zero, and
the L2 norm records the finite source values.

The 16 quantization levels have step size `2 * max_abs / 15`, so the maximum
per-element round-trip error is `max_abs / 15`. This is a storage-oriented
approximation: for correlated 384-dimensional vectors, the expected relative
dot-product error is on the order of the documented 15% bound rather than the
tighter error expected from INT8.

### Corrected dot product

Because the packed values are offset unsigned integers, a raw dot product alone
is not the float dot product. The implementation accumulates three integers:

```text
raw_dot = Σ qa[i] * qb[i]
sum_a   = Σ qa[i]
sum_b   = Σ qb[i]
```

It then applies the offset correction:

```text
raw_dot / (scale_a * scale_b)
    - max_abs_b * sum_a / scale_a
    - max_abs_a * sum_b / scale_b
    + dimensions * max_abs_a * max_abs_b
```

This is the expanded product of `(qa / scale_a - max_abs_a)` and
`(qb / scale_b - max_abs_b)`. It ensures scalar and NEON paths implement the
same dequantized result without allocating temporary float vectors.

The Arm kernel processes full packed bytes only, splits high and low nibbles,
and reduces the raw and marginal sums in parallel. It leaves both the byte tail
and an odd final high nibble to the scalar path so padding can never contribute
as data. Dimension mismatches, invalid scales, and too-short packed buffers
return `0.0`; malformed buffers dequantize to an empty vector.

## Binary vectors

Binary quantization stores a sign/threshold decision per dimension. For the
default threshold of zero, `v >= 0` maps to one and `v < 0` maps to zero.
`from_f32_with_threshold` generalizes that comparison. Non-finite values become
zero before comparison, so their output bit depends on the selected threshold.

Bits are packed most-significant first: dimension zero is bit 7 of byte zero,
dimension one is bit 6, and so on. The storage size is
`ceil(dimensions / 8)` bytes. Dequantization maps one to `+1.0` and zero to
`-1.0`; it is intentionally lossy and returns an empty vector if public fields
describe a buffer shorter than the required packed length.

Hamming distance is the population count of the XOR of two packed buffers. The
scalar implementation counts full 64-bit groups and remaining bytes; the NEON
implementation applies `vcnt` to 16-byte vectors and widens before summing.
The final partial byte needs special treatment: its valid dimensions occupy the
high `dimensions % 8` bits, so the low padding bits are masked out before their
popcount. This is necessary for odd dimension counts to produce the same result
as an unpacked sign comparison.

`hamming_distance_binary` returns `u32::MAX` for different dimensions or a
malformed packed buffer. A valid binary cosine approximation is:

```text
cosine_similarity_approx = 1 - 2 * hamming / dimensions
cosine_distance_approx   = 2 * hamming / dimensions
```

The cosine-distance method returns `0.0` for a zero-dimensional receiver.
Callers must reject a mismatch sentinel before treating the approximation as a
distance.

## Choosing an operation

Use the operation that matches the data and search stage:

| Situation | Recommended operation |
| --- | --- |
| Unit-normalized float embeddings | `dot_product` or `batch_dot_product` |
| Float embeddings with unknown norms | `cosine_similarity_fused` or `cosine_similarity` |
| ANN ranking where only L2 ordering matters | `squared_euclidean_distance` |
| One query against many same-tier compressed candidates | prepare once, then use a prepared batch API |
| Coarse candidate filter with maximum compression | binary Hamming/cosine approximation, followed by a higher-fidelity rerank |
| Reusing a caller-owned result buffer | a `*_into` prepared batch API |

Do not compare float SIMD output for bit equality with the scalar reference, do
not feed `i8::MIN` to an INT8 SIMD raw path, and do not count padding bits in
packed INT4 or binary vectors. Those are correctness requirements rather than
performance suggestions.

## Public API contracts

The low-level `simd` API is unstable in general, but its float operation
signatures and mismatch sentinels are consumed by the khive ANN indexes during
the 0.4.x line. `dot_product` returns `0.0` for a dimensional mismatch;
`cosine_similarity` returns `0.0` for a mismatch or empty input; and both L2
operations return `f32::MAX` for a mismatch. The stable ergonomic wrappers are
available under `lattice_embed::utils`.

For unit-normalized data, dot product is cosine similarity and
`squared_euclidean_distance` is `2 * (1 - dot)`. Squared L2 is the ANN hot
path because its monotonic relationship with L2 preserves mathematical ranking
without a square root. Floating-point reduction order may differ between SIMD
and scalar kernels, so near ties must not rely on an exact scalar bit pattern.

`dot_product_batch4` evaluates a query against four equally-sized candidates
and returns four zeroes when any candidate differs in length. The higher-level
batch routine uses that kernel only when a four-item group borrows the same
query slice (identical pointer and length); equal values in different slices do
not qualify for the reuse optimization.

## Kernel safety boundary

All unsafe float kernels are reached only after the dispatcher has selected an
available target feature. They use unaligned loads and stores, calculate vector
chunk counts by floor division before pointer arithmetic, and finish with safe
tails. These three properties are the memory-safety contract: changing a lane
width, unroll factor, or tail calculation must preserve them.

Wasm SIMD128 is selected at build time, not detected at runtime. Its loads are
alignment-independent by the WebAssembly specification. AVX and NEON paths may
use FMA or a different reduction tree, which explains numerical differences
from scalar execution without changing the operation's mathematical contract.

The NEON normalization kernel estimates reciprocal square root with
`vrsqrteq_f32` and two Newton--Raphson refinements. If a positive subnormal
norm produces a non-finite estimate it uses the scalar reciprocal square root
instead, preventing an otherwise valid vector from becoming non-finite.

## Raw INT8 input invariant

`dot_product_i8_raw` is an intentionally unchecked hot-path interface except
for debug assertions. Every input byte must be in `[-127, 127]`; `-128` is not
valid. The x86 signed-dot transformations negate bytes to construct a
sign-adjusted operand, and two's-complement negation leaves `-128` unchanged,
silently producing an incorrect result. This is numerical corruption rather
than memory unsafety. Values created by `QuantizedVector::from_f32` are already
clamped; external quantizers must map `-128` to `-127` before calling the raw
entry point.

Prepared-query internals use trusted INT8 functions only for those
constructor-owned vectors. They retain debug checks but deliberately avoid an
O(n) release scan on every candidate comparison.
