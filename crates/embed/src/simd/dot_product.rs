//! SIMD float dot-product kernels, including a one-query/four-candidate path.
//!
//! The public operations return zero for dimensional mismatches.
//!
//! See docs/simd.md for dispatch and batch-kernel design.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::*;

use std::sync::OnceLock;

use super::simd_config;

/// SIMD kernel function pointer type for f32 dot product.
pub type DotKernel = fn(&[f32], &[f32]) -> f32;

static DOT_PRODUCT_KERNEL: OnceLock<DotKernel> = OnceLock::new();

/// Resolve the best available f32 dot-product kernel once and return it.
///
/// Used by `batch_dot_product` to hoist SIMD dispatch out of batch loops.
#[inline]
pub fn resolved_dot_product_kernel() -> DotKernel {
    *DOT_PRODUCT_KERNEL.get_or_init(resolve_dot_product_kernel)
}

fn resolve_dot_product_kernel() -> DotKernel {
    let config = simd_config();

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx512f_enabled {
            return dot_product_avx512_kernel;
        }
        if config.avx2_enabled && config.fma_enabled {
            return dot_product_avx2_kernel;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            return dot_product_neon_kernel;
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if config.simd128_enabled() {
            return dot_product_simd128_kernel;
        }
    }

    dot_product_scalar
}

// ---------------------------------------------------------------------------
// Batch-4 dot product kernel (query vs. 4 candidates simultaneously)
// ---------------------------------------------------------------------------

/// SIMD kernel type for batch-4 f32 dot product.
///
/// Signature: (query, c0, c1, c2, c3) → [dot(q,c0), dot(q,c1), dot(q,c2), dot(q,c3)].
/// All slices must have equal length (enforced by `dot_product_batch4`).
pub type DotBatch4Kernel = fn(&[f32], &[f32], &[f32], &[f32], &[f32]) -> [f32; 4];

static DOT_PRODUCT_BATCH4_KERNEL: OnceLock<DotBatch4Kernel> = OnceLock::new();

/// Resolve the best available batch-4 f32 dot-product kernel once and return it.
///
/// Used by HNSW expansion loop and `batch_dot_product` for same-query chunks.
#[inline]
pub fn resolved_dot_product_batch4_kernel() -> DotBatch4Kernel {
    *DOT_PRODUCT_BATCH4_KERNEL.get_or_init(resolve_dot_product_batch4_kernel)
}

/// Compute dot product of one query against 4 candidates simultaneously.
///
/// Returns `[0.0; 4]` if any candidate length differs from the query length.
#[inline]
pub fn dot_product_batch4(
    query: &[f32],
    c0: &[f32],
    c1: &[f32],
    c2: &[f32],
    c3: &[f32],
) -> [f32; 4] {
    if query.len() != c0.len()
        || query.len() != c1.len()
        || query.len() != c2.len()
        || query.len() != c3.len()
    {
        debug_assert!(
            false,
            "dot_product_batch4: dimension mismatch (query={}, c0={}, c1={}, c2={}, c3={})",
            query.len(),
            c0.len(),
            c1.len(),
            c2.len(),
            c3.len()
        );
        return [0.0; 4];
    }
    resolved_dot_product_batch4_kernel()(query, c0, c1, c2, c3)
}

fn resolve_dot_product_batch4_kernel() -> DotBatch4Kernel {
    let config = simd_config();

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled && config.fma_enabled {
            return dot_product_batch4_avx2_kernel;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            return dot_product_batch4_neon_kernel;
        }
    }

    dot_product_batch4_scalar
}

/// Scalar batch-4 dot product fallback. Used when no SIMD is available.
fn dot_product_batch4_scalar(
    q: &[f32],
    c0: &[f32],
    c1: &[f32],
    c2: &[f32],
    c3: &[f32],
) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for i in 0..q.len() {
        let qi = q[i];
        out[0] += qi * c0[i];
        out[1] += qi * c1[i];
        out[2] += qi * c2[i];
        out[3] += qi * c3[i];
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_product_avx512_kernel(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only stored in DOT_PRODUCT_KERNEL when avx512f was detected at init time.
    unsafe { dot_product_avx512_unrolled(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_product_avx2_kernel(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only stored in DOT_PRODUCT_KERNEL when avx2+fma were detected at init time.
    if a.len() == 384 {
        unsafe { dot_product_384_avx2(a, b) }
    } else {
        unsafe { dot_product_avx2_8acc(a, b) }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_product_neon_kernel(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only stored in DOT_PRODUCT_KERNEL when neon was detected at init time (always true on aarch64).
    unsafe { dot_product_neon_unrolled(a, b) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn dot_product_simd128_kernel(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only stored in DOT_PRODUCT_KERNEL when compiled with the wasm32
    // `simd128` target feature (the `#[cfg(target_feature = "simd128")]` gate
    // above is compile-time, not runtime -- see `SimdConfig::simd128_enabled`).
    unsafe { dot_product_simd128_unrolled(a, b) }
}

/// Computes the float dot product, returning `0.0` for a dimensional mismatch.
///
/// See [`docs/simd.md`] (§Public API contracts) for ANN and normalization semantics.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    // Runtime length check to prevent UB in release builds
    if a.len() != b.len() {
        return 0.0;
    }
    debug_assert_eq!(a.len(), b.len());
    resolved_dot_product_kernel()(a, b)
}

/// Scalar dot product implementation.
#[inline]
pub(crate) fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Computes a four-accumulator dot product with AVX-512F.
///
/// # Safety
/// Caller must provide AVX-512F and equal slices; chunked unaligned loads stay in bounds.
/// See [`docs/simd.md`] (§Kernel safety boundary) for the shared kernel invariant.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_avx512_unrolled(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 16;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL; // 64 floats per iteration

    let n = a.len();
    debug_assert_eq!(n, b.len());
    let chunks = n / CHUNK_SIZE;

    // 4 independent accumulators to break dependency chains
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let a0 = _mm512_loadu_ps(a.as_ptr().add(base));
        let b0 = _mm512_loadu_ps(b.as_ptr().add(base));
        sum0 = _mm512_fmadd_ps(a0, b0, sum0);

        let a1 = _mm512_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = _mm512_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH));
        sum1 = _mm512_fmadd_ps(a1, b1, sum1);

        let a2 = _mm512_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = _mm512_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 2));
        sum2 = _mm512_fmadd_ps(a2, b2, sum2);

        let a3 = _mm512_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = _mm512_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 3));
        sum3 = _mm512_fmadd_ps(a3, b3, sum3);
    }

    // Combine accumulators (dependencies are introduced only once at the end)
    let sum01 = _mm512_add_ps(sum0, sum1);
    let sum23 = _mm512_add_ps(sum2, sum3);
    let sum_vec = _mm512_add_ps(sum01, sum23);

    let main_sum = horizontal_sum_avx512(sum_vec);

    // Handle remainder with single-register loop
    let main_processed = chunks * CHUNK_SIZE;
    let remaining = n - main_processed;
    let remaining_chunks = remaining / SIMD_WIDTH;

    let mut remainder_sum = _mm512_setzero_ps();
    for i in 0..remaining_chunks {
        let offset = main_processed + i * SIMD_WIDTH;
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(offset));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(offset));
        remainder_sum = _mm512_fmadd_ps(a_vec, b_vec, remainder_sum);
    }

    let mut total = main_sum + horizontal_sum_avx512(remainder_sum);

    // Final scalar remainder
    let scalar_start = main_processed + remaining_chunks * SIMD_WIDTH;
    for i in scalar_start..n {
        total += a[i] * b[i];
    }

    total
}

/// Horizontal sum of AVX-512 register (16 floats -> 1 float).
///
/// # Safety
///
/// Caller must ensure CPU supports AVX-512F (verified via `target_feature` gate).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn horizontal_sum_avx512(v: __m512) -> f32 {
    _mm512_reduce_add_ps(v)
}

/// Computes an eight-accumulator dot product with AVX2 and FMA.
///
/// # Safety
/// Caller must provide AVX2/FMA and equal slices; chunked unaligned loads stay in bounds.
/// See [`docs/simd.md`] (§Kernel safety boundary) for the shared kernel invariant.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_avx2_8acc(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 8;
    const UNROLL: usize = 8;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL; // 64 floats per iteration
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let chunks = n / CHUNK_SIZE;

    // 8 independent accumulators to break dependency chains
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    let mut sum4 = _mm256_setzero_ps();
    let mut sum5 = _mm256_setzero_ps();
    let mut sum6 = _mm256_setzero_ps();
    let mut sum7 = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let a0 = _mm256_loadu_ps(a.as_ptr().add(base));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(base));
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);

        let a1 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH));
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);

        let a2 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 2));
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);

        let a3 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 3));
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);

        let a4 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 4));
        let b4 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 4));
        sum4 = _mm256_fmadd_ps(a4, b4, sum4);

        let a5 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 5));
        let b5 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 5));
        sum5 = _mm256_fmadd_ps(a5, b5, sum5);

        let a6 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 6));
        let b6 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 6));
        sum6 = _mm256_fmadd_ps(a6, b6, sum6);

        let a7 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 7));
        let b7 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 7));
        sum7 = _mm256_fmadd_ps(a7, b7, sum7);
    }

    // Combine accumulators pairwise to reduce dependency chain depth
    let sum01 = _mm256_add_ps(sum0, sum1);
    let sum23 = _mm256_add_ps(sum2, sum3);
    let sum45 = _mm256_add_ps(sum4, sum5);
    let sum67 = _mm256_add_ps(sum6, sum7);
    let sum0123 = _mm256_add_ps(sum01, sum23);
    let sum4567 = _mm256_add_ps(sum45, sum67);
    let sum_vec = _mm256_add_ps(sum0123, sum4567);

    let sum = horizontal_sum_avx2(sum_vec);

    // Handle remainder with single-vector loop
    let main_processed = chunks * CHUNK_SIZE;
    let remaining = n - main_processed;
    let remaining_chunks = remaining / SIMD_WIDTH;

    let mut remainder_sum = _mm256_setzero_ps();
    for i in 0..remaining_chunks {
        let offset = main_processed + i * SIMD_WIDTH;
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(offset));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(offset));
        remainder_sum = _mm256_fmadd_ps(a_vec, b_vec, remainder_sum);
    }

    let mut total = sum + horizontal_sum_avx2(remainder_sum);

    // Final scalar remainder
    let scalar_start = main_processed + remaining_chunks * SIMD_WIDTH;
    for i in scalar_start..n {
        total += a[i] * b[i];
    }

    total
}

/// Computes the fixed-size 384-dimension AVX2/FMA dot product.
///
/// # Safety
/// Caller must provide AVX2/FMA and two 384-element slices.
/// See [`docs/simd.md`] (§Dot product) for the specialization rationale.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_384_avx2(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 8;
    // 384 / 8 = 48 iterations, processed as 6 groups of 8 for accumulator reuse
    const UNROLL: usize = 8;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL; // 64 floats per iteration
    const CHUNKS: usize = 384 / CHUNK_SIZE; // 6 full chunks
    const TAIL_ITERS: usize = (384 - CHUNKS * CHUNK_SIZE) / SIMD_WIDTH; // 0 remainder

    debug_assert_eq!(a.len(), 384);
    debug_assert_eq!(b.len(), 384);
    debug_assert_eq!(CHUNKS * CHUNK_SIZE + TAIL_ITERS * SIMD_WIDTH, 384);

    // 8 independent accumulators
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    let mut sum4 = _mm256_setzero_ps();
    let mut sum5 = _mm256_setzero_ps();
    let mut sum6 = _mm256_setzero_ps();
    let mut sum7 = _mm256_setzero_ps();

    // 6 full chunks of 64 elements = 384 elements total
    for i in 0..CHUNKS {
        let base = i * CHUNK_SIZE;

        let a0 = _mm256_loadu_ps(a.as_ptr().add(base));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(base));
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);

        let a1 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH));
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);

        let a2 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 2));
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);

        let a3 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 3));
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);

        let a4 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 4));
        let b4 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 4));
        sum4 = _mm256_fmadd_ps(a4, b4, sum4);

        let a5 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 5));
        let b5 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 5));
        sum5 = _mm256_fmadd_ps(a5, b5, sum5);

        let a6 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 6));
        let b6 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 6));
        sum6 = _mm256_fmadd_ps(a6, b6, sum6);

        let a7 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 7));
        let b7 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 7));
        sum7 = _mm256_fmadd_ps(a7, b7, sum7);
    }

    // Combine accumulators pairwise
    let sum01 = _mm256_add_ps(sum0, sum1);
    let sum23 = _mm256_add_ps(sum2, sum3);
    let sum45 = _mm256_add_ps(sum4, sum5);
    let sum67 = _mm256_add_ps(sum6, sum7);
    let sum0123 = _mm256_add_ps(sum01, sum23);
    let sum4567 = _mm256_add_ps(sum45, sum67);
    let sum_vec = _mm256_add_ps(sum0123, sum4567);

    horizontal_sum_avx2(sum_vec)
}

/// Horizontal sum of AVX2 register (8 floats -> 1 float).
///
/// # Safety
///
/// Caller must ensure CPU supports AVX2 (verified via `target_feature` gate).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Sum high and low 128-bit lanes
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(high, low);

    // Horizontal add within 128-bit
    let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
    let sums = _mm_add_ps(sum128, shuf); // [0+1,1+1,2+3,3+3]
    let shuf2 = _mm_movehl_ps(sums, sums); // [2+3,3+3,2+3,3+3]
    let sums2 = _mm_add_ss(sums, shuf2); // [0+1+2+3,...]

    _mm_cvtss_f32(sums2)
}

/// AVX2 batch-4 dot product kernel wrapper (routes to 384-specialized or general path).
///
/// # Safety
///
/// Only stored in `DOT_PRODUCT_BATCH4_KERNEL` when AVX2+FMA was detected at init time.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_product_batch4_avx2_kernel(
    q: &[f32],
    c0: &[f32],
    c1: &[f32],
    c2: &[f32],
    c3: &[f32],
) -> [f32; 4] {
    if q.len() == 384 {
        unsafe { dot_product_384_batch4_avx2(q, c0, c1, c2, c3) }
    } else {
        unsafe { dot_product_batch4_avx2(q, c0, c1, c2, c3) }
    }
}

/// Computes one 384-dimension query against four candidates with AVX2/FMA.
///
/// # Safety
/// Caller must provide AVX2/FMA and five 384-element slices.
/// See [`docs/simd.md`] (§Dot product) for the batch-kernel layout.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_384_batch4_avx2(
    q: &[f32],
    c0: &[f32],
    c1: &[f32],
    c2: &[f32],
    c3: &[f32],
) -> [f32; 4] {
    const W: usize = 8; // floats per AVX2 register
    const CHUNK: usize = W * 2; // 16 floats per loop (2 query loads reused across 4 candidates)
    const CHUNKS: usize = 384 / CHUNK; // 24 chunks, zero remainder

    debug_assert_eq!(q.len(), 384);

    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();
    let mut acc20 = _mm256_setzero_ps();
    let mut acc21 = _mm256_setzero_ps();
    let mut acc30 = _mm256_setzero_ps();
    let mut acc31 = _mm256_setzero_ps();

    for i in 0..CHUNKS {
        let base = i * CHUNK;
        let q0 = _mm256_loadu_ps(q.as_ptr().add(base));
        let q1 = _mm256_loadu_ps(q.as_ptr().add(base + W));

        acc00 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c0.as_ptr().add(base)), acc00);
        acc01 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c0.as_ptr().add(base + W)), acc01);
        acc10 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c1.as_ptr().add(base)), acc10);
        acc11 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c1.as_ptr().add(base + W)), acc11);
        acc20 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c2.as_ptr().add(base)), acc20);
        acc21 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c2.as_ptr().add(base + W)), acc21);
        acc30 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c3.as_ptr().add(base)), acc30);
        acc31 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c3.as_ptr().add(base + W)), acc31);
    }

    [
        horizontal_sum_avx2(_mm256_add_ps(acc00, acc01)),
        horizontal_sum_avx2(_mm256_add_ps(acc10, acc11)),
        horizontal_sum_avx2(_mm256_add_ps(acc20, acc21)),
        horizontal_sum_avx2(_mm256_add_ps(acc30, acc31)),
    ]
}

/// Computes one query against four candidates with AVX2/FMA.
///
/// # Safety
/// Caller must provide AVX2/FMA and five equal-length slices; bounds are chunked.
/// See [`docs/simd.md`] (§Dot product) for the reuse and accumulator strategy.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_batch4_avx2(
    q: &[f32],
    c0: &[f32],
    c1: &[f32],
    c2: &[f32],
    c3: &[f32],
) -> [f32; 4] {
    const W: usize = 8;
    const CHUNK: usize = W * 2; // 16 floats per loop

    let n = q.len();
    let chunks = n / CHUNK;

    let mut acc00 = _mm256_setzero_ps();
    let mut acc01 = _mm256_setzero_ps();
    let mut acc10 = _mm256_setzero_ps();
    let mut acc11 = _mm256_setzero_ps();
    let mut acc20 = _mm256_setzero_ps();
    let mut acc21 = _mm256_setzero_ps();
    let mut acc30 = _mm256_setzero_ps();
    let mut acc31 = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * CHUNK;
        let q0 = _mm256_loadu_ps(q.as_ptr().add(base));
        let q1 = _mm256_loadu_ps(q.as_ptr().add(base + W));

        acc00 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c0.as_ptr().add(base)), acc00);
        acc01 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c0.as_ptr().add(base + W)), acc01);
        acc10 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c1.as_ptr().add(base)), acc10);
        acc11 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c1.as_ptr().add(base + W)), acc11);
        acc20 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c2.as_ptr().add(base)), acc20);
        acc21 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c2.as_ptr().add(base + W)), acc21);
        acc30 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(c3.as_ptr().add(base)), acc30);
        acc31 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(c3.as_ptr().add(base + W)), acc31);
    }

    let mut out = [
        horizontal_sum_avx2(_mm256_add_ps(acc00, acc01)),
        horizontal_sum_avx2(_mm256_add_ps(acc10, acc11)),
        horizontal_sum_avx2(_mm256_add_ps(acc20, acc21)),
        horizontal_sum_avx2(_mm256_add_ps(acc30, acc31)),
    ];

    let scalar_start = chunks * CHUNK;
    for i in scalar_start..n {
        let qi = q[i];
        out[0] += qi * c0[i];
        out[1] += qi * c1[i];
        out[2] += qi * c2[i];
        out[3] += qi * c3[i];
    }

    out
}

/// Computes a four-accumulator dot product with NEON.
///
/// # Safety
/// Caller must run on aarch64 with equal slices; chunked loads stay in bounds.
/// See [`docs/simd.md`] (§Kernel safety boundary) for the shared kernel invariant.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_product_neon_unrolled(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 4;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL; // 16 floats per iteration
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let chunks = n / CHUNK_SIZE;

    // 4 independent accumulators
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let a0 = vld1q_f32(a.as_ptr().add(base));
        let b0 = vld1q_f32(b.as_ptr().add(base));
        sum0 = vfmaq_f32(sum0, a0, b0);

        let a1 = vld1q_f32(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = vld1q_f32(b.as_ptr().add(base + SIMD_WIDTH));
        sum1 = vfmaq_f32(sum1, a1, b1);

        let a2 = vld1q_f32(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = vld1q_f32(b.as_ptr().add(base + SIMD_WIDTH * 2));
        sum2 = vfmaq_f32(sum2, a2, b2);

        let a3 = vld1q_f32(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = vld1q_f32(b.as_ptr().add(base + SIMD_WIDTH * 3));
        sum3 = vfmaq_f32(sum3, a3, b3);
    }

    // Combine accumulators
    let sum01 = vaddq_f32(sum0, sum1);
    let sum23 = vaddq_f32(sum2, sum3);
    let sum_vec = vaddq_f32(sum01, sum23);

    let mut sum = horizontal_sum_neon(sum_vec);

    // Handle remainder with single-vector loop
    let main_processed = chunks * CHUNK_SIZE;
    let remaining = n - main_processed;
    let remaining_chunks = remaining / SIMD_WIDTH;

    let mut remainder_sum = vdupq_n_f32(0.0);
    for i in 0..remaining_chunks {
        let offset = main_processed + i * SIMD_WIDTH;
        let a_vec = vld1q_f32(a.as_ptr().add(offset));
        let b_vec = vld1q_f32(b.as_ptr().add(offset));
        remainder_sum = vfmaq_f32(remainder_sum, a_vec, b_vec);
    }

    sum += horizontal_sum_neon(remainder_sum);

    // Final scalar remainder
    let scalar_start = main_processed + remaining_chunks * SIMD_WIDTH;
    for i in scalar_start..n {
        sum += a[i] * b[i];
    }

    sum
}

/// Horizontal sum of NEON register (4 floats -> 1 float).
///
/// # Safety
///
/// Caller must ensure running on aarch64 (NEON is mandatory on this arch).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) unsafe fn horizontal_sum_neon(v: float32x4_t) -> f32 {
    vaddvq_f32(v)
}

/// Computes a four-accumulator dot product with wasm32 SIMD128.
///
/// # Safety
/// This function requires compile-time SIMD128 and equal slices; bounds are chunked.
/// See [`docs/simd.md`] (§Kernel safety boundary) for wasm and reassociation semantics.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn dot_product_simd128_unrolled(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 4;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL; // 16 floats per iteration

    let n = a.len();
    debug_assert_eq!(n, b.len());
    let chunks = n / CHUNK_SIZE;

    // 4 independent accumulators to break dependency chains
    let mut sum0 = f32x4_splat(0.0);
    let mut sum1 = f32x4_splat(0.0);
    let mut sum2 = f32x4_splat(0.0);
    let mut sum3 = f32x4_splat(0.0);

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let a0 = v128_load(a.as_ptr().add(base) as *const v128);
        let b0 = v128_load(b.as_ptr().add(base) as *const v128);
        sum0 = f32x4_add(sum0, f32x4_mul(a0, b0));

        let a1 = v128_load(a.as_ptr().add(base + SIMD_WIDTH) as *const v128);
        let b1 = v128_load(b.as_ptr().add(base + SIMD_WIDTH) as *const v128);
        sum1 = f32x4_add(sum1, f32x4_mul(a1, b1));

        let a2 = v128_load(a.as_ptr().add(base + SIMD_WIDTH * 2) as *const v128);
        let b2 = v128_load(b.as_ptr().add(base + SIMD_WIDTH * 2) as *const v128);
        sum2 = f32x4_add(sum2, f32x4_mul(a2, b2));

        let a3 = v128_load(a.as_ptr().add(base + SIMD_WIDTH * 3) as *const v128);
        let b3 = v128_load(b.as_ptr().add(base + SIMD_WIDTH * 3) as *const v128);
        sum3 = f32x4_add(sum3, f32x4_mul(a3, b3));
    }

    // Combine accumulators (dependencies are introduced only once at the end)
    let sum01 = f32x4_add(sum0, sum1);
    let sum23 = f32x4_add(sum2, sum3);
    let sum_vec = f32x4_add(sum01, sum23);

    let mut total = horizontal_sum_simd128(sum_vec);

    // Handle remainder with single-vector loop
    let main_processed = chunks * CHUNK_SIZE;
    let remaining = n - main_processed;
    let remaining_chunks = remaining / SIMD_WIDTH;

    let mut remainder_sum = f32x4_splat(0.0);
    for i in 0..remaining_chunks {
        let offset = main_processed + i * SIMD_WIDTH;
        let a_vec = v128_load(a.as_ptr().add(offset) as *const v128);
        let b_vec = v128_load(b.as_ptr().add(offset) as *const v128);
        remainder_sum = f32x4_add(remainder_sum, f32x4_mul(a_vec, b_vec));
    }

    total += horizontal_sum_simd128(remainder_sum);

    // Final scalar remainder
    let scalar_start = main_processed + remaining_chunks * SIMD_WIDTH;
    for i in scalar_start..n {
        total += a[i] * b[i];
    }

    total
}

/// Horizontal sum of a wasm32 SIMD128 register (4 floats -> 1 float).
///
/// # Safety
///
/// Caller must ensure the crate was compiled with the wasm32 `simd128` target
/// feature (this function only exists under `#[cfg(target_feature =
/// "simd128")]`).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub(crate) unsafe fn horizontal_sum_simd128(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v)
        + f32x4_extract_lane::<1>(v)
        + f32x4_extract_lane::<2>(v)
        + f32x4_extract_lane::<3>(v)
}

/// NEON batch-4 dot product kernel wrapper.
///
/// # Safety
///
/// Only stored in `DOT_PRODUCT_BATCH4_KERNEL` when NEON is detected (always on aarch64).
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_product_batch4_neon_kernel(
    q: &[f32],
    c0: &[f32],
    c1: &[f32],
    c2: &[f32],
    c3: &[f32],
) -> [f32; 4] {
    // SAFETY: only stored when NEON detected, which is mandatory on aarch64.
    unsafe { dot_product_batch4_neon(q, c0, c1, c2, c3) }
}

/// Computes one query against four candidates with NEON.
///
/// # Safety
/// Caller must run on aarch64 with five equal-length slices; bounds are chunked.
/// See [`docs/simd.md`] (§Dot product) for the reuse and accumulator strategy.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_product_batch4_neon(
    q: &[f32],
    c0: &[f32],
    c1: &[f32],
    c2: &[f32],
    c3: &[f32],
) -> [f32; 4] {
    const W: usize = 4; // floats per NEON register
    const CHUNK: usize = W * 2; // 8 floats per loop (2 NEON loads from query)

    let n = q.len();
    let chunks = n / CHUNK;

    let mut acc00 = vdupq_n_f32(0.0);
    let mut acc01 = vdupq_n_f32(0.0);
    let mut acc10 = vdupq_n_f32(0.0);
    let mut acc11 = vdupq_n_f32(0.0);
    let mut acc20 = vdupq_n_f32(0.0);
    let mut acc21 = vdupq_n_f32(0.0);
    let mut acc30 = vdupq_n_f32(0.0);
    let mut acc31 = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let base = i * CHUNK;
        let q0 = vld1q_f32(q.as_ptr().add(base));
        let q1 = vld1q_f32(q.as_ptr().add(base + W));

        acc00 = vfmaq_f32(acc00, q0, vld1q_f32(c0.as_ptr().add(base)));
        acc01 = vfmaq_f32(acc01, q1, vld1q_f32(c0.as_ptr().add(base + W)));
        acc10 = vfmaq_f32(acc10, q0, vld1q_f32(c1.as_ptr().add(base)));
        acc11 = vfmaq_f32(acc11, q1, vld1q_f32(c1.as_ptr().add(base + W)));
        acc20 = vfmaq_f32(acc20, q0, vld1q_f32(c2.as_ptr().add(base)));
        acc21 = vfmaq_f32(acc21, q1, vld1q_f32(c2.as_ptr().add(base + W)));
        acc30 = vfmaq_f32(acc30, q0, vld1q_f32(c3.as_ptr().add(base)));
        acc31 = vfmaq_f32(acc31, q1, vld1q_f32(c3.as_ptr().add(base + W)));
    }

    let mut out = [
        vaddvq_f32(vaddq_f32(acc00, acc01)),
        vaddvq_f32(vaddq_f32(acc10, acc11)),
        vaddvq_f32(vaddq_f32(acc20, acc21)),
        vaddvq_f32(vaddq_f32(acc30, acc31)),
    ];

    let scalar_start = chunks * CHUNK;
    for i in scalar_start..n {
        let qi = q[i];
        out[0] += qi * c0[i];
        out[1] += qi * c1[i];
        out[2] += qi * c2[i];
        out[3] += qi * c3[i];
    }

    out
}

/// Returns true only when all 4 pairs in a chunk share the same query pointer and all lengths match.
#[inline]
fn same_query_batch4(chunk: &[(&[f32], &[f32])]) -> bool {
    debug_assert_eq!(chunk.len(), 4);
    let q_ptr = chunk[0].0.as_ptr();
    let q_len = chunk[0].0.len();
    q_len == chunk[0].1.len()
        && chunk
            .iter()
            .all(|(q, c)| q.as_ptr() == q_ptr && q.len() == q_len && c.len() == q_len)
}

/// **Unstable**: batched dot-product dispatch with a same-query fast path.
pub fn batch_dot_product(pairs: &[(&[f32], &[f32])]) -> Vec<f32> {
    let pair_kernel = resolved_dot_product_kernel();
    let batch4_kernel = resolved_dot_product_batch4_kernel();
    let mut out = Vec::with_capacity(pairs.len());

    let mut chunks = pairs.chunks_exact(4);
    for chunk in &mut chunks {
        if same_query_batch4(chunk) {
            let q = chunk[0].0;
            let dots = batch4_kernel(q, chunk[0].1, chunk[1].1, chunk[2].1, chunk[3].1);
            out.extend_from_slice(&dots);
        } else {
            for &(a, b) in chunk {
                out.push(if a.len() == b.len() {
                    pair_kernel(a, b)
                } else {
                    0.0
                });
            }
        }
    }
    for &(a, b) in chunks.remainder() {
        out.push(if a.len() == b.len() {
            pair_kernel(a, b)
        } else {
            0.0
        });
    }
    out
}
