//! SIMD-accelerated dot product operations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

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

/// **Unstable**: SIMD dispatch layer; use `lattice_embed::utils::dot_product` for the stable wrapper.
///
/// For normalized vectors, this equals cosine similarity.
/// Returns 0.0 if vectors have different lengths.
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

/// AVX-512F-accelerated dot product using FMA with 4x unrolling and multiple accumulators.
///
/// Processes 64 floats per iteration (4 x 16 floats) with 4 independent accumulators
/// to break dependency chains and maximize throughput.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX-512F instructions (verified via `simd_config()`)
/// - `a` and `b` have equal length (checked by caller)
///
/// Memory safety:
/// - Uses `_mm512_loadu_ps` for unaligned loads (safe for any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk/remainder calculation:
///   `chunks = n / CHUNK_SIZE` (floor), so `chunks * CHUNK_SIZE <= n`.
///   `remaining_chunks = remaining / SIMD_WIDTH` (floor), so all SIMD loads stay in bounds.
/// - Final scalar loop iterates `scalar_start..n` using safe `a[i]` / `b[i]` indexing
///   and never reads past the end of the slice.
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

/// AVX2-accelerated dot product using 8 independent accumulators.
///
/// Processes 64 floats per iteration (8 x 8 floats) with 8 independent accumulators
/// to better hide FMA latency on modern x86 CPUs. AVX2 provides 16 YMM registers;
/// 8 accumulators + 1 A load + 1 B load = 10 registers, well within budget.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX2 and FMA instructions (verified via `simd_config()`)
/// - `a` and `b` have equal length (checked by caller)
///
/// Memory safety:
/// - Uses `_mm256_loadu_ps` for unaligned loads (safe for any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk/remainder calculation:
///   `chunks = n / CHUNK_SIZE` (floor), so `chunks * CHUNK_SIZE <= n`.
///   `remaining_chunks = remaining / SIMD_WIDTH` (floor), so all SIMD loads stay in bounds.
/// - Final scalar loop uses safe `a[i]` / `b[i]` indexing and never reads past slice end.
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

/// AVX2-accelerated dot product specialized for 384-dimension vectors.
///
/// 384 = 48 x 8, so 384d vectors divide evenly into 48 AVX2 iterations
/// with zero remainder. This eliminates all remainder handling branches.
/// Uses 8 accumulators across 6 iterations of 8 FMAs each (48 total).
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX2 and FMA instructions (verified via `simd_config()`)
/// - `a` and `b` have equal length == 384 (checked by caller)
///
/// Memory safety:
/// - Uses `_mm256_loadu_ps` for unaligned loads (safe for any alignment)
/// - Fixed iteration count (48) covers exactly 384 elements, no out-of-bounds
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

/// AVX2 batch-4 dot product specialized for 384-dimension vectors.
///
/// 384 = 24 × 16, processed exactly as 24 chunks (2 AVX2 loads per chunk) with no
/// scalar remainder, matching the existing `dot_product_384_avx2` specialization.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX2 and FMA (verified via dispatch table)
/// - All 5 slices have equal length == 384 (enforced by `dot_product_batch4`)
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

/// AVX2 batch-4 dot product for arbitrary-length vectors.
///
/// Processes 16 floats per loop (2 AVX2 query loads reused across 4 candidates),
/// with 2 accumulators per candidate to break FMA dependency chains.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX2 and FMA (verified via dispatch table)
/// - All 5 slices have equal length (enforced by `dot_product_batch4`)
///
/// Memory safety:
/// - `chunks * CHUNK <= q.len()` by construction (floor division)
/// - Scalar tail uses safe `q[i]` / `cN[i]` indexing within slice bounds
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

/// NEON-accelerated dot product with 4x unrolling and multiple accumulators.
///
/// Processes 16 floats per iteration (4 x 4 floats) with 4 independent accumulators.
///
/// # Safety
///
/// Caller must ensure:
/// - Running on aarch64 (NEON is mandatory, always available)
/// - `a` and `b` have equal length (checked by caller)
///
/// Memory safety:
/// - Uses `vld1q_f32` for loads (handles any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk/remainder calculation:
///   `chunks = n / CHUNK_SIZE` (floor), so `chunks * CHUNK_SIZE <= n`.
///   `remaining_chunks = remaining / SIMD_WIDTH` (floor), so all NEON loads stay in bounds.
/// - Final scalar loop uses safe `a[i]` / `b[i]` indexing and never reads past slice end.
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

/// NEON batch-4 dot product: one query vs. 4 candidates simultaneously.
///
/// Processes 8 floats per loop (2 NEON vector loads from query, reused across all
/// 4 candidates). Uses 2 accumulators per candidate to break vfmaq_f32 latency chains.
/// NEON provides 32 Q-registers; 8 accumulators + 2 query loads = 10 registers, no spill.
///
/// # Safety
///
/// Caller must ensure:
/// - Running on aarch64 (NEON is mandatory — always true on this arch)
/// - All 5 slices have equal length (enforced by `dot_product_batch4`)
///
/// Memory safety:
/// - `chunks * CHUNK <= q.len()` by construction
/// - Scalar tail uses safe `q[i]` / `cN[i]` indexing within slice bounds
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

/// **Unstable**: SIMD batch dispatch; use `lattice_embed::utils::batch_dot_product` for stable wrapper.
///
/// Uses the batch-4 SIMD kernel for consecutive same-query chunks (e.g., query-vs-N
/// search pattern where the left-hand slice is the same borrowed reference for every pair).
/// Falls back to the per-pair kernel for mixed or remainder inputs.
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
