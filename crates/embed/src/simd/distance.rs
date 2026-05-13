//! SIMD-accelerated distance operations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::simd_config;

#[cfg(target_arch = "x86_64")]
use super::dot_product::{horizontal_sum_avx2, horizontal_sum_avx512};

#[cfg(target_arch = "aarch64")]
use super::dot_product::horizontal_sum_neon;

#[inline(always)]
fn dispatch_squared(a: &[f32], b: &[f32]) -> f32 {
    let config = simd_config();

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx512f_enabled {
            return unsafe { squared_euclidean_distance_avx512_unrolled(a, b) };
        }
        if config.avx2_enabled && config.fma_enabled {
            return unsafe { squared_euclidean_distance_avx2_unrolled(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            return unsafe { squared_euclidean_distance_neon_unrolled(a, b) };
        }
    }

    squared_euclidean_distance_scalar(a, b)
}

/// **Unstable**: SIMD dispatch layer; use `lattice_embed::utils::euclidean_distance` for the stable wrapper.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    debug_assert_eq!(a.len(), b.len());
    dispatch_squared(a, b).sqrt()
}

/// **Unstable**: squared Euclidean distance — skips the final sqrt.
///
/// Ordering is preserved: `sq_dist(a,b) <= sq_dist(a,c) ↔ dist(a,b) <= dist(a,c)`.
/// Use this for HNSW graph comparisons where only ordering matters; apply `.sqrt()`
/// at the output boundary when the true L2 distance is required.
#[inline]
pub fn squared_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    debug_assert_eq!(a.len(), b.len());
    dispatch_squared(a, b)
}

/// Scalar squared Euclidean distance (sum of squared differences, no sqrt).
pub(crate) fn squared_euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
}

/// Scalar Euclidean distance (used by tests to validate SIMD results).
#[cfg(test)]
pub(crate) fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    squared_euclidean_distance_scalar(a, b).sqrt()
}

/// AVX-512F-accelerated squared Euclidean distance with 4x unrolling.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX-512F instructions (verified via `simd_config()`)
/// - `a` and `b` have equal length (checked by caller)
///
/// Memory safety:
/// - Uses `_mm512_loadu_ps` for unaligned loads (safe for any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk/remainder calculation
/// - Remainder loops use safe indexing
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn squared_euclidean_distance_avx512_unrolled(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 16;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;

    let n = a.len();
    debug_assert_eq!(n, b.len());
    let chunks = n / CHUNK_SIZE;

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let a0 = _mm512_loadu_ps(a.as_ptr().add(base));
        let b0 = _mm512_loadu_ps(b.as_ptr().add(base));
        let diff0 = _mm512_sub_ps(a0, b0);
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);

        let a1 = _mm512_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = _mm512_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH));
        let diff1 = _mm512_sub_ps(a1, b1);
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);

        let a2 = _mm512_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = _mm512_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 2));
        let diff2 = _mm512_sub_ps(a2, b2);
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);

        let a3 = _mm512_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = _mm512_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 3));
        let diff3 = _mm512_sub_ps(a3, b3);
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);
    }

    let sum_vec = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    let main_sum = horizontal_sum_avx512(sum_vec);

    // Handle remainder with single-register AVX-512F loop
    let main_processed = chunks * CHUNK_SIZE;
    let remaining = n - main_processed;
    let remaining_chunks = remaining / SIMD_WIDTH;

    let mut remainder_sum = _mm512_setzero_ps();
    for i in 0..remaining_chunks {
        let offset = main_processed + i * SIMD_WIDTH;
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(offset));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm512_sub_ps(a_vec, b_vec);
        remainder_sum = _mm512_fmadd_ps(diff, diff, remainder_sum);
    }

    let mut sum = main_sum + horizontal_sum_avx512(remainder_sum);

    // Final scalar remainder
    let scalar_start = main_processed + remaining_chunks * SIMD_WIDTH;
    for i in scalar_start..n {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum
}

/// AVX2-accelerated squared Euclidean distance with 4x unrolling.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX2 and FMA instructions (verified via `simd_config()`)
/// - `a` and `b` have equal length (checked by caller)
///
/// Memory safety:
/// - Uses `_mm256_loadu_ps` for unaligned loads (safe for any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk calculation
/// - Remainder loop uses safe indexing
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn squared_euclidean_distance_avx2_unrolled(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 8;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let chunks = n / CHUNK_SIZE;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let a0 = _mm256_loadu_ps(a.as_ptr().add(base));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(base));
        let diff0 = _mm256_sub_ps(a0, b0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

        let a1 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH));
        let diff1 = _mm256_sub_ps(a1, b1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);

        let a2 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 2));
        let diff2 = _mm256_sub_ps(a2, b2);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);

        let a3 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 3));
        let diff3 = _mm256_sub_ps(a3, b3);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
    }

    let sum_vec = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    let mut sum = horizontal_sum_avx2(sum_vec);

    // Handle remainder
    for i in (chunks * CHUNK_SIZE)..n {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum
}

/// NEON-accelerated squared Euclidean distance with 4x unrolling.
///
/// # Safety
///
/// Caller must ensure:
/// - Running on aarch64 (NEON is mandatory, always available)
/// - `a` and `b` have equal length (checked by caller)
///
/// Memory safety:
/// - Uses `vld1q_f32` for loads (handles any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk calculation
/// - Remainder loop uses safe indexing
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn squared_euclidean_distance_neon_unrolled(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 4;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let chunks = n / CHUNK_SIZE;

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let a0 = vld1q_f32(a.as_ptr().add(base));
        let b0 = vld1q_f32(b.as_ptr().add(base));
        let diff0 = vsubq_f32(a0, b0);
        sum0 = vfmaq_f32(sum0, diff0, diff0);

        let a1 = vld1q_f32(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = vld1q_f32(b.as_ptr().add(base + SIMD_WIDTH));
        let diff1 = vsubq_f32(a1, b1);
        sum1 = vfmaq_f32(sum1, diff1, diff1);

        let a2 = vld1q_f32(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = vld1q_f32(b.as_ptr().add(base + SIMD_WIDTH * 2));
        let diff2 = vsubq_f32(a2, b2);
        sum2 = vfmaq_f32(sum2, diff2, diff2);

        let a3 = vld1q_f32(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = vld1q_f32(b.as_ptr().add(base + SIMD_WIDTH * 3));
        let diff3 = vsubq_f32(a3, b3);
        sum3 = vfmaq_f32(sum3, diff3, diff3);
    }

    let sum_vec = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    let mut sum = horizontal_sum_neon(sum_vec);

    // Handle remainder
    for i in (chunks * CHUNK_SIZE)..n {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum
}
