//! SIMD-accelerated vector normalization.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::simd_config;

#[cfg(target_arch = "x86_64")]
use super::dot_product::{horizontal_sum_avx2, horizontal_sum_avx512};

#[cfg(target_arch = "aarch64")]
use super::dot_product::horizontal_sum_neon;

/// **Unstable**: SIMD dispatch layer; use `lattice_embed::utils::normalize` for the stable wrapper.
#[inline]
pub fn normalize(vector: &mut [f32]) {
    let config = simd_config();

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx512f_enabled {
            // SAFETY: Runtime feature detection verified AVX-512F. The mutable
            // slice is valid for the call lifetime; the callee uses unaligned
            // loads/stores and chunk/remainder bounds that stay inside the slice.
            return unsafe { normalize_avx512_unrolled(vector) };
        }
        if config.avx2_enabled && config.fma_enabled {
            // SAFETY: Runtime feature detection verified AVX2+FMA. The mutable
            // slice is valid for the call lifetime; the callee uses unaligned
            // loads/stores and chunk/remainder bounds that stay inside the slice.
            return unsafe { normalize_avx2_unrolled(vector) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is available on aarch64. The mutable slice is valid
            // for the call lifetime; the callee uses unaligned loads/stores and
            // bounded chunk/remainder loops that stay inside the slice.
            return unsafe { normalize_neon_unrolled(vector) };
        }
    }

    normalize_scalar(vector)
}

/// Scalar normalization.
pub(crate) fn normalize_scalar(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv_norm = 1.0 / norm;
        vector.iter_mut().for_each(|x| *x *= inv_norm);
    }
}

/// AVX-512F-accelerated normalization with 4x unrolling.
///
/// Performs two passes:
/// 1. Compute L2 norm
/// 2. Scale each element by 1 / norm
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX-512F instructions (verified via `simd_config()`)
///
/// Memory safety:
/// - Uses `_mm512_loadu_ps`/`_mm512_storeu_ps` for unaligned access
/// - Pointer arithmetic stays within slice bounds via chunk calculation
/// - Remainder loops use safe indexing
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn normalize_avx512_unrolled(vector: &mut [f32]) {
    const SIMD_WIDTH: usize = 16;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;

    let n = vector.len();
    let chunks = n / CHUNK_SIZE;
    let main_processed = chunks * CHUNK_SIZE;
    let remaining = n - main_processed;
    let remaining_chunks = remaining / SIMD_WIDTH;

    // First pass: compute L2 norm with 4 accumulators
    let mut norm0 = _mm512_setzero_ps();
    let mut norm1 = _mm512_setzero_ps();
    let mut norm2 = _mm512_setzero_ps();
    let mut norm3 = _mm512_setzero_ps();

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let v0 = _mm512_loadu_ps(vector.as_ptr().add(base));
        norm0 = _mm512_fmadd_ps(v0, v0, norm0);

        let v1 = _mm512_loadu_ps(vector.as_ptr().add(base + SIMD_WIDTH));
        norm1 = _mm512_fmadd_ps(v1, v1, norm1);

        let v2 = _mm512_loadu_ps(vector.as_ptr().add(base + SIMD_WIDTH * 2));
        norm2 = _mm512_fmadd_ps(v2, v2, norm2);

        let v3 = _mm512_loadu_ps(vector.as_ptr().add(base + SIMD_WIDTH * 3));
        norm3 = _mm512_fmadd_ps(v3, v3, norm3);
    }

    let norm_vec = _mm512_add_ps(_mm512_add_ps(norm0, norm1), _mm512_add_ps(norm2, norm3));

    // Remainder for norm calculation with single-register AVX-512F loop
    let mut norm_remainder = _mm512_setzero_ps();
    for i in 0..remaining_chunks {
        let offset = main_processed + i * SIMD_WIDTH;
        let v = _mm512_loadu_ps(vector.as_ptr().add(offset));
        norm_remainder = _mm512_fmadd_ps(v, v, norm_remainder);
    }

    let mut norm_sq = horizontal_sum_avx512(norm_vec) + horizontal_sum_avx512(norm_remainder);

    // Scalar tail for norm (recomputed inline to avoid cross-pass variable dependency)
    for i in (main_processed + remaining_chunks * SIMD_WIDTH)..n {
        norm_sq += vector[i] * vector[i];
    }

    let norm = norm_sq.sqrt();
    if norm == 0.0 {
        return;
    }

    let inv_norm = 1.0 / norm;
    let inv_norm_vec = _mm512_set1_ps(inv_norm);

    // Second pass: scale by inverse norm with 4x unrolling
    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let v0 = _mm512_loadu_ps(vector.as_ptr().add(base));
        _mm512_storeu_ps(
            vector.as_mut_ptr().add(base),
            _mm512_mul_ps(v0, inv_norm_vec),
        );

        let v1 = _mm512_loadu_ps(vector.as_ptr().add(base + SIMD_WIDTH));
        _mm512_storeu_ps(
            vector.as_mut_ptr().add(base + SIMD_WIDTH),
            _mm512_mul_ps(v1, inv_norm_vec),
        );

        let v2 = _mm512_loadu_ps(vector.as_ptr().add(base + SIMD_WIDTH * 2));
        _mm512_storeu_ps(
            vector.as_mut_ptr().add(base + SIMD_WIDTH * 2),
            _mm512_mul_ps(v2, inv_norm_vec),
        );

        let v3 = _mm512_loadu_ps(vector.as_ptr().add(base + SIMD_WIDTH * 3));
        _mm512_storeu_ps(
            vector.as_mut_ptr().add(base + SIMD_WIDTH * 3),
            _mm512_mul_ps(v3, inv_norm_vec),
        );
    }

    // Remainder for scaling with single-register AVX-512F loop
    for i in 0..remaining_chunks {
        let offset = main_processed + i * SIMD_WIDTH;
        let v = _mm512_loadu_ps(vector.as_ptr().add(offset));
        _mm512_storeu_ps(
            vector.as_mut_ptr().add(offset),
            _mm512_mul_ps(v, inv_norm_vec),
        );
    }

    // Final scalar remainder (recomputed inline to avoid cross-pass variable dependency)
    for i in (main_processed + remaining_chunks * SIMD_WIDTH)..n {
        vector[i] *= inv_norm;
    }
}

/// AVX2-accelerated normalization with 4x unrolling.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX2 and FMA instructions (verified via `simd_config()`)
///
/// Memory safety:
/// - Uses `_mm256_loadu_ps`/`_mm256_storeu_ps` for unaligned access (safe for any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk calculation
/// - Remainder loop uses safe indexing
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn normalize_avx2_unrolled(vector: &mut [f32]) {
    const SIMD_WIDTH: usize = 8;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    let n = vector.len();
    let chunks = n / CHUNK_SIZE;

    // First pass: compute L2 norm with 4 accumulators
    let mut norm0 = _mm256_setzero_ps();
    let mut norm1 = _mm256_setzero_ps();
    let mut norm2 = _mm256_setzero_ps();
    let mut norm3 = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let v0 = _mm256_loadu_ps(vector.as_ptr().add(base));
        norm0 = _mm256_fmadd_ps(v0, v0, norm0);

        let v1 = _mm256_loadu_ps(vector.as_ptr().add(base + SIMD_WIDTH));
        norm1 = _mm256_fmadd_ps(v1, v1, norm1);

        let v2 = _mm256_loadu_ps(vector.as_ptr().add(base + SIMD_WIDTH * 2));
        norm2 = _mm256_fmadd_ps(v2, v2, norm2);

        let v3 = _mm256_loadu_ps(vector.as_ptr().add(base + SIMD_WIDTH * 3));
        norm3 = _mm256_fmadd_ps(v3, v3, norm3);
    }

    let norm_vec = _mm256_add_ps(_mm256_add_ps(norm0, norm1), _mm256_add_ps(norm2, norm3));
    let mut norm_sq = horizontal_sum_avx2(norm_vec);

    // Remainder for norm calculation
    for i in (chunks * CHUNK_SIZE)..n {
        norm_sq += vector[i] * vector[i];
    }

    let norm = norm_sq.sqrt();
    if norm == 0.0 {
        return;
    }

    let inv_norm = 1.0 / norm;
    let inv_norm_vec = _mm256_set1_ps(inv_norm);

    // Second pass: divide by norm with 4x unrolling
    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let v0 = _mm256_loadu_ps(vector.as_ptr().add(base));
        _mm256_storeu_ps(
            vector.as_mut_ptr().add(base),
            _mm256_mul_ps(v0, inv_norm_vec),
        );

        let v1 = _mm256_loadu_ps(vector.as_ptr().add(base + SIMD_WIDTH));
        _mm256_storeu_ps(
            vector.as_mut_ptr().add(base + SIMD_WIDTH),
            _mm256_mul_ps(v1, inv_norm_vec),
        );

        let v2 = _mm256_loadu_ps(vector.as_ptr().add(base + SIMD_WIDTH * 2));
        _mm256_storeu_ps(
            vector.as_mut_ptr().add(base + SIMD_WIDTH * 2),
            _mm256_mul_ps(v2, inv_norm_vec),
        );

        let v3 = _mm256_loadu_ps(vector.as_ptr().add(base + SIMD_WIDTH * 3));
        _mm256_storeu_ps(
            vector.as_mut_ptr().add(base + SIMD_WIDTH * 3),
            _mm256_mul_ps(v3, inv_norm_vec),
        );
    }

    // Remainder for scaling
    for i in (chunks * CHUNK_SIZE)..n {
        vector[i] *= inv_norm;
    }
}

/// NEON-accelerated normalization with 4x unrolling.
///
/// # Safety
///
/// Caller must ensure:
/// - Running on aarch64 (NEON is mandatory, always available)
///
/// Memory safety:
/// - Uses `vld1q_f32`/`vst1q_f32` for loads/stores (handles any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk calculation
/// - Remainder loop uses safe iteration
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn normalize_neon_unrolled(vector: &mut [f32]) {
    const SIMD_WIDTH: usize = 4;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    let n = vector.len();
    let chunks = n / CHUNK_SIZE;

    // First pass: compute L2 norm with 4 accumulators
    let mut norm0 = vdupq_n_f32(0.0);
    let mut norm1 = vdupq_n_f32(0.0);
    let mut norm2 = vdupq_n_f32(0.0);
    let mut norm3 = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let v0 = vld1q_f32(vector.as_ptr().add(base));
        norm0 = vfmaq_f32(norm0, v0, v0);

        let v1 = vld1q_f32(vector.as_ptr().add(base + SIMD_WIDTH));
        norm1 = vfmaq_f32(norm1, v1, v1);

        let v2 = vld1q_f32(vector.as_ptr().add(base + SIMD_WIDTH * 2));
        norm2 = vfmaq_f32(norm2, v2, v2);

        let v3 = vld1q_f32(vector.as_ptr().add(base + SIMD_WIDTH * 3));
        norm3 = vfmaq_f32(norm3, v3, v3);
    }

    let norm_vec = vaddq_f32(vaddq_f32(norm0, norm1), vaddq_f32(norm2, norm3));
    let mut norm_sq = horizontal_sum_neon(norm_vec);

    // Remainder for norm calculation
    for val in vector.iter().skip(chunks * CHUNK_SIZE) {
        norm_sq += val * val;
    }

    let norm = norm_sq.sqrt();
    if norm == 0.0 {
        return;
    }

    let inv_norm = 1.0 / norm;
    let inv_norm_vec = vdupq_n_f32(inv_norm);

    // Second pass: divide by norm with 4x unrolling
    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let v0 = vld1q_f32(vector.as_ptr().add(base));
        vst1q_f32(vector.as_mut_ptr().add(base), vmulq_f32(v0, inv_norm_vec));

        let v1 = vld1q_f32(vector.as_ptr().add(base + SIMD_WIDTH));
        vst1q_f32(
            vector.as_mut_ptr().add(base + SIMD_WIDTH),
            vmulq_f32(v1, inv_norm_vec),
        );

        let v2 = vld1q_f32(vector.as_ptr().add(base + SIMD_WIDTH * 2));
        vst1q_f32(
            vector.as_mut_ptr().add(base + SIMD_WIDTH * 2),
            vmulq_f32(v2, inv_norm_vec),
        );

        let v3 = vld1q_f32(vector.as_ptr().add(base + SIMD_WIDTH * 3));
        vst1q_f32(
            vector.as_mut_ptr().add(base + SIMD_WIDTH * 3),
            vmulq_f32(v3, inv_norm_vec),
        );
    }

    // Remainder for scaling
    for val in vector.iter_mut().skip(chunks * CHUNK_SIZE) {
        *val *= inv_norm;
    }
}
