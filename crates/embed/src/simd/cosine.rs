//! SIMD-accelerated cosine similarity operations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use std::sync::OnceLock;

use super::simd_config;

#[cfg(target_arch = "x86_64")]
use super::dot_product::{horizontal_sum_avx2, horizontal_sum_avx512};

#[cfg(target_arch = "aarch64")]
use super::dot_product::horizontal_sum_neon;

type CosineKernel = fn(&[f32], &[f32]) -> f32;
static COSINE_KERNEL: OnceLock<CosineKernel> = OnceLock::new();

#[inline]
fn cosine_kernel() -> CosineKernel {
    *COSINE_KERNEL.get_or_init(resolve_cosine_kernel)
}

fn resolve_cosine_kernel() -> CosineKernel {
    let config = simd_config();

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx512f_enabled {
            return cosine_avx512_kernel;
        }
        if config.avx2_enabled && config.fma_enabled {
            return cosine_avx2_kernel;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            return cosine_neon_kernel;
        }
    }

    cosine_similarity_scalar
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn cosine_avx512_kernel(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only stored in COSINE_KERNEL when avx512f was detected at init time.
    unsafe { cosine_similarity_avx512_unrolled(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn cosine_avx2_kernel(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only stored in COSINE_KERNEL when avx2+fma were detected at init time.
    unsafe { cosine_similarity_avx2_unrolled(a, b) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn cosine_neon_kernel(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only stored in COSINE_KERNEL when neon was detected at init time (always true on aarch64).
    unsafe { cosine_similarity_neon_unrolled(a, b) }
}

/// **Unstable**: SIMD dispatch layer; use `lattice_embed::utils::cosine_similarity` for the stable wrapper.
///
/// For pre-normalized vectors (embeddings typically are), use `dot_product`
/// directly for better performance.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(!a.is_empty());
    cosine_kernel()(a, b)
}

/// Scalar cosine similarity.
pub(crate) fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// AVX-512F-accelerated cosine similarity with 4x unrolling.
///
/// Computes dot(a,b) / (|a| * |b|) in a single pass with 4 accumulators each.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX-512F instructions (verified via `simd_config()`)
/// - `a` and `b` have equal, non-zero length (checked by caller)
///
/// Memory safety:
/// - Uses `_mm512_loadu_ps` for unaligned loads (safe for any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk calculation
/// - Remainder loops use safe indexing after bounds checks
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn cosine_similarity_avx512_unrolled(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 16;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;

    let n = a.len();
    debug_assert_eq!(n, b.len());
    debug_assert!(n > 0);
    let chunks = n / CHUNK_SIZE;

    // 4 accumulators for each of 3 sums (dot, norm_a, norm_b)
    let mut dot0 = _mm512_setzero_ps();
    let mut dot1 = _mm512_setzero_ps();
    let mut dot2 = _mm512_setzero_ps();
    let mut dot3 = _mm512_setzero_ps();

    let mut na0 = _mm512_setzero_ps();
    let mut na1 = _mm512_setzero_ps();
    let mut na2 = _mm512_setzero_ps();
    let mut na3 = _mm512_setzero_ps();

    let mut nb0 = _mm512_setzero_ps();
    let mut nb1 = _mm512_setzero_ps();
    let mut nb2 = _mm512_setzero_ps();
    let mut nb3 = _mm512_setzero_ps();

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let a0 = _mm512_loadu_ps(a.as_ptr().add(base));
        let b0 = _mm512_loadu_ps(b.as_ptr().add(base));
        dot0 = _mm512_fmadd_ps(a0, b0, dot0);
        na0 = _mm512_fmadd_ps(a0, a0, na0);
        nb0 = _mm512_fmadd_ps(b0, b0, nb0);

        let a1 = _mm512_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = _mm512_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH));
        dot1 = _mm512_fmadd_ps(a1, b1, dot1);
        na1 = _mm512_fmadd_ps(a1, a1, na1);
        nb1 = _mm512_fmadd_ps(b1, b1, nb1);

        let a2 = _mm512_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = _mm512_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 2));
        dot2 = _mm512_fmadd_ps(a2, b2, dot2);
        na2 = _mm512_fmadd_ps(a2, a2, na2);
        nb2 = _mm512_fmadd_ps(b2, b2, nb2);

        let a3 = _mm512_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = _mm512_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 3));
        dot3 = _mm512_fmadd_ps(a3, b3, dot3);
        na3 = _mm512_fmadd_ps(a3, a3, na3);
        nb3 = _mm512_fmadd_ps(b3, b3, nb3);
    }

    // Combine accumulators
    let dot_vec = _mm512_add_ps(_mm512_add_ps(dot0, dot1), _mm512_add_ps(dot2, dot3));
    let na_vec = _mm512_add_ps(_mm512_add_ps(na0, na1), _mm512_add_ps(na2, na3));
    let nb_vec = _mm512_add_ps(_mm512_add_ps(nb0, nb1), _mm512_add_ps(nb2, nb3));

    // Handle remainder with single-register AVX-512F loop
    let main_processed = chunks * CHUNK_SIZE;
    let remaining = n - main_processed;
    let remaining_chunks = remaining / SIMD_WIDTH;

    let mut dot_remainder = _mm512_setzero_ps();
    let mut na_remainder = _mm512_setzero_ps();
    let mut nb_remainder = _mm512_setzero_ps();

    for i in 0..remaining_chunks {
        let offset = main_processed + i * SIMD_WIDTH;
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(offset));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(offset));

        dot_remainder = _mm512_fmadd_ps(a_vec, b_vec, dot_remainder);
        na_remainder = _mm512_fmadd_ps(a_vec, a_vec, na_remainder);
        nb_remainder = _mm512_fmadd_ps(b_vec, b_vec, nb_remainder);
    }

    let mut dot = horizontal_sum_avx512(dot_vec) + horizontal_sum_avx512(dot_remainder);
    let mut norm_a = horizontal_sum_avx512(na_vec) + horizontal_sum_avx512(na_remainder);
    let mut norm_b = horizontal_sum_avx512(nb_vec) + horizontal_sum_avx512(nb_remainder);

    // Final scalar remainder
    let scalar_start = main_processed + remaining_chunks * SIMD_WIDTH;
    for i in scalar_start..n {
        let av = a[i];
        let bv = b[i];
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    norm_a = norm_a.sqrt();
    norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// AVX2-accelerated cosine similarity with 4x unrolling.
///
/// Computes dot(a,b) / (|a| * |b|) in a single pass with 4 accumulators each.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX2 and FMA instructions (verified via `simd_config()`)
/// - `a` and `b` have equal, non-zero length (checked by caller)
///
/// Memory safety:
/// - Uses `_mm256_loadu_ps` for unaligned loads (safe for any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk calculation
/// - Remainder loop uses safe indexing
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn cosine_similarity_avx2_unrolled(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 8;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    let n = a.len();
    debug_assert_eq!(n, b.len());
    debug_assert!(n > 0);
    let chunks = n / CHUNK_SIZE;

    // 4 accumulators for each of 3 sums (dot, norm_a, norm_b)
    let mut dot0 = _mm256_setzero_ps();
    let mut dot1 = _mm256_setzero_ps();
    let mut dot2 = _mm256_setzero_ps();
    let mut dot3 = _mm256_setzero_ps();

    let mut na0 = _mm256_setzero_ps();
    let mut na1 = _mm256_setzero_ps();
    let mut na2 = _mm256_setzero_ps();
    let mut na3 = _mm256_setzero_ps();

    let mut nb0 = _mm256_setzero_ps();
    let mut nb1 = _mm256_setzero_ps();
    let mut nb2 = _mm256_setzero_ps();
    let mut nb3 = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        // Unroll 0
        let a0 = _mm256_loadu_ps(a.as_ptr().add(base));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(base));
        dot0 = _mm256_fmadd_ps(a0, b0, dot0);
        na0 = _mm256_fmadd_ps(a0, a0, na0);
        nb0 = _mm256_fmadd_ps(b0, b0, nb0);

        // Unroll 1
        let a1 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH));
        dot1 = _mm256_fmadd_ps(a1, b1, dot1);
        na1 = _mm256_fmadd_ps(a1, a1, na1);
        nb1 = _mm256_fmadd_ps(b1, b1, nb1);

        // Unroll 2
        let a2 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 2));
        dot2 = _mm256_fmadd_ps(a2, b2, dot2);
        na2 = _mm256_fmadd_ps(a2, a2, na2);
        nb2 = _mm256_fmadd_ps(b2, b2, nb2);

        // Unroll 3
        let a3 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 3));
        dot3 = _mm256_fmadd_ps(a3, b3, dot3);
        na3 = _mm256_fmadd_ps(a3, a3, na3);
        nb3 = _mm256_fmadd_ps(b3, b3, nb3);
    }

    // Combine accumulators
    let dot_vec = _mm256_add_ps(_mm256_add_ps(dot0, dot1), _mm256_add_ps(dot2, dot3));
    let na_vec = _mm256_add_ps(_mm256_add_ps(na0, na1), _mm256_add_ps(na2, na3));
    let nb_vec = _mm256_add_ps(_mm256_add_ps(nb0, nb1), _mm256_add_ps(nb2, nb3));

    let mut dot = horizontal_sum_avx2(dot_vec);
    let mut norm_a = horizontal_sum_avx2(na_vec);
    let mut norm_b = horizontal_sum_avx2(nb_vec);

    // Handle remainder
    let remainder_start = chunks * CHUNK_SIZE;
    for i in remainder_start..n {
        let av = a[i];
        let bv = b[i];
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    norm_a = norm_a.sqrt();
    norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// NEON-accelerated cosine similarity with 4x unrolling.
///
/// Computes dot(a,b) / (|a| * |b|) in a single pass with 4 accumulators each.
///
/// # Safety
///
/// Caller must ensure:
/// - Running on aarch64 (NEON is mandatory, always available)
/// - `a` and `b` have equal, non-zero length (checked by caller)
///
/// Memory safety:
/// - Uses `vld1q_f32` for loads (handles any alignment)
/// - Pointer arithmetic stays within slice bounds via chunk calculation
/// - Remainder loop uses safe indexing
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn cosine_similarity_neon_unrolled(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 4;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    let n = a.len();
    debug_assert_eq!(n, b.len());
    debug_assert!(n > 0);
    let chunks = n / CHUNK_SIZE;

    // 4 accumulators for each sum
    let mut dot0 = vdupq_n_f32(0.0);
    let mut dot1 = vdupq_n_f32(0.0);
    let mut dot2 = vdupq_n_f32(0.0);
    let mut dot3 = vdupq_n_f32(0.0);

    let mut na0 = vdupq_n_f32(0.0);
    let mut na1 = vdupq_n_f32(0.0);
    let mut na2 = vdupq_n_f32(0.0);
    let mut na3 = vdupq_n_f32(0.0);

    let mut nb0 = vdupq_n_f32(0.0);
    let mut nb1 = vdupq_n_f32(0.0);
    let mut nb2 = vdupq_n_f32(0.0);
    let mut nb3 = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let a0 = vld1q_f32(a.as_ptr().add(base));
        let b0 = vld1q_f32(b.as_ptr().add(base));
        dot0 = vfmaq_f32(dot0, a0, b0);
        na0 = vfmaq_f32(na0, a0, a0);
        nb0 = vfmaq_f32(nb0, b0, b0);

        let a1 = vld1q_f32(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = vld1q_f32(b.as_ptr().add(base + SIMD_WIDTH));
        dot1 = vfmaq_f32(dot1, a1, b1);
        na1 = vfmaq_f32(na1, a1, a1);
        nb1 = vfmaq_f32(nb1, b1, b1);

        let a2 = vld1q_f32(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = vld1q_f32(b.as_ptr().add(base + SIMD_WIDTH * 2));
        dot2 = vfmaq_f32(dot2, a2, b2);
        na2 = vfmaq_f32(na2, a2, a2);
        nb2 = vfmaq_f32(nb2, b2, b2);

        let a3 = vld1q_f32(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = vld1q_f32(b.as_ptr().add(base + SIMD_WIDTH * 3));
        dot3 = vfmaq_f32(dot3, a3, b3);
        na3 = vfmaq_f32(na3, a3, a3);
        nb3 = vfmaq_f32(nb3, b3, b3);
    }

    // Combine accumulators
    let dot_vec = vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3));
    let na_vec = vaddq_f32(vaddq_f32(na0, na1), vaddq_f32(na2, na3));
    let nb_vec = vaddq_f32(vaddq_f32(nb0, nb1), vaddq_f32(nb2, nb3));

    let mut dot = horizontal_sum_neon(dot_vec);
    let mut norm_a = horizontal_sum_neon(na_vec);
    let mut norm_b = horizontal_sum_neon(nb_vec);

    // Handle remainder
    let remainder_start = chunks * CHUNK_SIZE;
    for i in remainder_start..n {
        let av = a[i];
        let bv = b[i];
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    norm_a = norm_a.sqrt();
    norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// **Unstable**: SIMD batch dispatch; use `lattice_embed::utils::batch_cosine_similarity` for stable wrapper.
///
/// Resolves the SIMD cosine kernel once to hoist OnceLock dispatch out of the per-pair loop.
/// For unit-normalized inputs, callers should use `batch_dot_product` directly (or the
/// PreparedQuery/HNSW APIs which accept explicit normalization hints). Auto-detecting
/// unit-normalization here would require an O(N×dim) pre-scan costing as much as the
/// computation itself, so detection is left to the caller.
pub fn batch_cosine_similarity(pairs: &[(&[f32], &[f32])]) -> Vec<f32> {
    let kernel = cosine_kernel();
    pairs
        .iter()
        .map(|&(a, b)| {
            if a.len() != b.len() || a.is_empty() {
                0.0
            } else {
                kernel(a, b)
            }
        })
        .collect()
}
