//! SIMD cosine-similarity kernels and batch variants.
//!
//! See docs/simd.md for fused reductions and query-reuse behaviour.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::*;

use std::sync::OnceLock;

use super::simd_config;

#[cfg(target_arch = "x86_64")]
use super::dot_product::{horizontal_sum_avx2, horizontal_sum_avx512};

#[cfg(target_arch = "aarch64")]
use super::dot_product::horizontal_sum_neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use super::dot_product::horizontal_sum_simd128;

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

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if config.simd128_enabled() {
            return cosine_simd128_kernel;
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

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn cosine_simd128_kernel(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only stored in COSINE_KERNEL when compiled with the wasm32
    // `simd128` target feature (compile-time gate, see `SimdConfig::simd128_enabled`).
    unsafe { cosine_similarity_simd128_unrolled(a, b) }
}

/// Computes cosine similarity, returning `0.0` for a mismatch, empty input, or zero norm.
///
/// See [`docs/simd.md`](../../docs/simd.md#cosine-similarity) for fused reduction and normalized-input use.
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

/// Computes fused cosine similarity with AVX-512F.
///
/// # Safety
/// Caller must provide AVX-512F and equal, non-empty slices; chunked loads stay in bounds.
/// See [`docs/simd.md`](../../docs/simd.md#kernel-safety-boundary) for the shared kernel invariant.
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

    let dot_vec = _mm512_add_ps(_mm512_add_ps(dot0, dot1), _mm512_add_ps(dot2, dot3));
    let na_vec = _mm512_add_ps(_mm512_add_ps(na0, na1), _mm512_add_ps(na2, na3));
    let nb_vec = _mm512_add_ps(_mm512_add_ps(nb0, nb1), _mm512_add_ps(nb2, nb3));

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

/// Computes fused cosine similarity with AVX2 and FMA.
///
/// # Safety
/// Caller must provide AVX2/FMA and equal, non-empty slices; chunked loads stay in bounds.
/// See [`docs/simd.md`](../../docs/simd.md#kernel-safety-boundary) for the shared kernel invariant.
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

    let dot_vec = _mm256_add_ps(_mm256_add_ps(dot0, dot1), _mm256_add_ps(dot2, dot3));
    let na_vec = _mm256_add_ps(_mm256_add_ps(na0, na1), _mm256_add_ps(na2, na3));
    let nb_vec = _mm256_add_ps(_mm256_add_ps(nb0, nb1), _mm256_add_ps(nb2, nb3));

    let mut dot = horizontal_sum_avx2(dot_vec);
    let mut norm_a = horizontal_sum_avx2(na_vec);
    let mut norm_b = horizontal_sum_avx2(nb_vec);

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

/// Computes fused cosine similarity with NEON.
///
/// # Safety
/// Caller must run on aarch64 with equal, non-empty slices; chunked loads stay in bounds.
/// See [`docs/simd.md`](../../docs/simd.md#kernel-safety-boundary) for the shared kernel invariant.
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

    let dot_vec = vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3));
    let na_vec = vaddq_f32(vaddq_f32(na0, na1), vaddq_f32(na2, na3));
    let nb_vec = vaddq_f32(vaddq_f32(nb0, nb1), vaddq_f32(nb2, nb3));

    let mut dot = horizontal_sum_neon(dot_vec);
    let mut norm_a = horizontal_sum_neon(na_vec);
    let mut norm_b = horizontal_sum_neon(nb_vec);

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

/// Computes fused cosine similarity with wasm32 SIMD128.
///
/// # Safety
/// This function requires compile-time SIMD128 and equal, non-empty slices; bounds are chunked.
/// See [`docs/simd.md`](../../docs/simd.md#kernel-safety-boundary) for wasm and reassociation semantics.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn cosine_similarity_simd128_unrolled(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 4;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    let n = a.len();
    debug_assert_eq!(n, b.len());
    debug_assert!(n > 0);
    let chunks = n / CHUNK_SIZE;

    // 4 accumulators for each of 3 sums (dot, norm_a, norm_b)
    let mut dot0 = f32x4_splat(0.0);
    let mut dot1 = f32x4_splat(0.0);
    let mut dot2 = f32x4_splat(0.0);
    let mut dot3 = f32x4_splat(0.0);

    let mut na0 = f32x4_splat(0.0);
    let mut na1 = f32x4_splat(0.0);
    let mut na2 = f32x4_splat(0.0);
    let mut na3 = f32x4_splat(0.0);

    let mut nb0 = f32x4_splat(0.0);
    let mut nb1 = f32x4_splat(0.0);
    let mut nb2 = f32x4_splat(0.0);
    let mut nb3 = f32x4_splat(0.0);

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let a0 = v128_load(a.as_ptr().add(base) as *const v128);
        let b0 = v128_load(b.as_ptr().add(base) as *const v128);
        dot0 = f32x4_add(dot0, f32x4_mul(a0, b0));
        na0 = f32x4_add(na0, f32x4_mul(a0, a0));
        nb0 = f32x4_add(nb0, f32x4_mul(b0, b0));

        let a1 = v128_load(a.as_ptr().add(base + SIMD_WIDTH) as *const v128);
        let b1 = v128_load(b.as_ptr().add(base + SIMD_WIDTH) as *const v128);
        dot1 = f32x4_add(dot1, f32x4_mul(a1, b1));
        na1 = f32x4_add(na1, f32x4_mul(a1, a1));
        nb1 = f32x4_add(nb1, f32x4_mul(b1, b1));

        let a2 = v128_load(a.as_ptr().add(base + SIMD_WIDTH * 2) as *const v128);
        let b2 = v128_load(b.as_ptr().add(base + SIMD_WIDTH * 2) as *const v128);
        dot2 = f32x4_add(dot2, f32x4_mul(a2, b2));
        na2 = f32x4_add(na2, f32x4_mul(a2, a2));
        nb2 = f32x4_add(nb2, f32x4_mul(b2, b2));

        let a3 = v128_load(a.as_ptr().add(base + SIMD_WIDTH * 3) as *const v128);
        let b3 = v128_load(b.as_ptr().add(base + SIMD_WIDTH * 3) as *const v128);
        dot3 = f32x4_add(dot3, f32x4_mul(a3, b3));
        na3 = f32x4_add(na3, f32x4_mul(a3, a3));
        nb3 = f32x4_add(nb3, f32x4_mul(b3, b3));
    }

    let dot_vec = f32x4_add(f32x4_add(dot0, dot1), f32x4_add(dot2, dot3));
    let na_vec = f32x4_add(f32x4_add(na0, na1), f32x4_add(na2, na3));
    let nb_vec = f32x4_add(f32x4_add(nb0, nb1), f32x4_add(nb2, nb3));

    let mut dot = horizontal_sum_simd128(dot_vec);
    let mut norm_a = horizontal_sum_simd128(na_vec);
    let mut norm_b = horizontal_sum_simd128(nb_vec);

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

/// **Unstable**: batched cosine dispatch; callers supply normalization knowledge.
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

/// **Unstable**: fused single-pass cosine similarity.
///
/// For pre-normalized vectors, use `dot_product` directly.
#[inline]
pub fn cosine_similarity_fused(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    // All SIMD backends already perform a fused single-pass computation.
    // This entry point makes the guarantee explicit in the public API.
    cosine_kernel()(a, b)
}

/// The query-side norm for [`cosine_similarity_pre_normalized`].
///
/// Computed with the same dot-product kernel the scoring path uses, so the
/// hoisted norm and the per-candidate scoring agree bit for bit.
#[inline]
pub fn query_norm(query: &[f32]) -> f32 {
    if query.is_empty() {
        return 0.0;
    }
    super::dot_product::resolved_dot_product_kernel()(query, query).sqrt()
}

/// Cosine similarity against a query whose norm the caller already knows.
///
/// This is the shape a scan loop wants. `‖query‖` is fixed across every
/// candidate, but [`cosine_similarity`] takes only the two slices, so it
/// recomputes that norm on every call and spends a third accumulator stream
/// producing a value the caller already has. Hoisting it with [`query_norm`]
/// leaves two accumulators of real work per candidate.
///
/// Obtain `query_norm` from [`query_norm`]. Passing a value that is not
/// `‖query‖` rescales the result rather than erroring, which is what makes an
/// already-normalized query cheap: pass `1.0`.
///
/// Returns `0.0` on a length mismatch, an empty slice, or a zero norm on
/// either side, matching [`cosine_similarity`].
///
/// ```
/// use lattice_embed::simd::{cosine_similarity, cosine_similarity_pre_normalized, query_norm};
///
/// let q = [1.0_f32, 2.0, 3.0];
/// let c = [4.0_f32, 5.0, 6.0];
/// let n = query_norm(&q);
/// let a = cosine_similarity_pre_normalized(&q, &c, n);
/// let b = cosine_similarity(&q, &c);
/// assert!((a - b).abs() < 1e-6);
/// ```
#[inline]
pub fn cosine_similarity_pre_normalized(query: &[f32], candidate: &[f32], query_norm: f32) -> f32 {
    if query.len() != candidate.len() || query.is_empty() || query_norm == 0.0 {
        return 0.0;
    }
    cosine_pre_normalized_with(
        super::dot_product::resolved_dot_product_kernel(),
        query,
        candidate,
        query_norm,
    )
}

/// Shared body, taking an already-resolved kernel.
///
/// Scan loops resolve the kernel once and pass it in, so the per-candidate cost
/// stays two dot products with no `OnceLock` read in the loop.
#[inline]
fn cosine_pre_normalized_with(
    dot_kernel: super::dot_product::DotKernel,
    query: &[f32],
    candidate: &[f32],
    query_norm: f32,
) -> f32 {
    if query.len() != candidate.len() || query.is_empty() || query_norm == 0.0 {
        return 0.0;
    }
    let norm_c = dot_kernel(candidate, candidate).sqrt();
    if norm_c == 0.0 {
        return 0.0;
    }
    dot_kernel(query, candidate) / (query_norm * norm_c)
}

/// **Unstable**: one-query/many-candidate cosine similarity.
///
/// Results retain candidate order and use `0.0` for dimensional mismatches.
pub fn batch_cosine_one_vs_many(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    if query.is_empty() || candidates.is_empty() {
        return vec![0.0_f32; candidates.len()];
    }

    // Resolve the kernel once, outside the candidate loop.
    let dot_kernel = super::dot_product::resolved_dot_product_kernel();

    let norm_q = dot_kernel(query, query).sqrt();
    if norm_q == 0.0 {
        return vec![0.0_f32; candidates.len()];
    }

    candidates
        .iter()
        .map(|&c| cosine_pre_normalized_with(dot_kernel, query, c, norm_q))
        .collect()
}

#[cfg(test)]
mod pre_normalized_tests {
    use super::*;

    fn vecs(dim: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (
            (0..dim).map(|_| next()).collect(),
            (0..dim).map(|_| next()).collect(),
        )
    }

    /// The whole justification for the API: hoisting the query norm must not
    /// change the answer. If it did, callers could not substitute it into a
    /// scan loop.
    #[test]
    fn pre_normalized_agrees_with_cosine_similarity() {
        for dim in [1usize, 3, 4, 7, 8, 15, 16, 17, 31, 64, 127, 384, 768, 1536] {
            for seed in 0..4u32 {
                let (q, c) = vecs(dim, seed);
                let want = cosine_similarity(&q, &c);
                let got = cosine_similarity_pre_normalized(&q, &c, query_norm(&q));
                assert!(
                    (got - want).abs() <= 1e-4,
                    "dim={dim} seed={seed}: pre_normalized {got} vs cosine_similarity {want}"
                );
            }
        }
    }

    /// An already-normalized query is the cheap case: norm 1.0 must reduce to
    /// the plain dot product.
    #[test]
    fn unit_norm_query_reduces_to_dot_product() {
        let (q, c) = vecs(384, 7);
        let n = query_norm(&q);
        let unit: Vec<f32> = q.iter().map(|x| x / n).collect();

        let got = cosine_similarity_pre_normalized(&unit, &c, 1.0);
        let want = cosine_similarity(&unit, &c);
        assert!(
            (got - want).abs() <= 1e-4,
            "unit-norm shortcut disagreed: {got} vs {want}"
        );
    }

    #[test]
    fn degenerate_inputs_return_zero_not_nan() {
        let zero = vec![0.0f32; 8];
        let other = vec![1.0f32; 8];
        let short = vec![1.0f32; 3];

        assert_eq!(query_norm(&[]), 0.0);
        assert_eq!(cosine_similarity_pre_normalized(&zero, &other, 0.0), 0.0);
        assert_eq!(
            cosine_similarity_pre_normalized(&other, &zero, query_norm(&other)),
            0.0
        );
        assert_eq!(
            cosine_similarity_pre_normalized(&other, &short, query_norm(&other)),
            0.0
        );
        assert_eq!(cosine_similarity_pre_normalized(&[], &[], 1.0), 0.0);

        for v in [
            cosine_similarity_pre_normalized(&zero, &zero, 0.0),
            cosine_similarity_pre_normalized(&other, &other, query_norm(&other)),
        ] {
            assert!(v.is_finite(), "produced a non-finite similarity: {v}");
        }
    }

    /// `batch_cosine_one_vs_many` is now implemented on top of the new function,
    /// so it must still agree with per-pair `cosine_similarity`.
    #[test]
    fn batch_one_vs_many_still_matches_per_pair() {
        let (q, _) = vecs(256, 11);
        let cands: Vec<Vec<f32>> = (0..8).map(|s| vecs(256, 100 + s).1).collect();
        let refs: Vec<&[f32]> = cands.iter().map(Vec::as_slice).collect();

        let batch = batch_cosine_one_vs_many(&q, &refs);
        assert_eq!(batch.len(), refs.len());
        for (i, c) in refs.iter().enumerate() {
            let want = cosine_similarity(&q, c);
            assert!(
                (batch[i] - want).abs() <= 1e-4,
                "candidate {i}: batch {} vs per-pair {want}",
                batch[i]
            );
        }
    }
}
