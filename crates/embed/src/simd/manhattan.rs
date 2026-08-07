//! SIMD Manhattan (L1) distance kernels.
//!
//! Mirrors the squared-L2 kernels in `distance.rs`: same dispatch shape, same
//! 4x unroll, same remainder handling. The only difference is the per-lane
//! operation, `|a - b|` accumulated instead of `(a - b)^2`.
//!
//! There is no `fma` in these kernels, so the AVX2 path requires only `avx2`
//! rather than `avx2 + fma` the way the L2 kernel does.
//!
//! See docs/simd.md for distance semantics and ranking guidance.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::*;

use super::simd_config;

#[cfg(target_arch = "x86_64")]
use super::dot_product::{horizontal_sum_avx2, horizontal_sum_avx512};

#[cfg(target_arch = "aarch64")]
use super::dot_product::horizontal_sum_neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use super::dot_product::horizontal_sum_simd128;

#[inline(always)]
fn dispatch_manhattan(a: &[f32], b: &[f32]) -> f32 {
    let config = simd_config();

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx512f_enabled {
            return unsafe { manhattan_distance_avx512_unrolled(a, b) };
        }
        if config.avx2_enabled {
            return unsafe { manhattan_distance_avx2_unrolled(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            return unsafe { manhattan_distance_neon_unrolled(a, b) };
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if config.simd128_enabled() {
            return unsafe { manhattan_distance_simd128_unrolled(a, b) };
        }
    }

    manhattan_distance_scalar(a, b)
}

/// Computes Manhattan (L1) distance, the sum of absolute per-element differences.
///
/// Returns [`f32::MAX`] for a dimensional mismatch, matching
/// [`euclidean_distance`](super::euclidean_distance).
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    debug_assert_eq!(a.len(), b.len());
    dispatch_manhattan(a, b)
}

/// Scalar Manhattan distance.
pub(crate) fn manhattan_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
}

/// Computes L1 with AVX-512F.
///
/// # Safety
/// Caller must provide AVX-512F and equal slices; chunked unaligned loads stay in bounds.
/// See [`docs/simd.md`](../../docs/simd.md#kernel-safety-boundary) for the shared kernel invariant.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn manhattan_distance_avx512_unrolled(a: &[f32], b: &[f32]) -> f32 {
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
        sum0 = _mm512_add_ps(sum0, _mm512_abs_ps(_mm512_sub_ps(a0, b0)));

        let a1 = _mm512_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = _mm512_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH));
        sum1 = _mm512_add_ps(sum1, _mm512_abs_ps(_mm512_sub_ps(a1, b1)));

        let a2 = _mm512_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = _mm512_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 2));
        sum2 = _mm512_add_ps(sum2, _mm512_abs_ps(_mm512_sub_ps(a2, b2)));

        let a3 = _mm512_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = _mm512_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 3));
        sum3 = _mm512_add_ps(sum3, _mm512_abs_ps(_mm512_sub_ps(a3, b3)));
    }

    let sum_vec = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    let mut sum = horizontal_sum_avx512(sum_vec);

    for i in (chunks * CHUNK_SIZE)..n {
        sum += (a[i] - b[i]).abs();
    }

    sum
}

/// Computes L1 with AVX2.
///
/// There is no `_mm256_abs_ps`, so the sign bit is cleared with `andnot` against
/// a broadcast `-0.0` mask, which is the standard float-absolute idiom and needs
/// no FMA.
///
/// # Safety
/// Caller must provide AVX2 and equal slices; chunked unaligned loads stay in bounds.
/// See [`docs/simd.md`](../../docs/simd.md#kernel-safety-boundary) for the shared kernel invariant.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn manhattan_distance_avx2_unrolled(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 8;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let chunks = n / CHUNK_SIZE;

    // Clearing the sign bit: andnot(-0.0, x) keeps every bit except the sign.
    let sign_mask = _mm256_set1_ps(-0.0);

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let a0 = _mm256_loadu_ps(a.as_ptr().add(base));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(base));
        sum0 = _mm256_add_ps(sum0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a0, b0)));

        let a1 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH));
        sum1 = _mm256_add_ps(sum1, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a1, b1)));

        let a2 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 2));
        sum2 = _mm256_add_ps(sum2, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a2, b2)));

        let a3 = _mm256_loadu_ps(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(base + SIMD_WIDTH * 3));
        sum3 = _mm256_add_ps(sum3, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(a3, b3)));
    }

    let sum_vec = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    let mut sum = horizontal_sum_avx2(sum_vec);

    for i in (chunks * CHUNK_SIZE)..n {
        sum += (a[i] - b[i]).abs();
    }

    sum
}

/// Computes L1 with NEON.
///
/// # Safety
/// Caller must run on aarch64 with equal slices; chunked loads stay in bounds.
/// See [`docs/simd.md`](../../docs/simd.md#kernel-safety-boundary) for the shared kernel invariant.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn manhattan_distance_neon_unrolled(a: &[f32], b: &[f32]) -> f32 {
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
        sum0 = vaddq_f32(sum0, vabsq_f32(vsubq_f32(a0, b0)));

        let a1 = vld1q_f32(a.as_ptr().add(base + SIMD_WIDTH));
        let b1 = vld1q_f32(b.as_ptr().add(base + SIMD_WIDTH));
        sum1 = vaddq_f32(sum1, vabsq_f32(vsubq_f32(a1, b1)));

        let a2 = vld1q_f32(a.as_ptr().add(base + SIMD_WIDTH * 2));
        let b2 = vld1q_f32(b.as_ptr().add(base + SIMD_WIDTH * 2));
        sum2 = vaddq_f32(sum2, vabsq_f32(vsubq_f32(a2, b2)));

        let a3 = vld1q_f32(a.as_ptr().add(base + SIMD_WIDTH * 3));
        let b3 = vld1q_f32(b.as_ptr().add(base + SIMD_WIDTH * 3));
        sum3 = vaddq_f32(sum3, vabsq_f32(vsubq_f32(a3, b3)));
    }

    let sum_vec = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    let mut sum = horizontal_sum_neon(sum_vec);

    for i in (chunks * CHUNK_SIZE)..n {
        sum += (a[i] - b[i]).abs();
    }

    sum
}

/// Computes L1 with wasm32 SIMD128.
///
/// # Safety
/// This function requires compile-time SIMD128 and equal slices; bounds are chunked.
/// See [`docs/simd.md`](../../docs/simd.md#kernel-safety-boundary) for wasm and reassociation semantics.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn manhattan_distance_simd128_unrolled(a: &[f32], b: &[f32]) -> f32 {
    const SIMD_WIDTH: usize = 4;
    const UNROLL: usize = 4;
    const CHUNK_SIZE: usize = SIMD_WIDTH * UNROLL;
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let chunks = n / CHUNK_SIZE;

    let mut sum0 = f32x4_splat(0.0);
    let mut sum1 = f32x4_splat(0.0);
    let mut sum2 = f32x4_splat(0.0);
    let mut sum3 = f32x4_splat(0.0);

    for i in 0..chunks {
        let base = i * CHUNK_SIZE;

        let a0 = v128_load(a.as_ptr().add(base) as *const v128);
        let b0 = v128_load(b.as_ptr().add(base) as *const v128);
        sum0 = f32x4_add(sum0, f32x4_abs(f32x4_sub(a0, b0)));

        let a1 = v128_load(a.as_ptr().add(base + SIMD_WIDTH) as *const v128);
        let b1 = v128_load(b.as_ptr().add(base + SIMD_WIDTH) as *const v128);
        sum1 = f32x4_add(sum1, f32x4_abs(f32x4_sub(a1, b1)));

        let a2 = v128_load(a.as_ptr().add(base + SIMD_WIDTH * 2) as *const v128);
        let b2 = v128_load(b.as_ptr().add(base + SIMD_WIDTH * 2) as *const v128);
        sum2 = f32x4_add(sum2, f32x4_abs(f32x4_sub(a2, b2)));

        let a3 = v128_load(a.as_ptr().add(base + SIMD_WIDTH * 3) as *const v128);
        let b3 = v128_load(b.as_ptr().add(base + SIMD_WIDTH * 3) as *const v128);
        sum3 = f32x4_add(sum3, f32x4_abs(f32x4_sub(a3, b3)));
    }

    let sum_vec = f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3));
    let mut sum = horizontal_sum_simd128(sum_vec);

    for i in (chunks * CHUNK_SIZE)..n {
        sum += (a[i] - b[i]).abs();
    }

    sum
}

#[cfg(test)]
mod tests {
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

    /// Whichever kernel is dispatched must agree with the scalar reference.
    ///
    /// Dimensions straddle the 4/8/16-lane widths and the 4x-unrolled chunk
    /// sizes (16/32/64 elements) plus their remainders, so the tail loop is
    /// exercised rather than assumed.
    #[test]
    fn simd_matches_scalar_across_dimensions() {
        for dim in [
            0usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 384, 768, 1536,
        ] {
            for seed in 0..4u32 {
                let (a, b) = vecs(dim, seed);
                let got = manhattan_distance(&a, &b);
                let want = manhattan_distance_scalar(&a, &b);
                assert!(
                    (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                    "dim={dim} seed={seed}: simd {got} vs scalar {want}"
                );
            }
        }
    }

    #[test]
    fn known_values() {
        assert_eq!(manhattan_distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 9.0);
        // Sign is irrelevant: only the magnitude of each difference counts.
        assert_eq!(manhattan_distance(&[-1.0, 0.0], &[1.0, 0.0]), 2.0);
        assert_eq!(manhattan_distance(&[5.0], &[5.0]), 0.0);
    }

    #[test]
    fn dimension_mismatch_is_max() {
        assert_eq!(manhattan_distance(&[1.0, 2.0], &[1.0]), f32::MAX);
        assert_eq!(manhattan_distance(&[], &[1.0]), f32::MAX);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(manhattan_distance(&[], &[]), 0.0);
    }

    /// L1 must never be negative, whatever the operand order.
    #[test]
    fn symmetric_and_non_negative() {
        for dim in [1usize, 8, 17, 64, 384] {
            let (a, b) = vecs(dim, dim as u32);
            let ab = manhattan_distance(&a, &b);
            let ba = manhattan_distance(&b, &a);
            assert!(ab >= 0.0, "negative L1 at dim={dim}: {ab}");
            assert!(
                (ab - ba).abs() <= 1e-4,
                "asymmetric at dim={dim}: {ab} vs {ba}"
            );
        }
    }
}
