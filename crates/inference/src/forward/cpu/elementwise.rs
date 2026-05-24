// ===================================================================
// Elementwise operations — RMS norm, SiLU, element-wise mul — with SIMD fast paths
// ===================================================================

use super::simd::simd_config;

/// **Unstable**: RMS normalization in-place; may add SIMD path.
///
/// RMS normalization: x = x * gamma / rms(x), where rms = sqrt(mean(x^2) + eps).
///
/// Unlike LayerNorm, RMSNorm does not center (no beta/mean subtraction).
pub fn rms_norm(x: &mut [f32], gamma: &[f32], hidden: usize, eps: f32) {
    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is available on aarch64 and the runtime gate ensures this path.
            unsafe {
                rms_norm_neon(x, gamma, hidden, eps);
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled {
            // SAFETY: The runtime feature check guarantees AVX2 support.
            unsafe {
                rms_norm_avx2(x, gamma, hidden, eps);
                return;
            }
        }
    }

    rms_norm_scalar(x, gamma, hidden, eps);
}

/// **Unstable**: SiLU in-place; may gain SIMD path.
///
/// SiLU activation: x = x * sigmoid(x).
pub fn silu_inplace(x: &mut [f32]) {
    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is available on aarch64 and the runtime gate ensures this path.
            unsafe {
                silu_inplace_neon(x);
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled {
            // SAFETY: The runtime feature check guarantees AVX2 support.
            unsafe {
                silu_inplace_avx2(x);
                return;
            }
        }
    }

    silu_inplace_scalar(x);
}

/// **Unstable**: element-wise multiply in-place; may gain SIMD path.
///
/// Element-wise multiply: a *= b.
pub fn elementwise_mul(a: &mut [f32], b: &[f32]) {
    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is available on aarch64 and the runtime gate ensures this path.
            unsafe {
                elementwise_mul_neon(a, b);
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled {
            // SAFETY: The runtime feature check guarantees AVX2 support.
            unsafe {
                elementwise_mul_avx2(a, b);
                return;
            }
        }
    }

    elementwise_mul_scalar(a, b);
}

// ===================================================================
// Scalar fallbacks
// ===================================================================

pub fn rms_norm_scalar(x: &mut [f32], gamma: &[f32], hidden: usize, eps: f32) {
    let num_tokens = x.len() / hidden;
    debug_assert_eq!(x.len(), num_tokens * hidden);
    debug_assert_eq!(gamma.len(), hidden);

    for t in 0..num_tokens {
        let row = &mut x[t * hidden..(t + 1) * hidden];

        // Compute mean of squares.
        let mut sum_sq = 0.0f32;
        for &v in row.iter() {
            sum_sq += v * v;
        }
        let rms = (sum_sq / hidden as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Scale by gamma / rms.
        for (v, &g) in row.iter_mut().zip(gamma.iter()) {
            *v = *v * inv_rms * g;
        }
    }
}

#[inline]
pub fn silu_inplace_scalar(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * (1.0 / (1.0 + (-*v).exp()));
    }
}

#[inline]
pub fn elementwise_mul_scalar(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (av, &bv) in a.iter_mut().zip(b.iter()) {
        *av *= bv;
    }
}

// ===================================================================
// NEON fast exp (Schraudolph bit trick) — 4-wide float32x4_t
// ===================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fast_exp_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let x = vmaxq_f32(x, vdupq_n_f32(-87.0));
    let x = vminq_f32(x, vdupq_n_f32(88.0));
    let t = vfmaq_f32(vdupq_n_f32(1_065_353_216.0), x, vdupq_n_f32(12_102_203.0));
    vreinterpretq_f32_s32(vcvtq_s32_f32(t))
}

// ===================================================================
// AVX2 fast exp (Schraudolph bit trick) — 8-wide __m256
// ===================================================================

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn fast_exp_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.0));
    let scale = _mm256_set1_ps(12_102_203.0);
    let bias = _mm256_set1_ps(1_065_353_216.0);
    let t = _mm256_add_ps(_mm256_mul_ps(x, scale), bias);
    _mm256_castsi256_ps(_mm256_cvtps_epi32(t))
}

// ===================================================================
// NEON implementations — 4-wide float32x4_t
// ===================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rms_norm_neon(x: &mut [f32], gamma: &[f32], hidden: usize, eps: f32) {
    use std::arch::aarch64::*;

    let num_tokens = x.len() / hidden;
    debug_assert_eq!(x.len(), num_tokens * hidden);
    debug_assert_eq!(gamma.len(), hidden);

    let chunks = hidden / 4;
    let inv_hidden = 1.0f32 / hidden as f32;

    for t in 0..num_tokens {
        let row_ptr = x.as_mut_ptr().add(t * hidden);
        let gamma_ptr = gamma.as_ptr();

        // --- Step 1: Sum of squares via SIMD ---
        let mut vsum_sq = vdupq_n_f32(0.0);
        for c in 0..chunks {
            // SAFETY: c * 4 + 3 < chunks * 4 <= hidden, within row bounds.
            let v = vld1q_f32(row_ptr.add(c * 4) as *const f32);
            vsum_sq = vfmaq_f32(vsum_sq, v, v);
        }
        let mut sum_sq = vaddvq_f32(vsum_sq);
        for i in (chunks * 4)..hidden {
            let v = *row_ptr.add(i);
            sum_sq += v * v;
        }

        let rms = (sum_sq * inv_hidden + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let vinv_rms = vdupq_n_f32(inv_rms);

        // --- Step 2: Scale by gamma / rms using SIMD ---
        for c in 0..chunks {
            let off = c * 4;
            // SAFETY: off + 3 < chunks * 4 <= hidden, within both row and gamma bounds.
            let v = vld1q_f32(row_ptr.add(off) as *const f32);
            let g = vld1q_f32(gamma_ptr.add(off));
            let result = vmulq_f32(vmulq_f32(v, vinv_rms), g);
            vst1q_f32(row_ptr.add(off), result);
        }
        for i in (chunks * 4)..hidden {
            let v = *row_ptr.add(i);
            let g = *gamma_ptr.add(i);
            *row_ptr.add(i) = v * inv_rms * g;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn silu_inplace_neon(x: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = x.len();
    let chunks = n / 4;
    let ptr = x.as_mut_ptr();
    let vone = vdupq_n_f32(1.0);

    for c in 0..chunks {
        let off = c * 4;
        // SAFETY: off + 3 < chunks * 4 <= n, within slice bounds.
        let v = vld1q_f32(ptr.add(off) as *const f32);
        // sigmoid(v) = 1 / (1 + exp(-v))
        let neg_v = vnegq_f32(v);
        let exp_neg_v = fast_exp_neon(neg_v);
        let sigmoid = vdivq_f32(vone, vaddq_f32(vone, exp_neg_v));
        // silu = v * sigmoid(v)
        let result = vmulq_f32(v, sigmoid);
        vst1q_f32(ptr.add(off), result);
    }
    for i in (chunks * 4)..n {
        let v = *ptr.add(i);
        *ptr.add(i) = v * (1.0 / (1.0 + (-v).exp()));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn elementwise_mul_neon(a: &mut [f32], b: &[f32]) {
    use std::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_mut_ptr();
    let b_ptr = b.as_ptr();

    for c in 0..chunks {
        let off = c * 4;
        // SAFETY: off + 3 < chunks * 4 <= n, within both slice bounds.
        let av = vld1q_f32(a_ptr.add(off) as *const f32);
        let bv = vld1q_f32(b_ptr.add(off));
        vst1q_f32(a_ptr.add(off), vmulq_f32(av, bv));
    }
    for i in (chunks * 4)..n {
        *a_ptr.add(i) *= *b_ptr.add(i);
    }
}

// ===================================================================
// AVX2 implementations — 8-wide __m256
// ===================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rms_norm_avx2(x: &mut [f32], gamma: &[f32], hidden: usize, eps: f32) {
    use std::arch::x86_64::*;

    let num_tokens = x.len() / hidden;
    debug_assert_eq!(x.len(), num_tokens * hidden);
    debug_assert_eq!(gamma.len(), hidden);

    let chunks = hidden / 8;
    let inv_hidden = 1.0f32 / hidden as f32;

    for t in 0..num_tokens {
        let row_ptr = x.as_mut_ptr().add(t * hidden);
        let gamma_ptr = gamma.as_ptr();

        // --- Step 1: Sum of squares via SIMD ---
        let mut vsum_sq = _mm256_setzero_ps();
        for c in 0..chunks {
            // SAFETY: c * 8 + 7 < chunks * 8 <= hidden, within row bounds.
            let v = _mm256_loadu_ps(row_ptr.add(c * 8) as *const f32);
            vsum_sq = _mm256_add_ps(vsum_sq, _mm256_mul_ps(v, v));
        }
        // Horizontal sum of vsum_sq
        let hi = _mm256_extractf128_ps(vsum_sq, 1);
        let lo = _mm256_castps256_ps128(vsum_sq);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let mut sum_sq = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        for i in (chunks * 8)..hidden {
            let v = *row_ptr.add(i);
            sum_sq += v * v;
        }

        let rms = (sum_sq * inv_hidden + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let vinv_rms = _mm256_set1_ps(inv_rms);

        // --- Step 2: Scale by gamma / rms using SIMD ---
        for c in 0..chunks {
            let off = c * 8;
            // SAFETY: off + 7 < chunks * 8 <= hidden, within both row and gamma bounds.
            let v = _mm256_loadu_ps(row_ptr.add(off) as *const f32);
            let g = _mm256_loadu_ps(gamma_ptr.add(off) as *const f32);
            let result = _mm256_mul_ps(_mm256_mul_ps(v, vinv_rms), g);
            _mm256_storeu_ps(row_ptr.add(off), result);
        }
        for i in (chunks * 8)..hidden {
            let v = *row_ptr.add(i);
            let g = *gamma_ptr.add(i);
            *row_ptr.add(i) = v * inv_rms * g;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn silu_inplace_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = x.len();
    let chunks = n / 8;
    let ptr = x.as_mut_ptr();
    let vone = _mm256_set1_ps(1.0);

    for c in 0..chunks {
        let off = c * 8;
        // SAFETY: off + 7 < chunks * 8 <= n, within slice bounds.
        let v = _mm256_loadu_ps(ptr.add(off) as *const f32);
        // sigmoid(v) = 1 / (1 + exp(-v))
        let neg_v = _mm256_xor_ps(v, _mm256_set1_ps(-0.0));
        let exp_neg_v = fast_exp_avx2(neg_v);
        let sigmoid = _mm256_div_ps(vone, _mm256_add_ps(vone, exp_neg_v));
        // silu = v * sigmoid(v)
        let result = _mm256_mul_ps(v, sigmoid);
        _mm256_storeu_ps(ptr.add(off), result);
    }
    for i in (chunks * 8)..n {
        let v = *ptr.add(i);
        *ptr.add(i) = v * (1.0 / (1.0 + (-v).exp()));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn elementwise_mul_avx2(a: &mut [f32], b: &[f32]) {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let a_ptr = a.as_mut_ptr();
    let b_ptr = b.as_ptr();

    for c in 0..chunks {
        let off = c * 8;
        // SAFETY: off + 7 < chunks * 8 <= n, within both slice bounds.
        let av = _mm256_loadu_ps(a_ptr.add(off) as *const f32);
        let bv = _mm256_loadu_ps(b_ptr.add(off) as *const f32);
        _mm256_storeu_ps(a_ptr.add(off), _mm256_mul_ps(av, bv));
    }
    for i in (chunks * 8)..n {
        *a_ptr.add(i) *= *b_ptr.add(i);
    }
}
