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
    assert!(hidden > 0, "hidden must be > 0");
    assert!(
        x.len() % hidden == 0,
        "x.len() must be a multiple of hidden"
    );
    assert!(gamma.len() == hidden, "gamma.len() must equal hidden");

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
    assert!(a.len() == b.len(), "a.len() must equal b.len()");

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
// NEON implementations — 4-wide float32x4_t
// ===================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rms_norm_neon(x: &mut [f32], gamma: &[f32], hidden: usize, eps: f32) {
    use std::arch::aarch64::*;

    let num_tokens = x.len() / hidden;
    debug_assert_eq!(x.len(), num_tokens * hidden);
    debug_assert_eq!(gamma.len(), hidden);

    const UNROLL: usize = 4;
    const CHUNK: usize = 4 * UNROLL; // 16 floats per unrolled iteration
    let chunks = hidden / CHUNK;
    let inv_hidden = 1.0f32 / hidden as f32;

    for t in 0..num_tokens {
        let row_ptr = x.as_mut_ptr().add(t * hidden);
        let gamma_ptr = gamma.as_ptr();

        // --- Step 1: Sum of squares with 4 independent accumulators ---
        let mut sq0 = vdupq_n_f32(0.0);
        let mut sq1 = vdupq_n_f32(0.0);
        let mut sq2 = vdupq_n_f32(0.0);
        let mut sq3 = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let base = c * CHUNK;
            let v0 = vld1q_f32(row_ptr.add(base) as *const f32);
            sq0 = vfmaq_f32(sq0, v0, v0);
            let v1 = vld1q_f32(row_ptr.add(base + 4) as *const f32);
            sq1 = vfmaq_f32(sq1, v1, v1);
            let v2 = vld1q_f32(row_ptr.add(base + 8) as *const f32);
            sq2 = vfmaq_f32(sq2, v2, v2);
            let v3 = vld1q_f32(row_ptr.add(base + 12) as *const f32);
            sq3 = vfmaq_f32(sq3, v3, v3);
        }
        let combined = vaddq_f32(vaddq_f32(sq0, sq1), vaddq_f32(sq2, sq3));
        let mut sum_sq = vaddvq_f32(combined);
        for i in (chunks * CHUNK)..hidden {
            let v = *row_ptr.add(i);
            sum_sq += v * v;
        }

        let rms = (sum_sq * inv_hidden + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let vinv_rms = vdupq_n_f32(inv_rms);

        // --- Step 2: Scale by gamma / rms with 4x unrolling ---
        for c in 0..chunks {
            let base = c * CHUNK;
            let v0 = vld1q_f32(row_ptr.add(base) as *const f32);
            let g0 = vld1q_f32(gamma_ptr.add(base));
            vst1q_f32(row_ptr.add(base), vmulq_f32(vmulq_f32(v0, vinv_rms), g0));
            let v1 = vld1q_f32(row_ptr.add(base + 4) as *const f32);
            let g1 = vld1q_f32(gamma_ptr.add(base + 4));
            vst1q_f32(
                row_ptr.add(base + 4),
                vmulq_f32(vmulq_f32(v1, vinv_rms), g1),
            );
            let v2 = vld1q_f32(row_ptr.add(base + 8) as *const f32);
            let g2 = vld1q_f32(gamma_ptr.add(base + 8));
            vst1q_f32(
                row_ptr.add(base + 8),
                vmulq_f32(vmulq_f32(v2, vinv_rms), g2),
            );
            let v3 = vld1q_f32(row_ptr.add(base + 12) as *const f32);
            let g3 = vld1q_f32(gamma_ptr.add(base + 12));
            vst1q_f32(
                row_ptr.add(base + 12),
                vmulq_f32(vmulq_f32(v3, vinv_rms), g3),
            );
        }
        for i in (chunks * CHUNK)..hidden {
            let v = *row_ptr.add(i);
            let g = *gamma_ptr.add(i);
            *row_ptr.add(i) = v * inv_rms * g;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_exp_f32(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_145_75_f32;
    const LN2_LO: f32 = 1.428_606_8e-6_f32;

    let x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(88.72)), vdupq_n_f32(-87.33));

    // Range reduction: x = n*ln(2) + r, |r| <= ln(2)/2
    // Round to nearest via magic-number trick (avoids unstable intrinsics)
    let magic = vdupq_n_f32(12_582_912.0); // 1.5 * 2^23
    let biased = vaddq_f32(vmulq_f32(x, vdupq_n_f32(LOG2E)), magic);
    let n = vsubq_f32(biased, magic);
    let n_int = vcvtq_s32_f32(n);

    let r = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(LN2_HI)), n, vdupq_n_f32(LN2_LO));

    // exp(r) via Horner: 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120 + r/720)))))
    let p = vdupq_n_f32(1.0 / 720.0);
    let p = vfmaq_f32(vdupq_n_f32(1.0 / 120.0), p, r);
    let p = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), p, r);
    let p = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), p, r);
    let p = vfmaq_f32(vdupq_n_f32(0.5), p, r);
    let p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
    let p = vfmaq_f32(vdupq_n_f32(1.0), p, r);

    // Reconstruct: exp(x) = 2^n * exp(r)
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n_int, vdupq_n_s32(127))));
    vmulq_f32(p, pow2n)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn silu_inplace_neon(x: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = x.len();
    const UNROLL: usize = 4;
    const CHUNK: usize = 4 * UNROLL;
    let chunks = n / CHUNK;
    let ptr = x.as_mut_ptr();
    let vone = vdupq_n_f32(1.0);

    for c in 0..chunks {
        let base = c * CHUNK;

        let v0 = vld1q_f32(ptr.add(base) as *const f32);
        let v1 = vld1q_f32(ptr.add(base + 4) as *const f32);
        let v2 = vld1q_f32(ptr.add(base + 8) as *const f32);
        let v3 = vld1q_f32(ptr.add(base + 12) as *const f32);

        let e0 = neon_exp_f32(vnegq_f32(v0));
        let e1 = neon_exp_f32(vnegq_f32(v1));
        let e2 = neon_exp_f32(vnegq_f32(v2));
        let e3 = neon_exp_f32(vnegq_f32(v3));

        let s0 = vdivq_f32(vone, vaddq_f32(vone, e0));
        let s1 = vdivq_f32(vone, vaddq_f32(vone, e1));
        let s2 = vdivq_f32(vone, vaddq_f32(vone, e2));
        let s3 = vdivq_f32(vone, vaddq_f32(vone, e3));

        vst1q_f32(ptr.add(base), vmulq_f32(v0, s0));
        vst1q_f32(ptr.add(base + 4), vmulq_f32(v1, s1));
        vst1q_f32(ptr.add(base + 8), vmulq_f32(v2, s2));
        vst1q_f32(ptr.add(base + 12), vmulq_f32(v3, s3));
    }

    // 4-wide SIMD remainder
    let remaining = chunks * CHUNK;
    let simd_tail = (n - remaining) / 4;
    for c in 0..simd_tail {
        let off = remaining + c * 4;
        let v = vld1q_f32(ptr.add(off) as *const f32);
        let sig = vdivq_f32(vone, vaddq_f32(vone, neon_exp_f32(vnegq_f32(v))));
        vst1q_f32(ptr.add(off), vmulq_f32(v, sig));
    }

    for i in (remaining + simd_tail * 4)..n {
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
    const UNROLL: usize = 4;
    const CHUNK: usize = 4 * UNROLL; // 16 floats per iteration
    let chunks = n / CHUNK;
    let a_ptr = a.as_mut_ptr();
    let b_ptr = b.as_ptr();

    for c in 0..chunks {
        let base = c * CHUNK;
        let a0 = vld1q_f32(a_ptr.add(base) as *const f32);
        let b0 = vld1q_f32(b_ptr.add(base));
        vst1q_f32(a_ptr.add(base), vmulq_f32(a0, b0));
        let a1 = vld1q_f32(a_ptr.add(base + 4) as *const f32);
        let b1 = vld1q_f32(b_ptr.add(base + 4));
        vst1q_f32(a_ptr.add(base + 4), vmulq_f32(a1, b1));
        let a2 = vld1q_f32(a_ptr.add(base + 8) as *const f32);
        let b2 = vld1q_f32(b_ptr.add(base + 8));
        vst1q_f32(a_ptr.add(base + 8), vmulq_f32(a2, b2));
        let a3 = vld1q_f32(a_ptr.add(base + 12) as *const f32);
        let b3 = vld1q_f32(b_ptr.add(base + 12));
        vst1q_f32(a_ptr.add(base + 12), vmulq_f32(a3, b3));
    }
    for i in (chunks * CHUNK)..n {
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
        // sigmoid(v) = 1 / (1 + exp(-v)); use accurate exp for numerical correctness.
        let neg_v = _mm256_xor_ps(v, _mm256_set1_ps(-0.0));
        let neg_arr: [f32; 8] = core::mem::transmute(neg_v);
        let exp_arr = [
            neg_arr[0].exp(),
            neg_arr[1].exp(),
            neg_arr[2].exp(),
            neg_arr[3].exp(),
            neg_arr[4].exp(),
            neg_arr[5].exp(),
            neg_arr[6].exp(),
            neg_arr[7].exp(),
        ];
        let exp_neg_v: __m256 = core::mem::transmute(exp_arr);
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
