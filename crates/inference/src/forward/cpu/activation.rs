//! CPU activation and bias helpers, including tanh, GELU, add-bias, and fused add-bias-GELU paths.
// ===================================================================
// Fast tanh approximation: odd 13/6 rational, single precision
// ===================================================================

use super::simd::simd_config;

// Coefficients of the single-precision tanh rational used by Eigen
// (`generic_fast_tanh_float` in Eigen/src/Core/MathFunctionsImpl.h, MPL-2.0): a
// degree-13 odd numerator over a degree-6 even denominator, accurate to a couple
// of ulp on [-9, 9]. The scalar, NEON and AVX2 paths all evaluate this same
// formula.
const TANH_ALPHA_1: f32 = 4.8935246e-03;
const TANH_ALPHA_3: f32 = 6.3726195e-04;
const TANH_ALPHA_5: f32 = 1.48572235e-05;
const TANH_ALPHA_7: f32 = 5.1222973e-08;
const TANH_ALPHA_9: f32 = -8.604672e-11;
const TANH_ALPHA_11: f32 = 2.000188e-13;
const TANH_ALPHA_13: f32 = -2.7607684e-16;
const TANH_BETA_0: f32 = 4.893525e-03;
const TANH_BETA_2: f32 = 2.2684347e-03;
const TANH_BETA_4: f32 = 1.1853471e-04;
const TANH_BETA_6: f32 = 1.1982584e-06;

/// Inputs are clamped to `[-TANH_CLAMP, TANH_CLAMP]` before the rational is evaluated;
/// beyond it `tanh` is within 3e-8 of ±1.
const TANH_CLAMP: f32 = 9.0;

/// Fast tanh approximation: a 13/6 rational with inputs clamped to [-9, 9].
///
/// Max absolute error against `f64::tanh` is below 1e-6 over every finite `f32` input
/// (measured at 3.5e-7 on a dense sweep of [-12, 12] plus a stride sweep of all finite
/// `f32` bit patterns). The result is clamped to [-1, 1]. `NaN` propagates; `±inf` maps
/// to `±1`; `±0` is preserved.
#[inline]
pub fn fast_tanh(x: f32) -> f32 {
    // `f32::clamp` returns NaN for a NaN input.
    let x = x.clamp(-TANH_CLAMP, TANH_CLAMP);
    let x2 = x * x;
    let mut p = TANH_ALPHA_13;
    p = x2 * p + TANH_ALPHA_11;
    p = x2 * p + TANH_ALPHA_9;
    p = x2 * p + TANH_ALPHA_7;
    p = x2 * p + TANH_ALPHA_5;
    p = x2 * p + TANH_ALPHA_3;
    p = x2 * p + TANH_ALPHA_1;
    let num = x * p;
    let mut q = TANH_BETA_6;
    q = x2 * q + TANH_BETA_4;
    q = x2 * q + TANH_BETA_2;
    q = x2 * q + TANH_BETA_0;
    (num / q).clamp(-1.0, 1.0)
}

// ===================================================================
// GELU activation (in-place) — with SIMD fast path
// ===================================================================

/// **Unstable**: approximate GELU in-place; approximation polynomial may change.
///
/// Approximate GELU activation (in-place).
pub fn gelu(x: &mut [f32]) {
    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is available on aarch64 and the runtime gate ensures this path.
            unsafe {
                gelu_neon(x);
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled && config.fma_enabled {
            // SAFETY: The runtime feature checks above guarantee AVX2+FMA support.
            unsafe {
                gelu_avx2(x);
                return;
            }
        }
    }

    gelu_scalar(x);
}

#[inline]
pub fn gelu_scalar(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;

    for val in x.iter_mut() {
        let x3 = *val * *val * *val;
        let inner = SQRT_2_OVER_PI * (*val + COEFF * x3);
        *val = 0.5 * *val * (1.0 + fast_tanh(inner));
    }
}

/// SIMD vectorized form of [`fast_tanh`] for 4 NEON lanes (same rational, FMA evaluation).
/// Clamps the input to [-9, 9] and the output to [-1, 1]; `NaN` lanes propagate.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fast_tanh_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    // vminq/vmaxq propagate NaN.
    let x = vminq_f32(
        vmaxq_f32(x, vdupq_n_f32(-TANH_CLAMP)),
        vdupq_n_f32(TANH_CLAMP),
    );
    let x2 = vmulq_f32(x, x);

    let mut p = vdupq_n_f32(TANH_ALPHA_13);
    p = vfmaq_f32(vdupq_n_f32(TANH_ALPHA_11), x2, p);
    p = vfmaq_f32(vdupq_n_f32(TANH_ALPHA_9), x2, p);
    p = vfmaq_f32(vdupq_n_f32(TANH_ALPHA_7), x2, p);
    p = vfmaq_f32(vdupq_n_f32(TANH_ALPHA_5), x2, p);
    p = vfmaq_f32(vdupq_n_f32(TANH_ALPHA_3), x2, p);
    p = vfmaq_f32(vdupq_n_f32(TANH_ALPHA_1), x2, p);
    let num = vmulq_f32(x, p);

    let mut q = vdupq_n_f32(TANH_BETA_6);
    q = vfmaq_f32(vdupq_n_f32(TANH_BETA_4), x2, q);
    q = vfmaq_f32(vdupq_n_f32(TANH_BETA_2), x2, q);
    q = vfmaq_f32(vdupq_n_f32(TANH_BETA_0), x2, q);

    let result = vdivq_f32(num, q);

    let one = vdupq_n_f32(1.0);
    let neg_one = vdupq_n_f32(-1.0);
    vminq_f32(vmaxq_f32(result, neg_one), one)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gelu_neon(x: &mut [f32]) {
    use std::arch::aarch64::*;

    let sqrt_2_over_pi = vdupq_n_f32(0.797_884_6);
    let coeff = vdupq_n_f32(0.044_715);
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);

    let n = x.len();
    const UNROLL: usize = 4;
    const CHUNK: usize = 4 * UNROLL;
    let chunks = n / CHUNK;
    let ptr = x.as_mut_ptr();

    for c in 0..chunks {
        let base = c * CHUNK;

        let v0 = vld1q_f32(ptr.add(base) as *const f32);
        let v1 = vld1q_f32(ptr.add(base + 4) as *const f32);
        let v2 = vld1q_f32(ptr.add(base + 8) as *const f32);
        let v3 = vld1q_f32(ptr.add(base + 12) as *const f32);

        let i0 = vmulq_f32(
            sqrt_2_over_pi,
            vfmaq_f32(v0, coeff, vmulq_f32(vmulq_f32(v0, v0), v0)),
        );
        let i1 = vmulq_f32(
            sqrt_2_over_pi,
            vfmaq_f32(v1, coeff, vmulq_f32(vmulq_f32(v1, v1), v1)),
        );
        let i2 = vmulq_f32(
            sqrt_2_over_pi,
            vfmaq_f32(v2, coeff, vmulq_f32(vmulq_f32(v2, v2), v2)),
        );
        let i3 = vmulq_f32(
            sqrt_2_over_pi,
            vfmaq_f32(v3, coeff, vmulq_f32(vmulq_f32(v3, v3), v3)),
        );

        let t0 = fast_tanh_neon(i0);
        let t1 = fast_tanh_neon(i1);
        let t2 = fast_tanh_neon(i2);
        let t3 = fast_tanh_neon(i3);

        vst1q_f32(
            ptr.add(base),
            vmulq_f32(vmulq_f32(half, v0), vaddq_f32(one, t0)),
        );
        vst1q_f32(
            ptr.add(base + 4),
            vmulq_f32(vmulq_f32(half, v1), vaddq_f32(one, t1)),
        );
        vst1q_f32(
            ptr.add(base + 8),
            vmulq_f32(vmulq_f32(half, v2), vaddq_f32(one, t2)),
        );
        vst1q_f32(
            ptr.add(base + 12),
            vmulq_f32(vmulq_f32(half, v3), vaddq_f32(one, t3)),
        );
    }

    let remaining = chunks * CHUNK;
    let simd_tail = (n - remaining) / 4;
    for c in 0..simd_tail {
        let off = remaining + c * 4;
        let v = vld1q_f32(ptr.add(off) as *const f32);
        let inner = vmulq_f32(
            sqrt_2_over_pi,
            vfmaq_f32(v, coeff, vmulq_f32(vmulq_f32(v, v), v)),
        );
        let result = vmulq_f32(vmulq_f32(half, v), vaddq_f32(one, fast_tanh_neon(inner)));
        vst1q_f32(ptr.add(off), result);
    }

    for i in (remaining + simd_tail * 4)..n {
        let v = *ptr.add(i);
        let x3 = v * v * v;
        let inner = 0.797_884_6 * (v + 0.044_715 * x3);
        *ptr.add(i) = 0.5 * v * (1.0 + fast_tanh(inner));
    }
}

/// SIMD vectorized form of [`fast_tanh`] for 8 AVX2 lanes (same rational, FMA evaluation).
/// Clamps the input to [-9, 9] and the output to [-1, 1]; `NaN` lanes propagate.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fast_tanh_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    // min/max return their second operand when either input is NaN, so the value
    // being clamped goes second to let NaN propagate.
    let x = _mm256_max_ps(
        _mm256_set1_ps(-TANH_CLAMP),
        _mm256_min_ps(_mm256_set1_ps(TANH_CLAMP), x),
    );
    let x2 = _mm256_mul_ps(x, x);

    let mut p = _mm256_set1_ps(TANH_ALPHA_13);
    p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(TANH_ALPHA_11));
    p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(TANH_ALPHA_9));
    p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(TANH_ALPHA_7));
    p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(TANH_ALPHA_5));
    p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(TANH_ALPHA_3));
    p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(TANH_ALPHA_1));
    let num = _mm256_mul_ps(x, p);

    let mut q = _mm256_set1_ps(TANH_BETA_6);
    q = _mm256_fmadd_ps(x2, q, _mm256_set1_ps(TANH_BETA_4));
    q = _mm256_fmadd_ps(x2, q, _mm256_set1_ps(TANH_BETA_2));
    q = _mm256_fmadd_ps(x2, q, _mm256_set1_ps(TANH_BETA_0));

    let result = _mm256_div_ps(num, q);

    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);
    _mm256_max_ps(neg_one, _mm256_min_ps(one, result))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gelu_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;

    let sqrt_2_over_pi = _mm256_set1_ps(0.797_884_6);
    let coeff = _mm256_set1_ps(0.044_715);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);

    let chunks = x.len() / 8;
    let ptr = x.as_mut_ptr();

    for c in 0..chunks {
        let off = c * 8;
        let v = _mm256_loadu_ps(ptr.add(off) as *const f32);
        let v2 = _mm256_mul_ps(v, v);
        let v3 = _mm256_mul_ps(v2, v);

        // inner = sqrt_2_over_pi * (v + coeff * v^3)
        let inner = _mm256_mul_ps(sqrt_2_over_pi, _mm256_fmadd_ps(coeff, v3, v));

        let tanh_val = fast_tanh_avx2(inner);

        // gelu = 0.5 * v * (1 + tanh(inner))
        let result = _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, tanh_val));
        _mm256_storeu_ps(ptr.add(off), result);
    }

    // Scalar remainder
    for i in (chunks * 8)..x.len() {
        let v = *ptr.add(i);
        let x3 = v * v * v;
        let inner = 0.797_884_6 * (v + 0.044_715 * x3);
        *ptr.add(i) = 0.5 * v * (1.0 + fast_tanh(inner));
    }
}

// ===================================================================
// Add bias (in-place)
// ===================================================================

/// **Unstable**: add bias to each row in-place; may be merged with downstream operations.
///
/// Add bias to each row of a matrix (in-place).
pub fn add_bias(x: &mut [f32], bias: &[f32], dim: usize) {
    assert_eq!(bias.len(), dim, "add_bias: bias length must equal dim");
    assert_eq!(
        x.len() % dim,
        0,
        "add_bias: x length must be a multiple of dim"
    );

    for row in x.chunks_exact_mut(dim) {
        for (val, &b) in row.iter_mut().zip(bias.iter()) {
            *val += b;
        }
    }
}

// ===================================================================
// Fused add_bias + GELU (in-place) — single pass over data
// ===================================================================

/// **Unstable**: fused bias+GELU; fusion strategy and SIMD dispatch may change.
///
/// Fused bias addition and GELU activation. Performs `x[i] = gelu(x[i] + bias[i % dim])`
/// in a single pass, saving one full traversal of the data compared to calling
/// `add_bias` then `gelu` separately.
pub fn add_bias_gelu(x: &mut [f32], bias: &[f32], dim: usize) {
    assert_eq!(
        bias.len(),
        dim,
        "add_bias_gelu: bias length must equal dim before the SIMD kernels index it"
    );
    assert_eq!(
        x.len() % dim,
        0,
        "add_bias_gelu: x length must be a multiple of dim"
    );

    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is available on aarch64 and the runtime gate ensures this path.
            unsafe {
                add_bias_gelu_neon(x, bias, dim);
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled && config.fma_enabled {
            // SAFETY: The runtime feature checks above guarantee AVX2+FMA support.
            unsafe {
                add_bias_gelu_avx2(x, bias, dim);
                return;
            }
        }
    }

    add_bias_gelu_scalar(x, bias, dim);
}

pub fn add_bias_gelu_scalar(x: &mut [f32], bias: &[f32], dim: usize) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;

    for row in x.chunks_exact_mut(dim) {
        for (val, &b) in row.iter_mut().zip(bias.iter()) {
            let v = *val + b;
            let x3 = v * v * v;
            let inner = SQRT_2_OVER_PI * (v + COEFF * x3);
            *val = 0.5 * v * (1.0 + fast_tanh(inner));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_bias_gelu_neon(x: &mut [f32], bias: &[f32], dim: usize) {
    use std::arch::aarch64::*;

    let sqrt_2_over_pi = vdupq_n_f32(0.797_884_6);
    let coeff = vdupq_n_f32(0.044_715);
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);

    const UNROLL: usize = 4;
    const CHUNK: usize = 4 * UNROLL;
    let chunks = dim / CHUNK;

    for row in x.chunks_exact_mut(dim) {
        let ptr = row.as_mut_ptr();
        let b_ptr = bias.as_ptr();

        for c in 0..chunks {
            let base = c * CHUNK;
            let v0 = vaddq_f32(
                vld1q_f32(ptr.add(base) as *const f32),
                vld1q_f32(b_ptr.add(base)),
            );
            let v1 = vaddq_f32(
                vld1q_f32(ptr.add(base + 4) as *const f32),
                vld1q_f32(b_ptr.add(base + 4)),
            );
            let v2 = vaddq_f32(
                vld1q_f32(ptr.add(base + 8) as *const f32),
                vld1q_f32(b_ptr.add(base + 8)),
            );
            let v3 = vaddq_f32(
                vld1q_f32(ptr.add(base + 12) as *const f32),
                vld1q_f32(b_ptr.add(base + 12)),
            );

            let i0 = vmulq_f32(
                sqrt_2_over_pi,
                vfmaq_f32(v0, coeff, vmulq_f32(vmulq_f32(v0, v0), v0)),
            );
            let i1 = vmulq_f32(
                sqrt_2_over_pi,
                vfmaq_f32(v1, coeff, vmulq_f32(vmulq_f32(v1, v1), v1)),
            );
            let i2 = vmulq_f32(
                sqrt_2_over_pi,
                vfmaq_f32(v2, coeff, vmulq_f32(vmulq_f32(v2, v2), v2)),
            );
            let i3 = vmulq_f32(
                sqrt_2_over_pi,
                vfmaq_f32(v3, coeff, vmulq_f32(vmulq_f32(v3, v3), v3)),
            );

            let t0 = fast_tanh_neon(i0);
            let t1 = fast_tanh_neon(i1);
            let t2 = fast_tanh_neon(i2);
            let t3 = fast_tanh_neon(i3);

            vst1q_f32(
                ptr.add(base),
                vmulq_f32(vmulq_f32(half, v0), vaddq_f32(one, t0)),
            );
            vst1q_f32(
                ptr.add(base + 4),
                vmulq_f32(vmulq_f32(half, v1), vaddq_f32(one, t1)),
            );
            vst1q_f32(
                ptr.add(base + 8),
                vmulq_f32(vmulq_f32(half, v2), vaddq_f32(one, t2)),
            );
            vst1q_f32(
                ptr.add(base + 12),
                vmulq_f32(vmulq_f32(half, v3), vaddq_f32(one, t3)),
            );
        }

        let remaining = chunks * CHUNK;
        let simd_tail = (dim - remaining) / 4;
        for c in 0..simd_tail {
            let off = remaining + c * 4;
            let v = vaddq_f32(
                vld1q_f32(ptr.add(off) as *const f32),
                vld1q_f32(b_ptr.add(off)),
            );
            let inner = vmulq_f32(
                sqrt_2_over_pi,
                vfmaq_f32(v, coeff, vmulq_f32(vmulq_f32(v, v), v)),
            );
            vst1q_f32(
                ptr.add(off),
                vmulq_f32(vmulq_f32(half, v), vaddq_f32(one, fast_tanh_neon(inner))),
            );
        }

        for i in (remaining + simd_tail * 4)..dim {
            let v = *ptr.add(i) + *b_ptr.add(i);
            let x3 = v * v * v;
            let inner = 0.797_884_6 * (v + 0.044_715 * x3);
            *ptr.add(i) = 0.5 * v * (1.0 + fast_tanh(inner));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn add_bias_gelu_avx2(x: &mut [f32], bias: &[f32], dim: usize) {
    use std::arch::x86_64::*;

    let sqrt_2_over_pi = _mm256_set1_ps(0.797_884_6);
    let coeff = _mm256_set1_ps(0.044_715);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);

    let chunks8 = dim / 8;

    for row in x.chunks_exact_mut(dim) {
        let ptr = row.as_mut_ptr();
        let b_ptr = bias.as_ptr();

        for c in 0..chunks8 {
            let off = c * 8;
            // Load x and bias, fuse add
            let v = _mm256_add_ps(
                _mm256_loadu_ps(ptr.add(off) as *const f32),
                _mm256_loadu_ps(b_ptr.add(off)),
            );

            // GELU: 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 * v^3)))
            let v2 = _mm256_mul_ps(v, v);
            let v3 = _mm256_mul_ps(v2, v);
            let inner = _mm256_mul_ps(sqrt_2_over_pi, _mm256_fmadd_ps(coeff, v3, v));
            let tanh_val = fast_tanh_avx2(inner);
            let result = _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, tanh_val));
            _mm256_storeu_ps(ptr.add(off), result);
        }

        // Scalar remainder
        for i in (chunks8 * 8)..dim {
            let v = *ptr.add(i) + *b_ptr.add(i);
            let x3 = v * v * v;
            let inner = 0.797_884_6 * (v + 0.044_715 * x3);
            *ptr.add(i) = 0.5 * v * (1.0 + fast_tanh(inner));
        }
    }
}

#[cfg(test)]
mod guard_tests;

#[cfg(test)]
mod fast_tanh_tests;
