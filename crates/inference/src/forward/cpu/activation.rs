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
mod guard_tests {
    use super::*;

    #[test]
    fn add_bias_gelu_accepts_valid_lengths() {
        let dim = 4;
        let bias = vec![0.1, -0.2, 0.3, -0.4];
        let mut x = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0, 1.5]; // 2 rows × dim
        add_bias_gelu(&mut x, &bias, dim);
        assert_eq!(x.len(), 8);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "bias length must equal dim")]
    fn add_bias_gelu_rejects_short_bias() {
        // dim=4 needs bias.len()==4; a short bias would OOB the unsafe SIMD kernel.
        let mut x = vec![0.0; 8];
        let bias = vec![0.0; 3];
        add_bias_gelu(&mut x, &bias, 4);
    }

    #[test]
    #[should_panic(expected = "bias length must equal dim")]
    fn add_bias_rejects_short_bias() {
        // A short bias previously truncated silently (chunks_exact + zip),
        // leaving the trailing lanes unbiased. Fail closed in release now.
        let mut x = vec![0.0; 8];
        let bias = vec![0.0; 3];
        add_bias(&mut x, &bias, 4);
    }

    #[test]
    #[should_panic(expected = "multiple of dim")]
    fn add_bias_rejects_ragged_x() {
        // x.len() not a multiple of dim previously dropped the remainder row.
        let mut x = vec![0.0; 7];
        let bias = vec![0.0; 4];
        add_bias(&mut x, &bias, 4);
    }

    #[test]
    fn add_bias_accepts_valid_lengths() {
        let dim = 4;
        let bias = vec![0.1, -0.2, 0.3, -0.4];
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 rows × dim
        add_bias(&mut x, &bias, dim);
        assert_eq!(x, vec![1.1, 1.8, 3.3, 3.6, 5.1, 5.8, 7.3, 7.6]);
    }
}

#[cfg(test)]
mod fast_tanh_tests {
    use super::*;

    /// Absolute error bound of `fast_tanh` against `f64::tanh`, over every sweep input.
    const TANH_BOUND: f64 = 1e-6;

    /// Inputs: a dense grid over [-12, 12], the `f32` neighbours of the clamp points,
    /// zeros, tiny, subnormal and huge magnitudes, and a stride over every finite `f32`
    /// bit pattern (all exponents, both signs).
    fn sweep_inputs() -> Vec<f32> {
        let mut xs: Vec<f32> = (-120_000..=120_000)
            .map(|i| (f64::from(i) * 1e-4) as f32)
            .collect();
        for c in [TANH_CLAMP, -TANH_CLAMP] {
            let bits = c.to_bits();
            xs.extend([c, f32::from_bits(bits - 1), f32::from_bits(bits + 1)]);
        }
        xs.extend([
            0.0,
            -0.0,
            1e-30,
            -1e-30,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
            1e10,
            -1e10,
            f32::MAX,
            f32::MIN,
        ]);
        xs.extend(
            (0..=u32::MAX)
                .step_by(4099)
                .map(f32::from_bits)
                .filter(|v| v.is_finite()),
        );
        xs
    }

    fn gelu_f64(v: f32) -> f64 {
        let v = f64::from(v);
        let inner = (2.0 / std::f64::consts::PI).sqrt() * (v + 0.044_715 * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    }

    /// Runs the SIMD `fast_tanh` over `xs` in full vectors, zero-padding the last vector so
    /// remainder lanes go through the same kernel. `None` when the CPU lacks the feature.
    #[cfg(target_arch = "x86_64")]
    fn simd_fast_tanh(xs: &[f32]) -> Option<Vec<f32>> {
        use std::arch::x86_64::*;
        let config = simd_config();
        if !(config.avx2_enabled && config.fma_enabled) {
            return None;
        }
        let mut out = vec![0.0f32; xs.len()];
        for (src, dst) in xs.chunks(8).zip(out.chunks_mut(8)) {
            let mut lane = [0.0f32; 8];
            lane[..src.len()].copy_from_slice(src);
            // SAFETY: AVX2 and FMA were detected above; `lane` holds 8 f32.
            unsafe {
                let r = fast_tanh_avx2(_mm256_loadu_ps(lane.as_ptr()));
                _mm256_storeu_ps(lane.as_mut_ptr(), r);
            }
            dst.copy_from_slice(&lane[..src.len()]);
        }
        Some(out)
    }

    #[cfg(target_arch = "aarch64")]
    fn simd_fast_tanh(xs: &[f32]) -> Option<Vec<f32>> {
        use std::arch::aarch64::*;
        if !simd_config().neon_enabled {
            return None;
        }
        let mut out = vec![0.0f32; xs.len()];
        for (src, dst) in xs.chunks(4).zip(out.chunks_mut(4)) {
            let mut lane = [0.0f32; 4];
            lane[..src.len()].copy_from_slice(src);
            // SAFETY: NEON is available on aarch64; `lane` holds 4 f32.
            unsafe {
                let r = fast_tanh_neon(vld1q_f32(lane.as_ptr()));
                vst1q_f32(lane.as_mut_ptr(), r);
            }
            dst.copy_from_slice(&lane[..src.len()]);
        }
        Some(out)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn simd_fast_tanh(_xs: &[f32]) -> Option<Vec<f32>> {
        None
    }

    #[test]
    fn fast_tanh_scalar_matches_f64_tanh_over_full_range() {
        let xs = sweep_inputs();
        let mut worst = (0.0f64, 0.0f32);
        for &x in &xs {
            let y = fast_tanh(x);
            let err = (f64::from(y) - f64::from(x).tanh()).abs();
            assert!(
                err <= TANH_BOUND,
                "fast_tanh({x:e}) = {y:e}, f64 tanh = {:e}, abs_err = {err:e} (bound {TANH_BOUND:e})",
                f64::from(x).tanh(),
            );
            assert!(y.abs() <= 1.0, "fast_tanh({x:e}) = {y:e} leaves [-1, 1]");
            if err > worst.0 {
                worst = (err, x);
            }
        }
        eprintln!(
            "fast_tanh scalar: {} inputs, max abs err {:e} at x = {:e}",
            xs.len(),
            worst.0,
            worst.1
        );
    }

    #[test]
    fn fast_tanh_special_values() {
        assert_eq!(fast_tanh(f32::INFINITY), 1.0);
        assert_eq!(fast_tanh(f32::NEG_INFINITY), -1.0);
        assert!(fast_tanh(f32::NAN).is_nan(), "NaN must propagate");
        let pz = fast_tanh(0.0);
        assert!(pz == 0.0 && pz.is_sign_positive());
        let nz = fast_tanh(-0.0);
        assert!(nz == 0.0 && nz.is_sign_negative());
        for k in 1..=40 {
            let x = k as f32 * 0.25;
            assert_eq!(fast_tanh(-x), -fast_tanh(x), "odd symmetry at {x}");
        }
    }

    #[test]
    fn fast_tanh_simd_matches_scalar_and_f64() {
        let xs = sweep_inputs();
        let Some(simd) = simd_fast_tanh(&xs) else {
            eprintln!("fast_tanh SIMD path unavailable on this CPU; skipped");
            return;
        };
        let mut worst_vs_scalar = (0.0f32, 0.0f32);
        let mut worst_vs_f64 = (0.0f64, 0.0f32);
        for (&x, &y) in xs.iter().zip(&simd) {
            let err64 = (f64::from(y) - f64::from(x).tanh()).abs();
            if err64 > worst_vs_f64.0 {
                worst_vs_f64 = (err64, x);
            }
            assert!(
                err64 <= TANH_BOUND,
                "simd fast_tanh({x:e}) = {y:e}, abs_err vs f64 = {err64:e}"
            );
            let d = (y - fast_tanh(x)).abs();
            assert!(
                d <= 1e-6,
                "simd vs scalar at {x:e}: {y:e} vs {:e}",
                fast_tanh(x)
            );
            if d > worst_vs_scalar.0 {
                worst_vs_scalar = (d, x);
            }
        }
        eprintln!(
            "fast_tanh simd vs scalar: max diff {:e} at x = {:e}; vs f64: max abs err {:e} at x = {:e}",
            worst_vs_scalar.0, worst_vs_scalar.1, worst_vs_f64.0, worst_vs_f64.1
        );
    }

    #[test]
    fn fast_tanh_simd_remainder_lanes_and_special_values() {
        let pool = [
            -1e10,
            -9.5,
            -9.0,
            -3.25,
            -0.5,
            -1e-30,
            0.0,
            1e-30,
            0.5,
            3.25,
            9.0,
            9.5,
            1e10,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.125,
            -0.125,
        ];
        for n in 0..=pool.len() {
            let Some(simd) = simd_fast_tanh(&pool[..n]) else {
                return;
            };
            for (&x, &y) in pool[..n].iter().zip(&simd) {
                assert!((y - fast_tanh(x)).abs() <= 1e-6, "len {n}, x = {x:e}");
            }
        }
        let nan = simd_fast_tanh(&[f32::NAN, 1.0, f32::NAN]).unwrap();
        assert!(
            nan[0].is_nan() && nan[2].is_nan(),
            "NaN lanes must propagate"
        );
        assert!((nan[1] - 1.0f32.tanh()).abs() <= 1e-6);
        let signed_zero = simd_fast_tanh(&[-0.0]).unwrap();
        assert!(signed_zero[0] == 0.0 && signed_zero[0].is_sign_negative());
    }

    #[test]
    fn gelu_matches_f64_tanh_form_over_minus8_to_8() {
        // 16001 values: not a multiple of 8 or 4, so the SIMD remainder loops run.
        let vs: Vec<f32> = (-8000..=8000).map(|i| i as f32 * 1e-3).collect();
        assert_ne!(vs.len() % 8, 0);
        assert_ne!(vs.len() % 4, 0);

        let mut dispatched = vs.clone();
        gelu(&mut dispatched);
        let mut scalar = vs.clone();
        gelu_scalar(&mut scalar);
        let mut fused = vs.clone();
        let zero_bias = vec![0.0f32; vs.len()];
        add_bias_gelu(&mut fused, &zero_bias, vs.len());

        let mut worst = 0.0f64;
        for (i, &v) in vs.iter().enumerate() {
            let expected = gelu_f64(v);
            for (name, got) in [
                ("gelu", dispatched[i]),
                ("gelu_scalar", scalar[i]),
                ("add_bias_gelu", fused[i]),
            ] {
                let err = (f64::from(got) - expected).abs();
                assert!(
                    err <= 2e-6,
                    "{name}({v}) = {got:e}, f64 reference {expected:e}, abs_err = {err:e}"
                );
                worst = worst.max(err);
            }
        }
        eprintln!("gelu over [-8, 8]: max abs err {worst:e}");
    }
}
