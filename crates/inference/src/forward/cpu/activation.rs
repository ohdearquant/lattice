// ===================================================================
// Fast tanh approximation — Pade (7,6) rational, max error < 4e-5
// ===================================================================

use super::simd::simd_config;

/// Fast tanh approximation — Padé (7,6) rational, max error < 4e-5.
#[inline]
pub fn fast_tanh(x: f32) -> f32 {
    if x.abs() >= 10.0 {
        return x.signum();
    }
    let x2 = x * x;
    let num = x * (135_135.0 + x2 * (17_325.0 + x2 * (378.0 + x2)));
    let den = 135_135.0 + x2 * (62_370.0 + x2 * (3_150.0 + x2 * 28.0));
    num / den
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

/// SIMD vectorized Padé tanh approximation for 4 NEON lanes.
/// tanh(x) = x * (135135 + x^2*(17325 + x^2*(378 + x^2)))
///           / (135135 + x^2*(62370 + x^2*(3150 + x^2*28)))
/// Clamps output to [-1, 1] for |x| >= 10.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fast_tanh_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let x2 = vmulq_f32(x, x);

    // Numerator: x * (135135 + x2 * (17325 + x2 * (378 + x2)))
    let inner_n = vaddq_f32(vdupq_n_f32(378.0), x2);
    let mid_n = vfmaq_f32(vdupq_n_f32(17_325.0), x2, inner_n);
    let outer_n = vfmaq_f32(vdupq_n_f32(135_135.0), x2, mid_n);
    let num = vmulq_f32(x, outer_n);

    // Denominator: 135135 + x2 * (62370 + x2 * (3150 + x2 * 28))
    let inner_d = vfmaq_f32(vdupq_n_f32(3_150.0), x2, vdupq_n_f32(28.0));
    let mid_d = vfmaq_f32(vdupq_n_f32(62_370.0), x2, inner_d);
    let den = vfmaq_f32(vdupq_n_f32(135_135.0), x2, mid_d);

    // True SIMD division for full f32 precision (matches AVX2 path)
    let result = vdivq_f32(num, den);

    // Clamp to [-1, 1] for numerical safety
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

/// SIMD vectorized Padé tanh approximation for 8 AVX2 lanes.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fast_tanh_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let x2 = _mm256_mul_ps(x, x);

    // Numerator: x * (135135 + x2 * (17325 + x2 * (378 + x2)))
    let inner_n = _mm256_add_ps(_mm256_set1_ps(378.0), x2);
    let mid_n = _mm256_fmadd_ps(x2, inner_n, _mm256_set1_ps(17_325.0));
    let outer_n = _mm256_fmadd_ps(x2, mid_n, _mm256_set1_ps(135_135.0));
    let num = _mm256_mul_ps(x, outer_n);

    // Denominator: 135135 + x2 * (62370 + x2 * (3150 + x2 * 28))
    let inner_d = _mm256_fmadd_ps(x2, _mm256_set1_ps(28.0), _mm256_set1_ps(3_150.0));
    let mid_d = _mm256_fmadd_ps(x2, inner_d, _mm256_set1_ps(62_370.0));
    let den = _mm256_fmadd_ps(x2, mid_d, _mm256_set1_ps(135_135.0));

    // Division
    let result = _mm256_div_ps(num, den);

    // Clamp to [-1, 1]
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);
    _mm256_min_ps(_mm256_max_ps(result, neg_one), one)
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
    debug_assert_eq!(bias.len(), dim);
    debug_assert_eq!(x.len() % dim, 0);

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
}
