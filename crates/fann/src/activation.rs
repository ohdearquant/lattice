//! Activation functions for dense networks.
//!
//! Element-wise and batch evaluation cover the common nonlinearities; batch
//! Softmax normalizes the entire layer. ReLU variants use SIMD when available.
//!
//! See `docs/network.md` for formulas, numerical rules, and derivative limits.

/// Supported activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Activation {
    /// Linear activation (identity function): f(x) = x
    Linear,

    /// Sigmoid activation: f(x) = 1 / (1 + e^(-x))
    Sigmoid,

    /// Hyperbolic tangent: f(x) = tanh(x)
    Tanh,

    /// Rectified Linear Unit: f(x) = max(0, x)
    #[default]
    ReLU,

    /// Leaky ReLU: f(x) = x if x > 0, else alpha * x
    LeakyReLU(f32),

    /// Softmax (applied to entire layer output)
    Softmax,
}

/// NEON-accelerated ReLU: max(0, x) for 4 elements at a time.
///
/// # Safety
/// Must run on AArch64, where NEON is mandatory.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
unsafe fn simd_relu_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = values.len();
    let p = values.as_mut_ptr();
    debug_assert_eq!((p as usize) % core::mem::align_of::<f32>(), 0);
    let zero = vdupq_n_f32(0.0);
    let chunks = n / 4;
    let mut offset = 0usize;
    for _ in 0..chunks {
        debug_assert!(offset + 4 <= n);
        // SAFETY: offset + 4 <= n; p is valid for n elements
        let v = vld1q_f32(p.add(offset));
        let result = vmaxq_f32(v, zero);
        vst1q_f32(p.add(offset), result);
        offset += 4;
    }
    for i in offset..n {
        let val = *p.add(i);
        if val < 0.0 {
            *p.add(i) = 0.0;
        }
    }
}

/// AVX2-accelerated ReLU: max(0, x) for 8 elements at a time.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn simd_relu_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = values.len();
    let p = values.as_mut_ptr();
    debug_assert_eq!((p as usize) % core::mem::align_of::<f32>(), 0);
    let zero = _mm256_setzero_ps();
    let chunks = n / 8;
    let mut offset = 0usize;
    for _ in 0..chunks {
        debug_assert!(offset + 8 <= n);
        // SAFETY: offset + 8 <= n; p is valid for n elements
        let v = _mm256_loadu_ps(p.add(offset));
        let result = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(p.add(offset), result);
        offset += 8;
    }
    for i in offset..n {
        let val = *p.add(i);
        if val < 0.0 {
            *p.add(i) = 0.0;
        }
    }
}

/// NEON-accelerated LeakyReLU: x if x > 0, else alpha * x.
///
/// Uses vector selection to avoid per-lane branches.
///
/// # Safety
/// Must run on AArch64, where NEON is mandatory.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
unsafe fn simd_leaky_relu_neon(values: &mut [f32], alpha: f32) {
    use std::arch::aarch64::*;
    let n = values.len();
    let p = values.as_mut_ptr();
    debug_assert_eq!((p as usize) % core::mem::align_of::<f32>(), 0);
    let zero = vdupq_n_f32(0.0);
    let alpha_v = vdupq_n_f32(alpha);
    let chunks = n / 4;
    let mut offset = 0usize;
    for _ in 0..chunks {
        debug_assert!(offset + 4 <= n);
        // SAFETY: offset + 4 <= n; p is valid for n elements
        let v = vld1q_f32(p.add(offset));
        let scaled = vmulq_f32(v, alpha_v);
        let mask = vcgeq_f32(v, zero);
        let result = vbslq_f32(mask, v, scaled);
        vst1q_f32(p.add(offset), result);
        offset += 4;
    }
    for i in offset..n {
        let val = *p.add(i);
        if val < 0.0 {
            *p.add(i) = val * alpha;
        }
    }
}

/// AVX2-accelerated LeakyReLU: x if x > 0, else alpha * x.
///
/// Uses vector selection to avoid per-lane branches.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn simd_leaky_relu_avx2(values: &mut [f32], alpha: f32) {
    use std::arch::x86_64::*;
    let n = values.len();
    let p = values.as_mut_ptr();
    debug_assert_eq!((p as usize) % core::mem::align_of::<f32>(), 0);
    let zero = _mm256_setzero_ps();
    let alpha_v = _mm256_set1_ps(alpha);
    let chunks = n / 8;
    let mut offset = 0usize;
    for _ in 0..chunks {
        debug_assert!(offset + 8 <= n);
        // SAFETY: offset + 8 <= n; p is valid for n elements
        let v = _mm256_loadu_ps(p.add(offset));
        let scaled = _mm256_mul_ps(v, alpha_v);
        let mask = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
        let result = _mm256_blendv_ps(scaled, v, mask);
        _mm256_storeu_ps(p.add(offset), result);
        offset += 8;
    }
    for i in offset..n {
        let val = *p.add(i);
        if val < 0.0 {
            *p.add(i) = val * alpha;
        }
    }
}

/// Evaluates sigmoid without overflowing for large negative inputs.
#[inline]
fn stable_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

impl Activation {
    /// Default alpha for leaky ReLU.
    pub const DEFAULT_LEAKY_ALPHA: f32 = 0.01;

    /// Applies this activation to one value.
    ///
    /// Note: For Softmax, use `forward_batch` instead as it requires the full vector.
    #[inline]
    pub fn forward(&self, x: f32) -> f32 {
        match self {
            Activation::Linear => x,
            Activation::Sigmoid => stable_sigmoid(x),
            Activation::Tanh => x.tanh(),
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU(alpha) => {
                if x > 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            // A scalar Softmax is 1.0.
            Activation::Softmax => 1.0,
        }
    }

    /// Applies this activation in place to `values`.
    ///
    /// Softmax operates on the complete slice; ReLU variants use supported SIMD paths.
    /// See [`docs/network.md`](../docs/network.md#activationforward_batch) for semantics.
    #[inline]
    pub fn forward_batch(&self, values: &mut [f32]) {
        match self {
            Activation::Linear => {} // No-op
            Activation::Sigmoid => {
                for v in values.iter_mut() {
                    *v = stable_sigmoid(*v);
                }
            }
            Activation::Tanh => {
                for v in values.iter_mut() {
                    *v = v.tanh();
                }
            }
            Activation::ReLU => {
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                {
                    // SAFETY: NEON is mandatory on AArch64; the helper bounds-checks vector loads.
                    unsafe { simd_relu_neon(values) };
                }
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        // SAFETY: AVX2 was detected above.
                        unsafe { simd_relu_avx2(values) };
                    } else {
                        for v in values.iter_mut() {
                            *v = v.max(0.0);
                        }
                    }
                }
                #[cfg(not(all(
                    feature = "simd",
                    any(target_arch = "aarch64", target_arch = "x86_64")
                )))]
                {
                    for v in values.iter_mut() {
                        *v = v.max(0.0);
                    }
                }
            }
            Activation::LeakyReLU(alpha) => {
                let a = *alpha;
                #[cfg(all(feature = "simd", target_arch = "aarch64"))]
                {
                    // SAFETY: NEON is mandatory on AArch64; the helper bounds-checks vector loads.
                    unsafe { simd_leaky_relu_neon(values, a) };
                }
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        // SAFETY: AVX2 was detected above.
                        unsafe { simd_leaky_relu_avx2(values, a) };
                    } else {
                        for v in values.iter_mut() {
                            if *v < 0.0 {
                                *v *= a;
                            }
                        }
                    }
                }
                #[cfg(not(all(
                    feature = "simd",
                    any(target_arch = "aarch64", target_arch = "x86_64")
                )))]
                {
                    for v in values.iter_mut() {
                        if *v < 0.0 {
                            *v *= a;
                        }
                    }
                }
            }
            Activation::Softmax => {
                // Subtract the maximum to avoid exponent overflow.
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for v in values.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                if sum > 0.0 {
                    for v in values.iter_mut() {
                        *v /= sum;
                    }
                }
            }
        }
    }

    /// Computes a derivative from an activation output value.
    ///
    /// Softmax returns its Jacobian diagonal only.
    /// See [`docs/network.md`](../docs/network.md#activation-derivatives) for derivative scope.
    #[inline]
    pub fn derivative(&self, output: f32) -> f32 {
        match self {
            Activation::Linear => 1.0,
            Activation::Sigmoid => output * (1.0 - output),
            Activation::Tanh => 1.0 - output * output,
            Activation::ReLU => {
                if output > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::LeakyReLU(alpha) => {
                if output > 0.0 {
                    1.0
                } else {
                    *alpha
                }
            }
            // The full Jacobian is handled by output-layer backpropagation (ADR-023).
            Activation::Softmax => output * (1.0 - output),
        }
    }

    /// Replaces each output with its derivative.
    #[inline]
    pub fn derivative_batch(&self, outputs: &mut [f32]) {
        match self {
            Activation::Linear => {
                for v in outputs.iter_mut() {
                    *v = 1.0;
                }
            }
            Activation::Sigmoid => {
                for v in outputs.iter_mut() {
                    *v = *v * (1.0 - *v);
                }
            }
            Activation::Tanh => {
                for v in outputs.iter_mut() {
                    *v = 1.0 - *v * *v;
                }
            }
            Activation::ReLU => {
                for v in outputs.iter_mut() {
                    *v = if *v > 0.0 { 1.0 } else { 0.0 };
                }
            }
            Activation::LeakyReLU(alpha) => {
                let a = *alpha;
                for v in outputs.iter_mut() {
                    *v = if *v > 0.0 { 1.0 } else { a };
                }
            }
            Activation::Softmax => {
                // This exposes only the Softmax Jacobian diagonal.
                for v in outputs.iter_mut() {
                    *v = *v * (1.0 - *v);
                }
            }
        }
    }

    /// Returns whether this is Softmax.
    #[inline]
    pub fn is_softmax(&self) -> bool {
        matches!(self, Activation::Softmax)
    }

    /// Returns whether this activation has a bounded output range.
    #[inline]
    pub fn is_bounded(&self) -> bool {
        matches!(
            self,
            Activation::Sigmoid | Activation::Tanh | Activation::Softmax
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_linear() {
        let act = Activation::Linear;
        assert!(approx_eq(act.forward(0.5), 0.5));
        assert!(approx_eq(act.forward(-1.0), -1.0));
        assert!(approx_eq(act.derivative(0.5), 1.0));
    }

    #[test]
    fn test_sigmoid() {
        let act = Activation::Sigmoid;
        assert!(approx_eq(act.forward(0.0), 0.5));
        assert!(act.forward(10.0) > 0.99);
        assert!(act.forward(-10.0) < 0.01);
        assert!(approx_eq(act.derivative(0.5), 0.25));
    }

    #[test]
    fn test_tanh() {
        let act = Activation::Tanh;
        assert!(approx_eq(act.forward(0.0), 0.0));
        assert!(act.forward(3.0) > 0.99);
        assert!(act.forward(-3.0) < -0.99);
        assert!(approx_eq(act.derivative(0.0), 1.0));
    }

    #[test]
    fn test_relu() {
        let act = Activation::ReLU;
        assert!(approx_eq(act.forward(0.5), 0.5));
        assert!(approx_eq(act.forward(-0.5), 0.0));
        assert!(approx_eq(act.forward(0.0), 0.0));
        assert!(approx_eq(act.derivative(0.5), 1.0));
        assert!(approx_eq(act.derivative(0.0), 0.0));
    }

    #[test]
    fn test_leaky_relu() {
        let alpha = 0.1;
        let act = Activation::LeakyReLU(alpha);
        assert!(approx_eq(act.forward(0.5), 0.5));
        assert!(approx_eq(act.forward(-0.5), -0.05));
        assert!(approx_eq(act.derivative(0.5), 1.0));
        assert!(approx_eq(act.derivative(-0.5), alpha));
    }

    #[test]
    fn test_softmax_batch() {
        let act = Activation::Softmax;
        let mut values = vec![1.0, 2.0, 3.0];
        act.forward_batch(&mut values);

        let sum: f32 = values.iter().sum();
        assert!(approx_eq(sum, 1.0));

        assert!(values[0] < values[1]);
        assert!(values[1] < values[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let act = Activation::Softmax;
        let mut values = vec![1000.0, 1001.0, 1002.0];
        act.forward_batch(&mut values);

        let sum: f32 = values.iter().sum();
        assert!(approx_eq(sum, 1.0));
        assert!(values.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_forward_batch_consistency() {
        let activations = vec![
            Activation::Linear,
            Activation::Sigmoid,
            Activation::Tanh,
            Activation::ReLU,
            Activation::LeakyReLU(0.1),
        ];

        for act in activations {
            let values = vec![-1.0, 0.0, 0.5, 1.0];
            let single_results: Vec<f32> = values.iter().map(|&v| act.forward(v)).collect();

            let mut batch_values = values.clone();
            act.forward_batch(&mut batch_values);

            for (single, batch) in single_results.iter().zip(batch_values.iter()) {
                assert!(
                    approx_eq(*single, *batch),
                    "{act:?}: single={single}, batch={batch}"
                );
            }
        }
    }

    #[test]
    fn test_is_bounded() {
        assert!(!Activation::Linear.is_bounded());
        assert!(Activation::Sigmoid.is_bounded());
        assert!(Activation::Tanh.is_bounded());
        assert!(!Activation::ReLU.is_bounded());
        assert!(!Activation::LeakyReLU(0.1).is_bounded());
        assert!(Activation::Softmax.is_bounded());
    }

    #[test]
    fn test_default_is_relu() {
        assert_eq!(Activation::default(), Activation::ReLU);
    }
}
