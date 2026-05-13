//! Activation functions for neural networks
//!
//! Provides common activation functions with forward and derivative computations.
//! Optimized for fast inference with minimal branching.

/// Activation function types
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

// --- SIMD activation function helpers ---

/// NEON-accelerated ReLU: max(0, x) for 4 elements at a time.
///
/// # Safety
/// NEON must be available (mandatory on aarch64). Slice pointer must be valid.
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
    // Scalar tail
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
/// Caller must ensure AVX2 is available. Slice pointer must be valid.
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
    // Scalar tail
    for i in offset..n {
        let val = *p.add(i);
        if val < 0.0 {
            *p.add(i) = 0.0;
        }
    }
}

/// NEON-accelerated LeakyReLU: x if x > 0, else alpha * x.
///
/// Uses vbslq_f32 (bitwise select) to blend between x and alpha*x based on
/// whether x >= 0. This avoids branching entirely.
///
/// # Safety
/// NEON must be available (mandatory on aarch64). Slice pointer must be valid.
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
        // mask: all-ones where v >= 0, all-zeros where v < 0
        let mask = vcgeq_f32(v, zero);
        // Select: where mask is set, pick v; where clear, pick scaled
        let result = vbslq_f32(mask, v, scaled);
        vst1q_f32(p.add(offset), result);
        offset += 4;
    }
    // Scalar tail
    for i in offset..n {
        let val = *p.add(i);
        if val < 0.0 {
            *p.add(i) = val * alpha;
        }
    }
}

/// AVX2-accelerated LeakyReLU: x if x > 0, else alpha * x.
///
/// Uses _mm256_blendv_ps to select between x and alpha*x. The blend uses
/// the sign bit of x: positive keeps x, negative gets alpha*x.
///
/// # Safety
/// Caller must ensure AVX2 is available. Slice pointer must be valid.
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
        // Compare: mask lanes where v > 0 (all bits set for true lanes)
        let mask = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
        // Blend: pick v where mask is set, scaled where not
        let result = _mm256_blendv_ps(scaled, v, mask);
        _mm256_storeu_ps(p.add(offset), result);
        offset += 8;
    }
    // Scalar tail
    for i in offset..n {
        let val = *p.add(i);
        if val < 0.0 {
            *p.add(i) = val * alpha;
        }
    }
}

/// Numerically stable sigmoid: avoids overflow when x is large and negative.
///
/// For x >= 0 uses the standard formula 1/(1+e^-x).
/// For x < 0 rewrites as e^x/(1+e^x) so the exponent is always non-negative,
/// preventing the intermediate `-(-x).exp()` from overflowing to +Inf.
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
    /// Default leaky ReLU alpha value
    pub const DEFAULT_LEAKY_ALPHA: f32 = 0.01;

    /// Apply activation function to a single value (element-wise)
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
            // For single element, softmax is just 1.0
            Activation::Softmax => 1.0,
        }
    }

    /// Apply activation function to a batch of values (in-place)
    ///
    /// This is more efficient than calling `forward` repeatedly and handles
    /// Softmax correctly. Uses SIMD acceleration for ReLU and LeakyReLU on
    /// supported platforms when the `simd` feature is enabled.
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
                    // SAFETY: NEON is mandatory on aarch64. We process values
                    // in chunks of 4 using vld1q/vmaxq/vst1q, then handle
                    // the scalar tail. All pointer arithmetic stays within the
                    // slice bounds.
                    unsafe { simd_relu_neon(values) };
                }
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        // SAFETY: AVX2 verified above; function processes in
                        // 8-element chunks with scalar tail.
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
                    // SAFETY: NEON is mandatory on aarch64. Processes in chunks
                    // of 4, blending between x and alpha*x based on sign.
                    unsafe { simd_leaky_relu_neon(values, a) };
                }
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        // SAFETY: AVX2 verified above; processes in 8-element
                        // chunks with blendv for conditional alpha scaling.
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
                // Numerically stable softmax
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

    /// Compute derivative of activation function at given output value
    ///
    /// For backpropagation, we typically have the activation output `y = f(x)`
    /// and need `f'(x)`. For efficiency, many activation derivatives can be
    /// computed from the output `y` directly.
    ///
    /// Note: For Softmax, use `derivative_batch` with the full Jacobian.
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
            // TODO(FP-095): diagonal approximation of Softmax Jacobian; full Jacobian would be more accurate.
            Activation::Softmax => output * (1.0 - output),
        }
    }

    /// Compute derivatives for a batch of output values (in-place)
    ///
    /// Modifies `outputs` to contain the derivatives.
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
                // Simplified: diagonal of Jacobian
                // Full Jacobian: J[i,j] = s[i](delta[i,j] - s[j])
                for v in outputs.iter_mut() {
                    *v = *v * (1.0 - *v);
                }
            }
        }
    }

    /// Returns true if this activation is Softmax
    #[inline]
    pub fn is_softmax(&self) -> bool {
        matches!(self, Activation::Softmax)
    }

    /// Returns true if this activation has a bounded output range
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
        // Derivative at output 0.5
        assert!(approx_eq(act.derivative(0.5), 0.25));
    }

    #[test]
    fn test_tanh() {
        let act = Activation::Tanh;
        assert!(approx_eq(act.forward(0.0), 0.0));
        assert!(act.forward(3.0) > 0.99);
        assert!(act.forward(-3.0) < -0.99);
        // Derivative at output 0
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

        // Sum should be 1.0
        let sum: f32 = values.iter().sum();
        assert!(approx_eq(sum, 1.0));

        // Values should be in increasing order (since input was increasing)
        assert!(values[0] < values[1]);
        assert!(values[1] < values[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let act = Activation::Softmax;
        // Large values that would overflow without stability fix
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
