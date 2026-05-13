//! Neural network layer implementation
//!
//! A layer consists of a weight matrix, bias vector, and activation function.
//! Optimized for fast inference with pre-allocated buffers.

use crate::activation::Activation;
use crate::error::{FannError, FannResult, validate_layer_dimensions};

// --- SIMD dot product for matmul hot path ---

/// SIMD-accelerated dot product dispatching to platform-specific intrinsics.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "simd_dot_product: mismatched lengths");
    debug_assert_eq!((a.as_ptr() as usize) % core::mem::align_of::<f32>(), 0);
    debug_assert_eq!((b.as_ptr() as usize) % core::mem::align_of::<f32>(), 0);
    // SAFETY: NEON is mandatory on aarch64; slices have equal length (assert above)
    unsafe { simd_dot_product_neon(a, b) }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "simd_dot_product: mismatched lengths");
    debug_assert_eq!((a.as_ptr() as usize) % core::mem::align_of::<f32>(), 0);
    debug_assert_eq!((b.as_ptr() as usize) % core::mem::align_of::<f32>(), 0);
    // Try AVX-512 first, then AVX2+FMA, then scalar fallback
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: AVX-512F verified above; slices have equal length
        unsafe { simd_dot_product_avx512(a, b) }
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA verified above; slices have equal length
        unsafe { simd_dot_product_avx2(a, b) }
    } else {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// NEON dot product — 4x4-wide FMA (16 elements/iter) with 4 independent accumulators.
///
/// Uses 4 independent accumulator registers to hide the 4-cycle FMA latency on
/// typical ARM cores (Cortex-A76, Apple M-series). Each accumulator processes 4
/// f32 elements per iteration = 16 elements total per loop body.
///
/// # Safety
/// Caller must ensure `a.len() == b.len()`. NEON is mandatory on aarch64.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
unsafe fn simd_dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    debug_assert_eq!(n, b.len());
    debug_assert_eq!((pa as usize) % core::mem::align_of::<f32>(), 0);
    debug_assert_eq!((pb as usize) % core::mem::align_of::<f32>(), 0);

    // 4 independent accumulators to saturate FMA throughput
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    // Main loop: 16 elements per iteration (4 accumulators x 4 lanes)
    let chunks16 = n / 16;
    let mut offset = 0usize;
    for _ in 0..chunks16 {
        debug_assert!(offset + 16 <= n);
        // SAFETY: offset + 16 <= n because we iterate chunks16 = n/16 times.
        // pa/pb are valid for n elements. vld1q_f32 reads 4 contiguous f32s.
        let va0 = vld1q_f32(pa.add(offset));
        let vb0 = vld1q_f32(pb.add(offset));
        sum0 = vfmaq_f32(sum0, va0, vb0);

        let va1 = vld1q_f32(pa.add(offset + 4));
        let vb1 = vld1q_f32(pb.add(offset + 4));
        sum1 = vfmaq_f32(sum1, va1, vb1);

        let va2 = vld1q_f32(pa.add(offset + 8));
        let vb2 = vld1q_f32(pb.add(offset + 8));
        sum2 = vfmaq_f32(sum2, va2, vb2);

        let va3 = vld1q_f32(pa.add(offset + 12));
        let vb3 = vld1q_f32(pb.add(offset + 12));
        sum3 = vfmaq_f32(sum3, va3, vb3);

        offset += 16;
    }

    // Remainder loop: 4 elements per iteration using sum0 only
    let chunks4 = (n - offset) / 4;
    for _ in 0..chunks4 {
        debug_assert!(offset + 4 <= n);
        // SAFETY: offset + 4 <= n guaranteed by (n - offset) / 4 calculation
        let va = vld1q_f32(pa.add(offset));
        let vb = vld1q_f32(pb.add(offset));
        sum0 = vfmaq_f32(sum0, va, vb);
        offset += 4;
    }

    // Combine accumulators: (sum0+sum1) + (sum2+sum3)
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    let mut result = vaddvq_f32(sum0);

    // Scalar tail: remaining 0-3 elements
    for i in offset..n {
        // SAFETY: i < n, both slices have length n
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    result
}

/// AVX2+FMA dot product — 4x8-wide FMA (32 elements/iter) with 4 independent accumulators.
///
/// Uses 4 independent __m256 accumulators to hide FMA latency (typically 4-5 cycles
/// on Haswell/Skylake). Each accumulator processes 8 f32 elements per iteration =
/// 32 elements total per loop body.
///
/// # Safety
/// Caller must ensure AVX2+FMA are available and `a.len() == b.len()`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    debug_assert_eq!(n, b.len());
    debug_assert_eq!((pa as usize) % core::mem::align_of::<f32>(), 0);
    debug_assert_eq!((pb as usize) % core::mem::align_of::<f32>(), 0);

    // 4 independent accumulators to saturate FMA throughput
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    // Main loop: 32 elements per iteration (4 accumulators x 8 lanes)
    let chunks32 = n / 32;
    let mut offset = 0usize;
    for _ in 0..chunks32 {
        debug_assert!(offset + 32 <= n);
        // SAFETY: offset + 32 <= n because we iterate chunks32 = n/32 times.
        // pa/pb are valid for n elements. _mm256_loadu_ps reads 8 contiguous f32s.
        let va0 = _mm256_loadu_ps(pa.add(offset));
        let vb0 = _mm256_loadu_ps(pb.add(offset));
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

        let va1 = _mm256_loadu_ps(pa.add(offset + 8));
        let vb1 = _mm256_loadu_ps(pb.add(offset + 8));
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

        let va2 = _mm256_loadu_ps(pa.add(offset + 16));
        let vb2 = _mm256_loadu_ps(pb.add(offset + 16));
        sum2 = _mm256_fmadd_ps(va2, vb2, sum2);

        let va3 = _mm256_loadu_ps(pa.add(offset + 24));
        let vb3 = _mm256_loadu_ps(pb.add(offset + 24));
        sum3 = _mm256_fmadd_ps(va3, vb3, sum3);

        offset += 32;
    }

    // Remainder loop: 8 elements per iteration using sum0 only
    let chunks8 = (n - offset) / 8;
    for _ in 0..chunks8 {
        debug_assert!(offset + 8 <= n);
        // SAFETY: offset + 8 <= n guaranteed by (n - offset) / 8 calculation
        let va = _mm256_loadu_ps(pa.add(offset));
        let vb = _mm256_loadu_ps(pb.add(offset));
        sum0 = _mm256_fmadd_ps(va, vb, sum0);
        offset += 8;
    }

    // Combine accumulators: (sum0+sum1) + (sum2+sum3)
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    // Horizontal sum: 8 -> 4 -> 2 -> 1
    let high = _mm256_extractf128_ps(sum0, 1);
    let low = _mm256_castps256_ps128(sum0);
    let sum128 = _mm_add_ps(high, low);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    let mut result = _mm_cvtss_f32(sums2);

    // Scalar tail: remaining 0-7 elements
    for i in offset..n {
        // SAFETY: i < n, both slices have length n
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    result
}

/// AVX-512 dot product — 4x16-wide FMA (64 elements/iter) with 4 independent accumulators.
///
/// Uses 512-bit registers (16 f32 lanes each) with 4 independent accumulators =
/// 64 elements per loop iteration. Available on Skylake-X, Ice Lake, Zen 4+.
///
/// # Safety
/// Caller must ensure AVX-512F is available and `a.len() == b.len()`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(clippy::incompatible_msrv)]
unsafe fn simd_dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    debug_assert_eq!(n, b.len());
    debug_assert_eq!((pa as usize) % core::mem::align_of::<f32>(), 0);
    debug_assert_eq!((pb as usize) % core::mem::align_of::<f32>(), 0);

    // 4 independent accumulators to saturate FMA throughput
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();

    // Main loop: 64 elements per iteration (4 accumulators x 16 lanes)
    let chunks64 = n / 64;
    let mut offset = 0usize;
    for _ in 0..chunks64 {
        debug_assert!(offset + 64 <= n);
        // SAFETY: offset + 64 <= n because we iterate chunks64 = n/64 times.
        // pa/pb are valid for n elements. _mm512_loadu_ps reads 16 contiguous f32s.
        let va0 = _mm512_loadu_ps(pa.add(offset));
        let vb0 = _mm512_loadu_ps(pb.add(offset));
        sum0 = _mm512_fmadd_ps(va0, vb0, sum0);

        let va1 = _mm512_loadu_ps(pa.add(offset + 16));
        let vb1 = _mm512_loadu_ps(pb.add(offset + 16));
        sum1 = _mm512_fmadd_ps(va1, vb1, sum1);

        let va2 = _mm512_loadu_ps(pa.add(offset + 32));
        let vb2 = _mm512_loadu_ps(pb.add(offset + 32));
        sum2 = _mm512_fmadd_ps(va2, vb2, sum2);

        let va3 = _mm512_loadu_ps(pa.add(offset + 48));
        let vb3 = _mm512_loadu_ps(pb.add(offset + 48));
        sum3 = _mm512_fmadd_ps(va3, vb3, sum3);

        offset += 64;
    }

    // Remainder loop: 16 elements per iteration using sum0 only
    let chunks16 = (n - offset) / 16;
    for _ in 0..chunks16 {
        debug_assert!(offset + 16 <= n);
        // SAFETY: offset + 16 <= n guaranteed by (n - offset) / 16 calculation
        let va = _mm512_loadu_ps(pa.add(offset));
        let vb = _mm512_loadu_ps(pb.add(offset));
        sum0 = _mm512_fmadd_ps(va, vb, sum0);
        offset += 16;
    }

    // Combine accumulators: (sum0+sum1) + (sum2+sum3)
    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);

    // Horizontal sum via intrinsic
    let mut result = _mm512_reduce_add_ps(sum0);

    // Scalar tail: remaining 0-15 elements
    for i in offset..n {
        // SAFETY: i < n, both slices have length n
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    result
}

/// A single layer in a neural network
///
/// Computes: output = activation(input * weights + bias)
///
/// Memory layout:
/// - weights: [num_outputs * num_inputs] in row-major order
/// - biases: [num_outputs]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Layer {
    /// Number of input neurons
    num_inputs: usize,
    /// Number of output neurons
    num_outputs: usize,
    /// Weight matrix (row-major: output_idx * num_inputs + input_idx)
    weights: Vec<f32>,
    /// Bias vector
    biases: Vec<f32>,
    /// Activation function
    activation: Activation,
}

impl Layer {
    /// Create a new layer with given dimensions and activation
    ///
    /// Initializes weights using Xavier/Glorot initialization and biases to zero.
    ///
    /// # Errors
    ///
    /// Returns [`FannError::InvalidLayerDimensions`] if dimensions are zero.
    /// Returns [`FannError::ShapeTooLarge`] if the allocation would exceed safe limits.
    pub fn new(num_inputs: usize, num_outputs: usize, activation: Activation) -> FannResult<Self> {
        // Validate dimensions and allocation size
        validate_layer_dimensions(num_inputs, num_outputs)?;

        let weight_count = num_inputs * num_outputs;
        let mut weights = vec![0.0; weight_count];
        let biases = vec![0.0; num_outputs];

        // Xavier/Glorot initialization
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::SmallRng::from_entropy();
        let std_dev = (2.0 / (num_inputs + num_outputs) as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).map_err(|e| {
            FannError::InvalidDistributionParams(format!(
                "Failed to create Normal distribution with std_dev={std_dev}: {e}"
            ))
        })?;

        for w in weights.iter_mut() {
            *w = normal.sample(&mut rng) as f32;
        }

        Ok(Self {
            num_inputs,
            num_outputs,
            weights,
            biases,
            activation,
        })
    }

    /// Create a layer with provided weights and biases
    ///
    /// # Arguments
    /// * `num_inputs` - Number of input neurons
    /// * `num_outputs` - Number of output neurons
    /// * `weights` - Weight matrix in row-major order (length: num_inputs * num_outputs)
    /// * `biases` - Bias vector (length: num_outputs)
    /// * `activation` - Activation function
    ///
    /// # Errors
    ///
    /// Returns [`FannError::InvalidLayerDimensions`] if dimensions are zero.
    /// Returns [`FannError::ShapeTooLarge`] if the allocation would exceed safe limits.
    /// Returns [`FannError::WeightCountMismatch`] if weights length doesn't match dimensions.
    /// Returns [`FannError::BiasCountMismatch`] if biases length doesn't match outputs.
    pub fn with_weights(
        num_inputs: usize,
        num_outputs: usize,
        weights: Vec<f32>,
        biases: Vec<f32>,
        activation: Activation,
    ) -> FannResult<Self> {
        // Validate dimensions and allocation size
        validate_layer_dimensions(num_inputs, num_outputs)?;

        let expected_weights = num_inputs * num_outputs;
        if weights.len() != expected_weights {
            return Err(FannError::WeightCountMismatch {
                expected: expected_weights,
                actual: weights.len(),
            });
        }

        if biases.len() != num_outputs {
            return Err(FannError::BiasCountMismatch {
                expected: num_outputs,
                actual: biases.len(),
            });
        }

        Ok(Self {
            num_inputs,
            num_outputs,
            weights,
            biases,
            activation,
        })
    }

    /// Create a layer with zeros (useful for testing)
    ///
    /// # Errors
    ///
    /// Returns [`FannError::InvalidLayerDimensions`] if dimensions are zero.
    /// Returns [`FannError::ShapeTooLarge`] if the allocation would exceed safe limits.
    pub fn zeros(
        num_inputs: usize,
        num_outputs: usize,
        activation: Activation,
    ) -> FannResult<Self> {
        // Validate dimensions and allocation size
        validate_layer_dimensions(num_inputs, num_outputs)?;

        Ok(Self {
            num_inputs,
            num_outputs,
            weights: vec![0.0; num_inputs * num_outputs],
            biases: vec![0.0; num_outputs],
            activation,
        })
    }

    /// Create a new layer with seeded random initialization
    ///
    /// Uses Xavier/Glorot initialization with a deterministic RNG for reproducibility.
    ///
    /// # Arguments
    /// * `num_inputs` - Number of input neurons
    /// * `num_outputs` - Number of output neurons
    /// * `activation` - Activation function
    /// * `rng` - Seeded random number generator
    ///
    /// # Errors
    ///
    /// Returns [`FannError::InvalidLayerDimensions`] if dimensions are zero.
    /// Returns [`FannError::ShapeTooLarge`] if the allocation would exceed safe limits.
    pub fn new_with_rng<R: rand::Rng>(
        num_inputs: usize,
        num_outputs: usize,
        activation: Activation,
        rng: &mut R,
    ) -> FannResult<Self> {
        // Validate dimensions and allocation size
        validate_layer_dimensions(num_inputs, num_outputs)?;

        let weight_count = num_inputs * num_outputs;
        let mut weights = vec![0.0; weight_count];
        let biases = vec![0.0; num_outputs];

        // Xavier/Glorot initialization
        use rand_distr::{Distribution, Normal};
        let std_dev = (2.0 / (num_inputs + num_outputs) as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).map_err(|e| {
            FannError::InvalidDistributionParams(format!(
                "Failed to create Normal distribution with std_dev={std_dev}: {e}"
            ))
        })?;

        for w in weights.iter_mut() {
            *w = normal.sample(rng) as f32;
        }

        Ok(Self {
            num_inputs,
            num_outputs,
            weights,
            biases,
            activation,
        })
    }

    /// Forward pass through the layer
    ///
    /// Computes output = activation(input * weights + bias)
    /// Result is written to the output buffer.
    ///
    /// # Arguments
    /// * `input` - Input vector (length: num_inputs)
    /// * `output` - Pre-allocated output buffer (length: num_outputs)
    #[inline]
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> FannResult<()> {
        if input.len() != self.num_inputs {
            return Err(FannError::InputSizeMismatch {
                expected: self.num_inputs,
                actual: input.len(),
            });
        }

        if output.len() != self.num_outputs {
            return Err(FannError::OutputSizeMismatch {
                expected: self.num_outputs,
                actual: output.len(),
            });
        }

        // Matrix-vector multiplication: output = weights * input + bias
        // weights is row-major: weights[out_idx * num_inputs + in_idx]
        self.matmul_add(input, output);

        // Apply activation
        self.activation.forward_batch(output);

        Ok(())
    }

    /// Matrix-vector multiplication with bias addition
    ///
    /// Computes: output[i] = sum_j(weights[i,j] * input[j]) + bias[i]
    #[inline]
    fn matmul_add(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "simd")]
        {
            self.matmul_add_simd(input, output);
        }
        #[cfg(not(feature = "simd"))]
        {
            self.matmul_add_scalar(input, output);
        }
    }

    /// Scalar implementation of matrix-vector multiplication
    #[inline]
    #[cfg_attr(feature = "simd", allow(dead_code))]
    fn matmul_add_scalar(&self, input: &[f32], output: &mut [f32]) {
        for (out_idx, out_val) in output.iter_mut().enumerate() {
            let row_start = out_idx * self.num_inputs;
            let mut sum = self.biases[out_idx];
            for (in_idx, &in_val) in input.iter().enumerate() {
                sum += self.weights[row_start + in_idx] * in_val;
            }
            *out_val = sum;
        }
    }

    /// SIMD-optimized matrix-vector multiplication using platform intrinsics.
    ///
    /// Uses NEON on aarch64, AVX2+FMA on x86_64, scalar fallback elsewhere.
    /// Includes software prefetch hints for the next weight row to hide memory
    /// latency on large matrices.
    #[cfg(feature = "simd")]
    #[inline]
    #[allow(clippy::needless_range_loop)]
    fn matmul_add_simd(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            let num_outputs = output.len();
            let num_inputs = self.num_inputs;
            let weights_ptr = self.weights.as_ptr();
            for out_idx in 0..num_outputs {
                let row_start = out_idx * num_inputs;
                // Prefetch next row's weights into L1 cache to hide memory latency.
                // On the last row this prefetches slightly past the end, but
                // _mm_prefetch on x86 is a hint — out-of-bounds addresses are
                // silently ignored by the hardware.
                if out_idx + 1 < num_outputs {
                    let next_row_start = (out_idx + 1) * num_inputs;
                    debug_assert!(next_row_start < self.weights.len());
                    debug_assert_eq!((weights_ptr as usize) % core::mem::align_of::<f32>(), 0);
                    // SAFETY: _mm_prefetch is a hint; invalid addresses are
                    // silently discarded by the CPU. We only issue this when
                    // there is a next row.
                    unsafe {
                        use std::arch::x86_64::*;
                        _mm_prefetch(weights_ptr.add(next_row_start) as *const i8, _MM_HINT_T0);
                    }
                }
                let row = &self.weights[row_start..row_start + num_inputs];
                output[out_idx] = simd_dot_product(row, input) + self.biases[out_idx];
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            for (out_idx, out_val) in output.iter_mut().enumerate() {
                let row_start = out_idx * self.num_inputs;
                let row = &self.weights[row_start..row_start + self.num_inputs];
                *out_val = simd_dot_product(row, input) + self.biases[out_idx];
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            self.matmul_add_scalar(input, output);
        }
    }

    /// Get the number of input neurons
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Get the number of output neurons
    #[inline]
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Get the total number of parameters (weights + biases)
    #[inline]
    pub fn num_params(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    /// Get the activation function
    #[inline]
    pub fn activation(&self) -> Activation {
        self.activation
    }

    /// Get weights (for inspection/serialization)
    #[inline]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get biases (for inspection/serialization)
    #[inline]
    pub fn biases(&self) -> &[f32] {
        &self.biases
    }

    /// Get mutable weights (for training)
    #[inline]
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Get mutable biases (for training)
    #[inline]
    pub fn biases_mut(&mut self) -> &mut [f32] {
        &mut self.biases
    }

    /// Get weight at specific position
    #[inline]
    pub fn get_weight(&self, output_idx: usize, input_idx: usize) -> Option<f32> {
        if output_idx < self.num_outputs && input_idx < self.num_inputs {
            Some(self.weights[output_idx * self.num_inputs + input_idx])
        } else {
            None
        }
    }

    /// Set weight at specific position
    #[inline]
    pub fn set_weight(&mut self, output_idx: usize, input_idx: usize, value: f32) -> bool {
        if output_idx < self.num_outputs && input_idx < self.num_inputs {
            self.weights[output_idx * self.num_inputs + input_idx] = value;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(10, 5, Activation::ReLU).unwrap();
        assert_eq!(layer.num_inputs(), 10);
        assert_eq!(layer.num_outputs(), 5);
        assert_eq!(layer.weights().len(), 50);
        assert_eq!(layer.biases().len(), 5);
        assert_eq!(layer.num_params(), 55);
    }

    #[test]
    fn test_layer_zeros() {
        let layer = Layer::zeros(4, 3, Activation::Sigmoid).unwrap();
        assert!(layer.weights().iter().all(|&w| w == 0.0));
        assert!(layer.biases().iter().all(|&b| b == 0.0));
    }

    #[test]
    fn test_layer_with_weights() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let biases = vec![0.1, 0.2];
        let layer = Layer::with_weights(3, 2, weights, biases, Activation::Linear).unwrap();

        assert_eq!(layer.num_inputs(), 3);
        assert_eq!(layer.num_outputs(), 2);
    }

    #[test]
    fn test_layer_with_weights_mismatch() {
        let weights = vec![1.0, 2.0]; // Wrong size
        let biases = vec![0.1, 0.2];
        let result = Layer::with_weights(3, 2, weights, biases, Activation::Linear);
        assert!(matches!(result, Err(FannError::WeightCountMismatch { .. })));
    }

    #[test]
    fn test_layer_invalid_dimensions() {
        let result = Layer::new(0, 5, Activation::ReLU);
        assert!(matches!(
            result,
            Err(FannError::InvalidLayerDimensions { .. })
        ));

        let result = Layer::new(5, 0, Activation::ReLU);
        assert!(matches!(
            result,
            Err(FannError::InvalidLayerDimensions { .. })
        ));
    }

    #[test]
    fn test_forward_linear() {
        // Simple 2->2 layer with known weights
        let weights = vec![1.0, 0.0, 0.0, 1.0]; // Identity
        let biases = vec![0.0, 0.0];
        let layer = Layer::with_weights(2, 2, weights, biases, Activation::Linear).unwrap();

        let input = [3.0, 4.0];
        let mut output = [0.0, 0.0];
        layer.forward(&input, &mut output).unwrap();

        assert!((output[0] - 3.0).abs() < 1e-5);
        assert!((output[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_forward_with_bias() {
        let weights = vec![1.0, 1.0]; // 1x2 - sum inputs
        let biases = vec![0.5];
        let layer = Layer::with_weights(2, 1, weights, biases, Activation::Linear).unwrap();

        let input = [2.0, 3.0];
        let mut output = [0.0];
        layer.forward(&input, &mut output).unwrap();

        assert!((output[0] - 5.5).abs() < 1e-5); // 2 + 3 + 0.5
    }

    #[test]
    fn test_forward_relu() {
        let weights = vec![1.0, -1.0]; // 2x1 transposed
        let biases = vec![0.0, 0.0];
        let layer = Layer::with_weights(1, 2, weights, biases, Activation::ReLU).unwrap();

        let input = [2.0];
        let mut output = [0.0, 0.0];
        layer.forward(&input, &mut output).unwrap();

        assert!((output[0] - 2.0).abs() < 1e-5); // ReLU(2) = 2
        assert!((output[1] - 0.0).abs() < 1e-5); // ReLU(-2) = 0
    }

    #[test]
    fn test_forward_input_mismatch() {
        let layer = Layer::new(10, 5, Activation::ReLU).unwrap();
        let input = [1.0; 8]; // Wrong size
        let mut output = [0.0; 5];

        let result = layer.forward(&input, &mut output);
        assert!(matches!(result, Err(FannError::InputSizeMismatch { .. })));
    }

    #[test]
    fn test_get_set_weight() {
        let mut layer = Layer::zeros(3, 2, Activation::Linear).unwrap();

        assert!(layer.set_weight(0, 1, 42.0));
        assert_eq!(layer.get_weight(0, 1), Some(42.0));

        assert!(!layer.set_weight(10, 10, 1.0)); // Out of bounds
        assert_eq!(layer.get_weight(10, 10), None);
    }

    #[test]
    fn test_layer_oversized_allocation_rejected() {
        // 100_001 * 100_001 > 100M (MAX_ALLOWED_ELEMENTS)
        let result = Layer::new(100_001, 100_001, Activation::ReLU);
        assert!(matches!(result, Err(FannError::ShapeTooLarge { .. })));

        // Also test zeros
        let result = Layer::zeros(100_001, 100_001, Activation::ReLU);
        assert!(matches!(result, Err(FannError::ShapeTooLarge { .. })));
    }
}
