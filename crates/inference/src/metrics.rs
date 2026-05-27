//! Inference metrics infrastructure (ADR-061 Phase 1).
//!
//! Provides zero-cost-when-disabled metrics collection for forward passes.
//! The [`OnlineSoftmaxEntropy`] accumulator computes attention entropy in O(1)
//! space without materializing the full probability vector.

/// Controls what metrics are collected during inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetricsMode {
    /// Zero overhead — no metrics collected.
    #[default]
    Off,
    /// Cheap counters: per-layer wall time, input/output norms, token throughput.
    CheapOnline,
    /// Adds attention entropy and sparsity via online accumulators.
    /// Requires modified attention kernels.
    AttentionProfile,
}

/// Per-layer measurement collected during a forward pass.
#[derive(Debug, Clone, Default)]
pub struct LayerMetrics {
    /// Layer index.
    pub layer_idx: usize,
    /// Forward pass wall time in nanoseconds.
    pub forward_ns: u64,
    /// L2 norm of input hidden states (pre-layernorm).
    pub input_norm: f32,
    /// L2 norm of output hidden states (post-residual).
    pub output_norm: f32,
    /// Mean attention entropy across heads (only in `AttentionProfile` mode).
    pub entropy: Option<f32>,
}

/// Collects [`LayerMetrics`] for an entire forward pass.
#[derive(Debug)]
pub struct ForwardMetrics {
    /// Mode that was active when this collection was made.
    pub mode: MetricsMode,
    /// Per-layer records, in layer order.
    pub layers: Vec<LayerMetrics>,
    /// Total wall time for the forward pass in nanoseconds.
    pub total_ns: u64,
}

/// Online softmax entropy accumulator.
///
/// Computes H = -Σ pᵢ ln pᵢ without materializing the probability vector,
/// using the numerically stable online algorithm described in ADR-061.
///
/// The invariant maintained after each [`update`](OnlineSoftmaxEntropy::update) call:
/// - `m` = max(a₁ … aₜ)
/// - `l` = Σ exp(aₛ − m)
/// - `e` = Σ exp(aₛ − m) · aₛ
///
/// From these, entropy in nats is: H = ln(l) + m − e/l ≥ 0.
///
/// # Numerical stability
///
/// Logits in \[-100, 100\] produce exp differences in \[exp(-200), exp(200)\].
/// The running-max rescaling keeps intermediate values in a safe range for `f32`.
#[derive(Debug, Clone)]
pub struct OnlineSoftmaxEntropy {
    m: f32, // running max of logits seen so far
    l: f32, // running sum of exp(aₛ − m)
    e: f32, // running weighted accumulator: Σ exp(aₛ − m) · aₛ
    count: usize,
}

impl Default for OnlineSoftmaxEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineSoftmaxEntropy {
    /// Create a fresh accumulator.
    pub fn new() -> Self {
        Self {
            m: 0.0,
            l: 0.0,
            e: 0.0,
            count: 0,
        }
    }

    /// Feed one logit value into the accumulator.
    ///
    /// This is the hot path — no allocation, branch-minimal.
    pub fn update(&mut self, logit: f32) {
        if self.count == 0 {
            self.m = logit;
            self.l = 1.0;
            self.e = logit;
            self.count = 1;
            return;
        }
        if logit > self.m {
            // New max: rescale existing accumulators.
            let alpha = (self.m - logit).exp();
            self.l = alpha * self.l + 1.0;
            self.e = alpha * self.e + logit;
            self.m = logit;
        } else {
            // No max change: incorporate this logit with a relative weight.
            let w = (logit - self.m).exp();
            self.l += w;
            self.e += w * logit;
        }
        self.count += 1;
    }

    /// Finalize and return entropy in nats.
    ///
    /// Returns `0.0` if fewer than 2 values have been fed (degenerate distribution).
    /// The result is clamped to `≥ 0.0` to guard against floating-point underflow.
    pub fn entropy_nats(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        (self.l.ln() + self.m - self.e / self.l).max(0.0)
    }

    /// Entropy in bits (H / ln 2).
    pub fn entropy_bits(&self) -> f32 {
        self.entropy_nats() / std::f32::consts::LN_2
    }

    /// Entropy normalized by log(T) for cross-length comparability.
    ///
    /// Returns a value in \[0, 1\] where 1.0 means perfectly uniform
    /// over the T positions fed so far.
    pub fn normalized_entropy(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        self.entropy_nats() / (self.count as f32).ln()
    }

    /// Number of logit values fed so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset to initial state, reusing the allocation.
    pub fn reset(&mut self) {
        self.m = 0.0;
        self.l = 0.0;
        self.e = 0.0;
        self.count = 0;
    }
}

/// Compute the L2 norm of a slice.
///
/// Uses a simple sum-of-squares approach. The inner loop is written to be
/// auto-vectorizable (no horizontal reduction inside the loop).
///
/// # Examples
/// ```
/// # use lattice_inference::metrics::l2_norm;
/// let v = [3.0_f32, 4.0];
/// assert!((l2_norm(&v) - 5.0).abs() < 1e-5);
/// ```
pub fn l2_norm(data: &[f32]) -> f32 {
    let sum_sq: f32 = data.iter().map(|&x| x * x).sum();
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative tolerance for f32 comparisons.
    const TOL: f32 = 1e-5;

    fn assert_close(a: f32, b: f32, msg: &str) {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1e-8);
        assert!(
            diff / scale < TOL,
            "{msg}: got {a}, expected {b}, diff={diff}"
        );
    }

    #[test]
    fn test_metrics_mode_default() {
        assert_eq!(MetricsMode::default(), MetricsMode::Off);
    }

    #[test]
    fn test_entropy_uniform() {
        // N identical logits → uniform distribution → entropy = ln(N).
        for n in [2usize, 8, 128, 512] {
            let mut acc = OnlineSoftmaxEntropy::new();
            let logit = 0.5_f32; // arbitrary identical value
            for _ in 0..n {
                acc.update(logit);
            }
            let expected = (n as f32).ln();
            assert_close(
                acc.entropy_nats(),
                expected,
                &format!("uniform entropy N={n}"),
            );
        }
    }

    #[test]
    fn test_entropy_peaked() {
        // One large logit, rest tiny → distribution concentrates on one token → entropy ≈ 0.
        let mut acc = OnlineSoftmaxEntropy::new();
        acc.update(100.0);
        for _ in 0..127 {
            acc.update(-100.0);
        }
        // With this extreme split, entropy should be very close to 0.
        assert!(
            acc.entropy_nats() < 0.01,
            "peaked entropy should be near 0, got {}",
            acc.entropy_nats()
        );
    }

    #[test]
    fn test_entropy_two_values() {
        // 50/50 split: two equal logits → entropy = ln(2) ≈ 0.693.
        let mut acc = OnlineSoftmaxEntropy::new();
        acc.update(1.0);
        acc.update(1.0);
        assert_close(acc.entropy_nats(), std::f32::consts::LN_2, "50/50 entropy");
    }

    #[test]
    fn test_entropy_matches_naive() {
        // Compare online vs naive (materialize softmax then -sum p log p).
        // Use a deterministic sequence so the test is repeatable.
        let logits: Vec<f32> = (0..256)
            .map(|i| {
                // LCG-derived values in [-5, 5]
                let x = (i as u32)
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                ((x >> 8) as f32) / 16_777_216.0 * 10.0 - 5.0
            })
            .collect();

        // Online accumulator.
        let mut acc = OnlineSoftmaxEntropy::new();
        for &l in &logits {
            acc.update(l);
        }
        let online_h = acc.entropy_nats();

        // Naive reference: softmax → -sum p log p.
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let naive_h: f32 = exps
            .iter()
            .map(|&e| {
                let p = e / sum_exp;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum();

        // Allow up to 1e-4 absolute difference due to f32 accumulation order.
        let diff = (online_h - naive_h).abs();
        assert!(
            diff < 1e-4,
            "online={online_h} naive={naive_h} diff={diff} > 1e-4"
        );
    }

    #[test]
    fn test_entropy_single_value() {
        let mut acc = OnlineSoftmaxEntropy::new();
        acc.update(5.0);
        assert_eq!(acc.entropy_nats(), 0.0, "single value → 0");
        assert_eq!(acc.count(), 1);
    }

    #[test]
    fn test_entropy_extreme_logits() {
        // Verify stability at the stated bounds [-100, 100].
        let mut acc = OnlineSoftmaxEntropy::new();
        acc.update(100.0);
        acc.update(-100.0);
        // Should not produce NaN or inf.
        let h = acc.entropy_nats();
        assert!(
            h.is_finite(),
            "extreme logits produced non-finite entropy: {h}"
        );
        assert!(h >= 0.0, "entropy should be non-negative, got {h}");
    }

    #[test]
    fn test_entropy_reset() {
        let mut acc = OnlineSoftmaxEntropy::new();
        acc.update(1.0);
        acc.update(2.0);
        acc.reset();
        assert_eq!(acc.count(), 0);
        assert_eq!(acc.entropy_nats(), 0.0, "after reset, entropy is 0");

        // Reuse should give correct results.
        acc.update(1.0);
        acc.update(1.0);
        assert_close(
            acc.entropy_nats(),
            std::f32::consts::LN_2,
            "after reset reuse",
        );
    }

    #[test]
    fn test_entropy_bits_and_normalized() {
        let mut acc = OnlineSoftmaxEntropy::new();
        // 4 uniform logits → entropy = ln(4) nats = 2 bits, normalized = 1.0.
        for _ in 0..4 {
            acc.update(0.0);
        }
        assert_close(acc.entropy_bits(), 2.0, "4-uniform bits");
        assert_close(acc.normalized_entropy(), 1.0, "4-uniform normalized");
    }

    #[test]
    fn test_l2_norm_known() {
        let v = [3.0_f32, 4.0];
        assert_close(l2_norm(&v), 5.0, "3-4-5 triangle");

        let unit = [1.0_f32, 0.0, 0.0];
        assert_close(l2_norm(&unit), 1.0, "unit vector");

        let zeros = [0.0_f32; 16];
        assert_close(l2_norm(&zeros), 0.0, "zero vector");
    }

    #[test]
    fn test_l2_norm_larger() {
        // sum of squares = 896 * 1² = 896, so norm = sqrt(896) ≈ 29.933.
        let v = vec![1.0_f32; 896];
        let expected = (896.0_f32).sqrt();
        assert_close(l2_norm(&v), expected, "896-dim unit-filled");
    }

    #[test]
    fn test_layer_metrics_default() {
        let lm = LayerMetrics::default();
        assert_eq!(lm.layer_idx, 0);
        assert_eq!(lm.forward_ns, 0);
        assert_eq!(lm.input_norm, 0.0);
        assert_eq!(lm.output_norm, 0.0);
        assert!(lm.entropy.is_none());
    }
}
