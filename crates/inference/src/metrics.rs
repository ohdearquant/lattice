//! Inference metrics core seed (ADR-061).
//!
//! Provides zero-cost-when-disabled metrics collection for forward passes.
//! The `OnlineSoftmaxEntropy` accumulator computes attention entropy in O(1)
//! space without materializing the full probability vector.

/// Controls what metrics are collected during inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MetricsMode {
    /// Zero overhead — no metrics collected.
    #[default]
    Off,
    /// Cheap counters: per-layer wall time, input/output norms, token throughput.
    CheapOnline,
    /// Adds attention entropy and sparsity via online accumulators.
    /// Requires modified attention kernels.
    AttentionProfile,
    /// Full diagnostics: entropy histograms, sparsity maps, KV-page mass,
    /// pattern labels. Higher overhead; not for production hot paths.
    HeavyDiagnostics,
}

/// Per-layer measurement collected during a forward pass (ADR-061 D4 schema).
///
/// `update_ratio` and `block_influence` are the scoring signals consumed by
/// ADR-060 pruning. `entropy` holds per-head entropy in nats (one element per
/// attention head) when `mode >= AttentionProfile`. All `Option<Vec<_>>` fields
/// default to `None` so `Default` is allocation-free.
#[derive(Debug, Clone, Default)]
pub struct LayerMetrics {
    /// Mode active when this record was populated.
    pub mode: MetricsMode,
    /// Layer index.
    pub layer_idx: usize,
    /// Forward pass wall time in nanoseconds (ADR-061 `latency_ns`).
    pub latency_ns: u64,
    /// L2 norm of input hidden states (pre-layernorm).
    pub input_norm: f32,
    /// L2 norm of output hidden states (post-residual).
    pub output_norm: f32,
    /// Parameter update ratio: ‖Δθ‖ / ‖θ‖ for this layer. Consumed by ADR-060.
    pub update_ratio: f32,
    /// Block influence score used by ADR-060 pruning (higher = more important).
    pub block_influence: f32,
    /// Per-head attention entropy in nats (one entry per head).
    /// Populated in `AttentionProfile` or `HeavyDiagnostics` mode.
    pub entropy: Option<Vec<f32>>,
    /// Per-head sparsity fractions in [0, 1]. Populated in `HeavyDiagnostics`.
    pub sparsity: Option<Vec<f32>>,
    /// Per-page KV-cache mass. Populated in `HeavyDiagnostics`.
    pub kv_page_mass: Option<Vec<f32>>,
    /// Attention-pattern labels (e.g. "diagonal", "sink"). `HeavyDiagnostics`.
    pub pattern_label: Option<Vec<String>>,
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
/// using the shifted-logit variant of the online algorithm (ADR-061).
///
/// The invariant maintained after each [`update`](OnlineSoftmaxEntropy::update) call:
/// - `m` = max(a₁ … aₜ)
/// - `l` = Σ exp(aₛ − m)
/// - `r` = Σ exp(aₛ − m) · (aₛ − m)   ← shifted accumulator
///
/// From these, entropy in nats is: H = ln(l) − r/l ≥ 0.
///
/// The shifted form keeps `r` bounded by `l * (max_logit_range)` rather than
/// `l * max_abs_logit`, preventing overflow on extreme finite logits like `f32::MAX`.
///
/// # Non-finite inputs
///
/// All update methods require finite logits. Use [`try_update`] for recoverable
/// error handling; [`update`](OnlineSoftmaxEntropy::update) panics on non-finite input in both debug and
/// release builds (fail-fast finite precondition).
///
/// [`try_update`]: OnlineSoftmaxEntropy::try_update
#[derive(Debug, Clone)]
pub struct OnlineSoftmaxEntropy {
    m: f32, // running max of logits seen so far
    l: f32, // running sum of exp(aₛ − m)
    r: f32, // shifted accumulator: Σ exp(aₛ − m) · (aₛ − m)
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
            r: 0.0,
            count: 0,
        }
    }

    /// Feed one logit value into the accumulator.
    ///
    /// This is the hot path — no allocation, branch-minimal.
    ///
    /// # Panics
    ///
    /// Panics in both debug and release builds if `logit` is not finite.
    /// Use [`try_update`](OnlineSoftmaxEntropy::try_update) for a recoverable path.
    #[inline]
    pub fn update(&mut self, logit: f32) {
        assert!(logit.is_finite(), "logit must be finite, got {logit}");
        self.update_finite(logit);
    }

    /// Feed one logit value, returning an error if it is not finite.
    ///
    /// Leaves the accumulator unchanged on error so it remains valid for
    /// further use.
    #[inline]
    pub fn try_update(&mut self, logit: f32) -> Result<(), crate::error::InferenceError> {
        if !logit.is_finite() {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "attention entropy logit must be finite, got {logit}"
            )));
        }
        self.update_finite(logit);
        Ok(())
    }

    /// Inner finite-preconditioned update (recurrence body).
    ///
    /// Called only after the caller has verified `logit.is_finite()`.
    #[inline]
    fn update_finite(&mut self, logit: f32) {
        debug_assert!(logit.is_finite());
        if self.count == 0 {
            self.m = logit;
            self.l = 1.0;
            self.r = 0.0; // (logit - m) = 0 for first element
            self.count = 1;
            return;
        }
        if logit > self.m {
            // New max: rescale existing accumulators.
            let diff = self.m - logit; // negative
            let alpha = diff.exp();
            // Shift correction: old terms had (a_s - m_old), now need (a_s - m_new).
            // (a_s - m_new) = (a_s - m_old) + (m_old - m_new) = (a_s - m_old) + diff
            // So r_new = alpha * r_old + alpha * l_old * diff + (logit - logit) * 1
            //          = alpha * (r_old + l_old * diff)
            //
            // Guard: when the new logit dwarfs the old max (|diff| very large),
            // alpha underflows to 0.0. In that case the old terms vanish exactly,
            // so r_new = 0.0 rather than 0.0 * (finite + possibly-large) = NaN.
            self.r = if alpha == 0.0 {
                0.0
            } else {
                alpha * (self.r + self.l * diff)
            };
            self.l = alpha * self.l + 1.0;
            self.m = logit;
        } else {
            let shifted = logit - self.m; // non-positive
            let w = shifted.exp();
            self.l += w;
            // Guard: when shifted is very negative, w underflows to 0.0.
            // Computing 0.0 * shifted would give 0.0 * -inf = NaN in release
            // mode when logit << m. Use 0.0 directly to keep r clean.
            self.r += if w == 0.0 { 0.0 } else { w * shifted };
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
        // H = ln(l) - r/l, where r = Σ exp(aₛ - m)(aₛ - m) ≤ 0 always,
        // so -r/l ≥ 0, making H ≥ ln(l) ≥ 0 when l ≥ 1.
        (self.l.ln() - self.r / self.l).max(0.0)
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
        self.r = 0.0;
        self.count = 0;
    }
}

/// Compute the L2 norm of a slice.
///
/// Uses f64 accumulation to avoid overflow on finite vectors whose element-wise
/// squares would exceed `f32::MAX` (e.g. elements near `1e20`). The inner loop
/// is written to be auto-vectorizable.
///
/// # Examples
/// ```
/// # use lattice_inference::metrics::l2_norm;
/// let v = [3.0_f32, 4.0];
/// assert!((l2_norm(&v) - 5.0).abs() < 1e-5);
/// ```
#[inline]
pub fn l2_norm(data: &[f32]) -> f32 {
    let sum_sq: f64 = data
        .iter()
        .map(|&x| {
            let x = f64::from(x);
            x * x
        })
        .sum();
    sum_sq.sqrt() as f32
}

#[cfg(test)]
mod tests;
