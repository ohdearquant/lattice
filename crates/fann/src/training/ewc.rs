//! Elastic Weight Consolidation (EWC++) diagonal-Fisher forgetting guard.
//!
//! Prevents catastrophic forgetting during online or continual learning by
//! penalising updates to parameters that were important for prior tasks.
//! "Importance" is measured via an exponential moving average of squared
//! gradients — a diagonal approximation to the Fisher information matrix.
//!
//! Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in
//! neural networks", PNAS 2017; Chaudhry et al., "Efficient Lifelong Learning
//! with A-GEM", ICLR 2019 (EWC++ online variant).

use crate::error::{FannError, FannResult};

/// Diagonal Fisher information matrix for Elastic Weight Consolidation (EWC++).
///
/// Tracks per-parameter importance via an exponential moving average (EMA) of
/// squared gradients. Accumulate importance with [`observe_gradient`], fix the
/// reference point at a task boundary with [`set_anchor`], then either call
/// [`penalty_gradient`] to inject the EWC regularisation term into the backward
/// pass, or [`project_delta`] to null-space-damp a raw parameter update.
///
/// All operations on flat `&[f32]` slices — no network type dependency.
///
/// [`observe_gradient`]: DiagonalFisher::observe_gradient
/// [`set_anchor`]: DiagonalFisher::set_anchor
/// [`penalty_gradient`]: DiagonalFisher::penalty_gradient
/// [`project_delta`]: DiagonalFisher::project_delta
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiagonalFisher {
    /// Diagonal Fisher estimate: EMA of squared gradients, one entry per parameter.
    pub values: Vec<f32>,
    /// Anchor (reference) parameter vector captured at the end of a prior task.
    pub anchor: Vec<f32>,
    /// EMA decay factor ∈ (0, 1). Higher values give slower decay and longer memory.
    pub decay: f32,
}

impl DiagonalFisher {
    /// Create a new zeroed Fisher estimate for `num_params` parameters.
    ///
    /// Both `values` (Fisher diagonal) and `anchor` (reference parameters)
    /// are initialised to zero. Call [`observe_gradient`] to warm up the
    /// Fisher estimate, then [`set_anchor`] once at the task boundary before
    /// starting the next task.
    ///
    /// [`observe_gradient`]: DiagonalFisher::observe_gradient
    /// [`set_anchor`]: DiagonalFisher::set_anchor
    pub fn new(num_params: usize, decay: f32) -> Self {
        Self {
            values: vec![0.0; num_params],
            anchor: vec![0.0; num_params],
            decay,
        }
    }

    /// Update the diagonal Fisher estimate with one gradient observation.
    ///
    /// Applies the EWC++ online rule:
    ///
    /// ```text
    /// F_i ← decay · F_i  +  (1 − decay) · g_i²
    /// ```
    ///
    /// Call this after every backward pass on the prior-task distribution to
    /// maintain a running importance estimate. Returns an error if `grad.len()`
    /// does not equal the number of parameters this guard was created for.
    pub fn observe_gradient(&mut self, grad: &[f32]) -> FannResult<()> {
        if grad.len() != self.values.len() {
            return Err(FannError::InputSizeMismatch {
                expected: self.values.len(),
                actual: grad.len(),
            });
        }
        let one_minus_decay = 1.0 - self.decay;
        for (f, &g) in self.values.iter_mut().zip(grad.iter()) {
            // EMA of g² accumulates squared-gradient importance over time.
            *f = self.decay * *f + one_minus_decay * g * g;
        }
        Ok(())
    }

    /// Fix the anchor (reference) parameter vector at a task boundary.
    ///
    /// The EWC penalty measures deviation from these anchor parameters, so
    /// call this once after the prior task finishes training. Returns an
    /// error if `params.len()` does not match the guard's parameter count.
    pub fn set_anchor(&mut self, params: &[f32]) -> FannResult<()> {
        if params.len() != self.anchor.len() {
            return Err(FannError::InputSizeMismatch {
                expected: self.anchor.len(),
                actual: params.len(),
            });
        }
        self.anchor.copy_from_slice(params);
        Ok(())
    }

    /// Accumulate the EWC penalty gradient into `out`.
    ///
    /// The EWC regularisation loss is `(λ/2) Σ F_i (θ_i − θ*_i)²`.
    /// Its gradient with respect to θ_i is `λ · F_i · (θ_i − θ*_i)`,
    /// which is *added* into `out` — the caller subtracts the combined
    /// gradient in the descent step.
    ///
    /// Returns an error if `params` or `out` have a different length than
    /// the Fisher diagonal.
    pub fn penalty_gradient(&self, params: &[f32], lambda: f32, out: &mut [f32]) -> FannResult<()> {
        let n = self.values.len();
        if params.len() != n {
            return Err(FannError::InputSizeMismatch {
                expected: n,
                actual: params.len(),
            });
        }
        if out.len() != n {
            return Err(FannError::InputSizeMismatch {
                expected: n,
                actual: out.len(),
            });
        }
        // self.anchor.len() == n by construction (set_anchor length-checks any update).
        for (((v, anchor), p), out_elem) in self
            .values
            .iter()
            .zip(self.anchor.iter())
            .zip(params.iter())
            .zip(out.iter_mut())
        {
            // Gradient of (λ/2)·F_i·(θ_i − θ*_i)² is λ·F_i·(θ_i − θ*_i).
            *out_elem += lambda * v * (p - anchor);
        }
        Ok(())
    }

    /// Damp a raw parameter update `delta` toward the Fisher null-space.
    ///
    /// Diagonal approximation: parameters with high Fisher importance (large
    /// squared-gradient EMA = important to prior tasks) have their update
    /// magnitude scaled toward zero; parameters with near-zero Fisher pass
    /// through unchanged.  Full SVD null-space projection is deferred; this
    /// O(n) diagonal approximation is appropriate for online continual learning.
    ///
    /// Processes `min(delta.len(), self.values.len())` entries so the guard
    /// can be used on parameter sub-slices without error.
    ///
    /// Degenerate-Fisher guard: if `f_max < 1e-8` (no importance signal has
    /// accumulated yet) the function returns immediately — no projection, no
    /// division, no panic.
    pub fn project_delta(&self, delta: &mut [f32]) {
        let f_max = self.values.iter().copied().fold(0.0_f32, f32::max);

        // No importance signal has been observed yet — treat as identity.
        if f_max < 1e-8 {
            return;
        }

        // High-Fisher parameters are important to past tasks; damping their
        // update toward zero reduces forgetting.  Low-Fisher parameters
        // (values[i]/f_max ≈ 0) are free to change and pass through at scale ≈ 1.
        for (d, &v) in delta.iter_mut().zip(self.values.iter()) {
            *d *= (1.0 - v / f_max).max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A degenerate Fisher (all-zero values) must leave delta unchanged.
    ///
    /// Verifies the early-return path in project_delta — no projection, no panic.
    #[test]
    fn ewc_degenerate_fisher_is_identity() {
        let fisher = DiagonalFisher::new(4, 0.9);
        let mut delta = vec![1.0_f32, -2.0, 3.0, -4.0];
        let original = delta.clone();
        // values are all zero → f_max < 1e-8 → early return.
        fisher.project_delta(&mut delta);
        assert_eq!(
            delta, original,
            "degenerate Fisher must be identity on project_delta"
        );
    }

    /// High-Fisher entry blocks its parameter's update; zero-Fisher entry passes.
    #[test]
    fn ewc_high_fisher_blocks() {
        let mut fisher = DiagonalFisher::new(2, 0.0);
        // decay=0 → observe_gradient fully replaces values with g².
        fisher.observe_gradient(&[100.0, 0.0]).unwrap();
        // values == [10000.0, 0.0]; f_max == 10000.0.

        let mut delta = vec![1.0_f32, 1.0];
        fisher.project_delta(&mut delta);

        // values[0] == f_max → scale = (1 - 1.0).max(0) = 0 → blocked.
        assert!(
            delta[0].abs() < 1e-6,
            "high-Fisher entry should be blocked, got {}",
            delta[0]
        );
        // values[1] == 0 → scale = (1 - 0.0).max(0) = 1 → passes through.
        assert!(
            (delta[1] - 1.0).abs() < 1e-6,
            "zero-Fisher entry should pass through, got {}",
            delta[1]
        );
    }

    /// With anchor=0, params=[2], Fisher=1, lambda=1 → out accumulates +2.
    #[test]
    fn ewc_penalty_gradient_pulls_to_anchor() {
        let mut fisher = DiagonalFisher::new(1, 0.0);
        // decay=0 → values[0] = g² = 1.0 after one unit-gradient observation.
        fisher.observe_gradient(&[1.0]).unwrap();
        // anchor remains at zero (default).

        let params = vec![2.0_f32];
        let mut out = vec![0.0_f32];
        fisher.penalty_gradient(&params, 1.0, &mut out).unwrap();

        // Gradient = lambda * F[0] * (theta[0] - anchor[0]) = 1 * 1 * (2 - 0) = +2.
        // Positive: this term is subtracted in gradient descent, pulling theta
        // back toward the anchor and resisting forgetting.
        assert!(
            (out[0] - 2.0).abs() < 1e-6,
            "expected penalty gradient +2.0, got {}",
            out[0]
        );
    }

    /// Two EMA steps with decay=0.9 and unit gradient must match the closed-form.
    #[test]
    fn ewc_fisher_ema_accumulates() {
        let mut fisher = DiagonalFisher::new(1, 0.9);

        // Step 1: F = 0.9 * 0 + 0.1 * 1² = 0.1
        fisher.observe_gradient(&[1.0]).unwrap();
        assert!(
            (fisher.values[0] - 0.1).abs() < 1e-6,
            "after step 1 expected 0.1, got {}",
            fisher.values[0]
        );

        // Step 2: F = 0.9 * 0.1 + 0.1 * 1² = 0.09 + 0.10 = 0.19
        fisher.observe_gradient(&[1.0]).unwrap();
        assert!(
            (fisher.values[0] - 0.19).abs() < 1e-6,
            "after step 2 expected 0.19, got {}",
            fisher.values[0]
        );
    }

    /// observe_gradient with a wrong-length slice must return Err, not panic.
    #[test]
    fn ewc_length_mismatch_errors() {
        let mut fisher = DiagonalFisher::new(4, 0.9);
        let result = fisher.observe_gradient(&[1.0, 2.0]); // 2 != 4
        assert!(result.is_err(), "expected Err on length mismatch, got Ok");
    }

    /// Serialise then deserialise must produce a structurally equal value.
    #[cfg(feature = "serde")]
    #[test]
    fn ewc_fisher_roundtrips() {
        let mut fisher = DiagonalFisher::new(3, 0.9);
        fisher.observe_gradient(&[1.0, 2.0, 3.0]).unwrap();
        fisher.set_anchor(&[0.1, 0.2, 0.3]).unwrap();

        let json = serde_json::to_string(&fisher).unwrap();
        let recovered: DiagonalFisher = serde_json::from_str(&json).unwrap();

        assert_eq!(fisher.values, recovered.values);
        assert_eq!(fisher.anchor, recovered.anchor);
        assert!(
            (fisher.decay - recovered.decay).abs() < 1e-9,
            "decay mismatch after roundtrip"
        );
    }
}
