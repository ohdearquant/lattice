//! Elastic Weight Consolidation (EWC++) diagonal-Fisher forgetting guard.
//!
//! Tracks per-parameter importance from squared gradients and uses it to
//! penalise or damp changes to parameters important to earlier tasks.
//! The guard is independent of `Network` and operates on flat parameter slices.
//!
//! See `docs/training.md` for the EWC model, lifecycle, and update formulas.

use crate::error::{FannError, FannResult, validate_allocation_size};

/// A flat-slice EWC++ diagonal-Fisher forgetting guard.
///
/// Tracks EMA importance and a matching parameter anchor for each entry.
/// See [`docs/training.md`](../../docs/training.md#elastic-weight-consolidation-ewc) for the lifecycle and update formulas.
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
    /// Creates zeroed Fisher and anchor vectors for `num_params` parameters.
    ///
    /// Returns [`FannError::InvalidDistributionParams`] for non-finite or out-of-range `decay`, or [`FannError::ShapeTooLarge`] for an oversized allocation.
    /// See [`docs/training.md`](../../docs/training.md#diagonal-fisher-information-as-an-importance-estimate) for the EMA rule and decay rationale.
    pub fn new(num_params: usize, decay: f32) -> FannResult<Self> {
        // Decay must retain history and a positive fresh-gradient contribution — see docs/training.md.
        if !decay.is_finite() || decay <= 0.0 || decay >= 1.0 {
            return Err(FannError::InvalidDistributionParams(format!(
                "EWC decay must be finite and in the open interval (0, 1), got {decay}"
            )));
        }
        validate_allocation_size(num_params)?;
        Ok(Self {
            values: vec![0.0; num_params],
            anchor: vec![0.0; num_params],
            decay,
        })
    }

    /// Updates each Fisher entry from the corresponding gradient observation.
    ///
    /// Returns [`FannError::InputSizeMismatch`] unless `grad` matches the parameter count.
    /// See [`docs/training.md`](../../docs/training.md#diagonal-fisher-information-as-an-importance-estimate) for the EWC++ EMA rule.
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

    /// Adds the EWC penalty gradient for `params` into `out`.
    ///
    /// Returns [`FannError::InputSizeMismatch`] unless both slices match the Fisher length.
    /// See [`docs/training.md`](../../docs/training.md#anchor-penalty-gradient) for the formula and descent integration.
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

    /// Damp the common prefix of a raw parameter update by Fisher importance.
    ///
    /// Leaves `delta` unchanged when the estimate has no importance signal.
    /// See [`docs/training.md`](../../docs/training.md#null-space-delta-projection) for the scaling rule and trade-offs.
    pub fn project_delta(&self, delta: &mut [f32]) {
        let f_max = self.values.iter().copied().fold(0.0_f32, f32::max);

        // No importance signal has been observed yet — treat as identity.
        if f_max < 1e-8 {
            return;
        }

        // Damp high-Fisher updates while preserving low-Fisher ones — see docs/training.md.
        for (d, &v) in delta.iter_mut().zip(self.values.iter()) {
            *d *= (1.0 - v / f_max).max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::error::MAX_ALLOWED_ELEMENTS;

    /// A degenerate Fisher (all-zero values) must leave delta unchanged.
    ///
    /// Verifies the early-return path in project_delta — no projection, no panic.
    #[test]
    fn ewc_degenerate_fisher_is_identity() {
        let fisher = DiagonalFisher::new(4, 0.9).unwrap();
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
        let mut fisher = DiagonalFisher::new(2, 0.9).unwrap();
        // decay=0.9 → F[0] = 0.9*0 + 0.1*100² = 1000; F[1] = 0.
        fisher.observe_gradient(&[100.0, 0.0]).unwrap();

        let mut delta = vec![1.0_f32, 1.0];
        fisher.project_delta(&mut delta);

        // F[0] == f_max → scale = (1 - 1000/1000).max(0) = 0 → blocked.
        assert!(
            delta[0].abs() < 1e-6,
            "high-Fisher entry should be blocked, got {}",
            delta[0]
        );
        // F[1] == 0 → scale = (1 - 0/1000).max(0) = 1 → passes through.
        assert!(
            (delta[1] - 1.0).abs() < 1e-6,
            "zero-Fisher entry should pass through, got {}",
            delta[1]
        );
    }

    /// With anchor=0, params=[2], Fisher≈1, lambda=1 → out accumulates ≈+2.
    ///
    /// Uses decay=f32::EPSILON so (1−decay)≈1 and one unit-gradient observation
    /// gives F[0]≈1.0; the penalty gradient is thus ≈ lambda·F[0]·(θ−θ*)=2.
    /// This regression test pins the direction: positive gradient means the
    /// update (w -= lr·g) pulls θ back toward the anchor, not away from it.
    #[test]
    fn ewc_penalty_gradient_pulls_to_anchor() {
        let mut fisher = DiagonalFisher::new(1, f32::EPSILON).unwrap();
        // decay≈0 → values[0] ≈ g² = 1.0 after one unit-gradient observation.
        fisher.observe_gradient(&[1.0]).unwrap();
        // anchor remains at zero (default).

        let params = vec![2.0_f32];
        let mut out = vec![0.0_f32];
        fisher.penalty_gradient(&params, 1.0, &mut out).unwrap();

        // Gradient = lambda · F[0] · (theta[0] − anchor[0]) ≈ 1 · 1 · (2 − 0) = +2.
        // Positive: subtracted in gradient descent → θ moves toward anchor.
        assert!(
            (out[0] - 2.0).abs() < 1e-5,
            "expected penalty gradient ≈+2.0 (toward anchor), got {}",
            out[0]
        );
        // Direction check: penalty gradient must be strictly positive when θ > anchor.
        assert!(
            out[0] > 0.0,
            "penalty gradient must be positive when theta > anchor (toward anchor)"
        );
    }

    /// Two EMA steps with decay=0.9 and unit gradient must match the closed-form.
    #[test]
    fn ewc_fisher_ema_accumulates() {
        let mut fisher = DiagonalFisher::new(1, 0.9).unwrap();

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
        let mut fisher = DiagonalFisher::new(4, 0.9).unwrap();
        let result = fisher.observe_gradient(&[1.0, 2.0]); // 2 != 4
        assert!(result.is_err(), "expected Err on length mismatch, got Ok");
    }

    // ---- Allocation-bound guard tests ---------------------------------------

    /// Constructing with num_params > MAX_ALLOWED_ELEMENTS must return Err, not panic.
    ///
    /// Mutation that breaks this: removing the `validate_allocation_size` call.
    #[test]
    fn diagonal_fisher_new_too_large_returns_err() {
        let result = DiagonalFisher::new(MAX_ALLOWED_ELEMENTS + 1, 0.9);
        assert!(
            matches!(result, Err(FannError::ShapeTooLarge { .. })),
            "expected ShapeTooLarge error for num_params > MAX, got {result:?}"
        );
    }

    // ---- Decay-validation guard tests ---------------------------------------

    /// decay = 0.0 must be rejected (lower bound of the open interval).
    ///
    /// Mutation that breaks this: removing or inverting the `decay <= 0.0` check.
    #[test]
    fn diagonal_fisher_new_decay_zero_returns_err() {
        let result = DiagonalFisher::new(4, 0.0);
        assert!(
            matches!(result, Err(FannError::InvalidDistributionParams(_))),
            "expected InvalidDistributionParams for decay=0.0, got {result:?}"
        );
    }

    /// decay = 1.0 must be rejected (upper bound of the open interval).
    ///
    /// Mutation that breaks this: removing or inverting the `decay >= 1.0` check.
    #[test]
    fn diagonal_fisher_new_decay_one_returns_err() {
        let result = DiagonalFisher::new(4, 1.0);
        assert!(
            matches!(result, Err(FannError::InvalidDistributionParams(_))),
            "expected InvalidDistributionParams for decay=1.0, got {result:?}"
        );
    }

    /// decay = -0.5 must be rejected (below zero).
    ///
    /// Mutation that breaks this: removing the `decay <= 0.0` check.
    #[test]
    fn diagonal_fisher_new_decay_negative_returns_err() {
        let result = DiagonalFisher::new(4, -0.5);
        assert!(
            matches!(result, Err(FannError::InvalidDistributionParams(_))),
            "expected InvalidDistributionParams for decay=-0.5, got {result:?}"
        );
    }

    /// decay = NaN must be rejected.
    ///
    /// Mutation that breaks this: removing the `is_finite()` check.
    #[test]
    fn diagonal_fisher_new_decay_nan_returns_err() {
        let result = DiagonalFisher::new(4, f32::NAN);
        assert!(
            matches!(result, Err(FannError::InvalidDistributionParams(_))),
            "expected InvalidDistributionParams for decay=NaN, got {result:?}"
        );
    }

    /// decay = +Inf must be rejected.
    ///
    /// Mutation that breaks this: removing the `is_finite()` check.
    #[test]
    fn diagonal_fisher_new_decay_inf_returns_err() {
        let result = DiagonalFisher::new(4, f32::INFINITY);
        assert!(
            matches!(result, Err(FannError::InvalidDistributionParams(_))),
            "expected InvalidDistributionParams for decay=+Inf, got {result:?}"
        );
    }

    /// Serialise then deserialise must produce a structurally equal value.
    #[cfg(feature = "serde")]
    #[test]
    fn ewc_fisher_roundtrips() {
        let mut fisher = DiagonalFisher::new(3, 0.9).unwrap();
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
