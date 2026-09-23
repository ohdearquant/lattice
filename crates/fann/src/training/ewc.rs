//! Elastic Weight Consolidation (EWC++) diagonal-Fisher forgetting guard.
//!
//! Tracks per-parameter importance from squared gradients and uses it to
//! penalise or damp changes to parameters important to earlier tasks.
//! The guard is independent of `Network` and operates on flat parameter slices.
//!
//! See `docs/training.md` for the EWC model, lifecycle, and update formulas.

use crate::error::{FannError, FannResult, validate_allocation_size};

/// Damping strength for [`DiagonalFisher::project_delta`]'s Fisher-weighted shrinkage.
///
/// `project_delta` scales each coordinate by `1 / (1 + alpha * F_i / F_ref)`, where
/// `F_ref` is the mean Fisher value. `alpha == 1.0` means a coordinate whose Fisher
/// value equals the mean is damped to half its raw magnitude; larger `alpha` damps
/// every coordinate more without ever reaching zero for a finite Fisher value. See
/// `docs/training.md#fisher-weighted-delta-shrinkage` for the derivation and the
/// properties this constant preserves (never zeroes, monotonic, scale-invariant in F).
const PROJECT_DELTA_ALPHA: f32 = 1.0;

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
    /// See [`docs/training.md`](../../docs/training.md#anchor--penalty-gradient) for the formula and descent integration.
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
    /// Scales each coordinate by `1 / (1 + alpha * F_i / F_ref)`, the closed-form
    /// minimiser of `||d - delta||^2 + alpha * sum_i (F_i / F_ref) * d_i^2` — a
    /// per-coordinate ridge penalty proportional to relative Fisher importance.
    /// Unlike the max-normalised linear damping this replaced (issue #1575), this
    /// never zeroes a coordinate for a finite Fisher value: a uniform non-zero
    /// Fisher scales every coordinate by exactly `1 / (1 + alpha)`, higher `F_i`
    /// means strictly more damping, and the result is scale-invariant in `F`
    /// (multiplying every `F_i` by the same positive constant leaves `delta`
    /// unchanged, since only the ratio `F_i / F_ref` appears).
    ///
    /// Leaves `delta` unchanged when the estimate has no importance signal
    /// (`F_ref` below the guard threshold) or when the Fisher holds no values.
    /// See [`docs/training.md`](../../docs/training.md#fisher-weighted-delta-shrinkage) for the derivation and trade-offs.
    pub fn project_delta(&self, delta: &mut [f32]) {
        let n = self.values.len();
        if n == 0 {
            return;
        }
        let f_ref = self.values.iter().sum::<f32>() / n as f32;

        // No importance signal has been observed yet — treat as identity.
        if f_ref < 1e-8 {
            return;
        }

        // Fisher-weighted shrinkage: never reaches zero for finite F_i — see
        // docs/training.md and the doc comment above.
        for (d, &v) in delta.iter_mut().zip(self.values.iter()) {
            *d /= 1.0 + PROJECT_DELTA_ALPHA * (v / f_ref);
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

    /// High-Fisher entries are damped strongly, but never zeroed (issue #1575:
    /// the old max-normalised formula zeroed the argmax coordinate exactly).
    /// Zero-Fisher entries pass through unchanged.
    #[test]
    fn ewc_high_fisher_blocks() {
        let mut fisher = DiagonalFisher::new(5, 0.9).unwrap();
        // decay=0.9 → F[0] = 0.9*0 + 0.1*100² = 1000; F[1..5] = 0.
        fisher
            .observe_gradient(&[100.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();

        let mut delta = vec![1.0_f32; 5];
        fisher.project_delta(&mut delta);

        // F_ref = mean(1000, 0, 0, 0, 0) = 200 → scale[0] = 1/(1 + 1000/200) = 1/6.
        assert!(
            delta[0].abs() > 1e-6,
            "high-Fisher entry must not be zeroed, got {}",
            delta[0]
        );
        assert!(
            delta[0].abs() < 1.0,
            "high-Fisher entry must be damped, got {}",
            delta[0]
        );

        // F[1..5] == 0 → scale = 1/(1 + 0) = 1 → pass through unchanged.
        for &d in &delta[1..] {
            assert!(
                (d - 1.0).abs() < 1e-6,
                "zero-Fisher entry should pass through, got {d}"
            );
        }

        // Strong damping: the high-Fisher coordinate shrinks at least 5x more
        // than a zero-Fisher one (shrink factor = original magnitude / result).
        let shrink_high = 1.0_f32 / delta[0];
        let shrink_zero = 1.0_f32 / delta[1];
        assert!(
            shrink_high >= 5.0 * shrink_zero,
            "high-Fisher coordinate should shrink >=5x more than a zero-Fisher one: \
             shrink_high={shrink_high}, shrink_zero={shrink_zero}"
        );
    }

    /// A uniform non-zero Fisher scales every coordinate by exactly
    /// `1 / (1 + alpha)` — the defect this replaces zeroed every coordinate
    /// instead (issue #1575: `F_max` equals every entry, so `1 - F_i/F_max = 0`).
    #[test]
    fn ewc_uniform_nonzero_fisher_scales_by_one_over_one_plus_alpha() {
        let fisher = DiagonalFisher {
            values: vec![2.0_f32; 4],
            anchor: vec![0.0; 4],
            decay: 0.9,
        };
        let original = vec![1.0_f32, -2.0, 3.0, -4.0];
        let mut delta = original.clone();
        fisher.project_delta(&mut delta);

        let expected_scale = 1.0 / (1.0 + PROJECT_DELTA_ALPHA);
        for (d, orig) in delta.iter().zip(original.iter()) {
            assert!(
                (d - orig * expected_scale).abs() < 1e-5,
                "expected {} (= {orig} * {expected_scale}), got {d}",
                orig * expected_scale
            );
            assert!(d.abs() > 1e-6, "no coordinate should be zeroed, got {d}");
        }
    }

    /// First-step shape from `one_gradient_step` (crates/fann/src/training/router_update.rs):
    /// on the first step the Fisher EMA starts at zero, so `F_i` is proportional to
    /// `delta_i^2`. The largest-magnitude component must stay non-zero and must be
    /// damped more than the smaller components (issue #1575's reachable scenario).
    #[test]
    fn ewc_first_step_shape_damps_largest_component_without_zeroing() {
        let original = vec![10.0_f32, 1.0, -3.0];
        // F_i proportional to delta_i^2 (the constant of proportionality is
        // scale-invariant, so use 1.0 for the squared magnitude directly).
        let fisher = DiagonalFisher {
            values: original.iter().map(|d| d * d).collect(),
            anchor: vec![0.0; 3],
            decay: 0.9,
        };
        let mut delta = original.clone();
        fisher.project_delta(&mut delta);

        for (d, orig) in delta.iter().zip(original.iter()) {
            assert!(
                d.abs() > 1e-6,
                "no coordinate should be zeroed, got {d} (from {orig})"
            );
        }

        // Retained fraction per coordinate: d_i / original_i.
        let retained: Vec<f32> = delta
            .iter()
            .zip(original.iter())
            .map(|(d, o)| d / o)
            .collect();
        assert!(
            retained[0] < retained[1],
            "largest-magnitude component (idx 0) must be damped more than idx 1: {retained:?}"
        );
        assert!(
            retained[0] < retained[2],
            "largest-magnitude component (idx 0) must be damped more than idx 2: {retained:?}"
        );
    }

    /// Higher Fisher value means strictly more damping (holding all other
    /// coordinates, and hence F_ref, fixed by construction of the input vector).
    #[test]
    fn ewc_damping_monotonic_in_fisher_value() {
        let values = vec![0.0_f32, 1.0, 2.0, 5.0, 10.0];
        let fisher = DiagonalFisher {
            values: values.clone(),
            anchor: vec![0.0; values.len()],
            decay: 0.9,
        };
        let mut delta = vec![1.0_f32; values.len()];
        fisher.project_delta(&mut delta);

        for w in delta.windows(2) {
            assert!(
                w[1] < w[0],
                "damping must strictly increase with Fisher value: {delta:?}"
            );
        }
    }

    /// `project_delta` is scale-invariant in `F`: multiplying every Fisher value
    /// by the same positive constant must not change the result, since only the
    /// ratio `F_i / F_ref` appears in the scaling formula.
    #[test]
    fn ewc_project_delta_scale_invariant_in_fisher() {
        let values = vec![0.5_f32, 3.0, 7.5];
        let scaled: Vec<f32> = values.iter().map(|v| v * 1000.0).collect();
        let original = vec![2.0_f32, -1.5, 0.25];

        let fisher_a = DiagonalFisher {
            values,
            anchor: vec![0.0; 3],
            decay: 0.9,
        };
        let fisher_b = DiagonalFisher {
            values: scaled,
            anchor: vec![0.0; 3],
            decay: 0.9,
        };

        let mut delta_a = original.clone();
        let mut delta_b = original.clone();
        fisher_a.project_delta(&mut delta_a);
        fisher_b.project_delta(&mut delta_b);

        for (a, b) in delta_a.iter().zip(delta_b.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "project_delta must be scale-invariant in Fisher: {a} vs {b}"
            );
        }
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
