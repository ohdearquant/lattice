//! Elastic Weight Consolidation (EWC++) diagonal-Fisher forgetting guard.
//!
//! Tracks per-parameter importance from squared gradients and uses it to
//! penalise or damp changes to parameters important to earlier tasks.
//! The guard is independent of `Network` and operates on flat parameter slices.
//!
//! See `docs/training.md` for the EWC model, lifecycle, and update formulas.

use crate::error::{FannError, FannResult, validate_allocation_size};

/// Flat-slice EWC++ diagonal-Fisher forgetting guard.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiagonalFisher {
    /// Per-parameter Fisher estimate.
    pub values: Vec<f32>,
    /// Reference parameters from a prior task.
    pub anchor: Vec<f32>,
    /// EMA decay in `(0, 1)`.
    pub decay: f32,
}

impl DiagonalFisher {
    /// Creates zeroed Fisher and anchor vectors for `num_params` parameters.
    ///
    /// Rejects invalid decay or oversized allocations.
    pub fn new(num_params: usize, decay: f32) -> FannResult<Self> {
        // The EMA needs both retained and fresh-gradient terms.
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
    /// Rejects a gradient with the wrong parameter count.
    pub fn observe_gradient(&mut self, grad: &[f32]) -> FannResult<()> {
        if grad.len() != self.values.len() {
            return Err(FannError::InputSizeMismatch {
                expected: self.values.len(),
                actual: grad.len(),
            });
        }
        let one_minus_decay = 1.0 - self.decay;
        for (f, &g) in self.values.iter_mut().zip(grad.iter()) {
            *f = self.decay * *f + one_minus_decay * g * g;
        }
        Ok(())
    }

    /// Stores reference parameters at a task boundary.
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
    /// Rejects slices that differ from the Fisher length.
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
        for (((v, anchor), p), out_elem) in self
            .values
            .iter()
            .zip(self.anchor.iter())
            .zip(params.iter())
            .zip(out.iter_mut())
        {
            *out_elem += lambda * v * (p - anchor);
        }
        Ok(())
    }

    /// Damp the common prefix of a raw parameter update by Fisher importance.
    ///
    /// Leaves `delta` unchanged without an importance signal.
    pub fn project_delta(&self, delta: &mut [f32]) {
        let f_max = self.values.iter().copied().fold(0.0_f32, f32::max);

        if f_max < 1e-8 {
            return;
        }

        for (d, &v) in delta.iter_mut().zip(self.values.iter()) {
            *d *= (1.0 - v / f_max).max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::error::MAX_ALLOWED_ELEMENTS;

    #[test]
    fn ewc_degenerate_fisher_is_identity() {
        let fisher = DiagonalFisher::new(4, 0.9).unwrap();
        let mut delta = vec![1.0_f32, -2.0, 3.0, -4.0];
        let original = delta.clone();
        fisher.project_delta(&mut delta);
        assert_eq!(
            delta, original,
            "degenerate Fisher must be identity on project_delta"
        );
    }

    #[test]
    fn ewc_high_fisher_blocks() {
        let mut fisher = DiagonalFisher::new(2, 0.9).unwrap();
        fisher.observe_gradient(&[100.0, 0.0]).unwrap();

        let mut delta = vec![1.0_f32, 1.0];
        fisher.project_delta(&mut delta);

        assert!(
            delta[0].abs() < 1e-6,
            "high-Fisher entry should be blocked, got {}",
            delta[0]
        );
        assert!(
            (delta[1] - 1.0).abs() < 1e-6,
            "zero-Fisher entry should pass through, got {}",
            delta[1]
        );
    }

    #[test]
    fn ewc_penalty_gradient_pulls_to_anchor() {
        let mut fisher = DiagonalFisher::new(1, f32::EPSILON).unwrap();
        fisher.observe_gradient(&[1.0]).unwrap();

        let params = vec![2.0_f32];
        let mut out = vec![0.0_f32];
        fisher.penalty_gradient(&params, 1.0, &mut out).unwrap();

        assert!(
            (out[0] - 2.0).abs() < 1e-5,
            "expected penalty gradient ≈+2.0 (toward anchor), got {}",
            out[0]
        );
        assert!(
            out[0] > 0.0,
            "penalty gradient must be positive when theta > anchor (toward anchor)"
        );
    }

    #[test]
    fn ewc_fisher_ema_accumulates() {
        let mut fisher = DiagonalFisher::new(1, 0.9).unwrap();

        fisher.observe_gradient(&[1.0]).unwrap();
        assert!(
            (fisher.values[0] - 0.1).abs() < 1e-6,
            "after step 1 expected 0.1, got {}",
            fisher.values[0]
        );

        fisher.observe_gradient(&[1.0]).unwrap();
        assert!(
            (fisher.values[0] - 0.19).abs() < 1e-6,
            "after step 2 expected 0.19, got {}",
            fisher.values[0]
        );
    }

    #[test]
    fn ewc_length_mismatch_errors() {
        let mut fisher = DiagonalFisher::new(4, 0.9).unwrap();
        let result = fisher.observe_gradient(&[1.0, 2.0]); // 2 != 4
        assert!(result.is_err(), "expected Err on length mismatch, got Ok");
    }

    #[test]
    fn diagonal_fisher_new_too_large_returns_err() {
        let result = DiagonalFisher::new(MAX_ALLOWED_ELEMENTS + 1, 0.9);
        assert!(
            matches!(result, Err(FannError::ShapeTooLarge { .. })),
            "expected ShapeTooLarge error for num_params > MAX, got {result:?}"
        );
    }

    #[test]
    fn diagonal_fisher_new_decay_zero_returns_err() {
        let result = DiagonalFisher::new(4, 0.0);
        assert!(
            matches!(result, Err(FannError::InvalidDistributionParams(_))),
            "expected InvalidDistributionParams for decay=0.0, got {result:?}"
        );
    }

    #[test]
    fn diagonal_fisher_new_decay_one_returns_err() {
        let result = DiagonalFisher::new(4, 1.0);
        assert!(
            matches!(result, Err(FannError::InvalidDistributionParams(_))),
            "expected InvalidDistributionParams for decay=1.0, got {result:?}"
        );
    }

    #[test]
    fn diagonal_fisher_new_decay_negative_returns_err() {
        let result = DiagonalFisher::new(4, -0.5);
        assert!(
            matches!(result, Err(FannError::InvalidDistributionParams(_))),
            "expected InvalidDistributionParams for decay=-0.5, got {result:?}"
        );
    }

    #[test]
    fn diagonal_fisher_new_decay_nan_returns_err() {
        let result = DiagonalFisher::new(4, f32::NAN);
        assert!(
            matches!(result, Err(FannError::InvalidDistributionParams(_))),
            "expected InvalidDistributionParams for decay=NaN, got {result:?}"
        );
    }

    #[test]
    fn diagonal_fisher_new_decay_inf_returns_err() {
        let result = DiagonalFisher::new(4, f32::INFINITY);
        assert!(
            matches!(result, Err(FannError::InvalidDistributionParams(_))),
            "expected InvalidDistributionParams for decay=+Inf, got {result:?}"
        );
    }

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
