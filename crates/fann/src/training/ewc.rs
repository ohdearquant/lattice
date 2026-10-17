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
    ///
    /// **Non-finite Fisher fails closed, never propagates NaN/inf.**
    /// `observe_gradient` does not itself reject a non-finite gradient, so a
    /// caller can hand this method a Fisher containing NaN or +-inf (reachable
    /// from an unvalidated caller-supplied gradient upstream). The exact rule:
    /// `F_ref` is accumulated in `f64` (see below); if that `F_ref` is itself
    /// non-finite (a NaN entry poisons the sum into NaN, an infinite entry
    /// poisons it into +-inf), every coordinate of the common prefix — the
    /// first `min(self.values.len(), delta.len())` entries — is set to `0.0`
    /// — the reference magnitude cannot be trusted, so that part of the
    /// update is blocked rather than divided by, or allowed to propagate, a
    /// non-finite value. Elements of `delta` beyond the Fisher's length are
    /// left unchanged, as on every other path through this method. If
    /// `F_ref` is finite but one coordinate's own scale factor still comes
    /// out non-finite, **only that coordinate** is set to `0.0`. Both cases
    /// mirror the effect (not the exact per-coordinate boundary) of the
    /// max-normalised formula this replaced, which zeroed a coordinate
    /// whenever its own scale computation went non-finite.
    ///
    /// `F_ref` is accumulated in `f64` rather than `f32` so that many
    /// large-but-finite Fisher values cannot silently overflow the sum to
    /// `+inf` — which would otherwise disable damping entirely (`v / inf ==
    /// 0` for every finite `v`) without ever going through the non-finite
    /// path above.
    /// See [`docs/training.md`](../../docs/training.md#fisher-weighted-delta-shrinkage) for the derivation and trade-offs.
    pub fn project_delta(&self, delta: &mut [f32]) {
        let n = self.values.len();
        if n == 0 {
            return;
        }

        // f64 accumulation: see the "large-but-finite" note in the doc comment.
        let sum: f64 = self.values.iter().map(|&v| f64::from(v)).sum();
        let f_ref = sum / n as f64;

        if !f_ref.is_finite() {
            // A NaN entry poisons the sum into NaN; an infinite entry poisons
            // it into +-inf. F_ref cannot be trusted as a reference magnitude
            // either way — block the common prefix instead of dividing by (or
            // propagating) a non-finite value. Elements of `delta` beyond
            // `self.values.len()` are left unchanged, matching the zipped
            // per-coordinate loop below. Checked before the degenerate
            // "no signal" comparison below because `NaN < 1e-8` is false, so a
            // NaN F_ref would otherwise fall through to per-coordinate damping.
            let m = n.min(delta.len());
            delta[..m].fill(0.0);
            return;
        }

        // No importance signal has been observed yet — treat as identity.
        if f_ref < 1e-8 {
            return;
        }

        // Fisher-weighted shrinkage: never reaches zero for finite F_i — see
        // docs/training.md and the doc comment above.
        for (d, &v) in delta.iter_mut().zip(self.values.iter()) {
            let factor = 1.0 + f64::from(PROJECT_DELTA_ALPHA) * (f64::from(v) / f_ref);
            // `factor` can still be non-finite for a single coordinate even
            // when F_ref is finite (e.g. this entry's own value is +-inf but
            // canceled out of the f64 sum) — block just that coordinate.
            *d = if factor.is_finite() {
                (f64::from(*d) / factor) as f32
            } else {
                0.0
            };
        }
    }
}

#[cfg(test)]
mod tests;
