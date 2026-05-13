//! Numerically stable helpers for log-domain Sinkhorn updates.
//!
//! The core design rule in this crate is simple: never materialize the Gibbs
//! kernel `K = exp(-C / epsilon)` directly. Instead we keep all scaling vectors
//! in log space and only exponentiate when we intentionally recover transport
//! mass. These helpers centralize the handful of stable transforms that the
//! rest of the module relies on.

use super::math::{abs, exp, ln, log1p, sqrt as math_sqrt};

/// Sentinel representing `log(0)`.
///
/// **Unstable**: numerical internal; value is mathematically fixed but export path may change.
pub const LOG_ZERO: f32 = f32::NEG_INFINITY;

/// Conservative cutoff before `expf` underflows in `f32`.
///
/// `exp(-87.0)` is ~1.6e-38, still a representable positive `f32`. Actual
/// underflow is near the smallest subnormal value.
///
/// **Unstable**: tuning constant; value could change if benchmarking reveals a better cutoff.
pub const EXP_UNDERFLOW_CUTOFF: f32 = -103.97208;

/// Safe natural logarithm that clamps the input away from zero.
///
/// In balanced OT the marginals should be strictly positive. In production,
/// however, users often pass weights with zeros after filtering or importance
/// truncation. Clamping to a tiny floor makes the algorithm robust while keeping
/// the perturbation controlled and explicit.
///
/// **Unstable**: numerical internal; signature or clamping semantics may be adjusted.
#[inline]
pub fn safe_ln(value: f32, floor: f32) -> f32 {
    let clamped = if value.is_finite() {
        value.max(floor)
    } else {
        floor
    };
    ln(clamped)
}

/// Safe exponential used only when recovering transport mass or converting a
/// normalized log probability back to probability space.
///
/// **Unstable**: numerical internal; underflow cutoff value may be tuned.
#[inline]
pub fn safe_exp(log_value: f32) -> f32 {
    if log_value < EXP_UNDERFLOW_CUTOFF {
        0.0
    } else {
        exp(log_value)
    }
}

/// Stable `log(exp(a) + exp(b))`.
///
/// **Unstable**: core Sinkhorn arithmetic primitive; always correct but module placement may change.
#[inline]
pub fn logaddexp(a: f32, b: f32) -> f32 {
    if a == LOG_ZERO {
        return b;
    }
    if b == LOG_ZERO {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + log1p(exp(lo - hi))
}

/// One-pass online log-sum-exp accumulator.
///
/// Tracks `(max, sum_of_exp_offsets)` and finalizes as `max + ln(sum)`.
/// Compared with chaining `logaddexp`, this avoids one `log1p` call per term
/// (only one `ln` at the end), reducing transcendental pressure in inner Sinkhorn
/// loops by roughly 1×exp per element.
///
/// **Unstable**: internal optimization primitive used by solver inner loops.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OnlineLogSumExp {
    max: f32,
    sum: f32,
}

impl OnlineLogSumExp {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            max: f32::NEG_INFINITY,
            sum: 0.0,
        }
    }

    #[inline]
    pub(crate) fn push(&mut self, x: f32) {
        if self.max == f32::NEG_INFINITY {
            self.max = x;
            self.sum = 1.0;
        } else if x <= self.max {
            self.sum += exp(x - self.max);
        } else {
            self.sum = self.sum * exp(self.max - x) + 1.0;
            self.max = x;
        }
    }

    #[inline]
    pub(crate) fn finish(self) -> f32 {
        if self.sum == 0.0 {
            f32::NEG_INFINITY
        } else {
            self.max + ln(self.sum)
        }
    }
}

/// Stable log-sum-exp over a slice.
///
/// **Unstable**: numerical internal; may be replaced by the iterator-based variant.
pub fn logsumexp(values: &[f32]) -> f32 {
    let mut acc = LOG_ZERO;
    for &value in values {
        acc = logaddexp(acc, value);
    }
    acc
}

/// Stable log-sum-exp over an implicit iterator `0..len`.
///
/// This is particularly useful when the values are generated on the fly from a
/// cost matrix and we do not want to allocate a temporary buffer.
///
/// **Unstable**: convenience variant; may be unified with `logsumexp` via an `impl Iterator` parameter.
pub fn logsumexp_by<F>(len: usize, mut f: F) -> f32
where
    F: FnMut(usize) -> f32,
{
    let mut acc = LOG_ZERO;
    for idx in 0..len {
        acc = logaddexp(acc, f(idx));
    }
    acc
}

/// Subtracts `logsumexp(log_weights)` in place so that the exponentiated values
/// sum to one. Returns the removed normalizer.
///
/// **Unstable**: barycenter internal; in-place mutation signature may be revised.
pub fn normalize_log_weights(log_weights: &mut [f32]) -> f32 {
    let normalizer = logsumexp(log_weights);
    if normalizer.is_finite() {
        for value in log_weights {
            *value -= normalizer;
        }
    }
    normalizer
}

/// Maximum absolute difference between two same-length slices.
///
/// **Unstable**: barycenter convergence helper; may be inlined.
pub fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut max_delta = 0.0;
    for (&left, &right) in lhs.iter().zip(rhs.iter()) {
        let delta = abs(left - right);
        if delta > max_delta {
            max_delta = delta;
        }
    }
    max_delta
}

/// Converts log-probabilities to a freshly allocated probability vector.
///
/// **Unstable**: convenience converter; allocates — may be replaced with an iterator adapter.
pub fn exp_normalized(log_weights: &[f32]) -> Vec<f32> {
    log_weights.iter().map(|&value| safe_exp(value)).collect()
}

/// Mean and standard deviation in one pass using Welford's online algorithm.
/// Drift diagnostics use this for outlier detection.
///
/// **Unstable**: drift summary internal; may be moved to a dedicated stats module.
pub fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mut mean = 0.0f32;
    let mut m2 = 0.0f32;
    let mut count = 0.0f32;

    for &value in values {
        count += 1.0;
        let delta = value - mean;
        mean += delta / count;
        let delta2 = value - mean;
        m2 += delta * delta2;
    }

    let variance = if count > 1.0 { m2 / count } else { 0.0 };
    (mean, math_sqrt(variance.max(0.0)))
}

/// Median computed by sorting a scratch copy. Used only in high-level
/// drift summaries, so the extra allocation is acceptable.
///
/// **Unstable**: drift summary internal; sorting allocation may be avoided with a selection algorithm.
pub fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mut scratch = values.to_vec();
    scratch.sort_by(|left, right| {
        left.partial_cmp(right)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mid = scratch.len() / 2;
    if scratch.len() % 2 == 0 {
        0.5 * (scratch[mid - 1] + scratch[mid])
    } else {
        scratch[mid]
    }
}

/// Square root wrapper.
///
/// **Unstable**: thin shim over `f32::sqrt`; may be removed if callers use `f32::sqrt` directly.
#[inline]
pub fn sqrt(value: f32) -> f32 {
    math_sqrt(value)
}

/// Sum of a slice.
///
/// **Unstable**: convenience wrapper; may be removed in favor of iterator `.sum()` at call sites.
#[inline]
pub fn sum(values: &[f32]) -> f32 {
    values.iter().copied().sum()
}

/// Checks whether all values are finite.
///
/// **Unstable**: validation helper used in problem setup; may be inlined.
#[inline]
pub fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// Stable `x * ln(x / y) - x + y`, with the usual `0 ln 0 = 0` convention.
///
/// **Unstable**: KL-divergence term used only by `UnbalancedSinkhornSolver`; may move to `unbalanced.rs`.
pub fn kl_term(x: f32, y: f32, floor: f32) -> f32 {
    if x <= 0.0 {
        return y.max(0.0);
    }
    let x_clamped = x.max(floor);
    let y_clamped = y.max(floor);
    x_clamped * (safe_ln(x_clamped, floor) - safe_ln(y_clamped, floor)) - x_clamped + y_clamped
}
