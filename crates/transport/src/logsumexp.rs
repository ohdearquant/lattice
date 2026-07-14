//! Numerically stable helpers for log-domain Sinkhorn updates.
//!
//! Scaling vectors stay in log space; exponentiation only recovers transport mass.

use super::math::{abs, exp, ln, log1p, sqrt as math_sqrt};

/// Sentinel representing `log(0)`.
pub const LOG_ZERO: f32 = f32::NEG_INFINITY;

/// Conservative `f32` exponential underflow cutoff.
pub const EXP_UNDERFLOW_CUTOFF: f32 = -103.97208;

/// Natural logarithm with a caller-provided positive floor.
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
#[inline]
pub fn safe_exp(log_value: f32) -> f32 {
    if log_value < EXP_UNDERFLOW_CUTOFF {
        0.0
    } else {
        exp(log_value)
    }
}

/// Stable `log(exp(a) + exp(b))`.
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

/// One-pass log-sum-exp accumulator.
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
pub fn logsumexp(values: &[f32]) -> f32 {
    let mut acc = LOG_ZERO;
    for &value in values {
        acc = logaddexp(acc, value);
    }
    acc
}

/// Stable log-sum-exp over values generated for `0..len`.
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
pub fn normalize_log_weights(log_weights: &mut [f32]) -> f32 {
    let normalizer = logsumexp(log_weights);
    if normalizer.is_finite() {
        for value in log_weights {
            *value -= normalizer;
        }
    }
    normalizer
}

/// Maximum absolute difference, returning the first non-finite delta.
pub fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut max_delta = 0.0;
    for (&left, &right) in lhs.iter().zip(rhs.iter()) {
        let delta = abs(left - right);
        if !delta.is_finite() {
            return delta;
        }
        if delta > max_delta {
            max_delta = delta;
        }
    }
    max_delta
}

/// Converts log-probabilities to a freshly allocated probability vector.
pub fn exp_normalized(log_weights: &[f32]) -> Vec<f32> {
    log_weights.iter().map(|&value| safe_exp(value)).collect()
}

/// Mean and standard deviation in one pass using Welford's online algorithm.
/// Drift diagnostics use this for outlier detection.
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
    if scratch.len().is_multiple_of(2) {
        0.5 * (scratch[mid - 1] + scratch[mid])
    } else {
        scratch[mid]
    }
}

/// Square root wrapper.
#[inline]
pub fn sqrt(value: f32) -> f32 {
    math_sqrt(value)
}

/// Sum of a slice.
#[inline]
pub fn sum(values: &[f32]) -> f32 {
    values.iter().copied().sum()
}

/// Checks whether all values are finite.
#[inline]
pub fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// Stable `x * ln(x / y) - x + y`, with the usual `0 ln 0 = 0` convention.
pub fn kl_term(x: f32, y: f32, floor: f32) -> f32 {
    if x <= 0.0 {
        return y.max(0.0);
    }
    let x_clamped = x.max(floor);
    let y_clamped = y.max(floor);
    x_clamped * (safe_ln(x_clamped, floor) - safe_ln(y_clamped, floor)) - x_clamped + y_clamped
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Non-finite deltas must not appear converged.
    #[test]
    fn max_abs_diff_is_nan_and_inf_honest() {
        let clean = [1.0f32, 2.0, 3.0];

        let nan_side = [1.0f32, f32::NAN, 3.0];
        let d = max_abs_diff(&clean, &nan_side);
        assert!(
            !d.is_finite(),
            "max_abs_diff silently dropped a NaN operand (got {d}); the gate is blind"
        );

        let inf_side = [1.0f32, f32::INFINITY, 3.0];
        let di = max_abs_diff(&clean, &inf_side);
        assert!(
            !di.is_finite(),
            "max_abs_diff silently dropped an Inf operand (got {di}); the gate is blind"
        );
    }
}
