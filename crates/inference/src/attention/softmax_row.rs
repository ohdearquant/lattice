//! ADR-080 cluster C1: shared fail-closed softmax row-finalizer contract.
//!
//! Canonical semantics, established in [`crate::attention::gqa::apply_gqa_attention`]'s
//! fused softmax and [`crate::attention::decode::decode_attention`]'s online softmax:
//! a softmax row is zeroed **in full** when any input score is NaN, the row max is
//! `+inf`, or the accumulated denominator is non-positive or non-finite (fail-closed).
//! No site may silently drop a single lane, clamp a masked `-inf` sentinel into
//! finite probability mass, or emit a partially-normalized row.
//!
//! These helpers implement the shared *decision* logic only. Each call site keeps
//! its own backend-appropriate exp implementation (real [`f32::exp`], the
//! Schraudolph `fast_exp` bit-trick approximation, or a NEON/AVX2 vectorized
//! variant) per ADR-080's "no numeric behavior changes beyond the documented
//! fail-closed fixes" -- this module does not choose or perform exp for callers.
//! All functions here are allocation-free and `#[inline]`.

/// Scan `row` for its max value and whether any element is NaN.
///
/// A plain `f32::max`-based fold silently *drops* a lone NaN operand (IEEE
/// `maxNum` semantics: `f32::max(x, NaN) == x`), so a row containing exactly one
/// NaN score would otherwise report a finite max and let that NaN escape
/// detection entirely (it can then survive as a single bad lane instead of
/// failing the whole row closed). This scans explicitly so callers can fail
/// closed on NaN regardless of where it falls relative to the true max.
#[inline]
pub fn row_max_and_any_nan(row: &[f32]) -> (f32, bool) {
    let mut max_val = f32::NEG_INFINITY;
    let mut any_nan = false;
    for &v in row {
        if v.is_nan() {
            any_nan = true;
        } else {
            max_val = max_val.max(v);
        }
    }
    (max_val, any_nan)
}

/// True iff the row must fail closed based on the pre-exp scan alone: any NaN
/// present, or the row max is not finite (`+inf`, or `-inf` for a fully-masked
/// row with no valid key). Both cases zero the whole row per the canonical
/// contract, without needing to compute a single exp.
#[inline]
pub fn row_fails_closed_pre_exp(max_val: f32, any_nan: bool) -> bool {
    any_nan || !max_val.is_finite()
}

/// Exact-zero guard for a structurally-masked (`-inf`) lane.
///
/// Fast/approximate exp implementations (the Schraudolph `fast_exp` bit-trick
/// used across this crate) clamp their input to a finite floor for range
/// safety; without this guard a `-inf` mask sentinel round-trips through
/// `fast_exp(-inf - max)` into a small but nonzero weight instead of exact
/// `0.0` (#740). Callers combine this with their own exp:
/// `if is_masked_neg_inf(v) { 0.0 } else { fast_exp(v - max) }`.
#[inline]
pub fn is_masked_neg_inf(v: f32) -> bool {
    v == f32::NEG_INFINITY
}

/// Normalize `row` in place by `sum` if the row is well-formed (finite,
/// strictly positive denominator); otherwise zero the entire row. This is the
/// shared final decision every canonical site already applies
/// (`sum > 0.0 && sum.is_finite()`), extracted so every consolidated call site
/// shares one fail-closed implementation instead of reimplementing the branch.
#[inline]
pub fn finalize_row(row: &mut [f32], sum: f32) {
    if sum.is_finite() && sum > 0.0 {
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    } else {
        row.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_max_and_any_nan_ignores_nan_in_max_but_flags_it() {
        let row = [1.0f32, f32::NAN, 3.0, 2.0];
        let (max_val, any_nan) = row_max_and_any_nan(&row);
        assert_eq!(max_val, 3.0);
        assert!(any_nan);
    }

    #[test]
    fn row_max_and_any_nan_all_neg_inf() {
        let row = [f32::NEG_INFINITY; 4];
        let (max_val, any_nan) = row_max_and_any_nan(&row);
        assert_eq!(max_val, f32::NEG_INFINITY);
        assert!(!any_nan);
    }

    #[test]
    fn row_fails_closed_pre_exp_nan() {
        assert!(row_fails_closed_pre_exp(3.0, true));
    }

    #[test]
    fn row_fails_closed_pre_exp_pos_inf_max() {
        assert!(row_fails_closed_pre_exp(f32::INFINITY, false));
    }

    #[test]
    fn row_fails_closed_pre_exp_neg_inf_max() {
        assert!(row_fails_closed_pre_exp(f32::NEG_INFINITY, false));
    }

    #[test]
    fn row_fails_closed_pre_exp_finite_ok() {
        assert!(!row_fails_closed_pre_exp(5.0, false));
    }

    #[test]
    fn is_masked_neg_inf_exact() {
        assert!(is_masked_neg_inf(f32::NEG_INFINITY));
        assert!(!is_masked_neg_inf(-1.0e30));
        assert!(!is_masked_neg_inf(f32::NAN));
    }

    #[test]
    fn finalize_row_normalizes_well_formed_sum() {
        let mut row = [1.0f32, 1.0, 2.0];
        finalize_row(&mut row, 4.0);
        assert_eq!(row, [0.25, 0.25, 0.5]);
    }

    #[test]
    fn finalize_row_zeros_on_nan_sum() {
        let mut row = [1.0f32, f32::NAN, 2.0];
        finalize_row(&mut row, f32::NAN);
        assert_eq!(row, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn finalize_row_zeros_on_zero_sum() {
        let mut row = [0.0f32, 0.0, 0.0];
        finalize_row(&mut row, 0.0);
        assert_eq!(row, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn finalize_row_zeros_on_negative_sum() {
        let mut row = [1.0f32, 2.0];
        finalize_row(&mut row, -1.0);
        assert_eq!(row, [0.0, 0.0]);
    }

    #[test]
    fn finalize_row_zeros_on_infinite_sum() {
        let mut row = [1.0f32, 2.0];
        finalize_row(&mut row, f32::INFINITY);
        assert_eq!(row, [0.0, 0.0]);
    }
}
