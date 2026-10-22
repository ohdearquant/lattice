//! Q8_0 non-finite input guard.

use super::*;
// -------------------------------------------------------------------
// Q8_0 non-finite input guard — mutation-sensitive (PR #452)
//
// IEEE 754: `NaN > x` is always false, so a fold-max over a block
// containing NaN leaves `amax` unchanged, and the NaN element is
// silently cast to 0 by the `clamp`, producing a plausible-looking
// packed buffer with a wrong entry and no error.
//
// Mutation sensitivity: removing the `if !v.is_finite()` guard in
// `quantize_row_q8_0` converts both Err returns to Ok, making
// `result.is_err()` false and failing both assertions.
// -------------------------------------------------------------------

#[test]
fn test_quantize_row_q8_0_rejects_nan() {
    // A block with one NaN must return Err, not silently encode it as 0.
    let mut vals = vec![1.0f32; 32];
    vals[5] = f32::NAN;
    let result = quantize_row_q8_0(&vals);
    assert!(result.is_err(), "NaN in Q8_0 weight block must return Err");
}

#[test]
fn test_quantize_row_q8_0_rejects_inf() {
    // A block with +inf must return Err; is_finite() rejects NaN, +inf, and -inf.
    let mut vals = vec![1.0f32; 32];
    vals[15] = f32::INFINITY;
    let result = quantize_row_q8_0(&vals);
    assert!(result.is_err(), "+inf in Q8_0 weight block must return Err");
}
