use super::argmax_f32_first_wins;

/// Mutation sensitivity: replacing the strict `>` comparison with `>=`
/// (or switching to `Iterator::max_by`, which keeps the LAST tied
/// element) flips this to return index 2 instead of 1.
#[test]
fn tie_break_is_first_wins() {
    let logits = [0.0_f32, 1.0, 1.0, 0.5];
    assert_eq!(
        argmax_f32_first_wins(&logits),
        1,
        "tied maximum logits must resolve to the FIRST index, not the last"
    );
}

#[test]
fn no_tie_returns_true_max() {
    let logits = [0.1_f32, -2.0, 5.5, -0.001];
    assert_eq!(argmax_f32_first_wins(&logits), 2);
}

/// Mutation sensitivity: initializing `best_val` to `0.0` instead of
/// `NEG_INFINITY` would make this return 0 (all-negative logits never
/// beat a 0.0 floor) instead of the true argmax at index 1.
#[test]
fn all_negative_logits_still_find_true_max() {
    let logits = [-5.0_f32, -1.0, -3.0];
    assert_eq!(argmax_f32_first_wins(&logits), 1);
}

/// A NaN in a non-max position must never win: `NaN > best_val` is
/// always false, so the comparison silently skips it.
#[test]
fn nan_in_nonmax_position_is_skipped() {
    let logits = [1.0_f32, f32::NAN, 2.0];
    assert_eq!(argmax_f32_first_wins(&logits), 2);
}

/// All-NaN input must fail closed to index 0, never panic.
#[test]
fn all_nan_fails_closed_to_zero() {
    let logits = [f32::NAN, f32::NAN, f32::NAN];
    assert_eq!(argmax_f32_first_wins(&logits), 0);
}

/// Empty input must fail closed to index 0, never panic.
#[test]
fn empty_slice_fails_closed_to_zero() {
    let logits: [f32; 0] = [];
    assert_eq!(argmax_f32_first_wins(&logits), 0);
}

/// All-`NEG_INFINITY` (a fully-masked grammar/top-k step) must fail
/// closed to index 0.
#[test]
fn all_neg_infinity_fails_closed_to_zero() {
    let logits = [f32::NEG_INFINITY; 4];
    assert_eq!(argmax_f32_first_wins(&logits), 0);
}

#[test]
fn scalar_sampling_matches_previous_first_wins_loop() {
    fn previous(logits: &[f32]) -> u32 {
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        best_idx
    }
    let cases: &[&[f32]] = &[
        &[],
        &[1.0, 2.0, 2.0],
        &[1.0, f32::NAN, 2.0],
        &[f32::NAN, f32::NAN],
        &[f32::NEG_INFINITY, f32::NEG_INFINITY],
    ];
    for logits in cases {
        assert_eq!(super::argmax_f32_scalar(logits), previous(logits));
    }
}
