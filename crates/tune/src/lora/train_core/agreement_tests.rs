use super::{argmax_logit, fold_sequence_agreement};

#[test]
fn argmax_picks_the_greatest_and_keeps_the_first_of_a_tie() {
    assert_eq!(argmax_logit(&[0.1, 0.9, 0.3]), 1);
    assert_eq!(argmax_logit(&[5.0]), 0);
    assert_eq!(argmax_logit(&[-3.0, -1.0, -7.0]), 1);
    // Ties keep the FIRST index. A rewrite to `max_by(partial_cmp)` returns the
    // LAST on a tie, which would silently move every tied position's verdict.
    assert_eq!(argmax_logit(&[2.0, 2.0, 1.0]), 0);
    assert_eq!(argmax_logit(&[1.0, 2.0, 2.0]), 1);
    assert_eq!(argmax_logit(&[0.0, 0.0, 0.0]), 0);
}

#[test]
fn a_nan_never_wins_and_never_panics() {
    // Every comparison against NaN is false, so `>` can neither promote the NaN
    // nor be promoted past it. This is the degenerate input a diverged adapter
    // actually produces, and the obvious `partial_cmp().unwrap()` rewrite panics
    // on it instead of scoring the run.
    assert_eq!(argmax_logit(&[f32::NAN, 1.0, 2.0]), 2);
    assert_eq!(argmax_logit(&[1.0, f32::NAN, 2.0]), 2);
    assert_eq!(argmax_logit(&[2.0, 1.0, f32::NAN]), 0);
    // All-NaN has no defensible answer; it must still return, not panic.
    assert_eq!(argmax_logit(&[f32::NAN, f32::NAN]), 0);
}

#[test]
fn a_sequence_agrees_only_when_every_scored_position_agrees() {
    assert_eq!(
        fold_sequence_agreement([true, true, true].into_iter()),
        (3, 3, true)
    );
    assert_eq!(
        fold_sequence_agreement([true, false, true].into_iter()),
        (2, 3, false)
    );
    assert_eq!(
        fold_sequence_agreement([false, false].into_iter()),
        (0, 2, false)
    );
    // One disagreement at the very end still sinks the sequence: the rule is
    // every position, not most of them.
    assert_eq!(
        fold_sequence_agreement([true, true, false].into_iter()),
        (2, 3, false)
    );
}

#[test]
fn an_empty_sequence_does_not_agree() {
    // The flattering reading of "all positions agreed" is vacuously true on an
    // empty completion. If this ever returns true, an adapter that emits nothing
    // scores a perfect sequence rate, which is the failure that would read as a
    // result rather than as a bug.
    assert_eq!(
        fold_sequence_agreement(std::iter::empty::<bool>()),
        (0, 0, false)
    );
}
