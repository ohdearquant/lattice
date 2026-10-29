use crate::__internal::{check_prompt_not_empty, has_finite_logit, initial_rng_state};
use crate::error::InferenceError;

/// An all-blocking grammar mask must be detected by `has_finite_logit` so the
/// caller can return a typed error rather than silently emitting token 0.
///
/// When every logit is NEG_INFINITY, the sampler's non-finite-max short-circuit
/// returns token 0 (the lowest-id token after sorting an all-NEG_INFINITY
/// candidate set). The `has_finite_logit` guard catches this before the sampler
/// is invoked and allows the generation loop to return `InvalidInput` instead.
///
/// Mutation sensitivity:
///   Change `has_finite_logit` to always return `true` →
///   `assert!(!has_finite_logit(...))` fails. This proves the guard is load-bearing
///   for the empty-mask fail-closed path.
#[test]
fn grammar_all_blocked_mask_detected_by_has_finite_logit() {
    // Simulate a grammar engine that blocked every token.
    let all_blocked = vec![f32::NEG_INFINITY; 8];

    assert!(
        !has_finite_logit(&all_blocked),
        "all-NEG_INFINITY logit buffer must NOT pass the has_finite_logit guard; \
             the caller must return a typed error, not silently emit token 0"
    );

    // A buffer with a single finite logit must pass the guard.
    let mut one_allowed = vec![f32::NEG_INFINITY; 8];
    one_allowed[3] = 1.0_f32;
    assert!(
        has_finite_logit(&one_allowed),
        "a single finite logit must pass the guard (grammar still has valid tokens)"
    );
}

/// Mutation sensitivity: change `check_prompt_not_empty` to always return
/// `Ok(())` → this assertion fails, catching a regression that would let
/// an empty prompt silently pass through every entry point that calls it
/// (the CPU-vs-Metal split #856 fixes).
#[test]
fn check_prompt_not_empty_rejects_zero() {
    let result = check_prompt_not_empty(0);
    assert!(
        matches!(result, Err(InferenceError::Inference(ref msg)) if msg.contains("empty prompt")),
        "prompt_len == 0 must be rejected with Err(Inference(\"empty prompt\")); got {result:?}"
    );
}

/// Mutation sensitivity: change the guard to always return `Err(..)` →
/// this assertion fails, catching a regression that would reject every
/// non-empty-prompt caller too.
#[test]
fn check_prompt_not_empty_allows_nonzero() {
    assert!(
        check_prompt_not_empty(1).is_ok(),
        "prompt_len == 1 (a real prompt) must be allowed"
    );
}

#[test]
fn generation_seed_mapping_preserves_explicit_seed_values() {
    for (seed, expected) in [(0, 1), (1, 1), (17, 17), (u64::MAX, u64::MAX)] {
        assert_eq!(initial_rng_state(Some(seed)), expected);
    }
}

#[test]
fn masked_logit_guard_preserves_nan_and_positive_infinity_behavior() {
    assert!(!has_finite_logit(&[]));
    assert!(!has_finite_logit(&[f32::NAN, f32::NEG_INFINITY]));
    assert!(has_finite_logit(&[f32::INFINITY]));
    assert!(has_finite_logit(&[f32::NAN, 0.0]));
}

#[test]
fn empty_prompt_admission_keeps_the_exact_error() {
    match check_prompt_not_empty(0) {
        Err(InferenceError::Inference(message)) => assert_eq!(message, "empty prompt"),
        other => panic!("unexpected empty-prompt result: {other:?}"),
    }
}
