use super::has_finite_logit;

/// Mutation sensitivity: change `has_finite_logit` to always return `true`
/// → this assertion fails, catching the regression that would let an
/// all-blocked grammar mask silently reach the sampler.
#[test]
fn all_neg_infinity_has_no_finite_logit() {
    let logits = vec![f32::NEG_INFINITY; 8];
    assert!(
        !has_finite_logit(&logits),
        "an all-NEG_INFINITY buffer must report no finite logit"
    );
}

/// One surviving finite logit (even a very negative one, or NaN elsewhere)
/// must be enough to pass. Mutation sensitivity: change the comparison
/// from `>` to `>=` or invert it → this fails.
#[test]
fn one_finite_logit_among_neg_infinity_passes() {
    let mut logits = vec![f32::NEG_INFINITY; 8];
    logits[3] = -1e30; // finite, just very negative — still a legal token
    assert!(
        has_finite_logit(&logits),
        "one surviving finite logit must be detected"
    );
}

/// NaN is neither `> NEG_INFINITY` nor `<= NEG_INFINITY`; every NaN
/// comparison is false, so an all-NaN buffer must also fail closed.
#[test]
fn all_nan_has_no_finite_logit() {
    let logits = vec![f32::NAN; 8];
    assert!(
        !has_finite_logit(&logits),
        "an all-NaN buffer must also report no finite logit (NaN is not > NEG_INFINITY)"
    );
}

/// A normal, fully-unmasked logit buffer must pass.
#[test]
fn normal_logits_pass() {
    let logits = vec![0.1_f32, -2.0, 5.5, -0.001];
    assert!(
        has_finite_logit(&logits),
        "an ordinary logit buffer must report at least one finite logit"
    );
}
