use super::*;

/// The three facts that make the prefill-derived first token safe to route
/// through the ordinary per-step sequence.
///
/// ADR-090 D2 requires the first-token exception to be *characterized*
/// rather than inferred from later steps, because the canonical CPU entry
/// runs a differently shaped step 0: it advances grammar on the sampled ID
/// and never calls `apply_override` or `answer_budget_exhausted`, while
/// [`DecodePolicy::transition`] calls both. Reading both paths at
/// `model/qwen35/generation.rs:196-300` against `transition` shows every
/// control is either present at step 0 in an equivalent form or unable to
/// fire there at all. These tests pin the "unable to fire" half, which is
/// the half that is a property of this module rather than of the entry.
///
/// They exist because the shared driver may unify step 0 with later steps
/// only while these hold, and each would be broken by an ordinary-looking
/// change (a `>` relaxed to `>=`, a budget baseline moved) with no other
/// test in this crate going red.
fn cfg_with(max_new_tokens: usize, reasoning_budget: Option<usize>) -> GenerateConfig {
    GenerateConfig {
        max_new_tokens,
        reasoning_budget,
        enable_thinking: true,
        logprobs: None,
        ..Default::default()
    }
}

#[test]
fn force_close_think_cannot_fire_on_the_first_generated_token() {
    let close = 248_069_u32;

    // budget == 1 is the discriminating case and the one an existing test in
    // qwen35_config does not cover: it is where `generated_so_far >= budget`
    // comes closest to holding at step 0, so it is where a wrong baseline
    // would first show. It must still not fire.
    assert_eq!(
        force_close_think(Some(1), true, false, 0, Some(close)),
        None,
        "budget 1 must not force on the first generated token"
    );
    assert_eq!(
        force_close_think(Some(2), true, false, 0, Some(close)),
        None,
        "budget 2 must not force on the first generated token"
    );

    // Must-FIRE control. Without it this test would keep passing if
    // `force_close_think` were changed to return None unconditionally.
    assert_eq!(
        force_close_think(Some(1), true, false, 1, Some(close)),
        Some(close),
        "budget 1 must force once one token has been generated"
    );
}

#[test]
fn answer_budget_cannot_be_exhausted_by_the_first_generated_token() {
    let close = 248_069_u32;
    let cfg = cfg_with(1, Some(4));
    let mut logprobs = Vec::new();
    let logits = vec![0.0_f32; 8];

    // The only way step 0 sets `reasoning_end_len` is by emitting `</think>`
    // as the very first token, which gives end == 1 with exactly one token
    // generated. That is the earliest the answer budget could possibly be
    // exhausted, so it is the case to pin.
    let policy = DecodePolicy::init(
        &cfg,
        Some(close),
        &mut logprobs,
        close,
        &logits,
        cfg.temperature,
        1,
        false,
    );
    assert_eq!(
        policy.reasoning_end_len,
        Some(1),
        "a first token equal to the close ID must capture the reasoning end"
    );
    assert!(
        !policy.answer_budget_exhausted(1),
        "the answer budget must not be exhausted by the first generated token \
             while max_new_tokens >= 1"
    );

    // Must-FIRE control at the boundary: with max_new_tokens == 1 and the
    // reasoning block closed at length 1, length 2 is the first exhausted
    // length. Without this arm the assertion above would survive
    // `answer_budget_exhausted` being stubbed to false.
    assert!(
        policy.answer_budget_exhausted(2),
        "one answer token past the reasoning end must exhaust a budget of 1"
    );
}

#[test]
fn a_zero_length_request_never_reaches_the_first_token_path() {
    // The proof above is conditional on `max_new_tokens >= 1`. That holds
    // because the canonical entry returns before sampling when the request
    // asks for nothing (`model/qwen35/generation.rs`, the
    // `max_new_tokens == 0` guard). Pinned here as the premise it is: if the
    // guard ever moves, `answer_budget_exhausted(1)` with max_new_tokens 0
    // would be `0 >= 0`, i.e. true on the first token.
    let cfg = cfg_with(0, Some(4));
    let mut logprobs = Vec::new();
    let logits = vec![0.0_f32; 8];
    let policy = DecodePolicy::init(
        &cfg,
        Some(248_069),
        &mut logprobs,
        248_069,
        &logits,
        cfg.temperature,
        1,
        false,
    );
    assert!(
        policy.answer_budget_exhausted(1),
        "documents WHY the entry guard is load-bearing: with max_new_tokens 0 \
             the first token would exhaust the answer budget immediately"
    );
}

fn outcome_name(outcome: &StopCheckOutcome) -> &'static str {
    match outcome {
        StopCheckOutcome::Continue => "Continue",
        StopCheckOutcome::Stopped => "Stopped",
        StopCheckOutcome::Interrupted => "Interrupted",
    }
}

/// Builds a streaming policy for `stop_strings`, pushes `first_delta` through the
/// per-step check (accepting everything it emits), then runs `finish_stop(tail)` with
/// a sink answering `accept`. Returns the outcome, every string the flush handed the
/// sink (one entry per call), and the accumulated `text`.
fn finish_after(
    stop_strings: &[&str],
    first_delta: &str,
    tail: &str,
    accept: bool,
) -> (StopCheckOutcome, Vec<String>, String) {
    let cfg = GenerateConfig {
        stop_strings: stop_strings.iter().map(|s| (*s).to_string()).collect(),
        ..cfg_with(8, None)
    };
    let mut policy = DecodePolicy::construct(&cfg, None, 1, 1, true);
    let mut logprobs = Vec::new();
    let mut offsets = Vec::new();
    let mut text = String::new();
    let initial =
        policy.check_initial_stop(&mut logprobs, &mut text, &mut offsets, first_delta, |_| {
            true
        });
    assert_eq!(outcome_name(&initial), "Continue");
    let mut flushed = Vec::new();
    let outcome = policy.finish_stop(&mut text, tail, |s| {
        flushed.push(s.to_string());
        accept
    });
    (outcome, flushed, text)
}

#[test]
fn finish_stop_streaming_reports_a_rejected_tail_flush() {
    // "ZZ" holds back one byte, so "b" of "ab" is still unconfirmed when the loop ends
    // and reaches the sink only through the tail flush.
    let (outcome, flushed, text) = finish_after(&["ZZ"], "ab", "c", false);
    assert_eq!(flushed, vec!["bc".to_string()]);
    assert_eq!(outcome_name(&outcome), "Interrupted");
    assert_eq!(text, "abc");

    let (outcome, flushed, _) = finish_after(&["ZZ"], "ab", "c", true);
    assert_eq!(flushed, vec!["bc".to_string()]);
    assert_eq!(
        outcome_name(&outcome),
        "Continue",
        "control: the same flush accepted by the sink is not an interruption"
    );
}

#[test]
fn finish_stop_streaming_rejection_wins_over_a_stop_match_in_the_tail() {
    // The tail completes "ZZ", and the flush emits the text before it ("ab").
    let (outcome, flushed, text) = finish_after(&["ZZ"], "a", "bZZ", false);
    assert_eq!(flushed, vec!["ab".to_string()]);
    assert_eq!(
        outcome_name(&outcome),
        "Interrupted",
        "per-step stop_check reports Interrupted over Stopped; the tail flush must agree"
    );
    assert_eq!(text, "ab");

    let (outcome, _, _) = finish_after(&["ZZ"], "a", "bZZ", true);
    assert_eq!(
        outcome_name(&outcome),
        "Stopped",
        "control: an accepted flush that completes a stop match reports Stopped"
    );
}

#[test]
fn finish_stop_disabled_reports_a_rejected_tail_flush() {
    let run = |tail: &str, accept: bool| {
        let cfg = cfg_with(8, None);
        let mut policy = DecodePolicy::construct(&cfg, None, 1, 1, true);
        let mut text = String::new();
        let mut calls = 0;
        let outcome = policy.finish_stop(&mut text, tail, |_| {
            calls += 1;
            accept
        });
        (outcome, calls, text)
    };

    let (outcome, calls, text) = run("xy", false);
    assert_eq!(outcome_name(&outcome), "Interrupted");
    assert_eq!(calls, 1);
    assert_eq!(text, "xy");

    let (outcome, calls, _) = run("xy", true);
    assert_eq!(outcome_name(&outcome), "Continue", "control: accepted tail");
    assert_eq!(calls, 1);

    let (outcome, calls, _) = run("", false);
    assert_eq!(
        outcome_name(&outcome),
        "Continue",
        "an empty tail emits nothing, so nothing can be rejected"
    );
    assert_eq!(calls, 0);
}
