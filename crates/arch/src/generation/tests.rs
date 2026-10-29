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

fn policy_with_boundaries(
    max_new_tokens: usize,
    reasoning_budget: Option<usize>,
    enable_thinking: bool,
    close_id: Option<u32>,
    boundary_table: Option<std::sync::Arc<[bool]>>,
    first_emitted_id: u32,
) -> DecodePolicy {
    let cfg = GenerateConfig {
        max_new_tokens,
        reasoning_budget,
        enable_thinking,
        logprobs: None,
        ..Default::default()
    };
    let mut logprobs = Vec::new();
    DecodePolicy::init(
        &cfg,
        close_id,
        boundary_table,
        &mut logprobs,
        first_emitted_id,
        &[],
        0.0,
        1,
        false,
    )
}

fn emit_step(policy: &mut DecodePolicy, generated_len_before: usize, sampled_id: u32) -> u32 {
    let mut logprobs = Vec::new();
    let mut text = String::new();
    let mut offsets = Vec::new();
    match policy
        .transition(
            &mut logprobs,
            sampled_id,
            &[],
            0.0,
            generated_len_before,
            |_| Ok(true),
            |_| false,
            |_| {},
            |_| String::new(),
            &mut text,
            &mut offsets,
            |_, _| true,
        )
        .expect("the policy callbacks cannot fail")
    {
        StepOutcome::Emitted { token_id, .. } => token_id,
        _ => panic!("expected an emitted token"),
    }
}

#[test]
fn reasoning_close_waits_for_a_newline_after_the_budget() {
    let close_id = 9u32;
    let newline_id = 8u32;
    let table: std::sync::Arc<[bool]> = vec![
        false, false, false, false, false, false, false, false, true, false,
    ]
    .into();
    let mut policy = policy_with_boundaries(8, Some(4), true, Some(close_id), Some(table), 1);

    for generated_len_before in 1..6 {
        let emitted = emit_step(&mut policy, generated_len_before, 1);
        assert_ne!(emitted, close_id, "the close must wait past the budget");
    }
    assert_eq!(
        emit_step(&mut policy, 6, newline_id),
        newline_id,
        "the newline token itself is emitted before the close can fire"
    );
    assert_eq!(
        emit_step(&mut policy, 7, 1),
        close_id,
        "the step after the newline must force the close"
    );
}

#[test]
fn reasoning_close_falls_back_at_exactly_budget_plus_grace() {
    let close_id = 9u32;
    let table: std::sync::Arc<[bool]> = vec![false; 16].into();
    let mut policy = policy_with_boundaries(40, Some(4), true, Some(close_id), Some(table), 1);

    for generated_len_before in 4..36 {
        assert_ne!(
            emit_step(&mut policy, generated_len_before, 1),
            close_id,
            "without a newline the close must not fire before B + 32"
        );
    }
    assert_eq!(
        emit_step(&mut policy, 36, 1),
        close_id,
        "without a newline the close must fire at exactly B + 32"
    );
}

#[test]
fn reasoning_close_accepts_a_model_sampled_close_inside_the_window() {
    let close_id = 9u32;
    let table: std::sync::Arc<[bool]> = vec![false; 16].into();
    let mut policy = policy_with_boundaries(8, Some(4), true, Some(close_id), Some(table), 1);

    for generated_len_before in 1..4 {
        assert_ne!(emit_step(&mut policy, generated_len_before, 1), close_id);
    }
    assert_eq!(
        emit_step(&mut policy, 4, close_id),
        close_id,
        "a sampled close inside the window is a natural close"
    );
    assert_eq!(
        emit_step(&mut policy, 5, 3),
        3,
        "once naturally closed, no later step may force the close again"
    );
}

#[test]
fn reasoning_close_is_inert_without_an_active_budget() {
    let close_id = 9u32;
    let table: std::sync::Arc<[bool]> = vec![true; 16].into();
    for (budget, thinking) in [(None, true), (Some(0), true), (Some(4), false)] {
        let mut policy =
            policy_with_boundaries(8, budget, thinking, Some(close_id), Some(table.clone()), 1);
        assert_eq!(
            emit_step(&mut policy, 100, 3),
            3,
            "budget {budget:?} with thinking={thinking} must not force a close"
        );
        assert!(
            !policy.thinking_closed,
            "the inert path must leave the block state unchanged"
        );
    }
}

#[test]
fn reasoning_close_cannot_outlive_a_shorter_decode_cap() {
    let close_id = 9u32;
    let table: std::sync::Arc<[bool]> = vec![false; 16].into();
    let mut policy = policy_with_boundaries(2, Some(4), true, Some(close_id), Some(table), 1);
    assert_eq!(
        policy.cap(),
        7,
        "the decode cap must not grow for the grace window"
    );

    let mut generated_len = 1;
    while generated_len < policy.cap() {
        assert_ne!(
            emit_step(&mut policy, generated_len, 1),
            close_id,
            "the cap ends generation while the block is still open"
        );
        generated_len += 1;
    }
    assert_eq!(generated_len, 7);
    assert!(
        !policy.thinking_closed,
        "the cap must end the request before the deferred close"
    );
}

#[test]
fn reasoning_close_is_immediate_after_a_boundary_at_budget_minus_one() {
    let close_id = 9u32;
    let newline_id = 8u32;
    let table: std::sync::Arc<[bool]> = vec![
        false, false, false, false, false, false, false, false, true, false,
    ]
    .into();
    let mut policy = policy_with_boundaries(8, Some(4), true, Some(close_id), Some(table), 1);

    assert_eq!(emit_step(&mut policy, 1, 1), 1);
    assert_eq!(emit_step(&mut policy, 2, 1), 1);
    assert_eq!(emit_step(&mut policy, 3, newline_id), newline_id);
    assert_eq!(
        emit_step(&mut policy, 4, 1),
        close_id,
        "a newline emitted at step B - 1 must close at B without added delay"
    );
}

#[test]
fn force_close_think_cannot_fire_on_the_first_generated_token() {
    let close = 248_069_u32;

    // budget == 1 is the discriminating case and the one an existing test in
    // qwen35_config does not cover: it is where `generated_so_far >= budget`
    // comes closest to holding at step 0, so it is where a wrong baseline
    // would first show. It must still not fire.
    assert_eq!(
        force_close_think(Some(1), true, false, 0, Some(close), false),
        None,
        "budget 1 must not force on the first generated token"
    );
    assert_eq!(
        force_close_think(Some(2), true, false, 0, Some(close), false),
        None,
        "budget 2 must not force on the first generated token"
    );

    // Must-FIRE control. Without it this test would keep passing if
    // `force_close_think` were changed to return None unconditionally.
    assert_eq!(
        force_close_think(Some(1), true, false, 1, Some(close), true),
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
        None,
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
        None,
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
    let mut policy = DecodePolicy::construct(&cfg, None, None, 1, 1, true);
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
        let mut policy = DecodePolicy::construct(&cfg, None, None, 1, 1, true);
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
