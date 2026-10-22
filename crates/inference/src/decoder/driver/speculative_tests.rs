use super::*;
use crate::decoder::VerifiedRound;
use std::cell::Cell;
use std::collections::VecDeque;

const EOS: u32 = 999;

/// Replays prepared rounds and records what the driver asked of it.
struct ScriptedSession {
    first: u32,
    rounds: VecDeque<VerifiedRound>,
    first_reads: usize,
    seen_pending: Vec<u32>,
    rooms: Vec<usize>,
    finished: bool,
    finishes: Vec<FinishDisposition>,
}

fn round(committed: &[u32], next: u32) -> VerifiedRound {
    VerifiedRound {
        committed: committed.to_vec(),
        next: Some(next),
        cache_full: false,
    }
}

fn scripted(first: u32, rounds: Vec<VerifiedRound>) -> ScriptedSession {
    ScriptedSession {
        first,
        rounds: rounds.into(),
        first_reads: 0,
        seen_pending: Vec::new(),
        rooms: Vec::new(),
        finished: false,
        finishes: Vec::new(),
    }
}

impl SpeculativeSession for ScriptedSession {
    fn first_candidate(&mut self) -> u32 {
        self.first_reads += 1;
        self.first
    }

    fn advance(
        &mut self,
        pending: u32,
        room: usize,
        _is_stop: &dyn Fn(u32) -> bool,
    ) -> Result<VerifiedRound, InferenceError> {
        self.seen_pending.push(pending);
        self.rooms.push(room);
        self.rounds
            .pop_front()
            .ok_or_else(|| InferenceError::Inference("script exhausted".into()))
    }

    fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError> {
        self.finished = true;
        self.finishes.push(disposition);
        Ok(())
    }
}

fn drive(
    session: &mut ScriptedSession,
    gen_cfg: &GenerateConfig,
    cancel: &dyn Cancellation,
) -> Result<SpeculativeResult, InferenceError> {
    let mut text = String::new();
    let mut offsets = Vec::new();
    run_speculative(
        session,
        gen_cfg,
        EOS,
        cancel,
        |_| String::new(),
        &mut text,
        &mut offsets,
        |_, _| true,
    )
}

fn cfg(max_new_tokens: usize) -> GenerateConfig {
    GenerateConfig {
        max_new_tokens,
        temperature: 0.0,
        top_k: 1,
        repetition_penalty: 1.0,
        ..Default::default()
    }
}

#[test]
fn commits_in_order_and_continues_from_the_reported_continuation() {
    let mut session = scripted(1, vec![round(&[1, 2], 3), round(&[3], 4)]);
    let result = drive(&mut session, &cfg(3), &|| false).expect("run");
    assert_eq!(result.generated_ids, vec![1, 2, 3]);
    assert!(!result.stopped);
    assert_eq!(result.stop_reason, StopReason::Length);
    assert_eq!(session.seen_pending, vec![1, 3]);
    assert_eq!(session.rooms, vec![3, 1]);
    assert_eq!(
        result.trace,
        SpeculativeTrace {
            rounds: 2,
            offered: 3
        }
    );
    assert!(session.finished);
}

#[test]
fn a_zero_budget_reads_nothing() {
    let mut session = scripted(1, vec![]);
    let result = drive(&mut session, &cfg(0), &|| false).expect("run");
    assert!(result.generated_ids.is_empty());
    assert!(!result.stopped);
    assert_eq!(result.stop_reason, StopReason::Length);
    assert_eq!(session.first_reads, 0);
    assert!(session.seen_pending.is_empty());
}

#[test]
fn a_stop_token_as_the_first_candidate_ends_the_request_empty() {
    let mut session = scripted(EOS, vec![]);
    let result = drive(&mut session, &cfg(4), &|| false).expect("run");
    assert!(result.generated_ids.is_empty());
    assert!(result.stopped);
    assert_eq!(result.stop_reason, StopReason::Eos);
    assert!(session.seen_pending.is_empty());
}

#[test]
fn a_continuation_that_is_a_stop_token_ends_the_request_without_being_committed() {
    let mut session = scripted(1, vec![round(&[1, 2], 7)]);
    let mut gen_cfg = cfg(8);
    gen_cfg.stop_token_ids = vec![7];
    let result = drive(&mut session, &gen_cfg, &|| false).expect("run");
    assert_eq!(result.generated_ids, vec![1, 2]);
    assert!(result.stopped);
    assert_eq!(result.stop_reason, StopReason::Eos);
    assert_eq!(session.seen_pending, vec![1], "no round past the stop");
}

#[test]
fn a_stop_the_limit_cannot_reach_is_a_length_stop() {
    let mut session = scripted(1, vec![round(&[1, 2], EOS)]);
    let result = drive(&mut session, &cfg(2), &|| false).expect("run");
    assert_eq!(result.generated_ids, vec![1, 2]);
    assert!(!result.stopped, "the stop would land past the limit");
    assert_eq!(result.stop_reason, StopReason::Length);
}

#[test]
fn the_policy_refuses_a_stop_token_inside_a_committed_span() {
    let mut session = scripted(1, vec![round(&[1, EOS, 5], 6)]);
    let result = drive(&mut session, &cfg(8), &|| false).expect("run");
    assert_eq!(result.generated_ids, vec![1]);
    assert!(result.stopped);
    assert_eq!(result.stop_reason, StopReason::Eos);
    assert_eq!(
        result.trace.offered, 2,
        "the stop token is offered, the token after it is not"
    );
}

#[test]
fn a_span_longer_than_the_limit_is_cut_at_the_limit() {
    let mut session = scripted(1, vec![round(&[1, 2, 3], 4)]);
    let result = drive(&mut session, &cfg(2), &|| false).expect("run");
    assert_eq!(result.generated_ids, vec![1, 2]);
    assert_eq!(result.stop_reason, StopReason::Length);
    assert_eq!(result.trace.offered, 2);
}

#[test]
fn a_full_cache_outranks_the_length_limit() {
    let mut session = scripted(
        1,
        vec![VerifiedRound {
            committed: vec![1],
            next: None,
            cache_full: true,
        }],
    );
    let result = drive(&mut session, &cfg(1), &|| false).expect("run");
    assert_eq!(result.generated_ids, vec![1]);
    assert!(!result.stopped);
    assert_eq!(result.stop_reason, StopReason::KvFull);
    assert!(session.finished);
}

#[test]
fn a_round_without_a_continuation_or_a_full_cache_is_an_error() {
    let mut session = scripted(
        1,
        vec![VerifiedRound {
            committed: vec![1],
            next: None,
            cache_full: false,
        }],
    );
    assert!(drive(&mut session, &cfg(4), &|| false).is_err());
}

#[test]
fn cancellation_is_polled_before_each_round() {
    let polls = Cell::new(0usize);
    let cancel = || {
        polls.set(polls.get() + 1);
        polls.get() > 1
    };
    let mut session = scripted(1, vec![round(&[1], 2), round(&[2], 3)]);
    let result = drive(&mut session, &cfg(8), &cancel).expect("run");
    assert_eq!(result.generated_ids, vec![1]);
    assert!(!result.stopped);
    assert_eq!(result.stop_reason, StopReason::Interrupt);
    assert_eq!(
        session.seen_pending,
        vec![1],
        "the cancelled round never ran"
    );
    assert!(session.finished);
}

#[test]
fn controls_a_speculative_route_does_not_wire_are_refused_before_any_read() {
    let mut sampled = cfg(4);
    sampled.temperature = 0.8;
    let mut logprobs = cfg(4);
    logprobs.logprobs = Some(2);
    let mut stop_strings = cfg(4);
    stop_strings.stop_strings = vec!["x".into()];
    let mut penalised = cfg(4);
    penalised.repetition_penalty = 1.1;
    for gen_cfg in [sampled, logprobs, stop_strings, penalised] {
        let mut session = scripted(1, vec![round(&[1], 2)]);
        let err = drive(&mut session, &gen_cfg, &|| false).err();
        assert!(
            matches!(err, Some(InferenceError::InvalidInput(_))),
            "got {err:?}"
        );
        assert_eq!(session.first_reads, 0);
        assert!(!session.finished);
    }
}

#[test]
fn speculative_early_completion_finishes_once() {
    for (first, budget) in [(EOS, 4), (1, 0)] {
        let mut session = scripted(first, Vec::new());
        drive(&mut session, &cfg(budget), &|| false).expect("early completion");
        assert_eq!(session.finishes, [FinishDisposition::Reusable]);
        assert!(session.seen_pending.is_empty());
    }
}

#[test]
fn speculative_advance_failure_finishes_poisoned_once() {
    let mut session = scripted(1, Vec::new());
    let result = drive(&mut session, &cfg(4), &|| false);
    assert!(
        matches!(result, Err(InferenceError::Inference(message)) if message == "script exhausted")
    );
    assert_eq!(session.finishes, [FinishDisposition::Poisoned]);
}
