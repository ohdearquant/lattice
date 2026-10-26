//! The speculative extension of the session contract (ADR-090 D2, D6) and its driver. Internal:
//! only the Qwen Metal routes implement it, and the facade reaches it through `__internal`.

use super::driver::check_capabilities;
use super::{Cancellation, ExecutionCapabilities, FinishDisposition};
use crate::decode_budget::decode_cap;
use crate::error::InferenceError;
use crate::generation::{DecodePolicy, GenerateConfig, StepOutcome, TokenLogprob, TopLogprob};
use crate::stop_reason::StopReason;

// ---------------------------------------------------------------------------
// SpeculativeSession / VerifiedRound / SpeculativeTrace
// ---------------------------------------------------------------------------

/// What one speculative round hands the driver (ADR-090 D2, D6): the tokens the round
/// evaluated and committed, in order, and the target's own prediction that follows them.
///
/// Every token in `committed` is the target's greedy successor of the one before it, and
/// was evaluated by the target in this round: the pending token itself, then each draft
/// token the target agreed with. A draft token the target rejected never appears here; the
/// round has already restored the session to the state after the committed span. `next` is
/// the target's prediction for the position after the span. It has not been evaluated and
/// is not yet published: the driver applies the stop policy to it, then offers it as the
/// next round's pending token.
///
/// `committed` may end in a token the stop policy will refuse (an accepted draft that is a
/// stop token): the driver, not the session, decides what is published, so the session
/// reports it and stops there. `next` is `None` exactly when the round ends the request
/// itself, that is when `cache_full` is set or when the last committed token is such a
/// stop token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRound {
    pub committed: Vec<u32>,
    pub next: Option<u32>,
    /// The cache had no room for another round; `committed` holds the pending token alone.
    pub cache_full: bool,
}

/// The typed speculative extension of the execution boundary (ADR-090 D2, D6): a session
/// that verifies draft tokens against the target and returns verified spans, with its own
/// snapshot and rollback. It is a separate trait rather than optional methods on
/// [`DecoderSession`](super::DecoderSession): only the Qwen Metal routes have it, and a session without it cannot
/// be asked to speculate. The driver applies the shared decode policy to what it returns
/// ([`run_speculative`]); the session never publishes a token.
pub trait SpeculativeSession {
    /// The target's greedy prediction for the first output position, read from the prefill
    /// the caller already ran.
    fn first_candidate(&mut self) -> u32;

    /// Evaluates `pending` and verifies draft tokens after it, restoring the session to
    /// the committed span's end before it returns. `room` is the most tokens the driver
    /// can still publish, including `pending`: a session uses it only to avoid verifying
    /// past the length limit. `is_stop` is the driver's own stop predicate: a session uses
    /// it only to cut a draft chain short at a token the driver will refuse, never to
    /// decide what is committed.
    fn advance(
        &mut self,
        pending: u32,
        room: usize,
        is_stop: &dyn Fn(u32) -> bool,
    ) -> Result<VerifiedRound, InferenceError>;

    /// Ends the request. Called once, after the last round the driver ran.
    fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError>;
}

/// Ledger-transition counters for the speculative driver, the bypass detector the
/// per-token [`DriverTrace`](super::driver::DriverTrace) is for the ordinary loop: `offered` counts the tokens
/// the shared policy saw. A request whose output did not pass through the policy leaves
/// `offered` below the number of tokens it published.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeTrace {
    /// Rounds the session ran.
    pub rounds: usize,
    /// Committed tokens offered to the shared decode policy, published or refused.
    pub offered: usize,
}

/// Everything [`run_speculative`] produces. Logprobs and stop-string text are not
/// carried: the speculative routes refuse both, so the driver's own buffers for them stay
/// empty and the caller decodes the published ids.
pub struct SpeculativeResult {
    pub generated_ids: Vec<u32>,
    pub stopped: bool,
    pub stop_reason: StopReason,
    pub trace: SpeculativeTrace,
}

/// A speculative route verifies draft tokens by argmax, so it admits only the greedy
/// configuration its route predicate already selects it for. Refused here as well, in every
/// build, for the reason [`check_capabilities`] is a hard error: a driver that accepted any
/// configuration would let a caller believe sampling or a repetition penalty was applied.
fn check_greedy(gen_cfg: &GenerateConfig) -> Result<(), InferenceError> {
    if gen_cfg.temperature > 0.0 || gen_cfg.top_k > 1 || gen_cfg.repetition_penalty != 1.0 {
        return Err(InferenceError::InvalidInput(
            "a speculative route verifies draft tokens by argmax and admits only a greedy \
             request with no repetition penalty"
                .into(),
        ));
    }
    Ok(())
}

/// The speculative sibling of [`run`](super::driver::run) (ADR-090 D2, D6): one loop over a
/// [`SpeculativeSession`] that applies the same [`DecodePolicy`] to the tokens a round
/// verified and committed, and to nothing else.
///
/// **What the policy sees.** The session returns, per round, the tokens it evaluated and
/// the target's prediction that follows them. Each committed token goes through
/// [`DecodePolicy::transition_with_metadata`], in order, exactly once; a stop token is
/// refused there and ends the request, and the length limit is checked before each offer,
/// so a token past the limit is never published. The prediction that follows the span is
/// checked against the same stop predicate before it becomes the next pending token, and
/// not at all when the span has already filled the limit. A draft token the target
/// rejected is never in a span, so the policy never sees one.
///
/// **Rounds are the session's, not the policy's.** The loop runs a round whenever the limit
/// leaves room for the pending token, as the loops this replaces did, so a request whose
/// last token is the pending one still runs the round that evaluates it. That keeps the
/// forward passes, the end state and the stop reason at the cache boundary identical to
/// those loops; the ordinary driver's one-token-at-a-time shape would not.
///
/// **Controls.** The four optional controls ([`ExecutionCapabilities`]) are refused, since
/// a speculative route wires none of them, and so is a non-greedy request. Cancellation is
/// polled at the top of every round, the point where the ordinary driver polls before a
/// decode step; the prefill ran before this call, so there is no earlier point to poll.
///
/// `decode_delta`, `text`, `token_logprob_end_offsets` and `emit_confirmed` are the raw
/// I/O primitives [`run`](super::driver::run) takes, handed to the same policy call; a text sink that refuses a
/// delta ends the request with [`StopReason::Interrupt`].
#[allow(clippy::too_many_arguments)]
pub fn run_speculative(
    session: &mut dyn SpeculativeSession,
    gen_cfg: &GenerateConfig,
    eos_token_id: u32,
    cancel: &dyn Cancellation,
    decode_delta: impl FnMut(u32) -> String,
    text: &mut String,
    token_logprob_end_offsets: &mut Vec<usize>,
    emit_confirmed: impl FnMut(&str, u32) -> bool,
) -> Result<SpeculativeResult, InferenceError> {
    check_capabilities(ExecutionCapabilities::default(), gen_cfg)?;
    check_greedy(gen_cfg)?;
    let result = run_speculative_inner(
        session,
        gen_cfg,
        eos_token_id,
        cancel,
        decode_delta,
        text,
        token_logprob_end_offsets,
        emit_confirmed,
    );
    let disposition = match &result {
        Ok(result) if result.stop_reason != StopReason::Interrupt => FinishDisposition::Reusable,
        _ => FinishDisposition::Poisoned,
    };
    let finished = session.finish(disposition);
    result.and_then(|result| finished.map(|()| result))
}

#[allow(clippy::too_many_arguments)]
fn run_speculative_inner(
    session: &mut dyn SpeculativeSession,
    gen_cfg: &GenerateConfig,
    eos_token_id: u32,
    cancel: &dyn Cancellation,
    mut decode_delta: impl FnMut(u32) -> String,
    text: &mut String,
    token_logprob_end_offsets: &mut Vec<usize>,
    mut emit_confirmed: impl FnMut(&str, u32) -> bool,
) -> Result<SpeculativeResult, InferenceError> {
    let mut trace = SpeculativeTrace::default();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut token_logprobs: Vec<TokenLogprob> = Vec::new();
    let is_stop = |id: u32| id == eos_token_id || gen_cfg.stop_token_ids.contains(&id);

    // A zero budget generates nothing, before any token is read (the loops this replaces
    // returned here too, ahead of the prefill-derived candidate and its stop check).
    if gen_cfg.max_new_tokens == 0 {
        return Ok(SpeculativeResult {
            generated_ids,
            stopped: false,
            stop_reason: StopReason::Length,
            trace,
        });
    }

    let first = session.first_candidate();
    if is_stop(first) {
        return Ok(SpeculativeResult {
            generated_ids,
            stopped: true,
            stop_reason: StopReason::Eos,
            trace,
        });
    }

    // Capabilities leave no reasoning budget, so the cap is the plain token limit.
    let cap = decode_cap(gen_cfg.effective_reasoning_budget(), gen_cfg.max_new_tokens);
    generated_ids.reserve(cap);
    let mut policy = DecodePolicy::for_verified_stream(gen_cfg, false);
    let mut pending = first;
    let mut stopped = false;
    let mut stop_reason = StopReason::Length;

    'rounds: while generated_ids.len() < cap {
        if cancel.is_cancelled() {
            stop_reason = StopReason::Interrupt;
            break;
        }
        let round = session.advance(pending, cap - generated_ids.len(), &is_stop)?;
        trace.rounds += 1;

        for &token in &round.committed {
            if generated_ids.len() >= cap {
                break 'rounds;
            }
            trace.offered += 1;
            let generated_len_before = generated_ids.len();
            let outcome = policy.transition_with_metadata(
                &mut token_logprobs,
                token,
                generated_len_before,
                |_| Ok(true),
                &is_stop,
                |next_id| generated_ids.push(next_id),
                |_, _| -> Result<(f32, Vec<TopLogprob>), InferenceError> {
                    Err(InferenceError::Inference(
                        "a speculative route records no token metadata".into(),
                    ))
                },
                &mut decode_delta,
                text,
                token_logprob_end_offsets,
                |s, id| emit_confirmed(s, id),
            )?;
            match outcome {
                StepOutcome::GrammarStop => {
                    stopped = true;
                    stop_reason = StopReason::Grammar;
                    break 'rounds;
                }
                StepOutcome::Eos => {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                    break 'rounds;
                }
                StepOutcome::Stopped => {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                    break 'rounds;
                }
                StepOutcome::Interrupted => {
                    stop_reason = StopReason::Interrupt;
                    break 'rounds;
                }
                StepOutcome::Emitted {
                    answer_budget_exhausted,
                    ..
                } => {
                    if answer_budget_exhausted {
                        break 'rounds;
                    }
                }
            }
        }

        // The cache boundary outranks the length limit, as in the loops this replaces: a
        // request whose pending token fills the limit exactly still reports the full cache.
        if round.cache_full {
            stop_reason = StopReason::KvFull;
            break;
        }
        if generated_ids.len() >= cap {
            break;
        }
        let Some(next) = round.next else {
            return Err(InferenceError::Inference(
                "a speculative round ended without a continuation and without a full cache".into(),
            ));
        };
        if is_stop(next) {
            stopped = true;
            stop_reason = StopReason::Eos;
            break;
        }
        pending = next;
    }

    Ok(SpeculativeResult {
        generated_ids,
        stopped,
        stop_reason,
        trace,
    })
}

#[cfg(test)]
mod tests;
