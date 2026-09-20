//! The autoregressive driver (ADR-090 D1/D2, row C): one loop over `&mut dyn
//! DecoderSession` that drives the existing [`DecodePolicy`] unchanged.
//!
//! **Scope, stated once here rather than at every call site.** This driver
//! does not route grammar-constrained decoding, and callers must not invoke it
//! with `gen_cfg.grammar.is_some()` (checked with a `debug_assert!` below, and
//! enforced at the dispatch point in `model::qwen35::generation` by routing a
//! set `grammar` to the pre-existing inline implementation instead). Two
//! independent facts force this, not one:
//!
//! 1. Grammar *masking* needs mutable access to a `GrammarState` at the exact
//!    point a candidate is sampled, which happens inside `QwenCpuSession::select`
//!    over logits this module cannot see (`&mut dyn DecoderSession` has no
//!    such accessor, and D1 forbids downcasting to reach one).
//! 2. Grammar *advance* must run inside `DecodePolicy::transition`'s fixed
//!    internal order, before the EOS check -- which means it has to be a
//!    closure this driver passes to `transition`, and that closure would need
//!    to reach the *same* mutable `GrammarState` `select` used, across a call
//!    the driver does not own. Nothing in the row-A trait provides that reach
//!    without adding a method to it, which is a real API decision this row
//!    does not make unilaterally.
//!
//! This driver also does not route `gen_cfg.logprobs.is_some()`, for a
//! narrower version of the same reason: `DecodePolicy::init`/`transition`
//! (unchanged, per this row's own constraint) take raw `logits: &[f32]`
//! directly, and `&mut dyn DecoderSession` exposes no way to read the logits
//! a `select()` call just sampled from. Where `gen_cfg.logprobs.is_none()` is
//! asserted at entry, this driver passes an empty, never-read slice for that
//! parameter -- `DecodePolicy::record_logprob`'s own body returns before
//! touching it whenever `self.logprobs` is `None` (`crate::generation`,
//! `record_logprob`'s `let Some(top_n) = self.logprobs else { return; }`).
//!
//! Both gaps are named, not silently absorbed: see the row's report for why
//! neither is exercised by the golden or by any pre-existing decode-loop-level
//! test (grammar's own regression test only covers the pre-loop masking site
//! this row does not touch; no test anywhere sets `logprobs` through the
//! two loop helpers this row replaces).
use super::{AcceptedToken, DecoderSession, FinishDisposition, SelectionRequest};
use crate::error::InferenceError;
use crate::generation::{
    DecodePolicy, GenerateConfig, StepOutcome, StopCheckOutcome, TokenLogprob,
};
use crate::stop_reason::StopReason;

/// Ledger-transition counters the driver maintains as it runs: one prediction
/// is opened per `select` call, one is consumed per `decode` call. ADR-090
/// decomposition, "Open question 3, ANSWERED": the golden proves the tokens
/// are real; this proves they came through the driver. A step routed around
/// the driver (the bypass control this row's acceptance requires) leaves
/// `consumed` short of what a real run would have produced, in a way the
/// golden's token-id comparison cannot see by construction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DriverTrace {
    pub(crate) opened: usize,
    pub(crate) consumed: usize,
}

/// Everything [`run`] produces, mirroring the pieces `model::qwen35::generation`'s
/// `generate()` currently assembles by hand from `decode_loop`/`decode_loop_with_stops`'s
/// return value plus its own local `generated_ids`/`token_logprobs`.
pub(crate) struct DriverResult {
    pub(crate) generated_ids: Vec<u32>,
    pub(crate) token_logprobs: Vec<TokenLogprob>,
    pub(crate) stopped: bool,
    pub(crate) stop_reason: StopReason,
    /// Set only on [`StepOutcome::Stopped`] (a confirmed stop-string match);
    /// the stop-string caller uses it exactly as `decode_loop_with_stops` uses
    /// its own local of the same name, to skip a redundant tail-flush attempt.
    pub(crate) confirmed_stop_string_match: bool,
    pub(crate) trace: DriverTrace,
}

/// Runs prefill, then the prefill-derived first token, then the standard
/// per-step sequence for every following token, over `session` -- uniformly,
/// per ADR-090's decomposition "Open question 1, ANSWERED": every
/// `DecodePolicy::transition` control is either present at step 0 in an
/// equivalent form or provably unable to fire there, so folding step 0 into
/// the same `select`/`decode` cycle as later steps changes nothing observable.
/// Folding it in is not optional here: `AcceptedToken` can only be constructed
/// from a `PredictionId` minted by `PredictionLedger::open`, which only
/// `select()` can call, so the first `decode()` this loop issues (consuming
/// the first token's prediction) has no valid token to consume unless step 0
/// went through `select()` too.
///
/// `eos_token_id` is threaded in directly (rather than a `&Qwen35Config`)
/// because it is the one field `should_stop_token`-equivalent logic needs that
/// `GenerateConfig` does not already carry (`gen_cfg.stop_token_ids` covers
/// the rest); the driver otherwise has no reachable model handle at all,
/// deliberately, per D1 ("the driver never downcasts the session").
///
/// `decode_delta` / `text` / `token_logprob_end_offsets` / `emit_confirmed`
/// are the same raw I/O primitives `DecodePolicy::transition` /
/// `check_initial_stop` already take (`crate::generation::DecodePolicy`) --
/// pass a no-op `decode_delta` returning `String::new()` and throwaway
/// buffers for the fast (no stop-strings) path, and a real incremental
/// detokenizer plus the real output buffers for the stop-string path. This
/// function does not need to know which: `policy`'s own `StopMode` (fixed at
/// construction from the real `gen_cfg.stop_strings`, exactly as today)
/// decides whether the calls do real work or nothing, exactly as
/// `decode_loop` already relies on for its own throwaway values.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    session: &mut dyn DecoderSession,
    gen_cfg: &GenerateConfig,
    think_close_id: Option<u32>,
    prompt_ids: &[u32],
    eos_token_id: u32,
    streaming: bool,
    mut decode_delta: impl FnMut(u32) -> String,
    text: &mut String,
    token_logprob_end_offsets: &mut Vec<usize>,
    mut emit_confirmed: impl FnMut(&str, u32) -> bool,
) -> Result<DriverResult, InferenceError> {
    debug_assert!(
        gen_cfg.grammar.is_none() && gen_cfg.logprobs.is_none(),
        "decoder::driver::run does not route grammar or logprobs; see the module doc comment"
    );
    // D3: capabilities are negotiated per session, and "one driver over many sessions" (D1)
    // means a session that does not declare a control this call actually uses is a caller
    // bug, not this driver's problem to route around silently.
    let caps = *session.capabilities();
    debug_assert!(
        caps.stop_strings || gen_cfg.stop_strings.is_empty(),
        "session does not declare stop_strings support but gen_cfg.stop_strings is set"
    );
    debug_assert!(
        caps.reasoning_budget || gen_cfg.reasoning_budget.is_none(),
        "session does not declare reasoning_budget support but gen_cfg.reasoning_budget is set"
    );

    let cancel_never = || false;
    session.prefill(&cancel_never)?;

    let mut trace = DriverTrace::default();
    let mut all_ids: Vec<u32> = prompt_ids.to_vec();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut token_logprobs: Vec<TokenLogprob> = Vec::new();
    // Never read: `record_logprob` no-ops whenever `self.logprobs` is `None`,
    // which is asserted above for every call this driver makes.
    let no_logits: Vec<f32> = Vec::new();
    let is_eos = |id: u32| id == eos_token_id || gen_cfg.stop_token_ids.contains(&id);

    // --- Step 0: the prefill-derived first token. ---
    let request0 = SelectionRequest {
        config: gen_cfg,
        history: &all_ids,
        grammar: None,
    };
    let candidate0 = session.select(&request0)?;
    trace.opened += 1;

    if is_eos(candidate0.candidate_id) {
        session.finish(FinishDisposition::Reusable)?;
        return Ok(DriverResult {
            generated_ids: Vec::new(),
            token_logprobs: Vec::new(),
            stopped: true,
            stop_reason: StopReason::Eos,
            confirmed_stop_string_match: false,
            trace,
        });
    }

    generated_ids.push(candidate0.candidate_id);
    all_ids.push(candidate0.candidate_id);

    let mut policy = DecodePolicy::init(
        gen_cfg,
        think_close_id,
        &mut token_logprobs,
        candidate0.candidate_id,
        &no_logits,
        gen_cfg.temperature,
        generated_ids.len(),
        streaming,
    );

    // `pending` is the one prediction that has been opened (via `select`) but
    // not yet consumed (via `decode`). The loop below consumes the PREVIOUS
    // iteration's candidate before opening the current one, so exactly one
    // prediction is always left open when the loop ends -- see the comment
    // above the `finish` call for why that one is dropped rather than decoded.
    let mut pending = candidate0.prediction;

    let first_delta = decode_delta(candidate0.candidate_id);
    match policy.check_initial_stop(
        &mut token_logprobs,
        text,
        token_logprob_end_offsets,
        &first_delta,
        |s| emit_confirmed(s, candidate0.candidate_id),
    ) {
        StopCheckOutcome::Stopped => {
            session.finish(FinishDisposition::Reusable)?;
            return Ok(DriverResult {
                generated_ids,
                token_logprobs,
                stopped: true,
                stop_reason: StopReason::Eos,
                confirmed_stop_string_match: true,
                trace,
            });
        }
        StopCheckOutcome::Interrupted => {
            session.finish(FinishDisposition::Reusable)?;
            return Ok(DriverResult {
                generated_ids,
                token_logprobs,
                stopped: false,
                stop_reason: StopReason::Interrupt,
                confirmed_stop_string_match: false,
                trace,
            });
        }
        StopCheckOutcome::Continue => {}
    }

    let cap = policy.cap();
    let mut stopped = false;
    let mut stop_reason = StopReason::Length;
    let mut confirmed_stop_string_match = false;

    for _ in 1..cap {
        let accepted = AcceptedToken {
            final_id: *all_ids
                .last()
                .expect("all_ids holds the prompt plus at least the step-0 token"),
            prediction: pending,
        };
        session.decode(&accepted, &cancel_never)?;
        trace.consumed += 1;

        let request = SelectionRequest {
            config: gen_cfg,
            history: &all_ids,
            grammar: None,
        };
        let candidate = session.select(&request)?;
        trace.opened += 1;

        let generated_len_before = generated_ids.len();
        let outcome = policy.transition(
            &mut token_logprobs,
            candidate.candidate_id,
            &no_logits,
            gen_cfg.temperature,
            generated_len_before,
            |_next_id| true, // no grammar on this path; see the module doc comment
            &is_eos,
            |next_id| {
                generated_ids.push(next_id);
                all_ids.push(next_id);
            },
            &mut decode_delta,
            text,
            token_logprob_end_offsets,
            |s, id| emit_confirmed(s, id),
        );

        match outcome {
            StepOutcome::GrammarStop => {
                // Unreachable on this path (`grammar_advance` above always
                // returns `true`), handled for exhaustiveness.
                stopped = true;
                stop_reason = StopReason::Grammar;
                break;
            }
            StepOutcome::Eos => {
                stopped = true;
                stop_reason = StopReason::Eos;
                break;
            }
            StepOutcome::Stopped => {
                stopped = true;
                confirmed_stop_string_match = true;
                stop_reason = StopReason::Eos;
                break;
            }
            StepOutcome::Interrupted => {
                // Unreachable on this path (`generate()`'s two non-streaming
                // callers always pass an `emit_confirmed` that returns
                // `true`), handled for exhaustiveness/defense-in-depth --
                // mirrors the same note on `decode_loop_with_stops`.
                stop_reason = StopReason::Interrupt;
                break;
            }
            StepOutcome::Emitted {
                answer_budget_exhausted,
                ..
            } => {
                pending = candidate.prediction;
                if answer_budget_exhausted {
                    break;
                }
            }
        }
    }

    // The last opened prediction is dropped, not decoded, and that is the whole
    // reason this driver costs the same number of forward passes as the loops it
    // replaces. Every iteration above consumes the PREVIOUS candidate before
    // opening the current one, so when the loop ends, the final emitted token
    // still has an open prediction and no following iteration to consume it.
    // `decode()` on a session is a real forward pass; issuing one here to square
    // `consumed` with `generated_ids.len()` would add one full transformer step
    // to every request, whose logits nothing would ever read -- a measurable cost
    // paid to make a counter look tidy, and one that would land inside the very
    // measurement the next row exists to take. `finish` invalidates the open
    // prediction instead (`PredictionLedger::invalidate`), which is exactly what
    // the ledger's "ends at ... finish" contract already says happens.
    //
    // So the driver's standing invariant is `consumed == opened - 1`, and on a
    // natural finish `opened == generated_ids.len()`. A step routed around the
    // driver leaves BOTH short, which is what makes the trace a bypass detector.
    debug_assert_eq!(
        trace.consumed + 1,
        trace.opened,
        "exactly one prediction is open when the loop ends"
    );
    session.finish(FinishDisposition::Reusable)?;

    Ok(DriverResult {
        generated_ids,
        token_logprobs,
        stopped,
        stop_reason,
        confirmed_stop_string_match,
        trace,
    })
}
