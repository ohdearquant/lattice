//! The autoregressive driver (ADR-090 D1/D2, row C; grammar/logprobs routing
//! added by row R03): one loop over `&mut dyn DecoderSession` that drives the
//! existing [`DecodePolicy`] unchanged -- every canonical Qwen3.5 CPU
//! generate/stream request now runs through this driver, no exceptions.
//!
//! **Grammar.** Masking happens inside the session's own `select` (over logits
//! this module still cannot see -- D1 forbids downcasting to reach them), against
//! a `GrammarState` the session owns across its lifetime (constructed once,
//! mutated by every `select`/`advance_grammar` call -- see `QwenCpuSession`'s
//! module doc comment). Advance runs inside `DecodePolicy::transition_with_metadata`'s
//! fixed internal order, before the EOS check, via the `grammar_advance` callback
//! this driver wires to [`DecoderSession::advance_grammar`]. Step 0 (the
//! prefill-derived first token) has no `transition` call of its own to route
//! this through -- `DecodePolicy::init`/`init_with_metadata` build the policy but
//! run no per-step control sequence -- so this driver makes the identical
//! `advance_grammar` call directly, in the same position `generate_inline`'s
//! manual code made it (mask -> sample -> **advance** -> EOS check -> push).
//!
//! **Logprobs.** `DecodePolicy::init_with_metadata` / `transition_with_metadata`
//! (row R03 siblings of `init`/`transition`, sharing the same private engine --
//! see `crate::generation`) take a `record_metadata` callback instead of raw
//! `logits: &[f32]`, called at the identical point in the fixed order
//! (`init`'s / `transition`'s own doc comments) and ONLY when
//! `gen_cfg.logprobs` is `Some`. This driver's callback routes to
//! [`DecoderSession::metadata`], scored against the SAME prediction the
//! actually-emitted token's candidate came from -- D1's "Metadata identity"
//! role (the final token scored against the prediction's pre-advance view).
//!
//! **The `ended_without_reopening` trace exception.** The driver's standing
//! invariant is `consumed == opened - 1` (see the comment above the final
//! `debug_assert_eq!` below for the full derivation). Mid-loop grammar
//! exhaustion (`SelectOutcome::GrammarExhausted` returned from a `select` that
//! runs *after* that iteration's `decode()`) is the one termination mode that
//! does not fit it: `decode()` already ran (`consumed` incremented), but the
//! failed `select()` opened no new prediction, leaving `opened == consumed`
//! instead of `consumed + 1`. `ended_without_reopening` flags exactly this one
//! path so the final assertion can state both invariants instead of silently
//! weakening the general one.
use super::{
    AcceptedToken, Cancellation, DecoderSession, FinishDisposition, MetadataRequest, SelectOutcome,
    SelectionRequest,
};
use crate::error::InferenceError;
use crate::generation::{
    DecodePolicy, GenerateConfig, StepOutcome, StopCheckOutcome, TokenLogprob,
};
use crate::stop_reason::StopReason;
use std::cell::RefCell;

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
    cancel: &dyn Cancellation,
    mut decode_delta: impl FnMut(u32) -> String,
    text: &mut String,
    token_logprob_end_offsets: &mut Vec<usize>,
    mut emit_confirmed: impl FnMut(&str, u32) -> bool,
    mut on_prefill_end: impl FnMut(),
    finish_tail: impl FnOnce() -> String,
) -> Result<DriverResult, InferenceError> {
    // D3: capabilities are negotiated per session, and "one driver over many sessions" (D1)
    // means a session that does not declare a control this call actually uses is a caller
    // bug, not this driver's problem to route around silently.
    let caps = *session.capabilities();
    debug_assert!(
        caps.grammar || gen_cfg.grammar.is_none(),
        "session does not declare grammar support but gen_cfg.grammar is set"
    );
    debug_assert!(
        caps.logprobs || gen_cfg.logprobs.is_none(),
        "session does not declare logprobs support but gen_cfg.logprobs is set"
    );
    debug_assert!(
        caps.stop_strings || gen_cfg.stop_strings.is_empty(),
        "session does not declare stop_strings support but gen_cfg.stop_strings is set"
    );
    debug_assert!(
        caps.reasoning_budget || gen_cfg.reasoning_budget.is_none(),
        "session does not declare reasoning_budget support but gen_cfg.reasoning_budget is set"
    );

    // `RefCell<&mut dyn DecoderSession>`: `grammar_advance` and `record_metadata` below are
    // two independent closures that both need mutable session access, passed as sibling
    // arguments to the same `transition_with_metadata` call -- two simultaneous `&mut
    // session` captures the borrow checker rejects outright, even though the closures are
    // only ever called sequentially, never concurrently. Same pattern this crate already
    // uses for the identical shape (`FnMutCancellation`, `on_raw_event_cell`, `detok_cell` in
    // `model::qwen35::generation`'s `generate_streaming_via_driver`).
    let session = RefCell::new(session);

    // Cancellation checkpoint 1/3 (mirrors the pre-migration streaming entry's own
    // first checkpoint): before the prefill pass starts, so a client that already
    // disconnected never pays for it. `session.prefill`/`session.decode` below are
    // still called with an always-false `Cancellation` of their own -- this driver
    // owns the three streaming-contract checkpoints itself, at the exact points the
    // pre-migration loop checked them, rather than delegating to the session's
    // per-call check (which fires at a different point: immediately before its own
    // forward pass, not before/after the surrounding driver-level bookkeeping).
    if cancel.is_cancelled() {
        return Ok(DriverResult {
            generated_ids: Vec::new(),
            token_logprobs: Vec::new(),
            stopped: false,
            stop_reason: StopReason::Interrupt,
            confirmed_stop_string_match: false,
            trace: DriverTrace::default(),
        });
    }
    let cancel_never = || false;
    session.borrow_mut().prefill(&cancel_never)?;
    on_prefill_end();

    // Checkpoint 2/3: immediately after the prefill pass returns -- fired after
    // `on_prefill_end` (mirroring the pre-migration entry firing `PrefillEnd` before
    // this check, so a caller observing raw events still sees prefill end even on
    // this early-return path) and before paying for the first `select`.
    if cancel.is_cancelled() {
        return Ok(DriverResult {
            generated_ids: Vec::new(),
            token_logprobs: Vec::new(),
            stopped: false,
            stop_reason: StopReason::Interrupt,
            confirmed_stop_string_match: false,
            trace: DriverTrace::default(),
        });
    }

    let mut trace = DriverTrace::default();
    let mut all_ids: Vec<u32> = prompt_ids.to_vec();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut token_logprobs: Vec<TokenLogprob> = Vec::new();
    let is_eos = |id: u32| id == eos_token_id || gen_cfg.stop_token_ids.contains(&id);

    // --- Step 0: the prefill-derived first token. ---
    let request0 = SelectionRequest {
        config: gen_cfg,
        history: &all_ids,
    };
    let outcome0 = session.borrow_mut().select(&request0)?;
    let candidate0 = match outcome0 {
        // Blocked before any sample, and the grammar state is already a valid accept
        // state -- generate_inline's identical step-0 branch: no token is ever
        // sampled, so this is `stopped: true` (a completed grammar, not a rejected
        // one), unlike the advance-rejection case below. No prediction was opened
        // (`select` returned before calling `PredictionLedger::open`), so `trace`
        // stays at its untouched default -- there is nothing for the final
        // `debug_assert_eq!` to reconcile because this return skips it entirely,
        // exactly like the EOS-at-step-0 return below.
        SelectOutcome::GrammarExhausted => {
            session.borrow_mut().finish(FinishDisposition::Reusable)?;
            return Ok(DriverResult {
                generated_ids: Vec::new(),
                token_logprobs: Vec::new(),
                stopped: true,
                stop_reason: StopReason::Grammar,
                confirmed_stop_string_match: false,
                trace,
            });
        }
        SelectOutcome::Candidate(c) => c,
    };
    trace.opened += 1;

    // Grammar advance on the sampled candidate: generate_inline's manual
    // mask -> sample -> **advance** -> [reject: stopped=false] ->
    // is_complete_without_continuation -> EOS check -> push sequence. Step 0 has no
    // `transition` call to route this through (`DecodePolicy::init`/`init_with_metadata`
    // build the policy but run no per-step control sequence), so this driver makes the
    // identical call directly, in the same position. `grammar_output`'s `stop_reason` is
    // unconditionally `Grammar` regardless of its `stopped` argument (see
    // `model::qwen35::generation::grammar_output`), which this mirrors: a rejected
    // candidate at step 0 is `stopped: false` (no completed grammar, nothing to answer
    // with) -- distinct from the exhaustion-before-sampling case above.
    if !session
        .borrow_mut()
        .advance_grammar(candidate0.candidate_id)
    {
        session.borrow_mut().finish(FinishDisposition::Reusable)?;
        return Ok(DriverResult {
            generated_ids: Vec::new(),
            token_logprobs: Vec::new(),
            stopped: false,
            stop_reason: StopReason::Grammar,
            confirmed_stop_string_match: false,
            trace,
        });
    }
    let grammar_complete_at_step0 = session.borrow().grammar_complete_without_continuation();

    if is_eos(candidate0.candidate_id) {
        session.borrow_mut().finish(FinishDisposition::Reusable)?;
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

    let candidate0_prediction = candidate0.prediction;
    let mut policy = DecodePolicy::init_with_metadata(
        gen_cfg,
        think_close_id,
        &mut token_logprobs,
        candidate0.candidate_id,
        generated_ids.len(),
        streaming,
        |final_token, top_n| {
            session
                .borrow_mut()
                .metadata(
                    candidate0_prediction,
                    final_token,
                    &MetadataRequest {
                        top_logprobs: Some(top_n),
                    },
                )
                .map(|m| (m.final_logprob, m.top))
        },
    )?;

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
            session.borrow_mut().finish(FinishDisposition::Reusable)?;
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
            session.borrow_mut().finish(FinishDisposition::Reusable)?;
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
    // Set when step 0's advance succeeded AND immediately completed the grammar with
    // no further continuation (generate_inline's step-0 `grammar_complete` branch,
    // checked after `check_initial_stop` so a stop-string match still takes
    // precedence, matching the legacy ordering exactly). The loop is skipped
    // entirely and execution falls through to the natural-end tail-flush code below,
    // exactly as `generate_inline`'s manual `if grammar_complete { return ... }`
    // does -- except here the return is deferred to the bottom of this function so
    // the SAME tail-flush/finish sequence every other natural end uses applies here
    // too, rather than a third hand-written copy of it.
    let mut stopped = grammar_complete_at_step0;
    let mut stop_reason = if grammar_complete_at_step0 {
        StopReason::Grammar
    } else {
        StopReason::Length
    };
    let mut confirmed_stop_string_match = false;
    // Flags the one termination mode whose trace relationship is `opened == consumed`
    // rather than the standing `opened == consumed + 1` -- see the module doc comment.
    let mut ended_without_reopening = false;

    if !grammar_complete_at_step0 {
        for _ in 1..cap {
            // Checkpoint 3/3: top of every decode iteration, before this step's
            // forward pass -- mirrors the pre-migration streaming entry's per-iteration
            // checkpoint exactly (checked before `forward_step`, every iteration).
            if cancel.is_cancelled() {
                stop_reason = StopReason::Interrupt;
                break;
            }
            let accepted = AcceptedToken {
                final_id: *all_ids
                    .last()
                    .expect("all_ids holds the prompt plus at least the step-0 token"),
                prediction: pending,
            };
            session.borrow_mut().decode(&accepted, &cancel_never)?;
            trace.consumed += 1;

            let request = SelectionRequest {
                config: gen_cfg,
                history: &all_ids,
            };
            let outcome = session.borrow_mut().select(&request)?;
            let candidate = match outcome {
                // Mid-loop mirror of `decode_loop`/`decode_loop_with_stops`'s
                // mask-blocked-and-complete branch: no new prediction was opened this
                // iteration, so `trace.opened` stays where it was -- exactly matching
                // `trace.consumed` (this iteration's `decode()` did run), not the
                // standing `consumed + 1` invariant. `ended_without_reopening` flags
                // this for the final assertion below.
                SelectOutcome::GrammarExhausted => {
                    stopped = true;
                    stop_reason = StopReason::Grammar;
                    ended_without_reopening = true;
                    break;
                }
                SelectOutcome::Candidate(c) => c,
            };
            trace.opened += 1;

            let generated_len_before = generated_ids.len();
            let candidate_prediction = candidate.prediction;
            let outcome = policy.transition_with_metadata(
                &mut token_logprobs,
                candidate.candidate_id,
                generated_len_before,
                |next_id| session.borrow_mut().advance_grammar(next_id),
                &is_eos,
                |next_id| {
                    generated_ids.push(next_id);
                    all_ids.push(next_id);
                },
                |final_token, top_n| {
                    session
                        .borrow_mut()
                        .metadata(
                            candidate_prediction,
                            final_token,
                            &MetadataRequest {
                                top_logprobs: Some(top_n),
                            },
                        )
                        .map(|m| (m.final_logprob, m.top))
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
                    // Unreachable for `generate()`'s two non-streaming callers (their
                    // `emit_confirmed` always returns `true`); reachable for the
                    // streaming caller, whose `emit_confirmed` forwards `on_token`'s
                    // return value -- a caller that can no longer consume the stream
                    // (e.g. a dropped SSE receiver) stops generation here, same as the
                    // pre-migration streaming loop.
                    stop_reason = StopReason::Interrupt;
                    break;
                }
                StepOutcome::Emitted {
                    answer_budget_exhausted,
                    ..
                } => {
                    pending = candidate.prediction;
                    // Mirrors `decode_loop`/`decode_loop_with_stops`'s post-`Emitted`
                    // `grammar_complete_without_continuation` check, run BEFORE the
                    // answer-budget check -- proactively catching a grammar that just
                    // reached an accepting state with no legal continuation, rather
                    // than waiting for the next iteration's `select` to discover the
                    // same thing via `SelectOutcome::GrammarExhausted`. This path opened
                    // a real prediction this iteration (`trace.opened` above), so the
                    // standing `opened == consumed + 1` invariant holds here --
                    // `ended_without_reopening` is not set.
                    if session.borrow().grammar_complete_without_continuation() {
                        stopped = true;
                        stop_reason = StopReason::Grammar;
                        break;
                    }
                    if answer_budget_exhausted {
                        break;
                    }
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
    // `ended_without_reopening` (module doc comment) is the one termination mode
    // that legitimately breaks this: a mid-loop `SelectOutcome::GrammarExhausted`
    // runs `decode()` (incrementing `consumed`) but opens no new prediction, so
    // `opened == consumed` there instead of `consumed + 1`.
    if ended_without_reopening {
        debug_assert_eq!(
            trace.consumed, trace.opened,
            "mid-loop grammar exhaustion opens no new prediction the iteration it fires"
        );
    } else {
        debug_assert_eq!(
            trace.consumed + 1,
            trace.opened,
            "exactly one prediction is open when the loop ends"
        );
    }

    // Final flush of the natural end, run through the SAME `StopMode` matcher the
    // loop above used -- mirrors the pre-migration streaming loop's own
    // `policy.finish_stop` tail flush exactly, including on an EMPTY tail. Two
    // different buffers are released here and only one of them is `tail`:
    // `finish_tail` returns what the caller's incremental detokenizer held back for
    // UTF-8-boundary reasons, while `StopStringMatcher::push` separately holds back
    // `max_stop - 1` bytes on EVERY call so a stop string spanning a delta boundary
    // is never streamed early. `finish_stop` is what releases that second buffer, so
    // it must run even when `tail` is empty -- the ordinary case, since most
    // generations end on a clean UTF-8 boundary. Skipped when the loop already ended
    // via a confirmed stop-string match (nothing left to reconcile) or via a
    // caller/cancel interruption (the caller has stopped consuming; mirrors that
    // loop's `stopped_by_caller` gate, which it also sets on cancel).
    // Non-streaming callers pass a `finish_tail` returning an empty string AND run
    // in `StopMode::FullScan`, where `finish_stop` does nothing -- they do their own
    // post-return tail handling. Also covers the step-0 grammar-complete case: its
    // `decode_delta`/`check_initial_stop` call above already pushed the first
    // token's decoded text into `text` (or into the throwaway buffer, for the
    // fast/no-stop-strings path -- see `generate_via_driver`), so this flush behaves
    // identically to a one-token natural end reached via the loop.
    if stop_reason != StopReason::Interrupt && !confirmed_stop_string_match {
        let tail = finish_tail();
        // The id here is never read: every `emit_confirmed` this driver's callers
        // supply ignores it for the tail case (it is not tied to one sampled token),
        // same as the pre-migration `finish_stop` call site, whose `on_token` sink
        // takes no id at all.
        let tail_stopped = policy.finish_stop(text, &tail, |s| emit_confirmed(s, 0));
        if tail_stopped && !stopped {
            stopped = true;
            stop_reason = StopReason::Eos;
        }
    }

    session.borrow_mut().finish(FinishDisposition::Reusable)?;

    Ok(DriverResult {
        generated_ids,
        token_logprobs,
        stopped,
        stop_reason,
        confirmed_stop_string_match,
        trace,
    })
}
