//! The autoregressive driver (ADR-090 D1/D2, row C; grammar/logprobs routing
//! added by row R03, grammar ownership relocated here in this rework): one loop
//! over `&mut dyn DecoderSession` that drives the existing [`DecodePolicy`]
//! unchanged -- every canonical Qwen3.5 CPU generate/stream request now runs
//! through this driver, no exceptions.
//!
//! **Grammar.** This driver, not the session, owns the [`GrammarEngine`] and
//! [`GrammarState`] (built from `gen_cfg.grammar`; `None` when no grammar is
//! set): ADR-090 D1 names grammar transitions as the driver's responsibility,
//! and a per-session copy would be exactly the duplication this refactor exists
//! to remove -- every future session (Gemma CPU, Metal Qwen, ...) would
//! otherwise have to reimplement it. State needs interior mutability
//! (`RefCell`) because masking (`GrammarEngine::mask_logits`) takes `&mut
//! GrammarState` while [`DecoderSession::select`] only ever sees `&
//! SelectionRequest` -- see [`SelectionRequest::grammar_mask`]'s own doc
//! comment. Each step, this driver builds (once, before the loop) a closure
//! over that engine/state and hands the session a borrow of it through
//! `grammar_mask`; the session applies the mask to its own logits and reports
//! [`SelectOutcome::GrammarExhausted`] when every token is blocked, without
//! itself knowing whether that is a completed grammar or a real error --
//! resolving that ambiguity is this driver's job (`grammar_complete` below),
//! since only the driver still holds the engine and state. `advance` runs
//! inside `DecodePolicy::transition_with_metadata`'s fixed internal order,
//! before the EOS check, via the `grammar_advance` closure this driver builds
//! over its own owned state (no longer a session method). Step 0 (the
//! prefill-derived first token) has no `transition` call of its own to route
//! this through -- `DecodePolicy::init`/`init_with_metadata` build the policy but
//! run no per-step control sequence -- so this driver makes the identical
//! `advance` call directly, in the same position `generate_inline`'s manual
//! code made it (mask -> sample -> **advance** -> EOS check -> push).
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
//! Unchanged by this rework: logprobs stay session-scored, driver-called.
//!
//! **Capabilities are a hard error, not a debug assertion.** D3: capabilities
//! are negotiated per session, and "one driver over many sessions" (D1) means
//! a session that does not declare a control this call actually uses is a
//! caller bug this driver refuses outright (`InferenceError::InvalidInput`),
//! in every build -- a `debug_assert!` here would silently no-op in release
//! and let a session ignore a control it never claimed to support. See
//! `check_capabilities` below and its test module for the mutation-sensitive
//! proof (a fake session declaring `grammar: false` is refused; one declaring
//! `grammar: true` for the same request is not).
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
    AcceptedToken, Cancellation, DecoderSession, ExecutionCapabilities, FinishDisposition,
    GrammarMaskFn, MetadataRequest, SelectOutcome, SelectionRequest,
};
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use super::{SpeculativeSession, SpeculativeTrace};
use crate::error::InferenceError;
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use crate::generation::TopLogprob;
use crate::generation::{
    DecodePolicy, GenerateConfig, StepOutcome, StopCheckOutcome, TokenLogprob,
};
use crate::grammar::{GrammarEngine, pda::GrammarState};
use crate::model::qwen35_config::decode_cap;
use crate::stop_reason::StopReason;
#[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
use std::cell::Cell;
use std::cell::RefCell;

#[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
thread_local! {
    static TEST_DRIVER_RUN_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
pub(crate) fn test_driver_run_count() -> usize {
    TEST_DRIVER_RUN_COUNT.with(Cell::get)
}

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

/// Everything [`run`] produces: `model::qwen35::generation`'s
/// `generate_via_driver()`/`generate_streaming_via_driver()` read every field here
/// directly off the returned value, replacing the by-hand assembly the deleted
/// pre-driver `decode_loop`/`decode_loop_with_stops` functions used to require
/// from their own local `generated_ids`/`token_logprobs`.
pub(crate) struct DriverResult {
    pub(crate) generated_ids: Vec<u32>,
    pub(crate) token_logprobs: Vec<TokenLogprob>,
    pub(crate) stopped: bool,
    pub(crate) stop_reason: StopReason,
    /// Set only on [`StepOutcome::Stopped`] (a confirmed stop-string match);
    /// `generate_via_driver`'s stop-strings branch reads this exactly as the
    /// deleted `decode_loop_with_stops` read its own local of the same name,
    /// to skip a redundant tail-flush attempt.
    pub(crate) confirmed_stop_string_match: bool,
    pub(crate) trace: DriverTrace,
}

/// Refuses (`InferenceError::InvalidInput`) when `gen_cfg` requests a control `caps` does not
/// declare -- see this module's doc comment ("Capabilities are a hard error, not a debug
/// assertion"). Checked once, at the top of [`run`], before any session call.
fn check_capabilities(
    caps: ExecutionCapabilities,
    gen_cfg: &GenerateConfig,
) -> Result<(), InferenceError> {
    if !caps.grammar && gen_cfg.grammar.is_some() {
        return Err(InferenceError::InvalidInput(
            "session does not declare grammar support but gen_cfg.grammar is set".into(),
        ));
    }
    if !caps.logprobs && gen_cfg.logprobs.is_some() {
        return Err(InferenceError::InvalidInput(
            "session does not declare logprobs support but gen_cfg.logprobs is set".into(),
        ));
    }
    if !caps.stop_strings && !gen_cfg.stop_strings.is_empty() {
        return Err(InferenceError::InvalidInput(
            "session does not declare stop_strings support but gen_cfg.stop_strings is set".into(),
        ));
    }
    if !caps.reasoning_budget && gen_cfg.reasoning_budget.is_some() {
        return Err(InferenceError::InvalidInput(
            "session does not declare reasoning_budget support but gen_cfg.reasoning_budget is set"
                .into(),
        ));
    }
    Ok(())
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
/// `model::qwen35::generation::Qwen35Model::generate_via_driver`'s fast-path
/// branch already relies on for its own throwaway values.
///
/// `on_push` is called exactly once per token that becomes part of
/// `generated_ids` -- both the prefill-derived step-0 token and every
/// decode-loop token alike -- with `generated_ids.len()` immediately after
/// the push that grew it to that length, and before anything else runs for
/// that step: at step 0 that means before `DecodePolicy::init_with_metadata`
/// scores the token's logprob, and in the loop it means before
/// `DecodePolicy::transition_with_metadata`'s own `record_metadata` /
/// `capture_reasoning_end` / `decode_delta` sequence. This is the hook a
/// caller observing the true push boundary (e.g. a raw per-token
/// lifecycle event) must use instead of piggy-backing on `decode_delta`:
/// `decode_delta` runs later in the fixed per-step order, so anything
/// timed off it drifts whenever `record_metadata` does real (session-scored)
/// work, i.e. whenever `gen_cfg.logprobs` is set.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    session: &mut dyn DecoderSession,
    gen_cfg: &GenerateConfig,
    think_close_id: Option<u32>,
    prompt_ids: &[u32],
    eos_token_id: u32,
    streaming: bool,
    cancel: &dyn Cancellation,
    on_push: impl FnMut(usize),
    decode_delta: impl FnMut(u32) -> String,
    text: &mut String,
    token_logprob_end_offsets: &mut Vec<usize>,
    emit_confirmed: impl FnMut(&str, u32) -> bool,
    on_prefill_end: impl FnMut(),
    finish_tail: impl FnOnce() -> String,
) -> Result<DriverResult, InferenceError> {
    #[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
    TEST_DRIVER_RUN_COUNT.with(|count| count.set(count.get() + 1));

    check_capabilities(*session.capabilities(), gen_cfg)?;
    let mut prefill_started = false;
    let result = run_inner(
        session,
        gen_cfg,
        think_close_id,
        prompt_ids,
        eos_token_id,
        streaming,
        cancel,
        on_push,
        decode_delta,
        text,
        token_logprob_end_offsets,
        emit_confirmed,
        on_prefill_end,
        finish_tail,
        &mut prefill_started,
    );
    let interrupted = matches!(&result, Ok(result) if result.stop_reason == StopReason::Interrupt);
    let disposition = if prefill_started && (result.is_err() || interrupted) {
        FinishDisposition::Poisoned
    } else {
        FinishDisposition::Reusable
    };
    let finished = session.finish(disposition);
    result.and_then(|result| finished.map(|()| result))
}

fn run_inner(
    session: &mut dyn DecoderSession,
    gen_cfg: &GenerateConfig,
    think_close_id: Option<u32>,
    prompt_ids: &[u32],
    eos_token_id: u32,
    streaming: bool,
    cancel: &dyn Cancellation,
    mut on_push: impl FnMut(usize),
    mut decode_delta: impl FnMut(u32) -> String,
    text: &mut String,
    token_logprob_end_offsets: &mut Vec<usize>,
    mut emit_confirmed: impl FnMut(&str, u32) -> bool,
    mut on_prefill_end: impl FnMut(),
    finish_tail: impl FnOnce() -> String,
    prefill_started: &mut bool,
) -> Result<DriverResult, InferenceError> {
    // Driver-owned grammar engine + state (ADR-090 D1; moved off the session by this row's
    // rework -- see this module's doc comment). `grammar_state` needs interior mutability:
    // `GrammarEngine::mask_logits`/`advance` take `&mut GrammarState`, but the mask closure
    // below is reached through `&SelectionRequest` (shared borrow) and `grammar_advance` is
    // called from inside `DecodePolicy::transition_with_metadata` without a `&mut` path back
    // to this local. `None` on either side of these three bindings when `gen_cfg.grammar` is
    // `None` -- every one of them then behaves as documented "no grammar" default.
    let grammar_engine: Option<&GrammarEngine> = gen_cfg.grammar.as_deref();
    let grammar_state: Option<RefCell<GrammarState>> =
        grammar_engine.map(|engine| RefCell::new(engine.initial_state()));

    // Built once, borrowed by every `SelectionRequest` below (`Option<&dyn Fn(..)>` stays
    // `Copy`, so `SelectionRequest` does not have to give that up -- see its own doc
    // comment). Boxed because a `match`'s two arms are two different anonymous closure
    // types; the trait object is the point where they unify.
    let grammar_mask: Option<Box<GrammarMaskFn<'_>>> = match (grammar_engine, &grammar_state) {
        (Some(engine), Some(state)) => Some(Box::new(move |logits: &mut [f32]| {
            engine.mask_logits(&mut state.borrow_mut(), logits)?;
            Ok(())
        })),
        _ => None,
    };

    // Advances the driver-owned grammar state on the actually-emitted token, replacing the
    // removed `DecoderSession::advance_grammar`. `Ok(true)` (accept, nothing to advance) when no
    // grammar is set -- the same default that trait method used to return. A matcher stack
    // limit comes back as `Err`, never as a rejection.
    let grammar_advance = |next_id: u32| -> Result<bool, InferenceError> {
        match (grammar_engine, &grammar_state) {
            (Some(engine), Some(state)) => Ok(engine.advance(&mut state.borrow_mut(), next_id)?),
            _ => Ok(true),
        }
    };

    // Whether the grammar reached an accepting state with no further legal continuation, as
    // of the most recent `grammar_advance` call. Replaces the removed
    // `DecoderSession::grammar_complete_without_continuation`; `false` (never complete) when
    // no grammar is set. Called both to disambiguate `SelectOutcome::GrammarExhausted` (see
    // `SelectOutcome`'s doc comment) and, post-`Emitted`, as the same proactive check the
    // pre-driver loops made.
    let grammar_complete = || -> bool {
        match (grammar_engine, &grammar_state) {
            (Some(engine), Some(state)) => engine.is_complete_without_continuation(&state.borrow()),
            _ => false,
        }
    };

    // `RefCell<&mut dyn DecoderSession>`: `record_metadata` below and the surrounding
    // `select`/`decode` calls all need mutable session access from different
    // closures and call sites within this same function body -- two simultaneous `&mut
    // session` captures the borrow checker rejects outright, even though they are only ever
    // called sequentially, never concurrently. (`grammar_advance` above needs no session
    // access at all now -- the driver owns grammar state directly -- but the other closures
    // still do.) Same pattern this crate already uses for the identical shape
    // (`FnMutCancellation`, `on_raw_event_cell`, `detok_cell` in
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
    *prefill_started = true;
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
    // Same reservation the earlier per-path decode loops made: the most tokens
    // this request can ever emit is fixed up front by `gen_cfg`, independent of
    // anything the session or policy decides later, so reserve it before the
    // first push.
    let mut generated_ids: Vec<u32> = Vec::with_capacity(decode_cap(
        gen_cfg.effective_reasoning_budget(),
        gen_cfg.max_new_tokens,
    ));
    let mut token_logprobs: Vec<TokenLogprob> = Vec::new();
    let is_eos = |id: u32| id == eos_token_id || gen_cfg.stop_token_ids.contains(&id);

    // --- Step 0: the prefill-derived first token. ---
    let request0 = SelectionRequest {
        config: gen_cfg,
        history: &all_ids,
        grammar_mask: grammar_mask.as_deref(),
    };
    let outcome0 = session.borrow_mut().select(&request0)?;
    let candidate0 = match outcome0 {
        // `select` reported every token blocked but cannot itself tell a completed
        // grammar from a real dead end (`SelectOutcome`'s doc comment) -- this driver
        // still holds the engine/state, so it makes that call here. Blocked-and-complete
        // is generate_inline's identical step-0 branch: no token is ever sampled, so this
        // is `stopped: true` (a completed grammar, not a rejected one), unlike the
        // advance-rejection case below. No prediction was opened (`select` returned
        // before calling `PredictionLedger::open`), so `trace` stays at its untouched
        // default -- there is nothing for the final `debug_assert_eq!` to reconcile
        // because this return skips it entirely, exactly like the EOS-at-step-0 return
        // below. Blocked-and-NOT-complete mirrors the pre-driver loops' identical hard
        // error, just raised here instead of inside `select`.
        SelectOutcome::GrammarExhausted => {
            if !grammar_complete() {
                return Err(InferenceError::GrammarConstraintBlocked(
                    "grammar constraint blocked every token; \
                     no legal continuation exists in the current grammar state"
                        .into(),
                ));
            }
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
    // identical call directly, in the same position -- now against its own owned state
    // rather than through the removed `DecoderSession::advance_grammar`.
    // `grammar_output`'s `stop_reason` is unconditionally `Grammar` regardless of its
    // `stopped` argument (see `model::qwen35::generation::grammar_output`), which this
    // mirrors: a rejected candidate at step 0 is `stopped: false` (no completed grammar,
    // nothing to answer with) -- distinct from the exhaustion-before-sampling case above.
    if !grammar_advance(candidate0.candidate_id)? {
        return Ok(DriverResult {
            generated_ids: Vec::new(),
            token_logprobs: Vec::new(),
            stopped: false,
            stop_reason: StopReason::Grammar,
            confirmed_stop_string_match: false,
            trace,
        });
    }
    let grammar_complete_at_step0 = grammar_complete();

    if is_eos(candidate0.candidate_id) {
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
    on_push(generated_ids.len());

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
                grammar_mask: grammar_mask.as_deref(),
            };
            let outcome = session.borrow_mut().select(&request)?;
            let candidate = match outcome {
                // Mid-loop mirror of `decode_loop`/`decode_loop_with_stops`'s
                // mask-blocked-and-complete branch. As at step 0, `select` cannot itself
                // tell a completed grammar from a real dead end, so this driver resolves
                // it via its own owned state before deciding how the loop ends. The
                // completed case: no new prediction was opened this iteration, so
                // `trace.opened` stays where it was -- exactly matching `trace.consumed`
                // (this iteration's `decode()` did run), not the standing `consumed + 1`
                // invariant. `ended_without_reopening` flags this for the final assertion
                // below. The not-complete case raises the same hard error `select` used
                // to raise directly.
                SelectOutcome::GrammarExhausted => {
                    if !grammar_complete() {
                        return Err(InferenceError::GrammarConstraintBlocked(
                            "grammar constraint blocked every token; \
                             no legal continuation exists in the current grammar state"
                                .into(),
                        ));
                    }
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
                grammar_advance,
                &is_eos,
                |next_id| {
                    generated_ids.push(next_id);
                    all_ids.push(next_id);
                    on_push(generated_ids.len());
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
                    // grammar-complete check, run BEFORE the answer-budget check --
                    // proactively catching a grammar that just reached an accepting state
                    // with no legal continuation, rather than waiting for the next
                    // iteration's `select` to discover the same thing via
                    // `SelectOutcome::GrammarExhausted`. This path opened a real prediction
                    // this iteration (`trace.opened` above), so the standing
                    // `opened == consumed + 1` invariant holds here --
                    // `ended_without_reopening` is not set.
                    if grammar_complete() {
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
        //
        // A caller rejecting the flushed text ends the request exactly like the loop's
        // `StepOutcome::Interrupted` arm: `Interrupt`, `stopped: false`, and it wins over a
        // stop match the same tail completed (`finish_stop` applies that precedence).
        match policy.finish_stop(text, &tail, |s| emit_confirmed(s, 0)) {
            StopCheckOutcome::Interrupted => {
                stopped = false;
                stop_reason = StopReason::Interrupt;
            }
            StopCheckOutcome::Stopped if !stopped => {
                stopped = true;
                stop_reason = StopReason::Eos;
            }
            StopCheckOutcome::Stopped | StopCheckOutcome::Continue => {}
        }
    }

    Ok(DriverResult {
        generated_ids,
        token_logprobs,
        stopped,
        stop_reason,
        confirmed_stop_string_match,
        trace,
    })
}

/// Everything [`run_speculative`] produces. Logprobs and stop-string text are not
/// carried: the speculative routes refuse both, so the driver's own buffers for them stay
/// empty and the caller decodes the published ids.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(crate) struct SpeculativeResult {
    pub(crate) generated_ids: Vec<u32>,
    pub(crate) stopped: bool,
    pub(crate) stop_reason: StopReason,
    pub(crate) trace: SpeculativeTrace,
}

/// A speculative route verifies draft tokens by argmax, so it admits only the greedy
/// configuration its route predicate already selects it for. Refused here as well, in every
/// build, for the reason [`check_capabilities`] is a hard error: a driver that accepted any
/// configuration would let a caller believe sampling or a repetition penalty was applied.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
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

/// The speculative sibling of [`run`] (ADR-090 D2, D6): one loop over a
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
/// I/O primitives [`run`] takes, handed to the same policy call; a text sink that refuses a
/// delta ends the request with [`StopReason::Interrupt`].
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_speculative(
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

#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
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
mod tests {
    use super::*;
    use crate::decoder::{
        PredictionId, PredictionLedger, SelectionCandidate, StepStamp, TokenMetadata,
    };
    use crate::grammar::GrammarSpec;
    use std::sync::Arc;

    /// A one-token grammar (`root ::= "a"`) over a one-token vocabulary -- just enough for
    /// `gen_cfg.grammar` to be `Some`, which is all [`check_capabilities`] inspects. Neither
    /// test below reaches masking: `FakeSession::prefill` errors out before `select` is ever
    /// called, so the grammar is never actually run.
    fn a_trivial_grammar() -> Arc<GrammarEngine> {
        Arc::new(
            GrammarEngine::new(
                &GrammarSpec::Gbnf("root ::= \"a\"\n".into()),
                vec![b"a".to_vec()],
            )
            .expect("trivial one-token grammar must compile"),
        )
    }

    /// Declares a fixed [`ExecutionCapabilities`] and otherwise never runs: every method past
    /// `capabilities`/`prefill` panics if reached, so a test that gets past
    /// [`check_capabilities`] fails at `prefill`'s distinct sentinel error rather than at a
    /// silent no-op -- the two refusal points cannot be confused with each other.
    struct FakeSession {
        caps: ExecutionCapabilities,
    }

    impl DecoderSession for FakeSession {
        fn capabilities(&self) -> &ExecutionCapabilities {
            &self.caps
        }

        fn prefill(&mut self, _cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
            Err(InferenceError::Inference(
                "FakeSession::prefill reached -- check_capabilities let this request through"
                    .into(),
            ))
        }

        fn decode(
            &mut self,
            _accepted: &AcceptedToken,
            _cancel: &dyn Cancellation,
        ) -> Result<StepStamp, InferenceError> {
            unreachable!("FakeSession::prefill always errors before decode is reached")
        }

        fn select(
            &mut self,
            _request: &SelectionRequest<'_>,
        ) -> Result<SelectOutcome, InferenceError> {
            unreachable!("FakeSession::prefill always errors before select is reached")
        }

        fn metadata(
            &mut self,
            _prediction: PredictionId,
            _final_token: u32,
            _request: &MetadataRequest,
        ) -> Result<TokenMetadata, InferenceError> {
            unreachable!("FakeSession::prefill always errors before metadata is reached")
        }

        fn finish(&mut self, _disposition: FinishDisposition) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    fn run_with_capabilities(caps: ExecutionCapabilities) -> Result<DriverResult, InferenceError> {
        let mut session = FakeSession { caps };
        let gen_cfg = GenerateConfig {
            grammar: Some(a_trivial_grammar()),
            ..Default::default()
        };
        let cancel = || false;
        let mut text = String::new();
        let mut offsets = Vec::new();
        run(
            &mut session,
            &gen_cfg,
            None,
            &[0u32],
            999,
            false,
            &cancel,
            |_generated_len| {},
            |_next_id| String::new(),
            &mut text,
            &mut offsets,
            |_delta, _id| true,
            || {},
            String::new,
        )
    }

    // -----------------------------------------------------------------
    // Rework item 2: an undeclared-but-requested control is a hard
    // `InferenceError`, in every build -- not a `debug_assert!` a release
    // binary silently drops. A session declaring `grammar: false` while
    // `gen_cfg.grammar` is set must be refused before any session method
    // past `capabilities` is ever called.
    // -----------------------------------------------------------------

    #[test]
    fn undeclared_grammar_capability_is_a_hard_error() {
        let result = run_with_capabilities(ExecutionCapabilities {
            grammar: false,
            ..ExecutionCapabilities::default()
        });
        match result {
            Err(InferenceError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("grammar"),
                    "refusal message should name the missing capability, got: {msg}"
                );
            }
            Err(other_err) => panic!(
                "a session declaring grammar: false with gen_cfg.grammar set must be refused \
                 with InvalidInput before any session method past capabilities() runs; got a \
                 different error instead: {other_err:?}"
            ),
            Ok(_) => panic!(
                "a session declaring grammar: false with gen_cfg.grammar set must be refused; \
                 got Ok(_) instead"
            ),
        }
    }

    /// Passing control for the test above: the same request, against a session that DOES
    /// declare grammar support, must get past `check_capabilities` -- proven by reaching
    /// `FakeSession::prefill`'s distinct sentinel error rather than the capability refusal.
    /// Without this control, the test above could pass for the wrong reason (e.g. a
    /// `check_capabilities` that always refuses regardless of `caps`).
    #[test]
    fn declared_grammar_capability_passes_the_check() {
        let result = run_with_capabilities(ExecutionCapabilities {
            grammar: true,
            ..ExecutionCapabilities::default()
        });
        match result {
            Err(InferenceError::Inference(msg)) => {
                assert!(
                    msg.contains("FakeSession::prefill reached"),
                    "a session declaring grammar: true must get past check_capabilities and \
                     reach prefill; got a different error instead: {msg}"
                );
            }
            Err(other_err) => panic!(
                "expected FakeSession::prefill's sentinel error (proving check_capabilities let \
                 the request through); got a different error instead: {other_err:?}"
            ),
            Ok(_) => panic!(
                "expected FakeSession::prefill's sentinel error; got Ok(_) instead -- \
                 FakeSession::prefill should be unreachable-if-not-erroring"
            ),
        }
    }

    // -----------------------------------------------------------------
    // `on_push` vs `metadata` ordering, with `gen_cfg.logprobs` set. `on_push`
    // must fire for a token strictly before that token's `metadata` call (the
    // session-scored logprob lookup `DecodePolicy::transition_inner`'s
    // `record_metadata` callback routes to), for every generated token -- the
    // prefill-derived first token included, not just the decode-loop tokens.
    // -----------------------------------------------------------------

    /// Mutation sensitivity: `run`'s `on_push` call sites (step 0, and the push
    /// closure passed to `transition_with_metadata`) both fire before the metadata
    /// call for that same token; a change that fires `on_push` from `decode_delta`
    /// instead (the pre-fix design, which needed a caller-side counter because
    /// `decode_delta` receives only a token id, never the push-time length) would
    /// fire it *after* `metadata` at every step, since `transition_inner` calls
    /// `record_metadata` before `decode_delta` in its fixed order -- see that
    /// method's own doc comment. That reorders every pair in `events` to
    /// `[Metadata, Push(n), Metadata, Push(n+1), ...]`, and the `assert_eq!` below
    /// fails on the very first pair.
    #[test]
    fn on_push_fires_before_metadata_for_every_token_with_logprobs_enabled() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Event {
            Push(usize),
            Metadata,
        }

        /// Always accepts one fixed candidate token (never EOS, never a configured
        /// stop token), drives `PredictionLedger` correctly through select/decode,
        /// and records every `metadata` call into `events` -- enough to run a
        /// `gen_cfg.logprobs`-enabled request through several full decode-loop
        /// iterations and observe the `on_push`/`metadata` ordering for more than
        /// just the prefill-derived first token.
        struct RecordingSession {
            caps: ExecutionCapabilities,
            ledger: PredictionLedger,
            events: std::rc::Rc<std::cell::RefCell<Vec<Event>>>,
        }

        const CANDIDATE_ID: u32 = 7;

        impl DecoderSession for RecordingSession {
            fn capabilities(&self) -> &ExecutionCapabilities {
                &self.caps
            }

            fn prefill(&mut self, _cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
                Ok(StepStamp {
                    evaluated_len: 1,
                    prediction: None,
                })
            }

            fn decode(
                &mut self,
                accepted: &AcceptedToken,
                _cancel: &dyn Cancellation,
            ) -> Result<StepStamp, InferenceError> {
                self.ledger.consume(accepted.prediction)?;
                Ok(StepStamp {
                    evaluated_len: 2,
                    prediction: None,
                })
            }

            fn select(
                &mut self,
                _request: &SelectionRequest<'_>,
            ) -> Result<SelectOutcome, InferenceError> {
                Ok(SelectOutcome::Candidate(SelectionCandidate {
                    candidate_id: CANDIDATE_ID,
                    prediction: self.ledger.open(),
                }))
            }

            fn metadata(
                &mut self,
                prediction: PredictionId,
                final_token: u32,
                _request: &MetadataRequest,
            ) -> Result<TokenMetadata, InferenceError> {
                self.events.borrow_mut().push(Event::Metadata);
                Ok(TokenMetadata {
                    prediction,
                    final_token_id: final_token,
                    final_logprob: -0.1,
                    top: Vec::new(),
                })
            }

            fn finish(&mut self, _disposition: FinishDisposition) -> Result<(), InferenceError> {
                Ok(())
            }
        }

        let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::<Event>::new()));
        let mut session = RecordingSession {
            caps: ExecutionCapabilities {
                logprobs: true,
                ..ExecutionCapabilities::default()
            },
            ledger: PredictionLedger::new(),
            events: events.clone(),
        };
        // `logprobs: Some(_)` is what makes `record_metadata` a real, non-skipped
        // call at every step (see `DecodePolicy::transition_inner`'s doc comment) --
        // the ordering this test exists to pin has no observable effect otherwise.
        let gen_cfg = GenerateConfig {
            max_new_tokens: 3,
            logprobs: Some(0),
            ..Default::default()
        };
        let cancel = || false;
        let mut text = String::new();
        let mut offsets = Vec::new();
        let events_for_push = events.clone();

        let result = run(
            &mut session,
            &gen_cfg,
            None,
            &[0u32],
            999, // eos_token_id; CANDIDATE_ID (7) and the default stop_token_ids
            // (QWEN_CHAT_IM_END_TOKEN_ID, 248_046) never match it, so the loop
            // runs to gen_cfg.max_new_tokens rather than stopping early.
            false,
            &cancel,
            move |generated_len| {
                events_for_push
                    .borrow_mut()
                    .push(Event::Push(generated_len));
            },
            |_next_id| String::new(),
            &mut text,
            &mut offsets,
            |_delta, _id| true,
            || {},
            String::new,
        )
        .expect("a fixed non-EOS candidate with no grammar/stop-strings must run to the cap");

        assert_eq!(
            result.generated_ids,
            vec![CANDIDATE_ID; 3],
            "the fixed candidate must be accepted on every step"
        );

        let expected: Vec<Event> = (1..=3usize)
            .flat_map(|index| [Event::Push(index), Event::Metadata])
            .collect();
        assert_eq!(
            events.borrow().clone(),
            expected,
            "on_push must fire for token n strictly before that token's metadata call, for \
             every one of the 3 generated tokens (the prefill-derived first token included)"
        );
    }

    // -----------------------------------------------------------------
    // A caller rejecting the natural-end tail flush must end the request
    // with `Interrupt`, exactly as a mid-decode rejection does.
    // -----------------------------------------------------------------

    const TEXT_TOKEN: u32 = 7;
    const SCRIPTED_EOS: u32 = 999;

    /// Returns `script[i]` from the i-th `select` call (the last entry repeats), driving
    /// `PredictionLedger` correctly through select/decode.
    struct ScriptedSession {
        caps: ExecutionCapabilities,
        ledger: PredictionLedger,
        script: Vec<u32>,
        selects: usize,
    }

    impl DecoderSession for ScriptedSession {
        fn capabilities(&self) -> &ExecutionCapabilities {
            &self.caps
        }

        fn prefill(&mut self, _cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
            Ok(StepStamp {
                evaluated_len: 1,
                prediction: None,
            })
        }

        fn decode(
            &mut self,
            accepted: &AcceptedToken,
            _cancel: &dyn Cancellation,
        ) -> Result<StepStamp, InferenceError> {
            self.ledger.consume(accepted.prediction)?;
            Ok(StepStamp {
                evaluated_len: 2,
                prediction: None,
            })
        }

        fn select(
            &mut self,
            _request: &SelectionRequest<'_>,
        ) -> Result<SelectOutcome, InferenceError> {
            let index = self.selects.min(self.script.len() - 1);
            self.selects += 1;
            Ok(SelectOutcome::Candidate(SelectionCandidate {
                candidate_id: self.script[index],
                prediction: self.ledger.open(),
            }))
        }

        fn metadata(
            &mut self,
            _prediction: PredictionId,
            _final_token: u32,
            _request: &MetadataRequest,
        ) -> Result<TokenMetadata, InferenceError> {
            unreachable!("gen_cfg.logprobs is None, so metadata is never requested")
        }

        fn finish(&mut self, _disposition: FinishDisposition) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    /// Streams `script` with `stop_strings = ["ZZ"]`, every token decoding to "a". The sink
    /// returns false only on call number `reject_call` (1-based). Returns the result and
    /// every string the sink was handed, in order.
    fn run_streaming_script(
        script: Vec<u32>,
        max_new_tokens: usize,
        reject_call: usize,
    ) -> (DriverResult, Vec<String>) {
        let mut session = ScriptedSession {
            caps: ExecutionCapabilities {
                stop_strings: true,
                ..ExecutionCapabilities::default()
            },
            ledger: PredictionLedger::new(),
            script,
            selects: 0,
        };
        let gen_cfg = GenerateConfig {
            max_new_tokens,
            stop_strings: vec!["ZZ".to_string()],
            ..Default::default()
        };
        let cancel = || false;
        let mut text = String::new();
        let mut offsets = Vec::new();
        let mut emitted: Vec<String> = Vec::new();
        let result = run(
            &mut session,
            &gen_cfg,
            None,
            &[0u32],
            SCRIPTED_EOS,
            true,
            &cancel,
            |_generated_len| {},
            |_next_id| "a".to_string(),
            &mut text,
            &mut offsets,
            |delta, _id| {
                emitted.push(delta.to_string());
                emitted.len() != reject_call
            },
            || {},
            String::new,
        )
        .expect("a scripted non-grammar stream must not error");
        (result, emitted)
    }

    /// "ZZ" holds back one byte, so the third "a" of a 3-token run reaches the caller only
    /// through `finish_stop`: calls 1 and 2 are mid-decode, call 3 is the tail flush.
    #[test]
    fn rejected_tail_flush_after_length_cap_reports_interrupt() {
        let (control, emitted) = run_streaming_script(vec![TEXT_TOKEN], 3, usize::MAX);
        assert_eq!(
            emitted,
            vec!["a", "a", "a"],
            "fixture shape: 2 loop emits + 1 flush"
        );
        assert_eq!(
            control.stop_reason,
            StopReason::Length,
            "control: all accepted"
        );
        assert!(!control.stopped);

        let (result, emitted) = run_streaming_script(vec![TEXT_TOKEN], 3, 3);
        assert_eq!(emitted.len(), 3, "the rejected call must be the tail flush");
        assert_eq!(
            result.stop_reason,
            StopReason::Interrupt,
            "a sink rejecting the final flush must end the request as Interrupt, like a \
             mid-decode rejection"
        );
        assert!(!result.stopped, "Interrupt always reports stopped: false");
        assert!(!result.confirmed_stop_string_match);
    }

    /// Same, for a stream that ends on EOS (`stopped: true` before the flush): the rejected
    /// flush must still report `Interrupt` and clear `stopped`.
    #[test]
    fn rejected_tail_flush_after_eos_reports_interrupt() {
        let script = vec![TEXT_TOKEN, TEXT_TOKEN, SCRIPTED_EOS];
        let (control, emitted) = run_streaming_script(script.clone(), 8, usize::MAX);
        assert_eq!(
            emitted,
            vec!["a", "a"],
            "fixture shape: 1 loop emit + 1 flush"
        );
        assert_eq!(
            control.stop_reason,
            StopReason::Eos,
            "control: all accepted"
        );
        assert!(control.stopped);

        let (result, emitted) = run_streaming_script(script, 8, 2);
        assert_eq!(emitted.len(), 2, "the rejected call must be the tail flush");
        assert_eq!(result.stop_reason, StopReason::Interrupt);
        assert!(!result.stopped, "Interrupt always reports stopped: false");
    }
}

#[cfg(test)]
mod speculative_tests {
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
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use crate::decoder::{
        PredictionId, PredictionLedger, SelectionCandidate, StepStamp, TokenMetadata,
    };
    use std::cell::Cell;

    struct RecordingSession {
        caps: ExecutionCapabilities,
        ledger: PredictionLedger,
        prediction: Option<PredictionId>,
        calls: Vec<&'static str>,
        finishes: Vec<FinishDisposition>,
        failure: Option<&'static str>,
        finish_error: bool,
    }

    impl RecordingSession {
        fn new(failure: Option<&'static str>) -> Self {
            Self {
                caps: ExecutionCapabilities {
                    grammar: true,
                    logprobs: true,
                    stop_strings: true,
                    reasoning_budget: true,
                },
                ledger: PredictionLedger::new(),
                prediction: None,
                calls: Vec::new(),
                finishes: Vec::new(),
                failure,
                finish_error: false,
            }
        }

        fn record(&mut self, call: &'static str) -> Result<(), InferenceError> {
            self.calls.push(call);
            if self.failure == Some(call) {
                return Err(InferenceError::Inference(call.into()));
            }
            Ok(())
        }
    }

    impl DecoderSession for RecordingSession {
        fn capabilities(&self) -> &ExecutionCapabilities {
            &self.caps
        }

        fn prefill(&mut self, _cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
            self.record("prefill")?;
            Ok(StepStamp {
                evaluated_len: 1,
                prediction: None,
            })
        }

        fn decode(
            &mut self,
            accepted: &AcceptedToken,
            _cancel: &dyn Cancellation,
        ) -> Result<StepStamp, InferenceError> {
            self.record("decode")?;
            self.ledger.consume(accepted.prediction)?;
            Ok(StepStamp {
                evaluated_len: 2,
                prediction: None,
            })
        }

        fn select(
            &mut self,
            _request: &SelectionRequest<'_>,
        ) -> Result<SelectOutcome, InferenceError> {
            self.record("select")?;
            if self.failure == Some("grammar") {
                return Ok(SelectOutcome::GrammarExhausted);
            }
            let prediction = self.ledger.open();
            self.prediction = Some(prediction);
            Ok(SelectOutcome::Candidate(SelectionCandidate {
                candidate_id: 1,
                prediction,
            }))
        }

        fn metadata(
            &mut self,
            prediction: PredictionId,
            final_token: u32,
            _request: &MetadataRequest,
        ) -> Result<TokenMetadata, InferenceError> {
            self.record("metadata")?;
            Ok(TokenMetadata {
                prediction,
                final_token_id: final_token,
                final_logprob: 0.0,
                top: Vec::new(),
            })
        }

        fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError> {
            self.finishes.push(disposition);
            self.ledger.invalidate();
            self.record("finish")?;
            if self.finish_error {
                return Err(InferenceError::Inference("finish failed".into()));
            }
            Ok(())
        }
    }

    fn drive(
        session: &mut RecordingSession,
        cancel: &dyn Cancellation,
    ) -> Result<DriverResult, InferenceError> {
        let config = GenerateConfig {
            max_new_tokens: 2,
            logprobs: Some(0),
            ..Default::default()
        };
        run(
            session,
            &config,
            None,
            &[0],
            99,
            true,
            cancel,
            |_| {},
            |_| "a".into(),
            &mut String::new(),
            &mut Vec::new(),
            |_, _| true,
            || {},
            String::new,
        )
    }

    #[test]
    fn ordinary_completion_and_every_cancel_checkpoint_finish_once() {
        for cancel_at in [None, Some(0), Some(1), Some(2)] {
            let mut session = RecordingSession::new(None);
            let polls = Cell::new(0);
            let cancel = || {
                let at = polls.get();
                polls.set(at + 1);
                cancel_at == Some(at)
            };
            let result = drive(&mut session, &cancel).expect("driver result");
            assert_eq!(
                result.stop_reason,
                if cancel_at.is_some() {
                    StopReason::Interrupt
                } else {
                    StopReason::Length
                }
            );
            let expected = if matches!(cancel_at, Some(1 | 2)) {
                FinishDisposition::Poisoned
            } else {
                FinishDisposition::Reusable
            };
            assert_eq!(
                session.finishes,
                [expected],
                "cancel checkpoint {cancel_at:?}"
            );
            if let Some(prediction) = session.prediction {
                assert!(!session.ledger.is_live(prediction));
            }
            if cancel_at == Some(0) {
                assert_eq!(session.calls, ["finish"]);
            }
        }
    }

    #[test]
    fn ordinary_execution_errors_finish_poisoned_once() {
        for failure in ["prefill", "select", "decode", "metadata", "grammar"] {
            let mut session = RecordingSession::new(Some(failure));
            let result = drive(&mut session, &|| false);
            assert!(result.is_err(), "{failure}");
            assert!(session.calls.contains(&if failure == "grammar" {
                "select"
            } else {
                failure
            }));
            assert_eq!(session.finishes, [FinishDisposition::Poisoned], "{failure}");
            if let Some(prediction) = session.prediction {
                assert!(!session.ledger.is_live(prediction));
            }
        }
    }

    #[test]
    fn finish_errors_preserve_the_primary_execution_error() {
        let mut session = RecordingSession::new(Some("prefill"));
        session.finish_error = true;
        let result = drive(&mut session, &|| false);
        assert!(matches!(result, Err(InferenceError::Inference(message)) if message == "prefill"));
        assert_eq!(session.finishes, [FinishDisposition::Poisoned]);

        let mut session = RecordingSession::new(None);
        session.finish_error = true;
        let result = drive(&mut session, &|| false);
        assert!(
            matches!(result, Err(InferenceError::Inference(message)) if message == "finish failed")
        );
        assert_eq!(session.finishes, [FinishDisposition::Reusable]);
    }
}
