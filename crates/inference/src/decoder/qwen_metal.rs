//! `QwenMetalSession`: the ordinary Qwen3.5 Metal generation path behind the
//! object-safe [`super::DecoderSession`] boundary (ADR-090 D1).
//!
//! Wraps the work `MetalQwen35State::generate` (the direct entry) and
//! `MetalQwen35State::generate_streaming_with_cancel` (the streaming entry)
//! each did in their own decode loop. The streaming entry runs its requests
//! through this session and [`run_streaming`]; the direct entry still runs
//! its own loop, so the direct profile is constructed only by tests.
//!
//! **Owned state.** The session borrows the caller's `MetalQwen35State`
//! mutably for its whole life, so the GPU caches, scratch and the compact
//! route it engages cannot be observed by anyone else until the session is
//! dropped. It owns the prompt ids, the RNG state, the prediction ledger and
//! the most recent logit readback.
//!
//! **Readback mode, fixed at construction.** Which readback a forward pass
//! produces is decided once, in [`QwenMetalSession::new`], by the same
//! `plan_sampling_route` the two legacy loops call, plus the direct entry's
//! greedy zero-copy predicate:
//!
//! - [`ReadbackMode::Compact`]: the planner engaged a GPU top-k route; every
//!   forward pass (prefill included) leaves a candidate shortlist in the
//!   state's `compact_result` and returns no logits.
//! - [`ReadbackMode::GreedyArgmax`]: direct entry only, greedy with no
//!   compact route and no grammar. Prefill returns dense logits (the
//!   prefill-derived token is sampled from them, as the direct loop does);
//!   every decode step returns the argmax id straight from the GPU buffer.
//! - [`ReadbackMode::Dense`]: every other request; each forward pass returns
//!   the full-vocabulary logit row.
//!
//! The mode is never re-planned per step.
//!
//! **RNG.** Seeded from `GenerationPlan::rng_state`, the value
//! `prepare_generation` derives from `GenerateConfig::seed` and the value
//! both legacy loops destructure, and drawn only inside `select`, with the
//! legacy per-mode schedule: the prefill-derived token uses
//! `sample_from_candidates` (compact) or `sample_token` (dense and greedy
//! argmax), every later token uses `sample_decode_traced` (compact and dense)
//! or no draw at all (greedy argmax).
//!
//! **Compact-route teardown.** Construction engages the planned route on the
//! state; `disengage_compact_route` tears it down. `driver::run` calls
//! `finish` only on a completed request: it returns through `?` on every
//! error and returns early without `finish` on cancellation before or right
//! after prefill. The teardown therefore lives in `Drop`, with `finish`
//! calling the same idempotent method. The state is reachable only through
//! this session's exclusive borrow, which ends exactly when the session is
//! dropped, so no caller can observe an engaged route after any exit path,
//! unwinding included.
//!
//! **Not `Send`.** ADR-090 keeps the Metal session on the thread that
//! created it. The `metal` object wrappers `MetalQwen35State` holds are
//! themselves `Send`, so borrowing the state does not make the session
//! `!Send`; a raw-pointer `PhantomData` marker does, without an unsafe
//! auto-trait implementation. A compile-time control in the test module pins
//! that.

// Only this module's tests construct the direct profile; the direct entry is
// routed through the session separately.
#![cfg_attr(not(test), allow(dead_code))]

use super::driver;
use super::qwen_cpu::has_finite_logit;
use super::{
    AcceptedToken, Cancellation, DecoderSession, ExecutionCapabilities, FinishDisposition,
    MetadataRequest, PredictionError, PredictionId, PredictionLedger, SelectOutcome,
    SelectionCandidate, SelectionRequest, StepStamp, TokenMetadata,
};
use crate::error::InferenceError;
use crate::forward::metal_qwen35::{
    GpuTopkRoute, MetalQwen35State, SamplingRouteEnvironment, SamplingRoutePlan,
    apply_sampling_route_plan, mtp_route_active, plan_sampling_route, sample_decode_traced,
    sample_from_candidates, sample_token, self_spec_route_active,
};
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::model::qwen35::{
    GenerationPlan, check_logprobs_not_set, check_reasoning_budget_not_set,
};
use crate::sampling::compute_step_logprobs;
use crate::stop_reason::StopReason;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::detokenize::IncrementalDetokenizer;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;

/// Which ordinary Metal entry point the session reproduces. The two entries
/// differ in what they refuse and in how they read back decode logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalEntryProfile {
    /// `MetalQwen35State::generate`.
    Direct,
    /// `MetalQwen35State::generate_streaming_with_cancel`.
    Streaming,
}

/// The direct entry refuses `reasoning_budget` and `logprobs` before
/// tokenization-dependent work (`GenerationEntryContract::MetalDirect`) and
/// wires grammar masking and a stop-string matcher itself.
const DIRECT_CAPABILITIES: ExecutionCapabilities = ExecutionCapabilities {
    grammar: true,
    logprobs: false,
    stop_strings: true,
    reasoning_budget: false,
};

/// The streaming entry refuses none of the four controls
/// (`GenerationEntryContract::MetalStreaming`) and wires all of them.
const STREAMING_CAPABILITIES: ExecutionCapabilities = ExecutionCapabilities {
    grammar: true,
    logprobs: true,
    stop_strings: true,
    reasoning_budget: true,
};

/// Readback route for every forward pass of one request. See the module doc
/// comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadbackMode {
    Dense,
    Compact { route: GpuTopkRoute, top_k: usize },
    GreedyArgmax,
}

/// What the most recent forward pass left for `select` and `metadata`.
#[derive(Debug)]
enum Readback {
    /// No forward pass since construction, or the last one was consumed.
    Empty,
    /// A full-vocabulary logit row. Grammar masking is applied to it in place.
    Dense(Vec<f32>),
    /// Candidates live in the state's `compact_result`.
    Compact,
    /// The argmax id of the last decode step.
    Argmax(u32),
}

/// Derives the readback mode from the route plan. The greedy predicate is the
/// direct entry's own `greedy_fast` condition, which that entry evaluates once
/// per request after planning the route.
fn readback_mode(
    profile: MetalEntryProfile,
    plan: SamplingRoutePlan,
    gen_cfg: &GenerateConfig,
) -> ReadbackMode {
    if plan.use_compact {
        return ReadbackMode::Compact {
            route: plan.compact_route,
            top_k: plan.compact_topk,
        };
    }
    let greedy_fast = gen_cfg.temperature <= 0.0
        && gen_cfg.top_k <= 1
        && gen_cfg.repetition_penalty == 1.0
        && gen_cfg.grammar.is_none();
    if profile == MetalEntryProfile::Direct && greedy_fast {
        ReadbackMode::GreedyArgmax
    } else {
        ReadbackMode::Dense
    }
}

/// Grammar masking and logprob scoring read the full-vocabulary logit row, so
/// a request carrying either must have planned the dense readback. The
/// planner already refuses the compact route for both; this asserts it rather
/// than assuming it.
fn check_dense_for_grammar_and_logprobs(
    gen_cfg: &GenerateConfig,
    mode: ReadbackMode,
) -> Result<(), InferenceError> {
    if (gen_cfg.grammar.is_some() || gen_cfg.logprobs.is_some()) && mode != ReadbackMode::Dense {
        return Err(InferenceError::Inference(format!(
            "grammar and logprobs need the dense logit readback, but the route planner \
             selected {mode:?}"
        )));
    }
    Ok(())
}

/// Refuses a request the direct entry would hand to its MTP or GDN-first
/// self-speculative route rather than run through its ordinary loop. The
/// streaming entry has no such routes and runs every request through its
/// ordinary loop, so a streaming session admits the same request.
fn refuse_route_owned_elsewhere(
    profile: MetalEntryProfile,
    mtp_route: bool,
    self_spec_route: bool,
) -> Result<(), InferenceError> {
    if profile != MetalEntryProfile::Direct {
        return Ok(());
    }
    if mtp_route {
        return Err(InferenceError::InvalidInput(
            "this request selects the MTP draft/verify route (an MTP checkpoint, enable_mtp or \
             LATTICE_MTP, and a greedy configuration); the ordinary Metal decoder session does \
             not run it"
                .into(),
        ));
    }
    if self_spec_route {
        return Err(InferenceError::InvalidInput(
            "this request selects the GDN-first self-speculative route (LATTICE_SELF_SPEC and a \
             greedy configuration); the ordinary Metal decoder session does not run it"
                .into(),
        ));
    }
    Ok(())
}

/// The ordinary Qwen3.5 Metal generation path as a [`DecoderSession`]. See
/// the module doc comment for what it owns and why.
pub(crate) struct QwenMetalSession<'state> {
    state: &'state mut MetalQwen35State,
    capabilities: ExecutionCapabilities,
    mode: ReadbackMode,
    prompt_ids: Vec<u32>,
    rng_state: u64,
    temperature: f32,
    ledger: PredictionLedger,
    readback: Readback,
    /// The next `select` samples the prefill-derived token.
    prefill_token_pending: bool,
    /// The planned compact route is engaged on `state` and not yet torn down.
    route_engaged: bool,
    /// Keeps the session on the thread that created it (not `Send`, not `Sync`).
    thread_bound: PhantomData<*const ()>,
}

impl<'state> QwenMetalSession<'state> {
    /// Resets `state` for a new request and engages the readback route the
    /// current environment plans for `gen_cfg`.
    ///
    /// `plan` is the `prepare_generation` result for this request, under
    /// `GenerationEntryContract::MetalDirect` or `MetalStreaming` to match
    /// `profile`; its prompt ids and RNG state are taken as they are.
    ///
    /// # Errors
    ///
    /// Before any state mutation:
    /// - the direct profile refuses `reasoning_budget` and `logprobs` with the
    ///   errors the direct entry returns for them;
    /// - the direct profile refuses a request its entry routes to MTP or
    ///   GDN-first self-speculation;
    /// - a grammar or logprobs request whose planned mode is not dense is
    ///   refused as an internal invariant failure.
    pub(crate) fn new(
        state: &'state mut MetalQwen35State,
        plan: GenerationPlan,
        gen_cfg: &GenerateConfig,
        profile: MetalEntryProfile,
    ) -> Result<Self, InferenceError> {
        Self::with_route_environment(
            state,
            plan,
            gen_cfg,
            profile,
            SamplingRouteEnvironment::current(),
        )
    }

    fn with_route_environment(
        state: &'state mut MetalQwen35State,
        plan: GenerationPlan,
        gen_cfg: &GenerateConfig,
        profile: MetalEntryProfile,
        environment: SamplingRouteEnvironment,
    ) -> Result<Self, InferenceError> {
        if profile == MetalEntryProfile::Direct {
            check_reasoning_budget_not_set(gen_cfg)?;
            check_logprobs_not_set(gen_cfg)?;
        }

        let route = plan_sampling_route(gen_cfg, plan.prompt_ids.is_empty(), environment);
        let mode = readback_mode(profile, route, gen_cfg);
        check_dense_for_grammar_and_logprobs(gen_cfg, mode)?;

        let mtp_enabled = gen_cfg
            .enable_mtp
            .unwrap_or_else(|| crate::env_switch_enabled("LATTICE_MTP"));
        refuse_route_owned_elsewhere(
            profile,
            mtp_route_active(
                state.session.mtp.is_some(),
                mtp_enabled,
                gen_cfg,
                route.use_compact,
            ),
            self_spec_route_active(
                state.session.gdn_checkpoints.is_some(),
                crate::env_switch_enabled("LATTICE_SELF_SPEC"),
                gen_cfg,
                route.use_compact,
                state.engine.config.num_active_linear_attention_layers(),
            ),
        )?;

        state.reset_state();
        let route_engaged = apply_sampling_route_plan(
            route,
            &mut state.session.compact_route,
            &mut state.session.compact_topk,
            &mut state.session.compact_result,
        );

        Ok(Self {
            state,
            capabilities: match profile {
                MetalEntryProfile::Direct => DIRECT_CAPABILITIES,
                MetalEntryProfile::Streaming => STREAMING_CAPABILITIES,
            },
            mode,
            prompt_ids: plan.prompt_ids,
            rng_state: plan.rng_state,
            temperature: gen_cfg.temperature,
            ledger: PredictionLedger::new(),
            readback: Readback::Empty,
            prefill_token_pending: false,
            route_engaged,
            thread_bound: PhantomData,
        })
    }

    /// The readback mode fixed at construction.
    pub(crate) fn mode(&self) -> ReadbackMode {
        self.mode
    }

    fn disengage_route(&mut self) {
        if self.route_engaged {
            self.state.disengage_compact_route();
            self.route_engaged = false;
        }
    }
}

impl Drop for QwenMetalSession<'_> {
    fn drop(&mut self) {
        self.disengage_route();
    }
}

impl DecoderSession for QwenMetalSession<'_> {
    fn capabilities(&self) -> &ExecutionCapabilities {
        &self.capabilities
    }

    /// Batched prefill through `try_forward_prefill`, the fallible entry both
    /// legacy loops call: it refuses before any state mutation (a non-fresh
    /// session, an out-of-vocabulary id, a range beyond capacity, a
    /// multi-token prompt on an MoE model without LoRA), so an error here
    /// leaves nothing to undo except the route, which `Drop` tears down.
    fn prefill(&mut self, cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        if cancel.is_cancelled() {
            return Err(InferenceError::Inference("cancelled before prefill".into()));
        }
        self.ledger.reset();
        let logits = self.state.try_forward_prefill(&self.prompt_ids)?;
        self.readback = match self.mode {
            ReadbackMode::Compact { .. } => Readback::Compact,
            ReadbackMode::Dense | ReadbackMode::GreedyArgmax => Readback::Dense(logits),
        };
        self.prefill_token_pending = true;
        Ok(StepStamp {
            evaluated_len: self.state.session.position(),
            prediction: None,
        })
    }

    /// Consumes `accepted.prediction`, then runs one decode forward pass at
    /// the current cache position for the accepted final id, leaving that
    /// pass's readback for the next `select`. A full KV cache is refused
    /// before the forward pass; `prepare_generation`'s context-budget check
    /// makes that unreachable for a request it admitted.
    fn decode(
        &mut self,
        accepted: &AcceptedToken,
        cancel: &dyn Cancellation,
    ) -> Result<StepStamp, InferenceError> {
        if cancel.is_cancelled() {
            return Err(InferenceError::Inference("cancelled before decode".into()));
        }
        self.ledger.consume(accepted.prediction)?;
        self.readback = Readback::Empty;
        self.prefill_token_pending = false;

        let position = self.state.session.position();
        let capacity = self.state.max_context();
        if position >= capacity {
            return Err(InferenceError::Inference(format!(
                "KV cache is full ({position} of {capacity} positions); no decode step fits"
            )));
        }

        self.readback = match self.mode {
            ReadbackMode::Dense => {
                Readback::Dense(self.state.forward_step_decode(accepted.final_id, position))
            }
            ReadbackMode::Compact { .. } => {
                self.state.forward_step_decode(accepted.final_id, position);
                Readback::Compact
            }
            ReadbackMode::GreedyArgmax => Readback::Argmax(
                self.state
                    .forward_step_greedy_argmax(accepted.final_id, position),
            ),
        };

        Ok(StepStamp {
            evaluated_len: self.state.session.position(),
            prediction: None,
        })
    }

    /// Samples the next candidate from the current readback and opens a
    /// prediction for it. Only the dense readback can be grammar-masked; a
    /// mask arriving in another mode is refused rather than ignored.
    fn select(&mut self, request: &SelectionRequest<'_>) -> Result<SelectOutcome, InferenceError> {
        let prefill_token = self.prefill_token_pending;
        let candidate_id = match &mut self.readback {
            Readback::Empty => {
                return Err(InferenceError::Inference(
                    "select needs a readback from prefill or decode".into(),
                ));
            }
            Readback::Dense(logits) => {
                if let Some(mask) = request.grammar_mask {
                    // Decode-step masking is timed and the prefill-derived step is not,
                    // as in every other Metal decode loop.
                    let _signpost_grammar = (!prefill_token).then(|| {
                        crate::forward::signpost::interval(
                            crate::forward::signpost::Label::DecodeGrammarMask,
                        )
                    });
                    mask(logits)?;
                    if !has_finite_logit(logits) {
                        return Ok(SelectOutcome::GrammarExhausted);
                    }
                }
                if prefill_token {
                    sample_token(logits, request.config, request.history, &mut self.rng_state)
                } else {
                    sample_decode_traced(
                        None,
                        logits,
                        request.config,
                        request.history,
                        &mut self.rng_state,
                    )
                }
            }
            Readback::Compact | Readback::Argmax(_) if request.grammar_mask.is_some() => {
                return Err(InferenceError::Inference(format!(
                    "a grammar mask needs the dense logit readback, but this session reads back \
                     {:?}",
                    self.mode
                )));
            }
            Readback::Compact => {
                let candidates = self.state.session.compact_result.as_slice();
                if prefill_token {
                    sample_from_candidates(
                        candidates,
                        request.config,
                        request.history,
                        &mut self.rng_state,
                    )
                } else {
                    sample_decode_traced(
                        Some(candidates),
                        &[],
                        request.config,
                        request.history,
                        &mut self.rng_state,
                    )
                }
            }
            Readback::Argmax(id) => *id,
        };
        let prediction = self.ledger.open();
        Ok(SelectOutcome::Candidate(SelectionCandidate {
            candidate_id,
            prediction,
        }))
    }

    /// Scores `final_token` against the live prediction's dense logits, after
    /// any grammar mask `select` applied, with the same
    /// `compute_step_logprobs` the legacy streaming loop reaches through
    /// `DecodePolicy`. Only the dense readback has a full distribution to
    /// score.
    fn metadata(
        &mut self,
        prediction: PredictionId,
        final_token: u32,
        request: &MetadataRequest,
    ) -> Result<TokenMetadata, InferenceError> {
        if !self.ledger.is_live(prediction) {
            return Err(PredictionError::Stale.into());
        }
        let Readback::Dense(logits) = &self.readback else {
            return Err(InferenceError::Inference(format!(
                "token metadata needs the dense logit readback, but this session reads back {:?}",
                self.mode
            )));
        };
        let (final_logprob, top) = compute_step_logprobs(
            logits,
            final_token,
            self.temperature,
            request.top_logprobs.unwrap_or(0),
        );
        Ok(TokenMetadata {
            prediction,
            final_token_id: final_token,
            final_logprob,
            top,
        })
    }

    /// Tears down the compact route and ends the live prediction, whatever
    /// the disposition.
    fn finish(&mut self, _disposition: FinishDisposition) -> Result<(), InferenceError> {
        self.disengage_route();
        self.ledger.invalidate();
        self.readback = Readback::Empty;
        Ok(())
    }
}

/// The error the streaming entry returns when the grammar blocks every token
/// before the first one is emitted.
const STEP_ZERO_GRAMMAR_BLOCKED: &str = "grammar constraint blocked every token at step 0; \
     no legal first token exists in the current grammar state";

/// Adapts the streaming entry's `FnMut` cancellation poll to [`Cancellation`],
/// whose blanket impl covers only `Fn` closures. `driver::run` polls it
/// sequentially from one thread, so the borrow never conflicts.
struct FnMutCancellation<'a, F: FnMut() -> bool>(&'a RefCell<F>);

impl<F: FnMut() -> bool> Cancellation for FnMutCancellation<'_, F> {
    fn is_cancelled(&self) -> bool {
        (*self.0.borrow_mut())()
    }
}

/// Runs one `MetalQwen35State::generate_streaming_with_cancel` request through
/// [`driver::run`] over `session` and returns that entry's output.
///
/// `should_cancel` is polled at the driver's three checkpoints: before
/// prefill, right after it, and at the top of every decode iteration. Text
/// reaches `on_token` through an incremental detokenizer and the policy's
/// stop-string matcher, as it did in the entry's own loop.
///
/// Three parts of the entry's contract differ from what the driver reports,
/// and this function keeps the entry's:
///
/// 1. The text released by the natural-end flush reaches `on_token` with the
///    id of the last emitted token; the driver hands the flush the id 0.
/// 2. A sampled or budget-forced token the grammar rejects ends the request
///    with `stopped: false` and `StopReason::Grammar`, unless the flush then
///    completes a stop string, which reports `stopped: true` and
///    `StopReason::Eos`. The driver reports `stopped: true` for the
///    rejection and keeps `Grammar` through the flush. A rejection is the one
///    grammar stop that leaves the last opened prediction unpushed, so it is
///    read from the trace; the flush completed a stop string exactly when the
///    matcher withheld decoded text from `text`.
/// 3. A grammar that blocks every token before the first one is emitted keeps
///    the entry's step-0 message.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_streaming(
    session: &mut dyn DecoderSession,
    gen_cfg: &GenerateConfig,
    think_close_id: Option<u32>,
    prompt_ids: &[u32],
    eos_token_id: u32,
    tokenizer: &BpeTokenizer,
    mut on_token: impl FnMut(&str, u32) -> bool,
    should_cancel: impl FnMut() -> bool,
) -> Result<GenerateOutput, InferenceError> {
    let should_cancel = RefCell::new(should_cancel);
    let cancel = FnMutCancellation(&should_cancel);
    let detok = RefCell::new(IncrementalDetokenizer::new());
    let last_pushed: Cell<Option<u32>> = Cell::new(None);
    let decoded_len = Cell::new(0usize);
    let flushing_tail = Cell::new(false);
    let mut text = String::new();
    let mut token_logprob_end_offsets: Vec<usize> = Vec::new();

    let result = driver::run(
        session,
        gen_cfg,
        think_close_id,
        prompt_ids,
        eos_token_id,
        true,
        &cancel,
        |_generated_len| {},
        |next_id| {
            last_pushed.set(Some(next_id));
            let delta = detok.borrow_mut().push(tokenizer, next_id);
            decoded_len.set(decoded_len.get() + delta.len());
            delta
        },
        &mut text,
        &mut token_logprob_end_offsets,
        |delta, next_id| {
            let id = if flushing_tail.get() {
                last_pushed.get().unwrap_or(next_id)
            } else {
                next_id
            };
            on_token(delta, id)
        },
        || {},
        || {
            flushing_tail.set(true);
            let tail = detok.borrow_mut().finish();
            decoded_len.set(decoded_len.get() + tail.len());
            tail
        },
    );
    let result = match result {
        Err(InferenceError::GrammarConstraintBlocked(_)) if last_pushed.get().is_none() => {
            return Err(InferenceError::GrammarConstraintBlocked(
                STEP_ZERO_GRAMMAR_BLOCKED.into(),
            ));
        }
        other => other?,
    };

    let mut stopped = result.stopped;
    let mut stop_reason = result.stop_reason;
    if stop_reason == StopReason::Grammar && result.trace.opened == result.generated_ids.len() + 1 {
        let flush_completed_stop_string = text.len() < decoded_len.get();
        stopped = flush_completed_stop_string;
        if flush_completed_stop_string {
            stop_reason = StopReason::Eos;
        }
    }

    Ok(GenerateOutput {
        text,
        prompt_tokens: prompt_ids.len(),
        generated_tokens: result.generated_ids.len(),
        token_ids: result.generated_ids,
        stopped,
        stop_reason: Some(stop_reason),
        token_logprobs: result.token_logprobs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::driver;
    use crate::measurement::gpu_test_lock;
    use crate::model::qwen35::AttentionWeights;
    use crate::model::qwen35::{
        CommonLayerWeights, DenseFfnWeights, FeedForwardWeights, FullAttentionLayerWeights,
        ModelWeights,
    };
    use crate::model::qwen35_config::{LayerType, Qwen35Config};
    use crate::sampling::Candidate;
    use crate::stop_reason::StopReason;

    // -----------------------------------------------------------------
    // Compile-time control: the session must not be `Send` (ADR-090 D1).
    // `AmbiguousIfSend<_>` has one impl for every type and a second for
    // every `Send` type, so naming `some_item` through it only resolves when
    // exactly one impl applies, i.e. when the type is not `Send`.
    // -----------------------------------------------------------------

    trait AmbiguousIfSend<A> {
        fn some_item() {}
    }
    impl<T: ?Sized> AmbiguousIfSend<()> for T {}
    #[allow(dead_code)]
    struct IsSend;
    impl<T: ?Sized + Send> AmbiguousIfSend<IsSend> for T {}
    const _: fn() = || {
        let _ = <QwenMetalSession<'static> as AmbiguousIfSend<_>>::some_item;
    };

    // -----------------------------------------------------------------
    // Capabilities: each profile's values come from its entry's refusals.
    // -----------------------------------------------------------------

    #[test]
    fn capabilities_mirror_each_entry_profiles_refusals() {
        assert_eq!(
            DIRECT_CAPABILITIES,
            ExecutionCapabilities {
                grammar: true,
                logprobs: false,
                stop_strings: true,
                reasoning_budget: false,
            }
        );
        assert_eq!(
            STREAMING_CAPABILITIES,
            ExecutionCapabilities {
                grammar: true,
                logprobs: true,
                stop_strings: true,
                reasoning_budget: true,
            }
        );
    }

    // -----------------------------------------------------------------
    // Readback-mode derivation (no Metal device needed).
    // -----------------------------------------------------------------

    const COMPACT_ENV: SamplingRouteEnvironment = SamplingRouteEnvironment {
        compact: true,
        selection: false,
        approximate_top_p: false,
    };
    const DENSE_ENV: SamplingRouteEnvironment = SamplingRouteEnvironment {
        compact: false,
        selection: false,
        approximate_top_p: false,
    };

    fn greedy() -> GenerateConfig {
        GenerateConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..Default::default()
        }
    }

    fn sampled_block_topk() -> GenerateConfig {
        GenerateConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..Default::default()
        }
    }

    fn mode_for(
        profile: MetalEntryProfile,
        gen_cfg: &GenerateConfig,
        env: SamplingRouteEnvironment,
    ) -> ReadbackMode {
        readback_mode(profile, plan_sampling_route(gen_cfg, false, env), gen_cfg)
    }

    fn a_grammar() -> std::sync::Arc<crate::grammar::GrammarEngine> {
        std::sync::Arc::new(
            crate::grammar::GrammarEngine::new(
                &crate::grammar::GrammarSpec::Gbnf("root ::= \"a\"\n".into()),
                vec![b"a".to_vec()],
            )
            .expect("one-token grammar compiles"),
        )
    }

    #[test]
    fn readback_mode_follows_the_route_planner_and_the_direct_greedy_predicate() {
        use MetalEntryProfile::{Direct, Streaming};

        // Greedy with no compact route: only the direct entry has the argmax path.
        assert_eq!(
            mode_for(Direct, &greedy(), DENSE_ENV),
            ReadbackMode::GreedyArgmax
        );
        assert_eq!(
            mode_for(Streaming, &greedy(), DENSE_ENV),
            ReadbackMode::Dense
        );

        // The compact route wins over the greedy predicate on both entries.
        let block_argmax = ReadbackMode::Compact {
            route: GpuTopkRoute::BlockArgmax,
            top_k: 1,
        };
        assert_eq!(mode_for(Direct, &greedy(), COMPACT_ENV), block_argmax);
        assert_eq!(mode_for(Streaming, &greedy(), COMPACT_ENV), block_argmax);
        assert_eq!(
            mode_for(Streaming, &sampled_block_topk(), COMPACT_ENV),
            ReadbackMode::Compact {
                route: GpuTopkRoute::BlockTopK { local_k: 40 },
                top_k: 40,
            }
        );
        assert_eq!(
            mode_for(Direct, &sampled_block_topk(), DENSE_ENV),
            ReadbackMode::Dense
        );

        // Grammar and logprobs turn the compact route off in the planner.
        let grammar = GenerateConfig {
            grammar: Some(a_grammar()),
            ..greedy()
        };
        assert_eq!(mode_for(Direct, &grammar, COMPACT_ENV), ReadbackMode::Dense);
        let logprobs = GenerateConfig {
            logprobs: Some(2),
            ..greedy()
        };
        assert_eq!(
            mode_for(Streaming, &logprobs, COMPACT_ENV),
            ReadbackMode::Dense
        );
    }

    #[test]
    fn grammar_and_logprobs_are_refused_outside_the_dense_mode() {
        let compact = ReadbackMode::Compact {
            route: GpuTopkRoute::BlockArgmax,
            top_k: 1,
        };
        let grammar = GenerateConfig {
            grammar: Some(a_grammar()),
            ..greedy()
        };
        let logprobs = GenerateConfig {
            logprobs: Some(1),
            ..greedy()
        };
        for (gen_cfg, mode) in [
            (&grammar, compact),
            (&grammar, ReadbackMode::GreedyArgmax),
            (&logprobs, compact),
            (&logprobs, ReadbackMode::GreedyArgmax),
        ] {
            let refused = check_dense_for_grammar_and_logprobs(gen_cfg, mode);
            assert!(
                matches!(&refused, Err(InferenceError::Inference(msg)) if msg.contains("dense logit readback")),
                "{mode:?}: {refused:?}"
            );
        }
        // Controls: the dense mode admits both, and neither control needs it.
        assert!(check_dense_for_grammar_and_logprobs(&grammar, ReadbackMode::Dense).is_ok());
        assert!(check_dense_for_grammar_and_logprobs(&logprobs, ReadbackMode::Dense).is_ok());
        assert!(check_dense_for_grammar_and_logprobs(&greedy(), compact).is_ok());
    }

    #[test]
    fn only_the_direct_profile_refuses_routes_its_entry_hands_elsewhere() {
        use MetalEntryProfile::{Direct, Streaming};

        let mtp = refuse_route_owned_elsewhere(Direct, true, false);
        assert!(
            matches!(&mtp, Err(InferenceError::InvalidInput(msg)) if msg.contains("MTP")),
            "{mtp:?}"
        );
        let self_spec = refuse_route_owned_elsewhere(Direct, false, true);
        assert!(
            matches!(&self_spec, Err(InferenceError::InvalidInput(msg)) if msg.contains("self-speculative")),
            "{self_spec:?}"
        );
        // Controls: the ordinary direct request, and the streaming entry, which
        // runs these requests through its ordinary loop.
        assert!(refuse_route_owned_elsewhere(Direct, false, false).is_ok());
        assert!(refuse_route_owned_elsewhere(Streaming, true, true).is_ok());
    }

    // -----------------------------------------------------------------
    // Session-level tests over a tiny Metal state. Constructing the state
    // allocates GPU buffers and compiles pipelines; no test below runs a
    // forward pass. Each injects the readback a forward pass would leave.
    // -----------------------------------------------------------------

    const TINY_VOCAB: usize = 64;
    const TINY_CACHE: usize = 32;

    fn tiny_fixture() -> (Qwen35Config, ModelWeights) {
        let hidden = 512usize;
        let intermediate = 64usize;
        let cfg = Qwen35Config {
            hidden_size: hidden,
            num_hidden_layers: 1,
            vocab_size: TINY_VOCAB,
            intermediate_size: intermediate,
            rms_norm_eps: 1e-6,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            linear_num_key_heads: 1,
            linear_num_value_heads: Some(1),
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            full_attention_interval: 1,
            layer_types: vec![LayerType::FullAttention],
            layer_mask: vec![true],
            eos_token_id: (TINY_VOCAB - 1) as u32,
            max_position_embeddings: 128,
            quarot_rotation_seed: None,
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        };
        let common = CommonLayerWeights {
            input_layernorm: vec![1.0; hidden],
            post_attention_layernorm: vec![1.0; hidden],
            ffn: FeedForwardWeights::Dense(DenseFfnWeights {
                gate_proj: vec![0.0; intermediate * hidden],
                up_proj: vec![0.0; intermediate * hidden],
                down_proj: vec![0.0; hidden * intermediate],
            }),
        };
        let full = FullAttentionLayerWeights {
            q_proj: vec![0.0; 2 * cfg.full_q_dim() * hidden],
            k_proj: vec![0.0; cfg.full_kv_dim() * hidden],
            v_proj: vec![0.0; cfg.full_kv_dim() * hidden],
            o_proj: vec![0.0; hidden * cfg.full_q_dim()],
            q_norm: vec![1.0; cfg.head_dim],
            k_norm: vec![1.0; cfg.head_dim],
        };
        let weights = ModelWeights {
            embed_tokens: vec![0.0; TINY_VOCAB * hidden],
            lm_head: None,
            final_norm: vec![1.0; hidden],
            layers: vec![(AttentionWeights::Full(full), common)],
        };
        (cfg, weights)
    }

    fn plan(prompt_ids: Vec<u32>, rng_state: u64) -> GenerationPlan {
        GenerationPlan {
            prompt_len: prompt_ids.len(),
            required_capacity: prompt_ids.len() + 1,
            prompt_ids,
            rng_state,
        }
    }

    fn session<'s>(
        state: &'s mut MetalQwen35State,
        gen_cfg: &GenerateConfig,
        profile: MetalEntryProfile,
        env: SamplingRouteEnvironment,
    ) -> QwenMetalSession<'s> {
        QwenMetalSession::with_route_environment(
            state,
            plan(vec![1, 2, 3], 7),
            gen_cfg,
            profile,
            env,
        )
        .expect("an ordinary request constructs a session")
    }

    fn request<'a>(gen_cfg: &'a GenerateConfig, history: &'a [u32]) -> SelectionRequest<'a> {
        SelectionRequest {
            config: gen_cfg,
            history,
            grammar_mask: None,
        }
    }

    fn candidate(outcome: SelectOutcome) -> SelectionCandidate {
        match outcome {
            SelectOutcome::Candidate(candidate) => candidate,
            SelectOutcome::GrammarExhausted => panic!("no grammar is set"),
        }
    }

    /// The injected readback a dense forward pass would leave, peaked at `peak`.
    fn dense_row(peak: u32) -> Vec<f32> {
        let mut logits = vec![0.0f32; TINY_VOCAB];
        logits[peak as usize] = 10.0;
        logits
    }

    fn assert_route_disengaged(state: &MetalQwen35State) {
        assert_eq!(state.session.compact_route, GpuTopkRoute::CpuFallback);
        assert_eq!(state.session.compact_topk, 0);
    }

    #[test]
    fn decoder_session_mode_is_fixed_at_construction_and_engaged_on_the_state() {
        let Some(_) = metal::Device::system_default() else {
            return;
        };
        let _gpu = gpu_test_lock();
        let (cfg, weights) = tiny_fixture();
        let mut state =
            MetalQwen35State::new(&weights, &cfg, TINY_CACHE).expect("tiny Metal state");
        let gen_cfg = greedy();
        let history = [1u32, 2, 3];
        let mut s = session(
            &mut state,
            &gen_cfg,
            MetalEntryProfile::Streaming,
            COMPACT_ENV,
        );
        let planned = ReadbackMode::Compact {
            route: GpuTopkRoute::BlockArgmax,
            top_k: 1,
        };
        assert_eq!(s.mode(), planned);
        assert_eq!(s.state.session.compact_route, GpuTopkRoute::BlockArgmax);
        assert_eq!(s.state.session.compact_topk, 1);

        // Two selects over an injected compact readback: the mode and the
        // engaged route stay as planned, and the candidate comes from the
        // shortlist without any vocabulary-sized row.
        s.state.session.compact_result = vec![Candidate {
            token_id: 9,
            logit: 1.0,
        }];
        s.readback = Readback::Compact;
        s.prefill_token_pending = true;
        for _ in 0..2 {
            let c = candidate(s.select(&request(&gen_cfg, &history)).expect("select"));
            assert_eq!(c.candidate_id, 9);
            assert_eq!(s.mode(), planned);
            assert_eq!(s.state.session.compact_route, GpuTopkRoute::BlockArgmax);
            assert_eq!(s.state.session.compact_topk, 1);
        }
        drop(s);
        assert_route_disengaged(&state);

        // Control: the same request under the dense environment plans the
        // streaming dense mode and engages nothing.
        let s = session(
            &mut state,
            &gen_cfg,
            MetalEntryProfile::Streaming,
            DENSE_ENV,
        );
        assert_eq!(s.mode(), ReadbackMode::Dense);
        assert_eq!(s.state.session.compact_topk, 0);
    }

    #[test]
    fn decoder_session_ledger_refuses_stale_ids_and_keeps_one_live_prediction() {
        let Some(_) = metal::Device::system_default() else {
            return;
        };
        let _gpu = gpu_test_lock();
        let (cfg, weights) = tiny_fixture();
        let mut state =
            MetalQwen35State::new(&weights, &cfg, TINY_CACHE).expect("tiny Metal state");
        let gen_cfg = greedy();
        let history = [1u32, 2, 3];
        let mut s = session(
            &mut state,
            &gen_cfg,
            MetalEntryProfile::Streaming,
            DENSE_ENV,
        );
        s.readback = Readback::Dense(dense_row(5));
        s.prefill_token_pending = true;

        let first = candidate(s.select(&request(&gen_cfg, &history)).expect("select"));
        assert_eq!(first.candidate_id, 5);
        assert!(s.ledger.is_live(first.prediction));
        let meta = MetadataRequest {
            top_logprobs: Some(1),
        };
        // Control: the live prediction scores.
        s.metadata(first.prediction, first.candidate_id, &meta)
            .expect("metadata of the live prediction");

        // A second select supersedes the first: one live prediction at a time.
        let second = candidate(s.select(&request(&gen_cfg, &history)).expect("select"));
        assert!(s.ledger.is_live(second.prediction));
        assert!(!s.ledger.is_live(first.prediction));
        assert!(matches!(
            s.metadata(first.prediction, first.candidate_id, &meta),
            Err(InferenceError::Inference(msg)) if msg.contains("stale prediction id")
        ));

        // A stale id is refused by decode before any forward pass runs.
        let stale = AcceptedToken {
            final_id: first.candidate_id,
            prediction: first.prediction,
        };
        let before = s.state.session.position();
        let refused = s.decode(&stale, &|| false);
        assert!(
            matches!(&refused, Err(InferenceError::Inference(msg)) if msg.contains("stale prediction id")),
            "{refused:?}"
        );
        assert_eq!(s.state.session.position(), before);
        assert!(
            s.ledger.is_live(second.prediction),
            "the refusal must not consume the live prediction"
        );

        // `finish` ends the live prediction.
        s.finish(FinishDisposition::Reusable).expect("finish");
        assert!(!s.ledger.is_live(second.prediction));
    }

    #[test]
    fn decoder_session_select_refuses_a_grammar_mask_outside_the_dense_mode() {
        let Some(_) = metal::Device::system_default() else {
            return;
        };
        let _gpu = gpu_test_lock();
        let (cfg, weights) = tiny_fixture();
        let mut state =
            MetalQwen35State::new(&weights, &cfg, TINY_CACHE).expect("tiny Metal state");
        let gen_cfg = greedy();
        let history = [1u32, 2, 3];
        let mut s = session(&mut state, &gen_cfg, MetalEntryProfile::Direct, DENSE_ENV);
        assert_eq!(s.mode(), ReadbackMode::GreedyArgmax);
        let mask = |_: &mut [f32]| -> Result<(), InferenceError> { Ok(()) };
        let masked = SelectionRequest {
            config: &gen_cfg,
            history: &history,
            grammar_mask: Some(&mask),
        };

        s.readback = Readback::Argmax(4);
        assert!(matches!(
            s.select(&masked),
            Err(InferenceError::Inference(msg)) if msg.contains("grammar mask")
        ));
        // Control: the same readback without a mask selects the argmax id and
        // draws nothing.
        let rng_before = s.rng_state;
        let c = candidate(s.select(&request(&gen_cfg, &history)).expect("select"));
        assert_eq!(c.candidate_id, 4);
        assert_eq!(s.rng_state, rng_before);
    }

    #[test]
    fn decoder_session_construction_refuses_non_dense_grammar_or_direct_logprobs() {
        let Some(_) = metal::Device::system_default() else {
            return;
        };
        let _gpu = gpu_test_lock();
        let (cfg, weights) = tiny_fixture();
        let mut state =
            MetalQwen35State::new(&weights, &cfg, TINY_CACHE).expect("tiny Metal state");
        // The direct entry's own logprobs refusal, before any mutation.
        let logprobs = GenerateConfig {
            logprobs: Some(1),
            ..greedy()
        };
        let refused = QwenMetalSession::with_route_environment(
            &mut state,
            plan(vec![1, 2, 3], 7),
            &logprobs,
            MetalEntryProfile::Direct,
            COMPACT_ENV,
        );
        assert!(
            matches!(&refused, Err(InferenceError::InvalidInput(msg)) if msg.contains("logprobs")),
            "{:?}",
            refused.as_ref().err()
        );
        drop(refused);
        assert_route_disengaged(&state);

        // Streaming admits the same request and the planner keeps it dense.
        let s = QwenMetalSession::with_route_environment(
            &mut state,
            plan(vec![1, 2, 3], 7),
            &logprobs,
            MetalEntryProfile::Streaming,
            COMPACT_ENV,
        )
        .expect("streaming admits logprobs");
        assert_eq!(s.mode(), ReadbackMode::Dense);
    }

    /// A decode error after the prediction was consumed, injected by filling
    /// the KV cache; the session is then dropped the way a caller drops it
    /// after `driver::run` returns through `?`.
    #[test]
    fn decoder_session_tears_down_the_compact_route_on_a_mid_decode_error() {
        let Some(_) = metal::Device::system_default() else {
            return;
        };
        let _gpu = gpu_test_lock();
        let (cfg, weights) = tiny_fixture();
        let mut state =
            MetalQwen35State::new(&weights, &cfg, TINY_CACHE).expect("tiny Metal state");
        let gen_cfg = greedy();
        let history = [1u32, 2, 3];
        {
            let mut s = session(
                &mut state,
                &gen_cfg,
                MetalEntryProfile::Streaming,
                COMPACT_ENV,
            );
            s.state.session.compact_result = vec![Candidate {
                token_id: 9,
                logit: 1.0,
            }];
            s.readback = Readback::Compact;
            s.prefill_token_pending = true;
            let c = candidate(s.select(&request(&gen_cfg, &history)).expect("select"));

            let full = s.state.max_context();
            s.state.session.set_position(full);
            let failed = s.decode(
                &AcceptedToken {
                    final_id: c.candidate_id,
                    prediction: c.prediction,
                },
                &|| false,
            );
            assert!(
                matches!(&failed, Err(InferenceError::Inference(msg)) if msg.contains("KV cache is full")),
                "{failed:?}"
            );
            assert_eq!(
                s.state.session.compact_topk, 1,
                "control: the route is still engaged until the session is dropped"
            );
        }
        assert_route_disengaged(&state);
    }

    fn run_driver(
        s: &mut QwenMetalSession<'_>,
        gen_cfg: &GenerateConfig,
        prompt_ids: &[u32],
        cancel: &dyn Cancellation,
    ) -> Result<driver::DriverResult, InferenceError> {
        let mut text = String::new();
        let mut offsets = Vec::new();
        driver::run(
            s,
            gen_cfg,
            None,
            prompt_ids,
            (TINY_VOCAB - 1) as u32,
            true,
            cancel,
            |_| {},
            |_| String::new(),
            &mut text,
            &mut offsets,
            |_, _| true,
            || {},
            String::new,
        )
    }

    /// `driver::run` returns early without calling `finish` when cancelled
    /// before prefill, and through `?` when prefill fails. Neither runs a
    /// forward pass here: the cancel fires first, and `try_forward_prefill`
    /// refuses an out-of-vocabulary id before dispatch.
    #[test]
    fn decoder_session_tears_down_the_compact_route_when_the_driver_skips_finish() {
        let Some(_) = metal::Device::system_default() else {
            return;
        };
        let _gpu = gpu_test_lock();
        let (cfg, weights) = tiny_fixture();
        let mut state =
            MetalQwen35State::new(&weights, &cfg, TINY_CACHE).expect("tiny Metal state");
        let gen_cfg = greedy();

        {
            let mut s = session(
                &mut state,
                &gen_cfg,
                MetalEntryProfile::Streaming,
                COMPACT_ENV,
            );
            let result = run_driver(&mut s, &gen_cfg, &[1, 2, 3], &|| true)
                .expect("cancel before prefill is not an error");
            assert_eq!(result.stop_reason, StopReason::Interrupt);
            assert!(result.generated_ids.is_empty());
            assert_eq!(
                s.state.session.compact_topk, 1,
                "control: finish was skipped"
            );
        }
        assert_route_disengaged(&state);

        {
            let out_of_vocab = vec![TINY_VOCAB as u32 + 5];
            let mut s = QwenMetalSession::with_route_environment(
                &mut state,
                plan(out_of_vocab.clone(), 7),
                &gen_cfg,
                MetalEntryProfile::Streaming,
                COMPACT_ENV,
            )
            .expect("session");
            let failed = run_driver(&mut s, &gen_cfg, &out_of_vocab, &|| false);
            assert!(
                matches!(&failed, Err(InferenceError::InvalidInput(_))),
                "{:?}",
                failed.as_ref().err()
            );
            assert_eq!(
                s.state.session.compact_topk, 1,
                "control: finish was skipped"
            );
            assert_eq!(
                s.state.session.position(),
                0,
                "prefill refused before dispatch"
            );
        }
        assert_route_disengaged(&state);
    }

    // -----------------------------------------------------------------
    // `run_streaming` over a scripted session (no Metal device): the three
    // places the streaming entry's contract differs from what the driver
    // reports on its own.
    // -----------------------------------------------------------------

    /// Tokenizer ids: 0 decodes to "a"; 1 decodes to the lone byte 0xE4, an
    /// incomplete UTF-8 sequence the detokenizer holds until its final flush,
    /// which renders it as U+FFFD; 2 decodes to "x".
    fn scripted_tokenizer() -> crate::tokenizer::bpe::BpeTokenizer {
        let vocab = [("a", 0u32), ("\u{e4}", 1), ("x", 2)]
            .into_iter()
            .map(|(token, id)| (token.to_string(), id))
            .collect();
        crate::tokenizer::bpe::BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
            .expect("scripted tokenizer")
    }

    /// A grammar over the scripted ids, where the grammar reads id 0 as "a",
    /// id 1 as `second` and id 2 as "x".
    fn scripted_grammar(
        gbnf: &str,
        second: &[u8],
    ) -> std::sync::Arc<crate::grammar::GrammarEngine> {
        std::sync::Arc::new(
            crate::grammar::GrammarEngine::new(
                &crate::grammar::GrammarSpec::Gbnf(gbnf.into()),
                vec![b"a".to_vec(), second.to_vec(), b"x".to_vec()],
            )
            .expect("scripted grammar compiles"),
        )
    }

    const SCRIPTED_EOS: u32 = 99;
    const SCRIPTED_THINK_CLOSE: u32 = 2;

    /// Offers `script[i]` from the i-th `select` (the last entry repeats).
    /// A grammar mask is applied to a zero row over the three ids first, so a
    /// mask that blocks every id is reported as `GrammarExhausted`, as the
    /// real session reports it.
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
            request: &SelectionRequest<'_>,
        ) -> Result<SelectOutcome, InferenceError> {
            let candidate_id = self.script[self.selects.min(self.script.len() - 1)];
            self.selects += 1;
            if let Some(mask) = request.grammar_mask {
                let mut row = vec![0.0f32; 3];
                mask(&mut row)?;
                if !has_finite_logit(&row) {
                    return Ok(SelectOutcome::GrammarExhausted);
                }
            }
            Ok(SelectOutcome::Candidate(SelectionCandidate {
                candidate_id,
                prediction: self.ledger.open(),
            }))
        }

        fn metadata(
            &mut self,
            _prediction: PredictionId,
            _final_token: u32,
            _request: &MetadataRequest,
        ) -> Result<TokenMetadata, InferenceError> {
            unreachable!("no scripted request sets logprobs")
        }

        fn finish(&mut self, _disposition: FinishDisposition) -> Result<(), InferenceError> {
            Ok(())
        }
    }

    /// Streams `script` and returns the output and every `(text, id)` pair
    /// `on_token` received.
    fn stream_script(
        script: Vec<u32>,
        gen_cfg: &GenerateConfig,
    ) -> (Result<GenerateOutput, InferenceError>, Vec<(String, u32)>) {
        let mut session = ScriptedSession {
            caps: STREAMING_CAPABILITIES,
            ledger: PredictionLedger::new(),
            script,
            selects: 0,
        };
        let tokenizer = scripted_tokenizer();
        let mut calls = Vec::new();
        let output = run_streaming(
            &mut session,
            gen_cfg,
            Some(SCRIPTED_THINK_CLOSE),
            &[0],
            SCRIPTED_EOS,
            &tokenizer,
            |text, id| {
                calls.push((text.to_string(), id));
                true
            },
            || false,
        );
        (output, calls)
    }

    /// The natural-end flush releases the held 0xE4 byte as U+FFFD. It must
    /// reach `on_token` with the id of the last emitted token (1), not the
    /// placeholder id the driver hands the flush.
    #[test]
    fn streaming_tail_flush_reports_the_last_emitted_token_id() {
        let gen_cfg = GenerateConfig {
            max_new_tokens: 2,
            ..Default::default()
        };
        let (output, calls) = stream_script(vec![0, 1], &gen_cfg);
        let output = output.expect("an unconstrained scripted stream runs to its cap");
        assert_eq!(output.token_ids, vec![0, 1]);
        assert_eq!(output.stop_reason, Some(StopReason::Length));
        assert!(!output.stopped);
        assert_eq!(
            calls,
            vec![("a".to_string(), 0), ("\u{fffd}".to_string(), 1)],
            "fixture shape: one decode-step delta, then the flushed tail"
        );
        assert_eq!(output.text, "a\u{fffd}");
    }

    /// Budget forcing replaces the second token with the close id, which the
    /// grammar (`"b" "b"`, id 1 read as "b") rejects: the request ends with
    /// `StopReason::Grammar` and `stopped: false`. The flushed tail still
    /// reaches `on_token`, and completes no stop string.
    #[test]
    fn streaming_grammar_rejection_reports_not_stopped() {
        let gen_cfg = GenerateConfig {
            max_new_tokens: 4,
            enable_thinking: true,
            reasoning_budget: Some(1),
            grammar: Some(scripted_grammar("root ::= \"b\" \"b\"\n", b"b")),
            ..Default::default()
        };
        let (output, calls) = stream_script(vec![1], &gen_cfg);
        let output = output.expect("a grammar rejection is not an error");
        assert_eq!(
            output.token_ids,
            vec![1],
            "the forced close id is rejected before it is pushed"
        );
        assert_eq!(output.stop_reason, Some(StopReason::Grammar));
        assert!(
            !output.stopped,
            "a grammar rejection is not a stop condition"
        );
        assert_eq!(calls, vec![("\u{fffd}".to_string(), 1)]);
    }

    /// The same rejection, with a stop string the flushed tail completes: the
    /// request reports the stop string (`StopReason::Eos`, `stopped: true`),
    /// and the matched text never reaches `on_token`.
    #[test]
    fn streaming_grammar_rejection_then_flushed_stop_string_reports_eos() {
        let gen_cfg = GenerateConfig {
            max_new_tokens: 4,
            enable_thinking: true,
            reasoning_budget: Some(1),
            grammar: Some(scripted_grammar("root ::= \"b\" \"b\"\n", b"b")),
            stop_strings: vec!["\u{fffd}".to_string()],
            ..Default::default()
        };
        let (output, calls) = stream_script(vec![1], &gen_cfg);
        let output = output.expect("a grammar rejection is not an error");
        assert_eq!(output.token_ids, vec![1]);
        assert_eq!(output.stop_reason, Some(StopReason::Eos));
        assert!(output.stopped);
        assert!(calls.is_empty(), "the stop string is withheld: {calls:?}");
        assert_eq!(output.text, "");
    }

    /// A grammar that blocks every id before the first token keeps the
    /// entry's step-0 message; a dead end after the first token keeps the
    /// decode-step message (control: the step-0 message is not applied to it).
    #[test]
    fn streaming_grammar_block_messages_distinguish_step_zero() {
        let blocked_at_start = GenerateConfig {
            max_new_tokens: 4,
            grammar: Some(scripted_grammar("root ::= \"b\"\n", b"a")),
            ..Default::default()
        };
        let (output, calls) = stream_script(vec![1], &blocked_at_start);
        match output {
            Err(InferenceError::GrammarConstraintBlocked(message)) => {
                assert_eq!(message, STEP_ZERO_GRAMMAR_BLOCKED);
            }
            other => panic!("expected the step-0 grammar block, got {other:?}"),
        }
        assert!(calls.is_empty());

        let dead_end = GenerateConfig {
            max_new_tokens: 4,
            grammar: Some(scripted_grammar("root ::= \"b\" \"c\"\n", b"b")),
            ..Default::default()
        };
        let (output, calls) = stream_script(vec![1], &dead_end);
        match output {
            Err(InferenceError::GrammarConstraintBlocked(message)) => {
                assert!(
                    message.contains("no legal continuation"),
                    "a decode-step dead end keeps its own message, got {message:?}"
                );
            }
            other => panic!("expected the decode-step grammar block, got {other:?}"),
        }
        assert!(calls.is_empty(), "the held 0xE4 byte is never flushed");
    }

    // -----------------------------------------------------------------
    // Real-checkpoint parity: the session driven by `driver::run` against
    // the direct entry's own loop, on the same state. The streaming entry
    // itself runs through this session, so a streaming comparison here would
    // compare the driver with itself; the Metal generation golden replay pins
    // that entry instead. Runs only when the Metal generation golden
    // checkpoint is configured; skips otherwise unless the golden enforce
    // switch is on.
    // -----------------------------------------------------------------

    const MODEL_DIR_VAR: &str = "LATTICE_METAL_GENERATION_MODEL_DIR";
    const ENFORCE_VAR: &str = "LATTICE_METAL_GENERATION_GOLDEN_ENFORCE";
    const SKIP_MARKER: &str = "LATTICE_METAL_GENERATION_GOLDEN_SKIPPED";
    const PARITY_CACHE_LEN: usize = 2048;
    const PLAIN_PROMPT: &str = "The capital of France is";
    const GRAMMAR_PROMPT: &str = "<|im_start|>user\nReply with a JSON object whose key \"answer\" holds the number 42.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const GRAMMAR_SCHEMA: &str =
        r#"{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"]}"#;

    /// Environment switches that select a decode route, pinned for the whole
    /// parity run so an ambient value cannot move a case onto another route.
    const ROUTE_SWITCHES: &[(&str, &str)] = &[
        ("LATTICE_METAL_PATH_PROOF", "1"),
        ("LATTICE_COMPACT_TOPK", "0"),
        ("LATTICE_COMPACT_TOPK_SELECT", "0"),
        ("LATTICE_COMPACT_TOPP_APPROX", "0"),
        ("LATTICE_SELF_SPEC", "0"),
        ("LATTICE_MTP", "0"),
    ];

    static ROUTE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Pins [`ROUTE_SWITCHES`] and restores the prior values on drop. Writers
    /// are serialized by `ROUTE_ENV_LOCK`, held for the guard's life.
    struct RouteEnvironment {
        prior: Vec<(&'static str, Option<std::ffi::OsString>)>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl RouteEnvironment {
        fn pin() -> Self {
            let lock = ROUTE_ENV_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let prior = ROUTE_SWITCHES
                .iter()
                .map(|(name, _)| (*name, std::env::var_os(name)))
                .collect();
            for (name, value) in ROUTE_SWITCHES {
                // SAFETY: writers are serialized by ROUTE_ENV_LOCK, held by the guard.
                unsafe { std::env::set_var(name, value) };
            }
            Self { prior, _lock: lock }
        }

        fn set_compact(&self, on: bool) {
            // SAFETY: writers are serialized by ROUTE_ENV_LOCK, held by `self`.
            unsafe { std::env::set_var("LATTICE_COMPACT_TOPK", if on { "1" } else { "0" }) };
        }
    }

    impl Drop for RouteEnvironment {
        fn drop(&mut self) {
            for (name, value) in &self.prior {
                // SAFETY: writers are serialized by ROUTE_ENV_LOCK, still held here.
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }
    }

    /// Returns the checkpoint directory, or `None` after printing the skip
    /// marker. A missing checkpoint under the enforce switch, and a relative
    /// path always, fail the test.
    fn parity_checkpoint(test: &str) -> Option<std::path::PathBuf> {
        let enforce = crate::env_switch_enabled(ENFORCE_VAR);
        let skip = |reason: String| {
            assert!(!enforce, "{reason}, and {ENFORCE_VAR} is enabled");
            eprintln!("{SKIP_MARKER} test={test} reason={reason}");
            None
        };
        let Some(raw) = std::env::var_os(MODEL_DIR_VAR) else {
            return skip(format!("{MODEL_DIR_VAR} is unset"));
        };
        let path = std::path::PathBuf::from(&raw);
        assert!(
            path.is_absolute(),
            "{MODEL_DIR_VAR}={raw:?} is relative; cargo test runs test binaries with the crate \
             directory as CWD. Pass an absolute path."
        );
        if !path.exists() {
            return skip(format!("{MODEL_DIR_VAR}={raw:?} does not exist"));
        }
        if metal::Device::system_default().is_none() {
            return skip("no Metal device".to_string());
        }
        Some(path)
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ExpectedMode {
        Dense,
        Compact,
        GreedyArgmax,
    }

    struct ParityCase {
        name: &'static str,
        prompt: &'static str,
        compact_env: bool,
        /// `(temperature, top_k, top_p, seed)`.
        sampler: (f32, usize, f32, u64),
        max_new_tokens: usize,
        stop_strings: &'static [&'static str],
        grammar: bool,
        mode: ExpectedMode,
    }

    const GREEDY: (f32, usize, f32, u64) = (0.0, 1, 1.0, 1);
    const SAMPLED_DENSE: (f32, usize, f32, u64) = (0.8, 40, 0.9, 0x5EED_0001);
    const SAMPLED_BLOCK_TOPK: (f32, usize, f32, u64) = (0.8, 40, 1.0, 0x5EED_0002);

    const BASE_CASE: ParityCase = ParityCase {
        name: "",
        prompt: PLAIN_PROMPT,
        compact_env: false,
        sampler: GREEDY,
        max_new_tokens: 16,
        stop_strings: &[],
        grammar: false,
        mode: ExpectedMode::Dense,
    };

    const PARITY_CASES: &[ParityCase] = &[
        ParityCase {
            name: "direct_greedy_argmax",
            mode: ExpectedMode::GreedyArgmax,
            ..BASE_CASE
        },
        ParityCase {
            name: "direct_greedy_compact",
            compact_env: true,
            mode: ExpectedMode::Compact,
            ..BASE_CASE
        },
        ParityCase {
            name: "direct_sampled_dense",
            sampler: SAMPLED_DENSE,
            ..BASE_CASE
        },
        ParityCase {
            name: "direct_sampled_compact",
            compact_env: true,
            sampler: SAMPLED_BLOCK_TOPK,
            mode: ExpectedMode::Compact,
            ..BASE_CASE
        },
        ParityCase {
            name: "direct_grammar",
            prompt: GRAMMAR_PROMPT,
            compact_env: true,
            max_new_tokens: 48,
            grammar: true,
            ..BASE_CASE
        },
        ParityCase {
            name: "direct_stop_string",
            max_new_tokens: 48,
            stop_strings: &[".\n"],
            mode: ExpectedMode::GreedyArgmax,
            ..BASE_CASE
        },
    ];

    fn parity_config(
        case: &ParityCase,
        grammar: &std::sync::Arc<crate::grammar::GrammarEngine>,
    ) -> GenerateConfig {
        let (temperature, top_k, top_p, seed) = case.sampler;
        GenerateConfig {
            max_new_tokens: case.max_new_tokens,
            temperature,
            top_k,
            top_p,
            seed: Some(seed),
            min_p: 0.0,
            repetition_penalty: 1.0,
            enable_mtp: Some(false),
            grammar: case.grammar.then(|| std::sync::Arc::clone(grammar)),
            stop_strings: case.stop_strings.iter().map(|s| (*s).to_string()).collect(),
            ..Default::default()
        }
    }

    /// What both paths are compared on: the id stream, the stop disposition,
    /// the logprob ids, and the readback counters the path proof recorded.
    #[derive(Debug, PartialEq)]
    struct ParityRecord {
        token_ids: Vec<u32>,
        stop_reason: StopReason,
        stopped: bool,
        logprob_ids: Vec<u32>,
        top_logprob_ids: Vec<Vec<u32>>,
        logit_readback: String,
        hidden_readback: String,
    }

    fn parity_record(
        state: &MetalQwen35State,
        token_ids: Vec<u32>,
        stop_reason: StopReason,
        stopped: bool,
        token_logprobs: &[crate::generation::TokenLogprob],
    ) -> ParityRecord {
        ParityRecord {
            token_ids,
            stop_reason,
            stopped,
            logprob_ids: token_logprobs.iter().map(|t| t.token_id).collect(),
            top_logprob_ids: token_logprobs
                .iter()
                .map(|t| t.top.iter().map(|alt| alt.token_id).collect())
                .collect(),
            logit_readback: format!("{:?}", state.logit_readback_path_proof_snapshot()),
            hidden_readback: format!("{:?}", state.hidden_readback_path_proof_snapshot()),
        }
    }

    fn run_legacy(
        state: &mut MetalQwen35State,
        tokenizer: &crate::tokenizer::bpe::BpeTokenizer,
        case: &ParityCase,
        gen_cfg: &GenerateConfig,
    ) -> ParityRecord {
        state.reset_path_proof_counters();
        let output = state
            .generate(case.prompt, tokenizer, gen_cfg)
            .unwrap_or_else(|error| panic!("{}: legacy entry failed: {error}", case.name));
        let stop_reason = output
            .stop_reason
            .unwrap_or_else(|| panic!("{}: legacy entry reported no stop reason", case.name));
        parity_record(
            state,
            output.token_ids,
            stop_reason,
            output.stopped,
            &output.token_logprobs,
        )
    }

    fn run_session(
        state: &mut MetalQwen35State,
        tokenizer: &crate::tokenizer::bpe::BpeTokenizer,
        case: &ParityCase,
        gen_cfg: &GenerateConfig,
    ) -> ParityRecord {
        use crate::model::qwen35::{
            GenerationEntryContract, GenerationPreparation, prepare_generation,
            resolve_reasoning_close_token,
        };
        use crate::tokenizer::detokenize::IncrementalDetokenizer;

        state.reset_path_proof_counters();
        let vocab_size = state.engine.config.vocab_size;
        let eos_token_id = state.engine.config.eos_token_id;
        let plan = match prepare_generation(
            tokenizer,
            case.prompt,
            gen_cfg,
            vocab_size,
            state.max_context(),
            GenerationEntryContract::MetalDirect,
        )
        .unwrap_or_else(|error| panic!("{}: preparation failed: {error}", case.name))
        {
            GenerationPreparation::Ready(plan) => plan,
            GenerationPreparation::Complete(_) => {
                panic!("{}: preparation completed without decoding", case.name)
            }
        };
        let prompt_ids = plan.prompt_ids.clone();
        let think_close_id = resolve_reasoning_close_token(
            tokenizer,
            gen_cfg.reasoning_budget,
            gen_cfg.enable_thinking,
            vocab_size,
        )
        .unwrap_or_else(|error| panic!("{}: reasoning close token: {error}", case.name));

        let cancel = || false;
        let detok = std::cell::RefCell::new(IncrementalDetokenizer::new());
        let mut text = String::new();
        let mut offsets = Vec::new();

        let mut session = QwenMetalSession::new(state, plan, gen_cfg, MetalEntryProfile::Direct)
            .unwrap_or_else(|error| panic!("{}: session construction failed: {error}", case.name));
        let mode = session.mode();
        let result = driver::run(
            &mut session,
            gen_cfg,
            think_close_id,
            &prompt_ids,
            eos_token_id,
            false,
            &cancel,
            |_| {},
            |id| detok.borrow_mut().push(tokenizer, id),
            &mut text,
            &mut offsets,
            |_, _| true,
            || {},
            || detok.borrow_mut().finish(),
        )
        .unwrap_or_else(|error| panic!("{}: driver failed: {error}", case.name));
        drop(session);

        let observed = match mode {
            ReadbackMode::Dense => ExpectedMode::Dense,
            ReadbackMode::Compact { .. } => ExpectedMode::Compact,
            ReadbackMode::GreedyArgmax => ExpectedMode::GreedyArgmax,
        };
        assert_eq!(observed, case.mode, "{}: readback mode", case.name);
        assert_eq!(
            state.session.compact_topk, 0,
            "{}: route torn down",
            case.name
        );
        parity_record(
            state,
            result.generated_ids,
            result.stop_reason,
            result.stopped,
            &result.token_logprobs,
        )
    }

    #[test]
    fn decoder_session_matches_the_legacy_metal_entries_on_a_real_checkpoint() {
        use crate::model_format::{ModelFormat, detect_format};

        let test = "decoder_session_matches_the_legacy_metal_entries_on_a_real_checkpoint";
        let Some(model_dir) = parity_checkpoint(test) else {
            return;
        };
        let _gpu = gpu_test_lock();
        let env = RouteEnvironment::pin();

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = crate::tokenizer::bpe::BpeTokenizer::from_tokenizer_json(&tokenizer_path)
            .expect("checkpoint tokenizer");
        let (mut state, cfg) = match detect_format(&model_dir) {
            ModelFormat::Q4 => {
                let cfg = Qwen35Config::from_model_dir(&model_dir).expect("config.json");
                let state = MetalQwen35State::from_q4_dir(
                    &model_dir,
                    &tokenizer_path,
                    &cfg,
                    PARITY_CACHE_LEN,
                )
                .expect("Q4 Metal state");
                (state, cfg)
            }
            ModelFormat::Safetensors => {
                let model = crate::model::qwen35::Qwen35Model::from_safetensors(&model_dir)
                    .expect("safetensors model");
                let cfg = model.config().clone();
                let state = MetalQwen35State::new(model.weights(), &cfg, PARITY_CACHE_LEN)
                    .expect("safetensors Metal state");
                (state, cfg)
            }
            other => panic!(
                "{} holds no Qwen3.5 checkpoint: {other:?}",
                model_dir.display()
            ),
        };

        let spec =
            crate::grammar::GrammarSpec::json_schema_str(GRAMMAR_SCHEMA).expect("grammar schema");
        let vocab_bytes = tokenizer
            .vocab_bytes(cfg.vocab_size)
            .expect("vocabulary bytes");
        let grammar = std::sync::Arc::new(
            crate::grammar::GrammarEngine::new(&spec, vocab_bytes).expect("grammar engine"),
        );

        for case in PARITY_CASES {
            env.set_compact(case.compact_env);
            let gen_cfg = parity_config(case, &grammar);
            let legacy = run_legacy(&mut state, &tokenizer, case, &gen_cfg);
            let session = run_session(&mut state, &tokenizer, case, &gen_cfg);
            assert_eq!(
                session, legacy,
                "{}: session differs from the legacy entry",
                case.name
            );
            assert!(
                !legacy.token_ids.is_empty(),
                "{}: generated nothing",
                case.name
            );
            eprintln!(
                "decoder session parity {}: {} tokens, {:?}",
                case.name,
                legacy.token_ids.len(),
                legacy.stop_reason
            );
        }
    }
}
