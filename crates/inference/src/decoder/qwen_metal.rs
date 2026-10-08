//! `QwenMetalSession`: the ordinary Qwen3.5 Metal generation path behind the
//! object-safe [`super::DecoderSession`] boundary (ADR-090 D1).
//!
//! Wraps the work `MetalQwen35State::generate` (the direct entry) and
//! `MetalQwen35State::generate_streaming_with_cancel` (the streaming entry)
//! each did in their own decode loop. The streaming entry runs its requests
//! through this session and [`run_streaming`]; the direct entry runs its
//! ordinary requests through it and [`run_direct`]. The direct entry's MTP and
//! GDN-first self-speculative routes are refused by this session and run through
//! [`QwenMetalSpeculativeSession`] and [`run_speculative_direct`] instead; the
//! batch-GEMM MTP verifier keeps its own legacy loop.
//! [`MetalEntryProfile::PrefixCacheStreaming`] is the prefix-cache entry's
//! session: [`QwenMetalSession::over_restored_state`] builds it over a state the
//! caller restored to a reusable boundary, and it prefills only the suffix.
//!
//! **Owned state.** The session borrows the caller's `MetalQwen35State`
//! mutably for its whole life, so the GPU caches, scratch and the compact
//! route it engages cannot be observed by anyone else until the session is
//! dropped. It owns the prompt ids, the RNG state, the prediction ledger and
//! the most recent logit readback.
//!
//! **Readback mode, fixed at construction.** Which readback a forward pass
//! produces is decided once, in [`QwenMetalSession::new`], by
//! `plan_sampling_route`, which the direct entry also calls to choose between
//! its speculative routes and this session, plus the direct entry's greedy
//! zero-copy predicate:
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
//! `prepare_generation` derives from `GenerateConfig::seed`, and drawn only
//! inside `select`, with a per-mode schedule: the prefill-derived token uses
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

use super::driver::{self, DriverTrace};
use super::qwen_cpu::has_finite_logit;
use super::{
    AcceptedToken, Cancellation, DecoderSession, ExecutionCapabilities, FinishDisposition,
    MetadataRequest, PredictionError, PredictionId, PredictionLedger, SelectOutcome,
    SelectionCandidate, SelectionRequest, StepStamp, TokenMetadata,
};
use super::{SpeculativeSession, SpeculativeTrace, VerifiedRound};
use crate::error::InferenceError;
use crate::forward::metal_qwen35::{
    GpuTopkRoute, MetalQwen35State, SamplingRouteEnvironment, SamplingRoutePlan,
    SpeculativeMetrics, apply_sampling_route_plan, mtp_route_active, plan_sampling_route,
    sample_decode_traced, sample_from_candidates, sample_token, self_spec_route_active,
};
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::model::qwen35::stop_strings::earliest_stop_match;
use crate::model::qwen35::{
    GenerationPlan, check_logprobs_not_set, check_reasoning_budget_not_set,
};
use crate::sampling::compute_step_logprobs;
use crate::stop_reason::StopReason;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::detokenize::{IncrementalDetokenizer, decode_tokens};
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
    /// `MetalQwen35State::generate_streaming_with_prefix_cache_and_cancel`:
    /// the streaming loop over a state the caller restored to a reusable prefix
    /// boundary (or reset, for a full refill), prefilling only the suffix.
    PrefixCacheStreaming,
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

/// The prefix-cache entry refuses `logprobs` and `enable_mtp`
/// (`GenerationEntryContract::MetalPrefixCacheStreaming`) and wires grammar
/// masking, stop strings and the reasoning budget like the streaming entry.
const PREFIX_CACHE_CAPABILITIES: ExecutionCapabilities = ExecutionCapabilities {
    grammar: true,
    logprobs: false,
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
    profile: MetalEntryProfile,
    capabilities: ExecutionCapabilities,
    mode: ReadbackMode,
    prompt_ids: Vec<u32>,
    /// Where the suffix the prefix-cache profile prefills begins; 0 otherwise.
    suffix_start: usize,
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

    /// [`Self::new`] with the route environment the caller already read, so a
    /// caller that planned the route to choose this session plans the same one.
    ///
    /// The prefix-cache profile built here is a full refill: it resets the state
    /// and prefills the whole prompt through `forward_prefill_from` at position 0.
    pub(crate) fn with_route_environment(
        state: &'state mut MetalQwen35State,
        plan: GenerationPlan,
        gen_cfg: &GenerateConfig,
        profile: MetalEntryProfile,
        environment: SamplingRouteEnvironment,
    ) -> Result<Self, InferenceError> {
        Self::construct(state, plan, 0, gen_cfg, profile, environment, true)
    }

    /// The prefix-cache profile over a `state` the caller already restored to the
    /// reusable boundary `suffix_start` (its KV cursor and recurrent state), which
    /// this constructor leaves untouched. Prefill runs only
    /// `plan.prompt_ids[suffix_start..]`, at its absolute position. A caller whose
    /// plan is a full refill resets the state itself and passes 0.
    ///
    /// # Errors
    ///
    /// Before any state mutation: the prefix-cache entry's `logprobs` refusal, a
    /// non-dense grammar request as in [`Self::with_route_environment`], and a
    /// `suffix_start` beyond the prompt.
    pub(crate) fn over_restored_state(
        state: &'state mut MetalQwen35State,
        plan: GenerationPlan,
        suffix_start: usize,
        gen_cfg: &GenerateConfig,
        environment: SamplingRouteEnvironment,
    ) -> Result<Self, InferenceError> {
        Self::construct(
            state,
            plan,
            suffix_start,
            gen_cfg,
            MetalEntryProfile::PrefixCacheStreaming,
            environment,
            false,
        )
    }

    fn construct(
        state: &'state mut MetalQwen35State,
        plan: GenerationPlan,
        suffix_start: usize,
        gen_cfg: &GenerateConfig,
        profile: MetalEntryProfile,
        environment: SamplingRouteEnvironment,
        reset_state: bool,
    ) -> Result<Self, InferenceError> {
        match profile {
            MetalEntryProfile::Direct => {
                check_reasoning_budget_not_set(gen_cfg)?;
                check_logprobs_not_set(gen_cfg)?;
            }
            MetalEntryProfile::PrefixCacheStreaming => check_logprobs_not_set(gen_cfg)?,
            MetalEntryProfile::Streaming => {}
        }
        if suffix_start > plan.prompt_ids.len() {
            return Err(InferenceError::InvalidInput(format!(
                "suffix_start {suffix_start} is beyond the {} prompt tokens",
                plan.prompt_ids.len()
            )));
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

        if reset_state {
            state.reset_state();
        }
        let route_engaged = apply_sampling_route_plan(
            route,
            &mut state.session.compact_route,
            &mut state.session.compact_topk,
            &mut state.session.compact_result,
        );

        Ok(Self {
            state,
            profile,
            capabilities: match profile {
                MetalEntryProfile::Direct => DIRECT_CAPABILITIES,
                MetalEntryProfile::Streaming => STREAMING_CAPABILITIES,
                MetalEntryProfile::PrefixCacheStreaming => PREFIX_CACHE_CAPABILITIES,
            },
            mode,
            prompt_ids: plan.prompt_ids,
            suffix_start,
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
    #[cfg(test)]
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

    /// Batched prefill through `try_forward_prefill`, the fallible entry the
    /// direct and streaming loops call: it refuses before any state mutation (a
    /// non-fresh session, an out-of-vocabulary id, a range beyond capacity, a
    /// multi-token prompt on an MoE model without LoRA), so an error here
    /// leaves nothing to undo except the route, which `Drop` tears down. The
    /// prefix-cache profile prefills its suffix through `forward_prefill_from`
    /// at the restored boundary, as the prefix-cache loop does, because
    /// `try_forward_prefill` refuses any session that is not fresh.
    fn prefill(&mut self, cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        if cancel.is_cancelled() {
            return Err(InferenceError::Inference("cancelled before prefill".into()));
        }
        self.ledger.reset();
        let logits = if self.profile == MetalEntryProfile::PrefixCacheStreaming {
            self.state.forward_prefill_from(
                &self.prompt_ids[self.suffix_start..],
                self.suffix_start,
                false,
            )?
        } else {
            self.state.try_forward_prefill(&self.prompt_ids)?
        };
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
    /// `compute_step_logprobs` the CPU sessions' `metadata` also call; the
    /// driver reaches it through `DecodePolicy::transition_with_metadata`. Only
    /// the dense readback has a full distribution to score.
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

/// The error both entries return when the grammar blocks every token before
/// the first one is emitted.
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

/// How a request that ended with [`StopReason::Interrupt`] was cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InterruptSource {
    /// The cancellation poll returned true.
    Cancel,
    /// `on_token` refused text; `tail` is set when it refused the natural-end flush.
    Delivery { tail: bool },
}

/// What [`stream_through_driver`] reports beyond the entry's output.
struct StreamRun {
    output: GenerateOutput,
    trace: DriverTrace,
    /// Set exactly when `output.stop_reason` is `Interrupt`.
    interrupt: Option<InterruptSource>,
    confirmed_stop_string_match: bool,
    /// The natural-end flush completed a stop string. Meaningless, and false,
    /// when the flush did not run.
    flush_completed_stop_string: bool,
    /// A sampled or budget-forced token the grammar rejected ended the request,
    /// as the driver reported it (before the entry's `stopped` adaptation).
    grammar_rejection: bool,
    prefill_ran: bool,
}

/// Runs one streaming request through [`driver::run`] over `session` and
/// adapts the driver's result to the streaming entries' shared contract.
///
/// `should_cancel` is polled at the driver's three checkpoints: before
/// prefill, right after it, and at the top of every decode iteration. Text
/// reaches `on_token` through an incremental detokenizer and the policy's
/// stop-string matcher, as it did in the entries' own loops.
///
/// Three parts of the contract differ from what the driver reports, and this
/// function keeps the entries':
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
///    the entries' step-0 message.
#[allow(clippy::too_many_arguments)]
fn stream_through_driver(
    session: &mut dyn DecoderSession,
    gen_cfg: &GenerateConfig,
    think_close_id: Option<u32>,
    prompt_ids: &[u32],
    eos_token_id: u32,
    tokenizer: &BpeTokenizer,
    mut on_token: impl FnMut(&str, u32) -> bool,
    mut should_cancel: impl FnMut() -> bool,
) -> Result<StreamRun, InferenceError> {
    let cancel_fired = Cell::new(false);
    let should_cancel = RefCell::new(|| {
        let cancelled = should_cancel();
        if cancelled {
            cancel_fired.set(true);
        }
        cancelled
    });
    let cancel = FnMutCancellation(&should_cancel);
    let detok = RefCell::new(IncrementalDetokenizer::new());
    let last_pushed: Cell<Option<u32>> = Cell::new(None);
    let decoded_len = Cell::new(0usize);
    let flushing_tail = Cell::new(false);
    let delivery_refused: Cell<Option<bool>> = Cell::new(None);
    let prefill_ran = Cell::new(false);
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
            let accepted = on_token(delta, id);
            if !accepted {
                delivery_refused.set(Some(flushing_tail.get()));
            }
            accepted
        },
        || prefill_ran.set(true),
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

    let flush_completed_stop_string = result.stop_reason != StopReason::Interrupt
        && !result.confirmed_stop_string_match
        && text.len() < decoded_len.get();
    let grammar_rejection = result.stop_reason == StopReason::Grammar
        && result.trace.opened == result.generated_ids.len() + 1;
    let interrupt = (result.stop_reason == StopReason::Interrupt).then(|| {
        if cancel_fired.get() {
            InterruptSource::Cancel
        } else {
            InterruptSource::Delivery {
                tail: delivery_refused.get().unwrap_or(false),
            }
        }
    });

    let mut stopped = result.stopped;
    let mut stop_reason = result.stop_reason;
    if grammar_rejection {
        stopped = flush_completed_stop_string;
        if flush_completed_stop_string {
            stop_reason = StopReason::Eos;
        }
    }

    Ok(StreamRun {
        output: GenerateOutput {
            text,
            prompt_tokens: prompt_ids.len(),
            generated_tokens: result.generated_ids.len(),
            token_ids: result.generated_ids,
            stopped,
            stop_reason: Some(stop_reason),
            token_logprobs: result.token_logprobs,
        },
        trace: result.trace,
        interrupt,
        confirmed_stop_string_match: result.confirmed_stop_string_match,
        flush_completed_stop_string,
        grammar_rejection,
        prefill_ran: prefill_ran.get(),
    })
}

/// Runs one `MetalQwen35State::generate_streaming_with_cancel` request through
/// [`driver::run`] over `session` and returns that entry's output; see
/// [`stream_through_driver`] for the parts of its contract the driver does not
/// report.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_streaming(
    session: &mut dyn DecoderSession,
    gen_cfg: &GenerateConfig,
    think_close_id: Option<u32>,
    prompt_ids: &[u32],
    eos_token_id: u32,
    tokenizer: &BpeTokenizer,
    on_token: impl FnMut(&str, u32) -> bool,
    should_cancel: impl FnMut() -> bool,
) -> Result<GenerateOutput, InferenceError> {
    Ok(stream_through_driver(
        session,
        gen_cfg,
        think_close_id,
        prompt_ids,
        eos_token_id,
        tokenizer,
        on_token,
        should_cancel,
    )?
    .output)
}

/// What the cross-turn slot holds after a prefix-cache request, decided by the
/// path the request ended on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefixCommit {
    /// The slot stays empty: the restore already consumed the warm entry.
    Leave,
    /// Save the boundary live state represents. With `silent_step`, first
    /// forward the last pushed token so the next turn can reuse through it.
    Save { silent_step: bool },
}

/// A prefix-cache request's output, the driver trace that proves which route
/// ran it, and what to do with the cross-turn slot.
pub(crate) struct PrefixStreamRun {
    pub(crate) output: GenerateOutput,
    pub(crate) trace: DriverTrace,
    /// The suffix prefill ran; false for a cancel before it.
    pub(crate) prefill_ran: bool,
    pub(crate) commit: PrefixCommit,
}

/// Runs one prefix-cache streaming request through [`driver::run`] over a
/// session built with [`QwenMetalSession::over_restored_state`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_prefix_cache_streaming(
    session: &mut dyn DecoderSession,
    gen_cfg: &GenerateConfig,
    think_close_id: Option<u32>,
    prompt_ids: &[u32],
    eos_token_id: u32,
    tokenizer: &BpeTokenizer,
    on_token: impl FnMut(&str, u32) -> bool,
    should_cancel: impl FnMut() -> bool,
) -> Result<PrefixStreamRun, InferenceError> {
    let run = stream_through_driver(
        session,
        gen_cfg,
        think_close_id,
        prompt_ids,
        eos_token_id,
        tokenizer,
        on_token,
        should_cancel,
    )?;
    let commit = prefix_commit(&run);
    Ok(PrefixStreamRun {
        output: run.output,
        trace: run.trace,
        prefill_ran: run.prefill_ran,
        commit,
    })
}

/// The cross-turn slot rule of the prefix-cache entry, path by path. The
/// request's exit is read from what the shared stream core recorded, because
/// the entry's rules separate exits the driver reports alike:
///
/// - A cancel before the first token, a first token the caller refused, and a
///   refused natural-end flush leave the slot empty. A cancel at a decode-loop
///   top and a refused later token save the forwarded prefix, with no silent
///   step: the last pushed token was not forwarded.
/// - A confirmed stop-string match, and a flush that completes one however the
///   loop ended, leave the slot empty: the saved tokens would represent text
///   the caller never received.
/// - A grammar that blocks every token before the first, and is complete,
///   leaves the slot empty.
/// - Every other exit saves. The silent step runs only when the last pushed
///   token was not forwarded and no stop token, completed grammar or grammar
///   rejection ended the request.
fn prefix_commit(run: &StreamRun) -> PrefixCommit {
    let generated = run.output.generated_tokens;
    let reason = run.output.stop_reason;
    if reason == Some(StopReason::Interrupt) {
        return match run.interrupt {
            Some(InterruptSource::Cancel) if generated == 0 => PrefixCommit::Leave,
            Some(InterruptSource::Delivery { tail: true }) => PrefixCommit::Leave,
            Some(InterruptSource::Delivery { tail: false }) if run.trace.consumed == 0 => {
                PrefixCommit::Leave
            }
            _ => PrefixCommit::Save { silent_step: false },
        };
    }
    if run.confirmed_stop_string_match || run.flush_completed_stop_string {
        return PrefixCommit::Leave;
    }
    if reason == Some(StopReason::Grammar) && generated == 0 && run.trace.opened == 0 {
        return PrefixCommit::Leave;
    }
    PrefixCommit::Save {
        silent_step: generated > 0 && !run.output.stopped && !run.grammar_rejection,
    }
}

/// Runs one `MetalQwen35State::generate` request through [`driver::run`] over
/// `session` and returns that entry's output.
///
/// **Stop strings** use the driver's non-streaming full-scan mode, the mode the
/// CPU direct entry runs them in: each decoded delta is appended to the text and
/// scanned from the earliest byte a new match could start at, and a match
/// truncates the text and stops the request with `StopReason::Eos`. That is the
/// text the entry's own matcher accumulated. The natural-end flush then appends
/// the bytes the detokenizer held back and scans the whole text once more, as
/// the matcher's `finish` did: a stop string those bytes complete truncates the
/// text and leaves the stop disposition as the loop reported it. With no stop
/// strings the text is decoded from the generated ids in one pass.
///
/// Two parts of the entry's contract differ from what the driver reports, and
/// this function keeps the entry's:
///
/// 1. A sampled token the grammar rejects ends the request with
///    `stopped: false` and `StopReason::Grammar`; the driver reports
///    `stopped: true`. A rejection is the one grammar stop that leaves the last
///    opened prediction unpushed, so it is read from the trace.
/// 2. A grammar that blocks every token before the first one is emitted keeps
///    the entry's step-0 message.
pub(crate) fn run_direct(
    session: &mut dyn DecoderSession,
    gen_cfg: &GenerateConfig,
    prompt_ids: &[u32],
    eos_token_id: u32,
    tokenizer: &BpeTokenizer,
) -> Result<GenerateOutput, InferenceError> {
    let scan_stop_strings = !gen_cfg.stop_strings.is_empty();
    let mut detok = IncrementalDetokenizer::new();
    let pushed = Cell::new(0usize);
    let mut text = String::new();
    let mut token_logprob_end_offsets: Vec<usize> = Vec::new();
    let never_cancel = || false;

    let result = driver::run(
        session,
        gen_cfg,
        None,
        prompt_ids,
        eos_token_id,
        false,
        &never_cancel,
        |generated_len| pushed.set(generated_len),
        |next_id| {
            if scan_stop_strings {
                detok.push(tokenizer, next_id)
            } else {
                String::new()
            }
        },
        &mut text,
        &mut token_logprob_end_offsets,
        |_, _| true,
        || {},
        String::new,
    );
    let result = match result {
        Err(InferenceError::GrammarConstraintBlocked(_)) if pushed.get() == 0 => {
            return Err(InferenceError::GrammarConstraintBlocked(
                STEP_ZERO_GRAMMAR_BLOCKED.into(),
            ));
        }
        other => other?,
    };

    let text = if scan_stop_strings {
        if !result.confirmed_stop_string_match {
            let tail = detok.finish();
            text.push_str(&tail);
            if let Some(hit) = earliest_stop_match(&text, &gen_cfg.stop_strings) {
                text.truncate(hit);
            }
        }
        text
    } else {
        decode_tokens(tokenizer, &result.generated_ids)
    };

    let grammar_rejection = result.stop_reason == StopReason::Grammar
        && result.trace.opened == result.generated_ids.len() + 1;
    Ok(GenerateOutput {
        text,
        prompt_tokens: prompt_ids.len(),
        generated_tokens: result.generated_ids.len(),
        token_ids: result.generated_ids,
        stopped: result.stopped && !grammar_rejection,
        stop_reason: Some(result.stop_reason),
        token_logprobs: result.token_logprobs,
    })
}

/// The direct entry's MTP and GDN-first self-speculative routes as a
/// [`SpeculativeSession`] (ADR-090 D6): the session owns the forward passes, the
/// draft and the rollback, and hands the driver verified rounds. It borrows the
/// caller's state for its whole life, like [`QwenMetalSession`].
///
/// The batch-GEMM MTP verifier is not a session: it keeps the legacy loop in the
/// state and is reported as excluded.
pub(crate) struct QwenMetalSpeculativeSession<'state> {
    state: &'state mut MetalQwen35State,
    metrics: SpeculativeMetrics,
    first_candidate: u32,
    /// Keeps the session on the thread that created it (not `Send`, not `Sync`).
    thread_bound: PhantomData<*const ()>,
}

impl<'state> QwenMetalSpeculativeSession<'state> {
    /// Wraps a state that has already been reset and prefilled for this request.
    /// `prefill_logits` are the dense logits of the prompt's last position; their
    /// first-wins argmax is the first candidate.
    pub(crate) fn new(
        state: &'state mut MetalQwen35State,
        metrics: SpeculativeMetrics,
        prefill_logits: &[f32],
    ) -> Self {
        Self {
            state,
            metrics,
            first_candidate: crate::sampling::argmax_f32_first_wins(prefill_logits),
            thread_bound: PhantomData,
        }
    }
}

impl SpeculativeSession for QwenMetalSpeculativeSession<'_> {
    fn first_candidate(&mut self) -> u32 {
        self.first_candidate
    }

    fn advance(
        &mut self,
        pending: u32,
        room: usize,
        is_stop: &dyn Fn(u32) -> bool,
    ) -> Result<VerifiedRound, InferenceError> {
        Ok(self
            .state
            .speculative_round(&mut self.metrics, pending, room, is_stop))
    }

    fn finish(&mut self, _disposition: FinishDisposition) -> Result<(), InferenceError> {
        self.state.finish_speculative(&self.metrics);
        Ok(())
    }
}

/// Drives a [`QwenMetalSpeculativeSession`] through the shared speculative driver and
/// assembles the direct entry's output: the text is decoded from the committed ids in
/// one pass, as the loops this replaces did. The route is recorded on the state whether
/// the request completes or fails.
pub(crate) fn run_speculative_direct(
    session: &mut QwenMetalSpeculativeSession<'_>,
    gen_cfg: &GenerateConfig,
    prompt_len: usize,
    eos_token_id: u32,
    tokenizer: &BpeTokenizer,
) -> Result<GenerateOutput, InferenceError> {
    let mut text = String::new();
    let mut token_logprob_end_offsets: Vec<usize> = Vec::new();
    let never_cancel = || false;
    let result = driver::run_speculative(
        session,
        gen_cfg,
        eos_token_id,
        &never_cancel,
        |_| String::new(),
        &mut text,
        &mut token_logprob_end_offsets,
        |_, _| true,
    );
    let trace = result
        .as_ref()
        .map_or_else(|_| SpeculativeTrace::default(), |r| r.trace);
    let route = session.metrics.route();
    session.state.record_speculative_route(route, trace);
    let result = result?;
    Ok(GenerateOutput {
        text: decode_tokens(tokenizer, &result.generated_ids),
        prompt_tokens: prompt_len,
        generated_tokens: result.generated_ids.len(),
        token_ids: result.generated_ids,
        stopped: result.stopped,
        stop_reason: Some(result.stop_reason),
        token_logprobs: Vec::new(),
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
        assert_eq!(
            PREFIX_CACHE_CAPABILITIES,
            ExecutionCapabilities {
                grammar: true,
                logprobs: false,
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

    /// The tiny fixture with nonzero attention weights and embeddings, so a
    /// prefill's logits depend on the prefix rows already in the KV cache.
    fn varied_fixture() -> (Qwen35Config, ModelWeights) {
        let (cfg, mut weights) = tiny_fixture();
        let fill = |values: &mut Vec<f32>, salt: usize| {
            for (i, value) in values.iter_mut().enumerate() {
                *value = (((i * 31 + salt * 17) % 23) as f32 - 11.0) * 0.01;
            }
        };
        fill(&mut weights.embed_tokens, 1);
        let Some((AttentionWeights::Full(full), _)) = weights.layers.first_mut() else {
            panic!("the tiny fixture has one full-attention layer");
        };
        fill(&mut full.q_proj, 2);
        fill(&mut full.k_proj, 3);
        fill(&mut full.v_proj, 4);
        fill(&mut full.o_proj, 5);
        (cfg, weights)
    }

    /// A state holding the boundary an `ExactAppend` restore leaves: the KV rows
    /// and cursor of `prefix`, with the cursor at `prefix.len()`.
    fn state_restored_to(
        weights: &ModelWeights,
        cfg: &Qwen35Config,
        prefix: &[u32],
    ) -> MetalQwen35State {
        let mut state = MetalQwen35State::new(weights, cfg, TINY_CACHE).expect("tiny Metal state");
        state.try_forward_prefill(prefix).expect("prefix prefill");
        assert_eq!(state.session.position(), prefix.len());
        state
    }

    fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0;
        for (id, value) in logits.iter().enumerate() {
            if *value > logits[best] {
                best = id;
            }
        }
        best as u32
    }

    /// The prefix-cache session prefills the suffix of a restored state exactly
    /// as the prefix-cache loop does: through `forward_prefill_from` at the
    /// boundary, without resetting what the restore left. A batched suffix and a
    /// single-token suffix take different branches of that primitive, and the
    /// single-token one refuses any cursor that is not the boundary.
    #[test]
    fn prefix_session_prefill_on_restored_state_matches_legacy_suffix_prefill() {
        let Some(_) = metal::Device::system_default() else {
            return;
        };
        let _gpu = gpu_test_lock();
        let (cfg, weights) = varied_fixture();
        let prefix = [1u32, 2, 3];
        let gen_cfg = greedy();

        for prompt in [vec![1u32, 2, 3, 4, 5], vec![1u32, 2, 3, 4]] {
            let suffix_start = prefix.len();
            let mut legacy = state_restored_to(&weights, &cfg, &prefix);
            let expected = legacy
                .forward_prefill_from(&prompt[suffix_start..], suffix_start, false)
                .expect("legacy suffix prefill");
            assert!(
                expected.iter().any(|v| *v != expected[0]),
                "fixture shape: the readback must not be constant"
            );

            // Control: the fresh-prompt entry refuses the restored state, which is
            // why the prefix profile cannot prefill through it.
            let refused = state_restored_to(&weights, &cfg, &prefix).try_forward_prefill(&prompt);
            assert!(
                matches!(&refused, Err(InferenceError::InvalidInput(msg)) if msg.contains("fresh session")),
                "{refused:?}"
            );

            let mut state = state_restored_to(&weights, &cfg, &prefix);
            let mut s = QwenMetalSession::over_restored_state(
                &mut state,
                plan(prompt.clone(), 7),
                suffix_start,
                &gen_cfg,
                DENSE_ENV,
            )
            .expect("a restored prefix constructs a session");
            assert_eq!(s.mode(), ReadbackMode::Dense);
            assert_eq!(
                s.state.session.position(),
                suffix_start,
                "construction must leave the restored boundary alone"
            );

            let stamp = s.prefill(&|| false).expect("prefix prefill");
            assert_eq!(stamp.evaluated_len, prompt.len());
            assert_eq!(s.state.session.position(), legacy.session.position());
            match &s.readback {
                Readback::Dense(logits) => assert_eq!(logits, &expected),
                other => panic!("expected a dense readback, got {other:?}"),
            }
            let c = candidate(s.select(&request(&gen_cfg, &prompt)).expect("select"));
            assert_eq!(c.candidate_id, argmax(&expected));
        }
    }

    #[test]
    fn prefix_session_construction_refuses_logprobs_and_a_suffix_beyond_the_prompt() {
        let Some(_) = metal::Device::system_default() else {
            return;
        };
        let _gpu = gpu_test_lock();
        let (cfg, weights) = varied_fixture();
        let prefix = [1u32, 2, 3];
        let mut state = state_restored_to(&weights, &cfg, &prefix);

        let logprobs = GenerateConfig {
            logprobs: Some(1),
            ..greedy()
        };
        let refused = QwenMetalSession::over_restored_state(
            &mut state,
            plan(vec![1, 2, 3, 4], 7),
            3,
            &logprobs,
            COMPACT_ENV,
        )
        .err();
        assert!(
            matches!(&refused, Some(InferenceError::InvalidInput(msg)) if msg.contains("logprobs")),
            "{refused:?}"
        );
        let beyond = QwenMetalSession::over_restored_state(
            &mut state,
            plan(vec![1, 2, 3, 4], 7),
            5,
            &greedy(),
            COMPACT_ENV,
        )
        .err();
        assert!(
            matches!(&beyond, Some(InferenceError::InvalidInput(msg)) if msg.contains("suffix_start")),
            "{beyond:?}"
        );
        assert_eq!(state.session.position(), prefix.len());
        assert_route_disengaged(&state);

        // Control: the same state and a valid boundary construct, and the route the
        // environment plans is engaged.
        let s = QwenMetalSession::over_restored_state(
            &mut state,
            plan(vec![1, 2, 3, 4], 7),
            3,
            &greedy(),
            COMPACT_ENV,
        )
        .expect("a valid boundary constructs");
        assert_eq!(s.state.session.compact_topk, 1);
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
    // `run_direct` over a scripted session (no Metal device): the direct
    // entry's stop-string text and disposition, and the two places its
    // contract differs from what the driver reports on its own.
    // -----------------------------------------------------------------

    /// Runs `script` through `run_direct` with the direct entry's capabilities.
    fn direct_script(
        script: Vec<u32>,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, InferenceError> {
        let mut session = ScriptedSession {
            caps: DIRECT_CAPABILITIES,
            ledger: PredictionLedger::new(),
            script,
            selects: 0,
        };
        run_direct(
            &mut session,
            gen_cfg,
            &[0],
            SCRIPTED_EOS,
            &scripted_tokenizer(),
        )
    }

    fn with_stop_strings(max_new_tokens: usize, stop_strings: &[&str]) -> GenerateConfig {
        GenerateConfig {
            max_new_tokens,
            stop_strings: stop_strings.iter().map(|s| (*s).to_string()).collect(),
            ..Default::default()
        }
    }

    /// "ax" is completed only by the second token of the pair, so the match
    /// needs the text of both: the request stops on that token with
    /// `StopReason::Eos`, keeps it in `token_ids`, and returns the text before
    /// the match.
    #[test]
    fn direct_stop_string_spanning_two_tokens_stops_and_truncates() {
        let output = direct_script(vec![2, 0, 2, 0], &with_stop_strings(8, &["ax"]))
            .expect("a stop string is not an error");
        assert_eq!(output.token_ids, vec![2, 0, 2]);
        assert_eq!(output.text, "x");
        assert!(output.stopped);
        assert_eq!(output.stop_reason, Some(StopReason::Eos));

        // Control: without the stop string the same script runs to its cap.
        let control = direct_script(vec![2, 0, 2, 0], &with_stop_strings(4, &[]))
            .expect("an unconstrained script runs to its cap");
        assert_eq!(control.text, "xaxa");
        assert_eq!(control.stop_reason, Some(StopReason::Length));
    }

    /// The last token's lone 0xE4 byte is released only by the natural-end
    /// flush, as U+FFFD. A stop string that byte completes truncates the text
    /// but leaves the disposition the loop reported: `StopReason::Length` and
    /// `stopped: false`, as the entry's own matcher left them.
    #[test]
    fn direct_stop_string_completed_by_the_final_flush_truncates_and_keeps_the_disposition() {
        let output = direct_script(vec![0, 1], &with_stop_strings(2, &["\u{fffd}"]))
            .expect("a stop string is not an error");
        assert_eq!(output.token_ids, vec![0, 1]);
        assert_eq!(output.text, "a");
        assert!(!output.stopped);
        assert_eq!(output.stop_reason, Some(StopReason::Length));

        // Control: without the stop string the flushed byte stays in the text.
        let control = direct_script(vec![0, 1], &with_stop_strings(2, &[]))
            .expect("an unconstrained script runs to its cap");
        assert_eq!(control.text, "a\u{fffd}");
    }

    /// A stop string that never occurs leaves the text, the ids and the
    /// disposition exactly as a request without stop strings reports them.
    #[test]
    fn direct_stop_string_never_completed_changes_nothing() {
        let output = direct_script(vec![0, 2, 1], &with_stop_strings(3, &["zz"]))
            .expect("a stop string is not an error");
        let control = direct_script(vec![0, 2, 1], &with_stop_strings(3, &[]))
            .expect("an unconstrained script runs to its cap");
        assert_eq!(output.text, "ax\u{fffd}");
        assert_eq!(output.token_ids, control.token_ids);
        assert_eq!(output.text, control.text);
        assert_eq!(output.stopped, control.stopped);
        assert_eq!(output.stop_reason, Some(StopReason::Length));
        assert_eq!(output.stop_reason, control.stop_reason);
    }

    /// A token the grammar rejects ends the request with `StopReason::Grammar`
    /// and `stopped: false`; a grammar the emitted tokens complete reports
    /// `stopped: true` (control).
    #[test]
    fn direct_grammar_rejection_reports_not_stopped() {
        let gen_cfg = GenerateConfig {
            max_new_tokens: 4,
            grammar: Some(scripted_grammar("root ::= \"x\" \"x\"\n", b"b")),
            ..Default::default()
        };
        let rejected = direct_script(vec![2, 0], &gen_cfg).expect("a rejection is not an error");
        assert_eq!(rejected.token_ids, vec![2]);
        assert_eq!(rejected.text, "x");
        assert_eq!(rejected.stop_reason, Some(StopReason::Grammar));
        assert!(!rejected.stopped);

        let completed = direct_script(vec![2, 2], &gen_cfg).expect("a completed grammar");
        assert_eq!(completed.token_ids, vec![2, 2]);
        assert_eq!(completed.stop_reason, Some(StopReason::Grammar));
        assert!(completed.stopped);
    }

    /// A grammar that blocks every id before the first token keeps the
    /// entry's step-0 message; a dead end after the first token keeps the
    /// decode-step message (control).
    #[test]
    fn direct_grammar_block_messages_distinguish_step_zero() {
        let blocked_at_start = GenerateConfig {
            max_new_tokens: 4,
            grammar: Some(scripted_grammar("root ::= \"b\"\n", b"a")),
            ..Default::default()
        };
        match direct_script(vec![1], &blocked_at_start) {
            Err(InferenceError::GrammarConstraintBlocked(message)) => {
                assert_eq!(message, STEP_ZERO_GRAMMAR_BLOCKED);
            }
            other => panic!("expected the step-0 grammar block, got {other:?}"),
        }

        let dead_end = GenerateConfig {
            max_new_tokens: 4,
            grammar: Some(scripted_grammar("root ::= \"x\" \"c\"\n", b"b")),
            ..Default::default()
        };
        match direct_script(vec![2], &dead_end) {
            Err(InferenceError::GrammarConstraintBlocked(message)) => {
                assert!(
                    message.contains("no legal continuation"),
                    "a decode-step dead end keeps its own message, got {message:?}"
                );
            }
            other => panic!("expected the decode-step grammar block, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // `run_prefix_cache_streaming` over a scripted session: the cross-turn
    // slot rule, one case per exit the prefix-cache entry distinguishes.
    // -----------------------------------------------------------------

    /// What a prefix-cache request over `script` commits and how it ended.
    /// `cancel_from` makes the poll return true from that poll on;
    /// `refuse_from` makes `on_token` refuse from that delivery on.
    fn prefix_script(
        script: Vec<u32>,
        gen_cfg: &GenerateConfig,
        cancel_from: Option<u32>,
        refuse_from: Option<u32>,
    ) -> (PrefixStreamRun, Vec<(String, u32)>) {
        let mut session = ScriptedSession {
            caps: PREFIX_CACHE_CAPABILITIES,
            ledger: PredictionLedger::new(),
            script,
            selects: 0,
        };
        let tokenizer = scripted_tokenizer();
        let mut calls = Vec::new();
        let polls = Cell::new(0u32);
        let run = run_prefix_cache_streaming(
            &mut session,
            gen_cfg,
            Some(SCRIPTED_THINK_CLOSE),
            &[0],
            SCRIPTED_EOS,
            &tokenizer,
            |text, id| {
                calls.push((text.to_string(), id));
                refuse_from.is_none_or(|n| (calls.len() as u32) < n)
            },
            || {
                polls.set(polls.get() + 1);
                cancel_from.is_some_and(|n| polls.get() >= n)
            },
        )
        .expect("a scripted prefix-cache request runs");
        (run, calls)
    }

    fn capped(max_new_tokens: usize) -> GenerateConfig {
        GenerateConfig {
            max_new_tokens,
            ..Default::default()
        }
    }

    /// Ids: 0 reads "a", 1 renders U+FFFD only in the final flush, 2 reads "x"
    /// and is the budget-forced close id.
    #[test]
    fn prefix_commit_saves_the_exits_that_forwarded_their_tokens() {
        // A length stop: the last pushed token is not forwarded yet.
        let (run, _) = prefix_script(vec![0, 0], &capped(2), None, None);
        assert_eq!(run.output.stop_reason, Some(StopReason::Length));
        assert_eq!(run.commit, PrefixCommit::Save { silent_step: true });

        // A stop token inside the loop: every pushed token was forwarded.
        let (run, _) = prefix_script(vec![0, SCRIPTED_EOS], &capped(4), None, None);
        assert_eq!(run.output.token_ids, vec![0]);
        assert_eq!(run.commit, PrefixCommit::Save { silent_step: false });

        // A stop token on the first sample saves the prompt-only boundary.
        let (run, _) = prefix_script(vec![SCRIPTED_EOS], &capped(4), None, None);
        assert!(run.output.token_ids.is_empty());
        assert_eq!(run.commit, PrefixCommit::Save { silent_step: false });

        // A grammar the first token completes: the token was never forwarded.
        let completes = GenerateConfig {
            grammar: Some(scripted_grammar("root ::= \"a\"\n", b"b")),
            ..capped(4)
        };
        let (run, _) = prefix_script(vec![0], &completes, None, None);
        assert_eq!(run.output.stop_reason, Some(StopReason::Grammar));
        assert!(run.output.stopped);
        assert_eq!(run.commit, PrefixCommit::Save { silent_step: false });

        // A budget-forced token the grammar rejects: the loop top forwarded the
        // last pushed token, so the silent step must not forward it again.
        let rejects = GenerateConfig {
            enable_thinking: true,
            reasoning_budget: Some(1),
            grammar: Some(scripted_grammar("root ::= \"b\" \"b\"\n", b"b")),
            ..capped(4)
        };
        let (run, _) = prefix_script(vec![1], &rejects, None, None);
        assert_eq!(run.output.stop_reason, Some(StopReason::Grammar));
        assert!(!run.output.stopped);
        assert_eq!(run.commit, PrefixCommit::Save { silent_step: false });
    }

    #[test]
    fn prefix_commit_leaves_the_slot_empty_for_the_exits_that_saved_nothing() {
        // A cancel before prefill and one right after it, with the stats flag.
        let (run, _) = prefix_script(vec![0, 0], &capped(3), Some(1), None);
        assert_eq!(run.output.stop_reason, Some(StopReason::Interrupt));
        assert!(!run.prefill_ran);
        assert_eq!(run.commit, PrefixCommit::Leave);
        let (run, _) = prefix_script(vec![0, 0], &capped(3), Some(2), None);
        assert!(run.prefill_ran);
        assert_eq!(run.commit, PrefixCommit::Leave);

        // The caller refusing the first token's text.
        let (run, _) = prefix_script(vec![0, 0], &capped(3), None, Some(1));
        assert_eq!(run.output.token_ids, vec![0]);
        assert_eq!(run.commit, PrefixCommit::Leave);

        // The caller refusing the natural-end flush (control: a refusal of the
        // same text one delivery earlier would not be the tail).
        let (run, _) = prefix_script(vec![0, 1], &capped(2), None, Some(2));
        assert_eq!(run.output.stop_reason, Some(StopReason::Interrupt));
        assert_eq!(run.commit, PrefixCommit::Leave);

        // A stop string the first token completes, and one a later token does.
        let first = GenerateConfig {
            stop_strings: vec!["a".into()],
            ..capped(4)
        };
        let (run, _) = prefix_script(vec![0, 0], &first, None, None);
        assert_eq!(run.commit, PrefixCommit::Leave);
        let later = GenerateConfig {
            stop_strings: vec!["x".into()],
            ..capped(4)
        };
        let (run, _) = prefix_script(vec![0, 2], &later, None, None);
        assert_eq!(run.output.text, "a");
        assert_eq!(run.commit, PrefixCommit::Leave);

        // A grammar that blocks every token before the first one and is
        // complete without a continuation.
        let complete_at_start = GenerateConfig {
            grammar: Some(scripted_grammar("root ::= \"\"\n", b"b")),
            ..capped(4)
        };
        let (run, _) = prefix_script(vec![0], &complete_at_start, None, None);
        assert_eq!(run.output.stop_reason, Some(StopReason::Grammar));
        assert!(run.output.token_ids.is_empty());
        assert_eq!(run.commit, PrefixCommit::Leave);
    }

    /// A cancel at a decode-loop top and a refused later token save the
    /// forwarded prefix, with no silent step.
    #[test]
    fn prefix_commit_saves_the_forwarded_prefix_on_a_mid_request_interrupt() {
        // Polls 1 and 2 bracket prefill, poll 3 is the first loop top.
        let (run, _) = prefix_script(vec![0, 0, 0], &capped(4), Some(3), None);
        assert_eq!(run.output.stop_reason, Some(StopReason::Interrupt));
        assert_eq!(run.output.token_ids.len(), 1);
        assert_eq!(run.commit, PrefixCommit::Save { silent_step: false });

        // The second delivery refused: the first loop iteration's token.
        let (run, _) = prefix_script(vec![0, 0, 0], &capped(4), None, Some(2));
        assert_eq!(run.output.stop_reason, Some(StopReason::Interrupt));
        assert_eq!(run.output.token_ids.len(), 2);
        assert_eq!(run.commit, PrefixCommit::Save { silent_step: false });
    }

    /// The flush can complete a stop string. However the loop ended, a stop
    /// string the flush completes leaves the slot empty: after a stop token the
    /// exit is unchanged, but the caller's text was still truncated by the match.
    #[test]
    fn prefix_commit_reads_a_stop_string_completed_by_the_flush_per_exit() {
        let flush = |gen_cfg: GenerateConfig| GenerateConfig {
            stop_strings: vec!["\u{fffd}".into()],
            ..gen_cfg
        };

        let (run, _) = prefix_script(vec![0, 1], &flush(capped(2)), None, None);
        assert_eq!(run.output.stop_reason, Some(StopReason::Eos));
        assert!(run.output.stopped);
        assert_eq!(
            run.commit,
            PrefixCommit::Leave,
            "a length stop the flush ends"
        );

        let rejects = GenerateConfig {
            enable_thinking: true,
            reasoning_budget: Some(1),
            grammar: Some(scripted_grammar("root ::= \"b\" \"b\"\n", b"b")),
            ..capped(4)
        };
        let (run, _) = prefix_script(vec![1], &flush(rejects), None, None);
        assert_eq!(run.output.stop_reason, Some(StopReason::Eos));
        assert!(run.output.stopped);
        assert_eq!(
            run.commit,
            PrefixCommit::Leave,
            "a rejection the flush ends"
        );

        let (run, _) = prefix_script(vec![1, SCRIPTED_EOS], &flush(capped(4)), None, None);
        assert_eq!(run.output.stop_reason, Some(StopReason::Eos));
        assert!(run.output.stopped);
        assert_eq!(
            run.commit,
            PrefixCommit::Leave,
            "a stop token the flush then follows"
        );
    }
}
