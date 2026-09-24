//! `QwenCpuSession`: the first concrete [`super::DecoderSession`] (ADR-090 row B).
//!
//! Wraps `Qwen35Model`'s base CPU dense entry point (`generate`/`generate_streaming`,
//! `crates/inference/src/model/qwen35/generation.rs`) behind the object-safe session
//! boundary D1 introduces. No driver and no caller anywhere else in the crate: this row
//! wires the concrete session so the trait's object safety and the ledger's eligibility
//! rules are proven against real prefill/decode/sample computation, not only against the
//! `DummySession` in `super::tests`. Wiring `generate()`/`generate_streaming()` to route
//! through a driver built on this session is row C's work, kept apart from this row
//! deliberately: this session is exercised directly by tests.
//!
//! **Owned state**, read from what `Qwen35Model::generate` itself constructs and owns per
//! request: `gdn_states`, `kv_cache`, `scratch` (all reused unmodified from
//! `super::super::model::qwen35::{cache, forward}`), plus `prompt_ids`/`prompt_len`. The
//! model itself is borrowed immutably (`&'model Qwen35Model`), matching `generate`'s own
//! `&self` receiver.
//!
//! **Prediction ledger**: one per session (the obvious answer, and nothing here needs more
//! than one prefill's worth of live-prediction bookkeeping at a time). `prefill` does not
//! open a prediction -- it leaves the freshly evaluated logits in `scratch.logits`, exactly
//! where `generate()`'s own post-prefill sampling call reads them -- so the ledger has no
//! live prediction until the first `select`. Another prefill would need to invalidate
//! whatever prediction is still live (D2); this row's session is constructed fresh per
//! prompt (mirroring `generate()`'s own per-call allocation) and its `prefill` is called at
//! most once per construction, so `reset()`-on-reprefill is inherent-method territory for
//! a later row rather than exercised here.
//!
//! **RNG state**: `sample_token` (`model::qwen35::sampling`) takes `rng_state: &mut u64`
//! explicitly -- the draw is threaded, not global -- and `SelectionRequest` carries no RNG
//! slot of its own (`select` takes `&mut self`), so the u64 lives in the session, seeded
//! once at construction via the exact same `initial_rng_state` transform `generate()` uses.
//! Reusing that function (rather than re-deriving the seed-to-state mapping here) is what
//! keeps a given seed's draw schedule identical to the legacy path; row C's golden is what
//! will actually catch a divergence.
//!
//! **`metadata` temperature**: `compute_step_logprobs` (`crate::sampling`) is a pure
//! function of `(logits, token_id, temperature, top_n)` with no `GenerateConfig` access of
//! its own, and the trait's `metadata` signature carries no config either -- so the
//! temperature has to be captured at construction, same as the RNG seed. This session
//! stores it as a bare `f32` (not the whole `GenerateConfig`) since temperature is the only
//! field either `metadata` or the reused sampling call needs; `select`'s own temperature
//! comes from the caller's `SelectionRequest::config` on each call, same as `generate()`'s
//! `sample_token(.., gen_cfg, ..)`. Verified at the call site this session's construction
//! is meant to mirror: `generate()` passes `gen_cfg.temperature` into
//! `DecodePolicy::init`/`transition`, which is what reaches `record_logprob` ->
//! `compute_step_logprobs` -- the same value this session captures.
//!
//! **Grammar state (row R03, re-owned by `decoder::driver` in this rework)**: this session
//! holds no grammar engine or state of its own. ADR-090 D1 names the driver as the owner of
//! grammar transitions, so a per-model session that carried its own copy would be exactly the
//! duplication every future session (Gemma CPU, Metal Qwen, ...) would have to reimplement.
//! Each `select` call instead receives a borrowed mask through
//! [`SelectionRequest::grammar_mask`] (`None` when no grammar is set) and applies it to its
//! own logits before sampling, via [`has_finite_logit`] to detect "every token blocked" --
//! see `decoder::driver`'s module doc comment for who owns the engine/state and where
//! `advance`/`is_complete_without_continuation` are called from.

use super::{
    AcceptedToken, Cancellation, DecoderSession, ExecutionCapabilities, FinishDisposition,
    MetadataRequest, PredictionError, PredictionId, PredictionLedger, SelectOutcome,
    SelectionCandidate, SelectionRequest, StepStamp, TokenMetadata,
};
use crate::attention::gdn::GatedDeltaNetState;
use crate::error::InferenceError;
use crate::model::qwen35::Qwen35Model;
use crate::model::qwen35::{ForwardScratch, KvCache, force_serial_prefill, initial_rng_state};
use crate::model::qwen35::{prefill_tokens, sample_token};
use crate::sampling::compute_step_logprobs;

/// When a grammar engine blocks every token via `mask_logits`, every logit becomes
/// `NEG_INFINITY`. Without this guard the sampler's non-finite-max short-circuit would
/// silently emit token 0 (lowest id after sorting an all-NEG_INFINITY candidate set),
/// violating the grammar contract. `select` checks this before invoking the sampler and
/// returns [`SelectOutcome::GrammarExhausted`] or a typed error instead. Moved here from
/// `model::qwen35::generation` (row R03): this is the one site that still masks logits
/// before sampling; `model::qwen35::generation`'s own tests reach it via
/// `crate::decoder::qwen_cpu::has_finite_logit`, not a private copy.
pub(crate) fn has_finite_logit(logits: &[f32]) -> bool {
    logits.iter().any(|&l| l > f32::NEG_INFINITY)
}

/// `ExecutionCapabilities` for the base CPU dense entry point. All four true: this is the
/// same family `generate()`/`generate_streaming()` wire directly (see the doc comment on
/// [`super::ExecutionCapabilities`], which names this exact family for every field), as
/// opposed to the standalone `generate_q8`/`generate_f16`/`generate_q8_neon` wrappers that
/// fail closed via `check_*_not_set`. Declared narrower would be inventing caution the
/// underlying entry point does not have.
const CAPABILITIES: ExecutionCapabilities = ExecutionCapabilities {
    grammar: true,
    logprobs: true,
    stop_strings: true,
    reasoning_budget: true,
};

/// First concrete [`DecoderSession`]. See the module doc comment for what it owns and why.
pub(crate) struct QwenCpuSession<'model> {
    model: &'model Qwen35Model,
    gdn_states: Vec<GatedDeltaNetState>,
    kv_cache: KvCache,
    scratch: ForwardScratch,
    prompt_ids: Vec<u32>,
    prompt_len: usize,
    rng_state: u64,
    temperature: f32,
    ledger: PredictionLedger,
}

impl<'model> QwenCpuSession<'model> {
    /// Allocates fresh `gdn_states`/`kv_cache`/`scratch` for `prompt_ids`, mirroring what
    /// `Qwen35Model::generate` itself allocates per call. `temperature` and `seed` are the
    /// two `GenerateConfig` fields this session must capture at construction (see the module
    /// doc comment on why `metadata` and the RNG draw need them independent of any later
    /// `SelectionRequest`). No `grammar` parameter: the driver owns the grammar engine and
    /// state now (module doc comment) and hands this session a mask per step instead.
    pub(crate) fn new(
        model: &'model Qwen35Model,
        prompt_ids: Vec<u32>,
        temperature: f32,
        seed: Option<u64>,
    ) -> Self {
        let cfg = &model.config;
        let num_linear = cfg.num_linear_attention_layers();
        let num_full = cfg.num_full_attention_layers();
        let gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let prompt_len = prompt_ids.len();

        Self {
            model,
            gdn_states,
            kv_cache: KvCache::new(num_full),
            scratch: ForwardScratch::new(),
            prompt_ids,
            prompt_len,
            rng_state: initial_rng_state(seed),
            temperature,
            ledger: PredictionLedger::new(),
        }
    }
}

impl<'model> DecoderSession for QwenCpuSession<'model> {
    fn capabilities(&self) -> &ExecutionCapabilities {
        &CAPABILITIES
    }

    /// Reproduces `generate()`'s three-branch prefill choreography
    /// (`model::qwen35::generation.rs:~140-192`) exactly, over this session's own
    /// `gdn_states`/`kv_cache`/`scratch` instead of `generate`'s locals:
    ///
    /// 1. `force_serial_prefill()` (test-only escape hatch; `false` in a release binary) ->
    ///    serial `prefill_tokens`, never attempting the batched call.
    /// 2. Otherwise, batched first; on `Err(UnsupportedModel(_))` ONLY, fall back to serial
    ///    (the batched call is documented, and tested by
    ///    `test_batch_prefill_rejects_moe_without_panic` plus this row's own session-level
    ///    control, to refuse before mutating `gdn_states`/`kv_cache`).
    /// 3. Any other `Err` propagates unchanged.
    ///
    /// Does not sample: the post-prefill logits are left in `self.scratch.logits` for the
    /// first `select` to read, exactly where `generate()` leaves them before its own
    /// post-prefill `sample_token` call. `prediction: None` in the returned `StepStamp`
    /// reflects that nothing has been opened in the ledger yet.
    fn prefill(&mut self, cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        if cancel.is_cancelled() {
            return Err(InferenceError::Inference("cancelled before prefill".into()));
        }

        let vocab_size = self.model.config.vocab_size;
        let prefill_logits: Vec<f32> = if force_serial_prefill() {
            prefill_tokens(
                self.model,
                &self.prompt_ids,
                &mut self.gdn_states,
                &mut self.kv_cache,
                &mut self.scratch,
            );
            self.kv_cache.seq_len = self.prompt_len;
            self.scratch.logits[..vocab_size].to_vec()
        } else {
            match self.model.prefill_tokens_batched_for_generate(
                &self.prompt_ids,
                &mut self.gdn_states,
                &mut self.kv_cache,
            ) {
                Ok(logits) => logits,
                Err(InferenceError::UnsupportedModel(_)) => {
                    prefill_tokens(
                        self.model,
                        &self.prompt_ids,
                        &mut self.gdn_states,
                        &mut self.kv_cache,
                        &mut self.scratch,
                    );
                    self.kv_cache.seq_len = self.prompt_len;
                    self.scratch.logits[..vocab_size].to_vec()
                }
                Err(e) => return Err(e),
            }
        };

        // The batched path only mutates its own private `PrefillScratch`, so
        // `self.scratch.logits` can still be its initial zero-length `Vec::new()`.
        let cfg = &self.model.config;
        self.scratch.ensure_capacity(cfg, self.prompt_len);
        self.scratch.logits[..vocab_size].copy_from_slice(&prefill_logits);

        Ok(StepStamp {
            evaluated_len: self.kv_cache.seq_len,
            prediction: None,
        })
    }

    /// Consumes `accepted.prediction` (rejecting a stale/foreign/already-consumed id through
    /// `PredictionLedger::consume`, mapped via `From<PredictionError>`), then runs one
    /// `forward_step` at the current cache position -- the same `forward_step` call the
    /// pre-driver `decode_loop` used to make -- advancing `kv_cache.seq_len` and leaving
    /// the new position's logits in `scratch.logits` for the next `select`. Consumption
    /// happens before the forward pass so a stale id never advances session state.
    fn decode(
        &mut self,
        accepted: &AcceptedToken,
        cancel: &dyn Cancellation,
    ) -> Result<StepStamp, InferenceError> {
        if cancel.is_cancelled() {
            return Err(InferenceError::Inference("cancelled before decode".into()));
        }
        self.ledger.consume(accepted.prediction)?;

        let pos = self.kv_cache.seq_len;
        self.model.forward_step(
            accepted.final_id,
            pos,
            &mut self.gdn_states,
            &mut self.kv_cache,
            &mut self.scratch,
        );
        self.kv_cache.seq_len += 1;

        Ok(StepStamp {
            evaluated_len: self.kv_cache.seq_len,
            prediction: None,
        })
    }

    /// Masks the logits `prefill`/`decode` most recently left in `self.scratch` through
    /// `request.grammar_mask` (a no-op when `None` -- no grammar set), then samples via the
    /// exact `sample_token` `generate()` calls, and opens a new ledger prediction for the
    /// sampled candidate.
    ///
    /// Mirrors `generate_inline`'s / `decode_loop`'s mask-before-sample sequence, minus the
    /// part that moved to the driver: this method masks in place and fails closed via
    /// [`has_finite_logit`] when every token is blocked, but does not itself decide whether
    /// that is a completed grammar or a real error -- it always reports
    /// [`SelectOutcome::GrammarExhausted`] and leaves that call to whichever caller still
    /// holds the grammar engine and state (see `SelectOutcome`'s and `decoder::driver`'s
    /// doc comments). Grammar *advance* is likewise not this method's job any more: the
    /// driver calls it directly on its own owned state after the sampled candidate is known.
    fn select(&mut self, request: &SelectionRequest<'_>) -> Result<SelectOutcome, InferenceError> {
        let vocab_size = self.model.config.vocab_size;

        if let Some(mask) = request.grammar_mask {
            mask(&mut self.scratch.logits[..vocab_size])?;
            if !has_finite_logit(&self.scratch.logits[..vocab_size]) {
                return Ok(SelectOutcome::GrammarExhausted);
            }
        }

        let candidate_id = sample_token(
            &self.scratch.logits[..vocab_size],
            request.config,
            request.history,
            &mut self.rng_state,
        );
        let prediction = self.ledger.open();

        Ok(SelectOutcome::Candidate(SelectionCandidate {
            candidate_id,
            prediction,
        }))
    }

    /// Scores `final_token` against the prediction's logits via
    /// `crate::sampling::compute_step_logprobs` -- the same pure function
    /// `DecodePolicy::record_logprob` calls -- reusing it rather than writing a second
    /// scorer. `request.top_logprobs: None` is treated as `top_n = 0` (report the final
    /// token's log-probability with no alternatives) rather than refusing to score at all:
    /// the underlying call is total over `top_n`, so there is no reason to make this
    /// operation partial when the caller passed `None`. Rejects `prediction` if it is not
    /// (or no longer) the ledger's live prediction -- the eligibility check
    /// `PredictionLedger::is_live`'s own doc comment names `select`/`metadata` as the
    /// operations that must consult it.
    fn metadata(
        &mut self,
        prediction: PredictionId,
        final_token: u32,
        request: &MetadataRequest,
    ) -> Result<TokenMetadata, InferenceError> {
        if !self.ledger.is_live(prediction) {
            return Err(PredictionError::Stale.into());
        }

        let vocab_size = self.model.config.vocab_size;
        let top_n = request.top_logprobs.unwrap_or(0);
        let (final_logprob, top) = compute_step_logprobs(
            &self.scratch.logits[..vocab_size],
            final_token,
            self.temperature,
            top_n,
        );

        Ok(TokenMetadata {
            prediction,
            final_token_id: final_token,
            final_logprob,
            top,
        })
    }

    /// `Poisoned` invalidates whatever prediction is still live, through the same
    /// `PredictionLedger::invalidate` a cancellation or failure would use (D2: a poisoned
    /// session's typed state must never be made to look reusable by patching a sequence
    /// number, and the ledger is the piece of that typed state this row's session actually
    /// has). `Reusable` is a no-op: nothing else in this row's state needs to change for a
    /// completed-or-failed-before-mutation session to remain valid for another prefill.
    fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError> {
        if disposition == FinishDisposition::Poisoned {
            self.ledger.invalidate();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::GenerateConfig;
    use crate::model::qwen35::test_support::tiny_zero_model;
    use crate::tokenizer::common::Tokenizer;

    fn cancel_false() -> impl Fn() -> bool {
        || false
    }

    fn cancel_true() -> impl Fn() -> bool {
        || true
    }

    fn tokenize(model: &Qwen35Model, text: &str) -> Vec<u32> {
        let input = model.tokenizer.tokenize(text);
        input.input_ids[..input.real_length].to_vec()
    }

    // -----------------------------------------------------------------
    // Acceptance 1 (checkpoint-free half): one full step,
    // prefill -> select -> metadata -> decode, over a tiny zero-weight
    // dense model. All-zero weights make every logit zero, so greedy
    // (temperature 0.0) sampling is deterministic without needing a real
    // checkpoint for this shape of assertion.
    // -----------------------------------------------------------------

    #[test]
    fn full_step_lifecycle_over_tiny_dense_model() {
        let model = tiny_zero_model();
        let prompt_ids = tokenize(&model, "abc");
        assert!(
            !prompt_ids.is_empty(),
            "test tokenizer must produce >=1 id for \"abc\""
        );
        let prompt_len = prompt_ids.len();

        let mut session = QwenCpuSession::new(&model, prompt_ids.clone(), 0.0, Some(7));

        let cancel = cancel_false();
        let stamp = session
            .prefill(&cancel)
            .expect("prefill over a dense tiny model must succeed");
        assert_eq!(stamp.evaluated_len, prompt_len);
        assert_eq!(
            stamp.prediction, None,
            "prefill must not open a prediction; select does"
        );

        let history = prompt_ids.clone();
        let gen_cfg = GenerateConfig {
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let request = SelectionRequest {
            config: &gen_cfg,
            history: &history,
            grammar_mask: None,
        };
        let candidate = match session
            .select(&request)
            .expect("select over freshly prefilled logits must succeed")
        {
            SelectOutcome::Candidate(c) => c,
            SelectOutcome::GrammarExhausted => panic!("no grammar set on this session"),
        };

        // Ledger lifecycle, open half: live immediately after select.
        assert!(session.ledger.is_live(candidate.prediction));

        let meta_request = MetadataRequest {
            top_logprobs: Some(2),
        };
        let metadata = session
            .metadata(candidate.prediction, candidate.candidate_id, &meta_request)
            .expect("metadata over a live prediction must succeed");
        assert_eq!(metadata.final_token_id, candidate.candidate_id);
        assert_eq!(metadata.prediction, candidate.prediction);
        // All-zero logits -> uniform distribution over `vocab_size` -> every token's
        // reporting log-probability is exactly -ln(vocab_size).
        let vocab_size = model.config.vocab_size;
        let expected_logprob = -(vocab_size as f32).ln();
        assert!(
            (metadata.final_logprob - expected_logprob).abs() < 1e-4,
            "final_logprob={} expected={}",
            metadata.final_logprob,
            expected_logprob
        );
        // metadata does not consume: still live after scoring.
        assert!(session.ledger.is_live(candidate.prediction));

        let accepted = AcceptedToken {
            final_id: metadata.final_token_id,
            prediction: candidate.prediction,
        };
        let stamp2 = session
            .decode(&accepted, &cancel)
            .expect("decode of the freshly opened prediction must succeed");
        assert_eq!(stamp2.evaluated_len, prompt_len + 1);
        assert_eq!(stamp2.prediction, None);

        // Ledger lifecycle, consumed half: no longer live after decode.
        assert!(!session.ledger.is_live(candidate.prediction));

        session
            .finish(FinishDisposition::Reusable)
            .expect("finish(Reusable) must always succeed");
    }

    // -----------------------------------------------------------------
    // Acceptance 2: MoE fallback precedes mutation, at the session level
    // (both `gdn_states` and `kv_cache`, not just `kv_cache` the way
    // `test_batch_prefill_rejects_moe_without_panic` does). Checkpoint-free:
    // `is_moe()` is a pure config predicate, so flagging a tiny dense model
    // MoE-shaped via config alone is enough to reach the refusal branch
    // without needing real MoE weights.
    // -----------------------------------------------------------------

    #[test]
    fn moe_fallback_precedes_mutation_of_gdn_states_and_kv_cache() {
        let mut model = tiny_zero_model();
        model.config.num_experts = Some(1);
        model.config.num_experts_per_tok = Some(1);
        assert!(model.config.is_moe());

        let prompt_ids = tokenize(&model, "abc");
        assert!(!prompt_ids.is_empty());

        let mut session = QwenCpuSession::new(&model, prompt_ids, 0.0, Some(7));

        let pre_gdn: Vec<(Vec<f32>, Vec<f32>)> = session
            .gdn_states
            .iter()
            .map(|s| (s.s_matrices.clone(), s.conv_buffer.clone()))
            .collect();
        let pre_k = session.kv_cache.k.clone();
        let pre_v = session.kv_cache.v.clone();
        let pre_seq_len = session.kv_cache.seq_len;

        let err = session
            .model
            .prefill_tokens_batched_for_generate(
                &session.prompt_ids,
                &mut session.gdn_states,
                &mut session.kv_cache,
            )
            .expect_err("MoE-flagged config must refuse the batched path");
        assert!(matches!(err, InferenceError::UnsupportedModel(_)));

        let post_gdn: Vec<(Vec<f32>, Vec<f32>)> = session
            .gdn_states
            .iter()
            .map(|s| (s.s_matrices.clone(), s.conv_buffer.clone()))
            .collect();
        assert_eq!(
            pre_gdn, post_gdn,
            "gdn_states must be untouched by a refused batched prefill attempt \
             -- the doc comment on prefill_tokens_batched_for_generate names \
             gdn_states in the same breath as kv_cache, but the existing \
             function-level test never checks it"
        );
        assert_eq!(pre_k, session.kv_cache.k);
        assert_eq!(pre_v, session.kv_cache.v);
        assert_eq!(pre_seq_len, session.kv_cache.seq_len);
    }

    // -----------------------------------------------------------------
    // The escape hatch is a peer of the other two branches, not a modifier:
    // this session's own `prefill` must still take the serial path when
    // `force_serial_prefill()` is forced true, and produce the same
    // evaluated length / logits a batched-first prefill would (all-zero
    // weights make the two paths trivially equal here; parity under real
    // weights is what generation.rs's own delegation-parity test covers).
    // -----------------------------------------------------------------

    #[test]
    fn escape_hatch_forces_the_serial_branch() {
        use crate::model::qwen35::{FORCE_SERIAL_PREFILL, SERIAL_PREFILL_TEST_LOCK};
        let _guard = SERIAL_PREFILL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let model = tiny_zero_model();
        let prompt_ids = tokenize(&model, "abc");
        let prompt_len = prompt_ids.len();
        let mut session = QwenCpuSession::new(&model, prompt_ids, 0.0, Some(7));

        FORCE_SERIAL_PREFILL.store(true, std::sync::atomic::Ordering::SeqCst);
        let result = session.prefill(&cancel_false());
        FORCE_SERIAL_PREFILL.store(false, std::sync::atomic::Ordering::SeqCst);

        let stamp = result.expect("forced-serial prefill over a dense tiny model must succeed");
        assert_eq!(stamp.evaluated_len, prompt_len);
        assert_eq!(session.kv_cache.seq_len, prompt_len);
    }

    // -----------------------------------------------------------------
    // Ledger lifecycle mutation target: a cancelled decode must consume
    // nothing, and finish(Poisoned) must invalidate a still-live
    // prediction. Both are asserted with a before/after control, per the
    // fleet rule that a before/after comparison catches what a
    // post-state-only check does not.
    // -----------------------------------------------------------------

    #[test]
    fn cancelled_decode_consumes_nothing_and_finish_poisoned_invalidates() {
        let model = tiny_zero_model();
        let prompt_ids = tokenize(&model, "abc");
        let mut session = QwenCpuSession::new(&model, prompt_ids.clone(), 0.0, Some(7));
        session
            .prefill(&cancel_false())
            .expect("prefill must succeed");

        let gen_cfg = GenerateConfig {
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let request = SelectionRequest {
            config: &gen_cfg,
            history: &prompt_ids,
            grammar_mask: None,
        };
        let candidate = match session.select(&request).expect("select must succeed") {
            SelectOutcome::Candidate(c) => c,
            SelectOutcome::GrammarExhausted => panic!("no grammar set on this session"),
        };
        assert!(session.ledger.is_live(candidate.prediction)); // control: live before

        let accepted = AcceptedToken {
            final_id: candidate.candidate_id,
            prediction: candidate.prediction,
        };
        let cancelled = session.decode(&accepted, &cancel_true());
        assert!(cancelled.is_err(), "a cancelled decode must not succeed");
        assert!(
            session.ledger.is_live(candidate.prediction),
            "a cancelled decode must not consume the live prediction"
        );

        session
            .finish(FinishDisposition::Poisoned)
            .expect("finish must always return Ok");
        assert!(
            !session.ledger.is_live(candidate.prediction),
            "finish(Poisoned) must invalidate whatever prediction was live"
        );
    }

    // -----------------------------------------------------------------
    // Acceptance 1 (checkpoint-gated half): the same full-step lifecycle,
    // over the real dense Qwen3.5-0.8B checkpoint. Requires
    // LATTICE_INFERENCE_MODEL_DIR (bf16 safetensors; needs the `f16`
    // feature to load) and panics rather than silently skipping when unset
    // (`crate::test_support::require_checkpoint_dir`).
    // -----------------------------------------------------------------

    #[test]
    #[ignore = "requires local Qwen3.5 checkpoint: set LATTICE_INFERENCE_MODEL_DIR"]
    fn full_step_lifecycle_over_real_checkpoint() {
        let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");
        let model = Qwen35Model::from_safetensors(std::path::Path::new(&model_dir))
            .expect("dense Qwen3.5 checkpoint should load successfully");

        let prompt_ids = tokenize(&model, "The quick brown fox jumps over the lazy dog.");
        assert!(!prompt_ids.is_empty());
        let prompt_len = prompt_ids.len();

        let mut session = QwenCpuSession::new(&model, prompt_ids.clone(), 0.0, Some(1234));

        let cancel = cancel_false();
        let stamp = session
            .prefill(&cancel)
            .expect("prefill over the real checkpoint must succeed");
        assert_eq!(stamp.evaluated_len, prompt_len);
        assert_eq!(stamp.prediction, None);

        let gen_cfg = GenerateConfig {
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let request = SelectionRequest {
            config: &gen_cfg,
            history: &prompt_ids,
            grammar_mask: None,
        };
        let candidate = match session.select(&request).expect("select must succeed") {
            SelectOutcome::Candidate(c) => c,
            SelectOutcome::GrammarExhausted => panic!("no grammar set on this session"),
        };
        assert!(session.ledger.is_live(candidate.prediction)); // open at select

        let meta_request = MetadataRequest {
            top_logprobs: Some(3),
        };
        let metadata = session
            .metadata(candidate.prediction, candidate.candidate_id, &meta_request)
            .expect("metadata must succeed");
        assert_eq!(metadata.final_token_id, candidate.candidate_id);
        assert!(metadata.final_logprob.is_finite());
        assert_eq!(metadata.top.len(), 3);

        let accepted = AcceptedToken {
            final_id: metadata.final_token_id,
            prediction: candidate.prediction,
        };
        let stamp2 = session
            .decode(&accepted, &cancel)
            .expect("decode must succeed");
        assert_eq!(stamp2.evaluated_len, prompt_len + 1);
        assert!(
            !session.ledger.is_live(candidate.prediction),
            "consumed at decode"
        );

        session
            .finish(FinishDisposition::Reusable)
            .expect("finish must succeed");
    }
}
