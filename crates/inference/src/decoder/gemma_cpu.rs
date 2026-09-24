//! `GemmaCpuSession`: the Gemma 4 E2B text CPU concrete [`super::DecoderSession`]
//! (ADR-090 row R04), the Gemma sibling of [`super::qwen_cpu::QwenCpuSession`].
//!
//! Wraps `Gemma4Model`'s per-token `forward_step`
//! (`crates/inference/src/model/gemma4_model.rs`) behind the object-safe session
//! boundary D1 introduces. Unlike Qwen, Gemma has no batched-prefill fallback
//! ladder to reproduce: sequential prefill via `forward_step` is explicitly
//! permitted for this row (ADR-090 D2, "current Gemma text prefill may remain
//! sequential").
//!
//! **Owned state**: a fresh [`Gemma4KvCache`] and [`Gemma4Scratch`] per request
//! (mirroring what [`Gemma4Model::generate_greedy_with_probe`] itself allocates
//! per call), plus `prompt_ids` and an explicit `position` counter. Gemma's
//! cache has no crate-visible aggregate position: `Gemma4KvCache`'s per-layer
//! `seq_len` is donor/slot-resolved and a sliding layer's slot evicts at
//! `sliding_window` capacity, so it cannot serve as the RoPE/`forward_step`
//! position the way Qwen's flat `KvCache::seq_len` does. This session tracks
//! `position` itself, exactly as `Gemma4Model::generate_greedy`/
//! `generate_greedy_with_probe` already do in their own local loops.
//!
//! **Prediction ledger**: one per session, same rationale as `QwenCpuSession`.
//!
//! **RNG / sampling**: reuses [`crate::model::qwen35::sample_token`] (a thin,
//! model-agnostic wrapper over `crate::sampling::sample_full_logits`) and
//! [`crate::model::qwen35::initial_rng_state`] through their existing
//! crate-wide paths -- neither function reads anything Qwen-specific: both
//! take `&GenerateConfig`, raw logit/id slices, or a bare RNG state.
//!
//! **Capabilities**: all four `false`. This row implements full
//! sampling-policy selection (temperature/top-k/top-p/min-p/seed) but no
//! grammar masking, no per-token logprobs, no `stop_strings`, and no
//! reasoning-budget forcing. `metadata` below is a real, working
//! implementation (reused from `QwenCpuSession::metadata`'s exact approach)
//! rather than a stub, but is unreachable through `decoder::driver::run` today
//! because that driver's `check_capabilities` refuses any `gen_cfg.logprobs`
//! request before `select`/`metadata` are ever called on a session declaring
//! `logprobs: false`. A later row can flip these to `true` once each control
//! is actually wired and tested end to end.

use super::qwen_cpu::has_finite_logit;
use super::{
    AcceptedToken, Cancellation, DecoderSession, ExecutionCapabilities, FinishDisposition,
    MetadataRequest, PredictionError, PredictionId, PredictionLedger, SelectOutcome,
    SelectionCandidate, SelectionRequest, StepStamp, TokenMetadata,
};
use crate::error::InferenceError;
use crate::model::gemma4_cache::Gemma4KvCache;
use crate::model::gemma4_model::{Gemma4Model, Gemma4Scratch};
use crate::model::qwen35::{initial_rng_state, sample_token};
use crate::sampling::compute_step_logprobs;

/// `ExecutionCapabilities` for the Gemma 4 E2B text CPU session (ADR-090 row
/// R04): all `false`. Grammar masking, per-token logprobs, `stop_strings`, and
/// reasoning-budget forcing are not implemented against this session; a
/// `gen_cfg` requesting any of them is refused by `decoder::driver::run`'s
/// `check_capabilities` before this session's `prefill` is ever called
/// (proven end to end in `model::gemma4_model`'s test module, one test per
/// capability, through the crate's public `Gemma4Model::generate` entry
/// point).
const CAPABILITIES: ExecutionCapabilities = ExecutionCapabilities {
    grammar: false,
    logprobs: false,
    stop_strings: false,
    reasoning_budget: false,
};

/// Gemma 4 E2B text CPU concrete [`DecoderSession`]. See the module doc
/// comment for what it owns and why.
pub(crate) struct GemmaCpuSession<'model> {
    model: &'model Gemma4Model,
    cache: Gemma4KvCache,
    scratch: Gemma4Scratch,
    prompt_ids: Vec<u32>,
    position: usize,
    rng_state: u64,
    temperature: f32,
    ledger: PredictionLedger,
}

impl<'model> GemmaCpuSession<'model> {
    /// Allocates a fresh [`Gemma4KvCache`] sized to `max_seq_len` (the
    /// caller's own prompt-plus-decode-budget bound -- see
    /// `Gemma4Model::generate_via_driver`'s admission check) and a fresh
    /// [`Gemma4Scratch`], mirroring what
    /// `Gemma4Model::generate_greedy_with_probe` allocates per call.
    /// `temperature` and `seed` are captured for the same reason
    /// `QwenCpuSession::new` captures them (see that constructor's own doc
    /// comment): neither `select`'s reused sampling call nor `metadata`'s
    /// reused scoring call can read them from a later
    /// `SelectionRequest`/`MetadataRequest`.
    ///
    /// # Errors
    /// Propagates [`Gemma4KvCache::new`]'s structural admission errors
    /// (zero-sized dimension, a shared layer with no same-type donor, or a
    /// capacity/kv_dim product overflow).
    pub(crate) fn new(
        model: &'model Gemma4Model,
        prompt_ids: Vec<u32>,
        temperature: f32,
        seed: Option<u64>,
        max_seq_len: usize,
    ) -> Result<Self, InferenceError> {
        let cache = model.new_cache(max_seq_len)?;
        let scratch = Gemma4Scratch::new(&model.config);
        Ok(Self {
            model,
            cache,
            scratch,
            prompt_ids,
            position: 0,
            rng_state: initial_rng_state(seed),
            temperature,
            ledger: PredictionLedger::new(),
        })
    }
}

impl<'model> DecoderSession for GemmaCpuSession<'model> {
    fn capabilities(&self) -> &ExecutionCapabilities {
        &CAPABILITIES
    }

    /// Sequential prefill over `self.prompt_ids` via `Gemma4Model::forward_step`,
    /// one token at a time (ADR-090 D2 explicitly permits this for Gemma;
    /// there is no batched-prefill fallback ladder to reproduce here, unlike
    /// `QwenCpuSession::prefill`). Advances `self.position` by one per token.
    /// Does not sample: the last position's logits are left in
    /// `self.scratch.logits` for the first `select` to read, exactly like
    /// `Gemma4Model::generate_greedy`'s own prefill loop. `prediction: None`
    /// in the returned `StepStamp` reflects that nothing has been opened in
    /// the ledger yet.
    fn prefill(&mut self, cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        if cancel.is_cancelled() {
            return Err(InferenceError::Inference("cancelled before prefill".into()));
        }

        let prompt_len = self.prompt_ids.len();
        for i in 0..prompt_len {
            let token_id = self.prompt_ids[i];
            self.model.forward_step(
                token_id,
                self.position,
                &mut self.cache,
                &mut self.scratch,
                &[],
            )?;
            self.position += 1;
        }

        Ok(StepStamp {
            evaluated_len: self.position,
            prediction: None,
        })
    }

    /// Consumes `accepted.prediction` (rejecting a stale/foreign/already-consumed
    /// id through `PredictionLedger::consume`), then runs one `forward_step` at
    /// `self.position`, advancing it by one and leaving the new position's
    /// logits in `self.scratch.logits` for the next `select`. Consumption
    /// happens before the forward pass so a stale id never advances session
    /// state.
    fn decode(
        &mut self,
        accepted: &AcceptedToken,
        cancel: &dyn Cancellation,
    ) -> Result<StepStamp, InferenceError> {
        if cancel.is_cancelled() {
            return Err(InferenceError::Inference("cancelled before decode".into()));
        }
        self.ledger.consume(accepted.prediction)?;

        self.model.forward_step(
            accepted.final_id,
            self.position,
            &mut self.cache,
            &mut self.scratch,
            &[],
        )?;
        self.position += 1;

        Ok(StepStamp {
            evaluated_len: self.position,
            prediction: None,
        })
    }

    /// Masks the logits `prefill`/`decode` most recently left in
    /// `self.scratch` through `request.grammar_mask` (always `None` in
    /// practice: `CAPABILITIES.grammar` is `false`, so
    /// `decoder::driver::run` refuses any request with `gen_cfg.grammar` set
    /// before this method is ever called -- kept as a real, working branch
    /// rather than an `unreachable!()` so a future row that flips the
    /// capability does not have to rewrite this method), then samples via
    /// [`sample_token`] (the crate's existing sampling path:
    /// temperature/top-k/top-p/min-p/repetition-penalty/seed, all read from
    /// `request.config`/`self.rng_state`), and opens a new ledger prediction
    /// for the sampled candidate. Mirrors `QwenCpuSession::select` exactly.
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
    /// `crate::sampling::compute_step_logprobs`, exactly as
    /// `QwenCpuSession::metadata` does. Real and working (see this module's
    /// doc comment on why it is not a stub), though unreachable through the
    /// driver today since `CAPABILITIES.logprobs` is `false`.
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

    /// `Poisoned` invalidates whatever prediction is still live; `Reusable` is
    /// a no-op. Same rationale as `QwenCpuSession::finish`.
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
    use crate::model::gemma4_model::tiny_zero_model;

    fn cancel_false() -> impl Fn() -> bool {
        || false
    }

    fn cancel_true() -> impl Fn() -> bool {
        || true
    }

    // -----------------------------------------------------------------
    // Declared capabilities are honestly all false (this row implements
    // full sampling-policy selection but no grammar/logprobs/stop_strings/
    // reasoning_budget). The refusal mechanism itself is
    // `decoder::driver::run`'s generic `check_capabilities`, proven once for
    // any session in `driver`'s own test module; `model::gemma4_model`'s test
    // module exercises the refusal end to end for this session specifically,
    // through the public `Gemma4Model::generate` entry point.
    //
    // A whole-struct `assert_eq!` against an explicit expected literal, not
    // four `assert!(!CAPABILITIES.field)` calls: `CAPABILITIES` is a `const`,
    // so each individual field read is constant-folded at compile time and
    // clippy's `assertions_on_constants` flags `assert!(!CONST.field)` as
    // asserting a value already known at compile time ("this assertion has a
    // constant value"). `ExecutionCapabilities` derives `PartialEq` + `Debug`
    // (`decoder.rs`), so an `assert_eq!` on the whole struct is available and
    // is not a boolean-literal condition to that lint -- and it keeps this
    // test's original intent (all four fields pinned to `false` explicitly,
    // independent of whatever `ExecutionCapabilities::default()` happens to
    // mean) rather than weakening it to a `Default` comparison.
    // -----------------------------------------------------------------
    #[test]
    fn declared_capabilities_are_honestly_all_false() {
        assert_eq!(
            CAPABILITIES,
            ExecutionCapabilities {
                grammar: false,
                logprobs: false,
                stop_strings: false,
                reasoning_budget: false,
            }
        );
    }

    // -----------------------------------------------------------------
    // Acceptance: one full step, prefill -> select -> metadata -> decode,
    // over a tiny zero-weight synthetic Gemma 4 model (1 layer, non-shared
    // global attention -- see `tiny_zero_model`'s doc comment). All-zero
    // weights make every logit exactly zero (a deterministic uniform
    // distribution), the same trick `QwenCpuSession`'s own tiny-model test
    // relies on.
    // -----------------------------------------------------------------
    #[test]
    fn full_step_lifecycle_over_tiny_zero_model() {
        let model = tiny_zero_model();
        let prompt_ids = vec![2u32, 3u32];
        let vocab_size = model.config.vocab_size;

        let mut session = GemmaCpuSession::new(&model, prompt_ids.clone(), 0.0, Some(7), 16)
            .expect("session construction over the tiny model must succeed");

        let cancel = cancel_false();
        let stamp = session
            .prefill(&cancel)
            .expect("prefill over the tiny zero-weight model must succeed");
        assert_eq!(stamp.evaluated_len, prompt_ids.len());
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
        // All-zero weights -> every logit is exactly 0.0 -> greedy sampling
        // (temperature 0.0) picks the first (lowest-id) token deterministically.
        assert_eq!(candidate.candidate_id, 0);
        assert!(session.ledger.is_live(candidate.prediction));

        let meta_request = MetadataRequest {
            top_logprobs: Some(2),
        };
        let metadata = session
            .metadata(candidate.prediction, candidate.candidate_id, &meta_request)
            .expect("metadata over a live prediction must succeed");
        assert_eq!(metadata.final_token_id, candidate.candidate_id);
        assert_eq!(metadata.prediction, candidate.prediction);
        // All-zero logits -> uniform distribution over `vocab_size` -> every
        // token's reporting log-probability is exactly -ln(vocab_size).
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
        assert_eq!(stamp2.evaluated_len, prompt_ids.len() + 1);
        assert_eq!(stamp2.prediction, None);
        assert!(!session.ledger.is_live(candidate.prediction));

        session
            .finish(FinishDisposition::Reusable)
            .expect("finish(Reusable) must always succeed");
    }

    // -----------------------------------------------------------------
    // Cancellation before prefill: no forward pass runs, `self.position`
    // stays at its constructed value (0).
    // -----------------------------------------------------------------
    #[test]
    fn cancelled_before_prefill_returns_error_and_advances_nothing() {
        let model = tiny_zero_model();
        let mut session = GemmaCpuSession::new(&model, vec![2, 3], 0.0, Some(7), 16)
            .expect("session construction must succeed");

        let result = session.prefill(&cancel_true());
        assert!(result.is_err(), "a cancelled prefill must not succeed");
        assert_eq!(
            session.position, 0,
            "a cancelled prefill must not advance position or run forward_step"
        );
    }

    // -----------------------------------------------------------------
    // Ledger lifecycle mutation target, mid-decode cancellation: a cancelled
    // decode must consume nothing, and finish(Poisoned) must invalidate a
    // still-live prediction. Mirrors `QwenCpuSession`'s equivalent test with
    // the same before/after control.
    // -----------------------------------------------------------------
    #[test]
    fn cancelled_decode_consumes_nothing_and_finish_poisoned_invalidates() {
        let model = tiny_zero_model();
        let prompt_ids = vec![2u32, 3u32];
        let mut session = GemmaCpuSession::new(&model, prompt_ids.clone(), 0.0, Some(7), 16)
            .expect("session construction must succeed");
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
}
