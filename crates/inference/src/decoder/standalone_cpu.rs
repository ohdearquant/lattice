//! `StandaloneCpuSession`: one [`super::DecoderSession`] for the three standalone CPU
//! generation wrappers (`generate_f16`, `generate_q8`, `generate_q8_neon`).
//!
//! The wrappers differ only in the weight format and the single-token forward step they call,
//! so the session holds one [`StandaloneWeights`] value and dispatches on it. Everything else
//! the wrappers do is shared and lives here once: serial per-token prefill, the position and
//! `seq_len` bookkeeping, the prediction ledger, and sampling through `sample_token`.
//!
//! The session carries the wrappers' narrower contract. [`ExecutionCapabilities`] is all
//! `false`, so `driver::run` refuses grammar, logprobs, stop strings and a reasoning budget
//! before it calls the session, and `select` and `metadata` refuse the same controls if a
//! caller reaches them directly. `prepare_generation` with
//! `GenerationEntryContract::StandaloneCpu` still runs first and keeps its refusal text.
//!
//! Prefill is serial on purpose: each prompt token goes through the single-token forward step
//! at its own position, with `kv_cache.seq_len` advanced after every token but the last and
//! set to the prompt length at the end. That is what the wrappers do today, and it differs
//! from the batched prefill `QwenCpuSession` uses.
//!
//! Only the Q8 NEON format reserves KV capacity up front (`GenerationPlan::required_capacity`
//! times the full-attention KV width) and sizes its scratch for it. The f16 and Q8 wrappers
//! grow their caches on demand, and the session keeps that difference rather than unifying
//! allocation behaviour as a side effect.
//!
//! `generate_with_trace` is the entry the wrappers will call: it prepares the generation,
//! builds the session and runs `driver::run` over it, with no streaming, no cancellation and
//! no-op detokenizer hooks.

// Nothing outside the tests calls this module until the standalone wrappers route through the
// shared driver; the allowance is removed in the change that does that.
#![cfg_attr(not(test), allow(dead_code))]

use super::driver::{self, DriverTrace};
use super::{
    AcceptedToken, Cancellation, DecoderSession, ExecutionCapabilities, FinishDisposition,
    MetadataRequest, PredictionId, PredictionLedger, SelectOutcome, SelectionCandidate,
    SelectionRequest, StepStamp, TokenMetadata,
};
use crate::attention::gdn::GatedDeltaNetState;
use crate::error::InferenceError;
use crate::forward::cpu_f16::forward_step_f16;
use crate::forward::cpu_q8::forward_step_q8;
use crate::forward::neon_forward::{Q8NeonModel, forward_step_q8_neon};
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::model::qwen35::{
    ForwardScratch, GenerationEntryContract, GenerationPlan, GenerationPreparation, KvCache,
    decode_tokens, prepare_generation, sample_token,
};
use crate::model::qwen35_config::Qwen35Config;
use crate::rope::RopeTable;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::weights::f16_weights::F16ModelWeights;
use crate::weights::q8_weights::Q8ModelWeights;

/// Weights of one standalone wrapper, borrowed for the life of a session.
#[derive(Clone, Copy)]
pub(crate) enum StandaloneWeights<'w> {
    F16(&'w F16ModelWeights),
    Q8(&'w Q8ModelWeights),
    Q8Neon(&'w Q8NeonModel),
}

/// The standalone wrappers fail closed on all four optional controls, so none is declared.
const CAPABILITIES: ExecutionCapabilities = ExecutionCapabilities {
    grammar: false,
    logprobs: false,
    stop_strings: false,
    reasoning_budget: false,
};

pub(crate) struct StandaloneCpuSession<'w> {
    weights: StandaloneWeights<'w>,
    cfg: &'w Qwen35Config,
    rope: &'w RopeTable,
    gdn_states: Vec<GatedDeltaNetState>,
    kv_cache: KvCache,
    scratch: ForwardScratch,
    prompt_ids: Vec<u32>,
    prompt_len: usize,
    rng_state: u64,
    ledger: PredictionLedger,
}

impl<'w> StandaloneCpuSession<'w> {
    /// Allocates fresh recurrent state, KV cache and scratch for the planned prompt. The RNG
    /// state comes from the plan, which already normalized the request's seed.
    pub(crate) fn new(
        weights: StandaloneWeights<'w>,
        cfg: &'w Qwen35Config,
        rope: &'w RopeTable,
        plan: GenerationPlan,
    ) -> Self {
        let GenerationPlan {
            rng_state,
            prompt_ids,
            prompt_len,
            required_capacity,
        } = plan;

        let gdn_states: Vec<GatedDeltaNetState> = (0..cfg.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
        let mut scratch = ForwardScratch::new();

        if let StandaloneWeights::Q8Neon(_) = weights {
            kv_cache.reserve(required_capacity, cfg.full_kv_dim());
            scratch.ensure_capacity(cfg, required_capacity);
        }

        Self {
            weights,
            cfg,
            rope,
            gdn_states,
            kv_cache,
            scratch,
            prompt_ids,
            prompt_len,
            rng_state,
            ledger: PredictionLedger::new(),
        }
    }

    fn forward(&mut self, token_id: u32, position: usize) -> Result<(), InferenceError> {
        match self.weights {
            StandaloneWeights::F16(weights) => forward_step_f16(
                weights,
                self.cfg,
                self.rope,
                token_id,
                position,
                &mut self.gdn_states,
                &mut self.kv_cache,
                &mut self.scratch,
                None,
                None,
            ),
            StandaloneWeights::Q8(weights) => {
                forward_step_q8(
                    weights,
                    self.cfg,
                    self.rope,
                    token_id,
                    position,
                    &mut self.gdn_states,
                    &mut self.kv_cache,
                    &mut self.scratch,
                );
                Ok(())
            }
            StandaloneWeights::Q8Neon(model) => {
                forward_step_q8_neon(
                    model,
                    self.cfg,
                    self.rope,
                    token_id,
                    position,
                    &mut self.gdn_states,
                    &mut self.kv_cache,
                    &mut self.scratch,
                );
                Ok(())
            }
        }
    }
}

impl DecoderSession for StandaloneCpuSession<'_> {
    fn capabilities(&self) -> &ExecutionCapabilities {
        &CAPABILITIES
    }

    /// Runs every prompt token through the single-token forward step at its own position and
    /// leaves the last token's logits in `scratch.logits` for the first `select`. Opens no
    /// prediction, so the returned stamp carries none.
    fn prefill(&mut self, cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        if cancel.is_cancelled() {
            return Err(InferenceError::Inference("cancelled before prefill".into()));
        }

        for pos in 0..self.prompt_len {
            let token_id = self.prompt_ids[pos];
            self.forward(token_id, pos)?;
            if pos + 1 < self.prompt_len {
                self.kv_cache.seq_len += 1;
            }
        }
        self.kv_cache.seq_len = self.prompt_len;

        Ok(StepStamp {
            evaluated_len: self.kv_cache.seq_len,
            prediction: None,
        })
    }

    /// Consumes the accepted token's prediction before the forward pass, so a stale, foreign
    /// or already consumed id leaves the session state untouched, then evaluates the token at
    /// the current cache position.
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
        self.forward(accepted.final_id, pos)?;
        self.kv_cache.seq_len += 1;

        Ok(StepStamp {
            evaluated_len: self.kv_cache.seq_len,
            prediction: None,
        })
    }

    fn select(&mut self, request: &SelectionRequest<'_>) -> Result<SelectOutcome, InferenceError> {
        if request.grammar_mask.is_some() {
            return Err(InferenceError::InvalidInput(
                "session does not declare grammar support but a grammar mask was supplied".into(),
            ));
        }

        let candidate_id = sample_token(
            &self.scratch.logits[..self.cfg.vocab_size],
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

    fn metadata(
        &mut self,
        _prediction: PredictionId,
        _final_token: u32,
        _request: &MetadataRequest,
    ) -> Result<TokenMetadata, InferenceError> {
        Err(InferenceError::InvalidInput(
            "session does not declare logprobs support".into(),
        ))
    }

    fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError> {
        if disposition == FinishDisposition::Poisoned {
            self.ledger.invalidate();
        }
        Ok(())
    }
}

/// Prepares a generation under the standalone contract, builds the session for `weights` and
/// runs the shared driver over it.
pub(crate) fn generate_with_trace(
    weights: StandaloneWeights<'_>,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    rope: &RopeTable,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) -> Result<(GenerateOutput, DriverTrace), InferenceError> {
    let plan = match prepare_generation(
        tokenizer,
        prompt,
        gen_cfg,
        cfg.vocab_size,
        rope.max_positions(),
        GenerationEntryContract::StandaloneCpu,
    )? {
        GenerationPreparation::Ready(plan) => plan,
        GenerationPreparation::Complete(output) => return Ok((output, DriverTrace::default())),
    };
    let prompt_ids = plan.prompt_ids.clone();
    let prompt_len = plan.prompt_len;
    let mut session = StandaloneCpuSession::new(weights, cfg, rope, plan);

    let never_cancel = || false;
    let mut throwaway_text = String::new();
    let mut throwaway_offsets: Vec<usize> = Vec::new();
    let result = driver::run(
        &mut session,
        gen_cfg,
        None,
        &prompt_ids,
        cfg.eos_token_id,
        false,
        &never_cancel,
        |_generated_len| {},
        |_next_id| String::new(),
        &mut throwaway_text,
        &mut throwaway_offsets,
        |_delta, _next_id| true,
        || {},
        String::new,
    )?;

    let text = decode_tokens(tokenizer, &result.generated_ids);

    Ok((
        GenerateOutput {
            text,
            token_ids: result.generated_ids.clone(),
            prompt_tokens: prompt_len,
            generated_tokens: result.generated_ids.len(),
            stopped: result.stopped,
            stop_reason: Some(result.stop_reason),
            token_logprobs: result.token_logprobs,
        },
        result.trace,
    ))
}

/// Checks shared by the three wrapper test modules, whose model fixtures are private to each.
/// A wrapper's tests hand its fixture and its direct forward step to these helpers.
#[cfg(test)]
pub(crate) mod parity {
    use super::*;

    /// The wrapper's own single-token forward step, standing in for the session's dispatch.
    pub(crate) type ReferenceForward<'a> = dyn Fn(
            u32,
            usize,
            &mut [GatedDeltaNetState],
            &mut KvCache,
            &mut ForwardScratch,
        ) -> Result<(), InferenceError>
        + 'a;

    pub(crate) type LegacyGenerate<'a> =
        dyn Fn(&GenerateConfig) -> Result<GenerateOutput, InferenceError> + 'a;

    fn plan_for(
        cfg: &Qwen35Config,
        tokenizer: &BpeTokenizer,
        rope: &RopeTable,
        prompt: &str,
        gen_cfg: &GenerateConfig,
    ) -> GenerationPlan {
        match prepare_generation(
            tokenizer,
            prompt,
            gen_cfg,
            cfg.vocab_size,
            rope.max_positions(),
            GenerationEntryContract::StandaloneCpu,
        ) {
            Ok(GenerationPreparation::Ready(plan)) => plan,
            Ok(GenerationPreparation::Complete(_)) => {
                panic!("the request completes during preparation, so no session would run")
            }
            Err(error) => panic!("the request was refused during preparation: {error:?}"),
        }
    }

    /// The four configs the standalone wrappers pin generation goldens for, plus the step-0 stop
    /// and zero-budget cases that reach the driver's early returns.
    pub(crate) fn greedy_case() -> GenerateConfig {
        GenerateConfig {
            max_new_tokens: 6,
            temperature: 0.0,
            seed: Some(7),
            ..Default::default()
        }
    }

    pub(crate) fn deterministic_cases() -> Vec<(&'static str, GenerateConfig)> {
        let base = greedy_case();
        vec![
            ("greedy", base.clone()),
            (
                "stop_token",
                GenerateConfig {
                    stop_token_ids: vec![80],
                    ..base.clone()
                },
            ),
            (
                "stops_at_step_zero",
                GenerateConfig {
                    stop_token_ids: vec![36],
                    ..base.clone()
                },
            ),
            (
                "one_token",
                GenerateConfig {
                    max_new_tokens: 1,
                    ..base.clone()
                },
            ),
            (
                "zero_tokens",
                GenerateConfig {
                    max_new_tokens: 0,
                    ..base
                },
            ),
        ]
    }

    pub(crate) fn seeded_case() -> GenerateConfig {
        GenerateConfig {
            max_new_tokens: 6,
            temperature: 0.8,
            top_k: 5,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: Some(1234),
            ..Default::default()
        }
    }

    /// The session-driven result must equal the wrapper's own loop field for field, and the
    /// ledger counters must show the driver issued every step.
    pub(crate) fn assert_matches_legacy(
        legacy: &LegacyGenerate<'_>,
        weights: StandaloneWeights<'_>,
        cfg: &Qwen35Config,
        tokenizer: &BpeTokenizer,
        rope: &RopeTable,
        prompt: &str,
        name: &str,
        gen_cfg: &GenerateConfig,
    ) {
        let expected = legacy(gen_cfg).unwrap_or_else(|e| panic!("{name}: legacy loop: {e:?}"));
        let (actual, trace) = generate_with_trace(weights, cfg, tokenizer, rope, prompt, gen_cfg)
            .unwrap_or_else(|e| panic!("{name}: session: {e:?}"));

        assert_eq!(actual.token_ids, expected.token_ids, "{name}: token_ids");
        assert_eq!(actual.text, expected.text, "{name}: text");
        assert_eq!(
            actual.prompt_tokens, expected.prompt_tokens,
            "{name}: prompt_tokens"
        );
        assert_eq!(
            actual.generated_tokens, expected.generated_tokens,
            "{name}: generated_tokens"
        );
        assert_eq!(actual.stopped, expected.stopped, "{name}: stopped");
        assert_eq!(
            actual.stop_reason, expected.stop_reason,
            "{name}: stop_reason"
        );
        assert_eq!(
            actual.token_logprobs.len(),
            expected.token_logprobs.len(),
            "{name}: token_logprobs"
        );

        if gen_cfg.max_new_tokens > 0 {
            assert_eq!(
                trace.opened,
                actual.token_ids.len() + usize::from(actual.stopped),
                "{name}: one prediction is opened per sampled token, the stopping one included"
            );
            assert_eq!(
                trace.consumed + 1,
                trace.opened,
                "{name}: every prediction but the last is consumed by a decode step"
            );
        } else {
            assert_eq!(trace, DriverTrace::default(), "{name}: no step ran");
        }
    }

    /// Replays the session step by step against a hand-driven copy of the wrapper's own forward
    /// step and requires bit-identical logits at every position. Token outputs do not depend on
    /// the position and window the session feeds the forward step, the logits do.
    pub(crate) fn assert_logits_replay(
        weights: StandaloneWeights<'_>,
        cfg: &Qwen35Config,
        tokenizer: &BpeTokenizer,
        rope: &RopeTable,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        steps: usize,
        forward: &ReferenceForward<'_>,
    ) {
        let plan = plan_for(cfg, tokenizer, rope, prompt, gen_cfg);
        let prompt_ids = plan.prompt_ids.clone();
        let prompt_len = plan.prompt_len;
        let vocab = cfg.vocab_size;
        let mut session = StandaloneCpuSession::new(weights, cfg, rope, plan);

        let mut gdn_states: Vec<GatedDeltaNetState> = (0..cfg.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
        let mut scratch = ForwardScratch::new();

        for (pos, &token_id) in prompt_ids.iter().enumerate() {
            forward(token_id, pos, &mut gdn_states, &mut kv_cache, &mut scratch)
                .expect("reference prefill step");
            if pos + 1 < prompt_len {
                kv_cache.seq_len += 1;
            }
        }
        kv_cache.seq_len = prompt_len;

        let never = || false;
        let stamp = session.prefill(&never).expect("session prefill");
        assert_eq!(stamp.evaluated_len, kv_cache.seq_len, "prefill length");
        assert_logits_bits_equal(
            &session.scratch.logits[..vocab],
            &scratch.logits[..vocab],
            "prefill",
        );

        let mut history = prompt_ids;
        let mut previous: Vec<u32> = scratch.logits[..vocab]
            .iter()
            .map(|logit| logit.to_bits())
            .collect();
        for step in 0..steps {
            let request = SelectionRequest {
                config: gen_cfg,
                history: &history,
                grammar_mask: None,
            };
            let candidate = match session.select(&request).expect("session select") {
                SelectOutcome::Candidate(candidate) => candidate,
                SelectOutcome::GrammarExhausted => panic!("no grammar is set, so none can exhaust"),
            };
            history.push(candidate.candidate_id);

            let stamp = session
                .decode(
                    &AcceptedToken {
                        final_id: candidate.candidate_id,
                        prediction: candidate.prediction,
                    },
                    &never,
                )
                .expect("session decode");

            forward(
                candidate.candidate_id,
                kv_cache.seq_len,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
            )
            .expect("reference decode step");
            kv_cache.seq_len += 1;

            assert_eq!(
                stamp.evaluated_len, kv_cache.seq_len,
                "decode step {step}: evaluated length"
            );
            assert_logits_bits_equal(
                &session.scratch.logits[..vocab],
                &scratch.logits[..vocab],
                &format!("decode step {step}"),
            );

            let current: Vec<u32> = scratch.logits[..vocab]
                .iter()
                .map(|logit| logit.to_bits())
                .collect();
            assert_ne!(
                current, previous,
                "decode step {step}: the logits did not move, so the comparison above proves nothing"
            );
            previous = current;
        }
    }

    fn assert_logits_bits_equal(session: &[f32], reference: &[f32], what: &str) {
        assert_eq!(session.len(), reference.len(), "{what}: logits length");
        let differing = session
            .iter()
            .zip(reference)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing,
            0,
            "{what}: {differing} of {} logits differ in their bits",
            session.len()
        );
    }

    /// A prediction can be consumed once: the second decode with the same accepted token must be
    /// refused and must leave the cache position and logits as the first decode left them.
    pub(crate) fn assert_rejects_stale_prediction(
        weights: StandaloneWeights<'_>,
        cfg: &Qwen35Config,
        tokenizer: &BpeTokenizer,
        rope: &RopeTable,
        prompt: &str,
        gen_cfg: &GenerateConfig,
    ) {
        let plan = plan_for(cfg, tokenizer, rope, prompt, gen_cfg);
        let prompt_ids = plan.prompt_ids.clone();
        let vocab = cfg.vocab_size;
        let mut session = StandaloneCpuSession::new(weights, cfg, rope, plan);

        let never = || false;
        session.prefill(&never).expect("session prefill");
        let request = SelectionRequest {
            config: gen_cfg,
            history: &prompt_ids,
            grammar_mask: None,
        };
        let candidate = match session.select(&request).expect("session select") {
            SelectOutcome::Candidate(candidate) => candidate,
            SelectOutcome::GrammarExhausted => panic!("no grammar is set, so none can exhaust"),
        };
        let accepted = AcceptedToken {
            final_id: candidate.candidate_id,
            prediction: candidate.prediction,
        };

        session
            .decode(&accepted, &never)
            .expect("the first decode consumes the live prediction");
        let seq_len = session.kv_cache.seq_len;
        let logits: Vec<u32> = session.scratch.logits[..vocab]
            .iter()
            .map(|logit| logit.to_bits())
            .collect();

        match session.decode(&accepted, &never) {
            Err(InferenceError::Inference(message)) => assert!(
                message.contains("stale prediction id"),
                "unexpected refusal text: {message}"
            ),
            Err(other) => panic!("a stale prediction was refused with the wrong error: {other:?}"),
            Ok(_) => panic!("a consumed prediction was accepted a second time"),
        }
        assert_eq!(
            session.kv_cache.seq_len, seq_len,
            "the refused decode advanced the cache"
        );
        let after: Vec<u32> = session.scratch.logits[..vocab]
            .iter()
            .map(|logit| logit.to_bits())
            .collect();
        assert_eq!(after, logits, "the refused decode ran a forward pass");
    }

    /// A cancelled step fails before it touches the session: prefill leaves the cache empty, and
    /// a cancelled decode neither consumes the prediction nor moves the cache, so the same
    /// token still decodes afterwards.
    pub(crate) fn assert_cancellation_precedes_state_change(
        weights: StandaloneWeights<'_>,
        cfg: &Qwen35Config,
        tokenizer: &BpeTokenizer,
        rope: &RopeTable,
        prompt: &str,
        gen_cfg: &GenerateConfig,
    ) {
        let plan = plan_for(cfg, tokenizer, rope, prompt, gen_cfg);
        let prompt_ids = plan.prompt_ids.clone();
        let mut session = StandaloneCpuSession::new(weights, cfg, rope, plan);

        let cancelled = || true;
        let never = || false;
        // The Q8 NEON session sizes its scratch at construction, so an empty logits buffer is
        // not the untouched state; compare against the buffer as constructed instead.
        let logits_before: Vec<u32> = session
            .scratch
            .logits
            .iter()
            .map(|logit| logit.to_bits())
            .collect();
        assert!(session.prefill(&cancelled).is_err());
        assert_eq!(session.kv_cache.seq_len, 0, "cancelled prefill ran a step");
        let logits_after: Vec<u32> = session
            .scratch
            .logits
            .iter()
            .map(|logit| logit.to_bits())
            .collect();
        assert_eq!(
            logits_after, logits_before,
            "cancelled prefill ran a forward pass"
        );

        session.prefill(&never).expect("session prefill");
        let request = SelectionRequest {
            config: gen_cfg,
            history: &prompt_ids,
            grammar_mask: None,
        };
        let candidate = match session.select(&request).expect("session select") {
            SelectOutcome::Candidate(candidate) => candidate,
            SelectOutcome::GrammarExhausted => panic!("no grammar is set, so none can exhaust"),
        };
        let accepted = AcceptedToken {
            final_id: candidate.candidate_id,
            prediction: candidate.prediction,
        };
        let seq_len = session.kv_cache.seq_len;

        assert!(session.decode(&accepted, &cancelled).is_err());
        assert_eq!(
            session.kv_cache.seq_len, seq_len,
            "cancelled decode advanced the cache"
        );
        session
            .decode(&accepted, &never)
            .expect("the cancelled decode left the prediction live");
    }

    /// `select` and `metadata` refuse controls the session does not declare, and `select`
    /// still works for the same session when none is supplied.
    pub(crate) fn assert_refuses_undeclared_controls(
        weights: StandaloneWeights<'_>,
        cfg: &Qwen35Config,
        tokenizer: &BpeTokenizer,
        rope: &RopeTable,
        prompt: &str,
        gen_cfg: &GenerateConfig,
    ) {
        let plan = plan_for(cfg, tokenizer, rope, prompt, gen_cfg);
        let prompt_ids = plan.prompt_ids.clone();
        let mut session = StandaloneCpuSession::new(weights, cfg, rope, plan);
        let never = || false;
        session.prefill(&never).expect("session prefill");

        let mask = |_logits: &mut [f32]| -> Result<(), InferenceError> { Ok(()) };
        let masked = SelectionRequest {
            config: gen_cfg,
            history: &prompt_ids,
            grammar_mask: Some(&mask),
        };
        assert!(matches!(
            session.select(&masked),
            Err(InferenceError::InvalidInput(_))
        ));

        let plain = SelectionRequest {
            config: gen_cfg,
            history: &prompt_ids,
            grammar_mask: None,
        };
        let candidate = match session.select(&plain).expect("select without a mask") {
            SelectOutcome::Candidate(candidate) => candidate,
            SelectOutcome::GrammarExhausted => panic!("no grammar is set, so none can exhaust"),
        };
        let metadata = session.metadata(
            candidate.prediction,
            candidate.candidate_id,
            &MetadataRequest {
                top_logprobs: Some(0),
            },
        );
        assert!(matches!(metadata, Err(InferenceError::InvalidInput(_))));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn hollow_f16() -> F16ModelWeights {
        F16ModelWeights {
            embed_tokens: vec![],
            final_norm: vec![],
            layers: vec![],
        }
    }

    fn hollow_q8() -> Q8ModelWeights {
        Q8ModelWeights {
            embed_tokens: vec![],
            final_norm: vec![],
            layers: vec![],
        }
    }

    fn hollow_neon() -> Q8NeonModel {
        Q8NeonModel {
            embed_tokens: vec![],
            final_norm: vec![],
            lm_head_packed: vec![],
            lm_head_rows: 0,
            lm_head_cols: 0,
            layers: vec![],
        }
    }

    fn tokenizer() -> BpeTokenizer {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
            vocab.insert((*c).to_string(), i as u32);
        }
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("he".to_string(), "l".to_string()),
        ];
        BpeTokenizer::from_vocab_and_merges(vocab, merges).expect("test tokenizer builds")
    }

    fn plan(cfg: &Qwen35Config, rope: &RopeTable, max_new_tokens: usize) -> GenerationPlan {
        let gen_cfg = GenerateConfig {
            max_new_tokens,
            temperature: 0.0,
            seed: Some(7),
            ..Default::default()
        };
        match prepare_generation(
            &tokenizer(),
            "hello",
            &gen_cfg,
            cfg.vocab_size,
            rope.max_positions(),
            GenerationEntryContract::StandaloneCpu,
        ) {
            Ok(GenerationPreparation::Ready(plan)) => plan,
            other => panic!("the request must prepare a session: {other:?}"),
        }
    }

    fn with_each_hollow_session(check: impl Fn(&str, StandaloneCpuSession<'_>)) {
        let cfg = Qwen35Config::qwen35_2b();
        let rope = RopeTable::new(cfg.rope_dim(), 64, cfg.rope_theta);
        let f16 = hollow_f16();
        let q8 = hollow_q8();
        let neon = hollow_neon();
        for (name, weights) in [
            ("f16", StandaloneWeights::F16(&f16)),
            ("q8", StandaloneWeights::Q8(&q8)),
            ("q8_neon", StandaloneWeights::Q8Neon(&neon)),
        ] {
            let session = StandaloneCpuSession::new(weights, &cfg, &rope, plan(&cfg, &rope, 8));
            check(name, session);
        }
    }

    #[test]
    fn session_declares_none_of_the_optional_controls() {
        with_each_hollow_session(|name, session| {
            let caps = session.capabilities();
            assert_eq!(*caps, ExecutionCapabilities::default(), "{name}");
            assert!(!caps.grammar, "{name}: grammar");
            assert!(!caps.logprobs, "{name}: logprobs");
            assert!(!caps.stop_strings, "{name}: stop_strings");
            assert!(!caps.reasoning_budget, "{name}: reasoning_budget");
        });
    }

    /// Each control is refused by the driver's capability check, which runs before the first
    /// session call, so a session over empty weights is enough. The message text identifies the
    /// capability check as the refusing mechanism: the wrappers' own refusal wording differs.
    #[test]
    fn driver_refuses_every_unsupported_control_on_the_standalone_profile() {
        let grammar_spec = GrammarSpec::Gbnf("root ::= \"t\" | \"f\"\n".to_string());
        let engine = GrammarEngine::new(&grammar_spec, vec![b"t".to_vec(), b"f".to_vec()])
            .expect("trivial grammar compiles");

        let controls: Vec<(&str, GenerateConfig, &str)> = vec![
            (
                "grammar",
                GenerateConfig {
                    grammar: Some(Arc::new(engine)),
                    ..GenerateConfig::default()
                },
                "session does not declare grammar support but gen_cfg.grammar is set",
            ),
            (
                "logprobs",
                GenerateConfig {
                    logprobs: Some(0),
                    ..GenerateConfig::default()
                },
                "session does not declare logprobs support but gen_cfg.logprobs is set",
            ),
            (
                "stop_strings",
                GenerateConfig {
                    stop_strings: vec!["</s>".to_string()],
                    ..GenerateConfig::default()
                },
                "session does not declare stop_strings support but gen_cfg.stop_strings is set",
            ),
            (
                "reasoning_budget",
                GenerateConfig {
                    reasoning_budget: Some(16),
                    ..GenerateConfig::default()
                },
                "session does not declare reasoning_budget support but \
                 gen_cfg.reasoning_budget is set",
            ),
        ];

        with_each_hollow_session(|wrapper, mut session| {
            for (control, gen_cfg, message) in &controls {
                let never_cancel = || false;
                let mut text = String::new();
                let mut offsets: Vec<usize> = Vec::new();
                let result = driver::run(
                    &mut session,
                    gen_cfg,
                    None,
                    &[0, 1, 2],
                    999_999,
                    false,
                    &never_cancel,
                    |_| {},
                    |_| String::new(),
                    &mut text,
                    &mut offsets,
                    |_, _| true,
                    || {},
                    String::new,
                );
                match result {
                    Err(InferenceError::InvalidInput(actual)) => {
                        assert_eq!(&actual, message, "{wrapper}: {control}")
                    }
                    Err(other) => panic!("{wrapper}: {control}: wrong error {other:?}"),
                    Ok(_) => panic!("{wrapper}: {control}: the driver accepted the control"),
                }
            }
        });
    }

    #[test]
    fn neon_session_reserves_kv_for_prompt_plus_budget() {
        let cfg = Qwen35Config::qwen35_2b();
        let rope = RopeTable::new(cfg.rope_dim(), 64, cfg.rope_theta);
        let neon = hollow_neon();
        let plan = plan(&cfg, &rope, 8);
        let required = plan.prompt_len + 8 + 1;
        assert_eq!(plan.required_capacity, required, "plan capacity");
        assert!(
            cfg.num_full_attention_layers() > 0,
            "the fixture has no full-attention layer, so there is no cache to reserve"
        );

        let session =
            StandaloneCpuSession::new(StandaloneWeights::Q8Neon(&neon), &cfg, &rope, plan);

        let wanted = required * cfg.full_kv_dim();
        for (layer, (k, v)) in session
            .kv_cache
            .k
            .iter()
            .zip(&session.kv_cache.v)
            .enumerate()
        {
            assert!(k.capacity() >= wanted, "layer {layer}: k capacity");
            assert!(v.capacity() >= wanted, "layer {layer}: v capacity");
        }
        assert!(
            session.scratch.scores.len() >= cfg.num_attention_heads * (required + 1),
            "scratch was not sized for the planned capacity"
        );
    }

    #[test]
    fn q8_session_does_not_prereserve_kv() {
        let cfg = Qwen35Config::qwen35_2b();
        let rope = RopeTable::new(cfg.rope_dim(), 64, cfg.rope_theta);
        let q8 = hollow_q8();
        assert!(cfg.num_full_attention_layers() > 0);

        let session = StandaloneCpuSession::new(
            StandaloneWeights::Q8(&q8),
            &cfg,
            &rope,
            plan(&cfg, &rope, 8),
        );

        assert!(session.kv_cache.k.iter().all(|k| k.capacity() == 0));
        assert!(session.kv_cache.v.iter().all(|v| v.capacity() == 0));
        assert!(session.scratch.scores.is_empty());
    }

    #[test]
    fn f16_session_does_not_prereserve_kv() {
        let cfg = Qwen35Config::qwen35_2b();
        let rope = RopeTable::new(cfg.rope_dim(), 64, cfg.rope_theta);
        let f16 = hollow_f16();
        assert!(cfg.num_full_attention_layers() > 0);

        let session = StandaloneCpuSession::new(
            StandaloneWeights::F16(&f16),
            &cfg,
            &rope,
            plan(&cfg, &rope, 8),
        );

        assert!(session.kv_cache.k.iter().all(|k| k.capacity() == 0));
        assert!(session.kv_cache.v.iter().all(|v| v.capacity() == 0));
        assert!(session.scratch.scores.is_empty());
    }
}
