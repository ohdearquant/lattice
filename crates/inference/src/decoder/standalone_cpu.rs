//! `StandaloneCpuSession`: one [`super::DecoderSession`] for the three standalone CPU
//! generation wrappers (`generate_f16`, `generate_q8`, `generate_q8_neon`).
//!
//! The wrappers share the decode sequence and differ in weight format, forward step, and
//! KV/scratch allocation, so the session holds one [`StandaloneWeights`] value and dispatches
//! on it. The shared part lives here once: serial per-token prefill, the position and
//! `seq_len` bookkeeping, the prediction ledger, and sampling through `sample_token`.
//!
//! The session carries the wrappers' narrower contract. [`ExecutionCapabilities`] is all
//! `false`, so `driver::run` refuses grammar, logprobs, stop strings and a reasoning budget
//! before it calls the session. A caller that reaches the session directly gets the same
//! refusals from `select`, which rejects a grammar mask, logprobs, stop strings and a reasoning
//! budget, and from `metadata`, which rejects every logprobs request. `prepare_generation` with
//! `GenerationEntryContract::StandaloneCpu` still runs first and keeps its refusal text.
//!
//! A session runs one generation. `prefill` is refused the second time it is called, because
//! the recurrent state and KV cache it would write over belong to the first run; build a new
//! session for another generation.
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
//! `generate_with_trace` is the entry the wrappers call: it prepares the generation, builds
//! the session and runs `driver::run` over it, with no streaming, no cancellation and no-op
//! detokenizer hooks.

use super::driver::{self, DriverTrace};
use super::{
    AcceptedToken, Cancellation, DecoderSession, ExecutionCapabilities, FinishDisposition,
    MetadataRequest, PredictionId, PredictionLedger, SelectOutcome, SelectionCandidate,
    SelectionRequest, StepStamp, TokenMetadata,
};
use crate::attention::gdn::GatedDeltaNetState;
use crate::error::InferenceError;
use crate::forward::cpu_f16::forward_step_f16;
use crate::forward::cpu_q8::forward_step_q8_with_scratch;
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
use crate::weights::q8_weights::{Q8AttentionWeights, Q8MatmulScratch, Q8ModelWeights};

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
    q8_scratch: Q8MatmulScratch,
    prompt_ids: Vec<u32>,
    prompt_len: usize,
    rng_state: u64,
    ledger: PredictionLedger,
    prefilled: bool,
}

/// Resolves the Q8 matmul scratch column bound for `cfg` without allocating. The
/// constructor sizes the request scratch from this, and `generate_with_trace` runs it
/// ahead of `StandaloneCpuSession::new` so an overflowing config surfaces as a typed
/// error instead of the panic the infallible constructor would raise.
fn q8_scratch_max_cols(
    cfg: &Qwen35Config,
    model: &Q8ModelWeights,
) -> Result<usize, InferenceError> {
    let mut max_cols = cfg.hidden_size.max(cfg.intermediate_size);
    for (attention, _) in model.layers.iter().take(cfg.num_hidden_layers) {
        // Matrix metadata is checked later; it must not enlarge this early allocation.
        let columns = match attention {
            Q8AttentionWeights::Linear(_) => cfg.checked_linear_output_dim(),
            Q8AttentionWeights::Full(_) => cfg.checked_full_q_dim(),
        };
        max_cols = max_cols.max(columns?);
    }
    Ok(max_cols)
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

        let mut q8_scratch = Q8MatmulScratch::default();
        if let StandaloneWeights::Q8(model) = weights {
            // `generate_with_trace` resolves this first and propagates the typed error;
            // reaching the panic here means a caller built a session outside that entry.
            let max_cols = match q8_scratch_max_cols(cfg, model) {
                Ok(max_cols) => max_cols,
                Err(error) => panic!("{error}"),
            };
            if cfg.num_hidden_layers != 0
                && !model.layers.is_empty()
                && let Err(error) = q8_scratch.prepare(1, max_cols)
            {
                panic!("{error}");
            }
        }

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
            q8_scratch,
            prompt_ids,
            prompt_len,
            rng_state,
            ledger: PredictionLedger::new(),
            prefilled: false,
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
                forward_step_q8_with_scratch(
                    weights,
                    self.cfg,
                    self.rope,
                    token_id,
                    position,
                    &mut self.gdn_states,
                    &mut self.kv_cache,
                    &mut self.scratch,
                    &mut self.q8_scratch,
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
    ///
    /// The session runs one generation: a second call returns `InvalidInput` before it touches
    /// any state. A call refused for cancellation has not started, so it does not use up the
    /// session's one prefill.
    fn prefill(&mut self, cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError> {
        if self.prefilled {
            return Err(InferenceError::InvalidInput(
                "this session runs one generation; build a new session".into(),
            ));
        }
        if cancel.is_cancelled() {
            return Err(InferenceError::Inference("cancelled before prefill".into()));
        }
        self.prefilled = true;

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
        if request.config.logprobs.is_some() {
            return Err(InferenceError::InvalidInput(
                "session does not declare logprobs support but the selection config sets logprobs"
                    .into(),
            ));
        }
        if !request.config.stop_strings.is_empty() {
            return Err(InferenceError::InvalidInput(
                "session does not declare stop_strings support but the selection config sets \
                 stop_strings"
                    .into(),
            ));
        }
        if request.config.reasoning_budget.is_some() {
            return Err(InferenceError::InvalidInput(
                "session does not declare reasoning_budget support but the selection config sets \
                 reasoning_budget"
                    .into(),
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

    /// Ends the live prediction whatever the disposition, so a decode of the last selected
    /// token after `finish` is refused as stale. The session holds no other state to tear down.
    fn finish(&mut self, _disposition: FinishDisposition) -> Result<(), InferenceError> {
        self.ledger.invalidate();
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
    if let StandaloneWeights::Q8(model) = weights {
        // Validate the scratch sizing before the infallible constructor, which would
        // otherwise turn an overflowing config into the panic inside `new`.
        q8_scratch_max_cols(cfg, model)?;
    }
    let mut session = StandaloneCpuSession::new(weights, cfg, rope, plan);

    let never_cancel = || false;
    let mut throwaway_text = String::new();
    let mut throwaway_offsets: Vec<usize> = Vec::new();
    let result = driver::run(
        &mut session,
        gen_cfg,
        None,
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

#[cfg(test)]
#[path = "standalone_cpu_q8_scratch_checks.rs"]
pub(crate) mod q8_scratch_checks;

/// Checks shared by the three wrapper test modules, whose model fixtures are private to each.
/// A wrapper's tests hand its fixture and its direct forward step to these helpers.
#[cfg(test)]
pub(crate) mod parity;

#[cfg(test)]
mod tests;
