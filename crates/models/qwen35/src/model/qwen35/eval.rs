//! Language-model evaluation primitives for Qwen3.5: per-token NLLs and
//! strided sliding-window perplexity.
//!
//! Step 4a / 4b of [ADR-044](../../../../../docs/adr/ADR-044-quarot-rotated-quantization.md).
//! The CPU forward path's [`Qwen35Model::compute_perplexity`] is the baseline
//! that the Metal Q4 path ([`crate::forward::metal_qwen35::MetalQwen35State::compute_perplexity`])
//! compares against for the rotated-Q4 vs unrotated-Q4 delta measurement. Both
//! backends share [`run_strided_perplexity`] for the window walk + aggregation,
//! and [`log_softmax_nll`] for per-token cross-entropy.
//!
//! The strided methodology matches the [HuggingFace fixed-length-model perplexity
//! recipe](https://huggingface.co/docs/transformers/en/perplexity): each non-first
//! global token is scored exactly once. After the first window, every newly
//! scored target has **at least** `window - stride` and at most `window - 1`
//! preceding in-window tokens — overlap drives the lower bound up so the cold-
//! start bias of non-overlapping chunks is bounded out. Inside the first window,
//! targets ramp from 1 preceding token (target 1) to `window - 1`
//! (target `window - 1`). Context never crosses window boundaries: each window
//! resets the KV cache + GDN state.

use super::cache::{ForwardScratch, KvCache};
use super::model::Qwen35Model;
use crate::attention::gdn::{GatedDeltaNetState, qwen35_gdn_geometry};
use crate::forward::batch_prefill::not_cancelled;
use lattice_compute::forward::cpu::bf16_matmul::Bf16MatmulScratch;
use lattice_compute::forward::cpu::weight_matrix::WeightMatrix;
use lattice_compute::option_scoring::{OptionScores, check_letter_ids, option_scores_from_logits};
use lattice_inference_core::InferenceError;

/// **Unstable**: input for [`Qwen35Model::prefill_readout`].
#[derive(Debug)]
#[non_exhaustive]
pub enum PrefillInput<'a> {
    /// Raw token IDs from the model's vocabulary.
    TokenIds(&'a [u32]),
}

/// Project one captured hidden state onto the full vocabulary, then read the requested rows.
///
/// The returned values are read out of a full-vocabulary projection computed with the same call
/// [`Qwen35Model::forward_step`] makes, so each one equals the logit that call produces at this
/// position bit for bit on every CPU target. Projecting onto only the requested rows would not:
/// the CPU kernels round differently depending on the number of rows in the product.
fn project_readout_hidden(
    hidden: &[f32],
    head: &WeightMatrix,
    vocab: usize,
    rows: &[u32],
    full: &mut [f32],
    logits: &mut [f32],
    scratch: &mut Bf16MatmulScratch,
) -> Result<(), InferenceError> {
    let k = hidden.len();
    if k == 0 || vocab == 0 || !(1..=255).contains(&rows.len()) || logits.len() != rows.len() {
        return Err(InferenceError::Inference(
            "prefill_readout: inconsistent projection geometry".into(),
        ));
    }
    let head_span = vocab.checked_mul(k).ok_or_else(|| {
        InferenceError::Inference("prefill_readout: projection size overflow".into())
    })?;
    if head.len() < head_span {
        return Err(InferenceError::Inference(
            "prefill_readout: inconsistent head geometry".into(),
        ));
    }
    if full.len() < vocab {
        return Err(InferenceError::Inference(
            "prefill_readout: inconsistent projection geometry".into(),
        ));
    }
    let head_rows = head.len() / k;
    for &row_id in rows {
        let row = row_id as usize;
        if row >= vocab {
            return Err(InferenceError::Inference(format!(
                "prefill_readout: head row {row} out of range: head has {head_rows} rows"
            )));
        }
    }
    head.matmul_bt(hidden, &mut full[..vocab], 1, k, vocab, scratch);
    for (index, &row_id) in rows.iter().enumerate() {
        logits[index] = full[row_id as usize];
    }
    if let Some((index, &value)) = logits.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(InferenceError::Inference(format!(
            "prefill_readout: non-finite logit at selected row {index}: {value}"
        )));
    }
    Ok(())
}

/// Configuration for [`Qwen35Model::compute_perplexity`].
#[derive(Clone, Copy, Debug)]
pub struct PerplexityConfig {
    /// Context window length in tokens. Each window scores at most
    /// `window - 1` positions (the first token has no predecessor).
    pub window: usize,
    /// Tokens advanced between successive windows. Each global token is
    /// scored under the first window whose context covers it.
    pub stride: usize,
}

impl Default for PerplexityConfig {
    fn default() -> Self {
        Self {
            window: 512,
            stride: 256,
        }
    }
}

/// Result of [`Qwen35Model::compute_perplexity`].
#[derive(Clone, Debug)]
pub struct PerplexityReport {
    /// Perplexity: `exp(mean_nll)`. Lower is better.
    pub ppl: f64,
    /// Mean cross-entropy in nats per scored token: `total_nll / num_tokens_scored`.
    pub mean_nll: f64,
    /// Sum of NLLs across every scored token, in nats.
    pub total_nll: f64,
    /// Tokens that contributed to the average. With `stride < window` and
    /// `tokens.len() <= window` this equals `tokens.len() - 1`; otherwise
    /// it is the same value, since each non-first global token is scored
    /// exactly once.
    pub num_tokens_scored: usize,
    /// Number of windows the harness ran over the corpus.
    pub num_windows: usize,
    /// Window size passed in [`PerplexityConfig`], echoed for the report.
    pub window: usize,
    /// Stride passed in [`PerplexityConfig`], echoed for the report.
    pub stride: usize,
}

impl Qwen35Model {
    /// **Unstable**: compute per-position cross-entropy NLLs for an autoregressive
    /// forward pass over `tokens`.
    ///
    /// Returns a `Vec<f32>` of length `tokens.len() - 1`. The value at index `i`
    /// is `-log p(tokens[i + 1] | tokens[0..=i])` as computed by the model, where
    /// `p` is the softmax over the full vocabulary at decode step `i`.
    ///
    /// The first token has no predecessor to score against, so it is not
    /// represented in the output. The KV cache and GDN recurrent state are
    /// constructed fresh inside this function; the caller's state (if any) is
    /// not touched.
    ///
    /// Errors if `tokens.len() < 2`, if any `token_id >= cfg.vocab_size`, or
    /// if `tokens.len() > self.max_context()` (the precomputed RoPE table
    /// only covers positions `0..max_context`; the underlying `forward_step`
    /// would otherwise panic on out-of-range RoPE indexing).
    pub fn compute_token_nlls(&self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError> {
        let cfg = &self.config;
        if tokens.len() < 2 {
            return Err(InferenceError::Inference(format!(
                "compute_token_nlls: need at least 2 tokens, got {}",
                tokens.len()
            )));
        }
        let max_context = self.max_context();
        if tokens.len() > max_context {
            return Err(InferenceError::Inference(format!(
                "compute_token_nlls: tokens.len() ({}) exceeds RoPE capacity ({}); \
                 use a shorter window or load the model with a larger context table",
                tokens.len(),
                max_context
            )));
        }
        if let Some((bad_idx, &bad)) = tokens
            .iter()
            .enumerate()
            .find(|&(_, &t)| (t as usize) >= cfg.vocab_size)
        {
            return Err(InferenceError::Inference(format!(
                "compute_token_nlls: tokens[{bad_idx}]={bad} >= vocab_size {}",
                cfg.vocab_size
            )));
        }

        let num_linear = cfg.num_linear_attention_layers();
        let num_full = cfg.num_full_attention_layers();
        let gdn_geometry = qwen35_gdn_geometry(cfg)?;
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::from_geometry(&gdn_geometry))
            .collect();
        let mut kv_cache = KvCache::new(num_full);
        let mut scratch = ForwardScratch::new();

        let mut nlls = Vec::with_capacity(tokens.len() - 1);

        for (pos, &token_id) in tokens.iter().enumerate() {
            self.forward_step(token_id, pos, &mut gdn_states, &mut kv_cache, &mut scratch);

            if pos + 1 < tokens.len() {
                let next = tokens[pos + 1] as usize;
                let nll = log_softmax_nll(&scratch.logits[..cfg.vocab_size], next);
                nlls.push(nll);
            }

            if pos < tokens.len() - 1 {
                kv_cache.seq_len += 1;
            }
        }

        Ok(nlls)
    }

    /// **Unstable**: score a list of option letters with one prompt prefill (ADR-097).
    ///
    /// Runs the model over `prompt_ids` from a fresh recurrent and KV state, takes the logits at
    /// the last prompt position and reads them at `letter_ids`. There is no sampling and no decode
    /// step. The model is not modified: every call builds its own state, so repeat calls on the
    /// same inputs return bit-identical scores. This is the CPU counterpart of
    /// `MetalQwen35State::score_option_letters`; it takes `&self` because the CPU model keeps no
    /// session state.
    ///
    /// The logits come from [`Qwen35Model::forward_prompt_debug`], the existing serial
    /// last-position path, so they equal the logits a hand-driven `forward_step` loop leaves in
    /// its scratch buffer. See [`lattice_compute::option_scoring::resolve_option_letters`] for turning
    /// letters into ids.
    ///
    /// # Errors
    ///
    /// [`InferenceError::InvalidInput`] before any forward pass for an empty `prompt_ids`, a prompt
    /// longer than [`Self::max_context`], a prompt token outside the vocabulary, an empty
    /// `letter_ids`, a letter id outside the vocabulary or a repeated letter id.
    /// [`InferenceError::Inference`] when the last-position logits contain a non-finite value.
    pub fn score_option_letters(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<OptionScores, InferenceError> {
        self.check_option_scoring_input(prompt_ids, letter_ids)?;
        let logits = self.forward_prompt_debug(prompt_ids)?;
        option_scores_from_logits(&logits, letter_ids)
    }

    /// **Unstable**: [`Self::score_option_letters`] with the prompt run through the batched
    /// prefill that `generate` uses.
    ///
    /// The batched prefill runs each projection over the whole prompt as one matrix-matrix
    /// product instead of one matrix-vector product per token. Its last-position
    /// logits match the serial path's within floating-point rounding, not bit for bit, because
    /// the products accumulate in a different order. A configuration the batched prefill does
    /// not support (mixture-of-experts) takes the serial path, so the result then equals
    /// [`Self::score_option_letters`] exactly. Every call builds its own state.
    ///
    /// The batched prefill holds the whole prompt's activations at once, so its scratch memory
    /// grows linearly with the prompt length.
    ///
    /// # Errors
    ///
    /// The same as [`Self::score_option_letters`], plus any error the batched prefill returns
    /// for a prompt whose scratch would exceed addressable storage.
    pub fn score_option_letters_batched(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<OptionScores, InferenceError> {
        not_cancelled(self.score_option_letters_batched_with_cancel(
            prompt_ids,
            letter_ids,
            &mut || false,
        )?)
    }

    /// **Unstable**: [`Self::score_option_letters_batched`], asking `should_cancel` before each
    /// layer of the batched prefill. `Ok(None)` means it returned true and the row was not
    /// scored. The serial path a mixture-of-experts config takes does not ask it.
    ///
    /// # Errors
    ///
    /// The same as [`Self::score_option_letters_batched`].
    pub fn score_option_letters_batched_with_cancel(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Option<OptionScores>, InferenceError> {
        self.check_option_scoring_input(prompt_ids, letter_ids)?;
        let cfg = &self.config;
        let gdn_geometry = qwen35_gdn_geometry(cfg)?;
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..cfg.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::from_geometry(&gdn_geometry))
            .collect();
        let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
        let logits = match self.prefill_tokens_batched_for_generate_with_cancel(
            prompt_ids,
            &mut gdn_states,
            &mut kv_cache,
            should_cancel,
        ) {
            Ok(Some(logits)) => logits,
            Ok(None) => return Ok(None),
            Err(InferenceError::UnsupportedModel(_)) => self.forward_prompt_debug(prompt_ids)?,
            Err(error) => return Err(error),
        };
        option_scores_from_logits(&logits, letter_ids).map(Some)
    }

    /// **Unstable**: prefill a prompt prefix once, for scoring several prompts
    /// that continue it with [`Self::score_option_letters_after_prefix`].
    ///
    /// The prefix runs through the batched prefill and computes no logits.
    ///
    /// # Errors
    ///
    /// `InvalidInput` for an empty prefix, a token outside the vocabulary, or
    /// a prefix that leaves no position within the RoPE capacity for a
    /// continuation. `UnsupportedModel` for a mixture-of-experts config, which
    /// the batched prefill does not run.
    pub fn prefill_scoring_prefix(
        &self,
        prefix_ids: &[u32],
    ) -> Result<ScoringPrefix, InferenceError> {
        not_cancelled(self.prefill_scoring_prefix_with_cancel(prefix_ids, &mut || false)?)
    }

    /// **Unstable**: [`Self::prefill_scoring_prefix`], asking `should_cancel` before each layer
    /// of the prefill. `Ok(None)` means it returned true and no prefix was built.
    ///
    /// # Errors
    ///
    /// The same as [`Self::prefill_scoring_prefix`].
    pub fn prefill_scoring_prefix_with_cancel(
        &self,
        prefix_ids: &[u32],
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Option<ScoringPrefix>, InferenceError> {
        if prefix_ids.len() >= self.max_context() {
            return Err(InferenceError::InvalidInput(format!(
                "prefill_scoring_prefix: prefix_ids.len() ({}) leaves no position for a continuation within RoPE capacity ({})",
                prefix_ids.len(),
                self.max_context()
            )));
        }
        self.check_prompt_ids(prefix_ids)?;
        let cfg = &self.config;
        let gdn_geometry = qwen35_gdn_geometry(cfg)?;
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..cfg.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::from_geometry(&gdn_geometry))
            .collect();
        let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
        if !self.prefill_tokens_batched_state_only(
            prefix_ids,
            &mut gdn_states,
            &mut kv_cache,
            should_cancel,
        )? {
            return Ok(None);
        }
        Ok(Some(ScoringPrefix {
            ids: prefix_ids.to_vec(),
            gdn_states,
            kv_cache,
        }))
    }

    /// **Unstable**: score `letter_ids` at the last position of `prompt_ids`,
    /// a prompt that continues `prefix`, prefilling only the tokens after it.
    ///
    /// The prefix state is copied, so one prefix serves any number of calls.
    /// The scores match [`Self::score_option_letters_batched`] on the whole
    /// prompt within floating-point rounding.
    ///
    /// # Errors
    ///
    /// The checks of [`Self::score_option_letters`], and `InvalidInput` when
    /// `prompt_ids` does not start with the prefix or adds no token to it.
    pub fn score_option_letters_after_prefix(
        &self,
        prefix: &ScoringPrefix,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<OptionScores, InferenceError> {
        not_cancelled(self.score_option_letters_after_prefix_with_cancel(
            prefix,
            prompt_ids,
            letter_ids,
            &mut || false,
        )?)
    }

    /// **Unstable**: [`Self::score_option_letters_after_prefix`], asking `should_cancel`
    /// before each layer of the prefill. `Ok(None)` means it returned true and the row was not
    /// scored; `prefix` is unchanged either way.
    ///
    /// # Errors
    ///
    /// The same as [`Self::score_option_letters_after_prefix`].
    pub fn score_option_letters_after_prefix_with_cancel(
        &self,
        prefix: &ScoringPrefix,
        prompt_ids: &[u32],
        letter_ids: &[u32],
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Option<OptionScores>, InferenceError> {
        self.check_option_scoring_input(prompt_ids, letter_ids)?;
        let Some(suffix) = prompt_ids
            .strip_prefix(prefix.ids.as_slice())
            .filter(|suffix| !suffix.is_empty())
        else {
            return Err(InferenceError::InvalidInput(format!(
                "score_option_letters_after_prefix: the {}-token prompt does not continue the {}-token prefix with at least one token",
                prompt_ids.len(),
                prefix.ids.len()
            )));
        };
        let kv_dim = self.config.checked_full_kv_dim()?;
        let mut gdn_states = prefix.gdn_states.clone();
        let mut kv_cache = prefix.kv_cache.fork(prompt_ids.len(), kv_dim);
        let Some(logits) = self.prefill_tokens_batched_for_generate_with_cancel(
            suffix,
            &mut gdn_states,
            &mut kv_cache,
            should_cancel,
        )?
        else {
            return Ok(None);
        };
        option_scores_from_logits(&logits, letter_ids).map(Some)
    }

    /// **Unstable**: score several `(prompt_ids, letter_ids)` rows that share a
    /// prompt prefix, prefilling the shared prefix once.
    ///
    /// The longest prefix common to every row's prompt (leaving at least one
    /// token of each prompt after it) is prefilled once with
    /// [`Self::prefill_scoring_prefix`], and each row is scored from it with
    /// [`Self::score_option_letters_after_prefix`]. This is the layout of a
    /// multi-question decision request, where every row is the same state
    /// followed by one question. The scores match
    /// [`Self::score_option_letters_batched`] row by row within floating-point
    /// rounding.
    ///
    /// With a single row, or rows that share no prefix, each row is scored by
    /// [`Self::score_option_letters_batched`]. A mixture-of-experts config,
    /// which the batched prefill does not run, scores each row with the serial
    /// [`Self::score_option_letters`].
    ///
    /// # Errors
    ///
    /// Every row is checked as [`Self::score_option_letters`] checks its input
    /// before any forward pass; the first refusal is returned.
    pub fn score_option_rows_shared_prefix(
        &self,
        rows: &[(&[u32], &[u32])],
    ) -> Result<Vec<OptionScores>, InferenceError> {
        for (prompt_ids, letter_ids) in rows {
            self.check_option_scoring_input(prompt_ids, letter_ids)?;
        }
        if self.config.is_moe() {
            return rows
                .iter()
                .map(|(prompt_ids, letter_ids)| self.score_option_letters(prompt_ids, letter_ids))
                .collect();
        }
        let shared = shared_scoring_prefix_len(rows);
        if rows.len() < 2 || shared == 0 {
            return rows
                .iter()
                .map(|(prompt_ids, letter_ids)| {
                    self.score_option_letters_batched(prompt_ids, letter_ids)
                })
                .collect();
        }
        let prefix = self.prefill_scoring_prefix(&rows[0].0[..shared])?;
        rows.iter()
            .map(|(prompt_ids, letter_ids)| {
                self.score_option_letters_after_prefix(&prefix, prompt_ids, letter_ids)
            })
            .collect()
    }

    fn check_option_scoring_input(
        &self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<(), InferenceError> {
        if prompt_ids.is_empty() {
            return Err(InferenceError::InvalidInput(
                "score_option_letters: prompt_ids must not be empty".into(),
            ));
        }
        check_letter_ids(self.config.vocab_size, letter_ids)?;
        self.check_prompt_ids(prompt_ids)
    }

    fn check_prompt_ids(&self, prompt_ids: &[u32]) -> Result<(), InferenceError> {
        let vocab_size = self.config.vocab_size;
        if prompt_ids.is_empty() {
            return Err(InferenceError::InvalidInput(
                "score_option_letters: prompt_ids must not be empty".into(),
            ));
        }
        if prompt_ids.len() > self.max_context() {
            return Err(InferenceError::InvalidInput(format!(
                "score_option_letters: prompt_ids.len() ({}) exceeds RoPE capacity ({})",
                prompt_ids.len(),
                self.max_context()
            )));
        }
        if let Some((index, &token_id)) = prompt_ids
            .iter()
            .enumerate()
            .find(|&(_, &t)| t as usize >= vocab_size)
        {
            return Err(InferenceError::InvalidInput(format!(
                "score_option_letters: prompt_ids[{index}]={token_id} out of range: \
                 vocab_size is {vocab_size}"
            )));
        }
        Ok(())
    }

    /// **Unstable**: read selected logits at selected positions with one CPU prefill.
    ///
    /// Processes the complete input from fresh recurrent and KV state. Each result row
    /// corresponds to a position in `slots`, and contains the logits for `rows` in caller
    /// order. The hidden states are captured after final RMSNorm. Each slot's values are read
    /// out of a full-vocabulary projection computed with the same call `Self::forward_step`
    /// makes, so each equals the logit that call produces at this position bit for bit on every
    /// CPU target; projecting onto only the requested rows would not, because the CPU kernels
    /// round differently depending on the number of rows in the product. There is no sampling
    /// or decode step.
    ///
    /// # Errors
    ///
    /// [`InferenceError::InvalidInput`] before any forward pass for empty input, input longer
    /// than [`Self::max_context`], an input token outside the vocabulary, empty `slots`, a slot
    /// outside the input, slots that are not strictly increasing, empty `rows`, more than 255
    /// rows, a row outside the vocabulary, or duplicate rows.
    /// [`InferenceError::Inference`] for inconsistent head geometry or a non-finite logit.
    pub fn prefill_readout(
        &self,
        input: PrefillInput<'_>,
        slots: &[usize],
        rows: &[u32],
    ) -> Result<Vec<Vec<f32>>, InferenceError> {
        let PrefillInput::TokenIds(tokens) = input;
        let cfg = &self.config;
        if tokens.is_empty() {
            return Err(InferenceError::InvalidInput(
                "prefill_readout: tokens must not be empty".into(),
            ));
        }
        if tokens.len() > self.max_context() {
            return Err(InferenceError::InvalidInput(format!(
                "prefill_readout: tokens.len() ({}) exceeds RoPE capacity ({})",
                tokens.len(),
                self.max_context()
            )));
        }
        if let Some((index, &token_id)) = tokens
            .iter()
            .enumerate()
            .find(|&(_, &token)| token as usize >= cfg.vocab_size)
        {
            return Err(InferenceError::InvalidInput(format!(
                "prefill_readout: tokens[{index}]={token_id} out of range: vocab_size is {}",
                cfg.vocab_size
            )));
        }
        if slots.is_empty() {
            return Err(InferenceError::InvalidInput(
                "prefill_readout: slots must not be empty".into(),
            ));
        }
        if let Some(&slot) = slots.iter().find(|&&slot| slot >= tokens.len()) {
            return Err(InferenceError::InvalidInput(format!(
                "prefill_readout: slot {slot} out of range: input has {} tokens",
                tokens.len()
            )));
        }
        if slots.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(InferenceError::InvalidInput(
                "prefill_readout: slots must be strictly increasing".into(),
            ));
        }
        check_letter_ids(cfg.vocab_size, rows)?;
        if rows.len() > 255 {
            return Err(InferenceError::InvalidInput(format!(
                "prefill_readout: at most 255 rows are supported, got {}",
                rows.len()
            )));
        }

        let hidden = cfg.hidden_size;
        let mut full = vec![0.0; cfg.vocab_size];

        let gdn_geometry = qwen35_gdn_geometry(cfg)?;
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..cfg.num_linear_attention_layers())
            .map(|_| GatedDeltaNetState::from_geometry(&gdn_geometry))
            .collect();
        let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
        let mut scratch = ForwardScratch::new();
        let mut captured = vec![vec![0.0; hidden]; slots.len()];
        let mut next_slot = 0;
        for (position, &token_id) in tokens.iter().enumerate() {
            self.forward_hidden_step(
                token_id,
                position,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
            );
            if next_slot < slots.len() && position == slots[next_slot] {
                captured[next_slot].copy_from_slice(&scratch.hidden[..hidden]);
                next_slot += 1;
            }
            if position < tokens.len() - 1 {
                kv_cache.seq_len += 1;
            }
        }

        let mut result = vec![vec![0.0; rows.len()]; slots.len()];
        let head = self.weights.logits_weight();
        for (hidden_state, logits) in captured.iter().zip(&mut result) {
            project_readout_hidden(
                hidden_state,
                head,
                cfg.vocab_size,
                rows,
                &mut full,
                logits,
                &mut scratch.bf16,
            )?;
        }
        Ok(result)
    }

    /// **Unstable (train-backward)**: per-position final hidden states.
    ///
    /// Returns one `[hidden_size]` vector per input token: the residual stream
    /// after the final RMSNorm — i.e. the exact vector `lm_head` is applied to
    /// at that position. Base weights are frozen, so for lm_head-side LoRA these
    /// are fixed inputs and can be cached across optimizer steps.
    #[cfg(feature = "train-backward")]
    pub fn forward_final_hidden(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, InferenceError> {
        if tokens.len() < 2 {
            return Err(InferenceError::Inference(format!(
                "forward_final_hidden: need at least 2 tokens, got {}",
                tokens.len()
            )));
        }
        self.final_hidden_states(tokens)
    }

    /// Per-position final hidden states: one `[hidden_size]` vector per input
    /// token, the residual stream after the final RMSNorm, which is the exact
    /// vector `lm_head` is applied to at that position.
    ///
    /// This carries no feature gate. It is the shared core beneath both the
    /// training capture path (`Qwen35Model::forward_final_hidden`, which adds
    /// its own two-token minimum) and the stable pooled-embedding API
    /// ([`Qwen35Model::embed_tokens`]), so that a consumer building for
    /// inference only can reach hidden states without enabling `train-backward`.
    pub(crate) fn final_hidden_states(
        &self,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, InferenceError> {
        let cfg = &self.config;
        if tokens.is_empty() {
            return Err(InferenceError::Inference(
                "final_hidden_states: need at least 1 token, got 0".to_string(),
            ));
        }
        let max_context = self.max_context();
        if tokens.len() > max_context {
            return Err(InferenceError::Inference(format!(
                "final_hidden_states: tokens.len() ({}) exceeds RoPE capacity ({max_context})",
                tokens.len(),
            )));
        }
        if let Some((bad_idx, &bad)) = tokens
            .iter()
            .enumerate()
            .find(|&(_, &t)| (t as usize) >= cfg.vocab_size)
        {
            return Err(InferenceError::Inference(format!(
                "final_hidden_states: tokens[{bad_idx}]={bad} >= vocab_size {}",
                cfg.vocab_size
            )));
        }

        let hidden = cfg.hidden_size;
        let num_linear = cfg.num_linear_attention_layers();
        let num_full = cfg.num_full_attention_layers();
        let gdn_geometry = qwen35_gdn_geometry(cfg)?;
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::from_geometry(&gdn_geometry))
            .collect();
        let mut kv_cache = KvCache::new(num_full);
        let mut scratch = ForwardScratch::new();

        let mut hiddens = Vec::with_capacity(tokens.len());
        for (pos, &token_id) in tokens.iter().enumerate() {
            self.forward_step(token_id, pos, &mut gdn_states, &mut kv_cache, &mut scratch);
            hiddens.push(scratch.hidden[..hidden].to_vec());
            if pos < tokens.len() - 1 {
                kv_cache.seq_len += 1;
            }
        }

        Ok(hiddens)
    }

    /// **Unstable (train-backward)**: capture a Full-attention layer's I/O.
    ///
    /// Runs the real model forward over `tokens` and, for the given GQA `layer`,
    /// records per position the pre-input-layernorm residual `h_in` and the
    /// gated o_proj output. Returns `(inputs, outputs)`, each `[seq_len, hidden]`
    /// row-major. The diff test recomputes `normed = rms_norm(h_in)` then runs
    /// the materialised GQA forward; the trainer reuses `h_in` as the frozen
    /// residual. Verifies forward-matches-real-model, not just self-consistency.
    #[cfg(feature = "train-backward")]
    pub fn capture_attn_io(
        &self,
        tokens: &[u32],
        layer: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::Inference(
                "capture_attn_io: need at least 1 token".to_string(),
            ));
        }
        let cfg = &self.config;
        let max_context = self.max_context();
        if tokens.len() > max_context {
            return Err(InferenceError::Inference(format!(
                "capture_attn_io: tokens.len() ({}) exceeds RoPE capacity ({max_context})",
                tokens.len(),
            )));
        }
        if let Some((bad_idx, &bad)) = tokens
            .iter()
            .enumerate()
            .find(|&(_, &t)| (t as usize) >= cfg.vocab_size)
        {
            return Err(InferenceError::Inference(format!(
                "capture_attn_io: tokens[{bad_idx}]={bad} >= vocab_size {}",
                cfg.vocab_size
            )));
        }
        let num_linear = cfg.num_linear_attention_layers();
        let num_full = cfg.num_full_attention_layers();
        let gdn_geometry = qwen35_gdn_geometry(cfg)?;
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::from_geometry(&gdn_geometry))
            .collect();
        let mut kv_cache = KvCache::new(num_full);
        let mut scratch = ForwardScratch::new();
        scratch.capture_attn_layer = Some(layer);

        for (pos, &token_id) in tokens.iter().enumerate() {
            self.forward_step(token_id, pos, &mut gdn_states, &mut kv_cache, &mut scratch);
            if pos < tokens.len() - 1 {
                kv_cache.seq_len += 1;
            }
        }

        let hidden = cfg.hidden_size;
        let expected = tokens.len() * hidden;
        if scratch.captured_attn_input.len() != expected {
            return Err(InferenceError::Inference(format!(
                "capture_attn_io: layer {layer} captured {} elems, expected {expected} \
                 (is layer {layer} a Full-attention layer?)",
                scratch.captured_attn_input.len()
            )));
        }
        Ok((scratch.captured_attn_input, scratch.captured_attn_out))
    }

    /// **Unstable (train-backward)**: RoPE cos/sin tables for `seq_len` positions.
    ///
    /// Returns `(cos, sin)`, each `[seq_len * rope_dim/2]` row-major with layout
    /// `table[pos * half + i]`, matching `gqa_forward_with_cache`'s expectation
    /// and the real model's `apply_partial_rope` indexing (`cos_at(pos*half+i)`).
    ///
    /// Errors if `seq_len > self.max_context()`. The precomputed RoPE table only
    /// covers positions `0..max_context()`; without this guard the `cos_at`/`sin_at`
    /// calls inside the loop would panic on an out-of-range slice index.
    #[cfg(feature = "train-backward")]
    pub fn rope_cos_sin_tables(
        &self,
        seq_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), InferenceError> {
        let max_context = self.max_context();
        if seq_len > max_context {
            return Err(InferenceError::Inference(format!(
                "rope_cos_sin_tables: seq_len ({seq_len}) exceeds RoPE table capacity \
                 ({max_context}); load the model with a larger context table",
            )));
        }
        let half = self.config.rope_dim() / 2;
        let mut cos_t = Vec::with_capacity(seq_len * half);
        let mut sin_t = Vec::with_capacity(seq_len * half);
        for pos in 0..seq_len {
            for i in 0..half {
                cos_t.push(self.rope.cos_at(pos * half + i));
                sin_t.push(self.rope.sin_at(pos * half + i));
            }
        }
        Ok((cos_t, sin_t))
    }

    /// **Unstable**: compute strided sliding-window perplexity over `tokens`.
    ///
    /// Mirrors the HuggingFace fixed-length-model recipe: walk the corpus in
    /// overlapping windows of `cfg.window` tokens, advancing `cfg.stride` tokens
    /// per step, and aggregate NLLs across windows so each non-first global
    /// token is scored exactly once. Perplexity is
    /// `exp(total_nll / num_tokens_scored)`.
    ///
    /// Each window resets the KV cache + GDN state, so context never crosses a
    /// window boundary. After the first window, every newly scored target has
    /// **at least** `window - stride` and at most `window - 1` preceding
    /// in-window tokens. Inside the first window, targets ramp from 1 preceding
    /// token (target 1) up to `window - 1` (target `window - 1`); only the
    /// corpus's very first token is unscored, because it has no predecessor.
    ///
    /// `stride` must be strictly less than `window`. Equal values would
    /// produce disjoint windows whose first token (the global boundary token)
    /// has no predecessor inside its own window and would therefore go
    /// unscored — silently breaking the "scored exactly once" invariant.
    /// Callers that want maximally-disjoint windows should pass
    /// `stride = window - 1`.
    pub fn compute_perplexity(
        &self,
        tokens: &[u32],
        cfg: &PerplexityConfig,
    ) -> Result<PerplexityReport, InferenceError> {
        run_strided_perplexity(tokens, cfg, self.max_context(), |slice| {
            self.compute_token_nlls(slice)
        })
    }
}

/// Backend-agnostic strided sliding-window aggregator. Shared by
/// [`Qwen35Model::compute_perplexity`] (CPU forward path) and
/// [`crate::forward::metal_qwen35::MetalQwen35State::compute_perplexity`]
/// (Metal Q4 forward path) so both backends share the same validation,
/// window walk, and aggregation semantics.
///
/// `nll_fn(slice)` returns one NLL per non-first token of `slice` (i.e.
/// length `slice.len() - 1`). The aggregator owns the strided iteration,
/// the "score each non-first global token exactly once" invariant, and
/// the report construction. Each backend's `compute_token_nlls`
/// implementation is responsible for resetting recurrent state at the
/// start of each window — the aggregator hands disjoint slices but does
/// not touch the model.
///
/// `max_context` is the largest slice length the backend can serve in a
/// single window (the RoPE table size on CPU, the KV-cache capacity on
/// Metal). The aggregator validates the *effective* window
/// `min(cfg.window, tokens.len())` against this so an oversized window
/// on a short corpus still succeeds, matching HuggingFace's
/// `min(begin + max_length, seq_len)` clipping in the fixed-length-PPL
/// recipe.
pub(crate) fn run_strided_perplexity<F>(
    tokens: &[u32],
    cfg: &PerplexityConfig,
    max_context: usize,
    mut nll_fn: F,
) -> Result<PerplexityReport, InferenceError>
where
    F: FnMut(&[u32]) -> Result<Vec<f32>, InferenceError>,
{
    if cfg.window < 2 {
        return Err(InferenceError::Inference(format!(
            "compute_perplexity: window ({}) must be >= 2",
            cfg.window
        )));
    }
    if cfg.stride == 0 {
        return Err(InferenceError::Inference(
            "compute_perplexity: stride must be > 0".into(),
        ));
    }
    if cfg.stride >= cfg.window {
        return Err(InferenceError::Inference(format!(
            "compute_perplexity: stride ({}) must be < window ({}); \
             stride == window silently drops every window-boundary token",
            cfg.stride, cfg.window
        )));
    }
    if tokens.len() < 2 {
        return Err(InferenceError::Inference(format!(
            "compute_perplexity: need at least 2 tokens, got {}",
            tokens.len()
        )));
    }
    let effective_window = cfg.window.min(tokens.len());
    if effective_window > max_context {
        return Err(InferenceError::Inference(format!(
            "compute_perplexity: effective window ({} = min(window {}, tokens.len() {})) \
             exceeds backend context capacity ({}); use a shorter window or load the model \
             with a larger context table",
            effective_window,
            cfg.window,
            tokens.len(),
            max_context
        )));
    }

    let n = tokens.len();
    let mut total_nll: f64 = 0.0;
    let mut num_scored: usize = 0;
    let mut num_windows: usize = 0;
    // Highest global target index already scored (0 = nothing scored yet;
    // valid token indices are 1..n-1 inclusive, since target 0 is the
    // unscored first token).
    let mut last_scored: usize = 0;
    let mut begin: usize = 0;

    loop {
        let end = (begin + cfg.window).min(n);
        if end - begin < 2 {
            break;
        }

        let nlls = nll_fn(&tokens[begin..end])?;
        for (i, &nll) in nlls.iter().enumerate() {
            let global_target = begin + i + 1;
            if global_target > last_scored {
                total_nll += nll as f64;
                num_scored += 1;
            }
        }
        // Last NLL in this window predicts global token (end - 1).
        last_scored = end - 1;
        num_windows += 1;

        if end >= n {
            break;
        }
        begin += cfg.stride;
    }

    if num_scored == 0 {
        return Err(InferenceError::Inference(
            "compute_perplexity: scored zero tokens (corpus shorter than 2?)".into(),
        ));
    }

    let mean_nll = total_nll / (num_scored as f64);
    let ppl = mean_nll.exp();

    Ok(PerplexityReport {
        ppl,
        mean_nll,
        total_nll,
        num_tokens_scored: num_scored,
        num_windows,
        window: cfg.window,
        stride: cfg.stride,
    })
}

/// Numerically stable `-log softmax(logits)[target]`. Shared between the CPU
/// and Metal Q4 perplexity harnesses.
///
/// Computed as `log_sum_exp(logits) - logits[target]` with the standard
/// max-subtraction trick, so the running sum stays in `[0, vocab_size]`
/// even for logits with large magnitudes.
pub(crate) fn log_softmax_nll(logits: &[f32], target: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp: f64 = 0.0;
    for &l in logits {
        sum_exp += ((l - max) as f64).exp();
    }
    let log_sum_exp = (max as f64) + sum_exp.ln();
    (log_sum_exp - logits[target] as f64) as f32
}

/// **Unstable**: the model state after a prompt prefix, built by
/// [`Qwen35Model::prefill_scoring_prefix`] and read by
/// [`Qwen35Model::score_option_letters_after_prefix`].
#[derive(Clone)]
pub struct ScoringPrefix {
    ids: Vec<u32>,
    gdn_states: Vec<GatedDeltaNetState>,
    kv_cache: KvCache,
}

impl ScoringPrefix {
    /// The prefix's token ids.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
}

impl std::fmt::Debug for ScoringPrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScoringPrefix")
            .field("tokens", &self.ids.len())
            .finish_non_exhaustive()
    }
}

/// **Unstable**: the length of the prefix every row's prompt shares, capped
/// so that each prompt keeps at least one token after it. This is the prefix
/// [`Qwen35Model::score_option_rows_shared_prefix`] prefills once.
pub fn shared_scoring_prefix_len(rows: &[(&[u32], &[u32])]) -> usize {
    let Some(((first, _), rest)) = rows.split_first() else {
        return 0;
    };
    let mut len = first.len().saturating_sub(1);
    for (prompt_ids, _) in rest {
        len = len.min(prompt_ids.len().saturating_sub(1));
        len = first[..len]
            .iter()
            .zip(prompt_ids.iter())
            .take_while(|(a, b)| a == b)
            .count();
    }
    len
}

#[cfg(test)]
mod tests;
