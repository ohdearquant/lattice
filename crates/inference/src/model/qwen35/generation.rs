//! Qwen3.5 generation, streaming generation, prefill/decode loops, stop-streamer utilities, and stop-token helpers.
use super::cache::{ForwardScratch, KvCache};
use super::detokenize::{IncrementalDetokenizer, decode_tokens};
use super::model::Qwen35Model;
use super::sampling::sample_token;
use super::stop_strings::{
    StopStringMatcher, earliest_stop_match, earliest_stop_match_from, stop_scan_search_start,
};
use crate::attention::gdn::GatedDeltaNetState;
use crate::error::InferenceError;
use crate::grammar::pda::GrammarState;
use crate::model::qwen35_config::{
    GenerateConfig, GenerateOutput, Qwen35Config, TokenLogprob, decode_cap, force_close_think,
};
use crate::sampling::record_logprob;
use crate::stop_reason::StopReason;
use crate::tokenizer::common::Tokenizer;

/// Test-only toggle forcing the pre-delegation serial prefill path
/// (`prefill_tokens`) instead of `prefill_tokens_batched_for_generate`.
///
/// Exists solely so `generate` / `generate_streaming` tests can reproduce the
/// exact old-path token sequence in-process (no duplicated ~300-line copy of
/// `generate`'s body) and assert it against the new batched-prefill path.
/// Guarded by `#[cfg(test)]` end to end, so it does not exist in non-test
/// builds; production behaviour is unaffected. Tests that use it must hold
/// `SERIAL_PREFILL_TEST_LOCK` for the duration, since this is process-global
/// mutable state and `cargo test` runs tests in parallel by default.
#[cfg(test)]
pub(crate) static FORCE_SERIAL_PREFILL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Serializes tests that toggle `FORCE_SERIAL_PREFILL`, so they never race
/// against each other (racing against unrelated dense-model tests is
/// harmless: both prefill paths are required to produce identical output,
/// which is exactly the invariant this feature exists to preserve).
#[cfg(test)]
pub(crate) static SERIAL_PREFILL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
fn force_serial_prefill() -> bool {
    FORCE_SERIAL_PREFILL.load(std::sync::atomic::Ordering::SeqCst)
}

#[cfg(not(test))]
fn force_serial_prefill() -> bool {
    false
}

impl Qwen35Model {
    /// **Unstable**: autoregressive text generation with temperature/top-k/top-p sampling.
    pub fn generate(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, InferenceError> {
        let cfg = &self.config;

        let mut rng_state = initial_rng_state(gen_cfg.seed);

        let input = self.tokenizer.tokenize(prompt);
        let prompt_ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();
        let prompt_len = prompt_ids.len();

        if prompt_len == 0 {
            return Err(InferenceError::Inference("empty prompt".into()));
        }

        // max_new_tokens == 0 means "generate nothing": return before sampling so
        // we never emit a token the caller did not ask for. Mirrors the identical
        // guard in generate_streaming, which this function is otherwise a copy of.
        if gen_cfg.max_new_tokens == 0 {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: false,
                stop_reason: Some(StopReason::Length),
                token_logprobs: vec![],
            });
        }

        // Context preflight. apply_partial_rope indexes the precomputed cos/sin
        // table unchecked, so a position at or past max_context() is an
        // out-of-bounds slice access — a release panic, not a clean error. The
        // decode loop runs `1..decode_cap(reasoning_budget, max_new_tokens)`
        // (the budget-extended cap, equal to max_new_tokens when reasoning is
        // unbudgeted), reaching at most position prompt_len + cap - 2. We adopt
        // the stricter OpenAI-style "prompt plus requested completion fits the
        // window" bound prompt_len + cap <= max_context: it matches the HTTP
        // server (bin/lattice.rs) verbatim, so direct and HTTP generation agree
        // on when a request is too long. Strictly safe (it can only reject one
        // extra edge request, never admit a panic). Same guard in
        // generate_streaming — using decode_cap is what makes a budgeted request
        // (which decodes past max_new_tokens) preflight against its true reach.
        let max_context = self.max_context();
        let effective_new = decode_cap(gen_cfg.reasoning_budget, gen_cfg.max_new_tokens);
        if prompt_len.saturating_add(effective_new) > max_context {
            return Err(InferenceError::Inference(format!(
                "prompt ({prompt_len} tokens) plus max_new_tokens ({}) exceeds \
                 model context window ({max_context})",
                gen_cfg.max_new_tokens
            )));
        }

        let num_linear = cfg.num_linear_attention_layers();
        let num_full = cfg.num_full_attention_layers();
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let mut kv_cache = KvCache::new(num_full);
        let mut scratch = ForwardScratch::new();

        // Initialise per-request grammar state when grammar-constrained decoding
        // is requested. None when no grammar is set (zero-cost for unconstrained
        // generation). Mirrors the pattern in crate::generate::generate.
        let mut grammar_state: Option<GrammarState> =
            gen_cfg.grammar.as_ref().map(|g| g.initial_state());

        let mut generated_ids: Vec<u32> = Vec::with_capacity(effective_new);
        let mut all_ids = prompt_ids.clone();
        // Empty `Vec` costs no heap allocation until pushed to, so this is
        // zero-cost when `gen_cfg.logprobs` is `None` (the default path).
        let mut token_logprobs: Vec<TokenLogprob> = Vec::new();

        // Prompt prefill: try the batched (dense-config) path first, which
        // performs one layer pass over all prompt positions plus a single
        // final-token vocab projection, instead of `prompt_len` full
        // `forward_step` calls (each of which computes an unused vocab
        // projection for every non-final prompt token). Falls back to the
        // serial `prefill_tokens` loop for MoE (`UnsupportedModel`) *before*
        // any `gdn_states` / `kv_cache` mutation, so the fallback always
        // starts from pristine state. See
        // `Qwen35Model::prefill_tokens_batched_for_generate` for the
        // logits-equivalence argument.
        let prefill_logits: Vec<f32> = if force_serial_prefill() {
            // Test-only escape hatch (compiles to `false` unconditionally
            // outside `#[cfg(test)]`; see `force_serial_prefill` below) used by
            // the delegation parity test to reproduce the pre-delegation
            // behaviour for a byte-for-byte token comparison against the
            // batched path.
            prefill_tokens(
                self,
                &prompt_ids,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
            );
            kv_cache.seq_len = prompt_len;
            scratch.logits[..cfg.vocab_size].to_vec()
        } else {
            match self.prefill_tokens_batched_for_generate(
                &prompt_ids,
                &mut gdn_states,
                &mut kv_cache,
            ) {
                Ok(logits) => logits,
                Err(InferenceError::UnsupportedModel(_)) => {
                    prefill_tokens(
                        self,
                        &prompt_ids,
                        &mut gdn_states,
                        &mut kv_cache,
                        &mut scratch,
                    );
                    kv_cache.seq_len = prompt_len;
                    scratch.logits[..cfg.vocab_size].to_vec()
                }
                Err(e) => return Err(e),
            }
        };
        // `scratch` may not have been touched by the batched path (it only
        // mutates its own private `PrefillScratch`), so its `logits` buffer
        // can still be its initial zero-length `Vec::new()`. Ensure capacity
        // before copying the prefill result in, whichever path produced it.
        scratch.ensure_capacity(cfg, prompt_len);
        scratch.logits[..cfg.vocab_size].copy_from_slice(&prefill_logits);

        // Apply grammar mask on the post-prefill logit buffer before the first
        // sample. mask_logits sets every disallowed token to NEG_INFINITY in-place,
        // so the sampler only sees the grammar-permitted candidate set.
        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state) {
            engine.mask_logits(gs, &mut scratch.logits[..cfg.vocab_size]);
            // If the grammar blocked every token the sampler's non-finite-max
            // short-circuit would silently return token 0. Fail closed instead.
            if !has_finite_logit(&scratch.logits[..cfg.vocab_size]) {
                return Err(InferenceError::InvalidInput(
                    "grammar constraint blocked every token at step 0; \
                     no legal first token exists in the current grammar state"
                        .into(),
                ));
            }
        }

        let next_id = sample_token(
            &scratch.logits[..cfg.vocab_size],
            gen_cfg,
            &all_ids,
            &mut rng_state,
        );

        // Advance grammar state after sampling. advance() returns false when the
        // grammar has no valid continuation for the selected token, signalling the
        // end of grammar-constrained generation. Mirror the same early-return used
        // in crate::generate::generate for parity.
        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state)
            && !engine.advance(gs, next_id)
        {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: false,
                stop_reason: Some(StopReason::Grammar),
                token_logprobs: vec![],
            });
        }

        if should_stop_token(cfg, gen_cfg, next_id) {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: true,
                stop_reason: Some(StopReason::Eos),
                token_logprobs: vec![],
            });
        }

        generated_ids.push(next_id);
        all_ids.push(next_id);
        record_logprob(
            &mut token_logprobs,
            &scratch.logits[..cfg.vocab_size],
            next_id,
            gen_cfg.temperature,
            gen_cfg.logprobs,
        );

        // Budget forcing: resolve </think> once and seed thinking_closed from the
        // prefill token so budget=1 works. Mirrors generate_streaming exactly;
        // disabled (reasoning_budget=None) → None/false → no-op, byte-identical to
        // pre-feature behaviour (e2e-parity pinned). special_token_id resolves
        // </think> via the added-token map (special=false markers are present
        // there — distinct from the id_to_token detok path).
        let think_close_id = if gen_cfg.reasoning_budget.is_some() {
            self.tokenizer.special_token_id("</think>")
        } else {
            None
        };
        let thinking_closed_seed = Some(next_id) == think_close_id;

        if gen_cfg.stop_strings.is_empty() {
            // Fast path: no string-level stops. Behaviour byte-for-byte identical
            // to before this feature was added; the e2e-parity CI gate pins this.
            let (stopped, loop_stop_reason) = decode_loop(
                self,
                gen_cfg,
                &mut all_ids,
                &mut generated_ids,
                &mut rng_state,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
                &mut grammar_state,
                think_close_id,
                thinking_closed_seed,
                &mut token_logprobs,
            )?;

            let text = decode_tokens(&self.tokenizer, &generated_ids);

            Ok(GenerateOutput {
                text,
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                stopped,
                stop_reason: Some(loop_stop_reason),
                token_logprobs,
            })
        } else {
            // String-stop path: accumulate decoded text and check after every token.
            let mut detok = IncrementalDetokenizer::new();
            let first_delta = detok.push(&self.tokenizer, next_id);
            let mut full = first_delta;

            // Tracks, per recorded `token_logprobs` entry, the length of `full`
            // immediately after that token's delta landed — grown in lockstep
            // with token_logprobs (both gated on gen_cfg.logprobs.is_some()), so
            // a stop-string truncation can drop exactly the trailing entries
            // whose text didn't fully survive. See
            // `truncate_token_logprobs_to_retained_text`.
            let mut token_logprob_end_offsets: Vec<usize> = if token_logprobs.is_empty() {
                Vec::new()
            } else {
                vec![full.len()]
            };

            // Check stop strings after the first token.
            if let Some(hit) = earliest_stop_match(&full, &gen_cfg.stop_strings) {
                full.truncate(hit);
                truncate_token_logprobs_to_retained_text(
                    &mut token_logprobs,
                    &token_logprob_end_offsets,
                    hit,
                );
                // generated_ids already contains next_id; we cannot un-generate it,
                // so token_ids/generated_tokens reflect all tokens up to the match.
                return Ok(GenerateOutput {
                    text: full,
                    token_ids: generated_ids.clone(),
                    prompt_tokens: prompt_len,
                    generated_tokens: generated_ids.len(),
                    stopped: true,
                    stop_reason: Some(StopReason::Eos),
                    token_logprobs,
                });
            }

            let (stopped, loop_stop_reason) = decode_loop_with_stops(
                self,
                gen_cfg,
                &mut all_ids,
                &mut generated_ids,
                &mut rng_state,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
                &mut detok,
                &mut full,
                &mut grammar_state,
                think_close_id,
                thinking_closed_seed,
                &mut token_logprobs,
                &mut token_logprob_end_offsets,
            )?;

            Ok(GenerateOutput {
                text: full,
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                stopped,
                stop_reason: Some(loop_stop_reason),
                token_logprobs,
            })
        }
    }

    /// Streaming variant of [`generate`] — identical token sequence, but invokes
    /// `on_token` with incremental text deltas after each generated token.
    ///
    /// # Parity safety
    ///
    /// The body below is a deliberate copy of `generate` rather than a refactor of
    /// the shared path. This ensures that no change here can silently alter the
    /// non-streaming `generate` path, which is pinned by the e2e-parity CI gate
    /// (greedy token match vs HF transformers). `on_token` is the only addition.
    ///
    /// `should_cancel = || false` convenience form of
    /// [`Self::generate_streaming_with_cancel`]; both share this one
    /// implementation.
    pub fn generate_streaming(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<GenerateOutput, InferenceError> {
        self.generate_streaming_with_cancel(
            prompt,
            gen_cfg,
            |delta| {
                on_token(delta);
                true
            },
            || false,
        )
    }

    /// Cancellation-aware sibling of [`Self::generate_streaming`] (ADR-080 C2,
    /// ports the Metal `MetalQwen35State::generate_streaming_with_cancel`
    /// contract to the CPU backend, closing #744: previously `lattice.rs`'s CPU
    /// streaming path had no way to observe a client disconnect at all and ran
    /// to the token cap after the client left).
    ///
    /// `should_cancel` is polled independently of `on_token`: before the
    /// prefill pass starts, immediately after it returns, and at the top of
    /// every decode iteration — all before any further work runs for that
    /// step. `on_token` itself also stops generation the moment it returns
    /// `false` (the caller could not forward the delta, e.g. the SSE receiver
    /// was dropped). Either signal short-circuits the trailing
    /// incomplete-UTF-8 flush too, since the caller is no longer consuming
    /// the stream by then. Both stopping paths report
    /// `stopped: false, stop_reason: Some(StopReason::Interrupt)` — a
    /// cancellation is not an OpenAI "stop condition", matching the Metal
    /// contract exactly.
    pub fn generate_streaming_with_cancel<F, C>(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        mut on_token: F,
        mut should_cancel: C,
    ) -> Result<GenerateOutput, InferenceError>
    where
        F: FnMut(&str) -> bool,
        C: FnMut() -> bool,
    {
        let cfg = &self.config;

        let mut rng_state = initial_rng_state(gen_cfg.seed);

        let input = self.tokenizer.tokenize(prompt);
        let prompt_ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();
        let prompt_len = prompt_ids.len();

        if prompt_len == 0 {
            return Err(InferenceError::Inference("empty prompt".into()));
        }

        // max_new_tokens == 0 means "generate nothing": return before sampling so
        // we never emit a token the caller did not ask for.
        if gen_cfg.max_new_tokens == 0 {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: false,
                stop_reason: Some(StopReason::Length),
                token_logprobs: vec![],
            });
        }

        // Context preflight: see generate() for the full rationale and the exact
        // vs. adopted-bound discussion. apply_partial_rope indexes the RoPE table
        // unchecked, so a request past max_context() would panic in the decode
        // loop; this mirrors the HTTP server's total-token contract verbatim.
        // decode_cap accounts for a budgeted request decoding past max_new_tokens.
        let max_context = self.max_context();
        let effective_new = decode_cap(gen_cfg.reasoning_budget, gen_cfg.max_new_tokens);
        if prompt_len.saturating_add(effective_new) > max_context {
            return Err(InferenceError::Inference(format!(
                "prompt ({prompt_len} tokens) plus max_new_tokens ({}) exceeds \
                 model context window ({max_context})",
                gen_cfg.max_new_tokens
            )));
        }

        let num_linear = cfg.num_linear_attention_layers();
        let num_full = cfg.num_full_attention_layers();
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let mut kv_cache = KvCache::new(num_full);
        let mut scratch = ForwardScratch::new();

        // Per-request grammar state, mirroring the generate() path above.
        let mut grammar_state: Option<GrammarState> =
            gen_cfg.grammar.as_ref().map(|g| g.initial_state());

        let mut generated_ids: Vec<u32> = Vec::with_capacity(effective_new);
        let mut all_ids = prompt_ids.clone();
        // Empty `Vec` costs no heap allocation until pushed to, so this is
        // zero-cost when `gen_cfg.logprobs` is `None` (the default path).
        let mut token_logprobs: Vec<TokenLogprob> = Vec::new();

        // Checked independently of `on_token`: a client that disconnected
        // between dequeue and here must not pay for the (potentially large)
        // prefill pass below. Mirrors the Metal
        // `generate_streaming_with_cancel`'s first `should_cancel` checkpoint.
        if should_cancel() {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: false, // caller interrupted the stream, not a stop condition
                stop_reason: Some(StopReason::Interrupt),
                token_logprobs: vec![],
            });
        }

        // Prompt prefill: try the batched (dense-config) path first, which
        // performs one layer pass over all prompt positions plus a single
        // final-token vocab projection, instead of `prompt_len` full
        // `forward_step` calls (each of which computes an unused vocab
        // projection for every non-final prompt token). Falls back to the
        // serial `prefill_tokens` loop for MoE (`UnsupportedModel`) *before*
        // any `gdn_states` / `kv_cache` mutation, so the fallback always
        // starts from pristine state. See
        // `Qwen35Model::prefill_tokens_batched_for_generate` for the
        // logits-equivalence argument.
        let prefill_logits: Vec<f32> = if force_serial_prefill() {
            // Test-only escape hatch (compiles to `false` unconditionally
            // outside `#[cfg(test)]`; see `force_serial_prefill` below) used by
            // the delegation parity test to reproduce the pre-delegation
            // behaviour for a byte-for-byte token comparison against the
            // batched path.
            prefill_tokens(
                self,
                &prompt_ids,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
            );
            kv_cache.seq_len = prompt_len;
            scratch.logits[..cfg.vocab_size].to_vec()
        } else {
            match self.prefill_tokens_batched_for_generate(
                &prompt_ids,
                &mut gdn_states,
                &mut kv_cache,
            ) {
                Ok(logits) => logits,
                Err(InferenceError::UnsupportedModel(_)) => {
                    prefill_tokens(
                        self,
                        &prompt_ids,
                        &mut gdn_states,
                        &mut kv_cache,
                        &mut scratch,
                    );
                    kv_cache.seq_len = prompt_len;
                    scratch.logits[..cfg.vocab_size].to_vec()
                }
                Err(e) => return Err(e),
            }
        };
        // `scratch` may not have been touched by the batched path (it only
        // mutates its own private `PrefillScratch`), so its `logits` buffer
        // can still be its initial zero-length `Vec::new()`. Ensure capacity
        // before copying the prefill result in, whichever path produced it.
        scratch.ensure_capacity(cfg, prompt_len);
        scratch.logits[..cfg.vocab_size].copy_from_slice(&prefill_logits);

        // The prefill call itself cannot be interrupted mid-flight, so this
        // is the earliest point a disconnect that happened *during* prefill
        // can be observed -- before paying for grammar masking or sampling
        // on its output. Mirrors the Metal `generate_streaming_with_cancel`'s
        // second `should_cancel` checkpoint.
        if should_cancel() {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: false, // caller interrupted the stream, not a stop condition
                stop_reason: Some(StopReason::Interrupt),
                token_logprobs: vec![],
            });
        }

        // Grammar mask on the post-prefill logits, identical to the generate() path.
        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state) {
            engine.mask_logits(gs, &mut scratch.logits[..cfg.vocab_size]);
            if !has_finite_logit(&scratch.logits[..cfg.vocab_size]) {
                return Err(InferenceError::InvalidInput(
                    "grammar constraint blocked every token at step 0; \
                     no legal first token exists in the current grammar state"
                        .into(),
                ));
            }
        }

        let next_id = sample_token(
            &scratch.logits[..cfg.vocab_size],
            gen_cfg,
            &all_ids,
            &mut rng_state,
        );

        // Grammar advance after sampling the first token, mirroring generate().
        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state)
            && !engine.advance(gs, next_id)
        {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: false,
                stop_reason: Some(StopReason::Grammar),
                token_logprobs: vec![],
            });
        }

        if should_stop_token(cfg, gen_cfg, next_id) {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: true,
                stop_reason: Some(StopReason::Eos),
                token_logprobs: vec![],
            });
        }

        generated_ids.push(next_id);
        all_ids.push(next_id);
        record_logprob(
            &mut token_logprobs,
            &scratch.logits[..cfg.vocab_size],
            next_id,
            gen_cfg.temperature,
            gen_cfg.logprobs,
        );

        // Budget forcing setup: resolve the </think> token id once and seed
        // the thinking_closed state from the prefill token so budget=1 works.
        let think_close_id = if gen_cfg.reasoning_budget.is_some() {
            self.tokenizer.special_token_id("</think>")
        } else {
            None
        };
        let mut thinking_closed = Some(next_id) == think_close_id;
        // Tracks generated_ids length at the moment </think> was emitted, for the
        // answer-budget break. None when reasoning_budget is disabled (parity-safe).
        let mut reasoning_end_len: Option<usize> = None;
        // Capture close-point after prefill push (covers budget=1 edge case).
        if thinking_closed && reasoning_end_len.is_none() {
            reasoning_end_len = Some(generated_ids.len());
        }

        // Incremental detokenization: emit only complete-UTF-8 text deltas. A
        // byte-level BPE codepoint can span several tokens, so we buffer raw bytes
        // and never stream a partial codepoint (see IncrementalDetokenizer).
        let mut detok = IncrementalDetokenizer::new();

        if gen_cfg.stop_strings.is_empty() {
            // Fast path: no string-level stops. Behaviour byte-for-byte identical
            // to before this feature was added; the e2e-parity CI gate pins this.
            // `text` is the caller-owned full output — the detokenizer itself only
            // retains a small undecided UTF-8 boundary tail (see IncrementalDetokenizer).
            let mut text = String::new();
            let delta = detok.push(&self.tokenizer, next_id);
            if !delta.is_empty() {
                text.push_str(&delta);
                if !on_token(&delta) {
                    return Ok(GenerateOutput {
                        text,
                        token_ids: generated_ids.clone(),
                        prompt_tokens: prompt_len,
                        generated_tokens: generated_ids.len(),
                        stopped: false, // caller interrupted the stream, not a stop condition
                        stop_reason: Some(StopReason::Interrupt),
                        token_logprobs,
                    });
                }
            }

            let mut stopped = false;
            let mut stopped_by_caller = false;
            let mut stop_reason = StopReason::Length;
            // Decode loop (mirrors decode_loop free function exactly).
            // cap = rb + max_new_tokens when budgeting; max_new_tokens otherwise (parity-safe).
            let cap = decode_cap(gen_cfg.reasoning_budget, gen_cfg.max_new_tokens);
            for _ in 1..cap {
                // Checked before any per-step work, independent of whether this
                // iteration's delta ends up non-empty -- closes the gap where a
                // run of tokens decoding to an incomplete UTF-8 tail would
                // otherwise never reach the on_token check below.
                if should_cancel() {
                    stopped_by_caller = true;
                    stop_reason = StopReason::Interrupt;
                    break;
                }
                let pos = kv_cache.seq_len;
                let Some(&last_token) = all_ids.last() else {
                    return Err(InferenceError::Inference("empty generation state".into()));
                };

                self.forward_step(
                    last_token,
                    pos,
                    &mut gdn_states,
                    &mut kv_cache,
                    &mut scratch,
                );
                kv_cache.seq_len += 1;

                // Grammar mask before sampling; fail closed on an all-blocked step.
                if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state) {
                    engine.mask_logits(gs, &mut scratch.logits[..cfg.vocab_size]);
                    if !has_finite_logit(&scratch.logits[..cfg.vocab_size]) {
                        return Err(InferenceError::InvalidInput(
                            "grammar constraint blocked every token; \
                             no legal continuation exists in the current grammar state"
                                .into(),
                        ));
                    }
                }

                let sampled_id = sample_token(
                    &scratch.logits[..cfg.vocab_size],
                    gen_cfg,
                    &all_ids,
                    &mut rng_state,
                );

                // Budget forcing: override sampled token with </think> when the
                // reasoning budget is exhausted and the block is still open.
                let next_id = force_close_think(
                    gen_cfg.reasoning_budget,
                    gen_cfg.enable_thinking,
                    thinking_closed,
                    generated_ids.len(),
                    think_close_id,
                )
                .unwrap_or(sampled_id);

                // Grammar advance on the actually-emitted token (after any budget
                // override): a false return signals grammar completion. Set
                // stopped=true before breaking so the caller sees a grammar-terminal
                // stop as stopped=true, matching decode_loop's `return Ok(true)`.
                if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state)
                    && !engine.advance(gs, next_id)
                {
                    stopped = true;
                    stop_reason = StopReason::Grammar;
                    break;
                }

                // Track when the thinking block closes (natural or forced).
                if Some(next_id) == think_close_id {
                    thinking_closed = true;
                }

                if should_stop_token(cfg, gen_cfg, next_id) {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                    break;
                }

                generated_ids.push(next_id);
                all_ids.push(next_id);
                record_logprob(
                    &mut token_logprobs,
                    &scratch.logits[..cfg.vocab_size],
                    next_id,
                    gen_cfg.temperature,
                    gen_cfg.logprobs,
                );
                // Capture close-point after push so </think> is the last reasoning token.
                if thinking_closed && reasoning_end_len.is_none() {
                    reasoning_end_len = Some(generated_ids.len());
                }

                let delta = detok.push(&self.tokenizer, next_id);
                if !delta.is_empty() {
                    text.push_str(&delta);
                    if !on_token(&delta) {
                        stopped_by_caller = true;
                        stop_reason = StopReason::Interrupt;
                        break;
                    }
                }

                // Answer-budget break: stop once max_new_tokens answer tokens follow </think>.
                if let Some(end) = reasoning_end_len
                    && generated_ids.len().saturating_sub(end) >= gen_cfg.max_new_tokens
                {
                    break;
                }
            }

            // Flush any trailing incomplete bytes (generation truncated mid-codepoint)
            // so the streamed deltas concatenate to exactly the returned text. Skip
            // when the caller asked to stop -- it is no longer consuming the stream.
            if !stopped_by_caller {
                let tail = detok.finish();
                if !tail.is_empty() {
                    text.push_str(&tail);
                    on_token(&tail);
                }
            }

            Ok(GenerateOutput {
                text,
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                stopped,
                stop_reason: Some(stop_reason),
                token_logprobs,
            })
        } else {
            // String-stop path: use StopStreamer to hold back (max_stop - 1) bytes and
            // never emit a partial stop prefix before we can confirm it is not a match.
            let mut streamer = StopStringMatcher::new(&gen_cfg.stop_strings);

            // `StopStringMatcher::push`'s sink is `FnMut(&str)` (no return
            // value), so a caller-requested stop is threaded through a
            // captured flag instead -- mirrors the Metal
            // `generate_streaming_with_cancel`'s `caller_interrupted` idiom
            // exactly, for the same structural reason.
            let mut caller_interrupted = false;
            let first_delta = detok.push(&self.tokenizer, next_id);
            let stop_matched = streamer.push(&first_delta, &mut |s| {
                if !caller_interrupted && !on_token(s) {
                    caller_interrupted = true;
                }
            });
            if caller_interrupted {
                return Ok(GenerateOutput {
                    text: streamer.into_text(),
                    token_ids: generated_ids.clone(),
                    prompt_tokens: prompt_len,
                    generated_tokens: generated_ids.len(),
                    stopped: false, // caller interrupted the stream, not a stop condition
                    stop_reason: Some(StopReason::Interrupt),
                    token_logprobs,
                });
            }
            if stop_matched {
                // Stop matched in the very first token.
                // token_ids already contain next_id; cannot un-generate it.
                return Ok(GenerateOutput {
                    text: streamer.into_text(),
                    token_ids: generated_ids.clone(),
                    prompt_tokens: prompt_len,
                    generated_tokens: generated_ids.len(),
                    stopped: true,
                    stop_reason: Some(StopReason::Eos),
                    token_logprobs,
                });
            }

            let mut stopped = false;
            let mut stopped_by_caller = false;
            let mut stop_reason = StopReason::Length;
            // Decode loop for the string-stop path.
            // cap = rb + max_new_tokens when budgeting; max_new_tokens otherwise (parity-safe).
            let cap = decode_cap(gen_cfg.reasoning_budget, gen_cfg.max_new_tokens);
            for _ in 1..cap {
                if should_cancel() {
                    stopped_by_caller = true;
                    stop_reason = StopReason::Interrupt;
                    break;
                }
                let pos = kv_cache.seq_len;
                let Some(&last_token) = all_ids.last() else {
                    return Err(InferenceError::Inference("empty generation state".into()));
                };

                self.forward_step(
                    last_token,
                    pos,
                    &mut gdn_states,
                    &mut kv_cache,
                    &mut scratch,
                );
                kv_cache.seq_len += 1;

                // Grammar mask before sampling; fail closed on an all-blocked step.
                if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state) {
                    engine.mask_logits(gs, &mut scratch.logits[..cfg.vocab_size]);
                    if !has_finite_logit(&scratch.logits[..cfg.vocab_size]) {
                        return Err(InferenceError::InvalidInput(
                            "grammar constraint blocked every token; \
                             no legal continuation exists in the current grammar state"
                                .into(),
                        ));
                    }
                }

                let sampled_id = sample_token(
                    &scratch.logits[..cfg.vocab_size],
                    gen_cfg,
                    &all_ids,
                    &mut rng_state,
                );

                // Budget forcing: override sampled token with </think> when the
                // reasoning budget is exhausted and the block is still open.
                let next_id = force_close_think(
                    gen_cfg.reasoning_budget,
                    gen_cfg.enable_thinking,
                    thinking_closed,
                    generated_ids.len(),
                    think_close_id,
                )
                .unwrap_or(sampled_id);

                // Grammar advance on the actually-emitted token (after any budget
                // override): break cleanly when the grammar signals completion.
                if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state)
                    && !engine.advance(gs, next_id)
                {
                    stopped = true;
                    stop_reason = StopReason::Grammar;
                    break;
                }

                // Track when the thinking block closes (natural or forced).
                if Some(next_id) == think_close_id {
                    thinking_closed = true;
                }

                if should_stop_token(cfg, gen_cfg, next_id) {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                    break;
                }

                generated_ids.push(next_id);
                all_ids.push(next_id);
                record_logprob(
                    &mut token_logprobs,
                    &scratch.logits[..cfg.vocab_size],
                    next_id,
                    gen_cfg.temperature,
                    gen_cfg.logprobs,
                );
                // Capture close-point after push so </think> is the last reasoning token.
                if thinking_closed && reasoning_end_len.is_none() {
                    reasoning_end_len = Some(generated_ids.len());
                }

                let delta = detok.push(&self.tokenizer, next_id);
                let mut iter_interrupted = false;
                let stop_matched = streamer.push(&delta, &mut |s| {
                    if !iter_interrupted && !on_token(s) {
                        iter_interrupted = true;
                    }
                });
                if iter_interrupted {
                    stopped_by_caller = true;
                    stop_reason = StopReason::Interrupt;
                    break;
                }
                if stop_matched {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                    break;
                }

                // Answer-budget break: stop once max_new_tokens answer tokens follow </think>.
                if let Some(end) = reasoning_end_len
                    && generated_ids.len().saturating_sub(end) >= gen_cfg.max_new_tokens
                {
                    break;
                }
            }

            // Natural-end flush (no-op if a stop was already hit inside the loop).
            // Skip when the caller asked to stop -- it is no longer consuming
            // the stream, and `on_token`'s return value here would not change
            // why generation actually stopped.
            if !stopped_by_caller {
                streamer.finish(&detok.finish(), &mut |s| {
                    on_token(s);
                });
                // finish() may itself complete a stop in the tail bytes.
                if streamer.stopped() && !stopped {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                }
            }

            Ok(GenerateOutput {
                text: streamer.into_text(),
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                stopped,
                stop_reason: Some(stop_reason),
                token_logprobs,
            })
        }
    }
}

/// Returns `true` when at least one logit is strictly greater than
/// `f32::NEG_INFINITY` — i.e. the grammar mask leaves at least one legal token.
///
/// When a grammar engine blocks every token via `mask_logits`, every logit
/// becomes `NEG_INFINITY`. Without this guard the sampler's non-finite-max
/// short-circuit would silently emit token 0 (lowest id after sorting an
/// all-NEG_INFINITY candidate set), violating the grammar contract. Callers
/// check this before invoking the sampler and return a typed error instead.
fn has_finite_logit(logits: &[f32]) -> bool {
    logits.iter().any(|&l| l > f32::NEG_INFINITY)
}

fn initial_rng_state(seed: Option<u64>) -> u64 {
    match seed {
        Some(s) => {
            if s == 0 {
                1
            } else {
                s
            }
        }
        None => {
            use std::time::SystemTime;
            let t = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x12345678_9abcdef0);
            if t == 0 { 1 } else { t }
        }
    }
}

fn prefill_tokens(
    model: &Qwen35Model,
    prompt_ids: &[u32],
    gdn_states: &mut [GatedDeltaNetState],
    kv_cache: &mut KvCache,
    scratch: &mut ForwardScratch,
) {
    let prompt_len = prompt_ids.len();
    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        model.forward_step(token_id, pos, gdn_states, kv_cache, scratch);
        if pos < prompt_len - 1 {
            kv_cache.seq_len += 1;
        }
    }
}

/// Fast-path decode loop (no string stops). Budget forcing mirrors
/// `generate_streaming`'s inline fast-path loop exactly: when `reasoning_budget`
/// is disabled, `think_close_id` is `None`, `thinking_closed_seed` is `false`,
/// `decode_cap` equals `max_new_tokens`, and `force_close_think` returns `None`,
/// so the body is byte-identical to its pre-feature form (e2e-parity pinned).
#[allow(clippy::too_many_arguments)]
fn decode_loop(
    model: &Qwen35Model,
    gen_cfg: &GenerateConfig,
    all_ids: &mut Vec<u32>,
    generated_ids: &mut Vec<u32>,
    rng_state: &mut u64,
    gdn_states: &mut [GatedDeltaNetState],
    kv_cache: &mut KvCache,
    scratch: &mut ForwardScratch,
    grammar_state: &mut Option<GrammarState>,
    think_close_id: Option<u32>,
    thinking_closed_seed: bool,
    token_logprobs: &mut Vec<TokenLogprob>,
) -> Result<(bool, StopReason), InferenceError> {
    let cfg = &model.config;
    let mut thinking_closed = thinking_closed_seed;
    let mut reasoning_end_len: Option<usize> = if thinking_closed {
        Some(generated_ids.len())
    } else {
        None
    };
    let cap = decode_cap(gen_cfg.reasoning_budget, gen_cfg.max_new_tokens);
    for _ in 1..cap {
        let pos = kv_cache.seq_len;
        let Some(&last_token) = all_ids.last() else {
            return Err(InferenceError::Inference("empty generation state".into()));
        };

        model.forward_step(last_token, pos, gdn_states, kv_cache, scratch);
        kv_cache.seq_len += 1;

        // Grammar mask before sampling; fail closed when every token is blocked.
        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut *grammar_state) {
            engine.mask_logits(gs, &mut scratch.logits[..cfg.vocab_size]);
            if !has_finite_logit(&scratch.logits[..cfg.vocab_size]) {
                return Err(InferenceError::InvalidInput(
                    "grammar constraint blocked every token; \
                     no legal continuation exists in the current grammar state"
                        .into(),
                ));
            }
        }

        let sampled_id = sample_token(
            &scratch.logits[..cfg.vocab_size],
            gen_cfg,
            all_ids,
            rng_state,
        );

        // Budget forcing: override the sampled token with </think> when the
        // reasoning budget is exhausted and the block is still open.
        let next_id = force_close_think(
            gen_cfg.reasoning_budget,
            gen_cfg.enable_thinking,
            thinking_closed,
            generated_ids.len(),
            think_close_id,
        )
        .unwrap_or(sampled_id);

        // Grammar advance on the actually-emitted token (after any budget
        // override); a false return signals grammar completion.
        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut *grammar_state)
            && !engine.advance(gs, next_id)
        {
            return Ok((true, StopReason::Grammar));
        }

        // Track when the thinking block closes (natural or forced).
        if Some(next_id) == think_close_id {
            thinking_closed = true;
        }

        if should_stop_token(cfg, gen_cfg, next_id) {
            return Ok((true, StopReason::Eos));
        }

        generated_ids.push(next_id);
        all_ids.push(next_id);
        record_logprob(
            token_logprobs,
            &scratch.logits[..cfg.vocab_size],
            next_id,
            gen_cfg.temperature,
            gen_cfg.logprobs,
        );
        // Capture close-point after push so </think> is the last reasoning token.
        if thinking_closed && reasoning_end_len.is_none() {
            reasoning_end_len = Some(generated_ids.len());
        }

        // Answer-budget break: stop once max_new_tokens answer tokens follow </think>.
        if let Some(end) = reasoning_end_len
            && generated_ids.len().saturating_sub(end) >= gen_cfg.max_new_tokens
        {
            break;
        }
    }
    Ok((false, StopReason::Length))
}

/// String-stop variant of `decode_loop`. Called only when `gen_cfg.stop_strings` is non-empty.
///
/// Runs the autoregressive loop, appending each token's decoded text into `full`. After every
/// token it checks for the earliest occurrence of any stop string; on a hit it truncates `full`
/// and returns early. When no stop is hit the loop runs to `max_new_tokens - 1` (the first token
/// was already pushed by the caller before branching here).
///
/// Note: `generated_ids` and `all_ids` contain all tokens up to and including the token that
/// completed the stop match — we cannot un-generate a partial token after the fact.
#[allow(clippy::too_many_arguments)]
fn decode_loop_with_stops(
    model: &Qwen35Model,
    gen_cfg: &GenerateConfig,
    all_ids: &mut Vec<u32>,
    generated_ids: &mut Vec<u32>,
    rng_state: &mut u64,
    gdn_states: &mut [GatedDeltaNetState],
    kv_cache: &mut KvCache,
    scratch: &mut ForwardScratch,
    detok: &mut IncrementalDetokenizer,
    full: &mut String,
    grammar_state: &mut Option<GrammarState>,
    think_close_id: Option<u32>,
    thinking_closed_seed: bool,
    token_logprobs: &mut Vec<TokenLogprob>,
    token_logprob_end_offsets: &mut Vec<usize>,
) -> Result<(bool, StopReason), InferenceError> {
    let cfg = &model.config;
    let mut stopped = false;
    let mut stop_reason = StopReason::Length;
    let mut thinking_closed = thinking_closed_seed;
    let mut reasoning_end_len: Option<usize> = if thinking_closed {
        Some(generated_ids.len())
    } else {
        None
    };
    let cap = decode_cap(gen_cfg.reasoning_budget, gen_cfg.max_new_tokens);
    // Longest configured stop string, used to bound the per-token stop scan to
    // the suffix that could contain a new match (see `earliest_stop_match_from`).
    // `full` already reflects everything scanned before this function was
    // called (the pre-loop first-token check), so its current length is the
    // correct starting point for the "already scanned" prefix.
    let max_stop = gen_cfg
        .stop_strings
        .iter()
        .map(String::len)
        .max()
        .unwrap_or(1);
    for _ in 1..cap {
        let pos = kv_cache.seq_len;
        let Some(&last_token) = all_ids.last() else {
            return Err(InferenceError::Inference("empty generation state".into()));
        };

        model.forward_step(last_token, pos, gdn_states, kv_cache, scratch);
        kv_cache.seq_len += 1;

        // Grammar mask before sampling; fail closed when every token is blocked.
        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut *grammar_state) {
            engine.mask_logits(gs, &mut scratch.logits[..cfg.vocab_size]);
            if !has_finite_logit(&scratch.logits[..cfg.vocab_size]) {
                return Err(InferenceError::InvalidInput(
                    "grammar constraint blocked every token; \
                     no legal continuation exists in the current grammar state"
                        .into(),
                ));
            }
        }

        let sampled_id = sample_token(
            &scratch.logits[..cfg.vocab_size],
            gen_cfg,
            all_ids,
            rng_state,
        );

        // Budget forcing: override the sampled token with </think> when the
        // reasoning budget is exhausted and the block is still open.
        let next_id = force_close_think(
            gen_cfg.reasoning_budget,
            gen_cfg.enable_thinking,
            thinking_closed,
            generated_ids.len(),
            think_close_id,
        )
        .unwrap_or(sampled_id);

        // Grammar advance on the actually-emitted token (after any budget
        // override); a false return signals grammar completion.
        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut *grammar_state)
            && !engine.advance(gs, next_id)
        {
            stopped = true;
            stop_reason = StopReason::Grammar;
            break;
        }

        // Track when the thinking block closes (natural or forced).
        if Some(next_id) == think_close_id {
            thinking_closed = true;
        }

        if should_stop_token(cfg, gen_cfg, next_id) {
            stopped = true;
            stop_reason = StopReason::Eos;
            break;
        }

        generated_ids.push(next_id);
        all_ids.push(next_id);
        record_logprob(
            token_logprobs,
            &scratch.logits[..cfg.vocab_size],
            next_id,
            gen_cfg.temperature,
            gen_cfg.logprobs,
        );
        // Capture close-point after push so </think> is the last reasoning token.
        if thinking_closed && reasoning_end_len.is_none() {
            reasoning_end_len = Some(generated_ids.len());
        }

        let prev_len = full.len();
        let delta = detok.push(&model.tokenizer, next_id);
        if !delta.is_empty() {
            full.push_str(&delta);
        }

        // Keep the offset tracker in lockstep with token_logprobs' conditional
        // growth (record_logprob is a no-op unless gen_cfg.logprobs is set).
        if token_logprobs.len() > token_logprob_end_offsets.len() {
            token_logprob_end_offsets.push(full.len());
        }

        // Only the suffix that could contain a NEW match needs rescanning —
        // any match fully inside `full[..prev_len]` would already have been
        // found on a prior iteration (see `earliest_stop_match_from`'s doc
        // comment). Bounds per-token work instead of rescanning all of `full`.
        // Shared with `StopStringMatcher::push` via `stop_scan_search_start`
        // so both call sites derive the bound identically.
        let search_start = stop_scan_search_start(full, prev_len, max_stop);
        if let Some(hit) = earliest_stop_match_from(full, &gen_cfg.stop_strings, search_start) {
            full.truncate(hit);
            truncate_token_logprobs_to_retained_text(
                token_logprobs,
                token_logprob_end_offsets,
                hit,
            );
            stopped = true;
            stop_reason = StopReason::Eos;
            break;
        }

        // Answer-budget break: stop once max_new_tokens answer tokens follow </think>.
        if let Some(end) = reasoning_end_len
            && generated_ids.len().saturating_sub(end) >= gen_cfg.max_new_tokens
        {
            break;
        }
    }

    // Only flush detok tail when no stop was hit. A stop already truncated `full` to
    // the correct position; appending tail bytes from past the stop would corrupt it.
    if !stopped {
        let tail = detok.finish();
        if !tail.is_empty() {
            full.push_str(&tail);
            // The tail itself might complete a stop string.
            if let Some(hit) = earliest_stop_match(full, &gen_cfg.stop_strings) {
                full.truncate(hit);
                truncate_token_logprobs_to_retained_text(
                    token_logprobs,
                    token_logprob_end_offsets,
                    hit,
                );
                return Ok((true, StopReason::Eos));
            }
        }
        return Ok((false, StopReason::Length));
    }
    Ok((stopped, stop_reason))
}

/// Drops trailing `token_logprobs` entries whose decoded text extends past
/// `retained_len` (the text length after a stop-string match truncates the
/// output).
///
/// A stop match can complete mid-token or even mid-multi-token (an
/// incrementally-detokenized delta may itself span several sampled tokens),
/// so more than one trailing token can end up with text that no longer
/// appears in the truncated output. The OpenAI `logprobs.content` shape is
/// one entry per whole token; a token whose text was only partially retained
/// can't be represented as a partial entry, so it — and any token after it —
/// is dropped rather than left describing text the caller never receives in
/// `message.content` (#620 round-1 review finding).
///
/// `token_logprob_end_offsets[i]` must be the length of the accumulated
/// output text immediately after token `i`'s delta was appended, and the two
/// slices must be the same length (both grow in lockstep, gated on the same
/// `gen_cfg.logprobs.is_some()` condition — see call sites).
fn truncate_token_logprobs_to_retained_text(
    token_logprobs: &mut Vec<TokenLogprob>,
    token_logprob_end_offsets: &[usize],
    retained_len: usize,
) {
    debug_assert_eq!(token_logprobs.len(), token_logprob_end_offsets.len());
    let keep = token_logprob_end_offsets.partition_point(|&end| end <= retained_len);
    token_logprobs.truncate(keep);
}

/// Returns true when `token_id` is EOS or is in the `stop_token_ids` list.
///
/// `pub(crate)` — all callers (Q8, F16, NEON, batch-prefill generate paths) live
/// within this crate. Keeping the function crate-private avoids leaking a
/// low-level sampling helper as part of the public API.
pub(crate) fn should_stop_token(
    cfg: &Qwen35Config,
    gen_cfg: &GenerateConfig,
    token_id: u32,
) -> bool {
    token_id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&token_id)
}

/// Returns `Err(InvalidInput)` when `gen_cfg.grammar` is set, to prevent the
/// grammar field from being silently ignored on paths that have not yet wired
/// grammar masking (#397).
///
/// Grammar masking (`mask_logits` + `advance`) requires a per-step wiring loop
/// inside each generate path. The following paths have not yet been wired and
/// delegate to this guard to fail closed rather than silently producing
/// unconstrained output when a caller sets `gen_cfg.grammar`:
///
/// - `generate_q8` (`forward/cpu_q8.rs`)
/// - `generate_f16` (`forward/cpu_f16.rs`)
/// - `generate_q8_neon` (`forward/neon_forward.rs`)
/// - `Qwen35Model::generate_with_batch_prefill` (`forward/batch_prefill.rs`)
/// - `multimodal_generate_preflight` (`forward/metal_qwen35.rs`)
///
/// The base CPU `generate()` / `generate_streaming()` paths in this module wire
/// grammar directly and therefore do not call this guard.
pub(crate) fn check_grammar_not_set(gen_cfg: &GenerateConfig) -> Result<(), InferenceError> {
    if gen_cfg.grammar.is_some() {
        return Err(InferenceError::InvalidInput(
            "grammar-constrained decoding is not yet supported on this path; \
             use the Qwen3.5 CPU generate() / generate_streaming() or the generic \
             generate() in src/generate.rs, which implement grammar masking"
                .into(),
        ));
    }
    Ok(())
}

/// Sibling guard to [`check_grammar_not_set`]: fails closed instead of silently
/// dropping a `logprobs` request on a generation path that has not been wired
/// to capture per-step log-probabilities (#585). Same five paths, same
/// rationale — see `check_grammar_not_set` for the full list.
///
/// The base CPU `generate()` / `generate_streaming()` paths in this module
/// wire logprobs capture directly and therefore do not call this guard.
pub(crate) fn check_logprobs_not_set(gen_cfg: &GenerateConfig) -> Result<(), InferenceError> {
    if gen_cfg.logprobs.is_some() {
        return Err(InferenceError::InvalidInput(
            "per-token logprobs are not yet supported on this generation path; \
             use the Qwen3.5 CPU generate() / generate_streaming() or the Metal \
             generate_streaming(), which implement logprobs capture"
                .into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Grammar wiring — mutation-sensitive unit tests (#397)
    // -----------------------------------------------------------------------

    /// Grammar masking must block the argmax (highest-logit) token when the grammar
    /// forbids it, causing the sampler to select a lower-logit allowed token instead.
    ///
    /// Constructs a three-token grammar that allows "t" (index 0) and "f" (index 1)
    /// but forbids "x" (index 2). Logits are set so index 2 would win greedy argmax
    /// WITHOUT masking. With masking, index 2 must become NEG_INFINITY and greedy
    /// sampling must return index 1 (next highest allowed logit).
    ///
    /// Mutation sensitivity:
    ///   Remove the `engine.mask_logits(gs, ...)` call from the wiring site →
    ///   logits[2] stays 1000.0, greedy sampling returns 2, `assert_ne!(sampled, 2)`
    ///   fails. This proves the mask_logits call is load-bearing.
    #[test]
    fn grammar_masking_blocks_argmax_token() {
        use crate::grammar::{GrammarEngine, GrammarSpec};
        use std::sync::Arc;

        // Grammar: root ::= "t" | "f" — index 0 and 1 are valid, index 2 ("x") is not.
        let spec = GrammarSpec::Gbnf("root ::= \"t\" | \"f\"\n".to_string());
        let vocab = vec![b"t".to_vec(), b"f".to_vec(), b"x".to_vec()];
        let engine =
            Arc::new(GrammarEngine::new(&spec, vocab).expect("trivial grammar must compile"));

        let mut state = engine.initial_state();

        // Set logits so the forbidden token (index 2) has the highest value.
        // Without masking, greedy sampling would return 2.
        let mut logits = vec![1.0_f32, 2.0_f32, 1000.0_f32];
        engine.mask_logits(&mut state, &mut logits);

        // The forbidden token must be blocked.
        assert_eq!(
            logits[2],
            f32::NEG_INFINITY,
            "mask_logits must set the forbidden token to NEG_INFINITY"
        );

        // At least one allowed token must remain finite.
        assert!(
            has_finite_logit(&logits),
            "at least one allowed logit must survive the mask"
        );

        // Greedy sampling (temperature=0) must choose the highest ALLOWED logit,
        // which is index 1 ("f", logit 2.0), not the blocked index 2.
        let gen_cfg = GenerateConfig {
            temperature: 0.0, // greedy
            ..Default::default()
        };
        let mut rng = 1u64;
        let sampled = sample_token(&logits, &gen_cfg, &[], &mut rng);
        assert_ne!(
            sampled, 2,
            "blocked token must not be selected by the sampler"
        );
        assert_eq!(
            sampled, 1,
            "greedy must select the highest remaining allowed logit (index 1)"
        );
    }

    /// An all-blocking grammar mask must be detected by `has_finite_logit` so the
    /// caller can return a typed error rather than silently emitting token 0.
    ///
    /// When every logit is NEG_INFINITY, the sampler's non-finite-max short-circuit
    /// returns token 0 (the lowest-id token after sorting an all-NEG_INFINITY
    /// candidate set). The `has_finite_logit` guard catches this before the sampler
    /// is invoked and allows the generation loop to return `InvalidInput` instead.
    ///
    /// Mutation sensitivity:
    ///   Change `has_finite_logit` to always return `true` →
    ///   `assert!(!has_finite_logit(...))` fails. This proves the guard is load-bearing
    ///   for the empty-mask fail-closed path.
    #[test]
    fn grammar_all_blocked_mask_detected_by_has_finite_logit() {
        // Simulate a grammar engine that blocked every token.
        let all_blocked = vec![f32::NEG_INFINITY; 8];

        assert!(
            !has_finite_logit(&all_blocked),
            "all-NEG_INFINITY logit buffer must NOT pass the has_finite_logit guard; \
             the caller must return a typed error, not silently emit token 0"
        );

        // A buffer with a single finite logit must pass the guard.
        let mut one_allowed = vec![f32::NEG_INFINITY; 8];
        one_allowed[3] = 1.0_f32;
        assert!(
            has_finite_logit(&one_allowed),
            "a single finite logit must pass the guard (grammar still has valid tokens)"
        );
    }

    // -----------------------------------------------------------------------
    // Grammar wiring — end-to-end production-seam test (#397)
    // -----------------------------------------------------------------------

    /// Proves that `generate()` calls `mask_logits` at the post-prefill wiring
    /// site — i.e., the production call is real, not just the primitive tested
    /// by `grammar_masking_blocks_argmax_token`.
    ///
    /// Strategy: build a minimal synthetic model (4 layers, 64-dim hidden,
    /// 97-token vocab), then construct a grammar engine whose vocabulary table
    /// is 97 empty byte sequences. `VocabPartition::build` automatically rejects
    /// empty entries (they can never advance the PDA), so the precomputed bitmask
    /// for the initial state is all-zeros: `mask_logits` sets every one of the 97
    /// logit positions to `NEG_INFINITY`. `has_finite_logit` then fires the
    /// fail-closed guard inside `generate()`, which returns `Err(InvalidInput)`.
    ///
    /// Coverage: the post-prefill masking site in `generate()` (the
    /// `engine.mask_logits` call just before the first `sample_token`). The
    /// decode-loop wiring sites — inside `decode_loop` and the inline streaming
    /// loops — are reached only for tokens 2+ and are not separately covered
    /// here; they would require additional forward-step iterations that cannot be
    /// isolated without a controllable-output model.
    ///
    /// Mutation sensitivity: removing `engine.mask_logits(gs, ...)` from the
    /// post-prefill site leaves logits at their raw (finite) model values.
    /// `has_finite_logit` then returns `true`, no error is returned, and this
    /// test's `assert!(result.is_err())` fails — proving the call is load-bearing.
    ///
    /// The model-building helpers below mirror `lora_serving::build_model` in
    /// tests.rs; they are duplicated here to keep generation.rs self-contained
    /// without a cross-module test-support coupling.
    #[test]
    fn grammar_wiring_mask_logits_called_in_generate() {
        use crate::attention::gdn::GatedDeltaNetWeights;
        use crate::grammar::{GrammarEngine, GrammarSpec};
        use crate::lora_hook::NoopLoraHook;
        use crate::model::qwen35::{
            AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
            FullAttentionLayerWeights, ModelWeights,
        };
        use crate::model::qwen35_config::{LayerType, compute_layer_types};
        use crate::rope::RopeTable;
        use crate::tokenizer::bpe::BpeTokenizer;
        use std::sync::Arc;

        // Tiny 4-layer hybrid config (3 linear + 1 full), vocab_size must match
        // the grammar engine entry count so mask_logits covers all 97 logits.
        const H: usize = 64;
        const VOCAB: usize = 97;
        const I: usize = 128;
        const NUM_LAYERS: usize = 4;
        const FULL_INTERVAL: usize = 4;
        const HEAD_DIM: usize = 16;
        const LINEAR_KH: usize = 4;
        const KERNEL: usize = 4;

        let cfg = Qwen35Config {
            hidden_size: H,
            num_hidden_layers: NUM_LAYERS,
            vocab_size: VOCAB,
            intermediate_size: I,
            rms_norm_eps: 1e-6,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: HEAD_DIM,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            linear_num_key_heads: LINEAR_KH,
            linear_num_value_heads: Some(LINEAR_KH),
            linear_key_head_dim: HEAD_DIM,
            linear_value_head_dim: HEAD_DIM,
            linear_conv_kernel_dim: KERNEL,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            full_attention_interval: FULL_INTERVAL,
            layer_types: compute_layer_types(NUM_LAYERS, FULL_INTERVAL),
            layer_mask: vec![true; NUM_LAYERS],
            eos_token_id: (VOCAB - 1) as u32,
            max_position_embeddings: 1024,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
        };

        // Deterministic xorshift for reproducible synthetic weights.
        fn rand_vec(rng: &mut u64, len: usize) -> Vec<f32> {
            (0..len)
                .map(|_| {
                    *rng ^= *rng << 13;
                    *rng ^= *rng >> 7;
                    *rng ^= *rng << 17;
                    ((*rng >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * 0.02
                })
                .collect()
        }
        let mut rng = 0xA55E_u64 | 1;

        let qkv_dim = cfg.linear_qkv_dim();
        let out_dim = cfg.linear_output_dim();
        let q_dim = cfg.full_q_dim();
        let kv_dim = cfg.full_kv_dim();

        // layer_types = [Linear, Linear, Linear, Full] for interval=4.
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for lt in &cfg.layer_types {
            let common = CommonLayerWeights {
                input_layernorm: rand_vec(&mut rng, H),
                post_attention_layernorm: rand_vec(&mut rng, H),
                ffn: FeedForwardWeights::Dense(DenseFfnWeights {
                    gate_proj: rand_vec(&mut rng, I * H),
                    up_proj: rand_vec(&mut rng, I * H),
                    down_proj: rand_vec(&mut rng, H * I),
                }),
            };
            let attn = match lt {
                LayerType::LinearAttention => AttentionWeights::Linear(GatedDeltaNetWeights {
                    in_proj_qkv: rand_vec(&mut rng, qkv_dim * H),
                    in_proj_qkv_rows: qkv_dim,
                    in_proj_qkv_cols: H,
                    in_proj_z: rand_vec(&mut rng, out_dim * H),
                    in_proj_z_rows: out_dim,
                    in_proj_z_cols: H,
                    in_proj_b: rand_vec(&mut rng, LINEAR_KH * H),
                    in_proj_b_rows: LINEAR_KH,
                    in_proj_b_cols: H,
                    in_proj_a: rand_vec(&mut rng, LINEAR_KH * H),
                    in_proj_a_rows: LINEAR_KH,
                    in_proj_a_cols: H,
                    a_log: rand_vec(&mut rng, LINEAR_KH),
                    dt_bias: rand_vec(&mut rng, LINEAR_KH),
                    conv1d_weight: rand_vec(&mut rng, qkv_dim * KERNEL),
                    conv_dim: qkv_dim,
                    kernel_size: KERNEL,
                    norm_weight: rand_vec(&mut rng, out_dim),
                    out_proj: rand_vec(&mut rng, H * out_dim),
                    out_proj_rows: H,
                    out_proj_cols: out_dim,
                }),
                LayerType::FullAttention => AttentionWeights::Full(FullAttentionLayerWeights {
                    q_proj: rand_vec(&mut rng, 2 * q_dim * H),
                    k_proj: rand_vec(&mut rng, kv_dim * H),
                    v_proj: rand_vec(&mut rng, kv_dim * H),
                    o_proj: rand_vec(&mut rng, H * q_dim),
                    q_norm: rand_vec(&mut rng, HEAD_DIM),
                    k_norm: rand_vec(&mut rng, HEAD_DIM),
                }),
            };
            layers.push((attn, common));
        }

        // Minimal 7-token BPE tokenizer: "a" (id 1) serves as the one-token
        // prompt. The grammar fires before EOS (id 96) is ever needed.
        let tok_json = r#"{
  "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"<unk>":0,"a":1,"b":2,"c":3,"d":4,"e":5," ":6},"merges":[]}
}"#;
        let tokenizer =
            BpeTokenizer::from_tokenizer_json_str(tok_json).expect("test tokenizer parses");

        let rope = RopeTable::new(
            cfg.rope_dim(),
            cfg.max_position_embeddings.min(8192),
            cfg.rope_theta,
        );

        let model = Qwen35Model {
            config: cfg.clone(),
            weights: ModelWeights {
                embed_tokens: rand_vec(&mut rng, VOCAB * H),
                lm_head: None,
                final_norm: rand_vec(&mut rng, H),
                layers,
            },
            tokenizer,
            rope,
            lora: Box::new(NoopLoraHook),
        };

        // Grammar engine with VOCAB empty byte sequences. VocabPartition::build
        // skips empty entries, so the precomputed bitmask for the initial state is
        // all-zeros: mask_logits sets all VOCAB logits to NEG_INFINITY.
        let vocab_bytes: Vec<Vec<u8>> = vec![vec![]; VOCAB];
        let spec = GrammarSpec::Gbnf("root ::= \"ok\"\n".to_string());
        let engine = Arc::new(
            GrammarEngine::new(&spec, vocab_bytes).expect("grammar engine builds with empty vocab"),
        );

        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            temperature: 0.0,
            grammar: Some(engine),
            ..Default::default()
        };

        // The all-blocking grammar must trigger has_finite_logit inside generate()
        // and return Err(InvalidInput). Any other outcome is a wiring failure.
        let result = model.generate("a", &gen_cfg);
        assert!(
            matches!(result, Err(InferenceError::InvalidInput(_))),
            "grammar blocking every token must return Err(InvalidInput); got {result:?}"
        );
    }

    #[test]
    fn generate_config_default_stop_strings_empty() {
        let cfg = GenerateConfig::default();
        assert!(cfg.stop_strings.is_empty());
    }

    #[test]
    fn generate_config_stop_strings_field_explicit() {
        let cfg = GenerateConfig {
            stop_strings: vec!["</s>".to_string(), "\nUser:".to_string()],
            ..Default::default()
        };
        assert_eq!(cfg.stop_strings.len(), 2);
        assert_eq!(cfg.stop_strings[0], "</s>");
    }

    // -----------------------------------------------------------------------
    // StopReason mutation-sensitive tests (#456)
    // -----------------------------------------------------------------------
    //
    // Each test exercises one specific code path that sets stop_reason and
    // asserts the exact variant. If the wrong variant is assigned at that
    // site, the assertion fails — proving the assignment is load-bearing.
    //
    // Build helper: all-zero weights → all logits == 0.0 after any forward
    // pass → greedy sampling always picks token 0 (first-wins on equal logits).
    // This gives deterministic sampling without relying on random-weight outputs.

    const DEFAULT_TINY_TOK_JSON: &str = r#"{
  "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"<unk>":0,"a":1,"b":2,"c":3,"d":4,"e":5," ":6},"merges":[]}
}"#;

    fn build_tiny_zero_model() -> Qwen35Model {
        build_tiny_zero_model_tok(DEFAULT_TINY_TOK_JSON)
    }

    fn build_tiny_zero_model_tok(tok_json: &str) -> Qwen35Model {
        use crate::attention::gdn::GatedDeltaNetWeights;
        use crate::lora_hook::NoopLoraHook;
        use crate::model::qwen35::{
            AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
            FullAttentionLayerWeights, ModelWeights,
        };
        use crate::model::qwen35_config::{LayerType, compute_layer_types};
        use crate::rope::RopeTable;
        use crate::tokenizer::bpe::BpeTokenizer;

        const H: usize = 64;
        const VOCAB: usize = 97;
        const I: usize = 128;
        const NUM_LAYERS: usize = 4;
        const FULL_INTERVAL: usize = 4;
        const HEAD_DIM: usize = 16;
        const LINEAR_KH: usize = 4;
        const KERNEL: usize = 4;

        let cfg = Qwen35Config {
            hidden_size: H,
            num_hidden_layers: NUM_LAYERS,
            vocab_size: VOCAB,
            intermediate_size: I,
            rms_norm_eps: 1e-6,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: HEAD_DIM,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            linear_num_key_heads: LINEAR_KH,
            linear_num_value_heads: Some(LINEAR_KH),
            linear_key_head_dim: HEAD_DIM,
            linear_value_head_dim: HEAD_DIM,
            linear_conv_kernel_dim: KERNEL,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            full_attention_interval: FULL_INTERVAL,
            layer_types: compute_layer_types(NUM_LAYERS, FULL_INTERVAL),
            layer_mask: vec![true; NUM_LAYERS],
            eos_token_id: (VOCAB - 1) as u32,
            max_position_embeddings: 1024,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
        };

        let z = |len: usize| vec![0.0_f32; len];
        let qkv_dim = cfg.linear_qkv_dim();
        let out_dim = cfg.linear_output_dim();
        let q_dim = cfg.full_q_dim();
        let kv_dim = cfg.full_kv_dim();

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for lt in &cfg.layer_types {
            let common = CommonLayerWeights {
                input_layernorm: z(H),
                post_attention_layernorm: z(H),
                ffn: FeedForwardWeights::Dense(DenseFfnWeights {
                    gate_proj: z(I * H),
                    up_proj: z(I * H),
                    down_proj: z(H * I),
                }),
            };
            let attn = match lt {
                LayerType::LinearAttention => AttentionWeights::Linear(GatedDeltaNetWeights {
                    in_proj_qkv: z(qkv_dim * H),
                    in_proj_qkv_rows: qkv_dim,
                    in_proj_qkv_cols: H,
                    in_proj_z: z(out_dim * H),
                    in_proj_z_rows: out_dim,
                    in_proj_z_cols: H,
                    in_proj_b: z(LINEAR_KH * H),
                    in_proj_b_rows: LINEAR_KH,
                    in_proj_b_cols: H,
                    in_proj_a: z(LINEAR_KH * H),
                    in_proj_a_rows: LINEAR_KH,
                    in_proj_a_cols: H,
                    a_log: z(LINEAR_KH),
                    dt_bias: z(LINEAR_KH),
                    conv1d_weight: z(qkv_dim * KERNEL),
                    conv_dim: qkv_dim,
                    kernel_size: KERNEL,
                    norm_weight: z(out_dim),
                    out_proj: z(H * out_dim),
                    out_proj_rows: H,
                    out_proj_cols: out_dim,
                }),
                LayerType::FullAttention => AttentionWeights::Full(FullAttentionLayerWeights {
                    q_proj: z(2 * q_dim * H),
                    k_proj: z(kv_dim * H),
                    v_proj: z(kv_dim * H),
                    o_proj: z(H * q_dim),
                    q_norm: z(HEAD_DIM),
                    k_norm: z(HEAD_DIM),
                }),
            };
            layers.push((attn, common));
        }

        let tokenizer =
            BpeTokenizer::from_tokenizer_json_str(tok_json).expect("test tokenizer parses");
        let rope = RopeTable::new(
            cfg.rope_dim(),
            cfg.max_position_embeddings.min(8192),
            cfg.rope_theta,
        );

        Qwen35Model {
            config: cfg.clone(),
            weights: ModelWeights {
                embed_tokens: z(VOCAB * H),
                lm_head: None,
                final_norm: z(H),
                layers,
            },
            tokenizer,
            rope,
            lora: Box::new(NoopLoraHook),
        }
    }

    /// `max_new_tokens == 0` must set `stop_reason = Some(StopReason::Length)` on the
    /// early-return path that fires before any forward pass or token sampling.
    ///
    /// Mutation sensitivity: changing `StopReason::Length` in the `max_new_tokens == 0`
    /// guard to any other variant causes `assert_eq!(stop_reason, ...)` to fail.
    #[test]
    fn stop_reason_length_on_zero_max_tokens() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 0,
            temperature: 0.0,
            ..Default::default()
        };
        let result = model
            .generate("a", &gen_cfg)
            .expect("zero-token generate must succeed");
        assert_eq!(
            result.stop_reason,
            Some(StopReason::Length),
            "max_new_tokens == 0 must return StopReason::Length; got {:?}",
            result.stop_reason
        );
        assert_eq!(
            result.generated_tokens, 0,
            "zero max_new_tokens must produce no tokens"
        );
    }

    /// First sampled token matching a `stop_token_ids` entry must set
    /// `stop_reason = Some(StopReason::Eos)` on the early-return path.
    ///
    /// Zero-weight model → all logits == 0.0 → greedy picks token 0 (first-wins on ties).
    /// `stop_token_ids = [0]` causes `should_stop_token` to fire on the first decode step.
    ///
    /// Mutation sensitivity: changing `StopReason::Eos` at the `should_stop_token` return
    /// site to any other variant causes `assert_eq!(stop_reason, Some(StopReason::Eos))` to fail.
    #[test]
    fn stop_reason_eos_on_first_stop_token() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 5,
            temperature: 0.0,
            stop_token_ids: vec![0], // token 0 is greedy-sampled with all-zero logits
            ..Default::default()
        };
        let result = model
            .generate("a", &gen_cfg)
            .expect("eos-on-first-token generate must succeed");
        assert_eq!(
            result.stop_reason,
            Some(StopReason::Eos),
            "stop_token_ids match on first token must return StopReason::Eos; got {:?}",
            result.stop_reason
        );
    }

    /// Grammar `advance` returning `false` on the first sampled token must set
    /// `stop_reason = Some(StopReason::Grammar)`.
    ///
    /// Mechanism: the grammar engine has vocab_size = 1 (["t"]). `mask_logits` blocks
    /// token 0 ("t"); tokens 1..96 (beyond grammar vocab_size) stay at 0.0 and are
    /// finite. Greedy picks token 1. `advance(1)`: 1 >= grammar.vocab_size (1) → `false`.
    ///
    /// Mutation sensitivity: changing `StopReason::Grammar` at the `advance`-returns-false
    /// return site to any other variant causes the assertion to fail.
    #[test]
    fn stop_reason_grammar_on_advance_false() {
        use crate::grammar::{GrammarEngine, GrammarSpec};
        use std::sync::Arc;

        let model = build_tiny_zero_model();

        // vocab = ["t"] (size 1). Grammar root ::= "x" blocks token 0 via mask;
        // tokens 1..96 remain finite. Greedy picks token 1. advance(1): 1 >= 1 → false.
        let spec = GrammarSpec::Gbnf("root ::= \"x\"\n".to_string());
        let vocab = vec![b"t".to_vec()];
        let engine =
            Arc::new(GrammarEngine::new(&spec, vocab).expect("single-token grammar compiles"));

        let gen_cfg = GenerateConfig {
            max_new_tokens: 5,
            temperature: 0.0,
            grammar: Some(engine),
            stop_token_ids: vec![],
            ..Default::default()
        };
        let result = model
            .generate("a", &gen_cfg)
            .expect("grammar-advance-false generate must succeed");
        assert_eq!(
            result.stop_reason,
            Some(StopReason::Grammar),
            "grammar advance returning false must return StopReason::Grammar; got {:?}",
            result.stop_reason
        );
    }

    // Same tiny zero-weight model, but the tokenizer carries `</think>` as added
    // token id 7 so `special_token_id("</think>")` resolves and reasoning-budget
    // forcing can fire. `special:false` still makes it queryable — any added token
    // (regardless of the special flag) is inserted into the tokenizer's lookup.
    fn build_tiny_thinking_model() -> Qwen35Model {
        build_tiny_zero_model_tok(
            r#"{
  "version":"1.0","truncation":null,"padding":null,
  "added_tokens":[{"id":7,"content":"</think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"<unk>":0,"a":1,"b":2,"c":3,"d":4,"e":5," ":6},"merges":[]}
}"#,
        )
    }

    /// COMBINED grammar × reasoning-budget path: when the s1 budget forces `</think>`
    /// but the active grammar forbids that token, decoding must **fail closed** — stop
    /// with `StopReason::Grammar` and NOT emit the forbidden `</think>`.
    ///
    /// This pins the load-bearing weave in `decode_loop`: grammar `advance` runs on the
    /// budget-FORCED token (`next_id`), not the pre-force `sampled_id`. Setup: grammar
    /// `root ::= "aa"` with a 7-entry grammar vocab (ids 0..=6); the tokenizer carries
    /// `</think>` at id 7 (outside the grammar vocab). All-zero weights → greedy always
    /// picks token 0's argmax after masking. Post-prefill emits one `'a'` (id 1); the
    /// first decode-loop step has `generated_len == budget == 1`, so `force_close_think`
    /// overrides the sampled `'a'` with `</think>` (id 7). `advance(7)`:
    /// `7 >= grammar.vocab_size (7)` → `false` → `StopReason::Grammar`, before `</think>`
    /// is pushed.
    ///
    /// Mutation sensitivity: if `advance` were called on `sampled_id` (1, grammar-legal)
    /// instead of the forced `next_id` (7), `advance(1)` would succeed, `</think>` would
    /// be emitted, and `token_ids` would be `[1, 7]` — failing the `token_ids == [1]`
    /// assertion below.
    #[test]
    fn grammar_budget_forced_close_fails_closed() {
        use crate::grammar::{GrammarEngine, GrammarSpec};
        use std::sync::Arc;

        let model = build_tiny_thinking_model();
        let close_id = model
            .tokenizer
            .special_token_id("</think>")
            .expect("thinking model tokenizer resolves </think>");
        assert!(
            close_id >= 7,
            "test assumes </think> id ({close_id}) is outside the 7-token grammar vocab"
        );

        // root ::= "aa": grammar vocab ids 0..=6 (size 7). </think> (id 7) is out of the
        // grammar vocab, so advance(7) fail-closes.
        let spec = GrammarSpec::Gbnf("root ::= \"aa\"\n".to_string());
        let vocab: Vec<Vec<u8>> = vec![
            b"<unk>".to_vec(),
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
            b"d".to_vec(),
            b"e".to_vec(),
            b" ".to_vec(),
        ];
        let engine = Arc::new(GrammarEngine::new(&spec, vocab).expect("aa grammar compiles"));

        let gen_cfg = GenerateConfig {
            max_new_tokens: 5,
            temperature: 0.0,
            enable_thinking: true,
            reasoning_budget: Some(1),
            grammar: Some(engine),
            stop_token_ids: vec![],
            ..Default::default()
        };
        let result = model
            .generate("a", &gen_cfg)
            .expect("combined grammar+budget generate must succeed");

        assert_eq!(
            result.stop_reason,
            Some(StopReason::Grammar),
            "budget-forced </think> forbidden by grammar must stop with Grammar; got {:?}",
            result.stop_reason
        );
        assert_eq!(
            result.token_ids,
            vec![1],
            "fail-closed: the budget-forced </think> must NOT be emitted; only the \
             pre-force 'a' (id 1) survives. token_ids [1, 7] means advance ran on the \
             sampled token, not the forced token"
        );
        assert_eq!(result.generated_tokens, 1);
    }

    // -----------------------------------------------------------------------
    // generate_streaming_with_cancel mutation-sensitive tests (ADR-080 C2, #744)
    // -----------------------------------------------------------------------
    //
    // Zero-weight model: greedy sampling always picks token 0, which decodes to
    // the literal text "<unk>" -- a non-empty delta on every step, so both the
    // pre-loop first-token emission and every decode-loop iteration produce a
    // delta for `on_token` to observe. This gives fully deterministic
    // cancellation-checkpoint counting without relying on random-weight output.

    /// `should_cancel` returning `true` before the prefill pass starts (the
    /// very first checkpoint) must short-circuit generation entirely: no
    /// prefill, no sampling, `generated_tokens == 0`.
    ///
    /// Mutation sensitivity: removing this checkpoint (or its early return)
    /// makes generation fall through to prefill + sampling, producing
    /// `generated_tokens > 0` and failing the assertion below.
    #[test]
    fn generate_streaming_with_cancel_true_before_prefill_returns_interrupt() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 5,
            temperature: 0.0,
            ..Default::default()
        };
        // Cancel on the very first `should_cancel` call only (the pre-prefill
        // checkpoint), so this test pins THAT checkpoint specifically rather
        // than any-of-the-four checkpoints: if the pre-prefill check were
        // removed, the post-prefill checkpoint (call 2) would see `false` and
        // generation would run to completion instead of stopping at
        // `generated_tokens == 0`.
        let calls = std::cell::Cell::new(0usize);
        let result = model
            .generate_streaming_with_cancel(
                "a",
                &gen_cfg,
                |_delta| true,
                || {
                    let n = calls.get() + 1;
                    calls.set(n);
                    n == 1
                },
            )
            .expect("cancelled-before-prefill generate must succeed");
        assert!(
            !result.stopped,
            "a caller cancellation is not an OpenAI stop condition"
        );
        assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
        assert_eq!(result.generated_tokens, 0);
        assert!(result.text.is_empty());
    }

    /// `should_cancel` flipping to `true` at the top of a later decode-loop
    /// iteration (fast path, no `stop_strings`) must stop generation before
    /// the `max_new_tokens` cap is reached, keeping the tokens already emitted
    /// before the flip.
    ///
    /// Call count: checkpoint 1 (pre-prefill) = call 1, checkpoint 2
    /// (post-prefill) = call 2, first decode-loop top = call 3. Flipping true
    /// at call 3 means the loop never runs its body, so only the one
    /// pre-loop token (emitted right after prefill, before the decode loop
    /// starts) is generated.
    ///
    /// Mutation sensitivity: removing the decode-loop's `should_cancel` check
    /// lets generation run to `max_new_tokens` (10), failing
    /// `generated_tokens == 1`.
    #[test]
    fn generate_streaming_with_cancel_mid_decode_stops_early_fast_path() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 10,
            temperature: 0.0,
            ..Default::default()
        };
        let calls = std::cell::Cell::new(0usize);
        let result = model
            .generate_streaming_with_cancel(
                "a",
                &gen_cfg,
                |_delta| true,
                || {
                    let n = calls.get() + 1;
                    calls.set(n);
                    n >= 3
                },
            )
            .expect("mid-decode cancel generate must succeed");
        assert!(!result.stopped);
        assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
        assert_eq!(
            result.generated_tokens, 1,
            "should_cancel flipping true at the first decode-loop checkpoint must stop \
             after exactly the one pre-loop token; got {}",
            result.generated_tokens
        );
    }

    /// `on_token` returning `false` on the very first (pre-loop) delta must
    /// stop generation immediately, in the fast path (no `stop_strings`).
    ///
    /// Mutation sensitivity: dropping the `if !on_token(&delta)` early return
    /// after the pre-loop delta lets generation continue into the decode
    /// loop, failing `generated_tokens == 1`.
    #[test]
    fn generate_streaming_with_cancel_on_token_false_stops_generation_fast_path() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 10,
            temperature: 0.0,
            ..Default::default()
        };
        let result = model
            .generate_streaming_with_cancel("a", &gen_cfg, |_delta| false, || false)
            .expect("on_token-false generate must succeed");
        assert!(!result.stopped);
        assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
        assert_eq!(
            result.generated_tokens, 1,
            "on_token returning false on the very first delta must stop after exactly \
             one token; got {}",
            result.generated_tokens
        );
    }

    /// Same `on_token`-returns-`false` cancellation, but in the `stop_strings`
    /// path (`StopStringMatcher`'s sink has no return value, so cancellation
    /// is threaded through a captured `caller_interrupted` flag instead --
    /// this test pins that alternate code path independently of the fast
    /// path above).
    ///
    /// Mutation sensitivity: dropping the `caller_interrupted` check after
    /// the pre-loop `streamer.push` call lets generation continue into the
    /// decode loop, failing `generated_tokens == 1`.
    #[test]
    fn generate_streaming_with_cancel_on_token_false_stops_generation_stop_string_path() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 10,
            temperature: 0.0,
            stop_strings: vec!["ZZZZ".to_string()],
            ..Default::default()
        };
        let result = model
            .generate_streaming_with_cancel("a", &gen_cfg, |_delta| false, || false)
            .expect("on_token-false generate (stop-string path) must succeed");
        assert!(!result.stopped);
        assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
        assert_eq!(
            result.generated_tokens, 1,
            "on_token returning false on the very first delta must stop after exactly \
             one token in the stop-string path too; got {}",
            result.generated_tokens
        );
    }

    /// Same mid-decode `should_cancel` cancellation as the fast-path test
    /// above, but in the `stop_strings` path -- pins that the decode loop's
    /// `should_cancel` checkpoint is present in both branches, not just the
    /// fast path.
    ///
    /// Mutation sensitivity: removing the decode-loop's `should_cancel` check
    /// in the `stop_strings` branch lets generation run to `max_new_tokens`
    /// (10), failing `generated_tokens == 1`.
    #[test]
    fn generate_streaming_with_cancel_mid_decode_stops_early_stop_string_path() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 10,
            temperature: 0.0,
            stop_strings: vec!["ZZZZ".to_string()],
            ..Default::default()
        };
        let calls = std::cell::Cell::new(0usize);
        let result = model
            .generate_streaming_with_cancel(
                "a",
                &gen_cfg,
                |_delta| true,
                || {
                    let n = calls.get() + 1;
                    calls.set(n);
                    n >= 3
                },
            )
            .expect("mid-decode cancel generate (stop-string path) must succeed");
        assert!(!result.stopped);
        assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
        assert_eq!(
            result.generated_tokens, 1,
            "should_cancel flipping true at the first decode-loop checkpoint must stop \
             after exactly the one pre-loop token (stop-string path); got {}",
            result.generated_tokens
        );
    }

    /// Stop-string truncation must drop `token_logprobs` entries whose decoded
    /// text didn't fully survive the truncation, not leave them describing
    /// bytes the caller never sees in `text` (#620 round-1 review finding:
    /// `build_choice_logprobs` in lattice.rs builds `logprobs.content` directly
    /// off `token_logprobs`, so a stale entry there silently corrupts the
    /// OpenAI-compatible response).
    ///
    /// Zero-weight model → every sampled token is id 0, which decodes to the
    /// literal text "<unk>" (all 5 chars sit in the byte-level decoder's
    /// printable range, so `append_token_bytes` skips none of them as
    /// "special"). Two tokens concatenate to "<unk><unk>" (10 bytes); the stop
    /// string "k><unk" (6 bytes) first matches at byte 3 — BEFORE the token
    /// boundary at byte 5, so the match also clips into the FIRST token's own
    /// trailing bytes, not just the second token's. Both entries' text is
    /// therefore only partially retained, and both must be dropped — this is
    /// the multi-token-span case `truncate_token_logprobs_to_retained_text`
    /// exists to handle, not just "drop the most recent entry".
    ///
    /// Mutation sensitivity: without the truncation call, `token_logprobs`
    /// keeps both entries (len 2) even though `text` is only "<un" (3 bytes) —
    /// neither entry's full text is representable in the truncated output.
    /// With the fix, `token_logprobs` is empty.
    #[test]
    fn stop_string_truncation_drops_stale_token_logprobs() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 10,
            temperature: 0.0,
            logprobs: Some(0),
            stop_strings: vec!["k><unk".to_string()],
            ..Default::default()
        };
        let result = model
            .generate("a", &gen_cfg)
            .expect("stop-string generate must succeed");

        assert_eq!(
            result.text, "<un",
            "stop string must truncate at the first match; got {:?}",
            result.text
        );
        assert_eq!(
            result.generated_tokens, 2,
            "both tokens were sampled before the match completed (can't un-generate); \
             got {}",
            result.generated_tokens
        );
        assert!(
            result.token_logprobs.is_empty(),
            "both tokens' text was only partially retained after truncation, so both \
             logprobs entries must be dropped; got {} entries",
            result.token_logprobs.len()
        );
    }

    // -------------------------------------------------------------------
    // Public-prefill-delegation token parity (perf_hunt public-prefill
    // experiment). Requires a real dense Qwen3.5 checkpoint; set
    // LATTICE_INFERENCE_MODEL_DIR to a safetensors directory (e.g.
    // /Users/lion/.lattice/models/qwen3.5-0.8b). Ignored by default so CI
    // and plain `cargo test` runs never depend on local model files.
    // -------------------------------------------------------------------

    /// Runs greedy generation for `prompt` with both the pre-delegation
    /// serial prefill path and the batched-prefill delegation path on the
    /// same loaded model, asserting the generated token ids are identical.
    /// Serialized on `SERIAL_PREFILL_TEST_LOCK` because `FORCE_SERIAL_PREFILL`
    /// is process-global.
    fn assert_batched_prefill_matches_serial(
        model: &Qwen35Model,
        prompt: &str,
        max_new_tokens: usize,
    ) {
        let _guard = SERIAL_PREFILL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let gen_cfg = crate::model::qwen35_config::GenerateConfig {
            max_new_tokens,
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };

        FORCE_SERIAL_PREFILL.store(true, std::sync::atomic::Ordering::SeqCst);
        let serial = model.generate(prompt, &gen_cfg);
        FORCE_SERIAL_PREFILL.store(false, std::sync::atomic::Ordering::SeqCst);
        let batched = model.generate(prompt, &gen_cfg);

        let serial = serial.expect("serial-prefill generate must succeed");
        let batched = batched.expect("batched-prefill generate must succeed");

        assert_eq!(
            serial.token_ids, batched.token_ids,
            "batched-prefill delegation changed generated token ids for prompt {prompt:?}: \
             serial={:?} batched={:?}",
            serial.token_ids, batched.token_ids
        );
        assert_eq!(
            serial.text, batched.text,
            "batched-prefill delegation changed decoded text for prompt {prompt:?}"
        );
        assert_eq!(serial.stop_reason, batched.stop_reason);
        assert_eq!(serial.stopped, batched.stopped);
    }

    #[test]
    #[ignore = "requires local Qwen3.5 checkpoint: set LATTICE_INFERENCE_MODEL_DIR"]
    fn generate_batched_prefill_matches_serial_for_seeded_dense_prompt() {
        let Ok(model_dir) = std::env::var("LATTICE_INFERENCE_MODEL_DIR") else {
            return;
        };
        let model = Qwen35Model::from_safetensors(std::path::Path::new(&model_dir))
            .expect("dense Qwen3.5 model should load successfully");

        // 20 greedy tokens from a fixed prompt, per the perf_hunt experiment ask.
        assert_batched_prefill_matches_serial(
            &model,
            "The quick brown fox jumps over the lazy dog. In a distant future,",
            20,
        );
    }

    #[test]
    #[ignore = "requires local Qwen3.5 checkpoint: set LATTICE_INFERENCE_MODEL_DIR"]
    fn generate_streaming_batched_prefill_matches_nonstreaming_text() {
        let Ok(model_dir) = std::env::var("LATTICE_INFERENCE_MODEL_DIR") else {
            return;
        };
        let model = Qwen35Model::from_safetensors(std::path::Path::new(&model_dir))
            .expect("dense Qwen3.5 model should load successfully");

        let gen_cfg = crate::model::qwen35_config::GenerateConfig {
            max_new_tokens: 20,
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let prompt = "The quick brown fox jumps over the lazy dog. In a distant future,";

        let non_streaming = model
            .generate(prompt, &gen_cfg)
            .expect("non-streaming generate must succeed");

        let mut streamed_text = String::new();
        let streaming = model
            .generate_streaming(prompt, &gen_cfg, |delta| streamed_text.push_str(delta))
            .expect("streaming generate must succeed");

        assert_eq!(
            non_streaming.token_ids, streaming.token_ids,
            "streaming batched-prefill delegation diverged from non-streaming"
        );
        assert_eq!(non_streaming.text, streaming.text);
        assert_eq!(non_streaming.text, streamed_text);
    }

    #[test]
    #[ignore = "requires local Qwen3.5 checkpoint: set LATTICE_INFERENCE_MODEL_DIR"]
    fn generate_batched_prefill_matches_serial_across_prompt_lengths() {
        let Ok(model_dir) = std::env::var("LATTICE_INFERENCE_MODEL_DIR") else {
            return;
        };
        let model = Qwen35Model::from_safetensors(std::path::Path::new(&model_dir))
            .expect("dense Qwen3.5 model should load successfully");

        for words in [8usize, 64, 256] {
            let prompt = "hello ".repeat(words);
            assert_batched_prefill_matches_serial(&model, prompt.trim_end(), 5);
        }
    }

    /// A/B time-to-first-token sweep: serial (pre-delegation) prefill vs.
    /// batched-prefill delegation, same loaded model, back-to-back, for a
    /// range of prompt lengths. `max_new_tokens: 1` isolates prefill + first
    /// sample. Prints `ms` per path so the perf_hunt experiment report can
    /// quote the raw numbers; not a pass/fail gate (that's the parity tests
    /// above) — run with `--release --features f16 -- --ignored --nocapture`.
    #[test]
    #[ignore = "requires local Qwen3.5 checkpoint: set LATTICE_INFERENCE_MODEL_DIR; run --release"]
    fn public_prefill_ttft_ab_sweep() {
        let Ok(model_dir) = std::env::var("LATTICE_INFERENCE_MODEL_DIR") else {
            return;
        };
        let model = Qwen35Model::from_safetensors(std::path::Path::new(&model_dir))
            .expect("dense Qwen3.5 model should load successfully");

        let _guard = SERIAL_PREFILL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let gen_cfg = crate::model::qwen35_config::GenerateConfig {
            max_new_tokens: 1,
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };

        println!("words\tprompt_tokens\tserial_ms\tbatched_ms\tspeedup");
        for words in [64usize, 512, 2000] {
            let prompt = "hello ".repeat(words);
            let prompt = prompt.trim_end();

            // Warm the model/tokenizer once outside the timed region.
            FORCE_SERIAL_PREFILL.store(false, std::sync::atomic::Ordering::SeqCst);
            let _ = model.generate(prompt, &gen_cfg).unwrap();

            FORCE_SERIAL_PREFILL.store(true, std::sync::atomic::Ordering::SeqCst);
            let t0 = std::time::Instant::now();
            let serial = model.generate(prompt, &gen_cfg).expect("serial generate");
            let serial_ms = t0.elapsed().as_secs_f64() * 1000.0;

            FORCE_SERIAL_PREFILL.store(false, std::sync::atomic::Ordering::SeqCst);
            let t0 = std::time::Instant::now();
            let batched = model.generate(prompt, &gen_cfg).expect("batched generate");
            let batched_ms = t0.elapsed().as_secs_f64() * 1000.0;

            assert_eq!(
                serial.token_ids, batched.token_ids,
                "TTFT sweep: token mismatch at words={words}"
            );

            println!(
                "{words}\t{}\t{serial_ms:.1}\t{batched_ms:.1}\t{:.3}",
                serial.prompt_tokens,
                serial_ms / batched_ms.max(1e-6),
            );
        }
    }
}
