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
use crate::sampling::compute_step_logprobs;
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
        // `DecodePolicy::init` (codex round-2 major #2, PR #787) constructs
        // the policy AND records this prefill-derived first token's logprob
        // in the same call -- replaces the freestanding `record_logprob(...)`
        // call this site used to make independently of the policy.
        let mut policy = DecodePolicy::init(
            gen_cfg,
            think_close_id,
            &mut token_logprobs,
            next_id,
            &scratch.logits[..cfg.vocab_size],
            gen_cfg.temperature,
            generated_ids.len(),
            false,
        );

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
                &mut policy,
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
            let mut full = String::new();

            // Tracks, per recorded `token_logprobs` entry, the length of `full`
            // immediately after that token's delta landed — grown in lockstep
            // with token_logprobs (both gated on gen_cfg.logprobs.is_some()), so
            // a stop-string truncation can drop exactly the trailing entries
            // whose text didn't fully survive. See
            // `truncate_token_logprobs_to_retained_text`.
            let mut token_logprob_end_offsets: Vec<usize> = Vec::new();

            // Check stop strings after the first token (codex round-3 major
            // #1, PR #787 / Leo's ruling): routed through the policy's owned
            // stop-mode adapter (`check_initial_stop`) instead of the free
            // `earliest_stop_match` call this site used to make directly --
            // `full`/`token_logprob_end_offsets` are populated by the adapter
            // itself, not by this call site.
            if matches!(
                policy.check_initial_stop(
                    &mut token_logprobs,
                    &mut full,
                    &mut token_logprob_end_offsets,
                    &first_delta,
                    |_| true,
                ),
                StopCheckOutcome::Stopped
            ) {
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
                &mut policy,
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
    ///
    /// Thin wrapper over [`Self::generate_streaming_with_observer`] with a
    /// no-op raw-event observer, so every existing caller (production
    /// serving paths, tests) keeps this exact 4-argument signature and zero
    /// behavior change.
    pub fn generate_streaming_with_cancel<F, C>(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
    ) -> Result<GenerateOutput, InferenceError>
    where
        F: FnMut(&str) -> bool,
        C: FnMut() -> bool,
    {
        self.generate_streaming_with_observer(prompt, gen_cfg, on_token, should_cancel, |_| {})
    }

    /// [`Self::generate_streaming_with_cancel`] plus a raw generation-lifecycle
    /// observer (benchmark-overhaul row 2, codex round-1 blocker #1): fires
    /// [`RawGenEvent::PrefillEnd`] once, after the prefill forward pass has
    /// produced logits and *before* the first token is sampled, and
    /// [`RawGenEvent::RawToken`] once per token that actually becomes part of
    /// `GenerateOutput` (both the prefill-derived first token and every
    /// decode-loop token), in generation order, with a 1-based
    /// monotonically-increasing `index` equal to `generated_ids.len()` at
    /// the moment of firing.
    ///
    /// This is deliberately independent of `on_token`'s text deltas: the
    /// incremental UTF-8 detokenizer buffers incomplete multi-byte
    /// codepoints, so one text delta is not guaranteed to equal one sampled
    /// token, and `prefill_end` measured off the first delta would fire
    /// *after* sampling rather than before it. `on_raw_event` fires exactly
    /// at the raw-token boundary and is unaffected by detokenizer buffering,
    /// so a caller measuring prefill/decode timing (`--emit-phase-events` in
    /// `qwen35_generate.rs`) gets an event count that always equals
    /// `GenerateOutput.generated_tokens` by construction: it is fired from
    /// the exact same `push` control-flow point that increments
    /// `generated_ids`, never from a step that grammar-stops or EOS-stops
    /// before the token is pushed.
    pub fn generate_streaming_with_observer<F, C, O>(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        mut on_token: F,
        mut should_cancel: C,
        mut on_raw_event: O,
    ) -> Result<GenerateOutput, InferenceError>
    where
        F: FnMut(&str) -> bool,
        C: FnMut() -> bool,
        O: FnMut(RawGenEvent),
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

        // Prefill logits are ready and the first token has not been sampled
        // yet -- the true prefill/decode boundary (codex round-1 blocker #1).
        // Fired unconditionally here (not gated on the eventual grammar/EOS
        // outcome below), since prefill itself always completed by this
        // point regardless of what the first sampled token turns out to be.
        on_raw_event(RawGenEvent::PrefillEnd);

        // Test-only seam (codex round-2 medium, PR #882): stamp "first sample
        // entered" right at the point sampling actually begins, so a test can
        // assert PrefillEnd fired before this instant, not merely before the
        // RawToken callback further below. No-op outside `cfg(test)`.
        #[cfg(test)]
        test_record_first_sample_entry();

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
        // Raw-token event for the prefill-derived first token, fired from the
        // exact point it becomes part of `generated_ids` -- symmetric with
        // the decode-loop `push` closures below, so the event count always
        // equals `GenerateOutput.generated_tokens`.
        on_raw_event(RawGenEvent::RawToken {
            index: generated_ids.len(),
        });

        // Budget forcing setup: resolve the </think> token id once and seed
        // the thinking_closed state from the prefill token so budget=1 works.
        let think_close_id = if gen_cfg.reasoning_budget.is_some() {
            self.tokenizer.special_token_id("</think>")
        } else {
            None
        };
        // `DecodePolicy::init` (codex round-2 major #2, PR #787) constructs
        // the policy AND records this prefill-derived first token's logprob
        // in the same call -- replaces the freestanding `record_logprob(...)`
        // call this site used to make independently of the policy.
        let mut policy = DecodePolicy::init(
            gen_cfg,
            think_close_id,
            &mut token_logprobs,
            next_id,
            &scratch.logits[..cfg.vocab_size],
            gen_cfg.temperature,
            generated_ids.len(),
            true,
        );

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
            let mut throwaway_offsets: Vec<usize> = Vec::new();
            let delta = detok.push(&self.tokenizer, next_id);
            // Codex round-3 major #1, PR #787 / Leo's ruling: routed through
            // the policy's owned stop-mode adapter (always `Disabled` here,
            // since `gen_cfg.stop_strings` is empty) instead of a manual
            // `text.push_str` + `on_token` call.
            if matches!(
                policy.check_initial_stop(
                    &mut token_logprobs,
                    &mut text,
                    &mut throwaway_offsets,
                    &delta,
                    |s| on_token(s),
                ),
                StopCheckOutcome::Interrupted
            ) {
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

            let mut stopped = false;
            let mut stopped_by_caller = false;
            let mut stop_reason = StopReason::Length;
            // Decode loop (mirrors decode_loop free function exactly).
            // cap = rb + max_new_tokens when budgeting; max_new_tokens otherwise (parity-safe).
            let cap = policy.cap();
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

                // One atomic per-step transition (ADR-080 C3 / codex round-1
                // major #3, PR #787) -- see `DecodePolicy::transition`. Set
                // stopped=true on a grammar stop so the caller sees a
                // grammar-terminal stop as stopped=true, matching
                // decode_loop's `return Ok(true)`. `policy.stop_mode` is
                // always `Disabled` on this path (`gen_cfg.stop_strings` is
                // empty); the adapter still threads text through to
                // `on_token` and reports `Interrupted` when the caller's sink
                // can no longer consume output (codex round-3 major #1, PR
                // #787 / Leo's ruling).
                let generated_len_before = generated_ids.len();
                let outcome = policy.transition(
                    &mut token_logprobs,
                    sampled_id,
                    &scratch.logits[..cfg.vocab_size],
                    gen_cfg.temperature,
                    generated_len_before,
                    |next_id| {
                        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state) {
                            engine.advance(gs, next_id)
                        } else {
                            true
                        }
                    },
                    |next_id| should_stop_token(cfg, gen_cfg, next_id),
                    |next_id| {
                        generated_ids.push(next_id);
                        all_ids.push(next_id);
                        // Raw-token event fired from the same `push` point
                        // that increments `generated_ids` -- never reached on
                        // a grammar-stop/EOS step that returns before `push`
                        // (codex round-1 blocker #1).
                        on_raw_event(RawGenEvent::RawToken {
                            index: generated_ids.len(),
                        });
                    },
                    |next_id| detok.push(&self.tokenizer, next_id),
                    &mut text,
                    &mut throwaway_offsets,
                    |s, _next_id| on_token(s),
                );

                let answer_budget_exhausted = match outcome {
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
                    StepOutcome::Interrupted => {
                        stopped_by_caller = true;
                        stop_reason = StopReason::Interrupt;
                        break;
                    }
                    StepOutcome::Stopped => {
                        // Unreachable on this path (`policy.stop_mode` is
                        // always `Disabled` here -- no `stop_strings`
                        // configured -- and `Disabled`'s `stop_check` arm
                        // never returns `StopCheckOutcome::Stopped`), handled
                        // for exhaustiveness/defense-in-depth.
                        stopped = true;
                        stop_reason = StopReason::Eos;
                        break;
                    }
                    StepOutcome::Emitted {
                        answer_budget_exhausted,
                        ..
                    } => answer_budget_exhausted,
                };

                // Answer-budget break: stop once max_new_tokens answer tokens follow </think>.
                if answer_budget_exhausted {
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
            // String-stop path: `policy.stop_mode` is `StopMode::Streaming`
            // (constructed in `DecodePolicy::init` above from the same
            // non-empty `gen_cfg.stop_strings`), holding back (max_stop - 1)
            // bytes so a partial stop prefix is never emitted before it is
            // confirmed not to be a match.
            let mut text = String::new();
            let mut throwaway_offsets: Vec<usize> = Vec::new();
            let first_delta = detok.push(&self.tokenizer, next_id);
            let initial_outcome = policy.check_initial_stop(
                &mut token_logprobs,
                &mut text,
                &mut throwaway_offsets,
                &first_delta,
                |s| on_token(s),
            );
            if matches!(initial_outcome, StopCheckOutcome::Interrupted) {
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
            if matches!(initial_outcome, StopCheckOutcome::Stopped) {
                // Stop matched in the very first token.
                // token_ids already contain next_id; cannot un-generate it.
                return Ok(GenerateOutput {
                    text,
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
            let cap = policy.cap();
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

                // One atomic per-step transition (ADR-080 C3 / codex round-1
                // major #3, PR #787) -- see `DecodePolicy::transition`.
                // `policy.stop_mode` (fixed to `StopMode::Streaming` at
                // construction) owns the incremental byte-holdback
                // stop-string match itself now (codex round-3 major #1, PR
                // #787 / Leo's ruling) -- this call site supplies only
                // `decode_delta` (this loop's own detokenizer) and the
                // shared `text`/`throwaway_offsets` buffers, so it can no
                // longer independently choose to skip the real stop check.
                let generated_len_before = generated_ids.len();
                let outcome = policy.transition(
                    &mut token_logprobs,
                    sampled_id,
                    &scratch.logits[..cfg.vocab_size],
                    gen_cfg.temperature,
                    generated_len_before,
                    |next_id| {
                        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state) {
                            engine.advance(gs, next_id)
                        } else {
                            true
                        }
                    },
                    |next_id| should_stop_token(cfg, gen_cfg, next_id),
                    |next_id| {
                        generated_ids.push(next_id);
                        all_ids.push(next_id);
                        // Raw-token event fired from the same `push` point
                        // that increments `generated_ids` -- never reached on
                        // a grammar-stop/EOS step that returns before `push`
                        // (codex round-1 blocker #1).
                        on_raw_event(RawGenEvent::RawToken {
                            index: generated_ids.len(),
                        });
                    },
                    |next_id| detok.push(&self.tokenizer, next_id),
                    &mut text,
                    &mut throwaway_offsets,
                    |s, _next_id| on_token(s),
                );

                let answer_budget_exhausted = match outcome {
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
                    StepOutcome::Interrupted => {
                        stopped_by_caller = true;
                        stop_reason = StopReason::Interrupt;
                        break;
                    }
                    StepOutcome::Stopped => {
                        stopped = true;
                        stop_reason = StopReason::Eos;
                        break;
                    }
                    StepOutcome::Emitted {
                        answer_budget_exhausted,
                        ..
                    } => answer_budget_exhausted,
                };

                // Answer-budget break: stop once max_new_tokens answer tokens follow </think>.
                if answer_budget_exhausted {
                    break;
                }
            }

            // Natural-end flush (no-op if a stop was already hit inside the loop).
            // Skip when the caller asked to stop -- it is no longer consuming
            // the stream, and `on_token`'s return value here would not change
            // why generation actually stopped.
            if !stopped_by_caller {
                let tail_stopped = policy.finish_stop(&mut text, &detok.finish(), |s| on_token(s));
                // finish_stop may itself complete a stop in the tail bytes.
                if tail_stopped && !stopped {
                    stopped = true;
                    stop_reason = StopReason::Eos;
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
        }
    }
}

/// Raw generation-lifecycle event fired by
/// [`Qwen35Model::generate_streaming_with_observer`], independent of the
/// caller's text-delta callback and unaffected by incremental UTF-8
/// detokenizer buffering (benchmark-overhaul row 2, codex round-1 blocker
/// #1: a text delta is not guaranteed to equal one raw sampled token, and
/// measuring `prefill_end` off the first delta fires it after sampling
/// instead of before).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawGenEvent {
    /// Fired exactly once, after the prefill forward pass has produced
    /// logits and before the first token is sampled -- the true
    /// prefill/decode boundary.
    PrefillEnd,
    /// Fired once per token that actually becomes part of
    /// `GenerateOutput.token_ids` (never for a token that grammar-stops or
    /// EOS-stops before being pushed), in generation order. `index` is
    /// 1-based and monotonically increasing, equal to `generated_ids.len()`
    /// at the moment of firing -- so a caller counting these events always
    /// gets a count equal to `GenerateOutput.generated_tokens`.
    RawToken { index: usize },
}

// Test-only seam (codex round-2 medium finding, PR #882): the existing
// mutation-sensitive test below only proves `PrefillEnd` fires before the
// `RawToken` *callback*, not before `sample_token` itself. A `PrefillEnd`
// moved to just after `generated_ids.push` (after sampling, before the
// `RawToken` callback) would keep that test green while silently folding
// sampling time into the reported prefill interval. `test_record_first_sample_entry`
// is called from the exact point `sample_token` is about to run for the
// prefill-derived first token; a test's `on_raw_event` closure calls
// `test_mark_prefill_end_seen` when it observes `PrefillEnd`, and the two
// are compared to answer "did PrefillEnd fire before the *first sample*",
// not just "before the first raw-token callback". Zero production effect:
// every item here is `#[cfg(test)]` and compiles to nothing outside tests.
#[cfg(test)]
thread_local! {
    static TEST_PREFILL_END_SEEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static TEST_FIRST_SAMPLE_SAW_PREFILL_END: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Resets the seam's thread-local state. Call at the start of any test that
/// reads `test_take_first_sample_saw_prefill_end` afterward.
#[cfg(test)]
fn test_reset_sample_seam() {
    TEST_PREFILL_END_SEEN.with(|c| c.set(false));
    TEST_FIRST_SAMPLE_SAW_PREFILL_END.with(|c| c.set(None));
}

/// Called by a test's `on_raw_event` closure when it observes
/// `RawGenEvent::PrefillEnd`.
#[cfg(test)]
fn test_mark_prefill_end_seen() {
    TEST_PREFILL_END_SEEN.with(|c| c.set(true));
}

/// Called from production code (under `#[cfg(test)]`) immediately before the
/// first call to `sample_token`. Records, the first time only, whether
/// `PrefillEnd` had already been observed by that point.
#[cfg(test)]
fn test_record_first_sample_entry() {
    TEST_FIRST_SAMPLE_SAW_PREFILL_END.with(|seen_at_sample| {
        if seen_at_sample.get().is_none() {
            let seen = TEST_PREFILL_END_SEEN.with(std::cell::Cell::get);
            seen_at_sample.set(Some(seen));
        }
    });
}

/// Reads the value recorded by `test_record_first_sample_entry`.
#[cfg(test)]
fn test_take_first_sample_saw_prefill_end() -> Option<bool> {
    TEST_FIRST_SAMPLE_SAW_PREFILL_END.with(std::cell::Cell::get)
}

/// Outcome of [`DecodePolicy::transition`], the one per-step call every decode
/// loop drives through (ADR-080 C3 / codex round-1 major #3, PR #787).
pub(crate) enum StepOutcome {
    /// The (possibly budget-overridden) token was rejected by the backend's
    /// own grammar advance before ever being pushed. The loop must stop with
    /// `stopped = true`, `stop_reason = Grammar`.
    GrammarStop,
    /// The (possibly budget-overridden) token is EOS / a stop-token id and
    /// was never pushed. The loop must stop with `stopped = true`,
    /// `stop_reason = Eos`.
    Eos,
    /// [`DecodePolicy::stop_check`] — driven internally from `self.stop_mode`
    /// (codex round-3 major #1, PR #787 / Leo's ruling; see [`StopMode`]) —
    /// reported that a configured stop string matched as of this token. The
    /// token was pushed (via `push`) and every other backend-neutral
    /// per-step control already applied before `stop_check` ran; the loop
    /// must stop with `stopped = true`, `stop_reason = Eos`.
    Stopped,
    /// [`DecodePolicy::stop_check`] reported that the caller's
    /// streaming sink (`emit_confirmed`) can no longer consume output (e.g. a
    /// dropped SSE receiver) — not a stop condition. The loop must stop with
    /// `stopped = false`, `stop_reason = Interrupt`.
    Interrupted,
    /// The token was pushed (via the caller's `push` callback), `stop_check`
    /// reported [`StopCheckOutcome::Continue`], and every backend-neutral
    /// per-step control (logprobs, reasoning-end capture, answer-budget
    /// accounting) has already been applied for it.
    Emitted {
        /// The actually-emitted token id (post budget-override).
        token_id: u32,
        /// Whether the answer-budget window has closed as of this token —
        /// the loop should break after this iteration (in addition to its
        /// normal cap) when this is `true`.
        answer_budget_exhausted: bool,
    },
}

/// Outcome of the mandatory per-step stop-check [`DecodePolicy::transition`]
/// drives internally (codex round-3 major #1, PR #787: the round-2 mandatory
/// `stop_check` closure could still compile as a trivial
/// `|_, _| StopCheckOutcome::Continue` for a configuration that actually had
/// stop strings — an arbitrary outcome-producing closure cannot be forced to
/// consult real matcher state. `transition` no longer accepts one at all; see
/// [`StopMode`] and [`DecodePolicy::stop_check`]).
pub(crate) enum StopCheckOutcome {
    /// No stop-string match yet (or `stop_strings` is not configured for
    /// this generation at all) — keep decoding.
    Continue,
    /// A configured stop string matched as of this token. The callback has
    /// already truncated/finalized the backend's own accumulated output
    /// (text buffer or streaming sink) before returning this.
    Stopped,
    /// The caller's streaming sink (`on_token`) signaled it can no longer
    /// consume output — not a stop condition.
    Interrupted,
}

/// Backend-neutral decode-policy state (ADR-080 C3): reasoning-budget
/// accounting and logprobs formatting, shared by every canonical/streaming
/// decode loop — CPU [`decode_loop`], [`decode_loop_with_stops`], both
/// branches of [`Qwen35Model::generate_streaming_with_cancel`], and the Metal
/// `generate_streaming` / `generate_streaming_with_prefix_cache_and_cancel_inner`
/// loops in `crate::forward::metal_qwen35` — via one atomic per-step
/// transition ([`DecodePolicy::transition`]): each backend keeps
/// `forward_step`, grammar masking, sampling, and its own token vectors
/// (`generated_ids` / `all_ids` or the Metal equivalents) entirely to itself,
/// hands `transition` the token its own pipeline just sampled plus three
/// backend callbacks (grammar-advance, EOS/stop-token check, the push into
/// its own vectors) and raw per-token I/O primitives for the stop check
/// (`decode_delta`, a `text`/`token_logprob_end_offsets` buffer pair, and
/// `emit_confirmed` — see [`StopMode`] below), and gets back a
/// [`StepOutcome`] that already reflects budget-override, reasoning-block
/// tracking, logprobs recording, reasoning-end capture, the stop check, and
/// the answer-budget check — in that fixed order, every time, for every
/// site.
///
/// Before this struct existed, this exact bookkeeping (`think_close_id`
/// resolution, `thinking_closed` / `reasoning_end_len` tracking, the
/// `decode_cap` / `force_close_think` calls, and the answer-budget break
/// condition) was hand-duplicated across six independent decode loops —
/// exactly the drift ADR-080 C3 exists to prevent: a seventh loop could add
/// its own copy and silently diverge from the other six. The struct
/// originally exposed each of these as a separate method
/// (`apply_override` / `note_emitted` / `record_logprob` /
/// `capture_reasoning_end` / `answer_budget_exhausted`), which let a call
/// site choreograph a subset of them and skip another — codex round-1
/// blocker #1 was exactly that failure mode: the Metal prefix-cache loop
/// called four of the five and silently never called `record_logprob`.
/// `transition` replaces all five with the one call above; the five
/// constituent methods are now private to this module, so a caller in a
/// different module (e.g. `crate::forward::metal_qwen35`) cannot reach any of
/// them individually even by mistake — omitting `transition` is the only way
/// to skip a control, and doing so breaks every one of these behaviors at
/// once rather than silently dropping just one.
///
/// Stop-string matching (codex round-3 major #1, PR #787 / Leo's Option A
/// ruling): the streaming vs non-streaming consumption shapes genuinely
/// differ (incremental byte-holdback via [`StopStringMatcher`] vs full-text
/// rescan via `earliest_stop_match_from`), but which one applies — and
/// whether checking happens at all — is now [`StopMode`], a value chosen
/// exactly once from the real `gen_cfg.stop_strings` at [`DecodePolicy::init`]
/// time and stored privately on the policy. A caller can no longer supply a
/// closure that *decides* the stop outcome (round 2's `stop_check` parameter,
/// which could compile as a trivial `|_, _| Continue` for any configuration
/// regardless of what `stop_strings` actually held); it supplies only raw
/// per-token I/O primitives — a decoded delta (`decode_delta`) and a
/// confirmed-text sink (`emit_confirmed`) — and [`DecodePolicy::stop_check`]
/// (called from both [`DecodePolicy::check_initial_stop`], for the
/// prefill-derived first token, and `transition`, for every token after)
/// dispatches on `self.stop_mode` to decide, using the real adapter for that
/// mode, not caller-supplied decision logic.
pub(crate) struct DecodePolicy {
    reasoning_budget: Option<usize>,
    enable_thinking: bool,
    max_new_tokens: usize,
    logprobs: Option<usize>,
    think_close_id: Option<u32>,
    thinking_closed: bool,
    reasoning_end_len: Option<usize>,
    stop_mode: StopMode,
}

/// The stop-string check adapter a [`DecodePolicy`] owns (codex round-3
/// major #1, PR #787 / Leo's Option A ruling). The only place a value of this
/// type is ever produced is the private [`StopMode::for_config`], called once
/// from [`DecodePolicy::init`] on the real `gen_cfg.stop_strings` — there is
/// no public constructor, so a caller cannot independently choose (or swap
/// in) `Disabled` for a configuration that actually has stop strings: the
/// variant a given policy drives is fixed by the config it was built from,
/// not by anything a call site writes.
enum StopMode {
    /// `gen_cfg.stop_strings` was empty at construction — there is nothing to
    /// match, so [`DecodePolicy::stop_check`] only threads decoded text
    /// through to the caller's sink (still needed for streaming callers'
    /// `on_token`; a no-op for `decode_loop`, which has no text pipeline at
    /// all).
    Disabled,
    /// Streaming incremental byte-holdback: the owned [`StopStringMatcher`]
    /// ensures a partial match never reaches the caller's confirmed-text
    /// sink. Used by every streaming call site with `stop_strings` set (CPU
    /// `generate_streaming_with_cancel`'s stop-string branch, both Metal
    /// streaming loops).
    Streaming(StopStringMatcher),
    /// Non-streaming full-text rescan, bounded to the suffix that could
    /// contain a new match (`stop_scan_search_start`). Used only by CPU
    /// `decode_loop_with_stops` (via `Qwen35Model::generate`'s stop-string
    /// branch), which has no external consumer to hold text back from.
    FullScan {
        stop_strings: Vec<String>,
        max_stop: usize,
    },
}

impl StopMode {
    fn for_config(stop_strings: &[String], streaming: bool) -> Self {
        if stop_strings.is_empty() {
            StopMode::Disabled
        } else if streaming {
            StopMode::Streaming(StopStringMatcher::new(stop_strings))
        } else {
            let max_stop = stop_strings.iter().map(String::len).max().unwrap_or(1);
            StopMode::FullScan {
                stop_strings: stop_strings.to_vec(),
                max_stop,
            }
        }
    }
}

impl DecodePolicy {
    /// The first-step transition (codex round-2 major #2, PR #787 / Leo's
    /// ruling): constructs the policy AND atomically records the
    /// prefill-derived first token's logprob in the same call, so there is no
    /// longer any way to build a `DecodePolicy` without also recording its
    /// first token's logprob. Before this, `new()` only built the struct and
    /// left every call site to separately invoke the freestanding
    /// `crate::sampling::record_logprob` for that one token — three
    /// call sites (this module's `generate()` / `generate_streaming_with_cancel()`,
    /// and Metal's `generate_streaming`) duplicated that call independently,
    /// the exact drift pattern codex's round-1 blocker #1 already proved live
    /// once for the *other* four constituent methods (see the struct-level
    /// doc comment above).
    ///
    /// `think_close_id` is resolved by the caller (`tokenizer.special_token_id("</think>")`
    /// when `gen_cfg.reasoning_budget.is_some()`, `None` otherwise) since each backend
    /// reaches its tokenizer differently. `first_emitted_id` / `first_generated_len` seed
    /// `thinking_closed` / `reasoning_end_len` from the token already sampled and pushed
    /// before the decode loop starts (the prefill-derived first token), covering the
    /// `reasoning_budget == 1` edge case exactly as the six duplicated call sites did.
    /// `first_logits` / `temperature` are the same values the free-function
    /// `record_logprob` call used to take directly. `streaming` selects which
    /// [`StopMode`] a non-empty `gen_cfg.stop_strings` resolves to
    /// (`Streaming`'s incremental holdback vs `FullScan`'s full-text rescan;
    /// see [`StopMode::for_config`]) — pass `true` for every streaming caller
    /// (CPU `generate_streaming_with_cancel`, both Metal streaming loops),
    /// `false` for non-streaming callers (`Qwen35Model::generate`). An empty
    /// `stop_strings` always resolves to `Disabled` regardless of `streaming`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn init(
        gen_cfg: &GenerateConfig,
        think_close_id: Option<u32>,
        token_logprobs: &mut Vec<TokenLogprob>,
        first_emitted_id: u32,
        first_logits: &[f32],
        temperature: f32,
        first_generated_len: usize,
        streaming: bool,
    ) -> Self {
        let thinking_closed = Some(first_emitted_id) == think_close_id;
        let reasoning_end_len = if thinking_closed {
            Some(first_generated_len)
        } else {
            None
        };
        let policy = Self {
            reasoning_budget: gen_cfg.reasoning_budget,
            enable_thinking: gen_cfg.enable_thinking,
            max_new_tokens: gen_cfg.max_new_tokens,
            logprobs: gen_cfg.logprobs,
            think_close_id,
            thinking_closed,
            reasoning_end_len,
            stop_mode: StopMode::for_config(&gen_cfg.stop_strings, streaming),
        };
        policy.record_logprob(token_logprobs, first_logits, first_emitted_id, temperature);
        policy
    }

    /// Total decode-loop iteration cap (`rb + max_new_tokens + 1` when budgeted,
    /// `max_new_tokens` otherwise) — see [`decode_cap`].
    pub(crate) fn cap(&self) -> usize {
        decode_cap(self.reasoning_budget, self.max_new_tokens)
    }

    /// Overrides `sampled_id` with the forced `</think>` token when the reasoning
    /// budget is exhausted and the block is still open; a no-op pass-through
    /// otherwise. Call after sampling, before grammar-advance (the actually-emitted
    /// token, post-override, is what grammar must advance on).
    ///
    /// Private (codex round-1 major #3, PR #787): only reachable through
    /// [`DecodePolicy::transition`], which owns the full per-step ordering.
    fn apply_override(&self, generated_len: usize, sampled_id: u32) -> u32 {
        force_close_think(
            self.reasoning_budget,
            self.enable_thinking,
            self.thinking_closed,
            generated_len,
            self.think_close_id,
        )
        .unwrap_or(sampled_id)
    }

    /// Marks the thinking block closed when `next_id` (the actually-emitted,
    /// post-override token) is the `</think>` token. Call after grammar-advance
    /// succeeds, before the EOS/stop-token check — mirrors the original inline
    /// ordering across all six sites.
    ///
    /// Private (codex round-1 major #3, PR #787): only reachable through
    /// [`DecodePolicy::transition`], which owns the full per-step ordering.
    fn note_emitted(&mut self, next_id: u32) {
        if Some(next_id) == self.think_close_id {
            self.thinking_closed = true;
        }
    }

    /// Captures the answer-budget window start the first time the thinking block
    /// closes, using the generated-token count *after* the token was pushed (so
    /// `</think>` itself is the last reasoning token, not the first answer token).
    /// A no-op once already captured or while the block is still open.
    ///
    /// Private (codex round-1 major #3, PR #787): only reachable through
    /// [`DecodePolicy::transition`], which owns the full per-step ordering.
    fn capture_reasoning_end(&mut self, generated_len_after_push: usize) {
        if self.thinking_closed && self.reasoning_end_len.is_none() {
            self.reasoning_end_len = Some(generated_len_after_push);
        }
    }

    /// True once `max_new_tokens` answer tokens have followed the `</think>` close
    /// point — the decode loop should break on this, in addition to its normal cap.
    ///
    /// Private (codex round-1 major #3, PR #787): only reachable through
    /// [`DecodePolicy::transition`], which owns the full per-step ordering.
    fn answer_budget_exhausted(&self, generated_len: usize) -> bool {
        self.reasoning_end_len
            .is_some_and(|end| generated_len.saturating_sub(end) >= self.max_new_tokens)
    }

    /// Appends one decode step's logprob data to `token_logprobs` when
    /// `self.logprobs` requests it; a no-op otherwise (so callers can invoke
    /// it unconditionally on every step -- the softmax pass over the full
    /// vocabulary is paid only when logprobs were actually requested).
    ///
    /// This is the ONLY place in the crate that pushes onto a
    /// `token_logprobs: &mut Vec<TokenLogprob>` accumulator (codex round-3
    /// medium #2, PR #787 / Leo's ruling): `crate::sampling` exposes only
    /// the pure computation (`compute_step_logprobs`), not a freestanding
    /// "record" function a sibling decode call site could invoke directly
    /// to recreate the exact duplicate-choreography bug this method's
    /// privacy already closes for the other four constituent methods.
    ///
    /// Private (codex round-1 major #3, PR #787): only reachable through
    /// [`DecodePolicy::transition`] / [`DecodePolicy::init`], which own the
    /// full per-step ordering.
    fn record_logprob(
        &self,
        token_logprobs: &mut Vec<TokenLogprob>,
        logits: &[f32],
        token_id: u32,
        temperature: f32,
    ) {
        let Some(top_n) = self.logprobs else {
            return;
        };
        let (logprob, top) = compute_step_logprobs(logits, token_id, temperature, top_n);
        token_logprobs.push(TokenLogprob {
            token_id,
            logprob,
            top,
        });
    }

    /// The stop-check adapter dispatch (codex round-3 major #1, PR #787 /
    /// Leo's Option A ruling): drives whichever [`StopMode`] this policy was
    /// constructed with, given the caller's freshly decoded delta text for
    /// the current token. The caller supplies no decision logic at all — only
    /// the decoded text and a sink for whatever text is confirmed safe to
    /// release (`emit_confirmed`, called with the post-holdback-safe
    /// substring for `Streaming`, the raw delta for `Disabled`, never for
    /// `FullScan`, which has no external consumer). Shared by
    /// [`DecodePolicy::check_initial_stop`] (the prefill-derived first token,
    /// called once before the decode loop) and `transition` (every token
    /// after) — the same `self.stop_mode` instance is mutated across both
    /// calls, so `Streaming`'s incremental byte-holdback state carries over
    /// correctly from the first token onward, exactly as it did when each
    /// call site constructed and drove its own matcher by hand.
    ///
    /// Private (codex round-1 major #3 / round-3 major #1, PR #787): only
    /// reachable through the two methods above.
    fn stop_check(
        &mut self,
        token_logprobs: &mut Vec<TokenLogprob>,
        text: &mut String,
        token_logprob_end_offsets: &mut Vec<usize>,
        delta: &str,
        mut emit_confirmed: impl FnMut(&str) -> bool,
    ) -> StopCheckOutcome {
        match &mut self.stop_mode {
            StopMode::Disabled => {
                if delta.is_empty() {
                    return StopCheckOutcome::Continue;
                }
                text.push_str(delta);
                if emit_confirmed(delta) {
                    StopCheckOutcome::Continue
                } else {
                    StopCheckOutcome::Interrupted
                }
            }
            StopMode::Streaming(matcher) => {
                let mut interrupted = false;
                let stop_matched = matcher.push(delta, &mut |s| {
                    if !s.is_empty() {
                        text.push_str(s);
                        if !interrupted && !emit_confirmed(s) {
                            interrupted = true;
                        }
                    }
                });
                if interrupted {
                    StopCheckOutcome::Interrupted
                } else if stop_matched {
                    StopCheckOutcome::Stopped
                } else {
                    StopCheckOutcome::Continue
                }
            }
            StopMode::FullScan {
                stop_strings,
                max_stop,
            } => {
                let prev_len = text.len();
                if !delta.is_empty() {
                    text.push_str(delta);
                }
                // Keep the offset tracker in lockstep with token_logprobs'
                // conditional growth (record_logprob is a no-op unless
                // gen_cfg.logprobs is set).
                if token_logprobs.len() > token_logprob_end_offsets.len() {
                    token_logprob_end_offsets.push(text.len());
                }
                let search_start = stop_scan_search_start(text, prev_len, *max_stop);
                if let Some(hit) = earliest_stop_match_from(text, stop_strings, search_start) {
                    text.truncate(hit);
                    truncate_token_logprobs_to_retained_text(
                        token_logprobs,
                        token_logprob_end_offsets,
                        hit,
                    );
                    StopCheckOutcome::Stopped
                } else {
                    StopCheckOutcome::Continue
                }
            }
        }
    }

    /// Checks the prefill-derived first token's already-decoded delta text
    /// against this policy's stop-mode (codex round-3 major #1, PR #787 /
    /// Leo's ruling), before the decode loop starts — the first token is
    /// pushed and its logprob recorded by [`DecodePolicy::init`] outside
    /// `transition`'s per-step scope (it has no preceding grammar-advance /
    /// EOS check of its own to run through `transition` for), so its
    /// stop-string check needs its own entry point. Uses the SAME
    /// `self.stop_mode` instance `transition` will keep driving for every
    /// subsequent token, so `Streaming`'s byte-holdback state is continuous
    /// across the boundary — critical for a match that spans the first and
    /// second tokens, which a freshly-constructed second matcher would miss.
    pub(crate) fn check_initial_stop(
        &mut self,
        token_logprobs: &mut Vec<TokenLogprob>,
        text: &mut String,
        token_logprob_end_offsets: &mut Vec<usize>,
        delta: &str,
        emit_confirmed: impl FnMut(&str) -> bool,
    ) -> StopCheckOutcome {
        self.stop_check(
            token_logprobs,
            text,
            token_logprob_end_offsets,
            delta,
            emit_confirmed,
        )
    }

    /// The one per-step transition (ADR-080 C3 / codex round-1 major #3, PR
    /// #787): atomically applies, in the fixed order every decode loop
    /// requires, the reasoning-budget override, the backend's grammar-advance
    /// callback, the emitted-token bookkeeping, the backend's EOS/stop-token
    /// callback, the backend's push callback (into its own `generated_ids` /
    /// `all_ids` or Metal-equivalent vectors), per-token logprobs recording,
    /// reasoning-end capture, the owned stop-check adapter, and the
    /// answer-budget check.
    ///
    /// `grammar_advance` and `is_eos` are backend callbacks because grammar
    /// masking/advance and EOS/stop-token identification remain genuinely
    /// backend-specific per ADR-080 C3 scope — each backend owns its own
    /// `GrammarState` and `cfg.eos_token_id` / `stop_token_ids` wiring, and
    /// `grammar_advance` must run on the *actually-emitted* (post-override)
    /// token before `is_eos` sees it, exactly mirroring the inline ordering
    /// every site used before this method existed. `push` is a callback
    /// because the token vectors are owned by the caller and are also read on
    /// the *next* loop iteration (`all_ids.last()` feeds the next
    /// `forward_step`) — the caller cannot hand that ownership to the policy.
    ///
    /// `decode_delta` / `text` / `token_logprob_end_offsets` / `emit_confirmed`
    /// (codex round-3 major #1, PR #787 / Leo's ruling) replace round-2's
    /// arbitrary outcome-producing `stop_check` closure: the caller supplies
    /// only raw I/O (decode a token to text; a buffer to accumulate into; an
    /// offset tracker only `StopMode::FullScan` consults; a sink for
    /// confirmed-safe text), and [`DecodePolicy::stop_check`] — driven from
    /// `self.stop_mode`, fixed at construction from the real
    /// `gen_cfg.stop_strings` — decides the outcome. A call site can no
    /// longer claim `Continue` for a configuration that actually has stop
    /// strings, because it no longer produces the outcome at all.
    ///
    /// Returns [`StepOutcome::GrammarStop`] / [`StepOutcome::Eos`] without
    /// ever calling `push` when the token is rejected before emission
    /// (matching the existing contract that a stop token is never present in
    /// `token_ids`); [`StepOutcome::Stopped`] / [`StepOutcome::Interrupted`]
    /// when the stop-check adapter reports either outcome (the answer-budget
    /// check is skipped in both cases, matching every site's original control
    /// flow, which broke out of the loop before ever reaching it); or
    /// [`StepOutcome::Emitted`] once the token has been pushed and every
    /// remaining control, including a `Continue` stop-check, applied.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transition(
        &mut self,
        token_logprobs: &mut Vec<TokenLogprob>,
        sampled_id: u32,
        logits: &[f32],
        temperature: f32,
        generated_len_before: usize,
        mut grammar_advance: impl FnMut(u32) -> bool,
        mut is_eos: impl FnMut(u32) -> bool,
        mut push: impl FnMut(u32),
        mut decode_delta: impl FnMut(u32) -> String,
        text: &mut String,
        token_logprob_end_offsets: &mut Vec<usize>,
        mut emit_confirmed: impl FnMut(&str, u32) -> bool,
    ) -> StepOutcome {
        let next_id = self.apply_override(generated_len_before, sampled_id);

        if !grammar_advance(next_id) {
            return StepOutcome::GrammarStop;
        }

        self.note_emitted(next_id);

        if is_eos(next_id) {
            return StepOutcome::Eos;
        }

        push(next_id);
        let generated_len_after = generated_len_before + 1;

        self.record_logprob(token_logprobs, logits, next_id, temperature);
        self.capture_reasoning_end(generated_len_after);

        let delta = decode_delta(next_id);
        let stop_outcome = self.stop_check(
            token_logprobs,
            text,
            token_logprob_end_offsets,
            &delta,
            |s| emit_confirmed(s, next_id),
        );
        match stop_outcome {
            StopCheckOutcome::Stopped => return StepOutcome::Stopped,
            StopCheckOutcome::Interrupted => return StepOutcome::Interrupted,
            StopCheckOutcome::Continue => {}
        }

        StepOutcome::Emitted {
            token_id: next_id,
            answer_budget_exhausted: self.answer_budget_exhausted(generated_len_after),
        }
    }

    /// Natural-end flush (decode loop ended by cap/EOS/grammar-stop, not by
    /// `stop_check` reporting `Stopped`/`Interrupted`). `tail` is the
    /// detokenizer's own end-of-generation flush (`detok.finish()`).
    ///
    /// A no-op for `Disabled` beyond appending+emitting `tail` directly (there
    /// is nothing held back to reconcile) and for `FullScan` (the
    /// non-streaming caller owns its own tail-flush against its `full` buffer
    /// directly, e.g. `decode_loop_with_stops`, since it has no external
    /// consumer to hold text back from in the first place). Only `Streaming`
    /// mode's owned [`StopStringMatcher`] can be holding back up to
    /// `max_stop - 1` unconfirmed bytes that must be reconciled once the
    /// token source is exhausted — mirrors `StopStringMatcher::finish`
    /// exactly, since that is the only mode this call does real work for.
    ///
    /// Returns `true` when the tail flush itself completed a stop match
    /// (`Streaming` only; always `false` for `Disabled`/`FullScan`).
    pub(crate) fn finish_stop(
        &mut self,
        text: &mut String,
        tail: &str,
        mut emit_confirmed: impl FnMut(&str) -> bool,
    ) -> bool {
        match &mut self.stop_mode {
            StopMode::Disabled => {
                if !tail.is_empty() {
                    text.push_str(tail);
                    emit_confirmed(tail);
                }
                false
            }
            StopMode::Streaming(matcher) => {
                matcher.finish(tail, &mut |s| {
                    if !s.is_empty() {
                        text.push_str(s);
                        emit_confirmed(s);
                    }
                });
                matcher.stopped()
            }
            StopMode::FullScan { .. } => false,
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
    policy: &mut DecodePolicy,
    token_logprobs: &mut Vec<TokenLogprob>,
) -> Result<(bool, StopReason), InferenceError> {
    let cfg = &model.config;
    let cap = policy.cap();
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

        // One atomic per-step transition (ADR-080 C3 / codex round-1 major
        // #3, PR #787): budget override, grammar-advance callback, emitted
        // bookkeeping, EOS callback, push callback, logprobs, reasoning-end
        // capture, the stop-check adapter, and the answer-budget check, all
        // in the fixed required order -- see `DecodePolicy::transition`. This
        // function is only ever called when `gen_cfg.stop_strings` is empty
        // (see `generate()`'s branch), so `policy.stop_mode` is always
        // `StopMode::Disabled` here, and this function has no text/detok
        // pipeline of its own at all (it returns raw token ids, decoded once
        // at the very end by `decode_tokens`) -- `decode_delta`/`text`/
        // `token_logprob_end_offsets`/`emit_confirmed` are therefore
        // throwaway values the `Disabled` dispatch never populates
        // meaningfully (codex round-3 major #1, PR #787 / Leo's ruling: this
        // is honest, not an escape hatch -- the empty-stop_strings guarantee
        // now lives in `policy.stop_mode`, derived from the real config at
        // `DecodePolicy::init`, not in a caller-chosen closure).
        let generated_len_before = generated_ids.len();
        let mut throwaway_text = String::new();
        let mut throwaway_offsets: Vec<usize> = Vec::new();
        let outcome = policy.transition(
            token_logprobs,
            sampled_id,
            &scratch.logits[..cfg.vocab_size],
            gen_cfg.temperature,
            generated_len_before,
            |next_id| {
                if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut *grammar_state) {
                    engine.advance(gs, next_id)
                } else {
                    true
                }
            },
            |next_id| should_stop_token(cfg, gen_cfg, next_id),
            |next_id| {
                generated_ids.push(next_id);
                all_ids.push(next_id);
            },
            |_next_id| String::new(),
            &mut throwaway_text,
            &mut throwaway_offsets,
            |_delta, _next_id| true,
        );

        match outcome {
            StepOutcome::GrammarStop => return Ok((true, StopReason::Grammar)),
            StepOutcome::Eos => return Ok((true, StopReason::Eos)),
            StepOutcome::Stopped => return Ok((true, StopReason::Eos)),
            StepOutcome::Interrupted => return Ok((false, StopReason::Interrupt)),
            StepOutcome::Emitted {
                answer_budget_exhausted,
                ..
            } => {
                // Answer-budget break: stop once max_new_tokens answer tokens
                // follow </think>.
                if answer_budget_exhausted {
                    break;
                }
            }
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
    policy: &mut DecodePolicy,
    token_logprobs: &mut Vec<TokenLogprob>,
    token_logprob_end_offsets: &mut Vec<usize>,
) -> Result<(bool, StopReason), InferenceError> {
    let cfg = &model.config;
    let mut stopped = false;
    let mut stop_reason = StopReason::Length;
    let cap = policy.cap();
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

        // One atomic per-step transition (ADR-080 C3 / codex round-1 major
        // #3, PR #787) -- see `DecodePolicy::transition`. The stop-check
        // adapter (codex round-3 major #1, PR #787 / Leo's ruling) now owns
        // the rescan/truncate work this loop used to run itself in a
        // `stop_check` closure -- this call site supplies only `decode_delta`
        // (this loop's own detokenizer) and the shared `full`/
        // `token_logprob_end_offsets` buffers; `policy.stop_mode` (fixed to
        // `StopMode::FullScan` at construction, since this function is only
        // called when `gen_cfg.stop_strings` is non-empty) does the actual
        // rescan/truncate, not caller-supplied closure logic.
        let generated_len_before = generated_ids.len();
        let outcome = policy.transition(
            token_logprobs,
            sampled_id,
            &scratch.logits[..cfg.vocab_size],
            gen_cfg.temperature,
            generated_len_before,
            |next_id| {
                if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut *grammar_state) {
                    engine.advance(gs, next_id)
                } else {
                    true
                }
            },
            |next_id| should_stop_token(cfg, gen_cfg, next_id),
            |next_id| {
                generated_ids.push(next_id);
                all_ids.push(next_id);
            },
            |next_id| detok.push(&model.tokenizer, next_id),
            full,
            token_logprob_end_offsets,
            |_delta, _next_id| true,
        );

        let (_next_id, answer_budget_exhausted) = match outcome {
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
                stop_reason = StopReason::Eos;
                break;
            }
            StepOutcome::Interrupted => {
                // Unreachable on this path (`policy.stop_mode` is
                // `StopMode::FullScan` here, whose `stop_check` arm never
                // returns `StopCheckOutcome::Interrupted` -- only
                // `StopMode::Streaming`'s arm does, for the streaming call
                // sites), handled for exhaustiveness/defense-in-depth.
                stop_reason = StopReason::Interrupt;
                break;
            }
            StepOutcome::Emitted {
                token_id,
                answer_budget_exhausted,
            } => (token_id, answer_budget_exhausted),
        };

        // Answer-budget break: stop once max_new_tokens answer tokens follow
        // </think>. Computed inside `transition` and carried out via the
        // `Emitted` outcome above (`answer_budget_exhausted` is now private
        // to this module -- only `transition` may call it).
        if answer_budget_exhausted {
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
    if gen_cfg.disable_eos {
        // Benchmark determinism knob (`GenerateConfig::disable_eos`): force
        // continuation past EOS / configured stop-token ids so every trial
        // decodes the exact requested token count. Default `false` leaves
        // this function byte-for-byte unchanged (parity-safe).
        return false;
    }
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

/// Sibling guard to [`check_grammar_not_set`] / [`check_logprobs_not_set`]
/// (ADR-080 C3, #783): fails closed instead of silently dropping a
/// `stop_strings` request on a generation path that has not wired
/// string-level stop matching into its decode loop.
///
/// Callers: `generate_f16` (`forward/cpu_f16.rs`), `generate_q8`
/// (`forward/cpu_q8.rs`), `generate_q8_neon` (`forward/neon_forward.rs`),
/// `Qwen35Model::generate_with_batch_prefill` (`forward/batch_prefill.rs`).
///
/// The base CPU `generate()` / `generate_streaming()` paths in this module,
/// and the Metal `generate()` / `generate_streaming()` / `generate_multimodal`
/// family, all wire `stop_strings` matching directly and therefore do not
/// call this guard.
pub(crate) fn check_stop_strings_not_set(gen_cfg: &GenerateConfig) -> Result<(), InferenceError> {
    if !gen_cfg.stop_strings.is_empty() {
        return Err(InferenceError::InvalidInput(
            "stop_strings is not yet supported on this generation path; \
             use the Qwen3.5 CPU generate() / generate_streaming() or the Metal \
             generate() / generate_streaming(), which implement stop-string matching"
                .into(),
        ));
    }
    Ok(())
}

/// Sibling guard to [`check_stop_strings_not_set`] (ADR-080 C3, #783): fails
/// closed instead of silently dropping a `reasoning_budget` request on a
/// generation path that has not wired budget-forcing (`decode_cap` /
/// `force_close_think`) into its decode loop.
///
/// Same CPU caller list as [`check_stop_strings_not_set`]. On the Metal side,
/// the plain `generate()` entry point and `multimodal_generate_preflight`
/// (`generate_multimodal`) both call this guard directly; the MTP and
/// self-speculative greedy fast paths never see a set `reasoning_budget` in
/// the first place — their route predicates (`mtp_route_active` /
/// `self_spec_route_active`) exclude it, falling through to plain
/// `generate()`, which rejects it via this same guard.
///
/// The base CPU `generate()` / `generate_streaming()` paths in this module,
/// and the Metal `generate_streaming()` family, wire reasoning-budget forcing
/// directly and therefore do not call this guard.
pub(crate) fn check_reasoning_budget_not_set(
    gen_cfg: &GenerateConfig,
) -> Result<(), InferenceError> {
    if gen_cfg.reasoning_budget.is_some() {
        return Err(InferenceError::InvalidInput(
            "reasoning_budget is not yet supported on this generation path; \
             use the Qwen3.5 CPU generate() / generate_streaming() or the Metal \
             generate_streaming(), which implement reasoning-budget forcing"
                .into(),
        ));
    }
    Ok(())
}

/// Sibling guard to [`check_reasoning_budget_not_set`] (codex round-2 medium
/// #4, PR #787): fails closed instead of silently ignoring an active MTP
/// request on a generation path that never reads `gen_cfg.enable_mtp`.
///
/// Resolves `enable_mtp` exactly like the Metal `generate()` entry point
/// (`gen_cfg.enable_mtp.unwrap_or_else(|| LATTICE_MTP env set)`), so a caller
/// or environment combination that would activate MTP on the direct path is
/// rejected here too, rather than silently falling back to plain per-token
/// decode with no indication MTP was skipped.
///
/// Sole caller: the Metal cross-turn prefix-cache path
/// (`generate_streaming_with_prefix_cache_and_cancel`), which has no MTP
/// draft/verify wiring at all -- gated identically to that Metal-only
/// consumer (same gate as the `DecodePolicy`/`StepOutcome` re-export in
/// `mod.rs`) so non-metal-gpu builds don't carry an unused function.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) fn check_mtp_not_requested(gen_cfg: &GenerateConfig) -> Result<(), InferenceError> {
    let mtp_enabled = gen_cfg
        .enable_mtp
        .unwrap_or_else(|| std::env::var("LATTICE_MTP").is_ok());
    if mtp_enabled {
        return Err(InferenceError::InvalidInput(
            "enable_mtp (or LATTICE_MTP) is not supported on the cross-turn \
             prefix-cache generation path, which has no MTP draft/verify \
             wiring; use the Metal generate() / generate_streaming() paths, \
             which implement MTP"
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
    // check_stop_strings_not_set / check_reasoning_budget_not_set
    // (ADR-080 C3, #783) — mutation-sensitive unit tests for the shared guard
    // primitives every alternate CPU/Metal decode loop calls.
    // -----------------------------------------------------------------------

    /// Mutation sensitivity: change `check_stop_strings_not_set` to always
    /// return `Ok(())` → this assertion fails, catching a regression that
    /// would let a non-empty `stop_strings` silently pass through an
    /// unwired decode loop.
    #[test]
    fn check_stop_strings_not_set_rejects_nonempty() {
        let cfg = GenerateConfig {
            stop_strings: vec!["</s>".to_string()],
            ..Default::default()
        };
        let result = check_stop_strings_not_set(&cfg);
        assert!(
            matches!(result, Err(InferenceError::InvalidInput(_))),
            "non-empty stop_strings must be rejected with InvalidInput; got {result:?}"
        );
    }

    /// Mutation sensitivity: change the guard to always return `Err(..)` →
    /// this assertion fails, catching a regression that would reject every
    /// caller including the default (no stop strings requested) config.
    #[test]
    fn check_stop_strings_not_set_allows_empty() {
        assert!(
            check_stop_strings_not_set(&GenerateConfig::default()).is_ok(),
            "empty stop_strings (the default) must be allowed"
        );
    }

    /// Mutation sensitivity: change `check_reasoning_budget_not_set` to
    /// always return `Ok(())` → this assertion fails, catching a regression
    /// that would let a set `reasoning_budget` silently pass through an
    /// unwired decode loop.
    #[test]
    fn check_reasoning_budget_not_set_rejects_some() {
        let cfg = GenerateConfig {
            reasoning_budget: Some(128),
            ..Default::default()
        };
        let result = check_reasoning_budget_not_set(&cfg);
        assert!(
            matches!(result, Err(InferenceError::InvalidInput(_))),
            "Some(reasoning_budget) must be rejected with InvalidInput; got {result:?}"
        );
    }

    /// Mutation sensitivity: change the guard to always return `Err(..)` →
    /// this assertion fails, catching a regression that would reject every
    /// caller including the default (no reasoning budget requested) config.
    #[test]
    fn check_reasoning_budget_not_set_allows_none() {
        assert!(
            check_reasoning_budget_not_set(&GenerateConfig::default()).is_ok(),
            "reasoning_budget: None (the default) must be allowed"
        );
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

    /// `disable_eos: true` (the flagship CPU/Metal benchmark determinism
    /// knob -- `qwen35_generate --emit-phase-events`) must force
    /// continuation past a token that would otherwise stop generation on
    /// the very first step, so a benchmark trial always decodes the exact
    /// requested `max_new_tokens` count.
    ///
    /// Same zero-weight-model / `stop_token_ids = [0]` setup as
    /// `stop_reason_eos_on_first_stop_token` above (which proves the
    /// baseline DOES stop early without this flag) -- the only difference
    /// is `disable_eos: true`.
    ///
    /// Mutation sensitivity: removing the `if gen_cfg.disable_eos { return
    /// false; }` short-circuit at the top of `should_stop_token` makes this
    /// generation stop after 1 token instead of running to
    /// `max_new_tokens = 5`, failing both assertions below.
    #[test]
    fn disable_eos_forces_continuation_past_matching_stop_token() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 5,
            temperature: 0.0,
            stop_token_ids: vec![0], // would otherwise fire on the first greedy-sampled token
            disable_eos: true,
            ..Default::default()
        };
        let result = model
            .generate("a", &gen_cfg)
            .expect("disable_eos generate must succeed");
        assert_eq!(
            result.stop_reason,
            Some(StopReason::Length),
            "disable_eos must force the loop to run to max_new_tokens (StopReason::Length), \
             not stop early on a matching stop_token_ids entry; got {:?}",
            result.stop_reason
        );
        assert_eq!(
            result.generated_tokens, 5,
            "disable_eos must decode exactly max_new_tokens (5), got {}",
            result.generated_tokens
        );
    }

    /// Sibling of the test above, covering `should_stop_token`'s OTHER
    /// branch (`cfg.eos_token_id`, not `stop_token_ids`): `disable_eos`
    /// must also suppress a match against the model's own configured EOS
    /// id. `build_tiny_zero_model` sets `eos_token_id = VOCAB - 1 = 96`;
    /// the tiny tokenizer's `"a"` prompt plus all-zero logits normally
    /// greedy-samples token 0, not 96, so this test instead drives
    /// `should_stop_token` directly (the pure predicate `generate()`'s
    /// decode loop calls every step) rather than threading a real decode
    /// run through to a token-96 sample.
    #[test]
    fn disable_eos_suppresses_eos_token_id_match_in_should_stop_token() {
        let model = build_tiny_zero_model();
        let base_cfg = GenerateConfig {
            stop_token_ids: vec![],
            ..Default::default()
        };
        assert!(
            should_stop_token(&model.config, &base_cfg, model.config.eos_token_id),
            "sanity: without disable_eos, the model's own eos_token_id must stop generation"
        );
        let disabled_cfg = GenerateConfig {
            stop_token_ids: vec![],
            disable_eos: true,
            ..Default::default()
        };
        assert!(
            !should_stop_token(&model.config, &disabled_cfg, model.config.eos_token_id),
            "disable_eos: true must suppress a match against cfg.eos_token_id too"
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

    /// `should_cancel` returning `true` on its SECOND call only -- i.e. the
    /// pre-prefill checkpoint (call 1) sees `false` and lets prefill run,
    /// then the post-prefill checkpoint (call 2, immediately after prefill,
    /// before sampling) sees `true` and must stop before ANY token is
    /// sampled or emitted. This isolates the post-prefill checkpoint
    /// specifically (ADR-080 C2 round 2, codex round-1 medium finding #3):
    /// the pre-prefill test above (`n == 1`) cannot tell the two checkpoints
    /// apart, since removing the post-prefill one entirely still leaves that
    /// test green (its cancellation already fires at checkpoint 1).
    ///
    /// Mutation sensitivity: removing the post-prefill `if should_cancel()`
    /// guard (the one right after the prefill-logits copy, before grammar
    /// masking/sampling) lets generation fall through to sampling the first
    /// token, producing `generated_tokens > 0`, non-empty `text`, and at
    /// least one `on_token` callback -- failing all three assertions below.
    /// Verified by reverting that exact guard and re-running: see the PR
    /// body's mutation log.
    #[test]
    fn generate_streaming_with_cancel_true_after_prefill_returns_interrupt() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 5,
            temperature: 0.0,
            ..Default::default()
        };
        let calls = std::cell::Cell::new(0usize);
        let on_token_calls = std::cell::Cell::new(0usize);
        let result = model
            .generate_streaming_with_cancel(
                "a",
                &gen_cfg,
                |_delta| {
                    on_token_calls.set(on_token_calls.get() + 1);
                    true
                },
                || {
                    let n = calls.get() + 1;
                    calls.set(n);
                    n == 2
                },
            )
            .expect("cancelled-after-prefill generate must succeed");
        assert!(
            !result.stopped,
            "a caller cancellation is not an OpenAI stop condition"
        );
        assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
        assert_eq!(
            result.generated_tokens, 0,
            "post-prefill cancellation must stop before any token is sampled"
        );
        assert!(result.text.is_empty());
        assert_eq!(
            on_token_calls.get(),
            0,
            "post-prefill cancellation must stop before on_token is ever called"
        );
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
    /// Mutation sensitivity: dropping the interrupted check after the
    /// pre-loop `check_initial_stop` call lets generation continue into the
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

    // -----------------------------------------------------------------------
    // generate_streaming_with_observer / RawGenEvent mutation-sensitive tests
    // (benchmark-overhaul row 2, codex round-1 blocker #1, PR #882)
    // -----------------------------------------------------------------------

    /// `RawGenEvent::PrefillEnd` must fire exactly once, before the first
    /// `RawGenEvent::RawToken`, and every subsequent `RawToken` index must be
    /// monotonically increasing 1..=generated_tokens -- the raw prefill/decode
    /// boundary the CPU flagship smoke harness measures TTFT and decode
    /// throughput off, independent of the text-delta callback.
    ///
    /// Mutation sensitivity: moving the `on_raw_event(RawGenEvent::PrefillEnd)`
    /// call to after the first token is pushed (the pre-fix bug this test
    /// guards -- marking prefill_end off the first confirmed generated token
    /// instead of the true prefill/decode boundary) makes `events.first()`
    /// a `RawToken` instead of `PrefillEnd`, failing the first assertion.
    /// Removing any of the three `on_raw_event(RawGenEvent::RawToken { .. })`
    /// call sites (pre-loop first token, fast-path decode loop, stop-string
    /// decode loop) drops an index from the collected sequence, failing the
    /// monotonic-range assertion. Verified by reverting the fix and
    /// re-running: see the PR body's mutation log.
    #[test]
    fn raw_observer_prefill_end_precedes_first_raw_token_monotonic_index() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 3,
            temperature: 0.0,
            ..Default::default()
        };
        let events = std::cell::RefCell::new(Vec::<RawGenEvent>::new());
        let result = model
            .generate_streaming_with_observer(
                "a",
                &gen_cfg,
                |_delta| true,
                || false,
                |evt| events.borrow_mut().push(evt),
            )
            .expect("generation must succeed");

        let events = events.into_inner();
        assert_eq!(
            events.first(),
            Some(&RawGenEvent::PrefillEnd),
            "the very first raw event must be PrefillEnd -- prefill completed and logits \
             are ready before any token is sampled; got {events:?}"
        );
        let token_indices: Vec<usize> = events
            .iter()
            .skip(1)
            .map(|e| match e {
                RawGenEvent::RawToken { index } => *index,
                RawGenEvent::PrefillEnd => {
                    panic!("PrefillEnd must fire exactly once, at the very start: {events:?}")
                }
            })
            .collect();
        assert_eq!(result.generated_tokens, 3);
        assert_eq!(
            token_indices,
            vec![1, 2, 3],
            "RawToken events must be one per generated token, in generation order, with a \
             monotonically increasing 1-based index equal to generated_tokens so far"
        );
    }

    /// `RawGenEvent::PrefillEnd` must fire before `sample_token` is entered
    /// for the first time, not merely before the `RawToken` *callback*
    /// (codex round-2 medium finding, PR #882): the test above cannot
    /// distinguish "PrefillEnd fired before sampling" from "PrefillEnd
    /// fired after sampling but before the RawToken push", because both
    /// orderings produce the same `[PrefillEnd, RawToken, RawToken,
    /// RawToken]` event sequence. This test uses the `test_record_first_sample_entry`
    /// seam, planted at the exact point `sample_token` is called for the
    /// prefill-derived first token, to check the ordering codex's mutation
    /// actually exercises.
    ///
    /// Mutation sensitivity: moving `on_raw_event(RawGenEvent::PrefillEnd)`
    /// to immediately after `generated_ids.push(next_id)` (still before the
    /// `RawToken` callback -- the exact mutation codex applied in its round-2
    /// review) leaves the event-order test above green, but makes this test's
    /// `on_raw_event` closure observe `PrefillEnd` only *after*
    /// `test_record_first_sample_entry` already ran, so
    /// `test_take_first_sample_saw_prefill_end()` returns `Some(false)`
    /// instead of `Some(true)` and the assertion below fails. Verified by
    /// reverting the fix (see PR body's mutation log for this test).
    #[test]
    fn raw_observer_prefill_end_precedes_first_sample_not_just_first_raw_token() {
        test_reset_sample_seam();
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 3,
            temperature: 0.0,
            ..Default::default()
        };
        let result = model
            .generate_streaming_with_observer(
                "a",
                &gen_cfg,
                |_delta| true,
                || false,
                |evt| {
                    if evt == RawGenEvent::PrefillEnd {
                        test_mark_prefill_end_seen();
                    }
                },
            )
            .expect("generation must succeed");

        assert_eq!(result.generated_tokens, 3);
        assert_eq!(
            test_take_first_sample_saw_prefill_end(),
            Some(true),
            "PrefillEnd must have already fired by the moment sample_token is entered for \
             the prefill-derived first token -- not merely before the RawToken callback, \
             which a PrefillEnd emitted after sampling (but before the RawToken push) would \
             also satisfy while silently including sampling time in the reported prefill \
             interval (codex round-2 medium, PR #882)"
        );
    }

    /// One raw-token event fires per generated token even when the
    /// text-delta stream buffers an incomplete multi-byte UTF-8 sequence and
    /// therefore does NOT call `on_token` for that step at all (codex
    /// round-1 blocker #1's core claim: measuring prefill/decode boundaries
    /// off text deltas is not equivalent to measuring off raw sampled
    /// tokens, and can *lag* the true boundary by one or more tokens).
    ///
    /// Token id 0 is defined (via a custom tiny tokenizer vocab) to decode to
    /// the single raw byte `0xC2` -- the lead byte of a 2-byte UTF-8 sequence
    /// (`U+0080..=U+07FF`), which alone is an incomplete codepoint that
    /// `IncrementalDetokenizer` must buffer rather than emit (verified
    /// directly: pushing `0xC2` once in isolation yields an empty delta).
    /// The all-zero-weight tiny model always greedily samples token 0 (tied
    /// logits, first-wins), so the very *first* generated token -- the
    /// prefill-derived one, pushed into a fresh, empty detokenizer buffer --
    /// is guaranteed to buffer rather than emit: the fast decode path's
    /// `StopMode::Disabled` branch of `stop_check` skips calling `on_token`
    /// entirely whenever the resulting delta is empty (`generation.rs`:
    /// `if delta.is_empty() { return StopCheckOutcome::Continue; }`).
    ///
    /// This test snapshots `on_token`'s cumulative call count at the instant
    /// each `RawToken` event fires, rather than comparing final totals: a
    /// trailing `detok.finish()` flush (emitted once after the decode loop
    /// ends, for whatever incomplete bytes never got a chance to complete)
    /// also calls `on_token`, so the *final* on_token-call total is not a
    /// reliable signal here -- it can coincidentally equal
    /// `generated_tokens` even though a mid-generation step was buffered.
    /// The per-event snapshot sidesteps that confound entirely: at the
    /// moment `RawToken { index: 1 }` fires (in the observer, mid
    /// generation, before the decode loop or any flush has run), `on_token`
    /// must not yet have been called even once.
    ///
    /// Mutation sensitivity: reverting to the pre-fix design (emitting
    /// `token_available` from inside `on_token` instead of from
    /// `on_raw_event`) would silently miss this buffered first token
    /// entirely (its phase event would fire late, on whatever later step
    /// finally produces a non-empty delta, or not at all if generation ends
    /// first) -- this test's first-event snapshot fails immediately on that
    /// regression, and the raw-index-sequence assertion fails independently
    /// if any `on_raw_event(RawGenEvent::RawToken { .. })` call site is
    /// removed.
    #[test]
    fn raw_observer_fires_before_on_token_for_buffered_incomplete_utf8_first_token() {
        const UTF8_LEAD_BYTE_TOK_JSON: &str = r#"{
  "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"Â":0,"<unk>":1,"a":2},"merges":[]}
}"#;
        let model = build_tiny_zero_model_tok(UTF8_LEAD_BYTE_TOK_JSON);

        // Independent confirmation that token id 0 alone is genuinely
        // incomplete UTF-8, not just asserted by comment: pushing it once
        // into a fresh detokenizer must yield an empty delta.
        let mut probe = IncrementalDetokenizer::new();
        assert_eq!(
            probe.push(model.tokenizer(), 0),
            "",
            "token id 0 (raw byte 0xC2) must be an incomplete UTF-8 lead byte on its own -- \
             this test's premise depends on it"
        );

        let gen_cfg = GenerateConfig {
            max_new_tokens: 4,
            temperature: 0.0,
            ..Default::default()
        };
        let raw_indices = std::cell::RefCell::new(Vec::<usize>::new());
        let on_token_calls = std::cell::Cell::new(0usize);
        // Snapshot of `on_token_calls` at the instant each RawToken event
        // fires -- index i (0-based) corresponds to RawToken{index: i+1}.
        let snapshots_at_raw_event = std::cell::RefCell::new(Vec::<usize>::new());
        let result = model
            .generate_streaming_with_observer(
                "a",
                &gen_cfg,
                |_delta: &str| {
                    on_token_calls.set(on_token_calls.get() + 1);
                    true
                },
                || false,
                |evt| {
                    if let RawGenEvent::RawToken { index } = evt {
                        raw_indices.borrow_mut().push(index);
                        snapshots_at_raw_event
                            .borrow_mut()
                            .push(on_token_calls.get());
                    }
                },
            )
            .expect("generation over an incomplete-lead-byte vocab must still succeed");

        assert_eq!(
            result.generated_tokens, 4,
            "token id 0 (eos_token_id is 96 on this tiny model, never sampled) never \
             satisfies should_stop_token, so all 4 requested tokens must be generated"
        );
        assert_eq!(
            raw_indices.into_inner(),
            vec![1, 2, 3, 4],
            "one monotonically indexed RawToken event per generated token, regardless of \
             detokenizer buffering"
        );
        assert_eq!(
            snapshots_at_raw_event.borrow()[0],
            0,
            "on_token must not have been called yet at the moment the FIRST RawToken event \
             fires -- the prefill-derived first token's delta is buffered (empty) by the \
             incomplete-UTF-8 lead byte, so a phase-event trace measured off on_token would \
             have missed or mis-timed this token entirely; got {} prior on_token calls",
            snapshots_at_raw_event.borrow()[0]
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

    /// Codex round-3 medium #2 (PR #787 / Leo's ruling): with `gen_cfg.logprobs`
    /// left at its default (`None`), `DecodePolicy::record_logprob` (driven by
    /// both `init` for the prefill token and `transition` for every token
    /// after) must be a true no-op -- `token_logprobs` stays empty for the
    /// whole generation, not just for the truncated-text case the test above
    /// covers. Replaces `sampling.rs`'s now-removed
    /// `test_record_logprob_noop_when_not_requested`: that free function no
    /// longer exists (its mutation into `crate::sampling` was the whole
    /// point of the fix), so its behavioral contract is asserted here, at
    /// the only place that can still exercise it.
    ///
    /// Mutation sensitivity: removing the `let Some(top_n) = self.logprobs
    /// else { return; };` guard inside `DecodePolicy::record_logprob` makes
    /// every token here get an (incorrect) `TokenLogprob` entry, failing
    /// `token_logprobs.is_empty()`.
    #[test]
    fn decode_policy_record_logprob_noop_when_not_requested() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 3,
            temperature: 0.0,
            logprobs: None,
            ..Default::default()
        };
        let result = model
            .generate("a", &gen_cfg)
            .expect("plain generate must succeed");
        assert!(
            result.token_logprobs.is_empty(),
            "logprobs: None must record nothing across the whole generation \
             (prefill token via init, decode tokens via transition); got {} entries",
            result.token_logprobs.len()
        );
    }

    /// Codex round-2 major #2 (PR #787 / Leo's ruling): isolates
    /// `DecodePolicy::init`'s first-step logprob ownership from
    /// `transition`'s per-step logprob ownership, which
    /// `transition_records_one_logprob_per_generated_token` below already
    /// covers but does not itself distinguish. With `max_new_tokens: 1`,
    /// `decode_loop`'s cap is 1, so its `for _ in 1..cap` loop body never
    /// executes and `DecodePolicy::transition` is never called at all -- the
    /// entire generation consists of the one prefill-derived token `init`
    /// records. If that one `TokenLogprob` entry exists, it can only have
    /// come from `init`.
    ///
    /// Mutation sensitivity: removing the
    /// `policy.record_logprob(token_logprobs, first_logits, ...)` call
    /// inside `DecodePolicy::init` makes `token_logprobs` come back empty
    /// while `token_ids` still has 1 entry -- this test fails with a length
    /// mismatch (`0 != 1`) instead of passing.
    #[test]
    fn init_records_the_prefill_tokens_logprob_before_any_transition_call() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            temperature: 0.0,
            logprobs: Some(0),
            ..Default::default()
        };
        let result = model
            .generate("a", &gen_cfg)
            .expect("single-token generate must succeed");

        assert_eq!(
            result.generated_tokens, 1,
            "max_new_tokens: 1 must generate exactly the prefill-derived \
             first token and never enter decode_loop; got {}",
            result.generated_tokens
        );
        assert_eq!(
            result.token_logprobs.len(),
            1,
            "the sole generated token's logprob must be recorded by \
             DecodePolicy::init alone (transition is never called when \
             max_new_tokens == 1); got {} entries",
            result.token_logprobs.len()
        );
        assert_eq!(
            result.token_logprobs[0].token_id, result.token_ids[0],
            "the recorded logprob entry must describe the actual first token"
        );
    }

    /// Codex round-1 major #3 (PR #787): `DecodePolicy::transition` owns
    /// `record_logprob` internally now (the constituent method is private,
    /// only reachable through `transition`). This asserts the positive case
    /// the truncation test above does not: with no stop string to truncate
    /// anything, every generated token gets exactly one `TokenLogprob` entry,
    /// and its `token_id` matches the corresponding `token_ids` entry.
    ///
    /// Mutation sensitivity: commenting out the `self.record_logprob(...)`
    /// call inside `DecodePolicy::transition` makes `token_logprobs` come
    /// back empty while `token_ids` still has 3 entries -- this test fails
    /// with a length mismatch (`0 != 3`) instead of passing.
    #[test]
    fn transition_records_one_logprob_per_generated_token() {
        let model = build_tiny_zero_model();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 3,
            temperature: 0.0,
            logprobs: Some(0),
            ..Default::default()
        };
        let result = model
            .generate("a", &gen_cfg)
            .expect("plain generate must succeed");

        assert_eq!(
            result.token_logprobs.len(),
            result.token_ids.len(),
            "every generated token must get exactly one TokenLogprob entry \
             when logprobs is requested and nothing truncates the output; \
             got {} logprobs for {} tokens",
            result.token_logprobs.len(),
            result.token_ids.len()
        );
        for (i, (logprob, &token_id)) in result
            .token_logprobs
            .iter()
            .zip(result.token_ids.iter())
            .enumerate()
        {
            assert_eq!(
                logprob.token_id, token_id,
                "token_logprobs[{i}] must describe the token actually emitted \
                 at that position"
            );
        }
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
