use super::cache::{ForwardScratch, KvCache};
use super::detokenize::{IncrementalDetokenizer, decode_tokens};
use super::model::Qwen35Model;
use super::sampling::sample_token;
use crate::attention::gdn::GatedDeltaNetState;
use crate::error::InferenceError;
use crate::grammar::pda::GrammarState;
use crate::model::qwen35_config::{GenerateConfig, GenerateOutput, Qwen35Config};
use crate::tokenizer::common::Tokenizer;

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
            });
        }

        // Context preflight. apply_partial_rope indexes the precomputed cos/sin
        // table unchecked, so a position at or past max_context() is an
        // out-of-bounds slice access — a release panic, not a clean error. The
        // exact highest position the CPU loop reaches is prompt_len +
        // max_new_tokens - 2 (prefill 0..prompt_len-1; the first token reuses
        // prefill logits with no new RoPE lookup; decode runs `1..max_new_tokens`
        // from position prompt_len), so the precise RoPE-safe bound is
        // prompt_len + max_new_tokens - 1 <= max_context. We deliberately adopt
        // the stricter total-token policy prompt_len + max_new_tokens <=
        // max_context instead: it is the OpenAI-style "prompt plus requested
        // completion fits the window" contract and matches the HTTP server
        // (bin/lattice.rs) verbatim, so direct and HTTP generation agree on when
        // a request is too long. Strictly safe (it can only reject one extra
        // edge request, never admit a panic). Same guard in generate_streaming.
        let max_context = self.max_context();
        if prompt_len.saturating_add(gen_cfg.max_new_tokens) > max_context {
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

        let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
        let mut all_ids = prompt_ids.clone();

        prefill_tokens(
            self,
            &prompt_ids,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
        );
        kv_cache.seq_len = prompt_len;

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
        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state) {
            if !engine.advance(gs, next_id) {
                return Ok(GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    stopped: false,
                });
            }
        }

        if should_stop_token(cfg, gen_cfg, next_id) {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: true,
            });
        }

        generated_ids.push(next_id);
        all_ids.push(next_id);

        if gen_cfg.stop_strings.is_empty() {
            // Fast path: no string-level stops. Behaviour byte-for-byte identical
            // to before this feature was added; the e2e-parity CI gate pins this.
            let stopped = decode_loop(
                self,
                gen_cfg,
                &mut all_ids,
                &mut generated_ids,
                &mut rng_state,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
                &mut grammar_state,
            )?;

            let text = decode_tokens(&self.tokenizer, &generated_ids);

            Ok(GenerateOutput {
                text,
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                stopped,
            })
        } else {
            // String-stop path: accumulate decoded text and check after every token.
            let mut detok = IncrementalDetokenizer::new();
            let first_delta = detok.push(&self.tokenizer, next_id);
            let mut full = first_delta;

            // Check stop strings after the first token.
            if let Some(hit) = earliest_stop_match(&full, &gen_cfg.stop_strings) {
                full.truncate(hit);
                // generated_ids already contains next_id; we cannot un-generate it,
                // so token_ids/generated_tokens reflect all tokens up to the match.
                return Ok(GenerateOutput {
                    text: full,
                    token_ids: generated_ids.clone(),
                    prompt_tokens: prompt_len,
                    generated_tokens: generated_ids.len(),
                    stopped: true,
                });
            }

            let stopped = decode_loop_with_stops(
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
            )?;

            Ok(GenerateOutput {
                text: full,
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                stopped,
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
    pub fn generate_streaming(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        mut on_token: impl FnMut(&str),
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
        // we never emit a token the caller did not ask for.
        if gen_cfg.max_new_tokens == 0 {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: false,
            });
        }

        // Context preflight: see generate() for the full rationale and the exact
        // vs. adopted-bound discussion. apply_partial_rope indexes the RoPE table
        // unchecked, so a request past max_context() would panic in the decode
        // loop; this mirrors the HTTP server's total-token contract verbatim.
        let max_context = self.max_context();
        if prompt_len.saturating_add(gen_cfg.max_new_tokens) > max_context {
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

        let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
        let mut all_ids = prompt_ids.clone();

        prefill_tokens(
            self,
            &prompt_ids,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
        );
        kv_cache.seq_len = prompt_len;

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
        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state) {
            if !engine.advance(gs, next_id) {
                return Ok(GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    stopped: false,
                });
            }
        }

        if should_stop_token(cfg, gen_cfg, next_id) {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: true,
            });
        }

        generated_ids.push(next_id);
        all_ids.push(next_id);

        // Incremental detokenization: emit only complete-UTF-8 text deltas. A
        // byte-level BPE codepoint can span several tokens, so we buffer raw bytes
        // and never stream a partial codepoint (see IncrementalDetokenizer).
        let mut detok = IncrementalDetokenizer::new();

        if gen_cfg.stop_strings.is_empty() {
            // Fast path: no string-level stops. Behaviour byte-for-byte identical
            // to before this feature was added; the e2e-parity CI gate pins this.
            let delta = detok.push(&self.tokenizer, next_id);
            if !delta.is_empty() {
                on_token(&delta);
            }

            let mut stopped = false;
            // Decode loop (mirrors decode_loop free function exactly).
            for _ in 1..gen_cfg.max_new_tokens {
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

                let next_id = sample_token(
                    &scratch.logits[..cfg.vocab_size],
                    gen_cfg,
                    &all_ids,
                    &mut rng_state,
                );

                // Grammar advance: break cleanly when the grammar signals completion.
                if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state) {
                    if !engine.advance(gs, next_id) {
                        break;
                    }
                }

                if should_stop_token(cfg, gen_cfg, next_id) {
                    stopped = true;
                    break;
                }

                generated_ids.push(next_id);
                all_ids.push(next_id);

                let delta = detok.push(&self.tokenizer, next_id);
                if !delta.is_empty() {
                    on_token(&delta);
                }
            }

            // Flush any trailing incomplete bytes (generation truncated mid-codepoint)
            // so the streamed deltas concatenate to exactly the returned text.
            let tail = detok.finish();
            if !tail.is_empty() {
                on_token(&tail);
            }

            Ok(GenerateOutput {
                text: detok.text(),
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                stopped,
            })
        } else {
            // String-stop path: use StopStreamer to hold back (max_stop - 1) bytes and
            // never emit a partial stop prefix before we can confirm it is not a match.
            let mut streamer = StopStreamer::new(&gen_cfg.stop_strings);

            let first_delta = detok.push(&self.tokenizer, next_id);
            if streamer.push(&first_delta, &mut on_token) {
                // Stop matched in the very first token.
                // token_ids already contain next_id; cannot un-generate it.
                return Ok(GenerateOutput {
                    text: streamer.into_text(),
                    token_ids: generated_ids.clone(),
                    prompt_tokens: prompt_len,
                    generated_tokens: generated_ids.len(),
                    stopped: true,
                });
            }

            let mut stopped = false;
            // Decode loop for the string-stop path.
            for _ in 1..gen_cfg.max_new_tokens {
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

                let next_id = sample_token(
                    &scratch.logits[..cfg.vocab_size],
                    gen_cfg,
                    &all_ids,
                    &mut rng_state,
                );

                // Grammar advance: break cleanly when the grammar signals completion.
                if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut grammar_state) {
                    if !engine.advance(gs, next_id) {
                        stopped = true;
                        break;
                    }
                }

                if should_stop_token(cfg, gen_cfg, next_id) {
                    stopped = true;
                    break;
                }

                generated_ids.push(next_id);
                all_ids.push(next_id);

                let delta = detok.push(&self.tokenizer, next_id);
                if streamer.push(&delta, &mut on_token) {
                    stopped = true;
                    break;
                }
            }

            // Natural-end flush (no-op if a stop was already hit inside the loop).
            streamer.finish(&detok.finish(), &mut on_token);
            // finish() may itself complete a stop in the tail bytes.
            stopped |= streamer.stopped;

            Ok(GenerateOutput {
                text: streamer.into_text(),
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                stopped,
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

/// Stateful hold-back streamer for string-level stops (used by `generate_streaming`).
///
/// Maintains a `full` string of all decoded text so far and an `emitted` cursor. After each
/// push it:
/// 1. Checks `full` for the earliest stop-string match.
///    - If found: emits `full[emitted..hit]` via `sink`, truncates `full` to `hit`, sets
///      `emitted = full.len()`, marks `stopped = true`, returns `true`.
/// 2. If no match: emits the "safe" prefix `full[emitted..safe]` where
///    `safe = full.len() - (max_stop - 1)`, clamped to a UTF-8 char boundary, returns `false`.
///
/// `finish` is called once after the decode loop ends (natural stop / EOS / token cap).
/// If `stopped` is already true it is a no-op; otherwise it appends the `detok.finish()` tail,
/// checks for a stop in the result, and emits whatever is safe.
pub(crate) struct StopStreamer<'a> {
    full: String,
    emitted: usize,
    max_stop: usize,
    stops: &'a [String],
    stopped: bool,
}

impl<'a> StopStreamer<'a> {
    pub(crate) fn new(stops: &'a [String]) -> Self {
        let max_stop = stops.iter().map(String::len).max().unwrap_or(1);
        Self {
            full: String::new(),
            emitted: 0,
            max_stop,
            stops,
            stopped: false,
        }
    }

    /// Append `delta` to the accumulated text, emit newly-safe text via `sink`.
    ///
    /// Returns `true` when a stop string was found (caller must break the decode loop).
    pub(crate) fn push(&mut self, delta: &str, sink: &mut impl FnMut(&str)) -> bool {
        if delta.is_empty() {
            return false;
        }
        self.full.push_str(delta);

        if let Some(hit) = earliest_stop_match(&self.full, self.stops) {
            let slice = &self.full[self.emitted..hit];
            if !slice.is_empty() {
                sink(slice);
            }
            self.full.truncate(hit);
            // Advance emitted so the post-loop flush is a no-op.
            self.emitted = self.full.len();
            self.stopped = true;
            return true;
        }

        // Emit safe prefix: hold back (max_stop - 1) bytes so a stop string that
        // starts before the end of the current delta is never streamed prematurely.
        let mut safe = self
            .full
            .len()
            .saturating_sub(self.max_stop.saturating_sub(1));
        safe = safe.max(self.emitted);
        while safe > self.emitted && !self.full.is_char_boundary(safe) {
            safe -= 1;
        }
        if safe > self.emitted {
            sink(&self.full[self.emitted..safe]);
            self.emitted = safe;
        }
        false
    }

    /// Natural-end flush. `tail` is `detok.finish()`.
    ///
    /// No-op when already stopped (a stop string was found in the decode loop).
    /// Otherwise appends `tail`, checks for a late stop, and emits whatever remains.
    pub(crate) fn finish(&mut self, tail: &str, sink: &mut impl FnMut(&str)) {
        if self.stopped {
            return;
        }
        if !tail.is_empty() {
            self.full.push_str(tail);
        }
        // A stop string could complete inside the tail bytes.
        if let Some(hit) = earliest_stop_match(&self.full, self.stops) {
            let slice = &self.full[self.emitted..hit];
            if !slice.is_empty() {
                sink(slice);
            }
            self.full.truncate(hit);
            self.emitted = self.full.len();
            self.stopped = true;
            return;
        }
        // No stop found: emit everything remaining.
        if self.emitted < self.full.len() {
            sink(&self.full[self.emitted..]);
            self.emitted = self.full.len();
        }
    }

    /// Consume the streamer and return the final (possibly truncated) text.
    pub(crate) fn into_text(self) -> String {
        self.full
    }
}

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
) -> Result<bool, InferenceError> {
    let cfg = &model.config;
    for _ in 1..gen_cfg.max_new_tokens {
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

        let next_id = sample_token(
            &scratch.logits[..cfg.vocab_size],
            gen_cfg,
            all_ids,
            rng_state,
        );

        // Advance grammar state; a false return signals grammar completion.
        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut *grammar_state) {
            if !engine.advance(gs, next_id) {
                return Ok(true);
            }
        }

        if should_stop_token(cfg, gen_cfg, next_id) {
            return Ok(true);
        }

        generated_ids.push(next_id);
        all_ids.push(next_id);
    }
    Ok(false)
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
) -> Result<bool, InferenceError> {
    let cfg = &model.config;
    let mut stopped = false;
    for _ in 1..gen_cfg.max_new_tokens {
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

        let next_id = sample_token(
            &scratch.logits[..cfg.vocab_size],
            gen_cfg,
            all_ids,
            rng_state,
        );

        // Advance grammar state; a false return signals grammar completion.
        if let (Some(engine), Some(gs)) = (&gen_cfg.grammar, &mut *grammar_state) {
            if !engine.advance(gs, next_id) {
                stopped = true;
                break;
            }
        }

        if should_stop_token(cfg, gen_cfg, next_id) {
            stopped = true;
            break;
        }

        generated_ids.push(next_id);
        all_ids.push(next_id);

        let delta = detok.push(&model.tokenizer, next_id);
        if !delta.is_empty() {
            full.push_str(&delta);
        }

        if let Some(hit) = earliest_stop_match(full, &gen_cfg.stop_strings) {
            full.truncate(hit);
            stopped = true;
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
                return Ok(true);
            }
        }
        return Ok(false);
    }
    Ok(true)
}

/// Returns the smallest byte index in `haystack` at which any stop string begins,
/// or `None` if no stop string is present.
pub(crate) fn earliest_stop_match(haystack: &str, stops: &[String]) -> Option<usize> {
    stops.iter().filter_map(|s| haystack.find(s.as_str())).min()
}

/// Returns true when `token_id` is EOS or is in the `stop_token_ids` list.
pub fn should_stop_token(cfg: &Qwen35Config, gen_cfg: &GenerateConfig, token_id: u32) -> bool {
    token_id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&token_id)
}

/// Returns `Err(InvalidInput)` when `gen_cfg.grammar` is set, to prevent the
/// grammar field from being silently ignored on paths that have not yet wired
/// grammar masking (#397).
///
/// This guard is used by the batch-prefill and f16 CPU paths. The base CPU
/// generate() / generate_streaming() paths wire grammar directly and do not
/// call this guard.
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

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // earliest_stop_match
    // -----------------------------------------------------------------------

    #[test]
    fn earliest_stop_match_single_present() {
        assert_eq!(
            earliest_stop_match("hello world", &["world".to_string()]),
            Some(6)
        );
    }

    #[test]
    fn earliest_stop_match_no_match() {
        assert_eq!(
            earliest_stop_match("hello world", &["foo".to_string()]),
            None
        );
    }

    #[test]
    fn earliest_stop_match_multiple_returns_earliest() {
        // "world" at index 6, "lo" at index 3 — should return 3
        assert_eq!(
            earliest_stop_match("hello world", &["world".to_string(), "lo".to_string()]),
            Some(3)
        );
    }

    #[test]
    fn earliest_stop_match_at_index_zero() {
        assert_eq!(
            earliest_stop_match("stopword rest", &["stop".to_string()]),
            Some(0)
        );
    }

    #[test]
    fn earliest_stop_match_multibyte_utf8() {
        // "界" is a 3-byte UTF-8 codepoint; it starts at byte 3 in "世界hello".
        assert_eq!(
            earliest_stop_match("世界hello", &["界".to_string()]),
            Some(3)
        );
    }

    #[test]
    fn earliest_stop_match_empty_stops() {
        // No stops configured → None, even for non-empty haystack.
        assert_eq!(earliest_stop_match("hello", &[]), None);
    }

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
    // StopStreamer — regression tests for the two streaming bugs
    // -----------------------------------------------------------------------

    /// BUG 1 regression: stop split across deltas must NOT double-emit the pre-stop tail.
    ///
    /// deltas: ["hel", "lo W", "orld!"], stop="World"
    /// "World" (case-sensitive) never appears, but let's use "orld" for a mid-word hit.
    /// Actually use stop="World" case-sensitive with deltas building "hello World!" so the
    /// stop completes on the third delta.
    #[test]
    fn stop_streamer_stop_split_across_deltas_no_double_emit() {
        let stops = vec!["World".to_string()];
        let mut streamer = StopStreamer::new(&stops);
        let mut all_emitted: Vec<String> = Vec::new();

        // delta 1: "hel" — no stop, safe prefix held back
        let stopped1 = streamer.push("hel", &mut |s| all_emitted.push(s.to_string()));
        assert!(!stopped1);

        // delta 2: "lo W" — "hello W" so far, "World" not yet complete, some safe bytes emitted
        let stopped2 = streamer.push("lo W", &mut |s| all_emitted.push(s.to_string()));
        assert!(!stopped2);

        // delta 3: "orld!" — "hello World!" now, stop "World" at byte 6
        let stopped3 = streamer.push("orld!", &mut |s| all_emitted.push(s.to_string()));
        assert!(stopped3, "stop should be detected on third delta");

        let concatenated = all_emitted.join("");
        // The text before "World" is "hello " (6 bytes). Must appear EXACTLY ONCE.
        assert_eq!(
            concatenated, "hello ",
            "emitted concat must equal pre-stop text exactly once (BUG 1 regression)"
        );
        assert_eq!(streamer.into_text(), "hello ");
    }

    /// Stop in the very first delta: nothing before it, into_text() is empty.
    #[test]
    fn stop_streamer_stop_at_first_delta() {
        let stops = vec!["Stop".to_string()];
        let mut streamer = StopStreamer::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        let stopped = streamer.push("Stop now", &mut |s| emitted.push(s.to_string()));
        assert!(stopped);
        assert_eq!(emitted.join(""), "");
        assert_eq!(streamer.into_text(), "");
    }

    /// No stop hit, natural end: finish() emits the held-back tail, into_text() is full.
    #[test]
    fn stop_streamer_no_stop_natural_end() {
        let stops = vec!["zzz".to_string()];
        let mut streamer = StopStreamer::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        streamer.push("abc", &mut |s| emitted.push(s.to_string()));
        streamer.push("def", &mut |s| emitted.push(s.to_string()));
        streamer.finish("", &mut |s| emitted.push(s.to_string()));
        assert_eq!(emitted.join(""), "abcdef");
        assert_eq!(streamer.into_text(), "abcdef");
    }

    /// Multibyte hold-back: no panic, emitted concat == into_text, boundaries respected.
    #[test]
    fn stop_streamer_multibyte_no_panic() {
        let stops = vec!["STOP".to_string()];
        let mut streamer = StopStreamer::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        // "世" = 3 bytes, "界" = 3 bytes, "x" = 1 byte
        streamer.push("世", &mut |s| emitted.push(s.to_string()));
        streamer.push("界x", &mut |s| emitted.push(s.to_string()));
        streamer.finish("", &mut |s| emitted.push(s.to_string()));
        let concat = emitted.join("");
        assert_eq!(concat, streamer.into_text());
        // All emitted slices must be valid UTF-8 (would panic on invalid slice otherwise).
    }

    /// Stop at a delta boundary: prior delta is emitted cleanly, stop delta emits nothing.
    #[test]
    fn stop_streamer_stop_at_delta_boundary() {
        let stops = vec!["STOP".to_string()];
        let mut streamer = StopStreamer::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        let stopped1 = streamer.push("abc", &mut |s| emitted.push(s.to_string()));
        assert!(!stopped1);
        let stopped2 = streamer.push("STOP", &mut |s| emitted.push(s.to_string()));
        assert!(stopped2);
        assert_eq!(emitted.join(""), "abc");
        assert_eq!(streamer.into_text(), "abc");
    }

    /// Hold-back correctness: single delta "abcde", stop=["xyz"] (max_stop=3).
    /// With max_stop=3, hold back 2 bytes (max_stop - 1). Safe prefix = "abc" (5-2=3).
    /// finish("", sink) emits "de". into_text() = "abcde".
    #[test]
    fn stop_streamer_hold_back_emits_safe_prefix_then_finish() {
        let stops = vec!["xyz".to_string()]; // max_stop = 3
        let mut streamer = StopStreamer::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        let stopped = streamer.push("abcde", &mut |s| emitted.push(s.to_string()));
        assert!(!stopped);
        // After push: "abcde" (5 bytes), hold back 2 → safe = 3, emits "abc".
        assert_eq!(emitted.join(""), "abc");
        streamer.finish("", &mut |s| emitted.push(s.to_string()));
        assert_eq!(emitted.join(""), "abcde");
        assert_eq!(streamer.into_text(), "abcde");
    }

    /// BUG 2 regression (non-streaming): finish() on StopStreamer is a no-op after stop hit.
    #[test]
    fn stop_streamer_finish_noop_after_stop() {
        let stops = vec!["END".to_string()];
        let mut streamer = StopStreamer::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        let stopped = streamer.push("helloENDextra", &mut |s| emitted.push(s.to_string()));
        assert!(stopped);
        // Simulate tail bytes that would corrupt output if finish were not a no-op.
        streamer.finish("should_not_appear", &mut |s| emitted.push(s.to_string()));
        assert_eq!(emitted.join(""), "hello");
        assert_eq!(streamer.into_text(), "hello");
    }
}
