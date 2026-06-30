//! Text generation loop for decoder-only models.
//!
//! Provides the missing decode loop: prompt → tokenize → prefill → decode → detokenize.
//! Uses the flat KV cache for O(1) per-step attention and the sampling module for
//! temperature/top-k/top-p token selection.
//!
//! # Architecture
//!
//! ```text
//! prompt → tokenize → prefill(all tokens) → [decode(1 token) → sample → emit]* → detokenize
//!                        ↓                           ↓
//!                   fill KV cache              append to KV cache
//! ```

use crate::attention::gqa::GqaConfig;
use crate::error::InferenceError;
use crate::forward::cpu::{elementwise_mul, matmul_bt, rms_norm, silu_inplace};
use crate::grammar::GrammarEngine;
use crate::kv_cache::{FlatKVCache, FlatKVCacheConfig};
use crate::model::qwen::{QwenConfig, QwenModel};
use crate::sampling::{Sampler, SamplingConfig};
use std::sync::Arc;

/// **Unstable**: text generation configuration; fields and defaults are
/// subject to change as the generation API matures.
#[derive(Clone)]
pub struct GenerateConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Sampling configuration.
    pub sampling: SamplingConfig,
    /// EOS token ID (generation stops when this is produced).
    pub eos_token_id: Option<u32>,
    /// Whether to include the prompt in the output.
    pub include_prompt: bool,
    /// Optional grammar-constrained decoding engine (ADR-046).
    ///
    /// When set, `mask_logits` is called before sampling on every step.
    /// The `GrammarEngine` is shared via `Arc` so it can be reused across
    /// requests with the same schema without re-compilation.
    pub grammar: Option<Arc<GrammarEngine>>,
    /// Opt-in KV cache capacity cap in tokens (issue #12).
    ///
    /// When `None` (default), the cache is sized to `prompt_len + max_new_tokens`,
    /// which is the exact working set for this request — no over-allocation occurs.
    ///
    /// When `Some(n)`, the cache is capped to at most `n` tokens regardless of
    /// `max_new_tokens`. This is useful on memory-constrained devices where the
    /// caller knows the workload is short. If generation reaches the cap, it stops
    /// at that point rather than panicking.
    ///
    /// The cap is clamped to `[1, prompt_len + max_new_tokens]` at call time, so
    /// values that exceed the natural request size have no effect.
    pub kv_cache_capacity: Option<usize>,
}

impl std::fmt::Debug for GenerateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerateConfig")
            .field("max_new_tokens", &self.max_new_tokens)
            .field("sampling", &self.sampling)
            .field("eos_token_id", &self.eos_token_id)
            .field("include_prompt", &self.include_prompt)
            .field("grammar", &self.grammar.as_ref().map(|_| "<GrammarEngine>"))
            .field("kv_cache_capacity", &self.kv_cache_capacity)
            .finish()
    }
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: SamplingConfig::default(),
            eos_token_id: None,
            include_prompt: false,
            grammar: None,
            kv_cache_capacity: None,
        }
    }
}

/// **Unstable**: generation result; fields may be added or removed.
#[derive(Debug)]
pub struct GenerateOutput {
    /// The generated text (excluding prompt unless include_prompt is set).
    pub text: String,
    /// All generated token IDs (excluding prompt).
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Total tokens generated (not including prompt).
    pub generated_tokens: usize,
    /// Whether generation stopped due to EOS token.
    pub stopped_by_eos: bool,
}

// ---------------------------------------------------------------------------
// ForwardScratch: pre-allocated activation and logits buffers, reused across
// all tokens in a request. Eliminates 204 allocations/token (Round 0 baseline).
// ---------------------------------------------------------------------------

struct ForwardScratch {
    hidden: Vec<f32>,      // [≥ seq_len_cap * hidden_size]
    residual: Vec<f32>,    // [≥ seq_len_cap * hidden_size]
    qkv_buf: Vec<f32>,     // [≥ seq_len_cap * qkv_dim]
    q_buf: Vec<f32>,       // [≥ seq_len_cap * q_dim]
    k_buf: Vec<f32>,       // [≥ seq_len_cap * kv_dim]
    v_buf: Vec<f32>,       // [≥ seq_len_cap * kv_dim]
    attn_out: Vec<f32>,    // [≥ seq_len_cap * q_dim]
    gate_up_buf: Vec<f32>, // [≥ seq_len_cap * 2 * inter]
    gate_buf: Vec<f32>,    // [≥ seq_len_cap * inter]
    up_buf: Vec<f32>,      // [≥ seq_len_cap * inter]
    ffn_out: Vec<f32>,     // [≥ seq_len_cap * hidden_size]
    scores: Vec<f32>,      // [≥ num_heads * max_seq_len] — score_stride = max_seq_len
    logits: Vec<f32>,      // [≥ vocab_size]
    // Dequantization buffers: f16 KV cache → f32 for compute_attention.
    cached_k_f32: Vec<f32>, // [≥ max_seq_len * kv_dim]
    cached_v_f32: Vec<f32>, // [≥ max_seq_len * kv_dim]
}

impl ForwardScratch {
    fn new() -> Self {
        Self {
            hidden: Vec::new(),
            residual: Vec::new(),
            qkv_buf: Vec::new(),
            q_buf: Vec::new(),
            k_buf: Vec::new(),
            v_buf: Vec::new(),
            attn_out: Vec::new(),
            gate_up_buf: Vec::new(),
            gate_buf: Vec::new(),
            up_buf: Vec::new(),
            ffn_out: Vec::new(),
            scores: Vec::new(),
            logits: Vec::new(),
            cached_k_f32: Vec::new(),
            cached_v_f32: Vec::new(),
        }
    }

    // Grow buffers to fit seq_len_cap tokens and max_seq_len context.
    // Only allocates on the first call or when seq_len_cap > previous capacity.
    // Never shrinks — safe to call every token.
    fn ensure_capacity(&mut self, cfg: &QwenConfig, seq_len_cap: usize, max_seq_len: usize) {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let qkv_dim = q_dim + 2 * kv_dim;
        let inter = cfg.intermediate_size;

        grow(&mut self.hidden, seq_len_cap * h);
        grow(&mut self.residual, seq_len_cap * h);
        grow(&mut self.qkv_buf, seq_len_cap * qkv_dim);
        grow(&mut self.q_buf, seq_len_cap * q_dim);
        grow(&mut self.k_buf, seq_len_cap * kv_dim);
        grow(&mut self.v_buf, seq_len_cap * kv_dim);
        grow(&mut self.attn_out, seq_len_cap * q_dim);
        grow(&mut self.gate_up_buf, seq_len_cap * 2 * inter);
        grow(&mut self.gate_buf, seq_len_cap * inter);
        grow(&mut self.up_buf, seq_len_cap * inter);
        grow(&mut self.ffn_out, seq_len_cap * h);
        grow(&mut self.scores, cfg.num_attention_heads * max_seq_len);
        grow(&mut self.logits, cfg.vocab_size);
        // Dequant scratch: must hold up to max_seq_len * kv_dim f32 elements.
        let kv_dim = cfg.kv_dim();
        grow(&mut self.cached_k_f32, max_seq_len * kv_dim);
        grow(&mut self.cached_v_f32, max_seq_len * kv_dim);
    }
}

fn grow(buf: &mut Vec<f32>, n: usize) {
    if buf.len() < n {
        buf.resize(n, 0.0);
    }
}

// ---------------------------------------------------------------------------
// Public generate() entry point
// ---------------------------------------------------------------------------

/// Reject a `kv_cache_capacity` smaller than the prompt: prefill writes
/// `prompt_len` tokens in a single pass, so a cache that cannot hold the prompt
/// would overflow during prefill (issue #12). Only enforced when the caller
/// opted into a cap; `None` keeps the full `max_seq` sizing and is always valid.
fn check_kv_cache_capacity(
    requested: Option<usize>,
    effective_cap: usize,
    prompt_len: usize,
) -> Result<(), InferenceError> {
    if requested.is_some() && effective_cap < prompt_len {
        return Err(InferenceError::InvalidInput(format!(
            "kv_cache_capacity ({effective_cap}) is smaller than the prompt length \
             ({prompt_len}); the cache must hold at least the prompt"
        )));
    }
    Ok(())
}

/// Compute the maximum sequence length (prompt + generated) used to size the KV
/// cache, guarding against `usize` overflow on a pathological `max_new_tokens`.
///
/// A caller passing `max_new_tokens` near `usize::MAX` would otherwise wrap the
/// addition (release builds elide the overflow check), yielding a tiny `max_seq`,
/// an undersized cache, and a bare panic on the first prefill write into the
/// (now too small) cache buffer. Returning `InvalidInput` turns that latent panic
/// into a clean, caller-visible error.
///
/// This guards only the addition; [`check_alloc_capacity`] guards the subsequent
/// multiplication into per-buffer byte counts.
fn compute_max_seq(prompt_len: usize, max_new_tokens: usize) -> Result<usize, InferenceError> {
    prompt_len.checked_add(max_new_tokens).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "prompt_len ({prompt_len}) + max_new_tokens ({max_new_tokens}) overflows usize"
        ))
    })
}

/// Reject a request whose prompt plus generation would index the RoPE table past
/// its precomputed capacity, turning a latent release panic into a clean
/// `InvalidInput` (issue #467).
///
/// `RopeTable::apply` (rope.rs) indexes its precomputed cos/sin rows unchecked
/// (only a `debug_assert`), so applying RoPE at any position `>= max_context` is
/// an out-of-bounds slice access — a bare panic in release builds, not a
/// caller-visible error. Prefill uses positions `0..prompt_len` and the decode
/// loop reaches at most `prompt_len + max_new_tokens - 2`, so rejecting when
/// `prompt_len + max_new_tokens > max_context` keeps every applied position
/// strictly inside the table. The bound is conservative by one position (it also
/// rejects the exact-fit edge request) and mirrors the Qwen3.5 context preflight
/// in `model/qwen35/generation.rs`.
///
/// Independent of the KV/alloc guards ([`compute_max_seq`],
/// [`check_alloc_capacity`]), which size the cache rather than bound the RoPE
/// position. `saturating_add` avoids a `usize` wrap on a pathological
/// `max_new_tokens`; the wrapped-but-still-huge case is additionally caught by
/// [`compute_max_seq`] below.
fn check_context_window(
    prompt_len: usize,
    max_new_tokens: usize,
    max_context: usize,
) -> Result<(), InferenceError> {
    if prompt_len.saturating_add(max_new_tokens) > max_context {
        return Err(InferenceError::InvalidInput(format!(
            "prompt ({prompt_len} tokens) + max_new_tokens ({max_new_tokens}) exceeds \
             the model context window ({max_context})"
        )));
    }
    Ok(())
}

/// Validate that the KV cache and per-token scratch sized to `effective_cap` will
/// not overflow `usize` when their element counts are computed.
///
/// [`compute_max_seq`] guards the `prompt_len + max_new_tokens` *addition*, but a
/// huge-yet-non-overflowing result (e.g. `max_new_tokens = usize::MAX / 1024`)
/// still wraps the downstream `max_seq_len * dim` *multiplications* in cache
/// sizing ([`FlatKVCacheConfig::layer_capacity`]) and scratch sizing
/// (`ForwardScratch::ensure_capacity`), yielding undersized buffers and a panic on
/// the first write. Every length-scaled allocation is linear in `effective_cap`,
/// so summing their per-position element coefficients and checking the single
/// product `effective_cap * Σcoeff` bounds them all at once: each individual
/// product is `≤` the sum, so if the sum is overflow-safe every term is too.
///
/// The remaining scratch buffers scale with `prompt_len` (real tokenized input,
/// not caller-controlled to pathological sizes), so they are not guarded here.
fn check_alloc_capacity(
    cfg: &QwenConfig,
    num_layers: usize,
    effective_cap: usize,
) -> Result<(), InferenceError> {
    // Guard the base KV dimension product before using it in any further multiply.
    // A pathological config (e.g. num_key_value_heads = usize::MAX) can make the
    // raw `num_key_value_heads * head_dim` multiply wrap in release builds,
    // producing a silently tiny kv_dim that passes every downstream checked_mul
    // even though the real allocation would be absurd. Fail closed here so the
    // error is reported on the path that controls all subsequent buffer sizing.
    let kv_dim = cfg
        .num_key_value_heads
        .checked_mul(cfg.head_dim)
        .ok_or_else(|| {
            InferenceError::InvalidInput("num_key_value_heads * head_dim overflows usize".into())
        })?;
    // Guard the query-side dimension product. A config with num_attention_heads
    // near usize::MAX and a small head_dim overflows q_dim even when kv_dim is
    // safe — producing a tiny q_buf / attn_out / qkv_buf that causes an OOB
    // panic on the first prefill write into the undersized buffers.
    let q_dim = cfg
        .num_attention_heads
        .checked_mul(cfg.head_dim)
        .ok_or_else(|| {
            InferenceError::InvalidInput("num_attention_heads * head_dim overflows usize".into())
        })?;
    // Guard the fused QKV projection width (sizes qkv_buf). q_dim and kv_dim are
    // already validated above; this guards the final addition so a huge q_dim
    // combined with a non-trivial kv_dim cannot wrap the sum.
    let qkv_dim = kv_dim
        .checked_mul(2)
        .and_then(|two_kv| q_dim.checked_add(two_kv))
        .ok_or_else(|| {
            InferenceError::InvalidInput("q_dim + 2*kv_dim (qkv_dim) overflows usize".into())
        })?;
    // Guard every activation buffer that ForwardScratch::ensure_capacity sizes as
    // `seq_len_cap * DIM`. The invariant seq_len_cap <= effective_cap holds because
    // check_kv_cache_capacity rejects any cap smaller than prompt_len, and
    // ensure_capacity is called with at most prompt_len (prefill) or 1 (decode).
    // Guarding effective_cap * DIM up-front prevents any grow() call from wrapping
    // usize and silently undersizing a buffer before the first prefill write.
    //
    // Buffers covered and their per-token coefficients:
    //   hidden, residual, ffn_out  — hidden_size
    //   qkv_buf                    — qkv_dim  (= q_dim + 2*kv_dim)
    //   q_buf, attn_out            — q_dim    (= num_attention_heads * head_dim)
    //   k_buf, v_buf               — kv_dim   (= num_key_value_heads * head_dim)
    //   gate_up_buf                — 2 * intermediate_size
    //   gate_buf, up_buf           — intermediate_size
    let inter = cfg.intermediate_size;
    effective_cap.checked_mul(cfg.hidden_size).ok_or_else(|| {
        InferenceError::InvalidInput("effective_cap * hidden_size overflows usize".into())
    })?;
    effective_cap.checked_mul(qkv_dim).ok_or_else(|| {
        InferenceError::InvalidInput("effective_cap * qkv_dim overflows usize".into())
    })?;
    effective_cap.checked_mul(q_dim).ok_or_else(|| {
        InferenceError::InvalidInput("effective_cap * q_dim overflows usize".into())
    })?;
    effective_cap.checked_mul(kv_dim).ok_or_else(|| {
        InferenceError::InvalidInput("effective_cap * kv_dim overflows usize".into())
    })?;
    inter
        .checked_mul(2)
        .and_then(|two_inter| effective_cap.checked_mul(two_inter))
        .ok_or_else(|| {
            InferenceError::InvalidInput(
                "effective_cap * 2 * intermediate_size overflows usize".into(),
            )
        })?;
    effective_cap.checked_mul(inter).ok_or_else(|| {
        InferenceError::InvalidInput("effective_cap * intermediate_size overflows usize".into())
    })?;
    // Per-position element coefficients that scale with the cache length:
    //   KV cache (K+V across all layers): 2 * num_layers * kv_dim
    //   dequant scratch (cached_k_f32 + cached_v_f32): 2 * kv_dim
    //   attention scores: num_attention_heads
    let coeff = (|| -> Option<usize> {
        let cache = 2usize.checked_mul(num_layers)?.checked_mul(kv_dim)?;
        let dequant = 2usize.checked_mul(kv_dim)?;
        cache
            .checked_add(dequant)?
            .checked_add(cfg.num_attention_heads)
    })()
    .ok_or_else(|| InferenceError::InvalidInput("model dimensions overflow usize".into()))?;
    effective_cap.checked_mul(coeff).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "KV cache + scratch for max_seq ({effective_cap}) overflows usize"
        ))
    })?;
    Ok(())
}

/// Returns `true` when at least one logit is strictly greater than
/// `f32::NEG_INFINITY` — i.e. the grammar mask leaves at least one legal token.
///
/// A grammar engine calls `mask_logits` to set every disallowed position to
/// `NEG_INFINITY`. If it blocks every token, all logits become `NEG_INFINITY`
/// and the sampler's non-finite-max short-circuit silently emits token 0 — the
/// lowest-id token after sorting an all-NEG_INFINITY candidate set by ascending
/// token_id. That violates the grammar contract. This helper lets the caller
/// detect and reject the empty-mask case before invoking the sampler (#398).
fn has_finite_logit(logits: &[f32]) -> bool {
    logits.iter().any(|&l| l > f32::NEG_INFINITY)
}

/// **Unstable**: text generation entry point for Qwen3 models; the full
/// generation loop (prefill, decode, sampling) is under active design.
///
/// This is the main entry point for text generation. It handles:
/// 1. Tokenization of the prompt
/// 2. Prefill (full forward pass on prompt, populating KV cache)
/// 3. Decode loop (token-by-token generation with KV cache)
/// 4. Sampling (temperature, top-k, top-p, repetition penalty)
/// 5. Detokenization of the result
///
/// # Note on lm_head
///
/// Qwen3 models tie `lm_head` weights with `embed_tokens` (transposed).
/// The logits are computed as: `logits = hidden_state @ embed_tokens^T`.
/// This avoids loading a separate lm_head weight.
pub fn generate(
    model: &QwenModel,
    prompt: &str,
    config: &GenerateConfig,
) -> Result<GenerateOutput, InferenceError> {
    let cfg = model.config();
    let tokenizer = model.tokenizer();

    // 1. Tokenize prompt
    let tokenized = tokenizer.tokenize(prompt);
    let prompt_ids: Vec<u32> = tokenized.input_ids.clone();
    let prompt_len = tokenized.real_length;

    if prompt_len == 0 {
        return Err(InferenceError::InvalidInput("Empty prompt".into()));
    }

    // max_new_tokens == 0 means "generate nothing": return before prefill/sampling
    // so we never emit a token the caller did not ask for.
    if config.max_new_tokens == 0 {
        return Ok(GenerateOutput {
            text: String::new(),
            token_ids: Vec::new(),
            prompt_tokens: prompt_len,
            generated_tokens: 0,
            stopped_by_eos: false,
        });
    }

    // Context preflight: reject before any allocation if the request would drive
    // RoPE past its precomputed table. The KV/alloc guards below size the cache,
    // not the position index, so this is a distinct bound (see check_context_window).
    check_context_window(
        prompt_len,
        config.max_new_tokens,
        model.rope().max_positions(),
    )?;

    // 2. Initialize KV cache and scratch (allocate once per request)
    let max_seq = compute_max_seq(prompt_len, config.max_new_tokens)?;
    // Effective cache capacity: honour the caller's opt-in cap (issue #12).
    // Clamped to [1, max_seq] so over-large caps and zero are both safe.
    let effective_cap = config
        .kv_cache_capacity
        .map(|c| c.max(1).min(max_seq))
        .unwrap_or(max_seq);
    check_kv_cache_capacity(config.kv_cache_capacity, effective_cap, prompt_len)?;
    check_alloc_capacity(cfg, cfg.num_hidden_layers, effective_cap)?;
    let cache_cfg = FlatKVCacheConfig::for_qwen3(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        effective_cap,
    );
    // try_new validates every dimension product before any Vec allocation.
    // check_alloc_capacity (above) guards the same products, but try_new provides
    // a second layer at the exact allocation boundary — a defence against future
    // refactors that might move or remove the upstream guard.
    let mut cache = FlatKVCache::try_new(cache_cfg)?;
    let mut scratch = ForwardScratch::new();
    // Size scratch for the largest possible call (prefill at prompt_len tokens).
    // Decode calls use seq_len=1, which is always within this capacity.
    scratch.ensure_capacity(cfg, prompt_len.max(1), effective_cap);

    // 3. Initialize sampler and seed prompt history so prompt tokens are included
    // in repetition-penalty scoring from the very first generated token.
    let mut sampler = Sampler::new(config.sampling.clone());
    sampler.seed_history(&prompt_ids[..prompt_len]);

    // 4. Initialise grammar state if grammar-constrained decoding is requested.
    let mut grammar_state = config.grammar.as_ref().map(|g| g.initial_state());

    // 5. Prefill: run all prompt tokens through the model, populate KV cache.
    // The borrow of scratch.logits ends before sampling, so we can apply
    // grammar masking on scratch.logits directly before passing it to sample().
    forward_with_cache(
        model,
        &prompt_ids[..prompt_len],
        &mut cache,
        0,
        &mut scratch,
        effective_cap,
    )?;

    // Apply grammar masking on the logit buffer in-place before sampling.
    if let (Some(engine), Some(gs)) = (&config.grammar, &mut grammar_state) {
        engine.mask_logits(gs, &mut scratch.logits[..cfg.vocab_size]);
        // If the grammar blocked every token the sampler's non-finite-max
        // short-circuit would silently return token 0 (the lowest id after
        // sorting an all-NEG_INFINITY candidate set). Fail closed instead (#398).
        if !has_finite_logit(&scratch.logits[..cfg.vocab_size]) {
            return Err(InferenceError::InvalidInput(
                "grammar constraint blocked every token at step 0; \
                 no legal first token exists in the current grammar state"
                    .into(),
            ));
        }
    }

    // 6. Sample first token from the last position's logits.
    // Cap the preallocation hint at effective_cap (the real generation ceiling —
    // decode stops once the cache is full), not the raw max_new_tokens: a caller
    // with a small kv_cache_capacity and a huge max_new_tokens would otherwise
    // panic in Vec::with_capacity (capacity * 4 bytes > isize::MAX). effective_cap
    // is already validated allocation-safe by check_alloc_capacity above.
    let mut generated_ids: Vec<u32> = Vec::with_capacity(config.max_new_tokens.min(effective_cap));
    let first_token = sampler.sample(&scratch.logits[..cfg.vocab_size]);

    // Advance grammar state after sampling. The token is pushed to generated_ids
    // only after a successful advance so a grammar-rejected token (sampled despite
    // the mask, e.g. from rounding on a boundary logit) does not appear in the
    // output (#398).
    if let (Some(engine), Some(gs)) = (&config.grammar, &mut grammar_state) {
        if !engine.advance(gs, first_token) {
            return Ok(GenerateOutput {
                text: String::new(),
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                token_ids: vec![],
                stopped_by_eos: false,
            });
        }
    }
    generated_ids.push(first_token);

    let mut stopped_by_eos = false;
    if config.eos_token_id == Some(first_token) {
        stopped_by_eos = true;
    }

    // 7. Decode loop: one token at a time
    if !stopped_by_eos {
        for step in 0..config.max_new_tokens.saturating_sub(1) {
            // Stop cleanly when the cache is at capacity (set by kv_cache_capacity).
            if cache.is_full() {
                break;
            }
            let pos = prompt_len + step + 1; // +1 because first token already generated
            let input = [*generated_ids
                .last()
                .expect("invariant: first generated token exists before decode loop")];
            forward_with_cache(
                model,
                &input,
                &mut cache,
                pos - 1,
                &mut scratch,
                effective_cap,
            )?;

            // Apply grammar masking before sampling.
            if let (Some(engine), Some(gs)) = (&config.grammar, &mut grammar_state) {
                engine.mask_logits(gs, &mut scratch.logits[..cfg.vocab_size]);
                // Same empty-mask guard as the first-token path: an all-NEG_INFINITY
                // logit buffer would cause the sampler to silently emit token 0 (#398).
                if !has_finite_logit(&scratch.logits[..cfg.vocab_size]) {
                    return Err(InferenceError::InvalidInput(
                        "grammar constraint blocked every token; \
                         no legal continuation exists in the current grammar state"
                            .into(),
                    ));
                }
            }

            let token = sampler.sample(&scratch.logits[..cfg.vocab_size]);
            // Advance grammar state after sampling.
            if let (Some(engine), Some(gs)) = (&config.grammar, &mut grammar_state) {
                if !engine.advance(gs, token) {
                    break;
                }
            }
            generated_ids.push(token);

            if config.eos_token_id == Some(token) {
                stopped_by_eos = true;
                break;
            }
        }
    }

    // 7. Detokenize via the tokenizer's decode (BpeTokenizer implements
    // GPT-2 byte-level decode; only non-generative encoder tokenizers
    // return None, in which case text is empty but token_ids still carry
    // the result).
    let decoded = tokenizer.decode(&generated_ids).unwrap_or_default();
    let full_text = if config.include_prompt {
        format!("{prompt}{decoded}")
    } else {
        decoded
    };

    Ok(GenerateOutput {
        text: full_text,
        prompt_tokens: prompt_len,
        generated_tokens: generated_ids.len(),
        token_ids: generated_ids,
        stopped_by_eos,
    })
}

// ---------------------------------------------------------------------------
// forward_with_cache: core transformer forward pass
// ---------------------------------------------------------------------------

/// Run a forward pass on `input_ids`, populating the KV cache, and return
/// a borrowed slice into `scratch.logits` for the LAST token (shape `[vocab_size]`).
///
/// For prefill: `input_ids` is the full prompt, `start_pos` is 0.
/// For decode: `input_ids` is a single token, `start_pos` is the current position.
///
/// The returned slice borrows `scratch`; caller must drop it before re-borrowing.
fn forward_with_cache<'a>(
    model: &QwenModel,
    input_ids: &[u32],
    cache: &mut FlatKVCache,
    start_pos: usize,
    scratch: &'a mut ForwardScratch,
    max_seq_len: usize,
) -> Result<&'a [f32], InferenceError> {
    let cfg = model.config();
    let seq_len = input_ids.len();
    let hidden_size = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let qkv_dim = q_dim + 2 * kv_dim;
    let inter = cfg.intermediate_size;
    let head_dim = cfg.head_dim;

    // Grow scratch if needed (no-op on warm decode since prefill already sized it).
    scratch.ensure_capacity(cfg, seq_len, max_seq_len);

    let weights = model.weights();
    let rope = model.rope();

    // Embedding lookup → scratch.hidden
    for (i, &tok) in input_ids.iter().enumerate() {
        let tok = tok as usize;
        if tok >= cfg.vocab_size {
            return Err(InferenceError::InvalidInput(format!(
                "Token ID {tok} exceeds vocab size {}",
                cfg.vocab_size
            )));
        }
        let row = &weights.embed_tokens.data[tok * hidden_size..(tok + 1) * hidden_size];
        scratch.hidden[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(row);
    }

    let gqa_cfg = GqaConfig {
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim,
    };

    // Transformer layers
    for layer_idx in 0..cfg.num_hidden_layers {
        let lw = &weights.layers[layer_idx];

        // Pre-attention RMS norm (normalizes all seq_len tokens at once)
        scratch.residual[..seq_len * hidden_size]
            .copy_from_slice(&scratch.hidden[..seq_len * hidden_size]);
        rms_norm(
            &mut scratch.hidden[..seq_len * hidden_size],
            lw.input_layernorm_weight.data,
            hidden_size,
            cfg.rms_norm_eps,
        );

        // Fused QKV projection: [seq_len, hidden] × [qkv_dim, hidden]^T → [seq_len, qkv_dim]
        matmul_bt(
            &scratch.hidden[..seq_len * hidden_size],
            &lw.fused_qkv,
            &mut scratch.qkv_buf[..seq_len * qkv_dim],
            seq_len,
            hidden_size,
            qkv_dim,
        );

        // Scatter fused QKV into contiguous Q, K, V buffers
        for i in 0..seq_len {
            let qkv_row = i * qkv_dim;
            scratch.q_buf[i * q_dim..(i + 1) * q_dim]
                .copy_from_slice(&scratch.qkv_buf[qkv_row..qkv_row + q_dim]);
            scratch.k_buf[i * kv_dim..(i + 1) * kv_dim]
                .copy_from_slice(&scratch.qkv_buf[qkv_row + q_dim..qkv_row + q_dim + kv_dim]);
            scratch.v_buf[i * kv_dim..(i + 1) * kv_dim]
                .copy_from_slice(&scratch.qkv_buf[qkv_row + q_dim + kv_dim..qkv_row + qkv_dim]);
        }

        // Per-head QK RMS norm
        for i in 0..seq_len {
            for h in 0..cfg.num_attention_heads {
                let off = i * q_dim + h * head_dim;
                rms_norm(
                    &mut scratch.q_buf[off..off + head_dim],
                    lw.q_norm_weight.data,
                    head_dim,
                    cfg.rms_norm_eps,
                );
            }
            for h in 0..cfg.num_key_value_heads {
                let off = i * kv_dim + h * head_dim;
                rms_norm(
                    &mut scratch.k_buf[off..off + head_dim],
                    lw.k_norm_weight.data,
                    head_dim,
                    cfg.rms_norm_eps,
                );
            }
        }

        // RoPE on Q and K
        for i in 0..seq_len {
            let pos = start_pos + i;
            for h in 0..cfg.num_attention_heads {
                let off = i * q_dim + h * head_dim;
                rope.apply(&mut scratch.q_buf[off..off + head_dim], pos);
            }
            for h in 0..cfg.num_key_value_heads {
                let off = i * kv_dim + h * head_dim;
                rope.apply(&mut scratch.k_buf[off..off + head_dim], pos);
            }
        }

        // Append K, V to cache for this layer (convert f32→f16 on write).
        {
            let base_pos = cache.seq_len();
            let k_layer = cache.k_buffer_mut(layer_idx);
            for i in 0..seq_len {
                let dst_off = (base_pos + i) * kv_dim;
                for (j, &val) in scratch.k_buf[i * kv_dim..(i + 1) * kv_dim]
                    .iter()
                    .enumerate()
                {
                    k_layer[dst_off + j] = half::f16::from_f32(val);
                }
            }
            let v_layer = cache.v_buffer_mut(layer_idx);
            for i in 0..seq_len {
                let dst_off = (base_pos + i) * kv_dim;
                for (j, &val) in scratch.v_buf[i * kv_dim..(i + 1) * kv_dim]
                    .iter()
                    .enumerate()
                {
                    v_layer[dst_off + j] = half::f16::from_f32(val);
                }
            }
        }

        // Attention: Q(seq_len) @ K(cached)^T → softmax → @ V(cached)
        // Dequantize f16 KV cache into f32 scratch buffers before compute_attention.
        let cached_seq_len = cache.seq_len() + seq_len; // not yet advanced
        let k_end = cached_seq_len * kv_dim;
        for (i, &h) in cache.k_buffer(layer_idx)[..k_end].iter().enumerate() {
            scratch.cached_k_f32[i] = h.to_f32();
        }
        for (i, &h) in cache.v_buffer(layer_idx)[..k_end].iter().enumerate() {
            scratch.cached_v_f32[i] = h.to_f32();
        }

        // Split scratch borrows: attn_out + scores are mutable; q_buf, cached_k_f32,
        // cached_v_f32 are immutable. All are distinct Vec fields — no aliasing.
        {
            // SAFETY: q_buf, cached_k_f32, cached_v_f32, attn_out, scores are distinct
            // Vec fields — no aliasing. Raw pointers avoid simultaneous borrow conflicts.
            let q_ptr = scratch.q_buf.as_ptr();
            let ck_ptr = scratch.cached_k_f32.as_ptr();
            let cv_ptr = scratch.cached_v_f32.as_ptr();
            let attn_ptr = scratch.attn_out.as_mut_ptr();
            let scores_ptr = scratch.scores.as_mut_ptr();
            let attn_len = seq_len * q_dim;
            let scores_len = scratch.scores.len();
            let (q_slice, cached_k, cached_v, attn_slice, scores_slice) = unsafe {
                (
                    std::slice::from_raw_parts(q_ptr, seq_len * q_dim),
                    std::slice::from_raw_parts(ck_ptr, k_end),
                    std::slice::from_raw_parts(cv_ptr, k_end),
                    std::slice::from_raw_parts_mut(attn_ptr, attn_len),
                    std::slice::from_raw_parts_mut(scores_ptr, scores_len),
                )
            };
            compute_attention(
                attn_slice,
                q_slice,
                cached_k,
                cached_v,
                seq_len,
                cached_seq_len,
                start_pos,
                &gqa_cfg,
                scores_slice,
                max_seq_len,
            );
        }

        // O projection: [seq_len, q_dim] × [hidden_size, q_dim]^T → [seq_len, hidden_size]
        matmul_bt(
            &scratch.attn_out[..seq_len * q_dim],
            lw.o_proj_weight.data,
            &mut scratch.hidden[..seq_len * hidden_size],
            seq_len,
            q_dim,
            hidden_size,
        );

        // Residual connection
        for i in 0..seq_len * hidden_size {
            scratch.hidden[i] += scratch.residual[i];
        }

        // Post-attention RMS norm
        scratch.residual[..seq_len * hidden_size]
            .copy_from_slice(&scratch.hidden[..seq_len * hidden_size]);
        rms_norm(
            &mut scratch.hidden[..seq_len * hidden_size],
            lw.post_attention_layernorm_weight.data,
            hidden_size,
            cfg.rms_norm_eps,
        );

        // Fused gate+up projection: [seq_len, hidden] × [2*inter, hidden]^T → [seq_len, 2*inter]
        matmul_bt(
            &scratch.hidden[..seq_len * hidden_size],
            &lw.fused_gate_up,
            &mut scratch.gate_up_buf[..seq_len * 2 * inter],
            seq_len,
            hidden_size,
            2 * inter,
        );

        // Scatter gate and up
        for i in 0..seq_len {
            let gu_row = i * 2 * inter;
            scratch.gate_buf[i * inter..(i + 1) * inter]
                .copy_from_slice(&scratch.gate_up_buf[gu_row..gu_row + inter]);
            scratch.up_buf[i * inter..(i + 1) * inter]
                .copy_from_slice(&scratch.gate_up_buf[gu_row + inter..gu_row + 2 * inter]);
        }

        // SwiGLU: silu(gate) * up
        silu_inplace(&mut scratch.gate_buf[..seq_len * inter]);
        elementwise_mul(
            &mut scratch.gate_buf[..seq_len * inter],
            &scratch.up_buf[..seq_len * inter],
        );

        // Down projection: [seq_len, inter] × [hidden_size, inter]^T → [seq_len, hidden_size]
        matmul_bt(
            &scratch.gate_buf[..seq_len * inter],
            lw.down_proj_weight.data,
            &mut scratch.ffn_out[..seq_len * hidden_size],
            seq_len,
            inter,
            hidden_size,
        );

        // Residual
        for i in 0..seq_len * hidden_size {
            scratch.hidden[i] = scratch.residual[i] + scratch.ffn_out[i];
        }
    }

    // Advance KV cache position after all layers processed
    cache.advance_by(seq_len)?;

    // Final RMS norm on last token's hidden state
    let last_start = (seq_len - 1) * hidden_size;
    rms_norm(
        &mut scratch.hidden[last_start..last_start + hidden_size],
        weights.norm_weight.data,
        hidden_size,
        cfg.rms_norm_eps,
    );

    // Opt 3: logits projection via matmul_bt (Accelerate/NEON/AVX2 dispatch).
    // Replaces the per-vocab scalar dot-product loop.
    // embed_tokens [vocab_size, hidden_size] used as tied lm_head weights.
    matmul_bt(
        &scratch.hidden[last_start..last_start + hidden_size],
        weights.embed_tokens.data,
        &mut scratch.logits[..cfg.vocab_size],
        1,
        hidden_size,
        cfg.vocab_size,
    );

    Ok(&scratch.logits[..cfg.vocab_size])
}

// ---------------------------------------------------------------------------
// compute_attention: GQA multi-head attention with cached K/V
// ---------------------------------------------------------------------------

// NEON intrinsic helpers for the decode attention hot path (head_dim=128 fast path).

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// # Safety
/// `q` and `k` must each point to at least 128 contiguous, initialized f32 values.
unsafe fn dot_f32_neon_128(q: *const f32, k: *const f32) -> f32 {
    use std::arch::aarch64::*;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut d = 0usize;
    // head_dim=128: 8 outer iterations × 16 lanes = 128 f32, 4 accumulators break the dependency chain.
    while d < 128 {
        // SAFETY: caller guarantees q and k each have ≥128 f32 values starting at the pointer.
        let q0 = vld1q_f32(q.add(d));
        let k0 = vld1q_f32(k.add(d));
        let q1 = vld1q_f32(q.add(d + 4));
        let k1 = vld1q_f32(k.add(d + 4));
        let q2 = vld1q_f32(q.add(d + 8));
        let k2 = vld1q_f32(k.add(d + 8));
        let q3 = vld1q_f32(q.add(d + 12));
        let k3 = vld1q_f32(k.add(d + 12));
        acc0 = vfmaq_f32(acc0, q0, k0);
        acc1 = vfmaq_f32(acc1, q1, k1);
        acc2 = vfmaq_f32(acc2, q2, k2);
        acc3 = vfmaq_f32(acc3, q3, k3);
        d += 16;
    }
    let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    vaddvq_f32(acc)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// Accumulate one output head's weighted V sum from a pre-zeroed `out` buffer.
///
/// Computes `out[d] = Σ_{ki<kv_len} scores[ki] * v_base[ki*kv_row_stride + d]` for d in 0..128.
/// Accumulates across ki in NEON registers per 16-element output chunk, then stores once —
/// eliminating the read-modify-write traffic of the scalar loop.
///
/// # Safety
/// - `out` must point to at least 128 contiguous writable f32 values.
/// - `scores` must point to at least `kv_len` initialized f32 values.
/// - `v_base[ki * kv_row_stride + d]` must be valid for all `ki < kv_len`, `d < 128`.
unsafe fn accum_v_neon_128(
    out: *mut f32,
    scores: *const f32,
    v_base: *const f32,
    kv_len: usize,
    kv_row_stride: usize,
) {
    use std::arch::aarch64::*;
    let mut d = 0usize;
    // 128 / 16 = 8 outer iterations, each accumulating 16 output dimensions via 4 f32x4 registers.
    while d < 128 {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        for ki in 0..kv_len {
            // SAFETY: scores has kv_len entries; v_base row ki has at least kv_row_stride ≥ 128 f32.
            let w = vdupq_n_f32(*scores.add(ki));
            let v_row = v_base.add(ki * kv_row_stride + d);
            acc0 = vfmaq_f32(acc0, w, vld1q_f32(v_row));
            acc1 = vfmaq_f32(acc1, w, vld1q_f32(v_row.add(4)));
            acc2 = vfmaq_f32(acc2, w, vld1q_f32(v_row.add(8)));
            acc3 = vfmaq_f32(acc3, w, vld1q_f32(v_row.add(12)));
        }
        // SAFETY: out has ≥128 writable f32 values; d+12 < 128 for all loop iterations.
        vst1q_f32(out.add(d), acc0);
        vst1q_f32(out.add(d + 4), acc1);
        vst1q_f32(out.add(d + 8), acc2);
        vst1q_f32(out.add(d + 12), acc3);
        d += 16;
    }
}

/// Compute the dot product of `q[q_off..q_off+head_dim]` · `k[k_off..k_off+head_dim]`.
/// Uses NEON on aarch64 for head_dim=128; scalar otherwise.
#[inline(always)]
fn dot_f32_dispatch(q: &[f32], q_off: usize, k: &[f32], k_off: usize, head_dim: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if head_dim == 128 {
        // SAFETY: head_dim == 128 guarantees q[q_off..q_off+128] and k[k_off..k_off+128] are valid.
        return unsafe { dot_f32_neon_128(q.as_ptr().add(q_off), k.as_ptr().add(k_off)) };
    }
    let mut dot = 0.0f32;
    for d in 0..head_dim {
        dot += q[q_off + d] * k[k_off + d];
    }
    dot
}

/// Accumulate weighted V for one output head (zeroed on entry).
/// Uses NEON on aarch64 for head_dim=128; scalar otherwise.
#[inline(always)]
fn accum_v_dispatch(
    out: &mut [f32],
    out_off: usize,
    scores: &[f32],
    score_off: usize,
    v: &[f32],
    v_base_off: usize,
    kv_len: usize,
    kv_row_stride: usize,
    head_dim: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if head_dim == 128 {
        // SAFETY: head_dim == 128; out[out_off..out_off+128] is valid and pre-zeroed;
        // scores[score_off..score_off+kv_len] is valid;
        // v[v_base_off + ki*kv_row_stride .. +128] is valid for all ki < kv_len
        // because kv_h*head_dim + (kv_len-1)*kv_row_stride + 127 = v.len()-1.
        unsafe {
            accum_v_neon_128(
                out.as_mut_ptr().add(out_off),
                scores.as_ptr().add(score_off),
                v.as_ptr().add(v_base_off),
                kv_len,
                kv_row_stride,
            );
        }
        return;
    }
    for ki in 0..kv_len {
        let w = scores[score_off + ki];
        let v_off = v_base_off + ki * kv_row_stride;
        for d in 0..head_dim {
            out[out_off + d] += w * v[v_off + d];
        }
    }
}

/// Compute multi-head attention with GQA, supporting cached K/V and score scratch.
///
/// Q shape: [q_seq_len, num_heads * head_dim]
/// K shape: [kv_seq_len, num_kv_heads * head_dim]  (from cache)
/// V shape: [kv_seq_len, num_kv_heads * head_dim]  (from cache)
///
/// `scores_scratch` must have length >= `num_heads * score_stride`.
/// `score_stride` is the allocated width per head (>= kv_seq_len).
fn compute_attention(
    output: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_seq_len: usize,
    kv_seq_len: usize,
    start_pos: usize,
    cfg: &GqaConfig,
    scores_scratch: &mut [f32],
    score_stride: usize,
) {
    let head_dim = cfg.head_dim;
    let num_heads = cfg.num_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let groups = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    debug_assert!(kv_seq_len <= score_stride);
    debug_assert!(scores_scratch.len() >= num_heads * score_stride);

    // H1: decode fast path — share K and V loads across all Q heads in each GQA group.
    // In decode mode kv_seq_len == start_pos + 1, so every KV position is within the
    // causal window (ki <= start_pos == kv_seq_len - 1). No masking needed.
    // For groups=G this loads K and V once per position instead of G times,
    // saving (G-1)/G of the K/V bandwidth.
    if q_seq_len == 1 {
        output[..num_heads * head_dim].fill(0.0);

        // Phase 1: Q·K^T — each K row loaded once, dotted with all G query heads.
        // NEON dot product on aarch64 for head_dim=128; scalar fallback otherwise.
        for kv_h in 0..num_kv_heads {
            let group_start = kv_h * groups;
            for ki in 0..kv_seq_len {
                let k_off = ki * (num_kv_heads * head_dim) + kv_h * head_dim;
                for gi in 0..groups {
                    let h = group_start + gi;
                    let q_off = h * head_dim; // qi=0
                    scores_scratch[h * score_stride + ki] =
                        dot_f32_dispatch(q, q_off, k, k_off, head_dim) * scale;
                }
            }
        }

        // Phase 2: per-head online safe softmax (two passes instead of three).
        // Running (m, l) accumulation removes the separate max scan over scores.
        for h in 0..num_heads {
            let scores = &mut scores_scratch[h * score_stride..h * score_stride + kv_seq_len];
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            for &s in scores.iter() {
                let m_new = m.max(s);
                let alpha = if m == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m - m_new).exp()
                };
                l = l * alpha + (s - m_new).exp();
                m = m_new;
            }
            if l > 0.0 {
                let inv_l = 1.0 / l;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp() * inv_l;
                }
            } else {
                scores.fill(0.0);
            }
        }

        // Phase 3: V accumulation — per-head, NEON-accelerated on aarch64 for head_dim=128.
        // Accumulates in registers and stores once per 16-element chunk (no read-modify-write).
        let kv_row_stride = num_kv_heads * head_dim;
        for h in 0..num_heads {
            let kv_h = h / groups;
            accum_v_dispatch(
                output,
                h * head_dim,
                scores_scratch,
                h * score_stride,
                v,
                kv_h * head_dim,
                kv_seq_len,
                kv_row_stride,
                head_dim,
            );
        }
        return;
    }

    // Generic path: prefill and multi-token decode with causal masking.
    for h in 0..num_heads {
        let kv_h = h / groups;
        let head_score_base = h * score_stride;

        for qi in 0..q_seq_len {
            let q_off = qi * (num_heads * head_dim) + h * head_dim;
            // Slice scores for this head/query from the pre-allocated scratch.
            // The slice has length kv_seq_len; score_stride provides the stride between heads.
            let scores = &mut scores_scratch[head_score_base..head_score_base + kv_seq_len];

            // Compute Q @ K^T with causal masking
            let max_attend = start_pos + qi;
            for ki in 0..kv_seq_len {
                let k_off = ki * (num_kv_heads * head_dim) + kv_h * head_dim;
                if ki > max_attend {
                    scores[ki] = f32::NEG_INFINITY;
                } else {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[k_off + d];
                    }
                    scores[ki] = dot * scale;
                }
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for s in scores.iter_mut() {
                    *s *= inv;
                }
            } else {
                // Fail closed, mirroring the q_seq_len==1 online fast path (l > 0.0
                // guard above). A non-finite score (e.g. a NaN Q/K activation) makes
                // `sum` NaN; `NaN > 0.0` is false, so without this branch the
                // unnormalized NaN weights would flow into the V accumulation and
                // poison the attention output. Zeroing the row drops this query's
                // contribution instead of propagating garbage into the logits.
                scores.fill(0.0);
            }

            // Weighted sum of V
            let out_off = qi * (num_heads * head_dim) + h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for ki in 0..kv_seq_len {
                    let v_off = ki * (num_kv_heads * head_dim) + kv_h * head_dim;
                    val += scores[ki] * v[v_off + d];
                }
                output[out_off + d] = val;
            }
        }
    }
}

#[cfg(feature = "bench-internals")]
pub mod bench_support {
    use super::compute_attention;
    use crate::attention::gqa::GqaConfig;

    #[inline]
    pub fn compute_attention_for_bench(
        output: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_seq_len: usize,
        kv_seq_len: usize,
        start_pos: usize,
        cfg: &GqaConfig,
        scores_scratch: &mut [f32],
        score_stride: usize,
    ) {
        compute_attention(
            output,
            q,
            k,
            v,
            q_seq_len,
            kv_seq_len,
            start_pos,
            cfg,
            scores_scratch,
            score_stride,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_finite_logit_detects_all_neg_infinity() {
        // Regression test for the empty-mask fail-closed guard (#398).
        //
        // When a grammar engine blocks every token, all logits become
        // NEG_INFINITY. Without the guard, the sampler's non-finite-max
        // short-circuit silently emits token 0 (the lowest id after sorting
        // an all-NEG_INFINITY candidate set). The guard must detect this and
        // allow the caller to return a typed error instead.
        //
        // Mutation sensitivity: change has_finite_logit to always return true
        // → the first assertion fails because the guard would pass silently
        // on an all-blocked logit buffer.
        assert!(
            !has_finite_logit(&[f32::NEG_INFINITY; 8]),
            "all-NEG_INFINITY must fail the guard (empty grammar mask)"
        );
        // A single finite logit makes the mask non-empty; the guard passes.
        let mut mixed = vec![f32::NEG_INFINITY; 4];
        mixed[2] = 1.0_f32;
        assert!(
            has_finite_logit(&mixed),
            "a single finite logit must pass the guard"
        );
        // All-zero is a valid (uniform) distribution; token 0 wins by argmax.
        assert!(has_finite_logit(&[0.0_f32; 4]));
        // Positive infinity is still a finite-or-higher winner; not blocked.
        let mut with_inf = vec![f32::NEG_INFINITY; 4];
        with_inf[1] = f32::INFINITY;
        assert!(has_finite_logit(&with_inf));
    }

    #[test]
    fn test_generate_config_default() {
        let cfg = GenerateConfig::default();
        assert_eq!(cfg.max_new_tokens, 256);
        assert!(!cfg.include_prompt);
    }

    #[test]
    fn test_generate_output_struct() {
        let output = GenerateOutput {
            text: "Hello world".into(),
            token_ids: vec![1, 2, 3],
            prompt_tokens: 5,
            generated_tokens: 3,
            stopped_by_eos: false,
        };
        assert_eq!(output.generated_tokens, 3);
        assert!(!output.stopped_by_eos);
    }

    #[test]
    fn test_compute_max_seq_normal() {
        assert_eq!(compute_max_seq(10, 100).unwrap(), 110);
        assert_eq!(compute_max_seq(0, 0).unwrap(), 0);
        // Largest sum that still fits exactly.
        assert_eq!(compute_max_seq(1, usize::MAX - 1).unwrap(), usize::MAX);
    }

    #[test]
    fn test_compute_max_seq_overflow_is_error_not_panic() {
        // A pathological max_new_tokens near usize::MAX must surface a clean
        // InvalidInput error rather than wrapping the addition and panicking
        // later inside prefill_layer's capacity assertion (codex finding #2).
        let err = compute_max_seq(10, usize::MAX).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidInput(_)));
        let err = compute_max_seq(usize::MAX, 1).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }

    /// `check_context_window` must reject a request that would drive RoPE past the
    /// precomputed table — the public `generate` entry point would otherwise panic
    /// in release on the unchecked cos/sin index inside `RopeTable::apply`
    /// (issue #467). The message must contain "context window" (the shared wording
    /// of the Qwen3.5 / NEON / Q8 context preflights) so the abuse-path family stays
    /// greppable.
    ///
    /// Mutation check: deleting the guard returns `Ok(())`, failing `expect_err`;
    /// reverting `saturating_add` to `+` wraps `100 + usize::MAX` to `99 <= 4096`
    /// (release) or panics on overflow (debug), either way failing the test.
    #[test]
    fn test_check_context_window_over_capacity_is_error_not_panic() {
        let err = check_context_window(100, usize::MAX, 4096)
            .expect_err("over-capacity request must be rejected, not admitted to a panic");
        let msg = format!("{err}");
        assert!(
            msg.contains("context window"),
            "error should mention context window, got: {msg}"
        );
        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }

    /// The guard is exact at the capacity boundary: a request summing to exactly
    /// `max_context` is accepted (the largest request the conservative bound
    /// admits), one token past it is rejected.
    ///
    /// Mutation check: `>` → `>=` rejects the exact-fit request and fails the first
    /// assertion; `>` → `<` (or removing the guard) admits the over-capacity request
    /// and fails the second.
    #[test]
    fn test_check_context_window_boundary() {
        assert!(
            check_context_window(4000, 96, 4096).is_ok(),
            "prompt+max_new == max_context must be accepted"
        );
        assert!(
            check_context_window(4000, 97, 4096).is_err(),
            "prompt+max_new == max_context + 1 must be rejected"
        );
    }

    #[test]
    fn test_check_alloc_capacity_normal() {
        let cfg = QwenConfig::qwen3_embedding_0_6b();
        // A realistic context length must pass.
        assert!(check_alloc_capacity(&cfg, cfg.num_hidden_layers, 4096).is_ok());
        assert!(check_alloc_capacity(&cfg, cfg.num_hidden_layers, 262_144).is_ok());
        assert!(check_alloc_capacity(&cfg, cfg.num_hidden_layers, 0).is_ok());
    }

    /// check_alloc_capacity rejects a config whose num_key_value_heads * head_dim
    /// overflows usize, even when the wrapped (modular) result would be 0 and would
    /// otherwise silently pass every subsequent checked_mul in the coeff formula.
    ///
    /// Mutation-sensitivity contract:
    ///   Revert `check_alloc_capacity` to the unchecked `let kv_dim = cfg.kv_dim()`
    ///   form: `(usize::MAX / 4 + 1) * 4` wraps to 0 in release mode, the coeff
    ///   formula reduces to `num_attention_heads`, `effective_cap * coeff` does not
    ///   overflow, and the function returns Ok — causing the `unwrap_err()` call to
    ///   panic and the test to fail. FlatKVCache::try_new (generate.rs call site) is
    ///   the second guard at the allocation boundary that catches the same class of
    ///   overflow independently; this test covers the upstream guard.
    #[test]
    fn check_alloc_capacity_rejects_kv_dim_overflow() {
        // On 64-bit: (usize::MAX / 4 + 1) * 4 overflows to 0 in release mode
        // (wrapping arithmetic). The checked_mul guard must reject this before
        // the wrapped zero reaches any downstream computation.
        let overflow_kv_heads = usize::MAX / 4 + 1;
        let cfg = QwenConfig {
            vocab_size: 1,
            hidden_size: 1,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: overflow_kv_heads,
            head_dim: 4,
            intermediate_size: 1,
            max_position_embeddings: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let err = check_alloc_capacity(&cfg, 1, 1).unwrap_err();
        assert!(
            matches!(err, InferenceError::InvalidInput(_)),
            "expected InvalidInput on kv_dim overflow, got {err:?}"
        );
    }

    /// check_alloc_capacity rejects a config whose num_attention_heads * head_dim
    /// overflows usize while kv_dim remains safe (num_key_value_heads = 1).
    /// This closes the residual query-side gap found in the PR #449 cross-family review:
    /// the existing kv_dim guard let this config through, then q_dim wrapped silently
    /// in release mode, undersizing q_buf / attn_out / qkv_buf for the prefill write.
    ///
    /// Mutation-sensitivity contract:
    ///   Remove or bypass the `q_dim = cfg.num_attention_heads.checked_mul(cfg.head_dim)…`
    ///   guard: in release mode `(usize::MAX/4 + 1) * 4` wraps to 0, the function
    ///   returns Ok, and `unwrap_err()` panics — the test fails.  In debug mode the raw
    ///   multiplication panics instead, also failing the test.  Either outcome confirms the
    ///   guard is load-bearing.
    #[test]
    fn check_alloc_capacity_rejects_q_dim_overflow() {
        // kv_dim = 1 * 4 = 4 — passes the kv_dim guard.
        // q_dim = (usize::MAX/4 + 1) * 4 — overflows; the new q_dim guard must reject it.
        let overflow_q_heads = usize::MAX / 4 + 1;
        let cfg = QwenConfig {
            vocab_size: 1,
            hidden_size: 1,
            num_hidden_layers: 1,
            num_attention_heads: overflow_q_heads,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: 1,
            max_position_embeddings: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let err = check_alloc_capacity(&cfg, 1, 1).unwrap_err();
        assert!(
            matches!(err, InferenceError::InvalidInput(_)),
            "expected InvalidInput on q_dim overflow, got {err:?}"
        );
    }

    #[test]
    fn test_check_alloc_capacity_multiplication_overflow_is_error() {
        // The codex review of PR #291 found that guarding only compute_max_seq's
        // addition leaves the downstream `max_seq_len * kv_dim` multiplication
        // unchecked: a huge-yet-non-overflowing effective_cap (here usize::MAX/1024,
        // matching the reviewer's kv_dim=8*128 counterexample) wraps the cache/scratch
        // element count and panics on the first write. The guard must reject it.
        let cfg = QwenConfig::qwen3_embedding_0_6b();
        let effective_cap = usize::MAX / 1024;
        let err = check_alloc_capacity(&cfg, cfg.num_hidden_layers, effective_cap).unwrap_err();
        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }

    #[test]
    fn test_compute_attention_single_token() {
        // 1 query token, 1 KV token, 1 head, dim=4
        let cfg = GqaConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
        };
        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        let k = vec![1.0f32, 0.0, 0.0, 0.0];
        let v = vec![0.5f32, 0.5, 0.5, 0.5];
        let mut output = vec![0.0f32; 4];
        let mut scores = vec![0.0f32; 4]; // num_heads * score_stride

        compute_attention(&mut output, &q, &k, &v, 1, 1, 0, &cfg, &mut scores, 1);

        // Single token: softmax of single score = 1.0, so output = v
        for i in 0..4 {
            assert!(
                (output[i] - 0.5).abs() < 1e-5,
                "output[{i}] = {}",
                output[i]
            );
        }
    }

    #[test]
    fn test_compute_attention_causal_mask() {
        // 2 query tokens, 2 KV tokens, 1 head, dim=2
        let cfg = GqaConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
        };
        let q = vec![1.0f32, 0.0, 0.0, 1.0]; // 2 queries
        let k = vec![1.0f32, 0.0, 0.0, 1.0]; // 2 keys
        let v = vec![1.0f32, 0.0, 0.0, 1.0]; // 2 values
        let mut output = vec![0.0f32; 4];
        let mut scores = vec![0.0f32; 4]; // 1 head * stride 2

        compute_attention(&mut output, &q, &k, &v, 2, 2, 0, &cfg, &mut scores, 2);

        // Query 0 (pos=0) can only attend to key 0 → output[0:2] = v[0] = [1,0]
        assert!((output[0] - 1.0).abs() < 1e-3, "output[0] = {}", output[0]);
        assert!((output[1] - 0.0).abs() < 1e-3, "output[1] = {}", output[1]);
        // Query 1 (pos=1) attends to both keys → weighted average
    }

    #[test]
    fn test_compute_attention_prefill_nan_score_fails_closed() {
        // Generic prefill path (q_seq_len > 1) must fail closed on a non-finite
        // score row, mirroring the q_seq_len==1 online fast path. A NaN Q
        // activation makes the softmax `sum` NaN; without the else-branch guard
        // the unnormalized NaN weights would poison the attention output.
        let cfg = GqaConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
        };
        let q = vec![f32::NAN, 0.0]; // query 0 is NaN
        let k = vec![1.0f32, 1.0];
        let v = vec![7.0f32, 11.0];
        let mut output = vec![0.0f32; 2];
        let mut scores = vec![0.0f32; 2];

        compute_attention(&mut output, &q, &k, &v, 2, 2, 0, &cfg, &mut scores, 2);

        // Query 0's row is non-finite → output must be zeroed, not NaN/inf.
        assert!(
            output[0].is_finite(),
            "prefill attention must fail closed on NaN score, got {}",
            output[0]
        );
        assert_eq!(output[0], 0.0, "failed-closed row must be zeroed");
        // Query 1 has a finite score and must still produce a finite result.
        assert!(
            output[1].is_finite(),
            "finite query must be unaffected, got {}",
            output[1]
        );
    }

    /// Reference implementation of compute_attention that never takes the fast path.
    /// Used to verify parity of the q_seq_len==1 GQA fast path.
    fn compute_attention_ref(
        output: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_seq_len: usize,
        kv_seq_len: usize,
        start_pos: usize,
        cfg: &GqaConfig,
    ) {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let groups = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores_scratch = vec![0.0f32; num_heads * kv_seq_len];
        for h in 0..num_heads {
            let kv_h = h / groups;
            let head_score_base = h * kv_seq_len;
            for qi in 0..q_seq_len {
                let q_off = qi * (num_heads * head_dim) + h * head_dim;
                let scores = &mut scores_scratch[head_score_base..head_score_base + kv_seq_len];
                let max_attend = start_pos + qi;
                for ki in 0..kv_seq_len {
                    let k_off = ki * (num_kv_heads * head_dim) + kv_h * head_dim;
                    if ki > max_attend {
                        scores[ki] = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_off + d] * k[k_off + d];
                        }
                        scores[ki] = dot * scale;
                    }
                }
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_score).exp();
                    sum += *s;
                }
                if sum > 0.0 {
                    let inv = 1.0 / sum;
                    for s in scores.iter_mut() {
                        *s *= inv;
                    }
                }
                let out_off = qi * (num_heads * head_dim) + h * head_dim;
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for ki in 0..kv_seq_len {
                        let v_off = ki * (num_kv_heads * head_dim) + kv_h * head_dim;
                        val += scores[ki] * v[v_off + d];
                    }
                    output[out_off + d] = val;
                }
            }
        }
    }

    #[test]
    fn test_compute_attention_gqa_decode_parity() {
        // Verify q_seq_len=1 GQA fast path gives same result as the reference generic path.
        // Config: 4 Q heads, 2 KV heads (groups=2), head_dim=4.
        let cfg = GqaConfig {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
        };
        let kv_seq_len = 5usize;
        let num_heads = cfg.num_heads;
        let head_dim = cfg.head_dim;

        let q: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.15 - 0.5)
            .collect();
        let k: Vec<f32> = (0..kv_seq_len * cfg.num_kv_heads * head_dim)
            .map(|i| (i as f32) * 0.1 - 0.3)
            .collect();
        let v: Vec<f32> = (0..kv_seq_len * cfg.num_kv_heads * head_dim)
            .map(|i| (i as f32) * 0.07 + 0.05)
            .collect();

        let start_pos = kv_seq_len - 1; // all KV positions are within causal window

        let mut ref_output = vec![0.0f32; num_heads * head_dim];
        compute_attention_ref(&mut ref_output, &q, &k, &v, 1, kv_seq_len, start_pos, &cfg);

        let mut fast_output = vec![0.0f32; num_heads * head_dim];
        let mut scores = vec![0.0f32; num_heads * kv_seq_len];
        compute_attention(
            &mut fast_output,
            &q,
            &k,
            &v,
            1,
            kv_seq_len,
            start_pos,
            &cfg,
            &mut scores,
            kv_seq_len,
        );

        for i in 0..fast_output.len() {
            assert!(
                (fast_output[i] - ref_output[i]).abs() < 1e-5,
                "output[{i}]: fast={} ref={}",
                fast_output[i],
                ref_output[i]
            );
        }
    }

    #[test]
    fn test_forward_scratch_capacity() {
        let cfg = QwenConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let mut scratch = ForwardScratch::new();
        scratch.ensure_capacity(&cfg, 8, 64);

        assert!(scratch.hidden.len() >= 8 * 64);
        assert!(scratch.logits.len() >= 100);
        assert!(scratch.scores.len() >= 4 * 64);

        // Re-calling ensure_capacity with smaller seq_len is a no-op
        let old_len = scratch.hidden.len();
        scratch.ensure_capacity(&cfg, 1, 64);
        assert_eq!(scratch.hidden.len(), old_len);
    }

    // -----------------------------------------------------------------------
    // Issue #12: FlatKVCache allocation scales with kv_cache_capacity cap.
    //
    // Verifies:
    //   (a) A cap of 128 allocates far fewer bytes than a cap of 4096.
    //   (b) Default (None) matches the uncapped natural size.
    //   (c) kv_cache_capacity clamp: a cap larger than max_seq is silently ignored.
    //   (d) GenerateConfig::default() has kv_cache_capacity == None.
    // -----------------------------------------------------------------------

    #[test]
    fn kv_cache_capacity_below_prompt_len_is_rejected() {
        // requested cap smaller than the prompt → Err, not an OOB prefill panic
        assert!(check_kv_cache_capacity(Some(8), 8, 32).is_err());
        // cap exactly equal to prompt_len holds the prompt → Ok
        assert!(check_kv_cache_capacity(Some(32), 32, 32).is_ok());
        // cap larger than prompt → Ok
        assert!(check_kv_cache_capacity(Some(64), 64, 16).is_ok());
        // no cap requested → never rejected (default full-size behaviour)
        assert!(check_kv_cache_capacity(None, 100, 32).is_ok());
    }

    #[test]
    fn kv_cache_capacity_allocates_fewer_bytes() {
        use crate::kv_cache::{FlatKVCache, FlatKVCacheConfig};

        // Qwen3-0.6B-like parameters (28 layers, 8 KV heads, head_dim 128).
        let num_layers = 28usize;
        let num_kv_heads = 8usize;
        let head_dim = 128usize;

        let cap_small = 128usize;
        let cap_large = 4096usize;

        let bytes_small =
            FlatKVCacheConfig::for_qwen3(num_layers, num_kv_heads, head_dim, cap_small)
                .total_bytes();
        let bytes_large =
            FlatKVCacheConfig::for_qwen3(num_layers, num_kv_heads, head_dim, cap_large)
                .total_bytes();

        // cap=128 must allocate strictly fewer bytes than cap=4096
        assert!(
            bytes_small < bytes_large,
            "cap=128 bytes ({bytes_small}) should be less than cap=4096 bytes ({bytes_large})"
        );

        // The ratio must match the cap ratio exactly (linear scaling)
        assert_eq!(
            bytes_large / bytes_small,
            cap_large / cap_small,
            "allocation should scale linearly with cap"
        );

        // Concrete numbers for the record (f16 storage, 2 bytes per element)
        // bytes_small = 2 * 28 * 128 * (8*128) * 2 = 2 * 28 * 128 * 1024 * 2 = 14,680,064 (~14 MB)
        // bytes_large = 2 * 28 * 4096 * 1024 * 2  = 469,762,048 (~448 MB)
        let mb_small = bytes_small as f64 / (1024.0 * 1024.0);
        let mb_large = bytes_large as f64 / (1024.0 * 1024.0);
        assert!(
            mb_small < 20.0,
            "cap=128 should be under 20 MB, got {mb_small:.1} MB"
        );
        assert!(
            mb_large > 400.0,
            "cap=4096 should be over 400 MB, got {mb_large:.1} MB"
        );

        // Verify the FlatKVCache actually materializes these sizes
        let cache_small = FlatKVCache::new(FlatKVCacheConfig::for_qwen3(
            num_layers,
            num_kv_heads,
            head_dim,
            cap_small,
        ));
        let cache_large = FlatKVCache::new(FlatKVCacheConfig::for_qwen3(
            num_layers,
            num_kv_heads,
            head_dim,
            cap_large,
        ));
        assert_eq!(cache_small.memory_bytes(), bytes_small);
        assert_eq!(cache_large.memory_bytes(), bytes_large);
    }

    #[test]
    fn kv_cache_capacity_default_is_none() {
        let cfg = GenerateConfig::default();
        assert!(
            cfg.kv_cache_capacity.is_none(),
            "default must be None to preserve backward-compatible allocation"
        );
    }

    #[test]
    fn kv_cache_capacity_clamp_above_max_seq() {
        // A cap larger than max_seq must not panic; it is silently clamped.
        // We exercise the clamping formula directly without calling generate().
        let prompt_len = 10usize;
        let max_new_tokens = 50usize;
        let max_seq = prompt_len + max_new_tokens; // 60

        // Simulate the clamping logic from generate()
        let effective =
            |cap: Option<usize>| -> usize { cap.map(|c| c.max(1).min(max_seq)).unwrap_or(max_seq) };

        assert_eq!(effective(None), 60, "None -> full max_seq");
        assert_eq!(
            effective(Some(100)),
            60,
            "cap > max_seq -> clamped to max_seq"
        );
        assert_eq!(effective(Some(60)), 60, "cap == max_seq -> unchanged");
        assert_eq!(effective(Some(30)), 30, "cap < max_seq -> respected");
        assert_eq!(effective(Some(0)), 1, "cap=0 -> clamped to 1");
        assert_eq!(effective(Some(1)), 1, "cap=1 -> 1 (minimum)");
    }

    /// check_alloc_capacity rejects a config where every standalone dimension
    /// product (kv_dim, q_dim, qkv_dim) fits in usize, but
    /// effective_cap * qkv_dim wraps on 64-bit targets.
    ///
    /// Config: num_key_value_heads=1, head_dim=4 → kv_dim=4 (standalone guard passes).
    ///         num_attention_heads=1000, head_dim=4 → q_dim=4000 (standalone guard passes).
    ///         qkv_dim = 4000 + 2*4 = 4008 (addition guard passes).
    ///         hidden_size=1, intermediate_size=1 (all other per-token coefficients are 1).
    ///         effective_cap chosen so that effective_cap * 4008 > usize::MAX but
    ///         the pre-existing coeff guard (effective_cap * ~1016) does not overflow,
    ///         meaning only the new effective_cap * qkv_dim compound guard catches it.
    ///
    /// Mutation-sensitivity contract:
    ///   Remove the `effective_cap.checked_mul(qkv_dim)` guard line: the function
    ///   advances past it, the remaining guards (hidden_size=1, q_dim=4000,
    ///   kv_dim=4, 2*inter=2, inter=1, coeff≈1016) all pass within usize on 64-bit,
    ///   and the function returns Ok — causing `unwrap_err()` to panic and the test
    ///   to fail. This confirms the guard is load-bearing for this defect class.
    #[test]
    fn check_alloc_capacity_rejects_seq_scaled_qkv_overflow() {
        let cfg = QwenConfig {
            vocab_size: 1,
            hidden_size: 1,
            num_hidden_layers: 1,
            num_attention_heads: 1000,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: 1,
            max_position_embeddings: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        // On 64-bit: usize::MAX / 4008 < effective_cap, so effective_cap * 4008 overflows.
        // usize::MAX / 1016 > effective_cap, so the prior coeff guard would pass.
        let effective_cap = 4_602_481_056_314_759usize;
        let err = check_alloc_capacity(&cfg, 1, effective_cap).unwrap_err();
        assert!(
            matches!(err, InferenceError::InvalidInput(_)),
            "expected InvalidInput on seq-scaled qkv overflow, got {err:?}"
        );
    }

    /// A realistic model config with a modest context cap must not be falsely
    /// rejected by the new effective_cap * DIM compound guards.
    #[test]
    fn check_alloc_capacity_accepts_realistic_scratch_dims() {
        // Qwen3-Embedding-0.6B dimensions: hidden=1024, heads=16/8, head_dim=128,
        // intermediate=3072 — representative of deployed model shapes.
        let cfg = QwenConfig::qwen3_embedding_0_6b();
        assert!(
            check_alloc_capacity(&cfg, cfg.num_hidden_layers, 4_096).is_ok(),
            "4K-token cap rejected"
        );
        assert!(
            check_alloc_capacity(&cfg, cfg.num_hidden_layers, 32_768).is_ok(),
            "32K-token cap rejected"
        );
    }
}
