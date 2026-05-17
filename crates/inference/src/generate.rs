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
use crate::kv_cache::{FlatKVCache, FlatKVCacheConfig};
use crate::model::qwen::{QwenConfig, QwenModel};
use crate::sampling::{Sampler, SamplingConfig};

/// **Unstable**: text generation configuration; fields and defaults are
/// subject to change as the generation API matures.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Sampling configuration.
    pub sampling: SamplingConfig,
    /// EOS token ID (generation stops when this is produced).
    pub eos_token_id: Option<u32>,
    /// Whether to include the prompt in the output.
    pub include_prompt: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: SamplingConfig::default(),
            eos_token_id: None,
            include_prompt: false,
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

    // 2. Initialize KV cache and scratch (allocate once per request)
    let max_seq = prompt_len + config.max_new_tokens;
    let cache_cfg = FlatKVCacheConfig::for_qwen3(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        max_seq,
    );
    let mut cache = FlatKVCache::new(cache_cfg);
    let mut scratch = ForwardScratch::new();
    // Size scratch for the largest possible call (prefill at prompt_len tokens).
    // Decode calls use seq_len=1, which is always within this capacity.
    scratch.ensure_capacity(cfg, prompt_len.max(1), max_seq);

    // 3. Initialize sampler
    let mut sampler = Sampler::new(config.sampling.clone());

    // 4. Prefill: run all prompt tokens through the model, populate KV cache.
    // logits borrows scratch; borrow ends after sampler.sample(logits) below.
    let logits = forward_with_cache(
        model,
        &prompt_ids[..prompt_len],
        &mut cache,
        0,
        &mut scratch,
        max_seq,
    )?;

    // 5. Sample first token from the last position's logits
    let mut generated_ids: Vec<u32> = Vec::with_capacity(config.max_new_tokens);
    let first_token = sampler.sample(logits);
    // logits borrow on scratch ends here (NLL: last use above)
    generated_ids.push(first_token);

    let mut stopped_by_eos = false;
    if config.eos_token_id == Some(first_token) {
        stopped_by_eos = true;
    }

    // 6. Decode loop: one token at a time
    if !stopped_by_eos {
        for step in 0..config.max_new_tokens.saturating_sub(1) {
            let pos = prompt_len + step + 1; // +1 because first token already generated
            let input = [*generated_ids
                .last()
                .expect("invariant: first generated token exists before decode loop")];
            let logits =
                forward_with_cache(model, &input, &mut cache, pos - 1, &mut scratch, max_seq)?;

            let token = sampler.sample(logits);
            // logits borrow ends here; scratch re-borrowed next iteration
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

        // Append K, V to cache for this layer
        {
            let base_pos = cache.seq_len();
            let k_layer = cache.k_buffer_mut(layer_idx);
            for i in 0..seq_len {
                let dst_off = (base_pos + i) * kv_dim;
                k_layer[dst_off..dst_off + kv_dim]
                    .copy_from_slice(&scratch.k_buf[i * kv_dim..(i + 1) * kv_dim]);
            }
            let v_layer = cache.v_buffer_mut(layer_idx);
            for i in 0..seq_len {
                let dst_off = (base_pos + i) * kv_dim;
                v_layer[dst_off..dst_off + kv_dim]
                    .copy_from_slice(&scratch.v_buf[i * kv_dim..(i + 1) * kv_dim]);
            }
        }

        // Attention: Q(seq_len) @ K(cached)^T → softmax → @ V(cached)
        let cached_seq_len = cache.seq_len() + seq_len; // not yet advanced
        let k_end = cached_seq_len * kv_dim;
        let cached_k = &cache.k_buffer(layer_idx)[..k_end];
        let cached_v = &cache.v_buffer(layer_idx)[..k_end];

        // Split scratch borrows: attn_out + scores are mutable, q_buf is immutable.
        // These are disjoint fields; Rust allows simultaneous partial borrows.
        {
            // Capture immutable slice before mutable borrows.
            let q_slice = &scratch.q_buf[..seq_len * q_dim];
            // SAFETY: q_buf, attn_out, scores are distinct Vec fields — no aliasing.
            let attn_ptr = scratch.attn_out.as_mut_ptr();
            let scores_ptr = scratch.scores.as_mut_ptr();
            let attn_len = seq_len * q_dim;
            let scores_len = scratch.scores.len();
            let attn_slice = unsafe { std::slice::from_raw_parts_mut(attn_ptr, attn_len) };
            let scores_slice = unsafe { std::slice::from_raw_parts_mut(scores_ptr, scores_len) };
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
    cache.advance_by(seq_len);

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
}
