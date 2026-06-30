//! Q8-weight forward pass for Qwen3.5-2B.
//!
//! This module mirrors `qwen35_model::forward_step` and `gated_delta_net_fused::gated_delta_net_step_fused`
//! but uses `Q8ModelWeights` (per-row symmetric INT8 quantized) for all large projection matrices.
//! Activations, norms, and recurrent state remain in `f32`.
//!
//! The purpose is to quarter memory usage for weight-bound inference while preserving acceptable
//! numerical accuracy. The i8->f32 dequantization + Accelerate BLAS path in `matmul_bt_q8`
//! provides near-AMX throughput on Apple Silicon with 4x memory savings.

use crate::attention::gdn::{GatedDeltaNetState, sigmoid, softplus};
use crate::attention::gdn_fused::{
    GatedDeltaNetFusedScratch, conv1d_silu_fused, simd_decay_and_rank1_update, simd_gated_rms_norm,
    simd_l2_normalize, simd_matvec_transpose,
};
use crate::forward::cpu::{elementwise_mul, matmul_bt, silu_inplace};
use crate::model::qwen35::Qwen35Model;
use crate::model::qwen35::{
    ForwardScratch, KvCache, decode_tokens, qwen35_rms_norm, resize, sample_token,
    should_stop_token,
};
use crate::model::qwen35_config::{GenerateConfig, GenerateOutput, Qwen35Config};
use crate::rope::RopeTable;
use crate::stop_reason::StopReason;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer;
use crate::weights::q8_weights::{
    Q8AttentionWeights, Q8CommonLayerWeights, Q8FullAttentionLayerWeights, Q8GatedDeltaNetWeights,
    Q8ModelWeights, matmul_bt_q8,
};

// ---------------------------------------------------------------------------
// Public conversion entry point
// ---------------------------------------------------------------------------

/// **Unstable**: quantize Qwen35Model weights to Q8; quantization scheme may change.
///
/// Quantize all model weights from a loaded `Qwen35Model` into Q8 representation.
///
/// The embedding table and final norm remain f32 (embedding is tied to the LM head,
/// norms are numerically sensitive). All large projection matrices are quantized
/// per-row symmetric INT8.
///
/// Returns `Err(InferenceError::UnsupportedModel)` for checkpoints that contain MoE
/// layers, which Q8 quantization does not support.
pub fn quantize_from_model(
    model: &Qwen35Model,
) -> Result<Q8ModelWeights, crate::error::InferenceError> {
    let cfg = model.config.clone();
    crate::weights::q8_weights::quantize_model_weights(&model.weights, &cfg)
}

// ---------------------------------------------------------------------------
// GatedDeltaNet step (Q8 weights)
// ---------------------------------------------------------------------------

/// **Unstable**: Q8-weight GatedDeltaNet step; kernel interface evolving with quantization strategy.
///
/// Process a single token through the GatedDeltaNet layer using Q8 weight matrices.
///
/// Numerically equivalent to `gated_delta_net_step_fused` within Q8 quantization tolerance.
/// All five large projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj) use
/// `matmul_bt_q8`. Small vectors (a_log, dt_bias, conv1d_weight, norm_weight) remain f32.
///
/// `input`: hidden state `[hidden_size]`
/// `state`: mutable recurrent state for this layer
/// `weights`: layer weights with Q8 projection matrices
/// `cfg`: model config
/// `scratch`: reusable fused scratch buffers
/// `output`: output buffer `[hidden_size]`, written in-place
#[inline]
pub fn gated_delta_net_step_fused_q8(
    input: &[f32],
    state: &mut GatedDeltaNetState,
    weights: &Q8GatedDeltaNetWeights,
    cfg: &Qwen35Config,
    scratch: &mut GatedDeltaNetFusedScratch,
    output: &mut [f32],
) {
    let hidden = cfg.hidden_size;
    let num_heads = cfg.linear_num_key_heads;
    let value_heads = cfg.linear_num_value_heads();
    let ratio = value_heads / num_heads;
    let key_dim = cfg.linear_key_head_dim;
    let value_dim = cfg.linear_value_head_dim;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let kernel_size = cfg.linear_conv_kernel_dim;

    debug_assert_eq!(
        value_heads % num_heads,
        0,
        "value_heads must be divisible by key_heads"
    );
    debug_assert!(input.len() >= hidden);
    debug_assert!(output.len() >= hidden);

    scratch.ensure_capacity(qkv_dim, output_dim, value_heads, key_dim, value_dim);

    // 1. Projections (Q8 weights)
    matmul_bt_q8(
        input,
        &weights.in_proj_qkv,
        &mut scratch.qkv_proj[..qkv_dim],
        1,
        hidden,
        qkv_dim,
    );

    matmul_bt_q8(
        input,
        &weights.in_proj_z,
        &mut scratch.z_proj[..output_dim],
        1,
        hidden,
        output_dim,
    );

    matmul_bt_q8(
        input,
        &weights.in_proj_b,
        &mut scratch.beta_proj[..value_heads],
        1,
        hidden,
        value_heads,
    );

    matmul_bt_q8(
        input,
        &weights.in_proj_a,
        &mut scratch.alpha_proj[..value_heads],
        1,
        hidden,
        value_heads,
    );

    // sigmoid(beta)
    for b in &mut scratch.beta_proj[..value_heads] {
        *b = sigmoid(*b);
    }

    // 2. Fused conv1d + SiLU (f32 conv weights)
    conv1d_silu_fused(
        &scratch.qkv_proj[..qkv_dim],
        &mut state.conv_buffer,
        &weights.conv1d_weight,
        &mut scratch.conv_output[..qkv_dim],
        qkv_dim,
        kernel_size,
    );

    // 3-7. Per-head processing
    let q_total = num_heads * key_dim;
    let k_total = num_heads * key_dim;
    let v_offset = q_total + k_total;
    let scale = 1.0 / (key_dim as f32).sqrt();

    for h in 0..value_heads {
        let k_head = h / ratio;
        let q_start = k_head * key_dim;
        let k_start = q_total + k_head * key_dim;
        let v_start = v_offset + h * value_dim;

        scratch.q_head[..key_dim].copy_from_slice(&scratch.conv_output[q_start..q_start + key_dim]);
        scratch.k_head[..key_dim].copy_from_slice(&scratch.conv_output[k_start..k_start + key_dim]);
        let v = &scratch.conv_output[v_start..v_start + value_dim];

        // L2-normalize Q and K (SIMD-accelerated)
        simd_l2_normalize(&mut scratch.q_head[..key_dim]);
        simd_l2_normalize(&mut scratch.k_head[..key_dim]);

        // Decay gate (f32 weights: a_log, dt_bias). Clamp `a_log.exp()` to finite:
        // it overflows to +inf for a_log > ~88, and `inf * softplus(very_negative)=0.0`
        // is NaN that poisons the recurrent state. Mirrors gdn_fused::compute_decay_gate
        // (#314) — the inlined Q8 copy must keep the same clamp.
        let a = weights.a_log[h].exp().min(f32::MAX);
        let sp = softplus(scratch.alpha_proj[h] + weights.dt_bias[h]);
        let g = (-a * sp).exp();

        let s_offset = h * key_dim * value_dim;
        let s = &mut state.s_matrices[s_offset..s_offset + key_dim * value_dim];

        // Retrieve: kv_mem = S^T @ k (SIMD-accelerated)
        simd_matvec_transpose(
            s,
            &scratch.k_head[..key_dim],
            &mut scratch.kv_mem[..value_dim],
            key_dim,
            value_dim,
        );

        // Delta: (v - g * kv_mem) * beta
        let beta_h = scratch.beta_proj[h];
        for ((d, &vj), &mem) in scratch.delta[..value_dim]
            .iter_mut()
            .zip(&v[..value_dim])
            .zip(&scratch.kv_mem[..value_dim])
        {
            *d = (vj - mem * g) * beta_h;
        }

        // Fused decay + rank-1 update: S = g*S + outer(k, delta) (SIMD-accelerated)
        simd_decay_and_rank1_update(
            s,
            &scratch.k_head[..key_dim],
            &scratch.delta[..value_dim],
            g,
            key_dim,
            value_dim,
        );

        // Output: o = S^T @ q / sqrt(key_dim) (SIMD-accelerated)
        let out_start = h * value_dim;
        let out_head = &mut scratch.output_heads[out_start..out_start + value_dim];
        simd_matvec_transpose(s, &scratch.q_head[..key_dim], out_head, key_dim, value_dim);
        for val in out_head.iter_mut() {
            *val *= scale;
        }
    }

    // 8. Gated RMSNorm + output projection
    let gamma = &weights.norm_weight[..value_dim];
    debug_assert_eq!(gamma.len(), value_dim);

    for h in 0..value_heads {
        let start = h * value_dim;
        let end = start + value_dim;
        simd_gated_rms_norm(
            &scratch.output_heads[start..end],
            &scratch.z_proj[start..end],
            gamma,
            &mut scratch.gated_norm_buf[start..end],
            cfg.rms_norm_eps,
        );
    }

    // Output projection (Q8 weights)
    matmul_bt_q8(
        &scratch.gated_norm_buf[..output_dim],
        &weights.out_proj,
        &mut output[..hidden],
        1,
        output_dim,
        hidden,
    );
}

// ---------------------------------------------------------------------------
// Full attention step (Q8 weights)
// ---------------------------------------------------------------------------

/// Full GQA attention for a single token using Q8 weight matrices.
///
/// Input is read from `scratch.attn_out[..hidden]`, output written back to
/// `scratch.attn_out[..hidden]`.
fn full_attention_step_q8(
    weights: &Q8FullAttentionLayerWeights,
    cache_idx: usize,
    position: usize,
    kv_cache: &mut KvCache,
    scratch: &mut ForwardScratch,
    cfg: &Qwen35Config,
    rope: &RopeTable,
    hidden: usize,
) {
    // Read input from attn_out (where caller placed it)
    let input: Vec<f32> = scratch.attn_out[..hidden].to_vec();
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();
    let head_dim = cfg.head_dim;
    let num_q_heads = cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;
    let rope_dim = cfg.rope_dim();

    // Q projection produces [Q, gate] interleaved per head:
    // view(num_heads, head_dim*2) -> chunk(2) -> Q[num_heads, head_dim], gate[num_heads, head_dim]
    let q_proj_dim = 2 * q_dim;
    let mut q_and_gate = vec![0.0f32; q_proj_dim];
    matmul_bt_q8(
        &input,
        &weights.q_proj,
        &mut q_and_gate,
        1,
        hidden,
        q_proj_dim,
    );
    // Scatter per-head: each head has [Q_h, gate_h] of size head_dim*2
    let mut gate_z = vec![0.0f32; q_dim];
    for h in 0..num_q_heads {
        let src = h * head_dim * 2;
        let dst = h * head_dim;
        scratch.q_buf[dst..dst + head_dim].copy_from_slice(&q_and_gate[src..src + head_dim]);
        gate_z[dst..dst + head_dim]
            .copy_from_slice(&q_and_gate[src + head_dim..src + head_dim * 2]);
    }
    matmul_bt_q8(
        &input,
        &weights.k_proj,
        &mut scratch.k_buf[..kv_dim],
        1,
        hidden,
        kv_dim,
    );
    matmul_bt_q8(
        &input,
        &weights.v_proj,
        &mut scratch.v_buf[..kv_dim],
        1,
        hidden,
        kv_dim,
    );

    // Per-head QK-norm (Qwen3.5 RMSNorm: 1 + gamma, f32 norms)
    for h in 0..num_q_heads {
        let start = h * head_dim;
        qwen35_rms_norm(
            &mut scratch.q_buf[start..start + head_dim],
            &weights.q_norm,
            head_dim,
            cfg.rms_norm_eps,
        );
    }
    for h in 0..num_kv_heads {
        let start = h * head_dim;
        qwen35_rms_norm(
            &mut scratch.k_buf[start..start + head_dim],
            &weights.k_norm,
            head_dim,
            cfg.rms_norm_eps,
        );
    }

    // Partial RoPE: stride-half pairing (i, half+i) — matches apply_partial_rope / HF rotate_half
    let half = rope_dim / 2;
    for h in 0..num_q_heads {
        let start = h * head_dim;
        let base = position * half;
        for i in 0..half {
            let cos_val = rope.cos_at(base + i);
            let sin_val = rope.sin_at(base + i);
            let x0 = scratch.q_buf[start + i];
            let x1 = scratch.q_buf[start + half + i];
            scratch.q_buf[start + i] = x0 * cos_val - x1 * sin_val;
            scratch.q_buf[start + half + i] = x0 * sin_val + x1 * cos_val;
        }
    }
    for h in 0..num_kv_heads {
        let start = h * head_dim;
        let base = position * half;
        for i in 0..half {
            let cos_val = rope.cos_at(base + i);
            let sin_val = rope.sin_at(base + i);
            let x0 = scratch.k_buf[start + i];
            let x1 = scratch.k_buf[start + half + i];
            scratch.k_buf[start + i] = x0 * cos_val - x1 * sin_val;
            scratch.k_buf[start + half + i] = x0 * sin_val + x1 * cos_val;
        }
    }

    // Append to KV cache
    kv_cache.append_kv(
        cache_idx,
        &scratch.k_buf[..kv_dim],
        &scratch.v_buf[..kv_dim],
    );
    let cur_seq_len = kv_cache.seq_len + 1; // including current token

    // Compute attention: for each Q head, find its KV head, compute scaled dot-product
    let groups = num_q_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let k_cache = &kv_cache.k[cache_idx];
    let v_cache = &kv_cache.v[cache_idx];

    for qh in 0..num_q_heads {
        let kvh = qh / groups;
        let q_off = qh * head_dim;
        let q = &scratch.q_buf[q_off..q_off + head_dim];

        // Compute scores against all cached K vectors
        let scores_start = qh * cur_seq_len;
        let mut max_score = f32::NEG_INFINITY;

        for t in 0..cur_seq_len {
            let k_off = t * kv_dim + kvh * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[d] * k_cache[k_off + d];
            }
            let s = dot * scale;
            scratch.scores[scores_start + t] = s;
            if s > max_score {
                max_score = s;
            }
        }

        // Softmax. Fail closed on a non-finite score row: a NaN Q/K activation
        // (e.g. from an infinite Q8 scale via `0.0 * inf`) makes `sum_exp` NaN, and
        // `NaN > 0.0` is false, so without the else-branch the unnormalized NaN
        // weights would flow into the V accumulation and poison the logits. Mirrors
        // generate.rs::compute_attention (#409) and cpu_f16 moe_ffn_step_f16 (#411).
        let mut sum_exp = 0.0f32;
        for t in 0..cur_seq_len {
            let e = (scratch.scores[scores_start + t] - max_score).exp();
            scratch.scores[scores_start + t] = e;
            sum_exp += e;
        }
        if sum_exp > 0.0 {
            let inv_sum = 1.0 / sum_exp;
            for t in 0..cur_seq_len {
                scratch.scores[scores_start + t] *= inv_sum;
            }
        } else {
            scratch.scores[scores_start..scores_start + cur_seq_len].fill(0.0);
        }

        // Weighted sum of V
        let ctx_off = qh * head_dim;
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for t in 0..cur_seq_len {
                let v_off = t * kv_dim + kvh * head_dim;
                sum += scratch.scores[scores_start + t] * v_cache[v_off + d];
            }
            scratch.context[ctx_off + d] = sum;
        }
    }

    // Output gating: attn_output *= sigmoid(gate)
    for (ctx, &gz) in scratch.context[..q_dim].iter_mut().zip(&gate_z[..q_dim]) {
        let sig = 1.0 / (1.0 + (-gz).exp());
        *ctx *= sig;
    }

    // Output projection: context [1, q_dim] @ o_proj^T [hidden, q_dim] (Q8 weights)
    matmul_bt_q8(
        &scratch.context[..q_dim],
        &weights.o_proj,
        &mut scratch.attn_out[..hidden],
        1,
        q_dim,
        hidden,
    );
}

// ---------------------------------------------------------------------------
// FFN step (Q8 weights)
// ---------------------------------------------------------------------------

/// SwiGLU FFN step using Q8 weight matrices.
///
/// Input is read from `scratch.ffn_out[..hidden]`, output written back to
/// `scratch.ffn_out[..hidden]`.
#[inline]
fn ffn_step_q8(
    common: &Q8CommonLayerWeights,
    scratch: &mut ForwardScratch,
    cfg: &Qwen35Config,
    hidden: usize,
) {
    let inter = cfg.intermediate_size;

    // Read input from ffn_out (where caller placed it)
    let input: Vec<f32> = scratch.ffn_out[..hidden].to_vec();

    // gate = gate_proj(input), up = up_proj(input) (Q8 weights)
    matmul_bt_q8(
        &input,
        &common.gate_proj,
        &mut scratch.gate_buf[..inter],
        1,
        hidden,
        inter,
    );
    matmul_bt_q8(
        &input,
        &common.up_proj,
        &mut scratch.up_buf[..inter],
        1,
        hidden,
        inter,
    );

    // SwiGLU: silu(gate) * up
    silu_inplace(&mut scratch.gate_buf[..inter]);
    elementwise_mul(&mut scratch.gate_buf[..inter], &scratch.up_buf[..inter]);

    // down_proj (Q8 weights)
    matmul_bt_q8(
        &scratch.gate_buf[..inter],
        &common.down_proj,
        &mut scratch.ffn_out[..hidden],
        1,
        inter,
        hidden,
    );
}

// ---------------------------------------------------------------------------
// Forward step (Q8 weights)
// ---------------------------------------------------------------------------

/// Single-token forward pass using Q8 weight matrices.
///
/// Equivalent to `Qwen35Model::forward_step` but all large projection matrices
/// use `matmul_bt_q8`. Norms, recurrent state, and activations remain in `f32`.
/// The embedding table remains f32 because it is tied to the LM head.
///
/// Writes logits into `scratch.logits`.
pub(crate) fn forward_step_q8(
    weights: &Q8ModelWeights,
    cfg: &Qwen35Config,
    rope: &RopeTable,
    token_id: u32,
    position: usize,
    gdn_states: &mut [GatedDeltaNetState],
    kv_cache: &mut KvCache,
    scratch: &mut ForwardScratch,
) {
    let hidden = cfg.hidden_size;

    scratch.ensure_capacity(cfg, kv_cache.seq_len + 1);

    // Embedding lookup: f32 embed_tokens -> f32 hidden
    let embed_start = token_id as usize * hidden;
    scratch.hidden[..hidden]
        .copy_from_slice(&weights.embed_tokens[embed_start..embed_start + hidden]);

    let mut linear_idx = 0usize;
    let mut full_idx = 0usize;

    for layer_i in 0..cfg.num_hidden_layers {
        let (attn_weights, common) = &weights.layers[layer_i];

        // Save residual
        scratch.residual[..hidden].copy_from_slice(&scratch.hidden[..hidden]);

        // Pre-attention RMSNorm (Qwen3.5: 1 + gamma, f32 norms)
        qwen35_rms_norm(
            &mut scratch.hidden[..hidden],
            &common.input_layernorm,
            hidden,
            cfg.rms_norm_eps,
        );

        // Attention
        match attn_weights {
            Q8AttentionWeights::Linear(gdn_w) => {
                gated_delta_net_step_fused_q8(
                    &scratch.hidden[..hidden],
                    &mut gdn_states[linear_idx],
                    gdn_w,
                    cfg,
                    &mut scratch.gdn_scratch,
                    &mut scratch.attn_out[..hidden],
                );
                linear_idx += 1;
            }
            Q8AttentionWeights::Full(full_w) => {
                // Copy hidden to attn_out as temp input to avoid borrow conflict
                scratch.attn_out[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
                full_attention_step_q8(
                    full_w,
                    cache_idx_of(full_idx),
                    position,
                    kv_cache,
                    scratch,
                    cfg,
                    rope,
                    hidden,
                );
                full_idx += 1;
            }
        }

        // Residual connection
        for i in 0..hidden {
            scratch.hidden[i] = scratch.residual[i] + scratch.attn_out[i];
        }

        // Save residual for FFN
        scratch.residual[..hidden].copy_from_slice(&scratch.hidden[..hidden]);

        // Post-attention RMSNorm (Qwen3.5: 1 + gamma, f32 norms)
        qwen35_rms_norm(
            &mut scratch.hidden[..hidden],
            &common.post_attention_layernorm,
            hidden,
            cfg.rms_norm_eps,
        );

        // SwiGLU FFN: copy hidden into ffn_out as temp input to avoid borrow conflict
        scratch.ffn_out[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
        ffn_step_q8(common, scratch, cfg, hidden);

        // Residual connection
        for i in 0..hidden {
            scratch.hidden[i] = scratch.residual[i] + scratch.ffn_out[i];
        }
    }

    // Final RMSNorm (Qwen3.5: 1 + gamma, f32 norm)
    qwen35_rms_norm(
        &mut scratch.hidden[..hidden],
        &weights.final_norm,
        hidden,
        cfg.rms_norm_eps,
    );

    // Logits: hidden @ embed_tokens^T (tied weights, f32)
    // hidden [1, hidden] @ embed_tokens^T [hidden, vocab] = logits [1, vocab]
    // embed_tokens is [vocab, hidden] in row-major f32, so matmul_bt computes
    // hidden @ embed_tokens^T correctly.
    resize(&mut scratch.logits, cfg.vocab_size);
    matmul_bt(
        &scratch.hidden[..hidden],
        &weights.embed_tokens,
        &mut scratch.logits[..cfg.vocab_size],
        1,
        hidden,
        cfg.vocab_size,
    );
}

/// Identity function for cache index -- full_idx IS the cache index.
#[inline(always)]
fn cache_idx_of(full_idx: usize) -> usize {
    full_idx
}

// ---------------------------------------------------------------------------
// Generate (Q8 weights)
// ---------------------------------------------------------------------------

/// **Unstable**: Q8-weight generate; function signature will likely merge with model struct API.
///
/// Generate text from a prompt using Q8 weight matrices.
///
/// Equivalent to `Qwen35Model::generate` but calls `forward_step_q8` for all
/// forward passes. The tokenizer, RoPE table, and generate config are passed
/// explicitly since we operate as standalone functions rather than methods on
/// the model struct.
pub fn generate_q8(
    weights: &Q8ModelWeights,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    rope: &RopeTable,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) -> Result<GenerateOutput, crate::error::InferenceError> {
    // Initialize RNG
    let mut rng_state = match gen_cfg.seed {
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
    };

    // Tokenize prompt
    let input = tokenizer.tokenize(prompt);
    let prompt_ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();
    let prompt_len = prompt_ids.len();

    if prompt_len == 0 {
        return Err(crate::error::InferenceError::Inference(
            "empty prompt".into(),
        ));
    }

    // max_new_tokens == 0 is a valid request: "score this prompt, return no tokens".
    // Guard here so the decode loop (`for _ in 1..0`) never accidentally samples one
    // token from the prefill logits before realising the budget is exhausted.
    if gen_cfg.max_new_tokens == 0 {
        return Ok(GenerateOutput {
            text: String::new(),
            token_ids: vec![],
            prompt_tokens: prompt_len,
            generated_tokens: 0,
            stopped: false,
            stop_reason: Some(StopReason::Length),
        });
    }
    // Reject grammar configs before allocating any state. Grammar masking
    // (`mask_logits` + `advance`) is not wired into this generate loop; without
    // the guard the grammar field would be silently ignored, producing
    // unconstrained output despite a grammar being set (#397/#398).
    crate::model::qwen35::check_grammar_not_set(gen_cfg)?;

    // Context preflight. The RoPE cos/sin tables are indexed unchecked in
    // full_attention_step_q8 (`rope.cos_at(position * half + i)`), so a position
    // at or past the table capacity is an out-of-bounds slice access — a release
    // panic, not a clean error. Mirror Qwen35Model::generate's total-token policy
    // (prompt_len + max_new_tokens <= max_context) so direct and HTTP generation
    // agree on when a request is too long.
    let max_context = rope.max_positions();
    if prompt_len.saturating_add(gen_cfg.max_new_tokens) > max_context {
        return Err(crate::error::InferenceError::Inference(format!(
            "prompt ({prompt_len} tokens) plus max_new_tokens ({}) exceeds \
             model context window ({max_context})",
            gen_cfg.max_new_tokens
        )));
    }

    // Initialize states
    let num_linear = cfg.num_linear_attention_layers();
    let num_full = cfg.num_full_attention_layers();
    let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
        .map(|_| GatedDeltaNetState::new(cfg))
        .collect();
    let mut kv_cache = KvCache::new(num_full);
    let mut scratch = ForwardScratch::new();

    let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
    let mut all_ids = prompt_ids.clone();

    // Prefill: process prompt tokens one at a time through the recurrence
    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        forward_step_q8(
            weights,
            cfg,
            rope,
            token_id,
            pos,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
        );
        if pos < prompt_len - 1 {
            kv_cache.seq_len += 1;
        }
    }
    kv_cache.seq_len = prompt_len;

    // Sample from last prefill logits
    let next_id = sample_token(
        &scratch.logits[..cfg.vocab_size],
        gen_cfg,
        &all_ids,
        &mut rng_state,
    );

    if should_stop_token(cfg, gen_cfg, next_id) {
        return Ok(GenerateOutput {
            text: String::new(),
            token_ids: vec![],
            prompt_tokens: prompt_len,
            generated_tokens: 0,
            stopped: true,
            stop_reason: Some(StopReason::Eos),
        });
    }

    generated_ids.push(next_id);
    all_ids.push(next_id);

    let mut stopped = false;
    let mut stop_reason = StopReason::Length;
    // Autoregressive decode
    for _ in 1..gen_cfg.max_new_tokens {
        let pos = kv_cache.seq_len;
        // all_ids is seeded by the prompt before the loop, and the decode loop
        // only continues when the previous sample pushed a new id, so the
        // invariant `all_ids.is_empty() == false` should always hold here.
        // Return an error rather than panicking so library callers can handle it.
        let Some(&last_token) = all_ids.last() else {
            return Err(crate::error::InferenceError::Inference(
                "empty generation state".into(),
            ));
        };

        forward_step_q8(
            weights,
            cfg,
            rope,
            last_token,
            pos,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
        );
        kv_cache.seq_len += 1;

        let next_id = sample_token(
            &scratch.logits[..cfg.vocab_size],
            gen_cfg,
            &all_ids,
            &mut rng_state,
        );

        if should_stop_token(cfg, gen_cfg, next_id) {
            stopped = true;
            stop_reason = StopReason::Eos;
            break;
        }

        generated_ids.push(next_id);
        all_ids.push(next_id);
    }

    // Detokenize
    let text = decode_tokens(tokenizer, &generated_ids);

    Ok(GenerateOutput {
        text,
        token_ids: generated_ids.clone(),
        prompt_tokens: prompt_len,
        generated_tokens: generated_ids.len(),
        stopped,
        stop_reason: Some(stop_reason),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for #392: cpu Q8 RoPE must use stride-half pairing (i, half+i), not
    /// interleaved (2i, 2i+1).
    ///
    /// Design: call `full_attention_step_q8` with an identity K-projection so k_buf equals
    /// the quantized input (no quantization error in W_k).  Independently reproduce the same
    /// matmul + QK-norm + stride-half RoPE in the test body and compare against the post-call
    /// KV-cache.  The two RoPE paths agree to <1e-4 when the production loops are correct;
    /// reverting either loop to 2*i interleaved produces max_diff ~0.9 (observed during
    /// mutation verification).
    #[test]
    fn test_full_attn_step_q8_rope_stride_half_parity() {
        use crate::model::qwen35_config::LayerType;
        use crate::weights::q8_weights::quantize_matrix;

        let head_dim: usize = 32;
        let num_q_heads: usize = 1;
        let num_kv_heads: usize = 1;
        let hidden: usize = 64;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let position: usize = 3;

        let cfg = Qwen35Config {
            hidden_size: hidden,
            num_hidden_layers: 2,
            vocab_size: 128,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            num_attention_heads: num_q_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.5, // rope_dim = 16, half = 8
            rope_parameters: None,
            linear_num_key_heads: 2,
            linear_num_value_heads: Some(2),
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            linear_conv_kernel_dim: 4,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            full_attention_interval: 2,
            layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
            layer_mask: vec![true; 2],
            eos_token_id: 127,
            max_position_embeddings: 512,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
        };

        let rope_dim = cfg.rope_dim(); // = 16
        let half = rope_dim / 2; // = 8
        let rope = RopeTable::new(rope_dim, 512, cfg.rope_theta);

        // W_k = identity [kv_dim, hidden]: row j selects input[j] exactly (127/127 = 1.0).
        let mut k_proj_f32 = vec![0.0f32; kv_dim * hidden];
        for j in 0..kv_dim {
            k_proj_f32[j * hidden + j] = 1.0;
        }

        let zero_q8 = |rows: usize, cols: usize| -> crate::weights::q8_weights::Q8Matrix {
            quantize_matrix(&vec![0.0f32; rows * cols], rows, cols).unwrap()
        };
        // W_q = identity for first q_dim rows (Q part), zeros for next q_dim rows (gate part).
        // Row j selects input[j] exactly so scratch.q_buf is non-trivial and Q-loop mutation
        // changes the assertion result.
        let mut q_proj_f32 = vec![0.0f32; 2 * q_dim * hidden];
        for j in 0..q_dim {
            q_proj_f32[j * hidden + j] = 1.0;
        }
        let weights = Q8FullAttentionLayerWeights {
            q_proj: quantize_matrix(&q_proj_f32, 2 * q_dim, hidden).unwrap(),
            k_proj: quantize_matrix(&k_proj_f32, kv_dim, hidden).unwrap(),
            v_proj: zero_q8(kv_dim, hidden),
            o_proj: zero_q8(hidden, q_dim),
            q_norm: vec![0.0f32; head_dim],
            k_norm: vec![0.0f32; head_dim],
        };

        // Distinct non-trivial input values (positions 0..64 scaled to small floats).
        let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.07).collect();

        let mut scratch = ForwardScratch::new();
        scratch.ensure_capacity(&cfg, 2);
        scratch.attn_out[..hidden].copy_from_slice(&input);

        let mut kv_cache = KvCache::new(1);
        full_attention_step_q8(
            &weights,
            0,
            position,
            &mut kv_cache,
            &mut scratch,
            &cfg,
            &rope,
            hidden,
        );

        // Reference: reproduce the same matmul + QK-norm + stride-half RoPE.
        // Using the same production matmul and norm functions keeps the reference
        // in step with quantization, so the only source of divergence under mutation
        // is the RoPE pairing itself.

        // --- K reference ---
        let mut k_ref = vec![0.0f32; kv_dim];
        matmul_bt_q8(&input, &weights.k_proj, &mut k_ref, 1, hidden, kv_dim);
        qwen35_rms_norm(&mut k_ref, &weights.k_norm, head_dim, cfg.rms_norm_eps);

        // Stride-half reference (correct pairing).
        let base = position * half;
        for i in 0..half {
            let cos_val = rope.cos_at(base + i);
            let sin_val = rope.sin_at(base + i);
            let x0 = k_ref[i];
            let x1 = k_ref[half + i];
            k_ref[i] = x0 * cos_val - x1 * sin_val;
            k_ref[half + i] = x0 * sin_val + x1 * cos_val;
        }

        let k_cached = &kv_cache.k[0][..kv_dim];
        let max_k_diff = k_cached
            .iter()
            .zip(k_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_k_diff < 1e-4,
            "cpu Q8 K-loop stride-half RoPE diverges from reference: max_k_diff = {max_k_diff:.6}. \
             With interleaved pairing the diff is O(0.1-1). Bug: #392."
        );

        // --- Q reference (guards the Q loop mutation) ---
        // The production scatter copies q_and_gate[0..q_dim] → scratch.q_buf[0..q_dim]
        // for head 0 (num_q_heads=1).
        let mut q_and_gate_ref = vec![0.0f32; 2 * q_dim];
        matmul_bt_q8(
            &input,
            &weights.q_proj,
            &mut q_and_gate_ref,
            1,
            hidden,
            2 * q_dim,
        );
        let mut q_ref = q_and_gate_ref[..q_dim].to_vec();
        qwen35_rms_norm(&mut q_ref, &weights.q_norm, head_dim, cfg.rms_norm_eps);

        for i in 0..half {
            let cos_val = rope.cos_at(base + i);
            let sin_val = rope.sin_at(base + i);
            let x0 = q_ref[i];
            let x1 = q_ref[half + i];
            q_ref[i] = x0 * cos_val - x1 * sin_val;
            q_ref[half + i] = x0 * sin_val + x1 * cos_val;
        }

        let max_q_diff = scratch.q_buf[..q_dim]
            .iter()
            .zip(q_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_q_diff < 1e-4,
            "cpu Q8 Q-loop stride-half RoPE diverges from reference: max_q_diff = {max_q_diff:.6}. \
             With interleaved pairing the diff is O(0.1-1). Bug: #392."
        );
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn test_q8_forward_compiles() {
        // Verify the function signatures are correct by constructing the types
        // and calling the functions with a trivial (1-layer, tiny) config.
        let cfg = Qwen35Config::qwen35_2b();

        // Verify forward_step_q8 signature (pub(crate))
        let _fn_ptr: fn(
            &Q8ModelWeights,
            &Qwen35Config,
            &RopeTable,
            u32,
            usize,
            &mut [GatedDeltaNetState],
            &mut KvCache,
            &mut ForwardScratch,
        ) = forward_step_q8;

        // Verify gated_delta_net_step_fused_q8 signature
        let _gdn_fn_ptr: fn(
            &[f32],
            &mut GatedDeltaNetState,
            &Q8GatedDeltaNetWeights,
            &Qwen35Config,
            &mut GatedDeltaNetFusedScratch,
            &mut [f32],
        ) = gated_delta_net_step_fused_q8;

        // Verify generate_q8 returns the right type
        let _gen_fn_ptr: fn(
            &Q8ModelWeights,
            &Qwen35Config,
            &BpeTokenizer,
            &RopeTable,
            &str,
            &GenerateConfig,
        ) -> Result<GenerateOutput, crate::error::InferenceError> = generate_q8;

        // Verify the config helpers work
        assert!(cfg.num_full_attention_layers() > 0);
        assert!(cfg.num_linear_attention_layers() > 0);
        assert_eq!(
            cfg.num_full_attention_layers() + cfg.num_linear_attention_layers(),
            cfg.num_hidden_layers
        );
    }

    #[test]
    fn test_gdn_q8_step_with_zeros() {
        // Run the GDN Q8 step with zero weights/inputs to verify it doesn't crash.
        let cfg = Qwen35Config::qwen35_2b();
        let hidden = cfg.hidden_size;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let num_heads = cfg.linear_num_key_heads;
        let kernel_size = cfg.linear_conv_kernel_dim;

        let make_zero_q8 = |rows: usize, cols: usize| -> crate::weights::q8_weights::Q8Matrix {
            crate::weights::q8_weights::Q8Matrix {
                data: vec![0i8; rows * cols],
                scales: vec![1.0f32; rows],
                rows,
                cols,
            }
        };

        let weights = Q8GatedDeltaNetWeights {
            in_proj_qkv: make_zero_q8(qkv_dim, hidden),
            in_proj_z: make_zero_q8(output_dim, hidden),
            in_proj_b: make_zero_q8(num_heads, hidden),
            in_proj_a: make_zero_q8(num_heads, hidden),
            a_log: vec![0.0f32; num_heads],
            dt_bias: vec![0.0f32; num_heads],
            conv1d_weight: vec![0.0f32; qkv_dim * kernel_size],
            conv_dim: qkv_dim,
            kernel_size,
            norm_weight: vec![0.0f32; cfg.linear_value_head_dim],
            out_proj: make_zero_q8(hidden, output_dim),
        };

        let mut state = GatedDeltaNetState::new(&cfg);
        let mut scratch = GatedDeltaNetFusedScratch::default();
        let input = vec![0.0f32; hidden];
        let mut output = vec![0.0f32; hidden];

        gated_delta_net_step_fused_q8(
            &input,
            &mut state,
            &weights,
            &cfg,
            &mut scratch,
            &mut output,
        );

        // With all-zero weights and input, output should be all zeros
        for &v in &output[..hidden] {
            assert_eq!(
                v, 0.0,
                "zero weights + zero input should produce zero output"
            );
        }
    }

    /// Builds the tiny single-head full-attention Q8 fixture used by the
    /// fail-closed test (mirrors `test_full_attn_step_q8_rope_stride_half_parity`).
    fn tiny_full_attn_fixture() -> (Qwen35Config, RopeTable, Q8FullAttentionLayerWeights, usize) {
        use crate::model::qwen35_config::LayerType;
        use crate::weights::q8_weights::quantize_matrix;

        let head_dim: usize = 32;
        let num_q_heads: usize = 1;
        let num_kv_heads: usize = 1;
        let hidden: usize = 64;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let cfg = Qwen35Config {
            hidden_size: hidden,
            num_hidden_layers: 2,
            vocab_size: 128,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            num_attention_heads: num_q_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.5,
            rope_parameters: None,
            linear_num_key_heads: 2,
            linear_num_value_heads: Some(2),
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            linear_conv_kernel_dim: 4,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            full_attention_interval: 2,
            layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
            layer_mask: vec![true; 2],
            eos_token_id: 127,
            max_position_embeddings: 512,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
        };

        let rope = RopeTable::new(cfg.rope_dim(), 512, cfg.rope_theta);

        let mut k_proj_f32 = vec![0.0f32; kv_dim * hidden];
        for j in 0..kv_dim {
            k_proj_f32[j * hidden + j] = 1.0;
        }
        let zero_q8 = |rows: usize, cols: usize| -> crate::weights::q8_weights::Q8Matrix {
            quantize_matrix(&vec![0.0f32; rows * cols], rows, cols).unwrap()
        };
        let mut q_proj_f32 = vec![0.0f32; 2 * q_dim * hidden];
        for j in 0..q_dim {
            q_proj_f32[j * hidden + j] = 1.0;
        }
        let weights = Q8FullAttentionLayerWeights {
            q_proj: quantize_matrix(&q_proj_f32, 2 * q_dim, hidden).unwrap(),
            k_proj: quantize_matrix(&k_proj_f32, kv_dim, hidden).unwrap(),
            v_proj: zero_q8(kv_dim, hidden),
            o_proj: zero_q8(hidden, q_dim),
            q_norm: vec![0.0f32; head_dim],
            k_norm: vec![0.0f32; head_dim],
        };
        (cfg, rope, weights, hidden)
    }

    /// A non-finite attention score row (from an infinite Q8 scale) must fail
    /// closed instead of propagating NaN into the context and logits. Mutation
    /// check: deleting the `else { fill(0.0) }` branch leaves NaN in
    /// `scratch.context`, failing the assertion.
    #[test]
    fn test_full_attn_step_q8_nan_score_fails_closed() {
        let (cfg, rope, mut weights, hidden) = tiny_full_attn_fixture();
        let q_dim = cfg.full_q_dim();

        // Corrupt Q-projection row 0: zero data * infinite scale => NaN Q lane,
        // which qwen35_rms_norm spreads across head 0, yielding a NaN score.
        for d in weights.q_proj.data[0..hidden].iter_mut() {
            *d = 0;
        }
        weights.q_proj.scales[0] = f32::INFINITY;

        let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.07).collect();
        let mut scratch = ForwardScratch::new();
        scratch.ensure_capacity(&cfg, 2);
        scratch.attn_out[..hidden].copy_from_slice(&input);
        let mut kv_cache = KvCache::new(1);

        full_attention_step_q8(
            &weights,
            0,
            3,
            &mut kv_cache,
            &mut scratch,
            &cfg,
            &rope,
            hidden,
        );

        assert!(
            scratch.context[..q_dim].iter().all(|v| v.is_finite()),
            "Q8 attention must fail closed (zero context), not propagate NaN"
        );
    }

    /// A decay-gate overflow (`a_log.exp() == +inf`) combined with a softplus
    /// underflow (`== 0.0`) yields `inf * 0.0 == NaN`, which poisons the
    /// recurrent state unless `a` is clamped to `f32::MAX`. Mutation check:
    /// removing `.min(f32::MAX)` leaves NaN in `state.s_matrices`.
    #[test]
    fn test_gdn_q8_decay_gate_overflow_fails_closed() {
        let cfg = Qwen35Config::qwen35_2b();
        let hidden = cfg.hidden_size;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let num_heads = cfg.linear_num_key_heads;
        let kernel_size = cfg.linear_conv_kernel_dim;

        let make_zero_q8 = |rows: usize, cols: usize| -> crate::weights::q8_weights::Q8Matrix {
            crate::weights::q8_weights::Q8Matrix {
                data: vec![0i8; rows * cols],
                scales: vec![1.0f32; rows],
                rows,
                cols,
            }
        };

        let mut a_log = vec![0.0f32; num_heads];
        let mut dt_bias = vec![0.0f32; num_heads];
        a_log[0] = 100.0; // exp(100) overflows to +inf
        dt_bias[0] = -100.0; // softplus(-100) underflows to 0.0

        let weights = Q8GatedDeltaNetWeights {
            in_proj_qkv: make_zero_q8(qkv_dim, hidden),
            in_proj_z: make_zero_q8(output_dim, hidden),
            in_proj_b: make_zero_q8(num_heads, hidden),
            in_proj_a: make_zero_q8(num_heads, hidden),
            a_log,
            dt_bias,
            conv1d_weight: vec![0.0f32; qkv_dim * kernel_size],
            conv_dim: qkv_dim,
            kernel_size,
            norm_weight: vec![0.0f32; cfg.linear_value_head_dim],
            out_proj: make_zero_q8(hidden, output_dim),
        };

        let mut state = GatedDeltaNetState::new(&cfg);
        let mut scratch = GatedDeltaNetFusedScratch::default();
        let input = vec![0.0f32; hidden];
        let mut output = vec![0.0f32; hidden];

        gated_delta_net_step_fused_q8(
            &input,
            &mut state,
            &weights,
            &cfg,
            &mut scratch,
            &mut output,
        );

        assert!(
            state.s_matrices.iter().all(|v| v.is_finite()),
            "GDN recurrent state must stay finite when the decay gate overflows"
        );
        assert!(
            output[..hidden].iter().all(|v| v.is_finite()),
            "GDN output must stay finite when the decay gate overflows"
        );
    }

    /// Build a minimal Q8 config + weights with zero hidden layers so forward_step_q8
    /// skips all attention and MLP blocks. The embedding lookup and final-norm steps
    /// still execute, so embed_tokens and final_norm must have the right lengths.
    fn zero_layer_q8_fixture() -> (Qwen35Config, Q8ModelWeights, RopeTable, BpeTokenizer) {
        use std::collections::HashMap;

        let hidden = 4usize;
        let vocab = 8usize;

        let cfg = Qwen35Config {
            hidden_size: hidden,
            num_hidden_layers: 0,
            vocab_size: vocab,
            intermediate_size: 4,
            rms_norm_eps: 1e-6,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.5,
            rope_parameters: None,
            linear_num_key_heads: 1,
            linear_num_value_heads: Some(1),
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            full_attention_interval: 2,
            layer_types: vec![],
            layer_mask: vec![],
            // eos is 5 so that greedy token 0 is NOT eos — this lets tests for
            // stop_token_ids use token 0 as a distinct stop signal.
            eos_token_id: 5,
            max_position_embeddings: 512,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
        };

        // embed_tokens is [vocab, hidden] and also serves as the tied LM head.
        // All zeros → logits are all zeros → greedy sampling always picks token 0.
        let weights = Q8ModelWeights {
            embed_tokens: vec![0.0f32; vocab * hidden],
            final_norm: vec![0.0f32; hidden],
            layers: vec![],
        };

        // rope_dim = head_dim * partial_rotary_factor = 4 * 0.5 = 2.
        // No full-attention layers use the table, but max_positions must satisfy the
        // context preflight (prompt_len + max_new_tokens <= max_positions).
        let rope = RopeTable::new(2, 64, 10_000.0);

        let mut vocab_map: HashMap<String, u32> = HashMap::new();
        for (i, c) in ["h", "e", "l", "o", "w", "r", "d", "!"].iter().enumerate() {
            vocab_map.insert((*c).to_string(), i as u32);
        }
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("he".to_string(), "l".to_string()),
        ];
        let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab_map, merges).unwrap();

        (cfg, weights, rope, tokenizer)
    }

    /// `generate_q8` with `max_new_tokens == 0` must return zero generated tokens
    /// without running a forward pass or sampling anything.
    ///
    /// Mutation check: removing the `max_new_tokens == 0` early return causes
    /// the function to run prefill + sample one token from the logits, so
    /// `generated_tokens` becomes 1 instead of 0 and the assertion fails.
    #[test]
    fn test_generate_q8_max_new_tokens_zero_returns_empty() {
        let (cfg, weights, rope, tokenizer) = zero_layer_q8_fixture();

        let gen_cfg = GenerateConfig {
            max_new_tokens: 0,
            ..Default::default()
        };

        let out = generate_q8(&weights, &cfg, &tokenizer, &rope, "h", &gen_cfg)
            .expect("max_new_tokens=0 must succeed, not error");

        assert_eq!(
            out.generated_tokens, 0,
            "max_new_tokens=0 must produce zero generated tokens"
        );
        assert!(
            out.token_ids.is_empty(),
            "max_new_tokens=0 must produce an empty token list"
        );
        assert_eq!(out.prompt_tokens, 1, "prompt 'h' tokenizes to one token");
    }

    /// `generate_q8` must stop on a token in `stop_token_ids` even when that
    /// token differs from `eos_token_id`.
    ///
    /// Setup: all-zero weights → greedy sampling always picks token 0.
    /// Config has eos_token_id=5 (not 0) and stop_token_ids=[0].
    /// With the fix the first sampled token (0) hits the stop list and the
    /// function returns 0 generated tokens.
    ///
    /// Mutation check: reverting `should_stop_token` back to
    /// `next_id == cfg.eos_token_id` causes `0 == 5` to be false, so token 0
    /// is pushed to output and `generated_tokens` becomes ≥ 1.
    #[test]
    fn test_generate_q8_honors_stop_token_ids() {
        let (cfg, weights, rope, tokenizer) = zero_layer_q8_fixture();

        let gen_cfg = GenerateConfig {
            max_new_tokens: 4,
            stop_token_ids: vec![0], // token 0 is the stop signal, NOT eos (5)
            temperature: 0.0,        // greedy: all-zero logits always yield token 0
            ..Default::default()
        };

        let out = generate_q8(&weights, &cfg, &tokenizer, &rope, "h", &gen_cfg)
            .expect("generate_q8 must succeed with valid stop_token_ids");

        assert_eq!(
            out.generated_tokens, 0,
            "stop token 0 must halt generation before any token is emitted"
        );
        assert!(
            out.stopped,
            "stopped flag must be true when a stop token fires"
        );
    }

    /// `generate_q8` must reject a request whose prompt + max_new_tokens exceeds
    /// the RoPE table capacity with a clean error, not an out-of-bounds RoPE
    /// panic. The preflight returns before any weight is read, so an empty
    /// model is sufficient. Mutation check: removing the preflight makes this
    /// `expect_err` fail (the call would panic or run instead).
    #[test]
    fn test_generate_q8_rejects_context_overflow() {
        use std::collections::HashMap;

        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
            vocab.insert((*c).to_string(), i as u32);
        }
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("he".to_string(), "l".to_string()),
        ];
        let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

        let cfg = Qwen35Config::qwen35_2b();
        let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
        let weights = Q8ModelWeights {
            embed_tokens: vec![],
            final_norm: vec![],
            layers: vec![],
        };
        let gen_cfg = GenerateConfig {
            max_new_tokens: usize::MAX,
            ..Default::default()
        };

        let err = generate_q8(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg)
            .expect_err("request beyond context window must error, not panic");
        let msg = format!("{err}");
        assert!(
            msg.contains("context window"),
            "error must name the context window; got: {msg}"
        );
    }

    /// `generate_q8` must reject a `GenerateConfig` that sets `grammar` with a
    /// typed `InvalidInput` error before sampling any token (#397/#398).
    ///
    /// Before the fix, grammar was silently ignored and unconstrained output was
    /// produced. The guard now fires before any weight dereference or state
    /// allocation, so empty weight vecs are sufficient.
    ///
    /// Mutation sensitivity: removing the `check_grammar_not_set` call makes the
    /// function proceed past the guard and attempt to forward with empty weights,
    /// producing a panic or a non-`InvalidInput` error — this assert fails either way.
    #[test]
    fn generate_q8_rejects_grammar_config_before_sampling() {
        use crate::error::InferenceError;
        use crate::grammar::{GrammarEngine, GrammarSpec};
        use std::collections::HashMap;
        use std::sync::Arc;

        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
            vocab.insert((*c).to_string(), i as u32);
        }
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("he".to_string(), "l".to_string()),
        ];
        let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

        let cfg = Qwen35Config::qwen35_2b();
        let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
        let weights = Q8ModelWeights {
            embed_tokens: vec![],
            final_norm: vec![],
            layers: vec![],
        };

        let spec = GrammarSpec::Gbnf("root ::= \"t\" | \"f\"\n".to_string());
        let grammar_vocab = vec![b"t".to_vec(), b"f".to_vec()];
        let engine =
            GrammarEngine::new(&spec, grammar_vocab).expect("trivial grammar must compile");

        let gen_cfg = GenerateConfig {
            grammar: Some(Arc::new(engine)),
            ..Default::default()
        };

        let result = generate_q8(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg);
        assert!(
            matches!(result, Err(InferenceError::InvalidInput(_))),
            "generate_q8 must fail closed with InvalidInput when grammar is set (#397/#398); \
             got {result:?}"
        );
    }
}
