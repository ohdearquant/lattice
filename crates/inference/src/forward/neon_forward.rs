//! Q8 NEON forward pass for Qwen3.5-2B CPU inference.
//!
//! Unlike `q8_forward.rs` (which dequantizes to f32 then calls Accelerate BLAS),
//! this module uses the native NEON int8 matvec kernel from `q8_neon.rs` that
//! operates directly on Q8_0 packed weights without f32 conversion.
//!
//! Q8_0 format: blocks of 32 weights stored as `[f32_scale, 32x i8]` (36 bytes).
//! The NEON kernel uses `vmull_s8 + vpadalq_s16` to compute int8 dot products
//! without ever expanding to f32, avoiding the CPU format conversion trap.
//!
//! Architecture:
//! - Embedding lookup and norms: kept in f32 (small, element-wise)
//! - All large projection matmuls: Q8_0 NEON (`matmul_q8_neon`)
//! - GDN recurrence and GQA attention: f32 (operate on projected vectors)
//! - Logits projection (lm_head): Q8_0 NEON (vocab x hidden is the largest single matmul)

use crate::attention::gdn::{
    GatedDeltaNetState, GatedDeltaNetWeights, gated_rms_norm, l2_normalize_vec, sigmoid, softplus,
};
use crate::attention::gdn_fused::GatedDeltaNetFusedScratch;
use crate::forward::cpu::{elementwise_mul, silu_inplace};
use crate::forward::neon::{matmul_q8_neon_into, pack_weights_q8};
use crate::model::qwen35::{
    AttentionWeights, CommonLayerWeights, FeedForwardWeights, ForwardScratch,
    FullAttentionLayerWeights, KvCache, ModelWeights, decode_tokens, qwen35_rms_norm, resize,
    sample_token,
};
use crate::model::qwen35_config::{GenerateConfig, GenerateOutput, Qwen35Config};
use crate::rope::RopeTable;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer;

// -----------------------------------------------------------------------
// Q8 NEON weight structures
// -----------------------------------------------------------------------

/// **Unstable**: Q8_0 weights for a GatedDeltaNet layer; field set mirrors float layout and may change.
///
/// Q8_0-packed weights for a GatedDeltaNet (linear attention) layer.
///
/// Only the five large projection matrices are packed. Small vectors
/// (a_log, dt_bias, conv1d, norm) stay f32 since they are tiny and
/// numerically sensitive.
pub struct Q8NeonGdnWeights {
    /// QKV projection `[qkv_dim, hidden]` in Q8_0 format.
    pub in_proj_qkv_packed: Vec<u8>,
    pub in_proj_qkv_rows: usize,
    pub in_proj_qkv_cols: usize,

    /// Output gate projection `[output_dim, hidden]` in Q8_0 format.
    pub in_proj_z_packed: Vec<u8>,
    pub in_proj_z_rows: usize,
    pub in_proj_z_cols: usize,

    /// Update rate projection `[num_heads, hidden]` in Q8_0 format.
    pub in_proj_b_packed: Vec<u8>,
    pub in_proj_b_rows: usize,
    pub in_proj_b_cols: usize,

    /// Decay input projection `[num_heads, hidden]` in Q8_0 format.
    pub in_proj_a_packed: Vec<u8>,
    pub in_proj_a_rows: usize,
    pub in_proj_a_cols: usize,

    /// Output projection `[hidden, output_dim]` in Q8_0 format.
    pub out_proj_packed: Vec<u8>,
    pub out_proj_rows: usize,
    pub out_proj_cols: usize,

    // --- Small f32 weights (not quantized) ---
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub conv1d_weight: Vec<f32>,
    pub conv_dim: usize,
    pub kernel_size: usize,
    pub norm_weight: Vec<f32>,
}

/// **Unstable**: Q8_0 weights for a full GQA attention layer; field layout may change.
///
/// Q8_0-packed weights for a full-attention (GQA) layer.
pub struct Q8NeonFullAttnWeights {
    /// Q+gate projection `[2*q_dim, hidden]` in Q8_0 format.
    pub q_proj_packed: Vec<u8>,
    pub q_proj_rows: usize,
    pub q_proj_cols: usize,

    /// K projection `[kv_dim, hidden]` in Q8_0 format.
    pub k_proj_packed: Vec<u8>,
    pub k_proj_rows: usize,
    pub k_proj_cols: usize,

    /// V projection `[kv_dim, hidden]` in Q8_0 format.
    pub v_proj_packed: Vec<u8>,
    pub v_proj_rows: usize,
    pub v_proj_cols: usize,

    /// Output projection `[hidden, q_dim]` in Q8_0 format.
    pub o_proj_packed: Vec<u8>,
    pub o_proj_rows: usize,
    pub o_proj_cols: usize,

    // --- Small f32 weights (not quantized) ---
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// **Unstable**: Q8_0 common layer weights for MLP; field layout mirrors float weights.
///
/// Q8_0-packed common layer weights (MLP norms + projections).
pub struct Q8NeonCommonWeights {
    // Norms stay f32 (element-wise, tiny)
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,

    /// Gate projection `[intermediate, hidden]` in Q8_0 format.
    pub gate_proj_packed: Vec<u8>,
    pub gate_proj_rows: usize,
    pub gate_proj_cols: usize,

    /// Up projection `[intermediate, hidden]` in Q8_0 format.
    pub up_proj_packed: Vec<u8>,
    pub up_proj_rows: usize,
    pub up_proj_cols: usize,

    /// Down projection `[hidden, intermediate]` in Q8_0 format.
    pub down_proj_packed: Vec<u8>,
    pub down_proj_rows: usize,
    pub down_proj_cols: usize,
}

/// **Unstable**: per-layer attention weight storage for Q8_0 NEON; variant set tied to hybrid architecture.
///
/// Per-layer attention weight storage (Q8_0 NEON format).
pub enum Q8NeonAttentionWeights {
    /// GatedDeltaNet weights with Q8_0 projections.
    Linear(Q8NeonGdnWeights),
    /// Full GQA attention weights with Q8_0 projections.
    Full(Q8NeonFullAttnWeights),
}

/// **Unstable**: full model weights in Q8_0 NEON format; lm_head packing and layer layout may change.
///
/// All model weights in Q8_0 NEON format.
pub struct Q8NeonModel {
    /// Embedding table, kept in f32 (lookup, not matmul).
    pub embed_tokens: Vec<f32>,
    /// Final RMSNorm weights, kept in f32.
    pub final_norm: Vec<f32>,
    /// LM head `[vocab, hidden]` in Q8_0 format.
    /// Separate from embed_tokens because the NEON kernel needs packed format,
    /// while embed_tokens is used for lookup by index.
    pub lm_head_packed: Vec<u8>,
    pub lm_head_rows: usize,
    pub lm_head_cols: usize,
    /// Per-layer weights.
    pub layers: Vec<(Q8NeonAttentionWeights, Q8NeonCommonWeights)>,
}

// -----------------------------------------------------------------------
// Quantization: f32 ModelWeights -> Q8NeonModel
// -----------------------------------------------------------------------

/// Convert GDN weights to Q8_0 NEON packed format.
fn pack_gdn_weights(w: &GatedDeltaNetWeights) -> Q8NeonGdnWeights {
    Q8NeonGdnWeights {
        in_proj_qkv_packed: pack_weights_q8(&w.in_proj_qkv, w.in_proj_qkv_rows, w.in_proj_qkv_cols),
        in_proj_qkv_rows: w.in_proj_qkv_rows,
        in_proj_qkv_cols: w.in_proj_qkv_cols,

        in_proj_z_packed: pack_weights_q8(&w.in_proj_z, w.in_proj_z_rows, w.in_proj_z_cols),
        in_proj_z_rows: w.in_proj_z_rows,
        in_proj_z_cols: w.in_proj_z_cols,

        in_proj_b_packed: pack_weights_q8(&w.in_proj_b, w.in_proj_b_rows, w.in_proj_b_cols),
        in_proj_b_rows: w.in_proj_b_rows,
        in_proj_b_cols: w.in_proj_b_cols,

        in_proj_a_packed: pack_weights_q8(&w.in_proj_a, w.in_proj_a_rows, w.in_proj_a_cols),
        in_proj_a_rows: w.in_proj_a_rows,
        in_proj_a_cols: w.in_proj_a_cols,

        out_proj_packed: pack_weights_q8(&w.out_proj, w.out_proj_rows, w.out_proj_cols),
        out_proj_rows: w.out_proj_rows,
        out_proj_cols: w.out_proj_cols,

        a_log: w.a_log.clone(),
        dt_bias: w.dt_bias.clone(),
        conv1d_weight: w.conv1d_weight.clone(),
        conv_dim: w.conv_dim,
        kernel_size: w.kernel_size,
        norm_weight: w.norm_weight.clone(),
    }
}

/// Convert full-attention weights to Q8_0 NEON packed format.
fn pack_full_attn_weights(
    w: &FullAttentionLayerWeights,
    cfg: &Qwen35Config,
) -> Q8NeonFullAttnWeights {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();
    let q_proj_rows = 2 * q_dim; // Q + gate interleaved

    Q8NeonFullAttnWeights {
        q_proj_packed: pack_weights_q8(&w.q_proj, q_proj_rows, hidden),
        q_proj_rows,
        q_proj_cols: hidden,

        k_proj_packed: pack_weights_q8(&w.k_proj, kv_dim, hidden),
        k_proj_rows: kv_dim,
        k_proj_cols: hidden,

        v_proj_packed: pack_weights_q8(&w.v_proj, kv_dim, hidden),
        v_proj_rows: kv_dim,
        v_proj_cols: hidden,

        o_proj_packed: pack_weights_q8(&w.o_proj, hidden, q_dim),
        o_proj_rows: hidden,
        o_proj_cols: q_dim,

        q_norm: w.q_norm.clone(),
        k_norm: w.k_norm.clone(),
    }
}

/// Convert common layer weights (MLP) to Q8_0 NEON packed format.
fn pack_common_weights(w: &CommonLayerWeights, cfg: &Qwen35Config) -> Q8NeonCommonWeights {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;

    let (gate_proj, up_proj, down_proj) = match &w.ffn {
        FeedForwardWeights::Dense(dense) => (&dense.gate_proj, &dense.up_proj, &dense.down_proj),
        FeedForwardWeights::Moe(_) => {
            panic!("Q8 NEON packing is dense-only; MoE configs are not supported");
        }
    };

    Q8NeonCommonWeights {
        input_layernorm: w.input_layernorm.clone(),
        post_attention_layernorm: w.post_attention_layernorm.clone(),

        gate_proj_packed: pack_weights_q8(gate_proj, inter, hidden),
        gate_proj_rows: inter,
        gate_proj_cols: hidden,

        up_proj_packed: pack_weights_q8(up_proj, inter, hidden),
        up_proj_rows: inter,
        up_proj_cols: hidden,

        down_proj_packed: pack_weights_q8(down_proj, hidden, inter),
        down_proj_rows: hidden,
        down_proj_cols: inter,
    }
}

/// **Unstable**: quantize model weights to Q8_0 NEON format; packing strategy may change.
///
/// Quantize all model weights from f32 `ModelWeights` into Q8_0 NEON packed format.
///
/// The embedding table and norms remain f32 (embedding is used for token lookup,
/// norms are element-wise and numerically sensitive). All large projection matrices
/// are packed into Q8_0 format for native NEON int8 inference.
///
/// The lm_head (logits projection) is packed separately from the embedding table:
/// embed_tokens stays f32 for lookup, lm_head gets Q8_0 for the final matmul.
pub fn quantize_model(weights: &ModelWeights, cfg: &Qwen35Config) -> Q8NeonModel {
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;

    let layers = weights
        .layers
        .iter()
        .map(|(attn, common)| {
            let q8_attn = match attn {
                AttentionWeights::Linear(gdn_w) => {
                    Q8NeonAttentionWeights::Linear(pack_gdn_weights(gdn_w))
                }
                AttentionWeights::Full(full_w) => {
                    Q8NeonAttentionWeights::Full(pack_full_attn_weights(full_w, cfg))
                }
            };
            let q8_common = pack_common_weights(common, cfg);
            (q8_attn, q8_common)
        })
        .collect();

    // Pack the actual output projection; for Qwen3.6 this is untied lm_head.weight.
    let lm_head_packed = pack_weights_q8(weights.logits_weight(), vocab, hidden);

    Q8NeonModel {
        embed_tokens: weights.embed_tokens.clone(),
        final_norm: weights.final_norm.clone(),
        lm_head_packed,
        lm_head_rows: vocab,
        lm_head_cols: hidden,
        layers,
    }
}

// -----------------------------------------------------------------------
// GatedDeltaNet step (Q8 NEON projections, f32 recurrence)
// -----------------------------------------------------------------------

/// Process a single token through a GatedDeltaNet layer using Q8_0 NEON projections.
///
/// All five projections and all per-head temporaries write into `gdn_scratch`
/// and `x_q_scratch`; no heap allocations occur after warmup.
fn gdn_step_q8_neon(
    input: &[f32],
    state: &mut GatedDeltaNetState,
    weights: &Q8NeonGdnWeights,
    cfg: &Qwen35Config,
    gdn_scratch: &mut GatedDeltaNetFusedScratch,
    x_q_scratch: &mut Vec<i8>,
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

    gdn_scratch.ensure_capacity(qkv_dim, output_dim, num_heads, key_dim, value_dim);

    // 1. Projections (Q8 NEON) — zero allocations after warmup
    matmul_q8_neon_into(
        input,
        &weights.in_proj_qkv_packed,
        qkv_dim,
        hidden,
        &mut gdn_scratch.qkv_proj[..qkv_dim],
        x_q_scratch,
    );
    matmul_q8_neon_into(
        input,
        &weights.in_proj_z_packed,
        output_dim,
        hidden,
        &mut gdn_scratch.z_proj[..output_dim],
        x_q_scratch,
    );
    matmul_q8_neon_into(
        input,
        &weights.in_proj_b_packed,
        num_heads,
        hidden,
        &mut gdn_scratch.beta_proj[..num_heads],
        x_q_scratch,
    );
    matmul_q8_neon_into(
        input,
        &weights.in_proj_a_packed,
        num_heads,
        hidden,
        &mut gdn_scratch.alpha_proj[..num_heads],
        x_q_scratch,
    );

    // sigmoid(beta) in place
    for b in &mut gdn_scratch.beta_proj[..num_heads] {
        *b = sigmoid(*b);
    }

    // 2. Causal depthwise conv1d + SiLU
    let conv_dim = weights.conv_dim;
    let buf_len = kernel_size - 1;
    for ch in 0..conv_dim {
        let qkv_ch = gdn_scratch.qkv_proj[ch];
        let buf_start = ch * buf_len;
        for j in 0..buf_len.saturating_sub(1) {
            state.conv_buffer[buf_start + j] = state.conv_buffer[buf_start + j + 1];
        }
        if buf_len > 0 {
            state.conv_buffer[buf_start + buf_len - 1] = qkv_ch;
        }

        let mut acc =
            gdn_scratch.qkv_proj[ch] * weights.conv1d_weight[ch * kernel_size + kernel_size - 1];
        for k in 0..buf_len {
            acc += state.conv_buffer[buf_start + k] * weights.conv1d_weight[ch * kernel_size + k];
        }

        // SiLU
        let sig = 1.0 / (1.0 + (-acc).exp());
        gdn_scratch.conv_output[ch] = acc * sig;
    }

    // 3-7. Per-head recurrence — no local Vec allocations
    let q_total = num_heads * key_dim;
    let k_total = num_heads * key_dim;
    let v_offset = q_total + k_total;
    let scale = 1.0 / (key_dim as f32).sqrt();

    for h in 0..value_heads {
        let k_head = h / ratio;
        let q_start = k_head * key_dim;
        let k_start = q_total + k_head * key_dim;
        let v_start = v_offset + h * value_dim;

        gdn_scratch.q_head[..key_dim]
            .copy_from_slice(&gdn_scratch.conv_output[q_start..q_start + key_dim]);
        gdn_scratch.k_head[..key_dim]
            .copy_from_slice(&gdn_scratch.conv_output[k_start..k_start + key_dim]);

        l2_normalize_vec(&mut gdn_scratch.q_head[..key_dim]);
        l2_normalize_vec(&mut gdn_scratch.k_head[..key_dim]);

        // Decay gate
        let a_val = weights.a_log[k_head].exp();
        let sp = softplus(gdn_scratch.alpha_proj[k_head] + weights.dt_bias[k_head]);
        let g = (-a_val * sp).exp();

        let s_off = h * key_dim * value_dim;
        let s = &mut state.s_matrices[s_off..s_off + key_dim * value_dim];

        // Retrieve: kv_mem = S^T @ k
        for j in 0..value_dim {
            let mut dot = 0.0f32;
            for i in 0..key_dim {
                dot += s[i * value_dim + j] * gdn_scratch.k_head[i];
            }
            gdn_scratch.kv_mem[j] = dot;
        }

        // Delta: (v - g * kv_mem) * beta
        let beta_h = gdn_scratch.beta_proj[k_head];
        for j in 0..value_dim {
            let v_j = gdn_scratch.conv_output[v_start + j];
            gdn_scratch.delta[j] = (v_j - gdn_scratch.kv_mem[j] * g) * beta_h;
        }

        // Update: S = g*S + outer(k, delta)
        for i in 0..key_dim {
            for j in 0..value_dim {
                s[i * value_dim + j] =
                    g * s[i * value_dim + j] + gdn_scratch.k_head[i] * gdn_scratch.delta[j];
            }
        }

        // Output: o = S^T @ q / sqrt(key_dim)
        let out_start = h * value_dim;
        for j in 0..value_dim {
            let mut dot = 0.0f32;
            for i in 0..key_dim {
                dot += s[i * value_dim + j] * gdn_scratch.q_head[i];
            }
            gdn_scratch.output_heads[out_start + j] = dot * scale;
        }
    }

    // 8. Gated RMSNorm
    let gamma = &weights.norm_weight[..value_dim];
    for h in 0..value_heads {
        let start = h * value_dim;
        let end = start + value_dim;
        gated_rms_norm(
            &gdn_scratch.output_heads[start..end],
            &gdn_scratch.z_proj[start..end],
            gamma,
            &mut gdn_scratch.gated_norm_buf[start..end],
            cfg.rms_norm_eps,
        );
    }

    // 9. Output projection (Q8 NEON) — write directly into caller output
    matmul_q8_neon_into(
        &gdn_scratch.gated_norm_buf[..output_dim],
        &weights.out_proj_packed,
        hidden,
        output_dim,
        &mut output[..hidden],
        x_q_scratch,
    );
}

// -----------------------------------------------------------------------
// Full attention step (Q8 NEON projections, f32 attention)
// -----------------------------------------------------------------------

/// Full GQA attention for a single token using Q8_0 NEON projections.
///
/// Input is read from `scratch.attn_out[..hidden]`, output written back
/// to `scratch.attn_out[..hidden]`.
fn full_attention_step_q8_neon(
    weights: &Q8NeonFullAttnWeights,
    cache_idx: usize,
    position: usize,
    kv_cache: &mut KvCache,
    scratch: &mut ForwardScratch,
    cfg: &Qwen35Config,
    rope: &RopeTable,
    hidden: usize,
) {
    scratch.input_tmp[..hidden].copy_from_slice(&scratch.attn_out[..hidden]);
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();
    let head_dim = cfg.head_dim;
    let num_q_heads = cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;
    let rope_dim = cfg.rope_dim();

    // Q projection produces [Q, gate] interleaved per head
    let q_proj_dim = 2 * q_dim;
    matmul_q8_neon_into(
        &scratch.input_tmp[..hidden],
        &weights.q_proj_packed,
        q_proj_dim,
        hidden,
        &mut scratch.q_and_gate[..q_proj_dim],
        &mut scratch.x_q_scratch,
    );

    // Scatter per-head: each head has [Q_h, gate_h] of size head_dim*2
    for h in 0..num_q_heads {
        let src = h * head_dim * 2;
        let dst = h * head_dim;
        scratch.q_buf[dst..dst + head_dim]
            .copy_from_slice(&scratch.q_and_gate[src..src + head_dim]);
        scratch.gate_z[dst..dst + head_dim]
            .copy_from_slice(&scratch.q_and_gate[src + head_dim..src + head_dim * 2]);
    }

    // K and V projections — write directly into scratch buffers
    matmul_q8_neon_into(
        &scratch.input_tmp[..hidden],
        &weights.k_proj_packed,
        kv_dim,
        hidden,
        &mut scratch.k_buf[..kv_dim],
        &mut scratch.x_q_scratch,
    );
    matmul_q8_neon_into(
        &scratch.input_tmp[..hidden],
        &weights.v_proj_packed,
        kv_dim,
        hidden,
        &mut scratch.v_buf[..kv_dim],
        &mut scratch.x_q_scratch,
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

    // Partial RoPE: interleaved pairing, first rope_dim dimensions
    let half = rope_dim / 2;
    for h in 0..num_q_heads {
        let start = h * head_dim;
        let base = position * half;
        for i in 0..half {
            let cos_val = rope.cos_at(base + i);
            let sin_val = rope.sin_at(base + i);
            let x0 = scratch.q_buf[start + 2 * i];
            let x1 = scratch.q_buf[start + 2 * i + 1];
            scratch.q_buf[start + 2 * i] = x0 * cos_val - x1 * sin_val;
            scratch.q_buf[start + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
    for h in 0..num_kv_heads {
        let start = h * head_dim;
        let base = position * half;
        for i in 0..half {
            let cos_val = rope.cos_at(base + i);
            let sin_val = rope.sin_at(base + i);
            let x0 = scratch.k_buf[start + 2 * i];
            let x1 = scratch.k_buf[start + 2 * i + 1];
            scratch.k_buf[start + 2 * i] = x0 * cos_val - x1 * sin_val;
            scratch.k_buf[start + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }

    // Append to KV cache
    kv_cache.append_kv(
        cache_idx,
        &scratch.k_buf[..kv_dim],
        &scratch.v_buf[..kv_dim],
    );
    let cur_seq_len = kv_cache.seq_len + 1;

    // Compute attention scores and weighted sum (f32)
    let groups = num_q_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let k_cache = &kv_cache.k[cache_idx];
    let v_cache = &kv_cache.v[cache_idx];

    for qh in 0..num_q_heads {
        let kvh = qh / groups;
        let q_off = qh * head_dim;
        let q = &scratch.q_buf[q_off..q_off + head_dim];

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

        // Softmax
        let mut sum_exp = 0.0f32;
        for t in 0..cur_seq_len {
            let e = (scratch.scores[scores_start + t] - max_score).exp();
            scratch.scores[scores_start + t] = e;
            sum_exp += e;
        }
        let inv_sum = 1.0 / sum_exp;
        for t in 0..cur_seq_len {
            scratch.scores[scores_start + t] *= inv_sum;
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
    for d in 0..q_dim {
        let sig = 1.0 / (1.0 + (-scratch.gate_z[d]).exp());
        scratch.context[d] *= sig;
    }

    // Output projection (Q8 NEON) — write directly into attn_out
    matmul_q8_neon_into(
        &scratch.context[..q_dim],
        &weights.o_proj_packed,
        hidden,
        q_dim,
        &mut scratch.attn_out[..hidden],
        &mut scratch.x_q_scratch,
    );
}

// -----------------------------------------------------------------------
// FFN step (Q8 NEON projections)
// -----------------------------------------------------------------------

/// SwiGLU FFN step using Q8_0 NEON projections.
///
/// Input is read from `scratch.ffn_out[..hidden]`, output written back to
/// `scratch.ffn_out[..hidden]`.
#[inline]
fn ffn_step_q8_neon(common: &Q8NeonCommonWeights, scratch: &mut ForwardScratch, hidden: usize) {
    let inter = common.gate_proj_rows;

    // gate and up projections — write directly into scratch, reuse x_q_scratch
    matmul_q8_neon_into(
        &scratch.ffn_out[..hidden],
        &common.gate_proj_packed,
        inter,
        hidden,
        &mut scratch.gate_buf[..inter],
        &mut scratch.x_q_scratch,
    );
    matmul_q8_neon_into(
        &scratch.ffn_out[..hidden],
        &common.up_proj_packed,
        inter,
        hidden,
        &mut scratch.up_buf[..inter],
        &mut scratch.x_q_scratch,
    );

    // SwiGLU: silu(gate) * up
    silu_inplace(&mut scratch.gate_buf[..inter]);
    elementwise_mul(&mut scratch.gate_buf[..inter], &scratch.up_buf[..inter]);

    // down_proj — write directly into ffn_out
    matmul_q8_neon_into(
        &scratch.gate_buf[..inter],
        &common.down_proj_packed,
        hidden,
        inter,
        &mut scratch.ffn_out[..hidden],
        &mut scratch.x_q_scratch,
    );
}

// -----------------------------------------------------------------------
// Full forward step
// -----------------------------------------------------------------------

/// Single-token forward pass using Q8_0 NEON weight matrices.
///
/// Equivalent to `Qwen35Model::forward_step` but all large projection matrices
/// use the native NEON int8 kernel. Norms, recurrent state, attention scores,
/// and activations remain in f32.
///
/// Writes logits into `scratch.logits`.
pub(crate) fn forward_step_q8_neon(
    model: &Q8NeonModel,
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

    // Embedding lookup (f32)
    let embed_start = token_id as usize * hidden;
    scratch.hidden[..hidden]
        .copy_from_slice(&model.embed_tokens[embed_start..embed_start + hidden]);

    let mut linear_idx = 0usize;
    let mut full_idx = 0usize;

    for layer_i in 0..cfg.num_hidden_layers {
        let (attn_weights, common) = &model.layers[layer_i];

        // Save residual
        scratch.residual[..hidden].copy_from_slice(&scratch.hidden[..hidden]);

        // Pre-attention RMSNorm (Qwen3.5: 1 + gamma)
        qwen35_rms_norm(
            &mut scratch.hidden[..hidden],
            &common.input_layernorm,
            hidden,
            cfg.rms_norm_eps,
        );

        // Attention
        match attn_weights {
            Q8NeonAttentionWeights::Linear(gdn_w) => {
                gdn_step_q8_neon(
                    &scratch.hidden[..hidden],
                    &mut gdn_states[linear_idx],
                    gdn_w,
                    cfg,
                    &mut scratch.gdn_scratch,
                    &mut scratch.x_q_scratch,
                    &mut scratch.attn_out[..hidden],
                );
                linear_idx += 1;
            }
            Q8NeonAttentionWeights::Full(full_w) => {
                scratch.attn_out[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
                full_attention_step_q8_neon(
                    full_w, full_idx, position, kv_cache, scratch, cfg, rope, hidden,
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

        // Post-attention RMSNorm (Qwen3.5: 1 + gamma)
        qwen35_rms_norm(
            &mut scratch.hidden[..hidden],
            &common.post_attention_layernorm,
            hidden,
            cfg.rms_norm_eps,
        );

        // SwiGLU FFN
        scratch.ffn_out[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
        ffn_step_q8_neon(common, scratch, hidden);

        // Residual connection
        for i in 0..hidden {
            scratch.hidden[i] = scratch.residual[i] + scratch.ffn_out[i];
        }
    }

    // Final RMSNorm (Qwen3.5: 1 + gamma)
    qwen35_rms_norm(
        &mut scratch.hidden[..hidden],
        &model.final_norm,
        hidden,
        cfg.rms_norm_eps,
    );

    // Logits: hidden @ lm_head^T (Q8 NEON — biggest single matmul, write directly)
    resize(&mut scratch.logits, cfg.vocab_size);
    matmul_q8_neon_into(
        &scratch.hidden[..hidden],
        &model.lm_head_packed,
        model.lm_head_rows,
        model.lm_head_cols,
        &mut scratch.logits[..cfg.vocab_size],
        &mut scratch.x_q_scratch,
    );
}

// -----------------------------------------------------------------------
// Generate
// -----------------------------------------------------------------------

/// **Unstable**: Q8_0 NEON generate; function signature will likely merge with model struct API.
///
/// Generate text from a prompt using Q8_0 NEON weight matrices.
///
/// Equivalent to `Qwen35Model::generate` but calls `forward_step_q8_neon`
/// for all forward passes.
pub fn generate_q8_neon(
    model: &Q8NeonModel,
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

    // Initialize states
    let num_linear = cfg.num_linear_attention_layers();
    let num_full = cfg.num_full_attention_layers();
    let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
        .map(|_| GatedDeltaNetState::new(cfg))
        .collect();
    let mut kv_cache = KvCache::new(num_full);
    let mut scratch = ForwardScratch::new();
    let max_seq_len = prompt_len
        .saturating_add(gen_cfg.max_new_tokens)
        .saturating_add(1);

    kv_cache.reserve(max_seq_len, cfg.full_kv_dim());
    scratch.ensure_capacity(cfg, max_seq_len);

    let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
    let mut all_ids = prompt_ids.clone();

    // Prefill: process prompt tokens one at a time
    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        forward_step_q8_neon(
            model,
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

    if next_id == cfg.eos_token_id {
        return Ok(GenerateOutput {
            text: String::new(),
            token_ids: vec![],
            prompt_tokens: prompt_len,
            generated_tokens: 0,
        });
    }

    generated_ids.push(next_id);
    all_ids.push(next_id);

    // Autoregressive decode
    for _ in 1..gen_cfg.max_new_tokens {
        let pos = kv_cache.seq_len;
        let last_token = *all_ids
            .last()
            .expect("invariant: prompt or previous sample populated all_ids");

        forward_step_q8_neon(
            model,
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

        if next_id == cfg.eos_token_id {
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
    })
}

// -----------------------------------------------------------------------
// Bench-only support module
// -----------------------------------------------------------------------

/// Opaque benchmark fixtures for criterion benches.  Only compiled with
/// `--features bench-internals`; the default public API is unchanged.
#[cfg(feature = "bench-internals")]
pub mod bench_support {
    use super::*;
    use crate::model::qwen35_config::LayerType;
    use crate::rope::RopeTable;

    pub struct Q8ForwardBenchFixture {
        model: Q8NeonModel,
        cfg: Qwen35Config,
        rope: RopeTable,
    }

    pub struct Q8ForwardBenchState {
        gdn_states: Vec<GatedDeltaNetState>,
        kv_cache: KvCache,
        scratch: ForwardScratch,
    }

    fn xorshift64(state: &mut u64) -> f32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        (x & 0xFFFF) as f32 / 0x10000_u64 as f32 * 0.04 - 0.02
    }

    fn gen_weights_q8(n: usize, k: usize, rng: &mut u64) -> Vec<u8> {
        let floats: Vec<f32> = (0..n * k).map(|_| xorshift64(rng)).collect();
        pack_weights_q8(&floats, n, k)
    }

    impl Q8ForwardBenchFixture {
        /// Build a 2-layer (GDN + full-attention) synthetic Q8 model.
        ///
        /// All K-dimensions are multiples of 32 so Q8_0 packing is valid.
        /// Dimensions are small to keep fixture construction fast and bench
        /// iteration time dominated by the forward step, not memory traffic.
        pub fn synthetic_2layer() -> Self {
            let mut rng: u64 = 0xdeadbeef_cafebabe;

            let hidden: usize = 256;
            let vocab: usize = 8192;
            let inter: usize = 768;

            let num_attn_heads: usize = 4;
            let num_kv_heads: usize = 2;
            let head_dim: usize = 64;
            let q_dim = num_attn_heads * head_dim; // 256
            let kv_dim = num_kv_heads * head_dim; // 128

            let lin_key_heads: usize = 4;
            let lin_val_heads: usize = 4;
            let lin_key_dim: usize = 64;
            let lin_val_dim: usize = 64;
            // Q + K + V each have lin_key_heads * lin_key_dim dims
            let lin_qkv_dim = lin_key_heads * lin_key_dim * 2 + lin_val_heads * lin_val_dim; // 768
            let lin_output_dim = lin_val_heads * lin_val_dim; // 256
            let kernel_size: usize = 4;

            let cfg = Qwen35Config {
                hidden_size: hidden,
                num_hidden_layers: 2,
                vocab_size: vocab,
                intermediate_size: inter,
                rms_norm_eps: 1e-6,
                num_attention_heads: num_attn_heads,
                num_key_value_heads: num_kv_heads,
                head_dim,
                rope_theta: 10_000.0,
                partial_rotary_factor: 0.5,
                rope_parameters: None,
                linear_num_key_heads: lin_key_heads,
                linear_num_value_heads: Some(lin_val_heads),
                linear_key_head_dim: lin_key_dim,
                linear_value_head_dim: lin_val_dim,
                linear_conv_kernel_dim: kernel_size,
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
                eos_token_id: 8191,
                max_position_embeddings: 512,
                mtp_num_hidden_layers: 0,
                mtp_use_dedicated_embeddings: false,
            };

            let rope_dim = (head_dim as f32 * cfg.partial_rotary_factor) as usize; // 32
            let rope = RopeTable::new(rope_dim, cfg.max_position_embeddings, cfg.rope_theta);

            let gdn_weights = Q8NeonGdnWeights {
                in_proj_qkv_packed: gen_weights_q8(lin_qkv_dim, hidden, &mut rng),
                in_proj_qkv_rows: lin_qkv_dim,
                in_proj_qkv_cols: hidden,

                in_proj_z_packed: gen_weights_q8(lin_output_dim, hidden, &mut rng),
                in_proj_z_rows: lin_output_dim,
                in_proj_z_cols: hidden,

                in_proj_b_packed: gen_weights_q8(lin_key_heads, hidden, &mut rng),
                in_proj_b_rows: lin_key_heads,
                in_proj_b_cols: hidden,

                in_proj_a_packed: gen_weights_q8(lin_key_heads, hidden, &mut rng),
                in_proj_a_rows: lin_key_heads,
                in_proj_a_cols: hidden,

                out_proj_packed: gen_weights_q8(hidden, lin_output_dim, &mut rng),
                out_proj_rows: hidden,
                out_proj_cols: lin_output_dim,

                a_log: vec![0.0f32; lin_key_heads],
                dt_bias: vec![0.0f32; lin_key_heads],
                conv1d_weight: vec![0.01f32; lin_qkv_dim * kernel_size],
                conv_dim: lin_qkv_dim,
                kernel_size,
                norm_weight: vec![0.0f32; lin_val_dim],
            };

            let full_weights = Q8NeonFullAttnWeights {
                q_proj_packed: gen_weights_q8(2 * q_dim, hidden, &mut rng),
                q_proj_rows: 2 * q_dim,
                q_proj_cols: hidden,

                k_proj_packed: gen_weights_q8(kv_dim, hidden, &mut rng),
                k_proj_rows: kv_dim,
                k_proj_cols: hidden,

                v_proj_packed: gen_weights_q8(kv_dim, hidden, &mut rng),
                v_proj_rows: kv_dim,
                v_proj_cols: hidden,

                o_proj_packed: gen_weights_q8(hidden, q_dim, &mut rng),
                o_proj_rows: hidden,
                o_proj_cols: q_dim,

                q_norm: vec![0.0f32; head_dim],
                k_norm: vec![0.0f32; head_dim],
            };

            let make_common = |rng: &mut u64| Q8NeonCommonWeights {
                input_layernorm: vec![0.0f32; hidden],
                post_attention_layernorm: vec![0.0f32; hidden],
                gate_proj_packed: gen_weights_q8(inter, hidden, rng),
                gate_proj_rows: inter,
                gate_proj_cols: hidden,
                up_proj_packed: gen_weights_q8(inter, hidden, rng),
                up_proj_rows: inter,
                up_proj_cols: hidden,
                down_proj_packed: gen_weights_q8(hidden, inter, rng),
                down_proj_rows: hidden,
                down_proj_cols: inter,
            };
            let common0 = make_common(&mut rng);
            let common1 = make_common(&mut rng);

            let embed_tokens: Vec<f32> = (0..vocab * hidden)
                .map(|i| {
                    let mut s = (i as u64).wrapping_mul(0x9e3779b9_7f4a7c15);
                    s ^= s >> 33;
                    s &= 0xFFFF;
                    s as f32 / 0x10000_u64 as f32 * 0.04 - 0.02
                })
                .collect();
            let lm_head_packed = pack_weights_q8(&embed_tokens, vocab, hidden);

            let model = Q8NeonModel {
                embed_tokens,
                final_norm: vec![0.0f32; hidden],
                lm_head_packed,
                lm_head_rows: vocab,
                lm_head_cols: hidden,
                layers: vec![
                    (Q8NeonAttentionWeights::Linear(gdn_weights), common0),
                    (Q8NeonAttentionWeights::Full(full_weights), common1),
                ],
            };

            Self { model, cfg, rope }
        }

        /// Build a 24-layer Qwen35-2B-shaped Q8 model with synthetic weights.
        ///
        /// Uses all Qwen35-2B layer dimensions (hidden=2048, 18 GDN + 6 full layers,
        /// intermediate=6144) but with vocab_size=256 to avoid an impractical ~2GB
        /// embed_tokens allocation. The lm_head allocation per token is therefore
        /// smaller than the real-model scale; all attention projection allocations
        /// are exact Qwen35-2B shape.
        pub fn qwen35_24layer_shape() -> Self {
            let mut rng: u64 = 0x0123_4567_89ab_cdef;

            let mut cfg = Qwen35Config::qwen35_2b();
            cfg.vocab_size = 256;

            let hidden = cfg.hidden_size;
            let vocab = cfg.vocab_size;
            let inter = cfg.intermediate_size;
            let qkv_dim = cfg.linear_qkv_dim();
            let output_dim = cfg.linear_output_dim();
            let num_heads_lin = cfg.linear_num_key_heads;
            let lin_val_dim = cfg.linear_value_head_dim;
            let kernel_size = cfg.linear_conv_kernel_dim;
            let q_dim = cfg.full_q_dim();
            let kv_dim = cfg.full_kv_dim();

            let make_gdn = |rng: &mut u64| Q8NeonGdnWeights {
                in_proj_qkv_packed: gen_weights_q8(qkv_dim, hidden, rng),
                in_proj_qkv_rows: qkv_dim,
                in_proj_qkv_cols: hidden,
                in_proj_z_packed: gen_weights_q8(output_dim, hidden, rng),
                in_proj_z_rows: output_dim,
                in_proj_z_cols: hidden,
                in_proj_b_packed: gen_weights_q8(num_heads_lin, hidden, rng),
                in_proj_b_rows: num_heads_lin,
                in_proj_b_cols: hidden,
                in_proj_a_packed: gen_weights_q8(num_heads_lin, hidden, rng),
                in_proj_a_rows: num_heads_lin,
                in_proj_a_cols: hidden,
                out_proj_packed: gen_weights_q8(hidden, output_dim, rng),
                out_proj_rows: hidden,
                out_proj_cols: output_dim,
                a_log: vec![0.0f32; num_heads_lin],
                dt_bias: vec![0.0f32; num_heads_lin],
                conv1d_weight: vec![0.01f32; qkv_dim * kernel_size],
                conv_dim: qkv_dim,
                kernel_size,
                norm_weight: vec![0.0f32; lin_val_dim],
            };

            let make_full = |rng: &mut u64| Q8NeonFullAttnWeights {
                q_proj_packed: gen_weights_q8(2 * q_dim, hidden, rng),
                q_proj_rows: 2 * q_dim,
                q_proj_cols: hidden,
                k_proj_packed: gen_weights_q8(kv_dim, hidden, rng),
                k_proj_rows: kv_dim,
                k_proj_cols: hidden,
                v_proj_packed: gen_weights_q8(kv_dim, hidden, rng),
                v_proj_rows: kv_dim,
                v_proj_cols: hidden,
                o_proj_packed: gen_weights_q8(hidden, q_dim, rng),
                o_proj_rows: hidden,
                o_proj_cols: q_dim,
                q_norm: vec![0.0f32; cfg.head_dim],
                k_norm: vec![0.0f32; cfg.head_dim],
            };

            let make_common = |rng: &mut u64| Q8NeonCommonWeights {
                input_layernorm: vec![0.0f32; hidden],
                post_attention_layernorm: vec![0.0f32; hidden],
                gate_proj_packed: gen_weights_q8(inter, hidden, rng),
                gate_proj_rows: inter,
                gate_proj_cols: hidden,
                up_proj_packed: gen_weights_q8(inter, hidden, rng),
                up_proj_rows: inter,
                up_proj_cols: hidden,
                down_proj_packed: gen_weights_q8(hidden, inter, rng),
                down_proj_rows: hidden,
                down_proj_cols: inter,
            };

            let layers: Vec<(Q8NeonAttentionWeights, Q8NeonCommonWeights)> = cfg
                .layer_types
                .iter()
                .map(|lt| match lt {
                    LayerType::LinearAttention => (
                        Q8NeonAttentionWeights::Linear(make_gdn(&mut rng)),
                        make_common(&mut rng),
                    ),
                    LayerType::FullAttention => (
                        Q8NeonAttentionWeights::Full(make_full(&mut rng)),
                        make_common(&mut rng),
                    ),
                })
                .collect();

            let embed_tokens: Vec<f32> = (0..vocab * hidden)
                .map(|i| {
                    let mut s = (i as u64).wrapping_mul(0x9e3779b9_7f4a7c15);
                    s ^= s >> 33;
                    s &= 0xFFFF;
                    s as f32 / 0x10000_u64 as f32 * 0.04 - 0.02
                })
                .collect();
            let lm_head_packed = pack_weights_q8(&embed_tokens, vocab, hidden);

            let model = Q8NeonModel {
                embed_tokens,
                final_norm: vec![0.0f32; hidden],
                lm_head_packed,
                lm_head_rows: vocab,
                lm_head_cols: hidden,
                layers,
            };

            let rope_dim = cfg.rope_dim();
            let rope = RopeTable::new(rope_dim, cfg.max_position_embeddings, cfg.rope_theta);

            Self { model, cfg, rope }
        }

        /// Create fresh mutable state; runs `warm_len` steps outside the measured loop.
        pub fn state(&self, warm_len: usize) -> Q8ForwardBenchState {
            self.state_with_capacity(warm_len, 1)
        }

        /// Create fresh mutable state with enough cache/scratch for a measured token loop.
        pub fn state_with_capacity(
            &self,
            warm_len: usize,
            measured_tokens: usize,
        ) -> Q8ForwardBenchState {
            let num_linear = self.cfg.num_linear_attention_layers();
            let num_full = self.cfg.num_full_attention_layers();
            let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
                .map(|_| GatedDeltaNetState::new(&self.cfg))
                .collect();
            let mut kv_cache = KvCache::new(num_full);
            let mut scratch = ForwardScratch::new();
            let max_seq_len = warm_len.saturating_add(measured_tokens).saturating_add(1);

            kv_cache.reserve(max_seq_len, self.cfg.full_kv_dim());
            scratch.ensure_capacity(&self.cfg, max_seq_len);

            for pos in 0..warm_len {
                let token_id = (pos as u32) % (self.cfg.vocab_size as u32);
                forward_step_q8_neon(
                    &self.model,
                    &self.cfg,
                    &self.rope,
                    token_id,
                    pos,
                    &mut gdn_states,
                    &mut kv_cache,
                    &mut scratch,
                );
                kv_cache.seq_len += 1;
            }

            Q8ForwardBenchState {
                gdn_states,
                kv_cache,
                scratch,
            }
        }

        /// Run one `forward_step_q8_neon` and advance the sequence position.
        /// Returns `scratch.logits[0]` for `black_box` without exposing scratch types.
        pub fn step(&self, state: &mut Q8ForwardBenchState, token_id: u32) -> f32 {
            let pos = state.kv_cache.seq_len;
            forward_step_q8_neon(
                &self.model,
                &self.cfg,
                &self.rope,
                token_id % (self.cfg.vocab_size as u32),
                pos,
                &mut state.gdn_states,
                &mut state.kv_cache,
                &mut state.scratch,
            );
            state.kv_cache.seq_len += 1;
            state.scratch.logits[0]
        }
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen35_config::LayerType;

    /// Helper: create a zero Q8_0 packed weight buffer for [n, k].
    fn zero_packed(n: usize, k: usize) -> Vec<u8> {
        pack_weights_q8(&vec![0.0f32; n * k], n, k)
    }

    #[test]
    fn test_quantize_model_produces_valid_packed_sizes() {
        let cfg = Qwen35Config::qwen35_2b();
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let inter = cfg.intermediate_size;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let q_dim = cfg.full_q_dim();
        let kv_dim = cfg.full_kv_dim();

        // Q8_0 packed size: n * (k/32) * 36 bytes
        let q8_packed_size = |n: usize, k: usize| -> usize {
            assert_eq!(k % 32, 0, "k={k} must be multiple of 32");
            n * (k / 32) * 36
        };

        // Build a minimal ModelWeights with known-size zero tensors
        let gdn_w = GatedDeltaNetWeights {
            in_proj_qkv: vec![0.0; qkv_dim * hidden],
            in_proj_qkv_rows: qkv_dim,
            in_proj_qkv_cols: hidden,
            in_proj_z: vec![0.0; output_dim * hidden],
            in_proj_z_rows: output_dim,
            in_proj_z_cols: hidden,
            in_proj_b: vec![0.0; cfg.linear_num_key_heads * hidden],
            in_proj_b_rows: cfg.linear_num_key_heads,
            in_proj_b_cols: hidden,
            in_proj_a: vec![0.0; cfg.linear_num_key_heads * hidden],
            in_proj_a_rows: cfg.linear_num_key_heads,
            in_proj_a_cols: hidden,
            a_log: vec![0.0; cfg.linear_num_key_heads],
            dt_bias: vec![0.0; cfg.linear_num_key_heads],
            conv1d_weight: vec![0.0; qkv_dim * cfg.linear_conv_kernel_dim],
            conv_dim: qkv_dim,
            kernel_size: cfg.linear_conv_kernel_dim,
            norm_weight: vec![0.0; cfg.linear_value_head_dim],
            out_proj: vec![0.0; hidden * output_dim],
            out_proj_rows: hidden,
            out_proj_cols: output_dim,
        };

        let full_w = FullAttentionLayerWeights {
            q_proj: vec![0.0; 2 * q_dim * hidden],
            k_proj: vec![0.0; kv_dim * hidden],
            v_proj: vec![0.0; kv_dim * hidden],
            o_proj: vec![0.0; hidden * q_dim],
            q_norm: vec![0.0; cfg.head_dim],
            k_norm: vec![0.0; cfg.head_dim],
        };

        let make_common_w = || CommonLayerWeights {
            input_layernorm: vec![0.0; hidden],
            post_attention_layernorm: vec![0.0; hidden],
            ffn: crate::model::qwen35::FeedForwardWeights::Dense(
                crate::model::qwen35::DenseFfnWeights {
                    gate_proj: vec![0.0; inter * hidden],
                    up_proj: vec![0.0; inter * hidden],
                    down_proj: vec![0.0; hidden * inter],
                },
            ),
        };

        // Build one linear + one full layer to test both paths
        let weights = ModelWeights {
            embed_tokens: vec![0.0; vocab * hidden],
            lm_head: None,
            final_norm: vec![0.0; hidden],
            layers: vec![
                (AttentionWeights::Linear(gdn_w), make_common_w()),
                (AttentionWeights::Full(full_w), make_common_w()),
            ],
        };

        let q8 = quantize_model(&weights, &cfg);

        // Check lm_head packed size
        assert_eq!(q8.lm_head_packed.len(), q8_packed_size(vocab, hidden));
        assert_eq!(q8.lm_head_rows, vocab);
        assert_eq!(q8.lm_head_cols, hidden);

        // Check embed_tokens preserved
        assert_eq!(q8.embed_tokens.len(), vocab * hidden);

        // Check layer 0 (linear)
        match &q8.layers[0].0 {
            Q8NeonAttentionWeights::Linear(gdn) => {
                assert_eq!(
                    gdn.in_proj_qkv_packed.len(),
                    q8_packed_size(qkv_dim, hidden)
                );
                assert_eq!(
                    gdn.in_proj_z_packed.len(),
                    q8_packed_size(output_dim, hidden)
                );
                assert_eq!(
                    gdn.out_proj_packed.len(),
                    q8_packed_size(hidden, output_dim)
                );
                assert_eq!(gdn.a_log.len(), cfg.linear_num_key_heads);
            }
            Q8NeonAttentionWeights::Full(_) => panic!("expected Linear layer"),
        }

        // Check layer 1 (full)
        match &q8.layers[1].0 {
            Q8NeonAttentionWeights::Full(full) => {
                assert_eq!(full.q_proj_packed.len(), q8_packed_size(2 * q_dim, hidden));
                assert_eq!(full.k_proj_packed.len(), q8_packed_size(kv_dim, hidden));
                assert_eq!(full.v_proj_packed.len(), q8_packed_size(kv_dim, hidden));
                assert_eq!(full.o_proj_packed.len(), q8_packed_size(hidden, q_dim));
            }
            Q8NeonAttentionWeights::Linear(_) => panic!("expected Full layer"),
        }

        // Check common weights
        let c = &q8.layers[0].1;
        assert_eq!(c.gate_proj_packed.len(), q8_packed_size(inter, hidden));
        assert_eq!(c.up_proj_packed.len(), q8_packed_size(inter, hidden));
        assert_eq!(c.down_proj_packed.len(), q8_packed_size(hidden, inter));
        assert_eq!(c.input_layernorm.len(), hidden);
    }

    #[test]
    fn test_forward_step_q8_neon_zero_weights_produces_zero_logits() {
        let cfg = Qwen35Config::qwen35_2b();
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let inter = cfg.intermediate_size;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let num_heads = cfg.linear_num_key_heads;
        let q_dim = cfg.full_q_dim();
        let kv_dim = cfg.full_kv_dim();

        // Build a minimal 2-layer model (1 linear + 1 full) with zero weights
        let make_linear = || Q8NeonGdnWeights {
            in_proj_qkv_packed: zero_packed(qkv_dim, hidden),
            in_proj_qkv_rows: qkv_dim,
            in_proj_qkv_cols: hidden,
            in_proj_z_packed: zero_packed(output_dim, hidden),
            in_proj_z_rows: output_dim,
            in_proj_z_cols: hidden,
            in_proj_b_packed: zero_packed(num_heads, hidden),
            in_proj_b_rows: num_heads,
            in_proj_b_cols: hidden,
            in_proj_a_packed: zero_packed(num_heads, hidden),
            in_proj_a_rows: num_heads,
            in_proj_a_cols: hidden,
            out_proj_packed: zero_packed(hidden, output_dim),
            out_proj_rows: hidden,
            out_proj_cols: output_dim,
            a_log: vec![0.0; num_heads],
            dt_bias: vec![0.0; num_heads],
            conv1d_weight: vec![0.0; qkv_dim * cfg.linear_conv_kernel_dim],
            conv_dim: qkv_dim,
            kernel_size: cfg.linear_conv_kernel_dim,
            norm_weight: vec![0.0; cfg.linear_value_head_dim],
        };

        let make_full = || Q8NeonFullAttnWeights {
            q_proj_packed: zero_packed(2 * q_dim, hidden),
            q_proj_rows: 2 * q_dim,
            q_proj_cols: hidden,
            k_proj_packed: zero_packed(kv_dim, hidden),
            k_proj_rows: kv_dim,
            k_proj_cols: hidden,
            v_proj_packed: zero_packed(kv_dim, hidden),
            v_proj_rows: kv_dim,
            v_proj_cols: hidden,
            o_proj_packed: zero_packed(hidden, q_dim),
            o_proj_rows: hidden,
            o_proj_cols: q_dim,
            q_norm: vec![0.0; cfg.head_dim],
            k_norm: vec![0.0; cfg.head_dim],
        };

        let make_common = || Q8NeonCommonWeights {
            input_layernorm: vec![0.0; hidden],
            post_attention_layernorm: vec![0.0; hidden],
            gate_proj_packed: zero_packed(inter, hidden),
            gate_proj_rows: inter,
            gate_proj_cols: hidden,
            up_proj_packed: zero_packed(inter, hidden),
            up_proj_rows: inter,
            up_proj_cols: hidden,
            down_proj_packed: zero_packed(hidden, inter),
            down_proj_rows: hidden,
            down_proj_cols: inter,
        };

        // Build config for just 2 layers to keep test fast
        let mut test_cfg = cfg.clone();
        test_cfg.num_hidden_layers = 2;
        test_cfg.layer_types = vec![
            crate::model::qwen35_config::LayerType::LinearAttention,
            crate::model::qwen35_config::LayerType::FullAttention,
        ];

        let model = Q8NeonModel {
            embed_tokens: vec![0.0; vocab * hidden],
            final_norm: vec![0.0; hidden],
            lm_head_packed: zero_packed(vocab, hidden),
            lm_head_rows: vocab,
            lm_head_cols: hidden,
            layers: vec![
                (Q8NeonAttentionWeights::Linear(make_linear()), make_common()),
                (Q8NeonAttentionWeights::Full(make_full()), make_common()),
            ],
        };

        let rope_dim = test_cfg.rope_dim();
        let rope_max = test_cfg.max_position_embeddings.min(8192);
        let rope = RopeTable::new(rope_dim, rope_max, test_cfg.rope_theta);

        let num_linear = test_cfg.num_linear_attention_layers();
        let num_full = test_cfg.num_full_attention_layers();
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::new(&test_cfg))
            .collect();
        let mut kv_cache = KvCache::new(num_full);
        let mut scratch = ForwardScratch::new();

        forward_step_q8_neon(
            &model,
            &test_cfg,
            &rope,
            0, // token_id
            0, // position
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
        );

        // With all-zero weights and embeddings, all logits should be zero
        for (i, &v) in scratch.logits[..test_cfg.vocab_size].iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "logit[{i}] = {v}, expected ~0.0 with zero weights"
            );
        }
    }

    #[test]
    fn test_gdn_step_q8_neon_zero_weights_produces_zero_output() {
        let cfg = Qwen35Config::qwen35_2b();
        let hidden = cfg.hidden_size;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let num_heads = cfg.linear_num_key_heads;

        let weights = Q8NeonGdnWeights {
            in_proj_qkv_packed: zero_packed(qkv_dim, hidden),
            in_proj_qkv_rows: qkv_dim,
            in_proj_qkv_cols: hidden,
            in_proj_z_packed: zero_packed(output_dim, hidden),
            in_proj_z_rows: output_dim,
            in_proj_z_cols: hidden,
            in_proj_b_packed: zero_packed(num_heads, hidden),
            in_proj_b_rows: num_heads,
            in_proj_b_cols: hidden,
            in_proj_a_packed: zero_packed(num_heads, hidden),
            in_proj_a_rows: num_heads,
            in_proj_a_cols: hidden,
            out_proj_packed: zero_packed(hidden, output_dim),
            out_proj_rows: hidden,
            out_proj_cols: output_dim,
            a_log: vec![0.0; num_heads],
            dt_bias: vec![0.0; num_heads],
            conv1d_weight: vec![0.0; qkv_dim * cfg.linear_conv_kernel_dim],
            conv_dim: qkv_dim,
            kernel_size: cfg.linear_conv_kernel_dim,
            norm_weight: vec![0.0; cfg.linear_value_head_dim],
        };

        let mut state = GatedDeltaNetState::new(&cfg);
        let input = vec![0.0f32; hidden];
        let mut output = vec![0.0f32; hidden];
        let mut gdn_scratch = GatedDeltaNetFusedScratch::default();
        let mut x_q_scratch = Vec::new();

        gdn_step_q8_neon(
            &input,
            &mut state,
            &weights,
            &cfg,
            &mut gdn_scratch,
            &mut x_q_scratch,
            &mut output,
        );

        for (i, &v) in output[..hidden].iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "output[{i}] = {v}, expected 0.0 with zero weights + zero input"
            );
        }
    }

    #[test]
    fn test_ffn_step_q8_neon_zero_weights() {
        let cfg = Qwen35Config::qwen35_2b();
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;

        let common = Q8NeonCommonWeights {
            input_layernorm: vec![0.0; hidden],
            post_attention_layernorm: vec![0.0; hidden],
            gate_proj_packed: zero_packed(inter, hidden),
            gate_proj_rows: inter,
            gate_proj_cols: hidden,
            up_proj_packed: zero_packed(inter, hidden),
            up_proj_rows: inter,
            up_proj_cols: hidden,
            down_proj_packed: zero_packed(hidden, inter),
            down_proj_rows: hidden,
            down_proj_cols: inter,
        };

        let mut scratch = ForwardScratch::new();
        scratch.ensure_capacity(&cfg, 1);
        scratch.ffn_out[..hidden].fill(0.0);

        ffn_step_q8_neon(&common, &mut scratch, hidden);

        for (i, &v) in scratch.ffn_out[..hidden].iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "ffn_out[{i}] = {v}, expected 0.0 with zero weights"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Nonzero parity test: captures current allocating NEON logits as baseline.
    // After GDN/full-attention buffer migration (i2), re-run and verify the
    // first 16 logits are within 1e-6 of these captured constants.
    // -----------------------------------------------------------------------

    /// Small deterministic test model — 2 layers, all dims multiples of 32.
    fn make_nonzero_q8_neon_test_model() -> (Qwen35Config, Q8NeonModel, RopeTable) {
        let hidden: usize = 64;
        let vocab: usize = 128;
        let inter: usize = 128;
        let num_attn_heads: usize = 2;
        let num_kv_heads: usize = 1;
        let head_dim: usize = 32;
        let q_dim = num_attn_heads * head_dim; // 64
        let kv_dim = num_kv_heads * head_dim; // 32
        let lin_key_heads: usize = 2;
        let lin_val_heads: usize = 2;
        let lin_key_dim: usize = 32;
        let lin_val_dim: usize = 32;
        let lin_qkv_dim = lin_key_heads * lin_key_dim * 2 + lin_val_heads * lin_val_dim; // 192
        let lin_output_dim = lin_val_heads * lin_val_dim; // 64
        let kernel_size: usize = 4;

        let cfg = Qwen35Config {
            hidden_size: hidden,
            num_hidden_layers: 2,
            vocab_size: vocab,
            intermediate_size: inter,
            rms_norm_eps: 1e-6,
            num_attention_heads: num_attn_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.5,
            rope_parameters: None,
            linear_num_key_heads: lin_key_heads,
            linear_num_value_heads: Some(lin_val_heads),
            linear_key_head_dim: lin_key_dim,
            linear_value_head_dim: lin_val_dim,
            linear_conv_kernel_dim: kernel_size,
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
        };

        let rope_dim = (head_dim as f32 * cfg.partial_rotary_factor) as usize; // 16
        let rope = RopeTable::new(rope_dim, cfg.max_position_embeddings, cfg.rope_theta);

        // Deterministic weight generator: LCG producing small floats.
        let mut seed: u64 = 0xdead_beef_cafe_babe;
        let mut next_weight = |n: usize, k: usize| -> Vec<u8> {
            let floats: Vec<f32> = (0..n * k)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((seed >> 33) as f32 / u32::MAX as f32) * 0.04 - 0.02
                })
                .collect();
            pack_weights_q8(&floats, n, k)
        };

        let gdn_w = Q8NeonGdnWeights {
            in_proj_qkv_packed: next_weight(lin_qkv_dim, hidden),
            in_proj_qkv_rows: lin_qkv_dim,
            in_proj_qkv_cols: hidden,
            in_proj_z_packed: next_weight(lin_output_dim, hidden),
            in_proj_z_rows: lin_output_dim,
            in_proj_z_cols: hidden,
            in_proj_b_packed: next_weight(lin_key_heads, hidden),
            in_proj_b_rows: lin_key_heads,
            in_proj_b_cols: hidden,
            in_proj_a_packed: next_weight(lin_key_heads, hidden),
            in_proj_a_rows: lin_key_heads,
            in_proj_a_cols: hidden,
            out_proj_packed: next_weight(hidden, lin_output_dim),
            out_proj_rows: hidden,
            out_proj_cols: lin_output_dim,
            a_log: vec![0.0f32; lin_key_heads],
            dt_bias: vec![0.0f32; lin_key_heads],
            conv1d_weight: vec![0.01f32; lin_qkv_dim * kernel_size],
            conv_dim: lin_qkv_dim,
            kernel_size,
            norm_weight: vec![0.0f32; lin_val_dim],
        };

        let full_w = Q8NeonFullAttnWeights {
            q_proj_packed: next_weight(2 * q_dim, hidden),
            q_proj_rows: 2 * q_dim,
            q_proj_cols: hidden,
            k_proj_packed: next_weight(kv_dim, hidden),
            k_proj_rows: kv_dim,
            k_proj_cols: hidden,
            v_proj_packed: next_weight(kv_dim, hidden),
            v_proj_rows: kv_dim,
            v_proj_cols: hidden,
            o_proj_packed: next_weight(hidden, q_dim),
            o_proj_rows: hidden,
            o_proj_cols: q_dim,
            q_norm: vec![0.0f32; head_dim],
            k_norm: vec![0.0f32; head_dim],
        };

        let common_w = |seed: &mut u64| {
            let mut nw = |n: usize, k: usize| -> Vec<u8> {
                let floats: Vec<f32> = (0..n * k)
                    .map(|_| {
                        *seed = seed
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        ((*seed >> 33) as f32 / u32::MAX as f32) * 0.04 - 0.02
                    })
                    .collect();
                pack_weights_q8(&floats, n, k)
            };
            Q8NeonCommonWeights {
                input_layernorm: vec![0.0f32; hidden],
                post_attention_layernorm: vec![0.0f32; hidden],
                gate_proj_packed: nw(inter, hidden),
                gate_proj_rows: inter,
                gate_proj_cols: hidden,
                up_proj_packed: nw(inter, hidden),
                up_proj_rows: inter,
                up_proj_cols: hidden,
                down_proj_packed: nw(hidden, inter),
                down_proj_rows: hidden,
                down_proj_cols: inter,
            }
        };
        let common0 = common_w(&mut seed);
        let common1 = common_w(&mut seed);

        let embed_tokens: Vec<f32> = (0..vocab * hidden)
            .map(|i| {
                let mut s = (i as u64).wrapping_mul(0x9e3779b9_7f4a7c15);
                s ^= s >> 33;
                s &= 0xFFFF;
                s as f32 / 0x10000_u64 as f32 * 0.04 - 0.02
            })
            .collect();
        let lm_head_packed = pack_weights_q8(&embed_tokens, vocab, hidden);

        let model = Q8NeonModel {
            embed_tokens,
            final_norm: vec![0.0f32; hidden],
            lm_head_packed,
            lm_head_rows: vocab,
            lm_head_cols: hidden,
            layers: vec![
                (Q8NeonAttentionWeights::Linear(gdn_w), common0),
                (Q8NeonAttentionWeights::Full(full_w), common1),
            ],
        };

        (cfg, model, rope)
    }

    #[test]
    fn test_forward_step_q8_neon_into_migration_preserves_nonzero_logits() {
        let (cfg, model, rope) = make_nonzero_q8_neon_test_model();
        let num_linear = cfg.num_linear_attention_layers();
        let num_full = cfg.num_full_attention_layers();

        let run_two_steps = || -> Vec<f32> {
            let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
                .map(|_| GatedDeltaNetState::new(&cfg))
                .collect();
            let mut kv_cache = KvCache::new(num_full);
            let mut scratch = ForwardScratch::new();

            forward_step_q8_neon(
                &model,
                &cfg,
                &rope,
                7,
                0,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
            );
            kv_cache.seq_len += 1;
            forward_step_q8_neon(
                &model,
                &cfg,
                &rope,
                11,
                1,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
            );

            scratch.logits[..16].to_vec()
        };

        let logits_a = run_two_steps();
        let logits_b = run_two_steps();

        // Verify determinism (two identical runs must agree bitwise).
        assert_eq!(
            logits_a, logits_b,
            "forward_step_q8_neon is non-deterministic"
        );

        // Verify at least one non-zero logit (non-trivial model).
        assert!(
            logits_a.iter().any(|&v| v.abs() > 1e-9),
            "all 16 logits are zero — check weight generation"
        );

        // Baseline capture: hardcoded first 16 logits from the pre-migration allocating path.
        // Captured from `make_nonzero_q8_neon_test_model()` with seed 0xdead_beef_cafe_babe.
        // After i2 applies the zero-allocation migration, re-run this test; it must still pass.
        // To recapture: add `eprintln!("{:?}", &logits_a[..16]);` before this block, run test,
        // then update the array below.
        let expected: [f32; 16] = [
            -0.03680095,
            0.04277617,
            0.002856411,
            0.0072362088,
            -0.07445018,
            0.001189895,
            0.07319608,
            0.027846893,
            -0.02805086,
            -0.10936779,
            0.041943453,
            0.08327203,
            0.007971644,
            -0.079894274,
            0.01471648,
            0.028457977,
        ];
        for (i, (&actual, &exp)) in logits_a.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() <= 1e-6,
                "logit[{i}] mismatch: actual={actual:.8}, expected={exp:.8}"
            );
        }
    }

    #[test]
    fn test_all_projection_dims_are_multiples_of_32() {
        // Q8_0 requires K to be a multiple of 32. Verify all our dims qualify.
        let cfg = Qwen35Config::qwen35_2b();
        let dims_to_check = [
            ("hidden_size", cfg.hidden_size),
            ("intermediate_size", cfg.intermediate_size),
            ("full_q_dim", cfg.full_q_dim()),
            ("full_kv_dim", cfg.full_kv_dim()),
            ("2*full_q_dim", 2 * cfg.full_q_dim()),
            ("linear_qkv_dim", cfg.linear_qkv_dim()),
            ("linear_output_dim", cfg.linear_output_dim()),
        ];
        for (name, dim) in &dims_to_check {
            assert_eq!(
                dim % 32,
                0,
                "{name} = {dim} is not a multiple of 32 (required for Q8_0)"
            );
        }
    }
}
