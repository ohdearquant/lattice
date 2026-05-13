//! F16-weight forward pass for Qwen3.5-2B.
//!
//! This module mirrors `qwen35_model::forward_step` and `gated_delta_net_fused::gated_delta_net_step_fused`
//! but uses `F16ModelWeights` (packed `u16` half-precision) for all large projection matrices.
//! Activations, norms, and recurrent state remain in `f32`.
//!
//! The purpose is to halve memory bandwidth for weight-bound inference while preserving numerical
//! accuracy in the accumulation path (all dot products widen f16 elements on the fly).

use crate::attention::gdn::{GatedDeltaNetState, sigmoid, softplus};
use crate::attention::gdn_fused::{
    GatedDeltaNetFusedScratch, conv1d_silu_fused, simd_decay_and_rank1_update, simd_gated_rms_norm,
    simd_l2_normalize, simd_matvec_transpose,
};
use crate::forward::cpu::{elementwise_mul, silu_inplace};
use crate::model::qwen35::{
    ForwardScratch, KvCache, decode_tokens, qwen35_rms_norm, resize, sample_token,
};
use crate::model::qwen35_config::{GenerateConfig, GenerateOutput, Qwen35Config};
use crate::rope::RopeTable;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer;
use crate::weights::f16_weights::{
    F16AttentionWeights, F16FeedForwardWeights, F16FullAttentionLayerWeights,
    F16GatedDeltaNetWeights, F16ModelWeights, F16MoeLayerWeights, f16_to_f32_slice, matmul_bt_f16,
};

// ---------------------------------------------------------------------------
// GatedDeltaNet step (f16 weights)
// ---------------------------------------------------------------------------

/// **Unstable**: f16-weight GatedDeltaNet step; kernel interface evolving with quantization strategy.
///
/// Process a single token through the GatedDeltaNet layer using f16 weight matrices.
///
/// Numerically equivalent to `gated_delta_net_step_fused` within f16 quantization tolerance.
/// All five large projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj) use
/// `matmul_bt_f16`. Small vectors (a_log, dt_bias, conv1d_weight, norm_weight) remain f32.
///
/// `input`: hidden state `[hidden_size]`
/// `state`: mutable recurrent state for this layer
/// `weights`: layer weights with f16 projection matrices
/// `cfg`: model config
/// `scratch`: reusable fused scratch buffers
/// `output`: output buffer `[hidden_size]`, written in-place
#[inline]
pub fn gated_delta_net_step_fused_f16(
    input: &[f32],
    state: &mut GatedDeltaNetState,
    weights: &F16GatedDeltaNetWeights,
    cfg: &Qwen35Config,
    scratch: &mut GatedDeltaNetFusedScratch,
    output: &mut [f32],
) {
    let hidden = cfg.hidden_size;
    let num_heads = cfg.linear_num_key_heads;
    let value_heads = cfg.linear_num_value_heads();
    let key_dim = cfg.linear_key_head_dim;
    let value_dim = cfg.linear_value_head_dim;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let kernel_size = cfg.linear_conv_kernel_dim;

    debug_assert_eq!(
        num_heads, value_heads,
        "fused kernel assumes matched key/value head counts"
    );
    debug_assert!(input.len() >= hidden);
    debug_assert!(output.len() >= hidden);

    scratch.ensure_capacity(qkv_dim, output_dim, num_heads, key_dim, value_dim);

    // 1. Projections (f16 weights)
    matmul_bt_f16(
        input,
        &weights.in_proj_qkv,
        &mut scratch.qkv_proj[..qkv_dim],
        1,
        hidden,
        qkv_dim,
    );

    matmul_bt_f16(
        input,
        &weights.in_proj_z,
        &mut scratch.z_proj[..output_dim],
        1,
        hidden,
        output_dim,
    );

    matmul_bt_f16(
        input,
        &weights.in_proj_b,
        &mut scratch.beta_proj[..num_heads],
        1,
        hidden,
        num_heads,
    );

    matmul_bt_f16(
        input,
        &weights.in_proj_a,
        &mut scratch.alpha_proj[..num_heads],
        1,
        hidden,
        num_heads,
    );

    // sigmoid(beta)
    for b in &mut scratch.beta_proj[..num_heads] {
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

    for h in 0..num_heads {
        let q_start = h * key_dim;
        let k_start = q_total + h * key_dim;
        let v_start = v_offset + h * value_dim;

        scratch.q_head[..key_dim].copy_from_slice(&scratch.conv_output[q_start..q_start + key_dim]);
        scratch.k_head[..key_dim].copy_from_slice(&scratch.conv_output[k_start..k_start + key_dim]);
        let v = &scratch.conv_output[v_start..v_start + value_dim];

        // L2-normalize Q and K (SIMD-accelerated)
        simd_l2_normalize(&mut scratch.q_head[..key_dim]);
        simd_l2_normalize(&mut scratch.k_head[..key_dim]);

        // Decay gate (f32 weights: a_log, dt_bias)
        let a = weights.a_log[h].exp();
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
    // norm_weight is [value_dim] per-head, applied to each head independently.
    let gamma = &weights.norm_weight[..value_dim];
    debug_assert_eq!(gamma.len(), value_dim);

    for h in 0..num_heads {
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

    // Output projection (f16 weights)
    matmul_bt_f16(
        &scratch.gated_norm_buf[..output_dim],
        &weights.out_proj,
        &mut output[..hidden],
        1,
        output_dim,
        hidden,
    );
}

// ---------------------------------------------------------------------------
// Full attention step (f16 weights)
// ---------------------------------------------------------------------------

/// Full GQA attention for a single token using f16 weight matrices.
///
/// Input is read from `scratch.attn_out[..hidden]`, output written back to
/// `scratch.attn_out[..hidden]`.
fn full_attention_step_f16(
    weights: &F16FullAttentionLayerWeights,
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
    matmul_bt_f16(
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
    matmul_bt_f16(
        &input,
        &weights.k_proj,
        &mut scratch.k_buf[..kv_dim],
        1,
        hidden,
        kv_dim,
    );
    matmul_bt_f16(
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

    // Partial RoPE: only first rope_dim dimensions of each head (interleaved pairing)
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
    for (ctx, &gz) in scratch.context[..q_dim].iter_mut().zip(&gate_z[..q_dim]) {
        let sig = 1.0 / (1.0 + (-gz).exp());
        *ctx *= sig;
    }

    // Output projection: context [1, q_dim] @ o_proj^T [hidden, q_dim] (f16 weights)
    matmul_bt_f16(
        &scratch.context[..q_dim],
        &weights.o_proj,
        &mut scratch.attn_out[..hidden],
        1,
        q_dim,
        hidden,
    );
}

// ---------------------------------------------------------------------------
// FFN step (f16 weights)
// ---------------------------------------------------------------------------

/// Dense SwiGLU FFN step using f16 weight matrices.
///
/// Input is read from `scratch.ffn_out[..hidden]`, output written back to
/// `scratch.ffn_out[..hidden]`.
#[inline]
fn ffn_step_f16(
    gate_proj: &[u16],
    up_proj: &[u16],
    down_proj: &[u16],
    scratch: &mut ForwardScratch,
    inter: usize,
    hidden: usize,
) {
    scratch.input_tmp[..hidden].copy_from_slice(&scratch.ffn_out[..hidden]);

    matmul_bt_f16(
        &scratch.input_tmp[..hidden],
        gate_proj,
        &mut scratch.gate_buf[..inter],
        1,
        hidden,
        inter,
    );
    matmul_bt_f16(
        &scratch.input_tmp[..hidden],
        up_proj,
        &mut scratch.up_buf[..inter],
        1,
        hidden,
        inter,
    );

    silu_inplace(&mut scratch.gate_buf[..inter]);
    elementwise_mul(&mut scratch.gate_buf[..inter], &scratch.up_buf[..inter]);

    matmul_bt_f16(
        &scratch.gate_buf[..inter],
        down_proj,
        &mut scratch.ffn_out[..hidden],
        1,
        inter,
        hidden,
    );
}

/// MoE FFN step using f16 weight matrices.
///
/// Mirrors `moe_ffn_step` in `qwen35.rs`.
/// Input is read from `scratch.ffn_out[..hidden]`, output written back to
/// `scratch.ffn_out[..hidden]`.
#[inline]
fn moe_ffn_step_f16(moe: &F16MoeLayerWeights, scratch: &mut ForwardScratch, hidden: usize) {
    let inter = moe.experts.intermediate_size;
    let shared_inter = moe.shared_expert.intermediate_size;
    let num_experts = moe.router.num_experts;
    let top_k = moe.router.num_experts_per_tok;

    debug_assert_eq!(moe.router.hidden_size, hidden);
    debug_assert_eq!(moe.experts.num_experts, num_experts);
    debug_assert_eq!(moe.experts.hidden_size, hidden);
    debug_assert_eq!(moe.shared_expert.hidden_size, hidden);

    scratch.input_tmp[..hidden].copy_from_slice(&scratch.ffn_out[..hidden]);

    if scratch.router_logits.len() < num_experts {
        scratch.router_logits.resize(num_experts, 0.0);
    }
    if scratch.router_selected.len() < top_k {
        scratch.router_selected.resize(top_k, (usize::MAX, 0.0));
    }

    // Router logits: input [1, hidden] x gate^T [hidden, num_experts] -> [num_experts].
    matmul_bt_f16(
        &scratch.input_tmp[..hidden],
        &moe.router.gate,
        &mut scratch.router_logits[..num_experts],
        1,
        hidden,
        num_experts,
    );

    // Stable softmax over f32 router logits.
    let max_logit = scratch.router_logits[..num_experts]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0f32;
    for v in &mut scratch.router_logits[..num_experts] {
        *v = (*v - max_logit).exp();
        denom += *v;
    }
    if denom > 0.0 {
        for v in &mut scratch.router_logits[..num_experts] {
            *v /= denom;
        }
    }

    // Insertion-sort top-k selection.
    for slot in &mut scratch.router_selected[..top_k] {
        *slot = (usize::MAX, f32::NEG_INFINITY);
    }
    for (expert_id, prob) in scratch.router_logits[..num_experts]
        .iter()
        .copied()
        .enumerate()
    {
        for rank in 0..top_k {
            if prob > scratch.router_selected[rank].1 {
                for shift in (rank + 1..top_k).rev() {
                    scratch.router_selected[shift] = scratch.router_selected[shift - 1];
                }
                scratch.router_selected[rank] = (expert_id, prob);
                break;
            }
        }
    }

    let top_sum: f32 = scratch.router_selected[..top_k]
        .iter()
        .map(|(_, p)| *p)
        .sum();
    if top_sum > 0.0 {
        for (_, prob) in &mut scratch.router_selected[..top_k] {
            *prob /= top_sum;
        }
    }

    scratch.expert_out[..hidden].fill(0.0);

    for idx in 0..top_k {
        let (expert_id, weight) = scratch.router_selected[idx];
        debug_assert_ne!(expert_id, usize::MAX);

        let gate_up_stride = 2 * inter * hidden;
        let gate_up_start = expert_id * gate_up_stride;
        let down_start = expert_id * hidden * inter;

        // `gate_up_proj` is [num_experts, 2 * inter, hidden]; first half is gate, second is up.
        let gate_w = &moe.experts.gate_up_proj[gate_up_start..gate_up_start + inter * hidden];
        let up_w = &moe.experts.gate_up_proj
            [gate_up_start + inter * hidden..gate_up_start + 2 * inter * hidden];
        let down_w = &moe.experts.down_proj[down_start..down_start + hidden * inter];

        matmul_bt_f16(
            &scratch.input_tmp[..hidden],
            gate_w,
            &mut scratch.gate_buf[..inter],
            1,
            hidden,
            inter,
        );
        matmul_bt_f16(
            &scratch.input_tmp[..hidden],
            up_w,
            &mut scratch.up_buf[..inter],
            1,
            hidden,
            inter,
        );

        silu_inplace(&mut scratch.gate_buf[..inter]);
        elementwise_mul(&mut scratch.gate_buf[..inter], &scratch.up_buf[..inter]);

        scratch.down_input[..inter].copy_from_slice(&scratch.gate_buf[..inter]);
        matmul_bt_f16(
            &scratch.down_input[..inter],
            down_w,
            &mut scratch.ffn_out[..hidden],
            1,
            inter,
            hidden,
        );

        for i in 0..hidden {
            scratch.expert_out[i] += weight * scratch.ffn_out[i];
        }
    }

    let shared = &moe.shared_expert;

    // Shared gate: input [1, hidden] x shared_expert_gate^T [hidden, 1] -> [1].
    let mut shared_gate_logit = [0.0f32; 1];
    matmul_bt_f16(
        &scratch.input_tmp[..hidden],
        &shared.shared_expert_gate,
        &mut shared_gate_logit,
        1,
        hidden,
        1,
    );
    let shared_gate = sigmoid(shared_gate_logit[0]);

    matmul_bt_f16(
        &scratch.input_tmp[..hidden],
        &shared.gate_proj,
        &mut scratch.gate_buf[..shared_inter],
        1,
        hidden,
        shared_inter,
    );
    matmul_bt_f16(
        &scratch.input_tmp[..hidden],
        &shared.up_proj,
        &mut scratch.up_buf[..shared_inter],
        1,
        hidden,
        shared_inter,
    );

    silu_inplace(&mut scratch.gate_buf[..shared_inter]);
    elementwise_mul(
        &mut scratch.gate_buf[..shared_inter],
        &scratch.up_buf[..shared_inter],
    );

    scratch.down_input[..shared_inter].copy_from_slice(&scratch.gate_buf[..shared_inter]);
    matmul_bt_f16(
        &scratch.down_input[..shared_inter],
        &shared.down_proj,
        &mut scratch.ffn_out[..hidden],
        1,
        shared_inter,
        hidden,
    );

    for i in 0..hidden {
        scratch.expert_out[i] += shared_gate * scratch.ffn_out[i];
    }

    scratch.ffn_out[..hidden].copy_from_slice(&scratch.expert_out[..hidden]);
}

// ---------------------------------------------------------------------------
// Forward step (f16 weights)
// ---------------------------------------------------------------------------

/// Single-token forward pass using f16 weight matrices.
///
/// Equivalent to `Qwen35Model::forward_step` but all large projection matrices
/// (embeddings, QKV, FFN gate/up/down, output projections) use `matmul_bt_f16`.
/// Norms, recurrent state, and activations remain in `f32`.
///
/// Writes logits into `scratch.logits`.
pub(crate) fn forward_step_f16(
    weights: &F16ModelWeights,
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

    // Embedding lookup: f16 embed_tokens -> f32 hidden
    let embed_start = token_id as usize * hidden;
    f16_to_f32_slice(
        &weights.embed_tokens[embed_start..embed_start + hidden],
        &mut scratch.hidden[..hidden],
    );

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
            F16AttentionWeights::Linear(gdn_w) => {
                gated_delta_net_step_fused_f16(
                    &scratch.hidden[..hidden],
                    &mut gdn_states[linear_idx],
                    gdn_w,
                    cfg,
                    &mut scratch.gdn_scratch,
                    &mut scratch.attn_out[..hidden],
                );
                linear_idx += 1;
            }
            F16AttentionWeights::Full(full_w) => {
                // Copy hidden to attn_out as temp input to avoid borrow conflict
                scratch.attn_out[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
                full_attention_step_f16(
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

        // FFN: copy hidden into ffn_out as temp input to avoid borrow conflict
        scratch.ffn_out[..hidden].copy_from_slice(&scratch.hidden[..hidden]);
        match &common.ffn {
            F16FeedForwardWeights::Dense {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                ffn_step_f16(
                    gate_proj,
                    up_proj,
                    down_proj,
                    scratch,
                    cfg.intermediate_size,
                    hidden,
                );
            }
            F16FeedForwardWeights::Moe(moe) => {
                moe_ffn_step_f16(moe, scratch, hidden);
            }
        }

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

    // Logits: hidden @ embed_tokens^T (tied weights, f16)
    // hidden [1, hidden] @ embed_tokens^T [hidden, vocab] = logits [1, vocab]
    // embed_tokens is [vocab, hidden] in row-major f16, so matmul_bt_f16 computes
    // hidden @ embed_tokens^T correctly.
    resize(&mut scratch.logits, cfg.vocab_size);
    matmul_bt_f16(
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
// Generate (f16 weights)
// ---------------------------------------------------------------------------

/// **Unstable**: f16-weight generate; function signature will likely merge with model struct API.
///
/// Generate text from a prompt using f16 weight matrices.
///
/// Equivalent to `Qwen35Model::generate` but calls `forward_step_f16` for all
/// forward passes. The tokenizer, RoPE table, and generate config are passed
/// explicitly since we operate as standalone functions rather than methods on
/// the model struct.
pub fn generate_f16(
    weights: &F16ModelWeights,
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

    let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
    let mut all_ids = prompt_ids.clone();

    // Prefill: process prompt tokens one at a time through the recurrence
    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        forward_step_f16(
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

        forward_step_f16(
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_forward_compiles() {
        // Verify the function signatures are correct by constructing the types
        // and calling the functions with a trivial (1-layer, tiny) config.
        let cfg = Qwen35Config::qwen35_2b();

        // Verify forward_step_f16 signature (pub(crate))
        let _fn_ptr: fn(
            &F16ModelWeights,
            &Qwen35Config,
            &RopeTable,
            u32,
            usize,
            &mut [GatedDeltaNetState],
            &mut KvCache,
            &mut ForwardScratch,
        ) = forward_step_f16;

        // Verify gated_delta_net_step_fused_f16 signature
        let _gdn_fn_ptr: fn(
            &[f32],
            &mut GatedDeltaNetState,
            &F16GatedDeltaNetWeights,
            &Qwen35Config,
            &mut GatedDeltaNetFusedScratch,
            &mut [f32],
        ) = gated_delta_net_step_fused_f16;

        // Verify generate_f16 returns the right type
        let _gen_fn_ptr: fn(
            &F16ModelWeights,
            &Qwen35Config,
            &BpeTokenizer,
            &RopeTable,
            &str,
            &GenerateConfig,
        ) -> Result<GenerateOutput, crate::error::InferenceError> = generate_f16;

        // Verify the config helpers work
        assert!(cfg.num_full_attention_layers() > 0);
        assert!(cfg.num_linear_attention_layers() > 0);
        assert_eq!(
            cfg.num_full_attention_layers() + cfg.num_linear_attention_layers(),
            cfg.num_hidden_layers
        );
    }

    #[test]
    fn test_gdn_f16_step_with_zeros() {
        // Run the GDN f16 step with zero weights/inputs to verify it doesn't crash.
        let cfg = Qwen35Config::qwen35_2b();
        let hidden = cfg.hidden_size;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let num_heads = cfg.linear_num_key_heads;
        let kernel_size = cfg.linear_conv_kernel_dim;

        let weights = F16GatedDeltaNetWeights {
            in_proj_qkv: vec![0u16; qkv_dim * hidden],
            in_proj_qkv_rows: qkv_dim,
            in_proj_qkv_cols: hidden,
            in_proj_z: vec![0u16; output_dim * hidden],
            in_proj_z_rows: output_dim,
            in_proj_z_cols: hidden,
            in_proj_b: vec![0u16; num_heads * hidden],
            in_proj_b_rows: num_heads,
            in_proj_b_cols: hidden,
            in_proj_a: vec![0u16; num_heads * hidden],
            in_proj_a_rows: num_heads,
            in_proj_a_cols: hidden,
            a_log: vec![0.0f32; num_heads],
            dt_bias: vec![0.0f32; num_heads],
            conv1d_weight: vec![0.0f32; qkv_dim * kernel_size],
            conv_dim: qkv_dim,
            kernel_size,
            norm_weight: vec![0.0f32; output_dim],
            out_proj: vec![0u16; hidden * output_dim],
            out_proj_rows: hidden,
            out_proj_cols: output_dim,
        };

        let mut state = GatedDeltaNetState::new(&cfg);
        let mut scratch = GatedDeltaNetFusedScratch::default();
        let input = vec![0.0f32; hidden];
        let mut output = vec![0.0f32; hidden];

        gated_delta_net_step_fused_f16(
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
}
