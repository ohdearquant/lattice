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
    should_stop_token,
};
use crate::model::qwen35_config::{GenerateConfig, GenerateOutput, Qwen35Config};
use crate::rope::RopeTable;
use crate::stop_reason::StopReason;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer;
use crate::vision::multimodal::Qwen35VisionRequest;
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
    let ratio = value_heads / num_heads;
    let key_dim = cfg.linear_key_head_dim;
    let value_dim = cfg.linear_value_head_dim;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let kernel_size = cfg.linear_conv_kernel_dim;

    debug_assert!(input.len() >= hidden);
    debug_assert!(output.len() >= hidden);

    scratch.ensure_capacity(qkv_dim, output_dim, value_heads, key_dim, value_dim);

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
        &mut scratch.beta_proj[..value_heads],
        1,
        hidden,
        value_heads,
    );

    matmul_bt_f16(
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

        // Decay gate (f32 weights: a_log, dt_bias). Clamp exp(a_log) to finite
        // (mirror gdn.rs compute_decay_gate): a_log>~88 -> +inf, inf*0 = NaN poisons state.
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
    // norm_weight is [value_dim] per-head, applied to each head independently.
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
    mrope_cos_sin: Option<(&[f32], &[f32])>,
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

    // Partial RoPE: stride-half pairing (i, half+i) — matches apply_partial_rope / HF rotate_half.
    // When `mrope_cos_sin` is supplied (Qwen3.5 vision M-RoPE, ADR-069 S5b), the per-token
    // interleaved-axis cos/sin row replaces the 1-D `rope` table lookup; the rotation formula
    // is identical either way, so text-only decode (mrope_cos_sin=None) is untouched.
    let half = rope_dim / 2;
    for h in 0..num_q_heads {
        let start = h * head_dim;
        if let Some((cos_row, sin_row)) = mrope_cos_sin {
            for i in 0..half {
                let cos_val = cos_row[i];
                let sin_val = sin_row[i];
                let x0 = scratch.q_buf[start + i];
                let x1 = scratch.q_buf[start + half + i];
                scratch.q_buf[start + i] = x0 * cos_val - x1 * sin_val;
                scratch.q_buf[start + half + i] = x0 * sin_val + x1 * cos_val;
            }
        } else {
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
    }
    for h in 0..num_kv_heads {
        let start = h * head_dim;
        if let Some((cos_row, sin_row)) = mrope_cos_sin {
            for i in 0..half {
                let cos_val = cos_row[i];
                let sin_val = sin_row[i];
                let x0 = scratch.k_buf[start + i];
                let x1 = scratch.k_buf[start + half + i];
                scratch.k_buf[start + i] = x0 * cos_val - x1 * sin_val;
                scratch.k_buf[start + half + i] = x0 * sin_val + x1 * cos_val;
            }
        } else {
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

        for t in 0..cur_seq_len {
            let k_off = t * kv_dim + kvh * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[d] * k_cache[k_off + d];
            }
            scratch.scores[scores_start + t] = dot * scale;
        }

        // ADR-080 C1: route the fail-closed final decision through the
        // shared row-finalizer contract (#780). RED before the fix: the
        // bare `1.0 / sum_exp` had no guard, so a NaN/`+inf` cached score
        // propagated NaN into the context output instead of failing the
        // row closed. Mirrors the byte-identical duplicate in
        // `model::qwen35::forward::compute_attention_context`.
        let row = &mut scratch.scores[scores_start..scores_start + cur_seq_len];
        let (max_score, any_nan) = crate::attention::softmax_row::row_max_and_any_nan(row);
        if crate::attention::softmax_row::row_fails_closed_pre_exp(max_score, any_nan) {
            row.fill(0.0);
        } else {
            let mut sum_exp = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_score).exp();
                sum_exp += *v;
            }
            crate::attention::softmax_row::finalize_row(row, sum_exp);
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
    } else {
        // Fail closed on a non-finite denom (NaN/±inf router logit from a corrupt
        // f16 router gate weight or an upstream activation overflow), mirroring
        // the f32 router fix in qwen35/moe.rs::compute_router_probs and
        // generate.rs compute_attention (#409/#410). `max_logit` can stay finite
        // when only one lane is NaN (Rust `f32::max` ignores a single NaN), so
        // the NaN propagates into `denom` here and `denom > 0.0` is false.
        // Without this the router would leave un-normalized raw `exp` values and,
        // worse, an all-NaN row selects nothing below (`NaN > NEG_INF` is false),
        // leaving a `usize::MAX` sentinel that overflows expert indexing. Zeroing
        // drops the routed path (the shared expert still runs).
        scratch.router_logits[..num_experts].fill(0.0);
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
        // Defense in depth: never index expert weights with an unfilled sentinel
        // slot or a router-only expert id. A degenerate router row can leave
        // `(usize::MAX, _)` for a rank it could not fill, and a public f16 weight
        // set may declare more router experts than routed-expert storage; either
        // way `expert_id * gate_up_stride` would overflow / OOB the `moe.experts`
        // slices below. Bound on the storage count `moe.experts.num_experts`,
        // matching the f32 sibling (qwen35/moe.rs::accumulate_routed_experts).
        if expert_id >= moe.experts.num_experts {
            continue;
        }
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
    injected_embedding: Option<&[f32]>,
    mrope_cos_sin: Option<(&[f32], &[f32])>,
) -> Result<(), crate::error::InferenceError> {
    let hidden = cfg.hidden_size;

    scratch.ensure_capacity(cfg, kv_cache.seq_len + 1);

    match injected_embedding {
        // Qwen3.5 vision M-RoPE (ADR-069 S5b): REPLACE the token-embedding lookup with the
        // caller-supplied post-merger visual row at an `<|image_pad|>` slot (HF's
        // `masked_scatter` contract). Fail closed rather than poison the KV state with a
        // wrong-shape or non-finite row.
        Some(row) => {
            if row.len() != hidden {
                return Err(crate::error::InferenceError::InvalidInput(format!(
                    "injected_embedding length {} does not match hidden_size {hidden}",
                    row.len()
                )));
            }
            if let Some(bad) = row.iter().find(|v| !v.is_finite()) {
                return Err(crate::error::InferenceError::InvalidInput(format!(
                    "injected_embedding contains a non-finite value: {bad}"
                )));
            }
            scratch.hidden[..hidden].copy_from_slice(row);
        }
        None => {
            // Embedding lookup: f16 embed_tokens -> f32 hidden
            let embed_start = token_id as usize * hidden;
            f16_to_f32_slice(
                &weights.embed_tokens[embed_start..embed_start + hidden],
                &mut scratch.hidden[..hidden],
            );
        }
    }

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
                    mrope_cos_sin,
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

    Ok(())
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

    // #856: single shared preflight, unified with the Metal paths (which
    // used to accept an empty prompt and return an empty Ok) — see
    // `check_prompt_not_empty` (model::qwen35::generation).
    crate::model::qwen35::check_prompt_not_empty(prompt_len)?;

    // Sibling of `generate_multimodal_f16`'s input-id guard: tokenizer
    // output is bounded by the tokenizer's own vocabulary, but this driver accepts
    // `cfg` and `tokenizer` as independent parameters, so a mismatched pair can still
    // produce a prompt id at or past `cfg.vocab_size` and panic in `forward_step_f16`'s
    // embedding-table slice. Fail closed here, once, before any decoder state is
    // allocated.
    if let Some(&bad_id) = prompt_ids.iter().find(|&&id| id as usize >= cfg.vocab_size) {
        return Err(crate::error::InferenceError::InvalidInput(format!(
            "prompt contains out-of-vocabulary token id {bad_id} (vocab_size={})",
            cfg.vocab_size
        )));
    }

    // max_new_tokens == 0 means "generate nothing": return before sampling so
    // we never emit a token the caller did not ask for. Mirrors the guard in
    // Qwen35Model::generate (crates/inference/src/model/qwen35/generation.rs).
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

    // Reject grammar configs before allocating any state. Grammar masking
    // (`mask_logits` + `advance`) is not wired into this generate loop; without
    // the guard the grammar field would be silently ignored, producing
    // unconstrained output despite a grammar being set (#397/#398).
    crate::model::qwen35::check_grammar_not_set(gen_cfg)?;
    // Same rationale for logprobs capture, which is also not wired into this
    // generate loop (#585).
    crate::model::qwen35::check_logprobs_not_set(gen_cfg)?;
    // Same rationale for stop_strings matching and reasoning-budget forcing,
    // neither of which is wired into this generate loop (ADR-080 C3, #783).
    crate::model::qwen35::check_stop_strings_not_set(gen_cfg)?;
    crate::model::qwen35::check_reasoning_budget_not_set(gen_cfg)?;

    // Context preflight. The RoPE cos/sin tables are indexed unchecked in
    // forward_step_f16 (`rope.cos_at(base + i)`), so a position at or past the
    // table capacity is an out-of-bounds slice access — a release panic, not a
    // clean error. Mirror Qwen35Model::generate's total-token policy
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
        forward_step_f16(
            weights,
            cfg,
            rope,
            token_id,
            pos,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
            None,
            None,
        )?;
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
            token_logprobs: vec![],
        });
    }

    generated_ids.push(next_id);
    all_ids.push(next_id);

    let mut stopped = false;
    let mut stop_reason = StopReason::Length;
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
            None,
            None,
        )?;
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
        token_logprobs: vec![],
    })
}

// ---------------------------------------------------------------------------
// Generate multimodal (f16 weights, ADR-069 Stage 5b)
// ---------------------------------------------------------------------------

/// Greedy-decode a Qwen3.5 vision-language prompt through the CPU f16 forward
/// path (ADR-069 Stage 5b): the decoder splice on top of [`generate_f16`].
///
/// Mirrors `generate_f16`'s prefill/decode loop, but drives it from
/// [`crate::vision::multimodal::Qwen35VisionRequest`]'s already-expanded
/// `input_ids` instead of a tokenizer call, injects each post-merger visual
/// row at its `<|image_pad|>` slot (masked REPLACE, not add), and threads a
/// per-token M-RoPE cos/sin row into every full-attention (GQA) layer in
/// place of the 1-D `RopeTable`. GDN layers are untouched — they never
/// receive rope. Fails closed via `request.validate()` before any decoder
/// work begins.
///
/// No tokenizer is available here (the request already carries expanded
/// token ids), so `GenerateOutput.text` is always empty; callers that need
/// decoded text detokenize `token_ids` themselves.
pub fn generate_multimodal_f16(
    weights: &F16ModelWeights,
    cfg: &Qwen35Config,
    request: &crate::vision::multimodal::Qwen35VisionRequest,
    gen_cfg: &GenerateConfig,
) -> Result<GenerateOutput, crate::error::InferenceError> {
    request.validate().map_err(|e| {
        crate::error::InferenceError::InvalidInput(format!(
            "multimodal request failed validation: {e}"
        ))
    })?;

    // Caller-supplied `input_ids` must be bounded against the
    // checkpoint vocabulary before any decoder allocation/work begins — an
    // out-of-range id would otherwise panic in `forward_step_f16`'s embedding-table
    // slice (`token_id * hidden`) instead of failing closed.
    if let Some(&bad_id) = request
        .input_ids
        .iter()
        .find(|&&id| id as usize >= cfg.vocab_size)
    {
        return Err(crate::error::InferenceError::InvalidInput(format!(
            "input_ids contains out-of-vocabulary token id {bad_id} (vocab_size={})",
            cfg.vocab_size
        )));
    }

    let has_image = !request.image_grids.is_empty();

    // `request.validate()` proves only internal consistency —
    // bind the request to the *loaded checkpoint's* vision metadata before
    // selecting image slots or building M-RoPE tables, otherwise an internally
    // consistent request targeting the wrong checkpoint silently injects rows at
    // the wrong slots / applies image M-RoPE where HF would treat them as text.
    if has_image {
        let cfg_image_token_id = cfg.image_token_id.ok_or_else(|| {
            crate::error::InferenceError::InvalidInput(
                "multimodal request supplied but checkpoint has no image_token_id".to_string(),
            )
        })?;
        if cfg_image_token_id != request.image_token_id {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "request image_token_id {} does not match checkpoint image_token_id {cfg_image_token_id}",
                request.image_token_id
            )));
        }
        let vision_cfg = cfg.vision_config.as_ref().ok_or_else(|| {
            crate::error::InferenceError::InvalidInput(
                "multimodal request supplied but checkpoint has no vision_config".to_string(),
            )
        })?;
        if vision_cfg.spatial_merge_size != request.spatial_merge_size {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "request spatial_merge_size {} does not match checkpoint \
                 vision_config.spatial_merge_size {}",
                request.spatial_merge_size, vision_cfg.spatial_merge_size
            )));
        }
        if request.decoder_hidden_size != cfg.hidden_size {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "request decoder_hidden_size {} does not match checkpoint hidden_size {}",
                request.decoder_hidden_size, cfg.hidden_size
            )));
        }
        if vision_cfg.out_hidden_size != cfg.hidden_size {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "checkpoint vision_config.out_hidden_size {} does not match decoder \
                 hidden_size {}",
                vision_cfg.out_hidden_size, cfg.hidden_size
            )));
        }
    }

    let (positions, tables) = request.build_mrope_tables(cfg)?;

    // The M-RoPE table builder resolves `partial_rotary_factor`
    // from `cfg.rope_parameters`, while the attention loop derives its rotary
    // half-width from the separately public `cfg.rope_dim()`
    // (`cfg.partial_rotary_factor`). A constructible config where these diverge
    // must fail closed here, before the first forward pass indexes `cos_row`/
    // `sin_row` past the table's actual row width.
    let expected_rope_half = cfg.rope_dim() / 2;
    if tables.cos.iter().any(|row| row.len() != expected_rope_half)
        || tables.sin.iter().any(|row| row.len() != expected_rope_half)
    {
        return Err(crate::error::InferenceError::InvalidInput(format!(
            "M-RoPE table row width does not match decoder rotary half-width: expected \
             {expected_rope_half}"
        )));
    }

    let prompt_ids = &request.input_ids;
    let prompt_len = prompt_ids.len();
    crate::model::qwen35::check_prompt_not_empty(prompt_len)?;

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

    crate::model::qwen35::check_grammar_not_set(gen_cfg)?;
    crate::model::qwen35::check_logprobs_not_set(gen_cfg)?;
    crate::model::qwen35::check_stop_strings_not_set(gen_cfg)?;
    crate::model::qwen35::check_reasoning_budget_not_set(gen_cfg)?;

    let max_context = cfg.max_position_embeddings;
    if prompt_len.saturating_add(gen_cfg.max_new_tokens) > max_context {
        return Err(crate::error::InferenceError::Inference(format!(
            "prompt ({prompt_len} tokens) plus max_new_tokens ({}) exceeds \
             model context window ({max_context})",
            gen_cfg.max_new_tokens
        )));
    }

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

    let num_linear = cfg.num_linear_attention_layers();
    let num_full = cfg.num_full_attention_layers();
    let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
        .map(|_| GatedDeltaNetState::new(cfg))
        .collect();
    let mut kv_cache = KvCache::new(num_full);
    let mut scratch = ForwardScratch::new();

    // A request with no image runs needs no M-RoPE divergence: every position's 3 axes are
    // trivially equal to the sequential index (rope_delta=0), so the plain 1-D `RopeTable`
    // reproduces `build_cos_sin`'s output exactly in the f64-precomputed-table sense —
    // `generate_f16`'s own path — rather than the fresh f32 per-call computation, which is
    // only mathematically (not bit-) equivalent. Route text-only requests through the
    // unchanged 1-D table so they are bit-identical to `generate_f16`, and reserve the M-RoPE
    // table for requests that actually contain an image (where axes genuinely diverge).
    // `has_image` was already computed above (before `build_mrope_tables`) for the
    // checkpoint-binding guard; reused here rather than recomputed.
    let rope = RopeTable::new(cfg.rope_dim(), max_context, cfg.rope_theta);

    let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
    let mut all_ids = prompt_ids.clone();

    // Prefill: inject a post-merger visual row at each `<|image_pad|>` slot, in the same
    // sequential image-token order the request already concatenated `post_merger_rows` in
    // (HF's masked_scatter contract, ADR-069 RECON sec. 1).
    let mut visual_row = 0usize;
    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        let injected = if token_id == request.image_token_id {
            let start = visual_row * request.decoder_hidden_size;
            let end = start + request.decoder_hidden_size;
            visual_row += 1;
            Some(&request.post_merger_rows[start..end])
        } else {
            None
        };
        let cos_sin = if has_image {
            Some((tables.cos[pos].as_slice(), tables.sin[pos].as_slice()))
        } else {
            None
        };

        forward_step_f16(
            weights,
            cfg,
            &rope,
            token_id,
            pos,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
            injected,
            cos_sin,
        )?;
        if pos < prompt_len - 1 {
            kv_cache.seq_len += 1;
        }
    }
    kv_cache.seq_len = prompt_len;

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
            token_logprobs: vec![],
        });
    }

    generated_ids.push(next_id);
    all_ids.push(next_id);

    let mut stopped = false;
    let mut stop_reason = StopReason::Length;
    // Autoregressive decode: the KV-cache index (`physical_pos`) stays contiguous/physical,
    // while the M-RoPE coordinate is `physical_cache_len + rope_delta` (ADR-069 RECON sec. 3) —
    // they diverge whenever the prompt contained an image.
    for _ in 1..gen_cfg.max_new_tokens {
        let physical_pos = kv_cache.seq_len;
        let last_token = *all_ids
            .last()
            .expect("invariant: prompt or previous sample populated all_ids");

        // Owns the M-RoPE row buffers for this iteration so both branches below can borrow
        // from a value with the same lifetime as the `forward_step_f16` call.
        let decode_cos_sin;
        let mrope_cos_sin = if has_image {
            let decode_axis =
                crate::vision::qwen35_mrope::decode_position(physical_pos, positions.rope_delta)?;
            decode_cos_sin = request.build_decode_cos_sin(cfg, decode_axis)?;
            // Decode-time sibling of the prefill row-width guard: the prefill table's row-width
            // guard has no effect on this independently-built single-row decode
            // table — check it here too, before it reaches the attention loop's
            // `cos_row[i]`/`sin_row[i]` indexing.
            if decode_cos_sin.0.len() != expected_rope_half
                || decode_cos_sin.1.len() != expected_rope_half
            {
                return Err(crate::error::InferenceError::InvalidInput(format!(
                    "decode-time M-RoPE row width does not match decoder rotary \
                     half-width: expected {expected_rope_half}"
                )));
            }
            Some((decode_cos_sin.0.as_slice(), decode_cos_sin.1.as_slice()))
        } else {
            None
        };

        forward_step_f16(
            weights,
            cfg,
            &rope,
            last_token,
            physical_pos,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
            None,
            mrope_cos_sin,
        )?;
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

    Ok(GenerateOutput {
        text: String::new(),
        token_ids: generated_ids.clone(),
        prompt_tokens: prompt_len,
        generated_tokens: generated_ids.len(),
        stopped,
        stop_reason: Some(stop_reason),
        token_logprobs: vec![],
    })
}

// ---------------------------------------------------------------------------
// Pooled embedding extraction (vision-embed-pooling): image + text, same
// decoder + same pooling, so both land in the same vector space (GME-style).
// ---------------------------------------------------------------------------

/// How to collapse a prefill's per-position hidden states into one
/// fixed-size embedding vector.
///
/// **Retrieval quality with the base Qwen3.5-0.8B *instruct* checkpoint is
/// unvalidated.** GME-style pooled embeddings normally come from a
/// checkpoint that has been contrastively fine-tuned for retrieval
/// (image-text matching, hard-negative mining); the base instruct checkpoint
/// was never trained for that objective. What this module provides — and
/// what is tested — is the extraction *machinery*: pooling over the
/// verifiably correct positions, deterministically, into a unit-norm
/// vector. Picking (or fine-tuning) a checkpoint for retrieval quality is a
/// separate, later decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Mean over the hidden states at the request's `<|image_pad|>`
    /// positions (the visual tokens). For a text-only request — no image
    /// runs, so no pad tokens are present — this degrades to a mean over
    /// every position, i.e. an ordinary mean-pooled text embedding rather
    /// than an error.
    MeanVisualTokens,
    /// The hidden state at the last physical position — GME's
    /// text-embedding convention. Well-defined for both image and
    /// text-only requests.
    LastToken,
}

/// Run prefill ONLY (no sampling, no decode loop) over `request.input_ids`,
/// injecting each post-merger visual row at its `<|image_pad|>` slot exactly
/// as [`generate_multimodal_f16`] does, and return every position's final
/// hidden state (post-final-norm, pre-lm_head projection) as a flat
/// row-major `[seq_len * cfg.hidden_size]` buffer.
///
/// Shares `generate_multimodal_f16`'s request validation, checkpoint-binding
/// checks, and M-RoPE/injection wiring; the only behavioral difference is
/// that this never samples a token, so it takes no [`GenerateConfig`].
///
/// # Errors
///
/// See [`generate_multimodal_f16`]'s error conditions — the same
/// `request.validate()`, out-of-vocabulary, checkpoint-binding, and M-RoPE
/// table checks apply here, minus the generation-only checks (grammar,
/// logprobs, stop strings, reasoning budget) that do not apply to a
/// prefill-only call.
pub fn prefill_hidden_states_f16(
    weights: &F16ModelWeights,
    cfg: &Qwen35Config,
    request: &Qwen35VisionRequest,
) -> Result<Vec<f32>, crate::error::InferenceError> {
    request.validate().map_err(|e| {
        crate::error::InferenceError::InvalidInput(format!(
            "multimodal request failed validation: {e}"
        ))
    })?;

    if let Some(&bad_id) = request
        .input_ids
        .iter()
        .find(|&&id| id as usize >= cfg.vocab_size)
    {
        return Err(crate::error::InferenceError::InvalidInput(format!(
            "input_ids contains out-of-vocabulary token id {bad_id} (vocab_size={})",
            cfg.vocab_size
        )));
    }

    let has_image = !request.image_grids.is_empty();

    // Mirrors generate_multimodal_f16's checkpoint-binding guard: an
    // internally consistent request can still target the wrong checkpoint.
    if has_image {
        let cfg_image_token_id = cfg.image_token_id.ok_or_else(|| {
            crate::error::InferenceError::InvalidInput(
                "multimodal request supplied but checkpoint has no image_token_id".to_string(),
            )
        })?;
        if cfg_image_token_id != request.image_token_id {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "request image_token_id {} does not match checkpoint image_token_id {cfg_image_token_id}",
                request.image_token_id
            )));
        }
        let vision_cfg = cfg.vision_config.as_ref().ok_or_else(|| {
            crate::error::InferenceError::InvalidInput(
                "multimodal request supplied but checkpoint has no vision_config".to_string(),
            )
        })?;
        if vision_cfg.spatial_merge_size != request.spatial_merge_size {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "request spatial_merge_size {} does not match checkpoint \
                 vision_config.spatial_merge_size {}",
                request.spatial_merge_size, vision_cfg.spatial_merge_size
            )));
        }
        if request.decoder_hidden_size != cfg.hidden_size {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "request decoder_hidden_size {} does not match checkpoint hidden_size {}",
                request.decoder_hidden_size, cfg.hidden_size
            )));
        }
        if vision_cfg.out_hidden_size != cfg.hidden_size {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "checkpoint vision_config.out_hidden_size {} does not match decoder \
                 hidden_size {}",
                vision_cfg.out_hidden_size, cfg.hidden_size
            )));
        }
    }

    // Reject empty and over-context prompts before build_mrope_tables, which
    // otherwise materializes a position entry plus cos/sin rows per supplied
    // token — unbounded input must fail cheaply, not after that allocation.
    let prompt_len = request.input_ids.len();
    crate::model::qwen35::check_prompt_not_empty(prompt_len)?;
    let max_context = cfg.max_position_embeddings;
    if prompt_len > max_context {
        return Err(crate::error::InferenceError::Inference(format!(
            "prompt ({prompt_len} tokens) exceeds model context window ({max_context})"
        )));
    }

    let (_positions, tables) = request.build_mrope_tables(cfg)?;

    let expected_rope_half = cfg.rope_dim() / 2;
    if tables.cos.iter().any(|row| row.len() != expected_rope_half)
        || tables.sin.iter().any(|row| row.len() != expected_rope_half)
    {
        return Err(crate::error::InferenceError::InvalidInput(format!(
            "M-RoPE table row width does not match decoder rotary half-width: expected \
             {expected_rope_half}"
        )));
    }

    let prompt_ids = &request.input_ids;

    let num_linear = cfg.num_linear_attention_layers();
    let num_full = cfg.num_full_attention_layers();
    let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
        .map(|_| GatedDeltaNetState::new(cfg))
        .collect();
    let mut kv_cache = KvCache::new(num_full);
    let mut scratch = ForwardScratch::new();

    // Text-only requests route through the plain 1-D RopeTable (cos_sin =
    // None below), bit-identical to generate_f16/generate_multimodal_f16;
    // only image-bearing requests use the M-RoPE table.
    let rope = RopeTable::new(cfg.rope_dim(), max_context, cfg.rope_theta);

    let hidden = cfg.hidden_size;
    let mut hidden_states: Vec<f32> = Vec::with_capacity(prompt_len * hidden);

    let mut visual_row = 0usize;
    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        let injected = if token_id == request.image_token_id {
            let start = visual_row * request.decoder_hidden_size;
            let end = start + request.decoder_hidden_size;
            visual_row += 1;
            Some(&request.post_merger_rows[start..end])
        } else {
            None
        };
        let cos_sin = if has_image {
            Some((tables.cos[pos].as_slice(), tables.sin[pos].as_slice()))
        } else {
            None
        };

        forward_step_f16(
            weights,
            cfg,
            &rope,
            token_id,
            pos,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
            injected,
            cos_sin,
        )?;

        hidden_states.extend_from_slice(&scratch.hidden[..hidden]);

        if pos < prompt_len - 1 {
            kv_cache.seq_len += 1;
        }
    }
    kv_cache.seq_len = prompt_len;

    Ok(hidden_states)
}

/// Mean-pool `hidden_states` (flat row-major `[seq_len, hidden_size]`) over
/// `positions`. Panics only on internal misuse (empty `positions` or an
/// out-of-range index), never on caller input — callers of this private
/// helper always derive `positions` from a validated request.
fn mean_pool_rows(hidden_states: &[f32], hidden_size: usize, positions: &[usize]) -> Vec<f32> {
    debug_assert!(!positions.is_empty());
    let mut out = vec![0.0f32; hidden_size];
    for &p in positions {
        let row = &hidden_states[p * hidden_size..(p + 1) * hidden_size];
        for (o, &v) in out.iter_mut().zip(row) {
            *o += v;
        }
    }
    let n = positions.len() as f32;
    for o in &mut out {
        *o /= n;
    }
    out
}

/// L2-normalize `v` in place; a zero or non-finite norm leaves `v`
/// unchanged rather than dividing by zero/NaN.
fn l2_normalize_owned(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 && norm.is_finite() {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Collapse `hidden_states` into one `[hidden_size]` vector per
/// [`PoolingStrategy`]. `image_pad_positions` is the (possibly empty) list
/// of physical positions holding an `<|image_pad|>` token, in ascending
/// order.
fn pool_hidden_states(
    hidden_states: &[f32],
    hidden_size: usize,
    seq_len: usize,
    image_pad_positions: &[usize],
    pooling: PoolingStrategy,
) -> Vec<f32> {
    match pooling {
        PoolingStrategy::LastToken => {
            hidden_states[(seq_len - 1) * hidden_size..seq_len * hidden_size].to_vec()
        }
        PoolingStrategy::MeanVisualTokens => {
            if image_pad_positions.is_empty() {
                let all: Vec<usize> = (0..seq_len).collect();
                mean_pool_rows(hidden_states, hidden_size, &all)
            } else {
                mean_pool_rows(hidden_states, hidden_size, image_pad_positions)
            }
        }
    }
}

/// Run prefill over an image + text [`Qwen35VisionRequest`] and return a
/// pooled, L2-normalized embedding of length `cfg.hidden_size` (2048 for the
/// Qwen3.5-0.8B checkpoint).
///
/// See [`PoolingStrategy`] for the honesty note on retrieval quality: this
/// function is the extraction machinery (correct positions, deterministic,
/// unit-norm output), not a claim about embedding quality.
///
/// # Errors
///
/// See [`prefill_hidden_states_f16`].
pub fn embed_image_f16(
    weights: &F16ModelWeights,
    cfg: &Qwen35Config,
    request: &Qwen35VisionRequest,
    pooling: PoolingStrategy,
) -> Result<Vec<f32>, crate::error::InferenceError> {
    let hidden_states = prefill_hidden_states_f16(weights, cfg, request)?;
    let seq_len = request.input_ids.len();
    let image_pad_positions: Vec<usize> = request
        .input_ids
        .iter()
        .enumerate()
        .filter(|&(_, &id)| id == request.image_token_id)
        .map(|(i, _)| i)
        .collect();
    let pooled = pool_hidden_states(
        &hidden_states,
        cfg.hidden_size,
        seq_len,
        &image_pad_positions,
        pooling,
    );
    Ok(l2_normalize_owned(pooled))
}

/// Tokenize `prompt` and run it through the same decoder + pooling path as
/// [`embed_image_f16`] (as a text-only [`Qwen35VisionRequest`] with no image
/// runs), so text and image embeddings from the same checkpoint land in the
/// same vector space. Returns a pooled, L2-normalized embedding of length
/// `cfg.hidden_size`.
///
/// Requires a vision-language checkpoint: `cfg.rope_parameters` must carry
/// an `mrope_section` (only vision-language configs set one) even though no
/// image is present, because this routes through the same
/// [`Qwen35VisionRequest`]-shaped prefill as the image path rather than a
/// separate code path — that shared path is the whole point (same decoder,
/// same pooling, same space).
///
/// # Errors
///
/// Returns [`crate::error::InferenceError::InvalidInput`] if the tokenized
/// prompt is empty or contains an out-of-vocabulary or (surprisingly) an
/// `image_token_id` token. See [`prefill_hidden_states_f16`] for the
/// remaining error conditions.
pub fn embed_text_vlm_f16(
    weights: &F16ModelWeights,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    pooling: PoolingStrategy,
) -> Result<Vec<f32>, crate::error::InferenceError> {
    let input = tokenizer.tokenize(prompt);
    let prompt_ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();
    crate::model::qwen35::check_prompt_not_empty(prompt_ids.len())?;

    if let Some(&bad_id) = prompt_ids.iter().find(|&&id| id as usize >= cfg.vocab_size) {
        return Err(crate::error::InferenceError::InvalidInput(format!(
            "prompt contains out-of-vocabulary token id {bad_id} (vocab_size={})",
            cfg.vocab_size
        )));
    }

    // image_token_id is only needed here to shape a well-formed (imageless)
    // Qwen35VisionRequest; u32::MAX is an unreachable sentinel when the
    // checkpoint has no vision config at all (in which case the request
    // below will simply never see that id).
    let image_token_id = cfg.image_token_id.unwrap_or(u32::MAX);
    if prompt_ids.contains(&image_token_id) {
        return Err(crate::error::InferenceError::InvalidInput(
            "tokenized prompt unexpectedly contains the checkpoint's image_token_id".to_string(),
        ));
    }

    let request = Qwen35VisionRequest {
        input_ids: prompt_ids,
        image_grids: vec![],
        post_merger_rows: vec![],
        image_token_id,
        spatial_merge_size: cfg
            .vision_config
            .as_ref()
            .map(|v| v.spatial_merge_size)
            .unwrap_or(2),
        decoder_hidden_size: cfg.hidden_size,
    };

    embed_image_f16(weights, cfg, &request, pooling)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::type_complexity)]
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
            Option<&[f32]>,
            Option<(&[f32], &[f32])>,
        ) -> Result<(), crate::error::InferenceError> = forward_step_f16;

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

    /// Regression test for #392: cpu F16 RoPE must use stride-half pairing (i, half+i), not
    /// interleaved (2i, 2i+1).
    ///
    /// Design: call `full_attention_step_f16` with an identity K-projection so k_buf equals
    /// the input exactly (1.0 in f16 is exact, no rounding error on the diagonal).
    /// Independently reproduce the same matmul + QK-norm + stride-half RoPE in the test body
    /// and compare against the post-call KV-cache. The two paths agree to <1e-4 when the
    /// production loops are correct; reverting either loop to 2*i interleaved produces
    /// max_diff ~0.9 (observed during mutation verification).
    #[test]
    fn test_full_attn_step_f16_rope_stride_half_parity() {
        use crate::model::qwen35_config::LayerType;
        use crate::weights::f16_weights::f32_to_f16_slice;

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
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        };

        let rope_dim = cfg.rope_dim(); // = 16
        let half = rope_dim / 2; // = 8
        let rope = RopeTable::new(rope_dim, 512, cfg.rope_theta);

        // Helper: convert f32 slice to packed f16 Vec<u16>.
        let to_f16 = |src: &[f32]| -> Vec<u16> {
            let mut dst = vec![0u16; src.len()];
            f32_to_f16_slice(src, &mut dst);
            dst
        };

        // W_k = identity [kv_dim, hidden]: row j selects input[j] exactly (1.0 in f16 is exact).
        let mut k_proj_f32 = vec![0.0f32; kv_dim * hidden];
        for j in 0..kv_dim {
            k_proj_f32[j * hidden + j] = 1.0;
        }

        // W_q = identity for first q_dim rows (Q part), zeros for next q_dim rows (gate part).
        // Row j selects input[j] exactly so scratch.q_buf is non-trivial and Q-loop mutation
        // changes the assertion result.
        let mut q_proj_f32 = vec![0.0f32; 2 * q_dim * hidden];
        for j in 0..q_dim {
            q_proj_f32[j * hidden + j] = 1.0;
        }

        let weights = F16FullAttentionLayerWeights {
            q_proj: to_f16(&q_proj_f32),
            k_proj: to_f16(&k_proj_f32),
            v_proj: to_f16(&vec![0.0f32; kv_dim * hidden]),
            o_proj: to_f16(&vec![0.0f32; hidden * q_dim]),
            q_norm: vec![0.0f32; head_dim],
            k_norm: vec![0.0f32; head_dim],
        };

        // Distinct non-trivial input values (positions 0..64 scaled to small floats).
        let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.07).collect();

        let mut scratch = ForwardScratch::new();
        scratch.ensure_capacity(&cfg, 2);
        scratch.attn_out[..hidden].copy_from_slice(&input);

        let mut kv_cache = KvCache::new(1);
        full_attention_step_f16(
            &weights,
            0,
            position,
            &mut kv_cache,
            &mut scratch,
            &cfg,
            &rope,
            hidden,
            None,
        );

        // Reference: reproduce the same matmul + QK-norm + stride-half RoPE.
        // Using the same production matmul and norm keeps f16 rounding error identical on
        // both sides, so the only source of divergence under mutation is the RoPE pairing.

        // --- K reference ---
        let mut k_ref = vec![0.0f32; kv_dim];
        matmul_bt_f16(&input, &weights.k_proj, &mut k_ref, 1, hidden, kv_dim);
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
            "cpu F16 K-loop stride-half RoPE diverges from reference: max_k_diff = {max_k_diff:.6}. \
             With interleaved pairing the diff is O(0.1-1). Bug: #392."
        );

        // --- Q reference (guards the Q loop mutation) ---
        // The production scatter copies q_and_gate[0..q_dim] → scratch.q_buf[0..q_dim]
        // for head 0 (num_q_heads=1).
        let mut q_and_gate_ref = vec![0.0f32; 2 * q_dim];
        matmul_bt_f16(
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
            "cpu F16 Q-loop stride-half RoPE diverges from reference: max_q_diff = {max_q_diff:.6}. \
             With interleaved pairing the diff is O(0.1-1). Bug: #392."
        );
    }

    /// ADR-080 C1 (#780): `full_attention_step_f16`'s decode-attention row
    /// finalizer must fail closed on a NaN score, not propagate the bare
    /// `1.0 / sum_exp` into the context output. A prior cached position is
    /// poisoned (NaN K); the current position's own K/V stay finite. RED
    /// before the fix: the poisoned row's NaN leaked into every context lane.
    #[test]
    fn test_full_attn_step_f16_nan_cached_score_fails_closed() {
        use crate::model::qwen35_config::LayerType;
        use crate::weights::f16_weights::f32_to_f16_slice;

        let head_dim: usize = 32;
        let num_q_heads: usize = 1;
        let num_kv_heads: usize = 1;
        let hidden: usize = 64;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let position: usize = 1;

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
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        };

        let rope = RopeTable::new(cfg.rope_dim(), 512, cfg.rope_theta);

        let to_f16 = |src: &[f32]| -> Vec<u16> {
            let mut dst = vec![0u16; src.len()];
            f32_to_f16_slice(src, &mut dst);
            dst
        };

        // W_k = identity, W_q = identity for the Q half (gate half zero), W_v
        // = identity too (so the CURRENT token's V is a distinct, finite,
        // predictable vector rather than zero -- confirms the fail-closed
        // path isn't trivially passing because every V is zero anyway).
        let mut k_proj_f32 = vec![0.0f32; kv_dim * hidden];
        for j in 0..kv_dim {
            k_proj_f32[j * hidden + j] = 1.0;
        }
        let mut v_proj_f32 = vec![0.0f32; kv_dim * hidden];
        for j in 0..kv_dim {
            v_proj_f32[j * hidden + j] = 1.0;
        }
        let mut q_proj_f32 = vec![0.0f32; 2 * q_dim * hidden];
        for j in 0..q_dim {
            q_proj_f32[j * hidden + j] = 1.0;
        }

        let weights = F16FullAttentionLayerWeights {
            q_proj: to_f16(&q_proj_f32),
            k_proj: to_f16(&k_proj_f32),
            v_proj: to_f16(&v_proj_f32),
            o_proj: to_f16(&vec![0.0f32; hidden * q_dim]),
            q_norm: vec![0.0f32; head_dim],
            k_norm: vec![0.0f32; head_dim],
        };

        let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.07).collect();

        let mut scratch = ForwardScratch::new();
        scratch.ensure_capacity(&cfg, 2);
        scratch.attn_out[..hidden].copy_from_slice(&input);

        // Pre-load one poisoned cached position (position 0): NaN K, finite V.
        let mut kv_cache = KvCache::new(1);
        let mut poisoned_k = vec![0.0f32; kv_dim];
        poisoned_k[0] = f32::NAN;
        let finite_v = vec![5.0f32; kv_dim];
        kv_cache.append_kv(0, &poisoned_k, &finite_v);
        kv_cache.seq_len = 1;

        full_attention_step_f16(
            &weights,
            0,
            position,
            &mut kv_cache,
            &mut scratch,
            &cfg,
            &rope,
            hidden,
            None,
        );

        assert!(
            scratch.context[..head_dim].iter().all(|&v| v == 0.0),
            "expected exact-zero context for a NaN-poisoned cached score, \
             got {:?}",
            &scratch.context[..head_dim]
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

    /// A NaN reaching the f16 MoE router (corrupt f16 router gate weight or an
    /// upstream activation overflow) makes every router logit NaN. Before the
    /// fail-closed guards `moe_ffn_step_f16` left NaN probabilities (the
    /// `denom > 0.0` path was skipped), top-k selected nothing (`NaN > NEG_INF`
    /// is false), and the routed-expert loop indexed expert weights at
    /// `usize::MAX * stride` → overflow/OOB panic on the (bench-only public)
    /// `generate_f16` path. This is the f16 sibling of the f32 fix in
    /// qwen35/moe.rs (#410). The router must fail closed and no `usize::MAX`
    /// sentinel may reach accumulation.
    #[test]
    fn test_moe_ffn_step_f16_nan_router_fails_closed_no_panic() {
        use crate::weights::f16_weights::{
            F16, F16MoeLayerWeights, F16MoeRouter, F16RoutedExperts, F16SharedExpert,
        };
        let num_experts = 4usize;
        let hidden = 4usize;
        let inter = 2usize;
        let shared_inter = 2usize;
        let top_k = 2usize;

        let nan16 = F16::from_f32(f32::NAN).0;
        let zeros = |n: usize| vec![F16::from_f32(0.0).0; n];

        // Every router gate weight is NaN → every router logit is NaN.
        let router = F16MoeRouter::new(
            vec![nan16; num_experts * hidden],
            num_experts,
            top_k,
            hidden,
        )
        .unwrap();
        let experts = F16RoutedExperts::new(
            zeros(num_experts * 2 * inter * hidden),
            zeros(num_experts * hidden * inter),
            num_experts,
            hidden,
            inter,
        )
        .unwrap();
        let shared = F16SharedExpert::new(
            zeros(shared_inter * hidden),
            zeros(shared_inter * hidden),
            zeros(hidden * shared_inter),
            zeros(hidden),
            hidden,
            shared_inter,
        )
        .unwrap();
        let moe = F16MoeLayerWeights {
            router,
            experts,
            shared_expert: shared,
        };

        let mut scratch = ForwardScratch::new();
        let buf = inter.max(shared_inter);
        scratch.ffn_out.resize(hidden, 1.0);
        scratch.input_tmp.resize(hidden, 0.0);
        scratch.expert_out.resize(hidden, 0.0);
        scratch.gate_buf.resize(buf, 0.0);
        scratch.up_buf.resize(buf, 0.0);
        scratch.down_input.resize(buf, 0.0);
        scratch.router_logits.resize(num_experts, 0.0);
        scratch.router_selected.resize(top_k, (usize::MAX, 0.0));

        // Must not panic (was: usize::MAX expert_id → OOB slice / overflow).
        moe_ffn_step_f16(&moe, &mut scratch, hidden);

        assert!(
            scratch.router_selected[..top_k]
                .iter()
                .all(|(id, _)| *id < num_experts),
            "degenerate f16 router must not leave a usize::MAX sentinel selected"
        );
    }

    /// The non-obvious case the denom-else (not a max-only guard) is meant to
    /// catch: ONE router row is NaN while the rest are finite, so `max_logit`
    /// stays finite (Rust `f32::max` ignores a single NaN) but the NaN still
    /// lands in `denom`. A max-only guard (`if !max_logit.is_finite()`) would
    /// pass this and leave un-normalized raw `exp` mass; the denom-else fills
    /// the row with 0.0. f16 analogue of qwen35/moe.rs
    /// `test_moe_router_finite_max_nan_tail_fails_closed`.
    #[test]
    fn test_moe_ffn_step_f16_finite_max_nan_tail_fails_closed() {
        use crate::weights::f16_weights::{
            F16, F16MoeLayerWeights, F16MoeRouter, F16RoutedExperts, F16SharedExpert,
        };
        let num_experts = 4usize;
        let hidden = 4usize;
        let inter = 2usize;
        let shared_inter = 2usize;
        let top_k = 2usize;

        let nan16 = F16::from_f32(f32::NAN).0;
        let zeros = |n: usize| vec![F16::from_f32(0.0).0; n];

        // Only expert 0's gate row is NaN → logit[0] = NaN, logit[1..] finite,
        // so `max_logit` is finite but `denom` is NaN.
        let mut gate = zeros(num_experts * hidden);
        for w in &mut gate[..hidden] {
            *w = nan16;
        }
        let router = F16MoeRouter::new(gate, num_experts, top_k, hidden).unwrap();
        let experts = F16RoutedExperts::new(
            zeros(num_experts * 2 * inter * hidden),
            zeros(num_experts * hidden * inter),
            num_experts,
            hidden,
            inter,
        )
        .unwrap();
        let shared = F16SharedExpert::new(
            zeros(shared_inter * hidden),
            zeros(shared_inter * hidden),
            zeros(hidden * shared_inter),
            zeros(hidden),
            hidden,
            shared_inter,
        )
        .unwrap();
        let moe = F16MoeLayerWeights {
            router,
            experts,
            shared_expert: shared,
        };

        let mut scratch = ForwardScratch::new();
        let buf = inter.max(shared_inter);
        scratch.ffn_out.resize(hidden, 1.0);
        scratch.input_tmp.resize(hidden, 0.0);
        scratch.expert_out.resize(hidden, 0.0);
        scratch.gate_buf.resize(buf, 0.0);
        scratch.up_buf.resize(buf, 0.0);
        scratch.down_input.resize(buf, 0.0);
        scratch.router_logits.resize(num_experts, 0.0);
        scratch.router_selected.resize(top_k, (usize::MAX, 0.0));

        moe_ffn_step_f16(&moe, &mut scratch, hidden);

        assert!(
            scratch.router_logits[..num_experts]
                .iter()
                .all(|p| *p == 0.0),
            "finite-max + NaN-tail router row must fail closed to all-zero probs \
             (a max-only guard would miss this)"
        );
    }

    // NOTE: the storage-bound guard (`expert_id >= moe.experts.num_experts`) is
    // release-only defense-in-depth — in debug the `debug_assert_eq!(moe.experts
    // .num_experts, num_experts)` at the top of `moe_ffn_step_f16` fires first on
    // a router/expert count mismatch, so the guard cannot be exercised by a debug
    // unit test. It mirrors the f32 sibling guard (qwen35/moe.rs) and costs one
    // comparison; it hardens the bench-only public `generate_f16` path against a
    // manually constructed f16 weight set whose router declares more experts than
    // the routed-expert storage holds.

    /// Build a zero-layer F16 model fixture for generate_f16 unit tests.
    ///
    /// All-zero u16 (= f16 zero) embeddings → logits all 0 → greedy picks token 0.
    /// eos_token_id = 5 so that greedy token 0 is NOT eos, making stop_token_ids=[0]
    /// detectable as a distinct stop path.
    fn zero_layer_f16_fixture() -> (Qwen35Config, F16ModelWeights, RopeTable, BpeTokenizer) {
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
            // eos is 5 so that greedy token 0 is NOT eos — allows stop_token_ids=[0]
            // to be a distinct, detectable stop signal.
            eos_token_id: 5,
            max_position_embeddings: 512,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        };

        // embed_tokens is [vocab * hidden] packed u16 (f16 zeros = 0u16).
        // All zeros → logits all 0 → greedy always picks token 0.
        let weights = F16ModelWeights {
            embed_tokens: vec![0u16; vocab * hidden],
            final_norm: vec![0.0f32; hidden],
            layers: vec![],
        };

        // rope_dim = head_dim * partial_rotary_factor = 4 * 0.5 = 2.
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

    /// `generate_f16` must reject a request whose prompt + max_new_tokens exceeds
    /// the RoPE table capacity with a clean error, not an out-of-bounds RoPE index
    /// (in a real model) or a runaway allocation. The preflight returns before any
    /// forward pass, so the zero-layer fixture is sufficient. Mutation check:
    /// removing the preflight lets the zero-layer model run the decode to
    /// completion and return Ok (it has no RoPE-indexing attention layer), which
    /// trips the `expect_err` below — changing the test from PASS to FAIL. Mirrors
    /// `generate_q8`'s `test_generate_q8_rejects_context_overflow`.
    #[test]
    fn test_generate_f16_rejects_context_overflow() {
        let (cfg, weights, rope, tokenizer) = zero_layer_f16_fixture();
        let max_context = rope.max_positions(); // 64 from the fixture
        // "hello" is >= 1 token, so prompt_len + max_context > max_context.
        let gen_cfg = GenerateConfig {
            max_new_tokens: max_context,
            ..Default::default()
        };
        let err = generate_f16(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg)
            .expect_err("request beyond context window must error, not panic");
        let msg = format!("{err}");
        assert!(
            msg.contains("context window"),
            "error must name the context window; got: {msg}"
        );
    }

    /// `generate_f16` must stop on a token in `stop_token_ids` even when that
    /// token differs from `eos_token_id`.
    ///
    /// Setup: all-zero f16 weights → greedy sampling always picks token 0.
    /// Config has eos_token_id=5 (not 0) and stop_token_ids=[0].
    /// With the fix the first sampled token (0) hits the stop list and the
    /// function returns 0 generated tokens.
    ///
    /// Mutation check: reverting `should_stop_token` back to
    /// `next_id == cfg.eos_token_id` in either check causes `0 == 5` to be false,
    /// so token 0 is pushed to output and `generated_tokens` becomes ≥ 1.
    #[test]
    fn test_generate_f16_honors_stop_token_ids() {
        let (cfg, weights, rope, tokenizer) = zero_layer_f16_fixture();

        let gen_cfg = GenerateConfig {
            max_new_tokens: 4,
            stop_token_ids: vec![0], // token 0 is the stop signal, NOT eos (5)
            temperature: 0.0,        // greedy: all-zero logits always yield token 0
            ..Default::default()
        };

        let out = generate_f16(&weights, &cfg, &tokenizer, &rope, "h", &gen_cfg)
            .expect("generate_f16 must succeed with valid stop_token_ids");

        assert_eq!(
            out.generated_tokens, 0,
            "generate_f16 must stop immediately when the first greedy token (0) \
             is in stop_token_ids — got {} generated tokens instead",
            out.generated_tokens
        );
    }

    /// `generate_f16` must also stop when the stop token first appears in the
    /// **decode loop**, not only at the post-prefill check.
    ///
    /// Fixture: a "bouncing" 0-layer f16 model.
    ///   embed[0] = [-1, 1, 0, 0]  (token 0, f16)
    ///   embed[1] = [ 1, 1, 0, 0]  (token 1, f16)
    ///   final_norm gamma = [-2, 0, 0, 0]
    ///
    /// The negative gamma at dim-0 flips the sign of that component after RMSNorm,
    /// creating a "bounce" between tokens 0 and 1:
    ///   from token 1: hidden = [-√2, +√2, 0, 0] → logit[0] = 2√2 wins → generates 0
    ///   from token 0: hidden = [+√2, +√2, 0, 0] → logit[1] = 2√2 wins → generates 1
    ///
    /// Greedy sequence from prompt "e" (→ token 1, eos_token_id=5):
    ///   post-prefill  → token 0  (not stop=1)
    ///   decode step 1 → token 1  (stop) → decode-loop fires
    ///
    /// Mutation proof: reverting ONLY the decode-loop `should_stop_token` check
    /// (line 946 at time of writing) to `next_id == cfg.eos_token_id` leaves
    /// token 1 uncaught (1 ≠ eos=5), the sequence continues, and generated_tokens
    /// becomes ≥ 2 — failing the assertion below.
    #[test]
    fn test_generate_f16_honors_stop_token_ids_decode_loop() {
        use crate::weights::f16_weights::f32_to_f16_slice;
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
            // eos=5 so the stop at token 1 is detectable only via stop_token_ids.
            eos_token_id: 5,
            max_position_embeddings: 512,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        };

        // The negative gamma at dim-0 flips the sign of that component after
        // RMSNorm, creating a deterministic "bounce" between tokens 0 and 1.
        // from embed[1]=[1,1,0,0]: hidden→[-√2,+√2,0,0] → dot(embed[0]=[-1,1,..]) = 2√2 > 0
        // from embed[0]=[-1,1,0,0]: hidden→[+√2,+√2,0,0] → dot(embed[1]=[1,1,..]) = 2√2 > 0
        let embed_f32: Vec<f32> = {
            let mut v = vec![0.0f32; vocab * hidden];
            v[0] = -1.0; // token 0, dim 0
            v[1] = 1.0; // token 0, dim 1
            v[hidden] = 1.0; // token 1, dim 0
            v[hidden + 1] = 1.0; // token 1, dim 1
            v
        };
        let mut embed_f16 = vec![0u16; vocab * hidden];
        f32_to_f16_slice(&embed_f32, &mut embed_f16);

        let weights = F16ModelWeights {
            embed_tokens: embed_f16,
            final_norm: vec![-2.0f32, 0.0, 0.0, 0.0],
            layers: vec![],
        };

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

        let gen_cfg = GenerateConfig {
            max_new_tokens: 10,
            stop_token_ids: vec![1], // stop on token 1 mid-decode-loop; eos_token_id=5≠1
            temperature: 0.0,        // greedy: deterministic bouncing sequence
            ..Default::default()
        };

        // Prompt "e" → token 1.
        // Post-prefill generates token 0 (not stop=1).
        // Decode step 1 generates token 1 → decode-loop stop fires.
        let out = generate_f16(&weights, &cfg, &tokenizer, &rope, "e", &gen_cfg)
            .expect("generate_f16 must succeed");

        assert_eq!(
            out.generated_tokens, 1,
            "generate_f16 must stop at decode-loop step 1 when token 1 is in \
             stop_token_ids — got {} tokens; reverting only the decode-loop check \
             lets token 1 through and produces ≥ 2 tokens",
            out.generated_tokens
        );
        assert!(
            out.stopped,
            "generate_f16 must set stopped=true when the decode-loop stop fires"
        );
    }

    /// `generate_f16` must reject an empty prompt with a typed
    /// `Err(Inference("empty prompt"))` before any weight dereference or
    /// state allocation (#856): this is one of the three CPU forward paths
    /// the shared `check_prompt_not_empty` preflight unifies with the four
    /// Metal paths, which used to silently accept an empty prompt and
    /// return an empty `Ok`. See docs/generation-entrypoint-matrix.md row 2.
    ///
    /// The guard fires before any weight dereference, so empty weight vecs
    /// are sufficient (mirrors `generate_f16_rejects_grammar_config_before_sampling`
    /// below).
    ///
    /// Mutation sensitivity: reverting the `check_prompt_not_empty` call at
    /// this call site makes the function proceed past the guard with a
    /// zero-length prompt, either panicking in the prefill/decode loop
    /// (`all_ids.last()` on an empty vec) or producing a non-`Inference`
    /// error — this assert fails either way.
    #[test]
    fn generate_f16_rejects_empty_prompt() {
        use crate::error::InferenceError;
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
        let weights = F16ModelWeights {
            embed_tokens: vec![],
            final_norm: vec![],
            layers: vec![],
        };
        let gen_cfg = GenerateConfig::default();

        let result = generate_f16(&weights, &cfg, &tokenizer, &rope, "", &gen_cfg);
        assert!(
            matches!(result, Err(InferenceError::Inference(ref msg)) if msg.contains("empty prompt")),
            "generate_f16 must reject an empty prompt with Err(Inference(\"empty \
             prompt\")) (#856); got {result:?}"
        );
    }

    /// `generate_f16` must reject a `GenerateConfig` that sets `grammar` with a
    /// typed `InvalidInput` error before sampling any token (#397/#398).
    ///
    /// Before the fix, grammar was silently ignored and unconstrained output was
    /// produced. The guard fires before any weight dereference or state allocation,
    /// so empty weight vecs are sufficient.
    ///
    /// Mutation sensitivity: removing the `check_grammar_not_set` call makes the
    /// function proceed past the guard and attempt to forward with empty weights,
    /// producing a panic or a non-`InvalidInput` error — this assert fails either way.
    #[test]
    fn generate_f16_rejects_grammar_config_before_sampling() {
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
        let weights = F16ModelWeights {
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

        let result = generate_f16(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg);
        assert!(
            matches!(result, Err(InferenceError::InvalidInput(_))),
            "generate_f16 must fail closed with InvalidInput when grammar is set (#397/#398); \
             got {result:?}"
        );
    }

    /// `generate_f16` must reject a `GenerateConfig` that sets `stop_strings` with a
    /// typed `InvalidInput` error before sampling any token (ADR-080 C3, #783).
    ///
    /// Mutation sensitivity: removing the `check_stop_strings_not_set` call makes the
    /// function proceed past the guard and attempt to forward with empty weights,
    /// producing a panic or a non-`InvalidInput` error — this assert fails either way.
    #[test]
    fn generate_f16_rejects_stop_strings_config_before_sampling() {
        use crate::error::InferenceError;
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
        let weights = F16ModelWeights {
            embed_tokens: vec![],
            final_norm: vec![],
            layers: vec![],
        };

        let gen_cfg = GenerateConfig {
            stop_strings: vec!["</s>".to_string()],
            ..Default::default()
        };

        let result = generate_f16(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg);
        assert!(
            matches!(result, Err(InferenceError::InvalidInput(_))),
            "generate_f16 must fail closed with InvalidInput when stop_strings is set \
             (ADR-080 C3, #783); got {result:?}"
        );
    }

    /// `generate_f16` must reject a `GenerateConfig` that sets `reasoning_budget` with
    /// a typed `InvalidInput` error before sampling any token (ADR-080 C3, #783).
    ///
    /// Mutation sensitivity: removing the `check_reasoning_budget_not_set` call makes
    /// the function proceed past the guard and attempt to forward with empty weights,
    /// producing a panic or a non-`InvalidInput` error — this assert fails either way.
    #[test]
    fn generate_f16_rejects_reasoning_budget_config_before_sampling() {
        use crate::error::InferenceError;
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
        let weights = F16ModelWeights {
            embed_tokens: vec![],
            final_norm: vec![],
            layers: vec![],
        };

        let gen_cfg = GenerateConfig {
            reasoning_budget: Some(16),
            ..Default::default()
        };

        let result = generate_f16(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg);
        assert!(
            matches!(result, Err(InferenceError::InvalidInput(_))),
            "generate_f16 must fail closed with InvalidInput when reasoning_budget is set \
             (ADR-080 C3, #783); got {result:?}"
        );
    }

    /// `generate_f16` with `max_new_tokens == 0` must return zero generated tokens
    /// without running prefill or sampling anything (#612, 3rd recurrence of the
    /// #226/#456 bug class).
    ///
    /// The guard fires before any weight dereference or state allocation, so
    /// empty weight vecs are sufficient — mirrors the grammar-guard test above.
    ///
    /// Mutation sensitivity: removing the `max_new_tokens == 0` early return
    /// causes the function to run prefill (against empty weight vecs, which
    /// would panic) and sample one token, so `generated_tokens` becomes 1
    /// instead of 0 and the assertion below fails.
    #[test]
    fn generate_f16_max_new_tokens_zero_returns_empty() {
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
        let weights = F16ModelWeights {
            embed_tokens: vec![],
            final_norm: vec![],
            layers: vec![],
        };

        let gen_cfg = GenerateConfig {
            max_new_tokens: 0,
            ..Default::default()
        };

        let out = generate_f16(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg)
            .expect("max_new_tokens=0 must succeed, not error");

        assert_eq!(
            out.generated_tokens, 0,
            "max_new_tokens=0 must produce zero generated tokens"
        );
        assert!(
            out.token_ids.is_empty(),
            "max_new_tokens=0 must produce an empty token list"
        );
        assert_eq!(
            out.stop_reason,
            Some(StopReason::Length),
            "max_new_tokens=0 must report stop_reason=Length"
        );
    }

    // -----------------------------------------------------------------
    // ADR-069 Stage 5b: visual injection + M-RoPE splice (pure CPU, no
    // checkpoint required).
    // -----------------------------------------------------------------

    use crate::model::qwen35_config::{LayerType, RopeParams};
    use crate::weights::f16_weights::{
        F16CommonLayerWeights, F16FullAttentionLayerWeights, f32_to_f16_slice,
    };

    /// A minimal one-layer, full-attention-only (no GDN) config + f16 weight
    /// set: small enough to hand-construct, but with non-trivial (identity)
    /// Q/K/V/O projections so RoPE and embedding-source mutations actually
    /// move the logits, unlike an all-zero model.
    fn tiny_vision_splice_model() -> (Qwen35Config, F16ModelWeights) {
        let hidden = 8usize;
        let vocab = 4usize;

        let cfg = Qwen35Config {
            hidden_size: hidden,
            num_hidden_layers: 1,
            vocab_size: vocab,
            intermediate_size: 4,
            rms_norm_eps: 1e-6,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: hidden,
            rope_theta: 1.0e7,
            partial_rotary_factor: 1.0, // rope_dim=8, half=4 (matches production theta scale)
            rope_parameters: Some(RopeParams {
                rope_theta: 1.0e7,
                partial_rotary_factor: Some(1.0),
                mrope_section: Some(vec![2, 1, 1]),
                mrope_interleaved: Some(true),
            }),
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
            full_attention_interval: 1,
            layer_types: vec![LayerType::FullAttention],
            layer_mask: vec![true],
            eos_token_id: 999,
            max_position_embeddings: 512,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
            vision_config: None,
            image_token_id: Some(3),
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        };

        let to_f16 = |src: &[f32]| -> Vec<u16> {
            let mut dst = vec![0u16; src.len()];
            f32_to_f16_slice(src, &mut dst);
            dst
        };
        let identity = |rows: usize, cols: usize| -> Vec<f32> {
            let mut m = vec![0.0f32; rows * cols];
            for i in 0..rows.min(cols) {
                m[i * cols + i] = 1.0;
            }
            m
        };

        let embed_tokens_f32: Vec<f32> = (0..vocab * hidden)
            .map(|k| ((k % 11) as f32) * 0.05 - 0.2)
            .collect();

        let q_dim = hidden;
        let mut q_proj_f32 = vec![0.0f32; 2 * q_dim * hidden];
        q_proj_f32[..q_dim * hidden].copy_from_slice(&identity(q_dim, hidden));

        let full_weights = F16FullAttentionLayerWeights {
            q_proj: to_f16(&q_proj_f32),
            k_proj: to_f16(&identity(hidden, hidden)),
            v_proj: to_f16(&identity(hidden, hidden)),
            o_proj: to_f16(&identity(hidden, q_dim)),
            q_norm: vec![0.0f32; hidden],
            k_norm: vec![0.0f32; hidden],
        };

        let common = F16CommonLayerWeights {
            input_layernorm: vec![0.0f32; hidden],
            post_attention_layernorm: vec![0.0f32; hidden],
            ffn: F16FeedForwardWeights::Dense {
                gate_proj: to_f16(&vec![0.0f32; 4 * hidden]),
                up_proj: to_f16(&vec![0.0f32; 4 * hidden]),
                down_proj: to_f16(&vec![0.0f32; hidden * 4]),
            },
        };

        let weights = F16ModelWeights {
            embed_tokens: to_f16(&embed_tokens_f32),
            final_norm: vec![0.0f32; hidden],
            layers: vec![(F16AttentionWeights::Full(full_weights), common)],
        };

        (cfg, weights)
    }

    /// Injection reaches the decoder: a supplied embedding replaces the
    /// token-id lookup (not adds to it), mutating the supplied row changes
    /// the resulting logits, and mutating an unrelated vocab row leaves the
    /// injected-slot output unchanged (proves the None-path embedding table
    /// isn't consulted when `injected_embedding` is `Some`).
    #[test]
    fn injection_replaces_lookup_and_is_mutation_sensitive() {
        let (cfg, mut weights) = tiny_vision_splice_model();
        let hidden = cfg.hidden_size;

        // Compares the pre-lm_head hidden state (not `scratch.logits`): `embed_tokens` is
        // tied to the output projection too, so a logits comparison would show every vocab
        // row mutation regardless of whether the *input* lookup was ever consulted. The
        // decoder's final hidden state isolates exactly the quantity injection controls.
        let run = |weights: &F16ModelWeights, injected: Option<&[f32]>| -> Vec<f32> {
            let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
            let mut gdn_states: Vec<GatedDeltaNetState> = vec![];
            let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
            let mut scratch = ForwardScratch::new();
            forward_step_f16(
                weights,
                &cfg,
                &rope,
                0,
                0,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
                injected,
                None,
            )
            .expect("forward step succeeds");
            scratch.hidden[..hidden].to_vec()
        };

        let baseline = run(&weights, None);

        let mut visual_row = vec![0.9f32, -0.4, 0.2, 0.6, -0.1, 0.3, 0.7, -0.8];
        let injected_hidden = run(&weights, Some(&visual_row));
        assert_ne!(
            injected_hidden, baseline,
            "an injected embedding must produce a different hidden state than the token-id lookup"
        );

        // Mutate the supplied visual scalar: the image-pad-slot output must change.
        visual_row[0] += 1.0;
        let mutated_visual_hidden = run(&weights, Some(&visual_row));
        assert_ne!(
            mutated_visual_hidden, injected_hidden,
            "mutating the supplied visual row must change the injected-slot hidden state"
        );
        visual_row[0] -= 1.0; // restore

        // Mutate a non-pad vocab row in the embedding table: the injected-slot output
        // (still using the ORIGINAL visual_row) must be unchanged.
        let embed_start = hidden; // token id 1's row
        let mutated_row_f32 = vec![1.0f32; hidden];
        f32_to_f16_slice(
            &mutated_row_f32,
            &mut weights.embed_tokens[embed_start..embed_start + hidden],
        );
        let after_table_mutation = run(&weights, Some(&visual_row));
        assert_eq!(
            after_table_mutation, injected_hidden,
            "mutating an unrelated embedding-table row must not affect the injected slot"
        );

        // Sanity: length mismatch and non-finite values must fail closed.
        let mut gdn_states: Vec<GatedDeltaNetState> = vec![];
        let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
        let mut scratch = ForwardScratch::new();
        let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
        let short_row = vec![0.0f32; hidden - 1];
        assert!(
            forward_step_f16(
                &weights,
                &cfg,
                &rope,
                0,
                0,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
                Some(&short_row),
                None,
            )
            .is_err(),
            "wrong-length injected_embedding must be rejected"
        );
        let mut nan_row = vec![0.0f32; hidden];
        nan_row[2] = f32::NAN;
        assert!(
            forward_step_f16(
                &weights,
                &cfg,
                &rope,
                0,
                0,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
                Some(&nan_row),
                None,
            )
            .is_err(),
            "non-finite injected_embedding must be rejected"
        );
    }

    /// The cos/sin actually applied inside the wired GQA path
    /// (`full_attention_step_f16` via `forward_step_f16`) at an image-pad
    /// token matches `build_cos_sin`'s output at that same (t,h,w) position —
    /// exercised through the wired forward, not just the S5a unit builder.
    #[test]
    fn mrope_cos_sin_applied_in_wired_forward_matches_builder() {
        use crate::vision::qwen35_mrope::{MRopePositions, build_cos_sin};

        let (cfg, weights) = tiny_vision_splice_model();
        let hidden = cfg.hidden_size;
        let rope_params = cfg.rope_parameters.as_ref().unwrap();
        let mrope_section = rope_params.mrope_section.as_ref().unwrap();

        let positions = MRopePositions {
            positions: vec![(2, 3, 5)],
            rope_delta: 0,
        };
        let tables = build_cos_sin(
            &positions,
            cfg.head_dim,
            rope_params.partial_rotary_factor.unwrap(),
            rope_params.rope_theta as f32,
            mrope_section,
        )
        .expect("builds tables");
        let (cos_row, sin_row) = (tables.cos[0].as_slice(), tables.sin[0].as_slice());

        let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
        let mut gdn_states: Vec<GatedDeltaNetState> = vec![];
        let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
        let mut scratch = ForwardScratch::new();

        forward_step_f16(
            &weights,
            &cfg,
            &rope,
            1,
            0,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
            None,
            Some((cos_row, sin_row)),
        )
        .expect("forward step succeeds");

        // Independently reproduce the K rotation using the SAME cos/sin row (identity
        // k_proj means k_buf before rotation equals the RMSNorm'd embedding row).
        let mut k_ref = vec![0.0f32; hidden];
        f16_to_f32_slice(&weights.embed_tokens[hidden..2 * hidden], &mut k_ref);
        let zero_norm = vec![0.0f32; hidden];
        qwen35_rms_norm(&mut k_ref, &zero_norm, hidden, cfg.rms_norm_eps);
        let half = cfg.rope_dim() / 2;
        for i in 0..half {
            let x0 = k_ref[i];
            let x1 = k_ref[half + i];
            k_ref[i] = x0 * cos_row[i] - x1 * sin_row[i];
            k_ref[half + i] = x0 * sin_row[i] + x1 * cos_row[i];
        }

        let k_cached = &kv_cache.k[0][..hidden];
        let max_diff = k_cached
            .iter()
            .zip(k_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "wired M-RoPE rotation diverges from build_cos_sin's own row: max_diff={max_diff}"
        );
    }

    /// A text-only `Qwen35VisionRequest` (no image runs) driven through
    /// `generate_multimodal_f16` must be bit-identical to the same token
    /// sequence driven through `forward_step_f16` directly with the plain
    /// 1-D `RopeTable` (mirroring `generate_f16`'s own prefill+decode loop) —
    /// proving `generate_multimodal_f16`'s text-only path never falls
    /// through to the M-RoPE table when the request has no image.
    #[test]
    fn generate_multimodal_text_only_matches_plain_forward_bit_identical() {
        use crate::vision::multimodal::Qwen35VisionRequest;

        let (cfg, weights) = tiny_vision_splice_model();
        let input_ids: Vec<u32> = vec![0, 1, 2, 0];

        let request = Qwen35VisionRequest {
            input_ids: input_ids.clone(),
            image_grids: vec![],
            post_merger_rows: vec![],
            image_token_id: 3,
            spatial_merge_size: 2,
            decoder_hidden_size: cfg.hidden_size,
        };

        let gen_cfg = GenerateConfig {
            max_new_tokens: 2,
            temperature: 0.0,
            seed: Some(1),
            stop_token_ids: vec![],
            ..Default::default()
        };

        let multimodal_out = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg)
            .expect("text-only multimodal generate succeeds");

        // Reference: hand-drive forward_step_f16 exactly like generate_f16's own loop,
        // with the same seed and the plain 1-D RopeTable.
        let rope = RopeTable::new(cfg.rope_dim(), 512, cfg.rope_theta);
        let mut gdn_states: Vec<GatedDeltaNetState> = vec![];
        let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
        let mut scratch = ForwardScratch::new();
        for (pos, &token_id) in input_ids.iter().enumerate() {
            forward_step_f16(
                &weights,
                &cfg,
                &rope,
                token_id,
                pos,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
                None,
                None,
            )
            .expect("reference forward step succeeds");
            if pos < input_ids.len() - 1 {
                kv_cache.seq_len += 1;
            }
        }
        kv_cache.seq_len = input_ids.len();

        let mut rng_state = 1u64;
        let mut all_ids = input_ids.clone();
        let mut ref_ids = Vec::new();

        let next_id = sample_token(
            &scratch.logits[..cfg.vocab_size],
            &gen_cfg,
            &all_ids,
            &mut rng_state,
        );
        ref_ids.push(next_id);
        all_ids.push(next_id);

        for _ in 1..gen_cfg.max_new_tokens {
            let pos = kv_cache.seq_len;
            let last_token = *all_ids.last().unwrap();
            forward_step_f16(
                &weights,
                &cfg,
                &rope,
                last_token,
                pos,
                &mut gdn_states,
                &mut kv_cache,
                &mut scratch,
                None,
                None,
            )
            .expect("reference decode step succeeds");
            kv_cache.seq_len += 1;
            let next_id = sample_token(
                &scratch.logits[..cfg.vocab_size],
                &gen_cfg,
                &all_ids,
                &mut rng_state,
            );
            ref_ids.push(next_id);
            all_ids.push(next_id);
        }

        assert_eq!(
            multimodal_out.token_ids, ref_ids,
            "text-only generate_multimodal_f16 token ids must match the plain forward_step_f16 \
             reference bit-for-bit"
        );
    }

    /// `generate_multimodal_f16` fails closed on an invalid request (mismatched
    /// image-pad count vs grid) before any decoder work begins — it delegates to
    /// `Qwen35VisionRequest::validate()` rather than re-deriving the checks.
    #[test]
    fn generate_multimodal_f16_rejects_invalid_request() {
        use crate::vision::multimodal::Qwen35VisionRequest;

        let (cfg, weights) = tiny_vision_splice_model();
        let request = Qwen35VisionRequest {
            input_ids: vec![0, 3, 3, 1], // 2 image-pad tokens
            image_grids: vec![crate::vision::qwen35_vit::GridThw { t: 1, h: 4, w: 4 }], // needs 4
            post_merger_rows: vec![0.0f32; 2 * cfg.hidden_size],
            image_token_id: 3,
            spatial_merge_size: 2,
            decoder_hidden_size: cfg.hidden_size,
        };
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            ..Default::default()
        };
        assert!(
            generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg).is_err(),
            "a request whose image-pad count does not match its grid must be rejected"
        );
    }

    /// `generate_multimodal_f16` fails closed when the prompt plus
    /// `max_new_tokens` would exceed the model's context window, mirroring
    /// `generate_f16`'s own preflight.
    #[test]
    fn generate_multimodal_f16_rejects_context_overflow() {
        use crate::vision::multimodal::Qwen35VisionRequest;

        let (mut cfg, weights) = tiny_vision_splice_model();
        cfg.max_position_embeddings = 3;
        let request = Qwen35VisionRequest {
            input_ids: vec![0, 1, 2],
            image_grids: vec![],
            post_merger_rows: vec![],
            image_token_id: 3,
            spatial_merge_size: 2,
            decoder_hidden_size: cfg.hidden_size,
        };
        let gen_cfg = GenerateConfig {
            max_new_tokens: 5,
            ..Default::default()
        };
        assert!(
            generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg).is_err(),
            "prompt_len + max_new_tokens exceeding max_position_embeddings must be rejected"
        );
    }

    /// A caller-supplied `input_ids` entry at or past
    /// `cfg.vocab_size` must be rejected with `InvalidInput` before any decoder
    /// allocation/work, not panic in `forward_step_f16`'s embedding-table slice.
    /// Mutation check: removing the guard lets the request reach the None-branch
    /// embedding lookup, which indexes `embed_tokens[token_id * hidden..]` past
    /// the table's end and panics, turning this `expect_err` into a test-binary
    /// abort rather than a clean assertion failure — either way the test fails.
    #[test]
    fn generate_multimodal_f16_rejects_out_of_vocab_input_id() {
        use crate::vision::multimodal::Qwen35VisionRequest;

        let (cfg, weights) = tiny_vision_splice_model();
        let request = Qwen35VisionRequest {
            input_ids: vec![0, 1, cfg.vocab_size as u32], // last id == vocab_size (OOV)
            image_grids: vec![],
            post_merger_rows: vec![],
            image_token_id: 3,
            spatial_merge_size: 2,
            decoder_hidden_size: cfg.hidden_size,
        };
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            ..Default::default()
        };
        let err = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg)
            .expect_err("an out-of-vocabulary input_id must be rejected, not panic");
        assert!(
            matches!(err, crate::error::InferenceError::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    /// Sibling path of the input-id guard: `generate_f16` accepts `cfg` and
    /// `tokenizer` as independent parameters, so a mismatched pair can still
    /// tokenize a prompt into an id at or past `cfg.vocab_size`. Simulates that
    /// mismatch directly (a tokenizer vocab entry the fixture's 8-row embedding
    /// table cannot cover) rather than relying on tokenizer internals to produce
    /// an OOV id by accident.
    /// Mutation check: removing the guard lets `forward_step_f16`'s prefill call
    /// index `embed_tokens[8 * hidden..]` on a `vocab=8` table — out of bounds —
    /// so the panic (or, pre-guard, a wrong-but-silent read) replaces this clean
    /// `expect_err`, failing the test either way.
    #[test]
    fn test_generate_f16_rejects_out_of_vocab_prompt_id() {
        use std::collections::HashMap;

        let (cfg, weights, rope, _tokenizer) = zero_layer_f16_fixture();

        let mut vocab_map: HashMap<String, u32> = HashMap::new();
        for (i, c) in ["h", "e", "l", "o", "w", "r", "d", "!"].iter().enumerate() {
            vocab_map.insert((*c).to_string(), i as u32);
        }
        // OOV against the fixture's cfg.vocab_size=8 embedding table -- a
        // mismatched cfg/tokenizer pair, exactly the scenario the guard covers.
        vocab_map.insert("z".to_string(), cfg.vocab_size as u32);
        let mismatched_tokenizer = BpeTokenizer::from_vocab_and_merges(vocab_map, vec![])
            .expect("tokenizer with an OOV vocab entry still constructs");

        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            ..Default::default()
        };
        let err = generate_f16(&weights, &cfg, &mismatched_tokenizer, &rope, "z", &gen_cfg)
            .expect_err("an out-of-vocabulary prompt token id must be rejected, not panic");
        assert!(
            matches!(err, crate::error::InferenceError::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    /// Build the [`tiny_vision_splice_model`] fixture plus a populated
    /// `vision_config` (spatial_merge_size=2, out_hidden_size == decoder
    /// hidden_size) so checkpoint-binding mismatches are
    /// testable against a checkpoint that genuinely carries vision metadata.
    fn tiny_vision_splice_model_with_vision_cfg() -> (Qwen35Config, F16ModelWeights) {
        use crate::model::qwen35_config::VisionModelConfig;

        let (mut cfg, weights) = tiny_vision_splice_model();
        cfg.vision_config = Some(VisionModelConfig {
            depth: 1,
            hidden_size: 8,
            num_heads: 1,
            patch_size: 1,
            spatial_merge_size: 2,
            out_hidden_size: cfg.hidden_size,
            temporal_patch_size: 1,
            num_position_embeddings: 1,
            in_channels: 1,
            deepstack_visual_indexes: vec![],
            intermediate_size: None,
        });
        (cfg, weights)
    }

    /// An internally-consistent multimodal request whose
    /// `image_token_id` does not match the loaded checkpoint's must be rejected
    /// up front, before image slots are selected -- otherwise it silently
    /// injects/rotates at the wrong slots against a real checkpoint.
    /// Mutation check: removing just this guard branch (leaving the other FIX-3
    /// checks in place) lets the request reach the decoder unmodified; since
    /// `generate_multimodal_f16`'s own injection loop keys off the request's
    /// (not the checkpoint's) `image_token_id`, the run completes with `Ok`,
    /// flipping this `expect_err` to a failing assertion.
    #[test]
    fn generate_multimodal_f16_rejects_mismatched_image_token_id() {
        use crate::vision::multimodal::Qwen35VisionRequest;
        use crate::vision::qwen35_vit::GridThw;

        let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
        assert_eq!(cfg.image_token_id, Some(3));

        // Internally consistent (validate() passes): pad id 2, matching grid/rows,
        // but 2 != the checkpoint's image_token_id (3).
        let request = Qwen35VisionRequest {
            input_ids: vec![0, 2, 2, 2, 2, 1],
            image_grids: vec![GridThw { t: 1, h: 4, w: 4 }],
            post_merger_rows: vec![0.1f32; 4 * cfg.hidden_size],
            image_token_id: 2,
            spatial_merge_size: 2,
            decoder_hidden_size: cfg.hidden_size,
        };
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            ..Default::default()
        };
        let err = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg)
            .expect_err("mismatched image_token_id must be rejected, not silently run");
        let msg = format!("{err}");
        assert!(
            msg.contains("image_token_id"),
            "error must name image_token_id; got: {msg}"
        );
    }

    /// A request whose `spatial_merge_size` does not match the
    /// checkpoint's `vision_config.spatial_merge_size` must be rejected up front.
    /// Mutation check: removing just this guard branch lets an internally
    /// consistent (but checkpoint-mismatched) request run to completion (`Ok`),
    /// since nothing downstream re-derives merge size from `cfg.vision_config`.
    #[test]
    fn generate_multimodal_f16_rejects_mismatched_spatial_merge_size() {
        use crate::vision::multimodal::Qwen35VisionRequest;
        use crate::vision::qwen35_vit::GridThw;

        let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
        assert_eq!(cfg.vision_config.as_ref().unwrap().spatial_merge_size, 2);

        // merge_size=1 (checkpoint says 2): 1*4*4/1^2 = 16 post-merger rows,
        // internally consistent with the request's own spatial_merge_size.
        let mut input_ids = vec![0u32];
        input_ids.extend(std::iter::repeat_n(3u32, 16));
        input_ids.push(1);
        let request = Qwen35VisionRequest {
            input_ids,
            image_grids: vec![GridThw { t: 1, h: 4, w: 4 }],
            post_merger_rows: vec![0.1f32; 16 * cfg.hidden_size],
            image_token_id: 3,
            spatial_merge_size: 1,
            decoder_hidden_size: cfg.hidden_size,
        };
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            ..Default::default()
        };
        let err = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg)
            .expect_err("mismatched spatial_merge_size must be rejected, not silently run");
        let msg = format!("{err}");
        assert!(
            msg.contains("spatial_merge_size"),
            "error must name spatial_merge_size; got: {msg}"
        );
    }

    /// A request whose `decoder_hidden_size` does not match
    /// the checkpoint's `hidden_size` must be rejected up front by name, before
    /// image slots are selected -- not only by the unrelated, later
    /// `forward_step_f16` injected-row-length guard that happens to catch this
    /// specific case too. Asserting the error text names `decoder_hidden_size`
    /// keeps this test sensitive to *this* guard specifically.
    /// Mutation check: removing just this guard branch still leaves the request
    /// failing (via `forward_step_f16`'s unrelated length check deep in the
    /// prefill loop), but the error text no longer mentions
    /// `decoder_hidden_size` -- flipping the `contains` assertion to failing.
    #[test]
    fn generate_multimodal_f16_rejects_mismatched_decoder_hidden_size() {
        use crate::vision::multimodal::Qwen35VisionRequest;
        use crate::vision::qwen35_vit::GridThw;

        let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
        assert_eq!(cfg.hidden_size, 8);

        // decoder_hidden_size=4 (checkpoint hidden_size is 8): internally
        // consistent request (post_merger_rows sized to 4 rows * 4), but wrong
        // against this checkpoint.
        let request = Qwen35VisionRequest {
            input_ids: vec![0, 3, 3, 3, 3, 1],
            image_grids: vec![GridThw { t: 1, h: 4, w: 4 }],
            post_merger_rows: vec![0.1f32; 4 * 4],
            image_token_id: 3,
            spatial_merge_size: 2,
            decoder_hidden_size: 4,
        };
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            ..Default::default()
        };
        let err = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg)
            .expect_err("mismatched decoder_hidden_size must be rejected, not silently run");
        let msg = format!("{err}");
        assert!(
            msg.contains("decoder_hidden_size"),
            "error must name decoder_hidden_size (not just the unrelated downstream \
             injected-row-length message); got: {msg}"
        );
    }

    /// The M-RoPE table builder resolves
    /// `partial_rotary_factor` from `cfg.rope_parameters`, while the attention
    /// loop derives its rotary half-width from the separately public
    /// `cfg.rope_dim()` (`cfg.partial_rotary_factor`). A constructible config
    /// where these diverge (`head_dim=256`, `cfg.partial_rotary_factor=0.5` ->
    /// decoder half=64, but `rope_parameters.partial_rotary_factor=Some(0.25)`
    /// with section `[11,11,10]` -> table half=32) must fail closed here, before
    /// the first forward pass indexes `cos_row[32]`/`sin_row[32]` past the
    /// table's actual 32-lane row and panics.
    /// Mutation check: removing the guard lets `full_attention_step_f16` index
    /// the 32-lane row at `half=64`, panicking mid-attention instead of
    /// returning the clean `InvalidInput` this test expects.
    #[test]
    fn generate_multimodal_f16_rejects_mismatched_mrope_row_width() {
        use crate::model::qwen35_config::RopeParams;
        use crate::vision::multimodal::Qwen35VisionRequest;

        let (mut cfg, weights) = tiny_vision_splice_model();
        cfg.head_dim = 256;
        cfg.partial_rotary_factor = 0.5; // decoder rotary half = 64
        cfg.rope_parameters = Some(RopeParams {
            rope_theta: 1.0e7,
            partial_rotary_factor: Some(0.25), // table half = 32 (diverges from 64)
            mrope_section: Some(vec![11, 11, 10]),
            mrope_interleaved: Some(true),
        });

        // Text-only request: no image, so this exercises the guard on the
        // prefill table path alone (the decode-time sibling guard covers the
        // per-token `build_decode_cos_sin` path independently).
        let request = Qwen35VisionRequest {
            input_ids: vec![0, 1, 2],
            image_grids: vec![],
            post_merger_rows: vec![],
            image_token_id: 3,
            spatial_merge_size: 2,
            decoder_hidden_size: cfg.hidden_size,
        };
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            ..Default::default()
        };
        let err = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg).expect_err(
            "a config whose rope_parameters and rope_dim() disagree on rotary width must be \
             rejected, not panic at attention-lane indexing",
        );
        assert!(
            matches!(err, crate::error::InferenceError::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    // -----------------------------------------------------------------
    // Pooled embedding extraction (vision-embed-pooling)
    // -----------------------------------------------------------------

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na * nb)
    }

    /// A one-image request over [`tiny_vision_splice_model_with_vision_cfg`]:
    /// grid (1,4,4), merge_size=2 -> 4 post-merger rows, embedded in a short
    /// text scaffold `[0, <pad>x4, 1]`. `visual_rows` lets callers vary the
    /// "image content" while keeping every shape/id fixed.
    fn one_image_request(visual_rows: Vec<f32>) -> Qwen35VisionRequest {
        let mut input_ids = vec![0u32];
        input_ids.extend(std::iter::repeat_n(3u32, 4));
        input_ids.push(1);
        Qwen35VisionRequest {
            input_ids,
            image_grids: vec![crate::vision::qwen35_vit::GridThw { t: 1, h: 4, w: 4 }],
            post_merger_rows: visual_rows,
            image_token_id: 3,
            spatial_merge_size: 2,
            decoder_hidden_size: 8,
        }
    }

    #[test]
    fn embed_image_f16_is_deterministic() {
        let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
        let request = one_image_request(vec![0.3f32; 4 * cfg.hidden_size]);

        let v1 = embed_image_f16(&weights, &cfg, &request, PoolingStrategy::MeanVisualTokens)
            .expect("embed_image_f16 succeeds");
        let v2 = embed_image_f16(&weights, &cfg, &request, PoolingStrategy::MeanVisualTokens)
            .expect("embed_image_f16 succeeds");
        assert_eq!(v1, v2, "same input must produce an identical vector");

        let v3 = embed_image_f16(&weights, &cfg, &request, PoolingStrategy::LastToken)
            .expect("embed_image_f16 succeeds");
        let v4 = embed_image_f16(&weights, &cfg, &request, PoolingStrategy::LastToken)
            .expect("embed_image_f16 succeeds");
        assert_eq!(
            v3, v4,
            "same input must produce an identical vector (LastToken)"
        );
    }

    #[test]
    fn embed_image_f16_is_finite_and_unit_norm() {
        let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
        for pooling in [
            PoolingStrategy::MeanVisualTokens,
            PoolingStrategy::LastToken,
        ] {
            let request = one_image_request(vec![0.4f32; 4 * cfg.hidden_size]);
            let v = embed_image_f16(&weights, &cfg, &request, pooling)
                .expect("embed_image_f16 succeeds");
            assert_eq!(v.len(), cfg.hidden_size);
            assert!(
                v.iter().all(|x| x.is_finite()),
                "{pooling:?}: non-finite output"
            );
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "{pooling:?}: expected unit norm, got {norm}"
            );
        }
    }

    #[test]
    fn embed_image_f16_discriminates_different_images_but_matches_itself() {
        let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();

        let request_a = one_image_request(
            (0..4 * cfg.hidden_size)
                .map(|i| (i as f32) * 0.05 - 0.8)
                .collect(),
        );
        let request_b = one_image_request(
            (0..4 * cfg.hidden_size)
                .map(|i| -(i as f32) * 0.03 + 0.5)
                .collect(),
        );

        let emb_a = embed_image_f16(
            &weights,
            &cfg,
            &request_a,
            PoolingStrategy::MeanVisualTokens,
        )
        .expect("embed_image_f16 succeeds");
        let emb_a_again = embed_image_f16(
            &weights,
            &cfg,
            &request_a,
            PoolingStrategy::MeanVisualTokens,
        )
        .expect("embed_image_f16 succeeds");
        let emb_b = embed_image_f16(
            &weights,
            &cfg,
            &request_b,
            PoolingStrategy::MeanVisualTokens,
        )
        .expect("embed_image_f16 succeeds");

        let self_cos = cosine(&emb_a, &emb_a_again);
        assert!(
            (self_cos - 1.0).abs() < 1e-5,
            "an image embedded against itself must have cosine ~1.0, got {self_cos}"
        );

        let cross_cos = cosine(&emb_a, &emb_b);
        assert!(
            cross_cos < 0.999,
            "two different images must not collapse to near-identical embeddings, got cosine {cross_cos}"
        );
    }

    /// Direct unit test on the pooling primitive: pooling over the correct
    /// image-pad window vs. an off-by-one-shifted window over the SAME
    /// hidden-state matrix must produce different vectors. This is the
    /// mutation-sensitivity gate for the position-selection logic
    /// `embed_image_f16` relies on — an off-by-one bug in
    /// `image_pad_positions` (or a copy-pasted sibling that reintroduces
    /// one) changes the output instead of silently passing.
    #[test]
    fn pool_hidden_states_wrong_positions_change_the_output() {
        let hidden_size = 4;
        let seq_len = 6;
        // Row i is a constant-i vector, so shifting the pooled window by one
        // position is guaranteed to change the mean.
        let hidden_states: Vec<f32> = (0..seq_len)
            .flat_map(|i| std::iter::repeat_n(i as f32, hidden_size))
            .collect();

        let correct_positions = [1usize, 2, 3, 4];
        let off_by_one_positions = [2usize, 3, 4, 5];

        let correct = pool_hidden_states(
            &hidden_states,
            hidden_size,
            seq_len,
            &correct_positions,
            PoolingStrategy::MeanVisualTokens,
        );
        let wrong = pool_hidden_states(
            &hidden_states,
            hidden_size,
            seq_len,
            &off_by_one_positions,
            PoolingStrategy::MeanVisualTokens,
        );

        assert_ne!(
            correct, wrong,
            "pooling over an off-by-one-shifted position window must change the output"
        );
    }

    #[test]
    fn embed_image_f16_wrong_pad_run_placement_changes_embedding() {
        // Same checkpoint, same post-merger rows, but the image-pad run sits
        // at a different physical offset in input_ids (shifted by one text
        // token) -- the practical shape of a real "wrong positions" bug
        // (e.g. an off-by-one in scaffold assembly). The resulting pooled
        // embedding must differ: different M-RoPE coordinates and different
        // neighboring context both feed into it.
        let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
        let visual_rows = vec![0.25f32; 4 * cfg.hidden_size];

        let correct = one_image_request(visual_rows.clone());
        let mut shifted_ids = vec![0u32, 2]; // extra leading text token
        shifted_ids.extend(std::iter::repeat_n(3u32, 4));
        shifted_ids.push(1);
        let shifted = Qwen35VisionRequest {
            input_ids: shifted_ids,
            ..one_image_request(visual_rows)
        };

        let emb_correct =
            embed_image_f16(&weights, &cfg, &correct, PoolingStrategy::MeanVisualTokens)
                .expect("embed_image_f16 succeeds");
        let emb_shifted =
            embed_image_f16(&weights, &cfg, &shifted, PoolingStrategy::MeanVisualTokens)
                .expect("embed_image_f16 succeeds");

        assert_ne!(
            emb_correct, emb_shifted,
            "shifting the image-pad run's position in input_ids must change the pooled embedding"
        );
    }

    #[test]
    fn embed_text_vlm_f16_is_deterministic_and_unit_norm() {
        let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
        let mut vocab_map = std::collections::HashMap::new();
        for (i, c) in ["a", "b", "c"].iter().enumerate() {
            vocab_map.insert((*c).to_string(), i as u32);
        }
        let tokenizer =
            BpeTokenizer::from_vocab_and_merges(vocab_map, vec![]).expect("tokenizer constructs");

        for pooling in [
            PoolingStrategy::MeanVisualTokens,
            PoolingStrategy::LastToken,
        ] {
            let v1 = embed_text_vlm_f16(&weights, &cfg, &tokenizer, "abc", pooling)
                .expect("embed_text_vlm_f16 succeeds");
            let v2 = embed_text_vlm_f16(&weights, &cfg, &tokenizer, "abc", pooling)
                .expect("embed_text_vlm_f16 succeeds");
            assert_eq!(
                v1, v2,
                "{pooling:?}: same prompt must produce an identical vector"
            );
            assert_eq!(v1.len(), cfg.hidden_size);
            let norm: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "{pooling:?}: expected unit norm, got {norm}"
            );
        }
    }

    #[test]
    fn embed_text_vlm_f16_and_embed_image_f16_share_the_same_space() {
        // Not a quality claim (see PoolingStrategy's doc comment) -- just
        // proves both entry points route through the same decoder + pooling
        // call so their outputs are directly comparable vectors of the same
        // dimension, which is the structural property retrieval depends on.
        let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
        let mut vocab_map = std::collections::HashMap::new();
        vocab_map.insert("a".to_string(), 0u32);
        let tokenizer =
            BpeTokenizer::from_vocab_and_merges(vocab_map, vec![]).expect("tokenizer constructs");

        let text_emb =
            embed_text_vlm_f16(&weights, &cfg, &tokenizer, "a", PoolingStrategy::LastToken)
                .expect("embed_text_vlm_f16 succeeds");
        let image_emb = embed_image_f16(
            &weights,
            &cfg,
            &one_image_request(vec![0.1f32; 4 * cfg.hidden_size]),
            PoolingStrategy::LastToken,
        )
        .expect("embed_image_f16 succeeds");

        assert_eq!(text_emb.len(), image_emb.len());
        let cos = cosine(&text_emb, &image_emb);
        assert!(cos.is_finite());
    }

    #[test]
    fn embed_text_vlm_f16_rejects_prompt_colliding_with_image_token_id() {
        let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
        assert_eq!(cfg.image_token_id, Some(3));
        // Vocab entry "z" is deliberately assigned id 3, the checkpoint's
        // image_token_id -- a tokenized prompt must never silently contain it.
        let mut vocab_map = std::collections::HashMap::new();
        vocab_map.insert("z".to_string(), 3u32);
        let tokenizer =
            BpeTokenizer::from_vocab_and_merges(vocab_map, vec![]).expect("tokenizer constructs");

        let err = embed_text_vlm_f16(&weights, &cfg, &tokenizer, "z", PoolingStrategy::LastToken)
            .expect_err("a prompt colliding with image_token_id must be rejected");
        assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));
    }

    #[test]
    fn embed_image_f16_rejects_invalid_request() {
        let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
        let mut request = one_image_request(vec![0.1f32; 4 * cfg.hidden_size]);
        request.post_merger_rows.pop(); // now the wrong length
        assert!(
            embed_image_f16(&weights, &cfg, &request, PoolingStrategy::MeanVisualTokens).is_err()
        );
    }

    /// The context-window limit must be evaluated BEFORE `build_mrope_tables`
    /// materializes per-token position/cos/sin rows: an over-context request
    /// must fail cheaply, not after unbounded allocation work. The request
    /// here passes `Qwen35VisionRequest::validate` (run count, TOTAL pad
    /// count, and row-buffer length all line up) but carries per-run lengths
    /// [1, 3] against grids expecting [2, 2] — a mismatch only the M-RoPE
    /// builder detects — so getting the context-window error (not the
    /// builder's run-length error) proves the ordering.
    #[test]
    fn prefill_rejects_over_context_before_mrope_table_construction() {
        let (mut cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
        let base = one_image_request(vec![0.1f32; 4 * cfg.hidden_size]);
        let request = Qwen35VisionRequest {
            // Two pad runs of lengths 1 and 3 (total 4, matching the grids'
            // total merged rows), while each grid below expects a run of 2.
            input_ids: vec![0u32, 3, 1, 3, 3, 3, 1],
            image_grids: vec![
                crate::vision::qwen35_vit::GridThw { t: 1, h: 2, w: 4 },
                crate::vision::qwen35_vit::GridThw { t: 1, h: 2, w: 4 },
            ],
            ..base
        };
        request
            .validate()
            .expect("request must pass validation so only the builder would catch it");
        cfg.max_position_embeddings = request.input_ids.len() - 1;

        let err = prefill_hidden_states_f16(&weights, &cfg, &request)
            .expect_err("over-context request must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("context window"),
            "must fail on the context-window check, before M-RoPE table \
             construction; got: {msg}"
        );
    }

    /// Golden-reference check for `embed_image_f16`'s own image-pad
    /// position-selection wiring (not just the generic `pool_hidden_states`
    /// primitive, which `pool_hidden_states_wrong_positions_change_the_output`
    /// already covers in isolation): `one_image_request`'s fixed layout
    /// `[0, <pad>x4, 1]` puts the pad run at physical positions `[1,2,3,4]`
    /// by construction. This test independently derives the expected
    /// pooled/normalized vector from `prefill_hidden_states_f16` using that
    /// hand-known-correct window and asserts `embed_image_f16` matches it
    /// exactly.
    ///
    /// Mutation-sensitive: an off-by-one in `embed_image_f16`'s
    /// `image_pad_positions` computation (e.g. `.map(|(i, _)| i + 1)`) still
    /// passes `embed_image_f16_is_deterministic`,
    /// `_discriminates_different_images_but_matches_itself`, and
    /// `_wrong_pad_run_placement_changes_embedding` (all of them assert only
    /// relative properties -- determinism, discrimination, "differs from a
    /// differently-shaped request" -- that remain true under a consistently
    /// applied shift), but fails this golden check because the reference
    /// value is computed independently of that internal computation.
    #[test]
    fn embed_image_f16_matches_independently_computed_golden_pool() {
        let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
        let request = one_image_request(vec![0.37f32; 4 * cfg.hidden_size]);

        let hidden_states = prefill_hidden_states_f16(&weights, &cfg, &request)
            .expect("prefill_hidden_states_f16 succeeds");
        let known_correct_pad_positions = [1usize, 2, 3, 4]; // by construction of one_image_request
        let golden = l2_normalize_owned(mean_pool_rows(
            &hidden_states,
            cfg.hidden_size,
            &known_correct_pad_positions,
        ));

        let got = embed_image_f16(&weights, &cfg, &request, PoolingStrategy::MeanVisualTokens)
            .expect("embed_image_f16 succeeds");

        assert_eq!(
            got, golden,
            "embed_image_f16 must pool over exactly the known-correct image-pad positions"
        );
    }
}
