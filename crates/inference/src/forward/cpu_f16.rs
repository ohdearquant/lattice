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
use crate::decoder::standalone_cpu::{StandaloneWeights, generate_with_trace};
use crate::forward::cpu::{elementwise_mul, silu_inplace};
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::model::qwen35::{
    ForwardScratch, KvCache, qwen35_rms_norm, resize, sample_token, should_stop_token,
};
use crate::model::qwen35_config::Qwen35Config;
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
        // the f32 router fix in qwen35/moe.rs::compute_router_probs and the
        // shared attention row contract (#409/#410). `max_logit` can stay finite
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
/// Equivalent to `Qwen35Model::generate` but runs the shared decoder driver over a
/// standalone CPU session whose forward step is `forward_step_f16`. The tokenizer,
/// RoPE table, and generate config are passed explicitly since we operate as
/// standalone functions rather than methods on the model struct.
pub fn generate_f16(
    weights: &F16ModelWeights,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    rope: &RopeTable,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) -> Result<GenerateOutput, crate::error::InferenceError> {
    generate_with_trace(
        StandaloneWeights::F16(weights),
        cfg,
        tokenizer,
        rope,
        prompt,
        gen_cfg,
    )
    .map(|(output, _trace)| output)
}

// ---------------------------------------------------------------------------
// Generate multimodal (f16 weights, ADR-069 Stage 5b)
// ---------------------------------------------------------------------------

/// Greedy-decode a Qwen3.5 vision-language prompt through the CPU f16 forward
/// path (ADR-069 Stage 5b): the decoder splice on top of [`generate_f16`].
///
/// Mirrors the serial prefill and decode sequence `generate_f16` runs, but drives it from
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
        #[expect(
            clippy::expect_used,
            reason = "all_ids is seeded with the prompt before the loop and appended to on every iteration"
        )]
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
#[non_exhaustive]
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
mod tests;
