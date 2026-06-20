// Materialised causal GQA self-attention backward pass.
// Used for gradchecking and for the backward tape through layer-23.
//
// Models the real Qwen3.5 GATED attention: q_proj is [2*q_dim, hidden], its
// output is per-head interleaved [Q_h | gate_h], the gate is applied as
// context *= sigmoid(gate_z) before o_proj, and LoRA on q_proj spans the full
// 2*q_dim. The Q half is normed+roped; the gate half is raw (no norm, no rope).

use super::ops::{linear_vjp, lora_vjp, rope_backward};
use crate::attention::gated::{apply_sigmoid_gate, deinterleave_q_gate};

/// All caches needed for the GQA attention backward.
pub struct AttnCache {
    pub x_input: Vec<f32>,
    /// Raw q_proj(x)+LoRA Q-half, BEFORE q_norm — the actual input to the
    /// q_norm RMSNorm, required by its backward. Deinterleaved (q_dim).
    pub q_raw: Vec<f32>,
    /// Raw k_proj(x), BEFORE k_norm — the actual input to the k_norm RMSNorm.
    pub k_raw: Vec<f32>,
    pub q_pre_rope: Vec<f32>,
    pub k_pre_rope: Vec<f32>,
    pub v: Vec<f32>,
    pub q_h: Vec<Vec<f32>>,
    pub softmax_probs: Vec<Vec<f32>>,
    /// UNGATED attention context (before the sigmoid gate). The gate backward
    /// needs it: d_gate_z = d_context_gated · context_ungated · sigmoid'(gate_z).
    pub context: Vec<f32>,
    /// Raw deinterleaved gate (gate half of q_proj output, no norm/rope), q_dim.
    pub gate_z: Vec<f32>,
    pub h_q: Vec<f32>,
    pub h_v: Vec<f32>,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub rope_dim: usize,
    pub cos_vals: Vec<Vec<f32>>,
    pub sin_vals: Vec<Vec<f32>>,
}

/// Gradients of the GQA attention w.r.t. LoRA params and input.
pub struct AttnGrads {
    pub grad_a_q: Vec<f32>,
    pub grad_b_q: Vec<f32>,
    pub grad_a_v: Vec<f32>,
    pub grad_b_v: Vec<f32>,
    pub dx: Vec<f32>,
}

/// Full materialised GQA backward for a single sequence (prefill/training mode).
///
/// Shape conventions (all row-major f32):
///   w_q: [2*q_dim, hidden]  (per-head interleaved [Q_h | gate_h]; both used)
///   w_k: [kv_dim, hidden]
///   w_v: [kv_dim, hidden]
///   w_o: [hidden, q_dim]
///
/// LoRA is on q_proj and v_proj only (milestone scope).
/// Base weights are frozen — no weight grads.
///
/// Returns (AttnGrads, dx: [seq_len, hidden]) where dx is the gradient
/// flowing into this layer's input (for the residual stream).
#[allow(clippy::too_many_arguments)]
pub fn gqa_backward(
    dy_out: &[f32],
    cache: &AttnCache,
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    lora_a_q: Option<&[f32]>,
    lora_b_q: Option<&[f32]>,
    lora_a_v: Option<&[f32]>,
    lora_b_v: Option<&[f32]>,
    lora_rank: usize,
    lora_scale: f32,
) -> AttnGrads {
    let AttnCache {
        x_input,
        q_raw,
        k_raw,
        q_pre_rope: _,
        k_pre_rope, // holds POST-rope k (used for the dL/dq score backward)
        v,
        q_h, // holds POST-rope q (used for the dL/dk score backward)
        softmax_probs,
        context, // UNGATED attention context (for the gate backward)
        gate_z,  // raw deinterleaved gate (no norm/rope)
        h_q,     // LoRA q activation A_q·x
        h_v,     // LoRA v activation A_v·x
        num_q_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        rope_dim,
        cos_vals,
        sin_vals,
    } = cache;

    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let groups = num_q_heads / num_kv_heads;
    let scale = 1.0 / (*head_dim as f32).sqrt();
    let hidden = x_input.len() / seq_len;

    let mut grad_a_q = vec![0.0f32; lora_rank * hidden];
    // grad_B_q spans the full 2*q_dim q_proj output (Q rows + gate rows).
    let mut grad_b_q = vec![0.0f32; 2 * q_dim * lora_rank];
    let mut grad_a_v = vec![0.0f32; lora_rank * hidden];
    let mut grad_b_v = vec![0.0f32; kv_dim * lora_rank];
    let mut dx_total = vec![0.0f32; seq_len * hidden];

    // ---- Phase 1: score/context backward, GLOBAL accumulation ----
    // d_q_post_rope[t]: q at position t is consumed only by query t, written once.
    // d_k_post_rope[s] / d_v[s]: K/V at position s receive gradient from EVERY query
    // t >= s (causal mask), so these accumulate across the whole t-loop. The old code
    // sized these per-t (`(t+1)*kv_dim`) and read back only the diagonal [t] slice,
    // discarding every off-diagonal (t > s) contribution — the core gradient bug.
    let mut d_q_post_rope = vec![0.0f32; seq_len * q_dim];
    let mut d_k_post_rope = vec![0.0f32; seq_len * kv_dim];
    let mut d_v = vec![0.0f32; seq_len * kv_dim];
    // d_gate_z[t]: gradient w.r.t. the raw gate (gate half of q_proj output).
    let mut d_gate_z = vec![0.0f32; seq_len * q_dim];

    for t in 0..*seq_len {
        let dy_t = &dy_out[t * hidden..(t + 1) * hidden];
        // o_proj backward gives the gradient w.r.t. the GATED context.
        let d_context_gated = linear_vjp(w_o, dy_t, q_dim, hidden);
        // Split the sigmoid gate: context_gated = context_ungated · sigmoid(gate_z).
        //   d_context_ungated = d_context_gated · sigmoid(gate_z)
        //   d_gate_z          = d_context_gated · context_ungated · sigmoid'(gate_z)
        let mut d_context_t = vec![0.0f32; q_dim];
        for i in 0..q_dim {
            let g = 1.0 / (1.0 + (-gate_z[t * q_dim + i]).exp());
            d_context_t[i] = d_context_gated[i] * g;
            d_gate_z[t * q_dim + i] = d_context_gated[i] * context[t * q_dim + i] * g * (1.0 - g);
        }

        for qh in 0..*num_q_heads {
            let kvh = qh / groups;
            let q_off = qh * head_dim;
            let kv_off = kvh * head_dim;

            let d_ctx_h = &d_context_t[q_off..q_off + head_dim];
            let probs_h = &softmax_probs[t][qh * (t + 1)..(qh + 1) * (t + 1)];
            let q_post = &q_h[t]; // post-rope q for position t

            // d_v[s] += probs[s] * d_ctx_h  and  d_probs[s] = d_ctx_h · v[s]
            let mut d_probs = vec![0.0f32; t + 1];
            for s in 0..=t {
                let v_base = s * kv_dim + kv_off;
                let p = probs_h[s];
                let mut dp = 0.0f32;
                for d in 0..*head_dim {
                    d_v[v_base + d] += p * d_ctx_h[d];
                    dp += d_ctx_h[d] * v[v_base + d];
                }
                d_probs[s] = dp;
            }

            // softmax backward: d_score[s] = probs[s] * (d_probs[s] - Σ probs·d_probs)
            let sum_pd: f32 = probs_h
                .iter()
                .zip(d_probs.iter())
                .map(|(p, dp)| p * dp)
                .sum();
            for s in 0..=t {
                let d_score = probs_h[s] * (d_probs[s] - sum_pd) * scale;
                let k_base = s * kv_dim + kv_off;
                let q_base = t * q_dim + q_off;
                for d in 0..*head_dim {
                    // dL/dq_t += d_score · k_s   (k_pre_rope holds POST-rope k)
                    d_q_post_rope[q_base + d] += d_score * k_pre_rope[k_base + d];
                    // dL/dk_s += d_score · q_t   (global accumulate over all t >= s)
                    d_k_post_rope[k_base + d] += d_score * q_post[q_off + d];
                }
            }
        }
    }

    // ---- Phase 2: per-position rope → norm (RAW pre-norm input) → projection ----
    let eps = 1e-6f32;

    for t in 0..*seq_len {
        let x_t = &x_input[t * hidden..(t + 1) * hidden];
        let mut dx_t = vec![0.0f32; hidden];

        // ---- Q path ----
        let mut d_q_normed = vec![0.0f32; q_dim];
        for qh in 0..*num_q_heads {
            let start = qh * head_dim;
            let dq_head = &d_q_post_rope[t * q_dim + start..t * q_dim + start + head_dim];
            let back = rope_backward(dq_head, &cos_vals[t], &sin_vals[t], *rope_dim);
            d_q_normed[start..start + head_dim].copy_from_slice(&back);
        }
        // q_norm backward: RMSNorm input is q_raw (pre-norm), NOT the post-norm value.
        let mut d_q_raw = vec![0.0f32; q_dim];
        for qh in 0..*num_q_heads {
            let start = qh * head_dim;
            let q_head = &q_raw[t * q_dim + start..t * q_dim + start + head_dim];
            let mean_sq: f32 = q_head.iter().map(|xi| xi * xi).sum::<f32>() / *head_dim as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            let dg = &d_q_normed[start..start + head_dim];
            // Shifted weight (1 + gamma) to match the forward's q_norm.
            let sum_xwg: f32 = (0..*head_dim)
                .map(|j| q_head[j] * (1.0 + q_norm_w[j]) * dg[j])
                .sum();
            let inv3_over_d = inv_rms * inv_rms * inv_rms / *head_dim as f32;
            for j in 0..*head_dim {
                d_q_raw[start + j] =
                    (1.0 + q_norm_w[j]) * dg[j] * inv_rms - q_head[j] * inv3_over_d * sum_xwg;
            }
        }
        // Re-interleave the Q-half grad (d_q_raw, post norm/rope backward) with
        // the gate-half grad (d_gate_z, no norm/rope) into the full 2*q_dim
        // q_proj output gradient — the inverse of deinterleave_q_gate.
        let mut d_q_and_gate = vec![0.0f32; 2 * q_dim];
        for qh in 0..*num_q_heads {
            let qsrc = qh * head_dim;
            let dst = qh * 2 * head_dim;
            d_q_and_gate[dst..dst + head_dim].copy_from_slice(&d_q_raw[qsrc..qsrc + head_dim]);
            d_q_and_gate[dst + head_dim..dst + 2 * head_dim]
                .copy_from_slice(&d_gate_z[t * q_dim + qsrc..t * q_dim + qsrc + head_dim]);
        }
        let dx_q = linear_vjp(w_q, &d_q_and_gate, hidden, 2 * q_dim);
        for j in 0..hidden {
            dx_t[j] += dx_q[j];
        }
        if let (Some(la), Some(lb)) = (lora_a_q, lora_b_q) {
            let h_q_t = &h_q[t * lora_rank..(t + 1) * lora_rank];
            let (gb, ga, dx_lora) = lora_vjp(
                &d_q_and_gate,
                x_t,
                h_q_t,
                la,
                lb,
                lora_rank,
                hidden,
                2 * q_dim,
                lora_scale,
            );
            for k in 0..ga.len() {
                grad_a_q[k] += ga[k];
            }
            for k in 0..gb.len() {
                grad_b_q[k] += gb[k];
            }
            for j in 0..hidden {
                dx_t[j] += dx_lora[j];
            }
        }

        // ---- K path (no LoRA) ----
        let mut d_k_normed = vec![0.0f32; kv_dim];
        for kvh in 0..*num_kv_heads {
            let start = kvh * head_dim;
            let dk_head = &d_k_post_rope[t * kv_dim + start..t * kv_dim + start + head_dim];
            let back = rope_backward(dk_head, &cos_vals[t], &sin_vals[t], *rope_dim);
            d_k_normed[start..start + head_dim].copy_from_slice(&back);
        }
        let mut d_k_raw = vec![0.0f32; kv_dim];
        for kvh in 0..*num_kv_heads {
            let start = kvh * head_dim;
            let k_head = &k_raw[t * kv_dim + start..t * kv_dim + start + head_dim];
            let mean_sq: f32 = k_head.iter().map(|xi| xi * xi).sum::<f32>() / *head_dim as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            let dg = &d_k_normed[start..start + head_dim];
            // Shifted weight (1 + gamma) to match the forward's k_norm.
            let sum_xwg: f32 = (0..*head_dim)
                .map(|j| k_head[j] * (1.0 + k_norm_w[j]) * dg[j])
                .sum();
            let inv3_over_d = inv_rms * inv_rms * inv_rms / *head_dim as f32;
            for j in 0..*head_dim {
                d_k_raw[start + j] =
                    (1.0 + k_norm_w[j]) * dg[j] * inv_rms - k_head[j] * inv3_over_d * sum_xwg;
            }
        }
        let dx_k = linear_vjp(w_k, &d_k_raw, hidden, kv_dim);
        for j in 0..hidden {
            dx_t[j] += dx_k[j];
        }

        // ---- V path ----
        let d_v_t = &d_v[t * kv_dim..(t + 1) * kv_dim];
        let dx_v = linear_vjp(w_v, d_v_t, hidden, kv_dim);
        for j in 0..hidden {
            dx_t[j] += dx_v[j];
        }
        if let (Some(la), Some(lb)) = (lora_a_v, lora_b_v) {
            let h_v_t = &h_v[t * lora_rank..(t + 1) * lora_rank];
            let (gb, ga, dx_lora) = lora_vjp(
                d_v_t, x_t, h_v_t, la, lb, lora_rank, hidden, kv_dim, lora_scale,
            );
            for k in 0..ga.len() {
                grad_a_v[k] += ga[k];
            }
            for k in 0..gb.len() {
                grad_b_v[k] += gb[k];
            }
            for j in 0..hidden {
                dx_t[j] += dx_lora[j];
            }
        }

        for j in 0..hidden {
            dx_total[t * hidden + j] += dx_t[j];
        }
    }

    AttnGrads {
        grad_a_q,
        grad_b_q,
        grad_a_v,
        grad_b_v,
        dx: dx_total,
    }
}

// Compute the GQA forward and cache everything needed for the backward.
// Returns (output [seq_len, hidden], cache).
#[allow(clippy::too_many_arguments)]
pub fn gqa_forward_with_cache(
    x: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    lora_a_q: Option<&[f32]>,
    lora_b_q: Option<&[f32]>,
    lora_a_v: Option<&[f32]>,
    lora_b_v: Option<&[f32]>,
    lora_rank: usize,
    lora_scale: f32,
    seq_len: usize,
    hidden: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
    eps: f32,
) -> (Vec<f32>, AttnCache) {
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let groups = num_q_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let half = rope_dim / 2;

    assert_eq!(x.len(), seq_len * hidden);
    assert_eq!(w_q.len(), 2 * q_dim * hidden);
    assert_eq!(w_k.len(), kv_dim * hidden);
    assert_eq!(w_v.len(), kv_dim * hidden);
    assert_eq!(w_o.len(), hidden * q_dim);
    assert_eq!(cos_table.len(), seq_len * half);
    assert_eq!(sin_table.len(), seq_len * half);

    let mut q_pre_rope = vec![0.0f32; seq_len * q_dim];
    let mut k_pre_rope = vec![0.0f32; seq_len * kv_dim];
    let mut v = vec![0.0f32; seq_len * kv_dim];
    let mut gate_z = vec![0.0f32; seq_len * q_dim];
    let mut h_q = vec![0.0f32; seq_len * lora_rank];
    let mut h_v = vec![0.0f32; seq_len * lora_rank];

    for t in 0..seq_len {
        let x_t = &x[t * hidden..(t + 1) * hidden];

        // q+gate projection: full 2*q_dim rows of w_q, LoRA on the full output,
        // then deinterleave per head into Q (q_dim) and gate_z (q_dim).
        let mut q_and_gate_t = vec![0.0f32; 2 * q_dim];
        for i in 0..2 * q_dim {
            let row = &w_q[i * hidden..(i + 1) * hidden];
            q_and_gate_t[i] = row.iter().zip(x_t.iter()).map(|(a, b)| a * b).sum();
        }
        if let (Some(la), Some(lb)) = (lora_a_q, lora_b_q) {
            let h = &mut h_q[t * lora_rank..(t + 1) * lora_rank];
            for r in 0..lora_rank {
                h[r] = la[r * hidden..(r + 1) * hidden]
                    .iter()
                    .zip(x_t.iter())
                    .map(|(a, b)| a * b)
                    .sum();
            }
            for i in 0..2 * q_dim {
                let acc: f32 = lora_scale
                    * lb[i * lora_rank..(i + 1) * lora_rank]
                        .iter()
                        .zip(h.iter())
                        .map(|(b, hi)| b * hi)
                        .sum::<f32>();
                q_and_gate_t[i] += acc;
            }
        }
        deinterleave_q_gate(
            &q_and_gate_t,
            &mut q_pre_rope[t * q_dim..(t + 1) * q_dim],
            &mut gate_z[t * q_dim..(t + 1) * q_dim],
            num_q_heads,
            head_dim,
        );

        // k_proj
        let k_t = &mut k_pre_rope[t * kv_dim..(t + 1) * kv_dim];
        for i in 0..kv_dim {
            let row = &w_k[i * hidden..(i + 1) * hidden];
            k_t[i] = row.iter().zip(x_t.iter()).map(|(a, b)| a * b).sum();
        }

        // v_proj
        let v_t = &mut v[t * kv_dim..(t + 1) * kv_dim];
        for i in 0..kv_dim {
            let row = &w_v[i * hidden..(i + 1) * hidden];
            v_t[i] = row.iter().zip(x_t.iter()).map(|(a, b)| a * b).sum();
        }
        if let (Some(la), Some(lb)) = (lora_a_v, lora_b_v) {
            let h = &mut h_v[t * lora_rank..(t + 1) * lora_rank];
            for r in 0..lora_rank {
                h[r] = la[r * hidden..(r + 1) * hidden]
                    .iter()
                    .zip(x_t.iter())
                    .map(|(a, b)| a * b)
                    .sum();
            }
            for i in 0..kv_dim {
                let acc: f32 = lora_scale
                    * lb[i * lora_rank..(i + 1) * lora_rank]
                        .iter()
                        .zip(h.iter())
                        .map(|(b, hi)| b * hi)
                        .sum::<f32>();
                v_t[i] += acc;
            }
        }
    }

    // Cache the raw (pre-norm) projections — the q_norm/k_norm backward needs
    // its own input, which the in-place normalization below would destroy.
    let q_raw = q_pre_rope.clone();
    let k_raw = k_pre_rope.clone();

    // q_norm and k_norm (per-head RMSNorm). Qwen3.5 uses the SHIFTED weight
    // `(1 + gamma)`, matching `qwen35_rms_norm` — plain `gamma` would diverge
    // from the real model (a divergence the self-consistent gradcheck can't see).
    for t in 0..seq_len {
        for qh in 0..num_q_heads {
            let start = t * q_dim + qh * head_dim;
            let q_head = &mut q_pre_rope[start..start + head_dim];
            let mean_sq: f32 = q_head.iter().map(|xi| xi * xi).sum::<f32>() / head_dim as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for (j, qj) in q_head.iter_mut().enumerate() {
                *qj *= (1.0 + q_norm_w[j]) * inv_rms;
            }
        }
        for kvh in 0..num_kv_heads {
            let start = t * kv_dim + kvh * head_dim;
            let k_head = &mut k_pre_rope[start..start + head_dim];
            let mean_sq: f32 = k_head.iter().map(|xi| xi * xi).sum::<f32>() / head_dim as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for (j, kj) in k_head.iter_mut().enumerate() {
                *kj *= (1.0 + k_norm_w[j]) * inv_rms;
            }
        }
    }

    // Build per-position cos/sin
    let mut cos_vals: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    let mut sin_vals: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        cos_vals.push(cos_table[t * half..(t + 1) * half].to_vec());
        sin_vals.push(sin_table[t * half..(t + 1) * half].to_vec());
    }

    // Apply RoPE
    let mut q_after_rope: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    let mut k_after_rope = k_pre_rope.clone();
    for t in 0..seq_len {
        let mut q_t = q_pre_rope[t * q_dim..(t + 1) * q_dim].to_vec();
        for qh in 0..num_q_heads {
            let start = qh * head_dim;
            for i in 0..half {
                let c = cos_vals[t][i];
                let s = sin_vals[t][i];
                let x0 = q_t[start + i];
                let x1 = q_t[start + half + i];
                q_t[start + i] = x0 * c - x1 * s;
                q_t[start + half + i] = x0 * s + x1 * c;
            }
        }
        q_after_rope.push(q_t);

        let k_t = &mut k_after_rope[t * kv_dim..(t + 1) * kv_dim];
        for kvh in 0..num_kv_heads {
            let start = kvh * head_dim;
            let x0_vals: Vec<f32> = (0..half).map(|i| k_t[start + i]).collect();
            let x1_vals: Vec<f32> = (0..half).map(|i| k_t[start + half + i]).collect();
            for i in 0..half {
                let c = cos_vals[t][i];
                let s = sin_vals[t][i];
                k_t[start + i] = x0_vals[i] * c - x1_vals[i] * s;
                k_t[start + half + i] = x0_vals[i] * s + x1_vals[i] * c;
            }
        }
    }

    // Causal attention
    let mut softmax_probs: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    let mut context = vec![0.0f32; seq_len * q_dim];

    for t in 0..seq_len {
        let mut probs_all = vec![0.0f32; num_q_heads * (t + 1)];

        for qh in 0..num_q_heads {
            let kvh = qh / groups;
            let q_off = qh * head_dim;
            let kv_off = kvh * head_dim;
            let q = &q_after_rope[t][q_off..q_off + head_dim];

            let mut max_score = f32::NEG_INFINITY;
            let prob_start = qh * (t + 1);
            for s in 0..=t {
                let k = &k_after_rope[s * kv_dim + kv_off..s * kv_dim + kv_off + head_dim];
                let dot: f32 = q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum();
                let score = dot * scale;
                probs_all[prob_start + s] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            let mut sum_exp = 0.0f32;
            for s in 0..=t {
                let e = (probs_all[prob_start + s] - max_score).exp();
                probs_all[prob_start + s] = e;
                sum_exp += e;
            }
            let inv = 1.0 / sum_exp;
            for s in 0..=t {
                probs_all[prob_start + s] *= inv;
            }

            let ctx_off = t * q_dim + qh * head_dim;
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for s in 0..=t {
                    let v_off = s * kv_dim + kv_off;
                    sum += probs_all[prob_start + s] * v[v_off + d];
                }
                context[ctx_off + d] = sum;
            }
        }
        softmax_probs.push(probs_all);
    }

    // Sigmoid gate: cache the ungated context (needed by the gate backward),
    // then gate in place — context_gated[i] = context[i] * sigmoid(gate_z[i]).
    let context_ungated = context.clone();
    for t in 0..seq_len {
        apply_sigmoid_gate(
            &mut context[t * q_dim..(t + 1) * q_dim],
            &gate_z[t * q_dim..(t + 1) * q_dim],
        );
    }

    // O projection (on the gated context)
    let mut output = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let ctx_t = &context[t * q_dim..(t + 1) * q_dim];
        let out_t = &mut output[t * hidden..(t + 1) * hidden];
        for i in 0..hidden {
            let row = &w_o[i * q_dim..(i + 1) * q_dim];
            out_t[i] = row.iter().zip(ctx_t.iter()).map(|(a, b)| a * b).sum();
        }
    }

    let cache = AttnCache {
        x_input: x.to_vec(),
        q_raw,
        k_raw,
        q_pre_rope: q_pre_rope.clone(),
        k_pre_rope: k_after_rope,
        v,
        q_h: q_after_rope,
        softmax_probs,
        context: context_ungated,
        gate_z,
        h_q,
        h_v,
        num_q_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        rope_dim,
        cos_vals,
        sin_vals,
    };
    (output, cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(state: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            out.push(((x >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale);
        }
        out
    }

    /// GQA gradcheck: analytic grad_A,B vs central FD for a 2-layer all-GQA tiny fixture.
    #[test]
    fn gqa_lora_gradcheck() {
        let seq_len = 3;
        let hidden = 8;
        let num_q_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let rope_dim = 2;
        let lora_rank = 2;
        let lora_alpha = 4.0f32;
        let lora_scale = lora_alpha / lora_rank as f32;
        let eps_norm = 1e-6f32;

        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let half = rope_dim / 2;

        let mut rng = 0xDEAD_BEEF_u64;
        let w_q = rand_vec(&mut rng, 2 * q_dim * hidden, 0.1);
        let w_k = rand_vec(&mut rng, kv_dim * hidden, 0.1);
        let w_v = rand_vec(&mut rng, kv_dim * hidden, 0.1);
        let w_o = rand_vec(&mut rng, hidden * q_dim, 0.1);
        let q_norm_w = vec![1.0f32; head_dim];
        let k_norm_w = vec![1.0f32; head_dim];

        let x = rand_vec(&mut rng, seq_len * hidden, 0.2);
        let dy = rand_vec(&mut rng, seq_len * hidden, 0.1);

        // LoRA params for q and v. B MUST be non-zero: at B=0 the LoRA delta
        // B·(A·x) vanishes, the forward is independent of A, and grad_A is
        // identically zero on both sides — a vacuous check that validates nothing.
        // B_q spans the full 2*q_dim q_proj output; the gate rows being non-zero
        // de-vacuums the gate path (otherwise the sigmoid gate is untested).
        let lora_a_q = rand_vec(&mut rng, lora_rank * hidden, 0.05);
        let lora_b_q = rand_vec(&mut rng, 2 * q_dim * lora_rank, 0.05);
        let lora_a_v = rand_vec(&mut rng, lora_rank * hidden, 0.05);
        let lora_b_v = rand_vec(&mut rng, kv_dim * lora_rank, 0.05);

        // cos/sin table — build analytically
        let cos_table: Vec<f32> = (0..seq_len * half)
            .map(|i| {
                let pos = i / half;
                let dim = i % half;
                let theta = (pos as f32) / (10000f32.powf(2.0 * dim as f32 / rope_dim as f32));
                theta.cos()
            })
            .collect();
        let sin_table: Vec<f32> = (0..seq_len * half)
            .map(|i| {
                let pos = i / half;
                let dim = i % half;
                let theta = (pos as f32) / (10000f32.powf(2.0 * dim as f32 / rope_dim as f32));
                theta.sin()
            })
            .collect();

        let loss_fn = |a_q: &[f32], b_q: &[f32], a_v: &[f32], b_v: &[f32]| -> f32 {
            let (out, _) = gqa_forward_with_cache(
                &x,
                &w_q,
                &w_k,
                &w_v,
                &w_o,
                &q_norm_w,
                &k_norm_w,
                Some(a_q),
                Some(b_q),
                Some(a_v),
                Some(b_v),
                lora_rank,
                lora_scale,
                seq_len,
                hidden,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rope_dim,
                &cos_table,
                &sin_table,
                eps_norm,
            );
            dy.iter().zip(out.iter()).map(|(d, o)| d * o).sum()
        };

        let (out, cache) = gqa_forward_with_cache(
            &x,
            &w_q,
            &w_k,
            &w_v,
            &w_o,
            &q_norm_w,
            &k_norm_w,
            Some(&lora_a_q),
            Some(&lora_b_q),
            Some(&lora_a_v),
            Some(&lora_b_v),
            lora_rank,
            lora_scale,
            seq_len,
            hidden,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
            &cos_table,
            &sin_table,
            eps_norm,
        );
        let _ = out;

        let grads = gqa_backward(
            &dy,
            &cache,
            &w_q,
            &w_k,
            &w_v,
            &w_o,
            &q_norm_w,
            &k_norm_w,
            Some(&lora_a_q),
            Some(&lora_b_q),
            Some(&lora_a_v),
            Some(&lora_b_v),
            lora_rank,
            lora_scale,
        );

        let fd_eps = 1e-3f32;

        // FD for grad_A_q
        let mut fd_a_q = vec![0.0f32; lora_rank * hidden];
        for k in 0..lora_rank * hidden {
            let mut ap = lora_a_q.clone();
            let mut am = lora_a_q.clone();
            ap[k] += fd_eps;
            am[k] -= fd_eps;
            fd_a_q[k] = (loss_fn(&ap, &lora_b_q, &lora_a_v, &lora_b_v)
                - loss_fn(&am, &lora_b_q, &lora_a_v, &lora_b_v))
                / (2.0 * fd_eps);
        }

        // FD for grad_A_v
        let mut fd_a_v = vec![0.0f32; lora_rank * hidden];
        for k in 0..lora_rank * hidden {
            let mut av = lora_a_v.clone();
            let mut am = lora_a_v.clone();
            av[k] += fd_eps;
            am[k] -= fd_eps;
            fd_a_v[k] = (loss_fn(&lora_a_q, &lora_b_q, &av, &lora_b_v)
                - loss_fn(&lora_a_q, &lora_b_q, &am, &lora_b_v))
                / (2.0 * fd_eps);
        }

        // FD for grad_B_q (full 2*q_dim output rows)
        let mut fd_b_q = vec![0.0f32; 2 * q_dim * lora_rank];
        for k in 0..2 * q_dim * lora_rank {
            let mut bp = lora_b_q.clone();
            let mut bm = lora_b_q.clone();
            bp[k] += fd_eps;
            bm[k] -= fd_eps;
            fd_b_q[k] = (loss_fn(&lora_a_q, &bp, &lora_a_v, &lora_b_v)
                - loss_fn(&lora_a_q, &bm, &lora_a_v, &lora_b_v))
                / (2.0 * fd_eps);
        }

        // FD for grad_B_v
        let mut fd_b_v = vec![0.0f32; kv_dim * lora_rank];
        for k in 0..kv_dim * lora_rank {
            let mut bp = lora_b_v.clone();
            let mut bm = lora_b_v.clone();
            bp[k] += fd_eps;
            bm[k] -= fd_eps;
            fd_b_v[k] = (loss_fn(&lora_a_q, &lora_b_q, &lora_a_v, &bp)
                - loss_fn(&lora_a_q, &lora_b_q, &lora_a_v, &bm))
                / (2.0 * fd_eps);
        }

        let rel_err = |a: &[f32], b: &[f32]| -> f64 {
            let diff_sq: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| ((x - y) as f64).powi(2))
                .sum();
            let norm_sq: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>()
                + b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
            (2.0 * diff_sq / norm_sq.max(1e-30)).sqrt()
        };

        let err_aq = rel_err(&grads.grad_a_q, &fd_a_q);
        let err_av = rel_err(&grads.grad_a_v, &fd_a_v);
        let err_bq = rel_err(&grads.grad_b_q, &fd_b_q);
        let err_bv = rel_err(&grads.grad_b_v, &fd_b_v);
        eprintln!(
            "gqa_lora_gradcheck: grad_A_q={err_aq:.2e} grad_A_v={err_av:.2e} grad_B_q={err_bq:.2e} grad_B_v={err_bv:.2e}"
        );
        assert!(err_aq < 1e-2, "GQA grad_A_q rel_err {err_aq:.2e} >= 1e-2");
        assert!(err_av < 1e-2, "GQA grad_A_v rel_err {err_av:.2e} >= 1e-2");
        assert!(err_bq < 1e-2, "GQA grad_B_q rel_err {err_bq:.2e} >= 1e-2");
        assert!(err_bv < 1e-2, "GQA grad_B_v rel_err {err_bv:.2e} >= 1e-2");
    }
}
