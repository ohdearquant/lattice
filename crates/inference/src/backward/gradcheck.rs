//! Tiny-model gradient-checking utilities, including tiny weights, tiny forward/loss helpers, and `gradcheck_lora`.
// End-to-end gradcheck harness: tiny all-GQA 2-layer fixture,
// hidden=64, 4 Q-heads, 1 KV-head, head_dim=16, no GDN layers.
// Validates that the entire backward chain produces analytic grads
// matching central finite differences.

use super::{
    attention_gqa::{gqa_backward, gqa_forward_with_cache},
    ops::{linear_vjp, rmsnorm_backward, swiglu_backward},
    tape::{rms_norm_forward, swiglu_forward},
};

/// Tiny all-GQA transformer config for gradchecking.
#[derive(Clone)]
pub struct TinyConfig {
    pub hidden: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub num_layers: usize,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub eps: f32,
}

impl Default for TinyConfig {
    fn default() -> Self {
        Self {
            hidden: 64,
            num_q_heads: 4,
            num_kv_heads: 1,
            head_dim: 16,
            rope_dim: 8,
            inter: 128,
            vocab: 256,
            num_layers: 2,
            lora_rank: 4,
            lora_alpha: 8.0,
            eps: 1e-6,
        }
    }
}

/// Deterministic PRNG for generating test weights.
fn next_rng(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * 0.1
}

fn rand_vec(state: &mut u64, n: usize) -> Vec<f32> {
    (0..n).map(|_| next_rng(state)).collect()
}

/// All weights for the tiny 2-layer model.
pub struct TinyWeights {
    pub embed: Vec<f32>,
    /// Per layer: (w_q, w_k, w_v, w_o, q_norm, k_norm, pre_attn_norm, post_attn_norm, w_gate, w_up, w_down)
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub lora_a_q: Vec<f32>,
    pub lora_b_q: Vec<f32>,
    pub lora_a_v: Vec<f32>,
    pub lora_b_v: Vec<f32>,
}

pub struct LayerWeights {
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub pre_attn_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

pub fn make_tiny_weights(cfg: &TinyConfig) -> TinyWeights {
    let mut rng = 0xCAFE_BABE_u64;
    let q_dim = cfg.num_q_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    let lora_scale = cfg.lora_alpha / cfg.lora_rank as f32;
    let _ = lora_scale;

    let mut layers = Vec::new();
    for _ in 0..cfg.num_layers {
        layers.push(LayerWeights {
            w_q: rand_vec(&mut rng, 2 * q_dim * cfg.hidden),
            w_k: rand_vec(&mut rng, kv_dim * cfg.hidden),
            w_v: rand_vec(&mut rng, kv_dim * cfg.hidden),
            w_o: rand_vec(&mut rng, cfg.hidden * q_dim),
            q_norm: vec![1.0f32; cfg.head_dim],
            k_norm: vec![1.0f32; cfg.head_dim],
            pre_attn_norm: vec![1.0f32; cfg.hidden],
            post_attn_norm: vec![1.0f32; cfg.hidden],
            w_gate: rand_vec(&mut rng, cfg.inter * cfg.hidden),
            w_up: rand_vec(&mut rng, cfg.inter * cfg.hidden),
            w_down: rand_vec(&mut rng, cfg.hidden * cfg.inter),
        });
    }

    // NON-ZERO B: at B=0 the LoRA delta B·(A·x) vanishes, so the forward is
    // independent of A and both FD and analytic grad_A are identically zero —
    // a vacuous check. Random B makes the A-gradient path observable.
    let lora_a_q = rand_vec(&mut rng, cfg.lora_rank * cfg.hidden);
    // B_q spans the full 2*q_dim q_proj output (Q rows + gate rows).
    let lora_b_q = rand_vec(&mut rng, 2 * q_dim * cfg.lora_rank);
    let lora_a_v = rand_vec(&mut rng, cfg.lora_rank * cfg.hidden);
    let lora_b_v = rand_vec(&mut rng, kv_dim * cfg.lora_rank);

    TinyWeights {
        embed: rand_vec(&mut rng, cfg.vocab * cfg.hidden),
        layers,
        final_norm: vec![1.0f32; cfg.hidden],
        lm_head: rand_vec(&mut rng, cfg.vocab * cfg.hidden),
        lora_a_q,
        lora_b_q,
        lora_a_v,
        lora_b_v,
    }
}

/// Build RoPE cos/sin tables for tiny config.
fn build_rope_tables(cfg: &TinyConfig, seq_len: usize) -> (Vec<f32>, Vec<f32>) {
    let half = cfg.rope_dim / 2;
    let mut cos_t = Vec::with_capacity(seq_len * half);
    let mut sin_t = Vec::with_capacity(seq_len * half);
    for t in 0..seq_len {
        for d in 0..half {
            let theta = (t as f32) / (10000f32.powf(2.0 * d as f32 / cfg.rope_dim as f32));
            cos_t.push(theta.cos());
            sin_t.push(theta.sin());
        }
    }
    (cos_t, sin_t)
}

/// Run the full 2-layer tiny forward pass, returning (logits [vocab], all caches).
pub fn tiny_forward(
    token_ids: &[u32],
    weights: &TinyWeights,
    cfg: &TinyConfig,
    // only apply LoRA to last layer in this milestone scope
    apply_lora_layer: usize,
) -> (Vec<f32>, Vec<Vec<f32>>, Vec<f32>, f32) {
    let seq_len = token_ids.len();
    let (cos_t, sin_t) = build_rope_tables(cfg, seq_len);
    let lora_scale = cfg.lora_alpha / cfg.lora_rank as f32;

    // Embed all tokens: [seq_len, hidden]
    let mut hidden_seq: Vec<f32> = token_ids
        .iter()
        .flat_map(|&tid| {
            let start = tid as usize * cfg.hidden;
            weights.embed[start..start + cfg.hidden].iter().copied()
        })
        .collect();

    // Forward through each layer
    let mut layer_inputs: Vec<Vec<f32>> = Vec::new(); // residual stream inputs at each layer

    for l in 0..cfg.num_layers {
        layer_inputs.push(hidden_seq.clone());
        let lw = &weights.layers[l];
        let lora_a_q = if l == apply_lora_layer {
            Some(weights.lora_a_q.as_slice())
        } else {
            None
        };
        let lora_b_q = if l == apply_lora_layer {
            Some(weights.lora_b_q.as_slice())
        } else {
            None
        };
        let lora_a_v = if l == apply_lora_layer {
            Some(weights.lora_a_v.as_slice())
        } else {
            None
        };
        let lora_b_v = if l == apply_lora_layer {
            Some(weights.lora_b_v.as_slice())
        } else {
            None
        };

        let mut new_hidden = hidden_seq.clone();

        // Pre-attn RMSNorm: per-token
        let mut normed_pre_attn = vec![0.0f32; seq_len * cfg.hidden];
        for t in 0..seq_len {
            let x_t = &hidden_seq[t * cfg.hidden..(t + 1) * cfg.hidden];
            let (n, _) = rms_norm_forward(x_t, &lw.pre_attn_norm, cfg.eps);
            normed_pre_attn[t * cfg.hidden..(t + 1) * cfg.hidden].copy_from_slice(&n);
        }

        // GQA attention
        let (attn_out_seq, _cache) = gqa_forward_with_cache(
            &normed_pre_attn,
            &lw.w_q,
            &lw.w_k,
            &lw.w_v,
            &lw.w_o,
            &lw.q_norm,
            &lw.k_norm,
            lora_a_q,
            lora_b_q,
            lora_a_v,
            lora_b_v,
            cfg.lora_rank,
            lora_scale,
            seq_len,
            cfg.hidden,
            cfg.num_q_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.rope_dim,
            &cos_t,
            &sin_t,
            cfg.eps,
        );

        // Residual after attn
        for t in 0..seq_len {
            for j in 0..cfg.hidden {
                new_hidden[t * cfg.hidden + j] =
                    hidden_seq[t * cfg.hidden + j] + attn_out_seq[t * cfg.hidden + j];
            }
        }

        // Post-attn RMSNorm + SwiGLU FFN per token
        for t in 0..seq_len {
            let x_t = &new_hidden[t * cfg.hidden..(t + 1) * cfg.hidden].to_vec();
            let (normed, _inv) = rms_norm_forward(x_t, &lw.post_attn_norm, cfg.eps);
            let (ffn_out, _gp, _up) = swiglu_forward(
                &normed, &lw.w_gate, &lw.w_up, &lw.w_down, cfg.hidden, cfg.inter,
            );
            for j in 0..cfg.hidden {
                new_hidden[t * cfg.hidden + j] = x_t[j] + ffn_out[j];
            }
        }

        hidden_seq = new_hidden;
    }

    // Final norm (take last token only for loss)
    let last_t = seq_len - 1;
    let x_last = &hidden_seq[last_t * cfg.hidden..(last_t + 1) * cfg.hidden].to_vec();
    let (final_normed, inv_rms_final) = rms_norm_forward(x_last, &weights.final_norm, cfg.eps);

    // lm_head
    let mut logits = vec![0.0f32; cfg.vocab];
    for i in 0..cfg.vocab {
        logits[i] = weights.lm_head[i * cfg.hidden..(i + 1) * cfg.hidden]
            .iter()
            .zip(final_normed.iter())
            .map(|(a, b)| a * b)
            .sum();
    }

    (logits, layer_inputs, final_normed, inv_rms_final)
}

/// Compute scalar CE loss for a single next-token prediction (last position, target).
pub fn tiny_loss(token_ids: &[u32], target: u32, weights: &TinyWeights, cfg: &TinyConfig) -> f32 {
    let (logits, _, _, _) = tiny_forward(token_ids, weights, cfg, cfg.num_layers - 1);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    let log_prob = (logits[target as usize] - max) as f64 - sum_exp.ln();
    (-log_prob) as f32
}

/// Central-FD gradient check for LoRA A matrices (q and v) in the last layer.
///
/// Compares analytic backward with FD estimate. Passes if rel_err < tol.
pub fn gradcheck_lora(seq: &[u32], target: u32, cfg: &TinyConfig, fd_eps: f32, tol: f64) -> bool {
    let weights = make_tiny_weights(cfg);
    let lora_scale = cfg.lora_alpha / cfg.lora_rank as f32;
    let seq_len = seq.len();
    let (cos_t, sin_t) = build_rope_tables(cfg, seq_len);

    // Analytic backward:
    // 1. Forward through frozen prefix (layers 0..num_layers-2)
    // 2. Forward through last layer with cache
    // 3. CE backward → lm_head VJP → final_norm backward → attn+ffn backward

    // --- Step 1: frozen prefix forward ---
    let mut hidden_seq: Vec<f32> = seq
        .iter()
        .flat_map(|&tid| {
            let start = tid as usize * cfg.hidden;
            weights.embed[start..start + cfg.hidden].iter().copied()
        })
        .collect();

    let last_layer = cfg.num_layers - 1;
    for l in 0..last_layer {
        let lw = &weights.layers[l];
        let mut new_hidden = hidden_seq.clone();
        let mut normed_pre_attn = vec![0.0f32; seq_len * cfg.hidden];
        for t in 0..seq_len {
            let x_t = &hidden_seq[t * cfg.hidden..(t + 1) * cfg.hidden];
            let (n, _) = rms_norm_forward(x_t, &lw.pre_attn_norm, cfg.eps);
            normed_pre_attn[t * cfg.hidden..(t + 1) * cfg.hidden].copy_from_slice(&n);
        }
        let (attn_out_seq, _) = gqa_forward_with_cache(
            &normed_pre_attn,
            &lw.w_q,
            &lw.w_k,
            &lw.w_v,
            &lw.w_o,
            &lw.q_norm,
            &lw.k_norm,
            None,
            None,
            None,
            None,
            cfg.lora_rank,
            lora_scale,
            seq_len,
            cfg.hidden,
            cfg.num_q_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.rope_dim,
            &cos_t,
            &sin_t,
            cfg.eps,
        );
        for t in 0..seq_len {
            for j in 0..cfg.hidden {
                new_hidden[t * cfg.hidden + j] =
                    hidden_seq[t * cfg.hidden + j] + attn_out_seq[t * cfg.hidden + j];
            }
        }
        for t in 0..seq_len {
            let x_t = new_hidden[t * cfg.hidden..(t + 1) * cfg.hidden].to_vec();
            let (normed, _) = rms_norm_forward(&x_t, &lw.post_attn_norm, cfg.eps);
            let (ffn_out, _, _) = swiglu_forward(
                &normed, &lw.w_gate, &lw.w_up, &lw.w_down, cfg.hidden, cfg.inter,
            );
            for j in 0..cfg.hidden {
                new_hidden[t * cfg.hidden + j] = x_t[j] + ffn_out[j];
            }
        }
        hidden_seq = new_hidden;
    }

    let prefix_out = hidden_seq.clone();

    // --- Step 2: last layer forward with cache ---
    let lw = &weights.layers[last_layer];

    let mut normed_pre_attn = vec![0.0f32; seq_len * cfg.hidden];
    let mut inv_rms_pre_attn = vec![0.0f32; seq_len];
    for t in 0..seq_len {
        let x_t = &prefix_out[t * cfg.hidden..(t + 1) * cfg.hidden];
        let (n, inv) = rms_norm_forward(x_t, &lw.pre_attn_norm, cfg.eps);
        normed_pre_attn[t * cfg.hidden..(t + 1) * cfg.hidden].copy_from_slice(&n);
        inv_rms_pre_attn[t] = inv;
    }

    let (attn_out_seq, attn_cache) = gqa_forward_with_cache(
        &normed_pre_attn,
        &lw.w_q,
        &lw.w_k,
        &lw.w_v,
        &lw.w_o,
        &lw.q_norm,
        &lw.k_norm,
        Some(&weights.lora_a_q),
        Some(&weights.lora_b_q),
        Some(&weights.lora_a_v),
        Some(&weights.lora_b_v),
        cfg.lora_rank,
        lora_scale,
        seq_len,
        cfg.hidden,
        cfg.num_q_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.rope_dim,
        &cos_t,
        &sin_t,
        cfg.eps,
    );

    // After attn residual
    let mut post_attn = prefix_out.clone();
    for t in 0..seq_len {
        for j in 0..cfg.hidden {
            post_attn[t * cfg.hidden + j] += attn_out_seq[t * cfg.hidden + j];
        }
    }

    // Post-attn norm + SwiGLU
    let mut normed_pre_ffn = vec![0.0f32; seq_len * cfg.hidden];
    let mut inv_rms_pre_ffn = vec![0.0f32; seq_len];
    let mut gate_pres: Vec<Vec<f32>> = Vec::new();
    let mut up_pres: Vec<Vec<f32>> = Vec::new();
    let mut ffn_outs: Vec<Vec<f32>> = Vec::new();
    for t in 0..seq_len {
        let x_t = &post_attn[t * cfg.hidden..(t + 1) * cfg.hidden];
        let (n, inv) = rms_norm_forward(x_t, &lw.post_attn_norm, cfg.eps);
        normed_pre_ffn[t * cfg.hidden..(t + 1) * cfg.hidden].copy_from_slice(&n);
        inv_rms_pre_ffn[t] = inv;
        let (ffn_out, gp, up) =
            swiglu_forward(&n, &lw.w_gate, &lw.w_up, &lw.w_down, cfg.hidden, cfg.inter);
        gate_pres.push(gp);
        up_pres.push(up);
        ffn_outs.push(ffn_out);
    }

    let mut post_ffn = post_attn.clone();
    for t in 0..seq_len {
        for j in 0..cfg.hidden {
            post_ffn[t * cfg.hidden + j] += ffn_outs[t][j];
        }
    }

    // Final norm + lm_head (last token only)
    let last_t = seq_len - 1;
    let x_last = post_ffn[last_t * cfg.hidden..(last_t + 1) * cfg.hidden].to_vec();
    let (final_normed, inv_rms_final) = rms_norm_forward(&x_last, &weights.final_norm, cfg.eps);

    let mut logits = vec![0.0f32; cfg.vocab];
    for i in 0..cfg.vocab {
        logits[i] = weights.lm_head[i * cfg.hidden..(i + 1) * cfg.hidden]
            .iter()
            .zip(final_normed.iter())
            .map(|(a, b)| a * b)
            .sum();
    }

    // --- Step 3: Backward ---
    // CE backward → dlogits [vocab]
    let seq_logits: Vec<f32> = logits.clone(); // only last token for loss
    let n_comp = 1.0f32;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let mut d_logits = vec![0.0f32; cfg.vocab];
    for v in 0..cfg.vocab {
        let p = (logits[v] - max).exp() / sum_exp;
        let indicator = if v == target as usize { 1.0f32 } else { 0.0 };
        d_logits[v] = (p - indicator) / n_comp;
    }
    let _ = seq_logits;

    // lm_head VJP: d_final_normed [hidden] = lm_head^T d_logits
    let d_final_normed = linear_vjp(&weights.lm_head, &d_logits, cfg.hidden, cfg.vocab);

    // final_norm backward: d_x_last [hidden]
    let d_x_last = rmsnorm_backward(&x_last, &weights.final_norm, inv_rms_final, &d_final_normed);

    // Gradient flows into post_ffn[last_t], then backward through FFN (last token)
    let mut d_post_ffn = vec![0.0f32; seq_len * cfg.hidden];
    d_post_ffn[last_t * cfg.hidden..(last_t + 1) * cfg.hidden].copy_from_slice(&d_x_last);

    // FFN backward per token
    let mut d_post_attn = d_post_ffn.clone(); // residual: d_post_attn += ffn backward dx
    for t in 0..seq_len {
        let d_ffn_out = &d_post_ffn[t * cfg.hidden..(t + 1) * cfg.hidden];
        if d_ffn_out.iter().all(|&x| x == 0.0) {
            continue;
        }

        let (d_normed_ffn, _) = swiglu_backward(
            d_ffn_out,
            &gate_pres[t],
            &up_pres[t],
            &lw.w_down,
            &lw.w_gate,
            &lw.w_up,
            cfg.hidden,
            cfg.inter,
        );

        // post_attn_norm backward
        let x_t = &post_attn[t * cfg.hidden..(t + 1) * cfg.hidden];
        let d_x_t = rmsnorm_backward(x_t, &lw.post_attn_norm, inv_rms_pre_ffn[t], &d_normed_ffn);

        for j in 0..cfg.hidden {
            d_post_attn[t * cfg.hidden + j] += d_x_t[j]; // += because residual passes through
        }
    }

    // Attention backward: d_post_attn = d_residual_after_attn
    // Gradient to attn_out = same as d_post_attn; gradient to layer input (residual path) = same
    let d_attn_out = d_post_attn.clone(); // same grad flows to both branches (residual)
    let mut d_prefix_out = d_post_attn.clone(); // residual branch

    // pre_attn_norm backward + attn backward
    let attn_grads = gqa_backward(
        &d_attn_out,
        &attn_cache,
        &lw.w_q,
        &lw.w_k,
        &lw.w_v,
        &lw.w_o,
        &lw.q_norm,
        &lw.k_norm,
        Some(&weights.lora_a_q),
        Some(&weights.lora_b_q),
        Some(&weights.lora_a_v),
        Some(&weights.lora_b_v),
        cfg.lora_rank,
        lora_scale,
    );

    // pre_attn_norm backward (from attn dx) → prefix_out
    for t in 0..seq_len {
        let d_attn_dx_t = &attn_grads.dx[t * cfg.hidden..(t + 1) * cfg.hidden];
        let x_t = &prefix_out[t * cfg.hidden..(t + 1) * cfg.hidden];
        let d_x_t = rmsnorm_backward(x_t, &lw.pre_attn_norm, inv_rms_pre_attn[t], d_attn_dx_t);
        for j in 0..cfg.hidden {
            d_prefix_out[t * cfg.hidden + j] += d_x_t[j];
        }
    }

    // Analytic grads
    let analytic_a_q = attn_grads.grad_a_q;
    let analytic_a_v = attn_grads.grad_a_v;
    let analytic_b_q = attn_grads.grad_b_q;
    let analytic_b_v = attn_grads.grad_b_v;

    // --- FD check ---
    let q_dim = cfg.num_q_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    let fd_check = |n: usize, perturb: &dyn Fn(&mut TinyWeights, usize, f32)| -> Vec<f32> {
        let mut fd = vec![0.0f32; n];
        for k in 0..n {
            let mut wp = make_tiny_weights(cfg);
            // Copy relevant LoRA from canonical weights
            wp.lora_a_q.copy_from_slice(&weights.lora_a_q);
            wp.lora_b_q.copy_from_slice(&weights.lora_b_q);
            wp.lora_a_v.copy_from_slice(&weights.lora_a_v);
            wp.lora_b_v.copy_from_slice(&weights.lora_b_v);
            // Copy all layer weights too
            for ll in 0..cfg.num_layers {
                wp.layers[ll].w_q.copy_from_slice(&weights.layers[ll].w_q);
                wp.layers[ll].w_k.copy_from_slice(&weights.layers[ll].w_k);
                wp.layers[ll].w_v.copy_from_slice(&weights.layers[ll].w_v);
                wp.layers[ll].w_o.copy_from_slice(&weights.layers[ll].w_o);
                wp.layers[ll]
                    .w_gate
                    .copy_from_slice(&weights.layers[ll].w_gate);
                wp.layers[ll].w_up.copy_from_slice(&weights.layers[ll].w_up);
                wp.layers[ll]
                    .w_down
                    .copy_from_slice(&weights.layers[ll].w_down);
            }
            wp.embed.copy_from_slice(&weights.embed);
            wp.lm_head.copy_from_slice(&weights.lm_head);

            let mut wm = wp.clone(); // duplicate

            perturb(&mut wp, k, fd_eps);
            perturb(&mut wm, k, -fd_eps);

            let lp = tiny_loss(seq, target, &wp, cfg);
            let lm = tiny_loss(seq, target, &wm, cfg);
            fd[k] = (lp - lm) / (2.0 * fd_eps);
        }
        fd
    };

    let fd_a_q = fd_check(
        cfg.lora_rank * cfg.hidden,
        &|w: &mut TinyWeights, k: usize, delta: f32| {
            w.lora_a_q[k] += delta;
        },
    );
    let fd_a_v = fd_check(
        cfg.lora_rank * cfg.hidden,
        &|w: &mut TinyWeights, k: usize, delta: f32| {
            w.lora_a_v[k] += delta;
        },
    );
    let fd_b_q = fd_check(
        2 * q_dim * cfg.lora_rank,
        &|w: &mut TinyWeights, k: usize, delta: f32| {
            w.lora_b_q[k] += delta;
        },
    );
    let fd_b_v = fd_check(
        kv_dim * cfg.lora_rank,
        &|w: &mut TinyWeights, k: usize, delta: f32| {
            w.lora_b_v[k] += delta;
        },
    );

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

    let err_aq = rel_err(&analytic_a_q, &fd_a_q);
    let err_av = rel_err(&analytic_a_v, &fd_a_v);
    let err_bq = rel_err(&analytic_b_q, &fd_b_q);
    let err_bv = rel_err(&analytic_b_v, &fd_b_v);
    eprintln!(
        "end2end gradcheck: grad_A_q={err_aq:.2e} grad_A_v={err_av:.2e} grad_B_q={err_bq:.2e} grad_B_v={err_bv:.2e} tol={tol:.1e}"
    );

    err_aq < tol && err_av < tol && err_bq < tol && err_bv < tol
}

impl TinyWeights {
    fn clone(&self) -> Self {
        TinyWeights {
            embed: self.embed.clone(),
            layers: self
                .layers
                .iter()
                .map(|l| LayerWeights {
                    w_q: l.w_q.clone(),
                    w_k: l.w_k.clone(),
                    w_v: l.w_v.clone(),
                    w_o: l.w_o.clone(),
                    q_norm: l.q_norm.clone(),
                    k_norm: l.k_norm.clone(),
                    pre_attn_norm: l.pre_attn_norm.clone(),
                    post_attn_norm: l.post_attn_norm.clone(),
                    w_gate: l.w_gate.clone(),
                    w_up: l.w_up.clone(),
                    w_down: l.w_down.clone(),
                })
                .collect(),
            final_norm: self.final_norm.clone(),
            lm_head: self.lm_head.clone(),
            lora_a_q: self.lora_a_q.clone(),
            lora_b_q: self.lora_b_q.clone(),
            lora_a_v: self.lora_a_v.clone(),
            lora_b_v: self.lora_b_v.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_to_end_lora_gradcheck() {
        let cfg = TinyConfig::default();
        let seq = vec![1u32, 42, 7, 0];
        let target = 100u32;
        let passed = gradcheck_lora(&seq, target, &cfg, 1e-3, 5e-2);
        assert!(passed, "end-to-end LoRA gradcheck failed");
    }

    #[test]
    fn tiny_loss_finite() {
        let cfg = TinyConfig::default();
        let w = make_tiny_weights(&cfg);
        let seq = vec![0u32, 1, 2];
        let loss = tiny_loss(&seq, 3, &w, &cfg);
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        assert!(loss > 0.0, "NLL must be positive");
    }
}
