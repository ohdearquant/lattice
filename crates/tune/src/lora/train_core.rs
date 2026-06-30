#![allow(missing_docs)]

use lattice_inference::attention::gdn::GatedDeltaNetWeights;
use lattice_inference::attention::gdn_backward::{GdnSaved, gdn_backward, gdn_forward_save};
use lattice_inference::backward::attention_gqa::{AttnCache, gqa_backward, gqa_forward_with_cache};
use lattice_inference::backward::ops::{linear_vjp, rmsnorm_backward, swiglu_backward};
use lattice_inference::backward::tape::{rms_norm_forward, swiglu_forward};
use lattice_inference::model::qwen35_config::Qwen35Config;

use crate::error::{Result, TuneError};
use crate::lora::optimizer::AdamState;

pub const TOP_LAYER: usize = 23;

#[derive(Clone, Copy)]
pub struct Dims {
    pub hidden: usize,
    pub vocab: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub inter: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub eps: f32,
}

#[derive(Clone, Copy)]
pub struct GdnDims {
    num_kh: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    qkv_dim: usize,
    output_dim: usize,
    kernel_size: usize,
    scale: f32,
}

impl GdnDims {
    pub fn from_cfg(cfg: &Qwen35Config) -> Self {
        let key_dim = cfg.linear_key_head_dim;
        Self {
            num_kh: cfg.linear_num_key_heads,
            value_heads: cfg.linear_num_value_heads(),
            key_dim,
            value_dim: cfg.linear_value_head_dim,
            qkv_dim: cfg.linear_qkv_dim(),
            output_dim: cfg.linear_output_dim(),
            kernel_size: cfg.linear_conv_kernel_dim,
            scale: 1.0 / (key_dim as f32).sqrt(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MixerKind {
    Gqa,
    Gdn,
}

pub struct LayerW<'a> {
    pub kind: MixerKind,
    pub w_q: &'a [f32],
    pub w_k: &'a [f32],
    pub w_v: &'a [f32],
    pub w_o: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub gdn: Option<&'a GatedDeltaNetWeights>,
    pub pre_shift: Vec<f32>,
    pub post_shift: Vec<f32>,
    pub w_gate: &'a [f32],
    pub w_up: &'a [f32],
    pub w_down: &'a [f32],
    pub lora_slot: Option<usize>,
}

pub struct Head<'a> {
    pub lm_head: &'a [f32],
    pub final_shift: &'a [f32],
}

#[derive(Clone)]
pub struct LoraParams {
    pub a_q: Vec<f32>,
    pub b_q: Vec<f32>,
    pub a_v: Vec<f32>,
    pub b_v: Vec<f32>,
}

impl LoraParams {
    pub fn zeros(rank: usize, d: &Dims) -> Result<Self> {
        let a_q_len = rank.checked_mul(d.hidden).ok_or_else(|| {
            TuneError::Validation(format!(
                "rank({rank}) * hidden({}) overflows gradient buffer size",
                d.hidden
            ))
        })?;
        let b_q_len = (2usize)
            .checked_mul(d.q_dim)
            .and_then(|n| n.checked_mul(rank))
            .ok_or_else(|| {
                TuneError::Validation(format!(
                    "2 * q_dim({}) * rank({rank}) overflows gradient buffer size",
                    d.q_dim
                ))
            })?;
        let a_v_len = rank.checked_mul(d.hidden).ok_or_else(|| {
            TuneError::Validation(format!(
                "rank({rank}) * hidden({}) overflows gradient buffer size",
                d.hidden
            ))
        })?;
        let b_v_len = d.kv_dim.checked_mul(rank).ok_or_else(|| {
            TuneError::Validation(format!(
                "kv_dim({}) * rank({rank}) overflows gradient buffer size",
                d.kv_dim
            ))
        })?;
        Ok(Self {
            a_q: vec![0.0; a_q_len],
            b_q: vec![0.0; b_q_len],
            a_v: vec![0.0; a_v_len],
            b_v: vec![0.0; b_v_len],
        })
    }
}

pub type Grads = LoraParams;

pub struct SeqCtx {
    pub h_in: Vec<f32>,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub tokens: Vec<u32>,
    pub completion_start: usize,
    pub seq_len: usize,
}

enum MixerCache {
    Gqa(Box<AttnCache>),
    Gdn(Box<GdnSaved>),
}

struct LayerFwd {
    h_layer_in: Vec<f32>,
    inv_pre: Vec<f32>,
    mixer: MixerCache,
    h_mid: Vec<f32>,
    inv_ffn: Vec<f32>,
    gate_pre: Vec<Vec<f32>>,
    up_pre: Vec<Vec<f32>>,
}

struct HeadPos {
    t: usize,
    target: u32,
    logits: Vec<f32>,
    inv_final: f32,
}

pub struct FullFwd {
    layers: Vec<LayerFwd>,
    h_final: Vec<f32>,
    positions: Vec<HeadPos>,
}

fn lm_head_logits(lm_head: &[f32], final_normed: &[f32], hidden: usize, vocab: usize) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocab];
    for (i, l) in logits.iter_mut().enumerate() {
        *l = lm_head[i * hidden..(i + 1) * hidden]
            .iter()
            .zip(final_normed.iter())
            .map(|(w, x)| w * x)
            .sum();
    }
    logits
}

fn rmsnorm_seq(
    h: &[f32],
    shift: &[f32],
    hidden: usize,
    seq: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut normed = vec![0.0f32; seq * hidden];
    let mut inv = vec![0.0f32; seq];
    for t in 0..seq {
        let (n, iv) = rms_norm_forward(&h[t * hidden..(t + 1) * hidden], shift, eps);
        normed[t * hidden..(t + 1) * hidden].copy_from_slice(&n);
        inv[t] = iv;
    }
    (normed, inv)
}

fn position_nll(logits: &[f32], target: u32) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    let log_prob = (logits[target as usize] - max) as f64 - sum_exp.ln();
    (-log_prob) as f32
}

#[allow(clippy::too_many_arguments)]
pub fn forward_full(
    ctx: &SeqCtx,
    layers: &[LayerW<'_>],
    loras: &[LoraParams],
    head: &Head<'_>,
    d: &Dims,
    gdn_dims: &GdnDims,
    cfg: &Qwen35Config,
    rank: usize,
    scale: f32,
) -> Result<FullFwd> {
    let hidden = d.hidden;
    let seq = ctx.seq_len;
    let mut h = ctx.h_in.clone();
    let mut layers_fwd: Vec<LayerFwd> = Vec::with_capacity(layers.len());

    for lw in layers {
        let h_layer_in = h.clone();
        let (normed_pre, inv_pre) = rmsnorm_seq(&h, &lw.pre_shift, hidden, seq, d.eps);

        let (mixer_out, mixer_cache) = match lw.kind {
            MixerKind::Gqa => {
                let slot = lw.lora_slot.ok_or_else(|| {
                    TuneError::Validation("GQA layer is missing a LoRA slot".to_string())
                })?;
                let lora = &loras[slot];
                let (out, cache) = gqa_forward_with_cache(
                    &normed_pre,
                    lw.w_q,
                    lw.w_k,
                    lw.w_v,
                    lw.w_o,
                    lw.q_norm,
                    lw.k_norm,
                    Some(&lora.a_q),
                    Some(&lora.b_q),
                    Some(&lora.a_v),
                    Some(&lora.b_v),
                    rank,
                    scale,
                    seq,
                    hidden,
                    d.num_q_heads,
                    d.num_kv_heads,
                    d.head_dim,
                    d.rope_dim,
                    &ctx.cos,
                    &ctx.sin,
                    d.eps,
                );
                (out, MixerCache::Gqa(Box::new(cache)))
            }
            MixerKind::Gdn => {
                let gdn_w = lw.gdn.ok_or_else(|| {
                    TuneError::Validation("GDN layer is missing weights".to_string())
                })?;
                let mut saved = GdnSaved::new(
                    seq,
                    gdn_dims.num_kh,
                    gdn_dims.value_heads,
                    gdn_dims.key_dim,
                    gdn_dims.value_dim,
                    hidden,
                    gdn_dims.qkv_dim,
                    gdn_dims.output_dim,
                    gdn_dims.kernel_size,
                    gdn_dims.scale,
                    d.eps,
                );
                let mut out = vec![0.0f32; seq * hidden];
                gdn_forward_save(&normed_pre, gdn_w, cfg, &mut saved, &mut out);
                (out, MixerCache::Gdn(Box::new(saved)))
            }
        };

        let mut h_mid = h_layer_in.clone();
        for (a, b) in h_mid.iter_mut().zip(mixer_out.iter()) {
            *a += *b;
        }

        let (normed_ffn, inv_ffn) = rmsnorm_seq(&h_mid, &lw.post_shift, hidden, seq, d.eps);
        let mut gate_pre = Vec::with_capacity(seq);
        let mut up_pre = Vec::with_capacity(seq);
        let mut h_next = h_mid.clone();
        for t in 0..seq {
            let (ffn_out, gp, up) = swiglu_forward(
                &normed_ffn[t * hidden..(t + 1) * hidden],
                lw.w_gate,
                lw.w_up,
                lw.w_down,
                hidden,
                d.inter,
            );
            for (o, f) in h_next[t * hidden..(t + 1) * hidden]
                .iter_mut()
                .zip(ffn_out.iter())
            {
                *o += *f;
            }
            gate_pre.push(gp);
            up_pre.push(up);
        }

        layers_fwd.push(LayerFwd {
            h_layer_in,
            inv_pre,
            mixer: mixer_cache,
            h_mid,
            inv_ffn,
            gate_pre,
            up_pre,
        });
        h = h_next;
    }

    let mut positions = Vec::new();
    for t in (ctx.completion_start - 1)..ctx.seq_len - 1 {
        let (final_normed, inv_final) =
            rms_norm_forward(&h[t * hidden..(t + 1) * hidden], head.final_shift, d.eps);
        let logits = lm_head_logits(head.lm_head, &final_normed, hidden, d.vocab);
        positions.push(HeadPos {
            t,
            target: ctx.tokens[t + 1],
            logits,
            inv_final,
        });
    }

    Ok(FullFwd {
        layers: layers_fwd,
        h_final: h,
        positions,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn eval_chain_nll(
    caches: &[SeqCtx],
    layers: &[LayerW<'_>],
    loras: &[LoraParams],
    head: &Head<'_>,
    d: &Dims,
    gdn_dims: &GdnDims,
    cfg: &Qwen35Config,
    rank: usize,
    scale: f32,
) -> Result<f32> {
    let mut nll_sum = 0.0f64;
    let mut n = 0usize;
    for ctx in caches {
        let fwd = forward_full(ctx, layers, loras, head, d, gdn_dims, cfg, rank, scale)?;
        for p in &fwd.positions {
            nll_sum += position_nll(&p.logits, p.target) as f64;
            n += 1;
        }
    }
    Ok((nll_sum / n.max(1) as f64) as f32)
}

#[allow(clippy::too_many_arguments)]
pub fn nll_and_grads(
    fwd: &FullFwd,
    layers: &[LayerW<'_>],
    loras: &[LoraParams],
    head: &Head<'_>,
    d: &Dims,
    num_slots: usize,
    rank: usize,
    scale: f32,
) -> Result<(f32, usize, Vec<Grads>)> {
    let hidden = d.hidden;
    let seq = fwd.h_final.len() / hidden;
    let n_comp = fwd.positions.len().max(1) as f32;
    let mut grads: Vec<Grads> = (0..num_slots)
        .map(|_| LoraParams::zeros(rank, d))
        .collect::<Result<Vec<_>>>()?;

    let mut d_h = vec![0.0f32; seq * hidden];
    let mut nll_sum = 0.0f64;
    for p in &fwd.positions {
        nll_sum += position_nll(&p.logits, p.target) as f64;
        let max = p.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = p.logits.iter().map(|&l| (l - max).exp()).sum();
        let mut d_logits = vec![0.0f32; d.vocab];
        for (v, dl) in d_logits.iter_mut().enumerate() {
            let prob = (p.logits[v] - max).exp() / sum_exp;
            let indicator = if v == p.target as usize { 1.0 } else { 0.0 };
            *dl = (prob - indicator) / n_comp;
        }
        let d_final = linear_vjp(head.lm_head, &d_logits, hidden, d.vocab);
        let d_h_t = rmsnorm_backward(
            &fwd.h_final[p.t * hidden..(p.t + 1) * hidden],
            head.final_shift,
            p.inv_final,
            &d_final,
        );
        for (j, dv) in d_h_t.iter().enumerate() {
            d_h[p.t * hidden + j] += *dv;
        }
    }

    for li in (0..layers.len()).rev() {
        let lw = &layers[li];
        let lf = &fwd.layers[li];

        let mut d_h_mid = d_h.clone();
        for t in 0..seq {
            let d_ffn_out = &d_h[t * hidden..(t + 1) * hidden];
            let (d_normed_ffn, _dm) = swiglu_backward(
                d_ffn_out,
                &lf.gate_pre[t],
                &lf.up_pre[t],
                lw.w_down,
                lw.w_gate,
                lw.w_up,
                hidden,
                d.inter,
            );
            let d_hm = rmsnorm_backward(
                &lf.h_mid[t * hidden..(t + 1) * hidden],
                &lw.post_shift,
                lf.inv_ffn[t],
                &d_normed_ffn,
            );
            for (j, dv) in d_hm.iter().enumerate() {
                d_h_mid[t * hidden + j] += *dv;
            }
        }

        let d_normed_pre: Vec<f32> = match &lf.mixer {
            MixerCache::Gqa(cache) => {
                let slot = lw.lora_slot.ok_or_else(|| {
                    TuneError::Validation("GQA layer missing LoRA slot in backward".to_string())
                })?;
                let lora = &loras[slot];
                let g = gqa_backward(
                    &d_h_mid,
                    cache,
                    lw.w_q,
                    lw.w_k,
                    lw.w_v,
                    lw.w_o,
                    lw.q_norm,
                    lw.k_norm,
                    Some(&lora.a_q),
                    Some(&lora.b_q),
                    Some(&lora.a_v),
                    Some(&lora.b_v),
                    rank,
                    scale,
                );
                grads[slot] = Grads {
                    a_q: g.grad_a_q,
                    b_q: g.grad_b_q,
                    a_v: g.grad_a_v,
                    b_v: g.grad_b_v,
                };
                g.dx
            }
            MixerCache::Gdn(saved) => {
                let gdn_w = lw.gdn.ok_or_else(|| {
                    TuneError::Validation("GDN layer missing weights in backward".to_string())
                })?;
                let mut dx = vec![0.0f32; seq * hidden];
                gdn_backward(&d_h_mid, saved, gdn_w, &mut dx);
                dx
            }
        };

        let mut d_h_new = d_h_mid;
        for t in 0..seq {
            let d_hl = rmsnorm_backward(
                &lf.h_layer_in[t * hidden..(t + 1) * hidden],
                &lw.pre_shift,
                lf.inv_pre[t],
                &d_normed_pre[t * hidden..(t + 1) * hidden],
            );
            for (j, dv) in d_hl.iter().enumerate() {
                d_h_new[t * hidden + j] += *dv;
            }
        }
        d_h = d_h_new;
    }

    Ok((nll_sum as f32, fwd.positions.len(), grads))
}

pub fn shifted(gamma: &[f32]) -> Vec<f32> {
    gamma.iter().map(|g| 1.0 + g).collect()
}

pub fn rand_fill(rng: &mut u64, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            ((*rng >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn apply_adam_updates(
    adam: &mut AdamState,
    loras: &mut [LoraParams],
    grads: &[Grads],
    slot_layers: &[usize],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps_adam: f32,
) {
    for slot in 0..slot_layers.len() {
        let li = slot_layers[slot];
        adam.step(
            &format!("l{li}_a_q"),
            &mut loras[slot].a_q,
            &grads[slot].a_q,
            lr,
            beta1,
            beta2,
            eps_adam,
            0.0,
            false,
        );
        adam.step(
            &format!("l{li}_b_q"),
            &mut loras[slot].b_q,
            &grads[slot].b_q,
            lr,
            beta1,
            beta2,
            eps_adam,
            0.0,
            false,
        );
        adam.step(
            &format!("l{li}_a_v"),
            &mut loras[slot].a_v,
            &grads[slot].a_v,
            lr,
            beta1,
            beta2,
            eps_adam,
            0.0,
            false,
        );
        adam.step(
            &format!("l{li}_b_v"),
            &mut loras[slot].b_v,
            &grads[slot].b_v,
            lr,
            beta1,
            beta2,
            eps_adam,
            0.0,
            false,
        );
    }
}
