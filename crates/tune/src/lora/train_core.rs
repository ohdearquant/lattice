#![allow(missing_docs)]

//! Private backward tape for micro-LoRA training.
//!
//! The tape materializes a Qwen3.5 suffix, preserves mixer and FFN activations,
//! and uses a validated context to bind geometry, slots, rank/scale, and Adam.
//!
//! See `docs/lora-core.md` for the forward/backward walk and shape invariants.

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
    pub num_kh: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub qkv_dim: usize,
    pub output_dim: usize,
    pub kernel_size: usize,
    pub scale: f32,
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

/// Parameters or gradients for the five GDN LoRA projections.
/// Its fields mirror inference's `GdnGrads` for direct gradient transfer.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#gdnloraparams) for all shapes and the value-head invariant.
#[derive(Clone)]
pub struct GdnLoraParams {
    pub a_qkv: Vec<f32>,
    pub b_qkv: Vec<f32>,
    pub a_z: Vec<f32>,
    pub b_z: Vec<f32>,
    pub a_b: Vec<f32>,
    pub b_b: Vec<f32>,
    pub a_a: Vec<f32>,
    pub b_a: Vec<f32>,
    pub a_out: Vec<f32>,
    pub b_out: Vec<f32>,
}

impl GdnLoraParams {
    /// Build GDN LoRA arrays from checked shapes and caller-provided fillers.
    /// `zeros` delegates here so every initializer shares the same dimensions.
    /// See [`docs/lora-core.md`](../../docs/lora-core.md#gdnloraparams) for the drift-prevention rationale.
    pub fn shaped(
        rank: usize,
        hidden: usize,
        gd: &GdnDims,
        mut fill_a: impl FnMut(usize) -> Vec<f32>,
        mut fill_b: impl FnMut(usize) -> Vec<f32>,
    ) -> Result<Self> {
        let checked = |a: usize, b: usize, label: &str| -> Result<usize> {
            a.checked_mul(b).ok_or_else(|| {
                TuneError::Validation(format!(
                    "{label} overflows GDN LoRA gradient buffer size ({a} * {b})"
                ))
            })
        };
        Ok(Self {
            a_qkv: fill_a(checked(rank, hidden, "rank*hidden (a_qkv)")?),
            b_qkv: fill_b(checked(gd.qkv_dim, rank, "qkv_dim*rank (b_qkv)")?),
            a_z: fill_a(checked(rank, hidden, "rank*hidden (a_z)")?),
            b_z: fill_b(checked(gd.output_dim, rank, "output_dim*rank (b_z)")?),
            a_b: fill_a(checked(rank, hidden, "rank*hidden (a_b)")?),
            // Beta is projected per value head.
            b_b: fill_b(checked(gd.value_heads, rank, "value_heads*rank (b_b)")?),
            a_a: fill_a(checked(rank, hidden, "rank*hidden (a_a)")?),
            // Alpha is projected per value head.
            b_a: fill_b(checked(gd.value_heads, rank, "value_heads*rank (b_a)")?),
            a_out: fill_a(checked(rank, gd.output_dim, "rank*output_dim (a_out)")?),
            b_out: fill_b(checked(hidden, rank, "hidden*rank (b_out)")?),
        })
    }

    pub fn zeros(rank: usize, hidden: usize, gd: &GdnDims) -> Result<Self> {
        Self::shaped(rank, hidden, gd, |n| vec![0.0; n], |n| vec![0.0; n])
    }
}

/// Borrowed dimensions and model configuration used by tape entry points.
pub struct TapeGeometry<'a> {
    dims: &'a Dims,
    gdn_dims: &'a GdnDims,
    model: &'a Qwen35Config,
}

impl<'a> TapeGeometry<'a> {
    /// Bundle geometry references; [`TrainCtx::try_new`] validates consistency.
    pub fn new(dims: &'a Dims, gdn_dims: &'a GdnDims, model: &'a Qwen35Config) -> Self {
        Self {
            dims,
            gdn_dims,
            model,
        }
    }
}

/// Runtime LoRA rank and scale derived by [`TrainCtx::try_new`].
struct LoraExecution {
    rank: usize,
    scale: f32,
}

/// Mapping from GQA and GDN tape slots to global model-layer indices.
pub struct SlotLayout<'a> {
    gqa_layers: &'a [usize],
    gdn_layers: &'a [usize],
}

impl<'a> SlotLayout<'a> {
    /// Bundle slot-layer slices; [`TrainCtx::try_new`] validates them.
    pub fn new(gqa_layers: &'a [usize], gdn_layers: &'a [usize]) -> Self {
        Self {
            gqa_layers,
            gdn_layers,
        }
    }
}

/// Adam hyperparameters shared by every optimizer step in a training run.
pub struct AdamConfig {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl AdamConfig {
    /// Bundle Adam hyperparameters. Finiteness is checked by
    /// [`TrainCtx::try_new`], not here.
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
        }
    }
}

/// Validated immutable geometry, slot, LoRA, and Adam inputs for the tape.
pub struct TrainCtx<'a> {
    geometry: TapeGeometry<'a>,
    lora: LoraExecution,
    slots: SlotLayout<'a>,
    adam: AdamConfig,
}

impl<'a> TrainCtx<'a> {
    /// Construct a context after validating tape geometry, slots, and Adam values.
    /// Derives and stores execution scale from `alpha / rank`.
    /// See [`docs/lora-core.md`](../../docs/lora-core.md#trainctxtry_new) for every validation invariant.
    pub fn try_new(
        geometry: TapeGeometry<'a>,
        rank: usize,
        alpha: f32,
        slots: SlotLayout<'a>,
        adam: AdamConfig,
    ) -> Result<Self> {
        if rank == 0 {
            return Err(TuneError::Validation(
                "TrainCtx::try_new: rank must be > 0".to_string(),
            ));
        }
        if !alpha.is_finite() {
            return Err(TuneError::Validation(format!(
                "TrainCtx::try_new: alpha must be finite, got {alpha}"
            )));
        }
        for (name, v) in [
            ("learning_rate", adam.learning_rate),
            ("beta1", adam.beta1),
            ("beta2", adam.beta2),
            ("epsilon", adam.epsilon),
        ] {
            if !v.is_finite() {
                return Err(TuneError::Validation(format!(
                    "TrainCtx::try_new: adam.{name} must be finite, got {v}"
                )));
            }
        }

        let d = geometry.dims;
        let m = geometry.model;
        if !d.eps.is_finite() {
            return Err(TuneError::Validation(format!(
                "TrainCtx::try_new: Dims.eps must be finite, got {}",
                d.eps
            )));
        }
        if d.hidden != m.hidden_size
            || d.vocab != m.vocab_size
            || d.num_q_heads != m.num_attention_heads
            || d.num_kv_heads != m.num_key_value_heads
            || d.head_dim != m.head_dim
            || d.rope_dim != m.rope_dim()
            || d.inter != m.intermediate_size
            || d.q_dim != m.full_q_dim()
            || d.kv_dim != m.full_kv_dim()
            || (d.eps - m.rms_norm_eps).abs() > 1e-9
        {
            return Err(TuneError::Validation(
                "TrainCtx::try_new: Dims does not match TapeGeometry.model".to_string(),
            ));
        }
        let expected_gdn = GdnDims::from_cfg(m);
        let gd = geometry.gdn_dims;
        if !gd.scale.is_finite() {
            return Err(TuneError::Validation(format!(
                "TrainCtx::try_new: GdnDims.scale must be finite, got {}",
                gd.scale
            )));
        }
        if gd.num_kh != expected_gdn.num_kh
            || gd.value_heads != expected_gdn.value_heads
            || gd.key_dim != expected_gdn.key_dim
            || gd.value_dim != expected_gdn.value_dim
            || gd.qkv_dim != expected_gdn.qkv_dim
            || gd.output_dim != expected_gdn.output_dim
            || gd.kernel_size != expected_gdn.kernel_size
            || (gd.scale - expected_gdn.scale).abs() > 1e-9
        {
            return Err(TuneError::Validation(
                "TrainCtx::try_new: GdnDims does not match TapeGeometry.model".to_string(),
            ));
        }

        let num_hidden_layers = m.num_hidden_layers;
        let mut seen = std::collections::HashSet::new();
        for &li in slots.gqa_layers {
            if li >= num_hidden_layers {
                return Err(TuneError::Validation(format!(
                    "TrainCtx::try_new: gqa slot layer {li} out of range (model has {num_hidden_layers} layers)"
                )));
            }
            if !m.is_full_attention(li) {
                return Err(TuneError::Validation(format!(
                    "TrainCtx::try_new: gqa slot layer {li} is not a full-attention (GQA) layer \
                     in this model (mixer-kind/slot-index mismatch)"
                )));
            }
            if !seen.insert(li) {
                return Err(TuneError::Validation(format!(
                    "TrainCtx::try_new: duplicate gqa slot layer {li}"
                )));
            }
        }
        let mut seen_gdn = std::collections::HashSet::new();
        for &li in slots.gdn_layers {
            if li >= num_hidden_layers {
                return Err(TuneError::Validation(format!(
                    "TrainCtx::try_new: gdn slot layer {li} out of range (model has {num_hidden_layers} layers)"
                )));
            }
            if m.is_full_attention(li) {
                return Err(TuneError::Validation(format!(
                    "TrainCtx::try_new: gdn slot layer {li} is a full-attention (GQA) layer in \
                     this model, not GatedDeltaNet (mixer-kind/slot-index mismatch)"
                )));
            }
            if !seen_gdn.insert(li) {
                return Err(TuneError::Validation(format!(
                    "TrainCtx::try_new: duplicate gdn slot layer {li}"
                )));
            }
            if seen.contains(&li) {
                return Err(TuneError::Validation(format!(
                    "TrainCtx::try_new: layer {li} claimed by both gqa and gdn slot layouts \
                     (mixer-kind/slot-index mismatch)"
                )));
            }
        }

        let scale = alpha / rank as f32;
        Ok(Self {
            geometry,
            lora: LoraExecution { rank, scale },
            slots,
            adam,
        })
    }

    /// Number of GQA LoRA slots (== `slots.gqa_layers.len()`).
    pub fn num_gqa_slots(&self) -> usize {
        self.slots.gqa_layers.len()
    }

    /// Number of GDN LoRA slots (== `slots.gdn_layers.len()`).
    pub fn num_gdn_slots(&self) -> usize {
        self.slots.gdn_layers.len()
    }

    /// Global model layer index per GQA slot.
    fn gqa_layers(&self) -> &[usize] {
        self.slots.gqa_layers
    }

    /// Global model layer index per GDN slot.
    fn gdn_layers(&self) -> &[usize] {
        self.slots.gdn_layers
    }

    /// LoRA rank.
    fn rank(&self) -> usize {
        self.lora.rank
    }

    /// Execution-only LoRA scale (`alpha / rank`).
    fn scale(&self) -> f32 {
        self.lora.scale
    }

    /// Flat per-array dimensions.
    fn dims(&self) -> &Dims {
        self.geometry.dims
    }

    /// GDN-specific dimensions.
    fn gdn_dims(&self) -> &GdnDims {
        self.geometry.gdn_dims
    }

    /// The model config the geometry was derived from.
    fn model(&self) -> &Qwen35Config {
        self.geometry.model
    }

    /// Adam learning rate.
    fn learning_rate(&self) -> f32 {
        self.adam.learning_rate
    }

    /// Adam beta1.
    fn beta1(&self) -> f32 {
        self.adam.beta1
    }

    /// Adam beta2.
    fn beta2(&self) -> f32 {
        self.adam.beta2
    }

    /// Adam epsilon.
    fn epsilon(&self) -> f32 {
        self.adam.epsilon
    }
}

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

// lm_head is [vocab, hidden] row-major; the naive scalar form (one dot product
// per vocab row) pays vocab*hidden scalar FLOPs per completion token — by far
// the largest per-token cost in the tape (vocab reaches ~250k). Route through
// the same GEMM dispatch the forward path uses elsewhere (Accelerate/AMX on
// macOS, SIMD-tiled fallback otherwise): logits = final_normed @ lm_head^T,
// i.e. a matmul_bt with m=1 (#737 stage 1).
fn lm_head_logits(lm_head: &[f32], final_normed: &[f32], hidden: usize, vocab: usize) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocab];
    lattice_inference::forward::cpu::matmul_bt(
        final_normed,
        lm_head,
        &mut logits,
        1,
        hidden,
        vocab,
    );
    logits
}

#[cfg(test)]
mod lm_head_logits_tests {
    use super::*;

    // Pre-#737 form: one scalar dot product per vocab row. `matmul_bt`
    // reassociates the reduction (Accelerate/AMX on macOS, tiled/SIMD
    // elsewhere), so parity is checked to a tight relative tolerance
    // (matches PARITY_TOL in `lattice_inference::backward::ops`), not
    // bit-exact — see that module's tests for the same rationale.
    fn lm_head_logits_scalar(
        lm_head: &[f32],
        final_normed: &[f32],
        hidden: usize,
        vocab: usize,
    ) -> Vec<f32> {
        let mut logits = vec![0.0f32; vocab];
        for (v, out) in logits.iter_mut().enumerate() {
            let row = &lm_head[v * hidden..(v + 1) * hidden];
            let mut acc = 0.0f32;
            for j in 0..hidden {
                acc += row[j] * final_normed[j];
            }
            *out = acc;
        }
        logits
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f64 {
        let diff_sq: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| ((x - y) as f64).powi(2))
            .sum();
        let norm_sq: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
        (diff_sq / norm_sq.max(1e-30)).sqrt()
    }

    #[test]
    fn lm_head_logits_parity() {
        // Odd, unequal dims on purpose: a wrong transpose or swapped
        // dim-order (e.g. treating lm_head as [hidden, vocab] instead of
        // the actual [vocab, hidden]) would either panic on an out-of-bounds
        // slice or produce values that disagree with the scalar oracle —
        // it cannot silently pass by square-shape symmetry.
        const HIDDEN: usize = 7;
        const VOCAB: usize = 5;
        const PARITY_TOL: f64 = 1e-4;

        let lm_head: Vec<f32> = (0..VOCAB * HIDDEN)
            .map(|i| (i as f32 + 1.0) * 0.037 - 0.6)
            .collect();
        let final_normed: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 + 1.0) * 0.11 - 0.3).collect();

        let vectorized = lm_head_logits(&lm_head, &final_normed, HIDDEN, VOCAB);
        let scalar = lm_head_logits_scalar(&lm_head, &final_normed, HIDDEN, VOCAB);

        assert_eq!(vectorized.len(), VOCAB);
        let err = rel_err(&vectorized, &scalar);
        eprintln!("lm_head_logits parity rel_err={err:.2e}");
        assert!(
            err < PARITY_TOL,
            "lm_head_logits vectorized vs scalar rel_err {err:.2e} >= {PARITY_TOL:.2e}"
        );
    }
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

/// Positions the per-token FFN block must actually run for. Every layer's
/// mixer output at every position feeds the *next* layer (causal attention
/// keys/values, GDN recurrent state), so the FFN block — a pointwise
/// transform that feeds nothing but its own layer's output — must still run
/// at every position on every non-terminal layer. On the terminal layer,
/// though, only `completion_range` positions are ever read back out (the
/// head loop only computes logits there), so prompt positions' FFN output
/// would be computed and immediately discarded: skip them.
fn ffn_compute_range(
    is_last_layer: bool,
    seq: usize,
    completion_range: std::ops::Range<usize>,
) -> std::ops::Range<usize> {
    if is_last_layer {
        completion_range
    } else {
        0..seq
    }
}

pub fn forward_full(
    ctx: &SeqCtx,
    layers: &[LayerW<'_>],
    loras: &[LoraParams],
    gdn_loras: &[GdnLoraParams],
    head: &Head<'_>,
    train: &TrainCtx<'_>,
) -> Result<FullFwd> {
    let d = train.dims();
    let gdn_dims = train.gdn_dims();
    let cfg = train.model();
    let rank = train.rank();
    let scale = train.scale();
    let hidden = d.hidden;
    let seq = ctx.seq_len;
    let mut h = ctx.h_in.clone();
    let mut layers_fwd: Vec<LayerFwd> = Vec::with_capacity(layers.len());
    let num_layers = layers.len();
    // Only the completion positions' post-stack hidden state is ever read
    // (the head loop below indexes `h[t]` for `t` in this same range), so on
    // the terminal layer the pointwise FFN block need not run over prompt
    // positions — their output is computed and immediately discarded.
    let completion_range = (ctx.completion_start - 1)..(ctx.seq_len - 1);

    for (li, lw) in layers.iter().enumerate() {
        let is_last_layer = li + 1 == num_layers;
        let h_layer_in = h.clone();
        let (normed_pre, inv_pre) = rmsnorm_seq(&h, &lw.pre_shift, hidden, seq, d.eps);

        let (mixer_out, mixer_cache) = match lw.kind {
            MixerKind::Gqa => {
                let (lora_a_q, lora_b_q, lora_a_v, lora_b_v, eff_rank, eff_scale) =
                    match lw.lora_slot {
                        Some(slot) => {
                            let lora = &loras[slot];
                            (
                                Some(lora.a_q.as_slice()),
                                Some(lora.b_q.as_slice()),
                                Some(lora.a_v.as_slice()),
                                Some(lora.b_v.as_slice()),
                                rank,
                                scale,
                            )
                        }
                        // A layer above the trainable window: still runs
                        // frozen so the forward pass covers the model's
                        // full deployed suffix.
                        None => (None, None, None, None, 0, 0.0),
                    };
                let (out, cache) = gqa_forward_with_cache(
                    &normed_pre,
                    lw.w_q,
                    lw.w_k,
                    lw.w_v,
                    lw.w_o,
                    lw.q_norm,
                    lw.k_norm,
                    lora_a_q,
                    lora_b_q,
                    lora_a_v,
                    lora_b_v,
                    eff_rank,
                    eff_scale,
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
                match lw.lora_slot {
                    Some(slot) => {
                        let l = &gdn_loras[slot];
                        gdn_forward_save(
                            &normed_pre,
                            gdn_w,
                            cfg,
                            &mut saved,
                            &mut out,
                            Some(&l.a_qkv),
                            Some(&l.b_qkv),
                            Some(&l.a_z),
                            Some(&l.b_z),
                            Some(&l.a_b),
                            Some(&l.b_b),
                            Some(&l.a_a),
                            Some(&l.b_a),
                            Some(&l.a_out),
                            Some(&l.b_out),
                            rank,
                            scale,
                        );
                    }
                    None => {
                        gdn_forward_save(
                            &normed_pre,
                            gdn_w,
                            cfg,
                            &mut saved,
                            &mut out,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            0,
                            0.0,
                        );
                    }
                }
                (out, MixerCache::Gdn(Box::new(saved)))
            }
        };

        let mut h_mid = h_layer_in.clone();
        for (a, b) in h_mid.iter_mut().zip(mixer_out.iter()) {
            *a += *b;
        }

        let (normed_ffn, inv_ffn) = rmsnorm_seq(&h_mid, &lw.post_shift, hidden, seq, d.eps);
        let mut gate_pre: Vec<Vec<f32>> = vec![Vec::new(); seq];
        let mut up_pre: Vec<Vec<f32>> = vec![Vec::new(); seq];
        let mut h_next = h_mid.clone();
        for t in ffn_compute_range(is_last_layer, seq, completion_range.clone()) {
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
            gate_pre[t] = gp;
            up_pre[t] = up;
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

pub fn eval_chain_nll(
    caches: &[SeqCtx],
    layers: &[LayerW<'_>],
    loras: &[LoraParams],
    gdn_loras: &[GdnLoraParams],
    head: &Head<'_>,
    train: &TrainCtx<'_>,
) -> Result<f32> {
    let mut nll_sum = 0.0f64;
    let mut n = 0usize;
    for ctx in caches {
        let fwd = forward_full(ctx, layers, loras, gdn_loras, head, train)?;
        for p in &fwd.positions {
            nll_sum += position_nll(&p.logits, p.target) as f64;
            n += 1;
        }
    }
    Ok((nll_sum / n.max(1) as f64) as f32)
}

/// Compute reverse-mode NLL and gradients for the assembled tape.
/// Returns separate GQA and GDN slot gradients; either collection may be empty.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#nll_and_grads) for reverse-pass ordering.
pub fn nll_and_grads(
    fwd: &FullFwd,
    layers: &[LayerW<'_>],
    loras: &[LoraParams],
    head: &Head<'_>,
    train: &TrainCtx<'_>,
) -> Result<(f32, usize, Vec<Grads>, Vec<GdnLoraParams>)> {
    let d = train.dims();
    let gdn_dims = train.gdn_dims();
    let rank = train.rank();
    let scale = train.scale();
    let num_slots = train.num_gqa_slots();
    let num_gdn_slots = train.num_gdn_slots();
    let hidden = d.hidden;
    let seq = fwd.h_final.len() / hidden;
    let n_comp = fwd.positions.len().max(1) as f32;
    let mut grads: Vec<Grads> = (0..num_slots)
        .map(|_| LoraParams::zeros(rank, d))
        .collect::<Result<Vec<_>>>()?;
    let mut gdn_grads: Vec<GdnLoraParams> = (0..num_gdn_slots)
        .map(|_| GdnLoraParams::zeros(rank, hidden, gdn_dims))
        .collect::<Result<Vec<_>>>()?;

    let num_layers = layers.len();
    // Mirrors `forward_full`'s terminal-layer FFN skip: the completion
    // positions are exactly where the loss injects nonzero gradient into
    // `d_h`, so on the last layer the FFN backward at prompt positions
    // would multiply a structurally zero incoming gradient through.
    let completion_range = match (fwd.positions.first(), fwd.positions.last()) {
        (Some(first), Some(last)) => first.t..(last.t + 1),
        _ => 0..0,
    };

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
        let is_last_layer = li + 1 == num_layers;

        let mut d_h_mid = d_h.clone();
        for t in ffn_compute_range(is_last_layer, seq, completion_range.clone()) {
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
                let (lora_a_q, lora_b_q, lora_a_v, lora_b_v, eff_rank, eff_scale) =
                    match lw.lora_slot {
                        Some(slot) => {
                            let lora = &loras[slot];
                            (
                                Some(lora.a_q.as_slice()),
                                Some(lora.b_q.as_slice()),
                                Some(lora.a_v.as_slice()),
                                Some(lora.b_v.as_slice()),
                                rank,
                                scale,
                            )
                        }
                        None => (None, None, None, None, 0, 0.0),
                    };
                let g = gqa_backward(
                    &d_h_mid, cache, lw.w_q, lw.w_k, lw.w_v, lw.w_o, lw.q_norm, lw.k_norm,
                    lora_a_q, lora_b_q, lora_a_v, lora_b_v, eff_rank, eff_scale,
                );
                if let Some(slot) = lw.lora_slot {
                    grads[slot] = Grads {
                        a_q: g.grad_a_q,
                        b_q: g.grad_b_q,
                        a_v: g.grad_a_v,
                        b_v: g.grad_b_v,
                    };
                }
                g.dx
            }
            MixerCache::Gdn(saved) => {
                let gdn_w = lw.gdn.ok_or_else(|| {
                    TuneError::Validation("GDN layer missing weights in backward".to_string())
                })?;
                let g = gdn_backward(&d_h_mid, saved, gdn_w);
                if let Some(slot) = lw.lora_slot {
                    gdn_grads[slot] = GdnLoraParams {
                        a_qkv: g.grad_a_qkv,
                        b_qkv: g.grad_b_qkv,
                        a_z: g.grad_a_z,
                        b_z: g.grad_b_z,
                        a_b: g.grad_a_b,
                        b_b: g.grad_b_b,
                        a_a: g.grad_a_a,
                        b_a: g.grad_b_a,
                        a_out: g.grad_a_out,
                        b_out: g.grad_b_out,
                    };
                }
                g.dx
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

    Ok((nll_sum as f32, fwd.positions.len(), grads, gdn_grads))
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

pub fn apply_adam_updates(
    adam: &mut AdamState,
    loras: &mut [LoraParams],
    grads: &[Grads],
    train: &TrainCtx<'_>,
) {
    let slot_layers = train.gqa_layers();
    let lr = train.learning_rate();
    let beta1 = train.beta1();
    let beta2 = train.beta2();
    let eps_adam = train.epsilon();
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

/// Apply Adam updates to all ten factor arrays in each GDN LoRA slot.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#nll_and_grads) for the GDN slot layout.
pub fn apply_gdn_adam_updates(
    adam: &mut AdamState,
    gdn_loras: &mut [GdnLoraParams],
    gdn_grads: &[GdnLoraParams],
    train: &TrainCtx<'_>,
) {
    let slot_layers = train.gdn_layers();
    let lr = train.learning_rate();
    let beta1 = train.beta1();
    let beta2 = train.beta2();
    let eps_adam = train.epsilon();
    for slot in 0..slot_layers.len() {
        let li = slot_layers[slot];
        macro_rules! step {
            ($field:ident, $tag:literal) => {
                adam.step(
                    &format!("l{li}_gdn_{}", $tag),
                    &mut gdn_loras[slot].$field,
                    &gdn_grads[slot].$field,
                    lr,
                    beta1,
                    beta2,
                    eps_adam,
                    0.0,
                    false,
                );
            };
        }
        step!(a_qkv, "a_qkv");
        step!(b_qkv, "b_qkv");
        step!(a_z, "a_z");
        step!(b_z, "b_z");
        step!(a_b, "a_b");
        step!(b_b, "b_b");
        step!(a_a, "a_a");
        step!(b_a, "b_a");
        step!(a_out, "a_out");
        step!(b_out, "b_out");
    }
}

#[cfg(test)]
mod gdn_lora_tests {
    use super::*;

    fn tiny_gdn_dims() -> GdnDims {
        // hidden=16, num_kh=2, value_heads=4, key_dim=4, value_dim=4 — matches
        // the shapes exercised by gdn_backward's own multi-head gradcheck fixture.
        GdnDims {
            num_kh: 2,
            value_heads: 4,
            key_dim: 4,
            value_dim: 4,
            qkv_dim: 2 * (2 * 4) + 4 * 4, // 2*key_dim*num_kh + value_heads*value_dim = 32
            output_dim: 4 * 4,            // value_heads * value_dim = 16
            kernel_size: 3,
            scale: 0.5,
        }
    }

    #[test]
    fn gdn_lora_params_zeros_shapes_match_gdn_dims() {
        let gd = tiny_gdn_dims();
        let hidden = 16;
        let rank = 3;
        let p = GdnLoraParams::zeros(rank, hidden, &gd).expect("zeros should not overflow");

        assert_eq!(p.a_qkv.len(), rank * hidden);
        assert_eq!(p.b_qkv.len(), gd.qkv_dim * rank);
        assert_eq!(p.a_z.len(), rank * hidden);
        assert_eq!(p.b_z.len(), gd.output_dim * rank);
        assert_eq!(p.a_b.len(), rank * hidden);
        assert_eq!(p.b_b.len(), gd.value_heads * rank);
        assert_eq!(p.a_a.len(), rank * hidden);
        assert_eq!(p.b_a.len(), gd.value_heads * rank);
        assert_eq!(p.a_out.len(), rank * gd.output_dim);
        assert_eq!(p.b_out.len(), hidden * rank);
        // Zero-initialised.
        assert!(p.a_qkv.iter().all(|&v| v == 0.0));
        assert!(p.b_out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gdn_lora_params_zeros_rejects_overflowing_rank() {
        let gd = tiny_gdn_dims();
        // rank so large that rank * hidden overflows usize on any platform.
        let err = GdnLoraParams::zeros(usize::MAX / 2, 16, &gd);
        assert!(err.is_err(), "overflowing rank*hidden should be rejected");
    }

    #[test]
    fn apply_gdn_adam_updates_moves_nonzero_grad_params() {
        let gd = tiny_gdn_dims();
        let hidden = 16;
        let rank = 3;
        let mut gdn_loras = vec![GdnLoraParams::zeros(rank, hidden, &gd).unwrap()];
        // Non-zero, non-uniform gradient so the Adam step has a definite sign.
        let mut grads = GdnLoraParams::zeros(rank, hidden, &gd).unwrap();
        for (i, v) in grads.a_qkv.iter_mut().enumerate() {
            *v = 0.01 * (i as f32 + 1.0);
        }
        for (i, v) in grads.b_out.iter_mut().enumerate() {
            *v = -0.02 * (i as f32 + 1.0);
        }
        let gdn_grads = vec![grads];

        // apply_gdn_adam_updates only reads `train`'s gdn slot layout and Adam
        // hyperparameters (not its geometry), so a real preset's consistent
        // Dims/GdnDims/Qwen35Config triple is enough here — it need not match
        // `gd` (the tiny fixture the GdnLoraParams shapes above were built
        // from).
        let cfg = Qwen35Config::qwen35_0_8b();
        let dims = Dims {
            hidden: cfg.hidden_size,
            vocab: cfg.vocab_size,
            num_q_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_dim: cfg.rope_dim(),
            inter: cfg.intermediate_size,
            q_dim: cfg.full_q_dim(),
            kv_dim: cfg.full_kv_dim(),
            eps: cfg.rms_norm_eps,
        };
        let real_gdn_dims = GdnDims::from_cfg(&cfg);
        let gqa_layers: [usize; 0] = [];
        // Layer 7 was a full-attention (GQA) layer on the 0.8B preset — an
        // arbitrary index was fine before TrainCtx validated per-slot mixer
        // kind. Now the fixture must name an actual GatedDeltaNet layer.
        assert!(
            !cfg.is_full_attention(20),
            "layer 20 must be GDN on the 0.8B preset"
        );
        let gdn_layers = [20usize];
        let train = TrainCtx::try_new(
            TapeGeometry::new(&dims, &real_gdn_dims, &cfg),
            rank,
            (rank as f32) * 2.0,
            SlotLayout::new(&gqa_layers, &gdn_layers),
            AdamConfig::new(0.1, 0.9, 0.999, 1e-8),
        )
        .expect("valid TrainCtx");

        let mut adam = AdamState::new();
        apply_gdn_adam_updates(&mut adam, &mut gdn_loras, &gdn_grads, &train);

        // Adam moves params opposite the gradient sign: positive grad -> param
        // decreases from its zero init; negative grad -> param increases.
        assert!(
            gdn_loras[0].a_qkv.iter().all(|&v| v < 0.0),
            "a_qkv should move negative under positive grad: {:?}",
            gdn_loras[0].a_qkv
        );
        assert!(
            gdn_loras[0].b_out.iter().all(|&v| v > 0.0),
            "b_out should move positive under negative grad: {:?}",
            gdn_loras[0].b_out
        );
        // Untouched (zero-grad) arrays stay at their zero init.
        assert!(gdn_loras[0].a_z.iter().all(|&v| v == 0.0));
    }
}

#[cfg(test)]
mod train_ctx_tests {
    use super::*;

    /// A geometry triple that satisfies `TrainCtx::try_new`'s consistency
    /// check by construction: `Dims`/`GdnDims` are derived from the same
    /// `Qwen35Config` the way every real call site (`train.rs`,
    /// `train_grad_full.rs`) builds them.
    struct Geometry {
        cfg: Qwen35Config,
        dims: Dims,
        gdn_dims: GdnDims,
    }

    fn valid_geometry() -> Geometry {
        let cfg = Qwen35Config::qwen35_0_8b();
        let dims = Dims {
            hidden: cfg.hidden_size,
            vocab: cfg.vocab_size,
            num_q_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_dim: cfg.rope_dim(),
            inter: cfg.intermediate_size,
            q_dim: cfg.full_q_dim(),
            kv_dim: cfg.full_kv_dim(),
            eps: cfg.rms_norm_eps,
        };
        let gdn_dims = GdnDims::from_cfg(&cfg);
        Geometry {
            cfg,
            dims,
            gdn_dims,
        }
    }

    fn valid_adam() -> AdamConfig {
        AdamConfig::new(1e-3, 0.9, 0.999, 1e-8)
    }

    /// `TrainCtx` intentionally has no `Debug` impl (it borrows a `Qwen35Config`
    /// among other things, and nothing needs to print one), so
    /// `Result::unwrap_err` isn't available — extract the error message by hand.
    fn expect_err(result: Result<TrainCtx<'_>>) -> String {
        match result {
            Ok(_) => panic!("expected TrainCtx::try_new to reject this input"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn try_new_accepts_valid_inputs() {
        let g = valid_geometry();
        let gqa = [19usize, 23];
        let gdn = [20usize, 21, 22];
        let train = TrainCtx::try_new(
            TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
            8,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        )
        .expect("valid geometry/rank/slots/adam must construct a TrainCtx");
        assert_eq!(train.num_gqa_slots(), 2);
        assert_eq!(train.num_gdn_slots(), 3);
        assert_eq!(train.rank(), 8);
        assert!((train.scale() - 2.0).abs() < 1e-6); // alpha / rank = 16 / 8
    }

    /// Mutation-sensitive: an out-of-range GQA slot layer (>= num_hidden_layers)
    /// must be rejected — without this guard the tape indexes
    /// `model.weights.layers[idx]` (or an equivalent slot array) out of bounds.
    #[test]
    fn try_new_rejects_out_of_range_slot_layer() {
        let g = valid_geometry();
        let gqa = [g.cfg.num_hidden_layers]; // one past the last valid index
        let gdn: [usize; 0] = [];
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
            8,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        ));
        assert!(
            err.contains("out of range"),
            "expected out-of-range rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: a duplicate layer index within one slot list must
    /// be rejected — two slots claiming the same layer would silently alias
    /// the same LoRA weights to two different gradient accumulators.
    #[test]
    fn try_new_rejects_duplicate_slot_layer() {
        let g = valid_geometry();
        let gqa = [19usize, 19];
        let gdn: [usize; 0] = [];
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
            8,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        ));
        assert!(
            err.contains("duplicate"),
            "expected duplicate-layer rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: a layer index claimed by both the GQA and GDN
    /// slot lists is a mixer-kind conflict — the same layer cannot be
    /// trained as both mixer kinds in one materialised tape. (Layer 19 is
    /// full-attention on the 0.8B preset, so this also exercises the
    /// one-sided "gdn slot layer is actually GQA" rejection below — either
    /// way, `TrainCtx` must not construct.)
    #[test]
    fn try_new_rejects_mixer_kind_slot_index_mismatch() {
        let g = valid_geometry();
        let gqa = [19usize];
        let gdn = [19usize];
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
            8,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        ));
        assert!(
            err.contains("mixer-kind"),
            "expected mixer-kind/slot-index mismatch rejection; got: {err}"
        );
    }

    /// Mutation-sensitive, one-sided misclassification: layer 20 is
    /// GatedDeltaNet (not full-attention) on the 0.8B preset, so it must be
    /// rejected when claimed as a GQA slot even though nothing else conflicts.
    #[test]
    fn try_new_rejects_gqa_slot_layer_wrong_kind() {
        let g = valid_geometry();
        assert!(
            !g.cfg.is_full_attention(20),
            "fixture assumption: layer 20 must be GDN on the 0.8B preset"
        );
        let gqa = [20usize];
        let gdn: [usize; 0] = [];
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
            8,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        ));
        assert!(
            err.contains("mixer-kind"),
            "expected gqa-slot-wrong-kind rejection; got: {err}"
        );
    }

    /// Mutation-sensitive, one-sided misclassification: layer 19 is
    /// full-attention (GQA) on the 0.8B preset, so it must be rejected when
    /// claimed as a GDN slot even though nothing else conflicts.
    #[test]
    fn try_new_rejects_gdn_slot_layer_wrong_kind() {
        let g = valid_geometry();
        assert!(
            g.cfg.is_full_attention(19),
            "fixture assumption: layer 19 must be full-attention on the 0.8B preset"
        );
        let gqa: [usize; 0] = [];
        let gdn = [19usize];
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
            8,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        ));
        assert!(
            err.contains("mixer-kind"),
            "expected gdn-slot-wrong-kind rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: this exact swap was unrejected pre-fix —
    /// `gqa_layers = [20]` (actually GDN), `gdn_layers = [19]` (actually
    /// GQA) — both wrong, together. Must be rejected; before this fix
    /// `TrainCtx::try_new(...).is_ok()` on this input.
    #[test]
    fn try_new_rejects_gqa_gdn_slot_swap_19_20() {
        let g = valid_geometry();
        let gqa = [20usize];
        let gdn = [19usize];
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
            8,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        ));
        assert!(
            err.contains("mixer-kind"),
            "expected 19/20 swap rejection; got: {err}"
        );
    }

    #[test]
    fn try_new_rejects_zero_rank() {
        let g = valid_geometry();
        let gqa = [19usize];
        let gdn: [usize; 0] = [];
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
            0,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        ));
        assert!(
            err.contains("rank"),
            "expected rank-zero rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: non-finite alpha must be rejected before it can
    /// propagate into `scale = alpha / rank` and poison every LoRA-scaled
    /// forward/backward computation with NaN/inf.
    #[test]
    fn try_new_rejects_non_finite_alpha() {
        let g = valid_geometry();
        let gqa = [19usize];
        let gdn: [usize; 0] = [];
        for bad_alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = expect_err(TrainCtx::try_new(
                TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
                8,
                bad_alpha,
                SlotLayout::new(&gqa, &gdn),
                valid_adam(),
            ));
            assert!(
                err.contains("alpha"),
                "expected alpha rejection for {bad_alpha}; got: {err}"
            );
        }
    }

    /// Mutation-sensitive: non-finite Adam hyperparameters (learning rate,
    /// beta1, beta2, epsilon) must each be rejected — any one of them
    /// propagates NaN/inf into every optimizer step.
    #[test]
    fn try_new_rejects_non_finite_adam_hyperparams() {
        let g = valid_geometry();
        let gqa = [19usize];
        let gdn: [usize; 0] = [];
        let cases: [(&str, AdamConfig); 4] = [
            ("learning_rate", AdamConfig::new(f32::NAN, 0.9, 0.999, 1e-8)),
            ("beta1", AdamConfig::new(1e-3, f32::INFINITY, 0.999, 1e-8)),
            ("beta2", AdamConfig::new(1e-3, 0.9, f32::NEG_INFINITY, 1e-8)),
            ("epsilon", AdamConfig::new(1e-3, 0.9, 0.999, f32::NAN)),
        ];
        for (name, adam) in cases {
            let err = expect_err(TrainCtx::try_new(
                TapeGeometry::new(&g.dims, &g.gdn_dims, &g.cfg),
                8,
                16.0,
                SlotLayout::new(&gqa, &gdn),
                adam,
            ));
            assert!(err.contains(name), "expected {name} rejection; got: {err}");
        }
    }

    /// Mutation-sensitive: `Dims` built from a different model than
    /// `TapeGeometry.model` must be rejected — this is the geometry-
    /// consistency check the ten-argument signatures had no way to enforce.
    #[test]
    fn try_new_rejects_dims_model_mismatch() {
        let g = valid_geometry();
        let mut mismatched_dims = g.dims;
        mismatched_dims.hidden += 1;
        let gqa = [19usize];
        let gdn: [usize; 0] = [];
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&mismatched_dims, &g.gdn_dims, &g.cfg),
            8,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        ));
        assert!(
            err.contains("Dims"),
            "expected Dims/model mismatch rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: `GdnDims` built from a different model than
    /// `TapeGeometry.model` must be rejected, mirroring the `Dims` check.
    #[test]
    fn try_new_rejects_gdn_dims_model_mismatch() {
        let g = valid_geometry();
        let mut mismatched_gdn = g.gdn_dims;
        mismatched_gdn.value_heads += 1;
        let gqa = [19usize];
        let gdn: [usize; 0] = [];
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&g.dims, &mismatched_gdn, &g.cfg),
            8,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        ));
        assert!(
            err.contains("GdnDims"),
            "expected GdnDims/model mismatch rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: NaN bypasses `(a - b).abs() > 1e-9` (NaN
    /// comparisons are always false), so a non-finite `Dims.eps` would
    /// silently pass the tolerance check and poison every RMSNorm call in
    /// the tape. The finite check ahead of the tolerance comparison must
    /// catch it explicitly.
    #[test]
    fn try_new_rejects_non_finite_dims_eps() {
        let g = valid_geometry();
        let mut nan_dims = g.dims;
        nan_dims.eps = f32::NAN;
        let gqa = [19usize];
        let gdn: [usize; 0] = [];
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&nan_dims, &g.gdn_dims, &g.cfg),
            8,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        ));
        assert!(
            err.contains("eps"),
            "expected non-finite Dims.eps rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: same NaN-bypasses-tolerance hole as `Dims.eps`,
    /// for `GdnDims.scale` — a non-finite scale would poison every
    /// GatedDeltaNet forward/backward call in the tape.
    #[test]
    fn try_new_rejects_non_finite_gdn_scale() {
        let g = valid_geometry();
        let mut nan_gdn = g.gdn_dims;
        nan_gdn.scale = f32::NAN;
        let gqa = [19usize];
        let gdn: [usize; 0] = [];
        let err = expect_err(TrainCtx::try_new(
            TapeGeometry::new(&g.dims, &nan_gdn, &g.cfg),
            8,
            16.0,
            SlotLayout::new(&gqa, &gdn),
            valid_adam(),
        ));
        assert!(
            err.contains("scale"),
            "expected non-finite GdnDims.scale rejection; got: {err}"
        );
    }
}

#[cfg(test)]
mod ffn_compute_range_tests {
    use super::*;

    /// Non-terminal layers must still run the FFN block at every position:
    /// their output feeds the next layer's mixer (causal K/V, GDN state) at
    /// every position, not just the completion positions.
    #[test]
    fn non_terminal_layer_uses_full_sequence() {
        let range = ffn_compute_range(false, 10, 6..9);
        assert_eq!(range, 0..10);
    }

    /// Mutation-sensitive: if the terminal-layer branch is dropped (or
    /// inverted), the FFN block runs over the full prompt on the last layer
    /// every step — a confirmed O(prompt_length) regression, since only the
    /// completion positions' post-stack hidden state is ever read by the
    /// head loop.
    #[test]
    fn terminal_layer_uses_completion_positions_only() {
        let range = ffn_compute_range(true, 10, 6..9);
        assert_eq!(range, 6..9);
        assert_ne!(range, 0..10, "must not degrade to the full-sequence range");
    }

    #[test]
    fn terminal_layer_with_single_completion_position() {
        // A one-token completion (e.g. a 2-token pair) still only computes
        // FFN for that single position, not the whole prompt.
        let range = ffn_compute_range(true, 64, 62..63);
        assert_eq!(range, 62..63);
        assert_eq!(range.len(), 1);
    }
}
