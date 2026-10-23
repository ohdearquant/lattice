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
use lattice_inference::backward::tape::{rms_norm_forward, swiglu_forward_seq};
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
mod lm_head_logits_tests;

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

/// Splits a `[range.len(), width]` block of per-position rows into one `Vec` per sequence
/// position. Positions outside `range` get an empty `Vec`, never zeros: `swiglu_backward`
/// asserts `gate_pre.len() == inter`, so reading a position the forward did not compute
/// fails loudly instead of back-propagating through a zero activation. Takes `flat` by value
/// so it is freed as soon as its rows are copied out, before the next block is split.
fn rows_by_position(
    flat: Vec<f32>,
    range: &std::ops::Range<usize>,
    seq: usize,
    width: usize,
) -> Vec<Vec<f32>> {
    let mut rows = vec![Vec::new(); seq];
    for (i, t) in range.clone().enumerate() {
        rows[t] = flat[i * width..(i + 1) * width].to_vec();
    }
    rows
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
        let ffn_range = ffn_compute_range(is_last_layer, seq, completion_range.clone());
        let mut h_next = h_mid.clone();
        let (gate_pre, up_pre) = if ffn_range.is_empty() {
            (vec![Vec::new(); seq], vec![Vec::new(); seq])
        } else {
            let rows = ffn_range.start * hidden..ffn_range.end * hidden;
            let (ffn_out, gp, up) = swiglu_forward_seq(
                &normed_ffn[rows.clone()],
                lw.w_gate,
                lw.w_up,
                lw.w_down,
                ffn_range.len(),
                hidden,
                d.inter,
            );
            for (o, f) in h_next[rows].iter_mut().zip(ffn_out.iter()) {
                *o += *f;
            }
            (
                rows_by_position(gp, &ffn_range, seq, d.inter),
                rows_by_position(up, &ffn_range, seq, d.inter),
            )
        };

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

/// Index of the greatest logit. Ties keep the FIRST index, and no NaN can win a
/// position: the running best is a VALUE seeded at -inf rather than the element at
/// the running index, so a NaN never becomes the thing later elements are compared
/// against. The first draft did compare against `logits[best]`, which loses that
/// property at exactly one input -- a NaN in position 0 pins `best` at 0 forever,
/// because every later `v > NaN` is false -- and would have reported token 0 as the
/// argmax of a diverged row. An all-NaN row has no defensible answer and returns 0.
/// All of this is pinned by tests; a rewrite to `partial_cmp().unwrap()` would change
/// the tie rule silently and panic on the degenerate row this survives.
pub(crate) fn argmax_logit(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        if *v > best_v {
            best_v = *v;
            best = i;
        }
    }
    best
}

/// Fold one sequence's per-position agreements into counts and a verdict.
///
/// Returns `(correct, total, sequence_agreed)`. A sequence with NO scored positions
/// does NOT agree: counting it would let an empty completion raise the score, and
/// that is the direction which flatters, so it is the direction to refuse.
pub(crate) fn fold_sequence_agreement(
    agreements: impl Iterator<Item = bool>,
) -> (usize, usize, bool) {
    let mut correct = 0usize;
    let mut total = 0usize;
    for ok in agreements {
        total += 1;
        if ok {
            correct += 1;
        }
    }
    (correct, total, total > 0 && correct == total)
}

/// What one scoring pass measured: mean NLL plus two agreement counts.
///
/// Named fields rather than a tuple deliberately. Four of the five members are
/// `usize` and two of them are a correct-count beside its denominator, so a tuple
/// lets a caller transpose `positions_correct` with `positions` and get a plausible
/// ratio out of a silent compile. The numbers end up in a comparison table where
/// nobody re-derives them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChainScores {
    pub mean_nll: f32,
    pub positions_correct: usize,
    pub positions: usize,
    pub sequences_correct: usize,
    pub sequences: usize,
}

/// Mean NLL plus teacher-forced agreement over the same forward passes.
///
/// The agreement is TEACHER-FORCED: at each masked completion position the
/// argmax of the logits is compared with the gold token, and a sequence counts
/// as exact only when every one of its positions agrees. It is NOT a free-running
/// greedy decode, and the two can differ, because a free run conditions on its own
/// earlier tokens while this conditions on the gold ones. The distinction is stated
/// here rather than in a caller because the number is going into a comparison table
/// where the label is all anyone will read.
pub fn eval_chain_scores(
    caches: &[SeqCtx],
    layers: &[LayerW<'_>],
    loras: &[LoraParams],
    gdn_loras: &[GdnLoraParams],
    head: &Head<'_>,
    train: &TrainCtx<'_>,
) -> Result<ChainScores> {
    let mut nll_sum = 0.0f64;
    let mut n = 0usize;
    let mut pos_ok = 0usize;
    let mut seq_ok = 0usize;
    for ctx in caches {
        let fwd = forward_full(ctx, layers, loras, gdn_loras, head, train)?;
        for p in &fwd.positions {
            nll_sum += position_nll(&p.logits, p.target) as f64;
        }
        // p.target is u32, matching position_nll above; compare in usize.
        let (correct, total, agreed) = fold_sequence_agreement(
            fwd.positions
                .iter()
                .map(|p| argmax_logit(&p.logits) == p.target as usize),
        );
        pos_ok += correct;
        n += total;
        if agreed {
            seq_ok += 1;
        }
    }
    Ok(ChainScores {
        mean_nll: (nll_sum / n.max(1) as f64) as f32,
        positions_correct: pos_ok,
        positions: n,
        sequences_correct: seq_ok,
        sequences: caches.len(),
    })
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

/// Which Gated DeltaNet projections a training run targets.
///
/// The trainers used to target all five unconditionally while the Metal forward
/// accepts three, so an adapter could train to completion and then be refused at
/// load -- and the loss it minimised included a delta the server would never
/// apply. The selection therefore has to reach the OPTIMIZER, not just the
/// writer: dropping the two modules at save time would leave the saved adapter
/// different from the one the objective was computed for.
///
/// Both variants resolve their module names from `lattice_inference::lora_hook`,
/// which is the only copy of "which GDN modules can be served".
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GdnModuleSelection {
    /// The set the Metal forward can apply, and the default.
    ///
    /// Default because the failure this guards is silent at train time and shows
    /// up only at load, which is the worst place to learn about it -- after the
    /// compute has been spent.
    #[default]
    Served,
    /// All five GDN projections, including the two the fused recurrence kernel
    /// consumes.
    ///
    /// Trains a delta the Metal forward cannot serve. Legitimate for CPU-path
    /// work, which applies every projection, and it is the mutation control that
    /// proves the default is doing something: under `All` the refusal returns and
    /// `b_a`/`b_b` stop being zero.
    All,
}

impl GdnModuleSelection {
    /// The GDN module names this selection targets, in the trainers' emission order.
    #[must_use]
    pub fn modules(self) -> &'static [&'static str] {
        match self {
            Self::Served => &lattice_inference::lora_hook::GDN_LORA_MODULES_SERVABLE,
            Self::All => &lattice_inference::lora_hook::GDN_LORA_MODULES,
        }
    }

    /// Whether the fused-kernel projections `in_proj_a` and `in_proj_b` are trained.
    ///
    /// Named after what it decides rather than after the variant, because it is
    /// read at the four `a_b/b_b/a_a/b_a` optimizer steps and at the two
    /// `adapter_layers` inserts, and both sites are about those tensors.
    #[must_use]
    pub fn trains_fused_kernel_projections(self) -> bool {
        matches!(self, Self::All)
    }
}

/// Apply Adam updates to the GDN LoRA factor arrays selected by `selection`.
///
/// Under [`GdnModuleSelection::Served`] the `a_b`/`b_b`/`a_a`/`b_a` steps are
/// skipped rather than stepped-and-discarded. That is what makes the omission
/// from the saved adapter provably lossless: every `b_*` array is
/// zero-initialised, so an unstepped module contributes an exactly-zero delta,
/// and the adapter that is served is the adapter the loss was minimised for.
/// Skipping also keeps those tensors out of `AdamState`, so no moment estimates
/// accumulate for parameters nothing will read.
///
/// See [`docs/lora-core.md`](../../docs/lora-core.md#nll_and_grads) for the GDN slot layout.
pub fn apply_gdn_adam_updates(
    adam: &mut AdamState,
    gdn_loras: &mut [GdnLoraParams],
    gdn_grads: &[GdnLoraParams],
    train: &TrainCtx<'_>,
    selection: GdnModuleSelection,
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
        if selection.trains_fused_kernel_projections() {
            step!(a_b, "a_b");
            step!(b_b, "b_b");
            step!(a_a, "a_a");
            step!(b_a, "b_a");
        }
        step!(a_out, "a_out");
        step!(b_out, "b_out");
    }
}

#[cfg(test)]
mod gdn_lora_tests;

#[cfg(test)]
mod agreement_tests;

#[cfg(test)]
mod train_ctx_tests;

#[cfg(test)]
mod ffn_compute_range_tests;

#[cfg(test)]
mod ffn_seq_forward_tests;
