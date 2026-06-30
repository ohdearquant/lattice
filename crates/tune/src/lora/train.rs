//! Micro-LoRA training: CPU backward pass for a configurable layer range.
//!
//! Provides a self-contained training loop over caller-supplied token pairs.
//! Callers that need full training infrastructure (data loading, gradcheck,
//! validation) should use the `train_grad_full` binary instead.

use std::collections::HashMap;

use lattice_inference::attention::gdn::GatedDeltaNetWeights;
use lattice_inference::attention::gdn_backward::{GdnSaved, gdn_backward, gdn_forward_save};
use lattice_inference::backward::attention_gqa::{AttnCache, gqa_backward, gqa_forward_with_cache};
use lattice_inference::backward::ops::{linear_vjp, rmsnorm_backward, swiglu_backward};
use lattice_inference::backward::tape::{rms_norm_forward, swiglu_forward};
use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::model::qwen35_config::Qwen35Config;

use crate::error::{Result, TuneError};
use crate::lora::{AdamState, LoraAdapter, LoraConfig, LoraLayer};

const TOP_LAYER: usize = 23;

/// Upper bound on LoRA rank accepted by [`train_micro_lora`].
///
/// Practical LoRA ranks are 4–64; 512 is a generous safety valve that blocks
/// accidental overflow in buffer-size products (`rank * hidden`) while still
/// accommodating any foreseeable use case.
const MAX_LORA_RANK: usize = 512;

/// Upper bound on training steps accepted by [`train_micro_lora`].
///
/// 100 000 steps is well above any micro-LoRA use case and prevents an
/// accidentally large caller-supplied value from locking the process for hours.
const MAX_TRAIN_STEPS: usize = 100_000;

/// Upper bound on `max_seq_len` accepted by [`train_micro_lora`].
///
/// 8 192 tokens is a generous ceiling for a micro-training helper; the actual
/// model context window is the binding constraint at runtime.
const MAX_TRAIN_SEQ_LEN: usize = 8_192;

/// A single training example: tokenized text plus the index at which the
/// completion (supervised signal) begins.
pub struct TrainingPair {
    /// Full token sequence (prompt tokens followed by completion tokens).
    pub tokens: Vec<u32>,
    /// Index of the first completion token. Gradient is computed at positions
    /// `completion_start - 1 ..= tokens.len() - 2` (predicting `tokens[t+1]`).
    pub completion_start: usize,
}

/// Configuration for [`train_micro_lora`].
pub struct MicroLoraConfig {
    /// LoRA rank.
    pub rank: usize,
    /// LoRA alpha (effective scale is `alpha / rank`).
    pub alpha: f32,
    /// Index of the first materialised (trained) layer. Layers before this are
    /// run frozen by the model's own forward pass.
    pub first_layer: usize,
    /// Number of Adam steps.
    pub steps: usize,
    /// Adam learning rate.
    pub learning_rate: f32,
    /// Maximum sequence length in tokens. Pairs longer than this return an error.
    pub max_seq_len: usize,
}

impl Default for MicroLoraConfig {
    fn default() -> Self {
        MicroLoraConfig {
            rank: 4,
            alpha: 8.0,
            first_layer: 19,
            steps: 25,
            learning_rate: 1e-3,
            max_seq_len: 64,
        }
    }
}

/// Validate all caller-supplied inputs for [`train_micro_lora`].
///
/// Extracted so the validation logic can be tested without a real model
/// checkpoint. `vocab_size` and `num_hidden_layers` come from `model.config()`;
/// `num_hidden_layers` bounds the layer range the trainer will index, so a model
/// with too few layers is rejected before any layer access.
///
/// Returns `Err(TuneError::Validation)` on the first anomaly found.
pub(crate) fn validate_micro_lora_inputs(
    vocab_size: usize,
    num_hidden_layers: usize,
    pairs: &[TrainingPair],
    config: &MicroLoraConfig,
) -> Result<()> {
    if pairs.is_empty() {
        return Err(TuneError::Validation(
            "train_micro_lora requires at least one training pair".to_string(),
        ));
    }
    if config.rank == 0 {
        return Err(TuneError::Validation(
            "train_micro_lora: rank must be > 0".to_string(),
        ));
    }
    if config.rank > MAX_LORA_RANK {
        return Err(TuneError::Validation(format!(
            "train_micro_lora: rank {} exceeds maximum {}",
            config.rank, MAX_LORA_RANK
        )));
    }
    if config.steps > MAX_TRAIN_STEPS {
        return Err(TuneError::Validation(format!(
            "train_micro_lora: steps {} exceeds maximum {}",
            config.steps, MAX_TRAIN_STEPS
        )));
    }
    if config.max_seq_len > MAX_TRAIN_SEQ_LEN {
        return Err(TuneError::Validation(format!(
            "train_micro_lora: max_seq_len {} exceeds maximum {}",
            config.max_seq_len, MAX_TRAIN_SEQ_LEN
        )));
    }
    // first_layer and the hardcoded TOP_LAYER must both be valid indices into the
    // model's layer stack. train_micro_lora iterates `first_layer ..= TOP_LAYER`
    // and indexes `model.weights.layers[idx]` directly (through gqa_layer_weights /
    // gdn_layer_weights), which panics out of bounds. A model with fewer than
    // TOP_LAYER + 1 layers, or a first_layer past TOP_LAYER, must be rejected here,
    // before any model access in the layer loop below.
    if config.first_layer > TOP_LAYER {
        return Err(TuneError::Validation(format!(
            "train_micro_lora: first_layer {} exceeds maximum trainable layer index {TOP_LAYER}",
            config.first_layer
        )));
    }
    if num_hidden_layers <= TOP_LAYER {
        return Err(TuneError::Validation(format!(
            "train_micro_lora: model has {num_hidden_layers} layers but the trainer \
             requires at least {} (layer indices 0..={TOP_LAYER})",
            TOP_LAYER + 1
        )));
    }
    for (pair_idx, pair) in pairs.iter().enumerate() {
        if pair.tokens.len() < 2 {
            return Err(TuneError::Validation(format!(
                "train_micro_lora: pair {pair_idx} has {} tokens (minimum 2)",
                pair.tokens.len()
            )));
        }
        if pair.tokens.len() > config.max_seq_len {
            return Err(TuneError::Validation(format!(
                "train_micro_lora: pair {pair_idx} has {} tokens, exceeding max_seq_len {}",
                pair.tokens.len(),
                config.max_seq_len
            )));
        }
        // completion_start == 0 causes (completion_start - 1) to underflow in
        // forward_full's range `(completion_start - 1)..seq_len - 1`.
        if pair.completion_start == 0 {
            return Err(TuneError::Validation(format!(
                "train_micro_lora: pair {pair_idx} completion_start is 0 — must be >= 1"
            )));
        }
        // completion_start >= tokens.len() means no completion positions exist;
        // `tokens[t + 1]` at t = completion_start - 1 would be out of bounds.
        if pair.completion_start >= pair.tokens.len() {
            return Err(TuneError::Validation(format!(
                "train_micro_lora: pair {pair_idx} completion_start {} >= tokens.len() {}",
                pair.completion_start,
                pair.tokens.len()
            )));
        }
        // Out-of-vocab IDs panic at `logits[target as usize]` in forward_full.
        for (tok_idx, &tok) in pair.tokens.iter().enumerate() {
            if tok as usize >= vocab_size {
                return Err(TuneError::Validation(format!(
                    "train_micro_lora: pair {pair_idx} token[{tok_idx}]={tok} \
                     is out of vocabulary (vocab_size={vocab_size})"
                )));
            }
        }
    }
    Ok(())
}

/// Train a LoRA adapter with exact CPU gradients over the provided pairs.
///
/// Returns a [`LoraAdapter`] covering `q_proj` and `v_proj` in every GQA layer
/// in `[config.first_layer ..= 23]`. GDN layers in the same range are frozen
/// but their backward pass threads gradients through to lower GQA layers.
///
/// # Errors
///
/// Returns an error if:
/// - `pairs` is empty.
/// - Any pair has fewer than 2 tokens or exceeds `config.max_seq_len`.
/// - Any pair has `completion_start == 0` or `>= tokens.len()`.
/// - Any token ID is `>= vocab_size` (would panic on logit indexing).
/// - `config.rank == 0` or exceeds [`MAX_LORA_RANK`].
/// - `config.steps` exceeds [`MAX_TRAIN_STEPS`].
/// - `config.max_seq_len` exceeds [`MAX_TRAIN_SEQ_LEN`].
/// - `config.first_layer` exceeds the top trainable layer, or the model has
///   fewer layers than the trainer indexes.
/// - Buffer-size arithmetic overflows (rank × hidden, etc.).
/// - The model architecture contains unexpected layer types.
/// - The frozen-prefix capture fails.
///
/// # Implementation note
///
/// This function mirrors `train_grad_full`'s CPU forward/backward + Adam loop
/// (verified line-equivalent; intentionally duplicated because the binary's
/// private helpers are not importable from library code). A follow-up will
/// extract the shared tape into a library-private module.
pub fn train_micro_lora(
    model: &Qwen35Model,
    pairs: &[TrainingPair],
    config: &MicroLoraConfig,
) -> Result<LoraAdapter> {
    // Validate all caller-supplied inputs before any model access or allocation.
    // The extracted helper is separately testable without a real checkpoint.
    let vocab_size = model.config().vocab_size;
    let num_hidden_layers = model.config().num_hidden_layers;
    validate_micro_lora_inputs(vocab_size, num_hidden_layers, pairs, config)?;

    let cfg = model.config().clone();
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
    let rank = config.rank;
    let alpha = config.alpha;
    let scale = alpha / rank as f32;
    let first_layer = config.first_layer;

    // Head weights.
    let (lm_head_s, final_norm_s, _embed) = model.head_weights();
    let lm_head = lm_head_s.to_vec();
    let final_shift = shifted(final_norm_s);
    let head = Head {
        lm_head: &lm_head,
        final_shift: &final_shift,
    };

    // Build the materialised layer stack [first_layer ..= TOP_LAYER].
    let mut layers: Vec<LayerW> = Vec::new();
    let mut slot_layers: Vec<usize> = Vec::new();
    for layer_idx in first_layer..=TOP_LAYER {
        if let Some((w_q, w_k, w_v, w_o, q_norm, k_norm, pre, post, gate, up, down)) =
            model.gqa_layer_weights(layer_idx)
        {
            let slot = slot_layers.len();
            slot_layers.push(layer_idx);
            layers.push(LayerW {
                kind: MixerKind::Gqa,
                w_q,
                w_k,
                w_v,
                w_o,
                q_norm,
                k_norm,
                gdn: None,
                pre_shift: shifted(pre),
                post_shift: shifted(post),
                w_gate: gate,
                w_up: up,
                w_down: down,
                lora_slot: Some(slot),
            });
        } else if let Some((gdn, pre, post, gate, up, down)) = model.gdn_layer_weights(layer_idx) {
            layers.push(LayerW {
                kind: MixerKind::Gdn,
                w_q: &[],
                w_k: &[],
                w_v: &[],
                w_o: &[],
                q_norm: &[],
                k_norm: &[],
                gdn: Some(gdn),
                pre_shift: shifted(pre),
                post_shift: shifted(post),
                w_gate: gate,
                w_up: up,
                w_down: down,
                lora_slot: None,
            });
        } else {
            return Err(TuneError::Validation(format!(
                "layer {layer_idx} is neither a GQA nor GDN layer"
            )));
        }
    }
    let num_slots = slot_layers.len();
    if num_slots == 0 {
        return Err(TuneError::Validation(
            "no GQA layers in the configured range — nothing to train".to_string(),
        ));
    }

    // Capture the frozen-prefix output (h_in entering first_layer) and RoPE tables.
    // All pair bounds are already validated above before model access.
    let mut caches: Vec<SeqCtx> = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let (h_in, _) = model
            .capture_attn_io(&pair.tokens, first_layer)
            .map_err(|e| TuneError::Validation(e.to_string()))?;
        let seq_len = pair.tokens.len();
        let (cos, sin) = model
            .rope_cos_sin_tables(seq_len)
            .map_err(|e| TuneError::Validation(e.to_string()))?;
        caches.push(SeqCtx {
            h_in,
            cos,
            sin,
            tokens: pair.tokens.clone(),
            completion_start: pair.completion_start,
            seq_len,
        });
    }

    // Checked buffer sizes: verify each rank × dimension product before
    // allocating. These can only overflow with adversarial config values (rank
    // is already capped at MAX_LORA_RANK), but we guard explicitly so a future
    // caller cannot reach a capacity-overflow abort.
    let a_q_len = rank.checked_mul(dims.hidden).ok_or_else(|| {
        TuneError::Validation(format!(
            "rank({rank}) * hidden_size({}) overflows buffer size",
            dims.hidden
        ))
    })?;
    let b_q_len = (2usize)
        .checked_mul(dims.q_dim)
        .and_then(|n| n.checked_mul(rank))
        .ok_or_else(|| {
            TuneError::Validation(format!(
                "2 * q_dim({}) * rank({rank}) overflows buffer size",
                dims.q_dim
            ))
        })?;
    let a_v_len = rank.checked_mul(dims.hidden).ok_or_else(|| {
        TuneError::Validation(format!(
            "rank({rank}) * hidden_size({}) overflows buffer size",
            dims.hidden
        ))
    })?;
    let b_v_len = dims.kv_dim.checked_mul(rank).ok_or_else(|| {
        TuneError::Validation(format!(
            "kv_dim({}) * rank({rank}) overflows buffer size",
            dims.kv_dim
        ))
    })?;

    // Init LoRA: A ~ U(-1/sqrt(hidden), +1/sqrt(hidden)), B = 0.
    // Matches mlx_lm LoRALinear init for on-par convergence.
    let init_amp = 1.0 / (dims.hidden as f32).sqrt();
    let mut rng = 0xFEED_FACEu64;
    let mut loras: Vec<LoraParams> = (0..num_slots)
        .map(|_| LoraParams {
            a_q: rand_fill(&mut rng, a_q_len, init_amp),
            b_q: vec![0.0; b_q_len],
            a_v: rand_fill(&mut rng, a_v_len, init_amp),
            b_v: vec![0.0; b_v_len],
        })
        .collect();

    let mut adam = AdamState::new();
    let (beta1, beta2, eps_adam) = (0.9f32, 0.999f32, 1e-8f32);
    let lr = config.learning_rate;

    for step in 1..=config.steps {
        let ctx = &caches[(step - 1) % caches.len()];
        let fwd = forward_full(
            ctx, &layers, &loras, &head, &dims, &gdn_dims, &cfg, rank, scale,
        )?;
        let (_nll, _n, grads) =
            nll_and_grads(&fwd, &layers, &loras, &head, &dims, num_slots, rank, scale)?;

        for slot in 0..num_slots {
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

    // Assemble LoraAdapter from trained slot params.
    let mut adapter_layers = HashMap::new();
    for (s, &li) in slot_layers.iter().enumerate() {
        adapter_layers.insert(
            (li, "q_proj".to_string()),
            LoraLayer {
                a: loras[s].a_q.clone(),
                b: loras[s].b_q.clone(),
                d_in: dims.hidden,
                d_out: 2 * dims.q_dim,
                rank,
            },
        );
        adapter_layers.insert(
            (li, "v_proj".to_string()),
            LoraLayer {
                a: loras[s].a_v.clone(),
                b: loras[s].b_v.clone(),
                d_in: dims.hidden,
                d_out: dims.kv_dim,
                rank,
            },
        );
    }
    let lora_config = LoraConfig {
        rank,
        alpha,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
    };
    Ok(LoraAdapter::new(lora_config, adapter_layers))
}

// ─── Private types duplicated from train_grad_full.rs ────────────────────────
// The binary's helpers are private and cannot be imported from library code, so
// the forward/backward/Adam paths are reproduced here verbatim (line-equivalent
// to train_grad_full.rs as of this commit). Extract to a shared module when that
// refactor lands.

#[derive(Clone, Copy)]
struct Dims {
    hidden: usize,
    vocab: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    inter: usize,
    q_dim: usize,
    kv_dim: usize,
    eps: f32,
}

#[derive(Clone, Copy)]
struct GdnDims {
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
    fn from_cfg(cfg: &Qwen35Config) -> Self {
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

enum MixerKind {
    Gqa,
    Gdn,
}

/// Borrowed frozen weights for one materialised layer (lifetime tied to `model`).
/// Norm `*_shift` fields hold `(1 + gamma)` shifted layer norms; `q_norm`/`k_norm`
/// stay raw (shifted internally by `gqa_forward_with_cache`).
struct LayerW<'a> {
    kind: MixerKind,
    w_q: &'a [f32],
    w_k: &'a [f32],
    w_v: &'a [f32],
    w_o: &'a [f32],
    q_norm: &'a [f32],
    k_norm: &'a [f32],
    gdn: Option<&'a GatedDeltaNetWeights>,
    pre_shift: Vec<f32>,
    post_shift: Vec<f32>,
    w_gate: &'a [f32],
    w_up: &'a [f32],
    w_down: &'a [f32],
    lora_slot: Option<usize>,
}

struct Head<'a> {
    lm_head: &'a [f32],
    final_shift: &'a [f32],
}

/// Mutable LoRA factors for one GQA layer (q_proj + v_proj).
#[derive(Clone)]
struct LoraParams {
    a_q: Vec<f32>,
    b_q: Vec<f32>,
    a_v: Vec<f32>,
    b_v: Vec<f32>,
}

impl LoraParams {
    /// Allocate zero-initialised gradient accumulators with checked arithmetic.
    ///
    /// By the time this is called, `rank` has already been validated against
    /// `MAX_LORA_RANK` in the public API. The checked products here guard against
    /// future callers bypassing that validation or pathological dimension values.
    fn zeros(rank: usize, d: &Dims) -> Result<Self> {
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

/// LoRA gradient accumulators (same shape as [`LoraParams`]).
type Grads = LoraParams;

/// One sample's frozen context entering `first_layer`.
struct SeqCtx {
    h_in: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    tokens: Vec<u32>,
    completion_start: usize,
    seq_len: usize,
}

enum MixerCache {
    Gqa(Box<AttnCache>),
    Gdn(Box<GdnSaved>),
}

/// Saved forward activations for one materialised layer.
struct LayerFwd {
    h_layer_in: Vec<f32>,
    inv_pre: Vec<f32>,
    mixer: MixerCache,
    h_mid: Vec<f32>,
    inv_ffn: Vec<f32>,
    gate_pre: Vec<Vec<f32>>,
    up_pre: Vec<Vec<f32>>,
}

/// Intermediates at one completion position (for head backward).
struct HeadPos {
    t: usize,
    target: u32,
    logits: Vec<f32>,
    inv_final: f32,
}

struct FullFwd {
    layers: Vec<LayerFwd>,
    h_final: Vec<f32>,
    positions: Vec<HeadPos>,
}

// ─── Private helper functions ─────────────────────────────────────────────────

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

/// Shifted RMSNorm over a full sequence; returns `(normed [seq*hidden], inv [seq])`.
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

#[allow(clippy::too_many_arguments)]
fn forward_full(
    ctx: &SeqCtx,
    layers: &[LayerW],
    loras: &[LoraParams],
    head: &Head,
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

/// CE NLL for one position's logits against `target` (numerically stable).
fn position_nll(logits: &[f32], target: u32) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    let log_prob = (logits[target as usize] - max) as f64 - sum_exp.ln();
    (-log_prob) as f32
}

/// Full-depth backward over one sample. Returns `(nll_sum, n_comp, per-slot grads)`.
#[allow(clippy::too_many_arguments)]
fn nll_and_grads(
    fwd: &FullFwd,
    layers: &[LayerW],
    loras: &[LoraParams],
    head: &Head,
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

    // Head + CE backward.
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

    // Layer backward, top → first.
    for li in (0..layers.len()).rev() {
        let lw = &layers[li];
        let lf = &fwd.layers[li];

        // FFN backward. `h_next = h_mid + ffn_out` → residual skip seeds d_h_mid.
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

        // Mixer backward → `d_normed_pre`. `h_mid = h_layer_in + mixer_out`.
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

        // Pre-norm backward. `h_mid = h_layer_in + mixer_out` → residual skip seeds d_h_new.
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

/// Map a raw `gamma` slice to shifted RMSNorm weights `(1 + gamma)`.
fn shifted(gamma: &[f32]) -> Vec<f32> {
    gamma.iter().map(|g| 1.0 + g).collect()
}

/// Xorshift fill of `n` values in `[-amp, amp]`.
fn rand_fill(rng: &mut u64, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            ((*rng >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_pair_fields() {
        let pair = TrainingPair {
            tokens: vec![1, 2, 3, 4],
            completion_start: 2,
        };
        assert_eq!(pair.tokens, vec![1, 2, 3, 4]);
        assert_eq!(pair.completion_start, 2);
    }

    #[test]
    fn test_micro_lora_config_defaults() {
        let cfg = MicroLoraConfig::default();
        assert_eq!(cfg.rank, 4);
        assert!((cfg.alpha - 8.0).abs() < 1e-6);
        assert_eq!(cfg.first_layer, 19);
        assert_eq!(cfg.steps, 25);
        assert!((cfg.learning_rate - 1e-3).abs() < 1e-8);
        assert_eq!(cfg.max_seq_len, 64);
    }

    // Tests that require a real model checkpoint are marked #[ignore].
    // Run with:
    //   cargo test -p lattice-tune --features train-backward -- train --ignored

    #[ignore]
    #[test]
    fn placeholder_requires_real_model() {
        // Intentionally empty: a real model checkpoint is required to exercise
        // train_micro_lora. Populate this when running integration tests locally.
    }

    // ─── FIX 4: validate_micro_lora_inputs — no checkpoint required ─────────────
    //
    // All tests below call `validate_micro_lora_inputs` directly so they can run
    // without a real Qwen35Model checkpoint. Each test pins one guard; reverting
    // that guard from train.rs causes the corresponding test to fail.

    fn pair(tokens: Vec<u32>, completion_start: usize) -> TrainingPair {
        TrainingPair {
            tokens,
            completion_start,
        }
    }

    fn default_cfg() -> MicroLoraConfig {
        MicroLoraConfig {
            rank: 4,
            alpha: 8.0,
            first_layer: 19,
            steps: 10,
            learning_rate: 1e-3,
            max_seq_len: 64,
        }
    }

    /// Mutation-sensitive: if the `pairs.is_empty()` guard is removed, this fails.
    #[test]
    fn validation_rejects_empty_pairs() {
        let err = validate_micro_lora_inputs(1000, 24, &[], &default_cfg())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("at least one"),
            "expected empty-pairs rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: if the `tokens.len() < 2` guard is removed, this fails.
    #[test]
    fn validation_rejects_pair_with_one_token() {
        let pairs = [pair(vec![1], 0)];
        let err = validate_micro_lora_inputs(1000, 24, &pairs, &default_cfg())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("minimum 2"),
            "expected short-tokens rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: if the `completion_start == 0` guard is removed, the
    /// range `(completion_start - 1)..seq_len - 1` underflows to a huge integer
    /// and panics in forward_full — so this test must keep the guard in place.
    #[test]
    fn validation_rejects_completion_start_zero() {
        // Two tokens required to pass the length check first.
        let pairs = [pair(vec![1, 2], 0)];
        let err = validate_micro_lora_inputs(1000, 24, &pairs, &default_cfg())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("completion_start is 0"),
            "expected start==0 rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: if `completion_start >= tokens.len()` guard is removed,
    /// `tokens[t + 1]` at `t = completion_start - 1 = len - 1` is out-of-bounds.
    #[test]
    fn validation_rejects_completion_start_at_len() {
        let pairs = [pair(vec![1, 2, 3], 3)]; // completion_start == tokens.len()
        let err = validate_micro_lora_inputs(1000, 24, &pairs, &default_cfg())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("completion_start") && err.contains(">="),
            "expected out-of-range start rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: if the `tok >= vocab_size` guard is removed, the
    /// forward pass panics at `logits[target as usize]`.
    #[test]
    fn validation_rejects_out_of_vocab_token() {
        // vocab_size = 100; token 200 is out of range.
        let pairs = [pair(vec![0, 200, 1], 1)];
        let err = validate_micro_lora_inputs(100, 24, &pairs, &default_cfg())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("out of vocabulary"),
            "expected out-of-vocab rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: if `rank == 0` guard is removed, buffer products yield
    /// zero-length Vecs and forward_full divides by zero in `scale = alpha / rank`.
    #[test]
    fn validation_rejects_rank_zero() {
        let pairs = [pair(vec![1, 2], 1)];
        let mut cfg = default_cfg();
        cfg.rank = 0;
        let err = validate_micro_lora_inputs(1000, 24, &pairs, &cfg)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("rank must be > 0"),
            "expected rank-zero rejection; got: {err}"
        );
    }

    /// Confirm that valid inputs pass without error.
    #[test]
    fn validation_accepts_valid_inputs() {
        let pairs = [pair(vec![1, 2, 3], 1)];
        validate_micro_lora_inputs(1000, 24, &pairs, &default_cfg()).unwrap();
    }

    /// Mutation-sensitive: if the `num_hidden_layers <= TOP_LAYER` guard is
    /// removed, a model with fewer than TOP_LAYER + 1 layers reaches a direct
    /// `model.weights.layers[idx]` index in the layer loop and panics. The guard
    /// must reject it as a Validation error instead.
    #[test]
    fn validation_rejects_model_with_too_few_layers() {
        let pairs = [pair(vec![1, 2, 3], 1)];
        // 12-layer model: TOP_LAYER (23) would be out of bounds.
        let err = validate_micro_lora_inputs(1000, 12, &pairs, &default_cfg())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("requires at least"),
            "expected too-few-layers rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: if the `first_layer > TOP_LAYER` guard is removed, an
    /// out-of-range first_layer yields an inverted/empty layer range.
    #[test]
    fn validation_rejects_first_layer_above_top() {
        let pairs = [pair(vec![1, 2, 3], 1)];
        let mut cfg = default_cfg();
        cfg.first_layer = 24; // > TOP_LAYER (23)
        let err = validate_micro_lora_inputs(1000, 48, &pairs, &cfg)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("first_layer") && err.contains("exceeds"),
            "expected first_layer rejection; got: {err}"
        );
    }

    /// A model with at least TOP_LAYER + 1 layers and an in-range first_layer
    /// passes the new layer-bound checks.
    #[test]
    fn validation_accepts_valid_layer_bounds() {
        let pairs = [pair(vec![1, 2, 3], 1)];
        // 48-layer model with default first_layer = 19.
        validate_micro_lora_inputs(1000, 48, &pairs, &default_cfg()).unwrap();
    }
}
