//! CPU micro-LoRA training over a bounded Qwen3.5 layer suffix.
//!
//! The public helper trains GQA `q_proj` and `v_proj` weights with exact
//! gradients; GatedDeltaNet layers remain frozen while preserving gradient flow.
//! It validates all caller-controlled shape, range, and sequence bounds first.
//!
//! See `docs/lora-core.md` for the tape, loop, and configuration details.

use std::collections::HashMap;
use std::ops::RangeInclusive;

use lattice_inference::model::qwen35::Qwen35Model;

use crate::error::{Result, TuneError};
use crate::lora::train_core::{
    AdamConfig, Dims, GdnDims, GdnLoraParams, GdnModuleSelection, Head, LayerW, LoraParams,
    MixerKind, SeqCtx, SlotLayout, TapeGeometry, TrainCtx, apply_adam_updates,
    apply_gdn_adam_updates, forward_full, nll_and_grads, rand_fill, shifted,
};
use crate::lora::{AdamState, LoraAdapter, LoraConfig, LoraLayer};

/// Upper bound on LoRA rank accepted by [`train_micro_lora`].
/// Limits caller-controlled buffer-size products.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#train_micro_lora) for the safety-bound rationale.
const MAX_LORA_RANK: usize = 512;

/// Upper bound on training steps accepted by [`train_micro_lora`].
/// Limits unbounded caller-controlled work.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#train_micro_lora) for the safety-bound rationale.
const MAX_TRAIN_STEPS: usize = 100_000;

/// Upper bound on `max_seq_len` accepted by [`train_micro_lora`].
/// Limits tape allocation; the model context window remains authoritative.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#train_micro_lora) for the safety-bound rationale.
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
    /// Index of the last LoRA-adapted (trainable) layer, inclusive. Layers
    /// above this bound, through the model's true top layer, are still
    /// materialised and run frozen — the forward pass, loss, and gradients
    /// always cover the full deployed suffix from `first_layer` regardless
    /// of where the trainable window ends. `None` trains through the
    /// model's true top layer, leaving no frozen suffix above the trained
    /// window.
    pub last_layer: Option<usize>,
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
            last_layer: None,
            steps: 25,
            learning_rate: 1e-3,
            max_seq_len: 64,
        }
    }
}

/// Derive the trainable layer range, validating `first_layer`/`last_layer`
/// against the model's true top layer and, first, `num_hidden_layers`
/// (the config-declared layer count) against `loaded_layer_count` (the
/// weight vectors actually present). [`Qwen35Model::config_mut`] lets a
/// caller change `num_hidden_layers` independently of the loaded weights;
/// without this check a desynchronized config can derive a range that
/// indexes past the loaded layer-weight vectors and panics instead of
/// returning an error.
fn trainable_layer_range(
    num_hidden_layers: usize,
    loaded_layer_count: usize,
    first_layer: usize,
    last_layer: Option<usize>,
) -> Result<RangeInclusive<usize>> {
    if num_hidden_layers != loaded_layer_count {
        return Err(TuneError::Validation(format!(
            "train_micro_lora: config.num_hidden_layers ({num_hidden_layers}) does not \
             match the model's loaded layer count ({loaded_layer_count}) — configuration \
             and weights are desynchronized"
        )));
    }
    let top_layer = num_hidden_layers.checked_sub(1).ok_or_else(|| {
        TuneError::Validation("train_micro_lora: model has no hidden layers".to_string())
    })?;
    if first_layer > top_layer {
        return Err(TuneError::Validation(format!(
            "train_micro_lora: first_layer {first_layer} exceeds model top layer {top_layer}"
        )));
    }
    let end_layer = match last_layer {
        Some(last) => {
            if last < first_layer {
                return Err(TuneError::Validation(format!(
                    "train_micro_lora: last_layer {last} is before first_layer {first_layer}"
                )));
            }
            if last > top_layer {
                return Err(TuneError::Validation(format!(
                    "train_micro_lora: last_layer {last} exceeds model top layer {top_layer}"
                )));
            }
            last
        }
        None => top_layer,
    };
    Ok(first_layer..=end_layer)
}

/// Layer indices the tape must materialise, always ending at `top_layer`
/// regardless of where the trainable (LoRA-adapted) window ends. Bounding
/// this by the trainable window's own end instead would silently drop the
/// frozen suffix above it from the forward pass, so the computed loss and
/// gradients would reflect a network shorter than the one that is deployed.
fn materialized_layer_range(first_layer: usize, top_layer: usize) -> RangeInclusive<usize> {
    first_layer..=top_layer
}

/// Validate `train_micro_lora` inputs before model access or allocation.
/// Returns the first [`TuneError::Validation`] failure.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#train_micro_lora) for the preflight boundary.
pub(crate) fn validate_micro_lora_inputs(
    vocab_size: usize,
    num_hidden_layers: usize,
    loaded_layer_count: usize,
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
    trainable_layer_range(
        num_hidden_layers,
        loaded_layer_count,
        config.first_layer,
        config.last_layer,
    )?;
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

/// Train GQA `q_proj` and `v_proj` LoRA weights with exact CPU gradients.
/// GDN layers in the selected suffix remain frozen but propagate gradients.
/// Returns validation errors for invalid data, incompatible models, or unsafe sizes.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#train_micro_lora) for the tape, limits, and implementation rationale.
pub fn train_micro_lora(
    model: &Qwen35Model,
    pairs: &[TrainingPair],
    config: &MicroLoraConfig,
) -> Result<LoraAdapter> {
    train_micro_lora_impl(model, pairs, config, false, GdnModuleSelection::default())
}

/// Train GQA `q_proj`/`v_proj` LoRA weights, and — when `train_gdn` is
/// `true` — also train the five GDN LoRA projections (`in_proj_qkv`,
/// `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`) on GatedDeltaNet
/// layers inside the trainable window (`config.first_layer..=last_layer`).
/// With `train_gdn: false` this reproduces [`train_micro_lora`]'s
/// GQA-only, GDN-frozen behavior exactly.
///
/// A separate function rather than a new [`MicroLoraConfig`] field:
/// `MicroLoraConfig` is publicly, exhaustively constructible and
/// `cargo-semver-checks`' `constructible_struct_adds_field` gate covers
/// `lattice-tune`, so adding a field to it would be a breaking change
/// without an available major-version bump this cycle (the same
/// constraint that kept `min_p` off `SamplingConfig`/`GenerateConfig` in
/// `lattice-inference`).
/// See [`docs/lora-core.md`](../../docs/lora-core.md#train_micro_lora) for the tape, limits, and implementation rationale.
pub fn train_micro_lora_with_gdn(
    model: &Qwen35Model,
    pairs: &[TrainingPair],
    config: &MicroLoraConfig,
    train_gdn: bool,
) -> Result<LoraAdapter> {
    train_micro_lora_impl(
        model,
        pairs,
        config,
        train_gdn,
        GdnModuleSelection::default(),
    )
}

/// As [`train_micro_lora_with_gdn`], with the GDN module set chosen explicitly.
///
/// The default is [`GdnModuleSelection::Served`], and this is the way to ask for
/// [`GdnModuleSelection::All`]: five modules train, two of which the Metal
/// forward refuses to load. Legitimate on the CPU path, which applies every
/// projection, and required by the mutation control that proves the default set
/// is doing something.
///
/// A sibling function rather than a fifth parameter on
/// [`train_micro_lora_with_gdn`], for the same reason `train_gdn` itself is a
/// sibling of [`train_micro_lora`]: changing an exported signature is a breaking
/// change that `cargo-semver-checks` gates, and there is no major-version bump
/// available this cycle.
pub fn train_micro_lora_with_gdn_modules(
    model: &Qwen35Model,
    pairs: &[TrainingPair],
    config: &MicroLoraConfig,
    selection: GdnModuleSelection,
) -> Result<LoraAdapter> {
    train_micro_lora_impl(model, pairs, config, true, selection)
}

fn train_micro_lora_impl(
    model: &Qwen35Model,
    pairs: &[TrainingPair],
    config: &MicroLoraConfig,
    train_gdn: bool,
    gdn_modules: GdnModuleSelection,
) -> Result<LoraAdapter> {
    // Validate caller-controlled bounds before model access or allocation.
    let vocab_size = model.config().vocab_size;
    let num_hidden_layers = model.config().num_hidden_layers;
    let loaded_layer_count = model.loaded_layer_count();
    validate_micro_lora_inputs(
        vocab_size,
        num_hidden_layers,
        loaded_layer_count,
        pairs,
        config,
    )?;

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
    let first_layer = config.first_layer;

    // Head weights.
    let (lm_head_s, final_norm_s, _embed) = model.head_weights();
    let lm_head = lm_head_s.to_vec();
    let final_shift = shifted(final_norm_s);
    let head = Head {
        lm_head: &lm_head,
        final_shift: &final_shift,
    };

    // The trainable (LoRA-adapted) window may end before the model's true
    // top layer; the materialised stack always extends through the true top
    // so the forward pass matches the deployed full-depth model. Layers
    // above the trainable window are still materialised and run through
    // the tape, but frozen (no LoRA slot).
    let trainable_range = trainable_layer_range(
        num_hidden_layers,
        loaded_layer_count,
        first_layer,
        config.last_layer,
    )?;
    let trainable_last = *trainable_range.end();
    let top_layer = num_hidden_layers - 1;
    let mut layers: Vec<LayerW> = Vec::new();
    let mut slot_layers: Vec<usize> = Vec::new();
    let mut gdn_slot_layers: Vec<usize> = Vec::new();
    for layer_idx in materialized_layer_range(first_layer, top_layer) {
        let trainable = layer_idx <= trainable_last;
        if let Some((w_q, w_k, w_v, w_o, q_norm, k_norm, pre, post, gate, up, down)) =
            model.gqa_layer_weights(layer_idx)
        {
            let lora_slot = if trainable {
                let slot = slot_layers.len();
                slot_layers.push(layer_idx);
                Some(slot)
            } else {
                None
            };
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
                lora_slot,
            });
        } else if let Some((gdn, pre, post, gate, up, down)) = model.gdn_layer_weights(layer_idx) {
            let lora_slot = if train_gdn && trainable {
                let slot = gdn_slot_layers.len();
                gdn_slot_layers.push(layer_idx);
                Some(slot)
            } else {
                None
            };
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
                lora_slot,
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
    let num_gdn_slots = gdn_slot_layers.len();

    // Capture frozen-prefix state and RoPE tables after input validation.
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

    // Check every allocation product before constructing LoRA buffers.
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

    // Initialize A uniformly and B to zero; the initial adapter is a no-op.
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

    // Initialize GDN A uniformly and B to zero (same rationale as the GQA
    // loras above: the initial adapter is a no-op), continuing the same rng
    // stream position rather than reseeding it, matching train_grad_full's
    // zero_b_gdn_loras call convention.
    let mut gdn_loras: Vec<GdnLoraParams> = (0..num_gdn_slots)
        .map(|_| {
            GdnLoraParams::shaped(
                rank,
                dims.hidden,
                &gdn_dims,
                |n| rand_fill(&mut rng, n, init_amp),
                |n| vec![0.0; n],
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let mut adam = AdamState::new();
    let (beta1, beta2, eps_adam) = (0.9f32, 0.999f32, 1e-8f32);
    let lr = config.learning_rate;

    let train_ctx = TrainCtx::try_new(
        TapeGeometry::new(&dims, &gdn_dims, &cfg),
        rank,
        alpha,
        SlotLayout::new(&slot_layers, &gdn_slot_layers),
        AdamConfig::new(lr, beta1, beta2, eps_adam),
    )?;
    for step in 1..=config.steps {
        let ctx = &caches[(step - 1) % caches.len()];
        let fwd = forward_full(ctx, &layers, &loras, &gdn_loras, &head, &train_ctx)?;
        let (_nll, _n, grads, gdn_grads) = nll_and_grads(&fwd, &layers, &loras, &head, &train_ctx)?;

        apply_adam_updates(&mut adam, &mut loras, &grads, &train_ctx);
        apply_gdn_adam_updates(
            &mut adam,
            &mut gdn_loras,
            &gdn_grads,
            &train_ctx,
            gdn_modules,
        );
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
    let mut target_modules = vec!["q_proj".to_string(), "v_proj".to_string()];
    if num_gdn_slots > 0 {
        // From the selection, never a literal: a hand-written list here would be a
        // second copy of the servable-module set and would drift from the one the
        // Metal loader refuses by.
        target_modules.extend(gdn_modules.modules().iter().map(|m| (*m).to_string()));
        for (s, &li) in gdn_slot_layers.iter().enumerate() {
            let g = &gdn_loras[s];
            adapter_layers.insert(
                (li, "in_proj_qkv".to_string()),
                LoraLayer {
                    a: g.a_qkv.clone(),
                    b: g.b_qkv.clone(),
                    d_in: dims.hidden,
                    d_out: gdn_dims.qkv_dim,
                    rank,
                },
            );
            adapter_layers.insert(
                (li, "in_proj_z".to_string()),
                LoraLayer {
                    a: g.a_z.clone(),
                    b: g.b_z.clone(),
                    d_in: dims.hidden,
                    d_out: gdn_dims.output_dim,
                    rank,
                },
            );
            // Written only when they were TRAINED. Under the default they were
            // never stepped, so `b_b`/`b_a` are still exactly zero and the layer
            // would be an identity the loader refuses anyway.
            if gdn_modules.trains_fused_kernel_projections() {
                adapter_layers.insert(
                    (li, "in_proj_b".to_string()),
                    LoraLayer {
                        a: g.a_b.clone(),
                        b: g.b_b.clone(),
                        d_in: dims.hidden,
                        // beta is projected per VALUE head (matches the shipping
                        // gdn_fused forward and the f16 weight loader), not per
                        // key head (#792).
                        d_out: gdn_dims.value_heads,
                        rank,
                    },
                );
                adapter_layers.insert(
                    (li, "in_proj_a".to_string()),
                    LoraLayer {
                        a: g.a_a.clone(),
                        b: g.b_a.clone(),
                        d_in: dims.hidden,
                        // alpha is likewise projected per VALUE head.
                        d_out: gdn_dims.value_heads,
                        rank,
                    },
                );
            }
            adapter_layers.insert(
                (li, "out_proj".to_string()),
                LoraLayer {
                    a: g.a_out.clone(),
                    b: g.b_out.clone(),
                    d_in: gdn_dims.output_dim,
                    d_out: dims.hidden,
                    rank,
                },
            );
        }
    }
    let lora_config = LoraConfig {
        rank,
        alpha,
        target_modules,
        dtype: "f32".into(),
    };
    LoraAdapter::new(lora_config, adapter_layers)
}

#[cfg(test)]
mod tests;
