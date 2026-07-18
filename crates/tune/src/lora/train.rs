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
    AdamConfig, Dims, GdnDims, GdnLoraParams, Head, LayerW, LoraParams, MixerKind, SeqCtx,
    SlotLayout, TapeGeometry, TrainCtx, apply_adam_updates, forward_full, nll_and_grads, rand_fill,
    shifted,
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

    let mut adam = AdamState::new();
    let (beta1, beta2, eps_adam) = (0.9f32, 0.999f32, 1e-8f32);
    let lr = config.learning_rate;

    // This public entry point trains GQA slots only; GDN stays on the frozen path.
    let gdn_loras: Vec<GdnLoraParams> = Vec::new();
    let gdn_slot_layers: Vec<usize> = Vec::new();
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
        let (_nll, _n, grads, _gdn_grads) =
            nll_and_grads(&fwd, &layers, &loras, &head, &train_ctx)?;

        apply_adam_updates(&mut adam, &mut loras, &grads, &train_ctx);
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
    LoraAdapter::new(lora_config, adapter_layers)
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
        assert_eq!(cfg.last_layer, None);
        assert_eq!(cfg.steps, 25);
        assert!((cfg.learning_rate - 1e-3).abs() < 1e-8);
        assert_eq!(cfg.max_seq_len, 64);
    }

    // Tests that require a real model checkpoint are marked #[ignore].
    // Run with:
    //   cargo test -p lattice-tune --features train-backward -- train --ignored

    #[ignore = "requires a real model checkpoint; placeholder/documentation-only test"]
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
            last_layer: Some(23),
            steps: 10,
            learning_rate: 1e-3,
            max_seq_len: 64,
        }
    }

    /// Mutation-sensitive: if the `pairs.is_empty()` guard is removed, this fails.
    #[test]
    fn validation_rejects_empty_pairs() {
        let err = validate_micro_lora_inputs(1000, 24, 24, &[], &default_cfg())
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
        let err = validate_micro_lora_inputs(1000, 24, 24, &pairs, &default_cfg())
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
        let err = validate_micro_lora_inputs(1000, 24, 24, &pairs, &default_cfg())
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
        let err = validate_micro_lora_inputs(1000, 24, 24, &pairs, &default_cfg())
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
        let err = validate_micro_lora_inputs(100, 24, 24, &pairs, &default_cfg())
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
        let err = validate_micro_lora_inputs(1000, 24, 24, &pairs, &cfg)
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
        validate_micro_lora_inputs(1000, 24, 24, &pairs, &default_cfg()).unwrap();
    }

    #[test]
    fn trainable_range_uses_true_top_layer_when_last_layer_unset() {
        // `last_layer: None` explicitly opts into the full-suffix workload —
        // the range still extends to the model's true top layer.
        assert_eq!(
            trainable_layer_range(24, 24, 19, None)
                .unwrap()
                .collect::<Vec<_>>(),
            vec![19, 20, 21, 22, 23]
        );
        assert_eq!(
            trainable_layer_range(40, 40, 39, None)
                .unwrap()
                .collect::<Vec<_>>(),
            vec![39]
        );
        assert_eq!(
            trainable_layer_range(64, 64, 60, None)
                .unwrap()
                .collect::<Vec<_>>(),
            vec![60, 61, 62, 63]
        );
    }

    /// Mutation-sensitive: a model shorter than `first_layer` must be rejected
    /// before the materialized layer loop performs direct model access.
    #[test]
    fn validation_rejects_model_with_too_few_layers() {
        let pairs = [pair(vec![1, 2, 3], 1)];
        let err = validate_micro_lora_inputs(1000, 12, 12, &pairs, &default_cfg())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("first_layer") && err.contains("exceeds"),
            "expected too-few-layers rejection; got: {err}"
        );
    }

    /// Mutation-sensitive: if the derived-top guard is removed, an out-of-range
    /// first layer yields an inverted or empty layer range.
    #[test]
    fn validation_rejects_first_layer_above_top() {
        let pairs = [pair(vec![1, 2, 3], 1)];
        let mut cfg = default_cfg();
        cfg.first_layer = 40;
        cfg.last_layer = None;
        let err = validate_micro_lora_inputs(1000, 40, 40, &pairs, &cfg)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("first_layer") && err.contains("exceeds"),
            "expected first_layer rejection; got: {err}"
        );
    }

    /// A larger model with an in-range first layer passes the derived bounds.
    #[test]
    fn validation_accepts_valid_layer_bounds() {
        let pairs = [pair(vec![1, 2, 3], 1)];
        validate_micro_lora_inputs(1000, 48, 48, &pairs, &default_cfg()).unwrap();
    }

    // ─── REQUIRED FIX 1: config-vs-loaded-weights desync — panic prevention ────

    /// Mutation-sensitive: without the `num_hidden_layers != loaded_layer_count`
    /// guard, a config mutated (e.g. via `Qwen35Model::config_mut`) to claim more
    /// layers than are actually loaded derives a range that runs past the loaded
    /// layer-weight vectors, and `train_micro_lora`'s materialization loop panics
    /// indexing `weights.layers[layer_idx]` instead of returning an error.
    #[test]
    fn validation_rejects_desynchronized_config_vs_loaded_weights() {
        let pairs = [pair(vec![1, 2, 3], 1)];
        // config.num_hidden_layers (40) claims more layers than are actually
        // loaded (24) — e.g. after `Qwen35Model::config_mut().num_hidden_layers = 40`.
        let err = validate_micro_lora_inputs(1000, 40, 24, &pairs, &default_cfg())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("desynchronized"),
            "expected config-vs-weights desync rejection; got: {err}"
        );
    }

    #[test]
    fn trainable_range_rejects_desynchronized_config_vs_loaded_weights() {
        let err = trainable_layer_range(40, 24, 19, Some(23))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("desynchronized"),
            "expected config-vs-weights desync rejection; got: {err}"
        );
    }

    /// A config that matches the loaded weights passes the new guard.
    #[test]
    fn validation_accepts_synchronized_config_and_loaded_weights() {
        let pairs = [pair(vec![1, 2, 3], 1)];
        validate_micro_lora_inputs(1000, 24, 24, &pairs, &default_cfg()).unwrap();
    }

    // ─── default layer range matches the model's true top suffix ──────────────

    /// Mutation-sensitive: if `MicroLoraConfig::default()`'s `last_layer` is
    /// pinned back to `Some(23)`, a model deeper than 24 layers trains only
    /// `19..=23` and then applies the LM head immediately — silently omitting
    /// every layer after 23 and optimizing a computation different from what
    /// gets deployed. The default must derive the trained suffix from the
    /// loaded depth instead.
    #[test]
    fn default_layer_range_follows_true_top_suffix_on_deeper_model() {
        let cfg = MicroLoraConfig::default();
        let range = trainable_layer_range(40, 40, cfg.first_layer, cfg.last_layer).unwrap();
        assert_eq!(
            range.collect::<Vec<_>>(),
            (19..=39).collect::<Vec<_>>(),
            "default range must follow the model's true top layer, not stay pinned to 23"
        );
        // On the original 24-layer checkpoint the suffix is unchanged.
        let range24 = trainable_layer_range(24, 24, cfg.first_layer, cfg.last_layer).unwrap();
        assert_eq!(range24.collect::<Vec<_>>(), vec![19, 20, 21, 22, 23]);
    }

    /// Mutation-sensitive: a bounded trainable window (`last_layer =
    /// Some(23)`) must not truncate the materialised forward suffix on a
    /// model deeper than 24 layers. `train_micro_lora` uses
    /// `materialized_layer_range`, not `trainable_layer_range`, to build its
    /// `layers` tape — the trainable window only decides which materialised
    /// GQA layers get a LoRA slot. If the materialisation loop is reverted to
    /// stop at `trainable_last` (the pre-fix behavior), this range would end
    /// at 23 instead of 39, and every frozen layer above 23 would be missing
    /// from the tape.
    #[test]
    fn materialized_layer_range_reaches_true_top_past_a_bounded_trainable_window() {
        let materialized: Vec<usize> = materialized_layer_range(19, 39).collect();
        assert_eq!(materialized.first(), Some(&19));
        assert_eq!(
            materialized.last(),
            Some(&39),
            "materialization must reach the model's true top layer (39), \
             not stop at a bounded trainable window's end (23)"
        );
        assert_eq!(materialized.len(), 21);

        // The trainable window itself is still correctly bounded — only
        // layers up to 23 are eligible for a LoRA slot.
        let trainable = trainable_layer_range(40, 40, 19, Some(23)).unwrap();
        assert_eq!(*trainable.end(), 23);
    }
}
