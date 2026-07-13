//! CPU micro-LoRA training over a bounded Qwen3.5 layer suffix.
//!
//! The public helper trains GQA `q_proj` and `v_proj` weights with exact
//! gradients; GatedDeltaNet layers remain frozen while preserving gradient flow.
//! It validates all caller-controlled shape, range, and sequence bounds first.
//!
//! See `docs/lora-core.md` for the tape, loop, and configuration details.

use std::collections::HashMap;

use lattice_inference::model::qwen35::Qwen35Model;

use crate::error::{Result, TuneError};
use crate::lora::train_core::{
    AdamConfig, Dims, GdnDims, GdnLoraParams, Head, LayerW, LoraParams, MixerKind, SeqCtx,
    SlotLayout, TOP_LAYER, TapeGeometry, TrainCtx, apply_adam_updates, forward_full, nll_and_grads,
    rand_fill, shifted,
};
use crate::lora::{AdamState, LoraAdapter, LoraConfig, LoraLayer};

/// Upper bound on LoRA rank accepted by [`train_micro_lora`].
/// Limits caller-controlled buffer-size products.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#train_micro_lora) (§train_micro_lora) for the safety-bound rationale.
const MAX_LORA_RANK: usize = 512;

/// Upper bound on training steps accepted by [`train_micro_lora`].
/// Limits unbounded caller-controlled work.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#train_micro_lora) (§train_micro_lora) for the safety-bound rationale.
const MAX_TRAIN_STEPS: usize = 100_000;

/// Upper bound on `max_seq_len` accepted by [`train_micro_lora`].
/// Limits tape allocation; the model context window remains authoritative.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#train_micro_lora) (§train_micro_lora) for the safety-bound rationale.
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

/// Validate `train_micro_lora` inputs before model access or allocation.
/// Returns the first [`TuneError::Validation`] failure.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#train_micro_lora) (§train_micro_lora) for the preflight boundary.
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
    // Keep the materialized inclusive range valid before direct layer access.
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

/// Train GQA `q_proj` and `v_proj` LoRA weights with exact CPU gradients.
/// GDN layers in the selected suffix remain frozen but propagate gradients.
/// Returns validation errors for invalid data, incompatible models, or unsafe sizes.
/// See [`docs/lora-core.md`](../../docs/lora-core.md#train_micro_lora) (§train_micro_lora) for the tape, limits, and implementation rationale.
pub fn train_micro_lora(
    model: &Qwen35Model,
    pairs: &[TrainingPair],
    config: &MicroLoraConfig,
) -> Result<LoraAdapter> {
    // Validate caller-controlled bounds before model access or allocation.
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
