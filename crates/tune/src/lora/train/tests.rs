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

/// If the `pairs.is_empty()` guard is removed, this fails.
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

/// If the `tokens.len() < 2` guard is removed, this fails.
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

/// If the `completion_start == 0` guard is removed, the
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

/// If `completion_start >= tokens.len()` guard is removed,
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

/// If the `tok >= vocab_size` guard is removed, the
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

/// If `rank == 0` guard is removed, buffer products yield
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

/// A model shorter than `first_layer` must be rejected
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

/// If the derived-top guard is removed, an out-of-range
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

/// Without the `num_hidden_layers != loaded_layer_count`
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

/// If `MicroLoraConfig::default()`'s `last_layer` is
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

/// A bounded trainable window (`last_layer =
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
