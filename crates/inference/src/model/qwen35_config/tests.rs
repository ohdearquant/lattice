use super::*;
use crate::generation::GenerateConfig;
use crate::model::config_file::MAX_CONFIG_JSON_BYTES;

#[test]
fn test_config_construction_and_layer_types() {
    let cfg = Qwen35Config::qwen35_2b();

    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.vocab_size, 248_320);
    assert_eq!(cfg.layer_types.len(), 24);

    // Check the pattern: [lin, lin, lin, full] x 6
    assert_eq!(cfg.num_full_attention_layers(), 6);
    assert_eq!(cfg.num_linear_attention_layers(), 18);

    // Full attention at indices 3, 7, 11, 15, 19, 23
    let full_indices: Vec<usize> = cfg
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == LayerType::FullAttention)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(full_indices, vec![3, 7, 11, 15, 19, 23]);

    // Linear attention at all other indices
    for i in [
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
    ] {
        assert_eq!(cfg.layer_types[i], LayerType::LinearAttention);
    }
}

#[test]
fn test_generate_config_defaults() {
    let gen_cfg = GenerateConfig::default();
    assert_eq!(gen_cfg.max_new_tokens, 256);
    assert!((gen_cfg.temperature - 0.7).abs() < 1e-6);
    assert_eq!(gen_cfg.top_k, 50);
    assert!((gen_cfg.top_p - 0.9).abs() < 1e-6);
    assert!((gen_cfg.repetition_penalty - 1.1).abs() < 1e-6);
    assert!(
        gen_cfg.stop_token_ids.contains(&QWEN_CHAT_IM_END_TOKEN_ID),
        "default stop tokens must include im_end"
    );
}

#[test]
fn test_dimension_helpers() {
    let cfg = Qwen35Config::qwen35_2b();

    // Full attention dims
    assert_eq!(cfg.full_q_dim(), 8 * 256); // 2048
    assert_eq!(cfg.full_kv_dim(), 2 * 256); // 512
    assert_eq!(cfg.rope_dim(), 64); // 0.25 * 256

    // Linear attention dims (Qwen3.5-2B: 16 value heads)
    // Q: 16*128=2048, K: 16*128=2048, V: 16*128=2048 → total 6144
    assert_eq!(cfg.linear_qkv_dim(), 6144);
    assert_eq!(cfg.linear_output_dim(), 2048); // 16 * 128
}

#[test]
fn test_is_full_attention() {
    let cfg = Qwen35Config::qwen35_2b();
    assert!(!cfg.is_full_attention(0));
    assert!(!cfg.is_full_attention(1));
    assert!(!cfg.is_full_attention(2));
    assert!(cfg.is_full_attention(3));
    assert!(!cfg.is_full_attention(4));
    assert!(cfg.is_full_attention(7));
    assert!(cfg.is_full_attention(23));
    // Out of bounds returns false (linear)
    assert!(!cfg.is_full_attention(100));
}

#[test]
fn test_qwen36_hf_config_fixture_parse_fields() {
    let json = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen36_config.json"
    ));

    let cfg = Qwen35Config::from_config_json_str(json).expect("Qwen3.6 HF config parses");

    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.linear_num_value_heads(), 32);
    assert_eq!(cfg.num_hidden_layers, 40);
    assert_eq!(cfg.num_experts, Some(256));
    assert_eq!(cfg.num_experts_per_tok, Some(8));
    assert_eq!(cfg.moe_intermediate_size, Some(512));
    assert_eq!(cfg.shared_expert_intermediate_size, Some(512));
    assert!(!cfg.tie_word_embeddings);
    assert!(cfg.is_moe());

    // 12 additional field assertions.
    assert_eq!(cfg.vocab_size, 248320);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.intermediate_size, 6144);
    assert_eq!(cfg.linear_num_key_heads, 16);
    assert_eq!(cfg.linear_key_head_dim, 128);
    assert_eq!(cfg.linear_value_head_dim, 128);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert_eq!(cfg.full_attention_interval, 4);
    assert_eq!(cfg.eos_token_id, 248044_u32);
    assert_eq!(cfg.max_position_embeddings, 262144);
    assert_eq!(cfg.rope_theta, 10_000_000.0_f64);
    assert_eq!(cfg.partial_rotary_factor, 0.25_f32);

    let full_indices: Vec<usize> = cfg
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == LayerType::FullAttention)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(full_indices, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
}

#[test]
fn test_qwen36_27b_preset_dimensions() {
    let cfg = Qwen35Config::qwen36_27b();
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_hidden_layers, 64);
    assert_eq!(cfg.vocab_size, 248_320);
    assert_eq!(cfg.intermediate_size, 17408);
    assert_eq!(cfg.num_attention_heads, 24);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.head_dim, 256);
    assert!((cfg.rope_theta - 10_000_000.0_f64).abs() < 1.0);
    assert!((cfg.partial_rotary_factor - 0.25_f32).abs() < 1e-6);
    assert_eq!(cfg.eos_token_id, 248_044_u32);
    assert_eq!(cfg.max_position_embeddings, 262_144);
}

#[test]
fn test_qwen36_27b_preset_layer_types() {
    let cfg = Qwen35Config::qwen36_27b();
    assert_eq!(cfg.layer_types.len(), 64);
    assert_eq!(cfg.full_attention_interval, 4);
    assert_eq!(
        cfg.layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count(),
        48
    );
    assert_eq!(
        cfg.layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count(),
        16
    );
    // Every 4th layer (1-indexed) is full attention: indices 3, 7, 11, ..., 63
    for i in 0..64_usize {
        let expected = (i + 1) % 4 == 0;
        assert_eq!(
            cfg.layer_types[i] == LayerType::FullAttention,
            expected,
            "layer {i} type mismatch"
        );
    }
}

#[test]
fn test_qwen36_27b_preset_not_moe() {
    let cfg = Qwen35Config::qwen36_27b();
    assert!(!cfg.is_moe());
    assert!(cfg.num_experts.is_none());
    assert!(cfg.num_experts_per_tok.is_none());
    assert!(cfg.moe_intermediate_size.is_none());
    assert!(cfg.shared_expert_intermediate_size.is_none());
    assert!(cfg.router_aux_loss_coef.is_none());
    assert!(!cfg.output_router_logits);
}

#[test]
fn test_qwen36_27b_preset_gdn_fields() {
    let cfg = Qwen35Config::qwen36_27b();
    assert_eq!(cfg.linear_num_key_heads, 16);
    assert_eq!(cfg.linear_num_value_heads, Some(48));
    assert_eq!(cfg.linear_num_value_heads(), 48);
    assert_eq!(cfg.linear_key_head_dim, 128);
    assert_eq!(cfg.linear_value_head_dim, 128);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    // Untied embeddings — separate lm_head.weight
    assert!(!cfg.tie_word_embeddings);
    // MTP
    assert_eq!(cfg.mtp_num_hidden_layers, 1);
    assert!(!cfg.mtp_use_dedicated_embeddings);
}

#[test]
fn test_qwen36_27b_from_config_json() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let path = std::path::PathBuf::from(format!("{home}/.lattice/models/qwen3.6-27b/config.json"));
    if !path.exists() {
        return; // model not downloaded; skip
    }
    let cfg = Qwen35Config::from_config_json(&path).expect("27B config.json parses");
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_hidden_layers, 64);
    assert_eq!(cfg.vocab_size, 248_320);
    assert_eq!(cfg.intermediate_size, 17408);
    assert_eq!(cfg.num_attention_heads, 24);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.head_dim, 256);
    assert!(!cfg.tie_word_embeddings);
    assert_eq!(cfg.layer_types.len(), 64);
    assert!(!cfg.is_moe());
    assert_eq!(cfg.mtp_num_hidden_layers, 1);
    // rope_theta is nested under rope_parameters — from_config_json_str now extracts it.
    assert!(
        (cfg.rope_theta - 10_000_000.0_f64).abs() < 1.0,
        "rope_theta should be extracted from nested rope_parameters"
    );
}

// --- layer_mask tests ---

#[test]
fn test_layer_mask_default_all_true_27b() {
    let cfg = Qwen35Config::qwen36_27b();
    assert_eq!(cfg.layer_mask.len(), 64);
    assert!(cfg.layer_mask.iter().all(|&active| active));
    assert_eq!(cfg.num_active_layers(), 64);
    assert_eq!(cfg.num_active_linear_attention_layers(), 48);
    assert_eq!(cfg.num_active_full_attention_layers(), 16);
}

#[test]
fn test_num_active_layers_partial_mask() {
    let mut cfg = Qwen35Config::qwen36_27b();
    // Deactivate layer 0 (GDN), layer 3 (GQA), layer 4 (GDN).
    let mut mask = vec![true; 64];
    mask[0] = false;
    mask[3] = false;
    mask[4] = false;
    cfg.apply_layer_mask(mask);
    assert_eq!(cfg.num_active_layers(), 61);
    assert_eq!(cfg.num_active_linear_attention_layers(), 46);
    assert_eq!(cfg.num_active_full_attention_layers(), 15);
}

#[test]
#[should_panic(expected = "layer_mask length")]
fn test_apply_layer_mask_wrong_length_panics() {
    let mut cfg = Qwen35Config::qwen36_27b();
    cfg.apply_layer_mask(vec![true; 32]);
}

#[test]
fn test_pruned_config_preserves_fields() {
    let cfg = Qwen35Config::qwen36_27b();
    let mut mask = vec![true; 64];
    mask[5] = false;
    mask[10] = false;
    let pruned = cfg.pruned_config(mask.clone());
    assert_eq!(pruned.hidden_size, cfg.hidden_size);
    assert_eq!(pruned.num_hidden_layers, cfg.num_hidden_layers);
    assert_eq!(pruned.vocab_size, cfg.vocab_size);
    assert_eq!(pruned.layer_types, cfg.layer_types);
    assert_eq!(pruned.layer_mask, mask);
    assert_eq!(pruned.num_active_layers(), 62);
}

#[test]
#[should_panic(expected = "at least one active layer")]
fn test_apply_layer_mask_all_false_panics() {
    let mut cfg = Qwen35Config::qwen36_27b();
    cfg.apply_layer_mask(vec![false; 64]);
}

#[test]
fn test_layer_mask_normalizes_on_parse() {
    let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "full_attention_interval": 4,
                "eos_token_id": 1
            }
        }"#;
    let cfg = Qwen35Config::from_config_json_str(json).unwrap();
    assert_eq!(
        cfg.layer_mask.len(),
        4,
        "normalize_layer_mask must fill mask to num_hidden_layers"
    );
    assert!(
        cfg.layer_mask.iter().all(|&v| v),
        "normalized mask must be all-true"
    );
}

#[test]
fn test_zero_full_attention_interval_errors_not_panics() {
    // A config.json with full_attention_interval: 0 must return a clean
    // InferenceError, never panic. layer_types is omitted, so the container
    // #[serde(default)] fills it from the preset (whose length differs from
    // num_hidden_layers: 4), forcing the recompute branch that calls
    // compute_layer_types(4, 0) -> (i + 1) % 0 -> divide-by-zero panic
    // without the guard.
    let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "full_attention_interval": 0,
                "eos_token_id": 1
            }
        }"#;
    let result = Qwen35Config::from_config_json_str(json);
    assert!(
        result.is_err(),
        "full_attention_interval: 0 must yield an InferenceError, not panic"
    );
}

#[test]
fn test_zero_num_key_value_heads_errors_not_panics() {
    // An explicit num_key_value_heads: 0 survives serde but reaches a divide-by-zero
    // (`num_q_heads / num_kv_heads`) and a hard `assert!(num_kv_heads > 0)` in the GQA
    // forward path. Reject at parse time. Omitted fields fall back to the valid preset.
    // The substring assert proves THIS guard fired (not an unrelated default-zero field):
    // `#[serde(default)]` + `Default = qwen36_35b_a3b()` means every omitted field carries a
    // valid preset value, so the one explicit bad field is the only one that can trip.
    let json = r#"{"text_config": {"num_key_value_heads": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("num_key_value_heads: 0 must yield an InferenceError, not a panic")
        .to_string();
    assert!(
        err.contains("num_key_value_heads"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_indivisible_head_counts_error_not_panics() {
    // num_attention_heads not divisible by num_key_value_heads truncates the GQA group
    // count and over-runs the KV row (OOB read) on the unasserted release path.
    let json = r#"{"text_config": {"num_attention_heads": 3, "num_key_value_heads": 2}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("indivisible head counts must yield an InferenceError, not OOB/panic")
        .to_string();
    assert!(err.contains("divisible"), "wrong guard fired: {err}");
}

#[test]
fn test_zero_head_dim_errors() {
    let json = r#"{"text_config": {"head_dim": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("head_dim: 0 must yield an InferenceError")
        .to_string();
    assert!(err.contains("head_dim"), "wrong guard fired: {err}");
}

#[test]
fn test_zero_num_hidden_layers_errors() {
    let json = r#"{"text_config": {"num_hidden_layers": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("num_hidden_layers: 0 must yield an InferenceError")
        .to_string();
    assert!(
        err.contains("num_hidden_layers"),
        "wrong guard fired: {err}"
    );
}

/// A checkpoint-controlled `num_hidden_layers` far past any real Qwen3.5 depth must be
/// rejected with a typed error at config-parse time, before `compute_layer_types` /
/// `normalize_layer_mask` (both of which allocate a `Vec` proportional to
/// `num_hidden_layers`) ever run. Uses `usize::MAX` -- the most extreme value a
/// malicious `config.json` could carry -- to demonstrate the guard fires before any
/// layer-proportional allocation is attempted.
#[test]
fn test_extreme_num_hidden_layers_rejected_before_allocation() {
    let json = format!(
        r#"{{"text_config": {{"num_hidden_layers": {}}}}}"#,
        usize::MAX
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("num_hidden_layers: usize::MAX must yield an InferenceError, not a panic/OOM")
        .to_string();
    assert!(
        err.contains("num_hidden_layers") && err.contains("MAX_HIDDEN_LAYERS"),
        "wrong guard fired: {err}"
    );
}

/// Same DoS vector as above with a value (10_000_000) large enough to be an absurd
/// layer count but small enough that, absent the `MAX_HIDDEN_LAYERS` guard, the
/// resulting allocation would actually complete rather than aborting the process --
/// this is the config used for the mutation revert-check (see fix report).
#[test]
fn test_ten_million_hidden_layers_rejected() {
    let json = r#"{"text_config": {"num_hidden_layers": 10000000}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("num_hidden_layers: 10_000_000 must yield an InferenceError")
        .to_string();
    assert!(
        err.contains("num_hidden_layers") && err.contains("MAX_HIDDEN_LAYERS"),
        "wrong guard fired: {err}"
    );
}

/// `num_hidden_layers` is attacker-controlled `config.json` input consumed to size a
/// `Vec` (via `compute_layer_types`/`normalize_layer_mask`) before any tensor-level
/// validation runs. Rejects a layer count above `MAX_HIDDEN_LAYERS` as a typed error
/// at parse time, before any allocation is sized from it.
#[test]
fn test_num_hidden_layers_above_ceiling_errors_before_allocation() {
    let json = r#"{"text_config": {"num_hidden_layers": 10000000}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("num_hidden_layers above MAX_HIDDEN_LAYERS must yield an InferenceError")
        .to_string();
    assert!(
        err.contains("num_hidden_layers"),
        "wrong guard fired: {err}"
    );
}

// ──────────────────────────────────────────────────────────────────────
// Bounded layer_types / layer_mask deserialization
// ──────────────────────────────────────────────────────────────────────

/// `layer_types` sized past `MAX_HIDDEN_LAYERS` must be rejected during
/// deserialization itself -- before the post-parse `num_hidden_layers` guard runs.
/// `num_hidden_layers` is deliberately left at its default (40, well under the cap)
/// so an `Err` here can only come from the bounded-seq deserializer, not the
/// separate `num_hidden_layers > MAX_HIDDEN_LAYERS` check.
#[test]
fn test_layer_types_array_over_max_hidden_layers_rejected_at_deserialize() {
    let elems = vec!["\"linear_attention\""; MAX_HIDDEN_LAYERS + 1].join(",");
    let json = format!(r#"{{"text_config": {{"layer_types": [{elems}]}}}}"#);
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "a layer_types array of MAX_HIDDEN_LAYERS+1 elements must be rejected \
                 during deserialization, not allocated in full",
        )
        .to_string();
    assert!(
        err.contains("MAX_HIDDEN_LAYERS"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_layer_types_in_bounds_array_accepted() {
    let json = r#"{"text_config": {"num_hidden_layers": 3, "full_attention_interval": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention"]}}"#;
    let cfg = Qwen35Config::from_config_json_str(json)
        .expect("an in-bounds layer_types array must be accepted");
    assert_eq!(cfg.layer_types.len(), 3);
}

/// Same DoS vector as above for `layer_mask`, the sibling per-layer array.
#[test]
fn test_layer_mask_array_over_max_hidden_layers_rejected_at_deserialize() {
    let elems = vec!["true"; MAX_HIDDEN_LAYERS + 1].join(",");
    let json = format!(r#"{{"text_config": {{"layer_mask": [{elems}]}}}}"#);
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "a layer_mask array of MAX_HIDDEN_LAYERS+1 elements must be rejected \
                 during deserialization, not allocated in full",
        )
        .to_string();
    assert!(
        err.contains("MAX_HIDDEN_LAYERS"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_layer_mask_in_bounds_array_accepted() {
    let json = r#"{"text_config": {"num_hidden_layers": 3, "full_attention_interval": 4,
            "layer_mask": [true, false, true]}}"#;
    let cfg = Qwen35Config::from_config_json_str(json)
        .expect("an in-bounds layer_mask array must be accepted");
    assert_eq!(cfg.layer_mask, vec![true, false, true]);
}

// ──────────────────────────────────────────────────────────────────────
// hidden_size / vocab_size bounds
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_hidden_size_zero_errors() {
    let json = r#"{"text_config": {"hidden_size": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("hidden_size: 0 must yield an InferenceError")
        .to_string();
    assert!(err.contains("hidden_size"), "wrong guard fired: {err}");
}

#[test]
fn test_vocab_size_zero_errors() {
    let json = r#"{"text_config": {"vocab_size": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("vocab_size: 0 must yield an InferenceError")
        .to_string();
    assert!(err.contains("vocab_size"), "wrong guard fired: {err}");
}

/// A checkpoint declaring `embed_tokens` as `[huge_vocab, 0]` (zero-byte tensor,
/// passes shape checks) must be rejected at config-parse time via `MAX_VOCAB_SIZE`,
/// before the `logits` buffer resize (`cache.rs`) sees this value.
#[test]
fn test_vocab_size_over_max_errors() {
    let json = format!(
        r#"{{"text_config": {{"vocab_size": {}}}}}"#,
        MAX_VOCAB_SIZE + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("vocab_size above MAX_VOCAB_SIZE must yield an InferenceError")
        .to_string();
    assert!(
        err.contains("vocab_size") && err.contains("MAX_VOCAB_SIZE"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_vocab_size_at_max_accepted() {
    let json = format!(r#"{{"text_config": {{"vocab_size": {MAX_VOCAB_SIZE}}}}}"#);
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "vocab_size == MAX_VOCAB_SIZE must be accepted"
    );
}

// ──────────────────────────────────────────────────────────────────────
// head_dim bound (config-level budget, independent of layer mix)
// ──────────────────────────────────────────────────────────────────────

/// An all-linear-attention layer mix never calls `checked_full_q_dim` (the
/// full-attention Q projection sizing check), so `MAX_HEAD_DIM` must fire on its
/// own to bound `RopeTable::new`'s allocation regardless of layer mix.
#[test]
fn test_head_dim_over_max_rejected_all_linear_config() {
    let json = format!(
        r#"{{"text_config": {{"head_dim": {}, "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#,
        MAX_HEAD_DIM + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "head_dim above MAX_HEAD_DIM must yield an InferenceError even for an \
                 all-linear-attention config",
        )
        .to_string();
    assert!(
        err.contains("head_dim") && err.contains("MAX_HEAD_DIM"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_head_dim_at_max_accepted_all_linear_config() {
    let json = format!(
        r#"{{"text_config": {{"head_dim": {MAX_HEAD_DIM}, "num_hidden_layers": 2,
                "partial_rotary_factor": 0.25,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#
    );
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "head_dim == MAX_HEAD_DIM must be accepted for an all-linear-attention config"
    );
}

// ──────────────────────────────────────────────────────────────────────
// full-attention geometry global overflow bound (config-level budget,
// independent of layer mix)
// ──────────────────────────────────────────────────────────────────────

/// An all-linear-attention layer mix never loads full-attention tensors, so it never
/// reaches the loader's per-layer full-attention `q_dim` checks -- yet
/// `ForwardScratch::ensure_capacity` unconditionally computes `full_q_dim()` and
/// `2 * q_dim` for every config during generation. `num_attention_heads * head_dim`
/// must be rejected at parse time even when no full-attention layer is present.
///
/// Originally exercised the `checked_mul` overflow guard directly (an `usize::MAX`
/// `num_attention_heads` overflowing the `* head_dim` product). `MAX_ATTENTION_HEADS`
/// (added alongside `MAX_HIDDEN_SIZE` / `MAX_POSITION_EMBEDDINGS` in the allocation-scalar
/// sweep) now rejects any `num_attention_heads` above 8,192 before that multiplication
/// runs -- and since `MAX_ATTENTION_HEADS * MAX_HEAD_DIM` (8,192 * 2,048 = 16,777,216)
/// never overflows `usize`, the overflow branch is no longer reachable through
/// `num_attention_heads` alone. The `checked_mul` call stays in the source as
/// defense-in-depth (e.g. against a future `MAX_ATTENTION_HEADS` or `MAX_HEAD_DIM`
/// increase); this test now covers `MAX_ATTENTION_HEADS` catching the same extreme input.
#[test]
fn test_full_q_dim_overflow_rejected_all_linear_config() {
    let json = format!(
        r#"{{"text_config": {{"head_dim": 2, "num_attention_heads": {},
                "num_key_value_heads": 1, "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#,
        usize::MAX
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "an extreme num_attention_heads must yield an InferenceError even for an \
                 all-linear-attention config",
        )
        .to_string();
    assert!(
        err.contains("num_attention_heads") && err.contains("MAX_ATTENTION_HEADS"),
        "wrong guard fired: {err}"
    );
}

/// Same bypass as above, but driven through `num_key_value_heads` instead of
/// `num_attention_heads`. `num_attention_heads` must remain a multiple of
/// `num_key_value_heads` (an earlier guard), so both are set to `usize::MAX` here --
/// this still proves the KV-dimension geometry is bound by the same global check, since
/// `full_kv_dim` shares `head_dim` with `full_q_dim` and neither guard can be bypassed by
/// routing the overflow through the KV head count instead of the Q head count.
///
/// As with the Q-dim test above, `MAX_ATTENTION_HEADS` now catches this extreme
/// `num_attention_heads` (checked before `num_key_value_heads`) ahead of the
/// `checked_mul` overflow branch it originally targeted; see that test's doc comment.
#[test]
fn test_full_kv_dim_overflow_rejected_all_linear_config() {
    let json = format!(
        r#"{{"text_config": {{"head_dim": 2, "num_attention_heads": {max},
                "num_key_value_heads": {max}, "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#,
        max = usize::MAX
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "an extreme num_attention_heads / num_key_value_heads must yield an \
                 InferenceError even for an all-linear-attention config",
        )
        .to_string();
    assert!(
        err.contains("MAX_ATTENTION_HEADS"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_full_attention_geometry_accepted_full_attention_config() {
    // A real full-attention-carrying config (the default MoE preset, whose fields an
    // empty `text_config` object resolves) must still load.
    assert!(
        Qwen35Config::from_config_json_str(r#"{"text_config": {}}"#).is_ok(),
        "a valid full-attention-carrying config must be accepted"
    );
}

#[test]
fn test_full_attention_geometry_accepted_all_linear_config() {
    let json = r#"{"text_config": {"num_hidden_layers": 2,
            "layer_types": ["linear_attention", "linear_attention"]}}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "a valid all-linear-attention config must be accepted"
    );
}

// ──────────────────────────────────────────────────────────────────────
// full-attention geometry allocation BUDGET (config-level, independent of
// layer mix) -- distinct from the overflow-only checks above: a
// non-overflowing-but-enormous num_attention_heads * head_dim must also be
// rejected, since it drives ForwardScratch::ensure_capacity's q_buf /
// context allocations to exabyte scale without ever wrapping usize.
// ──────────────────────────────────────────────────────────────────────

/// head_dim=2048 (== MAX_HEAD_DIM, itself legal) with num_attention_heads=513 gives
/// full_q_dim = 1,050,624 -- one order of magnitude within usize (no overflow) but over
/// MAX_FULL_ATTENTION_DIM. num_key_value_heads stays small so full_kv_dim is nowhere
/// near the budget, isolating the q_dim branch specifically.
#[test]
fn test_full_q_dim_over_budget_rejected_all_linear_config() {
    let json = format!(
        r#"{{"text_config": {{"head_dim": {MAX_HEAD_DIM}, "num_attention_heads": 513,
                "num_key_value_heads": 1, "num_hidden_layers": 2,
                "partial_rotary_factor": 0.25,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "a non-overflowing but budget-exceeding full_q_dim must yield an \
                 InferenceError even for an all-linear-attention config",
        )
        .to_string();
    assert!(
        err.contains("q_dim") && err.contains("MAX_FULL_ATTENTION_DIM"),
        "wrong guard fired: {err}"
    );
}

/// Same allocation-budget bypass, but with num_attention_heads == num_key_value_heads so
/// full_kv_dim also exceeds the budget (kv_dim can never exceed q_dim given the
/// num_attention_heads-is-a-multiple-of-num_key_value_heads invariant enforced earlier in
/// this function, so this proves the KV-routed budget check is reachable via the same
/// global check rather than dead code, mirroring the existing overflow test pair above).
#[test]
fn test_full_kv_dim_over_budget_rejected_all_linear_config() {
    let json = format!(
        r#"{{"text_config": {{"head_dim": {MAX_HEAD_DIM}, "num_attention_heads": 513,
                "num_key_value_heads": 513, "num_hidden_layers": 2,
                "partial_rotary_factor": 0.25,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "a non-overflowing but budget-exceeding full_kv_dim must yield an \
                 InferenceError even for an all-linear-attention config",
        )
        .to_string();
    assert!(
        err.contains("MAX_FULL_ATTENTION_DIM"),
        "wrong guard fired: {err}"
    );
}

/// Boundary: head_dim=2048 (MAX_HEAD_DIM) * num_attention_heads=512 == exactly
/// MAX_FULL_ATTENTION_DIM (1,048,576) must be accepted -- guards against an off-by-one
/// in the new budget check.
#[test]
fn test_full_attention_dims_at_budget_accepted_all_linear_config() {
    let json = format!(
        r#"{{"text_config": {{"head_dim": {MAX_HEAD_DIM}, "num_attention_heads": 512,
                "num_key_value_heads": 1, "num_hidden_layers": 2,
                "partial_rotary_factor": 0.25,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#
    );
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "full_q_dim == MAX_FULL_ATTENTION_DIM must be accepted"
    );
}

// ──────────────────────────────────────────────────────────────────────
// MoE dimension bounds (num_experts / num_experts_per_tok)
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_moe_num_experts_zero_errors() {
    let json = r#"{"text_config": {"num_experts": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("num_experts: 0 must yield an InferenceError")
        .to_string();
    assert!(err.contains("num_experts"), "wrong guard fired: {err}");
}

#[test]
fn test_moe_num_experts_per_tok_zero_errors() {
    let json = r#"{"text_config": {"num_experts": 4, "num_experts_per_tok": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("num_experts_per_tok: 0 must yield an InferenceError")
        .to_string();
    assert!(
        err.contains("num_experts_per_tok"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_moe_num_experts_per_tok_over_num_experts_errors() {
    let json = r#"{"text_config": {"num_experts": 8, "num_experts_per_tok": 9}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("num_experts_per_tok > num_experts must yield an InferenceError")
        .to_string();
    assert!(
        err.contains("num_experts_per_tok") && err.contains("num_experts"),
        "wrong guard fired: {err}"
    );
}

/// A tiny zero-sized checkpoint can set `num_experts` to an extreme value; downstream,
/// `ForwardScratch::ensure_capacity` unconditionally resizes `router_logits` to
/// `num_experts`, so this must be bound at parse time.
#[test]
fn test_moe_num_experts_over_max_errors() {
    let json = format!(
        r#"{{"text_config": {{"num_experts": {}, "num_experts_per_tok": 1}}}}"#,
        MAX_NUM_EXPERTS + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("num_experts above MAX_NUM_EXPERTS must yield an InferenceError")
        .to_string();
    assert!(
        err.contains("num_experts") && err.contains("MAX_NUM_EXPERTS"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_moe_valid_config_accepted() {
    let json = r#"{"text_config": {"num_experts": 256, "num_experts_per_tok": 8}}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "a valid MoE config must be accepted"
    );
}

#[test]
fn test_moe_dense_config_unaffected() {
    // A dense (non-MoE) checkpoint leaves both MoE fields absent from config.json;
    // represent that explicitly with `null` so the struct-level `#[serde(default)]`
    // fallback (which pulls from the MoE preset) does not mask the dense case.
    let json = r#"{"text_config": {"num_experts": null, "num_experts_per_tok": null,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "linear_attention"]}}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "a dense config with no MoE fields must be unaffected by the MoE dimension checks"
    );
}

// ──────────────────────────────────────────────────────────────────────
// intermediate size bounds (intermediate_size / moe_intermediate_size /
// shared_expert_intermediate_size)
// ──────────────────────────────────────────────────────────────────────

/// `intermediate_size` is always present and drives MLP scratch buffer allocations on
/// every forward pass; a huge value paired with zero-sized tensors would otherwise pass
/// shape checks trivially before blowing up allocation at generation time.
#[test]
fn test_intermediate_size_over_max_errors() {
    let json = format!(
        r#"{{"text_config": {{"intermediate_size": {}}}}}"#,
        MAX_INTERMEDIATE_SIZE + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("intermediate_size above MAX_INTERMEDIATE_SIZE must yield an InferenceError")
        .to_string();
    assert!(
        err.contains("intermediate_size") && err.contains("MAX_INTERMEDIATE_SIZE"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_intermediate_size_at_max_accepted() {
    // hidden_size pinned to 1 (FIX 8: intermediate_size alone at MAX_INTERMEDIATE_SIZE,
    // paired with the preset's own default hidden_size, would otherwise breach the new
    // MAX_DENSE_FFN_TENSOR_BYTES product budget -- this test is about the per-field
    // intermediate_size cap in isolation, not the product budget covered separately).
    let json = format!(
        r#"{{"text_config": {{"intermediate_size": {MAX_INTERMEDIATE_SIZE}, "hidden_size": 1}}}}"#
    );
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "intermediate_size == MAX_INTERMEDIATE_SIZE must be accepted"
    );
}

/// `moe_intermediate_size` is gated on its own `Option` presence, independent of
/// `num_experts` -- explicitly set it (not just inherit the MoE preset default) so this
/// test actually exercises the if-let gate rather than silently passing through an
/// unrelated preset value.
#[test]
fn test_moe_intermediate_size_over_max_errors() {
    let json = format!(
        r#"{{"text_config": {{"moe_intermediate_size": {}}}}}"#,
        MAX_INTERMEDIATE_SIZE + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "moe_intermediate_size above MAX_INTERMEDIATE_SIZE must yield an InferenceError",
        )
        .to_string();
    assert!(
        err.contains("moe_intermediate_size") && err.contains("MAX_INTERMEDIATE_SIZE"),
        "wrong guard fired: {err}"
    );
}

/// Same as above for `shared_expert_intermediate_size`, explicitly present so the if-let
/// gate is genuinely exercised rather than trivially satisfied by the preset default.
#[test]
fn test_shared_expert_intermediate_size_over_max_errors() {
    let json = format!(
        r#"{{"text_config": {{"shared_expert_intermediate_size": {}}}}}"#,
        MAX_INTERMEDIATE_SIZE + 1
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "shared_expert_intermediate_size above MAX_INTERMEDIATE_SIZE must yield an \
                 InferenceError",
        )
        .to_string();
    assert!(
        err.contains("shared_expert_intermediate_size") && err.contains("MAX_INTERMEDIATE_SIZE"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_moe_and_shared_intermediate_size_at_max_accepted() {
    let json = format!(
        r#"{{"text_config": {{"moe_intermediate_size": {MAX_INTERMEDIATE_SIZE},
                "shared_expert_intermediate_size": {MAX_INTERMEDIATE_SIZE}}}}}"#
    );
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "moe_intermediate_size / shared_expert_intermediate_size == \
             MAX_INTERMEDIATE_SIZE must be accepted"
    );
}

#[test]
fn test_dense_config_no_moe_intermediate_fields_unaffected() {
    // A dense (non-MoE) checkpoint leaves both fields absent; represent that explicitly
    // with `null` so the struct-level `#[serde(default)]` fallback (which pulls from the
    // MoE preset) does not mask the dense case.
    let json = r#"{"text_config": {"moe_intermediate_size": null,
            "shared_expert_intermediate_size": null,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "linear_attention"]}}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "a dense config with no MoE intermediate fields must be unaffected by the \
             MAX_INTERMEDIATE_SIZE gated checks"
    );
}

#[test]
fn test_zero_linear_conv_kernel_dim_errors() {
    // `linear_conv_kernel_dim - 1` underflows usize (panics in debug, wraps to a ~16 EiB
    // allocation in release) in the GatedDeltaNet conv-buffer sizing.
    let json = r#"{"text_config": {"linear_conv_kernel_dim": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("linear_conv_kernel_dim: 0 must yield an InferenceError, not underflow")
        .to_string();
    assert!(
        err.contains("linear_conv_kernel_dim"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_zero_linear_num_key_heads_errors() {
    // gdn_fused divides `value_heads / linear_num_key_heads`; a parseable 0 is a hard
    // integer divide-by-zero panic deep in the GatedDeltaNet recurrence (#342).
    let json = r#"{"text_config": {"linear_num_key_heads": 0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("linear_num_key_heads: 0 must yield an InferenceError, not divide-by-zero")
        .to_string();
    assert!(
        err.contains("linear_num_key_heads"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_value_heads_below_key_heads_errors() {
    // value_heads < key_heads makes the integer `ratio = value_heads / key_heads == 0`, then
    // `h / ratio` is a divide-by-zero panic. value=1/key=16 (preset) hits ratio 0.
    let json = r#"{"text_config": {"linear_num_value_heads": 1}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("value_heads < key_heads must yield an InferenceError, not divide-by-zero")
        .to_string();
    assert!(
        err.contains("linear_num_value_heads"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_value_heads_multiple_of_key_heads_accepted() {
    // Boundary: the real GDN shape (key 16, value 32 → ratio 2) must pass the divisibility
    // guard. Guards against the new check wrongly rejecting legitimate asymmetric heads.
    let json = r#"{"text_config": {"linear_num_key_heads": 16, "linear_num_value_heads": 32}}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "key 16 / value 32 (ratio 2) is a real GDN config and must be accepted"
    );
}

#[test]
fn test_partial_rotary_factor_above_one_errors() {
    // rope_dim = (head_dim * factor); factor > 1 makes rope_dim exceed head_dim and
    // indexes head_vec[rope_dim/2 + i] out of bounds in apply_partial_rope.
    let json = r#"{"text_config": {"partial_rotary_factor": 3.0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("partial_rotary_factor > 1.0 must yield an InferenceError, not OOB")
        .to_string();
    assert!(
        err.contains("partial_rotary_factor"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn test_partial_rotary_factor_one_accepted() {
    // Boundary: factor == 1.0 makes rope_dim == head_dim (full rotary), which is in range.
    // Guards against an off-by-one in the [0.0, 1.0] range check.
    let json = r#"{"text_config": {"partial_rotary_factor": 1.0}}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "partial_rotary_factor == 1.0 (full rotary) must be accepted"
    );
}

#[test]
fn nested_rope_theta_applies_when_flat_is_absent() {
    // The defaulted flat field reads 1e7 here, so presence has to come from the raw JSON.
    let json = r#"{"text_config": {"rope_parameters": {"rope_theta": 500000.0}}}"#;
    let cfg = Qwen35Config::from_config_json_str(json).expect("nested rope_theta parses");
    assert_eq!(cfg.rope_theta, 500_000.0);
}

#[test]
fn equal_flat_and_nested_rope_theta_accepted() {
    let json = r#"{"text_config": {"rope_theta": 500000.0,
            "rope_parameters": {"rope_theta": 500000.0}}}"#;
    let cfg = Qwen35Config::from_config_json_str(json).expect("equal values agree");
    assert_eq!(cfg.rope_theta, 500_000.0);
}

#[test]
fn conflicting_flat_and_nested_rope_theta_errors() {
    let json = r#"{"text_config": {"rope_theta": 500000.0,
            "rope_parameters": {"rope_theta": 1000000.0}}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("disagreeing rope_theta values must not pick one")
        .to_string();
    assert!(
        err.contains("(500000)") && err.contains("(1000000)"),
        "error must name both values: {err}"
    );
}

#[test]
fn nested_partial_rotary_factor_applies_when_flat_is_absent() {
    let json = r#"{"text_config": {"rope_parameters": {"partial_rotary_factor": 0.5}}}"#;
    let cfg = Qwen35Config::from_config_json_str(json).expect("nested factor parses");
    assert_eq!(cfg.partial_rotary_factor, 0.5);
}

#[test]
fn conflicting_flat_and_nested_partial_rotary_factor_errors() {
    let json = r#"{"text_config": {"partial_rotary_factor": 0.25,
            "rope_parameters": {"partial_rotary_factor": 0.5}}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("disagreeing partial_rotary_factor values must not pick one")
        .to_string();
    assert!(
        err.contains("(0.25)") && err.contains("(0.5)"),
        "error must name both values: {err}"
    );
}

#[test]
fn equal_flat_and_nested_partial_rotary_factor_accepted() {
    let json = r#"{"text_config": {"partial_rotary_factor": 0.25,
            "rope_parameters": {"partial_rotary_factor": 0.25}}}"#;
    let cfg = Qwen35Config::from_config_json_str(json).expect("equal values agree");
    assert_eq!(cfg.partial_rotary_factor, 0.25);
}

// ──────────────────────────────────────────────────────────────────────
// Flat (text-only) config.json: geometry written at the top level
// ──────────────────────────────────────────────────────────────────────

/// One geometry literal, emitted in both HF shapes. A flat text-only checkpoint
/// (`Qwen3_5ForCausalLM`, transformers 5.x) writes these fields at the top level of
/// `config.json`; a vision-language checkpoint wraps the same fields in `text_config`.
/// Both forms are built from this single `F1_GEOMETRY` object so the test compares the
/// two forms rather than a remembered number.
///
/// The five MoE fields are written as explicit `null` to state the fixture's density in
/// the literal itself. Each MoE field carries its own field-level `#[serde(default)]`,
/// so an ABSENT field would also resolve to `None`; the nulls do not change the parsed
/// result, they only keep the fixture dense if the field-level defaults are ever removed (the
/// container-level default is the MoE preset).
const F1_GEOMETRY: &str = r#"
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "vocab_size": 151936,
        "intermediate_size": 6144,
        "rms_norm_eps": 1e-6,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                            "partial_rotary_factor": 0.25},
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                        "full_attention"],
        "num_experts": null,
        "num_experts_per_tok": null,
        "moe_intermediate_size": null,
        "shared_expert_intermediate_size": null,
        "router_aux_loss_coef": null,
        "tie_word_embeddings": false,
        "eos_token_id": 248044,
        "max_position_embeddings": 32768
    "#;

fn flat_config_json() -> String {
    format!(
        r#"{{"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
                "transformers_version": "5.17.0", {F1_GEOMETRY}}}"#
    )
}

fn wrapped_config_json() -> String {
    format!(
        r#"{{"architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5", "text_config": {{{F1_GEOMETRY}}}}}"#
    )
}

#[test]
fn flat_config_reads_the_same_geometry_as_its_wrapped_twin() {
    // A flat text-only config.json must yield the geometry its wrapped twin does, field
    // by field. `Qwen35Config` has no `PartialEq`, so the loader-visible geometry is
    // compared field by field.
    let flat = Qwen35Config::from_config_json_str(&flat_config_json()).expect("flat parses");
    let wrapped =
        Qwen35Config::from_config_json_str(&wrapped_config_json()).expect("wrapped parses");

    assert_eq!(flat.num_hidden_layers, wrapped.num_hidden_layers);
    assert_eq!(flat.hidden_size, wrapped.hidden_size);
    assert_eq!(flat.num_attention_heads, wrapped.num_attention_heads);
    assert_eq!(flat.num_key_value_heads, wrapped.num_key_value_heads);
    assert_eq!(flat.head_dim, wrapped.head_dim);
    assert_eq!(flat.intermediate_size, wrapped.intermediate_size);
    assert_eq!(flat.vocab_size, wrapped.vocab_size);
    assert_eq!(flat.linear_num_key_heads, wrapped.linear_num_key_heads);
    assert_eq!(flat.linear_num_value_heads, wrapped.linear_num_value_heads);
    assert_eq!(flat.linear_key_head_dim, wrapped.linear_key_head_dim);
    assert_eq!(flat.linear_value_head_dim, wrapped.linear_value_head_dim);
    assert_eq!(flat.linear_conv_kernel_dim, wrapped.linear_conv_kernel_dim);
    assert_eq!(
        flat.full_attention_interval,
        wrapped.full_attention_interval
    );
    assert_eq!(flat.layer_types, wrapped.layer_types);
    assert_eq!(flat.rope_theta, wrapped.rope_theta);
    assert_eq!(flat.partial_rotary_factor, wrapped.partial_rotary_factor);
    assert_eq!(flat.rope_dim(), wrapped.rope_dim());
    assert_eq!(flat.tie_word_embeddings, wrapped.tie_word_embeddings);
    assert_eq!(flat.eos_token_id, wrapped.eos_token_id);
    assert_eq!(
        flat.max_position_embeddings,
        wrapped.max_position_embeddings
    );

    // Anchor the comparison to the literal, not just to each other.
    assert_eq!(flat.num_hidden_layers, 4);
    assert_eq!(flat.hidden_size, 2048);
    assert_eq!(flat.head_dim, 128);
    assert_eq!(flat.vocab_size, 151_936);
}

#[test]
fn flat_config_is_not_the_moe_preset() {
    // Before this fix, a flat file silently inherited the Qwen3.6-35B-A3B MoE preset.
    // The preset sets `num_experts: Some(256)` (and the other MoE fields); a dense flat
    // file must carry none of them.
    let cfg = Qwen35Config::from_config_json_str(&flat_config_json()).expect("flat parses");
    let preset = Qwen35Config::qwen36_35b_a3b();

    // The preset the old fallback substituted carries the MoE router fields; the flat
    // file explicitly resolves them to `None`.
    assert_eq!(preset.num_experts, Some(256));
    assert!(
        !cfg.is_moe(),
        "a dense flat config must not resolve to the MoE preset"
    );
    assert_eq!(cfg.num_experts, None);
    assert_eq!(cfg.num_experts_per_tok, None);
    assert_eq!(cfg.moe_intermediate_size, None);
    assert_eq!(cfg.shared_expert_intermediate_size, None);
    assert_eq!(cfg.router_aux_loss_coef, None);
    assert!(!cfg.output_router_logits);
    assert_eq!(
        preset.num_hidden_layers, 40,
        "the preset's layer count is the one the old fallback substituted"
    );
    assert_ne!(
        cfg.num_hidden_layers, preset.num_hidden_layers,
        "a flat config must use its own layer count, not the preset's 40"
    );
    assert_eq!(
        preset.linear_num_value_heads,
        Some(32),
        "the preset's GDN value-head count is the MoE one"
    );
    assert_ne!(
        cfg.linear_num_value_heads, preset.linear_num_value_heads,
        "a flat config must use its own GDN value-head count, not the preset's"
    );
}

#[test]
fn flat_config_missing_both_geometry_keys_refuses() {
    for json in ["{}", r#"{"tie_word_embeddings": false}"#] {
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("a config with neither text_config nor flat geometry must refuse")
            .to_string();
        assert!(
            err.contains("no text_config object and no top-level"),
            "wrong refusal message for {json}: {err}"
        );
    }
}

#[test]
fn flat_config_rope_conflicts_are_refused_naming_the_flat_field() {
    // Same conflict detection as the wrapped form, but the flat fields are spelled at
    // the top level and the message must name them without a `text_config.` prefix.
    let theta_conflict = r#"{"num_hidden_layers": 4, "hidden_size": 2048,
            "rope_theta": 500000, "rope_parameters": {"rope_theta": 1000000}}"#;
    let err = Qwen35Config::from_config_json_str(theta_conflict)
        .expect_err("disagreeing flat and nested rope_theta must not pick one")
        .to_string();
    assert!(
        err.contains("rope_theta (500000)") && err.contains("(1000000)"),
        "error must name both values with the flat spelling: {err}"
    );
    assert!(
        !err.contains("text_config."),
        "a flat file's error must not claim a text_config prefix: {err}"
    );

    let factor_conflict = r#"{"num_hidden_layers": 4, "hidden_size": 2048,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"partial_rotary_factor": 0.5}}"#;
    let err = Qwen35Config::from_config_json_str(factor_conflict)
        .expect_err("disagreeing flat and nested partial_rotary_factor must not pick one")
        .to_string();
    assert!(
        err.contains("(0.25)") && err.contains("(0.5)"),
        "error must name both values: {err}"
    );
    assert!(
        !err.contains("text_config."),
        "a flat file's error must not claim a text_config prefix: {err}"
    );
}

#[test]
fn flat_config_with_only_num_hidden_layers_refuses() {
    // The top-level gate requires both keys; num_hidden_layers alone is not enough to
    // read geometry.
    let json = r#"{"num_hidden_layers": 24}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("a flat config with only num_hidden_layers must refuse")
        .to_string();
    assert!(
        err.contains("no text_config object and no top-level"),
        "wrong refusal message: {err}"
    );
}

#[test]
fn qwen35_0_8b_fixture_keeps_nested_rope_values() {
    let json = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen35_0_8b_config.json"
    ));
    let cfg = Qwen35Config::from_config_json_str(json).expect("fixture parses");
    assert_eq!(cfg.rope_theta, 10_000_000.0);
    assert_eq!(cfg.partial_rotary_factor, 0.25);
}

// ──────────────────────────────────────────────────────────────────────
// rope_dim invariant tests (issue #401)
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_odd_rope_dim_errors_not_panics() {
    // Mutation contract: removing the `rope_dim < 2 || rope_dim % 2 != 0` guard must
    // make this test FAIL (the call returns Ok instead of Err).
    //
    // head_dim=10, partial_rotary_factor=0.3 → rope_dim = (10 * 0.3) as usize = 3 (odd).
    // apply_partial_rope uses half = rope_dim / 2 = 1, rotating only pair (0,1) and
    // leaving dim 2 (inside the declared rotate range) silently unrotated — wrong output.
    let json = r#"{"text_config": {"head_dim": 10, "partial_rotary_factor": 0.3}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("odd rope_dim must yield an InferenceError, not silent wrong output")
        .to_string();
    assert!(err.contains("rope_dim"), "wrong guard fired: {err}");
}

#[test]
fn test_zero_rope_dim_errors_not_panics() {
    // Mutation contract: removing the `rope_dim < 2 || rope_dim % 2 != 0` guard must
    // make this test FAIL (the call returns Ok instead of Err).
    //
    // partial_rotary_factor=0.0 → rope_dim = 0.  RopeTable::new(0, ..) gives
    // max_positions()=0, so every non-empty-sequence call to the capacity-guarded APIs
    // rejects the input rather than applying a no-op — contrary to caller expectations.
    // Reject fail-closed until a dedicated no-RoPE dispatch path exists (Refs #401).
    let json = r#"{"text_config": {"partial_rotary_factor": 0.0}}"#;
    let err = Qwen35Config::from_config_json_str(json)
        .expect_err("zero rope_dim must yield an InferenceError, not capacity-zero surprise")
        .to_string();
    assert!(err.contains("rope_dim"), "wrong guard fired: {err}");
}

// `test_rope_dim_exceeds_head_dim_via_f32_rounding_errors` removed: it targeted
// `rope_dim() casting head_dim through f32 and rounding up past head_dim`, only
// observable at head_dim >= 2^24 (16,777,216). `MAX_HEAD_DIM` (2048) now rejects every
// head_dim in that range before `rope_dim()` is ever computed, so the f32-rounding edge
// is no longer reachable through any config this parser admits. `MAX_HEAD_DIM` itself is
// covered by `test_head_dim_over_max_rejected_all_linear_config` /
// `test_head_dim_at_max_accepted_all_linear_config`; the `rope_dim > cfg.head_dim` line
// stays in the source as defense-in-depth against a future `MAX_HEAD_DIM` increase.

#[test]
fn test_generate_config_enable_thinking_default_and_toggle() {
    let default_cfg = GenerateConfig::default();
    assert!(
        default_cfg.enable_thinking,
        "default must have thinking enabled"
    );

    let no_think = GenerateConfig {
        min_p: 0.0,
        enable_thinking: false,
        ..GenerateConfig::default()
    };
    assert!(!no_think.enable_thinking);
    // Other fields unaffected
    assert_eq!(no_think.max_new_tokens, 256);
    assert!((no_think.temperature - 0.7).abs() < 1e-6);
}

#[test]
fn test_qwen36_27b_layer_distribution_matches_config_json() {
    // Verify compute_layer_types(64, 4) produces the pattern from config.json:
    // [linear, linear, linear, full] × 16 times
    let types = compute_layer_types(64, 4);
    for chunk_start in (0..64).step_by(4) {
        assert_eq!(types[chunk_start], LayerType::LinearAttention);
        assert_eq!(types[chunk_start + 1], LayerType::LinearAttention);
        assert_eq!(types[chunk_start + 2], LayerType::LinearAttention);
        assert_eq!(types[chunk_start + 3], LayerType::FullAttention);
    }
}

#[test]
fn test_qwen35_config_backward_compat() {
    // Qwen3.5 config (no MoE fields in JSON) must still deserialize correctly.
    let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "vocab_size": 248320,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "rms_norm_eps": 0.000001,
                "intermediate_size": 6144,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "eos_token_id": 248044,
                "max_position_embeddings": 262144,
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25
            }
        }"#;

    let cfg = Qwen35Config::from_config_json_str(json).expect("backward-compat config parses");
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.linear_num_value_heads(), 16);
    assert!(!cfg.is_moe(), "Qwen3.5 must not be detected as MoE");
    assert!(
        cfg.tie_word_embeddings,
        "default tie_word_embeddings is true"
    );
    assert_eq!(cfg.num_experts, None);
    assert_eq!(cfg.num_experts_per_tok, None);
    assert_eq!(
        cfg.mtp_num_hidden_layers, 0,
        "Qwen3.5 mtp_num_hidden_layers must default to 0"
    );
    assert!(!cfg.mtp_use_dedicated_embeddings);
}

#[test]
fn qwen35_config_missing_max_position_embeddings_defaults_4096() {
    // #551: a config.json without max_position_embeddings must not
    // silently inherit the container-level Default's (qwen36_35b_a3b)
    // 262_144 context — it must resolve to the documented 4096 fallback.
    let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "vocab_size": 248320,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "rms_norm_eps": 0.000001,
                "intermediate_size": 6144,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "eos_token_id": 248044,
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25
            }
        }"#;

    let cfg = Qwen35Config::from_config_json_str(json)
        .expect("config without max_position_embeddings still parses");
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn test_qwen35_0_8b_preset_dimensions() {
    let cfg = Qwen35Config::qwen35_0_8b();
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.vocab_size, 248_320);
    assert_eq!(cfg.intermediate_size, 3584);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.rope_dim(), 64); // 256 * 0.25
    assert_eq!(cfg.linear_num_key_heads, 16);
    assert_eq!(cfg.linear_num_value_heads(), 16);
    assert_eq!(cfg.linear_key_head_dim, 128);
    assert_eq!(cfg.linear_value_head_dim, 128);
    assert_eq!(cfg.eos_token_id, 248_044);
    assert_eq!(cfg.max_position_embeddings, 262_144);
    assert_eq!(cfg.mtp_num_hidden_layers, 1);
    assert!(cfg.tie_word_embeddings);
    assert!(!cfg.is_moe(), "Qwen3.5-0.8B is dense, not MoE");
    // Same hybrid pattern as the 2B: [linear, linear, linear, full] x 6.
    assert_eq!(cfg.layer_types.len(), 24);
    assert_eq!(cfg.num_full_attention_layers(), 6);
    assert_eq!(cfg.num_linear_attention_layers(), 18);
}

#[test]
fn test_qwen35_0_8b_config_json_fixture_parses() {
    // Parse the real released config.json (downloaded verbatim) — proves
    // from_config_json handles the 0.8B checkpoint, not just my transcription.
    let json = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen35_0_8b_config.json"
    ));
    let cfg = Qwen35Config::from_config_json_str(json).expect("Qwen3.5-0.8B config.json parses");

    // Core dims must match the released checkpoint.
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.vocab_size, 248_320);
    assert_eq!(cfg.intermediate_size, 3584);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.linear_num_key_heads, 16);
    assert_eq!(cfg.linear_num_value_heads(), 16);
    assert_eq!(cfg.linear_key_head_dim, 128);
    assert_eq!(cfg.linear_value_head_dim, 128);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert_eq!(cfg.full_attention_interval, 4);
    assert_eq!(cfg.eos_token_id, 248_044);
    assert_eq!(cfg.max_position_embeddings, 262_144);
    assert_eq!(cfg.mtp_num_hidden_layers, 1);

    // Released checkpoint is a dense vision-language model: the MoE fields must still
    // resolve to None (0.8B has no MoE), while the vision_config / token-id / mrope
    // fields below are now parsed onto the config (ADR-069 S1) instead of being dropped.
    assert!(!cfg.is_moe(), "Qwen3.5-0.8B is dense, not MoE");

    // rope_theta and partial_rotary_factor are nested under rope_parameters
    // in this checkpoint; verify they resolve to the correct values.
    assert_eq!(cfg.rope_theta, 10_000_000.0);
    assert!((cfg.partial_rotary_factor - 0.25).abs() < 1e-6);
    assert_eq!(cfg.rope_dim(), 64);

    // layer_types comes from the explicit JSON array: [lin, lin, lin, full] x 6.
    assert_eq!(cfg.layer_types.len(), 24);
    assert_eq!(cfg.num_full_attention_layers(), 6);
    assert_eq!(cfg.num_linear_attention_layers(), 18);
    let full_indices: Vec<usize> = cfg
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == LayerType::FullAttention)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(full_indices, vec![3, 7, 11, 15, 19, 23]);

    // tie_word_embeddings is taken from the outer wrapper.
    assert!(cfg.tie_word_embeddings);

    // ADR-069 S1: vision_config, the four vision token ids, and the M-RoPE section
    // fields are top-level / text_config.rope_parameters siblings, respectively, and
    // must now round-trip onto the config (parsed but not yet consumed by forward).
    let vision = cfg
        .vision_config
        .as_ref()
        .expect("released checkpoint has a vision_config");
    assert_eq!(vision.depth, 12);
    assert_eq!(vision.hidden_size, 768);
    assert_eq!(vision.num_heads, 12);
    assert_eq!(vision.patch_size, 16);
    assert_eq!(vision.spatial_merge_size, 2);
    assert_eq!(vision.out_hidden_size, 1024);
    assert_eq!(vision.temporal_patch_size, 2);
    assert_eq!(vision.num_position_embeddings, 2304);
    assert_eq!(vision.in_channels, 3);
    assert!(vision.deepstack_visual_indexes.is_empty());

    assert_eq!(cfg.image_token_id, Some(248_056));
    assert_eq!(cfg.video_token_id, Some(248_057));
    assert_eq!(cfg.vision_start_token_id, Some(248_053));
    assert_eq!(cfg.vision_end_token_id, Some(248_054));

    let rope_params = cfg
        .rope_parameters
        .as_ref()
        .expect("released checkpoint nests rope_parameters under text_config");
    assert_eq!(rope_params.mrope_section, Some(vec![11, 11, 10]));
    assert_eq!(rope_params.mrope_interleaved, Some(true));
}

#[test]
fn test_text_only_config_has_no_vision_fields() {
    // A text-only config.json (no vision_config, no token ids, no mrope fields) must
    // still parse cleanly, with every vision-language field resolving to None — proving
    // the ADR-069 S1 additions don't regress the plain-text decoder path. Also asserts an
    // existing non-vision field to prove nothing else regressed in the same parse.
    let json = r#"{
            "text_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "vocab_size": 1000,
                "intermediate_size": 2048,
                "rms_norm_eps": 1e-6,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "eos_token_id": 999,
                "max_position_embeddings": 4096
            }
        }"#;
    let cfg = Qwen35Config::from_config_json_str(json).expect("text-only config.json still parses");

    assert_eq!(cfg.hidden_size, 1024, "non-vision field must still parse");
    assert!(cfg.vision_config.is_none());
    assert!(cfg.image_token_id.is_none());
    assert!(cfg.video_token_id.is_none());
    assert!(cfg.vision_start_token_id.is_none());
    assert!(cfg.vision_end_token_id.is_none());
    let rope_params = cfg.rope_parameters.clone().unwrap_or_default();
    assert!(rope_params.mrope_section.is_none());
    assert!(rope_params.mrope_interleaved.is_none());
}

/// Minimal text_config JSON (parses cleanly on its own, per
/// `test_text_only_config_has_no_vision_fields`) plus a top-level `vision_config`
/// object with every field valid except the ones the caller overrides.
fn config_json_with_vision(vision_config_body: &str) -> String {
    format!(
        r#"{{
                "text_config": {{
                    "hidden_size": 1024,
                    "num_hidden_layers": 4,
                    "vocab_size": 1000,
                    "intermediate_size": 2048,
                    "rms_norm_eps": 1e-6,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.25,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 16,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "full_attention_interval": 4,
                    "eos_token_id": 999,
                    "max_position_embeddings": 4096
                }},
                "vision_config": {vision_config_body}
            }}"#
    )
}

#[test]
fn parser_rejects_present_vision_config_with_depth_zero() {
    // A present-but-malformed vision_config (depth: 0) is syntactically valid JSON
    // and, before this fix, would silently load a truncated tensor set later. It must
    // be rejected here, at parse time.
    let json = config_json_with_vision(
        r#"{
                "depth": 0,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }"#,
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("depth: 0 vision_config must be rejected at parse time");
    assert!(
        err.to_string().contains("depth"),
        "error must name depth: {err}"
    );
}

#[test]
fn parser_rejects_present_vision_config_with_num_heads_zero() {
    let json = config_json_with_vision(
        r#"{
                "depth": 12,
                "hidden_size": 768,
                "num_heads": 0,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }"#,
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("num_heads: 0 vision_config must be rejected at parse time");
    assert!(
        err.to_string().contains("num_heads"),
        "error must name num_heads: {err}"
    );
}

/// The vision checkpoint loader mints ~12 tensor-name Strings per ViT block *before* any
/// tensor validation runs; an unbounded `depth` (previously only checked nonzero) drives
/// unbounded String allocation ahead of any shape check.
#[test]
fn parser_rejects_present_vision_config_with_depth_over_max() {
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": {},
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#,
        MAX_VISION_DEPTH + 1
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("depth above MAX_VISION_DEPTH must be rejected at parse time")
        .to_string();
    assert!(
        err.contains("depth") && err.contains("MAX_VISION_DEPTH"),
        "wrong guard fired: {err}"
    );
}

#[test]
fn parser_accepts_present_vision_config_with_depth_at_max() {
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": {MAX_VISION_DEPTH},
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
    ));
    assert!(
        Qwen35Config::from_config_json_str(&json).is_ok(),
        "depth == MAX_VISION_DEPTH must be accepted"
    );
}

// ──────────────────────────────────────────────────────────────────────
// config vector length bounds (mrope_section / deepstack_visual_indexes)
// ──────────────────────────────────────────────────────────────────────

/// `RopeParams::mrope_section` is deserialized as a `Vec<usize>` before any validation
/// runs; a huge declared length must be rejected at parse time rather than reaching an
/// uncontrolled allocation, even though the field is not yet consumed by the forward pass.
#[test]
fn parser_rejects_present_mrope_section_over_max() {
    let huge: Vec<String> = (0..MAX_CONFIG_VECTOR_LEN + 1)
        .map(|_| "0".to_string())
        .collect();
    let json = format!(
        r#"{{"text_config": {{"rope_parameters": {{"mrope_section": [{}]}}}}}}"#,
        huge.join(",")
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("mrope_section length above MAX_CONFIG_VECTOR_LEN must be rejected")
        .to_string();
    assert!(
        err.contains("mrope_section") && err.contains("MAX_CONFIG_VECTOR_LEN"),
        "wrong guard fired: {err}"
    );
}

/// Accept-case control: a realistic `mrope_section` (one entry per M-RoPE axis) must
/// still parse cleanly.
#[test]
fn parser_accepts_present_mrope_section_realistic_size() {
    let json = r#"{"text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}}}"#;
    let cfg =
        Qwen35Config::from_config_json_str(json).expect("realistic mrope_section must be accepted");
    assert_eq!(
        cfg.rope_parameters
            .expect("rope_parameters present")
            .mrope_section,
        Some(vec![11, 11, 10])
    );
}

/// `VisionModelConfig::deepstack_visual_indexes` is deserialized as a `Vec<usize>` before
/// any validation runs; a huge declared length must be rejected at parse time regardless
/// of whether the indexes are ever consumed.
#[test]
fn parser_rejects_present_deepstack_visual_indexes_over_max() {
    let huge: Vec<String> = (0..MAX_CONFIG_VECTOR_LEN + 1)
        .map(|_| "0".to_string())
        .collect();
    let json = config_json_with_vision(&format!(
        r#"{{
                "depth": 12,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3,
                "deepstack_visual_indexes": [{}]
            }}"#,
        huge.join(",")
    ));
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err("deepstack_visual_indexes length above MAX_CONFIG_VECTOR_LEN must be rejected")
        .to_string();
    assert!(
        err.contains("deepstack_visual_indexes") && err.contains("MAX_CONFIG_VECTOR_LEN"),
        "wrong guard fired: {err}"
    );
}

/// Accept-case control: a realistic `deepstack_visual_indexes` (a handful of in-range
/// layer indexes) must still parse cleanly.
#[test]
fn parser_accepts_present_deepstack_visual_indexes_realistic_size() {
    let json = config_json_with_vision(
        r#"{
                "depth": 12,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3,
                "deepstack_visual_indexes": [2, 5, 8]
            }"#,
    );
    let cfg = Qwen35Config::from_config_json_str(&json)
        .expect("realistic deepstack_visual_indexes must be accepted");
    assert_eq!(
        cfg.vision_config
            .expect("vision_config present")
            .deepstack_visual_indexes,
        vec![2, 5, 8]
    );
}

// ──────────────────────────────────────────────────────────────────────
// global linear-attention output-dim budget (independent of layer mix)
// ──────────────────────────────────────────────────────────────────────

/// An all-full-attention config never reaches `load_linear_attention_weights`'s per-layer
/// `checked_linear_output_dim` derivation, yet `ForwardScratch::ensure_capacity`
/// unconditionally derives `linear_output_dim()` for every config. A
/// `linear_num_value_heads` of `usize::MAX` must overflow-reject here rather than
/// panicking/wrapping downstream.
#[test]
fn parser_rejects_all_full_attention_config_with_overflowing_linear_output_dim() {
    let json = format!(
        r#"{{"text_config": {{
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "full_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 1,
            "linear_value_head_dim": {}
        }}}}"#,
        usize::MAX
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "overflowing linear_value_head_dim on an all-full-attention config must be \
                 rejected before any allocation",
        )
        .to_string();
    assert!(
        err.contains("output_dim") && err.contains("overflows"),
        "wrong guard fired: {err}"
    );
}

/// Same all-full-attention scenario, but with a huge, non-overflowing product that must
/// still be rejected against `MAX_LINEAR_OUTPUT_DIM`.
#[test]
fn parser_rejects_all_full_attention_config_with_linear_output_dim_over_max() {
    // `linear_num_value_heads` sits at its own `MAX_LINEAR_NUM_VALUE_HEADS` cap (so that
    // guard does not fire first); `linear_value_head_dim` is the free factor that pushes
    // the `linear_output_dim` product over `MAX_LINEAR_OUTPUT_DIM` while each individual
    // factor stays within its own cap.
    let value_heads = MAX_LINEAR_NUM_VALUE_HEADS;
    let value_dim = (MAX_LINEAR_OUTPUT_DIM / value_heads) + 1;
    let json = format!(
        r#"{{"text_config": {{
                "num_hidden_layers": 2,
                "layer_types": ["full_attention", "full_attention"],
                "linear_num_key_heads": 1,
                "linear_num_value_heads": {value_heads},
                "linear_key_head_dim": 1,
                "linear_value_head_dim": {value_dim}
            }}}}"#
    );
    let err = Qwen35Config::from_config_json_str(&json)
        .expect_err(
            "linear_output_dim above MAX_LINEAR_OUTPUT_DIM on an all-full-attention config \
                 must be rejected",
        )
        .to_string();
    assert!(
        err.contains("output_dim") && err.contains("MAX_LINEAR_OUTPUT_DIM"),
        "wrong guard fired: {err}"
    );
}

/// Accept-case control: realistic linear-attention dims on an all-full-attention config
/// must still parse cleanly -- the global budget must not reject real geometry.
#[test]
fn parser_accepts_all_full_attention_config_with_realistic_linear_dims() {
    let json = r#"{"text_config": {
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "full_attention"],
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128
        }}"#;
    assert!(
        Qwen35Config::from_config_json_str(json).is_ok(),
        "realistic linear-attention dims on an all-full-attention config must be accepted"
    );
}

// ──────────────────────────────────────────────────────────────────────
// KV-cache layer-count invariant tests (issue #170)
// ──────────────────────────────────────────────────────────────────────

/// KV cache layer count is the number of full-attention layers, NOT all layers.
///
/// This is the primary regression guard: a silent switch to all-24-layer KV
/// allocation would change `kv_cache_layer_count()` from 6 to 24 and cause
/// this test to fail.
#[test]
fn kv_layer_count_excludes_linear_layers() {
    let cfg = Qwen35Config::qwen35_0_8b();
    assert_eq!(
        cfg.kv_cache_layer_count(),
        6,
        "must be full-attention count"
    );
    assert_ne!(
        cfg.kv_cache_layer_count(),
        cfg.num_hidden_layers,
        "kv_cache_layer_count must not equal num_hidden_layers (would 4× decode memory)"
    );
}

/// Full + linear layers must sum to total hidden layers for every preset.
///
/// `compute_layer_types` produces exactly `num_hidden_layers` entries, each
/// either FullAttention or LinearAttention, so this invariant must always hold.
/// If any preset fails here it indicates a config bug.
#[test]
fn full_plus_linear_equals_total() {
    for (name, cfg) in [
        ("qwen35_0_8b", Qwen35Config::qwen35_0_8b()),
        ("qwen35_2b", Qwen35Config::qwen35_2b()),
        ("qwen36_35b_a3b", Qwen35Config::qwen36_35b_a3b()),
        ("qwen36_27b", Qwen35Config::qwen36_27b()),
    ] {
        assert_eq!(
            cfg.num_full_attention_layers() + cfg.num_linear_attention_layers(),
            cfg.num_hidden_layers,
            "{name}: full + linear must equal num_hidden_layers"
        );
    }
}

/// KV bytes per token for f16 matches the expected 12_288 B for qwen3.5-0.8B.
///
/// Formula: `num_full(6) × 2(K+V) × full_kv_dim(512) × dtype_bytes(2) = 12_288`.
/// At a 4096-token context this is 48 MiB.
#[test]
fn kv_bytes_per_token_f16() {
    let cfg = Qwen35Config::qwen35_0_8b();
    // 6 layers × 2 (K+V) × 512 (kv_dim) × 2 (f16) = 12_288 B/token = 48 MiB @ 4096 ctx
    assert_eq!(cfg.kv_bytes_per_token(2), 12_288);
}

/// Numeric identity: kv_bytes_per_token(1) == num_hidden_layers * head_dim for 0.8B.
///
/// `6 × 2 × 512 × 1 = 6_144 = 24 × 256`. This is a coincidence specific to
/// the 0.8B parameters (full_attention_interval=4, num_kv_heads=2); it does
/// NOT generalise across configs and must not be used as a formula.
#[test]
fn kv_bytes_per_token_identity() {
    let cfg = Qwen35Config::qwen35_0_8b();
    assert_eq!(cfg.kv_bytes_per_token(1), 6_144);
    // Coincidence: matches num_hidden_layers × head_dim for this specific preset only.
    assert_eq!(
        cfg.kv_bytes_per_token(1),
        cfg.num_hidden_layers * cfg.head_dim
    );
}

// ── decode_cap unit tests ────────────────────────────────────────────────

#[test]
fn decode_cap_none_budget_returns_max() {
    // Disabled path must be byte-identical to the pre-budget behaviour.
    assert_eq!(decode_cap(None, 512), 512);
    assert_eq!(decode_cap(None, 0), 0);
}

#[test]
fn decode_cap_zero_budget_returns_max() {
    // Some(0) is treated as disabled.
    assert_eq!(decode_cap(Some(0), 512), 512);
    assert_eq!(decode_cap(Some(0), 1), 1);
}

#[test]
fn decode_cap_nonzero_budget_adds_budgets() {
    // Worst case = rb + max_new_tokens + 1 (the +1 is the forced </think> delimiter).
    // Mutation-sensitive: revert to rb+max and these assertions fail.
    assert_eq!(decode_cap(Some(2048), 512), 2561);
    assert_eq!(decode_cap(Some(1), 1), 3);
    assert_eq!(decode_cap(Some(100), 200), 301);
}

#[test]
fn decode_cap_saturates_on_overflow() {
    // saturating_add must not wrap on usize::MAX inputs.
    assert_eq!(decode_cap(Some(usize::MAX), 1), usize::MAX);
    assert_eq!(decode_cap(Some(1), usize::MAX), usize::MAX);
}

// ── force_close_think unit tests ────────────────────────────────────────

#[test]
fn force_close_think_disabled_when_budget_none() {
    // budget=None → always returns None regardless of other args.
    assert_eq!(
        force_close_think(None, true, false, 100, Some(99), true),
        None,
        "None budget must disable forcing"
    );
}

#[test]
fn force_close_think_disabled_when_budget_zero() {
    // budget=Some(0) → budget > 0 guard fails → None.
    assert_eq!(
        force_close_think(Some(0), true, false, 100, Some(99), true),
        None,
        "budget=0 must disable forcing"
    );
}

#[test]
fn force_close_think_disabled_when_enable_thinking_false() {
    // enable_thinking=false → no reasoning block → forcing is a no-op.
    assert_eq!(
        force_close_think(Some(10), false, false, 20, Some(99), true),
        None,
        "enable_thinking=false must disable forcing"
    );
}

#[test]
fn force_close_think_disabled_when_already_closed() {
    // thinking_closed=true → block already closed → should not force again.
    assert_eq!(
        force_close_think(Some(10), true, true, 20, Some(99), true),
        None,
        "already-closed thinking block must not force again"
    );
}

#[test]
fn force_close_think_disabled_when_close_id_none() {
    // close_id=None → model has no </think> token → forcing is a no-op.
    assert_eq!(
        force_close_think(Some(10), true, false, 20, None, true),
        None,
        "close_id=None must disable forcing"
    );
}

#[test]
fn force_close_think_fires_at_budget_boundary() {
    let close_id = 248_069_u32;
    // generated_so_far == budget → should force (mutation: >= not >).
    assert_eq!(
        force_close_think(Some(10), true, false, 10, Some(close_id), true),
        Some(close_id),
        "must force when generated_so_far equals budget"
    );
    // generated_so_far > budget → also forces.
    assert_eq!(
        force_close_think(Some(10), true, false, 11, Some(close_id), true),
        Some(close_id),
        "must force when generated_so_far exceeds budget"
    );
}

#[test]
fn force_close_think_waits_for_a_boundary_within_the_grace_window() {
    let close_id = 248_069_u32;
    assert_eq!(
        force_close_think(Some(10), true, false, 10, Some(close_id), false),
        None,
        "the budget alone must not close before a boundary or the fallback"
    );
    assert_eq!(
        force_close_think(Some(10), true, false, 20, Some(close_id), true),
        Some(close_id),
        "a boundary after the budget must close"
    );
    assert_eq!(
        force_close_think(Some(10), true, false, 41, Some(close_id), false),
        None,
        "the fallback is exactly B + 32, not earlier"
    );
    assert_eq!(
        force_close_think(Some(10), true, false, 42, Some(close_id), false),
        Some(close_id),
        "the fallback must close at B + 32"
    );
}

#[test]
fn force_close_think_accepts_a_boundary_at_the_budget() {
    let close_id = 248_069_u32;
    assert_eq!(
        force_close_think(Some(10), true, false, 10, Some(close_id), true),
        Some(close_id),
        "a boundary at the budget must close without delay"
    );
}

#[test]
fn force_close_think_does_not_fire_before_budget() {
    let close_id = 248_069_u32;
    // generated_so_far < budget → must NOT force (mutation-sensitive boundary).
    assert_eq!(
        force_close_think(Some(10), true, false, 9, Some(close_id), false),
        None,
        "must not force when generated_so_far is one below budget"
    );
    assert_eq!(
        force_close_think(Some(10), true, false, 0, Some(close_id), false),
        None,
        "must not force when zero tokens generated"
    );
}

// -- from_model_dir: the single shared config-resolution policy (#923) --

#[test]
fn from_model_dir_errors_on_missing_config_json() {
    let tmp = tempfile::tempdir().unwrap();
    // Directory exists but has no config.json at all.
    let err = Qwen35Config::from_model_dir(tmp.path())
        .expect_err("a directory with no config.json must be a hard error");
    let msg = err.to_string();
    assert!(
        msg.contains("config.json"),
        "error must name the missing file: {msg}"
    );
    assert!(
        msg.contains(&tmp.path().display().to_string()),
        "error must name the offending directory: {msg}"
    );
}

#[test]
fn from_model_dir_loads_a_real_config_json() {
    let tmp = tempfile::tempdir().unwrap();
    let json = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen35_0_8b_config.json"
    ));
    std::fs::write(tmp.path().join("config.json"), json).unwrap();

    let cfg = Qwen35Config::from_model_dir(tmp.path())
        .expect("a directory with a valid config.json must load");
    assert_eq!(
        cfg.hidden_size, 1024,
        "must parse the real 0.8B config, not a preset"
    );
}

#[test]
fn from_model_dir_propagates_a_malformed_config_json_parse_error() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("config.json"), "not valid json {{{").unwrap();

    let err = Qwen35Config::from_model_dir(tmp.path())
        .expect_err("malformed config.json must still be a parse error, not a preset");
    assert!(
        err.to_string().contains("config.json") || err.to_string().contains("invalid Qwen"),
        "error must reflect the parse failure: {err}"
    );
}

#[test]
fn from_model_dir_rejects_oversized_config_json_before_read_to_string() {
    // FIX 5 regression: an oversized config.json must be rejected via the file's
    // metadata length, before `read_to_string` materializes it.
    let tmp = tempfile::tempdir().unwrap();
    let oversized = "x".repeat(MAX_CONFIG_JSON_BYTES as usize + 1);
    std::fs::write(tmp.path().join("config.json"), oversized).unwrap();

    let err = Qwen35Config::from_model_dir(tmp.path())
        .expect_err("an oversized config.json must be rejected");
    assert!(
        err.to_string().contains("MAX_CONFIG_JSON_BYTES"),
        "wrong guard fired: {err}"
    );

    // The validated sibling (`from_model_dir_validated` -> `from_config_json_validated`)
    // is an independent `read_to_string` call site (FIX 5 sibling-invocation-path) and
    // must be bounded the same way.
    let err_validated = Qwen35Config::from_model_dir_validated(tmp.path())
        .expect_err("an oversized config.json must be rejected via the validated sibling");
    assert!(
        err_validated.to_string().contains("MAX_CONFIG_JSON_BYTES"),
        "wrong guard fired: {err_validated}"
    );
}

mod validation_bounds;
