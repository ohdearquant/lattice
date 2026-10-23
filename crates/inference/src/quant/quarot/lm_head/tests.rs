use super::*;

fn insert_tensor(
    tensors: &mut HashMap<String, TensorEntry>,
    name: &str,
    shape: Vec<usize>,
    data: Vec<f64>,
) {
    tensors.insert(
        name.to_string(),
        TensorEntry {
            name: name.to_string(),
            shape,
            data,
        },
    );
}

/// Test vocab — small enough to keep `[vocab_size, hidden_size]` tensors
/// tractable per test (the qwen35_0_8b preset's 248_320×1024 ≈ 2 GB
/// would be wasteful here).
const TEST_VOCAB: usize = 64;

fn tied_qwen35_test_cfg() -> Qwen35Config {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    assert!(cfg.tie_word_embeddings, "qwen35_0_8b preset must be tied");
    cfg.vocab_size = TEST_VOCAB;
    cfg
}

fn untied_qwen35_test_cfg() -> Qwen35Config {
    let mut cfg = tied_qwen35_test_cfg();
    cfg.tie_word_embeddings = false;
    cfg
}

fn synthetic_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 11) as u32;
            (bits as f64 / u32::MAX as f64) - 0.5
        })
        .collect()
}

#[test]
fn tied_config_materializes_lm_head_as_clone_of_embed_tokens() {
    let cfg = tied_qwen35_test_cfg();
    let mut tensors = HashMap::new();
    let embed_data = synthetic_f64(cfg.vocab_size * cfg.hidden_size, 1);
    insert_tensor(
        &mut tensors,
        QWEN35_EMBED_TOKENS_NAME,
        vec![cfg.vocab_size, cfg.hidden_size],
        embed_data.clone(),
    );
    assert!(!tensors.contains_key(QWEN35_LM_HEAD_NAME));

    materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap();

    let lm = tensors
        .get(QWEN35_LM_HEAD_NAME)
        .expect("lm_head materialized");
    assert_eq!(lm.shape, vec![cfg.vocab_size, cfg.hidden_size]);
    assert_eq!(lm.data, embed_data);
    assert_eq!(lm.name, QWEN35_LM_HEAD_NAME);
    // embed_tokens unchanged.
    assert_eq!(tensors[QWEN35_EMBED_TOKENS_NAME].data, embed_data);
}

#[test]
fn untied_config_with_lm_head_present_is_noop() {
    let cfg = untied_qwen35_test_cfg();
    let mut tensors = HashMap::new();
    let embed_data = synthetic_f64(cfg.vocab_size * cfg.hidden_size, 2);
    let lm_data = synthetic_f64(cfg.vocab_size * cfg.hidden_size, 3);
    insert_tensor(
        &mut tensors,
        QWEN35_EMBED_TOKENS_NAME,
        vec![cfg.vocab_size, cfg.hidden_size],
        embed_data.clone(),
    );
    insert_tensor(
        &mut tensors,
        QWEN35_LM_HEAD_NAME,
        vec![cfg.vocab_size, cfg.hidden_size],
        lm_data.clone(),
    );

    materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap();

    // No-op: lm_head data preserved (NOT overwritten with embed_tokens).
    assert_eq!(tensors[QWEN35_LM_HEAD_NAME].data, lm_data);
    assert_eq!(tensors[QWEN35_EMBED_TOKENS_NAME].data, embed_data);
}

#[test]
fn tied_config_with_missing_embed_tokens_errors() {
    let cfg = tied_qwen35_test_cfg();
    let mut tensors = HashMap::new();
    let err = materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(QWEN35_EMBED_TOKENS_NAME),
        "unexpected error: {msg}"
    );
    assert!(msg.contains("tied config"), "unexpected error: {msg}");
}

#[test]
fn tied_config_with_lm_head_already_present_errors() {
    // Defensive check: tied flag + pre-existing lm_head is inconsistent.
    let cfg = tied_qwen35_test_cfg();
    let mut tensors = HashMap::new();
    insert_tensor(
        &mut tensors,
        QWEN35_EMBED_TOKENS_NAME,
        vec![cfg.vocab_size, cfg.hidden_size],
        vec![0.0; cfg.vocab_size * cfg.hidden_size],
    );
    insert_tensor(
        &mut tensors,
        QWEN35_LM_HEAD_NAME,
        vec![cfg.vocab_size, cfg.hidden_size],
        vec![0.0; cfg.vocab_size * cfg.hidden_size],
    );

    let err = materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("already in working set"),
        "unexpected error: {msg}"
    );
}

#[test]
fn untied_config_with_missing_lm_head_errors() {
    let cfg = untied_qwen35_test_cfg();
    let mut tensors = HashMap::new();
    insert_tensor(
        &mut tensors,
        QWEN35_EMBED_TOKENS_NAME,
        vec![cfg.vocab_size, cfg.hidden_size],
        vec![0.0; cfg.vocab_size * cfg.hidden_size],
    );
    let err = materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("untied config"), "unexpected error: {msg}");
    assert!(msg.contains(QWEN35_LM_HEAD_NAME), "unexpected error: {msg}");
}

#[test]
fn untied_config_with_lm_head_shape_mismatch_errors() {
    // A wrong-row-count `lm_head.weight` must be rejected at the converter
    // boundary — `qwen_required_tensor_names` is name-only and downstream
    // `fuse_rmsnorms` only checks cols. Without this check a malformed
    // matrix would slip into the rest of the pipeline.
    let cfg = untied_qwen35_test_cfg();
    let mut tensors = HashMap::new();
    insert_tensor(
        &mut tensors,
        QWEN35_LM_HEAD_NAME,
        vec![cfg.vocab_size + 1, cfg.hidden_size],
        vec![0.0; (cfg.vocab_size + 1) * cfg.hidden_size],
    );
    let err = materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("shape"), "unexpected error: {msg}");
    assert!(msg.contains(QWEN35_LM_HEAD_NAME), "unexpected error: {msg}");
}

#[test]
fn untied_config_with_lm_head_data_len_mismatch_errors() {
    // Shape says one thing; data buffer disagrees. Catches malformed
    // TensorEntry construction (shape and data can drift in principle).
    let cfg = untied_qwen35_test_cfg();
    let mut tensors = HashMap::new();
    insert_tensor(
        &mut tensors,
        QWEN35_LM_HEAD_NAME,
        vec![cfg.vocab_size, cfg.hidden_size],
        vec![0.0; cfg.vocab_size * cfg.hidden_size - 1],
    );
    let err = materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("data.len()"), "unexpected error: {msg}");
    assert!(msg.contains(QWEN35_LM_HEAD_NAME), "unexpected error: {msg}");
}

#[test]
fn tied_config_with_embed_tokens_shape_mismatch_errors() {
    let cfg = tied_qwen35_test_cfg();
    let mut tensors = HashMap::new();
    // Wrong vocab_size dimension.
    insert_tensor(
        &mut tensors,
        QWEN35_EMBED_TOKENS_NAME,
        vec![cfg.vocab_size + 1, cfg.hidden_size],
        vec![0.0; (cfg.vocab_size + 1) * cfg.hidden_size],
    );
    let err = materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("shape"), "unexpected error: {msg}");
    assert!(
        msg.contains(QWEN35_EMBED_TOKENS_NAME),
        "unexpected error: {msg}"
    );
    // Did NOT insert lm_head on failure.
    assert!(!tensors.contains_key(QWEN35_LM_HEAD_NAME));
}

#[test]
fn final_norm_fusion_target_uses_canonical_names() {
    let tgt = qwen35_final_norm_fusion_target();
    assert_eq!(tgt.norm_tensor, QWEN35_FINAL_NORM_NAME);
    assert_eq!(
        tgt.downstream_weights,
        vec![QWEN35_LM_HEAD_NAME.to_string()]
    );
}

#[test]
fn final_norm_fusion_target_norm_name_matches_loader() {
    // Round-trip: the final_norm name we emit must match what the loader
    // actually requests for this config.
    let cfg = tied_qwen35_test_cfg();
    let required = crate::model::qwen35::qwen_required_tensor_names(&cfg);
    let tgt = qwen35_final_norm_fusion_target();
    assert!(
        required.contains(&tgt.norm_tensor),
        "final_norm tensor `{}` not in qwen_required_tensor_names",
        tgt.norm_tensor
    );
}

#[test]
fn untie_word_embeddings_flips_true_to_false() {
    let mut cfg = tied_qwen35_test_cfg();
    assert!(cfg.tie_word_embeddings);
    untie_word_embeddings_in_cfg(&mut cfg);
    assert!(!cfg.tie_word_embeddings);
}

#[test]
fn untie_word_embeddings_is_idempotent_on_untied() {
    let mut cfg = untied_qwen35_test_cfg();
    assert!(!cfg.tie_word_embeddings);
    untie_word_embeddings_in_cfg(&mut cfg);
    assert!(!cfg.tie_word_embeddings);
}

/// End-to-end: after materialization + final-norm fusion + rotation
/// absorption, the materialized lm_head must carry BOTH transforms
/// while embed_tokens carries only the rotation. The two matrices must
/// not be byte-identical (they were before fusion + absorption ran).
#[test]
fn materialized_lm_head_diverges_from_embed_tokens_after_pipeline() {
    use crate::quant::quarot::hadamard::RandomizedHadamard;
    use crate::quant::quarot::pipeline::{absorb_rotations, fuse_rmsnorms};
    use crate::quant::quarot::plan::RotationPlan;

    let cfg = tied_qwen35_test_cfg();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let mut tensors = HashMap::new();
    insert_tensor(
        &mut tensors,
        QWEN35_EMBED_TOKENS_NAME,
        vec![vocab, hidden],
        synthetic_f64(vocab * hidden, 7),
    );
    insert_tensor(
        &mut tensors,
        QWEN35_FINAL_NORM_NAME,
        vec![hidden],
        synthetic_f64(hidden, 8),
    );

    materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap();
    // Pre-pipeline: clone of embed_tokens.
    assert_eq!(
        tensors[QWEN35_LM_HEAD_NAME].data,
        tensors[QWEN35_EMBED_TOKENS_NAME].data
    );

    let final_norm_target = qwen35_final_norm_fusion_target();
    fuse_rmsnorms(&mut tensors, std::slice::from_ref(&final_norm_target)).unwrap();

    let rotation = RandomizedHadamard::new(0xCAFE_BABE, hidden).unwrap();
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    absorb_rotations(&mut tensors, &plan, &rotation).unwrap();

    // Post-pipeline: lm_head must differ from embed_tokens because the
    // final-norm scale was folded into lm_head before rotation.
    assert_ne!(
        tensors[QWEN35_LM_HEAD_NAME].data, tensors[QWEN35_EMBED_TOKENS_NAME].data,
        "lm_head must carry the (1 + g_final) factor that embed_tokens does not"
    );
}

/// Real-fixture round-trip: the HF Qwen3.5-0.8B config.json has BOTH
/// `text_config.tie_word_embeddings: true` AND top-level
/// `tie_word_embeddings: true`. The parser gives top-level precedence,
/// so the converter must mutate both (or at minimum the top-level) to
/// make a reload see `tie_word_embeddings=false`.
#[test]
fn output_config_flip_updates_all_hf_tie_flags_and_reparses_untied() {
    let fixture_json = include_str!("../../../../tests/fixtures/qwen35_0_8b_config.json");

    let original: serde_json::Value = serde_json::from_str(fixture_json).unwrap();
    assert_eq!(original["tie_word_embeddings"].as_bool(), Some(true));
    assert_eq!(
        original["text_config"]["tie_word_embeddings"].as_bool(),
        Some(true)
    );

    let mutated = untie_word_embeddings_in_config_json(fixture_json).unwrap();
    let mutated_value: serde_json::Value = serde_json::from_str(&mutated).unwrap();
    assert_eq!(mutated_value["tie_word_embeddings"].as_bool(), Some(false));
    assert_eq!(
        mutated_value["text_config"]["tie_word_embeddings"].as_bool(),
        Some(false),
        "nested text_config.tie_word_embeddings must also be flipped"
    );

    // Round-trip: lattice parser must observe an untied config.
    let cfg = Qwen35Config::from_config_json_str(&mutated).unwrap();
    assert!(
        !cfg.tie_word_embeddings,
        "parser must see tie_word_embeddings=false after JSON mutation"
    );
}

#[test]
fn untie_in_config_json_inserts_top_level_when_only_nested_present() {
    // If the input has only nested text_config.tie_word_embeddings, the
    // top-level must still be inserted so the parser (top-level wins)
    // sees false.
    let json = r#"{"text_config": {"tie_word_embeddings": true, "hidden_size": 64}}"#;
    let mutated = untie_word_embeddings_in_config_json(json).unwrap();
    let v: serde_json::Value = serde_json::from_str(&mutated).unwrap();
    assert_eq!(v["tie_word_embeddings"].as_bool(), Some(false));
    assert_eq!(
        v["text_config"]["tie_word_embeddings"].as_bool(),
        Some(false)
    );
}

#[test]
fn untie_in_config_json_leaves_absent_nested_alone() {
    // If text_config exists but has no `tie_word_embeddings` field, the
    // helper does not invent one — the parser defaults the nested field
    // to true but the top-level (we set it to false) wins.
    let json = r#"{"text_config": {"hidden_size": 64}, "tie_word_embeddings": true}"#;
    let mutated = untie_word_embeddings_in_config_json(json).unwrap();
    let v: serde_json::Value = serde_json::from_str(&mutated).unwrap();
    assert_eq!(v["tie_word_embeddings"].as_bool(), Some(false));
    assert!(
        v["text_config"].get("tie_word_embeddings").is_none(),
        "nested field must not be inserted when originally absent"
    );
}

#[test]
fn untie_in_config_json_rejects_invalid_json() {
    let err = untie_word_embeddings_in_config_json("not json").unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("invalid JSON"), "unexpected error: {msg}");
}

#[test]
fn untie_in_config_json_rejects_non_object_root() {
    let err = untie_word_embeddings_in_config_json("[1, 2, 3]").unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("must be an object"), "unexpected error: {msg}");
}
