use lattice_arch::deepseek_v4_inspection::{inspect_root_config, inspect_root_json};
use serde_json::{Value, json};

const FIXTURE: &str = include_str!("fixtures/deepseek_v4_flash_dspark_config.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).unwrap()
}

fn rejects(value: &Value, expected: &str) {
    let error = inspect_root_json(&value.to_string())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(expected),
        "expected {expected:?}, got {error}"
    );
}

#[test]
fn pinned_root_preserves_geometry_and_extra_compression_entries() {
    let inspection = inspect_root_json(FIXTURE).unwrap();
    assert!(!inspection.supports_inference());
    assert_eq!(inspection.num_hidden_layers, 43);
    assert_eq!(inspection.num_nextn_predict_layers, 1);
    assert_eq!(inspection.compress_ratios.len(), 46);
    assert_eq!(&inspection.compress_ratios[43..], &[0, 0, 0]);
    assert_ne!(
        inspection.compress_ratios.len(),
        43 + inspection.num_nextn_predict_layers
    );
    assert_eq!(inspection.dspark_target_layer_ids, [40, 41, 42]);
    assert_eq!(
        (
            inspection.geometry.hidden_size,
            inspection.geometry.num_attention_heads,
            inspection.geometry.head_dim,
            inspection.geometry.num_key_value_heads
        ),
        (4096, 64, 512, 1)
    );
    assert_eq!(
        (
            inspection.geometry.vocab_size,
            inspection.geometry.max_position_embeddings
        ),
        (129280, 1048576)
    );
    assert_eq!(
        (
            inspection.dspark_block_size,
            inspection.dspark_noise_token_id,
            inspection.dspark_markov_rank
        ),
        (5, 128799, 256)
    );
    for field in [
        "compress_ratios",
        "quantization_config",
        "expert_dtype",
        "rope_scaling",
    ] {
        assert!(
            inspection
                .unsupported_fields
                .iter()
                .any(|name| name == field),
            "missing unsupported field {field}"
        );
    }
}

#[test]
fn inspector_reads_actual_bounded_root_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::write(&path, FIXTURE).unwrap();
    assert_eq!(
        inspect_root_config(&path).unwrap(),
        inspect_root_json(FIXTURE).unwrap()
    );
    std::fs::write(&path, b"\xff").unwrap();
    assert!(inspect_root_config(&path).is_err());
    assert!(inspect_root_config(dir.path()).is_err());
    let file = std::fs::File::create(&path).unwrap();
    file.set_len(8_388_609).unwrap();
    assert!(
        inspect_root_config(&path)
            .unwrap_err()
            .to_string()
            .contains("MAX_CONFIG_JSON_BYTES")
    );
}

#[test]
fn wrong_identity_and_schema_are_refused() {
    for (key, value) in [
        ("model_type", json!("other")),
        ("architectures", json!(["Other"])),
    ] {
        let mut v = fixture();
        v[key] = value;
        rejects(&v, "unexpected model_type or architecture");
    }
    assert!(inspect_root_json(r#"{"dim":4096,"n_layers":43,"n_mtp_layers":3}"#).is_err());
}

#[test]
fn required_fields_missing_null_and_wrong_type_are_refused() {
    let mut missing = fixture();
    missing.as_object_mut().unwrap().remove("hidden_size");
    rejects(&missing, "hidden_size");
    for invalid in [Value::Null, json!("4096"), json!(-1)] {
        let mut v = fixture();
        v["hidden_size"] = invalid;
        rejects(&v, "invalid root fields");
    }
}

#[test]
fn duplicate_keys_are_rejected_even_in_unknown_nested_fields() {
    for (before, after) in [
        (
            "\"model_type\": \"deepseek_v4\"",
            "\"model_type\": \"deepseek_v4\", \"model_type\": \"deepseek_v4\"",
        ),
        ("\"fmt\": \"e4m3\"", "\"fmt\": \"e4m3\", \"fmt\": \"e4m3\""),
    ] {
        assert!(FIXTURE.contains(before));
        let duplicate = FIXTURE.replacen(before, after, 1);
        assert!(
            inspect_root_json(&duplicate)
                .unwrap_err()
                .to_string()
                .contains("duplicate key")
        );
    }
}

#[test]
fn all_arrays_are_bounded_before_deserialization_allocation() {
    for key in [
        "compress_ratios",
        "dspark_target_layer_ids",
        "architectures",
    ] {
        let mut v = fixture();
        v[key] = json!(vec![0; 4097]);
        rejects(&v, "JSON array exceeds inspection limit");
    }
    let mut v = fixture();
    v["unknown_nested"] = json!({"items": vec![0; 4097]});
    rejects(&v, "JSON array exceeds inspection limit");
    let mut v = fixture();
    v["compress_ratios"] = json!(vec![0; 4096]);
    assert_eq!(
        inspect_root_json(&v.to_string())
            .unwrap()
            .compress_ratios
            .len(),
        4096
    );
}

#[test]
fn nested_object_key_count_is_bounded() {
    let mut v = fixture();
    let mut object = serde_json::Map::new();
    for id in 0..4097 {
        object.insert(id.to_string(), json!(0));
    }
    v["unknown_nested"] = Value::Object(object);
    rejects(&v, "JSON object exceeds inspection limit");
}

#[test]
fn unknown_fields_are_named_and_never_claim_inference_support() {
    let mut v = fixture();
    v["future_option"] = json!({"mode": "unknown"});
    v["compress_ratios"][0] = json!(999);
    v["quantization_config"]["fmt"] = json!("unknown_format");
    let result = inspect_root_json(&v.to_string()).unwrap();
    assert!(!result.supports_inference());
    for name in ["future_option", "compress_ratios", "quantization_config"] {
        assert!(result.unsupported_fields.iter().any(|field| field == name));
    }
    assert_eq!(result.compress_ratios[0], 999);
}

#[test]
fn nonzero_geometry_and_bounded_counts_are_enforced() {
    for key in [
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "head_dim",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "o_groups",
        "q_lora_rank",
        "o_lora_rank",
        "index_head_dim",
        "index_n_heads",
        "index_topk",
        "sliding_window",
        "hc_mult",
        "hc_sinkhorn_iters",
        "dspark_block_size",
        "dspark_markov_rank",
    ] {
        let mut v = fixture();
        v[key] = json!(0);
        rejects(&v, "must be positive");
    }
    let mut v = fixture();
    v["hidden_size"] = json!(16_777_217usize);
    rejects(&v, "metadata geometry exceeds inspection limit");
}

#[test]
fn positive_finite_epsilon_and_rope_parameters_are_required() {
    for key in [
        "rms_norm_eps",
        "hc_eps",
        "rope_theta",
        "compress_rope_theta",
    ] {
        let mut v = fixture();
        v[key] = json!(0);
        rejects(&v, "finite and positive");
    }
    for key in ["factor", "beta_fast", "beta_slow"] {
        let mut v = fixture();
        v["rope_scaling"][key] = json!(-1);
        rejects(&v, "finite and positive");
    }
    let mut v = fixture();
    v["rope_scaling"]["original_max_position_embeddings"] = json!(0);
    rejects(&v, "must be positive");
    let mut v = fixture();
    v["rope_scaling"]["type"] = json!("");
    rejects(&v, "must not be empty");
    let invalid = FIXTURE.replacen("\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": 1e9999", 1);
    assert_ne!(invalid, FIXTURE);
    assert!(inspect_root_json(&invalid).is_err());
}

#[test]
fn expert_and_hash_layer_bounds_are_enforced() {
    let mut v = fixture();
    v["num_experts_per_tok"] = json!(257);
    rejects(&v, "num_experts_per_tok exceeds");
    let mut v = fixture();
    v["num_hash_layers"] = json!(44);
    rejects(&v, "num_hash_layers exceeds");
}

#[test]
fn dspark_token_and_target_bounds_are_enforced() {
    let mut v = fixture();
    v["dspark_noise_token_id"] = json!(129280);
    rejects(&v, "outside vocabulary");
    let mut v = fixture();
    v["dspark_target_layer_ids"] = json!([]);
    rejects(&v, "must not be empty");
    for ids in [json!([40, 40]), json!([43])] {
        let mut v = fixture();
        v["dspark_target_layer_ids"] = ids;
        rejects(&v, "unique and in range");
    }
    let mut v = fixture();
    v["dspark_target_layer_ids"] = json!([42, 40, 41]);
    assert_eq!(
        inspect_root_json(&v.to_string())
            .unwrap()
            .dspark_target_layer_ids,
        [42, 40, 41]
    );
}

#[test]
fn rotary_geometry_and_index_geometry_are_enforced() {
    for dim in [0, 63, 514] {
        let mut v = fixture();
        v["qk_rope_head_dim"] = json!(dim);
        rejects(&v, "qk_rope_head_dim must");
    }
    let mut v = fixture();
    v["index_head_dim"] = json!(32);
    rejects(&v, "exceeds index_head_dim");
}

#[test]
fn compression_entries_cover_base_layers_without_inventing_draft_count() {
    let mut v = fixture();
    v["compress_ratios"] = json!(vec![0; 42]);
    rejects(&v, "shorter than base layer count");
    let mut v = fixture();
    v["compress_ratios"] = json!(vec![0; 43]);
    assert_eq!(
        inspect_root_json(&v.to_string())
            .unwrap()
            .compress_ratios
            .len(),
        43
    );
}

#[test]
fn shape_and_byte_arithmetic_overflow_is_refused() {
    let mut v = fixture();
    v["num_attention_heads"] = json!(usize::MAX);
    rejects(&v, "attention dimension overflow");
    let mut v = fixture();
    v["o_groups"] = json!(usize::MAX);
    v["o_lora_rank"] = json!(2);
    rejects(&v, "output group dimension is zero");
    let mut v = fixture();
    v["vocab_size"] = json!(usize::MAX);
    rejects(&v, "embedding byte count overflow");
    let mut v = fixture();
    v["num_attention_heads"] = json!(usize::MAX / 512);
    rejects(&v, "attention byte count overflow");
    let mut v = fixture();
    v["o_lora_rank"] = json!(usize::MAX);
    rejects(&v, "output rank overflow");
    let mut v = fixture();
    v["num_key_value_heads"] = json!(usize::MAX);
    rejects(&v, "key/value byte count overflow");
    let mut v = fixture();
    v["n_routed_experts"] = json!(usize::MAX);
    rejects(&v, "expert byte count overflow");
    let mut v = fixture();
    v["hidden_size"] = json!(usize::MAX);
    rejects(&v, "target projection overflow");
    let mut v = fixture();
    v["hidden_size"] = json!(usize::MAX);
    v["dspark_target_layer_ids"] = json!([40]);
    rejects(&v, "markov dimension overflow");
}
