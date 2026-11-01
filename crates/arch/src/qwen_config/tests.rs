use super::*;

#[test]
fn config_uses_top_level_values_over_nested_duplicates() {
    let json = r#"{
        "text_config": {"hidden_size": 128, "rope_theta": 42.0},
        "vocab_size": 151669, "hidden_size": 1024, "num_hidden_layers": 28,
        "num_attention_heads": 16, "num_key_value_heads": 8,
        "intermediate_size": 3072, "max_position_embeddings": 32768,
        "rope_theta": 1000000.0
    }"#;
    let config = QwenConfig::from_json_str(json).unwrap();
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.head_dim, 64);
    assert_eq!(config.rope_theta, 1_000_000.0);
}

#[test]
fn config_does_not_treat_a_string_value_as_a_key() {
    let json = r#"{
        "note": "hidden_size", "bait": 96,
        "vocab_size": 151669, "hidden_size": 1024, "num_hidden_layers": 28,
        "num_attention_heads": 16, "num_key_value_heads": 8,
        "intermediate_size": 3072, "max_position_embeddings": 32768
    }"#;
    let config = QwenConfig::from_json_str(json).unwrap();
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.head_dim, 64);
}

#[test]
fn config_retains_optional_defaults_and_required_field_errors() {
    let json = r#"{
        "vocab_size": 8, "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": 12, "max_position_embeddings": 128,
        "head_dim": "invalid", "rope_theta": false, "rms_norm_eps": null
    }"#;
    let config = QwenConfig::from_json_str(json).unwrap();
    assert_eq!(config.head_dim, 4);
    assert_eq!(config.rope_theta, 1_000_000.0);
    assert_eq!(config.rms_norm_eps, 1e-6);

    let missing = r#"{"num_attention_heads": "invalid", "num_key_value_heads": 1}"#;
    let error = QwenConfig::from_json_str(missing).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("config.json missing key num_attention_heads")
    );
}

#[test]
fn config_rejects_malformed_json_and_duplicate_top_level_key() {
    let duplicate = r#"{"num_attention_heads": 1, "num_attention_heads": 2}"#;
    let malformed = r#"{"num_attention_heads": 1"#;
    for json in [duplicate, malformed] {
        let error = QwenConfig::from_json_str(json).unwrap_err();
        assert!(error.to_string().contains("invalid config.json:"));
    }
}

fn width_test_config(
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
) -> String {
    format!(
        r#"{{"vocab_size":8,"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":{heads},"num_key_value_heads":{kv_heads},"head_dim":{head_dim},"intermediate_size":{intermediate},"max_position_embeddings":128}}"#
    )
}

fn assert_width_refusal(json: &str, expected: &str) {
    match QwenConfig::from_json_str(json) {
        Err(InferenceError::InvalidInput(message)) => assert_eq!(message, expected),
        other => panic!("expected named InvalidInput, got {other:?}"),
    }
}

#[test]
fn config_rejects_q_projection_width_overflow() {
    let json = width_test_config(usize::MAX, 1, 2, 12);
    assert_width_refusal(
        &json,
        "config.json: num_attention_heads * head_dim overflows usize",
    );
}

#[test]
fn config_rejects_doubled_kv_width_overflow() {
    let head_dim = usize::MAX / 2 + 1;
    let json = width_test_config(1, 1, head_dim, 12);
    assert_width_refusal(&json, "config.json: 2 * kv_dim overflows usize");
}

#[test]
fn config_rejects_packed_qkv_width_overflow() {
    let head_dim = usize::MAX / 4 + 1;
    let json = width_test_config(2, 1, head_dim, 12);
    assert_width_refusal(&json, "config.json: q_dim + 2 * kv_dim overflows usize");
}

#[test]
fn config_rejects_doubled_intermediate_width_overflow() {
    let json = width_test_config(2, 1, 4, usize::MAX / 2 + 1);
    assert_width_refusal(&json, "config.json: 2 * intermediate_size overflows usize");
}

#[test]
fn width_guards_preserve_valid_qwen_geometries() {
    let small = QwenConfig::from_json_str(&width_test_config(2, 1, 4, 12)).unwrap();
    assert_eq!((small.q_dim(), small.kv_dim()), (8, 4));
    let preset = QwenConfig::qwen3_embedding_0_6b();
    let json = format!(
        r#"{{"vocab_size":{},"hidden_size":{},"num_hidden_layers":{},"num_attention_heads":{},"num_key_value_heads":{},"head_dim":{},"intermediate_size":{},"max_position_embeddings":{}}}"#,
        preset.vocab_size,
        preset.hidden_size,
        preset.num_hidden_layers,
        preset.num_attention_heads,
        preset.num_key_value_heads,
        preset.head_dim,
        preset.intermediate_size,
        preset.max_position_embeddings,
    );
    let parsed = QwenConfig::from_json_str(&json).unwrap();
    assert_eq!(parsed.hidden_size, 1024);
    assert_eq!((parsed.q_dim(), parsed.kv_dim()), (2048, 1024));
}

#[test]
fn required_field_error_precedes_derived_width_overflow() {
    let json = format!(
        r#"{{"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":{},"num_key_value_heads":1,"head_dim":2,"intermediate_size":12,"max_position_embeddings":128}}"#,
        usize::MAX,
    );
    let error = QwenConfig::from_json_str(&json).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("config.json missing key vocab_size")
    );
}
