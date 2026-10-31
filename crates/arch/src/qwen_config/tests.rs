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
