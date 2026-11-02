use super::BertConfig;

const FIELDS: &str = r#""vocab_size":100,"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":16,"max_position_embeddings":128,"type_vocab_size":2,"layer_norm_eps":1e-12"#;

#[test]
fn top_level_field_ignores_nested_duplicate() {
    let json = format!(r#"{{"nested":{{"hidden_size":12}},{FIELDS}}}"#);
    assert_eq!(BertConfig::from_json_str(&json).unwrap().hidden_size, 8);
}

#[test]
fn top_level_field_ignores_string_key_decoy() {
    let json = format!(r#"{{"note":"hidden_size","decoy":12,{FIELDS}}}"#);
    assert_eq!(BertConfig::from_json_str(&json).unwrap().hidden_size, 8);
}

#[test]
fn quoted_numbers_preserve_lexical_conversion() {
    let json = format!("{{{FIELDS}}}")
        .replace("\"hidden_size\":8", "\"hidden_size\":\"8\"")
        .replace("\"layer_norm_eps\":1e-12", "\"layer_norm_eps\":\"1e-12\"");
    let config = BertConfig::from_json_str(&json).unwrap();
    assert_eq!(config.hidden_size, 8);
    assert_eq!(config.layer_norm_eps, 1e-12);
    let escaped = json.replace("\"hidden_size\":\"8\"", r#""hidden_size":"8\u0030""#);
    assert!(
        BertConfig::from_json_str(&escaped)
            .unwrap_err()
            .to_string()
            .contains("invalid usize for hidden_size:")
    );
    let nan = json.replace("\"layer_norm_eps\":\"1e-12\"", "\"layer_norm_eps\":\"NaN\"");
    assert!(
        BertConfig::from_json_str(&nan)
            .unwrap_err()
            .to_string()
            .contains("layer_norm_eps must be finite and > 0")
    );
}

#[test]
fn missing_null_and_invalid_values_keep_diagnostics_and_order() {
    let missing = format!("{{{FIELDS}}}").replace("\"vocab_size\":100,", "");
    assert!(
        BertConfig::from_json_str(&missing)
            .unwrap_err()
            .to_string()
            .contains("config.json missing key vocab_size")
    );
    for value in ["null", "true", "\"no\""] {
        let json = format!("{{{FIELDS}}}")
            .replace("\"vocab_size\":100", &format!("\"vocab_size\":{value}"));
        assert!(
            BertConfig::from_json_str(&json)
                .unwrap_err()
                .to_string()
                .contains("invalid usize for vocab_size:")
        );
    }
    let ordered = format!("{{{FIELDS}}}")
        .replace("\"vocab_size\":100", "\"vocab_size\":null")
        .replace("\"hidden_size\":8", "\"hidden_size\":false");
    assert!(
        BertConfig::from_json_str(&ordered)
            .unwrap_err()
            .to_string()
            .contains("invalid usize for vocab_size:")
    );
}

#[test]
fn malformed_duplicate_and_non_object_json_are_rejected() {
    let duplicate = format!("{{\"hidden_size\":9,{FIELDS}}}");
    for json in [duplicate, format!("{{{FIELDS}"), "[]".to_string()] {
        assert!(
            BertConfig::from_json_str(&json)
                .unwrap_err()
                .to_string()
                .contains("invalid config.json:")
        );
    }
}

fn rejects_field_replacement(old: &str, new: &str, expected: &str) {
    let json = format!("{{{FIELDS}}}").replace(old, new);
    let error = BertConfig::from_json_str(&json).unwrap_err().to_string();
    assert!(error.contains("invalid BERT config.json:"), "{error}");
    assert!(error.contains(expected), "{error}");
}

#[test]
fn rejects_zero_vocab_size() {
    rejects_field_replacement(
        "\"vocab_size\":100",
        "\"vocab_size\":0",
        "vocab_size must be > 0",
    );
}
#[test]
fn rejects_zero_hidden_size() {
    rejects_field_replacement(
        "\"hidden_size\":8",
        "\"hidden_size\":0",
        "hidden_size must be > 0",
    );
}
#[test]
fn rejects_zero_num_hidden_layers() {
    rejects_field_replacement(
        "\"num_hidden_layers\":1",
        "\"num_hidden_layers\":0",
        "num_hidden_layers must be > 0",
    );
}
#[test]
fn rejects_zero_num_attention_heads() {
    rejects_field_replacement(
        "\"num_attention_heads\":2",
        "\"num_attention_heads\":0",
        "num_attention_heads must be > 0",
    );
}
#[test]
fn rejects_zero_intermediate_size() {
    rejects_field_replacement(
        "\"intermediate_size\":16",
        "\"intermediate_size\":0",
        "intermediate_size must be > 0",
    );
}
#[test]
fn rejects_zero_max_position_embeddings() {
    rejects_field_replacement(
        "\"max_position_embeddings\":128",
        "\"max_position_embeddings\":0",
        "max_position_embeddings must be > 0",
    );
}
#[test]
fn rejects_zero_type_vocab_size() {
    rejects_field_replacement(
        "\"type_vocab_size\":2",
        "\"type_vocab_size\":0",
        "type_vocab_size must be > 0",
    );
}
#[test]
fn rejects_nondivisible_head_geometry() {
    rejects_field_replacement(
        "\"num_attention_heads\":2",
        "\"num_attention_heads\":3",
        "must be divisible by num_attention_heads",
    );
    rejects_field_replacement(
        "\"num_attention_heads\":2",
        "\"num_attention_heads\":16",
        "must be divisible by num_attention_heads",
    );
}
#[test]
fn rejects_nonpositive_epsilon() {
    for value in ["0", "-1"] {
        rejects_field_replacement(
            "\"layer_norm_eps\":1e-12",
            &format!("\"layer_norm_eps\":{value}"),
            "layer_norm_eps must be finite and > 0",
        );
    }
}
#[test]
fn rejects_nonfinite_epsilon() {
    for value in ["\"NaN\"", "\"inf\""] {
        rejects_field_replacement(
            "\"layer_norm_eps\":1e-12",
            &format!("\"layer_norm_eps\":{value}"),
            "layer_norm_eps must be finite and > 0",
        );
    }
}
#[test]
fn known_presets_validate_and_parser_errors_take_precedence() {
    for config in [
        BertConfig::bge_small(),
        BertConfig::bge_base(),
        BertConfig::bge_large(),
    ] {
        config.validate().unwrap();
    }
    let json = format!("{{{FIELDS}}}")
        .replace("\"hidden_size\":8", "\"hidden_size\":0")
        .replace("\"layer_norm_eps\":1e-12", "\"layer_norm_eps\":null");
    assert!(
        BertConfig::from_json_str(&json)
            .unwrap_err()
            .to_string()
            .contains("invalid f32 for layer_norm_eps:")
    );
}

#[test]
fn direct_validation_error_has_no_source_claim() {
    let mut config = BertConfig::bge_small();
    config.hidden_size = 0;
    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("hidden_size must be > 0"), "{error}");
    assert!(!error.contains("config.json"), "{error}");
    assert!(!error.contains("safetensors"), "{error}");
}
