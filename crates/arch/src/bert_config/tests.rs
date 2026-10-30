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
            .unwrap()
            .layer_norm_eps
            .is_nan()
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
