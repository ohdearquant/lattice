use super::*;

/// The shipped `text_config`, reduced to the keys this parser reads.
const SHIPPED: &str = r#"{
      "model_type": "embedding_gemma2",
      "text_config": {
        "attention_bias": false,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "embedding_dim": 768,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 512,
        "hidden_size_per_layer_input": 512,
        "intermediate_size": 2048,
        "layer_types": [
          "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
          "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
        ],
        "num_attention_heads": 4,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "per_layer_config": {
          "05": {"head_dim": 512, "num_key_value_heads": 1},
          "11": {"head_dim": 512, "num_key_value_heads": 1},
          "17": {"head_dim": 512, "num_key_value_heads": 1},
          "23": {"head_dim": 512, "num_key_value_heads": 1}
        },
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
          "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
          "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "sliding_window": 512,
        "vocab_size": 262144
      }
    }"#;

fn shipped_value() -> serde_json::Value {
    serde_json::from_str(SHIPPED).expect("fixture is valid JSON")
}

fn parse(value: &serde_json::Value) -> Result<EmbeddingGemma2Config, InferenceError> {
    EmbeddingGemma2Config::from_config_json_str(&value.to_string())
}

#[test]
fn parses_the_shipped_text_config() {
    let cfg = parse(&shipped_value()).expect("shipped config parses");
    assert_eq!(cfg.hidden_size, 512);
    assert_eq!(cfg.embedding_dim, 768);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.sliding_window, 512);
    assert_eq!(cfg.layer_types.len(), 24);
    let full: Vec<usize> = (0..24)
        .filter(|&i| cfg.layer_types[i] == EmbeddingGemma2LayerKind::Full)
        .collect();
    assert_eq!(full, vec![5, 11, 17, 23]);
    for i in 0..24 {
        let shape = cfg.layer_shapes[i];
        if full.contains(&i) {
            assert_eq!((shape.head_dim, shape.num_key_value_heads), (512, 1));
        } else {
            assert_eq!((shape.head_dim, shape.num_key_value_heads), (256, 2));
        }
    }
    assert_eq!(cfg.rope_theta_sliding, 10_000.0);
    assert_eq!(cfg.rope_theta_full, 1_000_000.0);
    assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
    assert_eq!(cfg.embed_scale, 512f32.sqrt());
    assert_eq!((cfg.bos_token_id, cfg.eos_token_id), (2, 1));
}

#[test]
fn a_config_without_the_text_config_wrapper_is_read_as_the_text_config() {
    let inner = shipped_value()["text_config"].clone();
    assert_eq!(
        parse(&inner).expect("bare text config parses"),
        parse(&shipped_value()).expect("wrapped config parses")
    );
}

#[test]
fn a_missing_required_field_is_named() {
    let mut value = shipped_value();
    value["text_config"]
        .as_object_mut()
        .expect("object")
        .remove("sliding_window");
    let err = parse(&value).expect_err("missing field must fail");
    assert!(err.to_string().contains("sliding_window"), "{err}");
}

#[test]
fn unsupported_values_are_rejected() {
    let cases: [(&str, serde_json::Value); 4] = [
        ("hidden_activation", serde_json::json!("gelu")),
        ("attention_bias", serde_json::json!(true)),
        ("num_key_value_heads", serde_json::json!(3)),
        ("head_dim", serde_json::json!(255)),
    ];
    for (key, replacement) in cases {
        let mut value = shipped_value();
        value["text_config"][key] = replacement;
        assert!(parse(&value).is_err(), "{key} change must be rejected");
    }
    let mut value = shipped_value();
    value["text_config"]["rope_parameters"]["full_attention"]["rope_type"] =
        serde_json::json!("yarn");
    assert!(
        parse(&value).is_err(),
        "non-default rope type must be rejected"
    );
}

#[test]
fn per_layer_config_outside_the_layer_range_is_rejected() {
    let mut value = shipped_value();
    value["text_config"]["per_layer_config"]["24"] =
        serde_json::json!({"head_dim": 512, "num_key_value_heads": 1});
    assert!(parse(&value).is_err());
}

#[test]
fn mixed_geometry_within_one_layer_type_is_rejected() {
    let mut value = shipped_value();
    value["text_config"]["per_layer_config"]["03"] =
        serde_json::json!({"head_dim": 128, "num_key_value_heads": 2});
    assert!(parse(&value).is_err());
}

#[test]
fn the_last_layer_is_forced_to_full_attention() {
    let mut value = shipped_value();
    value["text_config"]["layer_types"][23] = serde_json::json!("sliding_attention");
    let cfg = parse(&value).expect("parses");
    assert_eq!(cfg.layer_types[23], EmbeddingGemma2LayerKind::Full);
}

#[test]
fn special_token_ids_must_be_single_in_range_integers() {
    for bad in [
        serde_json::json!([1, 106]),
        serde_json::json!(-1),
        serde_json::json!(262144),
        serde_json::json!("2"),
    ] {
        let mut value = shipped_value();
        value["text_config"]["eos_token_id"] = bad.clone();
        assert!(
            parse(&value).is_err(),
            "eos_token_id {bad} must be rejected"
        );
    }
    let mut value = shipped_value();
    let text = value["text_config"].as_object_mut().expect("object");
    text.remove("bos_token_id");
    text.remove("eos_token_id");
    let cfg = parse(&value).expect("absent ids take the reference defaults");
    assert_eq!((cfg.bos_token_id, cfg.eos_token_id), (2, 1));
}
