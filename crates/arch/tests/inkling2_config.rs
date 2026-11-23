use lattice_arch::inkling2_config::{AttentionSpan, HeadGeometry, parse_config, parse_config_json};
use serde_json::{Value, json};

const FIXTURE: &str = include_str!("fixtures/inkling2_mini_config.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).unwrap()
}

fn text(value: &mut Value) -> &mut serde_json::Map<String, Value> {
    value["text_config"].as_object_mut().unwrap()
}

fn rejects(value: &Value, expected: &str) {
    let error = parse_config_json(&value.to_string())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(expected),
        "expected {expected:?}, got {error}"
    );
}

fn rejects_text_field(field: &str, replacement: Value, expected: &str) {
    let mut value = fixture();
    text(&mut value).insert(field.to_owned(), replacement);
    rejects(&value, expected);
}

#[test]
fn pinned_config_parses_with_its_declared_geometry() {
    let config = parse_config_json(FIXTURE).unwrap();
    let text = &config.text;
    assert_eq!(
        (
            text.hidden_size,
            text.num_hidden_layers,
            text.vocab_size,
            text.unpadded_vocab_size
        ),
        (2304, 24, 201_024, 200_058)
    );
    let expected_heads = HeadGeometry {
        num_attention_heads: 24,
        num_key_value_heads: 8,
        head_dim: 128,
    };
    assert_eq!(text.global_heads, expected_heads);
    assert_eq!(text.sliding_heads, expected_heads);
    assert_eq!(
        (
            text.sliding_window_size,
            text.d_rel,
            text.rel_extent,
            text.log_scaling_n_floor
        ),
        (1024, 16, 1024, 8192)
    );
    assert_eq!(
        (
            text.n_routed_experts,
            text.num_experts_per_tok,
            text.moe_intermediate_size
        ),
        (128, 8, 1152)
    );
    assert_eq!(text.conv_kernel_size, 4);
    assert_eq!(text.n_layers_per_attn_res_block, 6);
    assert_eq!(text.max_position_embeddings, 1_048_576);
    assert_eq!((text.log_scaling_alpha, text.rms_norm_eps), (0.1, 1e-6));
    // serde_json is built without `float_roundtrip`, so a 17-digit decimal may land one ulp
    // away from the correctly rounded f64.
    let threshold = 22.627416997969522_f64;
    assert!((text.qk_l2_clip_threshold - threshold).abs() <= f64::EPSILON * threshold);
    assert_eq!(
        (
            text.rel_l2_clip_threshold,
            text.final_logit_softcapping,
            text.mlp_gate_softcap,
            text.mlp_up_softcap,
            text.route_scale
        ),
        (8.0, 30.0, 4.0, 25.0, 8.0)
    );
    assert_eq!(
        (
            text.loop_config.core_start,
            text.loop_config.core_end,
            text.loop_config.num_passes,
            text.loop_config.gate_rank
        ),
        (6, 18, 3, 64)
    );
    assert_eq!(text.total_layer_applications(), 48);
    let global: Vec<usize> = (0..24)
        .filter(|&layer| text.span(layer) == Some(AttentionSpan::Global))
        .collect();
    assert_eq!(global, [5, 11, 17, 23]);
    assert_eq!(text.span(24), None);
    let starts: Vec<usize> = (0..24)
        .filter(|&layer| text.starts_attention_residual_block(layer))
        .collect();
    assert_eq!(starts, [0, 6, 12, 18]);
    assert_eq!(text.relative_extent(AttentionSpan::Global), 1024);
    assert_eq!(text.relative_extent(AttentionSpan::Sliding), 1024);
    assert_eq!(config.eos_token_id, 200_006);
    assert_eq!(config.image_token_id, 200_054);
    assert_eq!(
        config.unparsed_sections,
        ["vision_config", "mtp_config", "quantization_config"]
    );
}

#[test]
fn parser_reads_the_bounded_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::write(&path, FIXTURE).unwrap();
    assert_eq!(
        parse_config(&path).unwrap(),
        parse_config_json(FIXTURE).unwrap()
    );
}

#[test]
fn spans_and_extents_follow_their_own_fields() {
    let mut value = fixture();
    let fields = text(&mut value);
    fields.insert("swa_num_attention_heads".into(), json!(16));
    fields.insert("swa_num_key_value_heads".into(), json!(4));
    fields.insert("swa_head_dim".into(), json!(64));
    fields.insert("sliding_window_size".into(), json!(512));
    fields.insert("rel_extent".into(), json!(2048));
    fields.remove("local_layer_ids");
    let text = parse_config_json(&value.to_string()).unwrap().text;
    assert_eq!(
        text.heads(AttentionSpan::Sliding),
        HeadGeometry {
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 64,
        }
    );
    assert_eq!(text.heads(AttentionSpan::Global).head_dim, 128);
    assert_eq!(text.relative_extent(AttentionSpan::Sliding), 512);
    assert_eq!(text.relative_extent(AttentionSpan::Global), 2048);
}

#[test]
fn loop_length_changes_the_total_applications() {
    let mut value = fixture();
    value["text_config"]["loop_config"]["core_end"] = json!(12);
    value["text_config"]["loop_config"]["num_passes"] = json!(4);
    let text = parse_config_json(&value.to_string()).unwrap().text;
    assert_eq!(text.total_layer_applications(), 24 + 3 * 6);
}

#[test]
fn unknown_fields_are_refused_at_every_level() {
    let mut value = fixture();
    value["rope_scaling"] = json!({});
    rejects(&value, "unknown field `rope_scaling`");
    rejects_text_field("rope_theta", json!(10000.0), "unknown field `rope_theta`");
    let mut value = fixture();
    value["text_config"]["loop_config"]["exit_threshold"] = json!(0.5);
    rejects(&value, "unknown field `exit_threshold`");
}

#[test]
fn canonical_names_replace_the_aliases_but_never_join_them() {
    for (alias, canonical) in [
        ("model_max_length", "max_position_embeddings"),
        ("sconv_kernel_size", "conv_kernel_size"),
        ("intermediate_size", "moe_intermediate_size"),
        ("route_scale", "routed_scaling_factor"),
    ] {
        let mut value = fixture();
        let fields = text(&mut value);
        let moved = fields.remove(alias).unwrap();
        fields.insert(canonical.to_owned(), moved.clone());
        assert_eq!(
            parse_config_json(&value.to_string()).unwrap(),
            parse_config_json(FIXTURE).unwrap(),
            "{canonical} in place of {alias}"
        );
        let mut value = fixture();
        text(&mut value).insert(canonical.to_owned(), moved);
        rejects(&value, "duplicate field");
    }
}

#[test]
fn duplicate_json_keys_are_refused() {
    let doubled = FIXTURE.replacen(
        "\"hidden_size\": 2304,",
        "\"hidden_size\": 2304, \"hidden_size\": 4096,",
        1,
    );
    assert_ne!(doubled, FIXTURE);
    let error = parse_config_json(&doubled).unwrap_err().to_string();
    assert!(error.contains("duplicate key hidden_size"), "{error}");
}

#[test]
fn fields_the_model_does_not_read_must_hold_the_implemented_value() {
    for (field, wrong, expected) in [
        ("q_bias", json!(true), "q_bias = true is not supported"),
        ("o_bias", json!(true), "o_bias = true is not supported"),
        (
            "use_value_norm_no_gamma",
            json!(false),
            "use_value_norm_no_gamma = false is not supported",
        ),
        (
            "attn_res_normalize_values",
            json!(false),
            "attn_res_normalize_values = false is not supported",
        ),
        (
            "use_gate_bias",
            json!(false),
            "use_gate_bias = false is not supported",
        ),
        ("n_shared_experts", json!(1), "n_shared_experts must be 0"),
        (
            "scoring_func",
            json!("softmax"),
            "scoring_func \"softmax\" is not supported",
        ),
        (
            "hidden_act",
            json!("gelu"),
            "hidden_act \"gelu\" is not supported",
        ),
    ] {
        rejects_text_field(field, wrong, expected);
    }
    let mut value = fixture();
    value["text_config"]["loop_config"]["is_varloop"] = json!(true);
    rejects(&value, "loop_config.is_varloop = true is not supported");
}

#[test]
fn implemented_values_may_be_stated_or_omitted() {
    let mut value = fixture();
    let fields = text(&mut value);
    for field in [
        "q_bias",
        "o_bias",
        "use_value_norm_no_gamma",
        "attn_res_normalize_values",
        "use_gate_bias",
        "n_shared_experts",
        "local_layer_ids",
        "model_type",
        "torch_dtype",
    ] {
        assert!(fields.remove(field).is_some(), "{field}");
    }
    fields.insert("scoring_func".into(), json!("sigmoid"));
    fields.insert("hidden_act".into(), json!("silu"));
    value["text_config"]["loop_config"]
        .as_object_mut()
        .unwrap()
        .remove("is_varloop");
    assert_eq!(
        parse_config_json(&value.to_string()).unwrap(),
        parse_config_json(FIXTURE).unwrap()
    );
}

#[test]
fn layer_lists_must_match_the_every_sixth_layer_rule() {
    let mut ids: Vec<usize> = (0..24).filter(|layer| (layer + 1) % 6 != 0).collect();
    ids.push(23);
    rejects_text_field(
        "local_layer_ids",
        json!(ids),
        "local_layer_ids must list exactly",
    );
    let shifted: Vec<usize> = (0..24).filter(|layer| layer % 6 != 0).collect();
    rejects_text_field(
        "local_layer_ids",
        json!(shifted),
        "local_layer_ids must list exactly",
    );

    let types: Vec<&str> = (0..24)
        .map(|layer| {
            if (layer + 1) % 6 == 0 {
                "hybrid"
            } else {
                "hybrid_sliding"
            }
        })
        .collect();
    let mut value = fixture();
    text(&mut value).insert("layer_types".into(), json!(types));
    assert_eq!(
        parse_config_json(&value.to_string()).unwrap(),
        parse_config_json(FIXTURE).unwrap()
    );
    let mut wrong = types.clone();
    wrong.swap(4, 5);
    rejects_text_field(
        "layer_types",
        json!(wrong),
        "layer_types must place full attention",
    );
    rejects_text_field(
        "layer_types",
        json!(types[..23]),
        "layer_types must place full attention",
    );
}

#[test]
fn loop_bounds_are_enforced() {
    for (field, value, expected) in [
        ("num_passes", 0, "num_passes must be between 1 and 64"),
        ("num_passes", 65, "num_passes must be between 1 and 64"),
        ("core_start", 0, "core_start must be positive"),
        (
            "core_start",
            18,
            "core_start < core_end <= num_hidden_layers",
        ),
        ("core_end", 30, "core_start < core_end <= num_hidden_layers"),
        (
            "core_start",
            5,
            "must be multiples of n_layers_per_attn_res_block",
        ),
        (
            "core_end",
            17,
            "must be multiples of n_layers_per_attn_res_block",
        ),
        ("gate_rank", 0, "loop_config.gate_rank must be positive"),
    ] {
        let mut config = fixture();
        config["text_config"]["loop_config"][field] = json!(value);
        rejects(&config, expected);
    }
    let mut config = fixture();
    text(&mut config).remove("loop_config");
    rejects(&config, "missing field `loop_config`");
}

#[test]
fn geometry_is_validated() {
    rejects_text_field("hidden_size", json!(0), "hidden_size must be positive");
    rejects_text_field("d_rel", json!(0), "d_rel must be positive");
    rejects_text_field(
        "hidden_size",
        json!(16_777_217),
        "hidden_size exceeds the supported geometry",
    );
    rejects_text_field(
        "num_key_value_heads",
        json!(7),
        "num_attention_heads must be a multiple of num_key_value_heads",
    );
    rejects_text_field(
        "swa_num_key_value_heads",
        json!(5),
        "swa_num_attention_heads must be a multiple of swa_num_key_value_heads",
    );
    rejects_text_field(
        "unpadded_vocab_size",
        json!(201_025),
        "unpadded_vocab_size exceeds vocab_size",
    );
    rejects_text_field(
        "num_experts_per_tok",
        json!(129),
        "num_experts_per_tok exceeds n_routed_experts",
    );
    rejects_text_field("sconv_kernel_size", json!(65), "conv_kernel_size exceeds");
    rejects_text_field(
        "rms_norm_eps",
        json!(-1.0),
        "rms_norm_eps must be finite and positive",
    );
    rejects_text_field(
        "final_logit_softcapping",
        json!(0.0),
        "final_logit_softcapping must be finite and positive",
    );
    rejects_text_field(
        "route_scale",
        json!(-8.0),
        "route_scale must be finite and positive",
    );
    let mut value = fixture();
    text(&mut value).remove("head_dim");
    rejects(&value, "missing field `head_dim`");
}

#[test]
fn token_ids_must_fall_inside_the_unpadded_vocabulary() {
    let mut value = fixture();
    value["eos_token_id"] = json!(200_058);
    rejects(&value, "eos_token_id outside the unpadded vocabulary");
    let mut value = fixture();
    value["image_token_id"] = json!(200_058);
    rejects(&value, "image_token_id outside the unpadded vocabulary");
    let mut value = fixture();
    value["image_token_id"] = json!(7);
    assert_eq!(
        parse_config_json(&value.to_string())
            .unwrap()
            .image_token_id,
        7
    );
}

#[test]
fn identity_and_vision_width_are_checked() {
    let mut value = fixture();
    value["model_type"] = json!("inkling2");
    rejects(&value, "unexpected model_type or architecture");
    let mut value = fixture();
    value["architectures"] = json!(["Inkling2ForConditionalGeneration"]);
    rejects(&value, "unexpected model_type or architecture");
    rejects_text_field(
        "model_type",
        json!("inkling2_text"),
        "unexpected text_config.model_type",
    );

    let mut value = fixture();
    value["vision_config"]["text_hidden_size"] = json!(2048);
    rejects(&value, "does not match text_config.hidden_size");
    let mut value = fixture();
    value["vision_config"]
        .as_object_mut()
        .unwrap()
        .remove("text_hidden_size");
    rejects(&value, "vision_config.text_hidden_size is missing");
    let mut value = fixture();
    let root = value.as_object_mut().unwrap();
    root.remove("vision_config");
    root.remove("mtp_config");
    let config = parse_config_json(&value.to_string()).unwrap();
    assert_eq!(config.unparsed_sections, ["quantization_config"]);
    assert_eq!(config.text, parse_config_json(FIXTURE).unwrap().text);
}
