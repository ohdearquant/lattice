use super::*;
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gemma4")
        .join("e2b_config.json")
}

fn pinned_config_json() -> String {
    std::fs::read_to_string(fixture_path()).expect("read committed pinned target config.json")
}

fn pinned_config_value() -> serde_json::Value {
    serde_json::from_str(&pinned_config_json()).expect("pinned fixture is valid JSON")
}

#[test]
fn e2b_preset_matches_adr_082_dimensions() {
    let cfg = Gemma4Config::e2b();
    assert_eq!(cfg.num_hidden_layers, 35);
    assert_eq!(cfg.hidden_size, 1536);
    assert_eq!(cfg.vocab_size, 262_144);
    assert_eq!(cfg.intermediate_size, 6144);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.num_key_value_heads, 1);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.global_head_dim, 512);
    assert_eq!(cfg.sliding_window, 512);
    assert_eq!(cfg.num_kv_shared_layers, 20);
    assert_eq!(cfg.hidden_size_per_layer_input, 256);
    assert!(cfg.tie_word_embeddings);
    assert!(cfg.validate().is_ok());
}

#[test]
fn e2b_preset_layer_schedule_matches_g3() {
    let cfg = Gemma4Config::e2b();
    let global_indices: Vec<usize> = (0..cfg.num_hidden_layers)
        .filter(|&i| cfg.is_global_layer(i))
        .collect();
    assert_eq!(global_indices, vec![4, 9, 14, 19, 24, 29, 34]);
}

#[test]
fn e2b_preset_kv_sharing_matches_amendment_1() {
    let cfg = Gemma4Config::e2b();
    assert_eq!(cfg.first_kv_shared_layer_idx(), 15);
    let shared: Vec<usize> = (0..cfg.num_hidden_layers)
        .filter(|&i| cfg.is_kv_shared_layer(i))
        .collect();
    assert_eq!(shared.len(), 20);
    assert_eq!(shared[0], 15);
    assert_eq!(*shared.last().unwrap(), 34);
}

#[test]
fn use_double_wide_mlp_matches_kv_shared_set() {
    let cfg = Gemma4Config::e2b();
    for i in 0..cfg.num_hidden_layers {
        assert_eq!(cfg.use_double_wide_mlp(i), cfg.is_kv_shared_layer(i));
    }
    assert_eq!(cfg.mlp_intermediate_size(0), 6144);
    assert_eq!(cfg.mlp_intermediate_size(15), 12288);
    assert_eq!(cfg.mlp_intermediate_size(34), 12288);
    assert_eq!(cfg.mlp_intermediate_size(14), 6144);
}

#[test]
fn attn_head_dim_follows_layer_type_not_sharing() {
    let cfg = Gemma4Config::e2b();
    // Layer 4: global, non-shared -> global_head_dim.
    assert_eq!(cfg.attn_head_dim(4), 512);
    // Layer 15: sliding, shared -> head_dim (width is keyed on type, not sharing).
    assert_eq!(cfg.attn_head_dim(15), 256);
    // Layer 24: global, shared -> global_head_dim.
    assert_eq!(cfg.attn_head_dim(24), 512);
    // Layer 0: sliding, non-shared -> head_dim.
    assert_eq!(cfg.attn_head_dim(0), 256);
}

#[test]
fn pinned_config_fixture_parses_and_matches_adr_082() {
    let cfg = Gemma4Config::from_config_json_str(&pinned_config_json())
        .expect("committed pinned google/gemma-4-E2B-it config.json must parse");
    assert_eq!(cfg.hidden_size, 1536);
    assert_eq!(cfg.num_hidden_layers, 35);
    assert_eq!(cfg.vocab_size, 262_144);
    assert_eq!(cfg.intermediate_size, 6144);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.num_key_value_heads, 1);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.global_head_dim, 512);
    assert_eq!(cfg.sliding_window, 512);
    assert_eq!(cfg.num_kv_shared_layers, 20);
    assert_eq!(cfg.hidden_size_per_layer_input, 256);
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.rope_theta, 1_000_000.0);
    assert_eq!(cfg.rope_local_base_freq, 10_000.0);
    assert_eq!(cfg.partial_rotary_factor, 0.25);
    assert!(cfg.use_double_wide_mlp_raw);
    assert_eq!(cfg.hidden_activation, "gelu_pytorch_tanh");
    assert_eq!(cfg.final_logit_softcapping, 30.0);
    assert!(!cfg.attention_k_eq_v);
    assert!(!cfg.attention_bias);
    let global_indices: Vec<usize> = (0..cfg.num_hidden_layers)
        .filter(|&i| cfg.is_global_layer(i))
        .collect();
    assert_eq!(global_indices, vec![4, 9, 14, 19, 24, 29, 34]);
}

#[test]
fn absent_text_config_errors_naming_field() {
    let err = Gemma4Config::from_config_json_str("{}")
        .expect_err("an absent text_config must yield an InferenceError, not the E2B preset");
    assert!(
        err.to_string().contains("text_config"),
        "error must name text_config: {err}"
    );
}

#[test]
fn partial_text_config_errors_naming_missing_field() {
    // A text_config present but missing a forward-relevant field (here
    // rope_parameters) must be rejected naming that field, not silently
    // completed from the E2B preset.
    let json = r#"{"text_config": {"hidden_size": 1536}}"#;
    let err = Gemma4Config::from_config_json_str(json)
        .expect_err("a partial text_config must yield an InferenceError");
    assert!(
        err.to_string().contains("rope_parameters")
            || err.to_string().contains("num_hidden_layers"),
        "error must name a missing required field: {err}"
    );
}

#[test]
fn missing_nested_rope_full_attention_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["rope_parameters"]
        .as_object_mut()
        .unwrap()
        .remove("full_attention");
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("a missing rope_parameters.full_attention must yield an InferenceError");
    assert!(
        err.to_string().contains("full_attention"),
        "error must name full_attention: {err}"
    );
}

#[test]
fn missing_nested_rope_sliding_attention_theta_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["rope_parameters"]["sliding_attention"]
        .as_object_mut()
        .unwrap()
        .remove("rope_theta");
    let err = Gemma4Config::from_config_json_str(&json.to_string()).expect_err(
        "a missing rope_parameters.sliding_attention.rope_theta must yield an InferenceError",
    );
    assert!(
        err.to_string().contains("rope_theta"),
        "error must name rope_theta: {err}"
    );
}

#[test]
fn nested_rope_theta_mutation_is_observed_not_ignored() {
    let mut json = pinned_config_value();
    json["text_config"]["rope_parameters"]["full_attention"]["rope_theta"] =
        serde_json::json!(42.0);
    let cfg = Gemma4Config::from_config_json_str(&json.to_string())
        .expect("a plausible rope_theta mutation still parses");
    assert_eq!(
        cfg.rope_theta, 42.0,
        "nested rope_parameters.full_attention.rope_theta must flow through, not be \
             silently replaced by the E2B preset's 1_000_000.0"
    );
}

#[test]
fn use_double_wide_mlp_false_errors_naming_both_fields() {
    let mut json = pinned_config_value();
    json["text_config"]["use_double_wide_mlp"] = serde_json::json!(false);
    let err = Gemma4Config::from_config_json_str(&json.to_string()).expect_err(
        "use_double_wide_mlp=false while KV-shared layers exist must yield an InferenceError",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("use_double_wide_mlp"),
        "must name use_double_wide_mlp: {msg}"
    );
    assert!(
        msg.contains("is_kv_shared_layer") || msg.contains("num_kv_shared_layers"),
        "must name the KV-shared-layer field: {msg}"
    );
}

#[test]
fn odd_head_dim_errors_naming_field() {
    // gemma4_apply_rope's stride-half pairing requires even widths; an
    // odd head_dim must die here with a typed error, not deep in the
    // forward pass at the op-level assert.
    let mut json = pinned_config_value();
    json["text_config"]["head_dim"] = serde_json::json!(255);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("an odd head_dim must yield an InferenceError");
    let msg = err.to_string();
    assert!(
        msg.contains("head_dim (255)") && msg.contains("even"),
        "error must name head_dim and evenness: {msg}"
    );
}

#[test]
fn odd_global_head_dim_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["global_head_dim"] = serde_json::json!(511);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("an odd global_head_dim must yield an InferenceError");
    let msg = err.to_string();
    assert!(
        msg.contains("global_head_dim (511)") && msg.contains("even"),
        "error must name global_head_dim and evenness: {msg}"
    );
}

#[test]
fn wrong_hidden_activation_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["hidden_activation"] = serde_json::json!("gelu");
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("an unsupported hidden_activation must yield an InferenceError");
    assert!(
        err.to_string().contains("hidden_activation"),
        "error must name hidden_activation: {err}"
    );
}

#[test]
fn wrong_final_logit_softcapping_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["final_logit_softcapping"] = serde_json::json!(50.0);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("an unsupported final_logit_softcapping must yield an InferenceError");
    assert!(
        err.to_string().contains("final_logit_softcapping"),
        "error must name final_logit_softcapping: {err}"
    );
}

#[test]
fn tie_word_embeddings_false_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["tie_word_embeddings"] = serde_json::json!(false);
    let err = Gemma4Config::from_config_json_str(&json.to_string()).expect_err(
        "tie_word_embeddings=false must yield an InferenceError: this loader never reads \
             a distinct lm_head.weight",
    );
    assert!(
        err.to_string().contains("tie_word_embeddings"),
        "error must name tie_word_embeddings: {err}"
    );
}

#[test]
fn attention_k_eq_v_true_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["attention_k_eq_v"] = serde_json::json!(true);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("attention_k_eq_v=true must yield an InferenceError for this checkpoint");
    assert!(
        err.to_string().contains("attention_k_eq_v"),
        "error must name attention_k_eq_v: {err}"
    );
}

#[test]
fn attention_bias_true_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["attention_bias"] = serde_json::json!(true);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("attention_bias=true must yield an InferenceError for this checkpoint");
    assert!(
        err.to_string().contains("attention_bias"),
        "error must name attention_bias: {err}"
    );
}

#[test]
fn layer_types_length_34_errors_naming_field() {
    let mut json = pinned_config_value();
    let arr = json["text_config"]["layer_types"].as_array_mut().unwrap();
    arr.pop();
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("a length-34 layer_types must yield an InferenceError");
    assert!(
        err.to_string().contains("layer_types"),
        "error must name layer_types: {err}"
    );
}

#[test]
fn layer_types_length_36_errors_naming_field() {
    let mut json = pinned_config_value();
    let arr = json["text_config"]["layer_types"].as_array_mut().unwrap();
    arr.push(serde_json::json!("sliding_attention"));
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("a length-36 layer_types must yield an InferenceError");
    assert!(
        err.to_string().contains("layer_types"),
        "error must name layer_types: {err}"
    );
}

#[test]
fn layer_types_wrong_first_global_index_errors_naming_field() {
    let mut json = pinned_config_value();
    let arr = json["text_config"]["layer_types"].as_array_mut().unwrap();
    // Move the first global layer from index 4 to index 3.
    arr[3] = serde_json::json!("full_attention");
    arr[4] = serde_json::json!("sliding_attention");
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("a layer_types with the wrong first global index must be rejected");
    assert!(
        err.to_string().contains("layer_types"),
        "error must name layer_types: {err}"
    );
}

#[test]
fn layer_types_wrong_final_layer_type_errors_naming_field() {
    let mut json = pinned_config_value();
    let arr = json["text_config"]["layer_types"].as_array_mut().unwrap();
    // Layer 34 must be full_attention (final, forced global); flip it.
    let last = arr.len() - 1;
    arr[last] = serde_json::json!("sliding_attention");
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("a layer_types with the wrong final layer type must be rejected");
    assert!(
        err.to_string().contains("layer_types"),
        "error must name layer_types: {err}"
    );
}

#[test]
fn num_kv_shared_layers_exceeding_total_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["num_kv_shared_layers"] = serde_json::json!(36);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("num_kv_shared_layers > num_hidden_layers must yield an InferenceError");
    assert!(err.to_string().contains("num_kv_shared_layers"));
}

#[test]
fn zero_head_dim_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["head_dim"] = serde_json::json!(0);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("head_dim: 0 must yield an InferenceError");
    assert!(err.to_string().contains("head_dim"));
}

#[test]
fn zero_global_head_dim_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["global_head_dim"] = serde_json::json!(0);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("global_head_dim: 0 must yield an InferenceError");
    assert!(err.to_string().contains("global_head_dim"));
}

#[test]
fn partial_rotary_factor_zero_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] =
        serde_json::json!(0.0);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("partial_rotary_factor: 0.0 must yield an InferenceError");
    assert!(err.to_string().contains("partial_rotary_factor"));
}

#[test]
fn partial_rotary_factor_above_one_errors_naming_field() {
    let mut json = pinned_config_value();
    json["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] =
        serde_json::json!(1.5);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("partial_rotary_factor > 1.0 must yield an InferenceError");
    assert!(err.to_string().contains("partial_rotary_factor"));
}

#[test]
fn from_model_dir_errors_on_missing_config_json() {
    let tmp = tempfile::tempdir().unwrap();
    let err = Gemma4Config::from_model_dir(tmp.path())
        .expect_err("a directory with no config.json must be a hard error");
    let msg = err.to_string();
    assert!(msg.contains("config.json"));
    assert!(msg.contains(&tmp.path().display().to_string()));
}

#[test]
fn from_model_dir_loads_a_real_config_json() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("config.json"), pinned_config_json()).unwrap();
    let cfg = Gemma4Config::from_model_dir(tmp.path())
        .expect("a directory with a valid config.json must load");
    assert_eq!(cfg.hidden_size, 1536);
    assert_eq!(cfg.num_hidden_layers, 35);
}

// --- resolve_stop_token_ids (issue #1597: generate()/generate_with_trace()/
// generate_streaming_with_cancel() must stop on the checkpoint's real end-of-turn id, not
// only text_config's single eos_token_id) ---

#[test]
fn stop_ids_use_generation_config_list_when_present() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("config.json"), pinned_config_json()).unwrap();
    std::fs::write(
        tmp.path().join("generation_config.json"),
        serde_json::json!({"eos_token_id": [1, 106, 50]}).to_string(),
    )
    .unwrap();
    let ids = resolve_stop_token_ids(tmp.path(), 1);
    assert_eq!(ids, vec![1, 50, 106]);
}

#[test]
fn stop_ids_fall_back_to_top_level_config_json_list() {
    let tmp = tempfile::tempdir().unwrap();
    // Only config.json (top-level eos_token_id: [1, 106], per the pinned fixture) --
    // no generation_config.json in this directory at all.
    std::fs::write(tmp.path().join("config.json"), pinned_config_json()).unwrap();
    let ids = resolve_stop_token_ids(tmp.path(), 1);
    assert_eq!(ids, vec![1, 106]);
}

#[test]
fn stop_ids_fall_back_to_text_config_alone_when_neither_file_has_one() {
    let tmp = tempfile::tempdir().unwrap();
    let mut json = pinned_config_value();
    json.as_object_mut().unwrap().remove("eos_token_id");
    std::fs::write(tmp.path().join("config.json"), json.to_string()).unwrap();
    let ids = resolve_stop_token_ids(tmp.path(), 1);
    assert_eq!(ids, vec![1]);
}

#[test]
fn stop_ids_ignore_a_malformed_generation_config_and_fall_back() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("config.json"), pinned_config_json()).unwrap();
    // Not valid JSON at all.
    std::fs::write(tmp.path().join("generation_config.json"), "{not json").unwrap();
    let ids = resolve_stop_token_ids(tmp.path(), 1);
    assert_eq!(
        ids,
        vec![1, 106],
        "a malformed generation_config.json must be treated as absent, falling back to \
             config.json's top-level eos_token_id, not as a hard error"
    );
}

#[test]
fn stop_ids_ignore_a_generation_config_with_no_eos_field_and_fall_back() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("config.json"), pinned_config_json()).unwrap();
    std::fs::write(
        tmp.path().join("generation_config.json"),
        serde_json::json!({"max_length": 20}).to_string(),
    )
    .unwrap();
    let ids = resolve_stop_token_ids(tmp.path(), 1);
    assert_eq!(ids, vec![1, 106]);
}

#[test]
fn stop_ids_accept_a_single_integer_eos_token_id() {
    // 50 is absent from the top-level config.json fallback ([1, 106]), so
    // only the generation_config int path can produce it.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("config.json"), pinned_config_json()).unwrap();
    std::fs::write(
        tmp.path().join("generation_config.json"),
        serde_json::json!({"eos_token_id": 50}).to_string(),
    )
    .unwrap();
    let ids = resolve_stop_token_ids(tmp.path(), 1);
    assert_eq!(ids, vec![1, 50]);
}
