//! Gemma 4 E2B admission preflight (ADR-090 Gemma admission, issue #1598):
//! the loader must reject a non-null `use_bidirectional_attention` before
//! constructing a [`Gemma4Config`] or touching any weight file, admit the
//! pinned E2B fixture's absent/null value unchanged, and keep pinning the
//! existing zero-per-layer-embedding-width rejection (ADR-082 G9). Default
//! features, no checkpoint required -- every case here parses the committed
//! fixture (optionally mutated) or a config-only temporary directory.

use lattice_inference::model::gemma4_config::Gemma4Config;
use lattice_inference::model::gemma4_model::Gemma4Model;
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

/// A different attention profile (upstream Gemma variants that set
/// `use_bidirectional_attention` to a real value) must not silently load as
/// the pinned E2B geometry. Two shapes stand in for "any non-null value":
/// a string (the kind of tag such a variant might carry) and a bare bool
/// (the simplest non-null JSON shape). Everything else in the fixture,
/// including a positive `hidden_size_per_layer_input`, is left unchanged --
/// per ADR-090, a positive-PLE check alone cannot serve as an independent
/// stand-in for this admission gate.
#[test]
fn use_bidirectional_attention_non_null_is_rejected_naming_the_field() {
    for mode in [serde_json::json!("vision"), serde_json::json!(true)] {
        let mut json = pinned_config_value();
        json["text_config"]["use_bidirectional_attention"] = mode.clone();
        assert_eq!(
            json["text_config"]["hidden_size_per_layer_input"],
            serde_json::json!(256),
            "mutation must leave the pinned E2B PLE width untouched"
        );

        let err = Gemma4Config::from_config_json_str(&json.to_string()).expect_err(&format!(
            "use_bidirectional_attention={mode} must be rejected"
        ));
        assert!(
            err.to_string().contains("use_bidirectional_attention"),
            "error must name use_bidirectional_attention for mode {mode}: {err}"
        );
    }
}

/// The pinned E2B fixture ships `use_bidirectional_attention: null`, and a
/// checkpoint that omits the key entirely must be treated identically (the
/// raw field is `#[serde(default)]`) -- both are the one supported profile.
#[test]
fn use_bidirectional_attention_null_or_absent_is_admitted() {
    Gemma4Config::from_config_json_str(&pinned_config_json())
        .expect("the pinned E2B fixture (use_bidirectional_attention: null) must admit");

    let mut json = pinned_config_value();
    json["text_config"]
        .as_object_mut()
        .expect("text_config is a JSON object")
        .remove("use_bidirectional_attention");
    Gemma4Config::from_config_json_str(&json.to_string())
        .expect("an absent use_bidirectional_attention key must admit");
}

/// Pins the existing `hidden_size_per_layer_input == 0` rejection
/// (`gemma4_config.rs`'s per-layer-embedding-width guard) as part of the
/// same admission contract, disclosed as already passing before this PR.
#[test]
fn zero_per_layer_input_width_is_rejected() {
    let mut json = pinned_config_value();
    json["text_config"]["hidden_size_per_layer_input"] = serde_json::json!(0);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("hidden_size_per_layer_input: 0 must yield an InferenceError");
    assert!(
        err.to_string().contains("hidden_size_per_layer_input"),
        "error must name hidden_size_per_layer_input: {err}"
    );
}

/// The admission check must fire from config parsing alone, before any
/// weight file is touched. A temporary directory holding only a mutated
/// `config.json` (no `model.safetensors`) must fail on the attention-mode
/// field, never on the missing checkpoint file. The positive control in the
/// same test -- an unmodified fixture in an otherwise identical directory --
/// proves the negative arm is actually reaching the admission check rather
/// than failing for an unrelated reason: admission passes, and the very
/// next step (weight I/O) is what then reports the missing
/// `model.safetensors` (`Gemma4Model::from_safetensors`'s own directory
/// check).
#[test]
fn rejection_happens_before_any_weight_access() {
    let mode_dir = tempfile::tempdir().expect("create temp dir for the negative arm");
    let mut mode_json = pinned_config_value();
    mode_json["text_config"]["use_bidirectional_attention"] = serde_json::json!("vision");
    std::fs::write(mode_dir.path().join("config.json"), mode_json.to_string())
        .expect("write mutated config.json");
    let Err(mode_err) = Gemma4Model::from_safetensors(mode_dir.path()) else {
        panic!("a non-null use_bidirectional_attention must be rejected before weight I/O");
    };
    let mode_msg = mode_err.to_string();
    assert!(
        mode_msg.contains("use_bidirectional_attention"),
        "must name use_bidirectional_attention: {mode_msg}"
    );
    assert!(
        !mode_msg.contains("model.safetensors"),
        "an admission rejection must not read as a missing-weights error: {mode_msg}"
    );

    let control_dir = tempfile::tempdir().expect("create temp dir for the positive control");
    std::fs::write(control_dir.path().join("config.json"), pinned_config_json())
        .expect("write unmodified config.json");
    let Err(control_err) = Gemma4Model::from_safetensors(control_dir.path()) else {
        panic!("an unmodified config with no model.safetensors must fail at weight I/O");
    };
    let control_msg = control_err.to_string();
    assert!(
        control_msg.contains("model.safetensors"),
        "admission must have passed, leaving the missing-weights error as the next failure: \
         {control_msg}"
    );
}
