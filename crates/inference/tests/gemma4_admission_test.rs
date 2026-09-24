//! Gemma 4 E2B admission preflight (ADR-090 Gemma admission, issue #1598):
//! the loader must reject a non-null `use_bidirectional_attention` before
//! constructing a [`Gemma4Config`] or touching any weight file, admit the
//! pinned E2B fixture's absent/null value unchanged, and keep pinning the
//! existing zero-per-layer-embedding-width rejection (ADR-082 G9). Default
//! features, no checkpoint required -- every case here parses the committed
//! fixture (optionally mutated) or a config-only temporary directory.
//!
//! R04a (PR 2) extends the same admission gate with a target-decoder role
//! check (top-level `architectures`/`model_type`, `text_config.model_type`)
//! and explicit checks on semantic keys the pinned fixture carries that
//! [`Gemma4Config`] does not model at all (`enable_moe_block`,
//! `num_experts`, `top_k_experts`, `expert_intermediate_size`,
//! `vocab_size_per_layer_input`). All new checks live in the same private
//! raw-parse layer as the mode check; [`Gemma4Config`]'s public fields are
//! unchanged.

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

/// Role admission (ADR-090 Gemma admission, issue #1598 R04a): a
/// drafter/assistant Gemma 4 checkpoint is a materially different
/// architecture from the pinned E2B target decoder -- no own token
/// embedding table, MTP-style projection layers reading a *separate*
/// target model's embeddings -- and is registered in upstream tooling
/// under a different top-level `architectures` value
/// (`Gemma4AssistantForCausalLM`, real value read from
/// `ggml-org/llama.cpp`'s `conversion/gemma.py` via the GitHub API
/// 2026-09-23: `@ModelBase.register("Gemma4AssistantForCausalLM",
/// "Gemma4UnifiedAssistantForCausalLM")`, with examples including
/// `google/gemma-4-E2B-it-assistant` -- an assistant checkpoint at the
/// *same* E2B size as the pinned target, so size alone cannot distinguish
/// the role). Keeps otherwise-valid E2B geometry unchanged, isolating the
/// role signal the same way `use_bidirectional_attention_non_null_is_rejected_naming_the_field`
/// isolates the mode signal -- this is R04a's drafter-shaped negative
/// fixture (source-derived: no real Gemma 4 assistant checkpoint was
/// loaded or executed).
#[test]
fn wrong_top_level_architectures_is_rejected_naming_the_field() {
    let mut json = pinned_config_value();
    json["architectures"] = serde_json::json!(["Gemma4AssistantForCausalLM"]);
    assert_eq!(
        json["text_config"]["hidden_size_per_layer_input"],
        serde_json::json!(256),
        "mutation must leave the pinned E2B PLE width untouched"
    );
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("a drafter/assistant architectures value must be rejected");
    assert!(
        err.to_string().contains("architectures"),
        "error must name architectures: {err}"
    );
}

/// Top-level `model_type` admission (ADR-090 Gemma admission, issue #1598
/// R04a). The wrong value used here (`"gemma4_text"`) is not an invented
/// string: it is the *real* value that belongs one level down, at
/// `text_config.model_type` -- a plausible transcription mistake, not an
/// arbitrary probe.
#[test]
fn wrong_top_level_model_type_is_rejected_naming_the_field() {
    let mut json = pinned_config_value();
    json["model_type"] = serde_json::json!("gemma4_text");
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("an unsupported top-level model_type must be rejected");
    assert!(
        err.to_string().contains("model_type"),
        "error must name model_type: {err}"
    );
}

/// `text_config.model_type` admission (ADR-090 Gemma admission, issue
/// #1598 R04a). The wrong value used here (`"gemma4_audio"`) is not an
/// invented string: it is the real `model_type` of this same pinned
/// fixture's own `audio_config` sibling object, one level up.
#[test]
fn wrong_text_config_model_type_is_rejected_naming_the_field() {
    let mut json = pinned_config_value();
    assert_eq!(
        json["audio_config"]["model_type"],
        serde_json::json!("gemma4_audio"),
        "sanity: the fixture's own audio_config carries this sibling model_type"
    );
    json["text_config"]["model_type"] = serde_json::json!("gemma4_audio");
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("an unsupported text_config.model_type must be rejected");
    assert!(
        err.to_string().contains("model_type"),
        "error must name model_type: {err}"
    );
}

/// Semantic-key admission (ADR-090 Gemma admission, issue #1598 R04a):
/// `enable_moe_block` gates a Mixture-of-Experts forward path this loader
/// does not implement -- every layer runs the single dense GeGLU MLP
/// regardless of this flag. `true` must be rejected rather than silently
/// running the dense path against MoE-shaped weights.
#[test]
fn enable_moe_block_true_is_rejected_naming_the_field() {
    let mut json = pinned_config_value();
    json["text_config"]["enable_moe_block"] = serde_json::json!(true);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("enable_moe_block: true must yield an InferenceError");
    assert!(
        err.to_string().contains("enable_moe_block"),
        "error must name enable_moe_block: {err}"
    );
}

/// Semantic-key admission (ADR-090 Gemma admission, issue #1598 R04a): a
/// present `num_experts` is checked independently of `enable_moe_block`,
/// in case a checkpoint sets one without the other.
#[test]
fn num_experts_present_is_rejected_naming_the_field() {
    let mut json = pinned_config_value();
    json["text_config"]["num_experts"] = serde_json::json!(8);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("a present num_experts must yield an InferenceError");
    assert!(
        err.to_string().contains("num_experts"),
        "error must name num_experts: {err}"
    );
}

/// Semantic-key admission (ADR-090 Gemma admission, issue #1598 R04a):
/// same contract as `num_experts_present_is_rejected_naming_the_field`.
#[test]
fn top_k_experts_present_is_rejected_naming_the_field() {
    let mut json = pinned_config_value();
    json["text_config"]["top_k_experts"] = serde_json::json!(2);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("a present top_k_experts must yield an InferenceError");
    assert!(
        err.to_string().contains("top_k_experts"),
        "error must name top_k_experts: {err}"
    );
}

/// Semantic-key admission (ADR-090 Gemma admission, issue #1598 R04a):
/// same contract as `num_experts_present_is_rejected_naming_the_field`.
/// Named `expert_intermediate_size` to match the pinned fixture's own
/// spelling of this field (current upstream `transformers` `main` renamed
/// it `moe_intermediate_size`; llama.cpp's converter reads either name).
#[test]
fn expert_intermediate_size_present_is_rejected_naming_the_field() {
    let mut json = pinned_config_value();
    json["text_config"]["expert_intermediate_size"] = serde_json::json!(4096);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("a present expert_intermediate_size must yield an InferenceError");
    assert!(
        err.to_string().contains("expert_intermediate_size"),
        "error must name expert_intermediate_size: {err}"
    );
}

/// Semantic-key admission (ADR-090 Gemma admission, issue #1598 R04a):
/// `vocab_size_per_layer_input` is the row count `gemma4_loading`'s
/// tensor-shape derivation for `embed_tokens_per_layer` silently assumes
/// equals `vocab_size` -- the loader never reads this field at all. A
/// value other than the pinned fixture's own 262144 must be rejected here
/// rather than surfacing later as an opaque tensor-shape mismatch (or,
/// worse, an in-bounds but wrong read if the byte counts ever happened to
/// coincide). The wrong value used here (131072) is again not invented:
/// it is this same pinned fixture's own `max_position_embeddings`, a
/// plausible transcription mistake pulling in a sibling field's value.
#[test]
fn vocab_size_per_layer_input_mismatch_is_rejected_naming_the_field() {
    let mut json = pinned_config_value();
    assert_eq!(
        json["text_config"]["max_position_embeddings"],
        serde_json::json!(131_072),
        "sanity: the fixture's own max_position_embeddings carries this sibling value"
    );
    json["text_config"]["vocab_size_per_layer_input"] = serde_json::json!(131_072);
    let err = Gemma4Config::from_config_json_str(&json.to_string())
        .expect_err("a mismatched vocab_size_per_layer_input must yield an InferenceError");
    assert!(
        err.to_string().contains("vocab_size_per_layer_input"),
        "error must name vocab_size_per_layer_input: {err}"
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
///
/// A second negative arm covers the role check (a drafter/assistant
/// `architectures` value), so a role refusal is also shown to happen
/// before weight I/O, not only the mode refusal. Both negative arms share
/// the one positive control above them.
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

    let role_dir = tempfile::tempdir().expect("create temp dir for the role negative arm");
    let mut role_json = pinned_config_value();
    role_json["architectures"] = serde_json::json!(["Gemma4AssistantForCausalLM"]);
    std::fs::write(role_dir.path().join("config.json"), role_json.to_string())
        .expect("write mutated config.json");
    let Err(role_err) = Gemma4Model::from_safetensors(role_dir.path()) else {
        panic!("a drafter/assistant architectures value must be rejected before weight I/O");
    };
    let role_msg = role_err.to_string();
    assert!(
        role_msg.contains("architectures"),
        "must name architectures: {role_msg}"
    );
    assert!(
        !role_msg.contains("model.safetensors"),
        "an admission rejection must not read as a missing-weights error: {role_msg}"
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
