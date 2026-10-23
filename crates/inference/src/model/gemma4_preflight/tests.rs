use super::*;
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gemma4")
        .join("e2b_tensor_manifest.json")
}

fn load_language_model_tensors() -> HashMap<String, TensorInfo> {
    let data = std::fs::read_to_string(fixture_path()).expect("read committed Stage-0 fixture");
    let inventory: TensorInventory =
        serde_json::from_str(&data).expect("Stage-0 fixture parses as a TensorInventory");
    inventory
        .tensors
        .into_iter()
        .filter(|(name, _)| name.starts_with(LM_PREFIX))
        .collect()
}

#[test]
fn golden_preflight_passes_against_stage0_fixture() {
    let cfg = Gemma4Config::e2b();
    let tensors = load_language_model_tensors();
    let report = preflight_check(&cfg, &tensors).expect("Stage-0 fixture must preflight clean");

    // Exactly the 20 shared layers' K/V (+k_norm) tensors are skipped:
    // 20 layers * 3 tensors (k_proj, v_proj, k_norm) = 60.
    assert_eq!(report.skipped_shared_kv.len(), 60);
    for layer in 15..35 {
        for suffix in ["k_proj", "v_proj", "k_norm"] {
            let name = format!("{LM_PREFIX}layers.{layer}.self_attn.{suffix}.weight");
            assert!(
                report.skipped_shared_kv.contains(&name),
                "expected {name} in skipped_shared_kv"
            );
        }
    }
    // No non-shared layer's K/V shows up in the skip list.
    for layer in 0..15 {
        for suffix in ["k_proj", "v_proj", "k_norm"] {
            let name = format!("{LM_PREFIX}layers.{layer}.self_attn.{suffix}.weight");
            assert!(!report.skipped_shared_kv.contains(&name));
        }
    }
}

#[test]
fn golden_preflight_classifies_all_35_layers() {
    let cfg = Gemma4Config::e2b();
    let tensors = load_language_model_tensors();
    let report = preflight_check(&cfg, &tensors).unwrap();

    for layer in 0..35 {
        let q = format!("{LM_PREFIX}layers.{layer}.self_attn.q_proj.weight");
        assert!(report.loaded.contains(&q), "layer {layer} q_proj must load");
    }
    // 600 total lm tensors - 60 skipped shared K/V = 540 loaded.
    assert_eq!(report.loaded.len(), 540);
    assert_eq!(report.skipped_shared_kv.len(), 60);
    // Exhaustiveness: every supplied language-model tensor was consumed
    // exactly once, either loaded or skipped-as-shared-KV -- the input
    // inventory count equals loaded + skipped with nothing left over.
    assert_eq!(
        tensors.len(),
        report.loaded.len() + report.skipped_shared_kv.len()
    );
}

#[test]
fn mutation_unexpected_language_model_tensor_fails_naming_tensor() {
    let cfg = Gemma4Config::e2b();
    let mut tensors = load_language_model_tensors();
    let name = format!("{LM_PREFIX}layers.0.self_attn.mystery_proj.weight");
    tensors.insert(
        name.clone(),
        TensorInfo {
            dtype: GEMMA4_EXPECTED_DTYPE.to_string(),
            shape: vec![1],
        },
    );

    let err = preflight_check(&cfg, &tensors)
        .expect_err("an unexpected language-model tensor must fail closed as unconsumed");
    assert!(
        err.to_string().contains(&name),
        "error must name the unconsumed tensor {name}: {err}"
    );
}

#[test]
fn mutation_shared_layer_kv_missing_entry_is_tolerated_and_absent_from_skip_report() {
    let cfg = Gemma4Config::e2b();
    let mut tensors = load_language_model_tensors();
    let name = format!("{LM_PREFIX}layers.20.self_attn.k_proj.weight");
    tensors.remove(&name);

    let report = preflight_check(&cfg, &tensors)
        .expect("a missing shared-layer K/V entry must be tolerated, not fail closed");
    assert!(
        !report.skipped_shared_kv.contains(&name),
        "a missing shared-layer tensor must not appear in the skip report: {name}"
    );
    assert_eq!(report.skipped_shared_kv.len(), 59);
    assert_eq!(
        tensors.len(),
        report.loaded.len() + report.skipped_shared_kv.len(),
        "exhaustiveness must still hold with one fewer supplied and skipped tensor"
    );
}

#[test]
fn mutation_shared_layer_kv_shape_change_is_tolerated() {
    let cfg = Gemma4Config::e2b();
    let mut tensors = load_language_model_tensors();
    let name = format!("{LM_PREFIX}layers.20.self_attn.k_proj.weight");
    tensors.get_mut(&name).unwrap().shape = vec![1, 1];

    let report = preflight_check(&cfg, &tensors)
        .expect("a shared-layer K/V shape mutation must be tolerated, not fail closed");
    assert!(report.skipped_shared_kv.contains(&name));
}

#[test]
fn mutation_shared_layer_kv_dtype_change_is_tolerated() {
    let cfg = Gemma4Config::e2b();
    let mut tensors = load_language_model_tensors();
    let name = format!("{LM_PREFIX}layers.30.self_attn.v_proj.weight");
    tensors.get_mut(&name).unwrap().dtype = "F32".to_string();

    let report = preflight_check(&cfg, &tensors)
        .expect("a shared-layer K/V dtype mutation must be tolerated, not fail closed");
    assert!(report.skipped_shared_kv.contains(&name));
}

#[test]
fn mutation_nonshared_layer_missing_kv_fails_naming_tensor() {
    let cfg = Gemma4Config::e2b();
    let mut tensors = load_language_model_tensors();
    let name = format!("{LM_PREFIX}layers.10.self_attn.k_proj.weight");
    tensors.remove(&name);

    let err = preflight_check(&cfg, &tensors)
        .expect_err("a missing non-shared-layer k_proj must fail closed");
    assert!(
        err.to_string().contains(&name),
        "error must name the missing tensor {name}: {err}"
    );
}

#[test]
fn mutation_double_wide_mlp_kv_shared_mismatch_fails_naming_both_fields() {
    let mut cfg = Gemma4Config::e2b();
    // Shrink the shared set from {15..34} to {16..34}: layer 15 is now
    // (per config) NOT kv-shared, but the fixture's real layer 15
    // mlp.* tensors are still double-wide (12288) -- structural
    // disagreement between config and checkpoint.
    cfg.num_kv_shared_layers = 19;
    let tensors = load_language_model_tensors();

    let err = preflight_check(&cfg, &tensors)
        .expect_err("a config/checkpoint KV-sharing-set mismatch must fail closed");
    let msg = err.to_string();
    assert!(
        msg.contains("use_double_wide_mlp"),
        "must name use_double_wide_mlp: {msg}"
    );
    assert!(
        msg.contains("is_kv_shared_layer"),
        "must name is_kv_shared_layer: {msg}"
    );
}

#[test]
fn mutation_global_layer_wrong_head_width_fails_naming_tensor_and_shape() {
    let cfg = Gemma4Config::e2b();
    let mut tensors = load_language_model_tensors();
    // Layer 4 is global, non-shared: k_proj must be 512-wide. Mutate it
    // to the sliding width (256) instead.
    let name = format!("{LM_PREFIX}layers.4.self_attn.k_proj.weight");
    tensors.get_mut(&name).unwrap().shape = vec![256, 1536];

    let err = preflight_check(&cfg, &tensors)
        .expect_err("a global layer's k_proj at the wrong (sliding) width must fail closed");
    match err {
        InferenceError::ShapeMismatch {
            name: got_name,
            expected,
            actual,
        } => {
            assert_eq!(got_name, name);
            assert_eq!(expected, vec![512, 1536]);
            assert_eq!(actual, vec![256, 1536]);
        }
        other => {
            panic!("expected ShapeMismatch naming {name} and expected [512, 1536], got: {other}")
        }
    }
}

#[test]
fn mutation_wrong_dtype_fails_naming_tensor() {
    let cfg = Gemma4Config::e2b();
    let mut tensors = load_language_model_tensors();
    let name = format!("{LM_PREFIX}norm.weight");
    tensors.get_mut(&name).unwrap().dtype = "F16".to_string();

    let err =
        preflight_check(&cfg, &tensors).expect_err("a wrong top-level dtype must fail closed");
    assert!(
        err.to_string().contains(&name),
        "error must name {name}: {err}"
    );
}

#[test]
fn mutation_missing_top_level_tensor_fails_naming_tensor() {
    let cfg = Gemma4Config::e2b();
    let mut tensors = load_language_model_tensors();
    let name = format!("{LM_PREFIX}embed_tokens_per_layer.weight");
    tensors.remove(&name);

    let err = preflight_check(&cfg, &tensors).expect_err("a missing PLE table must fail closed");
    assert!(
        err.to_string().contains(&name),
        "error must name {name}: {err}"
    );
}

#[test]
fn duplicate_raw_tensor_inventory_key_is_rejected() {
    // Two raw JSON members naming the same tensor must be rejected before
    // collapsing into the HashMap, not silently resolved to whichever
    // value the raw JSON happened to visit last (mirrors PR #988's
    // `deserialize_weight_map_no_duplicates` test for the analogous
    // `weight_map` shape).
    let json = r#"{"tensors": {
            "model.language_model.norm.weight": {"dtype": "BF16", "shape": [1536]},
            "model.language_model.norm.weight": {"dtype": "F32", "shape": [1]}
        }}"#;
    let result: Result<TensorInventory, _> = serde_json::from_str(json);
    let err = result.expect_err("a duplicate raw tensor-name key must be rejected");
    assert!(err.to_string().contains("duplicate"));
    assert!(err.to_string().contains("model.language_model.norm.weight"));
}
