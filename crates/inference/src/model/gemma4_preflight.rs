//! Gemma 4 E2B safetensors loader preflight (ADR-082 stage 2).
//!
//! Classifies every language-model tensor a [`Gemma4Config`] implies against
//! a tensor inventory (e.g. a header-extracted safetensors manifest, such as
//! the committed Stage-0 fixture, PR #991), without loading any weight
//! bytes or constructing any forward-pass module.
//!
//! **Tolerate-and-skip contract (ADR-082 Amendment 1).** The checkpoint
//! carries `self_attn.{k_proj,v_proj,k_norm}` weights at *every* layer,
//! including the trailing `num_kv_shared_layers` layers whose runtime forward
//! pass never constructs those modules (they read shared state instead).
//! This preflight accepts those shared-layer tensors at whatever shape or
//! absence they appear in — it never requires them, never validates their
//! shape, and never wires them into a module. Every other tensor (including
//! `q_proj`/`o_proj`/`q_norm`, which are never shared) is validated for
//! presence, exact shape, and dtype, fail-closed.

use crate::error::InferenceError;
use crate::model::gemma4_config::{GEMMA4_EXPECTED_DTYPE, Gemma4Config};
use std::collections::{HashMap, HashSet};

/// One tensor entry from a safetensors header / manifest: dtype and shape.
/// Deliberately narrower than a real safetensors header entry (no
/// `data_offsets`) — extra JSON fields are ignored by serde, not rejected.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// Deserializes a raw tensor-name -> entry JSON object, rejecting a repeated
/// raw key before it collapses into the `HashMap`.
///
/// A safetensors header (or a manifest fixture derived from one, like
/// `tests/fixtures/gemma4/e2b_tensor_manifest.json`) is structurally the same
/// shape as the sharded-checkpoint `weight_map` in
/// `crate::weights::f32_weights`: tensor name -> value, one raw JSON object.
/// Ordinary map deserialization collapses a repeated raw key to whichever
/// value visits last, silently hiding a corrupted/duplicated header entry
/// from any inventory-count check run over the resulting `HashMap`. This
/// mirrors `deserialize_weight_map_no_duplicates` (PR #988) for that map
/// shape.
pub fn deserialize_tensor_inventory_no_duplicates<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, TensorInfo>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct InventoryVisitor;

    impl<'de> serde::de::Visitor<'de> for InventoryVisitor {
        type Value = HashMap<String, TensorInfo>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a map of tensor name to {dtype, shape}")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut out = HashMap::with_capacity(map.size_hint().unwrap_or(0));
            while let Some((name, entry)) = map.next_entry::<String, TensorInfo>()? {
                if out.insert(name.clone(), entry).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate tensor name in tensor inventory: {name}"
                    )));
                }
            }
            Ok(out)
        }
    }

    deserializer.deserialize_map(InventoryVisitor)
}

/// A tensor inventory keyed by raw name, with the same duplicate-raw-key
/// rejection as `SafetensorsIndex::weight_map`. Tests parse the committed
/// Stage-0 manifest fixture's `tensors` object into this type.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TensorInventory {
    #[serde(deserialize_with = "deserialize_tensor_inventory_no_duplicates")]
    pub tensors: HashMap<String, TensorInfo>,
}

/// Result of a successful preflight: which tensors would be loaded, and
/// which shared-layer K/V tensors were present but intentionally skipped.
#[derive(Debug, Clone, Default)]
pub struct PreflightReport {
    /// Tensor names validated (present, correct shape, correct dtype) and
    /// destined to be loaded.
    pub loaded: Vec<String>,
    /// Shared-layer `k_proj`/`v_proj`/`k_norm` tensor names that were
    /// present in the inventory and accepted-but-skipped per the
    /// tolerate-and-skip contract — never shape/dtype-validated, never
    /// wired into a module.
    pub skipped_shared_kv: Vec<String>,
}

const LM_PREFIX: &str = "model.language_model.";

/// Run the Stage-2 loader preflight: classify every tensor `cfg` implies
/// against `tensors`, fail closed on any missing/wrong-shape/wrong-dtype
/// required tensor, and tolerate-and-skip the KV-shared layers' K/V weights.
pub fn preflight_check(
    cfg: &Gemma4Config,
    tensors: &HashMap<String, TensorInfo>,
) -> Result<PreflightReport, InferenceError> {
    // Re-validate at this public boundary too: callers can build a
    // `Gemma4Config` directly (bypassing `from_config_json_str`), and a
    // malformed config must not silently derive nonsensical expected shapes
    // below instead of failing closed here.
    cfg.validate()?;

    let mut report = PreflightReport::default();
    let hidden = cfg.hidden_size;
    let per_layer_dim = cfg.hidden_size_per_layer_input;
    let ple_packed_dim = cfg.num_hidden_layers * per_layer_dim;

    check_required(
        tensors,
        &format!("{LM_PREFIX}embed_tokens.weight"),
        &[cfg.vocab_size, hidden],
        &mut report,
    )?;
    check_required(
        tensors,
        &format!("{LM_PREFIX}embed_tokens_per_layer.weight"),
        &[cfg.vocab_size, ple_packed_dim],
        &mut report,
    )?;
    check_required(
        tensors,
        &format!("{LM_PREFIX}norm.weight"),
        &[hidden],
        &mut report,
    )?;
    check_required(
        tensors,
        &format!("{LM_PREFIX}per_layer_model_projection.weight"),
        &[ple_packed_dim, hidden],
        &mut report,
    )?;
    check_required(
        tensors,
        &format!("{LM_PREFIX}per_layer_projection_norm.weight"),
        &[per_layer_dim],
        &mut report,
    )?;

    for layer in 0..cfg.num_hidden_layers {
        let prefix = format!("{LM_PREFIX}layers.{layer}.");
        let head_w = cfg.attn_head_dim(layer);
        let q_dim = cfg.num_attention_heads * head_w;
        let kv_dim = cfg.num_key_value_heads * head_w;

        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "post_per_layer_input_norm.weight",
        ] {
            check_required(
                tensors,
                &format!("{prefix}{suffix}"),
                &[hidden],
                &mut report,
            )?;
        }
        check_required(tensors, &format!("{prefix}layer_scalar"), &[1], &mut report)?;
        check_required(
            tensors,
            &format!("{prefix}per_layer_input_gate.weight"),
            &[per_layer_dim, hidden],
            &mut report,
        )?;
        check_required(
            tensors,
            &format!("{prefix}per_layer_projection.weight"),
            &[hidden, per_layer_dim],
            &mut report,
        )?;

        check_required(
            tensors,
            &format!("{prefix}self_attn.q_proj.weight"),
            &[q_dim, hidden],
            &mut report,
        )?;
        check_required(
            tensors,
            &format!("{prefix}self_attn.o_proj.weight"),
            &[hidden, q_dim],
            &mut report,
        )?;
        check_required(
            tensors,
            &format!("{prefix}self_attn.q_norm.weight"),
            &[head_w],
            &mut report,
        )?;

        check_mlp_geometry(cfg, layer, hidden, tensors, &prefix, &mut report)?;

        if cfg.is_kv_shared_layer(layer) {
            // Tolerate-and-skip: accept presence at any shape/dtype, never
            // require, never shape/dtype-validate, never wire into a module.
            for suffix in [
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.k_norm.weight",
            ] {
                let name = format!("{prefix}{suffix}");
                if tensors.contains_key(&name) {
                    report.skipped_shared_kv.push(name);
                }
            }
        } else {
            check_required(
                tensors,
                &format!("{prefix}self_attn.k_proj.weight"),
                &[kv_dim, hidden],
                &mut report,
            )?;
            check_required(
                tensors,
                &format!("{prefix}self_attn.v_proj.weight"),
                &[kv_dim, hidden],
                &mut report,
            )?;
            check_required(
                tensors,
                &format!("{prefix}self_attn.k_norm.weight"),
                &[head_w],
                &mut report,
            )?;
        }
    }

    // Exhaustiveness (ADR-082 Amendment 1): every
    // supplied language-model tensor must have been consumed above, either
    // by loading or by the shared-KV tolerate-and-skip category. A name
    // that matches neither -- an extra, misspelled, or otherwise unexpected
    // language-model tensor -- is unconsumed and must fail closed, naming
    // the tensor, rather than passing silently. Non-language-model tensors
    // (vision/audio towers) are out of this preflight's scope and are not
    // checked here.
    let consumed: HashSet<&str> = report
        .loaded
        .iter()
        .chain(report.skipped_shared_kv.iter())
        .map(String::as_str)
        .collect();
    let mut unconsumed: Vec<&str> = tensors
        .keys()
        .filter(|name| name.starts_with(LM_PREFIX) && !consumed.contains(name.as_str()))
        .map(String::as_str)
        .collect();
    unconsumed.sort_unstable();
    if let Some(&name) = unconsumed.first() {
        return Err(InferenceError::Inference(format!(
            "gemma4 preflight: unconsumed language-model tensor {name:?} was present in the \
             supplied inventory but not required by any layer or the shared-KV skip category \
             ({} unconsumed tensor(s) total)",
            unconsumed.len()
        )));
    }

    Ok(report)
}

/// Validate a required tensor's presence, exact shape, and dtype, recording
/// it in `report.loaded` on success. Fails closed naming the tensor on any
/// of: missing, wrong shape, or wrong dtype.
fn check_required(
    tensors: &HashMap<String, TensorInfo>,
    name: &str,
    expected_shape: &[usize],
    report: &mut PreflightReport,
) -> Result<(), InferenceError> {
    let entry = tensors
        .get(name)
        .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))?;
    if entry.shape != expected_shape {
        return Err(InferenceError::ShapeMismatch {
            name: name.to_string(),
            expected: expected_shape.to_vec(),
            actual: entry.shape.clone(),
        });
    }
    if entry.dtype != GEMMA4_EXPECTED_DTYPE {
        return Err(InferenceError::Inference(format!(
            "gemma4 preflight: tensor {name} has dtype {:?}, expected {GEMMA4_EXPECTED_DTYPE}",
            entry.dtype
        )));
    }
    report.loaded.push(name.to_string());
    Ok(())
}

/// Validate `layer`'s MLP tensors and cross-check the checkpoint's own
/// width marker against the config's claimed KV-sharing set.
///
/// Each of `gate_proj`/`up_proj`/`down_proj` must be exactly `intermediate`
/// or `2 * intermediate` wide (any other width is a plain shape-mismatch,
/// naming the tensor); all three must agree on which of those two widths
/// they are (else a same-layer disagreement error); and that agreed width
/// must match `cfg.use_double_wide_mlp(layer)` — ADR-082 Amendment 1's
/// "second structural observable" for the KV-shared layer set. A config
/// whose `num_kv_shared_layers` disagrees with the checkpoint's actual
/// `mlp.*` widths fails here, naming both `use_double_wide_mlp` and
/// `is_kv_shared_layer`, rather than surfacing as an opaque per-tensor shape
/// mismatch.
fn check_mlp_geometry(
    cfg: &Gemma4Config,
    layer: usize,
    hidden: usize,
    tensors: &HashMap<String, TensorInfo>,
    prefix: &str,
    report: &mut PreflightReport,
) -> Result<(), InferenceError> {
    let narrow = cfg.intermediate_size;
    let wide = cfg.intermediate_size * 2;

    let gate_wide = check_mlp_narrow_or_wide(
        tensors,
        &format!("{prefix}mlp.gate_proj.weight"),
        hidden,
        narrow,
        wide,
        MlpAxis::MlpDimFirst,
        report,
    )?;
    let up_wide = check_mlp_narrow_or_wide(
        tensors,
        &format!("{prefix}mlp.up_proj.weight"),
        hidden,
        narrow,
        wide,
        MlpAxis::MlpDimFirst,
        report,
    )?;
    let down_wide = check_mlp_narrow_or_wide(
        tensors,
        &format!("{prefix}mlp.down_proj.weight"),
        hidden,
        narrow,
        wide,
        MlpAxis::MlpDimSecond,
        report,
    )?;

    if gate_wide != up_wide || up_wide != down_wide {
        return Err(InferenceError::Inference(format!(
            "gemma4 preflight: layer {layer} mlp tensors disagree on width -- \
             gate_proj wide={gate_wide}, up_proj wide={up_wide}, down_proj wide={down_wide}"
        )));
    }

    let expected_wide = cfg.use_double_wide_mlp(layer);
    if gate_wide != expected_wide {
        return Err(InferenceError::Inference(format!(
            "gemma4 preflight: layer {layer} use_double_wide_mlp={expected_wide} disagrees with \
             is_kv_shared_layer({layer})={} (num_kv_shared_layers={}) -- the checkpoint's own \
             mlp.gate_proj width says observed double-wide={gate_wide}",
            cfg.is_kv_shared_layer(layer),
            cfg.num_kv_shared_layers,
        )));
    }

    Ok(())
}

#[derive(Clone, Copy)]
enum MlpAxis {
    /// `gate_proj`/`up_proj`: shape = `[mlp_dim, hidden]`.
    MlpDimFirst,
    /// `down_proj`: shape = `[hidden, mlp_dim]`.
    MlpDimSecond,
}

/// Validate one MLP tensor is exactly `narrow` or `wide` along its MLP-dim
/// axis (any other value is a shape mismatch naming the tensor), then
/// record it as loaded and return whether it was `wide`.
fn check_mlp_narrow_or_wide(
    tensors: &HashMap<String, TensorInfo>,
    name: &str,
    hidden: usize,
    narrow: usize,
    wide: usize,
    axis: MlpAxis,
    report: &mut PreflightReport,
) -> Result<bool, InferenceError> {
    let entry = tensors
        .get(name)
        .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))?;

    let (mlp_dim, other_dim, expected_shape_for_mismatch) = match axis {
        MlpAxis::MlpDimFirst => (
            entry.shape.first().copied(),
            entry.shape.get(1).copied(),
            vec![narrow, hidden],
        ),
        MlpAxis::MlpDimSecond => (
            entry.shape.get(1).copied(),
            entry.shape.first().copied(),
            vec![hidden, narrow],
        ),
    };

    let is_wide = mlp_dim == Some(wide);
    let is_narrow = mlp_dim == Some(narrow);
    if entry.shape.len() != 2 || other_dim != Some(hidden) || !(is_wide || is_narrow) {
        return Err(InferenceError::ShapeMismatch {
            name: name.to_string(),
            expected: expected_shape_for_mismatch,
            actual: entry.shape.clone(),
        });
    }
    if entry.dtype != GEMMA4_EXPECTED_DTYPE {
        return Err(InferenceError::Inference(format!(
            "gemma4 preflight: tensor {name} has dtype {:?}, expected {GEMMA4_EXPECTED_DTYPE}",
            entry.dtype
        )));
    }

    report.loaded.push(name.to_string());
    Ok(is_wide)
}

#[cfg(test)]
mod tests {
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
            other => panic!(
                "expected ShapeMismatch naming {name} and expected [512, 1536], got: {other}"
            ),
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

        let err =
            preflight_check(&cfg, &tensors).expect_err("a missing PLE table must fail closed");
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
}
