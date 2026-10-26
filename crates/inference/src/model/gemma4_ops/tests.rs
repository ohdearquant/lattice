use super::*;
use crate::model::qwen35::qwen35_rms_norm;
use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gemma4")
        .join("stage3")
}

fn load_json(name: &str) -> serde_json::Value {
    let path: PathBuf = fixture_dir().join(name);
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    serde_json::from_str(&data).unwrap_or_else(|e| panic!("parse fixture {}: {e}", path.display()))
}

fn manifest() -> serde_json::Value {
    load_json("manifest.json")
}

fn tolerance(op: &str) -> f32 {
    manifest()["ops"][op]["tolerance_max_abs_diff"]
        .as_f64()
        .unwrap_or_else(|| panic!("manifest missing tolerance for op {op}")) as f32
}

/// Predeclared minimum required max-abs-diff for a mutation (negative)
/// test to pass (manifest.json's `mutation_separation_floor`) -- a
/// margin above the golden's own tolerance, not merely `diff > tol`,
/// so a mutation one ULP past the tolerance boundary can't pass by
/// accident. See manifest.json's `mutation_separation_floor_note`.
fn mutation_separation_floor(op: &str) -> f32 {
    manifest()["ops"][op]["mutation_separation_floor"]
        .as_f64()
        .unwrap_or_else(|| panic!("manifest missing mutation_separation_floor for op {op}"))
        as f32
}

/// Read a tensor-ref field (`{"bin": "<file>", "shape": [...], ...}`,
/// written by `materialize_op`/`TensorRef` in
/// `scripts/gemma4_stage3_goldens.py`) as a `Vec<f32>`: raw
/// little-endian float32 bytes from the referenced `.bin` file.
fn load_bin(fx: &serde_json::Value, key: &str) -> Vec<f32> {
    let bin_name = fx[key]["bin"]
        .as_str()
        .unwrap_or_else(|| panic!("fixture field {key:?} is not a tensor ref (missing .bin)"));
    let path = fixture_dir().join(bin_name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "{} byte length must be a multiple of 4 (f32)",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn flatten_u32(v: &serde_json::Value) -> Vec<u32> {
    match v {
        serde_json::Value::Array(items) => items.iter().flat_map(flatten_u32).collect(),
        serde_json::Value::Number(n) => vec![n.as_u64().expect("integer fixture value") as u32],
        other => panic!("expected number or array in fixture, got {other:?}"),
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared slices must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn dims(v: &serde_json::Value, key: &str) -> Vec<usize> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture missing shape key {key}"))
        .iter()
        .map(|d| d.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn mutation_shifted_rms_norm_fails_golden() {
    // Fail-closed negative test (ADR-082 stage 3's declared negative
    // test): substituting lattice's Qwen3.5 shifted `(1 + gamma)`
    // RMSNorm for the standard op must exceed the golden's tolerance.
    let fx = load_json("rms_norm.json");
    let floor = mutation_separation_floor("rms_norm");
    let shape = dims(&fx, "shape");
    let hidden = shape[2];
    let eps = fx["eps"].as_f64().unwrap() as f32;
    let mut x = load_bin(&fx, "input");
    let gamma = load_bin(&fx, "weight");
    let expected = load_bin(&fx, "output");

    qwen35_rms_norm(&mut x, &gamma, hidden, eps);

    let diff = max_abs_diff(&x, &expected);
    assert!(
        diff >= floor,
        "shifted (1+gamma) RMSNorm must diverge from the standard-RMSNorm golden \
             by at least the predeclared mutation-separation floor \
             (diff {diff}, floor {floor}) -- this test is decorative if it doesn't"
    );
}

#[test]
fn scaled_embedding_matches_hf_golden() {
    let fx = load_json("scaled_embedding.json");
    let tol = tolerance("scaled_embedding");
    let hidden = fx["hidden"].as_u64().unwrap() as usize;
    let ids = flatten_u32(&fx["input_ids"]);
    let embed_weight = load_bin(&fx, "embed_weight");
    let expected = load_bin(&fx, "output");
    let expected_scale = fx["embed_scale"].as_f64().unwrap() as f32;
    assert!(
        ((hidden as f32).sqrt() - expected_scale).abs() < 1e-6,
        "fixture embed_scale must be sqrt(hidden_size)"
    );

    let mut out = vec![0f32; ids.len() * hidden];
    gemma4_scaled_embedding(&ids, &embed_weight, hidden, &mut out);

    let diff = max_abs_diff(&out, &expected);
    assert!(
        diff <= tol,
        "scaled_embedding max-abs-diff {diff} exceeds tolerance {tol}"
    );
}

#[test]
fn logit_softcap_matches_hf_golden() {
    let fx = load_json("logit_softcap.json");
    let tol = tolerance("logit_softcap");
    let cap = fx["cap"].as_f64().unwrap() as f32;
    let mut logits = load_bin(&fx, "input");
    let expected = load_bin(&fx, "output");

    gemma4_logit_softcap(&mut logits, cap);

    let diff = max_abs_diff(&logits, &expected);
    assert!(
        diff <= tol,
        "logit_softcap max-abs-diff {diff} exceeds tolerance {tol}"
    );
}

#[test]
fn mutation_disabled_softcap_fails_golden() {
    // Fail-closed negative test: leaving logits uncapped (identity) must
    // diverge from the capped golden beyond tolerance. The fixture's
    // logits deliberately span well past +/-cap so saturation is
    // material, not a rounding-noise-sized difference.
    let fx = load_json("logit_softcap.json");
    let floor = mutation_separation_floor("logit_softcap");
    let logits = load_bin(&fx, "input");
    let expected = load_bin(&fx, "output");

    let diff = max_abs_diff(&logits, &expected);
    assert!(
        diff >= floor,
        "uncapped logits must diverge from the softcapped golden by at least the \
             predeclared mutation-separation floor (diff {diff}, floor {floor})"
    );
}

#[test]
fn stage3_manifest_declares_all_ops() {
    let m = manifest();
    for op in [
        "rms_norm",
        "rms_norm_wide",
        "geglu_mlp",
        "scaled_embedding",
        "qk_norm_v_unscaled",
        "logit_softcap",
        "dual_rope",
    ] {
        assert!(m["ops"][op].is_object(), "manifest missing op {op}");
        let path = fixture_dir().join(m["ops"][op]["file"].as_str().unwrap());
        assert!(
            Path::new(&path).exists(),
            "manifest-declared fixture {op} missing on disk"
        );
    }
}
