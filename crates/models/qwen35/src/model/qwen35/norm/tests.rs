use super::qwen35_rms_norm;
use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../inference/tests/fixtures/gemma4/stage3")
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
