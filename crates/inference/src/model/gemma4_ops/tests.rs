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

// -- G6: standard RMSNorm ------------------------------------------------

#[test]
fn rms_norm_matches_hf_golden() {
    let fx = load_json("rms_norm.json");
    let tol = tolerance("rms_norm");
    let shape = dims(&fx, "shape");
    let hidden = shape[2];
    let eps = fx["eps"].as_f64().unwrap() as f32;
    let mut x = load_bin(&fx, "input");
    let gamma = load_bin(&fx, "weight");
    let expected = load_bin(&fx, "output");

    gemma4_rms_norm(&mut x, &gamma, hidden, eps);

    let diff = max_abs_diff(&x, &expected);
    assert!(
        diff <= tol,
        "rms_norm max-abs-diff {diff} exceeds tolerance {tol}"
    );
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

// -- Medium-4: RMSNorm at the real E2B hidden_size=1536 width ---------------

#[test]
fn rms_norm_wide_matches_hf_golden() {
    // Same op as `rms_norm_matches_hf_golden`, at the real (unscaled)
    // E2B hidden_size=1536 reduction width, demonstrating the bounded
    // f32 parity claim (see `gemma4_rms_norm`'s doc comment) holds at
    // real width, not only at the stage's 1/24-scaled synthetic width.
    let fx = load_json("rms_norm_wide.json");
    let tol = tolerance("rms_norm_wide");
    let shape = dims(&fx, "shape");
    let hidden = shape[2];
    assert_eq!(
        hidden, 1536,
        "rms_norm_wide fixture must use the real E2B hidden_size"
    );
    let eps = fx["eps"].as_f64().unwrap() as f32;
    let mut x = load_bin(&fx, "input");
    let gamma = load_bin(&fx, "weight");
    let expected = load_bin(&fx, "output");

    gemma4_rms_norm(&mut x, &gamma, hidden, eps);

    let diff = max_abs_diff(&x, &expected);
    assert!(
        diff <= tol,
        "rms_norm_wide max-abs-diff {diff} exceeds tolerance {tol}"
    );
}

// -- G7: GeGLU MLP ---------------------------------------------------------

#[test]
fn geglu_mlp_matches_hf_golden() {
    let fx = load_json("geglu_mlp.json");
    let tol = tolerance("geglu_mlp");
    let shape = dims(&fx, "shape");
    let (tokens, hidden) = (shape[0] * shape[1], shape[2]);
    let intermediate = fx["intermediate"].as_u64().unwrap() as usize;
    let x = load_bin(&fx, "input");
    let gate_w = load_bin(&fx, "gate_proj_weight");
    let up_w = load_bin(&fx, "up_proj_weight");
    let down_w = load_bin(&fx, "down_proj_weight");
    let expected = load_bin(&fx, "output");

    let mut out = vec![0f32; tokens * hidden];
    let mut gate_scratch = vec![0f32; tokens * intermediate];
    let mut up_scratch = vec![0f32; tokens * intermediate];
    gemma4_geglu_mlp(
        &x,
        &gate_w,
        &up_w,
        &down_w,
        tokens,
        hidden,
        intermediate,
        &mut gate_scratch,
        &mut up_scratch,
        &mut out,
    );

    let diff = max_abs_diff(&out, &expected);
    assert!(
        diff <= tol,
        "geglu_mlp max-abs-diff {diff} exceeds tolerance {tol}"
    );
}

// -- G10a: scaled embedding -------------------------------------------------

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

// -- G4: Q/K norm, V unscaled ------------------------------------------------

#[test]
fn qk_norm_v_unscaled_matches_hf_golden() {
    let fx = load_json("qk_norm_v_unscaled.json");
    let tol = tolerance("qk_norm_v_unscaled");
    let shape = dims(&fx, "shape");
    let head_dim = shape[3];
    let eps = fx["eps"].as_f64().unwrap() as f32;
    let mut q = load_bin(&fx, "q_input");
    let mut k = load_bin(&fx, "k_input");
    let mut v = load_bin(&fx, "v_input");
    let q_gamma = load_bin(&fx, "q_norm_weight");
    let k_gamma = load_bin(&fx, "k_norm_weight");
    let expected_q = load_bin(&fx, "q_output");
    let expected_k = load_bin(&fx, "k_output");
    let expected_v = load_bin(&fx, "v_output");

    gemma4_qk_norm_v_unscaled(&mut q, &mut k, &mut v, &q_gamma, &k_gamma, head_dim, eps);

    assert!(
        max_abs_diff(&q, &expected_q) <= tol,
        "q_norm exceeds tolerance {tol}"
    );
    assert!(
        max_abs_diff(&k, &expected_k) <= tol,
        "k_norm exceeds tolerance {tol}"
    );
    assert!(
        max_abs_diff(&v, &expected_v) <= tol,
        "v_norm exceeds tolerance {tol}"
    );
}

#[test]
fn mutation_scaled_v_norm_fails_golden() {
    // Fail-closed negative test: applying the Q/K-style *scaled* norm to
    // V (instead of the real unscaled norm) must diverge from the V
    // golden beyond tolerance.
    let fx = load_json("qk_norm_v_unscaled.json");
    let floor = mutation_separation_floor("qk_norm_v_unscaled");
    let shape = dims(&fx, "shape");
    let head_dim = shape[3];
    let eps = fx["eps"].as_f64().unwrap() as f32;
    let mut v = load_bin(&fx, "v_input");
    // Reuse q_norm_weight as a stand-in learned scale -- any non-trivial
    // weight demonstrates the mutation; the golden V weight is
    // (correctly) absent from the fixture since V is unscaled.
    let wrong_v_gamma = load_bin(&fx, "q_norm_weight");
    let expected_v = load_bin(&fx, "v_output");

    rms_norm(&mut v, &wrong_v_gamma, head_dim, eps);

    let diff = max_abs_diff(&v, &expected_v);
    assert!(
        diff >= floor,
        "a scaled V-norm must diverge from the unscaled-V golden by at least the \
             predeclared mutation-separation floor (diff {diff}, floor {floor})"
    );
}

// -- G10b: final logit softcap -----------------------------------------------

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

// -- G8: dual RoPE -----------------------------------------------------------

fn rope_output(
    fx: &serde_json::Value,
    input_key: &str,
    output_key: &str,
    shape_key: &str,
    theta: f64,
    partial_rotary_factor: Option<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let shape = dims(fx, shape_key);
    let (seq_len, heads, head_dim) = (shape[1], shape[2], shape[3]);
    let positions = flatten_u32(&fx["position_ids"]);
    let mut x = load_bin(fx, input_key);
    let expected = load_bin(fx, output_key);

    let inv_freq = gemma4_rope_inv_freq(head_dim, theta, partial_rotary_factor);
    let (cos, sin) = gemma4_rope_cos_sin(&inv_freq, &positions);
    gemma4_apply_rope(&mut x, &cos, &sin, seq_len, heads, head_dim);

    (x, expected)
}

#[test]
fn dual_rope_local_matches_hf_golden() {
    let fx = load_json("dual_rope.json");
    let tol = tolerance("dual_rope");
    let theta = fx["theta_local"].as_f64().unwrap();
    let (actual, expected) = rope_output(
        &fx,
        "local_input",
        "local_output",
        "shape_local",
        theta,
        None,
    );
    let diff = max_abs_diff(&actual, &expected);
    assert!(
        diff <= tol,
        "dual_rope (local) max-abs-diff {diff} exceeds tolerance {tol}"
    );
}

#[test]
fn dual_rope_global_matches_hf_golden() {
    let fx = load_json("dual_rope.json");
    let tol = tolerance("dual_rope");
    let theta = fx["theta_global"].as_f64().unwrap();
    let factor = fx["partial_rotary_factor"].as_f64().unwrap() as f32;
    let (actual, expected) = rope_output(
        &fx,
        "global_input",
        "global_output",
        "shape_global",
        theta,
        Some(factor),
    );
    let diff = max_abs_diff(&actual, &expected);
    assert!(
        diff <= tol,
        "dual_rope (global) max-abs-diff {diff} exceeds tolerance {tol}"
    );
}

#[test]
fn mutation_swapped_theta_fails_golden() {
    // Fail-closed negative test (ADR-082 stage 3's Risks section names
    // this exact failure mode): swapping local/global RoPE thetas must
    // make both outputs diverge from their respective goldens.
    let fx = load_json("dual_rope.json");
    let floor = mutation_separation_floor("dual_rope");
    let theta_local = fx["theta_local"].as_f64().unwrap();
    let theta_global = fx["theta_global"].as_f64().unwrap();
    let factor = fx["partial_rotary_factor"].as_f64().unwrap() as f32;

    // Local layer, but fed the global theta with no partial-rotary
    // truncation (a same-shape swap: local head_dim, wrong theta).
    let (actual_local, expected_local) = rope_output(
        &fx,
        "local_input",
        "local_output",
        "shape_local",
        theta_global,
        None,
    );
    let diff_local = max_abs_diff(&actual_local, &expected_local);
    assert!(
        diff_local >= floor,
        "local RoPE fed the global theta must diverge by at least the predeclared \
             mutation-separation floor (diff {diff_local}, floor {floor})"
    );

    // Global layer, but fed the local theta (still with the real
    // partial-rotary factor, isolating the theta swap).
    let (actual_global, expected_global) = rope_output(
        &fx,
        "global_input",
        "global_output",
        "shape_global",
        theta_local,
        Some(factor),
    );
    let diff_global = max_abs_diff(&actual_global, &expected_global);
    assert!(
        diff_global >= floor,
        "global RoPE fed the local theta must diverge by at least the predeclared \
             mutation-separation floor (diff {diff_global}, floor {floor})"
    );
}

#[test]
#[should_panic(expected = "head_dim must be even")]
fn apply_rope_rejects_odd_head_dim() {
    // The in-place stride-half loop pairs (i, half+i); an odd width has
    // an unpaired last lane the loop would silently leave un-rotated
    // (the removed two-pass form scaled it by cos). Reject, don't drift.
    let head_dim = 5;
    let mut x = vec![1.0_f32; head_dim];
    let cos = vec![1.0_f32; head_dim];
    let sin = vec![0.0_f32; head_dim];
    gemma4_apply_rope(&mut x, &cos, &sin, 1, 1, head_dim);
}

#[test]
fn apply_rope_bit_identical_to_two_pass_reference() {
    // Even-width exact-bit regression: the in-place pair loop must match
    // the removed materialize-rotate-half-then-elementwise form to the
    // bit on finite values (the claim the hot-path rewrite rests on).
    let (seq_len, heads, head_dim) = (3, 2, 8);
    let half = head_dim / 2;
    let n = seq_len * heads * head_dim;
    // Deterministic, sign-varied, non-round values.
    let val = |i: usize| ((i as f32) * 0.7311 - 3.1).sin() * 2.3;
    let x_orig: Vec<f32> = (0..n).map(val).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim).map(|i| val(i + 17)).collect();
    let sin: Vec<f32> = (0..seq_len * head_dim).map(|i| val(i + 41)).collect();

    // Reference: the old two-pass form.
    let mut expected = x_orig.clone();
    for t in 0..seq_len {
        let cos_row = &cos[t * head_dim..(t + 1) * head_dim];
        let sin_row = &sin[t * head_dim..(t + 1) * head_dim];
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            let row = &mut expected[base..base + head_dim];
            let mut rotated = vec![0.0_f32; head_dim];
            for i in 0..half {
                rotated[i] = -row[half + i];
                rotated[half + i] = row[i];
            }
            for i in 0..head_dim {
                row[i] = row[i] * cos_row[i] + rotated[i] * sin_row[i];
            }
        }
    }

    let mut actual = x_orig;
    gemma4_apply_rope(&mut actual, &cos, &sin, seq_len, heads, head_dim);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "lane {i}: in-place RoPE not bit-identical to two-pass reference \
                 ({a} vs {e})"
        );
    }
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
