//! Gemma 4 E2B text math kernels (ADR-082 stage 3, G6-G10).
//!
//! Pure per-op primitives -- no attention/cache/forward wiring (that is
//! stage 4, design-gated per the ADR). Each op is golden-tested against
//! `tests/fixtures/gemma4/stage3/` (HF `transformers.models.gemma4.
//! modeling_gemma4` reference, synthetic weights,
//! `scripts/gemma4_stage3_goldens.py`), at the tolerance predeclared in that
//! fixture set's `manifest.json` -- the tests below read the tolerance from
//! the manifest rather than hardcoding it, so a golden regeneration cannot
//! silently loosen the gate.
//!
//! ADR-058: additive-only. No shared kernel file in `forward/cpu/` is
//! edited; `gemma4_rms_norm` and `gemma4_geglu_mlp` call the existing
//! `crate::forward::cpu::{rms_norm, matmul_bt, elementwise_mul}` kernels
//! as-is (structurally unreachable from any other model's forward path --
//! this module has no callers yet), so no `make bench-compare` run is
//! needed for this PR.

use crate::forward::cpu::{elementwise_mul, matmul_bt, rms_norm};

// ---------------------------------------------------------------------------
// G6: standard RMSNorm.
// ---------------------------------------------------------------------------

/// Standard RMSNorm: `x * rsqrt(mean(x^2) + eps) * gamma` -- plain gamma
/// scaling, **not** lattice's Qwen3.5 shifted `(1 + gamma)` variant
/// (`crate::model::qwen35::qwen35_rms_norm`), which is this stage's
/// declared negative test (`tests::mutation_shifted_rms_norm_fails_golden`).
///
/// Reuses `crate::forward::cpu::rms_norm` as-is: this is **bounded f32
/// parity with `Gemma4RMSNorm.forward`, not bit-for-bit equivalence**. The
/// reference computes `pow(mean(x^2) + eps, -0.5)` (a single `pow` call);
/// this kernel computes `sqrt(mean(x^2) + eps)` then reciprocal (two
/// rounding steps), and its NEON path reduces `sum(x^2)` in 4-wide partial
/// sums merged pairwise -- a different accumulation order than either the
/// reference or this kernel's own scalar fallback. The golden fixture set
/// covers this at both a 1/24-scaled synthetic width (`rms_norm.json`,
/// hidden=64) and the real E2B `hidden_size=1536` (`rms_norm_wide.json`,
/// `tests::rms_norm_wide_matches_hf_golden`), each within its manifest's
/// predeclared tolerance -- see
/// `crates/inference/tests/fixtures/gemma4/stage3/manifest.json`'s
/// `rms_norm`/`rms_norm_wide` `tolerance_justification` for the structural
/// bound.
pub fn gemma4_rms_norm(x: &mut [f32], gamma: &[f32], hidden: usize, eps: f32) {
    rms_norm(x, gamma, hidden, eps);
}

// ---------------------------------------------------------------------------
// G7: GeGLU MLP.
// ---------------------------------------------------------------------------

/// Exact tanh-approximate GELU (`gelu_pytorch_tanh`), matching HF's
/// `ACT2FN["gelu_pytorch_tanh"]` to f32-tanh rounding.
///
/// Deliberately **not** `crate::forward::cpu::gelu`: that kernel's
/// `fast_tanh` is a Padé(7,6) rational approximation of tanh with max error
/// ~4e-5 (see its doc comment), which alone exceeds this stage's
/// predeclared 1e-5 max-abs-diff tolerance for `geglu_mlp` before any
/// matmul propagation. This uses `f32::tanh` (libm, correctly rounded to a
/// few ULPs) instead -- same GELU formula, exact tanh.
fn gelu_tanh_exact(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// GeGLU MLP: `down(gelu_tanh(gate(x)) * up(x))`
/// (`Gemma4TextMLP.forward`, `modeling_gemma4.py:1079-1081`).
///
/// `gate_w`/`up_w` are `[intermediate, hidden]` and `down_w` is
/// `[hidden, intermediate]` (`nn.Linear` weight layout, `[out, in]`) --
/// exactly the layout `crate::forward::cpu::matmul_bt` expects as its `B`
/// argument.
pub fn gemma4_geglu_mlp(
    x: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    tokens: usize,
    hidden: usize,
    intermediate: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), tokens * hidden, "x must be tokens*hidden");
    assert_eq!(
        gate_w.len(),
        intermediate * hidden,
        "gate_w must be intermediate*hidden"
    );
    assert_eq!(
        up_w.len(),
        intermediate * hidden,
        "up_w must be intermediate*hidden"
    );
    assert_eq!(
        down_w.len(),
        hidden * intermediate,
        "down_w must be hidden*intermediate"
    );
    assert_eq!(out.len(), tokens * hidden, "out must be tokens*hidden");

    let mut gate = vec![0f32; tokens * intermediate];
    let mut up = vec![0f32; tokens * intermediate];
    matmul_bt(x, gate_w, &mut gate, tokens, hidden, intermediate);
    matmul_bt(x, up_w, &mut up, tokens, hidden, intermediate);
    for v in gate.iter_mut() {
        *v = gelu_tanh_exact(*v);
    }
    elementwise_mul(&mut gate, &up);
    matmul_bt(&gate, down_w, out, tokens, intermediate, hidden);
}

// ---------------------------------------------------------------------------
// G10a: scaled embedding.
// ---------------------------------------------------------------------------

/// Scaled embedding lookup: `embedding[id] * sqrt(hidden_size)`
/// (`Gemma4TextScaledWordEmbedding.forward`,
/// `modeling_gemma4.py:1468-1469`; `embed_scale` is
/// `config.hidden_size**0.5`, a Python float, instantiated at
/// `modeling_gemma4.py:1601-1602` and applied after being cast to the
/// embedding weight's dtype -- computed and applied here in f32 throughout,
/// matching this stage's f32 reference).
pub fn gemma4_scaled_embedding(ids: &[u32], embed_weight: &[f32], hidden: usize, out: &mut [f32]) {
    assert_eq!(
        out.len(),
        ids.len() * hidden,
        "out must be ids.len()*hidden"
    );
    assert!(
        embed_weight.len().is_multiple_of(hidden),
        "embed_weight must be a whole number of hidden-sized rows"
    );
    let scale = (hidden as f32).sqrt();
    for (t, &id) in ids.iter().enumerate() {
        let row_start = id as usize * hidden;
        let row = &embed_weight[row_start..row_start + hidden];
        let out_row = &mut out[t * hidden..(t + 1) * hidden];
        for (o, &v) in out_row.iter_mut().zip(row.iter()) {
            *o = v * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// G4: Q/K norm with V unscaled.
// ---------------------------------------------------------------------------

/// Per-head RMSNorm applied to Q and K (each with a learned scale weight);
/// V gets the same RMSNorm shape but **unscaled** -- no learned weight,
/// mathematically `rms_norm(v, ones, head_dim, eps)`.
/// (`Gemma4TextAttention.__init__` constructs `q_norm`/`k_norm` with
/// `with_scale` defaulted `True` and `v_norm = Gemma4RMSNorm(head_dim, eps,
/// with_scale=False)` at `modeling_gemma4.py:1208-1212`; `forward` applies
/// `q_norm` then RoPE (`:1240-1241`), `k_norm` then RoPE (`:1257`), and
/// `v_norm` with **no** RoPE at all (`:1260`) -- V never rotates, see
/// [`gemma4_apply_rope`].)
///
/// Substituting a scaled norm for V (i.e. reusing `q_gamma`/`k_gamma`-style
/// weights on V) is this stage's V-norm negative test
/// (`tests::mutation_scaled_v_norm_fails_golden`).
pub fn gemma4_qk_norm_v_unscaled(
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    q_gamma: &[f32],
    k_gamma: &[f32],
    head_dim: usize,
    eps: f32,
) {
    rms_norm(q, q_gamma, head_dim, eps);
    rms_norm(k, k_gamma, head_dim, eps);
    let ones = vec![1.0f32; head_dim];
    rms_norm(v, &ones, head_dim, eps);
}

// ---------------------------------------------------------------------------
// G10b: final logit softcapping.
// ---------------------------------------------------------------------------

/// Final logit softcapping: `cap * tanh(logits / cap)`
/// (`modeling_gemma4.py:1889-1892`: divide, tanh, multiply, in that order --
/// mirrored here rather than the algebraically-equivalent `cap *
/// (logits/cap).tanh()` reassociation, though at this stage's tight
/// tolerance the two are indistinguishable).
pub fn gemma4_logit_softcap(logits: &mut [f32], cap: f32) {
    for v in logits.iter_mut() {
        *v /= cap;
        *v = v.tanh();
        *v *= cap;
    }
}

// ---------------------------------------------------------------------------
// G8: dual RoPE.
// ---------------------------------------------------------------------------

/// Build a per-layer-type RoPE inverse-frequency table of length
/// `head_dim / 2`.
///
/// Local (sliding) layers pass `partial_rotary_factor: None` (the
/// `rope_type="default"` path,
/// `Gemma4TextRotaryEmbedding.compute_default_rope_parameters`,
/// `modeling_gemma4.py:1120-1156`): every slot is
/// `1 / theta^(2i / head_dim)`.
///
/// Global (full-attention) layers pass `Some(partial_rotary_factor)` (the
/// `rope_type="proportional"` path, `_compute_proportional_rope_parameters`,
/// `transformers/modeling_rope_utils.py:187-252`): only the first
/// `(partial_rotary_factor * head_dim / 2) as usize` slots are the computed
/// frequency; the rest are **exactly zero**. A zero frequency makes
/// `cos=1, sin=0` for that dimension pair in [`gemma4_rope_cos_sin`] /
/// [`gemma4_apply_rope`] -- i.e. those dimensions pass through unrotated,
/// rather than being truncated from the tensor. `cos`/`sin` stay
/// `head_dim`-wide either way.
pub fn gemma4_rope_inv_freq(
    head_dim: usize,
    theta: f64,
    partial_rotary_factor: Option<f32>,
) -> Vec<f32> {
    assert!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "head_dim must be a positive even number"
    );
    let half = head_dim / 2;
    let rope_angles = match partial_rotary_factor {
        Some(factor) => ((factor * head_dim as f32) / 2.0) as usize,
        None => half,
    };
    (0..half)
        .map(|i| {
            if i < rope_angles {
                (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Build `[positions.len() * head_dim]` cos/sin tables from an inv_freq
/// table (`Gemma4TextRotaryEmbedding.forward`,
/// `modeling_gemma4.py:1163-1172`: `freqs = position ⊗ inv_freq`,
/// `emb = cat(freqs, freqs)`, `cos = emb.cos()`, `sin = emb.sin()` --
/// `attention_scaling` is `1.0` for both the `"default"` and
/// `"proportional"` RoPE types used here, so it is omitted).
pub fn gemma4_rope_cos_sin(inv_freq: &[f32], positions: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let head_dim = half * 2;
    let mut cos = vec![0f32; positions.len() * head_dim];
    let mut sin = vec![0f32; positions.len() * head_dim];
    for (t, &pos) in positions.iter().enumerate() {
        for i in 0..half {
            let angle = pos as f32 * inv_freq[i];
            let (s, c) = angle.sin_cos();
            cos[t * head_dim + i] = c;
            cos[t * head_dim + half + i] = c;
            sin[t * head_dim + i] = s;
            sin[t * head_dim + half + i] = s;
        }
    }
    (cos, sin)
}

/// Apply RoPE to a `[seq_len, heads, head_dim]` tensor (row-major, `head_dim`
/// fastest-varying) via `x * cos + rotate_half(x) * sin`
/// (`apply_rotary_pos_emb`, `modeling_gemma4.py:787-806`).
///
/// `rotate_half` (`modeling_gemma4.py:780-785`) pairs dimension `i` with
/// dimension `head_dim/2 + i` (**stride-half**: `x1 = x[..half]`,
/// `x2 = x[half..]`, `rotate = cat(-x2, x1)`), **not** the interleaved
/// `(2i, 2i+1)` convention -- this repo's worst historical bug class
/// (`CLAUDE.md` "RoPE pairing alt-forward variants"; ADR-082's own Risks
/// section names this exact failure mode). Verified directly against the
/// pinned source line-for-line above, not inferred from a comment.
///
/// Swapping the local/global `theta` passed to [`gemma4_rope_inv_freq`]
/// before calling this is this stage's dual-RoPE negative test
/// (`tests::mutation_swapped_theta_fails_golden`).
pub fn gemma4_apply_rope(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) {
    assert_eq!(
        x.len(),
        seq_len * heads * head_dim,
        "x must be seq_len*heads*head_dim"
    );
    assert_eq!(
        cos.len(),
        seq_len * head_dim,
        "cos must be seq_len*head_dim"
    );
    assert_eq!(
        sin.len(),
        seq_len * head_dim,
        "sin must be seq_len*head_dim"
    );
    let half = head_dim / 2;
    let mut rotated = vec![0f32; head_dim];
    for t in 0..seq_len {
        let cos_row = &cos[t * head_dim..(t + 1) * head_dim];
        let sin_row = &sin[t * head_dim..(t + 1) * head_dim];
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            let row = &mut x[base..base + head_dim];
            for i in 0..half {
                rotated[i] = -row[half + i];
                rotated[half + i] = row[i];
            }
            for i in 0..head_dim {
                row[i] = row[i] * cos_row[i] + rotated[i] * sin_row[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
        serde_json::from_str(&data)
            .unwrap_or_else(|e| panic!("parse fixture {}: {e}", path.display()))
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
        gemma4_geglu_mlp(
            &x,
            &gate_w,
            &up_w,
            &down_w,
            tokens,
            hidden,
            intermediate,
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
}
