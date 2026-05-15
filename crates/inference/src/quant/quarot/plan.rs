//! [`RotationPlan`]: which rotation is absorbed into which tensor, on which
//! side. This module owns the architecture-specific recipes (currently
//! Qwen3.5 hybrid) — the math primitives in [`super::rotation`] are
//! architecture-agnostic, this module says "in Qwen3.5, the attention
//! `q_proj` takes the residual-stream rotation on its input side."
//!
//! Step 3 of [ADR-044](../../../../../docs/adr/ADR-044-quarot-rotated-quantization.md)
//! glues this plan to SafeTensors I/O and the Q4 bridge.
//!
//! ## Scope
//!
//! v0 / step 3 handles the **residual-stream rotation only** — a single
//! rotation `R_res` of dimension `hidden_size`. Every linear layer that
//! reads or writes the residual stream gets `R_res` absorbed on the
//! corresponding side:
//!
//! | Tensor | Reads from residual | Writes to residual | Absorption |
//! |---|---|---|---|
//! | `attn.q_proj`, `attn.k_proj`, `attn.v_proj` | ✓ | — | input-side |
//! | `attn.o_proj` | — | ✓ | output-side |
//! | `mlp.gate_proj`, `mlp.up_proj` | ✓ | — | input-side |
//! | `mlp.down_proj` | — | ✓ | output-side |
//! | `lm_head` | ✓ | — | input-side |
//! | `embed_tokens` | — | ✓ | output-side |
//!
//! Plus pre- and post-norm RMSNorm weights that scale the residual: those
//! need `g ← R · g` (scale becomes rotated). RMSNorm only multiplies
//! element-wise so the rotation must be commuted through carefully — for
//! Hadamard rotations and RMSNorm specifically, this is the QuaRot
//! identity used in the paper.
//!
//! **Deferred to v1** (per ADR-044 §Scope, "Out of v0 entirely"):
//! - Per-head-dim rotations on QKV head spaces (improves activation
//!   quantization; not needed for weight-only Q4)
//! - MoE expert weights (DeepSeekMoE-style routed experts in Qwen3.5
//!   MoE layers — same absorption pattern but applied per expert slice)
//! - GatedDeltaNet linear-attention layers (different residual-stream
//!   interface; needs separate analysis)

use crate::error::InferenceError;
use crate::quant::quarot::hadamard::RandomizedHadamard;
use crate::quant::quarot::rotation::{
    absorb_input_rotation, absorb_input_rotation_f64, absorb_output_rotation,
    absorb_output_rotation_f64,
};

/// Which side of a linear layer's weight matrix gets the rotation absorbed.
///
/// `InputSide`: `W ← W · R^T`. Used when the layer reads from a residual
/// stream that gets pre-rotated by `R` upstream.
///
/// `OutputSide`: `W ← R · W`. Used when the layer writes to a residual
/// stream that should be pre-rotated by `R` for downstream consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsorptionSide {
    InputSide,
    OutputSide,
}

/// Plan for a single weight tensor: which rotation goes on which side.
///
/// A tensor with no rotation should not appear in the plan at all —
/// `RotationPlan::for_tensor` returns `None` for unplanned tensors so the
/// caller can decide whether to pass it through unchanged or warn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorRotation {
    pub side: AbsorptionSide,
    pub rotation_id: RotationId,
}

/// Stable identifier for a planned rotation. The actual [`RandomizedHadamard`]
/// is constructed once from `(seed, dim)` when the plan is materialized for
/// execution — the plan itself does not own rotations so it stays cheap to
/// clone and serialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RotationId {
    /// `R_res` of dimension `hidden_size`.
    ResidualStream,
}

/// Plan binding tensor-name patterns to [`TensorRotation`] entries.
///
/// Patterns are matched as suffixes of the tensor's SafeTensors name —
/// e.g., a pattern `"self_attn.q_proj.weight"` matches
/// `"model.layers.0.self_attn.q_proj.weight"` and
/// `"model.layers.23.self_attn.q_proj.weight"`. This is intentionally simple;
/// Qwen3.5's naming is regular enough that suffix-matching covers it
/// without a full glob engine.
#[derive(Debug, Clone)]
pub struct RotationPlan {
    rules: Vec<(String, TensorRotation)>,
}

impl RotationPlan {
    /// Plan for Qwen3.5-style decoder-only models with a single
    /// residual-stream rotation absorbed across every weight that touches
    /// the residual.
    ///
    /// Covers: GQA q_proj/k_proj/v_proj input-side, GQA o_proj output-side,
    /// dense MLP gate/up input-side and down output-side, embed_tokens
    /// output-side, lm_head input-side.
    ///
    /// Does NOT cover MoE expert weights (deferred — see module doc),
    /// GatedDeltaNet linear-attention layers (deferred), and per-head-dim
    /// rotations (deferred to v1).
    pub fn qwen35_residual_stream() -> Self {
        let rs = TensorRotation {
            side: AbsorptionSide::InputSide,
            rotation_id: RotationId::ResidualStream,
        };
        let rs_out = TensorRotation {
            side: AbsorptionSide::OutputSide,
            rotation_id: RotationId::ResidualStream,
        };
        Self {
            rules: vec![
                ("self_attn.q_proj.weight".into(), rs),
                ("self_attn.k_proj.weight".into(), rs),
                ("self_attn.v_proj.weight".into(), rs),
                ("self_attn.o_proj.weight".into(), rs_out),
                ("mlp.gate_proj.weight".into(), rs),
                ("mlp.up_proj.weight".into(), rs),
                ("mlp.down_proj.weight".into(), rs_out),
                ("embed_tokens.weight".into(), rs_out),
                ("lm_head.weight".into(), rs),
            ],
        }
    }

    /// Look up the rotation for a tensor by its SafeTensors name.
    pub fn for_tensor(&self, name: &str) -> Option<TensorRotation> {
        self.rules
            .iter()
            .find(|(pat, _)| name.ends_with(pat))
            .map(|(_, r)| *r)
    }

    /// Number of pattern rules in the plan. Useful for dimensional sanity
    /// checks but not for runtime dispatch.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

/// Apply a planned rotation to a single weight tensor (f32 in place).
///
/// Returns `Ok(true)` if the tensor matched a plan rule and was rotated,
/// `Ok(false)` if the tensor was not in the plan (caller decides whether
/// to pass through or warn), or an error on shape / dimension mismatch.
pub fn apply_tensor_rotation(
    name: &str,
    weight: &mut [f32],
    rows: usize,
    cols: usize,
    plan: &RotationPlan,
    residual_rotation: &RandomizedHadamard,
) -> Result<bool, InferenceError> {
    let Some(tr) = plan.for_tensor(name) else {
        return Ok(false);
    };
    let rotation = match tr.rotation_id {
        RotationId::ResidualStream => residual_rotation,
    };
    match tr.side {
        AbsorptionSide::InputSide => absorb_input_rotation(weight, rows, cols, rotation)?,
        AbsorptionSide::OutputSide => absorb_output_rotation(weight, rows, cols, rotation)?,
    }
    Ok(true)
}

/// `f64` variant of [`apply_tensor_rotation`]. Step 3 will use this when
/// running the absorption pass in f64 precision per ADR-044 §Risks.
pub fn apply_tensor_rotation_f64(
    name: &str,
    weight: &mut [f64],
    rows: usize,
    cols: usize,
    plan: &RotationPlan,
    residual_rotation: &RandomizedHadamard,
) -> Result<bool, InferenceError> {
    let Some(tr) = plan.for_tensor(name) else {
        return Ok(false);
    };
    let rotation = match tr.rotation_id {
        RotationId::ResidualStream => residual_rotation,
    };
    match tr.side {
        AbsorptionSide::InputSide => absorb_input_rotation_f64(weight, rows, cols, rotation)?,
        AbsorptionSide::OutputSide => absorb_output_rotation_f64(weight, rows, cols, rotation)?,
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35_plan_covers_dense_residual_stream_tensors() {
        let plan = RotationPlan::qwen35_residual_stream();
        assert_eq!(plan.rule_count(), 9);

        let cases = [
            (
                "model.layers.0.self_attn.q_proj.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.5.self_attn.k_proj.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.5.self_attn.v_proj.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.5.self_attn.o_proj.weight",
                AbsorptionSide::OutputSide,
            ),
            (
                "model.layers.23.mlp.gate_proj.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.23.mlp.up_proj.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.23.mlp.down_proj.weight",
                AbsorptionSide::OutputSide,
            ),
            ("model.embed_tokens.weight", AbsorptionSide::OutputSide),
            ("lm_head.weight", AbsorptionSide::InputSide),
        ];
        for (name, expected_side) in cases {
            let tr = plan
                .for_tensor(name)
                .unwrap_or_else(|| panic!("plan missed tensor {name}"));
            assert_eq!(tr.side, expected_side, "wrong side for {name}");
            assert_eq!(tr.rotation_id, RotationId::ResidualStream);
        }
    }

    #[test]
    fn qwen35_plan_misses_moe_expert_weights() {
        // Documents the v1 deferral — MoE expert tensors are NOT in the v0 plan.
        let plan = RotationPlan::qwen35_residual_stream();
        let moe_names = [
            "model.layers.0.mlp.experts.gate_up_proj.weight",
            "model.layers.0.mlp.experts.down_proj.weight",
        ];
        for name in moe_names {
            assert!(
                plan.for_tensor(name).is_none(),
                "MoE expert tensor {name} should not match v0 plan"
            );
        }
    }

    #[test]
    fn qwen35_plan_misses_gdn_linear_attention_weights() {
        // Documents the v1 deferral — GatedDeltaNet tensors are NOT in v0.
        let plan = RotationPlan::qwen35_residual_stream();
        let gdn_names = [
            "model.layers.0.linear_attn.A_log",
            "model.layers.0.linear_attn.dt_bias",
            "model.layers.0.linear_attn.in_proj.weight",
            "model.layers.0.linear_attn.out_proj.weight",
        ];
        for name in gdn_names {
            assert!(
                plan.for_tensor(name).is_none(),
                "GDN tensor {name} should not match v0 plan"
            );
        }
    }

    #[test]
    fn apply_tensor_rotation_skips_unplanned() {
        let plan = RotationPlan::qwen35_residual_stream();
        let hidden = 64;
        let r = RandomizedHadamard::new(7, hidden).unwrap();
        let mut weight = vec![1.0_f32; hidden * hidden];
        let weight_copy = weight.clone();

        let rotated = apply_tensor_rotation(
            "model.layers.0.linear_attn.in_proj.weight",
            &mut weight,
            hidden,
            hidden,
            &plan,
            &r,
        )
        .unwrap();
        assert!(!rotated, "unplanned tensor should report not rotated");
        assert_eq!(weight, weight_copy, "unplanned tensor must not be mutated");
    }

    #[test]
    fn apply_tensor_rotation_mlp_layer_pair_rotates_output() {
        // Dense MLP layer pair: gate_proj (input-side) + down_proj (output-side).
        // Rotation R is on the residual stream. After absorption:
        //   gate_proj' · (R · x) = gate_proj · x   (intermediate unchanged)
        //   down_proj' · intermediate = R · (down_proj · intermediate) = R · y_original
        // So the rotated pipeline output equals R · y_original, NOT y_original —
        // because down_proj writes BACK to the rotated residual stream.
        let hidden = 64;
        let intermediate = 128;
        let plan = RotationPlan::qwen35_residual_stream();
        let r = RandomizedHadamard::new(0xC0FFEE, hidden).unwrap();

        let mut state = 1_u64;
        let mut rand = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as u32 as f32 / u32::MAX as f32) - 0.5
        };
        let gate_proj: Vec<f32> = (0..intermediate * hidden).map(|_| rand()).collect();
        let down_proj: Vec<f32> = (0..hidden * intermediate).map(|_| rand()).collect();
        let x: Vec<f32> = (0..hidden).map(|_| rand()).collect();

        let matvec = |w: &[f32], r: usize, c: usize, v: &[f32]| -> Vec<f32> {
            (0..r)
                .map(|i| (0..c).map(|j| w[i * c + j] * v[j]).sum())
                .collect()
        };
        let intermediate_out = matvec(&gate_proj, intermediate, hidden, &x);
        let y_original = matvec(&down_proj, hidden, intermediate, &intermediate_out);
        let mut y_expected = y_original.clone();
        r.apply(&mut y_expected).unwrap();

        let mut gate_proj_abs = gate_proj.clone();
        let mut down_proj_abs = down_proj.clone();
        assert!(
            apply_tensor_rotation(
                "model.layers.0.mlp.gate_proj.weight",
                &mut gate_proj_abs,
                intermediate,
                hidden,
                &plan,
                &r,
            )
            .unwrap()
        );
        assert!(
            apply_tensor_rotation(
                "model.layers.0.mlp.down_proj.weight",
                &mut down_proj_abs,
                hidden,
                intermediate,
                &plan,
                &r,
            )
            .unwrap()
        );

        let mut x_rotated = x.clone();
        r.apply(&mut x_rotated).unwrap();
        let intermediate_rot = matvec(&gate_proj_abs, intermediate, hidden, &x_rotated);
        let y_rotated = matvec(&down_proj_abs, hidden, intermediate, &intermediate_rot);

        let max_abs_diff = y_expected
            .iter()
            .zip(y_rotated.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_diff < 1e-3,
            "MLP pair output should equal R · y_original: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn f64_apply_matches_f32_apply_pattern() {
        let plan = RotationPlan::qwen35_residual_stream();
        let hidden = 64;
        let r = RandomizedHadamard::new(11, hidden).unwrap();
        let mut weight_f32: Vec<f32> = (0..hidden * hidden)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let mut weight_f64: Vec<f64> = weight_f32.iter().map(|&v| f64::from(v)).collect();

        let r1 = apply_tensor_rotation(
            "model.layers.0.self_attn.q_proj.weight",
            &mut weight_f32,
            hidden,
            hidden,
            &plan,
            &r,
        )
        .unwrap();
        let r2 = apply_tensor_rotation_f64(
            "model.layers.0.self_attn.q_proj.weight",
            &mut weight_f64,
            hidden,
            hidden,
            &plan,
            &r,
        )
        .unwrap();
        assert!(r1 && r2);

        for (i, (a, b)) in weight_f32.iter().zip(weight_f64.iter()).enumerate() {
            let delta = (f64::from(*a) - b).abs();
            assert!(
                delta < 1e-5,
                "tensor[{i}]: f32={a} vs f64={b}, delta={delta}"
            );
        }
    }
}
