//! Library-level pipeline composing the QuaRot offline conversion stages
//! into reusable functions: **load** rotated-pipeline-input tensors as f64,
//! **fuse** shifted-RMSNorm `(1 + gamma)` into downstream linears, then
//! **absorb** the Hadamard rotation per [`RotationPlan`] into planned tensors.
//!
//! Step 3c-2 of ADR-044 (see `docs/adr/ADR-044-quarot-rotated-quantization.md`).
//!
//! ## What this module owns
//!
//! Pure library functions over a `HashMap<String, TensorEntry>` working set.
//! The pipeline mutates tensor data in place between stages; quantization
//! happens AFTER this pipeline by the caller (see step 3c-1's
//! `quantize_f64_to_q4` in `weights/q4_weights.rs`).
//!
//! ## What this module does NOT own (deferred to later sub-steps)
//!
//! - **`lm_head` materialization + final_norm fusion** (step 3c-3). When
//!   `tie_word_embeddings=true`, `lm_head.weight` is not present in the
//!   input SafeTensors; the binary must clone `embed_tokens.weight` and
//!   fuse `(1 + g_final)` into the materialized matrix BEFORE this
//!   pipeline absorbs the residual rotation onto it. Until 3c-3 lands,
//!   callers either (a) target untied configs only, or (b) pre-populate
//!   `lm_head.weight` in the working set before invoking
//!   [`absorb_rotations`].
//! - **Forward-equivalence assertion** with refuse-on-fail (step 3c-4).
//! - **Binary entry point + CLI** (step 3c-5).
//! - **MoE expert weights** — deferred to v1 per `plan.rs` §Deferred.
//!   [`qwen35_per_layer_fusion_plan`] refuses MoE configs explicitly;
//!   [`absorb_rotations`] would silently skip MoE expert tensors (no
//!   matching plan rule) — which is correctness-incomplete, not just
//!   slow, so do NOT feed an MoE config to this pipeline yet.
//!
//! ## Order of operations (MANDATORY)
//!
//! ```text
//!   1. load_tensors_f64(reader, names) → working_set
//!   2. fuse_rmsnorms(working_set, fusion_plan)
//!   3. absorb_rotations(working_set, rotation_plan, rotation)
//!   4. (caller) quantize_f64_to_q4 per planned tensor
//! ```
//!
//! Step 2 MUST happen before step 3. Diagonal scale `(1 + gamma)` does not
//! commute with Hadamard rotation; fusing after rotation produces wrong
//! outputs. See `rmsnorm_fusion.rs` module doc for the algebra.

use std::collections::HashMap;

use crate::error::InferenceError;
use crate::quant::quarot::hadamard::RandomizedHadamard;
use crate::quant::quarot::io::QuarotTensorReader;
use crate::quant::quarot::plan::{AbsorptionSide, RotationId, RotationPlan};
use crate::quant::quarot::rmsnorm_fusion::{
    RmsNormFusionTarget, fuse_shifted_rmsnorm_into_next_layer_f64, neutralize_rmsnorm_gamma_f64,
};
use crate::quant::quarot::rotation::{absorb_input_rotation_f64, absorb_output_rotation_f64};

/// One in-memory tensor in the conversion working set.
///
/// `data` is f64 throughout the rotation+fusion pipeline (per ADR-044
/// §Risks — quantize in f32, but keep transformation math in f64).
/// `shape` matches the SafeTensors header convention (row-major).
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

/// Read every tensor in `names` from `reader` as f64 and assemble a
/// working-set map keyed by tensor name.
///
/// Missing names propagate the reader's
/// [`InferenceError::MissingTensor`]; the working set is built in-order
/// and a partial set on error is dropped before return.
pub fn load_tensors_f64(
    reader: &QuarotTensorReader,
    names: &[String],
) -> Result<HashMap<String, TensorEntry>, InferenceError> {
    let mut out = HashMap::with_capacity(names.len());
    for name in names {
        let (data, shape) = reader.read_tensor_f64(name)?;
        out.insert(
            name.clone(),
            TensorEntry {
                name: name.clone(),
                shape,
                data,
            },
        );
    }
    Ok(out)
}

/// Apply every [`RmsNormFusionTarget`] in `plan`: for each entry, multiply
/// every listed downstream weight tensor by the norm's `(1 + gamma)` as a
/// column scale, then neutralize the norm to all-zeros so runtime
/// `(1 + 0) = 1` evaluates to identity.
///
/// Missing tensors (norm OR downstream) error out — the caller passed
/// `names` to [`load_tensors_f64`] that did not match this fusion plan,
/// which is always a converter bug.
///
/// Shape contract: every downstream weight is row-major `[rows × cols]`
/// where `cols` matches the norm's length. Mismatches return an error.
pub fn fuse_rmsnorms(
    tensors: &mut HashMap<String, TensorEntry>,
    plan: &[RmsNormFusionTarget],
) -> Result<(), InferenceError> {
    for target in plan {
        let gamma = tensors.get(&target.norm_tensor).ok_or_else(|| {
            InferenceError::Inference(format!(
                "fuse_rmsnorms: norm tensor `{}` not in working set",
                target.norm_tensor
            ))
        })?;
        if gamma.shape.len() != 1 {
            return Err(InferenceError::Inference(format!(
                "fuse_rmsnorms: norm tensor `{}` shape {:?} is not 1-D",
                target.norm_tensor, gamma.shape
            )));
        }
        let gamma_data = gamma.data.clone();
        let cols = gamma_data.len();

        for w_name in &target.downstream_weights {
            let w = tensors.get_mut(w_name).ok_or_else(|| {
                InferenceError::Inference(format!(
                    "fuse_rmsnorms: downstream weight `{w_name}` not in working set"
                ))
            })?;
            if w.shape.len() != 2 {
                return Err(InferenceError::Inference(format!(
                    "fuse_rmsnorms: downstream weight `{w_name}` shape {:?} is not 2-D",
                    w.shape
                )));
            }
            let (rows, w_cols) = (w.shape[0], w.shape[1]);
            if w_cols != cols {
                return Err(InferenceError::Inference(format!(
                    "fuse_rmsnorms: downstream weight `{w_name}` cols={w_cols} != \
                     norm `{}` len={cols}",
                    target.norm_tensor
                )));
            }
            fuse_shifted_rmsnorm_into_next_layer_f64(&mut w.data, rows, cols, &gamma_data)?;
        }

        let norm_entry = tensors.get_mut(&target.norm_tensor).ok_or_else(|| {
            InferenceError::Inference(format!(
                "fuse_rmsnorms: norm tensor `{}` disappeared from working set \
                 between gamma capture and neutralization; this should be unreachable",
                target.norm_tensor
            ))
        })?;
        neutralize_rmsnorm_gamma_f64(&mut norm_entry.data);
    }
    Ok(())
}

/// Apply every plan rule in `rotation_plan` that matches a tensor in the
/// working set, absorbing the corresponding rotation per [`AbsorptionSide`].
///
/// Tensors with no matching plan rule are left untouched — this includes
/// norms (already neutralized by [`fuse_rmsnorms`]), conv1d, A_log, etc.
///
/// The plan currently knows only [`RotationId::ResidualStream`]. The
/// rotation passed in MUST have `dim() == hidden_size` for the model.
/// Future plans with per-head rotations (v1) will pass a
/// `HashMap<RotationId, RandomizedHadamard>` instead.
///
/// # Errors
///
/// - A planned tensor's shape is not 2-D.
/// - Input-side absorption: tensor `cols != rotation.dim()`.
/// - Output-side absorption: tensor `rows != rotation.dim()`.
pub fn absorb_rotations(
    tensors: &mut HashMap<String, TensorEntry>,
    rotation_plan: &RotationPlan,
    rotation: &RandomizedHadamard,
) -> Result<(), InferenceError> {
    for entry in tensors.values_mut() {
        let Some(tensor_rotation) = rotation_plan.for_tensor(&entry.name) else {
            continue;
        };
        if tensor_rotation.rotation_id != RotationId::ResidualStream {
            return Err(InferenceError::Inference(format!(
                "absorb_rotations: tensor `{}` requested non-residual rotation \
                 `{:?}`; only `ResidualStream` is implemented (v1: per-head)",
                entry.name, tensor_rotation.rotation_id
            )));
        }
        if entry.shape.len() != 2 {
            return Err(InferenceError::Inference(format!(
                "absorb_rotations: tensor `{}` shape {:?} is not 2-D",
                entry.name, entry.shape
            )));
        }
        let (rows, cols) = (entry.shape[0], entry.shape[1]);
        match tensor_rotation.side {
            AbsorptionSide::InputSide => {
                absorb_input_rotation_f64(&mut entry.data, rows, cols, rotation)?;
            }
            AbsorptionSide::OutputSide => {
                absorb_output_rotation_f64(&mut entry.data, rows, cols, rotation)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen35_config::Qwen35Config;
    use crate::quant::quarot::rmsnorm_fusion::qwen35_per_layer_fusion_plan;

    fn synthetic_f64(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 11) as u32;
                (bits as f64 / u32::MAX as f64) - 0.5
            })
            .collect()
    }

    fn insert_tensor(
        tensors: &mut HashMap<String, TensorEntry>,
        name: &str,
        shape: Vec<usize>,
        data: Vec<f64>,
    ) {
        tensors.insert(
            name.to_string(),
            TensorEntry {
                name: name.to_string(),
                shape,
                data,
            },
        );
    }

    fn matvec_f64(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), cols);
        let mut y = vec![0.0_f64; rows];
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            y[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        }
        y
    }

    fn max_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn fuse_rmsnorms_zeros_norm_and_folds_into_downstream() {
        let mut tensors = HashMap::new();
        let cols = 4;
        let rows = 3;
        insert_tensor(
            &mut tensors,
            "n.weight",
            vec![cols],
            vec![0.0, 1.0, -0.5, 2.0],
        );
        insert_tensor(
            &mut tensors,
            "w.weight",
            vec![rows, cols],
            (0..(rows * cols)).map(|k| k as f64 + 1.0).collect(),
        );

        let plan = vec![RmsNormFusionTarget {
            norm_tensor: "n.weight".to_string(),
            downstream_weights: vec!["w.weight".to_string()],
        }];
        fuse_rmsnorms(&mut tensors, &plan).unwrap();

        // Norm zeroed
        assert_eq!(tensors["n.weight"].data, vec![0.0; cols]);
        // Downstream column-multiplied by (1 + gamma) = [1, 2, 0.5, 3]
        assert_eq!(tensors["w.weight"].data[0..4], [1.0, 4.0, 1.5, 12.0]);
        assert_eq!(tensors["w.weight"].data[4..8], [5.0, 12.0, 3.5, 24.0]);
        assert_eq!(tensors["w.weight"].data[8..12], [9.0, 20.0, 5.5, 36.0]);
    }

    #[test]
    fn fuse_rmsnorms_errors_on_missing_norm() {
        let mut tensors = HashMap::new();
        insert_tensor(&mut tensors, "w.weight", vec![3, 4], vec![0.0; 12]);
        let plan = vec![RmsNormFusionTarget {
            norm_tensor: "missing.weight".to_string(),
            downstream_weights: vec!["w.weight".to_string()],
        }];
        let err = fuse_rmsnorms(&mut tensors, &plan).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("missing.weight"), "unexpected error: {msg}");
    }

    #[test]
    fn fuse_rmsnorms_errors_on_missing_downstream() {
        let mut tensors = HashMap::new();
        insert_tensor(&mut tensors, "n.weight", vec![4], vec![0.0; 4]);
        let plan = vec![RmsNormFusionTarget {
            norm_tensor: "n.weight".to_string(),
            downstream_weights: vec!["absent.weight".to_string()],
        }];
        let err = fuse_rmsnorms(&mut tensors, &plan).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("absent.weight"), "unexpected error: {msg}");
    }

    #[test]
    fn fuse_rmsnorms_errors_on_dimension_mismatch() {
        let mut tensors = HashMap::new();
        insert_tensor(&mut tensors, "n.weight", vec![4], vec![0.0; 4]);
        insert_tensor(&mut tensors, "w.weight", vec![3, 5], vec![0.0; 15]);
        let plan = vec![RmsNormFusionTarget {
            norm_tensor: "n.weight".to_string(),
            downstream_weights: vec!["w.weight".to_string()],
        }];
        let err = fuse_rmsnorms(&mut tensors, &plan).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("cols=5"), "unexpected error: {msg}");
    }

    #[test]
    fn fuse_rmsnorms_errors_on_non_1d_norm() {
        let mut tensors = HashMap::new();
        insert_tensor(&mut tensors, "n.weight", vec![2, 2], vec![0.0; 4]);
        insert_tensor(&mut tensors, "w.weight", vec![3, 4], vec![0.0; 12]);
        let plan = vec![RmsNormFusionTarget {
            norm_tensor: "n.weight".to_string(),
            downstream_weights: vec!["w.weight".to_string()],
        }];
        let err = fuse_rmsnorms(&mut tensors, &plan).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not 1-D"), "unexpected error: {msg}");
    }

    #[test]
    fn absorb_rotations_applies_planned_sides() {
        let hidden = 16;
        let inter = 32;
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let r = RandomizedHadamard::new(0xCAFE_BABE, hidden).unwrap();

        let mut tensors = HashMap::new();
        let q_name = "model.language_model.layers.0.self_attn.q_proj.weight";
        let o_name = "model.language_model.layers.0.self_attn.o_proj.weight";
        let gate_name = "model.language_model.layers.0.mlp.gate_proj.weight";
        let down_name = "model.language_model.layers.0.mlp.down_proj.weight";

        // Input-side targets: shape [rows, hidden] (cols == hidden_size)
        let q_data = synthetic_f64(inter * hidden, 1);
        let gate_data = synthetic_f64(inter * hidden, 2);
        // Output-side targets: shape [hidden, ...] (rows == hidden_size)
        let o_data = synthetic_f64(hidden * inter, 3);
        let down_data = synthetic_f64(hidden * inter, 4);

        insert_tensor(&mut tensors, q_name, vec![inter, hidden], q_data.clone());
        insert_tensor(&mut tensors, o_name, vec![hidden, inter], o_data.clone());
        insert_tensor(
            &mut tensors,
            gate_name,
            vec![inter, hidden],
            gate_data.clone(),
        );
        insert_tensor(
            &mut tensors,
            down_name,
            vec![hidden, inter],
            down_data.clone(),
        );

        absorb_rotations(&mut tensors, &plan, &r).unwrap();

        // Input-side: forward y = W · x must equal forward of (W · R^T) · (R · x)
        let x = synthetic_f64(hidden, 5);
        let y_orig = matvec_f64(&q_data, inter, hidden, &x);
        let mut x_rot = x.clone();
        r.apply_f64(&mut x_rot).unwrap();
        let y_after = matvec_f64(&tensors[q_name].data, inter, hidden, &x_rot);
        let delta = max_abs_diff_f64(&y_orig, &y_after);
        assert!(
            delta < 1e-12,
            "q_proj input-side absorption diverged: {delta}"
        );

        // Output-side: forward of (R · W) · x must equal R · (W · x)
        let x2 = synthetic_f64(inter, 6);
        let y2_orig = matvec_f64(&o_data, hidden, inter, &x2);
        let mut y2_rot_expected = y2_orig.clone();
        r.apply_f64(&mut y2_rot_expected).unwrap();
        let y2_after = matvec_f64(&tensors[o_name].data, hidden, inter, &x2);
        let delta2 = max_abs_diff_f64(&y2_rot_expected, &y2_after);
        assert!(
            delta2 < 1e-12,
            "o_proj output-side absorption diverged: {delta2}"
        );
    }

    #[test]
    fn absorb_rotations_leaves_unplanned_tensors_untouched() {
        // `linear_attn.norm.weight` is unplanned (plain-gamma GDN norm; see plan.rs).
        let hidden = 16;
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let r = RandomizedHadamard::new(0xFEED_FACE, hidden).unwrap();

        let mut tensors = HashMap::new();
        let untouched_name = "model.language_model.layers.5.linear_attn.norm.weight";
        let untouched_data = synthetic_f64(hidden, 7);
        insert_tensor(
            &mut tensors,
            untouched_name,
            vec![hidden],
            untouched_data.clone(),
        );

        absorb_rotations(&mut tensors, &plan, &r).unwrap();

        assert_eq!(tensors[untouched_name].data, untouched_data);
    }

    #[test]
    fn absorb_rotations_errors_on_non_2d_planned_tensor() {
        let hidden = 16;
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let r = RandomizedHadamard::new(1, hidden).unwrap();

        let mut tensors = HashMap::new();
        // q_proj has a 2-D plan rule; pass it as 1-D to trigger the shape error.
        insert_tensor(
            &mut tensors,
            "model.language_model.layers.0.self_attn.q_proj.weight",
            vec![hidden],
            vec![0.0; hidden],
        );

        let err = absorb_rotations(&mut tensors, &plan, &r).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not 2-D"), "unexpected error: {msg}");
    }

    /// End-to-end algebra check: fuse-then-rotate-then-forward (rotated) equals
    /// original forward. Two-layer toy: input_layernorm → q_proj (input-side
    /// absorbed) and the residual is pre-rotated by R upstream.
    #[test]
    fn fuse_then_absorb_preserves_forward_pass() {
        let hidden = 16;
        let inter = 32;
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let r = RandomizedHadamard::new(0xABCDEF, hidden).unwrap();

        let q_name = "model.language_model.layers.0.self_attn.q_proj.weight";
        let n_name = "model.language_model.layers.0.input_layernorm.weight";

        let q_orig = synthetic_f64(inter * hidden, 11);
        let g_orig = synthetic_f64(hidden, 12);
        let x_normalized = synthetic_f64(hidden, 13);

        // Original forward: y = q_proj · ((1 + g) ⊙ normalize(h))
        let pre_q: Vec<f64> = x_normalized
            .iter()
            .zip(g_orig.iter())
            .map(|(xi, gi)| xi * (1.0 + gi))
            .collect();
        let y_orig = matvec_f64(&q_orig, inter, hidden, &pre_q);

        // Build working set, fuse, absorb.
        let mut tensors = HashMap::new();
        insert_tensor(&mut tensors, n_name, vec![hidden], g_orig.clone());
        insert_tensor(&mut tensors, q_name, vec![inter, hidden], q_orig.clone());

        let fusion_plan = vec![RmsNormFusionTarget {
            norm_tensor: n_name.to_string(),
            downstream_weights: vec![q_name.to_string()],
        }];
        fuse_rmsnorms(&mut tensors, &fusion_plan).unwrap();
        absorb_rotations(&mut tensors, &plan, &r).unwrap();

        // Runtime: normalize then rotate (norm is now zero, so `(1+0)·x = x`),
        // then q_proj_fused_rotated · (R · x).
        let mut x_rot = x_normalized.clone();
        r.apply_f64(&mut x_rot).unwrap();
        let y_after = matvec_f64(&tensors[q_name].data, inter, hidden, &x_rot);

        let delta = max_abs_diff_f64(&y_orig, &y_after);
        assert!(
            delta < 1e-12,
            "fuse + absorb pipeline diverged from original: {delta}"
        );
    }

    #[test]
    fn qwen35_fusion_plan_targets_resolve_within_required_tensor_set() {
        // Sanity: every fusion plan target name must be a tensor that the
        // loader would actually request. This is an integration check
        // between the fusion plan and the loader's required-name helper.
        let cfg = Qwen35Config::qwen35_0_8b();
        let required: std::collections::HashSet<String> =
            crate::model::qwen35::qwen_required_tensor_names(&cfg)
                .into_iter()
                .collect();
        let plan = qwen35_per_layer_fusion_plan(&cfg).unwrap();
        for target in &plan {
            assert!(
                required.contains(&target.norm_tensor),
                "{}",
                target.norm_tensor
            );
            for d in &target.downstream_weights {
                assert!(required.contains(d), "{d}");
            }
        }
    }
}
