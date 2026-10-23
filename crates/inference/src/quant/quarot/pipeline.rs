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
//! - **`lm_head` materialization + final_norm fusion** (step 3c-3). The
//!   shifted final RMSNorm `(1 + g_final)` must be folded into `lm_head`
//!   BEFORE [`absorb_rotations`] runs — the diagonal `D = diag(1 + g_final)`
//!   does not commute with the Hadamard rotation, so leaving the final
//!   norm online while rotating `lm_head` produces wrong logits (see
//!   `plan.rs` §Tied embeddings for the algebra). This applies to BOTH
//!   tied and untied configs:
//!     - Tied (`tie_word_embeddings=true`, Qwen3.5 default): `lm_head.weight`
//!       is absent from SafeTensors; the binary must clone `embed_tokens.weight`
//!       into a materialized `lm_head.weight`, fuse `(1 + g_final)` into it,
//!       then add a `final_norm → lm_head` fusion target and flip the output
//!       config to `tie_word_embeddings=false`.
//!     - Untied: `lm_head.weight` already exists; the caller still must add
//!       a `final_norm → lm_head` fusion target so the runtime norm becomes
//!       a no-op before rotation absorption.
//!
//!   Until step 3c-3 lands, **this module is not a complete model
//!   converter**. Callers can use it for sub-stage tests (RMSNorm fusion
//!   primitives, residual-stream rotation absorption on the planned
//!   tensors), but composing into a runnable QuaRot `.q4` artifact also
//!   requires the final-norm/`lm_head` work in 3c-3.
//! - **Forward-equivalence assertion** with refuse-on-fail (step 3c-4).
//! - **Binary entry point + CLI** (step 3c-5).
//! - **MoE expert weights** — deferred to v1 per `plan.rs` §Deferred.
//!   `qwen35_per_layer_fusion_plan` refuses MoE configs explicitly;
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
/// **Empty `downstream_weights` is rejected before any mutation.**
/// Neutralizing a norm without folding `(1 + gamma)` into a downstream
/// linear would silently change the model's output, so this is treated as
/// a converter bug rather than a no-op.
///
/// Shape contract: every downstream weight is row-major `[rows × cols]`
/// where `cols` matches the norm's length. Mismatches return an error.
pub fn fuse_rmsnorms(
    tensors: &mut HashMap<String, TensorEntry>,
    plan: &[RmsNormFusionTarget],
) -> Result<(), InferenceError> {
    for target in plan {
        if target.downstream_weights.is_empty() {
            return Err(InferenceError::Inference(format!(
                "fuse_rmsnorms: target for norm `{}` has empty `downstream_weights`; \
                 neutralizing the norm without folding the `(1 + gamma)` scale into \
                 a downstream linear would silently change the model's output",
                target.norm_tensor
            )));
        }
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
mod tests;
