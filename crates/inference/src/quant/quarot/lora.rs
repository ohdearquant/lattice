//! Rotation correction for LoRA adapters on QuaRot-converted models (ADR-045).
//!
//! When a LoRA adapter trained on an unrotated model is applied to a
//! QuaRot-converted base, both A and B matrices may need rotation correction
//! depending on the base projection's absorption side:
//!
//! **Input-side projections** (q/k/v_proj, gate/up_proj, in_proj_*):
//! The base weight absorbed `W ← W · R^T`, so the input activation is `R · h`.
//! Fix: `A_cr = A · R^T`. Then `B · A_cr · (R·h) = B · A · h` ✓
//!
//! **Output-side projections** (o_proj, down_proj, out_proj):
//! The base weight absorbed `W ← R · W`, so its output is in the rotated basis.
//! The LoRA delta must also be in the rotated basis.
//! Fix: `B_rot = R · B`. Then `B_rot · A · x = R · (B · A · x)` ✓
//!
//! Both corrections are exact (R is orthogonal) and absorbed at load time — zero
//! runtime cost.

use crate::error::InferenceError;
use crate::quant::quarot::hadamard::RandomizedHadamard;
use crate::quant::quarot::plan::{AbsorptionSide, RotationPlan};
use crate::quant::quarot::rotation::{absorb_input_rotation, absorb_output_rotation};

/// Counter-rotate a single LoRA A matrix for an input-side QuaRot projection.
///
/// Computes `A_cr = A · R^T` in-place, where A is row-major `(rank × d_in)`.
/// Applies R to each row — mathematically identical to input-side absorption.
///
/// # Errors
///
/// Returns an error if `a.len() != rank * d_in` or `rotation.dim() != d_in`.
pub fn counter_rotate_a(
    a: &mut [f32],
    rank: usize,
    d_in: usize,
    rotation: &RandomizedHadamard,
) -> Result<(), InferenceError> {
    absorb_input_rotation(a, rank, d_in, rotation)
}

/// Rotate a single LoRA B matrix for an output-side QuaRot projection.
///
/// Computes `B_rot = R · B` in-place, where B is row-major `(d_out × rank)`.
/// Applies R to each column — mathematically identical to output-side absorption.
///
/// After this, the LoRA delta `B_rot · A · x = R · (B · A · x)` is in the
/// rotated residual basis, matching the base projection's rotated output.
///
/// # Errors
///
/// Returns an error if `b.len() != d_out * rank` or `rotation.dim() != d_out`.
pub fn rotate_b_output_side(
    b: &mut [f32],
    d_out: usize,
    rank: usize,
    rotation: &RandomizedHadamard,
) -> Result<(), InferenceError> {
    absorb_output_rotation(b, d_out, rank, rotation)
}

/// Returns `true` if a LoRA module's A matrix needs counter-rotation (input-side).
pub fn needs_counter_rotation(plan: &RotationPlan, module: &str) -> bool {
    matches!(
        plan.absorption_for_module(module),
        Some(AbsorptionSide::InputSide)
    )
}

/// Returns `true` if a LoRA module's B matrix needs output-side rotation.
pub fn needs_b_rotation(plan: &RotationPlan, module: &str) -> bool {
    matches!(
        plan.absorption_for_module(module),
        Some(AbsorptionSide::OutputSide)
    )
}

/// What rotation was applied to a LoRA layer's matrices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraRotationApplied {
    /// A was counter-rotated: `A ← A · R^T` (input-side projection)
    CounterRotatedA,
    /// B was rotated: `B ← R · B` (output-side projection)
    RotatedB,
}

/// Summary of a rotation pass over an adapter's matrices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotationReport {
    /// (layer_idx, module, what_was_applied) for each processed layer.
    pub entries: Vec<(usize, String, LoraRotationApplied)>,
}

impl RotationReport {
    /// Number of layers where A was counter-rotated (input-side).
    pub fn num_a_rotated(&self) -> usize {
        self.entries
            .iter()
            .filter(|(_, _, r)| *r == LoraRotationApplied::CounterRotatedA)
            .count()
    }

    /// Number of layers where B was rotated (output-side).
    pub fn num_b_rotated(&self) -> usize {
        self.entries
            .iter()
            .filter(|(_, _, r)| *r == LoraRotationApplied::RotatedB)
            .count()
    }
}

/// A single LoRA layer's mutable references for rotation correction.
pub struct LoraLayerMut<'a> {
    /// Transformer layer index (0-based).
    pub layer_idx: usize,
    /// Projection module name (e.g., `"q_proj"`, `"o_proj"`).
    pub module: &'a str,
    /// A matrix, row-major `(rank × d_in)`. Modified for input-side modules.
    pub a: &'a mut [f32],
    /// B matrix, row-major `(d_out × rank)`. Modified for output-side modules.
    pub b: &'a mut [f32],
    /// LoRA rank (inner dimension).
    pub rank: usize,
    /// Input dimension of the base projection.
    pub d_in: usize,
    /// Output dimension of the base projection.
    pub d_out: usize,
}

/// Apply rotation corrections to all LoRA layers for a QuaRot-converted base.
///
/// For each layer:
/// - **Input-side** module: `A ← A · R^T` (counter-rotate A)
/// - **Output-side** module: `B ← R · B` (rotate B into residual basis)
/// - **Not in plan**: **ERROR** — refuses unknown targets to prevent silent
///   basis-composition failures on QuaRot bases
///
/// # Arguments
///
/// * `layers` — mutable references to each LoRA layer's A and B matrices
/// * `seed` — QuaRot seed (from the `.q4` artifact metadata)
/// * `hidden_dim` — hidden dimension (must be power of two; determines R's size)
/// * `plan` — rotation plan identifying absorption side per module
///
/// # Errors
///
/// Returns an error if:
/// - `hidden_dim` is not a valid Hadamard dimension
/// - Any matrix has inconsistent dimensions with the rotation
/// - Any module is not in the plan (fail-closed: unknown targets are rejected)
pub fn rotate_adapter_for_quarot(
    layers: Vec<LoraLayerMut<'_>>,
    seed: u64,
    hidden_dim: usize,
    plan: &RotationPlan,
) -> Result<RotationReport, InferenceError> {
    let rotation = RandomizedHadamard::new(seed, hidden_dim)?;
    let mut report = RotationReport {
        entries: Vec::with_capacity(layers.len()),
    };

    for layer in layers {
        match plan.absorption_for_module(layer.module) {
            Some(AbsorptionSide::InputSide) => {
                if layer.d_in != hidden_dim {
                    return Err(InferenceError::Inference(format!(
                        "rotate_adapter_for_quarot: layer {} module '{}' has d_in={} \
                         but hidden_dim={hidden_dim}",
                        layer.layer_idx, layer.module, layer.d_in
                    )));
                }
                counter_rotate_a(layer.a, layer.rank, layer.d_in, &rotation)?;
                report.entries.push((
                    layer.layer_idx,
                    layer.module.to_string(),
                    LoraRotationApplied::CounterRotatedA,
                ));
            }
            Some(AbsorptionSide::OutputSide) => {
                if layer.d_out != hidden_dim {
                    return Err(InferenceError::Inference(format!(
                        "rotate_adapter_for_quarot: layer {} module '{}' has d_out={} \
                         but hidden_dim={hidden_dim}",
                        layer.layer_idx, layer.module, layer.d_out
                    )));
                }
                rotate_b_output_side(layer.b, layer.d_out, layer.rank, &rotation)?;
                report.entries.push((
                    layer.layer_idx,
                    layer.module.to_string(),
                    LoraRotationApplied::RotatedB,
                ));
            }
            None => {
                return Err(InferenceError::Inference(format!(
                    "rotate_adapter_for_quarot: layer {} targets module '{}' which is \
                     not in the rotation plan. On a QuaRot base, unknown adapter targets \
                     risk silent basis-composition failures. Check module name spelling \
                     or update the plan.",
                    layer.layer_idx, layer.module
                )));
            }
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests;
