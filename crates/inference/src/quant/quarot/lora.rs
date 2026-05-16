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
    pub layer_idx: usize,
    pub module: &'a str,
    pub a: &'a mut [f32],
    pub b: &'a mut [f32],
    pub rank: usize,
    pub d_in: usize,
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
mod tests {
    use super::*;

    fn synthetic_vector(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 11) as u32;
                (bits as f32 / u32::MAX as f32) - 0.5
            })
            .collect()
    }

    fn matvec(mat: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), cols);
        (0..rows)
            .map(|r| {
                mat[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    // --- Input-side tests (A counter-rotation) ---

    #[test]
    fn input_side_counter_rotation_correctness() {
        // B · (A · R^T) · (R · h) == B · A · h
        let d_in = 64;
        let d_out = 32;
        let rank = 8;
        let seed = 0xC0FFEE_u64;

        let r = RandomizedHadamard::new(seed, d_in).unwrap();
        let a = synthetic_vector(rank * d_in, 1);
        let b = synthetic_vector(d_out * rank, 2);
        let h = synthetic_vector(d_in, 3);

        let a_h = matvec(&a, rank, d_in, &h);
        let original = matvec(&b, d_out, rank, &a_h);

        let mut a_cr = a.clone();
        counter_rotate_a(&mut a_cr, rank, d_in, &r).unwrap();
        let mut h_rot = h.clone();
        r.apply(&mut h_rot).unwrap();
        let a_cr_rh = matvec(&a_cr, rank, d_in, &h_rot);
        let counter_rotated = matvec(&b, d_out, rank, &a_cr_rh);

        let delta = max_abs_diff(&original, &counter_rotated);
        assert!(
            delta < 1e-4,
            "B·A_cr·(R·h) should equal B·A·h: max_abs_diff={delta}"
        );
    }

    #[test]
    fn input_side_naive_is_wrong() {
        let d_in = 64;
        let d_out = 16;
        let rank = 4;
        let seed = 42_u64;

        let r = RandomizedHadamard::new(seed, d_in).unwrap();
        let a = synthetic_vector(rank * d_in, 10);
        let b = synthetic_vector(d_out * rank, 11);
        let h = synthetic_vector(d_in, 12);

        let a_h = matvec(&a, rank, d_in, &h);
        let original = matvec(&b, d_out, rank, &a_h);

        let mut h_rot = h.clone();
        r.apply(&mut h_rot).unwrap();
        let a_rh = matvec(&a, rank, d_in, &h_rot);
        let naive = matvec(&b, d_out, rank, &a_rh);

        let delta = max_abs_diff(&original, &naive);
        assert!(
            delta > 0.1,
            "without counter-rotation the error should be large: max_abs_diff={delta}"
        );
    }

    // --- Output-side tests (B rotation) ---

    #[test]
    fn output_side_b_rotation_correctness() {
        // For output-side: base does W' = R·W, output is R·(W·x).
        // LoRA delta must also be rotated: (R·B)·A·x = R·(B·A·x) ✓
        let d_out = 64; // hidden_dim (output goes to residual stream)
        let rank = 8;
        let d_in_internal = 32; // internal dimension (not hidden_dim)
        let seed = 0xBEEF_u64;

        let r = RandomizedHadamard::new(seed, d_out).unwrap();
        let a = synthetic_vector(rank * d_in_internal, 50);
        let b = synthetic_vector(d_out * rank, 51);
        let x = synthetic_vector(d_in_internal, 52);

        // Original unrotated delta: B · A · x
        let a_x = matvec(&a, rank, d_in_internal, &x);
        let original_delta = matvec(&b, d_out, rank, &a_x);

        // Expected: R · (B · A · x)
        let mut expected = original_delta.clone();
        r.apply(&mut expected).unwrap();

        // With B_rot = R · B: B_rot · A · x should equal R · (B · A · x)
        let mut b_rot = b.clone();
        rotate_b_output_side(&mut b_rot, d_out, rank, &r).unwrap();
        let b_rot_a_x = matvec(&b_rot, d_out, rank, &a_x);

        let delta = max_abs_diff(&expected, &b_rot_a_x);
        assert!(
            delta < 1e-4,
            "(R·B)·A·x should equal R·(B·A·x): max_abs_diff={delta}"
        );
    }

    #[test]
    fn output_side_naive_skip_is_wrong() {
        // Without B rotation, the delta is in the wrong basis
        let d_out = 64;
        let rank = 4;
        let d_in_internal = 16;
        let seed = 123_u64;

        let r = RandomizedHadamard::new(seed, d_out).unwrap();
        let a = synthetic_vector(rank * d_in_internal, 60);
        let b = synthetic_vector(d_out * rank, 61);
        let x = synthetic_vector(d_in_internal, 62);

        // Unrotated delta
        let a_x = matvec(&a, rank, d_in_internal, &x);
        let unrotated_delta = matvec(&b, d_out, rank, &a_x);

        // Expected rotated delta
        let mut expected = unrotated_delta.clone();
        r.apply(&mut expected).unwrap();

        // The unrotated delta does NOT match the expected rotated delta
        let delta = max_abs_diff(&unrotated_delta, &expected);
        assert!(
            delta > 0.1,
            "skipping B rotation should produce large error: max_abs_diff={delta}"
        );
    }

    // --- Module lookup tests ---

    #[test]
    fn needs_counter_rotation_input_side_modules() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let input_side = [
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
        ];
        for module in input_side {
            assert!(
                needs_counter_rotation(&plan, module),
                "{module} should need A counter-rotation (input-side)"
            );
            assert!(
                !needs_b_rotation(&plan, module),
                "{module} should NOT need B rotation"
            );
        }
    }

    #[test]
    fn needs_b_rotation_output_side_modules() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let output_side = ["o_proj", "down_proj", "out_proj"];
        for module in output_side {
            assert!(
                needs_b_rotation(&plan, module),
                "{module} should need B rotation (output-side)"
            );
            assert!(
                !needs_counter_rotation(&plan, module),
                "{module} should NOT need A counter-rotation"
            );
        }
    }

    #[test]
    fn unknown_module_needs_neither() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        assert!(!needs_counter_rotation(&plan, "some_random_proj"));
        assert!(!needs_b_rotation(&plan, "some_random_proj"));
    }

    // --- Batch API tests ---

    #[test]
    fn rotate_adapter_batch_applies_both_sides() {
        let hidden_dim = 64;
        let rank = 4;
        let seed = 7_u64;
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();

        let mut a_q = synthetic_vector(rank * hidden_dim, 100);
        let mut b_q = synthetic_vector(hidden_dim * rank, 101); // d_out != hidden for q_proj (2*q_dim), but test uses hidden
        let mut a_o = synthetic_vector(rank * hidden_dim, 102);
        let mut b_o = synthetic_vector(hidden_dim * rank, 103);

        let a_q_orig = a_q.clone();
        let b_q_orig = b_q.clone();
        let a_o_orig = a_o.clone();
        let b_o_orig = b_o.clone();

        let layers = vec![
            LoraLayerMut {
                layer_idx: 0,
                module: "q_proj",
                a: a_q.as_mut_slice(),
                b: b_q.as_mut_slice(),
                rank,
                d_in: hidden_dim,
                d_out: hidden_dim,
            },
            LoraLayerMut {
                layer_idx: 0,
                module: "o_proj",
                a: a_o.as_mut_slice(),
                b: b_o.as_mut_slice(),
                rank,
                d_in: hidden_dim,
                d_out: hidden_dim,
            },
        ];

        let report = rotate_adapter_for_quarot(layers, seed, hidden_dim, &plan).unwrap();

        assert_eq!(report.num_a_rotated(), 1);
        assert_eq!(report.num_b_rotated(), 1);

        // q_proj: A was modified, B was NOT
        assert_ne!(a_q, a_q_orig);
        assert_eq!(b_q, b_q_orig);
        // o_proj: B was modified, A was NOT
        assert_eq!(a_o, a_o_orig);
        assert_ne!(b_o, b_o_orig);
    }

    #[test]
    fn rotate_adapter_unknown_module_rejected() {
        let hidden_dim = 64;
        let rank = 4;
        let seed = 7_u64;
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();

        let mut a = synthetic_vector(rank * hidden_dim, 200);
        let mut b = synthetic_vector(hidden_dim * rank, 201);

        let layers = vec![LoraLayerMut {
            layer_idx: 0,
            module: "conv1d",
            a: a.as_mut_slice(),
            b: b.as_mut_slice(),
            rank,
            d_in: hidden_dim,
            d_out: hidden_dim,
        }];

        let err = rotate_adapter_for_quarot(layers, seed, hidden_dim, &plan).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("conv1d") && msg.contains("not in the rotation plan"),
            "should reject unknown module: {msg}"
        );
    }

    #[test]
    fn rotate_adapter_misspelled_target_rejected() {
        let hidden_dim = 64;
        let rank = 4;
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();

        let mut a = synthetic_vector(rank * hidden_dim, 210);
        let mut b = synthetic_vector(hidden_dim * rank, 211);

        let layers = vec![LoraLayerMut {
            layer_idx: 3,
            module: "qproj", // misspelled — should be "q_proj"
            a: a.as_mut_slice(),
            b: b.as_mut_slice(),
            rank,
            d_in: hidden_dim,
            d_out: hidden_dim,
        }];

        let err = rotate_adapter_for_quarot(layers, 7, hidden_dim, &plan).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("qproj") && msg.contains("not in the rotation plan"),
            "should reject misspelled module: {msg}"
        );
    }

    #[test]
    fn rotate_adapter_input_side_dimension_mismatch_rejected() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let mut a = synthetic_vector(4 * 32, 300);
        let mut b = synthetic_vector(64 * 4, 301);

        let layers = vec![LoraLayerMut {
            layer_idx: 0,
            module: "q_proj",
            a: a.as_mut_slice(),
            b: b.as_mut_slice(),
            rank: 4,
            d_in: 32, // mismatch: hidden_dim=64 but d_in=32
            d_out: 64,
        }];

        let result = rotate_adapter_for_quarot(layers, 7, 64, &plan);
        assert!(result.is_err());
    }

    #[test]
    fn rotate_adapter_output_side_dimension_mismatch_rejected() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let mut a = synthetic_vector(4 * 64, 400);
        let mut b = synthetic_vector(32 * 4, 401);

        let layers = vec![LoraLayerMut {
            layer_idx: 0,
            module: "o_proj",
            a: a.as_mut_slice(),
            b: b.as_mut_slice(),
            rank: 4,
            d_in: 64,
            d_out: 32, // mismatch: hidden_dim=64 but d_out=32
        }];

        let result = rotate_adapter_for_quarot(layers, 7, 64, &plan);
        assert!(result.is_err());
    }

    #[test]
    fn counter_rotation_deterministic_across_calls() {
        let d_in = 64;
        let rank = 8;
        let seed = 99_u64;
        let a = synthetic_vector(rank * d_in, 500);

        let r = RandomizedHadamard::new(seed, d_in).unwrap();
        let mut a1 = a.clone();
        let mut a2 = a.clone();
        counter_rotate_a(&mut a1, rank, d_in, &r).unwrap();
        counter_rotate_a(&mut a2, rank, d_in, &r).unwrap();

        assert_eq!(a1, a2, "same seed must produce identical counter-rotation");
    }

    #[test]
    fn b_rotation_deterministic_across_calls() {
        let d_out = 64;
        let rank = 8;
        let seed = 99_u64;
        let b = synthetic_vector(d_out * rank, 600);

        let r = RandomizedHadamard::new(seed, d_out).unwrap();
        let mut b1 = b.clone();
        let mut b2 = b.clone();
        rotate_b_output_side(&mut b1, d_out, rank, &r).unwrap();
        rotate_b_output_side(&mut b2, d_out, rank, &r).unwrap();

        assert_eq!(b1, b2, "same seed must produce identical B rotation");
    }
}
