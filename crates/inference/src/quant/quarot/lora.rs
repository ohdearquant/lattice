//! Counter-rotation for LoRA adapters on QuaRot-converted models (ADR-045).
//!
//! When a LoRA adapter trained on an unrotated model is applied to a
//! QuaRot-converted base, the adapter's A matrices see the rotated
//! activation `R · h` instead of `h`. The fix is exact: replace A with
//! `A_cr = A · R^T` at adapter load time. Then at runtime:
//!
//! ```text
//! s · B · A_cr · (R · h) = s · B · (A · R^T) · (R · h) = s · B · A · h  ✓
//! ```
//!
//! The correction is absorbed once (microseconds) and adds zero runtime cost.

use crate::error::InferenceError;
use crate::quant::quarot::hadamard::RandomizedHadamard;
use crate::quant::quarot::plan::{AbsorptionSide, RotationPlan};
use crate::quant::quarot::rotation::absorb_input_rotation;

/// Counter-rotate a single LoRA A matrix for a QuaRot-converted base.
///
/// Computes `A_cr = A · R^T` in-place, where A is row-major `(rank × d_in)`.
///
/// Mathematically identical to input-side absorption: applying R to each row
/// of A yields `A · R^T` (see `rotation.rs` §Implementation for the proof).
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

/// Returns `true` if a LoRA module's A matrix needs counter-rotation.
///
/// A module needs counter-rotation iff the base weight was absorbed on the
/// input side (`W ← W · R^T`), because the adapter's A matrix then receives
/// the rotated activation `R · h` instead of `h`.
///
/// Output-side projections (o_proj, down_proj, out_proj) do NOT need
/// counter-rotation — their A matrices receive internal state that is
/// unaffected by the residual-stream rotation.
pub fn needs_counter_rotation(plan: &RotationPlan, module: &str) -> bool {
    matches!(
        plan.absorption_for_module(module),
        Some(AbsorptionSide::InputSide)
    )
}

/// Summary of a counter-rotation pass over an adapter's A matrices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterRotationReport {
    /// (layer_idx, module) pairs that were counter-rotated.
    pub rotated: Vec<(usize, String)>,
    /// (layer_idx, module) pairs skipped (output-side or not in plan).
    pub skipped: Vec<(usize, String)>,
}

/// Counter-rotate all A matrices in an adapter that target input-side projections.
///
/// Iterates over `layers` (each entry is `(layer_idx, module_name, a_matrix, rank, d_in)`)
/// and applies `A_cr = A · R^T` to those where the plan indicates input-side absorption.
///
/// This is the batch primitive. Callers holding a `LoraAdapter` (in `lattice-tune`)
/// should extract their layers into this format and call this function.
///
/// # Arguments
///
/// * `layers` — mutable iterator of `(layer_idx, module, &mut a, rank, d_in)`
/// * `seed` — QuaRot seed (from the `.q4` artifact's `config.json`)
/// * `hidden_dim` — hidden dimension (must be power of two; determines R's size)
/// * `plan` — rotation plan identifying which modules are input-side
///
/// # Errors
///
/// Returns an error if `hidden_dim` is not a valid Hadamard dimension or if
/// any A matrix has inconsistent dimensions.
pub fn counter_rotate_layers<'a, I>(
    layers: I,
    seed: u64,
    hidden_dim: usize,
    plan: &RotationPlan,
) -> Result<CounterRotationReport, InferenceError>
where
    I: IntoIterator<Item = (usize, &'a str, &'a mut [f32], usize, usize)>,
{
    let rotation = RandomizedHadamard::new(seed, hidden_dim)?;
    let mut report = CounterRotationReport {
        rotated: Vec::new(),
        skipped: Vec::new(),
    };

    for (layer_idx, module, a, rank, d_in) in layers {
        if needs_counter_rotation(plan, module) {
            if d_in != hidden_dim {
                return Err(InferenceError::Inference(format!(
                    "counter_rotate_layers: layer {layer_idx} module '{module}' has d_in={d_in} \
                     but hidden_dim={hidden_dim}"
                )));
            }
            counter_rotate_a(a, rank, d_in, &rotation)?;
            report.rotated.push((layer_idx, module.to_string()));
        } else {
            report.skipped.push((layer_idx, module.to_string()));
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

    #[test]
    fn counter_rotation_correctness() {
        // Validates: B · (A · R^T) · (R · h) == B · A · h
        let d_in = 64; // hidden_dim (power of two)
        let d_out = 32;
        let rank = 8;
        let seed = 0xC0FFEE_u64;

        let r = RandomizedHadamard::new(seed, d_in).unwrap();
        let a = synthetic_vector(rank * d_in, 1);
        let b = synthetic_vector(d_out * rank, 2);
        let h = synthetic_vector(d_in, 3);

        // Original: B · (A · h)
        let a_h = matvec(&a, rank, d_in, &h);
        let original = matvec(&b, d_out, rank, &a_h);

        // Counter-rotated path: B · (A_cr · (R · h))
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
    fn counter_rotation_naive_is_wrong() {
        // Without counter-rotation, B · A · (R · h) ≠ B · A · h
        let d_in = 64;
        let d_out = 16;
        let rank = 4;
        let seed = 42_u64;

        let r = RandomizedHadamard::new(seed, d_in).unwrap();
        let a = synthetic_vector(rank * d_in, 10);
        let b = synthetic_vector(d_out * rank, 11);
        let h = synthetic_vector(d_in, 12);

        // Original: B · A · h
        let a_h = matvec(&a, rank, d_in, &h);
        let original = matvec(&b, d_out, rank, &a_h);

        // Naive (no counter-rotation): B · A · (R · h)
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

    #[test]
    fn counter_rotation_is_exact_for_orthogonal_r() {
        // R^T · R = I, so the correction is exact (not approximate)
        let d_in = 128;
        let rank = 16;
        let seed = 0xDEAD_BEEF_u64;

        let r = RandomizedHadamard::new(seed, d_in).unwrap();
        let a = synthetic_vector(rank * d_in, 20);
        let h = synthetic_vector(d_in, 21);

        // A · h
        let a_h = matvec(&a, rank, d_in, &h);

        // A_cr · (R · h) = (A · R^T) · (R · h) should equal A · (R^T · R · h) = A · h
        let mut a_cr = a.clone();
        counter_rotate_a(&mut a_cr, rank, d_in, &r).unwrap();
        let mut h_rot = h.clone();
        r.apply(&mut h_rot).unwrap();
        let a_cr_rh = matvec(&a_cr, rank, d_in, &h_rot);

        let delta = max_abs_diff(&a_h, &a_cr_rh);
        assert!(
            delta < 5e-4,
            "A_cr·(R·h) should exactly equal A·h (orthogonal R): max_abs_diff={delta}"
        );
    }

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
                "{module} should need counter-rotation (input-side)"
            );
        }
    }

    #[test]
    fn needs_counter_rotation_output_side_modules() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let output_side = ["o_proj", "down_proj", "out_proj"];
        for module in output_side {
            assert!(
                !needs_counter_rotation(&plan, module),
                "{module} should NOT need counter-rotation (output-side)"
            );
        }
    }

    #[test]
    fn needs_counter_rotation_unknown_module() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        assert!(!needs_counter_rotation(&plan, "some_random_proj"));
        assert!(!needs_counter_rotation(&plan, "conv1d"));
    }

    #[test]
    fn counter_rotate_layers_batch() {
        let d_in = 64;
        let rank = 4;
        let seed = 7_u64;
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();

        let mut a_q = synthetic_vector(rank * d_in, 100);
        let mut a_o = synthetic_vector(rank * d_in, 101);
        let mut a_gate = synthetic_vector(rank * d_in, 102);
        let mut a_down = synthetic_vector(rank * d_in, 103);

        let a_q_orig = a_q.clone();
        let a_o_orig = a_o.clone();
        let a_gate_orig = a_gate.clone();
        let a_down_orig = a_down.clone();

        let layers: Vec<(usize, &str, &mut [f32], usize, usize)> = vec![
            (0, "q_proj", a_q.as_mut_slice(), rank, d_in),
            (0, "o_proj", a_o.as_mut_slice(), rank, d_in),
            (1, "gate_proj", a_gate.as_mut_slice(), rank, d_in),
            (1, "down_proj", a_down.as_mut_slice(), rank, d_in),
        ];

        let report = counter_rotate_layers(layers, seed, d_in, &plan).unwrap();

        assert_eq!(report.rotated.len(), 2);
        assert_eq!(report.skipped.len(), 2);
        assert!(report.rotated.contains(&(0, "q_proj".into())));
        assert!(report.rotated.contains(&(1, "gate_proj".into())));
        assert!(report.skipped.contains(&(0, "o_proj".into())));
        assert!(report.skipped.contains(&(1, "down_proj".into())));

        // Input-side modules were modified
        assert_ne!(a_q, a_q_orig);
        assert_ne!(a_gate, a_gate_orig);
        // Output-side modules were NOT modified
        assert_eq!(a_o, a_o_orig);
        assert_eq!(a_down, a_down_orig);
    }

    #[test]
    fn counter_rotate_layers_dimension_mismatch_rejected() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let mut a = synthetic_vector(4 * 32, 200);

        let layers: Vec<(usize, &str, &mut [f32], usize, usize)> =
            vec![(0, "q_proj", a.as_mut_slice(), 4, 32)]; // d_in=32 but hidden_dim=64

        let result = counter_rotate_layers(layers, 7, 64, &plan);
        assert!(result.is_err());
    }

    #[test]
    fn counter_rotation_deterministic_across_calls() {
        let d_in = 64;
        let rank = 8;
        let seed = 99_u64;
        let a = synthetic_vector(rank * d_in, 300);

        let r = RandomizedHadamard::new(seed, d_in).unwrap();
        let mut a1 = a.clone();
        let mut a2 = a.clone();
        counter_rotate_a(&mut a1, rank, d_in, &r).unwrap();
        counter_rotate_a(&mut a2, rank, d_in, &r).unwrap();

        assert_eq!(a1, a2, "same seed must produce identical counter-rotation");
    }

    #[test]
    fn counter_rotation_different_seeds_differ() {
        let d_in = 64;
        let rank = 8;
        let a = synthetic_vector(rank * d_in, 400);

        let r1 = RandomizedHadamard::new(1, d_in).unwrap();
        let r2 = RandomizedHadamard::new(2, d_in).unwrap();
        let mut a1 = a.clone();
        let mut a2 = a.clone();
        counter_rotate_a(&mut a1, rank, d_in, &r1).unwrap();
        counter_rotate_a(&mut a2, rank, d_in, &r2).unwrap();

        assert_ne!(a1, a2, "different seeds should produce different rotations");
    }
}
