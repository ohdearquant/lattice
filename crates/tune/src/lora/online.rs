//! One-event SGD refitting for a LoRA layer.
//!
//! [`adapt_step`] matches inference arithmetic, then updates A and B from the
//! squared-error residual for one `(input, target_delta)` observation.
//!
//! See ADR-057 for the decision record.
//! See `docs/lora-core.md` for the objective, gradients, and diagnostics.

use super::LoraAdapter;
use crate::error::TuneError;

/// Result of a single online adaptation step.
#[derive(Debug)]
pub struct AdaptStepResult {
    /// Sum of squared residuals `||delta - target_delta||²`.
    pub loss: f32,
    /// Euclidean norm of the concatenated dB and dA gradient vectors.
    pub grad_norm: f32,
}

/// Perform one SGD step on a single LoRA layer.
///
/// Computes `delta = scale * B @ (A @ input)`, measures the squared-error
/// residual against `target_delta`, back-propagates gradients through B and A,
/// then applies an in-place SGD update.
///
/// # Arguments
///
/// * `adapter` - Mutable adapter whose weights are updated in place.
/// * `layer_idx` - Transformer layer index (0-based).
/// * `module` - Module name, e.g. `"q_proj"`.
/// * `input` - Input activation; length must equal `d_in`.
/// * `target_delta` - Desired LoRA output correction; length must equal `d_out`.
/// * `learning_rate` - SGD step size.
///
/// # Errors
///
/// Returns [`TuneError::Training`] if no layer matches `(layer_idx, module)`.
/// Returns [`TuneError::DimensionMismatch`] if `input` or `target_delta` have wrong lengths.
pub fn adapt_step(
    adapter: &mut LoraAdapter,
    layer_idx: usize,
    module: &str,
    input: &[f32],
    target_delta: &[f32],
    learning_rate: f32,
) -> Result<AdaptStepResult, TuneError> {
    // Extract scale before any mutable borrow of adapter layers.
    let scale = adapter.config().scale();

    let lora = adapter
        .layers_mut()
        .get_mut(&(layer_idx, module.to_string()))
        .ok_or_else(|| TuneError::Training(format!("no LoRA layer for ({layer_idx}, {module})")))?;

    let rank = lora.rank;
    let d_in = lora.d_in;
    let d_out = lora.d_out;

    if input.len() != d_in {
        return Err(TuneError::DimensionMismatch {
            expected: d_in,
            actual: input.len(),
        });
    }
    if target_delta.len() != d_out {
        return Err(TuneError::DimensionMismatch {
            expected: d_out,
            actual: target_delta.len(),
        });
    }

    // Forward: intermediate = A @ input,  shape (rank,)
    // A is row-major (rank, d_in): intermediate[r] = sum_j A[r*d_in + j] * input[j]
    let mut intermediate = vec![0.0f32; rank];
    for r in 0..rank {
        let row = &lora.a[r * d_in..(r + 1) * d_in];
        intermediate[r] = row.iter().zip(input.iter()).map(|(a, x)| a * x).sum();
    }

    // Forward: delta = scale * B @ intermediate,  shape (d_out,)
    // B is row-major (d_out, rank): delta[i] = scale * sum_r B[i*rank + r] * intermediate[r]
    let mut delta = vec![0.0f32; d_out];
    for i in 0..d_out {
        let row = &lora.b[i * rank..(i + 1) * rank];
        let acc: f32 = row
            .iter()
            .zip(intermediate.iter())
            .map(|(b, h)| b * h)
            .sum();
        delta[i] = scale * acc;
    }

    // Residual and loss
    let residual: Vec<f32> = delta
        .iter()
        .zip(target_delta.iter())
        .map(|(d, t)| d - t)
        .collect();
    let loss: f32 = residual.iter().map(|r| r * r).sum();

    // dL/dB[i*rank + r] = 2 * scale * residual[i] * intermediate[r]
    let mut db = vec![0.0f32; d_out * rank];
    for i in 0..d_out {
        for r in 0..rank {
            db[i * rank + r] = 2.0 * scale * residual[i] * intermediate[r];
        }
    }

    // bt_residual[r] = sum_i B[i*rank + r] * residual[i]  (B^T @ residual)
    let mut bt_residual = vec![0.0f32; rank];
    for r in 0..rank {
        bt_residual[r] = (0..d_out).map(|i| lora.b[i * rank + r] * residual[i]).sum();
    }

    // dL/dA[r*d_in + j] = 2 * scale * bt_residual[r] * input[j]
    let mut da = vec![0.0f32; rank * d_in];
    for r in 0..rank {
        for j in 0..d_in {
            da[r * d_in + j] = 2.0 * scale * bt_residual[r] * input[j];
        }
    }

    let grad_norm: f32 = {
        let sq: f32 = db.iter().chain(da.iter()).map(|x| x * x).sum();
        sq.sqrt()
    };

    // SGD in-place update
    for (b_val, db_val) in lora.b.iter_mut().zip(db.iter()) {
        *b_val -= learning_rate * db_val;
    }
    for (a_val, da_val) in lora.a.iter_mut().zip(da.iter()) {
        *a_val -= learning_rate * da_val;
    }

    Ok(AdaptStepResult { loss, grad_norm })
}

#[cfg(test)]
mod tests;
