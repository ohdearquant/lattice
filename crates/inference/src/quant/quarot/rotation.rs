//! Absorb a [`RandomizedHadamard`] rotation into a linear-layer weight matrix
//! offline, so the runtime forward pass is unchanged after the input or output
//! activation gets rotated.
//!
//! See [ADR-044](../../../../../docs/adr/ADR-044-quarot-rotated-quantization.md) §Decision
//! for the absorption identity.
//!
//! ## Bias semantics — v0 scope
//!
//! This module rotates **weight matrices only**. It does not touch biases.
//! v0 targets bias-free linear projections (Qwen3 / Qwen3.5 attention QKV
//! and output projections, MLP up/gate/down — all bias-free in the
//! Llama-style architectures lattice currently supports). If a future
//! caller wires a model with biased projections, output-side absorption
//! is incomplete on its own: the output bias `b` would also need
//! `b ← R · b` so the post-bias activation matches `R · y`. Input-side
//! absorption is unaffected by biases (the bias adds AFTER the matmul).
//! A bias helper is a one-line follow-up when needed; left out of v0 to
//! keep the API surface minimal.
//!
//! ## Identity
//!
//! A linear layer (bias-free) computes `y = W · x`. If we rotate the
//! input by `R` (so the upstream activation becomes `R · x`), we want
//! the layer to produce the same output. Rewrite:
//!
//! ```text
//!   y = W · x = W · R^T · (R · x) = (W · R^T) · x'
//! ```
//!
//! where `x' = R · x` is the rotated input. So `W' := W · R^T` is the absorbed
//! weight: the layer now operates on the rotated input directly. We call this
//! **input-side absorption**.
//!
//! Similarly, if we want the output to be pre-rotated by `R` (so downstream
//! consumers see `R · y`), we set `W' := R · W`. We call this **output-side
//! absorption**.
//!
//! ## Implementation
//!
//! For `W` in row-major `[rows × cols]`, the row `W[i, :]` is the linear
//! functional that produces output coordinate `i`. Applying `R` to that row
//! vector computes `R · W[i, :]^T`, which equals the i-th row of `W · R^T`.
//! So **input-side absorption applies `R` to each row** of `W` in place.
//!
//! Output-side absorption is symmetric: each column `W[:, j]` becomes
//! `R · W[:, j]`, so we extract each column, apply `R`, and write back.
//!
//! Both routines support f32 (storage tier) and f64 (precision tier per
//! ADR-044 §Risks).

use crate::error::InferenceError;
use crate::quant::quarot::hadamard::RandomizedHadamard;

fn check_shape(
    fn_name: &str,
    weight_len: usize,
    rows: usize,
    cols: usize,
    rotation_dim: usize,
    rotation_dim_should_match: usize,
    matching_label: &str,
) -> Result<(), InferenceError> {
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        InferenceError::Inference(format!(
            "{fn_name}: rows*cols overflow (rows={rows}, cols={cols})"
        ))
    })?;
    if weight_len != expected {
        return Err(InferenceError::Inference(format!(
            "{fn_name}: weight length {weight_len} != rows*cols {expected}"
        )));
    }
    if rotation_dim != rotation_dim_should_match {
        return Err(InferenceError::Inference(format!(
            "{fn_name}: rotation.dim()={rotation_dim} but weight.{matching_label}={rotation_dim_should_match}"
        )));
    }
    Ok(())
}

/// Absorb `R` into the input side of a row-major `[rows × cols]` weight matrix
/// in place: `W ← W · R^T`. After this, a layer computing `y = W · x` produces
/// the same output when fed the pre-rotated input `x' = R · x`.
///
/// The rotation dimension must match `cols` (the layer's input dimension).
pub fn absorb_input_rotation(
    weight: &mut [f32],
    rows: usize,
    cols: usize,
    rotation: &RandomizedHadamard,
) -> Result<(), InferenceError> {
    check_shape(
        "absorb_input_rotation",
        weight.len(),
        rows,
        cols,
        rotation.dim(),
        cols,
        "cols",
    )?;
    for r in 0..rows {
        let row = &mut weight[r * cols..(r + 1) * cols];
        rotation.apply(row)?;
    }
    Ok(())
}

/// Absorb `R` into the output side of a row-major `[rows × cols]` weight matrix
/// in place: `W ← R · W`. After this, a layer computing `y = W · x` produces
/// the rotated output `R · y`.
///
/// The rotation dimension must match `rows` (the layer's output dimension).
pub fn absorb_output_rotation(
    weight: &mut [f32],
    rows: usize,
    cols: usize,
    rotation: &RandomizedHadamard,
) -> Result<(), InferenceError> {
    check_shape(
        "absorb_output_rotation",
        weight.len(),
        rows,
        cols,
        rotation.dim(),
        rows,
        "rows",
    )?;
    let mut column = vec![0.0_f32; rows];
    for c in 0..cols {
        for r in 0..rows {
            column[r] = weight[r * cols + c];
        }
        rotation.apply(&mut column)?;
        for r in 0..rows {
            weight[r * cols + c] = column[r];
        }
    }
    Ok(())
}

/// `f64` variant of [`absorb_input_rotation`].
///
/// Use during offline conversion when the f32 quantization budget cannot
/// afford the rotation's accumulated rounding error — see ADR-044 §Risks.
pub fn absorb_input_rotation_f64(
    weight: &mut [f64],
    rows: usize,
    cols: usize,
    rotation: &RandomizedHadamard,
) -> Result<(), InferenceError> {
    check_shape(
        "absorb_input_rotation_f64",
        weight.len(),
        rows,
        cols,
        rotation.dim(),
        cols,
        "cols",
    )?;
    for r in 0..rows {
        let row = &mut weight[r * cols..(r + 1) * cols];
        rotation.apply_f64(row)?;
    }
    Ok(())
}

/// `f64` variant of [`absorb_output_rotation`].
pub fn absorb_output_rotation_f64(
    weight: &mut [f64],
    rows: usize,
    cols: usize,
    rotation: &RandomizedHadamard,
) -> Result<(), InferenceError> {
    check_shape(
        "absorb_output_rotation_f64",
        weight.len(),
        rows,
        cols,
        rotation.dim(),
        rows,
        "rows",
    )?;
    let mut column = vec![0.0_f64; rows];
    for c in 0..cols {
        for r in 0..rows {
            column[r] = weight[r * cols + c];
        }
        rotation.apply_f64(&mut column)?;
        for r in 0..rows {
            weight[r * cols + c] = column[r];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
