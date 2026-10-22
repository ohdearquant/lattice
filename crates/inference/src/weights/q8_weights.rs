//! Per-row symmetric INT8 (Q8) weight quantization for Qwen3.5-2B.
//!
//! The model stores linear weights in row-major `[out_features, in_features]` order and
//! evaluates them through `matmul_bt`, which computes `C = A @ B^T`.
//! That layout makes **per-row / per-output-channel** scaling the natural choice:
//!
//! ```text
//! output[j] = scale[j] * sum_k(input[k] * q[j, k])
//! ```
//!
//! This module provides:
//! - a compact `Q8Matrix` representation,
//! - per-row symmetric quantization,
//! - a quantized `matmul_bt_q8` kernel,
//! - quantized mirrors of the major Qwen3.5-2B weight structures,
//! - conversion helpers, and
//! - memory accounting utilities.

#![forbid(unsafe_code)]

use crate::error::InferenceError;
use crate::weights::ingress::{IngestedTensor, validate_ingested_tensor};

pub use crate::forward::cpu::q8_matmul::matmul_bt_q8;
pub(crate) use crate::forward::cpu::q8_matmul::{Q8MatmulScratch, matmul_bt_q8_with_scratch};
pub use crate::model::qwen35::q8_conversion::{
    Q8AttentionWeights, Q8CommonLayerWeights, Q8FullAttentionLayerWeights,
    Q8GatedDeltaNetWeights, Q8ModelWeights, memory_report, quantize_gdn_weights,
};
pub(crate) use crate::model::qwen35::q8_conversion::{
    quantize_model_weights, validate_cfg_len, validate_gdn_shapes,
};

const Q8_MATRIX_SOURCE: &str = "in-memory Q8 quantization";

/// **Unstable**: per-row symmetric INT8 weight matrix; storage layout and field names may change.
///
/// A quantized weight matrix backed by `i8` values plus one scale per row.
#[derive(Debug, Clone)]
pub struct Q8Matrix {
    /// Quantized weight values in row-major `[rows, cols]` order.
    pub data: Vec<i8>,
    /// Per-row symmetric scale factors, one per output channel.
    pub scales: Vec<f32>,
    /// Number of rows (`out_features`).
    pub rows: usize,
    /// Number of columns (`in_features`).
    pub cols: usize,
}

/// **Unstable**: quantize a row-major f32 matrix into Q8; quantization scheme may change.
///
/// Quantize a single row-major f32 matrix into per-row symmetric Q8.
///
/// For each row `i`:
/// - `scale[i] = max(abs(row)) / 127.0`
/// - if the row is all zeros, `scale[i] = 1.0`
/// - `q[i, j] = clamp(round(w[i, j] / scale[i]), -128, 127)`
///
/// Returns `Err(InvalidInput)` if the declared geometry overflows or does not
/// match the source length, any source value is `NaN` or `±inf`, or a derived
/// scale is non-positive or non-finite. Validation is performed through the
/// shared weight-ingress contract so failures identify the tensor and first
/// offending element or row.
pub fn quantize_matrix(w: &[f32], rows: usize, cols: usize) -> Result<Q8Matrix, InferenceError> {
    quantize_named_matrix(w, rows, cols, Q8_MATRIX_SOURCE, "matrix")
}

pub(crate) fn validate_source_geometry(
    len: usize,
    rows: usize,
    cols: usize,
    source: &str,
    tensor_name: &str,
) -> Result<(), InferenceError> {
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "{source}: tensor {tensor_name} shape [{rows}, {cols}] overflows usize element count"
        ))
    })?;
    if len != expected {
        return Err(InferenceError::InvalidInput(format!(
            "{source}: tensor {tensor_name} (Q8) source element count {len} does not match shape \
             [{rows}, {cols}] (expected {expected})"
        )));
    }
    Ok(())
}

pub(crate) fn quantize_named_matrix(
    w: &[f32],
    rows: usize,
    cols: usize,
    source: &str,
    tensor_name: &str,
) -> Result<Q8Matrix, InferenceError> {
    validate_source_geometry(w.len(), rows, cols, source, tensor_name)?;

    let shape = [rows, cols];

    let mut data = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(rows);
    let mut has_nonfinite = false;

    // Single pass over the source matrix: finite-value validation is fused with the
    // max-abs scale computation (both already touch every element in row-major
    // order), instead of an earlier full pre-pass plus this loop re-reading the
    // same bytes. max(|x_i|) is order-independent and involves no FP accumulation,
    // so fusing cannot change the computed scale. A non-finite element still lets
    // `abs_v > max_abs` run (NaN comparisons are false, +-inf propagates), but the
    // resulting `max_abs`/`scale`/`data` are discarded below once `has_nonfinite`
    // triggers the exact pre-fusion diagnostic via a full re-scan (error path only).
    for row_idx in 0..rows {
        let start = row_idx * cols;
        let end = start + cols;
        let row = &w[start..end];

        let mut max_abs = 0.0f32;
        for &v in row {
            has_nonfinite |= !v.is_finite();
            let abs_v = v.abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
        }

        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales.push(scale);

        for &v in row {
            let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
            data.push(q);
        }
    }

    if has_nonfinite {
        // Re-scan (once, on the malformed-input path only) to reproduce the
        // exact shared-ingress diagnostic (offending element index and value).
        validate_ingested_tensor(IngestedTensor::q8_source(source, tensor_name, &shape, w))?;
        unreachable!("has_nonfinite implies validate_ingested_tensor rejects tensor {tensor_name}");
    }

    validate_ingested_tensor(IngestedTensor::q8(
        source,
        tensor_name,
        &shape,
        &data,
        &scales,
    ))?;

    Ok(Q8Matrix {
        data,
        scales,
        rows,
        cols,
    })
}

#[cfg(test)]
fn dequantize_matrix(q: &Q8Matrix) -> Vec<f32> {
    let mut out = vec![0.0f32; q.rows * q.cols];
    for row in 0..q.rows {
        let scale = q.scales[row];
        let start = row * q.cols;
        let end = start + q.cols;
        for idx in start..end {
            out[idx] = q.data[idx] as f32 * scale;
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod tests;
