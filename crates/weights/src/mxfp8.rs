//! CPU decoding of row-major MXFP8 blocks: FP8 E4M3 elements with E8M0 scales.
//!
//! In the OCP Microscaling format, 32 FP8 E4M3 ("FN", finite-only) elements
//! share one E8M0 scale. Elements are one byte each in row-major order and the
//! scales are row-major `[rows, cols / 32]`. This module does not identify or
//! load checkpoint tensors.

use crate::half_bits::f8_e4m3_bits_to_f32;
use crate::mxfp4::e8m0;
use lattice_inference_core::InferenceError;

/// Number of elements sharing one E8M0 scale.
pub const MXFP8_BLOCK_LEN: usize = 32;

fn decode_block_into(elements: &[u8], scale: u8, out: &mut [f32]) {
    let factor = e8m0(scale);
    for (output, &element) in out.iter_mut().zip(elements) {
        *output = f8_e4m3_bits_to_f32(element) * factor;
    }
}

fn validate_shape(
    rows: usize,
    cols: usize,
    elements: &[u8],
    scales: &[u8],
) -> Result<(), InferenceError> {
    let invalid = |message: &str| InferenceError::InvalidInput(message.into());
    if !cols.is_multiple_of(MXFP8_BLOCK_LEN) {
        return Err(invalid("MXFP8 columns must be a multiple of 32"));
    }
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("MXFP8 matrix shape overflow"))?;
    let blocks = rows
        .checked_mul(cols / MXFP8_BLOCK_LEN)
        .ok_or_else(|| invalid("MXFP8 block count overflow"))?;
    if elements.len() != count || scales.len() != blocks {
        return Err(invalid("MXFP8 buffer lengths do not match matrix shape"));
    }
    Ok(())
}

/// Decode one block into caller-provided storage without allocation.
///
/// E8M0 code 0 means 2^-127, code 254 means 2^127, and code 255 makes every
/// output NaN. The E4M3 NaN encodings (`0x7f`, `0xff`) decode to NaN. Products
/// outside the f32 range follow f32 multiplication.
pub fn decode_mxfp8_block(
    elements: &[u8; MXFP8_BLOCK_LEN],
    scale: u8,
    out: &mut [f32; MXFP8_BLOCK_LEN],
) {
    decode_block_into(elements, scale, out);
}

/// Decode a row-major matrix of complete MXFP8 blocks into f32.
///
/// `elements` is `[rows, cols]`, `scales` is `[rows, cols / 32]`, and `out`
/// is `[rows, cols]`. Empty dimensions require empty buffers. Validation
/// happens before any output is written; success allocates no memory.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] when `cols` is not a multiple of
/// 32, the shape overflows, or any slice length differs from the required one.
pub fn dequantize_mxfp8(
    rows: usize,
    cols: usize,
    elements: &[u8],
    scales: &[u8],
    out: &mut [f32],
) -> Result<(), InferenceError> {
    validate_shape(rows, cols, elements, scales)?;
    if out.len() != elements.len() {
        return Err(InferenceError::InvalidInput(
            "MXFP8 buffer lengths do not match matrix shape".into(),
        ));
    }
    for ((input, &scale), output) in elements
        .chunks_exact(MXFP8_BLOCK_LEN)
        .zip(scales)
        .zip(out.chunks_exact_mut(MXFP8_BLOCK_LEN))
    {
        decode_block_into(input, scale, output);
    }
    Ok(())
}

/// Multiply a row-major MXFP8 matrix by a vector without materializing it.
///
/// Computes `out[r] = sum over c of W[r, c] * x[c]`, where `W` is the matrix
/// [`dequantize_mxfp8`] produces from the same arguments. Each row accumulates
/// in f32 in column order, so the result is bit-identical to dequantizing and
/// taking a sequential dot product. `x` has `cols` elements and `out` has
/// `rows`. All inputs are validated before `out` is written; success
/// allocates no memory.
///
/// # Errors
///
/// The errors of [`dequantize_mxfp8`], plus an `x` or `out` length that does
/// not match the shape.
pub fn matvec_mxfp8(
    rows: usize,
    cols: usize,
    elements: &[u8],
    scales: &[u8],
    x: &[f32],
    out: &mut [f32],
) -> Result<(), InferenceError> {
    validate_shape(rows, cols, elements, scales)?;
    if x.len() != cols || out.len() != rows {
        return Err(InferenceError::InvalidInput(
            "MXFP8 matvec vector lengths do not match matrix shape".into(),
        ));
    }
    let row_blocks = cols / MXFP8_BLOCK_LEN;
    let mut block = [0.0f32; MXFP8_BLOCK_LEN];
    for (row, output) in out.iter_mut().enumerate() {
        let row_elements = &elements[row * cols..(row + 1) * cols];
        let row_scales = &scales[row * row_blocks..(row + 1) * row_blocks];
        let mut acc = 0.0f32;
        for ((input, &scale), xs) in row_elements
            .chunks_exact(MXFP8_BLOCK_LEN)
            .zip(row_scales)
            .zip(x.chunks_exact(MXFP8_BLOCK_LEN))
        {
            decode_block_into(input, scale, &mut block);
            for (weight, value) in block.iter().zip(xs) {
                acc += weight * value;
            }
        }
        *output = acc;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
