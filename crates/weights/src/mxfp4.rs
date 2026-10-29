//! CPU decoding of row-major MXFP4 E2M1 blocks with E8M0 scales.
//!
//! MXFP4 uses 32 elements per scale. The packed layout here places the first
//! element in the low nibble, as in the pinned DeepSeek-V4 converter; the
//! generic MX specification does not define this matrix byte layout.
//! An explicit loader also accepts paired U8 block/scale tensors in the
//! GPT-OSS Hugging Face layout; it does not infer quantization from metadata.

use crate::SafetensorsFile;
use lattice_inference_core::InferenceError;

/// Number of values sharing one E8M0 scale.
pub const MXFP4_BLOCK_LEN: usize = 32;
/// Number of packed E2M1 bytes in one block.
pub const MXFP4_BLOCK_BYTES: usize = MXFP4_BLOCK_LEN / 2;

const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

pub(super) fn e2m1(nibble: u8) -> f32 {
    let magnitude = E2M1_MAGNITUDES[usize::from(nibble & 7)];
    if nibble & 8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn e8m0(scale: u8) -> f32 {
    match scale {
        0 => f32::from_bits(0x0040_0000),
        255 => f32::NAN,
        value => f32::from_bits(u32::from(value) << 23),
    }
}

/// Decode one block into caller-provided storage without allocation.
///
/// Each byte contains the even-indexed element in its low nibble. E8M0 code
/// 0 means 2^-127, code 254 means 2^127, and code 255 makes all outputs NaN.
/// Finite values outside the f32 range overflow according to f32 multiplication.
pub fn decode_mxfp4_block(
    packed: &[u8; MXFP4_BLOCK_BYTES],
    scale: u8,
    out: &mut [f32; MXFP4_BLOCK_LEN],
) {
    decode_block_into(packed, scale, out);
}

fn decode_block_into(packed: &[u8], scale: u8, out: &mut [f32]) {
    let factor = e8m0(scale);
    for (index, &byte) in packed.iter().enumerate() {
        out[2 * index] = e2m1(byte & 0x0f) * factor;
        out[2 * index + 1] = e2m1(byte >> 4) * factor;
    }
}

/// Decode a row-major matrix of complete MXFP4 blocks into f32.
///
/// `packed` is `[rows, cols / 2]`, `scales` is `[rows, cols / 32]`,
/// and `out` is `[rows, cols]`. The caller supplies all output storage. Empty
/// dimensions are accepted when all supplied slices have the corresponding
/// empty length. No allocation occurs on success.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] when `cols` is not a multiple of
/// 32, shape multiplication overflows, or any slice length differs from the
/// exact required length. Validation happens before any output is written.
pub fn dequantize_mxfp4(
    rows: usize,
    cols: usize,
    packed: &[u8],
    scales: &[u8],
    out: &mut [f32],
) -> Result<(), InferenceError> {
    let invalid = |message: &str| InferenceError::InvalidInput(message.into());
    if !cols.is_multiple_of(MXFP4_BLOCK_LEN) {
        return Err(invalid("MXFP4 columns must be a multiple of 32"));
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("MXFP4 matrix shape overflow"))?;
    let blocks = rows
        .checked_mul(cols / MXFP4_BLOCK_LEN)
        .ok_or_else(|| invalid("MXFP4 block count overflow"))?;
    if packed.len() != elements / 2 || scales.len() != blocks || out.len() != elements {
        return Err(invalid("MXFP4 buffer lengths do not match matrix shape"));
    }
    for ((input, &scale), output) in packed
        .chunks_exact(MXFP4_BLOCK_BYTES)
        .zip(scales)
        .zip(out.chunks_exact_mut(MXFP4_BLOCK_LEN))
    {
        decode_block_into(input, scale, output);
    }
    Ok(())
}

/// Decode an explicitly selected Hugging Face GPT-OSS MXFP4 tensor pair.
///
/// `stem` identifies `<stem>_blocks` and `<stem>_scales` in the same file.
/// Both tensors must have dtype `U8`; blocks have shape `[..., groups, 16]`
/// and scales have shape `[..., groups]`. The caller supplies an output with
/// shape `[..., groups * 32]`, flattened in checkpoint order. This does not
/// transpose the result or infer that arbitrary U8 tensors are MXFP4. Scale
/// code 255 and finite overflow retain the decoder's NaN and infinity behavior.
///
/// # Errors
///
/// Returns an error for a missing tensor, wrong dtype or shape, overflowing
/// dimensions, or an output buffer with the wrong length. All validation
/// happens before the output is modified.
pub fn dequantize_mxfp4_safetensors(
    file: &SafetensorsFile,
    stem: &str,
    out: &mut [f32],
) -> Result<(), InferenceError> {
    let invalid = |message: &str| InferenceError::InvalidSafetensors(message.into());
    if stem.is_empty() {
        return Err(invalid("MXFP4 tensor stem must not be empty"));
    }
    let block_name = format!("{stem}_blocks");
    let scale_name = format!("{stem}_scales");
    let blocks = file.u8_payload(&block_name)?;
    let scales = file.u8_payload(&scale_name)?;
    let block_shape = file
        .tensor_shape(&block_name)
        .ok_or_else(|| InferenceError::MissingTensor(block_name.clone()))?;
    let scale_shape = file
        .tensor_shape(&scale_name)
        .ok_or_else(|| InferenceError::MissingTensor(scale_name.clone()))?;
    if block_shape.len() < 2
        || block_shape.last() != Some(&MXFP4_BLOCK_BYTES)
        || &block_shape[..block_shape.len() - 1] != scale_shape
    {
        return Err(invalid(
            "MXFP4 blocks/scales shapes do not match [..., groups, 16] and [..., groups]",
        ));
    }
    let groups = *scale_shape
        .last()
        .ok_or_else(|| invalid("MXFP4 scales need a groups dimension"))?;
    let cols = groups
        .checked_mul(MXFP4_BLOCK_LEN)
        .ok_or_else(|| invalid("MXFP4 column count overflow"))?;
    let rows = scale_shape[..scale_shape.len() - 1]
        .iter()
        .try_fold(1usize, |rows, &dim| {
            rows.checked_mul(dim)
                .ok_or_else(|| invalid("MXFP4 row count overflow"))
        })?;
    dequantize_mxfp4(rows, cols, blocks, scales, out)
}

#[cfg(test)]
mod safetensors_tests;

#[cfg(test)]
mod tests;
