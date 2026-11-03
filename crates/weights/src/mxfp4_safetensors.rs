//! Loader for paired MXFP4 block and scale tensors in a safetensors file.
//!
//! It selects an explicitly named `<stem>_blocks` / `<stem>_scales` pair in the
//! GPT-OSS Hugging Face layout and decodes it with `crate::mxfp4`. It does not
//! infer quantization from metadata.

use crate::SafetensorsFile;
use crate::mxfp4::{MXFP4_BLOCK_BYTES, MXFP4_BLOCK_LEN, dequantize_mxfp4};
use lattice_inference_core::InferenceError;

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
    let invalid_shape = |reason: &str| {
        invalid(&format!(
            "MXFP4 tensor {stem:?}: invalid blocks shape {block_shape:?} and scales shape {scale_shape:?}: {reason}"
        ))
    };
    if block_shape.len() < 2
        || block_shape.last() != Some(&MXFP4_BLOCK_BYTES)
        || &block_shape[..block_shape.len() - 1] != scale_shape
    {
        return Err(invalid_shape(
            "expected [..., groups, 16] and [..., groups]",
        ));
    }
    let groups = *scale_shape
        .last()
        .ok_or_else(|| invalid_shape("scales need a groups dimension"))?;
    let cols = groups
        .checked_mul(MXFP4_BLOCK_LEN)
        .ok_or_else(|| invalid_shape("column count overflow"))?;
    let rows = scale_shape[..scale_shape.len() - 1]
        .iter()
        .try_fold(1usize, |rows, &dim| {
            rows.checked_mul(dim)
                .ok_or_else(|| invalid_shape("row count overflow"))
        })?;
    dequantize_mxfp4(rows, cols, blocks, scales, out)
}

#[cfg(test)]
mod tests;
