//! CPU decoding of unpadded row-major NVFP4 1D blocks.
//!
//! Each group of 16 E2M1 values shares one UE4M3 scale, and the caller supplies
//! the tensor's global f32 dequantization multiplier. The low nibble is the
//! even-indexed value, as in NVIDIA ModelOpt packing. This is not a decoder for
//! swizzled scale tensors, 2D weight scaling, or checkpoint metadata.

use crate::mxfp4::e2m1;
use lattice_inference_core::InferenceError;

/// Number of values sharing a scale in this 1D layout.
pub const NVFP4_BLOCK_LEN: usize = 16;
/// Number of packed E2M1 bytes in one 1D block.
pub const NVFP4_BLOCK_BYTES: usize = NVFP4_BLOCK_LEN / 2;

fn validate_global(scale: f32) -> Result<(), InferenceError> {
    if !scale.is_finite() || scale.is_sign_negative() {
        return Err(InferenceError::InvalidInput(
            "NVFP4 global multiplier must be finite and nonnegative".into(),
        ));
    }
    Ok(())
}

fn validate_block_scale(scale: u8) -> Result<(), InferenceError> {
    if scale & 0x80 != 0 {
        return Err(InferenceError::InvalidInput(
            "NVFP4 UE4M3 scale must have a zero padding bit".into(),
        ));
    }
    Ok(())
}

fn ue4m3(scale: u8) -> f32 {
    let exponent = u32::from(scale >> 3);
    let fraction = u32::from(scale & 7);
    if exponent == 0 {
        (fraction as f32) * f32::from_bits(0x3b00_0000)
    } else if scale == 0x7f {
        f32::NAN
    } else {
        f32::from_bits(((exponent + 120) << 23) | (fraction << 20))
    }
}

fn decode_into(packed: &[u8], scale: u8, global: f32, out: &mut [f32]) {
    let block_scale = ue4m3(scale);
    for (index, &byte) in packed.iter().enumerate() {
        out[2 * index] = (e2m1(byte & 0x0f) * block_scale) * global;
        out[2 * index + 1] = (e2m1(byte >> 4) * block_scale) * global;
    }
}

/// Decode one 1D NVFP4 block without allocating.
///
/// `global` is the dequantization multiplier, not its reciprocal. The UE4M3
/// encoding `0x7f` is NaN, making the whole block NaN. Finite overflow follows
/// f32 multiplication. A zero global multiplier is accepted; negative zero is
/// refused along with other negative values.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] for a non-finite or negative global
/// multiplier or a scale whose padding bit is set; `out` remains untouched.
pub fn decode_nvfp4_block(
    packed: &[u8; NVFP4_BLOCK_BYTES],
    scale: u8,
    global: f32,
    out: &mut [f32; NVFP4_BLOCK_LEN],
) -> Result<(), InferenceError> {
    validate_global(global)?;
    validate_block_scale(scale)?;
    decode_into(packed, scale, global, out);
    Ok(())
}

/// Decode a row-major matrix of complete 1D NVFP4 blocks into f32.
///
/// `packed` is `[rows, cols / 2]`, `scales` is `[rows, cols / 16]`, and
/// `out` is `[rows, cols]`; each scale is an unswizzled UE4M3 byte. Empty
/// dimensions require empty buffers. The global multiplier and all block scales
/// are validated before output is written. Success allocates no memory.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] for an invalid global multiplier,
/// nonzero scale padding bits, columns not divisible by 16, overflowing shape,
/// or lengths that do not exactly match the matrix shape.
pub fn dequantize_nvfp4(
    rows: usize,
    cols: usize,
    packed: &[u8],
    scales: &[u8],
    global: f32,
    out: &mut [f32],
) -> Result<(), InferenceError> {
    let invalid = |message: &str| InferenceError::InvalidInput(message.into());
    validate_global(global)?;
    if !cols.is_multiple_of(NVFP4_BLOCK_LEN) {
        return Err(invalid("NVFP4 columns must be a multiple of 16"));
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("NVFP4 matrix shape overflow"))?;
    let blocks = rows
        .checked_mul(cols / NVFP4_BLOCK_LEN)
        .ok_or_else(|| invalid("NVFP4 block count overflow"))?;
    if packed.len() != elements / 2 || scales.len() != blocks || out.len() != elements {
        return Err(invalid("NVFP4 buffer lengths do not match matrix shape"));
    }
    for &scale in scales {
        validate_block_scale(scale)?;
    }
    for ((input, &scale), output) in packed
        .chunks_exact(NVFP4_BLOCK_BYTES)
        .zip(scales)
        .zip(out.chunks_exact_mut(NVFP4_BLOCK_LEN))
    {
        decode_into(input, scale, global, output);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
