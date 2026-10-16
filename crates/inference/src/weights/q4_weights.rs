//! Q4 per-block weight quantization for large models (e.g., Qwen3.6-27B).
//!
//! ## Format (v2 — asymmetric scale + bias, 20 bytes per block)
//!
//! Every 32 consecutive weights are packed into one [`Q4Block`] of 20 bytes:
//! - `scale: u16` — per-block scale, stored as an IEEE-754 f16 bit pattern
//! - `bias: u16`  — per-block bias (zero-point), stored as an IEEE-754 f16 bit pattern
//! - `packed: [u8; 16]` — 32 nibbles in **sequential-pairs** layout
//!
//! ### Nibble layout (sequential pairs — NOT llama.cpp split-half)
//!
//! ```text
//! byte[b] = (q[2b+1] << 4) | q[2b]     b ∈ 0..16
//! ```
//!
//! The low nibble holds `q[2b]`, the high nibble `q[2b+1]`. This matches the
//! nibble convention used by the `gemv_q4_decode` Metal kernel in
//! `forward/metal_qwen35.rs`.
//!
//! ### Dequantization (both encode modes share this)
//!
//! ```text
//! weight[2b]   = (byte[b] & 0x0F) as f32 * scale + bias
//! weight[2b+1] = (byte[b] >>  4)  as f32 * scale + bias
//! ```
//!
//! ### Encode modes (same on-disk layout)
//!
//! - **Asymmetric** (default): `scale = (max - min) / 15`, `bias = min`,
//!   `q[i] = clamp(round((weight[i] - min) / scale), 0, 15)`. Optimal for raw
//!   weights with a non-zero distributional center.
//! - **Symmetric** (Hadamard-rotated, zero-mean weights): `scale = abs_max / 7`,
//!   `bias = -8 * scale`, `q[i] = clamp(round(weight[i] / scale) + 8, 0, 15)`, so
//!   the shared dequant reduces to `(q - 8) * scale`.
//!
//! ## File format (`.q4`)
//!
//! ```text
//! magic        b"KHQ4"               4 bytes
//! version      2u32 LE               4 bytes   (v1 = legacy symmetric 18-byte blocks; rejected on load)
//! ndim         u32 LE                4 bytes
//! shape[i]     u64 LE × ndim
//! original_len u64 LE                8 bytes
//! blocks       [Q4Block; n_blocks]   n_blocks × 20 bytes
//! ```

// Q4 quantization operates on raw byte/u16 slices; unsafe is limited to
// the two transmute-equivalent slice casts in stream_quantize_shard and save/load.
#![allow(clippy::cast_possible_truncation)]

use crate::error::InferenceError;

/// One Q4_0 quantization block: 32 weights packed as 4-bit unsigned integers.
///
/// `scale` is stored as a raw IEEE-754 f16 bit pattern in a `u16` — the `half` crate
/// is not a dependency of `lattice-inference`. Use `q4_f32_to_f16` / `q4_f16_to_f32`.
///
/// `packed` holds 32 nibbles in **sequential-pairs** layout:
/// ```text
/// byte[b] = (q[2b+1] << 4) | q[2b]
/// ```
/// where `q[i] = clamp(round((weight[i] - bias) / scale), 0, 15)` for the default
/// asymmetric format (`bias` = per-block minimum). The legacy symmetric variant
/// fixes `bias = -8 * scale`, giving `q[i] = clamp(round(weight[i] / scale) + 8, 0, 15)`.
///
/// The scale may be negative. Lattice's own quantizer writes positive scales, but a
/// block repacked from another format keeps its source scale, whose sign is not fixed.
/// Ingress refuses only a zero or non-finite scale and a non-finite bias. A block's
/// smallest and largest weight are therefore `bias` and `bias + 15 * scale` in either
/// order, and a consumer that needs a range, an ordering or a magnitude (clamping, a
/// lookup table, a dynamic-range estimate) must take it from those two values, never
/// from the sign of `scale`. Every consumer is covered by a negative-scale fixture,
/// and a new consumer must pass it too.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q4Block {
    /// f16 bit pattern for the per-block scale — 2 bytes.
    pub scale: u16,
    /// f16 bit pattern for the per-block minimum (bias) — 2 bytes.
    /// Dequantization: `weight = nibble * scale + bias`.
    pub bias: u16,
    /// 32 nibbles packed as 16 bytes in sequential-pairs layout.
    pub packed: [u8; 16],
}

// Compile-time size assertion — must be exactly 20 bytes (2 + 2 + 16, no padding).
const _: () = assert!(std::mem::size_of::<Q4Block>() == 20);

/// Serialized size in bytes of one [`Q4Block`] (2 scale + 2 bias + 16 packed = 20).
///
/// Derive block-byte accounting from this constant rather than a magic literal so
/// it can never drift from the struct layout asserted above.
pub const Q4_BLOCK_BYTES: usize = std::mem::size_of::<Q4Block>();

/// Number of original weights packed into one [`Q4Block`] (32 nibbles).
///
/// Both the quantizer's block-count math and the ingress validator's
/// expected-block-count check must agree on this value — a single constant
/// keeps a future block-width change from updating one side and not the
/// other.
pub(crate) const Q4_BLOCK_WEIGHTS: usize = 32;

/// A Q4_0 quantized tensor.
///
/// Stores blocks, shape metadata, and the count of valid original weights (the last
/// block may be padded with zeros if `original_len` is not a multiple of 32).
#[derive(Debug, Clone)]
pub struct Q4Tensor {
    /// Quantized blocks, each covering 32 weights.
    pub blocks: Vec<Q4Block>,
    /// Original tensor shape (e.g., `[rows, cols]` for a 2-D weight matrix).
    pub shape: Vec<usize>,
    /// Number of valid original weights — may be less than `blocks.len() * 32`.
    pub original_len: usize,
}

// ---------------------------------------------------------------------------
// f16 ↔ f32 / bf16 → f32 helpers.
//
// Thin wrappers over the single always-compiled scalar decoder in
// `crate::weights::half_bits` (lattice#799) — kept as separate `q4_`-prefixed
// functions here only to preserve this module's existing call-site names
// and `pub(crate)` visibility; no conversion arithmetic lives in this file
// anymore.
// ---------------------------------------------------------------------------

/// Convert `f32` to IEEE-754 half-precision stored as a `u16` bit pattern.
///
/// Uses round-to-nearest-even for mantissa truncation. Handles ±0, ±∞, NaN,
/// subnormals, and overflow (→ ±∞).
#[inline]
pub(crate) fn q4_f32_to_f16(x: f32) -> u16 {
    crate::weights::half_bits::f32_to_f16_bits(x)
}

#[inline]
pub(crate) fn q4_f32_to_finite_f16(x: f32) -> Result<u16, u16> {
    crate::weights::half_bits::f32_to_finite_f16_bits(x)
}

/// Convert an IEEE-754 f16 bit pattern (`u16`) back to `f32`.
#[inline]
pub(crate) fn q4_f16_to_f32(bits: u16) -> f32 {
    crate::weights::half_bits::f16_bits_to_f32(bits)
}

/// Convert a BF16 bit pattern (`u16`) to `f32`.
///
/// BF16 has identical sign+exponent layout to f32; zero-extending the mantissa
/// is a lossless widening. Handles ±0, ±∞, NaN, and subnormals correctly.
#[inline]
fn bf16_to_f32(v: u16) -> f32 {
    crate::weights::half_bits::bf16_bits_to_f32(v)
}

/// Whether `scale` survives serialization as a finite, strictly positive f16.
///
/// A block's scale is written to disk as an f16 bit pattern, so what a reader
/// dequantizes with is `q4_f16_to_f32(q4_f32_to_f16(scale))`, not `scale`
/// itself. A positive f32 below f16's smallest subnormal (~5.96e-8) serializes
/// to exactly `+0.0`, and a value above f16's maximum (~65504) serializes to
/// infinity; either produces a block no reader can dequantize.
///
/// This is the single definition of "is this scale usable?" in this module.
/// [`q4_metadata_bits`] enforces it at serialization, and
/// [`quantize_block_with_mode_len`] tests the same condition when choosing a
/// block's scale, so a scale the quantizer emits can never be one the
/// serializer rejects for being unrepresentably small.
#[inline]
fn q4_scale_survives_f16(scale: f32) -> bool {
    let serialized = q4_f16_to_f32(q4_f32_to_f16(scale));
    serialized.is_finite() && serialized > 0.0
}

fn q4_metadata_bits(scale: f32, bias: f32) -> Result<(u16, u16), InferenceError> {
    let scale_bits = q4_f32_to_f16(scale);
    let bias_bits = q4_f32_to_f16(bias);
    let serialized_bias = q4_f16_to_f32(bias_bits);
    if !q4_scale_survives_f16(scale) {
        return Err(InferenceError::InvalidInput(format!(
            "Q4 scale {scale} is not representable as a finite, strictly positive f16 value"
        )));
    }
    if !serialized_bias.is_finite() {
        return Err(InferenceError::InvalidInput(format!(
            "Q4 bias {bias} is not representable as a finite f16 value"
        )));
    }
    Ok((scale_bits, bias_bits))
}

// ---------------------------------------------------------------------------
// Core block quantization
// ---------------------------------------------------------------------------

/// The scale a block is actually quantized and serialized with, given the
/// `candidate` its range implies (`abs_max / 7` symmetric, `range / 15`
/// asymmetric).
///
/// A block whose candidate scale cannot survive f16 serialization as a strictly
/// positive value carries no range a reader could reconstruct, so it takes the
/// degenerate fallback of `1.0`: every weight then quantizes to the same code
/// and dequantizes to the block's bias. For a block whose true range is on the
/// order of 1e-37 that leaves a reconstruction error of the same order —
/// identical to what an exactly-zero-range block has always produced, and far
/// below anything representable downstream.
///
/// Testing the candidate for f16 survivability rather than for exact equality
/// to zero is what makes this predicate agree with the one
/// [`q4_metadata_bits`] enforces. A range that is tiny but nonzero derives a
/// tiny-but-nonzero scale, which underflows to `+0.0` in f16 — degenerate in
/// exactly the sense the fallback exists for, but invisible to a `== 0.0` test.
///
/// The fallback deliberately covers only the underflow direction. A candidate
/// too *large* for f16 is a real range that cannot be represented, not an
/// absent one; substituting `1.0` there would silently mis-quantize a block, so
/// it is passed through for [`q4_metadata_bits`] to reject.
#[inline]
fn degenerate_safe_scale(candidate: f32) -> f32 {
    if candidate < 1.0 && !q4_scale_survives_f16(candidate) {
        1.0f32
    } else {
        candidate
    }
}

/// Quantize one block from only the first `valid_len` real values of `vals`;
/// the remaining `32 - valid_len` slots are caller-supplied zero padding used
/// solely to fill the fixed-size packing loop below.
///
/// Asymmetric mode derives `min_val`/`max_val` from the real `valid_len`
/// elements only, so a zero-padded tail block gets the same scale resolution
/// a full block would (padding zeros must never widen the range). Symmetric
/// mode folds `abs_max` over the full padded array unconditionally: zero
/// padding can never exceed a non-empty real element's absolute value, so the
/// result is bit-identical to folding over the real elements alone, and doing
/// it this way keeps the symmetric path source-identical to the pre-fix
/// version (see `quantize_f64_to_q4_symmetric_partial_block_is_bit_identical_to_padded_block`).
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if `valid_len` is not in `1..=32`,
/// or if any element of `vals` is non-finite. IEEE-754 `NaN > x` is always
/// false; a NaN silently leaves `abs_max`, `min_val`, or `max_val` unchanged,
/// yielding wrong-but-no-error quantization. Rejecting here means the error
/// points at the source weight rather than a downstream matmul.
#[inline]
fn quantize_block_with_mode_len(
    vals: &[f32; 32],
    valid_len: usize,
    symmetric: bool,
) -> Result<Q4Block, InferenceError> {
    if !(1..=32).contains(&valid_len) {
        return Err(InferenceError::InvalidInput(format!(
            "Q4 weight block valid_len {valid_len} must be in 1..=32"
        )));
    }

    for (i, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            return Err(InferenceError::InvalidInput(format!(
                "Q4 weight block element {i} contains a non-finite value ({v}); \
                 source weights must be finite"
            )));
        }
    }
    if symmetric {
        let abs_max = vals.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = degenerate_safe_scale(abs_max / 7.0);
        let inv_scale = 1.0 / scale;
        let bias = -8.0 * scale;
        let (scale_bits, bias_bits) = q4_metadata_bits(scale, bias)?;
        let mut packed = [0u8; 16];
        for b in 0..16 {
            let q0 = ((vals[2 * b] * inv_scale).round() + 8.0).clamp(0.0, 15.0) as u8;
            let q1 = ((vals[2 * b + 1] * inv_scale).round() + 8.0).clamp(0.0, 15.0) as u8;
            packed[b] = (q1 << 4) | (q0 & 0x0f);
        }
        Ok(Q4Block {
            scale: scale_bits,
            bias: bias_bits,
            packed,
        })
    } else {
        let real = &vals[..valid_len];
        let min_val = real.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = real.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = degenerate_safe_scale(range / 15.0);
        let (scale_bits, bias_bits) = q4_metadata_bits(scale, min_val)?;
        let inv_scale = 1.0 / scale;
        let mut packed = [0u8; 16];
        for b in 0..16 {
            let q0 = (((vals[2 * b] - min_val) * inv_scale).round()).clamp(0.0, 15.0) as u8;
            let q1 = (((vals[2 * b + 1] - min_val) * inv_scale).round()).clamp(0.0, 15.0) as u8;
            packed[b] = (q1 << 4) | (q0 & 0x0f);
        }
        Ok(Q4Block {
            scale: scale_bits,
            bias: bias_bits,
            packed,
        })
    }
}

// ---------------------------------------------------------------------------
// Public quantization API
// ---------------------------------------------------------------------------

/// Quantize a slice of f32 values into Q4_0 blocks.
///
/// The input is processed 32 elements at a time; the last block is zero-padded
/// if `src.len()` is not a multiple of 32.
///
/// Returns raw bytes containing tightly-packed [`Q4Block`]s (20 bytes each).
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if any value in `src` is non-finite
/// or the derived scale/bias cannot be represented as finite f16 metadata.
pub fn quantize_row_q4_0(src: &[f32]) -> Result<Vec<u8>, InferenceError> {
    let n_blocks = src.len().div_ceil(Q4_BLOCK_WEIGHTS);
    let mut out = Vec::with_capacity(n_blocks * 20);
    for chunk in src.chunks(32) {
        let mut vals = [0.0f32; 32];
        vals[..chunk.len()].copy_from_slice(chunk);
        let block = quantize_block_with_mode_len(&vals, chunk.len(), false)?;
        // SAFETY: Q4Block is #[repr(C)] with size 20; its alignment is 2 (the
        // alignment of the leading `scale: u16` per the Rust Reference's repr(C)
        // rule). Casting to `&[u8; 20]` is valid because the target element type
        // is `u8` (alignment 1 ≤ source alignment 2) and the source byte length
        // matches the destination length exactly.
        let bytes: &[u8; 20] = unsafe { &*std::ptr::from_ref(&block).cast() };
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

/// Dequantize Q4_0 blocks (raw bytes) back to f32 values.
///
/// Trailing bytes beyond the last complete 20-byte block are silently ignored;
/// the function returns `min(n_weights, (data.len() / 20) * 32)` values.
/// It never panics regardless of input length — inputs shorter than 20 bytes
/// return an empty `Vec`.
///
/// The caller is responsible for sizing `n_weights` appropriately:
/// if `n_weights > (data.len() / 20) * 32` the output is truncated to the
/// number of values that complete blocks can produce.
pub fn dequantize_row_q4_0(data: &[u8], n_weights: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_weights);
    for chunk in data.chunks_exact(20) {
        let scale = q4_f16_to_f32(u16::from_ne_bytes([chunk[0], chunk[1]]));
        let bias = q4_f16_to_f32(u16::from_ne_bytes([chunk[2], chunk[3]]));
        for b in 0..16 {
            let byte_val = chunk[4 + b];
            out.push((byte_val & 0x0f) as f32 * scale + bias);
            out.push((byte_val >> 4) as f32 * scale + bias);
        }
    }
    out.truncate(n_weights);
    out
}

/// Quantize a row-major f32 tensor into Q4_0 blocks, one row at a time.
///
/// `src` has shape `[rows, cols]`. Each row is quantized independently into
/// `cols.div_ceil(32)` blocks. Returns raw bytes (20 bytes per block).
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if the shape does not match `src`,
/// any source value is non-finite, or Q4 metadata is not representable in f16.
pub fn quantize_tensor_q4_0(
    src: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>, InferenceError> {
    let expected_len = rows.checked_mul(cols).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "Q4 tensor shape [{rows}, {cols}] overflows usize element count"
        ))
    })?;
    if src.len() != expected_len {
        return Err(InferenceError::InvalidInput(format!(
            "Q4 tensor source length {} does not match shape [{rows}, {cols}] \
             (expected {expected_len})",
            src.len()
        )));
    }
    let blocks_per_row = cols.div_ceil(Q4_BLOCK_WEIGHTS);
    let mut out = Vec::with_capacity(rows * blocks_per_row * 20);
    for row_idx in 0..rows {
        let row = &src[row_idx * cols..(row_idx + 1) * cols];
        out.extend_from_slice(&quantize_row_q4_0(row)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// BF16-input quantization API (for streaming model shards)
// ---------------------------------------------------------------------------

/// Validate that `shape.iter().product()` equals `data_len`.
///
/// SafeTensors' own `TensorView::new` rejects shape/data-size mismatches
/// (returns `InvalidTensorView`). The Q4 entry points keep the same
/// contract — without this check, a caller can produce a [`Q4Tensor`]
/// whose `shape` claims `[1, 96]` while `original_len` reads 64, and
/// `save_q4_file` will then write the inconsistent metadata into a `.q4`
/// header that downstream loaders (`write_merged_qkvz`, the Metal
/// runtime path) trust without re-verification. Uses `checked_mul` so
/// `usize` overflow on a malformed shape surfaces as a typed error rather
/// than wrapping to a plausible element count.
fn validate_shape_matches_data_len(shape: &[usize], data_len: usize) -> Result<(), InferenceError> {
    let numel = shape
        .iter()
        .try_fold(1_usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "Q4 tensor shape product overflows usize: shape={shape:?}"
            ))
        })?;
    if numel != data_len {
        return Err(InferenceError::InvalidInput(format!(
            "Q4 tensor shape product {numel} (shape={shape:?}) must equal data length {data_len}"
        )));
    }
    Ok(())
}

/// Quantize a BF16 tensor (raw `u16` slice) into a [`Q4Tensor`].
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if the shape does not match the
/// data or any BF16 value decodes to a non-finite f32 (NaN or ±inf).
pub fn quantize_bf16_to_q4(data: &[u16], shape: &[usize]) -> Result<Q4Tensor, InferenceError> {
    validate_shape_matches_data_len(shape, data.len())?;
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(Q4_BLOCK_WEIGHTS);
    let mut blocks = Vec::with_capacity(n_blocks);

    for chunk in data.chunks(32) {
        let mut vals = [0.0f32; 32];
        for (i, &v) in chunk.iter().enumerate() {
            vals[i] = bf16_to_f32(v);
        }
        blocks.push(quantize_block_with_mode_len(&vals, chunk.len(), false)?);
    }

    Ok(Q4Tensor {
        blocks,
        shape: shape.to_vec(),
        original_len,
    })
}

// ---------------------------------------------------------------------------
// QuaRot-pipeline quantization API (ADR-044 step 3c)
// ---------------------------------------------------------------------------

/// Quantize an `f32` tensor into a [`Q4Tensor`].
///
/// QuaRot offline-conversion entry point (ADR-044 §"Step 3c contract"). Prefer
/// this over [`quantize_bf16_to_q4`] when the source is the output of a
/// rotation pass and not a raw checkpoint, so the per-block `abs_max` is
/// computed from the same precision the upstream math produced rather than
/// from BF16-truncated values.
///
/// BF16's 7-bit mantissa is narrower than Q4_0's per-block scale resolution
/// (f16, 10-bit mantissa), so values pre-rounded to BF16 can sit on the wrong
/// side of a Q4 bin boundary or shift `abs_max` for the block. The f32 path
/// avoids that truncation.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if the shape does not match the
/// data or any value in `data` is non-finite.
pub fn quantize_f32_to_q4(data: &[f32], shape: &[usize]) -> Result<Q4Tensor, InferenceError> {
    validate_shape_matches_data_len(shape, data.len())?;
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(Q4_BLOCK_WEIGHTS);
    let mut blocks = Vec::with_capacity(n_blocks);

    for chunk in data.chunks(32) {
        let mut vals = [0.0f32; 32];
        vals[..chunk.len()].copy_from_slice(chunk);
        blocks.push(quantize_block_with_mode_len(&vals, chunk.len(), false)?);
    }

    Ok(Q4Tensor {
        blocks,
        shape: shape.to_vec(),
        original_len,
    })
}

/// Quantize an `f64` tensor into a [`Q4Tensor`] via f32 downcast.
///
/// Delegated wrapper around [`quantize_f32_to_q4`] for the QuaRot pipeline,
/// where rotation absorption runs in f64 per ADR-044 §Risks ("keep rotation
/// math in f64 [...] quantize in f32, store scales in f16 as before"). The
/// f32 downcast happens inside the per-block loop so callers do not allocate
/// an intermediate `Vec<f32>`.
///
/// **Intentionally f32-precision quantization.** This is NOT a true f64
/// quantizer — `abs_max`, the scale reciprocal, and the per-nibble round all
/// happen in f32, matching ADR-044 §Risks. The wrapper exists to avoid the
/// BF16 round-trip in [`quantize_bf16_to_q4`] and to skip an intermediate
/// f32 allocation at the call site, not to preserve f64 precision into the
/// nibble selection. Values within ~½ ULP of an f32 representation may
/// quantize to a different nibble than a hypothetical f64 reference would,
/// e.g., an exact f64 `0.5 - 1e-8` downcasts to f32 `0.5` and (with Rust's
/// `round` rounding halfway away from zero) lands on nibble 9 instead of 8.
/// QuaRot conversion accepts this — the dequantized magnitude is identical
/// at exact-midpoint values and rotated activations rarely sit on bin
/// boundaries.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if any f64 value is non-finite (NaN
/// or ±inf), or if the f32 downcast produces a non-finite value.
pub fn quantize_f64_to_q4(data: &[f64], shape: &[usize]) -> Result<Q4Tensor, InferenceError> {
    quantize_f64_to_q4_mode(data, shape, true) // symmetric — QuaRot-rotated weights are zero-mean
}

/// Quantize an `f64` tensor with explicit symmetry mode.
///
/// `symmetric=true` is required for Hadamard-rotated tensors (the rotation
/// makes them zero-mean, and asymmetric encoding wastes bits on a bias that
/// is approximately zero anyway, producing a 0.067·abs_max error on the zero
/// representation). Use `false` for raw weights with non-zero distributional
/// center.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if the shape does not match the
/// data or any value in `data` is non-finite.
pub fn quantize_f64_to_q4_mode(
    data: &[f64],
    shape: &[usize],
    symmetric: bool,
) -> Result<Q4Tensor, InferenceError> {
    validate_shape_matches_data_len(shape, data.len())?;
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(Q4_BLOCK_WEIGHTS);
    let mut blocks = Vec::with_capacity(n_blocks);

    for chunk in data.chunks(32) {
        let mut vals = [0.0f32; 32];
        for (i, &v) in chunk.iter().enumerate() {
            vals[i] = v as f32;
        }
        blocks.push(quantize_block_with_mode_len(&vals, chunk.len(), symmetric)?);
    }

    Ok(Q4Tensor {
        blocks,
        shape: shape.to_vec(),
        original_len,
    })
}

/// Dequantize all blocks of a [`Q4Tensor`] back to f32.
///
/// Output length equals `tensor.original_len` (zero-padded tail blocks are truncated).
pub fn dequantize_q4_to_f32(tensor: &Q4Tensor) -> Vec<f32> {
    let mut out = Vec::with_capacity(tensor.original_len);
    for block in &tensor.blocks {
        let scale = q4_f16_to_f32(block.scale);
        let bias = q4_f16_to_f32(block.bias);
        for b in 0..16 {
            let byte_val = block.packed[b];
            out.push((byte_val & 0x0f) as f32 * scale + bias);
            out.push((byte_val >> 4) as f32 * scale + bias);
        }
    }
    out.truncate(tensor.original_len);
    out
}

/// Quantize one BF16 shard (raw bytes, 2 bytes per value) into a `Vec<Q4Block>`.
///
/// Memory-efficient: the caller retains only one shard at a time.
///
/// # Errors
///
/// Returns an error if `bf16_bytes.len()` is odd (incomplete BF16 value).
pub fn stream_quantize_shard(
    bf16_bytes: &[u8],
) -> Result<Vec<Q4Block>, Box<dyn std::error::Error>> {
    if !bf16_bytes.len().is_multiple_of(2) {
        return Err("bf16_bytes length must be even (2 bytes per BF16 value)".into());
    }
    let n = bf16_bytes.len() / 2;
    let n_blocks = n.div_ceil(Q4_BLOCK_WEIGHTS);
    let mut blocks = Vec::with_capacity(n_blocks);

    for i in (0..bf16_bytes.len()).step_by(64) {
        let end = (i + 64).min(bf16_bytes.len());
        let chunk = &bf16_bytes[i..end];
        let mut vals = [0.0f32; 32];
        for (j, pair) in chunk.chunks_exact(2).enumerate() {
            let v = u16::from_ne_bytes([pair[0], pair[1]]);
            vals[j] = bf16_to_f32(v);
        }
        let valid_len = chunk.len() / 2;
        blocks.push(
            quantize_block_with_mode_len(&vals, valid_len, false)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?,
        );
    }

    Ok(blocks)
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Write a [`Q4Tensor`] to a `.q4` file.
///
/// File format:
/// ```text
/// magic        b"KHQ4"   4 bytes
/// version      2u32 LE   4 bytes
/// ndim         u32 LE    4 bytes
/// shape[i]     u64 LE × ndim
/// original_len u64 LE    8 bytes
/// blocks       [Q4Block; n]  n × 20 bytes
/// ```
pub fn save_q4_file(path: &std::path::Path, tensor: &Q4Tensor) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"KHQ4")?;
    f.write_all(&2u32.to_le_bytes())?;
    f.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;
    for &dim in &tensor.shape {
        f.write_all(&(dim as u64).to_le_bytes())?;
    }
    f.write_all(&(tensor.original_len as u64).to_le_bytes())?;
    // SAFETY: Q4Block is #[repr(C)] with size 20; its alignment is 2 (the
    // alignment of the leading `scale: u16` per the Rust Reference's repr(C)
    // rule). Casting to a `&[u8]` is valid because the target element type is
    // `u8` (alignment 1 ≤ source alignment 2). The resulting slice has length
    // `blocks.len() * 20` matching the source contiguous storage.
    let block_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            tensor.blocks.as_ptr().cast::<u8>(),
            tensor.blocks.len() * 20,
        )
    };
    f.write_all(block_bytes)
}

/// Header metadata returned by [`read_q4_header`] without allocating blocks.
pub struct Q4FileHeader {
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Number of valid original weights.
    pub original_len: usize,
    /// Byte offset in the file where the `Q4Block` payload starts.
    pub payload_offset: u64,
}

fn usize_from_u64(value: u64, what: &str) -> Result<usize, Box<dyn std::error::Error>> {
    usize::try_from(value).map_err(|_| format!("{what}: value {value} exceeds usize").into())
}

/// Validate a header-declared element count before allocating a buffer for it.
///
/// Custom `.q4`/`.f16` files carry untrusted `ndim`/`original_len`/`numel` fields
/// straight from disk. Without this guard, a crafted header can (a) overflow the
/// `count * elem_size` multiply (silently producing a wrong-sized buffer in release)
/// or (b) request an allocation far larger than the file, aborting the process with
/// an OOM. Both are denial-of-service / silent-corruption vectors on the
/// untrusted-checkpoint boundary (weight-loading sweep over #341/#342). A legitimate
/// payload is physically present in the file, so its byte length can never exceed
/// `file_len`; bounding by `file_len` therefore rejects only adversarial over-claims.
fn checked_alloc_bytes(
    count: usize,
    elem_size: usize,
    file_len: u64,
    what: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    let bytes = count
        .checked_mul(elem_size)
        .ok_or_else(|| format!("{what}: element count {count} × {elem_size} overflows usize"))?;
    if bytes as u64 > file_len {
        return Err(format!(
            "{what}: header claims {bytes} bytes but file is only {file_len} bytes"
        )
        .into());
    }
    Ok(bytes)
}

/// Byte length of `stream`, leaving its cursor position unchanged. `Seek`'s
/// own `stream_len` is not yet stable (`seek_stream_len`,
/// rust-lang/rust#59359), and this is the only length a generic
/// `Read + Seek` bound can observe -- there is no `.metadata()` to call once
/// the caller is no longer necessarily a concrete `std::fs::File`.
fn stream_len<S: std::io::Seek>(stream: &mut S) -> std::io::Result<u64> {
    use std::io::SeekFrom;
    let cur = stream.stream_position()?;
    let len = stream.seek(SeekFrom::End(0))?;
    stream.seek(SeekFrom::Start(cur))?;
    Ok(len)
}

/// Parse the header of a `.q4` file without reading the block payload.
///
/// On return the file cursor is positioned at the start of the block data.
///
/// # Errors
///
/// Returns an error on I/O failure, unrecognized magic bytes, or unsupported version.
pub fn read_q4_header<R: std::io::Read + std::io::Seek>(
    file: &mut R,
) -> Result<Q4FileHeader, Box<dyn std::error::Error>> {
    use std::io::{Read, SeekFrom};
    let file_len = stream_len(file)?;

    let (shape, original_len, payload_offset) = {
        let mut f = std::io::BufReader::new(&mut *file);

        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"KHQ4" {
            return Err("invalid magic: not a .q4 file".into());
        }

        let mut b4 = [0u8; 4];
        f.read_exact(&mut b4)?;
        let ver = u32::from_le_bytes(b4);
        if ver == 1 {
            return Err("legacy .q4 file (v1 symmetric format) — re-quantize with current quantize_q4 to produce v2 asymmetric blocks".into());
        }
        if ver != 2 {
            return Err(format!("unsupported .q4 file version: {ver}").into());
        }

        f.read_exact(&mut b4)?;
        let ndim = u32::from_le_bytes(b4) as usize;
        let shape_bytes = checked_alloc_bytes(ndim, 8, file_len, "shape dims")?;
        let mut shape = Vec::with_capacity(ndim);
        let mut b8 = [0u8; 8];
        for index in 0..ndim {
            f.read_exact(&mut b8)?;
            shape.push(usize_from_u64(
                u64::from_le_bytes(b8),
                &format!("shape dimension {index}"),
            )?);
        }

        f.read_exact(&mut b8)?;
        let original_len = usize_from_u64(u64::from_le_bytes(b8), "original_len")?;
        let payload_offset = 20u64
            .checked_add(shape_bytes as u64)
            .ok_or("Q4 payload offset overflows u64")?;
        (shape, original_len, payload_offset)
    };

    file.seek(SeekFrom::Start(payload_offset))?;

    Ok(Q4FileHeader {
        shape,
        original_len,
        payload_offset,
    })
}

/// Validate that `file_len` exactly covers the Q4 payload declared by
/// `header`, with no truncation or trailing data, and route the declared
/// tensor geometry through the shared ingress seam.
///
pub(crate) fn validate_q4_header_payload_bounds(
    header: &Q4FileHeader,
    file_len: u64,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let payload_bytes = header
        .original_len
        .div_ceil(Q4_BLOCK_WEIGHTS)
        .checked_mul(Q4_BLOCK_BYTES)
        .ok_or("Q4 block payload byte count overflows usize")? as u64;
    let required_len = header
        .payload_offset
        .checked_add(payload_bytes)
        .ok_or("Q4 payload end offset overflows u64")?;
    if file_len < required_len {
        return Err(format!(
            "{}: file truncated below Q4 block payload ({file_len} bytes < required {required_len})",
            path.display()
        )
        .into());
    }
    if file_len > required_len {
        return Err(format!(
            "{}: file has trailing bytes after Q4 block payload ({file_len} bytes > expected \
             {required_len})",
            path.display()
        )
        .into());
    }

    let source = path.display().to_string();
    crate::weights::ingress::validate_ingested_tensor(
        crate::weights::ingress::IngestedTensor::native_q4(
            &source,
            "native Q4 tensor",
            &header.shape,
            header.original_len,
            header.original_len.div_ceil(Q4_BLOCK_WEIGHTS),
        ),
    )?;
    Ok(())
}

/// Validate a `.q4` file's header, declared geometry, and exact on-disk
/// extent against `expected_shape` — without reading a single block byte.
///
/// This is the preflight every Q4 load path (CPU materializing load, Metal
/// no-copy mmap, MoE expert-cache mmap) shares: header parse plus
/// [`validate_q4_header_payload_bounds`] are both `O(1)` (fixed-size header
/// reads and a `file.metadata()` length check), so calling this before an
/// mmap never forces the payload's pages to be faulted in. Per-block
/// scale/bias finiteness is deliberately **not** checked here, because a
/// scan folded into a traversal the caller already performs is free while a
/// scan performed here is an extra pass over the same bytes.
///
/// This function therefore does **not** on its own make a file safe to hand
/// to a consumer that never reads the blocks. It is a preflight, not a
/// complete ingress check, and callers must additionally discharge the
/// per-block obligation described on [`validate_q4_block_metadata`] — either
/// by calling it from their own decode loop, or by calling
/// [`validate_q4_block_metadata_scan`] over the payload. Callers that obtain
/// the payload by memory-mapping go through [`open_and_mmap_q4_file`], whose
/// required `check` argument makes that choice unskippable.
///
/// # Errors
///
/// Returns an error on I/O failure, unrecognized magic bytes, unsupported
/// version, a truncated/oversized payload, or a shape mismatch against
/// `expected_shape`.
pub(crate) fn validate_q4_file<R: std::io::Read + std::io::Seek>(
    file: &mut R,
    path: &std::path::Path,
    expected_shape: Option<&[usize]>,
) -> Result<Q4FileHeader, Box<dyn std::error::Error>> {
    use std::io::SeekFrom;

    let header = read_q4_header(file)?;
    let file_len = stream_len(file)?;
    validate_q4_header_payload_bounds(&header, file_len, path)?;

    if let Some(expected_shape) = expected_shape {
        let source = path.display().to_string();
        let tensor_name = path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("native Q4 tensor");
        let block_count = header.original_len.div_ceil(Q4_BLOCK_WEIGHTS);
        crate::weights::ingress::validate_ingested_tensor(
            crate::weights::ingress::IngestedTensor::native_q4(
                &source,
                tensor_name,
                &header.shape,
                header.original_len,
                block_count,
            )
            .with_expected_shape(expected_shape),
        )?;
    }
    file.seek(SeekFrom::Start(header.payload_offset))?;
    Ok(header)
}

/// Validate one already-read Q4 block's scale/bias metadata (finite,
/// nonzero scale of either sign; finite bias) at the exact point a loader decodes
/// that block — never as a separate pre-scan pass.
///
/// `source`/`tensor_name`/`index` are provenance only, threaded through to
/// the error message so a rejected block names its file and position.
pub(crate) fn validate_q4_block_metadata(
    source: &str,
    tensor_name: &str,
    index: usize,
    scale_bits: u16,
    bias_bits: u16,
) -> Result<(), InferenceError> {
    crate::weights::ingress::validate_ingested_tensor(
        crate::weights::ingress::IngestedTensor::native_q4_block(
            source,
            tensor_name,
            index,
            scale_bits,
            bias_bits,
        ),
    )
}

/// Validate every Q4 block's scale/bias metadata in `payload`, one pass, for
/// consumers that never decode the blocks themselves.
///
/// `payload` is the block region only (the bytes at and after
/// [`Q4FileHeader::payload_offset`]); trailing bytes shorter than one block
/// are ignored here because [`validate_q4_header_payload_bounds`] has already
/// rejected any file whose extent is not an exact block multiple.
///
/// # Errors
///
/// Returns an error naming the first block whose scale is non-finite or
/// zero, or whose bias is non-finite.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn validate_q4_block_metadata_scan(
    source: &str,
    tensor_name: &str,
    payload: &[u8],
) -> Result<(), InferenceError> {
    for (index, chunk) in payload.chunks_exact(Q4_BLOCK_BYTES).enumerate() {
        let scale_bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
        let bias_bits = u16::from_ne_bytes([chunk[2], chunk[3]]);
        validate_q4_block_metadata(source, tensor_name, index, scale_bits, bias_bits)?;
    }
    Ok(())
}

/// How a caller of [`open_and_mmap_q4_file`] discharges the per-block
/// scale/bias obligation that [`validate_q4_file`] deliberately leaves open.
///
/// This is a required argument rather than a defaulted option so that adding
/// a new memory-mapping Q4 consumer forces an explicit answer to "who checks
/// the block metadata for these bytes?". A consumer that never traverses the
/// payload has no correct answer other than [`Q4BlockCheck::Now`].
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) enum Q4BlockCheck<'a> {
    /// Scan every block's scale/bias before the mapping is returned. Required
    /// for consumers that hand the mapped bytes to something other than a CPU
    /// decode loop — a no-copy GPU buffer, a DMA target, a raw slice — since
    /// nothing downstream of them will ever look at the metadata on the CPU.
    ///
    /// Costs one sequential pass over the payload, which for a no-copy
    /// mapping means faulting in pages the mapping was designed not to touch.
    Now { tensor_name: &'a str },
    /// The caller decodes every block it consumes and calls
    /// [`validate_q4_block_metadata`] inside that loop, so a separate pass
    /// would read the same bytes twice. `traversal` names the loop that
    /// discharges the obligation, so the claim can be checked against code
    /// rather than taken on trust.
    InCallerTraversal { traversal: &'static str },
}

/// Evidence that every block's scale/bias in a mapping was validated before
/// that mapping was handed to its consumer.
///
/// Only [`Q4BlockCheck::Now`] produces one. Constructors that publish mapped
/// bytes to something which will never decode them on the CPU take this by
/// value, so reaching such a constructor from a deferred check requires
/// visibly unwrapping a `None` rather than simply not writing a line.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) struct Q4BlocksChecked(());

/// Open `path`, run the [`validate_q4_file`] preflight against
/// `expected_shape`, discharge the per-block obligation per `check`, and
/// return the header, a read-only mapping of the whole file, and — when
/// `check` was [`Q4BlockCheck::Now`] — a [`Q4BlocksChecked`] witness.
///
/// Every memory-mapped Q4 consumer in this crate goes through here. That is
/// the point: the per-block scale/bias check is not something a mapping
/// consumer can reach the payload without having answered for, because the
/// only way to get the mapping is to pass a [`Q4BlockCheck`].
///
/// # Safety invariant
///
/// The returned mapping is read-only, and the model files must not be
/// modified while the process is running.
///
/// # Errors
///
/// Returns an error on I/O failure, a failed [`validate_q4_file`] preflight,
/// an mmap failure, a header whose `payload_offset` lies beyond the mapped
/// length, or — under [`Q4BlockCheck::Now`] — a block with a non-finite or
/// zero scale or a non-finite bias.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn open_and_mmap_q4_file(
    path: &std::path::Path,
    expected_shape: Option<&[usize]>,
    check: Q4BlockCheck<'_>,
) -> Result<(Q4FileHeader, memmap2::Mmap, Option<Q4BlocksChecked>), String> {
    let mut handle = crate::weights::mmap_trust::open_trusted_mmap_file(path)?;
    let header = validate_q4_file(&mut handle, path, expected_shape)
        .map_err(|e| format!("failed to validate Q4 payload {}: {e}", path.display()))?;

    let mmap = crate::weights::mmap_trust::map_and_verify_trusted(&handle)?;

    let payload = mmap.get(header.payload_offset as usize..).ok_or_else(|| {
        format!(
            "{}: payload_offset {} beyond mapped length {}",
            path.display(),
            header.payload_offset,
            mmap.len()
        )
    })?;

    let checked = match check {
        Q4BlockCheck::Now { tensor_name } => {
            let source = path.display().to_string();
            validate_q4_block_metadata_scan(&source, tensor_name, payload)
                .map_err(|e| e.to_string())?;
            Some(Q4BlocksChecked(()))
        }
        Q4BlockCheck::InCallerTraversal { traversal } => {
            debug_assert!(
                !traversal.is_empty(),
                "a deferred block check must name the traversal that discharges it"
            );
            None
        }
    };

    Ok((header, mmap, checked))
}

/// Load a [`Q4Tensor`] from a `.q4` file written by [`save_q4_file`].
///
/// # Errors
///
/// Returns an error on I/O failure, unrecognized magic bytes, or unsupported version.
pub fn load_q4_file(path: &std::path::Path) -> Result<Q4Tensor, Box<dyn std::error::Error>> {
    let f = crate::weights::mmap_trust::open_regular_file_no_hang(path)?;
    load_q4_from_open_file(f, path, None)
}

/// [`load_q4_file`] with the caller's required shape checked against the file's
/// own header, through the shared ingress seam, before any block is decoded.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn load_q4_file_checked(
    path: &std::path::Path,
    expected_shape: &[usize],
) -> Result<Q4Tensor, Box<dyn std::error::Error>> {
    let f = crate::weights::mmap_trust::open_regular_file_no_hang(path)?;
    load_q4_from_open_file(f, path, Some(expected_shape))
}

/// Parse a [`Q4Tensor`] from an already-open `.q4` file. Callers that resolved this file
/// through [`crate::weights::f32_weights::open_manifest_entry_once`] must read from that
/// opened fd rather than reopen by path -- see that function's docs. `path` is used only
/// for error messages and ingress provenance; this function never opens it.
pub(crate) fn load_q4_from_open_file(
    mut f: std::fs::File,
    path: &std::path::Path,
    expected_shape: Option<&[usize]>,
) -> Result<Q4Tensor, Box<dyn std::error::Error>> {
    use std::io::Read;
    let file_len = f.metadata()?.len();

    let header = validate_q4_file(&mut f, path, expected_shape)?;
    let n_blocks = header.original_len.div_ceil(Q4_BLOCK_WEIGHTS);

    let raw_len = checked_alloc_bytes(n_blocks, Q4_BLOCK_BYTES, file_len, "block payload")?;
    let mut raw = vec![0u8; raw_len];
    f.read_exact(&mut raw)?;

    // Single pass over the payload: decode each Q4Block AND validate its
    // scale/bias metadata here, rather than in a separate pre-scan over the
    // same bytes (`validate_q4_file` no longer does that scan — see its doc
    // comment). This is the only full-payload read `load_q4_file`/
    // `load_q4_file_checked` perform.
    let source = path.display().to_string();
    let tensor_name = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("native Q4 tensor");
    let mut blocks: Vec<Q4Block> = Vec::with_capacity(n_blocks);
    for (index, c) in raw.chunks_exact(Q4_BLOCK_BYTES).enumerate() {
        let scale = u16::from_ne_bytes([c[0], c[1]]);
        let bias = u16::from_ne_bytes([c[2], c[3]]);
        validate_q4_block_metadata(&source, tensor_name, index, scale, bias)?;
        let mut packed = [0u8; 16];
        packed.copy_from_slice(&c[4..20]);
        blocks.push(Q4Block {
            scale,
            bias,
            packed,
        });
    }

    Ok(Q4Tensor {
        blocks,
        shape: header.shape,
        original_len: header.original_len,
    })
}

/// Read a `.f16` (KHF1) file's header from an already-open handle, leaving the reader
/// positioned at the payload.
///
/// Takes the `File` rather than a path on purpose. A shape check is only meaningful for
/// the bytes it actually guards, so the header and the payload must come from the same
/// open handle; re-opening the pathname to read one and then the other lets the file be
/// replaced in between, and the validated header need not describe what gets materialized.
/// `display_path` is used only for error messages.
///
/// File format:
/// ```text
/// magic       b"KHF1"   4 bytes
/// version     1u32 LE   4 bytes
/// ndim        u32 LE    4 bytes
/// shape[i]    u64 LE × ndim
/// numel       u64 LE    8 bytes
/// data        [u16; numel]   numel × 2 bytes (IEEE-754 f16 bit patterns)
/// ```
fn read_f16_header(
    f: &mut std::fs::File,
    display_path: &str,
    file_len: u64,
) -> Result<(Vec<usize>, usize), Box<dyn std::error::Error>> {
    use std::io::Read;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"KHF1" {
        return Err(
            format!("invalid magic at {display_path}: expected KHF1, got {magic:?}").into(),
        );
    }

    let mut b4 = [0u8; 4];
    f.read_exact(&mut b4)?;
    if u32::from_le_bytes(b4) != 1 {
        return Err("unsupported .f16 file version".into());
    }

    f.read_exact(&mut b4)?;
    let ndim = u32::from_le_bytes(b4) as usize;
    checked_alloc_bytes(ndim, 8, file_len, "shape dims")?;
    let mut shape = Vec::with_capacity(ndim);
    let mut b8 = [0u8; 8];
    for index in 0..ndim {
        f.read_exact(&mut b8)?;
        shape.push(usize_from_u64(
            u64::from_le_bytes(b8),
            &format!("shape dimension {index}"),
        )?);
    }

    f.read_exact(&mut b8)?;
    let numel = usize_from_u64(u64::from_le_bytes(b8), "numel")?;

    let shape_product = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or("shape dims overflow usize")?;
    if shape_product != numel {
        return Err(format!(
            "{display_path}: shape product {shape_product} (shape={shape:?}) != numel {numel}"
        )
        .into());
    }

    Ok((shape, numel))
}

/// Why loading a `.f16` failed, when the caller needs to distinguish a shape
/// disagreement from every other failure in order to report it well.
#[derive(Debug)]
#[non_exhaustive]
pub enum F16LoadError {
    /// The header's declared shape disagreed with the shape the caller required.
    /// The payload was not read.
    ShapeMismatch {
        /// The shape the file's own header declares.
        declared: Vec<usize>,
    },
    /// Any other failure: I/O, bad magic, unsupported version, inconsistent header.
    Other(Box<dyn std::error::Error>),
}

impl std::fmt::Display for F16LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { declared } => {
                write!(f, "f16 header declares shape {declared:?}")
            }
            // Deliberately does NOT interpolate the wrapped error. `source()` below returns
            // that same error, and a consumer that walks the chain (or prints with `{:#}`)
            // would otherwise see the cause twice. `Display` states only what this wrapper
            // itself contributes; the cause is reached through `source()`.
            Self::Other(_) => write!(f, "f16 tensor could not be read"),
        }
    }
}

impl std::error::Error for F16LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            // Keep the wrapped cause reachable. `Display` alone flattens the chain, so a
            // caller that walks `source()` would otherwise see this wrapper as the root.
            Self::Other(e) => Some(e.as_ref()),
            Self::ShapeMismatch { .. } => None,
        }
    }
}

/// Load a KHF1 `.f16` tensor, requiring its header to declare `expected` before the
/// payload is read.
///
/// This exists instead of a public "read the shape" helper because reading the shape and
/// then loading the file are two opens of the same pathname, and nothing binds the shape
/// that was checked to the bytes that get materialized. Here the header is parsed and
/// compared on the same handle the payload is then read from, so passing the check is a
/// property of the actual tensor rather than of whatever happened to be at that path a
/// moment earlier.
///
/// # Errors
///
/// [`F16LoadError::ShapeMismatch`] if the header disagrees with `expected`, in which case
/// no payload is read; otherwise [`F16LoadError::Other`] on I/O failure, unrecognized
/// magic bytes, unsupported version, or malformed dimensions.
pub fn load_f16_tensor_file_expecting(
    path: &std::path::Path,
    expected: &[usize],
) -> Result<(Vec<f32>, Vec<usize>), F16LoadError> {
    let f = crate::weights::mmap_trust::open_regular_file_no_hang(path)
        .map_err(|e| F16LoadError::Other(e.into()))?;
    load_f16_tensor_from_open_file_expecting(f, &path.display().to_string(), expected)
}

/// [`load_f16_tensor_file_expecting`] for a handle the caller already opened.
///
/// Both properties hold at once here: the header is compared against `expected` on the
/// same handle the payload is read from, and that handle is the one the caller opened
/// (see [`crate::weights::f32_weights::open_manifest_entry_once`]) rather than a pathname
/// reopened afterwards. `display_path` is used only for error messages.
pub(crate) fn load_f16_tensor_from_open_file_expecting(
    mut f: std::fs::File,
    display_path: &str,
    expected: &[usize],
) -> Result<(Vec<f32>, Vec<usize>), F16LoadError> {
    let file_len = f
        .metadata()
        .map_err(|e| F16LoadError::Other(Box::new(e)))?
        .len();
    let (shape, numel) =
        read_f16_header(&mut f, display_path, file_len).map_err(F16LoadError::Other)?;
    if shape != expected {
        return Err(F16LoadError::ShapeMismatch { declared: shape });
    }
    read_f16_payload(&mut f, shape, numel, file_len, display_path, Some(expected))
        .map_err(F16LoadError::Other)
}

/// Load a tensor from a KHF1 `.f16` file, returning f32 values and shape.
///
/// File format:
/// ```text
/// magic       b"KHF1"   4 bytes
/// version     1u32 LE   4 bytes
/// ndim        u32 LE    4 bytes
/// shape[i]    u64 LE × ndim
/// numel       u64 LE    8 bytes
/// data        [u16; numel]   numel × 2 bytes (IEEE-754 f16 bit patterns)
/// ```
///
/// # Errors
///
/// Returns an error on I/O failure, malformed dimensions, unrecognized magic bytes, or
/// unsupported version.
pub fn load_f16_tensor_file(
    path: &std::path::Path,
) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let f = crate::weights::mmap_trust::open_regular_file_no_hang(path)?;
    load_f16_tensor_from_open_file(f, &path.display().to_string(), None)
}

/// [`load_f16_tensor_file`] with the caller's required shape checked through the
/// shared ingress seam.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn load_f16_tensor_file_checked(
    path: &std::path::Path,
    expected_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let f = crate::weights::mmap_trust::open_regular_file_no_hang(path)?;
    load_f16_tensor_from_open_file(f, &path.display().to_string(), Some(expected_shape))
}

/// Parse an f32 tensor from an already-open `.f16` file. Callers that resolved this file
/// through [`crate::weights::f32_weights::open_manifest_entry_once`] must read from that
/// opened fd rather than reopen by path -- see that function's docs.
/// `display_path` is used only for error messages and ingress provenance.
pub(crate) fn load_f16_tensor_from_open_file(
    mut f: std::fs::File,
    display_path: &str,
    expected_shape: Option<&[usize]>,
) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let file_len = f.metadata()?.len();
    let (shape, numel) = read_f16_header(&mut f, display_path, file_len)?;
    read_f16_payload(&mut f, shape, numel, file_len, display_path, expected_shape)
}

/// Read a `.f16` payload from a handle already positioned past its header, requiring
/// the file's length to match the header-declared extent exactly and routing the
/// decoded values through the shared ingress seam.
fn read_f16_payload(
    f: &mut std::fs::File,
    shape: Vec<usize>,
    numel: usize,
    file_len: u64,
    display_path: &str,
    expected_shape: Option<&[usize]>,
) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    use std::io::Read;
    let raw_len = checked_alloc_bytes(numel, 2, file_len, "f16 data")?;
    let shape_bytes = shape
        .len()
        .checked_mul(8)
        .ok_or("KHF1 shape byte count overflows usize")?;
    let payload_offset = 20u64
        .checked_add(shape_bytes as u64)
        .ok_or("KHF1 payload offset overflows u64")?;
    let required_len = payload_offset
        .checked_add(raw_len as u64)
        .ok_or("KHF1 payload end offset overflows u64")?;
    if file_len < required_len {
        return Err(format!(
            "{display_path}: file truncated below KHF1 payload ({file_len} bytes < required \
             {required_len})"
        )
        .into());
    }
    if file_len > required_len {
        return Err(format!(
            "{display_path}: file has trailing bytes after KHF1 payload ({file_len} bytes > \
             expected {required_len})"
        )
        .into());
    }
    let mut raw = vec![0u8; raw_len];
    f.read_exact(&mut raw)?;

    let values: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            q4_f16_to_f32(bits)
        })
        .collect();

    let tensor_name = std::path::Path::new(display_path)
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("native KHF1 tensor");
    let tensor = crate::weights::ingress::IngestedTensor::decoded_f32(
        display_path,
        tensor_name,
        &shape,
        "KHF1/F16",
        &values,
    );
    let tensor = if let Some(expected_shape) = expected_shape {
        tensor.with_expected_shape(expected_shape)
    } else {
        tensor
    };
    crate::weights::ingress::validate_ingested_tensor(tensor)?;

    Ok((values, shape))
}

// ---------------------------------------------------------------------------
// Merge-on-first-load `.q4` cache (`merged_qkvz_*.q4`) — content integrity
// ---------------------------------------------------------------------------
//
// `forward::metal_qwen35`'s Metal loader merges each GatedDeltaNet layer's
// `in_proj_qkv` and `in_proj_z` Q4 tensors into one `merged_qkvz_*.q4` file on
// first load so later loads can zero-copy mmap it like any other Q4 weight.
// The original cache-validity check compared only the merged file's *size*
// against the current source files' sizes (`#504`, second slice): a
// same-size stale or bit-rotted merged artifact would load silently. The
// functions below add a fail-closed content hash on top of that size check,
// mirroring `model::qwen`'s embedding-cache manifest guard (#504 first
// slice): hash the *current* source payloads and the merged file's on-disk
// payload, and only accept the cache when they match byte-for-byte via
// SHA-256. Any read/parse error is treated as invalid (reject, don't warn)
// so the caller always falls back to rebuilding the merge from trusted
// sources rather than trusting a file it could not fully verify.

/// Upper bound on a single `.q4` payload this module will read fully into
/// memory for content-integrity hashing (merge-on-first-load source
/// fingerprinting and merged-artifact verification). Generous relative to
/// any single per-layer GatedDeltaNet `in_proj_qkv`/`in_proj_z` tensor, while
/// still bounding the read so a corrupted or hostile file cannot drive an
/// unbounded allocation.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) const MAX_Q4_MERGE_PAYLOAD_LEN: u64 = 1 << 31; // 2 GiB

/// Read and validate a `.q4` file's header via [`read_q4_header`], then read
/// its payload (everything after the header) bounded by `max_len`.
///
/// The stat (via the header's file metadata) is only a fast-path; the read
/// itself is bounded via `take(max_len + 1)`, so a file that grows or is
/// swapped after the size check still cannot drive an allocation past the
/// cap. Mirrors `model::qwen::read_embedding_cache_file_bounded`.
///
/// Only compiled for tests or the `metal-gpu` feature: its sole caller is
/// the Metal merge-on-first-load `.q4` cache guard in
/// `forward::metal_qwen35`, which itself only exists under `metal-gpu`.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn read_q4_payload_bounded(
    path: &std::path::Path,
    max_len: u64,
) -> Result<(Q4FileHeader, Vec<u8>), Box<dyn std::error::Error>> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = crate::weights::mmap_trust::open_regular_file_no_hang(path)?;
    let header = read_q4_header(&mut file)?;
    let file_len = file.metadata()?.len();
    if file_len < header.payload_offset {
        return Err(format!(
            "{}: file truncated below header ({file_len} bytes < payload_offset {})",
            path.display(),
            header.payload_offset
        )
        .into());
    }
    let payload_len = file_len - header.payload_offset;
    if payload_len > max_len {
        return Err(format!(
            "{}: payload too large: {payload_len} bytes exceeds cap of {max_len} bytes",
            path.display()
        )
        .into());
    }

    file.seek(SeekFrom::Start(0))?;
    let header = validate_q4_file(&mut file, path, None)?;

    let mut buf = Vec::new();
    file.take(max_len.saturating_add(1)).read_to_end(&mut buf)?;
    if buf.len() as u64 > max_len {
        return Err(format!(
            "{}: payload too large: read exceeds cap of {max_len} bytes",
            path.display()
        )
        .into());
    }
    Ok((header, buf))
}

/// SHA-256 of `bytes`, formatted as lowercase hex. Mirrors
/// `model::qwen::embedding_cache_sha256_hex` / `download::sha256_hex`.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn q4_sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest.as_slice() {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

/// Expected byte length of a `merged_qkvz_*.q4` file built from `qkv_file_len`
/// and `z_file_len` (the on-disk lengths of the source `in_proj_qkv`/
/// `in_proj_z` files): a 36-byte header (`ndim=2`, `payload_offset=36` — all
/// source weight files are 2-D) plus both source payloads.
///
/// Returns `Err` instead of underflowing/panicking when a source file is
/// smaller than its own 36-byte header (truncated/corrupt source), so a
/// malformed source file fails closed here rather than wrapping to a bogus
/// huge `u64` via unchecked subtraction.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn merged_qkvz_expected_size(qkv_file_len: u64, z_file_len: u64) -> Result<u64, String> {
    const HEADER_LEN: u64 = 36;
    let qkv_payload_len = qkv_file_len.checked_sub(HEADER_LEN).ok_or_else(|| {
        format!("qkv source file too small: {qkv_file_len} bytes < {HEADER_LEN}-byte header")
    })?;
    let z_payload_len = z_file_len.checked_sub(HEADER_LEN).ok_or_else(|| {
        format!("z source file too small: {z_file_len} bytes < {HEADER_LEN}-byte header")
    })?;
    Ok(HEADER_LEN + qkv_payload_len + z_payload_len)
}

/// Content fingerprint (SHA-256 hex) of the *current* `qkv_path`/`z_path`
/// source payloads, in write order (qkv then z) — i.e. exactly the bytes
/// [`write_merged_qkvz`] would concatenate into a fresh merged file.
///
/// Reads both source payloads bounded by [`MAX_Q4_MERGE_PAYLOAD_LEN`].
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn merged_qkvz_source_fingerprint(
    qkv_path: &std::path::Path,
    z_path: &std::path::Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let (_qkv_hdr, mut qkv_payload) = read_q4_payload_bounded(qkv_path, MAX_Q4_MERGE_PAYLOAD_LEN)?;
    let (_z_hdr, z_payload) = read_q4_payload_bounded(z_path, MAX_Q4_MERGE_PAYLOAD_LEN)?;
    // Concatenate in write order (qkv then z) so this hashes exactly the
    // bytes `write_merged_qkvz` would produce as the merged payload.
    qkv_payload.extend_from_slice(&z_payload);
    Ok(q4_sha256_hex(&qkv_payload))
}

/// Content fingerprint (SHA-256 hex) of an existing merged file's on-disk
/// payload. Since [`write_merged_qkvz`] writes exactly `qkv_payload ||
/// z_payload` after its header, a valid, uncorrupted merged file's
/// fingerprint equals [`merged_qkvz_source_fingerprint`] computed from the
/// same source files at write time.
///
/// Reads the merged payload bounded by [`MAX_Q4_MERGE_PAYLOAD_LEN`].
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn merged_qkvz_file_fingerprint(
    merged_path: &std::path::Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let (_hdr, payload) = read_q4_payload_bounded(merged_path, MAX_Q4_MERGE_PAYLOAD_LEN)?;
    Ok(q4_sha256_hex(&payload))
}

/// Fail-closed validity check for a `merged_qkvz_*.q4` cache entry.
///
/// Returns `true` only when `merged_path` exists, its size matches
/// `expected_size`, and its content fingerprint matches a fingerprint
/// freshly derived from the *current* `qkv_path`/`z_path` source files. Any
/// I/O error, oversized payload, or malformed header on either side is
/// treated as invalid — this never warns-and-continues on a mismatch, it
/// always reports "rebuild from source" via `false`, so the caller
/// re-derives the merge from trusted inputs instead of trusting a merged
/// artifact it could not fully verify.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn merged_qkvz_cache_is_valid(
    merged_path: &std::path::Path,
    expected_size: u64,
    qkv_path: &std::path::Path,
    z_path: &std::path::Path,
) -> bool {
    let Ok(metadata) = std::fs::metadata(merged_path) else {
        return false;
    };
    if metadata.len() != expected_size {
        return false;
    }
    let Ok(source_fp) = merged_qkvz_source_fingerprint(qkv_path, z_path) else {
        return false;
    };
    let Ok(file_fp) = merged_qkvz_file_fingerprint(merged_path) else {
        return false;
    };
    source_fp == file_fp
}

/// Merge two Q4 files into a single concatenated Q4 file and write it to
/// `out_path`.
///
/// Reads only the raw bytes (no deserialization) and prepends a new KHQ4
/// header reflecting the merged shape. Uses a temp-file + atomic rename so a
/// crashed mid-write never leaves a partial file at the final path.
///
/// Returns `Err` if the model directory is read-only or I/O fails — callers
/// must fall back to the CPU concat path in that case.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn write_merged_qkvz(
    qkv_path: &std::path::Path,
    z_path: &std::path::Path,
    out_path: &std::path::Path,
) -> Result<(), String> {
    use std::io::Write;

    // Bounded reads with the same cap as the validator: a stat-then-
    // `read_to_end` here would let a source file that grows between the
    // metadata check and the read drive an unbounded allocation during a
    // cache rebuild.
    let (qkv_hdr, qkv_payload) = read_q4_payload_bounded(qkv_path, MAX_Q4_MERGE_PAYLOAD_LEN)
        .map_err(|e| format!("read {}: {e}", qkv_path.display()))?;
    let (z_hdr, z_payload) = read_q4_payload_bounded(z_path, MAX_Q4_MERGE_PAYLOAD_LEN)
        .map_err(|e| format!("read {}: {e}", z_path.display()))?;

    // Merged shape: rows = qkv_rows + z_rows, cols = hidden (shared). Both
    // source headers are untrusted on-disk data — a crafted or corrupt
    // rank-0 (`ndim = 0`) file would otherwise index `shape[0]` out of
    // bounds and panic here (denial of service), so both inputs must be
    // exactly 2-D before either dimension is read.
    if qkv_hdr.shape.len() != 2 {
        return Err(format!(
            "{}: qkv header shape {:?} is not 2-D (expected [rows, hidden])",
            qkv_path.display(),
            qkv_hdr.shape
        ));
    }
    if z_hdr.shape.len() != 2 {
        return Err(format!(
            "{}: z header shape {:?} is not 2-D (expected [rows, hidden])",
            z_path.display(),
            z_hdr.shape
        ));
    }
    if qkv_hdr.shape[1] != z_hdr.shape[1] {
        return Err(format!(
            "{}/{}: qkv hidden dimension {} does not match z hidden dimension {}",
            qkv_path.display(),
            z_path.display(),
            qkv_hdr.shape[1],
            z_hdr.shape[1]
        ));
    }
    let merged_rows = qkv_hdr.shape[0]
        .checked_add(z_hdr.shape[0])
        .ok_or("merged row count overflows usize")?;
    let cols = qkv_hdr.shape[1];
    let original_len = qkv_hdr
        .original_len
        .checked_add(z_hdr.original_len)
        .ok_or("merged original_len overflows usize")?;

    // Write to a temp file then rename atomically so partial writes are never trusted.
    let tmp = out_path.with_extension("q4.tmp");
    let write_result = (|| -> Result<(), String> {
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?,
        );
        // KHQ4 header: magic(4) + version(4) + ndim=2(4) + shape[0](8) + shape[1](8) + original_len(8)
        f.write_all(b"KHQ4").map_err(|e| e.to_string())?;
        // Version 2: asymmetric Q4 blocks (20 bytes each: scale + bias + 16 nibbles).
        f.write_all(&2u32.to_le_bytes())
            .map_err(|e| e.to_string())?;
        f.write_all(&2u32.to_le_bytes())
            .map_err(|e| e.to_string())?;
        f.write_all(&(merged_rows as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        f.write_all(&(cols as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        f.write_all(&(original_len as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        f.write_all(&qkv_payload).map_err(|e| e.to_string())?;
        f.write_all(&z_payload).map_err(|e| e.to_string())?;
        Ok(())
    })();

    if write_result.is_err() {
        let _ = std::fs::remove_file(&tmp);
        return write_result;
    }

    std::fs::rename(&tmp, out_path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename: {e}")
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Negative-scale Q4 fixture shared by every Q4 consumer's sign test.
///
/// Every block has a negative scale, so a consumer that drops the sign, clamps the
/// scale at zero or otherwise reads the scale's sign computes different weights. Each
/// consumer test runs the fixture, the same blocks with `|scale|`, and the same blocks
/// with a zero scale: the first must match the f64 reference, and the other two, which
/// are exactly what a sign-dropping or zero-clamping consumer would compute on the
/// fixture, must miss it by more than ten times the tolerance.
#[cfg(test)]
pub(crate) mod signed_scale_fixture;

#[cfg(test)]
mod tests;
