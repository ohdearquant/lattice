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

// ---------------------------------------------------------------------------
// Core block quantization
// ---------------------------------------------------------------------------

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
        let scale = if abs_max == 0.0 {
            1.0f32
        } else {
            abs_max / 7.0
        };
        let inv_scale = 1.0 / scale;
        let bias = -8.0 * scale;
        let mut packed = [0u8; 16];
        for b in 0..16 {
            let q0 = ((vals[2 * b] * inv_scale).round() + 8.0).clamp(0.0, 15.0) as u8;
            let q1 = ((vals[2 * b + 1] * inv_scale).round() + 8.0).clamp(0.0, 15.0) as u8;
            packed[b] = (q1 << 4) | (q0 & 0x0f);
        }
        Ok(Q4Block {
            scale: q4_f32_to_f16(scale),
            bias: q4_f32_to_f16(bias),
            packed,
        })
    } else {
        let real = &vals[..valid_len];
        let min_val = real.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = real.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range == 0.0 { 1.0f32 } else { range / 15.0 };
        let inv_scale = 1.0 / scale;
        let mut packed = [0u8; 16];
        for b in 0..16 {
            let q0 = (((vals[2 * b] - min_val) * inv_scale).round()).clamp(0.0, 15.0) as u8;
            let q1 = (((vals[2 * b + 1] - min_val) * inv_scale).round()).clamp(0.0, 15.0) as u8;
            packed[b] = (q1 << 4) | (q0 & 0x0f);
        }
        Ok(Q4Block {
            scale: q4_f32_to_f16(scale),
            bias: q4_f32_to_f16(min_val),
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
/// Returns [`InferenceError::InvalidInput`] if any value in `src` is non-finite.
pub fn quantize_row_q4_0(src: &[f32]) -> Result<Vec<u8>, InferenceError> {
    let n_blocks = src.len().div_ceil(32);
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
/// Returns [`InferenceError::InvalidInput`] if any value in `src` is non-finite.
pub fn quantize_tensor_q4_0(
    src: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>, InferenceError> {
    assert_eq!(
        src.len(),
        rows * cols,
        "src length does not match rows * cols"
    );
    let blocks_per_row = cols.div_ceil(32);
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

/// Assert that `shape.iter().product()` equals `data_len`.
///
/// SafeTensors' own `TensorView::new` rejects shape/data-size mismatches
/// (returns `InvalidTensorView`). The Q4 entry points keep the same
/// contract — without this check, a caller can produce a [`Q4Tensor`]
/// whose `shape` claims `[1, 96]` while `original_len` reads 64, and
/// `save_q4_file` will then write the inconsistent metadata into a `.q4`
/// header that downstream loaders (`write_merged_qkvz`, the Metal
/// runtime path) trust without re-verification. Uses `checked_mul` so
/// `usize` overflow on a malformed shape surfaces as a panic at
/// construction, not as a wraparound that aliases a valid length.
#[track_caller]
fn assert_shape_matches_data_len(shape: &[usize], data_len: usize) {
    let numel = shape
        .iter()
        .try_fold(1_usize, |acc, &d| acc.checked_mul(d))
        .unwrap_or_else(|| {
            panic!("shape product overflowed usize: shape={shape:?}");
        });
    assert_eq!(
        numel, data_len,
        "shape product {numel} (shape={shape:?}) must equal data length {data_len}"
    );
}

/// Quantize a BF16 tensor (raw `u16` slice) into a [`Q4Tensor`].
///
/// Panics if `shape.iter().product()` does not equal `data.len()`.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if any BF16 value decodes to a
/// non-finite f32 (NaN or ±inf).
pub fn quantize_bf16_to_q4(data: &[u16], shape: &[usize]) -> Result<Q4Tensor, InferenceError> {
    assert_shape_matches_data_len(shape, data.len());
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(32);
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
/// Panics if `shape.iter().product()` does not equal `data.len()`.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if any value in `data` is non-finite.
pub fn quantize_f32_to_q4(data: &[f32], shape: &[usize]) -> Result<Q4Tensor, InferenceError> {
    assert_shape_matches_data_len(shape, data.len());
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(32);
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
/// Panics if `shape.iter().product()` does not equal `data.len()`.
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
/// Returns [`InferenceError::InvalidInput`] if any value in `data` is non-finite.
pub fn quantize_f64_to_q4_mode(
    data: &[f64],
    shape: &[usize],
    symmetric: bool,
) -> Result<Q4Tensor, InferenceError> {
    assert_shape_matches_data_len(shape, data.len());
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(32);
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
    let n_blocks = n.div_ceil(32);
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

/// Parse the header of a `.q4` file without reading the block payload.
///
/// On return the file cursor is positioned at the start of the block data.
///
/// # Errors
///
/// Returns an error on I/O failure, unrecognized magic bytes, or unsupported version.
pub fn read_q4_header(file: &std::fs::File) -> Result<Q4FileHeader, Box<dyn std::error::Error>> {
    use std::io::Read;
    let file_len = file.metadata()?.len();
    let mut f = std::io::BufReader::new(file);

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
    checked_alloc_bytes(ndim, 8, file_len, "shape dims")?;
    let mut shape = Vec::with_capacity(ndim);
    let mut b8 = [0u8; 8];
    for _ in 0..ndim {
        f.read_exact(&mut b8)?;
        shape.push(u64::from_le_bytes(b8) as usize);
    }

    f.read_exact(&mut b8)?;
    let original_len = u64::from_le_bytes(b8) as usize;

    // payload_offset = 4 + 4 + 4 + ndim*8 + 8
    let payload_offset = (20 + ndim * 8) as u64;

    Ok(Q4FileHeader {
        shape,
        original_len,
        payload_offset,
    })
}

/// Validate that `file_len` bytes are enough to cover the full Q4 block
/// payload declared by `header`, without reading the payload itself.
///
/// [`load_q4_file`] fails closed on a truncated payload because its
/// `read_exact` for the block bytes returns an `Err` short of `n_blocks *
/// 20` bytes. The Metal no-copy mmap path (`forward::metal_qwen35::
/// mmap_q4_weight`) has no `read_exact` to fail — it hands the whole mmap
/// to the GPU — so this check is the sole gate standing between a
/// truncated on-disk `.q4` file and a Metal dispatch reading past the end
/// of the mapped payload.
///
/// Only compiled for tests or the `metal-gpu` feature: its sole caller is
/// the Metal no-copy `.q4` loader in `forward::metal_qwen35`.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn validate_q4_header_payload_bounds(
    header: &Q4FileHeader,
    file_len: u64,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let shape_product = header
        .shape
        .iter()
        .try_fold(1_usize, |acc, &d| acc.checked_mul(d))
        .ok_or("shape dims overflow usize")?;
    if shape_product != header.original_len {
        return Err(format!(
            "{}: shape product {shape_product} (shape={:?}) != original_len {}",
            path.display(),
            header.shape,
            header.original_len
        )
        .into());
    }

    let payload_bytes = header
        .original_len
        .div_ceil(32)
        .checked_mul(20)
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
    Ok(())
}

/// Load a [`Q4Tensor`] from a `.q4` file written by [`save_q4_file`].
///
/// # Errors
///
/// Returns an error on I/O failure, unrecognized magic bytes, or unsupported version.
pub fn load_q4_file(path: &std::path::Path) -> Result<Q4Tensor, Box<dyn std::error::Error>> {
    let f = std::fs::File::open(path)?;
    load_q4_from_open_file(f)
}

/// Parse a [`Q4Tensor`] from an already-open `.q4` file (FIX 7 fd-bind: callers that
/// resolved this file through [`crate::weights::f32_weights::open_contained_manifest_file`]
/// must read from that opened fd, not reopen by path -- see that function's docs).
pub(crate) fn load_q4_from_open_file(
    mut f: std::fs::File,
) -> Result<Q4Tensor, Box<dyn std::error::Error>> {
    use std::io::Read;
    let file_len = f.metadata()?.len();

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
    checked_alloc_bytes(ndim, 8, file_len, "shape dims")?;
    let mut shape = Vec::with_capacity(ndim);
    let mut b8 = [0u8; 8];
    for _ in 0..ndim {
        f.read_exact(&mut b8)?;
        shape.push(u64::from_le_bytes(b8) as usize);
    }

    f.read_exact(&mut b8)?;
    let original_len = u64::from_le_bytes(b8) as usize;

    // Fail closed on a header whose shape disagrees with its element count.
    // The quantize paths enforce `shape.product() == data.len()` via
    // `assert_shape_matches_data_len`; the loader must reject the same
    // inconsistency rather than return a tensor whose `shape` overstates the
    // block payload (downstream matmuls would read stale, out-of-range data).
    let shape_product = shape
        .iter()
        .try_fold(1_usize, |acc, &d| acc.checked_mul(d))
        .ok_or("shape dims overflow usize")?;
    if shape_product != original_len {
        return Err(format!(
            "shape product {shape_product} (shape={shape:?}) != original_len {original_len}"
        )
        .into());
    }

    let n_blocks = original_len.div_ceil(32);

    let raw_len = checked_alloc_bytes(n_blocks, 20, file_len, "block payload")?;
    let mut raw = vec![0u8; raw_len];
    f.read_exact(&mut raw)?;

    let blocks: Vec<Q4Block> = raw
        .chunks_exact(20)
        .map(|c| Q4Block {
            scale: u16::from_ne_bytes([c[0], c[1]]),
            bias: u16::from_ne_bytes([c[2], c[3]]),
            packed: c[4..20].try_into().expect("slice is exactly 16 bytes"),
        })
        .collect();

    Ok(Q4Tensor {
        blocks,
        shape,
        original_len,
    })
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
    let f = std::fs::File::open(path)?;
    load_f16_tensor_from_open_file(f, &path.display().to_string())
}

/// Parse an f32 tensor from an already-open `.f16` file (FIX 7 fd-bind: callers that
/// resolved this file through [`crate::weights::f32_weights::open_contained_manifest_file`]
/// must read from that opened fd, not reopen by path -- see that function's docs).
/// `display_path` is used only for error messages.
pub(crate) fn load_f16_tensor_from_open_file(
    mut f: std::fs::File,
    display_path: &str,
) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    use std::io::Read;
    let file_len = f.metadata()?.len();

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
    for _ in 0..ndim {
        f.read_exact(&mut b8)?;
        shape.push(u64::from_le_bytes(b8) as usize);
    }

    f.read_exact(&mut b8)?;
    let numel = u64::from_le_bytes(b8) as usize;

    let shape_product = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or("shape dims overflow usize")?;
    if shape_product != numel {
        return Err(
            format!("shape product {shape_product} (shape={shape:?}) != numel {numel}").into(),
        );
    }

    let raw_len = checked_alloc_bytes(numel, 2, file_len, "f16 data")?;
    let mut raw = vec![0u8; raw_len];
    f.read_exact(&mut raw)?;

    let values: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            q4_f16_to_f32(bits)
        })
        .collect();

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

    let file = std::fs::File::open(path)?;
    let header = read_q4_header(&file)?;
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

    let mut f = file;
    f.seek(SeekFrom::Start(header.payload_offset))?;
    let mut buf = Vec::new();
    f.take(max_len.saturating_add(1)).read_to_end(&mut buf)?;
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

    // Merged shape: rows = qkv_rows + z_rows, cols = hidden (shared)
    let merged_rows = qkv_hdr.shape[0] + z_hdr.shape[0];
    let cols = if qkv_hdr.shape.len() >= 2 {
        qkv_hdr.shape[1]
    } else {
        1
    };
    let original_len = qkv_hdr.original_len + z_hdr.original_len;

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

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test 1: Q4Block is exactly 20 bytes (scale + bias + 16 nibble bytes).
    // -----------------------------------------------------------------------
    #[test]
    fn test_q4_block_size() {
        assert_eq!(std::mem::size_of::<Q4Block>(), 20);
        let b = Q4Block {
            scale: 0,
            bias: 0,
            packed: [0u8; 16],
        };
        let base = std::ptr::from_ref(&b) as usize;
        let packed_off = std::ptr::from_ref(&b.packed) as usize - base;
        assert_eq!(
            packed_off, 4,
            "packed field must start at byte offset 4 (after scale + bias, no padding)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: All-zero roundtrip — zeros in must produce zeros out.
    // -----------------------------------------------------------------------
    #[test]
    fn test_quantize_dequantize_zeros() {
        let data = quantize_row_q4_0(&vec![0.0f32; 64]).unwrap();
        let out = dequantize_row_q4_0(&data, 64);
        assert_eq!(out.len(), 64);
        for v in &out {
            assert!(v.abs() < 1e-6, "expected ~0, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: Small positive values roundtrip within quantization tolerance.
    // -----------------------------------------------------------------------
    #[test]
    fn test_quantize_dequantize_small_values() {
        let src: Vec<f32> = (0..32).map(|i| i as f32 * 7.0 / 31.0).collect();
        let data = quantize_row_q4_0(&src).unwrap();
        let out = dequantize_row_q4_0(&data, 32);
        let max_err = src
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.5,
            "max abs error {max_err:.4} >= 0.5 for small values"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: Symmetric positive and negative values roundtrip.
    // -----------------------------------------------------------------------
    #[test]
    fn test_quantize_dequantize_symmetric() {
        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 15.5 * 7.0).collect();
        let data = quantize_row_q4_0(&src).unwrap();
        let out = dequantize_row_q4_0(&data, 32);
        let max_err = src
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.5,
            "max abs error {max_err:.4} >= 0.5 for symmetric values"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: max/min values map to nibbles 15/0 under asymmetric quantization.
    // -----------------------------------------------------------------------
    #[test]
    fn test_quantize_max_range() {
        // Block with w[0] = 7.0 (max, nibble 15) and w[1] = -7.0 (min, nibble 0), rest 0.
        let mut src = vec![0.0f32; 32];
        src[0] = 7.0;
        src[1] = -7.0;
        let data = quantize_row_q4_0(&src).unwrap();
        // Asymmetric: min=-7, max=7, scale = 14/15 ≈ 0.933
        // q[0] = round((7.0 - (-7.0)) / scale) = round(15) = 15 → low nibble
        // q[1] = round((-7.0 - (-7.0)) / scale) = round(0)  = 0  → high nibble
        // byte[0] = (0 << 4) | 15 = 0x0F.
        // Block layout: bytes 0..2 = scale, bytes 2..4 = bias, byte 4 = packed[0].
        let block_byte0 = data[4];
        assert_eq!(
            block_byte0 & 0x0f,
            15,
            "w[0]=7.0 (max) should produce low nibble 15"
        );
        assert_eq!(
            block_byte0 >> 4,
            0,
            "w[1]=-7.0 (min) should produce high nibble 0"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: Exactly 32 elements — single block roundtrip.
    // Values are in [-7, 7] so scale = 1.0 and max error is < 0.5 per step.
    // -----------------------------------------------------------------------
    #[test]
    fn test_quantize_single_block() {
        // Use values in [-7, 7] so scale = 7/7 = 1.0 and max quantization error = 0.5.
        let src: Vec<f32> = (0..32).map(|i| (i as f32 / 31.0) * 14.0 - 7.0).collect();
        let data = quantize_row_q4_0(&src).unwrap();
        assert_eq!(data.len(), 20, "single block must be 20 bytes");
        let out = dequantize_row_q4_0(&data, 32);
        assert_eq!(out.len(), 32);
        // With scale = 1.0 the max quantization error is 0.5 (half a step).
        // Use threshold 0.51 to account for f16 scale rounding.
        let max_err = src
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err <= 0.51,
            "max abs error {max_err:.4} > 0.51 for single block"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: 128 elements = 4 blocks.
    // -----------------------------------------------------------------------
    #[test]
    fn test_quantize_multiple_blocks() {
        let src: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 10.0).collect();
        let data = quantize_row_q4_0(&src).unwrap();
        assert_eq!(data.len(), 4 * 20, "4 blocks must be 80 bytes");
        let out = dequantize_row_q4_0(&data, 128);
        assert_eq!(out.len(), 128);
        let max_err = src
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.5,
            "max abs error {max_err:.4} >= 0.5 for multiple blocks"
        );
    }

    // -----------------------------------------------------------------------
    // Test 8: f32 → f16 → f32 roundtrip preserves value approximately.
    // -----------------------------------------------------------------------
    #[test]
    fn test_f16_roundtrip() {
        let values = [
            0.0f32,
            1.0,
            -1.0,
            0.5,
            -0.5,
            std::f32::consts::PI,
            100.0,
            -100.0,
            0.001,
            65504.0, // max finite f16
        ];
        for &v in &values {
            let bits = q4_f32_to_f16(v);
            let back = q4_f16_to_f32(bits);
            // f16 has ~3 decimal digits of precision; allow 0.2% relative error
            let rel_err = if v.abs() > 1e-4 {
                (v - back).abs() / v.abs()
            } else {
                (v - back).abs()
            };
            assert!(
                rel_err < 0.004,
                "f16 roundtrip failed for {v}: got {back}, rel_err={rel_err:.6}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 9: f16 helpers handle special values correctly.
    // -----------------------------------------------------------------------
    #[test]
    fn test_f16_special_values() {
        // +0 and -0
        assert_eq!(q4_f32_to_f16(0.0f32), 0x0000);
        assert_eq!(q4_f32_to_f16(-0.0f32), 0x8000);
        assert_eq!(q4_f16_to_f32(0x0000), 0.0f32);

        // +∞ and -∞
        let pos_inf = q4_f32_to_f16(f32::INFINITY);
        assert_eq!(pos_inf, 0x7c00);
        assert!(q4_f16_to_f32(pos_inf).is_infinite() && q4_f16_to_f32(pos_inf) > 0.0);

        let neg_inf = q4_f32_to_f16(f32::NEG_INFINITY);
        assert_eq!(neg_inf, 0xfc00);
        assert!(q4_f16_to_f32(neg_inf).is_infinite() && q4_f16_to_f32(neg_inf) < 0.0);

        // NaN round-trips to NaN
        let nan_bits = q4_f32_to_f16(f32::NAN);
        assert!(
            q4_f16_to_f32(nan_bits).is_nan(),
            "NaN should round-trip to NaN"
        );

        // Overflow → ±∞
        let overflow = q4_f32_to_f16(1.0e10f32);
        assert_eq!(overflow, 0x7c00, "overflow should produce +∞");
    }

    // -----------------------------------------------------------------------
    // Test 10: Nibble packing follows sequential-pairs layout.
    // -----------------------------------------------------------------------
    #[test]
    fn test_nibble_packing_order() {
        // Asymmetric block: w[0]=0.0, w[1]=7.0, rest 0.0.
        // min = 0, max = 7, scale = 7/15 ≈ 0.467, bias = 0.
        // q[0] = round((0-0)/scale) = 0  → low nibble 0
        // q[1] = round((7-0)/scale) = 15 → high nibble 15
        // byte[0] = (15 << 4) | 0 = 0xF0.
        // Layout: bytes 0..2 = scale, 2..4 = bias, 4 = packed[0].
        let mut src = vec![0.0f32; 32];
        src[0] = 0.0;
        src[1] = 7.0;
        let data = quantize_row_q4_0(&src).unwrap();
        let byte0 = data[4];
        assert_eq!(
            byte0, 0xF0,
            "byte[0] should be 0xF0 for w[0]=0.0 (nibble=0), w[1]=7.0 (nibble=15)"
        );

        // Dequant: nibble * scale + bias.
        let out = dequantize_row_q4_0(&data, 32);
        // weight[0] = 0 * 0.467 + 0 = 0 (exact)
        assert!(
            (out[0] - 0.0).abs() < 1e-3,
            "weight[0] should be ~0.0, got {}",
            out[0]
        );
        // weight[1] = 15 * scale + bias. With f16 scale rounding, ~7.0 ± 1 ULP.
        assert!(
            (out[1] - 7.0).abs() < 0.05,
            "weight[1] should be ~7.0, got {}",
            out[1]
        );
    }

    // -----------------------------------------------------------------------
    // Test 11: Multi-row per-row quantization via quantize_tensor_q4_0.
    // -----------------------------------------------------------------------
    #[test]
    fn test_quantize_tensor_rows() {
        let rows = 4usize;
        let cols = 64usize;
        let src: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 - 128.0) / 20.0)
            .collect();
        let data = quantize_tensor_q4_0(&src, rows, cols).unwrap();
        let blocks_per_row = cols.div_ceil(32); // 2 blocks per row of 64 cols
        assert_eq!(
            data.len(),
            rows * blocks_per_row * 20,
            "tensor bytes mismatch"
        );

        // Dequant each row and check roundtrip error.
        for row_idx in 0..rows {
            let row_bytes =
                &data[row_idx * blocks_per_row * 20..(row_idx + 1) * blocks_per_row * 20];
            let out = dequantize_row_q4_0(row_bytes, cols);
            let row_src = &src[row_idx * cols..(row_idx + 1) * cols];
            let max_err = row_src
                .iter()
                .zip(&out)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 0.5,
                "row {row_idx}: max abs error {max_err:.4} >= 0.5"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Additional tests (covering design doc test plan items 2–12 via Q4Tensor API)
    // -----------------------------------------------------------------------

    /// Build bf16 vals from f32 using the module's own helper.
    fn to_bf16(vals: &[f32]) -> Vec<u16> {
        vals.iter()
            .map(|&v| {
                // BF16 = upper 16 bits of f32
                let bits = v.to_bits();
                (bits >> 16) as u16
            })
            .collect()
    }

    fn bf16_round_trip(v: f32) -> f32 {
        bf16_to_f32((v.to_bits() >> 16) as u16)
    }

    #[test]
    fn test_quantize_dequantize_round_trip_zeros_bf16() {
        let data = vec![0u16; 64];
        let tensor = quantize_bf16_to_q4(&data, &[64]).unwrap();
        let out = dequantize_q4_to_f32(&tensor);
        assert_eq!(out.len(), 64);
        for v in &out {
            assert!(v.abs() < 1e-6, "expected ~0, got {v}");
        }
    }

    #[test]
    fn test_quantize_dequantize_round_trip_positive_bf16() {
        let f32_vals: Vec<f32> = (0..32).map(|i| i as f32 * 7.0 / 31.0).collect();
        let bf16_vals = to_bf16(&f32_vals);
        let tensor = quantize_bf16_to_q4(&bf16_vals, &[32]).unwrap();
        let out = dequantize_q4_to_f32(&tensor);
        // Compare against bf16-rounded originals (bf16 conversion is lossy at input).
        // Threshold 0.51 accounts for f16 scale rounding on top of the 0.5 quantization step.
        let max_err = f32_vals
            .iter()
            .zip(&out)
            .map(|(a, b)| (bf16_round_trip(*a) - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err <= 0.51, "max abs error {max_err:.4} > 0.51");
    }

    #[test]
    fn test_nibble_packing_byte_value_bf16() {
        // Asymmetric: w[0]=0.0, w[1]=7.0, rest 0.0
        // min=0, max=7, scale=7/15, bias=0. q[0]=0 (low), q[1]=15 (high). byte[0]=0xF0.
        let mut f32_vals = [0.0f32; 32];
        f32_vals[0] = 0.0;
        f32_vals[1] = 7.0;
        let bf16_vals = to_bf16(&f32_vals);
        let tensor = quantize_bf16_to_q4(&bf16_vals, &[32]).unwrap();
        assert_eq!(tensor.blocks.len(), 1);
        assert_eq!(
            tensor.blocks[0].packed[0], 0xF0,
            "byte[0] should be 0xF0 for w[0]=0.0 (nibble 0), w[1]=7.0 (nibble 15)"
        );
    }

    #[test]
    fn test_max_value_clamps_to_nibble_15() {
        // w[0]=100.0 → scale = 100/7 ≈ 14.28 → q[0] = round(7.0)+8 = 15
        let mut f32_vals = [0.0f32; 32];
        f32_vals[0] = 100.0;
        let bf16_vals = to_bf16(&f32_vals);
        let tensor = quantize_bf16_to_q4(&bf16_vals, &[32]).unwrap();
        let low_nibble = tensor.blocks[0].packed[0] & 0x0f;
        assert_eq!(low_nibble, 15, "weight[0]=100 should clamp to nibble 15");
    }

    #[test]
    fn test_block_boundary_continuity() {
        // Values chosen so that abs_max per block = 7.0 → scale = 1.0.
        // Every value is at least 1.0 above zero so no value rounds to nibble 8 (zero).
        // Block 0: all positive [1..7] repeated; block 1: all negative [-1..-7] repeated.
        let mut f32_vals = Vec::with_capacity(64);
        for i in 0..32 {
            f32_vals.push((i % 7) as f32 + 1.0);
        } // range [1, 7]
        for i in 0..32 {
            f32_vals.push(-((i % 7) as f32 + 1.0));
        } // range [-7, -1]
        let bf16_vals = to_bf16(&f32_vals);
        let tensor = quantize_bf16_to_q4(&bf16_vals, &[64]).unwrap();
        assert_eq!(tensor.blocks.len(), 2);
        let out = dequantize_q4_to_f32(&tensor);
        // All block-0 values are positive [1..7], scale≈1. Dequant ≥ (1-0.5)*1 = 0.5 > 0.
        for v in &out[0..32] {
            assert!(*v > 0.0, "block 0 weight should be positive, got {v}");
        }
        // All block-1 values are negative [-7..-1].
        for v in &out[32..64] {
            assert!(*v < 0.0, "block 1 weight should be negative, got {v}");
        }
    }

    #[test]
    fn test_save_load_round_trip() {
        let f32_vals: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 4.0).collect();
        let bf16_vals = to_bf16(&f32_vals);
        let original = quantize_bf16_to_q4(&bf16_vals, &[8, 8]).unwrap();
        let path = std::path::PathBuf::from("/tmp/test_q4_round_trip.q4");
        save_q4_file(&path, &original).unwrap();
        let loaded = load_q4_file(&path).unwrap();
        assert_eq!(loaded.shape, original.shape);
        assert_eq!(loaded.original_len, original.original_len);
        assert_eq!(loaded.blocks.len(), original.blocks.len());
        for (a, b) in original.blocks.iter().zip(&loaded.blocks) {
            assert_eq!(a.scale, b.scale, "scale mismatch after load");
            assert_eq!(a.packed, b.packed, "packed mismatch after load");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_stream_quantize_shard_matches_batch() {
        let f32_vals: Vec<f32> = (0..96).map(|i| i as f32 / 10.0).collect();
        let bf16_vals = to_bf16(&f32_vals);
        let batch_tensor = quantize_bf16_to_q4(&bf16_vals, &[96]).unwrap();
        // Convert bf16 u16s to raw bytes (native endian, matching stream_quantize_shard)
        let bf16_bytes: Vec<u8> = bf16_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let stream_blocks = stream_quantize_shard(&bf16_bytes).unwrap();
        assert_eq!(stream_blocks.len(), batch_tensor.blocks.len());
        for (a, b) in batch_tensor.blocks.iter().zip(&stream_blocks) {
            assert_eq!(a.scale, b.scale, "stream vs batch scale mismatch");
            assert_eq!(a.packed, b.packed, "stream vs batch packed mismatch");
        }
    }

    #[test]
    fn test_shape_preservation() {
        let shape = vec![4usize, 8, 4]; // 128 elements
        let data = vec![0u16; 128];
        let tensor = quantize_bf16_to_q4(&data, &shape).unwrap();
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.original_len, 128);
        assert_eq!(tensor.blocks.len(), 4); // 128 / 32 = 4

        let path = std::path::PathBuf::from("/tmp/test_q4_shape.q4");
        save_q4_file(&path, &tensor).unwrap();
        let loaded = load_q4_file(&path).unwrap();
        assert_eq!(loaded.shape, shape);
        assert_eq!(loaded.original_len, 128);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_round_trip_accuracy_tolerance() {
        // 1024 pseudo-random f32 in [-7, 7] using a simple LCG for reproducibility.
        // With scale ≈ 1.0 per block the theoretical max error per weight is 0.5
        // and expected MAE ≈ 0.25 for uniform random input.
        let mut state = 12345u64;
        let mut f32_vals = Vec::with_capacity(1024);
        for _ in 0..1024 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = ((state >> 32) as f32 / u32::MAX as f32) * 14.0 - 7.0;
            f32_vals.push(v);
        }
        let data = quantize_row_q4_0(&f32_vals).unwrap();
        let out = dequantize_row_q4_0(&data, 1024);
        assert_eq!(out.len(), 1024);
        let mae = f32_vals
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / 1024.0;
        // Threshold: Q4_0 with scale≈1.0 has MAE ≈ 0.25; allow 0.30 for block edge effects.
        assert!(
            mae < 0.30,
            "mean abs error {mae:.4} >= 0.30 (Q4 MAE for uniform [-7,7] expected ≈ 0.25)"
        );
    }

    // -----------------------------------------------------------------------
    // QuaRot pipeline entry points (ADR-044 step 3c-1)
    // -----------------------------------------------------------------------

    fn f32_to_bf16_bits(v: f32) -> u16 {
        // BF16 = top 16 bits of f32, round-to-nearest-even.
        let bits = v.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounding_bias = 0x7fff + lsb;
        ((bits.wrapping_add(rounding_bias)) >> 16) as u16
    }

    fn synthetic_f32_uniform(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (state >> 32) as f32 / u32::MAX as f32;
                u * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn quantize_f32_to_q4_shape_and_length() {
        let src = synthetic_f32_uniform(96, 17);
        let q = quantize_f32_to_q4(&src, &[3, 32]).unwrap();
        assert_eq!(q.shape, vec![3, 32]);
        assert_eq!(q.original_len, 96);
        assert_eq!(q.blocks.len(), 3, "96 elems = 3 full Q4 blocks");
    }

    #[test]
    fn quantize_f32_to_q4_pads_partial_block() {
        let src = synthetic_f32_uniform(40, 19);
        let q = quantize_f32_to_q4(&src, &[40]).unwrap();
        assert_eq!(q.original_len, 40);
        assert_eq!(q.blocks.len(), 2, "40 elems = 1 full + 1 partial Q4 block");
    }

    #[test]
    fn quantize_f32_to_q4_partial_block_uses_real_tail_min_max() {
        // Mutation-sensitive: a tail block of [5, 6, 7] zero-padded to 32
        // slots would (pre-fix) fold min/max over the padded zeros too,
        // yielding min=0/max=7 instead of the real min=5/max=7. This test
        // fails if the asymmetric path reverts to computing stats over the
        // padded [f32; 32] array instead of the real `chunk.len()` elements.
        let src = [5.0f32, 6.0, 7.0];
        let q = quantize_f32_to_q4(&src, &[3]).unwrap();
        assert_eq!(q.original_len, 3);
        assert_eq!(q.blocks.len(), 1);

        let block = q.blocks[0];
        assert_eq!(block.scale, q4_f32_to_f16(2.0f32 / 15.0));
        assert_eq!(block.bias, q4_f32_to_f16(5.0));
        assert_ne!(
            block.scale,
            q4_f32_to_f16(7.0f32 / 15.0),
            "partial tail scale must not include padded zero in max-min range"
        );
        assert_ne!(
            block.bias,
            q4_f32_to_f16(0.0),
            "partial tail bias must be the real tail min, not padded zero"
        );
    }

    #[test]
    fn quantize_f64_to_q4_symmetric_partial_block_is_bit_identical_to_padded_block() {
        // Symmetric mode must stay bit-identical to the old always-padded
        // path: adding zeros to a non-empty real chunk can never increase
        // abs_max, so the fixed length-aware helper must produce exactly the
        // same Q4Block as folding over the full zero-padded array.
        let src = [5.0f64, -6.0, 7.0];
        let mut padded = [0.0f32; 32];
        for (dst, src) in padded.iter_mut().zip(src.iter()) {
            *dst = *src as f32;
        }

        let expected = quantize_block_with_mode_len(&padded, 32, true).unwrap();
        let q = quantize_f64_to_q4_mode(&src, &[3], true).unwrap();

        assert_eq!(
            q.blocks[0], expected,
            "symmetric partial blocks must stay byte-identical to the old padded path"
        );
    }

    #[test]
    fn quantize_f64_to_q4_matches_f32_path_after_downcast() {
        // The f64 wrapper must agree byte-for-byte with the f32 entry under
        // the same symmetry mode. `quantize_f64_to_q4` defaults to symmetric
        // (Hadamard-rotated weights are zero-mean); the unrotated `quantize_
        // f32_to_q4` defaults to asymmetric. Both should produce identical
        // output when called with the same mode flag.
        let src_f64: Vec<f64> = synthetic_f32_uniform(256, 23)
            .into_iter()
            .map(f64::from)
            .collect();
        let src_f32: Vec<f32> = src_f64.iter().map(|&v| v as f32).collect();
        let q_f64 = quantize_f64_to_q4_mode(&src_f64, &[256], false).unwrap();
        let q_f32 = quantize_f32_to_q4(&src_f32, &[256]).unwrap();
        assert_eq!(q_f64.shape, q_f32.shape);
        assert_eq!(q_f64.original_len, q_f32.original_len);
        assert_eq!(
            q_f64.blocks.len(),
            q_f32.blocks.len(),
            "f64 path must produce same block count"
        );
        for (i, (a, b)) in q_f64.blocks.iter().zip(q_f32.blocks.iter()).enumerate() {
            assert_eq!(a.scale, b.scale, "block {i} scale mismatch");
            assert_eq!(a.bias, b.bias, "block {i} bias mismatch");
            assert_eq!(a.packed, b.packed, "block {i} packed mismatch");
        }
    }

    #[test]
    fn quantize_f32_to_q4_matches_bf16_path_when_input_is_bf16_castable() {
        // Control test: when the f32 input has zero mantissa entropy below the
        // BF16 truncation point (i.e., it was already bf16 -> f32), both paths
        // MUST produce identical Q4 tensors. This nails down the equivalence
        // so any divergence in the high-precision test below is provably
        // attributable to BF16 truncation, not to a behavioral difference
        // between the two quantize_* implementations.
        let bf16_bits: Vec<u16> = synthetic_f32_uniform(256, 29)
            .into_iter()
            .map(f32_to_bf16_bits)
            .collect();
        let f32_from_bf16: Vec<f32> = bf16_bits.iter().map(|&b| bf16_to_f32(b)).collect();

        let q_bf16 = quantize_bf16_to_q4(&bf16_bits, &[256]).unwrap();
        let q_f32 = quantize_f32_to_q4(&f32_from_bf16, &[256]).unwrap();
        assert_eq!(q_bf16.blocks.len(), q_f32.blocks.len());
        for (i, (a, b)) in q_bf16.blocks.iter().zip(q_f32.blocks.iter()).enumerate() {
            assert_eq!(a.scale, b.scale, "block {i} scale should match");
            assert_eq!(a.packed, b.packed, "block {i} packed should match");
        }
    }

    #[test]
    fn quantize_f32_to_q4_lower_error_than_bf16_path_on_high_precision_input() {
        // ADR-044 §"Step 3c contract" decision driver: when the source carries
        // >7 bits of mantissa entropy (e.g., the output of an f64 rotation
        // pass), the bf16 route discards information the f32 route preserves.
        //
        // Measurement: take 2048 pseudo-random f32 values uniform in [-1, 1]
        // (23-bit mantissa entropy). Quantize via both paths, dequantize, and
        // compare against the f32 source.
        //
        // Expectation: path (b) `quantize_f32_to_q4` produces strictly lower
        // max abs error AND lower mean abs error than path (a) f32->bf16->Q4.
        let src = synthetic_f32_uniform(2048, 31);
        let bf16_bits: Vec<u16> = src.iter().map(|&v| f32_to_bf16_bits(v)).collect();

        let q_bf16 = quantize_bf16_to_q4(&bf16_bits, &[2048]).unwrap();
        let q_f32 = quantize_f32_to_q4(&src, &[2048]).unwrap();
        let deq_bf16 = dequantize_q4_to_f32(&q_bf16);
        let deq_f32 = dequantize_q4_to_f32(&q_f32);

        let err = |reconstructed: &[f32]| -> (f32, f32) {
            let mut max_err = 0.0_f32;
            let mut sum_err = 0.0_f32;
            for (s, r) in src.iter().zip(reconstructed.iter()) {
                let e = (s - r).abs();
                max_err = max_err.max(e);
                sum_err += e;
            }
            (max_err, sum_err / src.len() as f32)
        };
        let (max_bf16, mean_bf16) = err(&deq_bf16);
        let (max_f32, mean_f32) = err(&deq_f32);

        // Self-documenting measurement print (visible via `cargo test -- --nocapture`).
        // Numbers feed the ADR-044 §"Step 3c contract" Q4 bridge decision record.
        eprintln!(
            "[3c-1 measurement] n=2048 source=f32 uniform [-1,1]: \
             f32_path mean_abs_err={mean_f32:.6} max_abs_err={max_f32:.6}; \
             bf16_path mean_abs_err={mean_bf16:.6} max_abs_err={max_bf16:.6}"
        );

        assert!(
            mean_f32 < mean_bf16,
            "f32 mean abs error ({mean_f32:.6}) should be < bf16 mean abs error ({mean_bf16:.6})"
        );
        assert!(
            max_f32 <= max_bf16,
            "f32 max abs error ({max_f32:.6}) should be <= bf16 max abs error ({max_bf16:.6})"
        );
    }

    // -----------------------------------------------------------------------
    // QuaRot composed rotated+Q4 forward gate (Issue #320)
    //
    // Exercises the full composition: absorb_rotations (offline rotation
    // absorption) → quantize_f64_to_q4 → dequantize_q4_to_f32 → matmul,
    // and asserts correctness against an independent f64 reference that
    // manually mirrors each step. Mutation-sensitive: perturbing the rotation
    // dispatch (pipeline.rs:226-230), absorption helpers (rotation.rs:161-164
    // or rotation.rs:184-192), Q4 symmetric scale (q4_weights.rs:277 or :280),
    // or Q4 symmetric mode flag (q4_weights.rs:523) must cause failure.
    // -----------------------------------------------------------------------
    #[test]
    fn quarot_rotated_q4_forward_matches_f64_reference() {
        use std::collections::HashMap;

        use crate::quant::quarot::hadamard::RandomizedHadamard;
        use crate::quant::quarot::pipeline::{TensorEntry, absorb_rotations};
        use crate::quant::quarot::plan::RotationPlan;

        const HIDDEN: usize = 32;
        const Q_ROWS: usize = 2; // q_proj input-side [2, 32]
        const O_ROWS: usize = 32; // o_proj output-side [32, 32]

        let q_name = "model.language_model.layers.0.self_attn.q_proj.weight";
        let o_name = "model.language_model.layers.0.self_attn.o_proj.weight";

        fn lcg_f64(n: usize, seed: u64) -> Vec<f64> {
            let mut state = seed;
            (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (state >> 32) as f64 / u32::MAX as f64 * 2.0 - 1.0
                })
                .collect()
        }

        fn matvec(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
            (0..rows)
                .map(|r| {
                    w[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(x)
                        .map(|(a, b)| a * b)
                        .sum()
                })
                .collect()
        }

        // NaN-honest max|a-b|: a `.fold(0.0, f64::max)` silently drops a NaN/Inf
        // operand (IEEE maxNum keeps the non-NaN side), letting a catastrophically
        // wrong output read as 0.0 and slip past a `<= tol` gate. Surface it instead.
        fn max_diff(a: &[f64], b: &[f64]) -> f64 {
            let mut max = 0.0_f64;
            for (x, y) in a.iter().zip(b) {
                let d = (x - y).abs();
                if !d.is_finite() {
                    return d;
                }
                if d > max {
                    max = d;
                }
            }
            max
        }

        // Independent symmetric Q4 dequant reference: re-implements
        // quantize_block_with_mode(symmetric=true) + dequantize_q4_to_f32
        // so that a bug in either production function is visible as a mismatch.
        fn ref_q4_dequant(data: &[f64]) -> Vec<f64> {
            let mut out = Vec::with_capacity(data.len());
            for chunk in data.chunks(32) {
                let f32s: Vec<f32> = chunk.iter().map(|&v| v as f32).collect();
                let abs_max = f32s.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
                let scale_f32 = if abs_max == 0.0 {
                    1.0_f32
                } else {
                    abs_max / 7.0
                };
                let bias_f32 = -8.0_f32 * scale_f32;
                // Round-trip through f16 storage exactly as quantize_block_with_mode does.
                let scale_dq = q4_f16_to_f32(q4_f32_to_f16(scale_f32));
                let bias_dq = q4_f16_to_f32(q4_f32_to_f16(bias_f32));
                let inv_scale = 1.0 / scale_f32;
                for &v in &f32s {
                    let nibble = ((v * inv_scale).round() + 8.0).clamp(0.0, 15.0) as u8;
                    out.push(f64::from(nibble as f32 * scale_dq + bias_dq));
                }
            }
            out
        }

        let q_data_orig = lcg_f64(Q_ROWS * HIDDEN, 0x1111_1111_1111_1111);
        let o_data_orig = lcg_f64(O_ROWS * HIDDEN, 0x2222_2222_2222_2222);
        let x_q = lcg_f64(HIDDEN, 0x3333_3333_3333_3333);
        let x_o = lcg_f64(HIDDEN, 0x4444_4444_4444_4444);

        let rotation = RandomizedHadamard::new(0x3200_0001, HIDDEN).expect("rotation init");
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();

        // ---- Production path: absorb_rotations + quantize_f64_to_q4 + dequantize ----
        let mut tensors: HashMap<String, TensorEntry> = HashMap::new();
        tensors.insert(
            q_name.to_string(),
            TensorEntry {
                name: q_name.to_string(),
                shape: vec![Q_ROWS, HIDDEN],
                data: q_data_orig.clone(),
            },
        );
        tensors.insert(
            o_name.to_string(),
            TensorEntry {
                name: o_name.to_string(),
                shape: vec![O_ROWS, HIDDEN],
                data: o_data_orig.clone(),
            },
        );

        absorb_rotations(&mut tensors, &plan, &rotation).expect("absorb_rotations");

        let q_q4 =
            quantize_f64_to_q4(&tensors[q_name].data, &[Q_ROWS, HIDDEN]).expect("q_proj quantize");
        let o_q4 =
            quantize_f64_to_q4(&tensors[o_name].data, &[O_ROWS, HIDDEN]).expect("o_proj quantize");

        // Shape and block-count sanity: fail loudly if the Q4 bridge is broken
        assert_eq!(q_q4.shape, vec![Q_ROWS, HIDDEN], "q_proj shape");
        assert_eq!(q_q4.original_len, Q_ROWS * HIDDEN, "q_proj original_len");
        assert_eq!(q_q4.blocks.len(), Q_ROWS, "[2,32] must produce 2 Q4 blocks");
        assert_eq!(o_q4.shape, vec![O_ROWS, HIDDEN], "o_proj shape");
        assert_eq!(o_q4.original_len, O_ROWS * HIDDEN, "o_proj original_len");
        assert_eq!(
            o_q4.blocks.len(),
            O_ROWS,
            "[32,32] must produce 32 Q4 blocks"
        );

        let q_deq: Vec<f64> = dequantize_q4_to_f32(&q_q4)
            .into_iter()
            .map(f64::from)
            .collect();
        let o_deq: Vec<f64> = dequantize_q4_to_f32(&o_q4)
            .into_iter()
            .map(f64::from)
            .collect();

        let prod_y_q = matvec(&q_deq, Q_ROWS, HIDDEN, &x_q);
        let prod_y_o = matvec(&o_deq, O_ROWS, HIDDEN, &x_o);

        // ---- Reference path: manual rotation + independent Q4 dequant ----

        // Input-side: apply rotation row-by-row (mirrors absorb_input_rotation_f64)
        let mut q_ref = q_data_orig.clone();
        for r in 0..Q_ROWS {
            rotation
                .apply_f64(&mut q_ref[r * HIDDEN..(r + 1) * HIDDEN])
                .expect("q_proj row rotation");
        }

        // Output-side: apply rotation column-by-column (mirrors absorb_output_rotation_f64)
        let mut o_ref = o_data_orig.clone();
        let mut col_buf = vec![0.0_f64; O_ROWS];
        for c in 0..HIDDEN {
            for r in 0..O_ROWS {
                col_buf[r] = o_ref[r * HIDDEN + c];
            }
            rotation
                .apply_f64(&mut col_buf)
                .expect("o_proj col rotation");
            for r in 0..O_ROWS {
                o_ref[r * HIDDEN + c] = col_buf[r];
            }
        }

        let q_ref_deq = ref_q4_dequant(&q_ref);
        let o_ref_deq = ref_q4_dequant(&o_ref);

        let ref_y_q = matvec(&q_ref_deq, Q_ROWS, HIDDEN, &x_q);
        let ref_y_o = matvec(&o_ref_deq, O_ROWS, HIDDEN, &x_o);

        // ---- Assert ----
        let max_q = max_diff(&prod_y_q, &ref_y_q);
        let max_o = max_diff(&prod_y_o, &ref_y_o);

        eprintln!("[quarot_q4_gate] max_abs_diff q_proj={max_q:.2e} o_proj={max_o:.2e}");

        assert!(
            max_q <= 1e-5,
            "q_proj forward max_abs_diff {max_q:.2e} > 1e-5: composed rotated+Q4 path is broken"
        );
        assert!(
            max_o <= 1e-5,
            "o_proj forward max_abs_diff {max_o:.2e} > 1e-5: composed rotated+Q4 path is broken"
        );
    }

    #[test]
    #[should_panic(expected = "shape product")]
    fn quantize_f32_to_q4_rejects_shape_data_mismatch() {
        let data = synthetic_f32_uniform(64, 41);
        // shape claims 96 elements; data has 64 → must panic.
        let _ = quantize_f32_to_q4(&data, &[3, 32]);
    }

    #[test]
    #[should_panic(expected = "shape product")]
    fn quantize_f64_to_q4_rejects_shape_data_mismatch() {
        let data: Vec<f64> = synthetic_f32_uniform(64, 43)
            .into_iter()
            .map(f64::from)
            .collect();
        let _ = quantize_f64_to_q4(&data, &[3, 32]);
    }

    #[test]
    #[should_panic(expected = "shape product")]
    fn quantize_bf16_to_q4_rejects_shape_data_mismatch() {
        // Lock the same contract on the pre-existing BF16 entry point — the
        // SafeTensors source format rejects shape/data mismatches and the Q4
        // bridge must not silently weaken that invariant.
        let data: Vec<u16> = (0..64).map(|i| i as u16).collect();
        let _ = quantize_bf16_to_q4(&data, &[3, 32]);
    }

    #[test]
    #[should_panic(expected = "overflowed usize")]
    fn quantize_f32_to_q4_rejects_shape_product_overflow() {
        let data = vec![0.0_f32; 32];
        // usize::MAX * 2 overflows; checked_mul must catch it before the
        // length comparison aliases to a valid length by wraparound.
        let _ = quantize_f32_to_q4(&data, &[usize::MAX, 2]);
    }

    #[test]
    fn quantize_f32_to_q4_block_layout_matches_quantize_row() {
        // Sanity: for input that is an exact multiple of 32, the entry should
        // produce the same per-block byte layout as `quantize_row_q4_0`
        // (which the existing kernels are already validated against).
        let src = synthetic_f32_uniform(128, 37);
        let q = quantize_f32_to_q4(&src, &[128]).unwrap();
        let row_bytes = quantize_row_q4_0(&src).unwrap();
        assert_eq!(row_bytes.len(), q.blocks.len() * 20);
        // SAFETY: Q4Block is #[repr(C)] size 20 (scale + bias + 16 nibbles),
        // alignment 2; byte-cast is valid because target element type is u8.
        let q_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(q.blocks.as_ptr().cast::<u8>(), q.blocks.len() * 20)
        };
        assert_eq!(q_bytes, row_bytes.as_slice());
    }

    // -----------------------------------------------------------------------
    // Tests for dequantize_row_q4_0 robustness (issue #263)
    //
    // These tests verify that dequantize_row_q4_0 does NOT panic on
    // misaligned or undersized inputs. The function uses chunks_exact(20)
    // which silently ignores trailing bytes, so removing the assert_eq!
    // alignment check makes the behaviour well-defined on any input.
    // -----------------------------------------------------------------------

    /// Misaligned input (25 bytes = 1 complete block + 5 remainder bytes) must not panic.
    /// The 5 trailing bytes are ignored; only the 1 complete block (32 values) is returned.
    #[test]
    fn dequantize_row_q4_0_misaligned_does_not_panic() {
        // Build a valid 1-block (20-byte) buffer by quantizing 32 known values.
        let src: Vec<f32> = (0..32).map(|i| (i as f32 / 31.0) * 14.0 - 7.0).collect();
        let mut buf = quantize_row_q4_0(&src).unwrap(); // exactly 20 bytes
        assert_eq!(buf.len(), 20);
        // Append 5 garbage bytes — total 25, which is NOT a multiple of 20.
        buf.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0xFF]);
        assert_eq!(buf.len(), 25);

        // Must not panic; chunks_exact(20) stops after the first complete block.
        let out = dequantize_row_q4_0(&buf, 32);

        // Should return exactly 32 values (one block worth).
        assert_eq!(out.len(), 32);

        // Round-trip tolerance: same threshold used by test_quantize_single_block.
        // With scale ≈ 1.0 (range = 14.0, 15 steps) the max error is ≤ 0.51.
        let max_err = src
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err <= 0.51,
            "max abs error {max_err:.4} > 0.51 for single-block misaligned input"
        );
    }

    /// Input shorter than one block (10 bytes < 20) must return an empty Vec.
    #[test]
    fn dequantize_row_q4_0_truncated_below_one_block() {
        let buf = vec![0xABu8; 10]; // 10 bytes — not even one complete block
        // Must not panic; chunks_exact(20) produces zero chunks → empty output.
        let out = dequantize_row_q4_0(&buf, 32);
        assert!(
            out.is_empty(),
            "expected empty Vec for sub-block input, got {} values",
            out.len()
        );
    }

    /// Clean 2-block (40-byte) input with n_weights=64 still returns 64 correct values.
    /// This is a regression guard: removing the assert must not break the happy path.
    #[test]
    fn dequantize_row_q4_0_exact_blocks_unchanged() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let data = quantize_row_q4_0(&src).unwrap();
        assert_eq!(data.len(), 40, "2-block input must be 40 bytes");
        let out = dequantize_row_q4_0(&data, 64);
        assert_eq!(out.len(), 64);
        let max_err = src
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.5,
            "max abs error {max_err:.4} >= 0.5 for exact 2-block input"
        );
    }

    // -----------------------------------------------------------------------
    // Adversarial header guards (weight-loading sweep): a crafted .q4/.f16
    // header must yield a clean Err, never an integer-overflow buffer or a
    // process-aborting OOM allocation.
    // -----------------------------------------------------------------------

    #[test]
    fn test_q4_rejects_huge_ndim() {
        // ndim = u32::MAX → unguarded Vec::with_capacity(ndim) is a ~34 GB OOM.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHQ4");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        let path = std::path::PathBuf::from("/tmp/test_q4_huge_ndim.q4");
        std::fs::write(&path, &buf).unwrap();
        let r = load_q4_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            r.is_err(),
            "u32::MAX ndim must be rejected, not OOM-aborted"
        );
    }

    #[test]
    fn test_read_q4_header_rejects_huge_ndim() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHQ4");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        let path = std::path::PathBuf::from("/tmp/test_q4_header_huge_ndim.q4");
        std::fs::write(&path, &buf).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let r = read_q4_header(&file);
        std::fs::remove_file(&path).ok();
        assert!(
            r.is_err(),
            "u32::MAX ndim in read_q4_header must be rejected"
        );
    }

    #[test]
    fn test_q4_rejects_huge_original_len() {
        // original_len = 2^62 → unguarded n_blocks*20 is a ~2.9 EB OOM.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHQ4");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
        buf.extend_from_slice(&4u64.to_le_bytes()); // shape[0]
        buf.extend_from_slice(&(1u64 << 62).to_le_bytes()); // original_len
        let path = std::path::PathBuf::from("/tmp/test_q4_huge_len.q4");
        std::fs::write(&path, &buf).unwrap();
        let r = load_q4_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            r.is_err(),
            "2^62 original_len must be rejected, not OOM-aborted"
        );
    }

    #[test]
    fn test_q4_rejects_shape_product_mismatch() {
        // shape product (4*16=64) disagrees with original_len (32): the header
        // claims twice as many elements as the block payload covers. The
        // quantize paths reject this via assert_shape_matches_data_len; the
        // loader must too, with a clean Err rather than a Q4Tensor whose shape
        // overstates its data (downstream matmuls would read stale elements).
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHQ4");
        buf.extend_from_slice(&2u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u32.to_le_bytes()); // ndim
        buf.extend_from_slice(&4u64.to_le_bytes()); // shape[0]
        buf.extend_from_slice(&16u64.to_le_bytes()); // shape[1] → product 64
        buf.extend_from_slice(&32u64.to_le_bytes()); // original_len (≠ 64)
        buf.extend_from_slice(&[0u8; 20]); // one valid-size block payload
        let path = std::path::PathBuf::from("/tmp/test_q4_shape_mismatch.q4");
        std::fs::write(&path, &buf).unwrap();
        let r = load_q4_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            r.is_err(),
            "shape product 64 != original_len 32 must be rejected"
        );
    }

    // -----------------------------------------------------------------------
    // validate_q4_header_payload_bounds (issue #540): the Metal no-copy mmap
    // loader has no `read_exact` to fail short on a truncated block payload
    // the way `load_q4_file` does, so this helper is the sole fail-closed
    // gate before a `.q4` file's mmap is handed to Metal dispatch.
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_q4_header_payload_bounds_rejects_truncated_payload() {
        // original_len=64 → 2 blocks × 20 bytes = 40 required payload bytes;
        // file ends exactly at payload_offset (zero payload bytes present).
        let header = Q4FileHeader {
            shape: vec![64],
            original_len: 64,
            payload_offset: 28,
        };
        let r = validate_q4_header_payload_bounds(&header, 28, &std::path::PathBuf::from("t.q4"));
        assert!(
            r.is_err(),
            "file truncated to payload_offset must be rejected"
        );
    }

    #[test]
    fn test_validate_q4_header_payload_bounds_rejects_one_byte_short() {
        let header = Q4FileHeader {
            shape: vec![64],
            original_len: 64,
            payload_offset: 28,
        };
        // Required length is payload_offset (28) + 40 = 68; one byte short.
        let r = validate_q4_header_payload_bounds(&header, 67, &std::path::PathBuf::from("t.q4"));
        assert!(r.is_err(), "payload one byte short of required must fail");
    }

    #[test]
    fn test_validate_q4_header_payload_bounds_accepts_exact_length() {
        let header = Q4FileHeader {
            shape: vec![64],
            original_len: 64,
            payload_offset: 28,
        };
        let r = validate_q4_header_payload_bounds(&header, 68, &std::path::PathBuf::from("t.q4"));
        assert!(
            r.is_ok(),
            "file with exactly the required payload bytes must be accepted: {r:?}"
        );
    }

    #[test]
    fn test_validate_q4_header_payload_bounds_rejects_shape_mismatch() {
        let header = Q4FileHeader {
            shape: vec![4, 16], // product 64
            original_len: 32,   // disagrees with shape product
            payload_offset: 36,
        };
        let r =
            validate_q4_header_payload_bounds(&header, 1_000, &std::path::PathBuf::from("t.q4"));
        assert!(
            r.is_err(),
            "shape product != original_len must be rejected before a payload-length check"
        );
    }

    #[test]
    fn test_validate_q4_header_payload_bounds_rejects_huge_original_len_overflow() {
        // original_len near usize::MAX must not panic on overflow in the
        // block-count/byte-count arithmetic; it must return a clean Err.
        let header = Q4FileHeader {
            shape: vec![usize::MAX],
            original_len: usize::MAX,
            payload_offset: 28,
        };
        let r =
            validate_q4_header_payload_bounds(&header, 1_000, &std::path::PathBuf::from("t.q4"));
        assert!(
            r.is_err(),
            "huge original_len must be rejected, not panic on overflow"
        );
    }

    #[test]
    fn test_f16_rejects_huge_numel() {
        // numel = 2^63 → unguarded numel*2 overflows usize to 0, silently
        // returning ([], [shape]) — wrong data with no error. Must be Err now.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHF1");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
        buf.extend_from_slice(&(1u64 << 63).to_le_bytes()); // shape[0]
        buf.extend_from_slice(&(1u64 << 63).to_le_bytes()); // numel
        let path = std::path::PathBuf::from("/tmp/test_f16_huge_numel.f16");
        std::fs::write(&path, &buf).unwrap();
        let r = load_f16_tensor_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            r.is_err(),
            "2^63 numel must be rejected, not silently truncated to empty"
        );
    }

    #[test]
    fn test_f16_rejects_shape_numel_mismatch() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHF1");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // ndim
        buf.extend_from_slice(&2u64.to_le_bytes()); // shape[0]
        buf.extend_from_slice(&2u64.to_le_bytes()); // shape[1]
        buf.extend_from_slice(&1u64.to_le_bytes()); // numel
        buf.extend_from_slice(&0u16.to_le_bytes()); // one valid f16 payload value
        let path = std::path::PathBuf::from("/tmp/test_f16_shape_numel_mismatch.f16");
        std::fs::write(&path, &buf).unwrap();
        let r = load_f16_tensor_file(&path);
        std::fs::remove_file(&path).ok();
        let err = r.expect_err("shape product != numel must be rejected");
        assert!(
            err.to_string().contains("shape product"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_f16_rejects_huge_ndim() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHF1");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        let path = std::path::PathBuf::from("/tmp/test_f16_huge_ndim.f16");
        std::fs::write(&path, &buf).unwrap();
        let r = load_f16_tensor_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            r.is_err(),
            "u32::MAX ndim in .f16 must be rejected, not OOM-aborted"
        );
    }

    #[test]
    fn test_q4_rejects_original_len_near_usize_max() {
        // original_len = usize::MAX - 3, with a single-dim shape equal to
        // original_len so shape_product == original_len and the loader
        // reaches the block-payload guard (not the earlier shape-mismatch
        // guard). n_blocks*20 does not itself overflow u64 at this
        // magnitude, so this exercises the file_len-bound branch of
        // checked_alloc_bytes: it must return a clean Err, never panic or
        // attempt a multi-exabyte allocation. Removing the `checked_mul`
        // guard (reverting to `n_blocks * 20`) does not panic here either
        // since the multiply itself doesn't overflow — this test instead
        // proves the *file_len bound* check is load-bearing on its own.
        let huge = usize::MAX - 3;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHQ4");
        buf.extend_from_slice(&2u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
        buf.extend_from_slice(&(huge as u64).to_le_bytes()); // shape[0] == original_len
        buf.extend_from_slice(&(huge as u64).to_le_bytes()); // original_len
        let path = std::path::PathBuf::from("/tmp/test_q4_original_len_near_usize_max.q4");
        std::fs::write(&path, &buf).unwrap();
        let r = load_q4_file(&path);
        std::fs::remove_file(&path).ok();
        let err = r.expect_err("original_len near usize::MAX must be rejected, not panic/OOM");
        let msg = err.to_string();
        assert!(
            msg.contains("block payload") || msg.contains("header claims"),
            "expected the block-payload allocation guard to fire, got: {msg}"
        );
    }

    #[test]
    fn test_f16_rejects_numel_whose_byte_count_exceeds_file_len() {
        // numel = usize::MAX / 4: numel*2 does NOT overflow (≈ 2^62), so the
        // checked_mul branch passes and rejection can only come from the
        // file_len-bound branch of checked_alloc_bytes. This pins that branch
        // for the f16 loader specifically — the assertion below must not
        // accept "overflows usize", or a deleted file_len check would go
        // unnoticed (the overflow branch is pinned separately by
        // test_f16_rejects_numel_that_wraps_to_small_value_on_overflow).
        let huge = usize::MAX / 4;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHF1");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
        buf.extend_from_slice(&(huge as u64).to_le_bytes()); // shape[0]
        buf.extend_from_slice(&(huge as u64).to_le_bytes()); // numel
        let path = std::path::PathBuf::from("/tmp/test_f16_numel_exceeds_file_len.f16");
        std::fs::write(&path, &buf).unwrap();
        let r = load_f16_tensor_file(&path);
        std::fs::remove_file(&path).ok();
        let err = r.expect_err("oversized f16 numel must be rejected, not panic/OOM");
        let msg = err.to_string();
        assert!(
            msg.contains("f16 data") && msg.contains("header claims"),
            "expected the f16-data file_len-bound guard to fire, got: {msg}"
        );
    }

    #[test]
    fn test_f16_rejects_numel_that_wraps_to_small_value_on_overflow() {
        // numel = usize::MAX/2 + 5: numel*2 overflows u64 and wraps to a
        // *small* residual (10, mod 2^64) that would sail past the
        // file_len-bound check if `checked_mul` were replaced by a plain
        // wrapping multiply — a silent-corruption bug (an ~empty read
        // reported as success with the wrong shape/numel) rather than the
        // OOM/panic the near-usize::MAX test above guards against. This is
        // the scenario `checked_mul` uniquely defends: the bound check alone
        // cannot catch it because the wrapped byte count looks small.
        let huge = usize::MAX / 2 + 5;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHF1");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
        buf.extend_from_slice(&(huge as u64).to_le_bytes()); // shape[0]
        buf.extend_from_slice(&(huge as u64).to_le_bytes()); // numel
        // Trailing filler: `huge * 2` wraps to a small residue (10 bytes) if
        // `checked_mul` is bypassed, so pad enough real bytes that a buggy
        // wrapping multiply would successfully `read_exact` a plausible
        // (wrong) buffer instead of also failing on a short read — isolating
        // the assertion to the overflow guard itself, not an incidental
        // short-file error.
        buf.extend_from_slice(&[0xABu8; 64]);
        let path =
            std::path::PathBuf::from("/tmp/test_f16_numel_wraps_to_small_value_on_overflow.f16");
        std::fs::write(&path, &buf).unwrap();
        let r = load_f16_tensor_file(&path);
        std::fs::remove_file(&path).ok();
        let err = r.expect_err(
            "numel whose ×2 wraps to a small value must still be rejected via checked_mul, \
             not silently accepted as a tiny (wrong) allocation",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("overflows usize"),
            "expected the checked_mul overflow branch specifically, got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // Non-finite input guard — mutation-sensitive tests (Finding 1, PR #452)
    //
    // IEEE-754: `NaN > x` and `NaN < x` are always false, so a plain
    // `f32::max` / `f32::min` fold over a block that contains NaN silently
    // ignores the NaN element and computes scale from the finite elements
    // only. The NaN then quantizes to nibble 0 via a saturating cast, so
    // no panic occurs and the caller receives a plausible-looking Q4Block
    // with a silently wrong entry. The guard at the top of
    // `quantize_block_with_mode` must catch this before the fold.
    //
    // Mutation sensitivity: removing the `if !v.is_finite()` guard converts
    // both `Err` returns below to `Ok`, turning `result.is_err()` → false
    // and failing the assertion.
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantize_block_rejects_nan_input() {
        // Block with one NaN among otherwise-valid weights must return Err.
        let mut vals = vec![1.0f32; 32];
        vals[7] = f32::NAN;
        let result = quantize_row_q4_0(&vals);
        assert!(
            result.is_err(),
            "NaN in weight block must be rejected with InvalidInput"
        );
    }

    #[test]
    fn test_quantize_block_rejects_inf_input() {
        // Block with one +inf element must return Err; the guard covers both
        // +inf and -inf via `is_finite()` (which returns false for any
        // non-finite value, including NaN, +inf, and -inf).
        let mut vals = vec![1.0f32; 32];
        vals[15] = f32::INFINITY;
        let result = quantize_row_q4_0(&vals);
        assert!(
            result.is_err(),
            "+inf in weight block must be rejected with InvalidInput"
        );
    }

    // -----------------------------------------------------------------------
    // Merge-on-first-load `merged_qkvz_*.q4` cache — content-integrity guard
    // (#504 remaining slice: "Merged-Q4 cache: compatibility check is
    // size-only — no content integrity on the merged artifact.")
    //
    // Mutation sensitivity: `merged_qkvz_cache_is_valid`'s size check alone
    // (the pre-fix behavior) would accept a same-size tampered/stale merged
    // file. `test_merged_qkvz_cache_rejects_same_size_corrupted_payload` and
    // `test_merged_qkvz_cache_rejects_same_size_stale_source` are the
    // discriminating tests: reverting the fingerprint comparison back to a
    // bare `metadata.len() == expected_size` check makes both pass
    // incorrectly (`is_valid` would wrongly return `true`), so they fail
    // under the reverted code. Verified manually per the task's mutation-test
    // protocol — see the session report for the reverse-apply/touch/restore
    // proof.
    // -----------------------------------------------------------------------

    /// Write a minimal valid 2-D `.q4` source file (`shape = [rows, cols]`,
    /// `rows * cols` must be a multiple of 32) with content derived from
    /// `seed` so distinct seeds produce distinct payload bytes.
    fn write_test_q4_source(path: &std::path::Path, rows: usize, cols: usize, seed: f32) {
        let n = rows * cols;
        let f32_vals: Vec<f32> = (0..n).map(|i| (i as f32 + seed) % 7.0 - 3.0).collect();
        let bf16_vals = to_bf16(&f32_vals);
        let tensor = quantize_bf16_to_q4(&bf16_vals, &[rows, cols]).unwrap();
        save_q4_file(path, &tensor).unwrap();
    }

    /// Build a fresh temp-dir-scoped triple of (qkv_path, z_path, merged_path)
    /// for one test, so parallel `cargo test` runs never collide on the same
    /// file. `qkv_seed`/`z_seed` control the source payload content.
    fn merge_test_paths(
        name: &str,
    ) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("lattice_test_merged_qkvz_{name}"));
        std::fs::create_dir_all(&dir).unwrap();
        (dir.join("qkv.q4"), dir.join("z.q4"), dir.join("merged.q4"))
    }

    #[test]
    fn test_merged_qkvz_expected_size_computes_correctly() {
        // 36-byte header each; qkv payload = 100 bytes, z payload = 40 bytes.
        let expected = merged_qkvz_expected_size(136, 76).unwrap();
        assert_eq!(expected, 36 + 100 + 40);
    }

    #[test]
    fn test_merged_qkvz_expected_size_rejects_truncated_source() {
        // A source file shorter than its own 36-byte header must fail
        // closed (`Err`), not underflow/panic via unchecked subtraction.
        let err = merged_qkvz_expected_size(20, 136).unwrap_err();
        assert!(
            err.contains("too small"),
            "expected a too-small error, got: {err}"
        );
    }

    #[test]
    fn test_write_merged_qkvz_then_cache_is_valid() {
        let (qkv_p, z_p, merged_p) = merge_test_paths("valid");
        write_test_q4_source(&qkv_p, 4, 8, 1.0);
        write_test_q4_source(&z_p, 4, 8, 5.0);

        write_merged_qkvz(&qkv_p, &z_p, &merged_p).unwrap();

        let qkv_len = std::fs::metadata(&qkv_p).unwrap().len();
        let z_len = std::fs::metadata(&z_p).unwrap().len();
        let expected_size = merged_qkvz_expected_size(qkv_len, z_len).unwrap();

        assert!(
            merged_qkvz_cache_is_valid(&merged_p, expected_size, &qkv_p, &z_p),
            "freshly written merged cache must validate against its own sources"
        );

        std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
    }

    #[test]
    fn test_write_merged_qkvz_rejects_oversized_source_payload() {
        // A source file whose payload exceeds MAX_Q4_MERGE_PAYLOAD_LEN must
        // fail closed BEFORE any payload allocation — the rebuild path uses
        // the same bounded reader as the validator. `set_len` produces a
        // sparse file, so this asserts the cap without 2 GiB of disk I/O.
        let (qkv_p, z_p, merged_p) = merge_test_paths("oversized_source");
        write_test_q4_source(&qkv_p, 4, 8, 1.0);
        write_test_q4_source(&z_p, 4, 8, 5.0);

        let f = std::fs::File::options().write(true).open(&qkv_p).unwrap();
        f.set_len(36 + MAX_Q4_MERGE_PAYLOAD_LEN + 1).unwrap();
        drop(f);

        let err = write_merged_qkvz(&qkv_p, &z_p, &merged_p)
            .expect_err("oversized source payload must be rejected, not read to EOF");
        assert!(
            err.contains("payload too large"),
            "expected a payload-cap error, got: {err}"
        );
        assert!(
            !merged_p.exists(),
            "no merged artifact may be produced from a rejected source"
        );

        std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
    }

    #[test]
    fn test_merged_qkvz_cache_rejects_missing_file() {
        let (qkv_p, z_p, merged_p) = merge_test_paths("missing");
        write_test_q4_source(&qkv_p, 4, 8, 1.0);
        write_test_q4_source(&z_p, 4, 8, 5.0);
        let qkv_len = std::fs::metadata(&qkv_p).unwrap().len();
        let z_len = std::fs::metadata(&z_p).unwrap().len();
        let expected_size = merged_qkvz_expected_size(qkv_len, z_len).unwrap();

        // merged_p was never written.
        assert!(!merged_qkvz_cache_is_valid(
            &merged_p,
            expected_size,
            &qkv_p,
            &z_p
        ));

        std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
    }

    #[test]
    fn test_merged_qkvz_cache_rejects_wrong_size() {
        let (qkv_p, z_p, merged_p) = merge_test_paths("wrongsize");
        write_test_q4_source(&qkv_p, 4, 8, 1.0);
        write_test_q4_source(&z_p, 4, 8, 5.0);
        write_merged_qkvz(&qkv_p, &z_p, &merged_p).unwrap();

        // Append a stray byte, making the on-disk size disagree with
        // `expected_size`.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&merged_p)
                .unwrap();
            f.write_all(&[0xAA]).unwrap();
        }

        let qkv_len = std::fs::metadata(&qkv_p).unwrap().len();
        let z_len = std::fs::metadata(&z_p).unwrap().len();
        let expected_size = merged_qkvz_expected_size(qkv_len, z_len).unwrap();

        assert!(!merged_qkvz_cache_is_valid(
            &merged_p,
            expected_size,
            &qkv_p,
            &z_p
        ));

        std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
    }

    #[test]
    fn test_merged_qkvz_cache_rejects_truncated_file() {
        let (qkv_p, z_p, merged_p) = merge_test_paths("truncated");
        write_test_q4_source(&qkv_p, 4, 8, 1.0);
        write_test_q4_source(&z_p, 4, 8, 5.0);
        write_merged_qkvz(&qkv_p, &z_p, &merged_p).unwrap();

        let full_len = std::fs::metadata(&merged_p).unwrap().len();
        let bytes = std::fs::read(&merged_p).unwrap();
        std::fs::write(&merged_p, &bytes[..bytes.len() - 10]).unwrap();

        let qkv_len = std::fs::metadata(&qkv_p).unwrap().len();
        let z_len = std::fs::metadata(&z_p).unwrap().len();
        let expected_size = merged_qkvz_expected_size(qkv_len, z_len).unwrap();
        assert_eq!(expected_size, full_len, "sanity: source sizes unchanged");

        assert!(!merged_qkvz_cache_is_valid(
            &merged_p,
            expected_size,
            &qkv_p,
            &z_p
        ));

        std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
    }

    #[test]
    fn test_merged_qkvz_cache_rejects_same_size_corrupted_payload() {
        // Same-size bit flip inside the merged payload — the pre-fix
        // size-only check would accept this file unchanged. This is the
        // mutation-sensitive case for the content-integrity fix itself.
        let (qkv_p, z_p, merged_p) = merge_test_paths("corrupted");
        write_test_q4_source(&qkv_p, 4, 8, 1.0);
        write_test_q4_source(&z_p, 4, 8, 5.0);
        write_merged_qkvz(&qkv_p, &z_p, &merged_p).unwrap();

        let qkv_len = std::fs::metadata(&qkv_p).unwrap().len();
        let z_len = std::fs::metadata(&z_p).unwrap().len();
        let expected_size = merged_qkvz_expected_size(qkv_len, z_len).unwrap();
        let full_len = std::fs::metadata(&merged_p).unwrap().len();
        assert_eq!(
            full_len, expected_size,
            "sanity: size unchanged by corruption"
        );

        // Flip one byte well inside the payload region (after the 36-byte
        // header) without changing the file's length.
        let mut bytes = std::fs::read(&merged_p).unwrap();
        let flip_at = bytes.len() - 5;
        bytes[flip_at] ^= 0xFF;
        std::fs::write(&merged_p, &bytes).unwrap();

        assert_eq!(
            std::fs::metadata(&merged_p).unwrap().len(),
            expected_size,
            "sanity: byte flip must not change file size"
        );

        assert!(
            !merged_qkvz_cache_is_valid(&merged_p, expected_size, &qkv_p, &z_p),
            "a same-size, bit-flipped merged payload must fail the content-integrity check \
             even though the size-only check would have accepted it"
        );

        std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
    }

    #[test]
    fn test_merged_qkvz_cache_rejects_same_size_stale_source() {
        // The z source file changes content (e.g. a re-quantize with
        // different weights) but keeps the exact same byte length, so the
        // merged filename (which encodes only sizes) and `expected_size`
        // are both unchanged. The stale merged cache must still be
        // rejected once content is checked.
        let (qkv_p, z_p, merged_p) = merge_test_paths("stale");
        write_test_q4_source(&qkv_p, 4, 8, 1.0);
        write_test_q4_source(&z_p, 4, 8, 5.0);
        write_merged_qkvz(&qkv_p, &z_p, &merged_p).unwrap();

        let qkv_len = std::fs::metadata(&qkv_p).unwrap().len();
        let z_len_before = std::fs::metadata(&z_p).unwrap().len();

        // Re-write z with different content but the same shape (same size).
        write_test_q4_source(&z_p, 4, 8, 99.0);
        let z_len_after = std::fs::metadata(&z_p).unwrap().len();
        assert_eq!(
            z_len_before, z_len_after,
            "sanity: same shape must produce the same file size"
        );

        let expected_size = merged_qkvz_expected_size(qkv_len, z_len_after).unwrap();
        assert_eq!(
            std::fs::metadata(&merged_p).unwrap().len(),
            expected_size,
            "sanity: merged file size still matches (source size unchanged)"
        );

        assert!(
            !merged_qkvz_cache_is_valid(&merged_p, expected_size, &qkv_p, &z_p),
            "a same-size stale source must invalidate the merged cache once content is checked"
        );

        std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
    }

    #[test]
    fn test_merged_qkvz_source_fingerprint_matches_file_fingerprint_after_write() {
        let (qkv_p, z_p, merged_p) = merge_test_paths("fingerprint");
        write_test_q4_source(&qkv_p, 4, 8, 2.0);
        write_test_q4_source(&z_p, 4, 8, 6.0);
        write_merged_qkvz(&qkv_p, &z_p, &merged_p).unwrap();

        let source_fp = merged_qkvz_source_fingerprint(&qkv_p, &z_p).unwrap();
        let file_fp = merged_qkvz_file_fingerprint(&merged_p).unwrap();
        assert_eq!(
            source_fp, file_fp,
            "a freshly written merged file's payload fingerprint must equal its sources' fingerprint"
        );

        std::fs::remove_dir_all(merged_p.parent().unwrap()).ok();
    }
}
