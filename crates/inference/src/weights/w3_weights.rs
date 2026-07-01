//! W3 per-block weight quantization for MLP-only decode-bandwidth reduction (issue #420).
//!
//! ## Format (v1 — asymmetric scale + bias, 16 bytes per block)
//!
//! Every 32 consecutive weights are packed into one [`W3Block`] of 16 bytes:
//! - `scale: u16` — per-block scale, stored as an IEEE-754 f16 bit pattern
//! - `bias: u16`  — per-block bias (zero-point), stored as an IEEE-754 f16 bit pattern
//! - `packed: [u8; 12]` — 32 unsigned 3-bit codes, sequential bit-packed, LSB-first
//!
//! This mirrors `q4_weights::Q4Block` (see that module's doc comment for the
//! design rationale of asymmetric scale+bias): same 32-weight group size, same
//! `weight = code * scale + bias` dequant math, same `K % 32 == 0` Metal
//! dispatch invariant. Only the code width (3 bits vs 4) and the packing
//! layout change, because 3-bit codes are not byte/nibble aligned.
//!
//! ### Bit layout (sequential, little-endian, LSB-first within `packed`)
//!
//! ```text
//! bit_offset = i * 3
//! byte_index = bit_offset / 8
//! shift      = bit_offset % 8
//! code[i] occupies bits [shift, shift+2] of the 16-bit little-endian word
//! formed by packed[byte_index] | (packed[byte_index + 1] << 8).
//! ```
//!
//! Concretely, the first four bytes are:
//! ```text
//! byte0: q0[0:2], q1[0:2], q2[0:1]
//! byte1: q2[2],   q3[0:2], q4[0:2], q5[0]
//! byte2: q5[1:2], q6[0:2], q7[0:2]
//! byte3: q8[0:2], q9[0:2], q10[0:1]
//! ```
//!
//! ### Dequantization
//!
//! ```text
//! weight[i] = code[i] as f32 * scale + bias
//! ```
//!
//! ### Encoding
//!
//! Asymmetric only (see ADR-420-1 in `.khive/reports` design doc for why):
//! `scale = (max - min) / 7`, `bias = min`,
//! `code[i] = clamp(round((weight[i] - min) / scale), 0, 7)`.
//!
//! ## File format (`.w3`)
//!
//! ```text
//! magic        b"KHW3"               4 bytes
//! version      1u32 LE               4 bytes
//! ndim         u32 LE                4 bytes
//! shape[i]     u64 LE × ndim
//! original_len u64 LE                8 bytes
//! blocks       [W3Block; n_blocks]   n_blocks × 16 bytes
//! ```
//!
//! ## Scope
//!
//! This module quantizes only dense MLP `gate_proj`/`up_proj`/`down_proj`
//! tensors ([`is_w3_mlp_tensor_name`]). Attention, GDN, embeddings, and
//! `lm_head` remain on the existing Q4/f16 path — see
//! `.khive/reports/w3_mlp_420_design.md` §"Non-goals".
//!
//! ## Status (see `impl_report.md` for the authoritative done-vs-designed split)
//!
//! DONE: CPU pack/dequant, `.w3` file I/O, MLP tensor-name classification.
//! DESIGNED, NOT IMPLEMENTED: `quantize_w3_mlp` converter binary, the mixed
//! W3/Q4 Metal loader (`from_w3_mlp_dir`), and the `gemv_w3_decode` /
//! `gemm_w3` Metal kernels. Do not wire this module into any Metal forward
//! path until those land — there is no dispatch code yet that consumes
//! [`W3Tensor`] or `.w3` files.

// W3 quantization operates on raw byte/u16 slices; unsafe is limited to the
// transmute-equivalent slice casts in save/load, mirroring q4_weights.rs.
#![allow(clippy::cast_possible_truncation)]

use crate::error::InferenceError;

/// Number of weights covered by one [`W3Block`].
pub const W3_GROUP_SIZE: usize = 32;
/// Packed 3-bit-code payload size in bytes (32 codes × 3 bits / 8 = 12 bytes).
pub const W3_PACKED_BYTES: usize = 12;
/// Total on-disk/in-memory size of one [`W3Block`] (2 + 2 + 12).
pub const W3_BLOCK_SIZE: usize = 16;

/// One W3 quantization block: 32 weights packed as unsigned 3-bit codes.
///
/// `scale`/`bias` are raw IEEE-754 f16 bit patterns in a `u16`, reusing the
/// same convention as [`super::q4_weights::Q4Block`] (`half` is not a
/// dependency of `lattice-inference`).
///
/// `packed` holds 32 codes in sequential bit-packed LSB-first layout — see
/// the module doc comment for the exact bit-offset formula. Dequant:
/// `weight[i] = code[i] as f32 * scale + bias`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct W3Block {
    /// f16 bit pattern for the per-block scale — 2 bytes.
    pub scale: u16,
    /// f16 bit pattern for the per-block minimum (bias) — 2 bytes.
    pub bias: u16,
    /// 32 unsigned 3-bit codes packed as 12 bytes.
    pub packed: [u8; W3_PACKED_BYTES],
}

// Compile-time size assertion — must be exactly 16 bytes (2 + 2 + 12, no padding).
const _: () = assert!(std::mem::size_of::<W3Block>() == W3_BLOCK_SIZE);

/// A W3-quantized tensor.
#[derive(Debug, Clone)]
pub struct W3Tensor {
    /// Quantized blocks, each covering 32 weights.
    pub blocks: Vec<W3Block>,
    /// Original tensor shape (e.g., `[rows, cols]` for a 2-D weight matrix).
    pub shape: Vec<usize>,
    /// Number of valid original weights — may be less than `blocks.len() * 32`.
    pub original_len: usize,
}

// ---------------------------------------------------------------------------
// Reuse of Q4's f16 <-> f32 / bf16 -> f32 helpers.
//
// The design doc explicitly allows either sharing the Q4 helpers or copying
// them locally "as Q4 already does in local modules." Sharing avoids
// duplicating ~90 lines of bit-twiddling and keeps a single source of truth
// for the f16 conversion; both modules live in the same crate so `pub(crate)`
// visibility on the Q4 helpers is sufficient.
// ---------------------------------------------------------------------------
use super::q4_weights::{q4_f16_to_f32 as w3_f16_to_f32, q4_f32_to_f16 as w3_f32_to_f16};

#[inline]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

// ---------------------------------------------------------------------------
// Bit packing / unpacking for 32 unsigned 3-bit codes <-> 12 bytes.
// ---------------------------------------------------------------------------

/// Pack 32 unsigned 3-bit codes (each `<= 7`) into 12 bytes, sequential
/// LSB-first layout. Codes above 7 are masked to their low 3 bits.
fn pack_w3_codes(codes: &[u8; 32]) -> [u8; W3_PACKED_BYTES] {
    let mut packed = [0u8; W3_PACKED_BYTES];
    for (i, &code) in codes.iter().enumerate() {
        let bit_offset = i * 3;
        let byte_index = bit_offset / 8;
        let shift = bit_offset % 8;
        let v = u16::from(code & 0x7);
        let lo = u16::from(packed[byte_index]);
        let hi = if byte_index + 1 < W3_PACKED_BYTES {
            u16::from(packed[byte_index + 1])
        } else {
            0
        };
        let word = (lo | (hi << 8)) | (v << shift);
        packed[byte_index] = (word & 0xff) as u8;
        if byte_index + 1 < W3_PACKED_BYTES {
            packed[byte_index + 1] = ((word >> 8) & 0xff) as u8;
        }
    }
    packed
}

/// Unpack 12 bytes into 32 unsigned 3-bit codes (`0..=7`).
fn unpack_w3_codes(packed: &[u8; W3_PACKED_BYTES]) -> [u8; 32] {
    let mut codes = [0u8; 32];
    for (i, code) in codes.iter_mut().enumerate() {
        let bit_offset = i * 3;
        let byte_index = bit_offset / 8;
        let shift = bit_offset % 8;
        let lo = u16::from(packed[byte_index]);
        let hi = if byte_index + 1 < W3_PACKED_BYTES {
            u16::from(packed[byte_index + 1])
        } else {
            0
        };
        let word = lo | (hi << 8);
        *code = ((word >> shift) & 0x7) as u8;
    }
    codes
}

// ---------------------------------------------------------------------------
// Core block quantization
// ---------------------------------------------------------------------------

/// Quantize exactly 32 f32 values into one [`W3Block`] using asymmetric mode.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if any value in `vals` is
/// non-finite. IEEE-754 `NaN > x` is always false, so a NaN would silently
/// leave the min/max accumulators unchanged and produce wrong-but-no-error
/// quantization (mirrors `q4_weights::quantize_block_with_mode`).
#[inline]
fn quantize_block_w3(vals: &[f32; 32]) -> Result<W3Block, InferenceError> {
    for (i, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            return Err(InferenceError::InvalidInput(format!(
                "W3 weight block element {i} contains a non-finite value ({v}); \
                 source weights must be finite"
            )));
        }
    }
    let min_val = vals.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    let scale = if range == 0.0 { 1.0f32 } else { range / 7.0 };
    let inv_scale = 1.0 / scale;
    let mut codes = [0u8; 32];
    for (i, &v) in vals.iter().enumerate() {
        codes[i] = (((v - min_val) * inv_scale).round()).clamp(0.0, 7.0) as u8;
    }
    Ok(W3Block {
        scale: w3_f32_to_f16(scale),
        bias: w3_f32_to_f16(min_val),
        packed: pack_w3_codes(&codes),
    })
}

// ---------------------------------------------------------------------------
// Public quantization API
// ---------------------------------------------------------------------------

/// Quantize a slice of f32 values into W3 blocks.
///
/// The input is processed 32 elements at a time; the last block is
/// zero-padded if `src.len()` is not a multiple of 32.
///
/// Returns raw bytes containing tightly-packed [`W3Block`]s (16 bytes each).
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if any value in `src` is non-finite.
pub fn quantize_row_w3(src: &[f32]) -> Result<Vec<u8>, InferenceError> {
    let n_blocks = src.len().div_ceil(W3_GROUP_SIZE);
    let mut out = Vec::with_capacity(n_blocks * W3_BLOCK_SIZE);
    for chunk in src.chunks(W3_GROUP_SIZE) {
        let mut vals = [0.0f32; 32];
        vals[..chunk.len()].copy_from_slice(chunk);
        let block = quantize_block_w3(&vals)?;
        // SAFETY: W3Block is #[repr(C)] with size 16; its alignment is 2 (the
        // alignment of the leading `scale: u16` per repr(C) rules). Casting to
        // `&[u8; 16]` is valid because the target element type is `u8`
        // (alignment 1 <= source alignment 2) and lengths match exactly.
        let bytes: &[u8; W3_BLOCK_SIZE] = unsafe { &*std::ptr::from_ref(&block).cast() };
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

/// Dequantize W3 blocks (raw bytes) back to f32 values.
///
/// Trailing bytes beyond the last complete 16-byte block are silently
/// ignored; the function returns `min(n_weights, (data.len() / 16) * 32)`
/// values and never panics regardless of input length.
pub fn dequantize_row_w3(data: &[u8], n_weights: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_weights);
    for chunk in data.chunks_exact(W3_BLOCK_SIZE) {
        let scale = w3_f16_to_f32(u16::from_ne_bytes([chunk[0], chunk[1]]));
        let bias = w3_f16_to_f32(u16::from_ne_bytes([chunk[2], chunk[3]]));
        let packed: [u8; W3_PACKED_BYTES] = chunk[4..16]
            .try_into()
            .expect("chunk slice is exactly W3_PACKED_BYTES");
        let codes = unpack_w3_codes(&packed);
        for code in codes {
            out.push(f32::from(code) * scale + bias);
        }
    }
    out.truncate(n_weights);
    out
}

/// Quantize a row-major f32 tensor into W3 blocks, one row at a time.
///
/// `src` has shape `[rows, cols]`. Each row is quantized independently into
/// `cols.div_ceil(32)` blocks. Returns raw bytes (16 bytes per block).
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if `src.len() != rows * cols`
/// (via checked multiplication) or if any value in `src` is non-finite.
pub fn quantize_tensor_w3(
    src: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>, InferenceError> {
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "quantize_tensor_w3: rows {rows} * cols {cols} overflows usize"
        ))
    })?;
    if src.len() != expected {
        return Err(InferenceError::ShapeMismatch {
            name: "quantize_tensor_w3 src".to_string(),
            expected: vec![rows, cols],
            actual: vec![src.len()],
        });
    }
    let blocks_per_row = cols.div_ceil(W3_GROUP_SIZE);
    let mut out = Vec::with_capacity(rows * blocks_per_row * W3_BLOCK_SIZE);
    for row_idx in 0..rows {
        let row = &src[row_idx * cols..(row_idx + 1) * cols];
        out.extend_from_slice(&quantize_row_w3(row)?);
    }
    Ok(out)
}

/// Validate that `shape.iter().product()` equals `data_len`, using checked
/// multiplication so a malformed shape surfaces as an error rather than a
/// `usize` wraparound aliasing a valid length (mirrors
/// `q4_weights::assert_shape_matches_data_len`, but returns `Result` per the
/// design's public-API error-handling rule instead of panicking).
fn checked_shape_matches_data_len(
    shape: &[usize],
    data_len: usize,
) -> Result<usize, InferenceError> {
    let Some(numel) = shape.iter().try_fold(1_usize, |acc, &d| acc.checked_mul(d)) else {
        return Err(InferenceError::InvalidInput(format!(
            "shape product overflowed usize: shape={shape:?}"
        )));
    };
    if numel != data_len {
        return Err(InferenceError::ShapeMismatch {
            name: "w3 tensor shape".to_string(),
            expected: shape.to_vec(),
            actual: vec![data_len],
        });
    }
    Ok(numel)
}

/// Quantize a BF16 tensor (raw `u16` slice) into a [`W3Tensor`].
///
/// # Errors
///
/// Returns [`InferenceError::ShapeMismatch`] if `shape.iter().product() !=
/// data.len()`, or [`InferenceError::InvalidInput`] if any BF16 value decodes
/// to a non-finite f32 (NaN or ±inf).
pub fn quantize_bf16_to_w3(data: &[u16], shape: &[usize]) -> Result<W3Tensor, InferenceError> {
    checked_shape_matches_data_len(shape, data.len())?;
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(W3_GROUP_SIZE);
    let mut blocks = Vec::with_capacity(n_blocks);

    for chunk in data.chunks(W3_GROUP_SIZE) {
        let mut vals = [0.0f32; 32];
        for (i, &v) in chunk.iter().enumerate() {
            vals[i] = bf16_to_f32(v);
        }
        blocks.push(quantize_block_w3(&vals)?);
    }

    Ok(W3Tensor {
        blocks,
        shape: shape.to_vec(),
        original_len,
    })
}

/// Quantize an `f32` tensor into a [`W3Tensor`].
///
/// # Errors
///
/// Returns [`InferenceError::ShapeMismatch`] if `shape.iter().product() !=
/// data.len()`, or [`InferenceError::InvalidInput`] if any value is non-finite.
pub fn quantize_f32_to_w3(data: &[f32], shape: &[usize]) -> Result<W3Tensor, InferenceError> {
    checked_shape_matches_data_len(shape, data.len())?;
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(W3_GROUP_SIZE);
    let mut blocks = Vec::with_capacity(n_blocks);

    for chunk in data.chunks(W3_GROUP_SIZE) {
        let mut vals = [0.0f32; 32];
        vals[..chunk.len()].copy_from_slice(chunk);
        blocks.push(quantize_block_w3(&vals)?);
    }

    Ok(W3Tensor {
        blocks,
        shape: shape.to_vec(),
        original_len,
    })
}

/// Dequantize all blocks of a [`W3Tensor`] back to f32.
///
/// Output length equals `tensor.original_len` (zero-padded tail blocks are truncated).
pub fn dequantize_w3_to_f32(tensor: &W3Tensor) -> Vec<f32> {
    let mut out = Vec::with_capacity(tensor.original_len);
    for block in &tensor.blocks {
        let scale = w3_f16_to_f32(block.scale);
        let bias = w3_f16_to_f32(block.bias);
        let codes = unpack_w3_codes(&block.packed);
        for code in codes {
            out.push(f32::from(code) * scale + bias);
        }
    }
    out.truncate(tensor.original_len);
    out
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Write a [`W3Tensor`] to a `.w3` file.
///
/// File format:
/// ```text
/// magic        b"KHW3"   4 bytes
/// version      1u32 LE   4 bytes
/// ndim         u32 LE    4 bytes
/// shape[i]     u64 LE × ndim
/// original_len u64 LE    8 bytes
/// blocks       [W3Block; n]  n × 16 bytes
/// ```
///
/// # Errors
///
/// Returns [`InferenceError::Io`] on any write failure.
pub fn save_w3_file(path: &std::path::Path, tensor: &W3Tensor) -> Result<(), InferenceError> {
    use std::io::Write;
    let mut f = std::fs::File::create(path).map_err(InferenceError::Io)?;
    f.write_all(b"KHW3").map_err(InferenceError::Io)?;
    f.write_all(&1u32.to_le_bytes())
        .map_err(InferenceError::Io)?;
    f.write_all(&(tensor.shape.len() as u32).to_le_bytes())
        .map_err(InferenceError::Io)?;
    for &dim in &tensor.shape {
        f.write_all(&(dim as u64).to_le_bytes())
            .map_err(InferenceError::Io)?;
    }
    f.write_all(&(tensor.original_len as u64).to_le_bytes())
        .map_err(InferenceError::Io)?;
    // SAFETY: W3Block is #[repr(C)] with size 16, alignment 2; casting to
    // `&[u8]` is valid because the target element type is `u8` (alignment
    // 1 <= source alignment 2). Resulting slice length is
    // `blocks.len() * 16`, matching the source contiguous storage.
    let block_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            tensor.blocks.as_ptr().cast::<u8>(),
            tensor.blocks.len() * W3_BLOCK_SIZE,
        )
    };
    f.write_all(block_bytes).map_err(InferenceError::Io)
}

/// Header metadata returned by [`read_w3_header`] without allocating blocks.
pub struct W3FileHeader {
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Number of valid original weights.
    pub original_len: usize,
    /// Byte offset in the file where the `W3Block` payload starts.
    pub payload_offset: u64,
}

/// Validate a header-declared element count before allocating a buffer for
/// it, bounding by the physical file length (mirrors
/// `q4_weights::checked_alloc_bytes`) so a crafted `.w3` header cannot
/// overflow a `count * elem_size` multiply or trigger an OOM abort.
fn checked_alloc_bytes(
    count: usize,
    elem_size: usize,
    file_len: u64,
    what: &str,
) -> Result<usize, InferenceError> {
    let bytes = count.checked_mul(elem_size).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "{what}: element count {count} × {elem_size} overflows usize"
        ))
    })?;
    if bytes as u64 > file_len {
        return Err(InferenceError::InvalidInput(format!(
            "{what}: header claims {bytes} bytes but file is only {file_len} bytes"
        )));
    }
    Ok(bytes)
}

/// Parse the header of a `.w3` file without reading the block payload.
///
/// # Errors
///
/// Returns [`InferenceError::Io`] on I/O failure, or
/// [`InferenceError::InvalidSafetensors`] on unrecognized magic bytes or
/// unsupported version.
pub fn read_w3_header(file: &std::fs::File) -> Result<W3FileHeader, InferenceError> {
    use std::io::Read;
    let file_len = file.metadata().map_err(InferenceError::Io)?.len();
    let mut f = std::io::BufReader::new(file);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(InferenceError::Io)?;
    if &magic != b"KHW3" {
        return Err(InferenceError::InvalidSafetensors(
            "invalid magic: not a .w3 file".to_string(),
        ));
    }

    let mut b4 = [0u8; 4];
    f.read_exact(&mut b4).map_err(InferenceError::Io)?;
    let ver = u32::from_le_bytes(b4);
    if ver != 1 {
        return Err(InferenceError::InvalidSafetensors(format!(
            "unsupported .w3 file version: {ver}"
        )));
    }

    f.read_exact(&mut b4).map_err(InferenceError::Io)?;
    let ndim = u32::from_le_bytes(b4) as usize;
    checked_alloc_bytes(ndim, 8, file_len, "shape dims")?;
    let mut shape = Vec::with_capacity(ndim);
    let mut b8 = [0u8; 8];
    for _ in 0..ndim {
        f.read_exact(&mut b8).map_err(InferenceError::Io)?;
        shape.push(u64::from_le_bytes(b8) as usize);
    }

    f.read_exact(&mut b8).map_err(InferenceError::Io)?;
    let original_len = u64::from_le_bytes(b8) as usize;

    // payload_offset = 4 (magic) + 4 (version) + 4 (ndim) + ndim*8 + 8 (original_len)
    let payload_offset = (20 + ndim * 8) as u64;

    Ok(W3FileHeader {
        shape,
        original_len,
        payload_offset,
    })
}

/// Load a [`W3Tensor`] from a `.w3` file written by [`save_w3_file`].
///
/// # Errors
///
/// Returns [`InferenceError::Io`] on I/O failure;
/// [`InferenceError::InvalidSafetensors`] on unrecognized magic bytes or
/// unsupported version; [`InferenceError::ShapeMismatch`] if the header's
/// shape product disagrees with `original_len`; or
/// [`InferenceError::InvalidInput`] if the header claims more bytes than the
/// file physically contains.
pub fn load_w3_file(path: &std::path::Path) -> Result<W3Tensor, InferenceError> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(InferenceError::Io)?;
    let file_len = f.metadata().map_err(InferenceError::Io)?.len();

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(InferenceError::Io)?;
    if &magic != b"KHW3" {
        return Err(InferenceError::InvalidSafetensors(
            "invalid magic: not a .w3 file".to_string(),
        ));
    }

    let mut b4 = [0u8; 4];
    f.read_exact(&mut b4).map_err(InferenceError::Io)?;
    let ver = u32::from_le_bytes(b4);
    if ver != 1 {
        return Err(InferenceError::InvalidSafetensors(format!(
            "unsupported .w3 file version: {ver}"
        )));
    }

    f.read_exact(&mut b4).map_err(InferenceError::Io)?;
    let ndim = u32::from_le_bytes(b4) as usize;
    checked_alloc_bytes(ndim, 8, file_len, "shape dims")?;
    let mut shape = Vec::with_capacity(ndim);
    let mut b8 = [0u8; 8];
    for _ in 0..ndim {
        f.read_exact(&mut b8).map_err(InferenceError::Io)?;
        shape.push(u64::from_le_bytes(b8) as usize);
    }

    f.read_exact(&mut b8).map_err(InferenceError::Io)?;
    let original_len = u64::from_le_bytes(b8) as usize;

    let shape_product = checked_shape_matches_data_len(&shape, original_len)?;
    let _ = shape_product;

    let n_blocks = original_len.div_ceil(W3_GROUP_SIZE);
    let raw_len = checked_alloc_bytes(n_blocks, W3_BLOCK_SIZE, file_len, "block payload")?;
    let mut raw = vec![0u8; raw_len];
    f.read_exact(&mut raw).map_err(InferenceError::Io)?;

    let blocks: Vec<W3Block> = raw
        .chunks_exact(W3_BLOCK_SIZE)
        .map(|c| W3Block {
            scale: u16::from_ne_bytes([c[0], c[1]]),
            bias: u16::from_ne_bytes([c[2], c[3]]),
            packed: c[4..16]
                .try_into()
                .expect("chunk slice is exactly W3_PACKED_BYTES"),
        })
        .collect();

    Ok(W3Tensor {
        blocks,
        shape,
        original_len,
    })
}

// ---------------------------------------------------------------------------
// MLP tensor selection
// ---------------------------------------------------------------------------

/// Returns `true` only for dense MLP `gate_proj`, `up_proj`, and `down_proj`
/// weight tensors — the sole W3 quantization target per issue #420.
///
/// Returns `false` for MoE routed-expert tensors (`mlp.experts.*`), MoE
/// shared-expert tensors (`mlp.shared_expert.*`), attention projections, GDN
/// projections, embeddings, `lm_head`, and any non-`.weight` tensor. Fails
/// closed: an unrecognized or ambiguous name is never classified as W3 MLP.
pub fn is_w3_mlp_tensor_name(name: &str) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    if name.contains(".experts.") || name.contains("shared_expert") {
        return false;
    }
    if !name.contains(".mlp.") {
        return false;
    }
    name.ends_with("mlp.gate_proj.weight")
        || name.ends_with("mlp.up_proj.weight")
        || name.ends_with("mlp.down_proj.weight")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_w3_block_size_and_offset() {
        assert_eq!(std::mem::size_of::<W3Block>(), 16);
        let b = W3Block {
            scale: 0,
            bias: 0,
            packed: [0u8; W3_PACKED_BYTES],
        };
        let base = std::ptr::from_ref(&b) as usize;
        let packed_off = std::ptr::from_ref(&b.packed) as usize - base;
        assert_eq!(
            packed_off, 4,
            "packed field must start at byte offset 4 (after scale + bias, no padding)"
        );
    }

    #[test]
    fn test_pack_known_codes_matches_design_byte_layout() {
        // Codes 0..7 repeated four times (32 values), matching the design
        // doc's worked example enough to hand-verify the first bytes.
        // q0=0, q1=0, q2=7 (all zero except q2) exercises the byte0/byte1
        // straddle described in the module doc comment.
        let mut codes = [0u8; 32];
        codes[2] = 7; // 0b111, offset=6: bits 6,7 of byte0 + bit0 of byte1
        let packed = pack_w3_codes(&codes);
        // q2 low 2 bits (0b11) at bits 6-7 of byte0 -> byte0 = 0b1100_0000 = 0xC0
        assert_eq!(
            packed[0], 0xC0,
            "byte0 should hold q2's low 2 bits at bits 6-7"
        );
        // q2 high bit (0b1) at bit0 of byte1 -> byte1 = 0x01
        assert_eq!(packed[1], 0x01, "byte1 should hold q2's high bit at bit0");

        let back = unpack_w3_codes(&packed);
        assert_eq!(back, codes);
    }

    #[test]
    fn test_pack_unpack_all_codes_roundtrip() {
        // Every code 0..=7 in sequence, repeated to fill 32 slots.
        let mut codes = [0u8; 32];
        for (i, c) in codes.iter_mut().enumerate() {
            *c = (i % 8) as u8;
        }
        let packed = pack_w3_codes(&codes);
        let back = unpack_w3_codes(&packed);
        assert_eq!(back, codes, "3-bit pack/unpack must round-trip exactly");
    }

    #[test]
    fn test_pack_max_codes_all_seven() {
        let codes = [7u8; 32];
        let packed = pack_w3_codes(&codes);
        // All bits set: 32*3 = 96 bits = 12 bytes, all 0xFF.
        assert_eq!(packed, [0xFFu8; W3_PACKED_BYTES]);
        let back = unpack_w3_codes(&packed);
        assert_eq!(back, codes);
    }

    #[test]
    fn test_pack_zero_codes() {
        let codes = [0u8; 32];
        let packed = pack_w3_codes(&codes);
        assert_eq!(packed, [0u8; W3_PACKED_BYTES]);
    }

    #[test]
    fn test_quantize_dequantize_zeros() {
        let data = quantize_row_w3(&vec![0.0f32; 64]).unwrap();
        let out = dequantize_row_w3(&data, 64);
        assert_eq!(out.len(), 64);
        for v in &out {
            assert!(v.abs() < 1e-6, "expected ~0, got {v}");
        }
    }

    #[test]
    fn test_quantize_dequantize_monotonic_range_half_step_tolerance() {
        // Range [-7, 7] -> scale = 14/7 = 2.0, max quantization error = 1.0 (half step).
        let src: Vec<f32> = (0..32).map(|i| (i as f32 / 31.0) * 14.0 - 7.0).collect();
        let data = quantize_row_w3(&src).unwrap();
        let out = dequantize_row_w3(&data, 32);
        let max_err = src
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err <= 1.01,
            "max abs error {max_err:.4} > 1.01 for monotonic range (half-step = 1.0)"
        );
    }

    #[test]
    fn test_quantize_row_w3_single_block_size() {
        let src = vec![1.0f32; 32];
        let data = quantize_row_w3(&src).unwrap();
        assert_eq!(data.len(), W3_BLOCK_SIZE, "single block must be 16 bytes");
    }

    #[test]
    fn test_quantize_row_w3_multiple_blocks() {
        let src: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 10.0).collect();
        let data = quantize_row_w3(&src).unwrap();
        assert_eq!(data.len(), 4 * W3_BLOCK_SIZE, "4 blocks must be 64 bytes");
        let out = dequantize_row_w3(&data, 128);
        assert_eq!(out.len(), 128);
    }

    #[test]
    fn test_max_min_map_to_codes_7_and_0() {
        let mut src = vec![0.0f32; 32];
        src[0] = 7.0;
        src[1] = -7.0;
        let data = quantize_row_w3(&src).unwrap();
        let packed: [u8; W3_PACKED_BYTES] = data[4..16].try_into().unwrap();
        let codes = unpack_w3_codes(&packed);
        assert_eq!(codes[0], 7, "max value should map to code 7");
        assert_eq!(codes[1], 0, "min value should map to code 0");
    }

    #[test]
    fn test_reject_nan() {
        let mut vals = vec![1.0f32; 32];
        vals[7] = f32::NAN;
        let result = quantize_row_w3(&vals);
        assert!(result.is_err(), "NaN must be rejected");
    }

    #[test]
    fn test_reject_infinity() {
        let mut vals = vec![1.0f32; 32];
        vals[15] = f32::INFINITY;
        let result = quantize_row_w3(&vals);
        assert!(result.is_err(), "+inf must be rejected");
    }

    #[test]
    fn test_reject_neg_infinity() {
        let mut vals = vec![1.0f32; 32];
        vals[3] = f32::NEG_INFINITY;
        let result = quantize_row_w3(&vals);
        assert!(result.is_err(), "-inf must be rejected");
    }

    #[test]
    fn test_dequantize_row_w3_truncates_to_n_weights() {
        let src = vec![2.0f32; 64];
        let data = quantize_row_w3(&src).unwrap();
        let out = dequantize_row_w3(&data, 40);
        assert_eq!(out.len(), 40, "must truncate to requested n_weights");
    }

    #[test]
    fn test_dequantize_row_w3_does_not_panic_on_partial_trailing_bytes() {
        let src = vec![1.0f32; 32];
        let mut data = quantize_row_w3(&src).unwrap();
        data.extend_from_slice(&[0xDE, 0xAD, 0xBE]); // 3 trailing garbage bytes
        let out = dequantize_row_w3(&data, 32);
        assert_eq!(
            out.len(),
            32,
            "trailing partial block must be ignored, not panic"
        );
    }

    #[test]
    fn test_dequantize_row_w3_empty_on_short_input() {
        let out = dequantize_row_w3(&[0xABu8; 5], 32);
        assert!(
            out.is_empty(),
            "input shorter than one block must yield empty output"
        );
    }

    #[test]
    fn test_quantize_tensor_w3_shape_mismatch_rejected() {
        let src = vec![0.0f32; 64];
        let result = quantize_tensor_w3(&src, 3, 32); // 96 != 64
        assert!(result.is_err(), "rows*cols mismatch must be rejected");
    }

    #[test]
    fn test_quantize_tensor_w3_rows() {
        let rows = 4usize;
        let cols = 64usize;
        let src: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 - 128.0) / 20.0)
            .collect();
        let data = quantize_tensor_w3(&src, rows, cols).unwrap();
        let blocks_per_row = cols.div_ceil(W3_GROUP_SIZE); // 2 blocks per row
        assert_eq!(data.len(), rows * blocks_per_row * W3_BLOCK_SIZE);
    }

    #[test]
    fn test_quantize_bf16_to_w3_rejects_shape_mismatch() {
        let data: Vec<u16> = (0..64).map(|i| i as u16).collect();
        let result = quantize_bf16_to_w3(&data, &[3, 32]); // 96 != 64
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_f32_to_w3_rejects_shape_product_overflow() {
        let data = vec![0.0f32; 32];
        let result = quantize_f32_to_w3(&data, &[usize::MAX, 2]);
        assert!(result.is_err(), "shape product overflow must be rejected");
    }

    #[test]
    fn test_quantize_bf16_to_w3_and_dequantize_roundtrip() {
        let f32_vals: Vec<f32> = (0..32).map(|i| i as f32 * 7.0 / 31.0).collect();
        let bf16_vals: Vec<u16> = f32_vals
            .iter()
            .map(|&v| (v.to_bits() >> 16) as u16)
            .collect();
        let tensor = quantize_bf16_to_w3(&bf16_vals, &[32]).unwrap();
        assert_eq!(tensor.shape, vec![32]);
        assert_eq!(tensor.original_len, 32);
        let out = dequantize_w3_to_f32(&tensor);
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn test_quantize_f32_to_w3_shape_preserved() {
        let shape = vec![4usize, 8, 4]; // 128 elements
        let data = vec![0.0f32; 128];
        let tensor = quantize_f32_to_w3(&data, &shape).unwrap();
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.original_len, 128);
        assert_eq!(tensor.blocks.len(), 4); // 128 / 32 = 4
    }

    #[test]
    fn test_save_load_round_trip() {
        let f32_vals: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 4.0).collect();
        let original = quantize_f32_to_w3(&f32_vals, &[8, 8]).unwrap();
        let path = std::path::PathBuf::from("/tmp/test_w3_round_trip.w3");
        save_w3_file(&path, &original).unwrap();
        let loaded = load_w3_file(&path).unwrap();
        assert_eq!(loaded.shape, original.shape);
        assert_eq!(loaded.original_len, original.original_len);
        assert_eq!(loaded.blocks.len(), original.blocks.len());
        for (a, b) in original.blocks.iter().zip(&loaded.blocks) {
            assert_eq!(a.scale, b.scale);
            assert_eq!(a.bias, b.bias);
            assert_eq!(a.packed, b.packed);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_w3_file_rejects_bad_magic() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"XXXX");
        buf.extend_from_slice(&1u32.to_le_bytes());
        let path = std::path::PathBuf::from("/tmp/test_w3_bad_magic.w3");
        std::fs::write(&path, &buf).unwrap();
        let r = load_w3_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(r.is_err(), "bad magic must be rejected");
    }

    #[test]
    fn test_load_w3_file_rejects_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHW3");
        buf.extend_from_slice(&99u32.to_le_bytes());
        let path = std::path::PathBuf::from("/tmp/test_w3_bad_version.w3");
        std::fs::write(&path, &buf).unwrap();
        let r = load_w3_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(r.is_err(), "unsupported version must be rejected");
    }

    #[test]
    fn test_load_w3_file_rejects_shape_product_mismatch() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHW3");
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u32.to_le_bytes()); // ndim
        buf.extend_from_slice(&4u64.to_le_bytes()); // shape[0]
        buf.extend_from_slice(&16u64.to_le_bytes()); // shape[1] -> product 64
        buf.extend_from_slice(&32u64.to_le_bytes()); // original_len (!= 64)
        buf.extend_from_slice(&[0u8; W3_BLOCK_SIZE]); // one valid-size block
        let path = std::path::PathBuf::from("/tmp/test_w3_shape_mismatch.w3");
        std::fs::write(&path, &buf).unwrap();
        let r = load_w3_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            r.is_err(),
            "shape product 64 != original_len 32 must be rejected"
        );
    }

    #[test]
    fn test_load_w3_file_rejects_truncated_payload() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHW3");
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
        buf.extend_from_slice(&64u64.to_le_bytes()); // shape[0] = 64
        buf.extend_from_slice(&64u64.to_le_bytes()); // original_len = 64 -> needs 2 blocks = 32 bytes
        buf.extend_from_slice(&[0u8; 10]); // far short of 32 bytes
        let path = std::path::PathBuf::from("/tmp/test_w3_truncated.w3");
        std::fs::write(&path, &buf).unwrap();
        let r = load_w3_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(r.is_err(), "truncated payload must be rejected");
    }

    #[test]
    fn test_read_w3_header_rejects_huge_ndim() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHW3");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        let path = std::path::PathBuf::from("/tmp/test_w3_huge_ndim.w3");
        std::fs::write(&path, &buf).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let r = read_w3_header(&file);
        std::fs::remove_file(&path).ok();
        assert!(r.is_err(), "u32::MAX ndim must be rejected");
    }

    #[test]
    fn test_read_w3_header_payload_offset() {
        let f32_vals: Vec<f32> = vec![0.0f32; 64];
        let tensor = quantize_f32_to_w3(&f32_vals, &[2, 32]).unwrap();
        let path = std::path::PathBuf::from("/tmp/test_w3_header_offset.w3");
        save_w3_file(&path, &tensor).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let header = read_w3_header(&file).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(header.shape, vec![2, 32]);
        assert_eq!(header.original_len, 64);
        // 20 + ndim(2)*8 = 36
        assert_eq!(header.payload_offset, 36);
    }

    // -----------------------------------------------------------------------
    // is_w3_mlp_tensor_name
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_w3_mlp_tensor_name_matches_dense_mlp() {
        assert!(is_w3_mlp_tensor_name(
            "model.language_model.layers.0.mlp.gate_proj.weight"
        ));
        assert!(is_w3_mlp_tensor_name(
            "model.language_model.layers.5.mlp.up_proj.weight"
        ));
        assert!(is_w3_mlp_tensor_name(
            "model.language_model.layers.12.mlp.down_proj.weight"
        ));
    }

    #[test]
    fn test_is_w3_mlp_tensor_name_rejects_moe_experts() {
        assert!(!is_w3_mlp_tensor_name(
            "model.language_model.layers.0.mlp.experts.gate_proj.weight"
        ));
        assert!(!is_w3_mlp_tensor_name(
            "model.language_model.layers.0.mlp.experts.down_proj.weight"
        ));
    }

    #[test]
    fn test_is_w3_mlp_tensor_name_rejects_shared_expert() {
        assert!(!is_w3_mlp_tensor_name(
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"
        ));
        assert!(!is_w3_mlp_tensor_name(
            "model.language_model.layers.0.mlp.shared_expert_gate.weight"
        ));
    }

    #[test]
    fn test_is_w3_mlp_tensor_name_rejects_attention_and_gdn() {
        assert!(!is_w3_mlp_tensor_name(
            "model.language_model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(!is_w3_mlp_tensor_name(
            "model.language_model.layers.0.self_attn.o_proj.weight"
        ));
        assert!(!is_w3_mlp_tensor_name(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        ));
        assert!(!is_w3_mlp_tensor_name(
            "model.language_model.layers.0.linear_attn.in_proj_z.weight"
        ));
    }

    #[test]
    fn test_is_w3_mlp_tensor_name_rejects_embed_and_lm_head() {
        assert!(!is_w3_mlp_tensor_name(
            "model.language_model.embed_tokens.weight"
        ));
        assert!(!is_w3_mlp_tensor_name("lm_head.weight"));
    }

    #[test]
    fn test_is_w3_mlp_tensor_name_rejects_norms_and_non_weight() {
        assert!(!is_w3_mlp_tensor_name(
            "model.language_model.layers.0.mlp.norm.weight"
        ));
        assert!(!is_w3_mlp_tensor_name(
            "model.language_model.layers.0.mlp.gate_proj.bias"
        ));
    }
}
