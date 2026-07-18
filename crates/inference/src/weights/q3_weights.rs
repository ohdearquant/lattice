//! Q3 per-block weight quantization for MLP projections (ADR-072 P1, #420).
//!
//! ## Why 3-bit, MLP-only
//!
//! Decode is weight-bandwidth-bound; MLP GEMMs (gate/up/down) are the single
//! largest weight-bandwidth group in a Qwen3.5 decode profile (~35%). Cutting
//! MLP weight bytes from Q4's 5.0 bpw to a 3-bit tier at **4.0 bpw** is a ~20%
//! traffic cut on that group. Apple GPUs have no low-precision matrix unit, so
//! W3 is a storage/bandwidth play: blocks dequantize to f16 for compute, they
//! do not lower compute cost. Attention and GDN stay Q4/Q8 (role-aware
//! precision is a separate lever, #423) — this module only defines the format.
//!
//! ## Format (`Q3Block`, group-32, 16 bytes = 4.0 bpw)
//!
//! Every 32 consecutive weights are packed into one [`Q3Block`] of 16 bytes:
//! - `scale: u16` — per-block scale, stored as an IEEE-754 f16 bit pattern
//! - `bias: u16`  — per-block bias (zero-point), stored as an IEEE-754 f16 bit pattern
//! - `packed: [u8; 12]` — 32 × 3-bit values in **plane-split 2+1** layout
//!
//! ### Plane-split 2+1 packing (NOT a dense bit-stream)
//!
//! 3-bit values do not byte-align. Rather than a dense 96-bit stream (where a
//! value straddles a byte boundary at a non-aligned offset — GPU-hostile), each
//! 3-bit value is split into a low 2-bit plane and a high 1-bit plane, each
//! byte-aligned and independently addressable:
//!
//! ```text
//! packed[0..8]  low-2-bit plane : byte[b] = q2[4b] | q2[4b+1]<<2 | q2[4b+2]<<4 | q2[4b+3]<<6
//! packed[8..12] high-1-bit plane: byte[8+b] = Σ_{k∈0..8} hi[8b+k] << k
//! dequant of value i: q[i] = low2(i) | (hi(i) << 2)
//!   low2(i) = (packed[i/4]     >> ((i%4)*2)) & 0x3
//!   hi(i)   = (packed[8 + i/8] >>  (i%8))    & 0x1
//! ```
//!
//! Both planes are read with aligned loads and constant shifts, no cross-byte
//! straddle — the structure llama.cpp's sub-4-bit K-quants use for the same
//! reason. Signed off as the on-disk layout (design note 2026-07-14).
//!
//! ### Dequantization (both encode modes share this)
//!
//! ```text
//! weight[i] = q[i] as f32 * scale + bias
//! ```
//!
//! ### Encode modes (same on-disk layout)
//!
//! - **Asymmetric** (default, 3-bit → 8 levels): `scale = (max - min) / 7`,
//!   `bias = min`, `q[i] = clamp(round((weight[i] - min) / scale), 0, 7)`.
//!   Optimal for raw weights with a non-zero distributional center.
//! - **Symmetric** (Hadamard-rotated, zero-mean weights): `scale = abs_max / 3.5`,
//!   `bias = -4 * scale`, `q[i] = clamp(round(weight[i] / scale) + 4, 0, 7)`, so
//!   the shared dequant reduces to `(q - 4) * scale`.
//!
//! ## File format (`.q3`)
//!
//! ```text
//! magic        b"KHQ3"               4 bytes
//! version      1u32 LE               4 bytes
//! ndim         u32 LE                4 bytes
//! shape[i]     u64 LE × ndim
//! original_len u64 LE                8 bytes
//! blocks       [Q3Block; n_blocks]   n_blocks × 16 bytes
//! ```

// Q3 quantization operates on raw byte/u16 slices; unsafe is limited to the
// two transmute-equivalent slice casts in quantize_row_q3_0 and save_q3_file.
#![allow(clippy::cast_possible_truncation)]

use crate::error::InferenceError;

/// One Q3 quantization block: 32 weights packed as 3-bit unsigned integers.
///
/// `scale`/`bias` are stored as raw IEEE-754 f16 bit patterns in `u16` — the
/// `half` crate is not a dependency of `lattice-inference`. Use
/// [`q3_f32_to_f16`] / [`q3_f16_to_f32`].
///
/// `packed` holds 32 3-bit values in **plane-split 2+1** layout (see the module
/// docs): a byte-aligned low-2-bit plane in `packed[0..8]` and a byte-aligned
/// high-1-bit plane in `packed[8..12]`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q3Block {
    /// f16 bit pattern for the per-block scale — 2 bytes.
    pub scale: u16,
    /// f16 bit pattern for the per-block bias (zero-point) — 2 bytes.
    /// Dequantization: `weight = q * scale + bias`.
    pub bias: u16,
    /// 32 × 3-bit values packed as 12 bytes in plane-split 2+1 layout.
    pub packed: [u8; 12],
}

// Compile-time size assertion — must be exactly 16 bytes (2 + 2 + 12, no padding).
const _: () = assert!(std::mem::size_of::<Q3Block>() == 16);

/// Serialized size in bytes of one [`Q3Block`] (2 scale + 2 bias + 12 packed = 16).
///
/// Derive block-byte accounting from this constant rather than a magic literal so
/// it can never drift from the struct layout asserted above.
pub const Q3_BLOCK_BYTES: usize = std::mem::size_of::<Q3Block>();

/// A Q3 quantized tensor.
///
/// Stores blocks, shape metadata, and the count of valid original weights (the
/// last block may be padded with zeros if `original_len` is not a multiple of 32).
#[derive(Debug, Clone)]
pub struct Q3Tensor {
    /// Quantized blocks, each covering 32 weights.
    pub blocks: Vec<Q3Block>,
    /// Original tensor shape (e.g., `[rows, cols]` for a 2-D weight matrix).
    pub shape: Vec<usize>,
    /// Number of valid original weights — may be less than `blocks.len() * 32`.
    pub original_len: usize,
}

// ---------------------------------------------------------------------------
// f16 ↔ f32 / bf16 → f32 helpers.
//
// Thin wrappers over the single always-compiled scalar decoder in
// `crate::weights::half_bits` (lattice#799) — kept as separate `q3_`-prefixed
// functions here only to preserve this module's call-site names; no conversion
// arithmetic lives in this file.
// ---------------------------------------------------------------------------

/// Convert `f32` to IEEE-754 half-precision stored as a `u16` bit pattern.
#[inline]
pub(crate) fn q3_f32_to_f16(x: f32) -> u16 {
    crate::weights::half_bits::f32_to_f16_bits(x)
}

/// Convert an IEEE-754 f16 bit pattern (`u16`) back to `f32`.
#[inline]
pub(crate) fn q3_f16_to_f32(bits: u16) -> f32 {
    crate::weights::half_bits::f16_bits_to_f32(bits)
}

/// Convert a BF16 bit pattern (`u16`) to `f32`.
#[inline]
fn bf16_to_f32(v: u16) -> f32 {
    crate::weights::half_bits::bf16_bits_to_f32(v)
}

// ---------------------------------------------------------------------------
// Plane-split 2+1 pack / unpack
// ---------------------------------------------------------------------------

/// Pack 32 3-bit values (each already clamped to `0..=7`) into the plane-split
/// 2+1 layout (12 bytes). See the module docs for the bit layout.
#[inline]
fn pack_plane_split(q: &[u8; 32]) -> [u8; 12] {
    let mut packed = [0u8; 12];
    // low-2-bit plane: bytes 0..8, four values per byte.
    for b in 0..8 {
        packed[b] = (q[4 * b] & 0x3)
            | ((q[4 * b + 1] & 0x3) << 2)
            | ((q[4 * b + 2] & 0x3) << 4)
            | ((q[4 * b + 3] & 0x3) << 6);
    }
    // high-1-bit plane: bytes 8..12, eight values per byte.
    for b in 0..4 {
        let mut byte = 0u8;
        for k in 0..8 {
            byte |= ((q[8 * b + k] >> 2) & 0x1) << k;
        }
        packed[8 + b] = byte;
    }
    packed
}

/// Unpack the 3-bit value at index `i` (`0..32`) from a plane-split 12-byte block.
#[inline]
fn unpack_plane_split(packed: &[u8], i: usize) -> u8 {
    let low2 = (packed[i / 4] >> ((i % 4) * 2)) & 0x3;
    let hi = (packed[8 + i / 8] >> (i % 8)) & 0x1;
    low2 | (hi << 2)
}

// ---------------------------------------------------------------------------
// Core block quantization
// ---------------------------------------------------------------------------

/// Quantize one block from only the first `valid_len` real values of `vals`;
/// the remaining `32 - valid_len` slots are caller-supplied zero padding used
/// solely to fill the fixed-size packing loop.
///
/// Asymmetric mode derives `min_val`/`max_val` from the real `valid_len`
/// elements only, so a zero-padded tail block gets the same scale resolution a
/// full block would (padding zeros must never widen the range). Symmetric mode
/// folds `abs_max` over the full padded array unconditionally: zero padding can
/// never exceed a non-empty real element's absolute value, so the result is
/// bit-identical to folding over the real elements alone.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if `valid_len` is not in `1..=32`,
/// or if any element of `vals` is non-finite. IEEE-754 `NaN > x` is always
/// false; a NaN silently leaves `abs_max`/`min_val`/`max_val` unchanged,
/// yielding wrong-but-no-error quantization. Rejecting here points the error at
/// the source weight rather than a downstream matmul.
#[inline]
fn quantize_block_with_mode_len(
    vals: &[f32; 32],
    valid_len: usize,
    symmetric: bool,
) -> Result<Q3Block, InferenceError> {
    if !(1..=32).contains(&valid_len) {
        return Err(InferenceError::InvalidInput(format!(
            "Q3 weight block valid_len {valid_len} must be in 1..=32"
        )));
    }

    for (i, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            return Err(InferenceError::InvalidInput(format!(
                "Q3 weight block element {i} contains a non-finite value ({v}); \
                 source weights must be finite"
            )));
        }
    }

    if symmetric {
        let abs_max = vals.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max == 0.0 {
            1.0f32
        } else {
            abs_max / 3.5
        };
        let inv_scale = 1.0 / scale;
        let bias = -4.0 * scale;
        let mut q = [0u8; 32];
        for (i, slot) in q.iter_mut().enumerate() {
            *slot = ((vals[i] * inv_scale).round() + 4.0).clamp(0.0, 7.0) as u8;
        }
        Ok(Q3Block {
            scale: q3_f32_to_f16(scale),
            bias: q3_f32_to_f16(bias),
            packed: pack_plane_split(&q),
        })
    } else {
        let real = &vals[..valid_len];
        let min_val = real.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = real.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range == 0.0 { 1.0f32 } else { range / 7.0 };
        let inv_scale = 1.0 / scale;
        let mut q = [0u8; 32];
        for (i, slot) in q.iter_mut().enumerate() {
            *slot = (((vals[i] - min_val) * inv_scale).round()).clamp(0.0, 7.0) as u8;
        }
        Ok(Q3Block {
            scale: q3_f32_to_f16(scale),
            bias: q3_f32_to_f16(min_val),
            packed: pack_plane_split(&q),
        })
    }
}

// ---------------------------------------------------------------------------
// Public quantization API
// ---------------------------------------------------------------------------

/// Quantize a slice of f32 values into Q3 blocks (asymmetric mode).
///
/// The input is processed 32 elements at a time; the last block is zero-padded
/// if `src.len()` is not a multiple of 32. Returns raw bytes containing
/// tightly-packed [`Q3Block`]s (16 bytes each).
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if any value in `src` is non-finite.
pub fn quantize_row_q3_0(src: &[f32]) -> Result<Vec<u8>, InferenceError> {
    let n_blocks = src.len().div_ceil(32);
    let mut out = Vec::with_capacity(n_blocks * Q3_BLOCK_BYTES);
    for chunk in src.chunks(32) {
        let mut vals = [0.0f32; 32];
        vals[..chunk.len()].copy_from_slice(chunk);
        let block = quantize_block_with_mode_len(&vals, chunk.len(), false)?;
        // SAFETY: Q3Block is #[repr(C)] with size 16; its alignment is 2 (the
        // alignment of the leading `scale: u16` per the Rust Reference's repr(C)
        // rule). Casting to `&[u8; 16]` is valid because the target element type
        // is `u8` (alignment 1 ≤ source alignment 2) and the source byte length
        // matches the destination length exactly.
        let bytes: &[u8; Q3_BLOCK_BYTES] = unsafe { &*std::ptr::from_ref(&block).cast() };
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

/// Dequantize Q3 blocks (raw bytes) back to f32 values.
///
/// Trailing bytes beyond the last complete 16-byte block are silently ignored;
/// the function returns `min(n_weights, (data.len() / 16) * 32)` values. It
/// never panics regardless of input length — inputs shorter than 16 bytes
/// return an empty `Vec`.
pub fn dequantize_row_q3_0(data: &[u8], n_weights: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_weights);
    for chunk in data.chunks_exact(Q3_BLOCK_BYTES) {
        let scale = q3_f16_to_f32(u16::from_ne_bytes([chunk[0], chunk[1]]));
        let bias = q3_f16_to_f32(u16::from_ne_bytes([chunk[2], chunk[3]]));
        let packed = &chunk[4..Q3_BLOCK_BYTES];
        for i in 0..32 {
            out.push(unpack_plane_split(packed, i) as f32 * scale + bias);
        }
    }
    out.truncate(n_weights);
    out
}

/// Quantize a row-major f32 tensor into Q3 blocks, one row at a time.
///
/// `src` has shape `[rows, cols]`. Each row is quantized independently into
/// `cols.div_ceil(32)` blocks. Returns raw bytes (16 bytes per block).
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if any value in `src` is non-finite.
pub fn quantize_tensor_q3_0(
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
    let mut out = Vec::with_capacity(rows * blocks_per_row * Q3_BLOCK_BYTES);
    for row_idx in 0..rows {
        let row = &src[row_idx * cols..(row_idx + 1) * cols];
        out.extend_from_slice(&quantize_row_q3_0(row)?);
    }
    Ok(out)
}

/// Assert that `shape.iter().product()` equals `data_len`.
///
/// Mirrors `q4_weights`'s guard: a [`Q3Tensor`] whose `shape` disagrees with its
/// element count would write inconsistent metadata into a `.q3` header that
/// downstream loaders trust without re-verification. Uses `checked_mul` so a
/// malformed shape surfaces as a panic at construction, not a wraparound.
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

/// Quantize a BF16 tensor (raw `u16` slice) into a [`Q3Tensor`] (asymmetric mode).
///
/// Panics if `shape.iter().product()` does not equal `data.len()`.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if any BF16 value decodes to a
/// non-finite f32.
pub fn quantize_bf16_to_q3(data: &[u16], shape: &[usize]) -> Result<Q3Tensor, InferenceError> {
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

    Ok(Q3Tensor {
        blocks,
        shape: shape.to_vec(),
        original_len,
    })
}

/// Quantize an `f32` tensor into a [`Q3Tensor`] (asymmetric mode).
///
/// Prefer this over [`quantize_bf16_to_q3`] when the source is higher precision
/// than BF16, so per-block `min`/`max` are computed from the real precision
/// rather than from BF16-truncated values.
///
/// Panics if `shape.iter().product()` does not equal `data.len()`.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if any value in `data` is non-finite.
pub fn quantize_f32_to_q3(data: &[f32], shape: &[usize]) -> Result<Q3Tensor, InferenceError> {
    assert_shape_matches_data_len(shape, data.len());
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(32);
    let mut blocks = Vec::with_capacity(n_blocks);

    for chunk in data.chunks(32) {
        let mut vals = [0.0f32; 32];
        vals[..chunk.len()].copy_from_slice(chunk);
        blocks.push(quantize_block_with_mode_len(&vals, chunk.len(), false)?);
    }

    Ok(Q3Tensor {
        blocks,
        shape: shape.to_vec(),
        original_len,
    })
}

/// Quantize an `f32` tensor with explicit symmetry mode.
///
/// `symmetric=true` is required for Hadamard-rotated tensors (zero-mean); it
/// wastes no bits on an approximately-zero bias. Use `false` for raw weights
/// with a non-zero distributional center.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if any value in `data` is non-finite.
pub fn quantize_f32_to_q3_mode(
    data: &[f32],
    shape: &[usize],
    symmetric: bool,
) -> Result<Q3Tensor, InferenceError> {
    assert_shape_matches_data_len(shape, data.len());
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(32);
    let mut blocks = Vec::with_capacity(n_blocks);

    for chunk in data.chunks(32) {
        let mut vals = [0.0f32; 32];
        vals[..chunk.len()].copy_from_slice(chunk);
        blocks.push(quantize_block_with_mode_len(&vals, chunk.len(), symmetric)?);
    }

    Ok(Q3Tensor {
        blocks,
        shape: shape.to_vec(),
        original_len,
    })
}

/// Dequantize all blocks of a [`Q3Tensor`] back to f32.
///
/// Output length equals `tensor.original_len` (zero-padded tail blocks are truncated).
pub fn dequantize_q3_to_f32(tensor: &Q3Tensor) -> Vec<f32> {
    let mut out = Vec::with_capacity(tensor.original_len);
    for block in &tensor.blocks {
        let scale = q3_f16_to_f32(block.scale);
        let bias = q3_f16_to_f32(block.bias);
        for i in 0..32 {
            out.push(unpack_plane_split(&block.packed, i) as f32 * scale + bias);
        }
    }
    out.truncate(tensor.original_len);
    out
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Write a [`Q3Tensor`] to a `.q3` file. See the module docs for the layout.
pub fn save_q3_file(path: &std::path::Path, tensor: &Q3Tensor) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"KHQ3")?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;
    for &dim in &tensor.shape {
        f.write_all(&(dim as u64).to_le_bytes())?;
    }
    f.write_all(&(tensor.original_len as u64).to_le_bytes())?;
    // SAFETY: Q3Block is #[repr(C)] with size 16; its alignment is 2 (the
    // alignment of the leading `scale: u16` per the Rust Reference's repr(C)
    // rule). Casting to a `&[u8]` is valid because the target element type is
    // `u8` (alignment 1 ≤ source alignment 2). The resulting slice has length
    // `blocks.len() * 16` matching the source contiguous storage.
    let block_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            tensor.blocks.as_ptr().cast::<u8>(),
            tensor.blocks.len() * Q3_BLOCK_BYTES,
        )
    };
    f.write_all(block_bytes)
}

/// Restrict Q3 loading to MLP gate/up/down projections (ADR-072 P1 scope).
///
/// The W3 format is a decode-bandwidth play for the MLP GEMM group only —
/// attention and GDN stay Q4/Q8 (role-aware precision, #423). Any tensor name
/// outside the MLP projection set is rejected closed rather than silently
/// accepted, so a mixed checkpoint can never route an attention/GDN weight
/// through the 3-bit path by mistake.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if `tensor_name` does not name an
/// MLP `gate_proj` / `up_proj` / `down_proj` / fused `gate_up_proj` tensor.
pub fn validate_q3_mlp_role(tensor_name: &str) -> Result<(), InferenceError> {
    const MLP_PROJ_SUFFIXES: [&str; 4] = [
        ".mlp.gate_proj",
        ".mlp.up_proj",
        ".mlp.down_proj",
        ".mlp.gate_up_proj",
    ];
    // Real checkpoints (safetensors index + the Q4 loader's call sites, e.g.
    // `{prefix}.mlp.gate_proj.weight`) always carry a terminal `.weight`; a
    // name without it — a LoRA adapter (`...gate_proj.lora_A.weight`), a
    // renamed backup (`...up_proj_backup`), or a trailing-suffix typo
    // (`...down_proj.weight.extra`) — must fail closed rather than match on
    // substring alone.
    let is_mlp_proj = tensor_name.strip_suffix(".weight").is_some_and(|stem| {
        MLP_PROJ_SUFFIXES
            .iter()
            .any(|suffix| stem.ends_with(suffix))
    });
    if is_mlp_proj {
        Ok(())
    } else {
        Err(InferenceError::InvalidInput(format!(
            "Q3 weight format is restricted to MLP gate/up/down projections \
             (ADR-072 P1); tensor '{tensor_name}' is outside that set and must \
             not be loaded as Q3"
        )))
    }
}

/// Header metadata returned by [`read_q3_header`] without allocating blocks.
pub struct Q3FileHeader {
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Number of valid original weights.
    pub original_len: usize,
    /// Byte offset in the file where the `Q3Block` payload starts.
    pub payload_offset: u64,
}

/// Validate a header-declared element count before allocating a buffer for it.
///
/// Custom `.q3` files carry untrusted `ndim`/`original_len` fields straight from
/// disk. Without this guard a crafted header can overflow a `count * elem_size`
/// multiply (silent wrong-sized buffer in release) or request an allocation far
/// larger than the file (OOM abort). A legitimate payload is physically present
/// in the file, so bounding by `file_len` rejects only adversarial over-claims.
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

/// Parse the header of a `.q3` file without reading the block payload.
///
/// On return the file cursor is positioned at the start of the block data.
///
/// # Errors
///
/// Returns an error on I/O failure, unrecognized magic bytes, or unsupported version.
pub fn read_q3_header(
    file: &mut std::fs::File,
) -> Result<Q3FileHeader, Box<dyn std::error::Error>> {
    use std::io::{Read, Seek, SeekFrom};
    let file_len = file.metadata()?.len();

    let (shape, original_len, payload_offset) = {
        let mut f = std::io::BufReader::new(&mut *file);

        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"KHQ3" {
            return Err("invalid magic: not a .q3 file".into());
        }

        let mut b4 = [0u8; 4];
        f.read_exact(&mut b4)?;
        let ver = u32::from_le_bytes(b4);
        if ver != 1 {
            return Err(format!("unsupported .q3 file version: {ver}").into());
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

        // payload_offset = 4 (magic) + 4 (version) + 4 (ndim) + ndim*8 + 8 (original_len)
        let payload_offset = (20 + ndim * 8) as u64;

        (shape, original_len, payload_offset)
        // `f` (the BufReader) is dropped here, before the seek below, so the
        // seek acts on the raw `File` and is not undone by buffered lookahead.
    };

    file.seek(SeekFrom::Start(payload_offset))?;

    Ok(Q3FileHeader {
        shape,
        original_len,
        payload_offset,
    })
}

/// Validate a [`Q3FileHeader`] against the file's actual byte length before the
/// payload is handed to a no-copy mmap buffer (the Metal loader never reads the
/// payload itself, so a missing bounds check here would let a truncated
/// on-disk checkpoint reach GPU dispatch and read past the mapped region).
///
/// Mirrors `q4_weights::validate_q4_header_payload_bounds` with the 16-byte
/// Q3 block size in place of Q4's 20.
///
/// # Errors
///
/// Returns an error if `shape.iter().product()` disagrees with `original_len`,
/// or if `file_len` is smaller than `payload_offset + n_blocks * 16`.
///
/// Only called from `mmap_q3_weight` (crates/inference/src/forward/metal_qwen35.rs)
/// today, which live checkpoint loading does not yet reach.
#[allow(dead_code)]
pub(crate) fn validate_q3_header_payload_bounds(
    header: &Q3FileHeader,
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
        .checked_mul(Q3_BLOCK_BYTES)
        .ok_or("Q3 block payload byte count overflows usize")? as u64;
    let required_len = header
        .payload_offset
        .checked_add(payload_bytes)
        .ok_or("Q3 payload end offset overflows u64")?;
    if file_len < required_len {
        return Err(format!(
            "{}: file truncated below Q3 block payload ({file_len} bytes < required {required_len})",
            path.display()
        )
        .into());
    }
    Ok(())
}

/// Load a [`Q3Tensor`] from a `.q3` file written by [`save_q3_file`].
///
/// # Errors
///
/// Returns an error on I/O failure, unrecognized magic bytes, unsupported
/// version, or a header whose shape product disagrees with `original_len`.
pub fn load_q3_file(path: &std::path::Path) -> Result<Q3Tensor, Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let file_len = f.metadata()?.len();

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"KHQ3" {
        return Err("invalid magic: not a .q3 file".into());
    }

    let mut b4 = [0u8; 4];
    f.read_exact(&mut b4)?;
    let ver = u32::from_le_bytes(b4);
    if ver != 1 {
        return Err(format!("unsupported .q3 file version: {ver}").into());
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

    // Fail closed on a header whose shape disagrees with its element count: a
    // tensor whose `shape` overstates the block payload would let downstream
    // matmuls read stale, out-of-range data.
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
    let raw_len = checked_alloc_bytes(n_blocks, Q3_BLOCK_BYTES, file_len, "block payload")?;
    let mut raw = vec![0u8; raw_len];
    f.read_exact(&mut raw)?;

    let blocks: Vec<Q3Block> = raw
        .chunks_exact(Q3_BLOCK_BYTES)
        .map(|c| Q3Block {
            scale: u16::from_ne_bytes([c[0], c[1]]),
            bias: u16::from_ne_bytes([c[2], c[3]]),
            packed: c[4..Q3_BLOCK_BYTES]
                .try_into()
                .expect("slice is exactly 12 bytes"),
        })
        .collect();

    Ok(Q3Tensor {
        blocks,
        shape,
        original_len,
    })
}

// ---------------------------------------------------------------------------
// CPU reference GEMV / GEMM — parity oracle for the Metal Q3 kernels
// ---------------------------------------------------------------------------

/// CPU reference GEMV: `y[n] = sum_k x[k] * dequant(qweight)[n, k]`.
///
/// `qweight` is `N` rows of packed Q3 blocks in the same row-major-per-row
/// layout the Metal kernels read (`row_bytes = (K/32) * 16` per row, produced
/// by calling [`quantize_row_q3_0`] once per output row). This is the
/// straight-line f32 dequant-then-dot reference the Metal `gemv_q3_decode`
/// kernel is checked against; it does no tiling and no reduced precision.
///
/// # Panics
///
/// Panics if `K` is not a multiple of 32, or if `qweight.len()` does not equal
/// `N * (K / 32) * Q3_BLOCK_BYTES`.
pub fn gemv_q3_reference(x: &[f32], qweight: &[u8], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % 32, 0, "K must be divisible by 32 for Q3 GEMV");
    let row_bytes = (k / 32) * Q3_BLOCK_BYTES;
    assert_eq!(
        qweight.len(),
        n * row_bytes,
        "qweight length does not match N * (K/32) * Q3_BLOCK_BYTES"
    );
    let mut y = vec![0.0f32; n];
    for (row_idx, row_bytes_slice) in qweight.chunks_exact(row_bytes).enumerate() {
        let w = dequantize_row_q3_0(row_bytes_slice, k);
        y[row_idx] = x.iter().zip(w.iter()).map(|(&xv, &wv)| xv * wv).sum();
    }
    y
}

/// CPU reference GEMM: `Y[m, n] = sum_k X[m, k] * dequant(qweight)[n, k]`.
///
/// `X` is row-major `[M, K]`, `qweight` is `N` rows of packed Q3 blocks (same
/// per-row layout as [`gemv_q3_reference`]), `Y` is row-major `[M, N]`. The
/// parity oracle for the Metal `gemm_q3_tiled` kernel.
///
/// # Panics
///
/// Panics if `K` is not a multiple of 32, or if `qweight.len()` does not equal
/// `N * (K / 32) * Q3_BLOCK_BYTES`, or if `x.len()` does not equal `M * K`.
pub fn gemm_q3_reference(x: &[f32], qweight: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % 32, 0, "K must be divisible by 32 for Q3 GEMM");
    assert_eq!(x.len(), m * k, "x length does not match M * K");
    let row_bytes = (k / 32) * Q3_BLOCK_BYTES;
    assert_eq!(
        qweight.len(),
        n * row_bytes,
        "qweight length does not match N * (K/32) * Q3_BLOCK_BYTES"
    );
    let w_deq: Vec<Vec<f32>> = qweight
        .chunks_exact(row_bytes)
        .map(|row| dequantize_row_q3_0(row, k))
        .collect();
    let mut y = vec![0.0f32; m * n];
    for mi in 0..m {
        let xrow = &x[mi * k..(mi + 1) * k];
        for ni in 0..n {
            y[mi * n + ni] = xrow
                .iter()
                .zip(w_deq[ni].iter())
                .map(|(&xv, &wv)| xv * wv)
                .sum();
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The largest per-weight dequant error is one half-step of the block's
    /// scale (rounding) plus the f16 scale/bias storage error. For a block whose
    /// range is `R` over 7 asymmetric steps, `scale = R/7`, so the round error is
    /// bounded by `scale/2 = R/14`. This helper returns that bound with a small
    /// f16 slack factor.
    fn asym_err_bound(min: f32, max: f32) -> f32 {
        let scale = (max - min) / 7.0;
        scale / 2.0 + scale.abs() * 1e-3 + 1e-6
    }

    #[test]
    fn q3_block_is_sixteen_bytes() {
        assert_eq!(Q3_BLOCK_BYTES, 16);
        assert_eq!(std::mem::size_of::<Q3Block>(), 16);
    }

    #[test]
    fn plane_split_pack_unpack_roundtrips_all_values() {
        // Every one of the 32 slots must round-trip every 3-bit level 0..=7.
        for base in 0u8..8 {
            let mut q = [0u8; 32];
            for (i, slot) in q.iter_mut().enumerate() {
                *slot = ((base as usize + i) % 8) as u8;
            }
            let packed = pack_plane_split(&q);
            for (i, &expected) in q.iter().enumerate() {
                assert_eq!(
                    unpack_plane_split(&packed, i),
                    expected,
                    "value {i} (={expected}) did not survive plane-split pack/unpack"
                );
            }
        }
    }

    #[test]
    fn plane_split_mutation_sensitive_high_bit() {
        // A value >= 4 sets the high-1-bit plane. If the high plane is dropped
        // (a plausible packing bug), the unpacked value loses its +4 and this
        // assertion fails — the test guards the 2+1 split, not just the low plane.
        let mut q = [0u8; 32];
        q[5] = 6; // 0b110: low2 = 2, hi = 1
        q[17] = 7; // 0b111: low2 = 3, hi = 1
        let packed = pack_plane_split(&q);
        assert_eq!(unpack_plane_split(&packed, 5), 6);
        assert_eq!(unpack_plane_split(&packed, 17), 7);
        // The high plane byte for i=5 is packed[8 + 5/8] = packed[8], bit 5 set.
        assert_ne!(packed[8] & (1 << 5), 0, "high bit for value 5 must be set");
        // The high plane byte for i=17 is packed[8 + 17/8] = packed[10], bit 1 set.
        assert_ne!(
            packed[10] & (1 << 1),
            0,
            "high bit for value 17 must be set"
        );
    }

    #[test]
    fn asymmetric_roundtrip_within_scale_half() {
        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 12.0) * 0.37).collect();
        let bytes = quantize_row_q3_0(&src).unwrap();
        assert_eq!(bytes.len(), Q3_BLOCK_BYTES);
        let back = dequantize_row_q3_0(&bytes, src.len());
        let min = src.iter().copied().fold(f32::INFINITY, f32::min);
        let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let bound = asym_err_bound(min, max);
        for (i, (&a, &b)) in src.iter().zip(back.iter()).enumerate() {
            assert!(
                (a - b).abs() <= bound,
                "elem {i}: |{a} - {b}| = {} exceeds bound {bound}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn symmetric_roundtrip_zero_mean() {
        // Zero-mean, symmetric distribution — the symmetric encode mode's target.
        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.1).collect();
        let t = quantize_f32_to_q3_mode(&src, &[32], true).unwrap();
        let back = dequantize_q3_to_f32(&t);
        let abs_max = src.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        // Symmetric scale = abs_max / 3.5; per-step error bounded by scale/2 plus slack.
        let bound = (abs_max / 3.5) / 2.0 + abs_max * 1e-3 + 1e-6;
        for (i, (&a, &b)) in src.iter().zip(back.iter()).enumerate() {
            assert!(
                (a - b).abs() <= bound,
                "elem {i}: |{a} - {b}| = {} exceeds bound {bound}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn partial_block_uses_real_tail_min_max() {
        // Only 3 real values; the remaining 29 slots are zero padding that must
        // NOT widen the range used to derive the scale.
        let src = vec![10.0f32, 10.5, 11.0];
        let t = quantize_f32_to_q3(&src, &[3]).unwrap();
        assert_eq!(t.blocks.len(), 1);
        assert_eq!(t.original_len, 3);
        let back = dequantize_q3_to_f32(&t);
        assert_eq!(back.len(), 3);
        let bound = asym_err_bound(10.0, 11.0);
        for (i, (&a, &b)) in src.iter().zip(back.iter()).enumerate() {
            assert!(
                (a - b).abs() <= bound,
                "elem {i}: |{a} - {b}| = {} exceeds bound {bound}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn constant_block_is_exact() {
        // A zero-range block sets scale = 1.0, bias = min; every weight maps to
        // q = 0 and dequantizes back to exactly `bias`.
        let src = vec![3.5f32; 32];
        let t = quantize_f32_to_q3(&src, &[32]).unwrap();
        let back = dequantize_q3_to_f32(&t);
        for &b in &back {
            assert_eq!(b, 3.5);
        }
    }

    #[test]
    fn rejects_non_finite() {
        let mut src = vec![0.1f32; 32];
        src[7] = f32::NAN;
        assert!(quantize_f32_to_q3(&src, &[32]).is_err());
        src[7] = f32::INFINITY;
        assert!(quantize_f32_to_q3(&src, &[32]).is_err());
    }

    #[test]
    #[should_panic(expected = "must equal data length")]
    fn rejects_shape_data_mismatch() {
        let src = vec![0.1f32; 32];
        // shape claims 64 elements but data has 32.
        let _ = quantize_f32_to_q3(&src, &[64]);
    }

    #[test]
    fn bf16_path_matches_f32_path_on_bf16_castable_input() {
        // Values exactly representable in bf16 must quantize identically via
        // both entry points (bf16 decode is lossless for these).
        let bf16_bits: Vec<u16> = (0..64).map(|i| ((i as u16) << 7) | 0x3C00).collect();
        let f32_vals: Vec<f32> = bf16_bits.iter().map(|&b| bf16_to_f32(b)).collect();
        let via_bf16 = quantize_bf16_to_q3(&bf16_bits, &[64]).unwrap();
        let via_f32 = quantize_f32_to_q3(&f32_vals, &[64]).unwrap();
        assert_eq!(via_bf16.blocks, via_f32.blocks);
    }

    #[test]
    fn save_load_roundtrip_preserves_blocks_and_shape() {
        let src: Vec<f32> = (0..96).map(|i| (i as f32 - 40.0) * 0.05).collect();
        let t = quantize_f32_to_q3(&src, &[3, 32]).unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lattice_q3_roundtrip_{}.q3", std::process::id()));
        save_q3_file(&path, &t).unwrap();

        // Header parses independently of the block payload.
        let mut file = std::fs::File::open(&path).unwrap();
        let header = read_q3_header(&mut file).unwrap();
        assert_eq!(header.shape, vec![3, 32]);
        assert_eq!(header.original_len, 96);
        assert_eq!(header.payload_offset, 20 + 2 * 8);

        let loaded = load_q3_file(&path).unwrap();
        assert_eq!(loaded.shape, t.shape);
        assert_eq!(loaded.original_len, t.original_len);
        assert_eq!(loaded.blocks, t.blocks);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_q3_header_leaves_cursor_at_payload_offset() {
        // Regression for the doc promise on `read_q3_header`: the cursor of
        // the SAME file handle must sit at `payload_offset` on return, not
        // wherever the internal BufReader's last fill left the shared fd.
        // Reading the first block directly off `file` (no re-open, no seek)
        // must reproduce the first block written by `save_q3_file`.
        let src: Vec<f32> = (0..96).map(|i| (i as f32 - 40.0) * 0.05).collect();
        let t = quantize_f32_to_q3(&src, &[3, 32]).unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lattice_q3_cursor_{}.q3", std::process::id()));
        save_q3_file(&path, &t).unwrap();

        let mut file = std::fs::File::open(&path).unwrap();
        let header = read_q3_header(&mut file).unwrap();

        assert_eq!(
            std::io::Seek::stream_position(&mut file).unwrap(),
            header.payload_offset,
            "cursor must sit at payload_offset after read_q3_header returns"
        );

        let mut first_block_bytes = [0u8; Q3_BLOCK_BYTES];
        std::io::Read::read_exact(&mut file, &mut first_block_bytes).unwrap();
        let expected: &[u8; Q3_BLOCK_BYTES] = unsafe { &*std::ptr::from_ref(&t.blocks[0]).cast() };
        assert_eq!(
            &first_block_bytes, expected,
            "reading off the same handle post-header must yield the first block, not header tail"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lattice_q3_badmagic_{}.q3", std::process::id()));
        std::fs::write(&path, b"KHQ4\x01\x00\x00\x00").unwrap();
        assert!(load_q3_file(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn four_bpw_storage() {
        // 32 weights in 16 bytes = 4.0 bits per weight (the ADR-072 W3 target).
        let bits_per_weight = (Q3_BLOCK_BYTES * 8) as f32 / 32.0;
        assert_eq!(bits_per_weight, 4.0);
    }

    // -----------------------------------------------------------------------
    // MLP-role fail-closed loader gate
    // -----------------------------------------------------------------------

    #[test]
    fn mlp_role_accepts_gate_up_down_and_fused() {
        assert!(validate_q3_mlp_role("model.layers.3.mlp.gate_proj.weight").is_ok());
        assert!(validate_q3_mlp_role("model.layers.3.mlp.up_proj.weight").is_ok());
        assert!(validate_q3_mlp_role("model.layers.3.mlp.down_proj.weight").is_ok());
        assert!(validate_q3_mlp_role("model.layers.3.mlp.gate_up_proj.weight").is_ok());
    }

    #[test]
    fn mlp_role_rejects_attention_and_gdn_tensors() {
        // Attention and GDN weights must stay Q4/Q8 (ADR-072 P1 scope) — any Q3
        // claim on these tensor names must fail closed, not silently load.
        for name in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.layers.0.mlp.experts.gate_up_proj", // MoE routed experts, not the dense MLP path
            "lm_head.weight",
            "model.embed_tokens.weight",
        ] {
            assert!(
                validate_q3_mlp_role(name).is_err(),
                "expected '{name}' to be rejected as outside the Q3 MLP role set"
            );
        }
    }

    #[test]
    fn mlp_role_rejects_near_miss_suffixes() {
        // Exact-suffix contract: `.contains()` on the role substring would
        // wrongly accept these — a LoRA adapter branch, a renamed sibling
        // tensor, and a trailing extra suffix after `.weight`.
        for name in [
            "model.layers.3.mlp.gate_proj.lora_A.weight",
            "model.layers.3.mlp.up_proj_backup.weight",
            "model.layers.3.mlp.up_proj_backup",
            "model.layers.3.mlp.down_proj.weight.extra",
        ] {
            assert!(
                validate_q3_mlp_role(name).is_err(),
                "expected near-miss '{name}' to be rejected by the exact-suffix gate"
            );
        }
    }

    #[test]
    fn mlp_role_requires_terminal_weight_suffix() {
        // Real checkpoints always name dense MLP projections with a terminal
        // `.weight` (see the Q4 loader's `{prefix}.mlp.gate_proj.weight` call
        // sites in metal_qwen35.rs); a role-correct name missing it is not a
        // real tensor name and must fail closed.
        for name in [
            "model.layers.3.mlp.gate_proj",
            "model.layers.3.mlp.up_proj",
            "model.layers.3.mlp.down_proj",
            "model.layers.3.mlp.gate_up_proj",
        ] {
            assert!(
                validate_q3_mlp_role(name).is_err(),
                "expected '{name}' (missing '.weight') to be rejected"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Header payload-bounds fail-closed gate
    // -----------------------------------------------------------------------

    #[test]
    fn header_bounds_rejects_truncated_payload() {
        let header = Q3FileHeader {
            shape: vec![64],
            original_len: 64, // 2 blocks * 16 bytes = 32 bytes required
            payload_offset: 20,
        };
        // File is one byte short of the required 20 + 32 = 52 bytes.
        let err = validate_q3_header_payload_bounds(&header, 51, std::path::Path::new("t.q3"));
        assert!(err.is_err());
    }

    #[test]
    fn header_bounds_accepts_exact_payload() {
        let header = Q3FileHeader {
            shape: vec![64],
            original_len: 64,
            payload_offset: 20,
        };
        assert!(
            validate_q3_header_payload_bounds(&header, 52, std::path::Path::new("t.q3")).is_ok()
        );
    }

    #[test]
    fn header_bounds_rejects_shape_mismatch() {
        let header = Q3FileHeader {
            shape: vec![63], // disagrees with original_len
            original_len: 64,
            payload_offset: 20,
        };
        assert!(
            validate_q3_header_payload_bounds(&header, 1_000, std::path::Path::new("t.q3"))
                .is_err()
        );
    }

    // -----------------------------------------------------------------------
    // CPU reference GEMV/GEMM — parity oracle sanity + mutation sensitivity
    // -----------------------------------------------------------------------

    fn synth_q3_weight_matrix(seed: u64, n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
        let mut rng = seed;
        let mut next = || -> f32 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 11) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let weights_f32: Vec<f32> = (0..n * k).map(|_| next()).collect();
        let mut packed = Vec::with_capacity(n * (k / 32) * Q3_BLOCK_BYTES);
        let mut deq = Vec::with_capacity(n * k);
        for row in weights_f32.chunks_exact(k) {
            let row_packed = quantize_row_q3_0(row).unwrap();
            deq.extend_from_slice(&dequantize_row_q3_0(&row_packed, k));
            packed.extend_from_slice(&row_packed);
        }
        (packed, deq)
    }

    #[test]
    fn gemv_reference_matches_naive_dequant_dot_product() {
        let (n, k) = (17usize, 64usize);
        let (packed, deq) = synth_q3_weight_matrix(0x1234_5678, n, k);
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 0.32).collect();
        let y = gemv_q3_reference(&x, &packed, n, k);
        for (ni, &yv) in y.iter().enumerate() {
            let expect: f32 = x
                .iter()
                .zip(&deq[ni * k..(ni + 1) * k])
                .map(|(&xv, &wv)| xv * wv)
                .sum();
            assert!((yv - expect).abs() < 1e-3, "row {ni}: {yv} vs {expect}");
        }
    }

    #[test]
    fn gemm_reference_matches_gemv_reference_per_row() {
        // GEMM with M=1 must reduce to GEMV — cross-checks the two oracles
        // against each other in addition to the naive-dot check above.
        let (n, k) = (9usize, 96usize);
        let (packed, _deq) = synth_q3_weight_matrix(0xC0FF_EE11, n, k);
        let x: Vec<f32> = (0..k).map(|i| ((i * 7) % 13) as f32 * 0.05 - 0.3).collect();
        let y_gemv = gemv_q3_reference(&x, &packed, n, k);
        let y_gemm = gemm_q3_reference(&x, &packed, 1, n, k);
        assert_eq!(y_gemv, y_gemm);
    }

    #[test]
    fn gemv_reference_mutation_sensitive_high_plane_bit() {
        // Flip one high-plane bit in the packed weight buffer (the same class
        // of bug the Stage-1 `plane_split_mutation_sensitive_high_bit` test
        // guards at the pack/unpack level) and assert the GEMV parity oracle's
        // output changes — this is the oracle the Metal kernel differential
        // test compares against, so it must itself be sensitive to packing
        // corruption or a corrupted kernel could pass parity vacuously.
        let (n, k) = (4usize, 32usize);
        let (mut packed, _deq) = synth_q3_weight_matrix(0xFEED_FACE, n, k);
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.02 - 0.3).collect();
        let y_before = gemv_q3_reference(&x, &packed, n, k);

        // Row 0's block: bytes [0..16) = [scale(2) bias(2) low2(8) hi(4)].
        // Flip a bit in the high-plane byte at packed offset 4+8+0 = 12.
        packed[12] ^= 0x01;

        let y_after = gemv_q3_reference(&x, &packed, n, k);
        assert_ne!(
            y_before[0], y_after[0],
            "flipping a high-plane bit must change the dequantized GEMV output \
             for the row whose block it belongs to"
        );
    }
}
