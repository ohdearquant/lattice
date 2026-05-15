//! Q4_0 per-block weight quantization for large models (e.g., Qwen3.6-27B).
//!
//! ## Format
//!
//! Every 32 consecutive weights are packed into one [`Q4Block`]:
//! - `scale: u16` — the per-block scale stored as an IEEE-754 f16 bit pattern
//! - `packed: [u8; 16]` — 32 nibbles in **sequential-pairs** layout
//!
//! ### Nibble layout (sequential pairs — NOT llama.cpp split-half)
//!
//! ```text
//! byte[b] = (q[2b+1] << 4) | q[2b]     b ∈ 0..16
//! ```
//!
//! where `q[i] = clamp(round(weight[i] / scale) + 8, 0, 15)` and `scale = abs_max / 7`.
//! This matches the nibble convention used by the existing `gemv_q4_decode` Metal kernel
//! in `forward/metal_qwen35.rs`.
//!
//! ### Dequantization
//!
//! ```text
//! weight[2b]   = (byte[b] & 0x0F) as f32 - 8.0) * scale
//! weight[2b+1] = ((byte[b] >>  4) as f32 - 8.0) * scale
//! ```
//!
//! ## File format (`.q4`)
//!
//! ```text
//! magic       b"KHQ4"           4 bytes
//! version     1u32 LE           4 bytes
//! ndim        u32 LE            4 bytes
//! shape[i]    u64 LE × ndim
//! original_len u64 LE           8 bytes
//! blocks      [Q4Block; n_blocks]   n_blocks × 18 bytes
//! ```

// Q4 quantization operates on raw byte/u16 slices; unsafe is limited to
// the two transmute-equivalent slice casts in stream_quantize_shard and save/load.
#![allow(clippy::cast_possible_truncation)]

/// One Q4_0 quantization block: 32 weights packed as 4-bit unsigned integers.
///
/// `scale` is stored as a raw IEEE-754 f16 bit pattern in a `u16` — the `half` crate
/// is not a dependency of `lattice-inference`. Use [`q4_f32_to_f16`] / [`q4_f16_to_f32`].
///
/// `packed` holds 32 nibbles in **sequential-pairs** layout:
/// ```text
/// byte[b] = (q[2b+1] << 4) | q[2b]
/// ```
/// where `q[i] = clamp(round(weight[i] / scale) + 8, 0, 15)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q4Block {
    /// f16 bit pattern for the per-block scale — 2 bytes.
    pub scale: u16,
    /// 32 nibbles packed as 16 bytes in sequential-pairs layout.
    pub packed: [u8; 16],
}

// Compile-time size assertion — must be exactly 18 bytes (2 + 16, no padding).
const _: () = assert!(std::mem::size_of::<Q4Block>() == 18);

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
// Module-local f16 ↔ f32 helpers (no `half` crate dependency).
// Mirrors the implementation in `forward/metal_qwen35.rs:1907–2034`.
// ---------------------------------------------------------------------------

/// Convert `f32` to IEEE-754 half-precision stored as a `u16` bit pattern.
///
/// Uses round-to-nearest-even for mantissa truncation. Handles ±0, ±∞, NaN,
/// subnormals, and overflow (→ ±∞).
#[inline]
pub(crate) fn q4_f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    // Inf or NaN
    if exp == 0xff {
        if frac == 0 {
            return sign | 0x7c00; // ±∞
        }
        // NaN: preserve payload, ensure quiet bit is set.
        let mut payload = ((frac >> 13) as u16) & 0x03ff;
        if payload == 0 {
            payload = 1;
        }
        payload |= 0x0200;
        return sign | 0x7c00 | payload;
    }

    // Zero or f32 subnormal (underflows to f16 zero)
    if exp == 0 {
        return sign;
    }

    let exp32 = exp - 127; // unbiased exponent

    // Overflow → ±∞
    if exp32 > 15 {
        return sign | 0x7c00;
    }

    // Normal f16 range
    if exp32 >= -14 {
        let mut exp16 = (exp32 + 15) as u16;
        let mut frac16 = round_shift_right_even(frac, 13) as u16;
        // Mantissa overflow: carry into exponent
        if frac16 == 0x0400 {
            frac16 = 0;
            exp16 += 1;
            if exp16 >= 0x1f {
                return sign | 0x7c00;
            }
        }
        return sign | (exp16 << 10) | frac16;
    }

    // Subnormal f16 range
    let mant = frac | 0x0080_0000;
    let shift = (-exp32 - 1) as u32;
    if shift >= 32 {
        return sign;
    }
    let frac16 = round_shift_right_even(mant, shift) as u16;
    if frac16 == 0 {
        return sign;
    }
    if frac16 == 0x0400 {
        return sign | 0x0400; // smallest normal f16
    }
    sign | frac16
}

/// Round-to-nearest-even right shift for mantissa truncation.
#[inline]
fn round_shift_right_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    if shift >= 32 {
        return 0;
    }
    let base = value >> shift;
    let mask = (1u32 << shift) - 1;
    let remainder = value & mask;
    let half = 1u32 << (shift - 1);
    if remainder > half || (remainder == half && (base & 1) != 0) {
        base + 1
    } else {
        base
    }
}

/// Convert an IEEE-754 f16 bit pattern (`u16`) back to `f32`.
#[inline]
pub(crate) fn q4_f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;

    let f32_bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, _) => {
            // Subnormal: find leading 1, normalize.
            let mut mant = frac;
            let mut e = -14i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (mant << 13)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000, // ±∞
        (0x1f, _) => (sign << 31) | 0x7f80_0000 | (frac << 13), // NaN
        _ => (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 13),
    };

    f32::from_bits(f32_bits)
}

// ---------------------------------------------------------------------------
// BF16 helper (for BF16-format shard loading)
// ---------------------------------------------------------------------------

/// Convert a BF16 bit pattern (`u16`) to `f32`.
///
/// BF16 has identical sign+exponent layout to f32; zero-extending the mantissa
/// is a lossless widening. Handles ±0, ±∞, NaN, and subnormals correctly.
#[inline]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

// ---------------------------------------------------------------------------
// Core block quantization
// ---------------------------------------------------------------------------

/// Quantize exactly 32 f32 values into one [`Q4Block`].
///
/// Uses scale = `abs_max / 7.0` (if `abs_max == 0` uses `1.0`).
/// Nibble layout: sequential pairs — `byte[b] = (q[2b+1] << 4) | q[2b]`.
#[inline]
fn quantize_block(vals: &[f32; 32]) -> Q4Block {
    let abs_max = vals.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max == 0.0 {
        1.0f32
    } else {
        abs_max / 7.0
    };
    let inv_scale = 1.0 / scale;

    let mut packed = [0u8; 16];
    for b in 0..16 {
        let q0 = ((vals[2 * b] * inv_scale).round() + 8.0).clamp(0.0, 15.0) as u8;
        let q1 = ((vals[2 * b + 1] * inv_scale).round() + 8.0).clamp(0.0, 15.0) as u8;
        // Sequential-pairs: low nibble = even-indexed weight, high nibble = odd-indexed weight.
        packed[b] = (q1 << 4) | (q0 & 0x0f);
    }

    Q4Block {
        scale: q4_f32_to_f16(scale),
        packed,
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
/// Returns raw bytes containing tightly-packed [`Q4Block`]s (18 bytes each).
pub fn quantize_row_q4_0(src: &[f32]) -> Vec<u8> {
    let n_blocks = src.len().div_ceil(32);
    let mut out = Vec::with_capacity(n_blocks * 18);
    for chunk in src.chunks(32) {
        let mut vals = [0.0f32; 32];
        vals[..chunk.len()].copy_from_slice(chunk);
        let block = quantize_block(&vals);
        // SAFETY: Q4Block is #[repr(C)] with size 18; its alignment is 2 (the
        // alignment of the leading `scale: u16` per the Rust Reference's repr(C)
        // rule). Casting to `&[u8; 18]` is valid because the target element type
        // is `u8` (alignment 1 ≤ source alignment 2) and the source byte length
        // matches the destination length exactly.
        let bytes: &[u8; 18] = unsafe { &*std::ptr::from_ref(&block).cast() };
        out.extend_from_slice(bytes);
    }
    out
}

/// Dequantize Q4_0 blocks (raw bytes) back to f32 values.
///
/// `data` must be a multiple of 18 bytes (one block per 18 bytes).
/// Returns exactly `n_weights` values; the caller is responsible for ensuring
/// `n_weights <= (data.len() / 18) * 32`.
pub fn dequantize_row_q4_0(data: &[u8], n_weights: usize) -> Vec<f32> {
    assert_eq!(
        data.len() % 18,
        0,
        "data length must be a multiple of 18 (Q4Block size)"
    );
    let mut out = Vec::with_capacity(n_weights);
    for chunk in data.chunks_exact(18) {
        let scale_bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
        let scale = q4_f16_to_f32(scale_bits);
        for b in 0..16 {
            let byte_val = chunk[2 + b];
            out.push(((byte_val & 0x0f) as f32 - 8.0) * scale);
            out.push(((byte_val >> 4) as f32 - 8.0) * scale);
        }
    }
    out.truncate(n_weights);
    out
}

/// Quantize a row-major f32 tensor into Q4_0 blocks, one row at a time.
///
/// `src` has shape `[rows, cols]`. Each row is quantized independently into
/// `cols.div_ceil(32)` blocks. Returns raw bytes (18 bytes per block).
pub fn quantize_tensor_q4_0(src: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(
        src.len(),
        rows * cols,
        "src length does not match rows * cols"
    );
    let blocks_per_row = cols.div_ceil(32);
    let mut out = Vec::with_capacity(rows * blocks_per_row * 18);
    for row_idx in 0..rows {
        let row = &src[row_idx * cols..(row_idx + 1) * cols];
        out.extend_from_slice(&quantize_row_q4_0(row));
    }
    out
}

// ---------------------------------------------------------------------------
// BF16-input quantization API (for streaming model shards)
// ---------------------------------------------------------------------------

/// Quantize a BF16 tensor (raw `u16` slice) into a [`Q4Tensor`].
///
/// `shape.iter().product()` must equal `data.len()`.
pub fn quantize_bf16_to_q4(data: &[u16], shape: &[usize]) -> Q4Tensor {
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(32);
    let mut blocks = Vec::with_capacity(n_blocks);

    for chunk in data.chunks(32) {
        let mut vals = [0.0f32; 32];
        for (i, &v) in chunk.iter().enumerate() {
            vals[i] = bf16_to_f32(v);
        }
        blocks.push(quantize_block(&vals));
    }

    Q4Tensor {
        blocks,
        shape: shape.to_vec(),
        original_len,
    }
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
/// `shape.iter().product()` must equal `data.len()`.
pub fn quantize_f32_to_q4(data: &[f32], shape: &[usize]) -> Q4Tensor {
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(32);
    let mut blocks = Vec::with_capacity(n_blocks);

    for chunk in data.chunks(32) {
        let mut vals = [0.0f32; 32];
        vals[..chunk.len()].copy_from_slice(chunk);
        blocks.push(quantize_block(&vals));
    }

    Q4Tensor {
        blocks,
        shape: shape.to_vec(),
        original_len,
    }
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
/// `shape.iter().product()` must equal `data.len()`.
pub fn quantize_f64_to_q4(data: &[f64], shape: &[usize]) -> Q4Tensor {
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(32);
    let mut blocks = Vec::with_capacity(n_blocks);

    for chunk in data.chunks(32) {
        let mut vals = [0.0f32; 32];
        for (i, &v) in chunk.iter().enumerate() {
            vals[i] = v as f32;
        }
        blocks.push(quantize_block(&vals));
    }

    Q4Tensor {
        blocks,
        shape: shape.to_vec(),
        original_len,
    }
}

/// Dequantize all blocks of a [`Q4Tensor`] back to f32.
///
/// Output length equals `tensor.original_len` (zero-padded tail blocks are truncated).
pub fn dequantize_q4_to_f32(tensor: &Q4Tensor) -> Vec<f32> {
    let mut out = Vec::with_capacity(tensor.original_len);
    for block in &tensor.blocks {
        let scale = q4_f16_to_f32(block.scale);
        for b in 0..16 {
            let byte_val = block.packed[b];
            out.push(((byte_val & 0x0f) as f32 - 8.0) * scale);
            out.push(((byte_val >> 4) as f32 - 8.0) * scale);
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
    if bf16_bytes.len() % 2 != 0 {
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
        blocks.push(quantize_block(&vals));
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
/// version      1u32 LE   4 bytes
/// ndim         u32 LE    4 bytes
/// shape[i]     u64 LE × ndim
/// original_len u64 LE    8 bytes
/// blocks       [Q4Block; n]  n × 18 bytes
/// ```
pub fn save_q4_file(path: &std::path::Path, tensor: &Q4Tensor) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"KHQ4")?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;
    for &dim in &tensor.shape {
        f.write_all(&(dim as u64).to_le_bytes())?;
    }
    f.write_all(&(tensor.original_len as u64).to_le_bytes())?;
    // SAFETY: Q4Block is #[repr(C)] with size 18; its alignment is 2 (the
    // alignment of the leading `scale: u16` per the Rust Reference's repr(C)
    // rule). Casting to a `&[u8]` is valid because the target element type is
    // `u8` (alignment 1 ≤ source alignment 2). The resulting slice has length
    // `blocks.len() * 18` matching the source contiguous storage.
    let block_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            tensor.blocks.as_ptr().cast::<u8>(),
            tensor.blocks.len() * 18,
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

/// Parse the header of a `.q4` file without reading the block payload.
///
/// On return the file cursor is positioned at the start of the block data.
///
/// # Errors
///
/// Returns an error on I/O failure, unrecognized magic bytes, or unsupported version.
pub fn read_q4_header(file: &std::fs::File) -> Result<Q4FileHeader, Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut f = std::io::BufReader::new(file);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"KHQ4" {
        return Err("invalid magic: not a .q4 file".into());
    }

    let mut b4 = [0u8; 4];
    f.read_exact(&mut b4)?;
    if u32::from_le_bytes(b4) != 1 {
        return Err("unsupported .q4 file version".into());
    }

    f.read_exact(&mut b4)?;
    let ndim = u32::from_le_bytes(b4) as usize;
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

/// Load a [`Q4Tensor`] from a `.q4` file written by [`save_q4_file`].
///
/// # Errors
///
/// Returns an error on I/O failure, unrecognized magic bytes, or unsupported version.
pub fn load_q4_file(path: &std::path::Path) -> Result<Q4Tensor, Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"KHQ4" {
        return Err("invalid magic: not a .q4 file".into());
    }

    let mut b4 = [0u8; 4];
    f.read_exact(&mut b4)?;
    if u32::from_le_bytes(b4) != 1 {
        return Err("unsupported .q4 file version".into());
    }

    f.read_exact(&mut b4)?;
    let ndim = u32::from_le_bytes(b4) as usize;
    let mut shape = Vec::with_capacity(ndim);
    let mut b8 = [0u8; 8];
    for _ in 0..ndim {
        f.read_exact(&mut b8)?;
        shape.push(u64::from_le_bytes(b8) as usize);
    }

    f.read_exact(&mut b8)?;
    let original_len = u64::from_le_bytes(b8) as usize;
    let n_blocks = original_len.div_ceil(32);

    let mut raw = vec![0u8; n_blocks * 18];
    f.read_exact(&mut raw)?;

    let blocks: Vec<Q4Block> = raw
        .chunks_exact(18)
        .map(|c| Q4Block {
            scale: u16::from_ne_bytes([c[0], c[1]]),
            packed: c[2..18].try_into().expect("slice is exactly 16 bytes"),
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
/// Returns an error on I/O failure, unrecognized magic bytes, or unsupported version.
pub fn load_f16_tensor_file(
    path: &std::path::Path,
) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"KHF1" {
        return Err(format!(
            "invalid magic at {}: expected KHF1, got {:?}",
            path.display(),
            magic
        )
        .into());
    }

    let mut b4 = [0u8; 4];
    f.read_exact(&mut b4)?;
    if u32::from_le_bytes(b4) != 1 {
        return Err("unsupported .f16 file version".into());
    }

    f.read_exact(&mut b4)?;
    let ndim = u32::from_le_bytes(b4) as usize;
    let mut shape = Vec::with_capacity(ndim);
    let mut b8 = [0u8; 8];
    for _ in 0..ndim {
        f.read_exact(&mut b8)?;
        shape.push(u64::from_le_bytes(b8) as usize);
    }

    f.read_exact(&mut b8)?;
    let numel = u64::from_le_bytes(b8) as usize;

    let mut raw = vec![0u8; numel * 2];
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test 1: Q4Block is exactly 18 bytes with no padding after scale.
    // -----------------------------------------------------------------------
    #[test]
    fn test_q4_block_size() {
        assert_eq!(std::mem::size_of::<Q4Block>(), 18);
        let b = Q4Block {
            scale: 0,
            packed: [0u8; 16],
        };
        let base = std::ptr::from_ref(&b) as usize;
        let packed_off = std::ptr::from_ref(&b.packed) as usize - base;
        assert_eq!(
            packed_off, 2,
            "packed field must start at byte offset 2 (no padding after scale)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: All-zero roundtrip — zeros in must produce zeros out.
    // -----------------------------------------------------------------------
    #[test]
    fn test_quantize_dequantize_zeros() {
        let data = quantize_row_q4_0(&vec![0.0f32; 64]);
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
        let data = quantize_row_q4_0(&src);
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
        let data = quantize_row_q4_0(&src);
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
    // Test 5: Values at ±max_abs are quantized to nibble 15 / nibble 1.
    // -----------------------------------------------------------------------
    #[test]
    fn test_quantize_max_range() {
        // Block with w[0] = 7.0 (max, nibble 15) and w[1] = -7.0 (min, nibble 1), rest 0.
        let mut src = vec![0.0f32; 32];
        src[0] = 7.0;
        src[1] = -7.0;
        let data = quantize_row_q4_0(&src);
        // scale = 7.0 / 7 = 1.0
        // q[0] = round(7.0/1.0) + 8 = 15 → low nibble of byte[0]
        // q[1] = round(-7.0/1.0) + 8 = 1  → high nibble of byte[0]
        // byte[0] = (1 << 4) | 15 = 0x1F
        let block_byte0 = data[2]; // bytes 0..2 = scale u16, byte 2 = packed[0]
        assert_eq!(
            block_byte0 & 0x0f,
            15,
            "w[0]=7.0 should produce low nibble 15"
        );
        assert_eq!(
            block_byte0 >> 4,
            1,
            "w[1]=-7.0 should produce high nibble 1"
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
        let data = quantize_row_q4_0(&src);
        assert_eq!(data.len(), 18, "single block must be 18 bytes");
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
        let data = quantize_row_q4_0(&src);
        assert_eq!(data.len(), 4 * 18, "4 blocks must be 72 bytes");
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
            0.0f32, 1.0, -1.0, 0.5, -0.5, 3.14159, 100.0, -100.0, 0.001,
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
        // Block: w[0]=0.0, w[1]=7.0, rest 0.0
        // abs_max = 7.0, scale = 1.0
        // q[0] = round(0.0/1.0) + 8 = 8  → nibble 0x8
        // q[1] = round(7.0/1.0) + 8 = 15 → nibble 0xF
        // byte[0] = (0xF << 4) | (0x8 & 0x0F) = 0xF8
        let mut src = vec![0.0f32; 32];
        src[0] = 0.0;
        src[1] = 7.0;
        let data = quantize_row_q4_0(&src);
        let byte0 = data[2]; // packed[0] starts at offset 2 (after scale u16)
        assert_eq!(
            byte0, 0xF8,
            "byte[0] should be 0xF8 for w[0]=0.0 (nibble=8), w[1]=7.0 (nibble=15)"
        );

        // Verify dequant matches sequential pairs convention
        // low nibble → weight[0]: (0x8 - 8) * 1.0 = 0.0
        // high nibble → weight[1]: (0xF - 8) * 1.0 = 7.0
        let out = dequantize_row_q4_0(&data, 32);
        assert!(
            (out[0] - 0.0).abs() < 1e-6,
            "weight[0] should be 0.0, got {}",
            out[0]
        );
        assert!(
            (out[1] - 7.0).abs() < 1e-6,
            "weight[1] should be 7.0, got {}",
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
        let data = quantize_tensor_q4_0(&src, rows, cols);
        let blocks_per_row = cols.div_ceil(32); // 2 blocks per row of 64 cols
        assert_eq!(
            data.len(),
            rows * blocks_per_row * 18,
            "tensor bytes mismatch"
        );

        // Dequant each row and check roundtrip error.
        for row_idx in 0..rows {
            let row_bytes =
                &data[row_idx * blocks_per_row * 18..(row_idx + 1) * blocks_per_row * 18];
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
        let tensor = quantize_bf16_to_q4(&data, &[64]);
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
        let tensor = quantize_bf16_to_q4(&bf16_vals, &[32]);
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
        // w[0]=0.0, w[1]=7.0 → scale=1.0 → byte[0] = 0xF8
        let mut f32_vals = [0.0f32; 32];
        f32_vals[0] = 0.0;
        f32_vals[1] = 7.0;
        let bf16_vals = to_bf16(&f32_vals);
        let tensor = quantize_bf16_to_q4(&bf16_vals, &[32]);
        assert_eq!(tensor.blocks.len(), 1);
        assert_eq!(
            tensor.blocks[0].packed[0], 0xF8,
            "byte[0] should be 0xF8 for w[0]=0.0, w[1]=7.0 with scale=1.0"
        );
    }

    #[test]
    fn test_max_value_clamps_to_nibble_15() {
        // w[0]=100.0 → scale = 100/7 ≈ 14.28 → q[0] = round(7.0)+8 = 15
        let mut f32_vals = [0.0f32; 32];
        f32_vals[0] = 100.0;
        let bf16_vals = to_bf16(&f32_vals);
        let tensor = quantize_bf16_to_q4(&bf16_vals, &[32]);
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
        let tensor = quantize_bf16_to_q4(&bf16_vals, &[64]);
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
        let original = quantize_bf16_to_q4(&bf16_vals, &[8, 8]);
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
        let batch_tensor = quantize_bf16_to_q4(&bf16_vals, &[96]);
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
        let tensor = quantize_bf16_to_q4(&data, &shape);
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
        let data = quantize_row_q4_0(&f32_vals);
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
        let q = quantize_f32_to_q4(&src, &[3, 32]);
        assert_eq!(q.shape, vec![3, 32]);
        assert_eq!(q.original_len, 96);
        assert_eq!(q.blocks.len(), 3, "96 elems = 3 full Q4 blocks");
    }

    #[test]
    fn quantize_f32_to_q4_pads_partial_block() {
        let src = synthetic_f32_uniform(40, 19);
        let q = quantize_f32_to_q4(&src, &[40]);
        assert_eq!(q.original_len, 40);
        assert_eq!(q.blocks.len(), 2, "40 elems = 1 full + 1 partial Q4 block");
    }

    #[test]
    fn quantize_f64_to_q4_matches_f32_path_after_downcast() {
        // The f64 wrapper must agree byte-for-byte with the f32 entry after the
        // caller does the downcast explicitly. This locks down the contract
        // that the f64 path is a thin convenience wrapper, not a different
        // quantization recipe.
        let src_f64: Vec<f64> = synthetic_f32_uniform(256, 23)
            .into_iter()
            .map(f64::from)
            .collect();
        let src_f32: Vec<f32> = src_f64.iter().map(|&v| v as f32).collect();
        let q_f64 = quantize_f64_to_q4(&src_f64, &[256]);
        let q_f32 = quantize_f32_to_q4(&src_f32, &[256]);
        assert_eq!(q_f64.shape, q_f32.shape);
        assert_eq!(q_f64.original_len, q_f32.original_len);
        assert_eq!(
            q_f64.blocks.len(),
            q_f32.blocks.len(),
            "f64 path must produce same block count"
        );
        for (i, (a, b)) in q_f64.blocks.iter().zip(q_f32.blocks.iter()).enumerate() {
            assert_eq!(a.scale, b.scale, "block {i} scale mismatch");
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

        let q_bf16 = quantize_bf16_to_q4(&bf16_bits, &[256]);
        let q_f32 = quantize_f32_to_q4(&f32_from_bf16, &[256]);
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

        let q_bf16 = quantize_bf16_to_q4(&bf16_bits, &[2048]);
        let q_f32 = quantize_f32_to_q4(&src, &[2048]);
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

    #[test]
    fn quantize_f32_to_q4_block_layout_matches_quantize_row() {
        // Sanity: for input that is an exact multiple of 32, the entry should
        // produce the same per-block byte layout as `quantize_row_q4_0`
        // (which the existing kernels are already validated against).
        let src = synthetic_f32_uniform(128, 37);
        let q = quantize_f32_to_q4(&src, &[128]);
        let row_bytes = quantize_row_q4_0(&src);
        assert_eq!(row_bytes.len(), q.blocks.len() * 18);
        // SAFETY: Q4Block is #[repr(C)] size 18, alignment 2 (from leading
        // `scale: u16`); byte-cast is valid because target element type is u8.
        let q_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(q.blocks.as_ptr().cast::<u8>(), q.blocks.len() * 18)
        };
        assert_eq!(q_bytes, row_bytes.as_slice());
    }
}
