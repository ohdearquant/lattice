//! Ternary matmul kernels for BitNet b1.58 inference.
//!
//! BitNet b1.58 uses weights quantized to {-1, 0, +1}. Since no multiplication
//! is needed (only add, subtract, or skip), this yields both memory savings
//! (2 bits per weight) and computational savings on CPU.
//!
//! # Weight packing (I2_S format)
//!
//! Each weight is encoded in 2 bits:
//! - `00` = 0 (skip)
//! - `01` = +1 (add)
//! - `10` = -1 (subtract)
//! - `11` = unused/reserved
//!
//! Within a `u8`, four weights are packed as:
//! - bits `[1:0]` = weight 0
//! - bits `[3:2]` = weight 1
//! - bits `[5:4]` = weight 2
//! - bits `[7:6]` = weight 3
//!
//! Each row carries a per-tensor scale `alpha = mean(|w_float|)` from the
//! original float weights (before rounding to ternary).
//!
//! # Activation quantization
//!
//! Before matmul, activations are quantized to int8 with symmetric range
//! `[-127, 127]` (matching Q8_0 convention in NEON and Metal paths):
//! ```text
//! gamma = max(|x|) / 127
//! x_q[i] = clamp(round(x[i] / gamma), -127, 127) as i8
//! ```
//!
//! The symmetric range avoids dequantization asymmetry (`-128` has no
//! positive counterpart). Since `gamma = max(|x|) / 127`, the scaled
//! values are mathematically bounded to `[-127, 127]` before rounding,
//! so the `-128` bound is unreachable either way — we use `-127` to
//! match the other kernels and document the intent explicitly.
//!
//! The final output is: `y = alpha * gamma * dot(x_q, w_ternary)`

use crate::error::InferenceError;

// ---------------------------------------------------------------------------
// I2_S weight encoding
// ---------------------------------------------------------------------------

/// 2-bit encoding: 0 → skip, 1 → +1, 2 → -1.
const ENC_ZERO: u8 = 0b00;
const ENC_POS: u8 = 0b01;
const ENC_NEG: u8 = 0b10;

/// Encode a single ternary value (-1, 0, +1) into its 2-bit representation.
#[inline(always)]
fn encode_ternary(v: i8) -> u8 {
    match v {
        1 => ENC_POS,
        -1 => ENC_NEG,
        _ => ENC_ZERO,
    }
}

/// Decode a 2-bit field back to the ternary value.
#[inline(always)]
fn decode_ternary(bits: u8) -> i8 {
    match bits & 0x03 {
        ENC_POS => 1,
        ENC_NEG => -1,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Packing / unpacking
// ---------------------------------------------------------------------------

/// **Unstable**: I2_S packing helper; byte layout tied to BitNet b1.58 spec.
///
/// Number of packed bytes needed for `k` ternary weights.
#[inline]
pub fn packed_row_bytes(k: usize) -> usize {
    k.div_ceil(4)
}

/// Pack a row of float weights into ternary I2_S format.
///
/// Each weight is rounded to the nearest of {-1, 0, +1} by sign(w) * round(|w| / alpha),
/// where alpha = mean(|w|) over the row. This matches the BitNet b1.58 quantization:
///   w_ternary = RoundClip(w / alpha, -1, 1)
///
/// Returns `(packed_bytes, alpha)` where `alpha` is the per-row scale factor.
fn pack_row(row: &[f32]) -> Result<(Vec<u8>, f32), InferenceError> {
    if let Some((index, value)) = row.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(InferenceError::InvalidInput(format!(
            "BitNet weight row contains non-finite value {value} at index {index}"
        )));
    }

    // Compute alpha = mean(|w|)
    let abs_sum: f32 = row.iter().map(|v| v.abs()).sum();
    let alpha = if row.is_empty() {
        0.0
    } else {
        abs_sum / row.len() as f32
    };

    let k = row.len();
    let num_bytes = packed_row_bytes(k);
    let mut packed = vec![0u8; num_bytes];

    if alpha > 0.0 {
        let inv_alpha = 1.0 / alpha;
        for (i, &w) in row.iter().enumerate() {
            let scaled = (w * inv_alpha).round().clamp(-1.0, 1.0) as i8;
            let enc = encode_ternary(scaled);
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            packed[byte_idx] |= enc << bit_offset;
        }
    }

    Ok((packed, alpha))
}

/// **Unstable**: pack float weights into I2_S ternary format; packing convention may change.
///
/// Pack an `n x k` float weight matrix into ternary I2_S format.
///
/// Returns `(packed_bytes, alphas)` where:
/// - `packed_bytes`: length `n * packed_row_bytes(k)`, rows stored contiguously
/// - `alphas`: per-row scale factors, length `n`
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] when the matrix geometry does not
/// match `weights` or any weight is non-finite.
pub fn pack_ternary(
    weights: &[f32],
    n: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<f32>), InferenceError> {
    let expected_len = n.checked_mul(k).ok_or_else(|| {
        InferenceError::InvalidInput(format!("BitNet weight geometry overflows: n={n}, k={k}"))
    })?;
    if weights.len() != expected_len {
        return Err(InferenceError::InvalidInput(format!(
            "BitNet weights length {} does not match n*k ({n}*{k}={expected_len})",
            weights.len()
        )));
    }

    let row_bytes = packed_row_bytes(k);
    let packed_len = n.checked_mul(row_bytes).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "BitNet packed geometry overflows: n={n}, row_bytes={row_bytes}"
        ))
    })?;
    let mut packed = vec![0u8; packed_len];
    let mut alphas = vec![0.0f32; n];

    for row_idx in 0..n {
        let row = &weights[row_idx * k..(row_idx + 1) * k];
        let (row_packed, alpha) = pack_row(row)?;
        packed[row_idx * row_bytes..(row_idx + 1) * row_bytes].copy_from_slice(&row_packed);
        alphas[row_idx] = alpha;
    }

    Ok((packed, alphas))
}

/// **Unstable**: unpack a single I2_S ternary weight; for debugging and testing only.
///
/// Unpack a single ternary weight from the packed buffer.
///
/// `row_idx` is the output row, `col_idx` is the position within that row.
#[inline]
pub fn unpack_weight(packed: &[u8], k: usize, row_idx: usize, col_idx: usize) -> i8 {
    let row_bytes = packed_row_bytes(k);
    let base = row_idx * row_bytes;
    let byte_idx = base + col_idx / 4;
    let bit_offset = (col_idx % 4) * 2;
    decode_ternary(packed[byte_idx] >> bit_offset)
}

// ---------------------------------------------------------------------------
// Activation quantization (absmax int8)
// ---------------------------------------------------------------------------

/// **Unstable**: quantize f32 activations to int8 absmax; quantization range convention may change.
///
/// Quantize an f32 activation vector to int8 with absmax scaling.
///
/// Returns `(x_quantized, gamma)` where `gamma = max(|x|) / 127`.
/// Reconstruction: `x[i] ~ x_q[i] as f32 * gamma`.
///
/// Uses the symmetric `[-127, 127]` range matching NEON and Metal Q8_0
/// kernels. The scaling `gamma = max(|x|) / 127` mathematically bounds
/// `v * inv_gamma` to `[-127, 127]`, so the clamp is defensive — but
/// pinning to `-127` keeps this kernel numerically aligned with the
/// other quantization paths (see module docs for details).
pub fn quantize_activation(x: &[f32]) -> (Vec<i8>, f32) {
    let abs_max = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if abs_max == 0.0 {
        return (vec![0i8; x.len()], 0.0);
    }
    let gamma = abs_max / 127.0;
    let inv_gamma = 1.0 / gamma;
    let quantized: Vec<i8> = x
        .iter()
        .map(|&v| (v * inv_gamma).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (quantized, gamma)
}

// ---------------------------------------------------------------------------
// Scalar ternary matvec
// ---------------------------------------------------------------------------

/// **Unstable**: scalar ternary matvec fallback; interface mirrors the NEON path.
///
/// Ternary matrix-vector product (scalar fallback).
///
/// Computes `output[i] = alpha[i] * x_scale * sum_j(x_q[j] * w_ternary[i,j])`
/// for each output row `i` in `0..n`.
///
/// # Arguments
/// - `x_q`: quantized activation vector, length `k`
/// - `x_scale`: activation scale (gamma)
/// - `packed_w`: packed ternary weights, `n * packed_row_bytes(k)` bytes
/// - `alphas`: per-row weight scales, length `n`
/// - `n`: number of output rows
/// - `k`: number of columns (input dimension)
/// - `output`: output buffer, length `n`
pub fn matvec_ternary_scalar(
    x_q: &[i8],
    x_scale: f32,
    packed_w: &[u8],
    alphas: &[f32],
    n: usize,
    k: usize,
    output: &mut [f32],
) {
    let row_bytes = packed_row_bytes(k);
    // Release-active, overflow-first, packed-ternary-layout contract (ADR-080 C4, held
    // finding): this was previously `debug_assert!`-gated only. In release, the tail
    // remainder loop below iterates `x_q.iter().enumerate().take(k).skip(rem_start)` —
    // `.take()` silently truncates rather than panicking when `x_q` is shorter than `k`,
    // so a too-short activation slice produced a silently wrong (under-summed) result
    // instead of a deterministic panic. Validated before any indexing below.
    crate::forward::cpu::validate_ternary_matvec_args(
        x_q.len(),
        alphas.len(),
        packed_w.len(),
        output.len(),
        n,
        k,
        row_bytes,
        "matvec_ternary_scalar",
    );

    for row in 0..n {
        let row_base = row * row_bytes;
        let mut acc: i32 = 0;

        // Process 4 weights per byte.
        let full_bytes = k / 4;
        for byte_idx in 0..full_bytes {
            let byte_val = packed_w[row_base + byte_idx];
            let x_base = byte_idx * 4;

            // Unroll the 4 weights in this byte.
            let w0 = decode_ternary(byte_val);
            let w1 = decode_ternary(byte_val >> 2);
            let w2 = decode_ternary(byte_val >> 4);
            let w3 = decode_ternary(byte_val >> 6);

            // Branchless: multiply by ternary is just conditional add/sub.
            acc += w0 as i32 * x_q[x_base] as i32;
            acc += w1 as i32 * x_q[x_base + 1] as i32;
            acc += w2 as i32 * x_q[x_base + 2] as i32;
            acc += w3 as i32 * x_q[x_base + 3] as i32;
        }

        // Handle remainder weights (k not a multiple of 4).
        let rem_start = full_bytes * 4;
        if rem_start < k {
            let byte_val = packed_w[row_base + full_bytes];
            for (j, &xj) in x_q.iter().enumerate().take(k).skip(rem_start) {
                let bit_offset = (j % 4) * 2;
                let w = decode_ternary(byte_val >> bit_offset);
                acc += w as i32 * xj as i32;
            }
        }

        output[row] = alphas[row] * x_scale * acc as f32;
    }
}

// ---------------------------------------------------------------------------
// NEON ternary matvec
// ---------------------------------------------------------------------------

/// **Unstable**: NEON ternary matvec; intrinsic selection and loop unroll factor may change.
///
/// NEON-accelerated ternary matrix-vector product.
///
/// Processes 16 int8 activations per NEON iteration. For each packed byte
/// (4 weights), extracts the 2-bit fields and uses conditional add/sub on
/// the int8 activation values. Accumulates into i32 lanes, then reduces.
///
/// # Safety
///
/// Caller must ensure this runs only on a target with NEON enabled. `x_q`,
/// `alphas`, `packed_w`, and `output` must satisfy the dimensions described
/// by `n` and `k`; the function validates those invariants in all builds
/// (release-active, via `validate_ternary_matvec_args`) before using unchecked
/// indexing in the SIMD loop.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn matvec_ternary_neon(
    x_q: &[i8],
    x_scale: f32,
    packed_w: &[u8],
    alphas: &[f32],
    n: usize,
    k: usize,
    output: &mut [f32],
) {
    use std::arch::aarch64::*;

    let row_bytes = packed_row_bytes(k);
    // Release-active, overflow-first, packed-ternary-layout contract (ADR-080 C4, held
    // finding): shared with `matvec_ternary_scalar` so both the safe scalar wrapper and
    // this unsafe NEON kernel enforce the identical contract instead of independently
    // re-deriving (and drifting on) the same bounds check. This kernel additionally reads
    // via `get_unchecked` below, so a missed/weakened check here is memory-unsafe, not
    // merely a wrong-result risk.
    crate::forward::cpu::validate_ternary_matvec_args(
        x_q.len(),
        alphas.len(),
        packed_w.len(),
        output.len(),
        n,
        k,
        row_bytes,
        "matvec_ternary_neon",
    );

    // Masks for extracting 2-bit fields from packed bytes.
    let mask_2bit = vdupq_n_u8(0x03);

    for (row, alpha) in alphas.iter().enumerate().take(n) {
        let row_base = row * row_bytes;
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);

        // Process 32 weights per iteration (8 packed bytes = 32 weights).
        // Each packed byte contains 4 weights.
        // We load 8 bytes, expand to 32 decoded ternary i8 values,
        // multiply with 32 activation i8 values, and accumulate.
        let chunks_32 = k / 32;
        for chunk in 0..chunks_32 {
            let w_offset = row_base + chunk * 8;
            let x_offset = chunk * 32;

            // Load 8 packed bytes (32 weights).
            // We process byte-by-byte since the packing is dense.
            // For each byte, extract 4 ternary weights.
            let mut ternary = [0i8; 32];
            for bi in 0..8 {
                let byte_val = *packed_w.get_unchecked(w_offset + bi);
                ternary[bi * 4] = decode_ternary(byte_val);
                ternary[bi * 4 + 1] = decode_ternary(byte_val >> 2);
                ternary[bi * 4 + 2] = decode_ternary(byte_val >> 4);
                ternary[bi * 4 + 3] = decode_ternary(byte_val >> 6);
            }

            // Load 32 ternary weights as 2x int8x16.
            let w_lo = vld1q_s8(ternary.as_ptr());
            let w_hi = vld1q_s8(ternary.as_ptr().add(16));

            // Load 32 activations as 2x int8x16.
            let x_lo = vld1q_s8(x_q.as_ptr().add(x_offset));
            let x_hi = vld1q_s8(x_q.as_ptr().add(x_offset + 16));

            // Multiply ternary * activation (i8 * i8 → i16, then widen to i32).
            // vmull_s8 takes low 8 lanes, produces 8x i16.
            let prod_lo_lo = vmull_s8(vget_low_s8(w_lo), vget_low_s8(x_lo));
            let prod_lo_hi = vmull_s8(vget_high_s8(w_lo), vget_high_s8(x_lo));
            let prod_hi_lo = vmull_s8(vget_low_s8(w_hi), vget_low_s8(x_hi));
            let prod_hi_hi = vmull_s8(vget_high_s8(w_hi), vget_high_s8(x_hi));

            // Widen i16 → i32 and accumulate.
            acc0 = vaddq_s32(acc0, vpaddlq_s16(prod_lo_lo));
            acc1 = vaddq_s32(acc1, vpaddlq_s16(prod_lo_hi));
            acc2 = vaddq_s32(acc2, vpaddlq_s16(prod_hi_lo));
            acc3 = vaddq_s32(acc3, vpaddlq_s16(prod_hi_hi));
        }

        // Reduce 4x i32x4 → scalar i32.
        let sum4 = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
        let mut acc_scalar: i32 = vaddvq_s32(sum4);

        // Handle remainder (weights after the last full chunk of 32).
        let rem_start = chunks_32 * 32;
        for j in rem_start..k {
            let byte_idx = row_base + j / 4;
            let bit_offset = (j % 4) * 2;
            let w = decode_ternary(*packed_w.get_unchecked(byte_idx) >> bit_offset);
            acc_scalar += w as i32 * *x_q.get_unchecked(j) as i32;
        }

        *output.get_unchecked_mut(row) = alpha * x_scale * acc_scalar as f32;
    }

    // Suppress unused variable warning — mask_2bit is reserved for a future
    // branchless decode path that extracts 2-bit fields directly in NEON
    // registers instead of using scalar decode_ternary.
    let _ = mask_2bit;
}

// ---------------------------------------------------------------------------
// Public dispatch
// ---------------------------------------------------------------------------

/// **Unstable**: ternary matmul with SIMD dispatch; dispatch logic and quantization may evolve.
///
/// Ternary matrix-vector multiply with automatic SIMD dispatch.
///
/// Quantizes the activation vector, then dispatches to the NEON or scalar kernel.
/// Returns the output vector of length `n`.
///
/// # Arguments
/// - `x`: f32 activation vector, length `k`
/// - `packed_w`: packed ternary weight matrix, `n * packed_row_bytes(k)` bytes
/// - `alphas`: per-row weight scales, length `n`
/// - `n`: number of output rows
/// - `k`: number of input columns
pub fn matmul_ternary(x: &[f32], packed_w: &[u8], alphas: &[f32], n: usize, k: usize) -> Vec<f32> {
    let row_bytes = packed_row_bytes(k);
    // Release-active, overflow-first, packed-ternary-layout contract (ADR-080 C4) — the
    // same shared validator `matvec_ternary_scalar`/`matvec_ternary_neon` use, checked here
    // too since `output` (freshly allocated below) doesn't exist yet at this point.
    crate::forward::cpu::validate_ternary_matvec_args(
        x.len(),
        alphas.len(),
        packed_w.len(),
        n, // output not yet allocated; its length will be exactly n below
        n,
        k,
        row_bytes,
        "matmul_ternary",
    );

    let (x_q, gamma) = quantize_activation(x);
    let mut output = vec![0.0f32; n];

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64.
        unsafe {
            matvec_ternary_neon(&x_q, gamma, packed_w, alphas, n, k, &mut output);
        }
        output
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        matvec_ternary_scalar(&x_q, gamma, packed_w, alphas, n, k, &mut output);
        output
    }
}

// ---------------------------------------------------------------------------
// Reference f32 matvec (for testing)
// ---------------------------------------------------------------------------

/// Reference float32 matrix-vector multiply (for validation).
///
/// Computes `output[i] = sum_j(weights[i*k + j] * x[j])`.
#[cfg(test)]
fn matvec_f32_reference(x: &[f32], weights: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += weights[i * k + j] * x[j];
        }
        output[i] = sum;
    }
    output
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests;
