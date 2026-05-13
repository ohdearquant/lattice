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
//! - bits [1:0] = weight 0
//! - bits [3:2] = weight 1
//! - bits [5:4] = weight 2
//! - bits [7:6] = weight 3
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
fn pack_row(row: &[f32]) -> (Vec<u8>, f32) {
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

    (packed, alpha)
}

/// **Unstable**: pack float weights into I2_S ternary format; packing convention may change.
///
/// Pack an `n x k` float weight matrix into ternary I2_S format.
///
/// Returns `(packed_bytes, alphas)` where:
/// - `packed_bytes`: length `n * packed_row_bytes(k)`, rows stored contiguously
/// - `alphas`: per-row scale factors, length `n`
pub fn pack_ternary(weights: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(weights.len(), n * k, "weights length must be n*k");

    let row_bytes = packed_row_bytes(k);
    let mut packed = vec![0u8; n * row_bytes];
    let mut alphas = vec![0.0f32; n];

    for row_idx in 0..n {
        let row = &weights[row_idx * k..(row_idx + 1) * k];
        let (row_packed, alpha) = pack_row(row);
        packed[row_idx * row_bytes..(row_idx + 1) * row_bytes].copy_from_slice(&row_packed);
        alphas[row_idx] = alpha;
    }

    (packed, alphas)
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
    debug_assert_eq!(x_q.len(), k);
    debug_assert_eq!(alphas.len(), n);
    debug_assert!(output.len() >= n);

    let row_bytes = packed_row_bytes(k);

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
/// by `n` and `k`; the function checks those invariants in debug builds before
/// using unchecked indexing in the SIMD loop.
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

    debug_assert_eq!(x_q.len(), k);
    debug_assert_eq!(alphas.len(), n);
    debug_assert!(output.len() >= n);
    debug_assert!(packed_w.len() >= n * packed_row_bytes(k));

    let row_bytes = packed_row_bytes(k);

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
    assert_eq!(x.len(), k, "activation length must equal k");
    assert_eq!(alphas.len(), n, "alphas length must equal n");
    assert_eq!(
        packed_w.len(),
        n * packed_row_bytes(k),
        "packed weights size mismatch"
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
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // Encoding / decoding roundtrip
    // -------------------------------------------------------------------

    #[test]
    fn test_encode_decode_roundtrip() {
        assert_eq!(decode_ternary(encode_ternary(0)), 0);
        assert_eq!(decode_ternary(encode_ternary(1)), 1);
        assert_eq!(decode_ternary(encode_ternary(-1)), -1);
    }

    #[test]
    fn test_decode_ternary_masks_correctly() {
        // Only the lowest 2 bits should matter.
        assert_eq!(decode_ternary(0b11_10_01_00 >> 0), 0);
        assert_eq!(decode_ternary(0b11_10_01_00 >> 2), 1);
        assert_eq!(decode_ternary(0b11_10_01_00 >> 4), -1);
        // 0b11 maps to 0 (unused encoding).
        assert_eq!(decode_ternary(0b11), 0);
    }

    // -------------------------------------------------------------------
    // Pack / unpack roundtrip
    // -------------------------------------------------------------------

    #[test]
    fn test_pack_unpack_roundtrip_simple() {
        // Create weights that are already exactly ternary: {-1, 0, 1}
        // When alpha = mean(|w|), the ternary values will quantize back to themselves
        // if the original values are +alpha, -alpha, or 0.
        let alpha = 0.5; // pick any positive scale
        let k = 8;
        let n = 2;
        // Row 0: [+1, -1, 0, +1, -1, -1, +1, 0] (as floats scaled by alpha)
        // Row 1: [0, +1, +1, -1, 0, 0, -1, +1]
        let row0_ternary: [i8; 8] = [1, -1, 0, 1, -1, -1, 1, 0];
        let row1_ternary: [i8; 8] = [0, 1, 1, -1, 0, 0, -1, 1];

        let mut weights = vec![0.0f32; n * k];
        for (i, &t) in row0_ternary.iter().enumerate() {
            weights[i] = t as f32 * alpha;
        }
        for (i, &t) in row1_ternary.iter().enumerate() {
            weights[k + i] = t as f32 * alpha;
        }

        let (packed, alphas) = pack_ternary(&weights, n, k);

        // Alpha should be close to the mean absolute value.
        // Row 0: 6 nonzero * alpha / 8 = 0.375
        let expected_alpha0 = 6.0 * alpha / 8.0;
        assert!(
            (alphas[0] - expected_alpha0).abs() < 1e-6,
            "alpha[0]={} expected={}",
            alphas[0],
            expected_alpha0
        );

        // Unpack and verify ternary values match.
        for j in 0..k {
            let w0 = unpack_weight(&packed, k, 0, j);
            assert_eq!(
                w0, row0_ternary[j],
                "row=0 col={}: got {} expected {}",
                j, w0, row0_ternary[j]
            );
        }
        for j in 0..k {
            let w1 = unpack_weight(&packed, k, 1, j);
            assert_eq!(
                w1, row1_ternary[j],
                "row=1 col={}: got {} expected {}",
                j, w1, row1_ternary[j]
            );
        }
    }

    #[test]
    fn test_pack_unpack_non_multiple_of_4() {
        // k=7: not a multiple of 4.
        let k = 7;
        let n = 1;
        // All +1 weights.
        let weights = vec![1.0f32; k];
        let (packed, alphas) = pack_ternary(&weights, n, k);

        assert!(alphas[0] > 0.0);
        assert_eq!(packed.len(), packed_row_bytes(k)); // ceil(7/4) = 2 bytes

        for j in 0..k {
            assert_eq!(unpack_weight(&packed, k, 0, j), 1, "col={}", j);
        }
    }

    // -------------------------------------------------------------------
    // Activation quantization
    // -------------------------------------------------------------------

    #[test]
    fn test_quantize_activation_basic() {
        let x = vec![1.0, -1.0, 0.5, -0.5, 0.0];
        let (q, gamma) = quantize_activation(&x);

        // gamma = 1.0 / 127 ~ 0.00787
        assert!((gamma - 1.0 / 127.0).abs() < 1e-6);
        assert_eq!(q[0], 127); // 1.0 / gamma = 127
        assert_eq!(q[1], -127); // -1.0 / gamma = -127
        // 0.5 / gamma = 63.5, rounds to 64
        assert!((q[2] as i32 - 64).unsigned_abs() <= 1);
        assert!((q[3] as i32 + 64).unsigned_abs() <= 1);
        assert_eq!(q[4], 0);
    }

    #[test]
    fn test_quantize_activation_all_zero() {
        let x = vec![0.0; 10];
        let (q, gamma) = quantize_activation(&x);
        assert_eq!(gamma, 0.0);
        assert!(q.iter().all(|&v| v == 0));
    }

    /// Regression test for #1428: activation quantization must use the
    /// symmetric `[-127, 127]` range matching NEON/Metal Q8_0. `-128`
    /// must never appear in the output, even for negative extremes.
    #[test]
    fn test_quantize_activation_symmetric_range() {
        // Extreme negative value should produce -127, not -128.
        let x = vec![-1.0, 1.0, -0.99999, 0.99999];
        let (q, _gamma) = quantize_activation(&x);
        assert_eq!(q[0], -127, "symmetric range: -1.0 must map to -127");
        assert_eq!(q[1], 127, "symmetric range: +1.0 must map to +127");
        for &v in &q {
            assert!(
                v >= -127,
                "quantized value {v} violates symmetric [-127, 127] range"
            );
        }
    }

    #[test]
    fn test_quantize_activation_roundtrip_approx() {
        let x = vec![0.3, -0.7, 1.5, -2.0, 0.0, 0.01];
        let (q, gamma) = quantize_activation(&x);
        // Reconstruct and check error for entries large enough to survive
        // the quantization bucket size. The int8 step is gamma ~ absmax/127,
        // so values smaller than ~2*gamma have poor relative precision.
        let bucket = 2.0 * gamma;
        for (i, &original) in x.iter().enumerate() {
            let reconstructed = q[i] as f32 * gamma;
            if original.abs() > bucket {
                let rel_error = (reconstructed - original).abs() / original.abs();
                assert!(
                    rel_error < 0.02,
                    "index {}: original={} reconstructed={} rel_error={}",
                    i,
                    original,
                    reconstructed,
                    rel_error
                );
            }
        }
    }

    // -------------------------------------------------------------------
    // Scalar matvec
    // -------------------------------------------------------------------

    #[test]
    fn test_matvec_scalar_simple() {
        // 2x4 weight matrix, all +1.
        let n = 2;
        let k = 4;
        let weights = vec![1.0f32; n * k];
        let (packed, alphas) = pack_ternary(&weights, n, k);

        // Activation: [1.0, 2.0, 3.0, 4.0]
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let (x_q, gamma) = quantize_activation(&x);

        let mut output = vec![0.0f32; n];
        matvec_ternary_scalar(&x_q, gamma, &packed, &alphas, n, k, &mut output);

        // Expected: alpha * gamma * sum(x_q) for each row.
        // sum of x = 10.0. With quantization, the result should be close to 10.0 * alpha_scale.
        // alpha = mean(|1.0|) = 1.0, gamma = 4.0/127.
        // Quantized: x_q = [32, 64, 95, 127] (approx).
        // dot = sum(x_q) since all weights are +1.
        // result ~ alpha * gamma * dot ~ 1.0 * (4.0/127) * (32+64+95+127) ~ 10.0
        for i in 0..n {
            assert!(
                (output[i] - 10.0).abs() < 0.5,
                "row {}: got {}, expected ~10.0",
                i,
                output[i]
            );
        }
    }

    #[test]
    fn test_matvec_scalar_identity_pattern() {
        // 4x4 matrix: row i has weight +1 only at column i, rest 0.
        // This is like an identity matrix in ternary.
        let n = 4;
        let k = 4;
        let mut weights = vec![0.0f32; n * k];
        // Only nonzero on diagonal, so we set diagonal to 1.0.
        for i in 0..n {
            weights[i * k + i] = 1.0;
        }
        let (packed, alphas) = pack_ternary(&weights, n, k);

        let x = vec![10.0, 20.0, 30.0, 40.0];
        let (x_q, gamma) = quantize_activation(&x);

        let mut output = vec![0.0f32; n];
        matvec_ternary_scalar(&x_q, gamma, &packed, &alphas, n, k, &mut output);

        // Each output[i] ~ alpha[i] * gamma * x_q[i].
        // alpha[i] = mean(|row_i|) = 1.0/4 = 0.25.
        // Due to ternary quantization, scaling won't give exact original values,
        // but the relative ordering should be preserved.
        for i in 0..n {
            assert!(
                output[i] > 0.0,
                "row {} should be positive, got {}",
                i,
                output[i]
            );
        }
        // Check ordering: output[3] > output[2] > output[1] > output[0]
        for i in 0..n - 1 {
            assert!(
                output[i + 1] > output[i],
                "ordering violated: output[{}]={} should be < output[{}]={}",
                i,
                output[i],
                i + 1,
                output[i + 1]
            );
        }
    }

    #[test]
    fn test_matvec_scalar_all_zero_weights() {
        let n = 3;
        let k = 8;
        let weights = vec![0.0f32; n * k];
        let (packed, alphas) = pack_ternary(&weights, n, k);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (x_q, gamma) = quantize_activation(&x);

        let mut output = vec![999.0f32; n];
        matvec_ternary_scalar(&x_q, gamma, &packed, &alphas, n, k, &mut output);

        for i in 0..n {
            assert_eq!(output[i], 0.0, "row {} should be 0, got {}", i, output[i]);
        }
    }

    #[test]
    fn test_matvec_scalar_all_negative_weights() {
        // All weights = -1. Output should be negative sum of activations.
        let n = 1;
        let k = 8;
        let weights = vec![-1.0f32; n * k];
        let (packed, alphas) = pack_ternary(&weights, n, k);

        let x = vec![1.0; k]; // sum = 8.0
        let (x_q, gamma) = quantize_activation(&x);

        let mut output = vec![0.0f32; n];
        matvec_ternary_scalar(&x_q, gamma, &packed, &alphas, n, k, &mut output);

        // Should be approximately -8.0
        assert!(
            (output[0] + 8.0).abs() < 0.5,
            "got {}, expected ~-8.0",
            output[0]
        );
    }

    #[test]
    fn test_matvec_scalar_matches_f32_reference() {
        // Generate a small matrix with mixed ternary-like weights.
        let n = 4;
        let k = 16;
        let mut weights = vec![0.0f32; n * k];
        // Fill with a deterministic pattern.
        for i in 0..n {
            for j in 0..k {
                let idx = i * k + j;
                weights[idx] = match idx % 5 {
                    0 => 0.5,
                    1 => -0.5,
                    2 => 0.0,
                    3 => 0.7,
                    _ => -0.3,
                };
            }
        }

        let x: Vec<f32> = (0..k).map(|i| (i as f32 - 8.0) * 0.1).collect();

        // Reference: direct f32 matmul.
        let ref_output = matvec_f32_reference(&x, &weights, n, k);

        // Ternary path.
        let ternary_output = matmul_ternary(
            &x,
            &{
                let (p, _) = pack_ternary(&weights, n, k);
                p
            },
            &{
                let (_, a) = pack_ternary(&weights, n, k);
                a
            },
            n,
            k,
        );

        // The ternary quantization loses precision, so we check that the
        // results are in the same ballpark (within ~30% relative or 0.5 abs).
        for i in 0..n {
            let abs_err = (ternary_output[i] - ref_output[i]).abs();
            let scale = ref_output[i].abs().max(1.0);
            assert!(
                abs_err / scale < 0.5,
                "row {}: ternary={} ref={} err={}",
                i,
                ternary_output[i],
                ref_output[i],
                abs_err
            );
        }
    }

    #[test]
    fn test_matvec_scalar_non_multiple_of_4() {
        // k=7: ensure remainder handling works.
        let n = 2;
        let k = 7;
        let weights = vec![1.0f32; n * k];
        let (packed, alphas) = pack_ternary(&weights, n, k);

        let x: Vec<f32> = (0..k).map(|i| (i + 1) as f32).collect(); // [1..7]
        let (x_q, gamma) = quantize_activation(&x);

        let mut output = vec![0.0f32; n];
        matvec_ternary_scalar(&x_q, gamma, &packed, &alphas, n, k, &mut output);

        // sum(1..=7) = 28.0
        for i in 0..n {
            assert!(
                (output[i] - 28.0).abs() < 1.5,
                "row {}: got {}, expected ~28.0",
                i,
                output[i]
            );
        }
    }

    // -------------------------------------------------------------------
    // NEON kernel tests (aarch64 only)
    // -------------------------------------------------------------------

    #[cfg(target_arch = "aarch64")]
    mod neon_tests {
        use super::*;

        #[test]
        fn test_neon_matches_scalar_small() {
            let n = 4;
            let k = 32; // exactly one NEON chunk.
            let mut weights = vec![0.0f32; n * k];
            for i in 0..n * k {
                weights[i] = match i % 3 {
                    0 => 0.8,
                    1 => -0.6,
                    _ => 0.0,
                };
            }
            let (packed, alphas) = pack_ternary(&weights, n, k);

            let x: Vec<f32> = (0..k).map(|i| (i as f32 - 16.0) * 0.1).collect();
            let (x_q, gamma) = quantize_activation(&x);

            let mut scalar_out = vec![0.0f32; n];
            let mut neon_out = vec![0.0f32; n];

            matvec_ternary_scalar(&x_q, gamma, &packed, &alphas, n, k, &mut scalar_out);
            // SAFETY: These tests are compiled only for aarch64, where NEON is
            // available, and the buffers were built for the exact n/k dimensions.
            unsafe {
                matvec_ternary_neon(&x_q, gamma, &packed, &alphas, n, k, &mut neon_out);
            }

            for i in 0..n {
                assert!(
                    (neon_out[i] - scalar_out[i]).abs() < 1e-6,
                    "row {}: neon={} scalar={}",
                    i,
                    neon_out[i],
                    scalar_out[i]
                );
            }
        }

        #[test]
        fn test_neon_matches_scalar_large() {
            // k=256 = 8 NEON chunks.
            let n = 8;
            let k = 256;
            let mut weights = vec![0.0f32; n * k];
            for i in 0..n * k {
                // Deterministic pattern.
                weights[i] = match (i * 7 + 3) % 5 {
                    0 => 1.0,
                    1 => -1.0,
                    2 => 0.5,
                    3 => -0.5,
                    _ => 0.0,
                };
            }
            let (packed, alphas) = pack_ternary(&weights, n, k);

            let x: Vec<f32> = (0..k)
                .map(|i| ((i * 13 % 100) as f32 - 50.0) * 0.01)
                .collect();
            let (x_q, gamma) = quantize_activation(&x);

            let mut scalar_out = vec![0.0f32; n];
            let mut neon_out = vec![0.0f32; n];

            matvec_ternary_scalar(&x_q, gamma, &packed, &alphas, n, k, &mut scalar_out);
            // SAFETY: These tests are compiled only for aarch64, where NEON is
            // available, and the buffers were built for the exact n/k dimensions.
            unsafe {
                matvec_ternary_neon(&x_q, gamma, &packed, &alphas, n, k, &mut neon_out);
            }

            for i in 0..n {
                assert!(
                    (neon_out[i] - scalar_out[i]).abs() < 1e-4,
                    "row {}: neon={} scalar={}",
                    i,
                    neon_out[i],
                    scalar_out[i]
                );
            }
        }

        #[test]
        fn test_neon_matches_scalar_with_remainder() {
            // k=100: not a multiple of 32, exercises the remainder path.
            let n = 3;
            let k = 100;
            let weights: Vec<f32> = (0..n * k)
                .map(|i| match i % 4 {
                    0 => 0.9,
                    1 => -0.7,
                    2 => 0.0,
                    _ => 0.4,
                })
                .collect();
            let (packed, alphas) = pack_ternary(&weights, n, k);

            let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 0.5).collect();
            let (x_q, gamma) = quantize_activation(&x);

            let mut scalar_out = vec![0.0f32; n];
            let mut neon_out = vec![0.0f32; n];

            matvec_ternary_scalar(&x_q, gamma, &packed, &alphas, n, k, &mut scalar_out);
            // SAFETY: These tests are compiled only for aarch64, where NEON is
            // available, and the buffers were built for the exact n/k dimensions.
            unsafe {
                matvec_ternary_neon(&x_q, gamma, &packed, &alphas, n, k, &mut neon_out);
            }

            for i in 0..n {
                assert!(
                    (neon_out[i] - scalar_out[i]).abs() < 1e-4,
                    "row {}: neon={} scalar={}",
                    i,
                    neon_out[i],
                    scalar_out[i]
                );
            }
        }

        #[test]
        fn test_neon_all_zero_weights() {
            let n = 2;
            let k = 64;
            let weights = vec![0.0f32; n * k];
            let (packed, alphas) = pack_ternary(&weights, n, k);

            let x = vec![5.0f32; k];
            let (x_q, gamma) = quantize_activation(&x);

            let mut output = vec![999.0f32; n];
            // SAFETY: These tests are compiled only for aarch64, where NEON is
            // available, and the buffers were built for the exact n/k dimensions.
            unsafe {
                matvec_ternary_neon(&x_q, gamma, &packed, &alphas, n, k, &mut output);
            }

            for i in 0..n {
                assert_eq!(output[i], 0.0, "row {} should be 0, got {}", i, output[i]);
            }
        }

        #[test]
        fn test_neon_all_positive_weights() {
            let n = 1;
            let k = 64;
            let weights = vec![1.0f32; n * k]; // all +1
            let (packed, alphas) = pack_ternary(&weights, n, k);

            let x = vec![1.0f32; k]; // sum = 64
            let (x_q, gamma) = quantize_activation(&x);

            let mut output = vec![0.0f32; n];
            // SAFETY: These tests are compiled only for aarch64, where NEON is
            // available, and the buffers were built for the exact n/k dimensions.
            unsafe {
                matvec_ternary_neon(&x_q, gamma, &packed, &alphas, n, k, &mut output);
            }

            assert!(
                (output[0] - 64.0).abs() < 1.5,
                "got {}, expected ~64.0",
                output[0]
            );
        }
    }

    // -------------------------------------------------------------------
    // matmul_ternary dispatch
    // -------------------------------------------------------------------

    #[test]
    fn test_matmul_ternary_dispatch() {
        let n = 4;
        let k = 32;
        let weights = vec![1.0f32; n * k];
        let (packed, alphas) = pack_ternary(&weights, n, k);

        let x = vec![1.0f32; k];
        let output = matmul_ternary(&x, &packed, &alphas, n, k);

        assert_eq!(output.len(), n);
        for i in 0..n {
            assert!(
                (output[i] - k as f32).abs() < 1.5,
                "row {}: got {}, expected ~{}",
                i,
                output[i],
                k
            );
        }
    }

    // -------------------------------------------------------------------
    // packed_row_bytes
    // -------------------------------------------------------------------

    #[test]
    fn test_packed_row_bytes() {
        assert_eq!(packed_row_bytes(0), 0);
        assert_eq!(packed_row_bytes(1), 1);
        assert_eq!(packed_row_bytes(4), 1);
        assert_eq!(packed_row_bytes(5), 2);
        assert_eq!(packed_row_bytes(8), 2);
        assert_eq!(packed_row_bytes(9), 3);
        assert_eq!(packed_row_bytes(2560), 640); // BitNet hidden_size
    }
}
