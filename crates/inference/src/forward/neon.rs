//! Native Q8_0 NEON dotprod matvec kernel.
//!
//! Computes `y[N] = x[K] @ W[N,K]^T` where:
//! - `x` is quantized to int8 with a single global scale
//! - `W` is in Q8_0 format: blocks of `[f32_scale, 32×i8]` (36 bytes per 32 weights)
//!
//! Uses ARM NEON `vdotq_s32` to process 16 int8 elements per instruction
//! WITHOUT converting to f32 first. This avoids the CPU format conversion trap
//! that made dequant-then-BLAS slower than native f32.

/// Q8_0 block: 4-byte f32 scale + 32 int8 weights = 36 bytes.
const QK8_0: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 4 + QK8_0;

#[inline]
fn load_q8_scale(block: &[u8]) -> f32 {
    f32::from_ne_bytes([block[0], block[1], block[2], block[3]])
}

fn validate_q8_args(
    x_len: usize,
    weights_len: usize,
    num_rows: usize,
    k: usize,
    output_len: usize,
) -> (usize, usize) {
    assert_eq!(k % QK8_0, 0, "k must be a multiple of 32 for Q8_0");
    assert!(x_len >= k, "x_q is shorter than k");
    let blocks = k / QK8_0;
    let row_stride = blocks
        .checked_mul(Q8_0_BLOCK_BYTES)
        .expect("row stride overflow");
    let needed = num_rows
        .checked_mul(row_stride)
        .expect("weights size overflow");
    assert!(
        weights_len >= needed,
        "weights buffer too small for {num_rows} rows x {k} cols"
    );
    assert!(output_len >= num_rows, "output buffer too small");
    (blocks, row_stride)
}

/// **Unstable**: Q8_0 quantize f32 to int8 with global scale; quantization range may change.
///
/// Quantize an f32 vector to int8 with a single global scale.
///
/// Returns `(quantized_vector, scale)` where `x[i] ≈ quantized[i] * scale`.
pub fn quantize_vec_q8(x: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = x.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs == 0.0 {
        return (vec![0; x.len()], 0.0);
    }
    let x_scale = max_abs / 127.0;
    let inv = 1.0 / x_scale;
    let x_q: Vec<i8> = x
        .iter()
        .map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (x_q, x_scale)
}

/// **Unstable**: Q8_0 matvec scalar fallback; used on non-AArch64 or for testing.
///
/// Scalar fallback for non-AArch64 or testing.
pub fn matvec_q8_scalar(
    x_q: &[i8],
    x_scale: f32,
    weights: &[u8],
    num_rows: usize,
    k: usize,
    output: &mut [f32],
) {
    let (blocks, row_stride) =
        validate_q8_args(x_q.len(), weights.len(), num_rows, k, output.len());
    output[..num_rows].fill(0.0);

    for row in 0..num_rows {
        let row_bytes = &weights[row * row_stride..(row + 1) * row_stride];
        let mut sum = 0.0f32;
        let mut x_off = 0;
        let mut w_off = 0;
        for _ in 0..blocks {
            let x_block = &x_q[x_off..x_off + QK8_0];
            let block = &row_bytes[w_off..w_off + Q8_0_BLOCK_BYTES];
            let mut dot = 0i32;
            for i in 0..QK8_0 {
                dot += (x_block[i] as i32) * (block[4 + i] as i8 as i32);
            }
            sum += (dot as f32) * load_q8_scale(block) * x_scale;
            x_off += QK8_0;
            w_off += Q8_0_BLOCK_BYTES;
        }
        output[row] = sum;
    }
}

/// NEON int8 matmul kernel: 4 output rows at a time, stable intrinsics.
///
/// Uses vmull_s8 + vpadalq_s16 instead of vdotq_s32 (unstable in stable Rust).
/// Processes 16 int8 elements per load pair without f32 conversion.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (all AArch64 does).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matvec_q8_neon_inner(
    x_q: &[i8],
    x_scale: f32,
    weights: &[u8],
    num_rows: usize,
    k: usize,
    output: &mut [f32],
) {
    use std::arch::aarch64::*;

    /// Compute i8 dot product of two 16-byte vectors using stable NEON.
    /// vmull_s8 (low 8) + vmlal_s8 (high 8) → int16x8, then vpaddlq → int32x4.
    #[inline(always)]
    unsafe fn dot_i8x16(a: int8x16_t, b: int8x16_t, acc: int32x4_t) -> int32x4_t {
        let a_lo = vget_low_s8(a);
        let a_hi = vget_high_s8(a);
        let b_lo = vget_low_s8(b);
        let b_hi = vget_high_s8(b);
        // 8 products (low) + 8 products (high) → int16x8
        let prod = vmull_s8(a_lo, b_lo);
        let prod = vmlal_s8(prod, a_hi, b_hi);
        // Pairwise widen-add int16x8 → int32x4, accumulate
        vpadalq_s16(acc, prod)
    }

    let (blocks, row_stride) =
        validate_q8_args(x_q.len(), weights.len(), num_rows, k, output.len());
    output[..num_rows].fill(0.0);

    let mut row = 0usize;

    // Process 4 rows at a time for better register utilisation.
    while row + 4 <= num_rows {
        let row0 = &weights[(row) * row_stride..(row + 1) * row_stride];
        let row1 = &weights[(row + 1) * row_stride..(row + 2) * row_stride];
        let row2 = &weights[(row + 2) * row_stride..(row + 3) * row_stride];
        let row3 = &weights[(row + 3) * row_stride..(row + 4) * row_stride];

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let mut x_off = 0usize;
        let mut w_off = 0usize;

        for _ in 0..blocks {
            let x_block = &x_q[x_off..x_off + QK8_0];
            let b0 = &row0[w_off..w_off + Q8_0_BLOCK_BYTES];
            let b1 = &row1[w_off..w_off + Q8_0_BLOCK_BYTES];
            let b2 = &row2[w_off..w_off + Q8_0_BLOCK_BYTES];
            let b3 = &row3[w_off..w_off + Q8_0_BLOCK_BYTES];

            // Load 32 int8 input values as 2×16-byte vectors.
            let xv0 = vld1q_s8(x_block[0..16].as_ptr());
            let xv1 = vld1q_s8(x_block[16..32].as_ptr());

            // Load 32 int8 weights for each of 4 rows.
            let w00 = vld1q_s8(b0[4..20].as_ptr() as *const i8);
            let w01 = vld1q_s8(b0[20..36].as_ptr() as *const i8);
            let w10 = vld1q_s8(b1[4..20].as_ptr() as *const i8);
            let w11 = vld1q_s8(b1[20..36].as_ptr() as *const i8);
            let w20 = vld1q_s8(b2[4..20].as_ptr() as *const i8);
            let w21 = vld1q_s8(b2[20..36].as_ptr() as *const i8);
            let w30 = vld1q_s8(b3[4..20].as_ptr() as *const i8);
            let w31 = vld1q_s8(b3[20..36].as_ptr() as *const i8);

            // Dot products: vmull_s8 + vmlal_s8 → int16x8, vpadalq → int32x4.
            let zero = vdupq_n_s32(0);
            let acc0 = dot_i8x16(w01, xv1, dot_i8x16(w00, xv0, zero));
            let acc1 = dot_i8x16(w11, xv1, dot_i8x16(w10, xv0, zero));
            let acc2 = dot_i8x16(w21, xv1, dot_i8x16(w20, xv0, zero));
            let acc3 = dot_i8x16(w31, xv1, dot_i8x16(w30, xv0, zero));

            // Horizontal sum of 4 int32 lanes → single i32, then scale.
            sum0 += (vaddvq_s32(acc0) as f32) * load_q8_scale(b0) * x_scale;
            sum1 += (vaddvq_s32(acc1) as f32) * load_q8_scale(b1) * x_scale;
            sum2 += (vaddvq_s32(acc2) as f32) * load_q8_scale(b2) * x_scale;
            sum3 += (vaddvq_s32(acc3) as f32) * load_q8_scale(b3) * x_scale;

            x_off += QK8_0;
            w_off += Q8_0_BLOCK_BYTES;
        }

        output[row] = sum0;
        output[row + 1] = sum1;
        output[row + 2] = sum2;
        output[row + 3] = sum3;
        row += 4;
    }

    // Tail: process remaining rows one at a time.
    while row < num_rows {
        let row_bytes = &weights[row * row_stride..(row + 1) * row_stride];
        let mut sum = 0.0f32;
        let mut x_off = 0usize;
        let mut w_off = 0usize;

        for _ in 0..blocks {
            let x_block = &x_q[x_off..x_off + QK8_0];
            let block = &row_bytes[w_off..w_off + Q8_0_BLOCK_BYTES];
            let xv0 = vld1q_s8(x_block[0..16].as_ptr());
            let xv1 = vld1q_s8(x_block[16..32].as_ptr());
            let wv0 = vld1q_s8(block[4..20].as_ptr() as *const i8);
            let wv1 = vld1q_s8(block[20..36].as_ptr() as *const i8);

            let zero = vdupq_n_s32(0);
            let acc = dot_i8x16(wv1, xv1, dot_i8x16(wv0, xv0, zero));

            sum += (vaddvq_s32(acc) as f32) * load_q8_scale(block) * x_scale;
            x_off += QK8_0;
            w_off += Q8_0_BLOCK_BYTES;
        }

        output[row] = sum;
        row += 1;
    }
}

/// **Unstable**: Q8_0 matmul with NEON dispatch; dispatch logic and format may change.
///
/// Safe wrapper: quantize x once, dispatch best kernel, return y.
///
/// ```rust,ignore
/// let y = matmul_q8_neon(&x, &packed_weights, num_rows, k);
/// ```
pub fn matmul_q8_neon(x: &[f32], weights: &[u8], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(x.len(), k, "x.len() must equal k");
    let (x_q, x_scale) = quantize_vec_q8(x);
    let mut output = vec![0.0f32; n];

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on AArch64, no runtime detection needed.
        // SAFETY: This block is compiled only for aarch64 where NEON is
        // available, and validate_q8_args checks buffer lengths and k lane width.
        unsafe {
            matvec_q8_neon_inner(&x_q, x_scale, weights, n, k, &mut output);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        matvec_q8_scalar(&x_q, x_scale, weights, n, k, &mut output);
    }

    output
}

/// **Unstable**: Q8_0 quantize into a pre-allocated buffer; may change with block format.
///
/// Quantize `x` into `x_q`, reusing its allocated capacity. Returns x_scale.
pub fn quantize_vec_q8_into(x: &[f32], x_q: &mut Vec<i8>) -> f32 {
    let max_abs = x.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs == 0.0 {
        x_q.resize(x.len(), 0);
        return 0.0;
    }
    let x_scale = max_abs / 127.0;
    let inv = 1.0 / x_scale;
    x_q.resize(x.len(), 0);
    for (dst, &v) in x_q.iter_mut().zip(x.iter()) {
        *dst = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    x_scale
}

/// **Unstable**: allocation-free Q8_0 matmul; dispatch logic may change.
///
/// Quantizes `x` into `x_q_scratch` (reusing capacity), then writes the
/// matmul result directly into `output` without allocating a new Vec.
pub fn matmul_q8_neon_into(
    x: &[f32],
    weights: &[u8],
    n: usize,
    k: usize,
    output: &mut [f32],
    x_q_scratch: &mut Vec<i8>,
) {
    assert_eq!(x.len(), k, "x.len() must equal k");
    let x_scale = quantize_vec_q8_into(x, x_q_scratch);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 always has NEON; validate_q8_args checks buffer sizes.
        unsafe {
            matvec_q8_neon_inner(x_q_scratch, x_scale, weights, n, k, output);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        matvec_q8_scalar(x_q_scratch, x_scale, weights, n, k, output);
    }
}

/// **Unstable**: pack f32 weights into Q8_0 block format; block layout may change.
///
/// Pack f32 weight matrix `[N, K]` into Q8_0 format.
///
/// Returns packed bytes where each row is `K/32` blocks of `[f32_scale, 32×i8]`.
pub fn pack_weights_q8(weights: &[f32], n: usize, k: usize) -> Vec<u8> {
    assert_eq!(weights.len(), n * k);
    assert_eq!(k % QK8_0, 0);
    let blocks_per_row = k / QK8_0;
    let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;
    let mut packed = vec![0u8; n * row_bytes];

    for row in 0..n {
        let row_data = &weights[row * k..(row + 1) * k];
        let dst = &mut packed[row * row_bytes..(row + 1) * row_bytes];

        for b in 0..blocks_per_row {
            let block_data = &row_data[b * QK8_0..(b + 1) * QK8_0];
            let max_abs = block_data
                .iter()
                .copied()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };
            let inv = if scale == 0.0 { 0.0 } else { 1.0 / scale };

            let off = b * Q8_0_BLOCK_BYTES;
            dst[off..off + 4].copy_from_slice(&scale.to_ne_bytes());
            for i in 0..QK8_0 {
                dst[off + 4 + i] = (block_data[i] * inv).round().clamp(-127.0, 127.0) as i8 as u8;
            }
        }
    }

    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_vec_zeros() {
        let (q, s) = quantize_vec_q8(&[0.0; 64]);
        assert_eq!(s, 0.0);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_quantize_vec_roundtrip() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let (q, s) = quantize_vec_q8(&x);
        assert!(s > 0.0);
        for (i, (&orig, &quant)) in x.iter().zip(q.iter()).enumerate() {
            let recovered = quant as f32 * s;
            let diff = (orig - recovered).abs();
            assert!(diff < s + 1e-6, "mismatch at {i}: {orig} vs {recovered}");
        }
    }

    #[test]
    fn test_pack_weights_roundtrip() {
        let n = 4;
        let k = 64;
        let w: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let packed = pack_weights_q8(&w, n, k);
        assert_eq!(packed.len(), n * (k / QK8_0) * Q8_0_BLOCK_BYTES);
    }

    #[test]
    fn test_scalar_matches_reference() {
        let n = 8;
        let k = 64;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 - 32.0) * 0.02).collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7 + 3) % 19) as f32 * 0.05 - 0.45)
            .collect();

        // Reference: f32 matmul
        let mut ref_out = vec![0.0f32; n];
        for row in 0..n {
            let mut sum = 0.0;
            for col in 0..k {
                sum += x[col] * w[row * k + col];
            }
            ref_out[row] = sum;
        }

        // Q8 scalar
        let (x_q, x_scale) = quantize_vec_q8(&x);
        let packed = pack_weights_q8(&w, n, k);
        let mut q8_out = vec![0.0f32; n];
        matvec_q8_scalar(&x_q, x_scale, &packed, n, k, &mut q8_out);

        for i in 0..n {
            let diff = (ref_out[i] - q8_out[i]).abs();
            let tol = 0.05 * ref_out[i].abs().max(1.0);
            assert!(
                diff < tol,
                "row {i}: ref={} q8={} diff={}",
                ref_out[i],
                q8_out[i],
                diff
            );
        }
    }

    #[test]
    fn test_neon_matches_scalar() {
        let n = 17; // not divisible by 4 — tests tail handling
        let k = 128;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 + 5) % 23) as f32 * 0.04 - 0.46)
            .collect();

        let (x_q, x_scale) = quantize_vec_q8(&x);
        let packed = pack_weights_q8(&w, n, k);

        let mut scalar_out = vec![0.0f32; n];
        matvec_q8_scalar(&x_q, x_scale, &packed, n, k, &mut scalar_out);

        let neon_out = matmul_q8_neon(&x, &packed, n, k);

        for i in 0..n {
            let diff = (scalar_out[i] - neon_out[i]).abs();
            assert!(
                diff < 1e-4,
                "row {i}: scalar={} neon={} diff={}",
                scalar_out[i],
                neon_out[i],
                diff
            );
        }
    }

    #[test]
    fn test_safe_wrapper_basic() {
        let n = 4;
        let k = 32;
        let x = vec![1.0f32; k];
        let w: Vec<f32> = (0..n * k)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let packed = pack_weights_q8(&w, n, k);
        let out = matmul_q8_neon(&x, &packed, n, k);
        assert_eq!(out.len(), n);
        // Each row alternates 0.5/-0.5, dot with all-1s = 0.0
        for &v in &out {
            assert!(v.abs() < 0.5, "expected near-zero, got {v}");
        }
    }

    #[test]
    fn test_large_matrix() {
        let n = 2048;
        let k = 2048;
        let x: Vec<f32> = (0..k).map(|i| ((i % 100) as f32 - 50.0) * 0.001).collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.01)
            .collect();
        let packed = pack_weights_q8(&w, n, k);
        let out = matmul_q8_neon(&x, &packed, n, k);
        assert_eq!(out.len(), n);
        // Smoke test: no NaN/Inf
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_matmul_q8_neon_into_matches_allocating_wrapper() {
        let n = 96;
        let k = 64;
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 7 + 3) % 19) as f32 * 0.05 - 0.45)
            .collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 + 7) % 23) as f32 * 0.04 - 0.46)
            .collect();
        let packed = pack_weights_q8(&w, n, k);

        let expected = matmul_q8_neon(&x, &packed, n, k);

        let mut out = vec![0.0f32; n];
        let mut x_q = Vec::new();
        matmul_q8_neon_into(&x, &packed, n, k, &mut out, &mut x_q);

        assert_eq!(
            expected, out,
            "matmul_q8_neon_into must produce bitwise-identical output to matmul_q8_neon"
        );
    }
}
