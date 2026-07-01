//! CPU decode-path GEMV kernel for W3-packed dense MLP weights (issue #420).
use crate::error::InferenceError;
use crate::weights::q4_weights::q4_f16_to_f32;
use crate::weights::w3_weights::{W3_BLOCK_SIZE, W3_GROUP_SIZE, W3_PACKED_BYTES};

/// Compute `y[0..n] = x[0..k] @ W3[n, k]^T` with f32 accumulation.
///
/// `w3` is raw W3 payload bytes only (not the file header): row-major
/// `[n][k / 32][W3_BLOCK_SIZE]`, where each 16-byte block holds a 2-byte f16
/// scale, a 2-byte f16 bias, and 12 bytes of sequential LSB-first packed
/// 3-bit codes (see [`crate::weights::w3_weights`]).
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if `k` is `0` or not a multiple
/// of the W3 group size (32), or if `x`, `y`, or `w3` are shorter than the
/// dimensions require.
pub fn gemv_w3_decode(
    x: &[f32],
    w3: &[u8],
    y: &mut [f32],
    n: usize,
    k: usize,
) -> Result<(), InferenceError> {
    if k == 0 || k % W3_GROUP_SIZE != 0 {
        return Err(InferenceError::InvalidInput(format!(
            "gemv_w3_decode: k={k} must be a nonzero multiple of the W3 group size \
             ({W3_GROUP_SIZE})"
        )));
    }
    if x.len() < k {
        return Err(InferenceError::InvalidInput(format!(
            "gemv_w3_decode: x has length {} but k={k} requires at least that many elements",
            x.len()
        )));
    }
    if y.len() < n {
        return Err(InferenceError::InvalidInput(format!(
            "gemv_w3_decode: y has length {} but n={n} requires at least that many elements",
            y.len()
        )));
    }
    let blocks_per_row = k / W3_GROUP_SIZE;
    let row_bytes = blocks_per_row * W3_BLOCK_SIZE;
    let required_w3_bytes = n * row_bytes;
    if w3.len() < required_w3_bytes {
        return Err(InferenceError::InvalidInput(format!(
            "gemv_w3_decode: w3 payload has {} bytes but n={n}, k={k} requires at least {} bytes",
            w3.len(),
            required_w3_bytes
        )));
    }

    for r in 0..n {
        let row = &w3[r * row_bytes..(r + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &row[b * W3_BLOCK_SIZE..(b + 1) * W3_BLOCK_SIZE];
            let scale = q4_f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let bias = q4_f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let packed: &[u8; W3_PACKED_BYTES] = block[4..4 + W3_PACKED_BYTES].try_into().unwrap();

            let x_block = &x[b * W3_GROUP_SIZE..(b + 1) * W3_GROUP_SIZE];
            let mut code_dot = 0.0f32;
            let mut x_sum = 0.0f32;
            for i in 0..W3_GROUP_SIZE {
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
                let code = ((word >> shift) & 0x7) as u8;
                let xv = x_block[i];
                code_dot += f32::from(code) * xv;
                x_sum += xv;
            }
            acc += code_dot * scale + x_sum * bias;
        }
        y[r] = acc;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::cpu::matmul_bt_scalar;
    use crate::weights::w3_weights::{W3Tensor, dequantize_w3_to_f32, quantize_f32_to_w3};

    fn tensor_to_raw_payload(tensor: &W3Tensor) -> Vec<u8> {
        let mut out = Vec::with_capacity(tensor.blocks.len() * W3_BLOCK_SIZE);
        for block in &tensor.blocks {
            out.extend_from_slice(&block.scale.to_le_bytes());
            out.extend_from_slice(&block.bias.to_le_bytes());
            out.extend_from_slice(&block.packed);
        }
        out
    }

    /// Simple xorshift-style PRNG so tests stay deterministic without adding
    /// a `rand` dependency to this module's test path.
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    }

    fn random_vals(seed: u64, count: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut state = seed.wrapping_add(1);
        (0..count)
            .map(|_| {
                let bits = lcg_next(&mut state);
                let unit = (bits >> 11) as f64 / (1u64 << 53) as f64;
                lo + (hi - lo) * unit as f32
            })
            .collect()
    }

    fn quantize_row_major(n: usize, k: usize, seed: u64) -> (Vec<f32>, W3Tensor) {
        let data = random_vals(seed, n * k, -2.0, 2.0);
        let tensor = quantize_f32_to_w3(&data, &[n, k]).unwrap();
        (data, tensor)
    }

    // Measured bound (tester, issue #420): running this test's full n x k
    // sweep (n in {1,2,5,16}, k in {32,64,96,128,256}, values in [-2,2], 20
    // seeds/combination = 400 samples) on this machine gave observed maxima
    // of overall_max_abs_diff=2.3841858e-5 and overall_max_rel_diff=2.0197e-3
    // (these two numbers are each maxed independently over all 400 samples,
    // so they are not necessarily from the same sample -- every individual
    // sample satisfies max_abs_diff<=1e-4 OR max_rel_diff<=1e-5, which is
    // what the per-sample assert below checks). The abs bound (1e-4) is
    // ~4x looser than the observed abs max; the rel bound (1e-5) is
    // deliberately not the binding constraint here since small-magnitude
    // reference values make relative error noisy -- the abs bound is what
    // actually carries the guarantee for this kernel. Run with
    // `--nocapture` to reproduce and see the printed measurement.
    #[test]
    fn test_gemv_w3_decode_matches_dequantized_reference() {
        let mut overall_max_abs = 0.0f32;
        let mut overall_max_rel = 0.0f32;
        for &n in &[1usize, 2, 5, 16] {
            for &k in &[32usize, 64, 96, 128, 256] {
                for seed_offset in 0..20u64 {
                    let seed = (n * 1000 + k) as u64 * 1000 + seed_offset;
                    let (_orig, tensor) = quantize_row_major(n, k, seed);
                    let w3_bytes = tensor_to_raw_payload(&tensor);
                    let w_deq = dequantize_w3_to_f32(&tensor);

                    let x = random_vals(999 + seed, k, -2.0, 2.0);
                    let mut y = vec![0.0f32; n];
                    gemv_w3_decode(&x, &w3_bytes, &mut y, n, k).unwrap();

                    let mut y_ref = vec![0.0f32; n];
                    matmul_bt_scalar(&x, &w_deq, &mut y_ref, 1, k, n);

                    let mut max_abs_diff = 0.0f32;
                    let mut max_rel_diff = 0.0f32;
                    for i in 0..n {
                        let abs_diff = (y[i] - y_ref[i]).abs();
                        max_abs_diff = max_abs_diff.max(abs_diff);
                        let rel_diff = abs_diff / y_ref[i].abs().max(1e-6);
                        max_rel_diff = max_rel_diff.max(rel_diff);
                    }
                    overall_max_abs = overall_max_abs.max(max_abs_diff);
                    overall_max_rel = overall_max_rel.max(max_rel_diff);
                    assert!(
                        max_abs_diff <= 1e-4 || max_rel_diff <= 1e-5,
                        "n={n} k={k} seed={seed}: max_abs_diff={max_abs_diff:.8} \
                         max_rel_diff={max_rel_diff:.8} exceeds the honest bound"
                    );
                }
            }
        }
        eprintln!(
            "gemv_w3_decode differential sweep: overall_max_abs_diff={overall_max_abs:.8e} \
             overall_max_rel_diff={overall_max_rel:.8e} (bound: 1e-4 abs / 1e-5 rel)"
        );
    }

    #[test]
    fn test_gemv_w3_decode_rejects_k_zero() {
        let x = [0.0f32; 1];
        let w3 = [0u8; 16];
        let mut y = [0.0f32; 1];
        assert!(gemv_w3_decode(&x, &w3, &mut y, 1, 0).is_err());
    }

    #[test]
    fn test_gemv_w3_decode_rejects_k_not_multiple_of_group_size() {
        let x = [0.0f32; 33];
        let w3 = [0u8; 16];
        let mut y = [0.0f32; 1];
        assert!(gemv_w3_decode(&x, &w3, &mut y, 1, 33).is_err());
    }

    #[test]
    fn test_gemv_w3_decode_rejects_short_x() {
        let x = [0.0f32; 16];
        let w3 = [0u8; 16];
        let mut y = [0.0f32; 1];
        assert!(gemv_w3_decode(&x, &w3, &mut y, 1, 32).is_err());
    }

    #[test]
    fn test_gemv_w3_decode_rejects_short_y() {
        let x = [0.0f32; 32];
        let w3 = [0u8; 16];
        let mut y: [f32; 0] = [];
        assert!(gemv_w3_decode(&x, &w3, &mut y, 1, 32).is_err());
    }

    #[test]
    fn test_gemv_w3_decode_rejects_short_w3_payload() {
        let x = [0.0f32; 32];
        let w3 = [0u8; 8]; // needs 16
        let mut y = [0.0f32; 1];
        assert!(gemv_w3_decode(&x, &w3, &mut y, 1, 32).is_err());
    }

    #[test]
    fn test_gemv_w3_decode_mutation_flips_result() {
        let (_orig, tensor) = quantize_row_major(1, 64, 42);
        let mut w3_bytes = tensor_to_raw_payload(&tensor);
        let x = random_vals(7, 64, -2.0, 2.0);

        let mut y_before = [0.0f32; 1];
        gemv_w3_decode(&x, &w3_bytes, &mut y_before, 1, 64).unwrap();

        // Flip one byte inside the packed codes of the first block (offset 4..16).
        w3_bytes[4] ^= 0xff;

        let mut y_after = [0.0f32; 1];
        gemv_w3_decode(&x, &w3_bytes, &mut y_after, 1, 64).unwrap();

        let diff = (y_before[0] - y_after[0]).abs();
        assert!(
            diff > 1e-4,
            "expected mutation to change the result by more than 1e-4, got diff={diff:.8}"
        );
    }
}
