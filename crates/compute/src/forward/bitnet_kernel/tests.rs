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
    assert_eq!(decode_ternary(0b11_10_01_00), 0);
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

    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

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
    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

    assert!(alphas[0] > 0.0);
    assert_eq!(packed.len(), packed_row_bytes(k)); // ceil(7/4) = 2 bytes

    for j in 0..k {
        assert_eq!(unpack_weight(&packed, k, 0, j), 1, "col={j}");
    }
}

#[test]
fn pack_ternary_rejects_non_finite_weights() {
    for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let result = pack_ternary(&[0.25, non_finite, -0.5, 1.0], 1, 4);
        assert!(
            matches!(result, Err(InferenceError::InvalidInput(_))),
            "non-finite weight {non_finite} must be rejected"
        );
    }
}

#[test]
fn pack_ternary_rejects_mismatched_geometry() {
    let result = pack_ternary(&[0.25, -0.5, 1.0], 1, 4);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "weights whose length differs from n*k must be rejected"
    );
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
                "index {i}: original={original} reconstructed={reconstructed} rel_error={rel_error}"
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
    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

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
    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

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
    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

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
    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

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
            let (p, _) = pack_ternary(&weights, n, k).expect("finite test weights");
            p
        },
        &{
            let (_, a) = pack_ternary(&weights, n, k).expect("finite test weights");
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
    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

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
        let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

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
        let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

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
        let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

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

    // ADR-080 C4 held finding: `matvec_ternary_neon` previously validated with
    // `debug_assert!` only, independently of the scalar path. Shared
    // `validate_ternary_matvec_args` now gates both; release-active here matters
    // most since this kernel reads via `get_unchecked`.
    #[test]
    #[should_panic(expected = "x_q too short for k")]
    fn matvec_ternary_neon_rejects_short_x_q() {
        let n = 2;
        let k = 32;
        let weights = vec![1.0f32; n * k];
        let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");
        let x_q = vec![1i8; k - 1]; // too short
        let mut output = vec![0.0f32; n];
        // SAFETY: never reached — the validator panics before any `get_unchecked` read.
        unsafe {
            matvec_ternary_neon(&x_q, 1.0, &packed, &alphas, n, k, &mut output);
        }
    }

    #[test]
    fn test_neon_all_zero_weights() {
        let n = 2;
        let k = 64;
        let weights = vec![0.0f32; n * k];
        let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

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
        let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

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
    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");

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
// validate_ternary_matvec_args: release-active rejection (ADR-080 C4 held finding —
// the scalar path previously silently truncated a too-short `x`/`x_q` via
// `.take(k).skip(rem_start)` in the remainder loop instead of panicking).
// -------------------------------------------------------------------

#[test]
#[should_panic(expected = "x_q too short for k")]
fn matmul_ternary_rejects_short_activation() {
    let n = 2;
    let k = 32;
    let weights = vec![1.0f32; n * k];
    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");
    let x = vec![1.0f32; k - 1]; // too short
    let _ = matmul_ternary(&x, &packed, &alphas, n, k);
}

#[test]
#[should_panic(expected = "alphas too short for n")]
fn matmul_ternary_rejects_short_alphas() {
    let n = 2;
    let k = 32;
    let weights = vec![1.0f32; n * k];
    let (packed, _alphas) = pack_ternary(&weights, n, k).expect("finite test weights");
    let x = vec![1.0f32; k];
    let short_alphas = vec![1.0f32; n - 1];
    let _ = matmul_ternary(&x, &packed, &short_alphas, n, k);
}

#[test]
#[should_panic(expected = "packed_w too short for n*packed_row_bytes")]
fn matmul_ternary_rejects_short_packed_w() {
    let n = 2;
    let k = 32;
    let weights = vec![1.0f32; n * k];
    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");
    let short_packed = &packed[..packed.len() - 1];
    let x = vec![1.0f32; k];
    let _ = matmul_ternary(&x, short_packed, &alphas, n, k);
}

#[test]
#[should_panic(expected = "x_q too short for k")]
fn matvec_ternary_scalar_rejects_short_x_q() {
    let n = 2;
    let k = 32;
    let weights = vec![1.0f32; n * k];
    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");
    let x_q = vec![1i8; k - 1]; // too short
    let mut output = vec![0.0f32; n];
    matvec_ternary_scalar(&x_q, 1.0, &packed, &alphas, n, k, &mut output);
}

#[test]
#[should_panic(expected = "x_q too short for k")]
fn matvec_ternary_scalar_rejects_short_x_q_remainder_path() {
    // #796 round 1 finding 4: k=32/len=31 above triggers an incidental native
    // out-of-bounds panic from the full-byte loop (`x_q[31]`) rather than exercising the
    // silent-truncation remainder path (k % 4 != 0, lines ~253-261 above), which only
    // runs for `rem_start < k`. Use k=5 (rem_start=4, one remainder weight) with
    // x_q.len()=4 so, with the validator reverse-mutated out, the remainder loop's
    // `.take(k).skip(rem_start)` would silently iterate zero elements and the function
    // would return normally instead of panicking — the guard is what makes this red.
    let n = 2;
    let k = 5;
    let weights = vec![1.0f32; n * k];
    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");
    let x_q = vec![1i8; k - 1]; // too short, ends inside the remainder range
    let mut output = vec![0.0f32; n];
    matvec_ternary_scalar(&x_q, 1.0, &packed, &alphas, n, k, &mut output);
}

#[test]
fn matvec_ternary_scalar_accepts_oversized_buffers() {
    let n = 2;
    let k = 32;
    let weights = vec![1.0f32; n * k];
    let (packed, alphas) = pack_ternary(&weights, n, k).expect("finite test weights");
    let x_q = vec![1i8; k + 4]; // oversized
    let mut output = vec![0.0f32; n + 4]; // oversized
    matvec_ternary_scalar(&x_q, 1.0, &packed, &alphas, n, k, &mut output);
    for row in output.iter().take(n) {
        assert!((*row - k as f32).abs() < 1.5);
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
