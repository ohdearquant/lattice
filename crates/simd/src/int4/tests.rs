use super::*;

fn generate_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ ((dim as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (0..dim)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                .wrapping_add(i as u64);
            let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            unit * 2.0 - 1.0
        })
        .collect()
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn test_int4_quantize_explicit_simd_matches_scalar_and_is_dispatched() {
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    let params = Int4Params {
        scale: 7.5,
        max_abs: 1.0,
    };
    let tie_params = Int4Params {
        scale: 1.0,
        max_abs: 7.5,
    };
    let tie_input = [-7.5, -1.0, 1.0, 7.5, -1.0, 1.0, 0.0, -0.0];
    let scalar_ties = quantize_int4_scalar(&tie_input, tie_params);
    assert_eq!(scalar_ties, [0x07, 0x9f, 0x79, 0x88]);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: baseline aarch64 provides NEON; the kernel bounds every access.
    let simd_ties = unsafe { quantize_int4_neon(&tie_input, tie_params) };
    #[cfg(target_arch = "x86_64")]
    // SAFETY: AVX2 was detected above; the kernel bounds every access.
    let simd_ties = unsafe { quantize_int4_avx2(&tie_input, tie_params) };
    assert_eq!(
        simd_ties, scalar_ties,
        "explicit SIMD must preserve ties-away rounding"
    );

    for dim in [0usize, 1, 3, 4, 7, 8, 9, 31, 32, 33, 383, 384, 385] {
        let mut input = generate_vector(dim, 900 + dim as u64);
        if dim > 0 {
            input[0] = f32::NAN;
        }
        if dim > 1 {
            input[1] = f32::INFINITY;
        }
        if dim > 2 {
            input[2] = f32::NEG_INFINITY;
        }
        if dim > 3 {
            let boundary = -1.0 + 0.5 / params.scale;
            input[3] = f32::from_bits(boundary.to_bits() - 1);
        }
        if dim > 4 {
            let boundary = -1.0 + 0.5 / params.scale;
            input[4] = f32::from_bits(boundary.to_bits() + 1);
        }

        let scalar = quantize_int4_scalar(&input, params);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: baseline aarch64 provides NEON; the kernel bounds every access.
        let simd = unsafe { quantize_int4_neon(&input, params) };
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2 was detected above; the kernel bounds every access.
        let simd = unsafe { quantize_int4_avx2(&input, params) };
        assert_eq!(simd, scalar, "explicit SIMD mismatch at dim={dim}");
    }

    let input = generate_vector(385, 1_063);
    let before = INT4_QUANTIZE_SIMD_HITS.with(std::cell::Cell::get);
    let quantized = Int4Vector::from_f32(&input);
    let after = INT4_QUANTIZE_SIMD_HITS.with(std::cell::Cell::get);
    assert_eq!(
        after,
        before + 1,
        "Int4Vector::from_f32 did not execute its explicit SIMD quantizer"
    );
    assert_eq!(
        quantized.data,
        quantize_int4_scalar(&input, quantized.params)
    );
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn test_int4_max_abs_explicit_simd_matches_scalar() {
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    for dim in [0usize, 1, 3, 4, 7, 8, 9, 31, 32, 33, 383, 384, 385] {
        let mut input = generate_vector(dim, 1_100 + dim as u64);
        if dim > 0 {
            input[0] = f32::NAN;
        }
        if dim > 1 {
            input[1] = f32::INFINITY;
        }
        if dim > 2 {
            input[2] = f32::NEG_INFINITY;
        }

        let scalar = max_abs_finite_scalar(&input);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: baseline aarch64 provides NEON; the kernel bounds every access.
        let simd = unsafe { max_abs_finite_neon(&input) };
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2 was detected above; the kernel bounds every access.
        let simd = unsafe { max_abs_finite_avx2(&input) };
        assert_eq!(simd, scalar, "finite max-abs mismatch at dim={dim}");
    }

    let input = generate_vector(385, 1_063);
    let before = INT4_MAX_ABS_SIMD_HITS.with(std::cell::Cell::get);
    let params = Int4Params::from_vector(&input);
    let after = INT4_MAX_ABS_SIMD_HITS.with(std::cell::Cell::get);
    assert_eq!(
        after,
        before + 1,
        "Int4Params::from_vector did not execute its explicit SIMD reducer"
    );
    assert_eq!(params.max_abs, max_abs_finite_scalar(&input));
}

#[test]
fn test_int4_roundtrip_accuracy() {
    let original = generate_vector(384, 42);
    let quantized = Int4Vector::from_f32(&original);
    let dequantized = quantized.to_f32();

    assert_eq!(dequantized.len(), original.len());

    // INT4 has only 16 levels, so error is larger than INT8.
    // Max error should be within 1/15 of the range.
    let max_abs = original
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    let expected_max_error = 2.0 * max_abs / 15.0;

    for (i, (orig, deq)) in original.iter().zip(dequantized.iter()).enumerate() {
        let error = (orig - deq).abs();
        assert!(
            error <= expected_max_error + 1e-5,
            "INT4 roundtrip error too large at index {i}: orig={orig}, deq={deq}, error={error}, max_allowed={expected_max_error}"
        );
    }
}

#[test]
fn test_int4_packing_correctness() {
    // Verify nibble packing: even index -> high nibble, odd -> low
    let v = vec![0.5, -0.5, 0.0, 1.0]; // 4 values -> 2 packed bytes
    let q = Int4Vector::from_f32(&v);
    assert_eq!(q.data.len(), 2);
    assert_eq!(q.dims, 4);

    // Verify roundtrip preserves approximate values
    let deq = q.to_f32();
    assert_eq!(deq.len(), 4);
    // 0.5 should map to roughly the right region
    assert!((deq[0] - 0.5).abs() < 0.15, "deq[0]={}", deq[0]);
    assert!((deq[1] - (-0.5)).abs() < 0.15, "deq[1]={}", deq[1]);
}

#[test]
fn test_int4_odd_dimensions() {
    // Odd number of dimensions: last nibble has a padding zero
    let v = generate_vector(383, 77);
    let q = Int4Vector::from_f32(&v);
    assert_eq!(q.data.len(), 192); // ceil(383/2) = 192
    assert_eq!(q.dims, 383);

    let deq = q.to_f32();
    assert_eq!(deq.len(), 383);
}

#[test]
fn test_int4_zero_vector() {
    let v = vec![0.0; 384];
    let q = Int4Vector::from_f32(&v);
    let deq = q.to_f32();
    for &val in &deq {
        assert!(
            val.abs() < 1e-5,
            "Zero vector should dequantize to near-zero"
        );
    }
}

#[test]
fn test_int4_dot_product_vs_f32() {
    // Use correlated vectors so the true dot product is large relative to noise.
    // For uncorrelated random vectors, the expected dot product is ~0 while
    // quantization noise is O(dims * step^2), so relative error is unbounded.
    let a = generate_vector(384, 101);
    let b: Vec<f32> = a
        .iter()
        .enumerate()
        .map(|(i, &x)| x + 0.2 * (i as f32 * 0.3).sin())
        .collect();

    // f32 reference
    let f32_dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();

    let qa = Int4Vector::from_f32(&a);
    let qb = Int4Vector::from_f32(&b);
    let int4_dot = qa.dot_product(&qb);

    // INT4 has 16 levels; for correlated vectors the relative error should be
    // within ~15% (quantization step = 2*max_abs/15 per component).
    let rel_error = (f32_dot - int4_dot).abs() / f32_dot.abs().max(1.0);
    assert!(
        rel_error < 0.15,
        "INT4 dot product relative error too large: f32={f32_dot}, int4={int4_dot}, rel_error={rel_error}"
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_packed_scalar_matches_neon_exact() {
    // Directly compare dot_product_int4_packed_scalar against
    // dot_product_int4_neon_unrolled on integer tuples. Both return
    // (raw_dot, sum_a, sum_b) as i32 — integer domain, exact equality expected.
    // This is the executing parity proof for the non-aarch64 fallback kernel.
    for dim in [1usize, 3, 31, 127, 383, 384] {
        let a_f32 = generate_vector(dim, 500 + dim as u64);
        let b_f32 = generate_vector(dim, 600 + dim as u64);
        let qa = Int4Vector::from_f32(&a_f32);
        let qb = Int4Vector::from_f32(&b_f32);

        let scalar_result = dot_product_int4_packed_scalar(&qa.data, &qb.data, dim);
        // SAFETY: aarch64 always has NEON; data slices are correctly sized by
        // Int4Vector::from_f32 (len = dims.div_ceil(2)).
        let neon_result = unsafe { dot_product_int4_neon_unrolled(&qa.data, &qb.data, dim) };

        assert_eq!(
            scalar_result, neon_result,
            "packed_scalar vs NEON integer mismatch at dim={dim}: scalar={scalar_result:?}, neon={neon_result:?}"
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_int4_neon_matches_dequantized_scalar() {
    for dim in [1, 2, 31, 64, 127, 384, 768] {
        let a = generate_vector(dim, 501);
        let b = generate_vector(dim, 777);
        let qa = Int4Vector::from_f32(&a);
        let qb = Int4Vector::from_f32(&b);

        let a_deq = qa.to_f32();
        let b_deq = qb.to_f32();
        let expected: f32 = a_deq.iter().zip(b_deq.iter()).map(|(&x, &y)| x * y).sum();
        let got = qa.dot_product(&qb);

        assert!(
            (expected - got).abs() < 1e-4,
            "INT4 NEON mismatch for dim={dim}: expected={expected}, got={got}"
        );
    }
}

#[test]
fn test_int4_cosine_similarity() {
    let a = generate_vector(384, 301);
    let b = generate_vector(384, 302);

    let qa = Int4Vector::from_f32(&a);
    let qb = Int4Vector::from_f32(&b);
    let int4_cos = qa.cosine_similarity(&qb);

    // Compute f32 reference cosine
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let f32_cos = dot / (norm_a * norm_b);

    assert!(
        (f32_cos - int4_cos).abs() < 0.1,
        "INT4 cosine too far from f32: f32={f32_cos}, int4={int4_cos}"
    );
}

#[test]
fn test_int4_memory_savings() {
    let v = generate_vector(384, 999);
    let q = Int4Vector::from_f32(&v);

    // f32: 384 * 4 = 1536 bytes
    // INT4: ceil(384/2) = 192 bytes = 8x compression
    assert_eq!(q.data.len(), 192);
    assert_eq!(v.len() * 4, 1536);
}

#[test]
fn test_int4_nan_inf_handling() {
    let v = vec![
        1.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        0.5,
        0.0,
        -0.3,
    ];
    let q = Int4Vector::from_f32(&v);
    let deq = q.to_f32();
    assert_eq!(deq.len(), 8);
    // NaN and Inf should be treated as 0
    // The dequantized value for the "0" slot should be near -max_abs + something,
    // but the key invariant is no panics and finite output.
    for &val in &deq {
        assert!(val.is_finite(), "Dequantized value should be finite");
    }
}

// --- Issue #211 regression tests -------------------------------------------------

#[test]
fn test_int4_to_f32_short_data_returns_empty() {
    // dims=128 requires 64 bytes; supply only 4.
    let q = Int4Vector {
        dims: 128,
        data: vec![0xFFu8; 4],
        params: Int4Params {
            scale: 7.5,
            max_abs: 1.0,
        },
        norm: 1.0,
    };
    let result = q.to_f32();
    assert!(
        result.is_empty(),
        "to_f32 on malformed Int4Vector must return empty Vec"
    );
}

#[test]
fn test_int4_to_f32_exact_length_works() {
    // Exactly the right number of bytes — must succeed and not index OOB.
    let v: Vec<f32> = (0..128).map(|i| (i as f32) / 64.0 - 1.0).collect();
    let q = Int4Vector::from_f32(&v);
    let deq = q.to_f32();
    assert_eq!(deq.len(), 128);
    for &val in &deq {
        assert!(val.is_finite());
    }
}
