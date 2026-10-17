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
fn test_binary_quantize_explicit_simd_matches_scalar_and_is_dispatched() {
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    for threshold in [0.0, 0.25, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
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

            let packed_len = dim.div_ceil(8);
            let scalar = quantize_binary_scalar(&input, threshold, packed_len);
            #[cfg(target_arch = "aarch64")]
            // SAFETY: baseline aarch64 provides NEON; the kernel bounds every access.
            let simd = unsafe { quantize_binary_neon(&input, threshold, packed_len) };
            #[cfg(target_arch = "x86_64")]
            // SAFETY: AVX2 was detected above; the kernel bounds every access.
            let simd = unsafe { quantize_binary_avx2(&input, threshold, packed_len) };
            assert_eq!(
                simd, scalar,
                "explicit SIMD mismatch at dim={dim}, threshold={threshold}"
            );
        }
    }

    let input = generate_vector(385, 1_063);
    let before = BINARY_QUANTIZE_SIMD_HITS.with(std::cell::Cell::get);
    let quantized = BinaryVector::from_f32(&input);
    let after = BINARY_QUANTIZE_SIMD_HITS.with(std::cell::Cell::get);
    assert_eq!(
        after,
        before + 1,
        "BinaryVector::from_f32 did not execute its explicit SIMD quantizer"
    );
    assert_eq!(
        quantized.data,
        quantize_binary_scalar(&input, 0.0, input.len().div_ceil(8))
    );
}

#[test]
fn test_binary_quantize_basic() {
    let v = vec![0.5, -0.3, 0.0, -1.0, 1.0, 0.1, -0.1, 0.9];
    let bv = BinaryVector::from_f32(&v);
    assert_eq!(bv.data.len(), 1); // 8 dims -> 1 byte
    assert_eq!(bv.dims, 8);

    // Expected bits (MSB first): 1, 0, 1, 0, 1, 1, 0, 1 = 0b10101101 = 0xAD
    assert_eq!(bv.data[0], 0xAD, "packed bits: {:08b}", bv.data[0]);
}

#[test]
fn test_binary_roundtrip() {
    let v = vec![0.5, -0.3, 0.0, -1.0, 1.0, 0.1, -0.1, 0.9];
    let bv = BinaryVector::from_f32(&v);
    let deq = bv.to_f32();

    // Binary: positive -> +1.0, negative -> -1.0
    assert_eq!(deq, vec![1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0]);
}

#[test]
fn test_binary_hamming_distance() {
    // Same vector should have 0 Hamming distance
    let v = generate_vector(384, 42);
    let bv = BinaryVector::from_f32(&v);
    assert_eq!(bv.hamming_distance(&bv), 0);

    // Opposite sign vector should have max Hamming distance
    let neg_v: Vec<f32> = v.iter().map(|x| -x).collect();
    let neg_bv = BinaryVector::from_f32(&neg_v);
    // Some values might be exactly 0.0, which maps to +1 in both cases
    // So Hamming may not be exactly 384
    let hamming = bv.hamming_distance(&neg_bv);
    // But it should be close to 384 for random vectors
    assert!(hamming > 350, "hamming={hamming}, expected close to 384");
}

#[test]
fn test_binary_cosine_approx_identical() {
    let v = generate_vector(384, 55);
    let bv = BinaryVector::from_f32(&v);
    let cos_dist = bv.cosine_distance_approx(&bv);
    assert!(
        cos_dist.abs() < 1e-5,
        "Identical binary vectors should have 0 cosine distance, got {cos_dist}"
    );
}

#[test]
fn test_binary_cosine_approx_quality() {
    let a = generate_vector(384, 101);
    let b = generate_vector(384, 202);

    // f32 reference cosine
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let f32_cos = dot / (norm_a * norm_b);

    let ba = BinaryVector::from_f32(&a);
    let bb = BinaryVector::from_f32(&b);
    let bin_cos = ba.cosine_similarity_approx(&bb);

    // Binary is a rough approximation -- within 0.3 is acceptable for pre-filtering
    assert!(
        (f32_cos - bin_cos).abs() < 0.35,
        "Binary cosine too far from f32: f32={f32_cos}, binary={bin_cos}"
    );
}

#[test]
fn test_binary_memory_savings() {
    let v = generate_vector(384, 999);
    let bv = BinaryVector::from_f32(&v);

    // f32: 384 * 4 = 1536 bytes
    // Binary: ceil(384/8) = 48 bytes = 32x compression
    assert_eq!(bv.data.len(), 48);
}

#[test]
fn test_binary_non_multiple_of_8_dims() {
    // 385 dims -> ceil(385/8) = 49 bytes
    let v = generate_vector(385, 77);
    let bv = BinaryVector::from_f32(&v);
    assert_eq!(bv.data.len(), 49);
    assert_eq!(bv.dims, 385);

    // Roundtrip should preserve all 385 values
    let deq = bv.to_f32();
    assert_eq!(deq.len(), 385);
}

#[test]
fn test_binary_with_threshold() {
    let v = vec![0.5, 0.3, 0.1, -0.1, -0.3, -0.5, 0.7, 0.2];
    // With threshold 0.25, only values >= 0.25 map to 1
    let bv = BinaryVector::from_f32_with_threshold(&v, 0.25);
    let deq = bv.to_f32();
    // Expected: 1, 1, -1, -1, -1, -1, 1, -1
    assert_eq!(deq, vec![1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0]);
}

#[test]
fn test_binary_nan_inf_handling() {
    let v = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        -1.0,
        0.0,
        0.5,
        -0.5,
    ];
    let bv = BinaryVector::from_f32(&v);
    let deq = bv.to_f32();
    assert_eq!(deq.len(), 8);
    for &val in &deq {
        assert!(val == 1.0 || val == -1.0, "Binary should produce +/-1.0");
    }
}

#[test]
fn test_hamming_scalar_vs_neon_parity() {
    // Generate two distinct binary vectors and verify both paths give same result
    let a = generate_vector(384, 111);
    let b = generate_vector(384, 222);
    let ba = BinaryVector::from_f32(&a);
    let bb = BinaryVector::from_f32(&b);

    let scalar_result = hamming_distance_scalar(&ba.data, &bb.data, ba.dims);
    let dispatch_result = ba.hamming_distance(&bb);

    assert_eq!(
        scalar_result, dispatch_result,
        "Scalar and dispatched Hamming should match"
    );
}

// --- Issue #211 regression tests -------------------------------------------------

#[test]
fn test_hamming_short_data_returns_max() {
    // dims=128 requires 16 bytes; supply only 4 — must return u32::MAX, not OOB.
    let a = BinaryVector {
        dims: 128,
        data: vec![0xFFu8; 4],
        norm: 1.0,
    };
    let b = BinaryVector {
        dims: 128,
        data: vec![0x00u8; 4],
        norm: 1.0,
    };
    assert_eq!(
        hamming_distance_binary(&a, &b),
        u32::MAX,
        "Short data must yield u32::MAX, not an OOB read"
    );
}

#[test]
fn test_hamming_one_side_short_returns_max() {
    // a is correct length, b is too short.
    let a = BinaryVector {
        dims: 128,
        data: vec![0xFFu8; 16],
        norm: 1.0,
    };
    let b = BinaryVector {
        dims: 128,
        data: vec![0x00u8; 8],
        norm: 1.0,
    };
    assert_eq!(hamming_distance_binary(&a, &b), u32::MAX);
}

#[test]
fn test_hamming_correct_data_still_works() {
    // Verify the guard does not break the normal path.
    let v = generate_vector(128, 42);
    let bv = BinaryVector::from_f32(&v);
    assert_eq!(bv.hamming_distance(&bv), 0);
}

#[test]
fn test_binary_to_f32_short_data_returns_empty() {
    // dims=128 requires 16 bytes; supply only 4.
    let bv = BinaryVector {
        dims: 128,
        data: vec![0xFFu8; 4],
        norm: 1.0,
    };
    let result = bv.to_f32();
    assert!(
        result.is_empty(),
        "to_f32 on malformed BinaryVector must return empty Vec"
    );
}

#[test]
fn test_binary_to_f32_exact_length_works() {
    // Exactly the right number of bytes — must succeed.
    let v = generate_vector(128, 7);
    let bv = BinaryVector::from_f32(&v);
    let deq = bv.to_f32();
    assert_eq!(deq.len(), 128);
}

// --- Issue #249 regression: padding-bit masking in Hamming distance -------------

/// Vectors with 12 dimensions use 2 bytes, with 4 padding bits in byte 1 (the low
/// nibble). Two vectors that agree on all 12 real dimensions but differ in their
/// padding bits must report Hamming distance 0, not 4.
///
/// Before the fix, XOR-popcount over the raw final byte counted those 4 spurious
/// differing padding bits. After the fix, the mask `0xFF << (8 - 4) = 0xF0` zeroes
/// the low nibble before counting, giving the correct answer.
#[test]
fn test_hamming_ignores_padding_bits() {
    // 12 dims → 2 bytes; the last 4 bits of byte 1 are padding.
    // Build via pub fields so we can inject arbitrary padding.
    let clean = BinaryVector {
        dims: 12,
        // byte 1: top nibble = 4 valid dims, low nibble = 0 (zero padding)
        data: vec![0b10101010u8, 0b11110000u8],
        norm: 1.0,
    };
    let dirty = BinaryVector {
        dims: 12,
        // same valid bits in top nibble of byte 1, non-zero garbage in low-nibble padding
        data: vec![0b10101010u8, 0b11111111u8],
        norm: 1.0,
    };

    // Both vectors agree on all 12 real dimensions; Hamming distance must be 0.
    assert_eq!(
        hamming_distance_scalar(&clean.data, &dirty.data, 12),
        0,
        "scalar: padding bits must not be counted"
    );
    assert_eq!(
        clean.hamming_distance(&dirty),
        0,
        "dispatch: padding bits must not be counted"
    );

    // Cosine distance approximation must also be 0.0 for identical valid bits.
    assert_eq!(
        clean.cosine_distance_approx(&dirty),
        0.0,
        "cosine_distance_approx: padding bits must not be counted"
    );
}

/// Verifies that the partial-byte mask counts real differing bits correctly.
///
/// byte 0: 0b10101010 ^ 0b01010101 = 0b11111111 → 8 differing bits.
/// byte 1 (masked with 0xF0): (0b11110000 ^ 0b00000000) & 0xF0 = 0b11110000 → 4 bits.
/// Total: 12, which equals `dims` (every real dimension differs).
#[test]
fn test_hamming_partial_byte_count() {
    let a = BinaryVector {
        dims: 12,
        data: vec![0b10101010u8, 0b11110000u8],
        norm: 1.0,
    };
    let b = BinaryVector {
        dims: 12,
        data: vec![0b01010101u8, 0b00000000u8],
        norm: 1.0,
    };

    assert_eq!(hamming_distance_scalar(&a.data, &b.data, 12), 12);
    assert_eq!(a.hamming_distance(&b), 12);
}
