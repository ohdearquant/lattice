//! Edge case tests for SIMD vector operations.
//!
//! Tests critical edge cases that are easy to get wrong in SIMD implementations:
//! - Zero vectors
//! - Unit vectors
//! - Very large/small values
//! - NaN and Infinity
//! - Empty vectors
//! - Mismatched lengths
//! - Single element vectors

use lattice_embed::simd::{
    QuantizedVector, batch_cosine_similarity, batch_dot_product, cosine_similarity, dot_product,
    euclidean_distance, normalize,
};

// ============================================================================
// ZERO VECTOR TESTS
// ============================================================================

#[test]
fn test_dot_product_zero_vectors() {
    let zero = vec![0.0f32; 384];
    let nonzero = vec![1.0f32; 384];

    // Zero dot zero = 0
    assert_eq!(dot_product(&zero, &zero), 0.0);

    // Zero dot nonzero = 0
    assert_eq!(dot_product(&zero, &nonzero), 0.0);
}

#[test]
fn test_cosine_similarity_zero_vector() {
    let zero = vec![0.0f32; 384];
    let nonzero: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();

    // Cosine with zero vector should return 0 (not NaN)
    let sim = cosine_similarity(&zero, &nonzero);
    assert_eq!(sim, 0.0, "Cosine with zero vector should be 0");

    // Zero with zero should return 0 (not NaN)
    let sim_zero = cosine_similarity(&zero, &zero);
    assert_eq!(sim_zero, 0.0, "Cosine of two zero vectors should be 0");
}

#[test]
fn test_normalize_zero_vector() {
    let mut zero = vec![0.0f32; 384];
    normalize(&mut zero);

    // Normalizing zero vector should leave it unchanged (not produce NaN)
    for val in &zero {
        assert!(val.is_finite(), "Normalized zero vector should be finite");
        assert_eq!(*val, 0.0, "Zero vector should remain zero after normalize");
    }
}

#[test]
fn test_euclidean_distance_zero_vectors() {
    let zero = vec![0.0f32; 384];
    let nonzero = vec![1.0f32; 384];

    // Distance from zero to zero = 0
    assert_eq!(euclidean_distance(&zero, &zero), 0.0);

    // Distance from zero to ones = sqrt(384)
    let expected = (384.0f32).sqrt();
    let dist = euclidean_distance(&zero, &nonzero);
    assert!(
        (dist - expected).abs() < 1e-4,
        "Expected {expected}, got {dist}"
    );
}

// ============================================================================
// UNIT VECTOR TESTS
// ============================================================================

#[test]
fn test_unit_vectors_cosine_similarity() {
    // Create standard basis vectors (unit vectors)
    let e1: Vec<f32> = (0..384).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let e2: Vec<f32> = (0..384).map(|i| if i == 1 { 1.0 } else { 0.0 }).collect();
    let e1_neg: Vec<f32> = (0..384).map(|i| if i == 0 { -1.0 } else { 0.0 }).collect();

    // Identical unit vectors = 1.0
    let sim = cosine_similarity(&e1, &e1);
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "Identical unit vectors should have similarity 1.0, got {sim}"
    );

    // Orthogonal unit vectors = 0.0
    let sim_ortho = cosine_similarity(&e1, &e2);
    assert!(
        sim_ortho.abs() < 1e-5,
        "Orthogonal unit vectors should have similarity 0.0, got {sim_ortho}"
    );

    // Opposite unit vectors = -1.0
    let sim_opp = cosine_similarity(&e1, &e1_neg);
    assert!(
        (sim_opp + 1.0).abs() < 1e-5,
        "Opposite unit vectors should have similarity -1.0, got {sim_opp}"
    );
}

#[test]
fn test_normalized_dot_product_equals_cosine() {
    // For normalized vectors, dot product == cosine similarity
    let mut a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut b: Vec<f32> = (0..384).map(|i| (i as f32 * 0.2).cos()).collect();

    normalize(&mut a);
    normalize(&mut b);

    let dot = dot_product(&a, &b);
    let cos = cosine_similarity(&a, &b);

    assert!(
        (dot - cos).abs() < 1e-5,
        "For normalized vectors, dot product ({dot}) should equal cosine similarity ({cos})"
    );
}

// ============================================================================
// EXTREME VALUE TESTS
// ============================================================================

#[test]
fn test_very_small_values() {
    // Test with values small enough to avoid underflow (1e-15 range)
    let tiny = vec![1e-15f32; 384];
    let also_tiny = vec![2e-15f32; 384];

    // Should still compute correct relative values
    let dot = dot_product(&tiny, &also_tiny);
    let expected: f32 = (1e-15 * 2e-15) * 384.0;
    let abs_error = (dot - expected).abs();
    assert!(
        abs_error < 1e-28,
        "Dot product of tiny values: expected {expected:.2e}, got {dot:.2e}, error {abs_error:.2e}"
    );

    // Cosine should still work (both point same direction)
    let cos = cosine_similarity(&tiny, &also_tiny);
    assert!(
        (cos - 1.0).abs() < 1e-5,
        "Parallel tiny vectors should have cosine 1.0, got {cos}"
    );
}

#[test]
fn test_underflow_detection() {
    // Document behavior with extreme underflow (1e-30 * 1e-30 = 1e-60 -> 0 in f32)
    let extreme_tiny = vec![1e-30f32; 384];
    let also_extreme = vec![2e-30f32; 384];

    // Dot product underflows to 0
    let dot = dot_product(&extreme_tiny, &also_extreme);
    assert_eq!(dot, 0.0, "Extreme small values should underflow to 0");

    // Cosine similarity should handle this gracefully (not NaN)
    let cos = cosine_similarity(&extreme_tiny, &also_extreme);
    // May be 0, NaN, or 1.0 depending on implementation - just check it's finite or zero-handled
    assert!(
        cos == 0.0 || cos.is_finite(),
        "Underflow case should not produce unexpected value, got {cos}"
    );
}

#[test]
fn test_moderately_large_values() {
    // Test with values large enough to be meaningful but not overflow
    // f32::MAX is ~3.4e38, so 1e15^2 * 384 = ~3.8e32 is safe
    let large = vec![1e15f32; 384];
    let also_large = vec![1e15f32; 384];

    // Cosine of identical direction should be 1.0
    let cos = cosine_similarity(&large, &also_large);
    assert!(
        (cos - 1.0).abs() < 1e-4,
        "Identical large vectors should have cosine 1.0, got {cos}"
    );
}

#[test]
fn test_overflow_detection() {
    // Document behavior with extreme overflow (1e30^2 * 384 -> inf)
    let extreme_large = vec![1e30f32; 384];
    let also_extreme = vec![1e30f32; 384];

    // Dot product overflows to infinity
    let dot = dot_product(&extreme_large, &also_extreme);
    assert!(
        dot.is_infinite(),
        "Extreme large values should overflow to infinity, got {dot}"
    );

    // Cosine similarity produces NaN (inf / inf) - this is expected mathematical behavior
    let cos = cosine_similarity(&extreme_large, &also_extreme);
    assert!(
        cos.is_nan(),
        "Overflow case produces NaN (inf/inf), got {cos}"
    );
}

#[test]
fn test_mixed_magnitude_values() {
    // Vector with both large and small components
    let mut mixed: Vec<f32> = (0..384)
        .map(|i| if i % 2 == 0 { 1e-10 } else { 1e10 })
        .collect();
    let original = mixed.clone();

    normalize(&mut mixed);

    // Should produce finite, normalized vector
    let norm: f32 = mixed.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Mixed magnitude vector should normalize to unit length, got {norm}"
    );

    // All values should be finite
    for (i, val) in mixed.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Value at index {i} should be finite, got {val} (original: {})",
            original[i]
        );
    }
}

// ============================================================================
// NAN AND INFINITY TESTS
// ============================================================================

#[test]
fn test_nan_handling_dot_product() {
    let normal: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut with_nan = normal.clone();
    with_nan[100] = f32::NAN;

    // NaN in input should propagate to output
    let result = dot_product(&with_nan, &normal);
    assert!(result.is_nan(), "Dot product with NaN should produce NaN");
}

#[test]
fn test_nan_handling_cosine() {
    let normal: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut with_nan = normal.clone();
    with_nan[100] = f32::NAN;

    let result = cosine_similarity(&with_nan, &normal);
    assert!(
        result.is_nan(),
        "Cosine similarity with NaN should produce NaN"
    );
}

#[test]
fn test_infinity_handling() {
    let normal: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut with_inf = normal.clone();
    with_inf[100] = f32::INFINITY;

    // Infinity should propagate appropriately
    let dot = dot_product(&with_inf, &normal);
    assert!(
        dot.is_infinite() || dot.is_nan(),
        "Dot product with infinity should be infinite or NaN"
    );
}

// ============================================================================
// EMPTY AND EDGE LENGTH TESTS
// ============================================================================

#[test]
fn test_empty_vectors() {
    let empty: Vec<f32> = vec![];

    // Empty vectors should handle gracefully
    let dot = dot_product(&empty, &empty);
    assert_eq!(dot, 0.0, "Dot product of empty vectors should be 0");

    let cos = cosine_similarity(&empty, &empty);
    assert_eq!(cos, 0.0, "Cosine of empty vectors should be 0");

    let dist = euclidean_distance(&empty, &empty);
    assert_eq!(dist, 0.0, "Euclidean distance of empty vectors should be 0");
}

#[test]
fn test_single_element_vectors() {
    let a = vec![3.0f32];
    let b = vec![4.0f32];

    let dot = dot_product(&a, &b);
    assert!((dot - 12.0).abs() < 1e-5, "Expected 12.0, got {dot}");

    let cos = cosine_similarity(&a, &b);
    assert!(
        (cos - 1.0).abs() < 1e-5,
        "Parallel 1D vectors should have cosine 1.0"
    );

    let dist = euclidean_distance(&a, &b);
    assert!(
        (dist - 1.0).abs() < 1e-5,
        "Expected distance 1.0, got {dist}"
    );
}

#[test]
fn test_mismatched_length_vectors() {
    let a = vec![1.0f32; 384];
    let b = vec![1.0f32; 256];

    // Mismatched lengths should return safe values (not panic)
    let dot = dot_product(&a, &b);
    assert_eq!(
        dot, 0.0,
        "Mismatched lengths should return 0 for dot product"
    );

    let cos = cosine_similarity(&a, &b);
    assert_eq!(cos, 0.0, "Mismatched lengths should return 0 for cosine");

    let dist = euclidean_distance(&a, &b);
    assert_eq!(
        dist,
        f32::MAX,
        "Mismatched lengths should return MAX for distance"
    );
}

// ============================================================================
// ALIGNMENT EDGE CASES (SIMD boundary conditions)
// ============================================================================

#[test]
fn test_simd_alignment_boundaries() {
    // Test dimensions around SIMD chunk boundaries
    // AVX2: 8 floats, NEON: 4 floats, with 4x unrolling = 32 (AVX2) or 16 (NEON)
    let test_dims = [
        1, 2, 3, 4, 5, 6, 7, 8, // Single NEON vector and below
        9, 15, 16, 17, // NEON chunk boundary
        31, 32, 33, // AVX2 unrolled chunk boundary
        63, 64, 65, // Double AVX2 chunk
        127, 128, 129, // Larger boundaries
        383, 384, 385, // Common embedding dimension
        511, 512, 513, // Power of 2 boundary
        767, 768, 769, // BGE-base dimension
        1023, 1024, 1025, // BGE-large dimension
    ];

    for &dim in &test_dims {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.17).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).cos()).collect();

        // Compute reference scalar results
        let dot_expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_expected = if norm_a > 0.0 && norm_b > 0.0 {
            dot_expected / (norm_a * norm_b)
        } else {
            0.0
        };

        // Test SIMD implementations match
        let dot = dot_product(&a, &b);
        assert!(
            (dot - dot_expected).abs() < 1e-3 * dot_expected.abs().max(1.0),
            "Dim {dim}: dot product mismatch: {dot} vs {dot_expected}"
        );

        let cos = cosine_similarity(&a, &b);
        assert!(
            (cos - cos_expected).abs() < 1e-4,
            "Dim {dim}: cosine mismatch: {cos} vs {cos_expected}"
        );

        // Test normalize produces unit vector
        let mut v = a.clone();
        normalize(&mut v);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a > 1e-10 {
            assert!(
                (norm_after - 1.0).abs() < 1e-5,
                "Dim {dim}: normalize failed, got norm {norm_after}"
            );
        }
    }
}

#[test]
fn test_unaligned_offset_dot_product_matches_scalar() {
    #[repr(align(64))]
    struct AlignedBuffer<const N: usize>([f32; N]);

    const DIM: usize = 384;
    let mut a_storage = AlignedBuffer([0.0; DIM + 1]);
    let mut b_storage = AlignedBuffer([0.0; DIM + 1]);

    for i in 0..=DIM {
        a_storage.0[i] = (i as f32 * 0.17).sin() * 0.5;
        b_storage.0[i] = (i as f32 * 0.11).cos() * 0.25;
    }

    let a = &a_storage.0[1..];
    let b = &b_storage.0[1..];
    assert_ne!((a.as_ptr() as usize) % 64, 0);
    assert_ne!((b.as_ptr() as usize) % 64, 0);

    let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let actual = dot_product(a, b);
    let tolerance = 1e-4 * expected.abs().max(1.0);

    assert!(
        (actual - expected).abs() <= tolerance,
        "unaligned offset dot product mismatch: actual={actual}, expected={expected}"
    );
}

// ============================================================================
// NEGATIVE VALUE TESTS
// ============================================================================

#[test]
fn test_all_negative_values() {
    let neg: Vec<f32> = (0..384)
        .map(|i| -((i as f32 * 0.1).sin().abs() + 0.1))
        .collect();

    // Cosine with itself should still be 1.0
    let cos = cosine_similarity(&neg, &neg);
    assert!(
        (cos - 1.0).abs() < 1e-5,
        "Negative vector cosine with itself should be 1.0, got {cos}"
    );

    // Normalize should work
    let mut normalized = neg.clone();
    normalize(&mut normalized);
    let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Normalized negative vector should have unit length"
    );
}

#[test]
fn test_opposite_vectors() {
    let pos: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();
    let neg: Vec<f32> = pos.iter().map(|x| -x).collect();

    let cos = cosine_similarity(&pos, &neg);
    assert!(
        (cos + 1.0).abs() < 1e-5,
        "Opposite vectors should have cosine -1.0, got {cos}"
    );

    let dist = euclidean_distance(&pos, &neg);
    // Distance = 2 * norm(pos)
    let norm_pos: f32 = pos.iter().map(|x| x * x).sum::<f32>().sqrt();
    let expected_dist = 2.0 * norm_pos;
    assert!(
        (dist - expected_dist).abs() < 1e-3,
        "Distance between opposite vectors: expected {expected_dist}, got {dist}"
    );
}

// ============================================================================
// BATCH OPERATION EDGE CASES
// ============================================================================

#[test]
fn test_batch_empty() {
    let pairs: Vec<(&[f32], &[f32])> = vec![];

    let cos_results = batch_cosine_similarity(&pairs);
    assert!(cos_results.is_empty());

    let dot_results = batch_dot_product(&pairs);
    assert!(dot_results.is_empty());
}

#[test]
fn test_batch_single_pair() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];
    let pairs: Vec<(&[f32], &[f32])> = vec![(&a, &b)];

    let cos_results = batch_cosine_similarity(&pairs);
    assert_eq!(cos_results.len(), 1);
    assert!((cos_results[0] - cosine_similarity(&a, &b)).abs() < 1e-6);

    let dot_results = batch_dot_product(&pairs);
    assert_eq!(dot_results.len(), 1);
    assert!((dot_results[0] - dot_product(&a, &b)).abs() < 1e-6);
}

#[test]
fn test_batch_with_varying_lengths() {
    // Different pairs with different lengths (some mismatched)
    let a1 = vec![1.0f32; 10];
    let b1 = vec![2.0f32; 10];
    let a2 = vec![1.0f32; 20];
    let b2 = vec![2.0f32; 15]; // Mismatched

    let pairs: Vec<(&[f32], &[f32])> = vec![(&a1, &b1), (&a2, &b2)];

    let results = batch_cosine_similarity(&pairs);
    assert_eq!(results.len(), 2);
    assert!((results[0] - 1.0).abs() < 1e-5, "Parallel vectors");
    assert_eq!(results[1], 0.0, "Mismatched lengths should return 0");
}

// ============================================================================
// QUANTIZATION EDGE CASES
// ============================================================================

#[test]
fn test_quantization_zero_vector() {
    let zero = vec![0.0f32; 384];
    let quantized = QuantizedVector::from_f32(&zero);

    // All quantized values should be 0
    for val in &quantized.data {
        assert_eq!(*val, 0, "Zero vector should quantize to all zeros");
    }

    // Norm should be 0
    assert_eq!(quantized.norm, 0.0, "Zero vector norm should be 0");

    // Dequantized should be zeros
    let dequantized = quantized.to_f32();
    for val in &dequantized {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_quantization_single_value_vector() {
    let uniform = vec![0.5f32; 384];
    let quantized = QuantizedVector::from_f32(&uniform);

    // All quantized values should be the same
    let first = quantized.data[0];
    for val in &quantized.data {
        assert_eq!(*val, first, "Uniform vector should quantize uniformly");
    }
}

#[test]
fn test_quantization_max_values() {
    // Vector with values at quantization limits
    let max_val = vec![1.0f32; 384];
    let quantized = QuantizedVector::from_f32(&max_val);

    // All should quantize to 127 (max int8 for symmetric quantization)
    for val in &quantized.data {
        assert_eq!(*val, 127, "Max values should quantize to 127");
    }
}

#[test]
fn test_quantization_min_values() {
    let min_val = vec![-1.0f32; 384];
    let quantized = QuantizedVector::from_f32(&min_val);

    // All should quantize to -127 (min int8 for symmetric quantization)
    for val in &quantized.data {
        assert_eq!(*val, -127, "Min values should quantize to -127");
    }
}

#[test]
fn test_quantization_with_nan() {
    let mut with_nan = vec![0.5f32; 384];
    with_nan[100] = f32::NAN;

    let quantized = QuantizedVector::from_f32(&with_nan);

    // NaN should be treated as 0
    assert_eq!(quantized.data[100], 0, "NaN should quantize to 0");

    // Norm should exclude NaN
    assert!(
        quantized.norm.is_finite(),
        "Norm should be finite even with NaN input"
    );
}

#[test]
fn test_quantization_with_infinity() {
    let mut with_inf = vec![0.5f32; 384];
    with_inf[100] = f32::INFINITY;

    let quantized = QuantizedVector::from_f32(&with_inf);

    // Infinity should be treated as 0
    assert_eq!(quantized.data[100], 0, "Infinity should quantize to 0");
}

#[test]
fn test_quantized_dot_product_zero_vectors() {
    let zero = vec![0.0f32; 384];
    let nonzero: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();

    let q_zero = QuantizedVector::from_f32(&zero);
    let q_nonzero = QuantizedVector::from_f32(&nonzero);

    let dot = q_zero.dot_product(&q_nonzero);
    assert_eq!(dot, 0.0, "Dot product with quantized zero should be 0");
}

#[test]
fn test_quantized_cosine_zero_norm() {
    let zero = vec![0.0f32; 384];
    let nonzero: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();

    let q_zero = QuantizedVector::from_f32(&zero);
    let q_nonzero = QuantizedVector::from_f32(&nonzero);

    // Cosine with zero vector should be 0 (not NaN)
    let cos = q_zero.cosine_similarity(&q_nonzero);
    assert_eq!(cos, 0.0, "Cosine with zero quantized vector should be 0");
}

#[test]
fn test_quantized_mismatched_lengths() {
    let a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();

    let q_a = QuantizedVector::from_f32(&a);
    let q_b = QuantizedVector::from_f32(&b);

    let dot = q_a.dot_product(&q_b);
    assert_eq!(
        dot, 0.0,
        "Mismatched quantized vector lengths should return 0"
    );
}
