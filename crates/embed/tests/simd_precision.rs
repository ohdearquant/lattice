//! Numerical precision tests for SIMD vector operations.
//!
//! These tests verify that SIMD implementations maintain acceptable precision
//! compared to scalar reference implementations across various input distributions.

#![allow(clippy::manual_range_contains)]
#![allow(clippy::needless_range_loop)]

use lattice_embed::simd::{
    QuantizationParams, QuantizedVector, cosine_similarity, cosine_similarity_i8, dot_product,
    dot_product_i8, euclidean_distance, normalize,
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Generate deterministic "random" vector using hash-based approach.
fn generate_vector(dim: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..dim)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            let hash = hasher.finish();
            // Map to [-1, 1]
            (hash as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Scalar reference implementation of dot product.
fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Scalar reference implementation of cosine similarity.
fn scalar_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Scalar reference implementation of normalization.
fn scalar_normalize(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv_norm = 1.0 / norm;
        vector.iter_mut().for_each(|x| *x *= inv_norm);
    }
}

/// Scalar reference implementation of Euclidean distance.
fn scalar_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

// ============================================================================
// DOT PRODUCT PRECISION TESTS
// ============================================================================

#[test]
fn test_dot_product_precision_uniform_distribution() {
    // Test with uniformly distributed values in [-1, 1]
    for dim in [64, 128, 256, 384, 512, 768, 1024] {
        let a = generate_vector(dim, 12345);
        let b = generate_vector(dim, 67890);

        let simd = dot_product(&a, &b);
        let scalar = scalar_dot_product(&a, &b);

        let rel_error = if scalar.abs() > 1e-10 {
            ((simd - scalar) / scalar).abs()
        } else {
            (simd - scalar).abs()
        };

        assert!(
            rel_error < 1e-5,
            "Dim {dim}: dot product relative error too large: {rel_error} (simd={simd}, scalar={scalar})"
        );
    }
}

#[test]
fn test_dot_product_precision_small_values() {
    // Test with very small values to check for underflow issues
    for dim in [128, 384, 1024] {
        let a: Vec<f32> = generate_vector(dim, 111)
            .iter()
            .map(|x| x * 1e-10)
            .collect();
        let b: Vec<f32> = generate_vector(dim, 222)
            .iter()
            .map(|x| x * 1e-10)
            .collect();

        let simd = dot_product(&a, &b);
        let scalar = scalar_dot_product(&a, &b);

        // For very small values, use absolute error
        let abs_error = (simd - scalar).abs();
        assert!(
            abs_error < 1e-15,
            "Dim {dim}: small value dot product error: {abs_error}"
        );
    }
}

#[test]
fn test_dot_product_precision_large_values() {
    // Test with large values to check for overflow handling
    for dim in [128, 384, 1024] {
        let a: Vec<f32> = generate_vector(dim, 333).iter().map(|x| x * 1e10).collect();
        let b: Vec<f32> = generate_vector(dim, 444).iter().map(|x| x * 1e10).collect();

        let simd = dot_product(&a, &b);
        let scalar = scalar_dot_product(&a, &b);

        let rel_error = ((simd - scalar) / scalar.abs().max(1e-10)).abs();
        assert!(
            rel_error < 1e-4,
            "Dim {dim}: large value dot product relative error: {rel_error}"
        );
    }
}

#[test]
fn test_dot_product_precision_alternating_signs() {
    // Test with alternating positive/negative to stress cancellation
    for dim in [128, 384, 1024] {
        let a: Vec<f32> = (0..dim)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let b: Vec<f32> = (0..dim)
            .map(|i| if i % 3 == 0 { 0.5 } else { -0.5 })
            .collect();

        let simd = dot_product(&a, &b);
        let scalar = scalar_dot_product(&a, &b);

        let abs_error = (simd - scalar).abs();
        assert!(
            abs_error < 1e-4,
            "Dim {dim}: alternating signs error: {abs_error}"
        );
    }
}

// ============================================================================
// COSINE SIMILARITY PRECISION TESTS
// ============================================================================

#[test]
fn test_cosine_precision_random_vectors() {
    for dim in [64, 128, 256, 384, 512, 768, 1024] {
        for seed in 0..10 {
            let a = generate_vector(dim, seed * 1000);
            let b = generate_vector(dim, seed * 1000 + 500);

            let simd = cosine_similarity(&a, &b);
            let scalar = scalar_cosine_similarity(&a, &b);

            let abs_error = (simd - scalar).abs();
            assert!(
                abs_error < 1e-5,
                "Dim {dim} seed {seed}: cosine error too large: {abs_error}"
            );

            // Verify bounds
            assert!(
                simd >= -1.0 - 1e-5 && simd <= 1.0 + 1e-5,
                "Cosine out of bounds: {simd}"
            );
        }
    }
}

#[test]
fn test_cosine_precision_near_parallel() {
    // Test vectors that are nearly parallel (small angle)
    for dim in [128, 384, 1024] {
        let base = generate_vector(dim, 999);
        let perturbation: Vec<f32> = generate_vector(dim, 888)
            .iter()
            .map(|x| x * 0.001)
            .collect();
        let nearly_parallel: Vec<f32> = base
            .iter()
            .zip(perturbation.iter())
            .map(|(b, p)| b + p)
            .collect();

        let simd = cosine_similarity(&base, &nearly_parallel);
        let scalar = scalar_cosine_similarity(&base, &nearly_parallel);

        let abs_error = (simd - scalar).abs();
        assert!(
            abs_error < 1e-5,
            "Dim {dim}: near-parallel cosine error: {abs_error}"
        );

        // Should be very close to 1.0
        assert!(
            simd > 0.99,
            "Near-parallel vectors should have high similarity, got {simd}"
        );
    }
}

#[test]
fn test_cosine_precision_near_orthogonal() {
    // Test vectors that are nearly orthogonal (90 degrees)
    for dim in [128, 384, 1024] {
        // Create orthogonal basis vectors
        let mut a = vec![0.0f32; dim];
        let mut b = vec![0.0f32; dim];

        // Set up orthogonal components
        for i in 0..dim / 2 {
            a[i * 2] = 1.0;
            b[i * 2 + 1] = 1.0;
        }

        // Add tiny perturbation to break perfect orthogonality
        for i in 0..dim {
            a[i] += 0.0001 * (i as f32 * 0.1).sin();
        }

        let simd = cosine_similarity(&a, &b);
        let scalar = scalar_cosine_similarity(&a, &b);

        let abs_error = (simd - scalar).abs();
        assert!(
            abs_error < 1e-4,
            "Dim {dim}: near-orthogonal cosine error: {abs_error}"
        );

        // Should be close to 0.0
        assert!(
            simd.abs() < 0.1,
            "Near-orthogonal vectors should have low similarity, got {simd}"
        );
    }
}

// ============================================================================
// NORMALIZATION PRECISION TESTS
// ============================================================================

#[test]
fn test_normalize_precision_random() {
    for dim in [64, 128, 256, 384, 512, 768, 1024] {
        let original = generate_vector(dim, 54321);

        let mut simd_v = original.clone();
        let mut scalar_v = original.clone();

        normalize(&mut simd_v);
        scalar_normalize(&mut scalar_v);

        // Check element-wise precision
        for i in 0..dim {
            let abs_error = (simd_v[i] - scalar_v[i]).abs();
            assert!(
                abs_error < 1e-6,
                "Dim {dim} index {i}: normalize element error: {abs_error}"
            );
        }

        // Check resulting norms
        let simd_norm: f32 = simd_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scalar_norm: f32 = scalar_v.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (simd_norm - 1.0).abs() < 1e-6,
            "SIMD normalized vector should have unit length, got {simd_norm}"
        );
        assert!(
            (scalar_norm - 1.0).abs() < 1e-6,
            "Scalar normalized vector should have unit length, got {scalar_norm}"
        );
    }
}

#[test]
fn test_normalize_preserves_direction() {
    let original = generate_vector(384, 98765);
    let mut normalized = original.clone();
    normalize(&mut normalized);

    // Cosine between original and normalized should be 1.0
    let cos = cosine_similarity(&original, &normalized);
    assert!(
        (cos - 1.0).abs() < 1e-5,
        "Normalized vector should point in same direction, cos = {cos}"
    );
}

// ============================================================================
// EUCLIDEAN DISTANCE PRECISION TESTS
// ============================================================================

#[test]
fn test_euclidean_precision_random() {
    for dim in [64, 128, 256, 384, 512, 768, 1024] {
        let a = generate_vector(dim, 11111);
        let b = generate_vector(dim, 22222);

        let simd = euclidean_distance(&a, &b);
        let scalar = scalar_euclidean_distance(&a, &b);

        let rel_error = ((simd - scalar) / scalar.max(1e-10)).abs();
        assert!(
            rel_error < 1e-4,
            "Dim {dim}: Euclidean distance relative error: {rel_error}"
        );
    }
}

#[test]
fn test_euclidean_triangle_inequality() {
    // Verify triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    let a = generate_vector(384, 1);
    let b = generate_vector(384, 2);
    let c = generate_vector(384, 3);

    let d_ab = euclidean_distance(&a, &b);
    let d_bc = euclidean_distance(&b, &c);
    let d_ac = euclidean_distance(&a, &c);

    assert!(
        d_ac <= d_ab + d_bc + 1e-4, // Small epsilon for floating point
        "Triangle inequality violated: d(a,c)={d_ac} > d(a,b)={d_ab} + d(b,c)={d_bc}"
    );
}

#[test]
fn test_euclidean_symmetry() {
    let a = generate_vector(384, 100);
    let b = generate_vector(384, 200);

    let d_ab = euclidean_distance(&a, &b);
    let d_ba = euclidean_distance(&b, &a);

    assert!(
        (d_ab - d_ba).abs() < 1e-6,
        "Distance should be symmetric: d(a,b)={d_ab} != d(b,a)={d_ba}"
    );
}

// ============================================================================
// QUANTIZATION PRECISION TESTS
// ============================================================================

#[test]
fn test_quantization_roundtrip_precision() {
    for dim in [128, 384, 768, 1024] {
        let original = generate_vector(dim, 33333);
        let quantized = QuantizedVector::from_f32(&original);
        let dequantized = quantized.to_f32();

        // Calculate max absolute error
        let max_error: f32 = original
            .iter()
            .zip(dequantized.iter())
            .map(|(o, d)| (o - d).abs())
            .fold(0.0f32, f32::max);

        // Quantization error should be bounded by scale/127
        let max_abs = original.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let expected_max_error = max_abs / 127.0 + 1e-6;

        assert!(
            max_error <= expected_max_error * 1.1, // 10% margin
            "Dim {dim}: quantization error {max_error} exceeds expected {expected_max_error}"
        );
    }
}

#[test]
fn test_quantized_dot_product_precision() {
    for dim in [128, 384, 768, 1024] {
        let a = generate_vector(dim, 44444);
        let b = generate_vector(dim, 55555);

        let float_dot = dot_product(&a, &b);
        let q_a = QuantizedVector::from_f32(&a);
        let q_b = QuantizedVector::from_f32(&b);
        let i8_dot = dot_product_i8(&q_a, &q_b);

        // Quantized dot product should be within ~2-3% of float
        let rel_error = if float_dot.abs() > 1e-6 {
            ((float_dot - i8_dot) / float_dot).abs()
        } else {
            (float_dot - i8_dot).abs()
        };

        assert!(
            rel_error < 0.05, // 5% tolerance
            "Dim {dim}: i8 dot product relative error {rel_error} (float={float_dot}, i8={i8_dot})"
        );
    }
}

#[test]
fn test_quantized_cosine_precision() {
    for dim in [128, 384, 768, 1024] {
        let a = generate_vector(dim, 66666);
        let b = generate_vector(dim, 77777);

        let float_cos = cosine_similarity(&a, &b);
        let q_a = QuantizedVector::from_f32(&a);
        let q_b = QuantizedVector::from_f32(&b);
        let i8_cos = cosine_similarity_i8(&q_a, &q_b);

        let abs_error = (float_cos - i8_cos).abs();
        assert!(
            abs_error < 0.02, // 0.02 tolerance for cosine
            "Dim {dim}: i8 cosine error {abs_error} (float={float_cos}, i8={i8_cos})"
        );
    }
}

#[test]
fn test_quantization_params_range() {
    // Test that quantization params correctly capture the range
    let vector: Vec<f32> = vec![-0.8, 0.0, 0.3, 0.9, -0.5];
    let params = QuantizationParams::from_vector(&vector);

    assert_eq!(params.min_val, -0.8);
    assert_eq!(params.max_val, 0.9);

    // Scale should map max_abs (0.9) to 127
    let expected_scale = 127.0 / 0.9;
    assert!(
        (params.scale - expected_scale).abs() < 1e-4,
        "Scale should be {expected_scale}, got {}",
        params.scale
    );
}

#[test]
fn test_quantization_relative_ordering_preserved() {
    // Verify that quantization preserves similarity ordering
    // This is critical for nearest neighbor search

    let query = generate_vector(384, 10000);
    let similar: Vec<f32> = query
        .iter()
        .enumerate()
        .map(|(i, x)| x + 0.05 * ((i as f32) * 0.7).sin())
        .collect();
    let dissimilar = generate_vector(384, 20000);

    let float_sim_similar = cosine_similarity(&query, &similar);
    let float_sim_dissimilar = cosine_similarity(&query, &dissimilar);

    // Verify float ordering
    assert!(
        float_sim_similar > float_sim_dissimilar,
        "Test setup: similar vector should have higher float similarity"
    );

    let q_query = QuantizedVector::from_f32(&query);
    let q_similar = QuantizedVector::from_f32(&similar);
    let q_dissimilar = QuantizedVector::from_f32(&dissimilar);

    let i8_sim_similar = cosine_similarity_i8(&q_query, &q_similar);
    let i8_sim_dissimilar = cosine_similarity_i8(&q_query, &q_dissimilar);

    // Quantized ordering should match float ordering
    assert!(
        i8_sim_similar > i8_sim_dissimilar - 0.01, // Small tolerance
        "Quantization should preserve relative ordering: i8 similar={i8_sim_similar}, dissimilar={i8_sim_dissimilar}"
    );
}

// ============================================================================
// ACCUMULATION PRECISION TESTS
// ============================================================================

#[test]
fn test_kahan_summation_not_needed() {
    // Verify that the 4-accumulator approach provides sufficient precision
    // without needing Kahan summation (which would be slower)

    // Create a vector where simple summation would lose precision
    let mut v: Vec<f32> = vec![1e10; 4]; // Large values
    v.extend(vec![1e-10; 1000]); // Many small values

    let expected_dot: f32 = 4.0 * 1e20 + 1000.0 * 1e-20;

    let result = dot_product(&v, &v);

    // The 4-accumulator approach should still be reasonably accurate
    let rel_error = ((result - expected_dot) / expected_dot).abs();
    assert!(
        rel_error < 0.01, // 1% tolerance
        "Dot product precision with mixed magnitudes: {rel_error}"
    );
}

#[test]
fn test_long_vector_precision() {
    // Test very long vectors where accumulated error could grow
    let dim = 8192; // Much larger than typical embeddings
    let a = generate_vector(dim, 88888);
    let b = generate_vector(dim, 99999);

    let simd = dot_product(&a, &b);
    let scalar = scalar_dot_product(&a, &b);

    let rel_error = if scalar.abs() > 1e-10 {
        ((simd - scalar) / scalar).abs()
    } else {
        (simd - scalar).abs()
    };

    assert!(
        rel_error < 1e-4,
        "Long vector dot product error: {rel_error}"
    );
}
