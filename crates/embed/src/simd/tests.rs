//! Tests for SIMD operations.

use super::*;

const AVX512_TEST_DIMS: [usize; 7] = [7, 15, 64, 128, 384, 768, 1536];

fn generate_random_vector(dim: usize) -> Vec<f32> {
    generate_random_vector_seeded(dim, 0x1234_5678_9abc_def0)
}

fn generate_random_vector_seeded(dim: usize, seed: u64) -> Vec<f32> {
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

fn assert_close(actual: f32, expected: f32, abs_tol: f32, rel_tol: f32, context: &str) {
    let diff = (actual - expected).abs();
    let limit = abs_tol.max(rel_tol * expected.abs().max(1.0));
    assert!(
        diff <= limit,
        "{context}: diff {diff} exceeds tolerance {limit} (actual={actual}, expected={expected})"
    );
}

#[test]
fn test_dot_product_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = dot_product(&a, &b);
    let expected: f32 = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 32.0
    assert!((result - expected).abs() < 1e-5);
}

#[test]
fn test_dot_product_384_dim() {
    let a = generate_random_vector_seeded(384, 1);
    let b = generate_random_vector_seeded(384, 2);

    let simd_result = dot_product(&a, &b);
    let scalar_result = dot_product::dot_product_scalar(&a, &b);

    assert_close(
        simd_result,
        scalar_result,
        1e-4,
        1e-4,
        "384-dim dot product SIMD vs scalar",
    );
}

#[test]
fn test_avx512_dot_product_dimensions() {
    for dim in AVX512_TEST_DIMS {
        let a = generate_random_vector_seeded(dim, 11);
        let b = generate_random_vector_seeded(dim, 29);

        let simd_result = dot_product(&a, &b);
        let scalar_result = dot_product::dot_product_scalar(&a, &b);

        assert_close(
            simd_result,
            scalar_result,
            1e-4,
            1e-4,
            &format!("dot_product dim {dim}"),
        );
    }
}

#[test]
fn test_cosine_similarity_identical() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = cosine_similarity(&a, &a);
    assert!(
        (result - 1.0).abs() < 1e-5,
        "Identical vectors should have similarity 1.0, got {result}"
    );
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let result = cosine_similarity(&a, &b);
    assert!(
        result.abs() < 1e-5,
        "Orthogonal vectors should have 0 similarity"
    );
}

#[test]
fn test_cosine_similarity_384_dim() {
    let a = generate_random_vector_seeded(384, 3);
    let b = generate_random_vector_seeded(384, 5);

    let simd_result = cosine_similarity(&a, &b);
    let scalar_result = cosine::cosine_similarity_scalar(&a, &b);

    assert_close(
        simd_result,
        scalar_result,
        1e-5,
        1e-4,
        "384-dim cosine SIMD vs scalar",
    );
}

#[test]
fn test_avx512_cosine_similarity_dimensions() {
    for dim in AVX512_TEST_DIMS {
        let a = generate_random_vector_seeded(dim, 41);
        let b = generate_random_vector_seeded(dim, 73);

        let simd_result = cosine_similarity(&a, &b);
        let scalar_result = cosine::cosine_similarity_scalar(&a, &b);

        assert_close(
            simd_result,
            scalar_result,
            1e-5,
            1e-4,
            &format!("cosine_similarity dim {dim}"),
        );
    }
}

#[test]
fn test_normalize() {
    let mut v = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    normalize(&mut v);

    let magnitude: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (magnitude - 1.0).abs() < 1e-5,
        "Normalized vector should have magnitude 1.0, got {magnitude}"
    );
}

#[test]
fn test_normalize_384_dim() {
    let mut simd_v = generate_random_vector_seeded(384, 7);
    let mut scalar_v = simd_v.clone();

    normalize(&mut simd_v);
    normalize::normalize_scalar(&mut scalar_v);

    for (i, (s, sc)) in simd_v.iter().zip(scalar_v.iter()).enumerate() {
        assert_close(*s, *sc, 1e-5, 1e-5, &format!("normalize 384 idx {i}"));
    }
}

#[test]
fn test_avx512_normalize_dimensions() {
    for dim in AVX512_TEST_DIMS {
        let mut simd_v = generate_random_vector_seeded(dim, 101);
        let mut scalar_v = simd_v.clone();

        normalize(&mut simd_v);
        normalize::normalize_scalar(&mut scalar_v);

        for (i, (s, sc)) in simd_v.iter().zip(scalar_v.iter()).enumerate() {
            assert_close(*s, *sc, 1e-5, 1e-5, &format!("normalize dim {dim} idx {i}"));
        }

        let simd_norm: f32 = simd_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_close(
            simd_norm,
            1.0,
            1e-5,
            1e-5,
            &format!("normalize dim {dim} norm"),
        );
    }
}

#[test]
fn test_euclidean_distance() {
    let a = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let result = euclidean_distance(&a, &b);
    assert!(
        (result - 5.0).abs() < 1e-5,
        "Expected distance 5.0, got {result}"
    );
}

#[test]
fn test_euclidean_distance_384_dim() {
    let a = generate_random_vector_seeded(384, 13);
    let b = generate_random_vector_seeded(384, 17);

    let simd_result = euclidean_distance(&a, &b);
    let scalar_result = distance::euclidean_distance_scalar(&a, &b);

    assert_close(
        simd_result,
        scalar_result,
        1e-3,
        1e-4,
        "384-dim Euclidean SIMD vs scalar",
    );
}

#[test]
fn test_avx512_euclidean_distance_dimensions() {
    for dim in AVX512_TEST_DIMS {
        let a = generate_random_vector_seeded(dim, 211);
        let b = generate_random_vector_seeded(dim, 307);

        let simd_result = euclidean_distance(&a, &b);
        let scalar_result = distance::euclidean_distance_scalar(&a, &b);

        assert_close(
            simd_result,
            scalar_result,
            1e-3,
            1e-4,
            &format!("euclidean_distance dim {dim}"),
        );
    }
}

#[test]
fn test_avx512f_config_detection() {
    let config = SimdConfig::detect();

    #[cfg(target_arch = "x86_64")]
    {
        let runtime_detected = std::is_x86_feature_detected!("avx512f");
        assert_eq!(
            config.avx512f_enabled, runtime_detected,
            "SimdConfig avx512f_enabled should match runtime detection"
        );
        if config.avx512vnni_enabled {
            assert!(
                config.avx512f_enabled,
                "AVX-512 VNNI should imply AVX-512F support"
            );
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        assert!(
            !config.avx512f_enabled,
            "AVX-512F should only be reported on x86_64"
        );
    }
}

#[test]
fn test_simd_config_detection() {
    let config = SimdConfig::detect();
    // Just verify it doesn't panic and show capabilities
    println!(
        "AVX512F: {}, AVX2: {}, FMA: {}, AVX512-VNNI: {}, NEON: {}",
        config.avx512f_enabled,
        config.avx2_enabled,
        config.fma_enabled,
        config.avx512vnni_enabled,
        config.neon_enabled
    );

    #[cfg(target_arch = "x86_64")]
    if config.avx512vnni_enabled {
        assert!(config.avx512f_enabled, "AVX512-VNNI requires AVX512F");
    }

    // On aarch64 (Apple Silicon, Graviton, etc.), NEON should be enabled
    #[cfg(target_arch = "aarch64")]
    assert!(config.neon_enabled, "NEON should be available on aarch64");
}

#[test]
fn test_batch_operations() {
    let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..10)
        .map(|i| {
            (
                generate_random_vector_seeded(384, i as u64 + 1),
                generate_random_vector_seeded(384, i as u64 + 1001),
            )
        })
        .collect();

    let pair_refs: Vec<(&[f32], &[f32])> = pairs
        .iter()
        .map(|(a, b)| (a.as_slice(), b.as_slice()))
        .collect();

    let results = batch_cosine_similarity(&pair_refs);
    assert_eq!(results.len(), 10);

    // Verify each result is in valid range
    for sim in results {
        assert!((-1.0..=1.0).contains(&sim));
    }
}

#[test]
fn test_non_multiple_of_8() {
    // Test with dimensions that aren't multiples of 8 or 16.
    for dim in [7, 15, 100, 383, 385] {
        let a = generate_random_vector_seeded(dim, 23);
        let b = generate_random_vector_seeded(dim, 47);

        let simd_result = cosine_similarity(&a, &b);
        let scalar_result = cosine::cosine_similarity_scalar(&a, &b);

        assert_close(
            simd_result,
            scalar_result,
            1e-5,
            1e-4,
            &format!("non-multiple dim {dim}"),
        );
    }
}

// ========================================================================
// INT8 QUANTIZATION TESTS
// ========================================================================

#[test]
fn test_quantization_roundtrip() {
    let original = generate_random_vector(384);
    let quantized = QuantizedVector::from_f32(&original);
    let dequantized = quantized.to_f32();

    // Check approximate equality (quantization introduces some error)
    for (orig, deq) in original.iter().zip(dequantized.iter()) {
        assert!(
            (orig - deq).abs() < 0.02,
            "Quantization error too large: {orig} vs {deq}"
        );
    }
}

#[test]
fn test_i8_dot_product_accuracy() {
    let a = generate_random_vector_seeded(384, 131);
    let b = generate_random_vector_seeded(384, 149);

    let float_result = dot_product(&a, &b);

    let a_q = QuantizedVector::from_f32(&a);
    let b_q = QuantizedVector::from_f32(&b);
    let i8_result = dot_product_i8(&a_q, &b_q);

    let normalized_error = (float_result - i8_result).abs() / float_result.abs().max(1.0);
    assert!(
        normalized_error < 0.05,
        "int8 dot product error too large: float={float_result}, i8={i8_result}, error={normalized_error}"
    );
}

#[test]
fn test_i8_cosine_similarity_accuracy() {
    let a = generate_random_vector_seeded(384, 163);
    let b = generate_random_vector_seeded(384, 181);

    let float_result = cosine_similarity(&a, &b);

    let a_q = QuantizedVector::from_f32(&a);
    let b_q = QuantizedVector::from_f32(&b);
    let i8_result = cosine_similarity_i8(&a_q, &b_q);

    // int8 cosine similarity should be very close to float32
    assert!(
        (float_result - i8_result).abs() < 0.02,
        "int8 cosine similarity error too large: float={float_result}, i8={i8_result}"
    );
}

#[test]
fn test_i8_memory_savings() {
    let v = generate_random_vector(384);
    let q = QuantizedVector::from_f32(&v);

    // float32: 384 * 4 = 1536 bytes
    // int8: 384 * 1 = 384 bytes
    assert_eq!(v.len() * 4, 1536, "float32 should be 1536 bytes");
    assert_eq!(q.data.len(), 384, "int8 should be 384 bytes");
    // 4x memory reduction
}

// ============================================================================
// PROPTEST - Property-based tests for embedding dimension invariants (#451)
// ============================================================================

// ========================================================================
// BATCH-4 KERNEL PARITY TESTS
// ========================================================================

#[test]
fn test_batch4_dot_matches_per_pair_basic() {
    // Small known vectors so correctness is obvious.
    let q = vec![1.0_f32, 0.5, -0.5, 0.25, 0.0, -1.0, 0.75, 0.1];
    let c0 = vec![1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let c1 = vec![0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let c2 = vec![-1.0_f32, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
    let c3 = vec![0.5_f32, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5];

    let batch = dot_product_batch4(&q, &c0, &c1, &c2, &c3);
    let expected = [
        dot_product(&q, &c0),
        dot_product(&q, &c1),
        dot_product(&q, &c2),
        dot_product(&q, &c3),
    ];

    for i in 0..4 {
        assert!(
            (batch[i] - expected[i]).abs() < 1e-5,
            "batch4[{i}] = {}, per-pair = {}",
            batch[i],
            expected[i]
        );
    }
}

#[test]
fn test_batch4_dot_matches_per_pair_384dim() {
    let q = generate_random_vector_seeded(384, 1001);
    let c0 = generate_random_vector_seeded(384, 2001);
    let c1 = generate_random_vector_seeded(384, 3001);
    let c2 = generate_random_vector_seeded(384, 4001);
    let c3 = generate_random_vector_seeded(384, 5001);

    let batch = dot_product_batch4(&q, &c0, &c1, &c2, &c3);
    let expected = [
        dot_product(&q, &c0),
        dot_product(&q, &c1),
        dot_product(&q, &c2),
        dot_product(&q, &c3),
    ];

    for i in 0..4 {
        let tol = 1e-4 * expected[i].abs().max(1.0);
        assert!(
            (batch[i] - expected[i]).abs() <= tol,
            "384d batch4[{i}] = {:.8}, per-pair = {:.8}, delta = {:.2e}",
            batch[i],
            expected[i],
            (batch[i] - expected[i]).abs()
        );
    }
}

#[test]
fn test_batch4_dot_matches_per_pair_odd_dims() {
    // 100 is not a multiple of 8; exercises the scalar tail.
    let q = generate_random_vector_seeded(100, 7777);
    let c0 = generate_random_vector_seeded(100, 7778);
    let c1 = generate_random_vector_seeded(100, 7779);
    let c2 = generate_random_vector_seeded(100, 7780);
    let c3 = generate_random_vector_seeded(100, 7781);

    let batch = dot_product_batch4(&q, &c0, &c1, &c2, &c3);
    let expected = [
        dot_product(&q, &c0),
        dot_product(&q, &c1),
        dot_product(&q, &c2),
        dot_product(&q, &c3),
    ];

    for i in 0..4 {
        let tol = 1e-4 * expected[i].abs().max(1.0);
        assert!(
            (batch[i] - expected[i]).abs() <= tol,
            "100d batch4[{i}] = {:.8}, per-pair = {:.8}",
            batch[i],
            expected[i]
        );
    }
}

#[test]
fn test_batch_dot_product_same_query_matches_per_pair() {
    // Constructs a same-query batch (N=16 candidates, all with the same query).
    // batch_dot_product should take the batch-4 SIMD path for chunks of 4
    // and produce the same values as per-pair calls.
    let query = generate_random_vector_seeded(384, 9001);
    let candidates: Vec<Vec<f32>> = (0..16)
        .map(|i| generate_random_vector_seeded(384, 9100 + i as u64))
        .collect();

    // Build pairs where every left-hand side is the same query borrow.
    let pairs: Vec<(&[f32], &[f32])> = candidates
        .iter()
        .map(|c| (query.as_slice(), c.as_slice()))
        .collect();

    let batch_results = batch_dot_product(&pairs);
    let per_pair_results: Vec<f32> = candidates.iter().map(|c| dot_product(&query, c)).collect();

    assert_eq!(batch_results.len(), 16);
    for i in 0..16 {
        let tol = 1e-4 * per_pair_results[i].abs().max(1.0);
        assert!(
            (batch_results[i] - per_pair_results[i]).abs() <= tol,
            "batch_dot_product[{i}] = {:.8}, per-pair = {:.8}, delta = {:.2e}",
            batch_results[i],
            per_pair_results[i],
            (batch_results[i] - per_pair_results[i]).abs()
        );
    }
}

#[test]
fn test_batch_dot_product_unit_vectors_equals_cosine() {
    // For unit-normalized vectors, dot product == cosine similarity.
    // This validates the Round-3 fast path: normalized cosine routes through dot.
    let mut q = generate_random_vector_seeded(384, 42);
    let mut c = generate_random_vector_seeded(384, 43);
    normalize(&mut q);
    normalize(&mut c);

    let dot = dot_product(&q, &c);
    let cos = cosine_similarity(&q, &c);

    assert!(
        (dot - cos).abs() < 1e-5,
        "unit-vector dot = {dot:.8}, cosine = {cos:.8}, delta = {:.2e}",
        (dot - cos).abs()
    );
}

mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating embedding dimensions (common model sizes + edge cases)
    fn embedding_dimension_strategy() -> impl Strategy<Value = usize> {
        prop_oneof![
            // Common embedding model dimensions
            Just(128),
            Just(384),  // bge-small-en-v1.5
            Just(768),  // bge-base-en-v1.5
            Just(1024), // bge-large-en-v1.5
            Just(1536),
            // Edge cases for SIMD alignment
            Just(1),    // Minimum
            Just(7),    // Non-aligned small
            Just(8),    // AVX alignment
            Just(15),   // Non-aligned
            Just(16),   // Double AVX / one AVX-512 register
            Just(31),   // Non-aligned
            Just(32),   // 4x AVX
            Just(63),   // Non-aligned
            Just(64),   // 8x AVX / 4x AVX-512
            Just(383),  // Off-by-one small
            Just(385),  // Off-by-one large
            Just(1023), // Off-by-one 1024
            // Random dimensions
            (1usize..2048usize),
        ]
    }

    proptest! {
        /// Property: Dot product of vector with itself equals squared L2 norm
        #[test]
        fn prop_dot_product_self_equals_squared_norm(dim in embedding_dimension_strategy()) {
            let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
            let dot_self = dot_product(&v, &v);
            let squared_norm: f32 = v.iter().map(|x| x * x).sum();
            prop_assert!((dot_self - squared_norm).abs() < 1e-4 * squared_norm.abs().max(1.0),
                "dot(v,v) should equal ||v||^2: got {} vs {}", dot_self, squared_norm);
        }

        /// Property: Cosine similarity of identical vectors is 1.0
        #[test]
        fn prop_cosine_identical_is_one(dim in embedding_dimension_strategy()) {
            let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).cos()).collect();
            // Skip zero vectors
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assume!(norm > 1e-6);

            let sim = cosine_similarity(&v, &v);
            prop_assert!((sim - 1.0).abs() < 1e-5,
                "cosine(v, v) should be 1.0, got {}", sim);
        }

        /// Property: Normalized vectors have unit length
        #[test]
        fn prop_normalize_produces_unit_vector(dim in embedding_dimension_strategy()) {
            let mut v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).sin()).collect();
            let original_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assume!(original_norm > 1e-6); // Skip near-zero vectors

            normalize(&mut v);
            let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!((norm_after - 1.0).abs() < 1e-5,
                "normalized vector should have unit length, got {}", norm_after);
        }

        /// Property: Euclidean distance is non-negative
        #[test]
        fn prop_euclidean_distance_non_negative(dim in embedding_dimension_strategy()) {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).cos()).collect();

            let dist = euclidean_distance(&a, &b);
            prop_assert!(dist >= 0.0, "Euclidean distance should be non-negative, got {}", dist);
        }

        /// Property: Euclidean distance to self is zero
        #[test]
        fn prop_euclidean_distance_self_is_zero(dim in embedding_dimension_strategy()) {
            let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).sin()).collect();
            let dist = euclidean_distance(&v, &v);
            prop_assert!(dist.abs() < 1e-5, "distance(v, v) should be 0, got {}", dist);
        }

        /// Property: SIMD and scalar dot product implementations are equivalent
        #[test]
        fn prop_simd_scalar_dot_product_equivalent(dim in embedding_dimension_strategy()) {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.17).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).cos()).collect();

            let simd_result = dot_product(&a, &b);
            let scalar_result = dot_product::dot_product_scalar(&a, &b);

            let tolerance = 1e-4 * scalar_result.abs().max(1.0);
            prop_assert!((simd_result - scalar_result).abs() < tolerance,
                "SIMD and scalar dot product differ: {} vs {}", simd_result, scalar_result);
        }

        /// Property: SIMD and scalar cosine similarity are equivalent
        #[test]
        fn prop_simd_scalar_cosine_equivalent(dim in embedding_dimension_strategy()) {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.13).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.19).cos()).collect();

            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assume!(norm_a > 1e-6 && norm_b > 1e-6);

            let simd_result = cosine_similarity(&a, &b);
            let scalar_result = cosine::cosine_similarity_scalar(&a, &b);

            prop_assert!((simd_result - scalar_result).abs() < 1e-4,
                "SIMD and scalar cosine differ: {} vs {}", simd_result, scalar_result);
        }

        /// Property: SIMD and scalar normalization are equivalent
        #[test]
        fn prop_simd_scalar_normalize_equivalent(dim in embedding_dimension_strategy()) {
            let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.11).sin()).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assume!(norm > 1e-6);

            let mut simd_v = v.clone();
            let mut scalar_v = v.clone();

            normalize(&mut simd_v);
            normalize::normalize_scalar(&mut scalar_v);

            for (i, (s, sc)) in simd_v.iter().zip(scalar_v.iter()).enumerate() {
                prop_assert!((s - sc).abs() < 1e-5,
                    "Mismatch at index {}: SIMD {} vs scalar {}", i, s, sc);
            }
        }

        /// Property: SIMD and scalar Euclidean distance are equivalent
        #[test]
        fn prop_simd_scalar_euclidean_equivalent(dim in embedding_dimension_strategy()) {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.07).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.11).cos()).collect();

            let simd_result = euclidean_distance(&a, &b);
            let scalar_result = distance::euclidean_distance_scalar(&a, &b);

            let tolerance = 1e-3 * scalar_result.abs().max(1.0);
            prop_assert!((simd_result - scalar_result).abs() < tolerance,
                "SIMD and scalar Euclidean differ: {} vs {}", simd_result, scalar_result);
        }

        /// Property: Cosine similarity is bounded between -1 and 1
        #[test]
        fn prop_cosine_bounded(dim in embedding_dimension_strategy()) {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.29).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.31).cos()).collect();

            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assume!(norm_a > 1e-6 && norm_b > 1e-6);

            let sim = cosine_similarity(&a, &b);
            prop_assert!((-1.0..=1.0).contains(&sim),
                "Cosine similarity should be in [-1, 1], got {}", sim);
        }

        /// Property: int8 quantization preserves relative similarity ordering
        #[test]
        fn prop_i8_quantization_preserves_similarity_order(dim in 32usize..512usize) {
            // Generate 3 vectors where a is more similar to b than to c
            let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin() + 0.1 * (i as f32 * 0.3).cos()).collect();
            let c: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.5).cos()).collect();

            let sim_ab_float = cosine_similarity(&a, &b);
            let sim_ac_float = cosine_similarity(&a, &c);

            // Only test if there's meaningful difference
            prop_assume!((sim_ab_float - sim_ac_float).abs() > 0.1);

            let a_q = QuantizedVector::from_f32(&a);
            let b_q = QuantizedVector::from_f32(&b);
            let c_q = QuantizedVector::from_f32(&c);

            let sim_ab_i8 = cosine_similarity_i8(&a_q, &b_q);
            let sim_ac_i8 = cosine_similarity_i8(&a_q, &c_q);

            // Relative ordering should be preserved
            if sim_ab_float > sim_ac_float {
                prop_assert!(sim_ab_i8 > sim_ac_i8 - 0.1,
                    "i8 should preserve order: ab={} > ac={} but i8: ab={}, ac={}",
                    sim_ab_float, sim_ac_float, sim_ab_i8, sim_ac_i8);
            }
        }
    }
}
