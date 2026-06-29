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

/// NaN-norm guard: a vector containing a NaN element has a NaN L2 norm. The
/// scalar path leaves it unchanged (`norm > 0.0` is false for NaN), and every
/// SIMD path must agree byte-for-byte rather than scaling by a NaN inv_norm.
///
/// The dims (129, 145, 200) are deliberately NOT multiples of the unrolled
/// chunk sizes (NEON 16, AVX2 32, AVX-512 64), so each leaves a nonzero scalar
/// tail. A NaN is placed once in the SIMD main body (index 5) and once in the
/// scalar tail (index `dim - 1`), confirming the early-return guard fires no
/// matter which accumulation loop the NaN flows through.
#[test]
fn test_normalize_nan_norm_matches_scalar() {
    for dim in [129usize, 145, 200] {
        for nan_at in [5usize, dim - 1] {
            let mut original = generate_random_vector_seeded(dim, 0x_4e_61_4e ^ dim as u64);
            original[nan_at] = f32::NAN;

            let mut simd_v = original.clone();
            let mut scalar_v = original.clone();

            normalize(&mut simd_v); // dispatches to the platform SIMD path
            normalize::normalize_scalar(&mut scalar_v);

            for (i, (s, sc)) in simd_v.iter().zip(scalar_v.iter()).enumerate() {
                assert_eq!(
                    s.to_bits(),
                    sc.to_bits(),
                    "NaN-norm SIMD vs scalar mismatch at [{i}] (dim={dim}, nan_at={nan_at}): {s} vs {sc}"
                );
                assert_eq!(
                    s.to_bits(),
                    original[i].to_bits(),
                    "NaN-norm vector must be left unchanged at [{i}] (dim={dim}, nan_at={nan_at}): {s} vs {}",
                    original[i]
                );
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_neon_normalize_matches_scalar_multiple_dims() {
    for dim in [1usize, 3, 16, 31, 127, 384, 768, 1536] {
        let mut simd_v = generate_random_vector_seeded(dim, 700 + dim as u64);
        let mut scalar_v = simd_v.clone();

        normalize(&mut simd_v); // dispatches to NEON on aarch64
        normalize::normalize_scalar(&mut scalar_v);

        for (i, (s, sc)) in simd_v.iter().zip(scalar_v.iter()).enumerate() {
            assert_close(
                *s,
                *sc,
                1e-5,
                1e-5,
                &format!("neon normalize dim={dim} idx={i}"),
            );
        }
    }
}

/// Differential test: vrsqrteq_f32 + Newton–Raphson vs scalar reference.
///
/// Two Newton steps on top of `vrsqrteq_f32` converge to full f32 precision
/// (~2^-23 relative).  The max absolute per-element error vs the scalar path
/// (which uses `f32::sqrt` + division) must stay below 1e-6 — issue #149's
/// accuracy gate — across representative embedding dimensions.
#[cfg(target_arch = "aarch64")]
#[test]
fn test_neon_rsqrt_newton_accuracy() {
    for dim in [16usize, 64, 128, 384, 768, 1024, 1536] {
        let mut neon_v = generate_random_vector_seeded(dim, 0x_dead_beef + dim as u64);
        let mut scalar_v = neon_v.clone();

        normalize(&mut neon_v);
        normalize::normalize_scalar(&mut scalar_v);

        let max_diff = neon_v
            .iter()
            .zip(scalar_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_diff < 1e-6,
            "vrsqrteq_f32 + Newton vs scalar: dim={dim}, max_abs_diff={max_diff} (limit=1e-6)"
        );
    }
}

/// Subnormal-norm guard: ‖v‖ ≲ 7e-20 drives norm_sq subnormal, where
/// `vrsqrteq`/Newton overflow to inf.  The NEON path must fall back to the
/// scalar reciprocal and stay finite + byte-consistent with `normalize_scalar`.
#[cfg(target_arch = "aarch64")]
#[test]
fn test_neon_normalize_subnormal_norm_finite() {
    let dim = 64usize;
    let mut neon_v = vec![1e-21_f32; dim];
    let mut scalar_v = neon_v.clone();

    normalize(&mut neon_v);
    normalize::normalize_scalar(&mut scalar_v);

    for (i, (&a, &b)) in neon_v.iter().zip(scalar_v.iter()).enumerate() {
        assert!(a.is_finite(), "NEON normalize non-finite at [{i}]: {a}");
        assert!(
            (a - b).abs() < 1e-5,
            "subnormal-norm NEON vs scalar mismatch at [{i}]: {a} vs {b}"
        );
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

/// Consumer contract for khive ANN indexes (`khive-hnsw`, `khive-vamana`; ADR-012).
///
/// Locks the guarantees the downstream indexes depend on so the surface cannot
/// silently drift:
/// 1. **Signature stability** — the four distance fns keep the
///    `(&[f32], &[f32]) -> f32` shape (compile-time gate).
/// 2. **Numerical stability** — `squared_euclidean_distance` tracks the scalar
///    reference within FP tolerance across the queried dims (384 is the live
///    embed dim).
/// 3. **Ordering invariant** — squared-L2 sorts identically to true L2, the only
///    property an ANN graph relies on.
/// 4. **Degenerate-input behaviour** — the documented length-mismatch returns.
#[test]
fn test_simd_distance_consumer_contract() {
    // The `(&[f32], &[f32]) -> f32` shape khive ANN indexes bind to.
    type DistFn = fn(&[f32], &[f32]) -> f32;

    // (1) Signature gate: fails to compile if any signature drifts.
    let _contract: [DistFn; 4] = [
        squared_euclidean_distance,
        euclidean_distance,
        dot_product,
        cosine_similarity,
    ];

    // (2) squared_euclidean_distance == scalar reference, within FP tolerance.
    for dim in AVX512_TEST_DIMS {
        let a = generate_random_vector_seeded(dim, 911);
        let b = generate_random_vector_seeded(dim, 1013);

        let simd_result = squared_euclidean_distance(&a, &b);
        let scalar_result = distance::squared_euclidean_distance_scalar(&a, &b);

        assert_close(
            simd_result,
            scalar_result,
            1e-3,
            1e-4,
            &format!("squared_euclidean_distance contract dim {dim}"),
        );
    }

    // (3) Ordering invariant: sorting by squared-L2 matches sorting by true L2.
    let query = generate_random_vector_seeded(384, 1);
    let candidates: Vec<Vec<f32>> = (0..16)
        .map(|i| generate_random_vector_seeded(384, 100 + i as u64))
        .collect();
    let order_by = |dist: DistFn| {
        let mut idx: Vec<usize> = (0..candidates.len()).collect();
        idx.sort_by(|&x, &y| {
            dist(&query, &candidates[x])
                .partial_cmp(&dist(&query, &candidates[y]))
                .unwrap()
        });
        idx
    };
    assert_eq!(
        order_by(squared_euclidean_distance),
        order_by(euclidean_distance),
        "squared_euclidean_distance must preserve true-L2 ordering (ANN invariant)"
    );

    // (4) Documented degenerate-input behaviour is part of the contract.
    let a4 = [1.0_f32, 2.0, 3.0, 4.0];
    let b3 = [1.0_f32, 2.0, 3.0];
    assert_eq!(squared_euclidean_distance(&a4, &b3), f32::MAX);
    assert_eq!(euclidean_distance(&a4, &b3), f32::MAX);
    assert_eq!(dot_product(&a4, &b3), 0.0);
    assert_eq!(cosine_similarity(&a4, &b3), 0.0);

    // cosine_similarity additionally documents empty-input → 0.0 (explicit guard).
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
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
    assert_eq!(q.len(), 384, "int8 should be 384 bytes");
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

// ============================================================================
// OPTIMIZATION CORRECTNESS TESTS (added with perf(embed) optimizations)
// ============================================================================

/// Verify that `cosine_similarity_fused` produces the same result as `cosine_similarity`.
///
/// Both routes through the same SIMD kernel; this test documents the guarantee
/// that the fused single-pass path matches the dispatch path at every dimension.
#[test]
fn test_cosine_fused_matches_separate() {
    for &dim in &[7usize, 15, 64, 128, 384, 768, 1536] {
        let a = generate_random_vector_seeded(dim, 0xf00d_dead + dim as u64);
        let b = generate_random_vector_seeded(dim, 0xbeef_cafe + dim as u64);

        let separate = cosine_similarity(&a, &b);
        let fused = cosine_similarity_fused(&a, &b);

        let diff = (separate - fused).abs();
        assert!(
            diff < 1e-6,
            "cosine_similarity_fused vs cosine_similarity at dim={dim}: separate={separate}, fused={fused}, diff={diff}"
        );
    }
}

/// Edge cases for `cosine_similarity_fused`: zero vectors, mismatched lengths, identical vectors.
#[test]
fn test_cosine_fused_edge_cases() {
    // Empty input
    let result = cosine_similarity_fused(&[], &[]);
    assert_eq!(result, 0.0, "fused: empty slices should return 0.0");

    // Length mismatch
    let a = vec![1.0_f32, 2.0, 3.0];
    let b = vec![1.0_f32, 2.0];
    assert_eq!(
        cosine_similarity_fused(&a, &b),
        0.0,
        "fused: length mismatch should return 0.0"
    );

    // Identical non-zero vector -> similarity 1.0
    let v = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let sim = cosine_similarity_fused(&v, &v);
    assert!(
        (sim - 1.0).abs() < 1e-6,
        "fused: identical vectors should have similarity 1.0, got {sim}"
    );
}

/// Verify that `batch_cosine_one_vs_many` matches per-pair `cosine_similarity` calls.
#[test]
fn test_batch_cosine_one_vs_many_matches_per_pair() {
    let query = generate_random_vector_seeded(384, 0x1111_2222);
    let candidates: Vec<Vec<f32>> = (0..8)
        .map(|i| generate_random_vector_seeded(384, 0x3333_0000 + i as u64))
        .collect();
    let candidate_refs: Vec<&[f32]> = candidates.iter().map(Vec::as_slice).collect();

    let batch = batch_cosine_one_vs_many(&query, &candidate_refs);
    assert_eq!(batch.len(), 8);

    for (i, c) in candidates.iter().enumerate() {
        let expected = cosine_similarity(&query, c);
        let diff = (batch[i] - expected).abs();
        assert!(
            diff < 1e-5,
            "batch_cosine_one_vs_many[{i}] at dim=384: batch={}, per-pair={}, diff={}",
            batch[i],
            expected,
            diff
        );
    }
}

/// `batch_cosine_one_vs_many` returns zeros for mismatched dimensions, not panics.
#[test]
fn test_batch_cosine_one_vs_many_dim_mismatch() {
    let query = vec![1.0_f32, 2.0, 3.0];
    let good = vec![4.0_f32, 5.0, 6.0];
    let bad = vec![1.0_f32, 2.0]; // wrong length
    let refs: Vec<&[f32]> = vec![good.as_slice(), bad.as_slice()];

    let results = batch_cosine_one_vs_many(&query, &refs);
    assert_eq!(results.len(), 2);
    // good candidate should produce a valid similarity
    assert!(
        results[0].is_finite(),
        "good candidate should produce finite similarity"
    );
    // bad candidate (dim mismatch) should return 0.0
    assert_eq!(results[1], 0.0, "dim-mismatch candidate should return 0.0");
}

/// Verify that the i8 dot product raw kernel (with prefetch) produces the same result
/// as the scalar reference across all standard embedding dimensions.
#[test]
fn test_i8_dot_unrolled_matches_original() {
    for &dim in &[7usize, 15, 16, 64, 128, 384, 768, 1536] {
        let a_f32 = generate_random_vector_seeded(dim, 0xaaaa_0000 + dim as u64);
        let b_f32 = generate_random_vector_seeded(dim, 0xbbbb_0000 + dim as u64);

        let a_q = QuantizedVector::from_f32(&a_f32);
        let b_q = QuantizedVector::from_f32(&b_f32);

        // Scalar reference: direct i32 accumulation over raw i8 slices.
        let scalar: f32 = a_q
            .data()
            .iter()
            .zip(b_q.data().iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum::<i32>() as f32;

        // SIMD path (with prefetch via dot_product_i8_raw).
        let simd = dot_product_i8_raw(a_q.data(), b_q.data());

        let diff = (simd - scalar).abs();
        assert!(
            diff <= 1.0,
            "i8 dot product (SIMD with prefetch) vs scalar at dim={dim}: simd={simd}, scalar={scalar}, diff={diff}"
        );
    }
}
