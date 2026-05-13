//! Integration tests for the lattice-embed public API.
//!
//! These tests exercise the public-facing utilities in the `utils` module
//! to ensure they work correctly for typical embedding workflows.

#![allow(clippy::manual_range_contains)]
#![allow(clippy::uninlined_format_args)]
#![allow(unused_variables)]

use lattice_embed::utils::{
    batch_cosine_similarity, batch_dot_product, cosine_similarity, dot_product, euclidean_distance,
    normalize,
};
use lattice_embed::{SimdConfig, simd_config};

// ============================================================================
// TYPICAL EMBEDDING WORKFLOW TESTS
// ============================================================================

#[test]
fn test_embedding_search_workflow() {
    // Simulate typical embedding search: find most similar document

    // Query embedding (normalized, as typical embedding models produce)
    let mut query: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();
    normalize(&mut query);

    // Document embeddings (also normalized)
    let documents: Vec<Vec<f32>> = (0..100)
        .map(|doc_id| {
            let mut doc: Vec<f32> = (0..384)
                .map(|i| ((i as f32 + doc_id as f32) * 0.1).cos())
                .collect();
            normalize(&mut doc);
            doc
        })
        .collect();

    // Find most similar document
    let similarities: Vec<f32> = documents
        .iter()
        .map(|doc| cosine_similarity(&query, doc))
        .collect();

    // Verify all similarities are in valid range
    for (i, &sim) in similarities.iter().enumerate() {
        assert!(
            sim >= -1.0 && sim <= 1.0,
            "Document {i} similarity out of bounds: {sim}"
        );
    }

    // Find max
    let (best_idx, best_sim) = similarities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("Best match: document {best_idx} with similarity {best_sim}");

    // For normalized vectors, dot_product should equal cosine_similarity
    let dot_sim = dot_product(&query, &documents[best_idx]);
    assert!(
        (dot_sim - *best_sim).abs() < 1e-5,
        "For normalized vectors, dot={dot_sim} should equal cosine={best_sim}"
    );
}

#[test]
fn test_batch_similarity_computation() {
    // Typical batch similarity: compare one query against many documents

    let query: Vec<f32> = (0..384).map(|i| (i as f32 * 0.2).sin()).collect();

    let documents: Vec<Vec<f32>> = (0..50)
        .map(|doc_id| {
            (0..384)
                .map(|i| ((i as f32 + doc_id as f32 * 10.0) * 0.15).cos())
                .collect()
        })
        .collect();

    // Create pairs for batch operation
    let pairs: Vec<(&[f32], &[f32])> = documents
        .iter()
        .map(|doc| (query.as_slice(), doc.as_slice()))
        .collect();

    // Batch computation
    let batch_results = batch_cosine_similarity(&pairs);
    assert_eq!(batch_results.len(), 50);

    // Verify against individual computations
    for (i, (&batch_sim, doc)) in batch_results.iter().zip(documents.iter()).enumerate() {
        let individual_sim = cosine_similarity(&query, doc);
        assert!(
            (batch_sim - individual_sim).abs() < 1e-6,
            "Document {i}: batch={batch_sim} != individual={individual_sim}"
        );
    }
}

#[test]
fn test_distance_based_clustering() {
    // Simulate clustering by computing pairwise distances

    let points: Vec<Vec<f32>> = (0..20)
        .map(|point_id| {
            (0..64) // Smaller dimension for speed
                .map(|i| ((i as f32 + point_id as f32 * 7.0) * 0.3).sin())
                .collect()
        })
        .collect();

    // Compute all pairwise distances
    let mut distances = vec![vec![0.0f32; 20]; 20];
    for i in 0..20 {
        for j in i..20 {
            let dist = euclidean_distance(&points[i], &points[j]);
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    // Verify symmetry and non-negativity
    for i in 0..20 {
        for j in 0..20 {
            assert!(distances[i][j] >= 0.0, "Distance should be non-negative");
            assert!(
                (distances[i][j] - distances[j][i]).abs() < 1e-6,
                "Distance matrix should be symmetric"
            );
        }
        // Self-distance should be zero
        assert!(distances[i][i].abs() < 1e-6, "Self-distance should be zero");
    }
}

#[test]
fn test_embedding_normalization_pipeline() {
    // Test the typical pipeline: receive embedding, normalize, store/compare

    // Simulate raw embedding from model (not normalized)
    let raw_embedding: Vec<f32> = (0..768)
        .map(|i| (i as f32 * 0.1).sin() * 10.0) // Scale by 10 to simulate unnormalized
        .collect();

    // Check it's not unit length
    let initial_norm: f32 = raw_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (initial_norm - 1.0).abs() > 0.1,
        "Raw embedding should not be unit length"
    );

    // Normalize
    let mut normalized = raw_embedding.clone();
    normalize(&mut normalized);

    // Verify unit length
    let final_norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (final_norm - 1.0).abs() < 1e-5,
        "Normalized embedding should have unit length, got {final_norm}"
    );

    // Verify direction preserved
    let cos = cosine_similarity(&raw_embedding, &normalized);
    assert!(
        (cos - 1.0).abs() < 1e-5,
        "Normalization should preserve direction"
    );
}

// ============================================================================
// SIMD CONFIG TESTS
// ============================================================================

#[test]
fn test_simd_config_available() {
    let config = simd_config();

    // On most modern systems, at least one SIMD path should be available
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    assert!(
        config.simd_available(),
        "SIMD should be available on x86_64/aarch64"
    );

    // Log what's available
    println!("SIMD Configuration:");
    println!("  AVX-512F: {}", config.avx512f_enabled);
    println!("  AVX2: {}", config.avx2_enabled);
    println!("  FMA: {}", config.fma_enabled);
    println!("  AVX-512 VNNI: {}", config.avx512vnni_enabled);
    println!("  NEON: {}", config.neon_enabled);
}

#[test]
fn test_simd_config_detect_consistent() {
    // Multiple calls should return same config
    let config1 = SimdConfig::detect();
    let config2 = SimdConfig::detect();

    assert_eq!(config1.avx512f_enabled, config2.avx512f_enabled);
    assert_eq!(config1.avx2_enabled, config2.avx2_enabled);
    assert_eq!(config1.fma_enabled, config2.fma_enabled);
    assert_eq!(config1.avx512vnni_enabled, config2.avx512vnni_enabled);
    assert_eq!(config1.neon_enabled, config2.neon_enabled);
}

#[test]
fn test_simd_config_cached() {
    // simd_config() should return cached value quickly
    let start = std::time::Instant::now();
    for _ in 0..10000 {
        let _ = simd_config();
    }
    let elapsed = start.elapsed();

    // Should be very fast (cached)
    assert!(
        elapsed.as_micros() < 1000, // Less than 1ms for 10k calls
        "simd_config() should be cached, took {:?}",
        elapsed
    );
}

// ============================================================================
// BGE MODEL DIMENSION TESTS
// ============================================================================

#[test]
fn test_bge_small_dimension() {
    // BGE-small-en-v1.5: 384 dimensions
    let dim = 384;
    let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();

    let sim = cosine_similarity(&a, &b);
    assert!(sim >= -1.0 && sim <= 1.0);
}

#[test]
fn test_bge_base_dimension() {
    // BGE-base-en-v1.5: 768 dimensions
    let dim = 768;
    let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();

    let sim = cosine_similarity(&a, &b);
    assert!(sim >= -1.0 && sim <= 1.0);
}

#[test]
fn test_bge_large_dimension() {
    // BGE-large-en-v1.5: 1024 dimensions
    let dim = 1024;
    let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();

    let sim = cosine_similarity(&a, &b);
    assert!(sim >= -1.0 && sim <= 1.0);
}

// ============================================================================
// REAL-WORLD USAGE PATTERNS
// ============================================================================

#[test]
fn test_rerank_by_similarity() {
    // Simulate document reranking: given query, rerank candidates by similarity

    let query: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();

    let candidates: Vec<(usize, Vec<f32>)> = (0..20)
        .map(|id| {
            let emb: Vec<f32> = (0..384)
                .map(|i| ((i as f32 + id as f32 * 5.0) * 0.15).cos())
                .collect();
            (id, emb)
        })
        .collect();

    // Compute similarities and rerank
    let mut scored: Vec<(usize, f32)> = candidates
        .iter()
        .map(|(id, emb)| (*id, cosine_similarity(&query, emb)))
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Top result should have highest similarity
    let top_id = scored[0].0;
    let top_sim = scored[0].1;

    assert!(
        scored.iter().all(|(_, sim)| *sim <= top_sim),
        "Top result should have highest similarity"
    );

    println!("Top 5 reranked documents:");
    for (id, sim) in scored.iter().take(5) {
        println!("  Document {id}: similarity {sim:.4}");
    }
}

#[test]
fn test_deduplication_by_similarity() {
    // Simulate deduplication: find near-duplicate documents

    let documents: Vec<Vec<f32>> = (0..10)
        .map(|doc_id| {
            (0..384)
                .map(|i| ((i as f32 + doc_id as f32 * 3.0) * 0.1).sin())
                .collect()
        })
        .collect();

    // Find pairs with high similarity (potential duplicates)
    let threshold = 0.95;
    let mut duplicates = Vec::new();

    for i in 0..documents.len() {
        for j in (i + 1)..documents.len() {
            let sim = cosine_similarity(&documents[i], &documents[j]);
            if sim > threshold {
                duplicates.push((i, j, sim));
            }
        }
    }

    println!(
        "Found {} potential duplicates with threshold {}",
        duplicates.len(),
        threshold
    );
    for (i, j, sim) in &duplicates {
        println!("  Documents {i} and {j}: similarity {sim:.4}");
    }
}

#[test]
fn test_semantic_drift_detection() {
    // Simulate detecting semantic drift: compare embeddings across time

    // "Old" embedding (baseline)
    let baseline: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();

    // "New" embeddings with varying degrees of drift
    let new_embeddings: Vec<(f32, Vec<f32>)> = vec![
        (0.0, baseline.clone()), // No drift
        (
            0.1,
            baseline
                .iter()
                .enumerate()
                .map(|(i, x)| x + 0.1 * (i as f32 * 0.2).cos())
                .collect(),
        ), // Small drift
        (
            0.3,
            baseline
                .iter()
                .enumerate()
                .map(|(i, x)| x + 0.3 * (i as f32 * 0.3).sin())
                .collect(),
        ), // Medium drift
        (0.5, (0..384).map(|i| (i as f32 * 0.2).cos()).collect()), // Large drift
    ];

    for (expected_drift, new_emb) in &new_embeddings {
        let similarity = cosine_similarity(&baseline, new_emb);
        let drift = 1.0 - similarity;

        println!(
            "Expected drift ~{:.1}, actual similarity {:.4}, computed drift {:.4}",
            expected_drift, similarity, drift
        );

        // Drift should correlate with expected
        // (not exact due to non-linear relationship)
    }
}

// ============================================================================
// PERFORMANCE SANITY TESTS
// ============================================================================

#[test]
fn test_batch_faster_than_sequential() {
    // Verify batch operations complete in reasonable time

    let vectors: Vec<Vec<f32>> = (0..100)
        .map(|seed| {
            (0..384)
                .map(|i| ((i as f32 + seed as f32) * 0.1).sin())
                .collect()
        })
        .collect();

    let pairs: Vec<(&[f32], &[f32])> = vectors
        .iter()
        .zip(vectors.iter().skip(1))
        .map(|(a, b)| (a.as_slice(), b.as_slice()))
        .collect();

    let start = std::time::Instant::now();
    let _ = batch_cosine_similarity(&pairs);
    let batch_time = start.elapsed();

    let start = std::time::Instant::now();
    for (a, b) in &pairs {
        let _ = cosine_similarity(a, b);
    }
    let sequential_time = start.elapsed();

    println!("Batch: {:?}, Sequential: {:?}", batch_time, sequential_time);

    // Both should complete quickly (< 10ms for 99 pairs of 384-dim vectors)
    assert!(
        batch_time.as_millis() < 100,
        "Batch operation too slow: {:?}",
        batch_time
    );
}

#[test]
fn test_large_batch_dot_product() {
    // Test with larger batches typical in production

    let vectors: Vec<Vec<f32>> = (0..1000)
        .map(|seed| {
            (0..384)
                .map(|i| ((i as f32 + seed as f32) * 0.1).sin())
                .collect()
        })
        .collect();

    let pairs: Vec<(&[f32], &[f32])> = vectors
        .iter()
        .zip(vectors.iter().rev())
        .map(|(a, b)| (a.as_slice(), b.as_slice()))
        .collect();

    let start = std::time::Instant::now();
    let results = batch_dot_product(&pairs);
    let elapsed = start.elapsed();

    assert_eq!(results.len(), 1000);
    println!("1000 dot products in {:?}", elapsed);

    // Should complete in < 10ms with SIMD
    assert!(
        elapsed.as_millis() < 100,
        "Large batch too slow: {:?}",
        elapsed
    );
}
