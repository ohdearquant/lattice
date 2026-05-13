//! Cosine similarity, nearest-neighbor search, and other vector operations.
//!
//! Run with:
//!   cargo run -p lattice-embed --example similarity
//!
//! This example does not require model weights: it exercises the SIMD vector
//! math in `lattice_embed::utils` directly on synthetic f32 data.

use lattice_embed::types::{DistanceMetric, EmbeddingKey, VectorDType, VectorNorm};
use lattice_embed::utils::{
    batch_cosine_similarity, batch_dot_product, cosine_similarity, dot_product, euclidean_distance,
    normalize,
};

fn main() {
    // -------------------------------------------------------------------------
    // 1. Cosine similarity basics
    // -------------------------------------------------------------------------
    println!("=== Cosine similarity ===");

    let identical = vec![1.0_f32, 2.0, 3.0];
    let sim_same = cosine_similarity(&identical, &identical);
    println!("identical vectors: {:.4}", sim_same); // 1.0

    let a = vec![1.0_f32, 0.0, 0.0];
    let b = vec![0.0_f32, 1.0, 0.0];
    let sim_ortho = cosine_similarity(&a, &b);
    println!("orthogonal vectors: {:.4}", sim_ortho); // 0.0

    let c = vec![1.0_f32, 0.0];
    let d = vec![-1.0_f32, 0.0];
    let sim_opposite = cosine_similarity(&c, &d);
    println!("opposite vectors: {:.4}", sim_opposite); // -1.0
    println!();

    // -------------------------------------------------------------------------
    // 2. Normalization and dot product as cosine shortcut
    // -------------------------------------------------------------------------
    println!("=== Normalization + dot product ===");

    let mut v1 = vec![3.0_f32, 4.0]; // magnitude = 5
    let original_mag: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("before normalize: magnitude = {:.4}", original_mag);

    normalize(&mut v1);
    let normalized_mag: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("after normalize:  magnitude = {:.4}", normalized_mag); // 1.0

    // For unit-norm vectors, dot == cosine (and is faster — skips norm computation)
    let mut v2 = vec![0.0_f32, 1.0];
    normalize(&mut v2);

    let dot = dot_product(&v1, &v2);
    let cosine = cosine_similarity(&v1, &v2);
    println!("dot product  = {:.6}", dot);
    println!("cosine sim   = {:.6}", cosine);
    println!("difference   = {:.2e}", (dot - cosine).abs()); // essentially 0
    println!();

    // -------------------------------------------------------------------------
    // 3. Euclidean distance
    // -------------------------------------------------------------------------
    println!("=== Euclidean distance ===");

    let origin = vec![0.0_f32, 0.0, 0.0];
    let point = vec![1.0_f32, 2.0, 2.0]; // distance = sqrt(1+4+4) = 3
    let dist = euclidean_distance(&origin, &point);
    println!("distance from origin to [1,2,2]: {:.4}", dist); // 3.0
    println!();

    // -------------------------------------------------------------------------
    // 4. Nearest-neighbor search (brute force)
    // -------------------------------------------------------------------------
    println!("=== Nearest-neighbor search ===");

    // Synthetic 384-dim embeddings, each normalized
    const DIMS: usize = 384;
    let query: Vec<f32> = {
        let mut v: Vec<f32> = (0..DIMS).map(|i| (i as f32 * 0.1).sin()).collect();
        normalize(&mut v);
        v
    };

    let corpus: Vec<Vec<f32>> = (0..20usize)
        .map(|doc_id| {
            let mut v: Vec<f32> = (0..DIMS)
                .map(|i| ((i as f32 + doc_id as f32 * 7.3) * 0.12).cos())
                .collect();
            normalize(&mut v);
            v
        })
        .collect();

    // Scalar loop (simple)
    let (best_idx, best_sim) = corpus
        .iter()
        .enumerate()
        .map(|(i, doc)| (i, cosine_similarity(&query, doc)))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("Best match: document {best_idx} (cosine = {best_sim:.4})");

    // Batch variant — internally uses the same SIMD path but lets the dispatcher
    // amortize any overhead over many pairs at once
    let pairs: Vec<(&[f32], &[f32])> = corpus
        .iter()
        .map(|doc| (query.as_slice(), doc.as_slice()))
        .collect();
    let batch_sims = batch_cosine_similarity(&pairs);

    let (batch_best_idx, batch_best_sim) = batch_sims
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    assert_eq!(best_idx, batch_best_idx);
    println!("Batch confirms: document {batch_best_idx} (cosine = {batch_best_sim:.4})");
    println!();

    // -------------------------------------------------------------------------
    // 5. Batch dot product
    // -------------------------------------------------------------------------
    println!("=== Batch dot product ===");

    let dot_pairs: Vec<(&[f32], &[f32])> = corpus
        .iter()
        .map(|doc| (query.as_slice(), doc.as_slice()))
        .collect();
    let dots = batch_dot_product(&dot_pairs);

    // For unit-norm vectors, dot products should match cosine similarities
    let max_diff: f32 = dots
        .iter()
        .zip(&batch_sims)
        .map(|(d, c)| (d - c).abs())
        .fold(0.0_f32, f32::max);

    println!(
        "Max |dot - cosine| across {} pairs: {:.2e}",
        dots.len(),
        max_diff
    );
    assert!(max_diff < 1e-4, "dot should equal cosine for unit vectors");
    println!();

    // -------------------------------------------------------------------------
    // 6. EmbeddingKey for index routing
    // -------------------------------------------------------------------------
    println!("=== EmbeddingKey ===");

    let key = EmbeddingKey::new(
        "bge-small-en-v1.5",
        "v1.5",
        384,
        DistanceMetric::Cosine,
        VectorDType::F32,
        VectorNorm::Unit,
    );

    println!("model:    {}", key.model);
    println!("revision: {}", key.revision);
    println!("dims:     {}", key.dims);
    println!("metric:   {:?}", key.metric);

    let canonical = key.canonical_bytes();
    println!("canonical bytes: {} bytes", canonical.len());

    // Two keys built from identical inputs must have identical canonical bytes
    let key2 = EmbeddingKey::new(
        "bge-small-en-v1.5",
        "v1.5",
        384,
        DistanceMetric::Cosine,
        VectorDType::F32,
        VectorNorm::Unit,
    );
    assert_eq!(key.canonical_bytes(), key2.canonical_bytes());
    println!("canonical bytes are deterministic: ok");

    // Different model => different bytes
    let key3 = EmbeddingKey::new(
        "bge-base-en-v1.5",
        "v1.5",
        768,
        DistanceMetric::Cosine,
        VectorDType::F32,
        VectorNorm::Unit,
    );
    assert_ne!(key.canonical_bytes(), key3.canonical_bytes());
    println!("different model => different bytes: ok");
    println!();

    println!("Done.");
}
