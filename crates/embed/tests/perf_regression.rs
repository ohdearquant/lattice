//! Performance regression tests for lattice-embed.
//!
//! These are pass/fail time-bounded assertions, not benchmarks.
//! Bounds are deliberately generous (50-100x over expected) to remain
//! stable across debug builds and CI variance.

use std::time::Instant;

use lattice_embed::utils::{cosine_similarity, dot_product, normalize};

#[test]
fn cosine_similarity_not_catastrophically_slow() {
    // cosine_similarity at dim=384, 10 000 iterations must finish in < 2s.
    // SIMD path: ~90ns/iter -> 10k iters ~900µs; 2s gives >2000x slack.
    // Scalar debug path: ~650ns/iter -> 10k iters ~6.5ms; still well under 2s.
    let dim = 384usize;
    let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.002 + 0.1).collect();

    let start = Instant::now();
    for _ in 0..10_000 {
        let _sim = cosine_similarity(&a, &b);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 2,
        "cosine_similarity regression: 10k iters at dim=384 took {elapsed:?}, expected < 2s"
    );
}

#[test]
fn dot_product_not_catastrophically_slow() {
    // dot_product at dim=768, 10 000 iterations must finish in < 2s.
    let dim = 768usize;
    let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..dim).map(|i| 1.0 / (i as f32 + 1.0)).collect();

    let start = Instant::now();
    for _ in 0..10_000 {
        let _dp = dot_product(&a, &b);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 2,
        "dot_product regression: 10k iters at dim=768 took {elapsed:?}, expected < 2s"
    );
}

#[test]
fn normalize_not_catastrophically_slow() {
    // normalize at dim=384, 5 000 iterations must finish in < 2s.
    let dim = 384usize;

    let start = Instant::now();
    for i in 0..5_000 {
        // Re-initialize each iter to keep the vector non-zero.
        let mut v: Vec<f32> = (0..dim).map(|j| ((i + j) as f32) * 0.001 + 0.1).collect();
        normalize(&mut v);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 2,
        "normalize regression: 5000 iters at dim=384 took {elapsed:?}, expected < 2s"
    );
}
