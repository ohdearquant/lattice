//! Benchmarks for embedding utilities and services.
//!
//! Run with: `cargo bench -p lattice-embed`
//!
//! For embedding service benchmarks (requires model download):
//! `cargo bench -p lattice-embed --features local -- --ignored`
//!
//! Results are saved to `target/criterion/`.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_embed::utils::{cosine_similarity, euclidean_distance, normalize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Embedding dimension for BGE-small model.
const DIM_384: usize = 384;

/// Embedding dimension for BGE-base model.
const DIM_768: usize = 768;

/// Embedding dimension for BGE-large model.
const DIM_1024: usize = 1024;

/// Generate a deterministic random vector of the specified dimension.
fn generate_vector(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            (hasher.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Generate N random vectors of the specified dimension.
fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count).map(|i| generate_vector(dim, i as u64)).collect()
}

/// Benchmark cosine similarity for different vector dimensions.
fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for dim in [DIM_384, DIM_768, DIM_1024] {
        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            let a = generate_vector(dim, 42);
            let b_vec = generate_vector(dim, 123);

            b.iter(|| black_box(cosine_similarity(black_box(&a), black_box(&b_vec))));
        });
    }

    group.finish();
}

/// Benchmark vector normalization for different dimensions.
fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");

    for dim in [DIM_384, DIM_768, DIM_1024] {
        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            // Clone fresh vector each iteration since normalize is in-place
            let template = generate_vector(dim, 42);

            b.iter(|| {
                let mut v = template.clone();
                normalize(black_box(&mut v));
                black_box(v)
            });
        });
    }

    group.finish();
}

/// Benchmark Euclidean distance for different vector dimensions.
fn bench_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance");

    for dim in [DIM_384, DIM_768, DIM_1024] {
        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            let a = generate_vector(dim, 42);
            let b_vec = generate_vector(dim, 123);

            b.iter(|| black_box(euclidean_distance(black_box(&a), black_box(&b_vec))));
        });
    }

    group.finish();
}

/// Benchmark batch cosine similarity (computing similarity for many vector pairs).
fn bench_batch_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_cosine_similarity");

    for batch_size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let vectors_a = generate_vectors(batch_size, DIM_384);
                let vectors_b = generate_vectors(batch_size, DIM_384);

                b.iter(|| {
                    let sims: Vec<f32> = vectors_a
                        .iter()
                        .zip(vectors_b.iter())
                        .map(|(a, b)| cosine_similarity(a, b))
                        .collect();
                    black_box(sims)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch normalization.
fn bench_batch_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_normalize");

    for batch_size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let template = generate_vectors(batch_size, DIM_384);

                b.iter(|| {
                    let mut vectors = template.clone();
                    for v in &mut vectors {
                        normalize(v);
                    }
                    black_box(vectors)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Embedding Service Benchmarks (require fastembed model download)
// ============================================================================
// These benchmarks are marked #[ignore] because they require:
// 1. The fastembed model to be downloaded (~100MB)
// 2. Significant CPU time for model loading and inference
//
// Run with: cargo bench -p lattice-embed --features local -- --ignored
// ============================================================================

#[cfg(feature = "local")]
mod service_benchmarks {
    use super::*;
    use lattice_embed::{EmbeddingModel, EmbeddingService, LocalEmbeddingService};
    use std::sync::Arc;
    use tokio::runtime::Runtime;

    /// Create a runtime for async benchmarks.
    fn create_runtime() -> Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create runtime")
    }

    /// Benchmark single text embedding latency.
    ///
    /// This measures the time to embed a single text, which includes:
    /// - Text preprocessing
    /// - Model inference
    /// - Result extraction
    ///
    /// Note: First call includes model loading overhead.
    pub fn bench_embed_single(c: &mut Criterion) {
        let mut group = c.benchmark_group("embed_single");
        group.sample_size(10); // Reduced sample size due to model overhead

        let rt = create_runtime();
        let service = Arc::new(LocalEmbeddingService::new());

        // Pre-warm: load the model
        rt.block_on(async {
            let _ = service
                .embed_one("warmup text", EmbeddingModel::BgeSmallEnV15)
                .await;
        });

        let texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming software development",
            "Vector embeddings capture semantic meaning of text",
        ];

        for (i, text) in texts.iter().enumerate() {
            let service = service.clone();
            let text = text.to_string();

            group.bench_function(BenchmarkId::new("text", i), |b| {
                b.to_async(&rt).iter(|| {
                    let svc = service.clone();
                    let t = text.clone();
                    async move {
                        black_box(
                            svc.embed_one(&t, EmbeddingModel::BgeSmallEnV15)
                                .await
                                .expect("embed failed"),
                        )
                    }
                });
            });
        }

        group.finish();
    }

    /// Benchmark batch embedding throughput.
    ///
    /// Measures how embedding time scales with batch size.
    /// Larger batches are generally more efficient due to batched inference.
    pub fn bench_embed_batch(c: &mut Criterion) {
        let mut group = c.benchmark_group("embed_batch");
        group.sample_size(10); // Reduced sample size due to model overhead

        let rt = create_runtime();
        let service = Arc::new(LocalEmbeddingService::new());

        // Pre-warm: load the model
        rt.block_on(async {
            let _ = service
                .embed_one("warmup text", EmbeddingModel::BgeSmallEnV15)
                .await;
        });

        for batch_size in [1, 10, 100] {
            let texts: Vec<String> = (0..batch_size)
                .map(|i| format!("This is sample text number {i} for embedding benchmarks"))
                .collect();

            let service = service.clone();
            group.throughput(Throughput::Elements(batch_size as u64));

            group.bench_with_input(
                BenchmarkId::from_parameter(batch_size),
                &texts,
                |b, texts| {
                    let svc = service.clone();
                    b.to_async(&rt).iter(|| {
                        let svc = svc.clone();
                        let t = texts.clone();
                        async move {
                            black_box(
                                svc.embed(&t, EmbeddingModel::BgeSmallEnV15)
                                    .await
                                    .expect("embed failed"),
                            )
                        }
                    });
                },
            );
        }

        group.finish();
    }

    /// Benchmark model loading/switching time.
    ///
    /// This measures the cold-start latency when loading a model.
    /// Model switching is expensive and should be avoided in production.
    pub fn bench_model_loading(c: &mut Criterion) {
        let mut group = c.benchmark_group("model_loading");
        group.sample_size(10); // Model loading is expensive

        let rt = create_runtime();

        group.bench_function("bge_small_cold_start", |b| {
            b.to_async(&rt).iter(|| async {
                // Create fresh service to measure cold start
                let service = LocalEmbeddingService::new();
                black_box(
                    service
                        .embed_one("cold start test", EmbeddingModel::BgeSmallEnV15)
                        .await
                        .expect("embed failed"),
                )
            });
        });

        group.finish();
    }

    criterion_group! {
        name = service_benches;
        config = Criterion::default();
        targets = bench_embed_single, bench_embed_batch, bench_model_loading
    }
}

// Core utility benchmarks (always run)
criterion_group!(
    utility_benches,
    bench_cosine_similarity,
    bench_normalize,
    bench_euclidean_distance,
    bench_batch_cosine_similarity,
    bench_batch_normalize,
);

// Conditional service benchmarks
#[cfg(feature = "local")]
criterion_main!(utility_benches, service_benchmarks::service_benches);

#[cfg(not(feature = "local"))]
criterion_main!(utility_benches);
