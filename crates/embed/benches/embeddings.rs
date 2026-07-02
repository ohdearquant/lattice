//! Benchmarks for embedding utilities and services.
//!
//! Run with: `cargo bench -p lattice-embed`
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

// Core utility benchmarks (always run)
criterion_group!(
    utility_benches,
    bench_cosine_similarity,
    bench_normalize,
    bench_euclidean_distance,
    bench_batch_cosine_similarity,
    bench_batch_normalize,
);

criterion_main!(utility_benches);
