//! Head-to-head comparison: lattice-embed vs SimSIMD
//!
//! Run with: `RUSTFLAGS="-C target-cpu=native" cargo bench -p lattice-embed --bench simsimd_comparison`
//!
//! This benchmark provides a comparison against the industry-standard SimSIMD library.
//!
//! ## Methodology Notes
//!
//! **API overhead differences**: SimSIMD's Rust bindings return `Option<f64>` requiring
//! unwrap/handling on each call. lattice assumes pre-validated slices in hot paths and
//! returns `f32` directly. This means SimSIMD pays ~2-5ns of additional per-call overhead
//! for Option handling and bounds checking that lattice does not.
//!
//! **Metric**: SimSIMD returns cosine DISTANCE (1 - similarity), so we convert to
//! similarity for comparison. This adds a subtraction operation to SimSIMD timings.
//!
//! For production use where slices are validated once at the API boundary, lattice's
//! approach is appropriate. For defensive programming where each call needs validation,
//! SimSIMD's approach is safer.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_embed::simd::{self, QuantizedVector, SimdConfig, dot_product_i8_raw};
use simsimd::SpatialSimilarity;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Embedding dimensions to test (matching common embedding models).
const DIMENSIONS: [usize; 4] = [384, 768, 1024, 1536];

/// Generate a deterministic random vector.
fn generate_vector(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            (hasher.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Generate a normalized vector.
fn generate_normalized_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = generate_vector(dim, seed);
    simd::normalize(&mut v);
    v
}

// ============================================================================
// COSINE SIMILARITY: lattice vs SimSIMD
// ============================================================================

fn bench_cosine_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_lattice_vs_simsimd");

    // Report SIMD capabilities
    let config = SimdConfig::detect();
    println!(
        "\n=== SIMD Capabilities: AVX2={}, FMA={}, AVX512-VNNI={}, NEON={} ===\n",
        config.avx2_enabled, config.fma_enabled, config.avx512vnni_enabled, config.neon_enabled
    );

    for dim in DIMENSIONS {
        let a = generate_vector(dim, 42);
        let b = generate_vector(dim, 123);

        group.throughput(Throughput::Elements(dim as u64));

        // lattice implementation
        group.bench_with_input(BenchmarkId::new("lattice", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::cosine_similarity(black_box(&a), black_box(&b))));
        });

        // SimSIMD implementation (returns cosine DISTANCE as f64, so we convert to similarity)
        group.bench_with_input(BenchmarkId::new("simsimd", dim), &dim, |bench, _| {
            bench.iter(|| {
                // SimSIMD returns distance (1 - similarity), convert back to similarity
                black_box(1.0 - f32::cosine(black_box(&a), black_box(&b)).unwrap_or(0.0) as f32)
            });
        });
    }

    group.finish();
}

// ============================================================================
// DOT PRODUCT: lattice vs SimSIMD
// ============================================================================

fn bench_dot_product_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_lattice_vs_simsimd");

    for dim in DIMENSIONS {
        let a = generate_normalized_vector(dim, 42);
        let b = generate_normalized_vector(dim, 123);

        group.throughput(Throughput::Elements(dim as u64));

        // lattice implementation
        group.bench_with_input(BenchmarkId::new("lattice", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::dot_product(black_box(&a), black_box(&b))));
        });

        // SimSIMD implementation (inner product, returns f64)
        group.bench_with_input(BenchmarkId::new("simsimd", dim), &dim, |bench, _| {
            bench.iter(|| black_box(f32::dot(black_box(&a), black_box(&b)).unwrap_or(0.0) as f32));
        });
    }

    group.finish();
}

// ============================================================================
// EUCLIDEAN DISTANCE: lattice vs SimSIMD
// ============================================================================

fn bench_euclidean_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_lattice_vs_simsimd");

    for dim in DIMENSIONS {
        let a = generate_vector(dim, 42);
        let b = generate_vector(dim, 123);

        group.throughput(Throughput::Elements(dim as u64));

        // lattice implementation
        group.bench_with_input(BenchmarkId::new("lattice", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::euclidean_distance(black_box(&a), black_box(&b))));
        });

        // SimSIMD implementation (sqeuclidean returns squared distance as f64)
        group.bench_with_input(BenchmarkId::new("simsimd", dim), &dim, |bench, _| {
            bench.iter(|| {
                black_box(
                    f32::sqeuclidean(black_box(&a), black_box(&b))
                        .unwrap_or(0.0)
                        .sqrt() as f32,
                )
            });
        });
    }

    group.finish();
}

// ============================================================================
// INT8 COMPARISON (if SimSIMD supports it)
// ============================================================================

fn bench_int8_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_lattice_vs_simsimd");

    for dim in DIMENSIONS {
        let a_f32 = generate_normalized_vector(dim, 42);
        let b_f32 = generate_normalized_vector(dim, 123);

        // lattice int8 quantized
        let a_q = QuantizedVector::from_f32(&a_f32);
        let b_q = QuantizedVector::from_f32(&b_f32);

        // Convert to i8 slices for SimSIMD
        let a_i8: Vec<i8> = a_q.data.clone();
        let b_i8: Vec<i8> = b_q.data.clone();

        group.throughput(Throughput::Elements(dim as u64));

        // lattice int8 dot product: public path with release-mode invariant assertions
        group.bench_with_input(BenchmarkId::new("lattice_i8_dot", dim), &dim, |bench, _| {
            bench.iter(|| black_box(a_q.dot_product(black_box(&b_q))));
        });

        // lattice int8 raw kernel: no invariant scans, pure dispatch + SIMD compute
        group.bench_with_input(
            BenchmarkId::new("lattice_i8_dot_raw", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    black_box(dot_product_i8_raw(
                        black_box(&a_q.data),
                        black_box(&b_q.data),
                    ))
                });
            },
        );

        // SimSIMD i8 dot product (returns f64)
        group.bench_with_input(BenchmarkId::new("simsimd_i8_dot", dim), &dim, |bench, _| {
            bench.iter(|| black_box(i8::dot(black_box(&a_i8), black_box(&b_i8)).unwrap_or(0.0)));
        });
    }

    group.finish();
}

// ============================================================================
// BATCH COMPARISON (1000 vectors search)
// ============================================================================

fn bench_batch_search_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_search_1000");

    let vectors: Vec<Vec<f32>> = (0..1000)
        .map(|i| generate_normalized_vector(384, i as u64))
        .collect();
    let query = generate_normalized_vector(384, 99999);

    group.throughput(Throughput::Elements(1000));

    // lattice per-pair cosine_similarity loop
    group.bench_function("lattice_f32", |bench| {
        bench.iter(|| {
            let results: Vec<f32> = vectors
                .iter()
                .map(|v| simd::cosine_similarity(&query, v))
                .collect();
            black_box(results)
        });
    });

    // lattice batch_cosine_similarity with same-query pairs (public batch API)
    let pair_refs: Vec<(&[f32], &[f32])> = vectors
        .iter()
        .map(|v| (query.as_slice(), v.as_slice()))
        .collect();
    group.bench_function("lattice_batch_cosine", |bench| {
        bench.iter(|| black_box(simd::batch_cosine_similarity(&pair_refs)));
    });

    // SimSIMD batch search (returns distance, convert to similarity)
    group.bench_function("simsimd_f32", |bench| {
        bench.iter(|| {
            let results: Vec<f32> = vectors
                .iter()
                .map(|v| 1.0 - f32::cosine(&query, v).unwrap_or(0.0) as f32)
                .collect();
            black_box(results)
        });
    });

    // lattice int8 batch search
    let vectors_i8: Vec<QuantizedVector> = vectors
        .iter()
        .map(|v| QuantizedVector::from_f32(v))
        .collect();
    let query_i8 = QuantizedVector::from_f32(&query);

    group.bench_function("lattice_i8", |bench| {
        bench.iter(|| {
            let results: Vec<f32> = vectors_i8
                .iter()
                .map(|v| query_i8.cosine_similarity(v))
                .collect();
            black_box(results)
        });
    });

    group.finish();
}

// ============================================================================
// ACCURACY VERIFICATION
// ============================================================================

fn verify_accuracy() {
    println!("\n=== Accuracy Verification ===\n");
    println!("NOTE: SimSIMD's 'cosine' returns DISTANCE (1-similarity), so we convert.\n");

    for dim in [384, 768] {
        let a = generate_vector(dim, 42);
        let b = generate_vector(dim, 123);

        let lattice_cos = simd::cosine_similarity(&a, &b) as f64;
        // SimSIMD returns cosine DISTANCE, convert to similarity
        let simsimd_cos = 1.0 - f32::cosine(&a, &b).unwrap_or(0.0);

        let lattice_dot = simd::dot_product(&a, &b) as f64;
        let simsimd_dot = f32::dot(&a, &b).unwrap_or(0.0);

        println!("dim={dim}");
        println!(
            "  cosine_sim: lattice={:.6}, simsimd={:.6}, diff={:.2e}",
            lattice_cos,
            simsimd_cos,
            (lattice_cos - simsimd_cos).abs()
        );
        println!(
            "  dot:        lattice={:.6}, simsimd={:.6}, diff={:.2e}",
            lattice_dot,
            simsimd_dot,
            (lattice_dot - simsimd_dot).abs()
        );
    }
    println!();
}

fn bench_with_accuracy_check(c: &mut Criterion) {
    verify_accuracy();
    bench_cosine_comparison(c);
}

criterion_group!(
    comparison_benches,
    bench_with_accuracy_check,
    bench_dot_product_comparison,
    bench_euclidean_comparison,
    bench_int8_comparison,
    bench_batch_search_comparison,
);

criterion_main!(comparison_benches);
