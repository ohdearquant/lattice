//! SIMD vs Scalar comparison benchmarks.
//!
//! Run with: `RUSTFLAGS="-C target-cpu=native" cargo bench -p lattice-embed -- simd`
//!
//! This benchmark explicitly compares SIMD-accelerated implementations against
//! scalar fallbacks to measure the speedup.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_embed::simd::{
    self, BinaryVector, Int4Vector, NormalizationHint, PreparedQuery, PreparedQueryWithMeta,
    QuantizationTier, QuantizedData, QuantizedVector, SimdConfig, approximate_cosine_distance,
    approximate_cosine_distance_prepared, approximate_cosine_distance_prepared_with_meta,
    approximate_dot_product_prepared,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Embedding dimensions to test.
const DIMENSIONS: [usize; 4] = [384, 768, 1024, 1536];

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

/// Generate a normalized random vector.
fn generate_normalized_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = generate_vector(dim, seed);
    simd::normalize(&mut v);
    v
}

// ============================================================================
// SCALAR IMPLEMENTATIONS (for comparison)
// ============================================================================

fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize_scalar(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv_norm = 1.0 / norm;
        vector.iter_mut().for_each(|x| *x *= inv_norm);
    }
}

fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ============================================================================
// COMPARISON BENCHMARKS
// ============================================================================

/// Compare SIMD vs scalar cosine similarity.
fn bench_cosine_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_cosine_similarity");

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

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| black_box(cosine_similarity_scalar(black_box(&a), black_box(&b))));
        });

        // SIMD (auto-detected)
        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::cosine_similarity(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

/// Compare SIMD vs scalar dot product (for normalized vectors).
fn bench_dot_product_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_dot_product");

    for dim in DIMENSIONS {
        // Use normalized vectors (common case for embeddings)
        let a = generate_normalized_vector(dim, 42);
        let b = generate_normalized_vector(dim, 123);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| black_box(dot_product_scalar(black_box(&a), black_box(&b))));
        });

        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::dot_product(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

/// Compare SIMD vs scalar normalization.
///
/// Note: Uses iter_batched to properly separate setup (vector copy) from
/// the measured operation (normalization), avoiding heap allocation in timing.
fn bench_normalize_simd_vs_scalar(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("simd_normalize");

    for dim in DIMENSIONS {
        let template = generate_vector(dim, 42);

        group.throughput(Throughput::Elements(dim as u64));

        let template_scalar = template.clone();
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter_batched(
                || template_scalar.clone(),
                |mut v| {
                    normalize_scalar(black_box(&mut v));
                    v
                },
                BatchSize::SmallInput,
            );
        });

        let template_simd = template.clone();
        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |bench, _| {
            bench.iter_batched(
                || template_simd.clone(),
                |mut v| {
                    simd::normalize(black_box(&mut v));
                    v
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Compare SIMD vs scalar Euclidean distance.
fn bench_euclidean_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_euclidean_distance");

    for dim in DIMENSIONS {
        let a = generate_vector(dim, 42);
        let b = generate_vector(dim, 123);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| black_box(euclidean_distance_scalar(black_box(&a), black_box(&b))));
        });

        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::euclidean_distance(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

/// Batch cosine similarity benchmark.
fn bench_batch_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_batch_cosine");

    for batch_size in [10, 100, 1000] {
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
            .map(|i| {
                (
                    generate_vector(384, i as u64),
                    generate_vector(384, i as u64 + 10000),
                )
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        // Scalar loop
        group.bench_with_input(
            BenchmarkId::new("scalar_loop", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| {
                    let results: Vec<f32> = pairs
                        .iter()
                        .map(|(a, b)| cosine_similarity_scalar(a, b))
                        .collect();
                    black_box(results)
                });
            },
        );

        // SIMD batch
        let pair_refs: Vec<(&[f32], &[f32])> = pairs
            .iter()
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("simd_batch", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| black_box(simd::batch_cosine_similarity(&pair_refs)));
            },
        );
    }

    group.finish();
}

/// Throughput test: operations per second at 384-dim (BGE-small).
fn bench_throughput_384(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_throughput_384");
    group.throughput(Throughput::Elements(1));

    let a = generate_normalized_vector(384, 42);
    let b = generate_normalized_vector(384, 123);

    group.bench_function("dot_product", |bench| {
        bench.iter(|| black_box(simd::dot_product(black_box(&a), black_box(&b))));
    });

    group.bench_function("cosine_similarity", |bench| {
        bench.iter(|| black_box(simd::cosine_similarity(black_box(&a), black_box(&b))));
    });

    let template = generate_vector(384, 42);
    group.bench_function("normalize", |bench| {
        bench.iter(|| {
            let mut v = template.clone();
            simd::normalize(black_box(&mut v));
            black_box(v)
        });
    });

    group.bench_function("euclidean_distance", |bench| {
        bench.iter(|| black_box(simd::euclidean_distance(black_box(&a), black_box(&b))));
    });

    group.finish();
}

// ============================================================================
// INT8 QUANTIZATION BENCHMARKS
// ============================================================================

/// Benchmark int8 quantization process.
fn bench_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_quantization");

    for dim in DIMENSIONS {
        let v = generate_normalized_vector(dim, 42);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("quantize", dim), &dim, |bench, _| {
            bench.iter(|| black_box(QuantizedVector::from_f32(black_box(&v))));
        });
    }

    group.finish();
}

/// Compare float32 vs int8 cosine similarity.
fn bench_int8_vs_float32(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_vs_float32_cosine");

    for dim in DIMENSIONS {
        let a = generate_normalized_vector(dim, 42);
        let b = generate_normalized_vector(dim, 123);

        // Pre-quantize vectors (amortized in real usage)
        let a_q = QuantizedVector::from_f32(&a);
        let b_q = QuantizedVector::from_f32(&b);

        group.throughput(Throughput::Elements(dim as u64));

        // Float32 SIMD
        group.bench_with_input(BenchmarkId::new("float32_simd", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::cosine_similarity(black_box(&a), black_box(&b))));
        });

        // Int8 quantized
        group.bench_with_input(BenchmarkId::new("int8", dim), &dim, |bench, _| {
            bench.iter(|| black_box(a_q.cosine_similarity(black_box(&b_q))));
        });
    }

    group.finish();
}

/// Int8 batch cosine similarity.
fn bench_int8_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_batch_cosine");

    for batch_size in [10, 100, 1000] {
        // Generate and pre-quantize
        let pairs_f32: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
            .map(|i| {
                (
                    generate_normalized_vector(384, i as u64),
                    generate_normalized_vector(384, i as u64 + 10000),
                )
            })
            .collect();

        let pairs_i8: Vec<(QuantizedVector, QuantizedVector)> = pairs_f32
            .iter()
            .map(|(a, b)| (QuantizedVector::from_f32(a), QuantizedVector::from_f32(b)))
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        // Float32 SIMD batch
        let pair_refs: Vec<(&[f32], &[f32])> = pairs_f32
            .iter()
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("float32_simd", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| black_box(simd::batch_cosine_similarity(&pair_refs)));
            },
        );

        // Int8 loop
        group.bench_with_input(
            BenchmarkId::new("int8_loop", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| {
                    let results: Vec<f32> = pairs_i8
                        .iter()
                        .map(|(a, b)| a.cosine_similarity(b))
                        .collect();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Memory comparison: float32 vs int8 with search simulation.
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_size");

    for dim in DIMENSIONS {
        let v = generate_normalized_vector(dim, 42);
        let v_q = QuantizedVector::from_f32(&v);

        let f32_bytes = v.len() * std::mem::size_of::<f32>();
        let i8_bytes = v_q.data.len() * std::mem::size_of::<i8>() + 16; // + overhead

        println!(
            "dim={}: float32={}B, int8={}B, ratio={:.1}x",
            dim,
            f32_bytes,
            i8_bytes,
            f32_bytes as f64 / i8_bytes as f64
        );
    }

    // Throughput benchmark with memory pressure
    let vectors_f32: Vec<Vec<f32>> = (0..1000)
        .map(|i| generate_normalized_vector(384, i as u64))
        .collect();

    let vectors_i8: Vec<QuantizedVector> = vectors_f32
        .iter()
        .map(|v| QuantizedVector::from_f32(v))
        .collect();

    let query = generate_normalized_vector(384, 99999);
    let query_q = QuantizedVector::from_f32(&query);

    group.throughput(Throughput::Elements(1000));

    group.bench_function("search_1000_float32", |bench| {
        bench.iter(|| {
            let results: Vec<f32> = vectors_f32
                .iter()
                .map(|v| simd::cosine_similarity(&query, v))
                .collect();
            black_box(results)
        });
    });

    group.bench_function("search_1000_int8", |bench| {
        bench.iter(|| {
            let results: Vec<f32> = vectors_i8
                .iter()
                .map(|v| query_q.cosine_similarity(v))
                .collect();
            black_box(results)
        });
    });

    group.finish();
}

// ============================================================================
// INT4 QUANTIZATION BENCHMARKS
// ============================================================================

fn bench_int4_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("int4_cosine_distance");

    for dim in DIMENSIONS {
        let a = generate_normalized_vector(dim, 42);
        let b = generate_normalized_vector(dim, 123);

        let a_q = Int4Vector::from_f32(&a);
        let b_q = Int4Vector::from_f32(&b);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("float32_simd", dim), &dim, |bench, _| {
            bench.iter(|| black_box(1.0 - simd::cosine_similarity(black_box(&a), black_box(&b))));
        });

        group.bench_with_input(BenchmarkId::new("int4", dim), &dim, |bench, _| {
            bench.iter(|| black_box(a_q.cosine_distance(black_box(&b_q))));
        });
    }

    group.finish();
}

// ============================================================================
// BINARY QUANTIZATION BENCHMARKS
// ============================================================================

fn bench_binary_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_cosine_distance");

    for dim in DIMENSIONS {
        let a = generate_normalized_vector(dim, 42);
        let b = generate_normalized_vector(dim, 123);

        let a_q = BinaryVector::from_f32(&a);
        let b_q = BinaryVector::from_f32(&b);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("float32_simd", dim), &dim, |bench, _| {
            bench.iter(|| black_box(1.0 - simd::cosine_similarity(black_box(&a), black_box(&b))));
        });

        group.bench_with_input(BenchmarkId::new("binary", dim), &dim, |bench, _| {
            bench.iter(|| black_box(a_q.cosine_distance_approx(black_box(&b_q))));
        });
    }

    group.finish();
}

// ============================================================================
// BATCH DOT PRODUCT BENCHMARKS (H3: resolve kernel once per batch)
// ============================================================================

fn bench_batch_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_batch_dot_product");

    for batch_size in [10, 100, 1000] {
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
            .map(|i| {
                (
                    generate_vector(384, i as u64),
                    generate_vector(384, i as u64 + 10000),
                )
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar_loop", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| {
                    let results: Vec<f32> = pairs
                        .iter()
                        .map(|(a, b)| dot_product_scalar(a, b))
                        .collect();
                    black_box(results)
                });
            },
        );

        let pair_refs: Vec<(&[f32], &[f32])> = pairs
            .iter()
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("simd_batch", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| black_box(simd::batch_dot_product(&pair_refs)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// PREPARED QUERY BENCHMARKS (H1: quantize query once per tier vs per call)
// ============================================================================

/// Demonstrates the cost of per-call query re-quantization in tiered loops.
///
/// "per_call": current `approximate_cosine_distance` path — re-quantizes on every call.
/// "query_once": query is pre-quantized once; inner SIMD kernel called directly.
///
/// This is the primary benchmark for the prepared-query optimization in tier.rs.
fn bench_prepared_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier_prepared_query");
    const COUNT: usize = 1000;
    const DIM: usize = 384;

    let query_f32 = generate_vector(DIM, 42);

    let stored_f32: Vec<Vec<f32>> = (0..COUNT)
        .map(|i| generate_vector(DIM, i as u64 + 1))
        .collect();

    let stored_int8: Vec<QuantizedVector> = stored_f32
        .iter()
        .map(|v| QuantizedVector::from_f32(v))
        .collect();
    let stored_int8_data: Vec<QuantizedData> = stored_int8
        .iter()
        .map(|q| QuantizedData::Int8(q.clone()))
        .collect();

    let stored_int4: Vec<Int4Vector> = stored_f32.iter().map(|v| Int4Vector::from_f32(v)).collect();
    let stored_int4_data: Vec<QuantizedData> = stored_int4
        .iter()
        .map(|q| QuantizedData::Int4(q.clone()))
        .collect();

    let stored_binary: Vec<BinaryVector> = stored_f32
        .iter()
        .map(|v| BinaryVector::from_f32(v))
        .collect();
    let stored_binary_data: Vec<QuantizedData> = stored_binary
        .iter()
        .map(|q| QuantizedData::Binary(q.clone()))
        .collect();

    group.throughput(Throughput::Elements(COUNT as u64));

    // INT8: current path re-quantizes query on each call
    group.bench_function("int8_query_per_call/1000", |bench| {
        bench.iter(|| {
            let results: Vec<f32> = stored_int8_data
                .iter()
                .map(|s| approximate_cosine_distance(black_box(&query_f32), s))
                .collect();
            black_box(results)
        });
    });

    // INT8: prepared query via public API — quantize once, call approximate_cosine_distance_prepared
    let pre_q_int8 = PreparedQuery::from_f32(&query_f32, QuantizationTier::Int8);
    group.bench_function("int8_query_once/1000", |bench| {
        bench.iter(|| {
            let results: Vec<f32> = stored_int8_data
                .iter()
                .map(|s| approximate_cosine_distance_prepared(black_box(&pre_q_int8), s))
                .collect();
            black_box(results)
        });
    });

    // INT4: current path re-quantizes query on each call
    group.bench_function("int4_query_per_call/1000", |bench| {
        bench.iter(|| {
            let results: Vec<f32> = stored_int4_data
                .iter()
                .map(|s| approximate_cosine_distance(black_box(&query_f32), s))
                .collect();
            black_box(results)
        });
    });

    // INT4: prepared query via public API
    let pre_q_int4 = PreparedQuery::from_f32(&query_f32, QuantizationTier::Int4);
    group.bench_function("int4_query_once/1000", |bench| {
        bench.iter(|| {
            let results: Vec<f32> = stored_int4_data
                .iter()
                .map(|s| approximate_cosine_distance_prepared(black_box(&pre_q_int4), s))
                .collect();
            black_box(results)
        });
    });

    // Binary: current path re-quantizes query on each call
    group.bench_function("binary_query_per_call/1000", |bench| {
        bench.iter(|| {
            let results: Vec<f32> = stored_binary_data
                .iter()
                .map(|s| approximate_cosine_distance(black_box(&query_f32), s))
                .collect();
            black_box(results)
        });
    });

    // Binary: prepared query via public API
    let pre_q_binary = PreparedQuery::from_f32(&query_f32, QuantizationTier::Binary);
    group.bench_function("binary_query_once/1000", |bench| {
        bench.iter(|| {
            let results: Vec<f32> = stored_binary_data
                .iter()
                .map(|s| approximate_cosine_distance_prepared(black_box(&pre_q_binary), s))
                .collect();
            black_box(results)
        });
    });

    group.finish();
}

// ============================================================================
// RAW INT8 DOT BENCHMARKS (H4/H5: isolate dispatch overhead in raw i8 kernel)
// ============================================================================

/// Isolates the raw INT8 dot product dispatch path.
///
/// Compares:
/// - `dot_product_i8`: public wrapper with release-mode invariant scans + dispatch
/// - `dot_product_i8_raw`: raw slice path — dispatch only, no invariant scans
fn bench_dot_product_i8_raw(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_raw_dot_product");

    for dim in [127usize, 128, 129, 384, 768, 1024] {
        let a_q = QuantizedVector::from_f32(&generate_vector(dim, 42));
        let b_q = QuantizedVector::from_f32(&generate_vector(dim, 123));

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("dot_product_i8", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::dot_product_i8(black_box(&a_q), black_box(&b_q))));
        });

        group.bench_with_input(
            BenchmarkId::new("dot_product_i8_raw", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    black_box(simd::dot_product_i8_raw(
                        black_box(&a_q.data),
                        black_box(&b_q.data),
                    ))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// PREPARED INT8 DOT PRODUCT BENCHMARKS
// ============================================================================

/// Benchmarks prepared vs per-call INT8 dot product paths.
///
/// `per_call`: re-quantizes query f32→INT8 on every invocation.
/// `prepared`: query is pre-quantized; uses `dot_product_i8_trusted` (no release assertions).
///
/// Also covers the SIMD boundary dims (127/128/129) where tail handling matters.
fn bench_int8_prepared_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_prepared_dot_product");

    for dim in [127usize, 128, 129, 384, 768, 1024] {
        let query_f32 = generate_vector(dim, 42);
        let stored_q = QuantizedVector::from_f32(&generate_vector(dim, 123));
        let stored_data = QuantizedData::Int8(stored_q);

        group.throughput(Throughput::Elements(dim as u64));

        // Per-call path: approximate_dot_product re-quantizes query on every call.
        group.bench_with_input(BenchmarkId::new("per_call", dim), &dim, |bench, _| {
            bench.iter(|| {
                black_box(simd::approximate_dot_product(
                    black_box(&query_f32),
                    black_box(&stored_data),
                ))
            });
        });

        // Prepared path: query quantized once; dispatch goes through dot_product_i8_trusted.
        let prepared = PreparedQuery::from_f32(&query_f32, QuantizationTier::Int8);
        group.bench_with_input(BenchmarkId::new("prepared", dim), &dim, |bench, _| {
            bench.iter(|| {
                black_box(approximate_dot_product_prepared(
                    black_box(&prepared),
                    black_box(&stored_data),
                ))
            });
        });
    }

    group.finish();
}

// ============================================================================
// ROUND 3: NORMALIZED COSINE FAST PATH BENCHMARKS (H1 baseline)
// ============================================================================

/// Baseline for normalized-cosine fast path (Round 3, Hypothesis 1).
///
/// Compares full cosine (two norm ops + division) vs dot product on pre-normalized
/// unit vectors at [384, 768, 1024] dims. Shows the theoretical speedup available.
/// After i2 adds `approximate_cosine_distance_prepared_with_meta`, it should
/// approach dot-product speed on the same unit corpus.
fn bench_normalized_cosine_fast_path(c: &mut Criterion) {
    const BENCH_DIMS: [usize; 3] = [384, 768, 1024];
    let mut group = c.benchmark_group("simd_normalized_cosine_fast_path");

    for dim in BENCH_DIMS {
        let a = generate_normalized_vector(dim, 42);
        let b = generate_normalized_vector(dim, 123);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("cosine_full", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::cosine_similarity(black_box(&a), black_box(&b))));
        });

        group.bench_with_input(BenchmarkId::new("dot_product", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::dot_product(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

// ============================================================================
// ROUND 3: SQUARED EUCLIDEAN FAST PATH BENCHMARKS (H3 baseline)
// ============================================================================

/// Baseline for squared-L2 fast path (Round 3, Hypothesis 3).
///
/// Measures `euclidean_distance` (with final sqrt) at [384, 768, 1024] dims.
/// After i2 adds `simd::squared_euclidean_distance`, this becomes a before/after
/// comparison: squared variant skips the sqrt and is used for HNSW internal ordering.
fn bench_squared_euclidean_fast_path(c: &mut Criterion) {
    const BENCH_DIMS: [usize; 3] = [384, 768, 1024];
    let mut group = c.benchmark_group("simd_squared_euclidean_fast_path");

    for dim in BENCH_DIMS {
        let a = generate_vector(dim, 42);
        let b = generate_vector(dim, 123);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("euclidean_full", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::euclidean_distance(black_box(&a), black_box(&b))));
        });

        group.bench_with_input(
            BenchmarkId::new("squared_euclidean", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    black_box(simd::squared_euclidean_distance(
                        black_box(&a),
                        black_box(&b),
                    ))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// ROUND 3: PREPARED QUERY NORMALIZED COSINE BENCHMARKS (H1 batch baseline)
// ============================================================================

/// Baseline for prepared normalized-cosine batch path (Round 3, Hypothesis 1).
///
/// Measures `approximate_cosine_distance_prepared` on `QuantizedData::Full` unit
/// vectors at [384, 768, 1024] dims (1000 stored vectors). After i2 adds
/// `approximate_cosine_distance_prepared_with_meta` with `VectorNorm::Unit`, the
/// new path should improve by at least 15% for 768/1024 by routing to dot product.
fn bench_prepared_query_normalized_cosine(c: &mut Criterion) {
    const BENCH_DIMS: [usize; 3] = [384, 768, 1024];
    const STORED_COUNT: usize = 1000;
    let mut group = c.benchmark_group("simd_prepared_query_normalized_cosine");

    for dim in BENCH_DIMS {
        let query_unit = generate_normalized_vector(dim, 999);

        let stored_full: Vec<QuantizedData> = (0..STORED_COUNT)
            .map(|i| QuantizedData::Full(generate_normalized_vector(dim, i as u64)))
            .collect();

        group.throughput(Throughput::Elements(STORED_COUNT as u64));

        let prepared_full = PreparedQuery::from_f32(&query_unit, QuantizationTier::Full);
        group.bench_with_input(
            BenchmarkId::new("prepared_full_cosine", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    let results: Vec<f32> = stored_full
                        .iter()
                        .map(|s| approximate_cosine_distance_prepared(black_box(&prepared_full), s))
                        .collect();
                    black_box(results)
                });
            },
        );

        let query_vec = query_unit.clone();
        group.bench_with_input(
            BenchmarkId::new("dot_product_loop", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    let results: Vec<f32> = stored_full
                        .iter()
                        .map(|s| match s {
                            QuantizedData::Full(v) => {
                                simd::dot_product(black_box(&query_vec), black_box(v))
                            }
                            _ => 0.0,
                        })
                        .collect();
                    black_box(results)
                });
            },
        );

        let meta = PreparedQueryWithMeta::from_f32(
            &query_unit,
            QuantizationTier::Full,
            NormalizationHint::Unit,
        );
        group.bench_with_input(
            BenchmarkId::new("prepared_meta_unit", dim),
            &dim,
            |bench, _| {
                bench.iter(|| {
                    let results: Vec<f32> = stored_full
                        .iter()
                        .map(|s| {
                            approximate_cosine_distance_prepared_with_meta(
                                black_box(&meta),
                                s,
                                NormalizationHint::Unit,
                            )
                        })
                        .collect();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// ROUND 4: QUERY-VS-N BATCH DOT PRODUCT BENCHMARKS (Hypothesis 2 baseline)
// ============================================================================

/// Baseline for query-vs-N batch dot product (Round 4, Hypothesis 2).
///
/// One fixed query against N candidates. Current path uses `simd::dot_product`
/// per pair (pair_loop) or `simd::batch_dot_product` with same-query pairs
/// (simd_batch). After i2 adds `dot_product_batch4`, the simd_batch path routes
/// through the batch-4 kernel for 4-vector chunks. These numbers are the "before".
fn bench_query_batch_dot_product(c: &mut Criterion) {
    const BENCH_DIMS: [usize; 3] = [128, 384, 768];
    const BATCH_SIZES: [usize; 4] = [4, 16, 64, 256];
    let mut group = c.benchmark_group("simd_query_batch_dot_product");

    for dim in BENCH_DIMS {
        let query = generate_normalized_vector(dim, 42);
        let candidates: Vec<Vec<f32>> = (0..256)
            .map(|i| generate_vector(dim, i as u64 + 1))
            .collect();

        for &count in &BATCH_SIZES {
            let cands = &candidates[..count];
            group.throughput(Throughput::Elements(count as u64));

            group.bench_with_input(
                BenchmarkId::new("pair_loop", format!("{dim}d_{count}c")),
                &count,
                |bench, _| {
                    bench.iter(|| {
                        let results: Vec<f32> = cands
                            .iter()
                            .map(|c| simd::dot_product(black_box(&query), black_box(c)))
                            .collect();
                        black_box(results)
                    });
                },
            );

            let pair_refs: Vec<(&[f32], &[f32])> = cands
                .iter()
                .map(|c| (query.as_slice(), c.as_slice()))
                .collect();

            group.bench_with_input(
                BenchmarkId::new("simd_batch", format!("{dim}d_{count}c")),
                &count,
                |bench, _| {
                    bench.iter(|| black_box(simd::batch_dot_product(black_box(&pair_refs))));
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// ROUND 4: NORMALIZED QUERY BATCH COSINE BENCHMARKS (Hypothesis 3 baseline)
// ============================================================================

/// Baseline for normalized query-vs-N batch cosine (Round 4, Hypothesis 3).
///
/// One unit-normalized query against N unit-normalized candidates. After Round 4,
/// `batch_cosine_similarity` should detect all-unit pairs and route to
/// `batch_dot_product`, which then uses the batch-4 dot kernel. pair_loop_dot
/// is the theoretical lower bound (pure dot on pre-normalized vectors).
fn bench_batch_cosine_normalized_query(c: &mut Criterion) {
    const BENCH_DIMS: [usize; 3] = [384, 768, 1024];
    const BATCH_SIZES: [usize; 5] = [4, 16, 64, 256, 1000];
    let mut group = c.benchmark_group("simd_batch_cosine_normalized_query");

    for dim in BENCH_DIMS {
        let query = generate_normalized_vector(dim, 42);
        let candidates: Vec<Vec<f32>> = (0..1000)
            .map(|i| generate_normalized_vector(dim, i as u64 + 1))
            .collect();

        for &count in &BATCH_SIZES {
            let cands = &candidates[..count];
            group.throughput(Throughput::Elements(count as u64));

            group.bench_with_input(
                BenchmarkId::new("pair_loop_dot", format!("{dim}d_{count}c")),
                &count,
                |bench, _| {
                    bench.iter(|| {
                        let results: Vec<f32> = cands
                            .iter()
                            .map(|c| simd::dot_product(black_box(&query), black_box(c)))
                            .collect();
                        black_box(results)
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("pair_loop_cosine", format!("{dim}d_{count}c")),
                &count,
                |bench, _| {
                    bench.iter(|| {
                        let results: Vec<f32> = cands
                            .iter()
                            .map(|c| simd::cosine_similarity(black_box(&query), black_box(c)))
                            .collect();
                        black_box(results)
                    });
                },
            );

            let pair_refs: Vec<(&[f32], &[f32])> = cands
                .iter()
                .map(|c| (query.as_slice(), c.as_slice()))
                .collect();

            group.bench_with_input(
                BenchmarkId::new("simd_batch", format!("{dim}d_{count}c")),
                &count,
                |bench, _| {
                    bench.iter(|| black_box(simd::batch_cosine_similarity(black_box(&pair_refs))));
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// ROUND 4: NON-NORMALIZED QUERY BATCH COSINE BENCHMARKS (Hypothesis 3 baseline)
// ============================================================================

/// Baseline for non-normalized query-vs-N batch cosine (Round 4, Hypothesis 3).
///
/// One non-normalized query against N non-normalized candidates. Isolates the
/// case where `batch_cosine_similarity` must compute full cosine per pair (no unit
/// shortcut). After Round 4, chunks of 4 route through `cosine_similarity_batch4_*`
/// kernels that fuse dot and norm accumulation into a single pass.
fn bench_batch_cosine_non_normalized_query(c: &mut Criterion) {
    const BENCH_DIMS: [usize; 3] = [384, 768, 1024];
    const BATCH_SIZES: [usize; 5] = [4, 16, 64, 256, 1000];
    let mut group = c.benchmark_group("simd_batch_cosine_non_normalized_query");

    for dim in BENCH_DIMS {
        let query = generate_vector(dim, 42);
        let candidates: Vec<Vec<f32>> = (0..1000)
            .map(|i| generate_vector(dim, i as u64 + 1))
            .collect();

        for &count in &BATCH_SIZES {
            let cands = &candidates[..count];
            group.throughput(Throughput::Elements(count as u64));

            group.bench_with_input(
                BenchmarkId::new("pair_loop", format!("{dim}d_{count}c")),
                &count,
                |bench, _| {
                    bench.iter(|| {
                        let results: Vec<f32> = cands
                            .iter()
                            .map(|c| simd::cosine_similarity(black_box(&query), black_box(c)))
                            .collect();
                        black_box(results)
                    });
                },
            );

            let pair_refs: Vec<(&[f32], &[f32])> = cands
                .iter()
                .map(|c| (query.as_slice(), c.as_slice()))
                .collect();

            group.bench_with_input(
                BenchmarkId::new("simd_batch", format!("{dim}d_{count}c")),
                &count,
                |bench, _| {
                    bench.iter(|| black_box(simd::batch_cosine_similarity(black_box(&pair_refs))));
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    simd_benches,
    bench_cosine_simd_vs_scalar,
    bench_dot_product_simd_vs_scalar,
    bench_normalize_simd_vs_scalar,
    bench_euclidean_simd_vs_scalar,
    bench_batch_cosine,
    bench_throughput_384,
    bench_quantization,
    bench_int8_vs_float32,
    bench_int8_batch,
    bench_memory_usage,
    bench_int4_cosine_distance,
    bench_binary_cosine_distance,
    bench_batch_dot_product,
    bench_prepared_query,
    bench_dot_product_i8_raw,
    bench_int8_prepared_dot_product,
    bench_normalized_cosine_fast_path,
    bench_squared_euclidean_fast_path,
    bench_prepared_query_normalized_cosine,
    bench_query_batch_dot_product,
    bench_batch_cosine_normalized_query,
    bench_batch_cosine_non_normalized_query,
);

criterion_main!(simd_benches);
