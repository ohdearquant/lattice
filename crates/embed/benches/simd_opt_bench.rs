//! Benchmark for optimized SIMD embedding distance kernels.
//!
//! Covers the optimizations added in perf(embed):
//! - `cosine_similarity_fused` (explicit single-pass API)
//! - `batch_cosine_one_vs_many` (pre-computed query norm, one-vs-N)
//! - `dot_product_i8_raw` (i8 dot product with software prefetch)
//!
//! Dimensions benchmarked: 128, 384, 768, 1536 (common embedding model sizes).
//!
//! Run with: `cargo bench -p lattice-embed --bench simd_opt_bench`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lattice_embed::simd::{
    QuantizedVector, batch_cosine_one_vs_many, batch_cosine_similarity, cosine_similarity,
    cosine_similarity_fused, dot_product_i8_raw,
};

/// Deterministic pseudo-random vector generator (no external deps).
fn gen_vec(dim: usize, seed: u64) -> Vec<f32> {
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

const DIMS: [usize; 4] = [128, 384, 768, 1536];

/// Benchmark `cosine_similarity` (existing, single-pass SIMD) vs
/// `cosine_similarity_fused` (same kernel, explicit API guarantee).
fn bench_cosine_fused_vs_original(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for &dim in &DIMS {
        let a = gen_vec(dim, 0x1111);
        let b = gen_vec(dim, 0x2222);

        group.bench_with_input(BenchmarkId::new("original", dim), &dim, |bencher, _| {
            bencher.iter(|| {
                std::hint::black_box(cosine_similarity(
                    std::hint::black_box(&a),
                    std::hint::black_box(&b),
                ))
            });
        });

        group.bench_with_input(BenchmarkId::new("fused", dim), &dim, |bencher, _| {
            bencher.iter(|| {
                std::hint::black_box(cosine_similarity_fused(
                    std::hint::black_box(&a),
                    std::hint::black_box(&b),
                ))
            });
        });
    }

    group.finish();
}

/// Benchmark `batch_cosine_similarity` (per-pair, no shared norm) vs
/// `batch_cosine_one_vs_many` (shared query norm, pre-computed once).
///
/// Both run 16 candidates against one query at each dimension to make
/// the norm-reuse benefit measurable.
fn bench_batch_cosine_one_vs_many(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_cosine");
    const N_CANDIDATES: usize = 16;

    for &dim in &DIMS {
        let query = gen_vec(dim, 0xaaaa);
        let candidates: Vec<Vec<f32>> = (0..N_CANDIDATES)
            .map(|i| gen_vec(dim, 0xbbbb + i as u64))
            .collect();

        // Prepare per-pair format for batch_cosine_similarity
        let pairs: Vec<(&[f32], &[f32])> = candidates
            .iter()
            .map(|c| (query.as_slice(), c.as_slice()))
            .collect();

        // Prepare candidate-refs format for batch_cosine_one_vs_many
        let candidate_refs: Vec<&[f32]> = candidates.iter().map(Vec::as_slice).collect();

        group.bench_with_input(BenchmarkId::new("per_pair_16", dim), &dim, |bencher, _| {
            bencher.iter(|| {
                std::hint::black_box(batch_cosine_similarity(std::hint::black_box(&pairs)))
            });
        });

        group.bench_with_input(
            BenchmarkId::new("one_vs_many_16", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| {
                    std::hint::black_box(batch_cosine_one_vs_many(
                        std::hint::black_box(&query),
                        std::hint::black_box(&candidate_refs),
                    ))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark the i8 raw dot product kernel at each dimension.
///
/// Compares the SIMD path (with software prefetch) against the pure scalar reference,
/// demonstrating the speedup ratio.
fn bench_i8_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("i8_dot_product");

    for &dim in &DIMS {
        let a_f32 = gen_vec(dim, 0xcccc);
        let b_f32 = gen_vec(dim, 0xdddd);
        let a_q = QuantizedVector::from_f32(&a_f32);
        let b_q = QuantizedVector::from_f32(&b_f32);

        // SIMD path (dispatch: NEON/AVX2/AVX-512 with prefetch)
        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |bencher, _| {
            bencher.iter(|| {
                std::hint::black_box(dot_product_i8_raw(
                    std::hint::black_box(a_q.data()),
                    std::hint::black_box(b_q.data()),
                ))
            });
        });

        // Scalar reference path
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bencher, _| {
            bencher.iter(|| {
                let sum: i32 = std::hint::black_box(a_q.data())
                    .iter()
                    .zip(std::hint::black_box(b_q.data()).iter())
                    .map(|(&x, &y)| x as i32 * y as i32)
                    .sum();
                std::hint::black_box(sum as f32)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_fused_vs_original,
    bench_batch_cosine_one_vs_many,
    bench_i8_dot_product,
);
criterion_main!(benches);
