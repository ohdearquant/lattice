//! Criterion benchmarks for block influence scoring (ADR-060 Phase 1).
//!
//! Two benchmark groups:
//!
//! * `cosine_similarity` — per-observation throughput (one token pair at a time).
//! * `block_influence_scoring` — full single-token sweep over all layers.
//! * `accumulator_calibration` — realistic calibration workload: N tokens per
//!   layer via [`BlockInfluenceAccumulator`], simulating offline calibration.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::pruning::{
    BlockInfluenceAccumulator, cosine_similarity, score_from_hidden_states,
};

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

fn random_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..len).map(|_| rng.next_f32()).collect()
}

/// Per-observation throughput: one cosine_similarity call per iteration.
fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50);

    for hidden_dim in [896usize, 2048, 4096] {
        let a = random_vec(hidden_dim, 42);
        let b = random_vec(hidden_dim, 99);
        group.throughput(Throughput::Elements(hidden_dim as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(hidden_dim),
            &hidden_dim,
            |bench, _| {
                bench.iter(|| {
                    black_box(cosine_similarity(
                        black_box(a.as_slice()),
                        black_box(b.as_slice()),
                    ))
                });
            },
        );
    }
    group.finish();
}

/// Per-layer sweep: score_from_hidden_states over a full model.
fn bench_score_from_hidden_states(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_influence_scoring");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50);

    for (n_layers, hidden_dim) in [(24usize, 896usize), (36, 2048)] {
        let states: Vec<Vec<f32>> = (0..=n_layers)
            .map(|i| random_vec(hidden_dim, 42 + i as u64))
            .collect();
        let label = format!("{n_layers}L_d{hidden_dim}");
        group.throughput(Throughput::Elements(n_layers as u64));
        group.bench_with_input(BenchmarkId::new("layers", &label), &states, |bench, s| {
            bench.iter(|| black_box(score_from_hidden_states(black_box(s))));
        });
    }
    group.finish();
}

/// Synthetic scorer workload: accumulate N tokens per layer via BlockInfluenceAccumulator.
///
/// This measures scorer throughput with generated random vectors — not actual
/// calibration cost, which would require a real forward pass and activation capture.
fn bench_accumulator_calibration(c: &mut Criterion) {
    let mut group = c.benchmark_group("accumulator_calibration");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(4));
    group.sample_size(30);

    // Simulate calibration for a 24-layer Qwen3-0.6B (hidden_dim=896).
    let n_layers = 24usize;
    let hidden_dim = 896usize;

    for n_tokens in [128usize, 512] {
        // Pre-generate token vectors for all layers.
        // input[layer][token] and output[layer][token].
        let inputs: Vec<Vec<Vec<f32>>> = (0..n_layers)
            .map(|layer| {
                (0..n_tokens)
                    .map(|tok| random_vec(hidden_dim, (layer * n_tokens + tok) as u64))
                    .collect()
            })
            .collect();
        let outputs: Vec<Vec<Vec<f32>>> = (0..n_layers)
            .map(|layer| {
                (0..n_tokens)
                    .map(|tok| random_vec(hidden_dim, 1_000_000 + (layer * n_tokens + tok) as u64))
                    .collect()
            })
            .collect();

        let label = format!("{n_layers}L_d{hidden_dim}_{n_tokens}tok");
        group.throughput(Throughput::Elements((n_layers * n_tokens) as u64));
        group.bench_with_input(
            BenchmarkId::new("tokens_per_layer", &label),
            &n_tokens,
            |bench, _| {
                bench.iter(|| {
                    let scores: Vec<_> = (0..n_layers)
                        .map(|layer| {
                            let mut acc = BlockInfluenceAccumulator::new(black_box(layer));
                            for tok in 0..n_tokens {
                                acc.update(
                                    black_box(&inputs[layer][tok]),
                                    black_box(&outputs[layer][tok]),
                                );
                            }
                            acc.finalize()
                        })
                        .collect();
                    black_box(scores)
                });
            },
        );
    }
    group.finish();
}

fn bench_all(c: &mut Criterion) {
    bench_cosine_similarity(c);
    bench_score_from_hidden_states(c);
    bench_accumulator_calibration(c);
}

criterion_group!(pruning, bench_all);
criterion_main!(pruning);
