//! Criterion benchmarks for inference metrics infrastructure (ADR-061 Phase 1).
//!
//! Measures:
//!   - [`OnlineSoftmaxEntropy`] throughput across seq-len sizes used in practice
//!   - [`l2_norm`] throughput for hidden sizes used by Qwen3 (896) and LLaMA (4096)
//!   - Naive entropy baseline (materialize softmax, then -Σ p log p) for overhead ratio
//!
//! The overhead ratio online_entropy_ns / naive_entropy_ns is reported via
//! [`print_overhead_ratio`] after each paired bench group.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::metrics::{OnlineSoftmaxEntropy, l2_norm};

// ---------------------------------------------------------------------------
// Deterministic LCG — same as elementwise_cpu_bench for consistency.
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 32) as u32
    }

    /// Returns a value in [-10, 10] to stress both branches of the online algorithm.
    fn next_logit(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32) * 20.0 - 10.0
    }

    fn next_f32_unit(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

fn logit_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..len).map(|_| rng.next_logit()).collect()
}

fn unit_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..len).map(|_| rng.next_f32_unit() * 2.0 - 1.0).collect()
}

// ---------------------------------------------------------------------------
// Naive entropy reference (materialize softmax, then -Σ p ln p).
// ---------------------------------------------------------------------------

fn naive_entropy_nats(logits: &[f32]) -> f32 {
    if logits.len() < 2 {
        return 0.0;
    }
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0_f32;
    let mut exps: Vec<f32> = logits
        .iter()
        .map(|&l| {
            let e = (l - max_l).exp();
            sum_exp += e;
            e
        })
        .collect();
    // Normalize in-place.
    let inv_sum = 1.0 / sum_exp;
    exps.iter_mut().for_each(|e| *e *= inv_sum);
    // -Σ p ln p.
    exps.iter()
        .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
        .sum()
}

// ---------------------------------------------------------------------------
// Benchmarks: online entropy
// ---------------------------------------------------------------------------

fn bench_online_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_entropy");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50);

    for seq_len in [128usize, 512, 2048, 8192] {
        let logits = logit_vec(seq_len, 42);
        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, _| {
            let mut acc = OnlineSoftmaxEntropy::new();
            b.iter(|| {
                acc.reset();
                for &l in black_box(logits.as_slice()) {
                    acc.update(l);
                }
                black_box(acc.entropy_nats())
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: naive entropy (reference baseline for overhead ratio)
// ---------------------------------------------------------------------------

fn bench_naive_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("naive_entropy");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50);

    for seq_len in [128usize, 512, 2048, 8192] {
        let logits = logit_vec(seq_len, 42);
        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, _| {
            b.iter(|| black_box(naive_entropy_nats(black_box(logits.as_slice()))));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: l2_norm
// ---------------------------------------------------------------------------

fn bench_l2_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_norm");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50);

    for hidden in [896usize, 4096] {
        let vec = unit_vec(hidden, 99);
        group.throughput(Throughput::Elements(hidden as u64));
        group.bench_with_input(BenchmarkId::from_parameter(hidden), &hidden, |b, _| {
            b.iter(|| black_box(l2_norm(black_box(vec.as_slice()))));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Overhead ratio: print after benchmarking (informational, not Criterion).
// ---------------------------------------------------------------------------
//
// Criterion does not expose raw ns values at bench-time, so we compute the
// ratio ourselves via `std::time::Instant` with enough iterations for stability.

fn print_overhead_ratio() {
    use std::hint::black_box as bb;
    use std::time::Instant;

    let iters = 2_000_usize;
    println!("\n=== online_entropy / naive_entropy overhead ratio ===");

    for seq_len in [128usize, 512, 2048, 8192] {
        let logits = logit_vec(seq_len, 42);

        // Online — prevent the loop from being elided.
        let mut acc = OnlineSoftmaxEntropy::new();
        let t0 = Instant::now();
        for _ in 0..iters {
            acc.reset();
            for &l in bb(logits.as_slice()) {
                acc.update(bb(l));
            }
            bb(acc.entropy_nats());
        }
        let online_ns = t0.elapsed().as_nanos() / iters as u128;

        // Naive.
        let t1 = Instant::now();
        for _ in 0..iters {
            bb(naive_entropy_nats(bb(logits.as_slice())));
        }
        let naive_ns = t1.elapsed().as_nanos() / iters as u128;

        let ratio = online_ns as f64 / naive_ns.max(1) as f64;
        println!(
            "  seq_len={seq_len:5}: online={online_ns:6} ns, naive={naive_ns:6} ns, ratio={ratio:.3}"
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Criterion entry points
// ---------------------------------------------------------------------------

fn bench_all(c: &mut Criterion) {
    bench_online_entropy(c);
    bench_naive_entropy(c);
    bench_l2_norm(c);
    // Print overhead ratio once at the end of the benchmark run.
    print_overhead_ratio();
}

criterion_group!(metrics, bench_all);
criterion_main!(metrics);
