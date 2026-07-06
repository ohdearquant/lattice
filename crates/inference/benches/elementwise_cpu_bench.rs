//! Criterion benches for CPU elementwise / norm / activation / softmax kernels.
//!
//! Gated by ADR-058 — used as the reference baseline for the perf-regression CI
//! workflow. The bench list is intentionally narrow: only the ops that appear
//! in the per-token decode CPU path on hidden sizes the Qwen3.5-0.8B uses.
//!
//! Deterministic LCG inputs make run-to-run noise come from the runner, not
//! from the data.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::forward::cpu::{
    add_bias_gelu, elementwise_mul, gelu, layer_norm, residual_add_layer_norm, rms_norm,
    silu_inplace, softmax_attention,
};
use std::time::Duration;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32) - 0.5
    }
}

fn random_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..len).map(|_| rng.next_f32()).collect()
}

fn bench_rms_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rms_norm");
    for hidden in [896usize, 4096] {
        let gamma = random_vec(hidden, 1);
        let buf_proto = random_vec(hidden, 2);
        group.throughput(Throughput::Elements(hidden as u64));
        group.bench_with_input(BenchmarkId::from_parameter(hidden), &hidden, |b, &h| {
            let mut buf = buf_proto.clone();
            b.iter(|| {
                rms_norm(black_box(&mut buf), black_box(&gamma), h, 1e-6);
            });
        });
    }
    group.finish();
}

fn bench_layer_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_norm");
    for hidden in [896usize, 4096] {
        let gamma = random_vec(hidden, 3);
        let beta = random_vec(hidden, 4);
        let buf_proto = random_vec(hidden, 5);
        group.throughput(Throughput::Elements(hidden as u64));
        group.bench_with_input(BenchmarkId::from_parameter(hidden), &hidden, |b, &h| {
            let mut buf = buf_proto.clone();
            b.iter(|| {
                layer_norm(
                    black_box(&mut buf),
                    black_box(&gamma),
                    black_box(&beta),
                    h,
                    1e-6,
                );
            });
        });
    }
    group.finish();
}

fn bench_silu_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("silu_inplace");
    for hidden in [896usize, 4096] {
        let buf_proto = random_vec(hidden, 6);
        group.throughput(Throughput::Elements(hidden as u64));
        group.bench_with_input(BenchmarkId::from_parameter(hidden), &hidden, |b, _| {
            let mut buf = buf_proto.clone();
            b.iter(|| {
                silu_inplace(black_box(&mut buf));
            });
        });
    }
    group.finish();
}

fn bench_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu");
    for hidden in [896usize, 4096] {
        let buf_proto = random_vec(hidden, 7);
        group.throughput(Throughput::Elements(hidden as u64));
        group.bench_with_input(BenchmarkId::from_parameter(hidden), &hidden, |b, _| {
            let mut buf = buf_proto.clone();
            b.iter(|| {
                gelu(black_box(&mut buf));
            });
        });
    }
    group.finish();
}

fn bench_add_bias_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_bias_gelu");
    for hidden in [896usize, 4096] {
        let bias = random_vec(hidden, 8);
        let buf_proto = random_vec(hidden, 9);
        group.throughput(Throughput::Elements(hidden as u64));
        group.bench_with_input(BenchmarkId::from_parameter(hidden), &hidden, |b, &h| {
            let mut buf = buf_proto.clone();
            b.iter(|| {
                add_bias_gelu(black_box(&mut buf), black_box(&bias), h);
            });
        });
    }
    group.finish();
}

fn bench_softmax_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_attention");
    // Buffer shape is (num_heads, seq_len, seq_len) — full attention matrix.
    // Single-head keeps the bench focused on the kernel itself, not multi-head dispatch.
    for seq_len in [128usize, 512] {
        let num_heads = 1usize;
        let total = num_heads * seq_len * seq_len;
        let buf_proto = random_vec(total, 10);
        group.throughput(Throughput::Elements((seq_len * seq_len) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, &s| {
            let mut buf = buf_proto.clone();
            b.iter(|| {
                softmax_attention(black_box(&mut buf), s, num_heads);
            });
        });
    }
    group.finish();
}

fn bench_residual_add_layer_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("residual_add_layer_norm");
    // BERT sublayer epilogue shapes this fusion actually runs at inside
    // `BertModel::forward`/`forward_with_hook`: bge-small (hidden=384) and
    // bert-base (hidden=768), both at a 128-token sequence length.
    for (hidden, seq_len) in [(384usize, 128usize), (768, 128)] {
        let total = hidden * seq_len;
        let gamma = random_vec(hidden, 20);
        let beta = random_vec(hidden, 21);
        let out_proto = random_vec(total, 22);
        let dense = random_vec(total, 23);
        let label = format!("hidden{hidden}_seq{seq_len}");

        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(BenchmarkId::new("fused", &label), &hidden, |b, &h| {
            let mut out = out_proto.clone();
            b.iter(|| {
                residual_add_layer_norm(
                    black_box(&mut out),
                    black_box(&dense),
                    black_box(&gamma),
                    black_box(&beta),
                    h,
                    1e-6,
                );
            });
        });
        // Unfused reference: the current pre-#676 epilogue shape (add loop
        // then layer_norm), minus the trailing buffer copy -- so this isolates
        // the kernel-level delta; the copy elimination is separate upside.
        group.bench_with_input(BenchmarkId::new("unfused", &label), &hidden, |b, &h| {
            let mut out = out_proto.clone();
            b.iter(|| {
                for i in 0..total {
                    out[i] += dense[i];
                }
                layer_norm(
                    black_box(&mut out),
                    black_box(&gamma),
                    black_box(&beta),
                    h,
                    1e-6,
                );
            });
        });
    }
    group.finish();
}

fn bench_elementwise_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_mul");
    let hidden = 4096usize;
    let b_vec = random_vec(hidden, 11);
    let buf_proto = random_vec(hidden, 12);
    group.throughput(Throughput::Elements(hidden as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(hidden),
        &hidden,
        |bencher, _| {
            let mut buf = buf_proto.clone();
            bencher.iter(|| {
                elementwise_mul(black_box(&mut buf), black_box(&b_vec));
            });
        },
    );
    group.finish();
}

fn criterion_cfg() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3))
        .sample_size(50)
}

criterion_group! {
    name = elementwise_cpu;
    config = criterion_cfg();
    targets =
        bench_rms_norm,
        bench_layer_norm,
        bench_silu_inplace,
        bench_gelu,
        bench_add_bias_gelu,
        bench_softmax_attention,
        bench_elementwise_mul,
        bench_residual_add_layer_norm,
}
criterion_main!(elementwise_cpu);
