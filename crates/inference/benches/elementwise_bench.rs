//! Criterion benchmarks for SIMD elementwise operations.
//!
//! Measures `rms_norm`, `silu_inplace`, and `elementwise_mul` throughput at
//! hidden sizes representative of deployed models:
//!   896  — Qwen3.5-0.6B / Qwen2-0.5B
//!   2048 — Qwen3.5-1.5B
//!   4096 — Qwen3.5-7B / LLaMA-3-8B
//!
//! Run with:
//!   cargo bench -p lattice-inference --bench elementwise_bench

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lattice_inference::attention::decode::decode_attention;
use lattice_inference::attention::gqa::GqaConfig;
use lattice_inference::forward::cpu::{
    add_bias_gelu, elementwise_mul, gelu, layer_norm, rms_norm, silu_inplace, softmax_attention,
};

// ---------------------------------------------------------------------------
// Deterministic test-data generator (LCG, avoids pulling in random crates)
// ---------------------------------------------------------------------------

fn lcg_f32_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0x6c62_272e_07bb_0142;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (state >> 32) as f32 / u32::MAX as f32;
        out.push(unit * 0.04 - 0.02);
    }
    out
}

// ---------------------------------------------------------------------------
// rms_norm benchmarks
// ---------------------------------------------------------------------------

fn bench_rms_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rms_norm");
    let eps = 1e-6f32;
    let num_tokens = 8usize;

    for &hidden in &[896usize, 2048, 4096] {
        let gamma = lcg_f32_vec(hidden, 0xA000_0000_0000_0001);
        let bytes = num_tokens * hidden * std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", hidden), &hidden, |b, &h| {
            let mut x = lcg_f32_vec(num_tokens * h, 0xB000_0000_0000_0001);
            b.iter(|| {
                rms_norm(
                    std::hint::black_box(&mut x),
                    std::hint::black_box(&gamma),
                    std::hint::black_box(h),
                    std::hint::black_box(eps),
                );
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// silu_inplace benchmarks
// ---------------------------------------------------------------------------

fn bench_silu(c: &mut Criterion) {
    let mut group = c.benchmark_group("silu_inplace");

    for &hidden in &[896usize, 2048, 4096] {
        let bytes = hidden * std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", hidden), &hidden, |b, &h| {
            let mut x = lcg_f32_vec(h, 0xC000_0000_0000_0001);
            b.iter(|| {
                silu_inplace(std::hint::black_box(&mut x));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// elementwise_mul benchmarks
// ---------------------------------------------------------------------------

fn bench_elementwise_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_mul");

    for &hidden in &[896usize, 2048, 4096] {
        // Two slices read + one write per element.
        let bytes = hidden * 3 * std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", hidden), &hidden, |b, &h| {
            let mut a = lcg_f32_vec(h, 0xD000_0000_0000_0001);
            let b_vec = lcg_f32_vec(h, 0xE000_0000_0000_0001);
            b.iter(|| {
                elementwise_mul(std::hint::black_box(&mut a), std::hint::black_box(&b_vec));
            });
        });
    }
    group.finish();
}

fn bench_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu");

    for &hidden in &[896usize, 2048, 4096] {
        let bytes = hidden * std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", hidden), &hidden, |b, &h| {
            let mut x = lcg_f32_vec(h, 0xF000_0000_0000_0001);
            b.iter(|| {
                gelu(std::hint::black_box(&mut x));
            });
        });
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_attention");
    let num_heads = 8usize;

    for &seq_len in &[32usize, 64, 128] {
        let size = num_heads * seq_len * seq_len;
        let bytes = size * std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", seq_len), &seq_len, |b, &s| {
            let mut x = lcg_f32_vec(num_heads * s * s, 0xA100_0000_0000_0001);
            b.iter(|| {
                softmax_attention(
                    std::hint::black_box(&mut x),
                    std::hint::black_box(s),
                    std::hint::black_box(num_heads),
                );
            });
        });
    }
    group.finish();
}

fn bench_add_bias_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_bias_gelu");

    for &hidden in &[896usize, 2048, 4096] {
        let bytes = hidden * std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", hidden), &hidden, |b, &h| {
            let mut x = lcg_f32_vec(h, 0xF100_0000_0000_0001);
            let bias = lcg_f32_vec(h, 0xF200_0000_0000_0001);
            b.iter(|| {
                add_bias_gelu(
                    std::hint::black_box(&mut x),
                    std::hint::black_box(&bias),
                    std::hint::black_box(h),
                );
            });
        });
    }
    group.finish();
}

fn bench_layer_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_norm");
    let eps = 1e-6f32;
    let num_tokens = 8usize;

    for &hidden in &[896usize, 2048, 4096] {
        let gamma = lcg_f32_vec(hidden, 0xA200_0000_0000_0001);
        let beta = lcg_f32_vec(hidden, 0xA300_0000_0000_0001);
        let bytes = num_tokens * hidden * std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", hidden), &hidden, |b, &h| {
            let mut x = lcg_f32_vec(num_tokens * h, 0xA400_0000_0000_0001);
            b.iter(|| {
                layer_norm(
                    std::hint::black_box(&mut x),
                    std::hint::black_box(&gamma),
                    std::hint::black_box(&beta),
                    std::hint::black_box(h),
                    std::hint::black_box(eps),
                );
            });
        });
    }
    group.finish();
}

fn bench_decode_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_attention");
    let cfg = GqaConfig {
        num_heads: 16,
        num_kv_heads: 2,
        head_dim: 128,
    };

    for &kv_len in &[128usize, 512, 2048] {
        let q = lcg_f32_vec(cfg.q_dim(), 0xDA00_0000_0000_0001);
        let k = lcg_f32_vec(kv_len * cfg.kv_dim(), 0xDB00_0000_0000_0001);
        let v = lcg_f32_vec(kv_len * cfg.kv_dim(), 0xDC00_0000_0000_0001);

        let bytes = (cfg.q_dim() + 2 * kv_len * cfg.kv_dim()) * std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("gqa_16h2kv_hd128", kv_len),
            &kv_len,
            |b, &kl| {
                let mut out = vec![0.0f32; cfg.q_dim()];
                let mut scores = vec![0.0f32; cfg.num_heads * kl];
                b.iter(|| {
                    decode_attention(
                        std::hint::black_box(&q),
                        std::hint::black_box(&k),
                        std::hint::black_box(&v),
                        std::hint::black_box(&mut out),
                        std::hint::black_box(&mut scores),
                        std::hint::black_box(kl),
                        std::hint::black_box(cfg),
                        std::hint::black_box(kl),
                    );
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rms_norm,
    bench_silu,
    bench_elementwise_mul,
    bench_gelu,
    bench_softmax,
    bench_add_bias_gelu,
    bench_layer_norm,
    bench_decode_attention
);
criterion_main!(benches);
