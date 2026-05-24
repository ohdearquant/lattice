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
use lattice_inference::forward::cpu::{elementwise_mul, rms_norm, silu_inplace};

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

criterion_group!(benches, bench_rms_norm, bench_silu, bench_elementwise_mul);
criterion_main!(benches);
