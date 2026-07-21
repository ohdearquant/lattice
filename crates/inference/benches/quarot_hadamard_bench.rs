//! `block_hadamard` Criterion bench group.
//!
//! Measures [`BlockHadamard`] at the real non-power-of-two axis dimensions
//! QuaRot rotation actually targets — Qwen3.5-0.8B's `intermediate_size`
//! (3584 = 2^9 * 7) and Qwen3.6-27B's `intermediate_size` (17408 = 2^11 * 17)
//! — with construction (`BlockHadamard::new`: per-block seed derivation
//! plus one contiguous signs-buffer reservation) and apply
//! (`BlockHadamard::apply`, the per-block butterfly transform) measured as
//! separate benchmark IDs so a regression in either phase is individually
//! attributable.
//!
//! ```bash
//! cargo bench -p lattice-inference --bench quarot_hadamard_bench -- "block_hadamard"
//! ```
//!
//! Always compiled (no feature gate) — `BlockHadamard` is pure CPU code
//! with no GPU/checkpoint dependency, unlike the GPU-backed bench groups
//! elsewhere in this crate.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use lattice_inference::quant::quarot::hadamard::BlockHadamard;

/// (dimension name, `n`, `block_size`) — both dims are genuinely
/// non-power-of-two (the case `BlockHadamard` exists to serve; a
/// power-of-two axis would use `RandomizedHadamard` directly), and
/// `block_size = 256` divides both exactly (3584 / 256 = 14,
/// 17408 / 256 = 68).
const CASES: &[(&str, usize, usize)] = &[
    ("qwen35_0_8b_intermediate_3584", 3584, 256),
    ("qwen36_27b_intermediate_17408", 17408, 256),
];

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_hadamard_construction");
    for &(label, n, block_size) in CASES {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(n, block_size),
            |b, &(n, block_size)| {
                b.iter(|| {
                    let bh =
                        BlockHadamard::new(black_box(0x5EED), black_box(n), black_box(block_size))
                            .unwrap();
                    black_box(bh)
                });
            },
        );
    }
    group.finish();
}

fn bench_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_hadamard_apply");
    for &(label, n, block_size) in CASES {
        let bh = BlockHadamard::new(0x5EED, n, block_size).unwrap();
        let data = vec![1.0_f32; n];
        group.bench_with_input(BenchmarkId::from_parameter(label), &data, |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut buf| {
                    bh.apply(black_box(&mut buf)).unwrap();
                    black_box(buf)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(block_hadamard, bench_construction, bench_apply);
criterion_main!(block_hadamard);
