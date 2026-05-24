//! Benchmark for the batch LoRA training loop.
//!
//! Measures [`train_lora`] throughput as a function of sample count (batch_size
//! parameter is informational; the loop is sequential).
//!
//! Groups:
//!   lora/train_loop — num_epochs=5, batch_size ∈ {10, 50, 100}

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use lattice_tune::lora::{LoraAdapter, LoraConfig, LoraLayer};
use lattice_tune::{LoraTrainConfig, TrainSample, train_lora};
use std::collections::HashMap;

const D_IN: usize = 64;
const D_OUT: usize = 64;
const RANK: usize = 4;
const NUM_EPOCHS: usize = 5;

/// Build a deterministic pseudo-random f32 from two indices (no external rand crate).
fn pseudo_f32(seed: u64, idx: u64) -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    (seed, idx).hash(&mut h);
    // Map to (-0.5, 0.5) so the initial weights are centred.
    h.finish() as f32 / u64::MAX as f32 - 0.5
}

fn make_adapter() -> LoraAdapter {
    let config = LoraConfig {
        rank: RANK,
        alpha: RANK as f32, // scale = 1.0
        target_modules: vec!["q_proj".into()],
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0usize, "q_proj".to_string()),
        LoraLayer {
            // A: (rank, d_in)
            a: (0..(RANK * D_IN))
                .map(|i| pseudo_f32(1, i as u64) * 0.01)
                .collect(),
            // B: (d_out, rank)
            b: (0..(D_OUT * RANK))
                .map(|i| pseudo_f32(2, i as u64) * 0.01)
                .collect(),
            d_in: D_IN,
            d_out: D_OUT,
            rank: RANK,
        },
    );
    LoraAdapter::new(config, layers)
}

fn make_samples(n: usize) -> Vec<TrainSample> {
    (0..n)
        .map(|i| TrainSample {
            layer_idx: 0,
            module: "q_proj".to_string(),
            input: (0..D_IN)
                .map(|j| pseudo_f32(3 + i as u64, j as u64))
                .collect(),
            target_delta: (0..D_OUT)
                .map(|j| pseudo_f32(1000 + i as u64, j as u64) * 0.1)
                .collect(),
        })
        .collect()
}

fn bench_train_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("lora/train_loop");

    for &n_samples in &[10usize, 50, 100] {
        let samples = make_samples(n_samples);

        group.bench_with_input(
            BenchmarkId::new("num_samples", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    // Re-create adapter each iteration so weights start fresh.
                    let mut adapter = make_adapter();
                    let config = LoraTrainConfig {
                        learning_rate: 0.01,
                        num_epochs: NUM_EPOCHS,
                        batch_size: n_samples,
                    };
                    train_lora(
                        black_box(&mut adapter),
                        black_box(&samples),
                        black_box(&config),
                    )
                    .expect("train_lora must not fail in bench")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_train_loop);
criterion_main!(benches);
