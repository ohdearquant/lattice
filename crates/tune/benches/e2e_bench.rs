//! End-to-end LoRA lifecycle benchmark: train → apply at varied ranks.
//!
//! Measures the full `train_lora` + `apply_lora` cycle for a realistic
//! adapter size (d_in=64, d_out=64, 50 samples, 10 epochs) across
//! ranks [2, 4, 8].
//!
//! Group: `lora/e2e_rank`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use lattice_tune::lora::{LoraAdapter, LoraConfig, LoraLayer, apply_lora};
use lattice_tune::{LoraTrainConfig, TrainSample, train_lora};
use std::collections::HashMap;

const D_IN: usize = 64;
const D_OUT: usize = 64;
const N_SAMPLES: usize = 50;
const NUM_EPOCHS: usize = 10;

/// Deterministic pseudo-random f32 in (-0.5, 0.5) — no external rand crate.
fn pseudo_f32(seed: u64, idx: u64) -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    (seed, idx).hash(&mut h);
    h.finish() as f32 / u64::MAX as f32 - 0.5
}

fn make_adapter(rank: usize) -> LoraAdapter {
    let config = LoraConfig {
        rank,
        alpha: rank as f32, // scale = 1.0
        target_modules: vec!["q_proj".into()],
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0usize, "q_proj".to_string()),
        LoraLayer {
            a: (0..rank * D_IN)
                .map(|i| pseudo_f32(1, i as u64) * 0.02)
                .collect(),
            b: (0..D_OUT * rank)
                .map(|i| pseudo_f32(2, i as u64) * 0.02)
                .collect(),
            d_in: D_IN,
            d_out: D_OUT,
            rank,
        },
    );
    LoraAdapter::new(config, layers)
}

fn make_samples() -> Vec<TrainSample> {
    (0..N_SAMPLES)
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

/// Fixed input vector for the apply phase.
fn make_input() -> Vec<f32> {
    (0..D_IN).map(|i| pseudo_f32(9999, i as u64)).collect()
}

fn bench_e2e_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("lora/e2e_rank");

    let samples = make_samples();
    let input = make_input();

    for &rank in &[2usize, 4, 8] {
        group.bench_with_input(BenchmarkId::new("rank", rank), &rank, |b, &rank| {
            b.iter(|| {
                // Train phase — re-create adapter each iteration for fresh weights.
                let mut adapter = make_adapter(rank);
                let train_cfg = LoraTrainConfig {
                    learning_rate: 0.01,
                    num_epochs: NUM_EPOCHS,
                    batch_size: N_SAMPLES,
                };
                train_lora(
                    black_box(&mut adapter),
                    black_box(&samples),
                    black_box(&train_cfg),
                )
                .expect("train_lora must not fail in bench");

                // Apply phase — one forward pass with the trained adapter.
                let layer = adapter
                    .layers
                    .get(&(0usize, "q_proj".to_string()))
                    .expect("layer must exist");
                let scale = adapter.config.scale();
                let mut out = vec![0.0f32; D_OUT];
                apply_lora(
                    black_box(layer),
                    black_box(scale),
                    black_box(&input),
                    black_box(&mut out),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_e2e_rank);
criterion_main!(benches);
