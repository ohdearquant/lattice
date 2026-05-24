//! Performance regression tests for lattice-tune.
//!
//! These are pass/fail time-bounded assertions, not benchmarks.
//! Bounds are deliberately generous (50-100x over expected) to remain
//! stable across debug builds and CI variance.

use std::collections::HashMap;
use std::time::Instant;

use lattice_tune::{LoraAdapter, LoraConfig, LoraLayer, LoraTrainConfig, TrainSample, train_lora};

fn make_adapter(rank: usize, d_in: usize, d_out: usize) -> LoraAdapter {
    let config = LoraConfig {
        rank,
        alpha: rank as f32,
        target_modules: vec!["q_proj".to_string()],
    };
    let mut layers = HashMap::new();
    // A: (rank, d_in), B: (d_out, rank) — initialize with small values
    let a: Vec<f32> = (0..rank * d_in).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..d_out * rank).map(|i| (i as f32) * 0.01).collect();
    layers.insert(
        (0, "q_proj".to_string()),
        LoraLayer {
            a,
            b,
            d_in,
            d_out,
            rank,
        },
    );
    LoraAdapter::new(config, layers)
}

fn make_samples(n: usize, d_in: usize, d_out: usize) -> Vec<TrainSample> {
    (0..n)
        .map(|i| {
            let fi = i as f32;
            TrainSample {
                layer_idx: 0,
                module: "q_proj".to_string(),
                input: (0..d_in).map(|j| fi * 0.01 + j as f32 * 0.001).collect(),
                target_delta: (0..d_out).map(|j| fi * 0.005 + j as f32 * 0.002).collect(),
            }
        })
        .collect()
}

#[test]
fn train_lora_not_catastrophically_slow() {
    // rank=2, d_in=16, d_out=8, 20 samples, 5 epochs must finish in < 2s.
    // Expected: the inner loop is simple matmul + SGD; 100 adapt_step calls
    // at tiny dimensions should complete in <<1ms total.
    // 2s gives >1000x slack over expected debug runtime.
    let mut adapter = make_adapter(2, 16, 8);
    let samples = make_samples(20, 16, 8);
    let config = LoraTrainConfig {
        learning_rate: 0.01,
        num_epochs: 5,
        batch_size: 4,
    };

    let start = Instant::now();
    let result = train_lora(&mut adapter, &samples, &config).expect("train_lora failed");
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_secs() < 2,
        "train_lora regression: 20 samples × 5 epochs took {elapsed:?}, expected < 2s"
    );
    // Sanity: result shape is consistent
    assert_eq!(result.epoch_losses.len(), 5);
    assert_eq!(result.total_steps, 100);
}

#[test]
fn train_lora_repeated_not_catastrophically_slow() {
    // Run train_lora 50 times (rank=2, d_in=8, d_out=4, 10 samples, 2 epochs).
    // 50 runs × ~2ms expected each = ~100ms total; 5s gives 50x slack in debug.
    let config = LoraTrainConfig {
        learning_rate: 0.01,
        num_epochs: 2,
        batch_size: 4,
    };
    let samples = make_samples(10, 8, 4);

    let start = Instant::now();
    for _ in 0..50 {
        let mut adapter = make_adapter(2, 8, 4);
        train_lora(&mut adapter, &samples, &config).expect("train_lora failed");
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_secs() < 5,
        "train_lora repeated regression: 50 runs took {elapsed:?}, expected < 5s"
    );
}
