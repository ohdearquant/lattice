//! Batch LoRA training loop (ADR-057, perf-opt-train-loop).
//!
//! Provides [`train_lora`] which iterates over a slice of [`TrainSample`]s for
//! multiple epochs, calling [`adapt_step`] for each sample and accumulating per-epoch
//! MSE loss statistics.

use super::{LoraAdapter, online::adapt_step};
use crate::error::TuneError;

/// Configuration for a batch LoRA training run.
#[derive(Debug, Clone)]
pub struct LoraTrainConfig {
    /// SGD step size applied inside each [`adapt_step`] call.
    pub learning_rate: f32,
    /// Number of full passes over the sample set.
    pub num_epochs: usize,
    /// Batch size (informational; does not affect the current sequential loop).
    pub batch_size: usize,
}

/// A single training sample: one (input, target_delta) pair for one LoRA layer.
#[derive(Debug, Clone)]
pub struct TrainSample {
    /// Transformer layer index (0-based).
    pub layer_idx: usize,
    /// Module name, e.g. `"q_proj"`.
    pub module: String,
    /// Input activation vector; length must equal the layer's `d_in`.
    pub input: Vec<f32>,
    /// Target LoRA output correction; length must equal the layer's `d_out`.
    pub target_delta: Vec<f32>,
}

/// Summary statistics from a completed [`train_lora`] run.
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// Average MSE loss recorded in the final epoch.
    pub final_loss: f32,
    /// Average MSE loss per epoch (length == `num_epochs`).
    pub epoch_losses: Vec<f32>,
    /// Total number of [`adapt_step`] calls executed (`num_epochs * samples.len()`).
    pub total_steps: usize,
}

/// Run a multi-epoch batch LoRA training loop.
///
/// Iterates over all `samples` in order for each epoch (no shuffling, ensuring
/// deterministic behaviour), calling [`adapt_step`] once per sample and
/// accumulating the per-step MSE loss.
///
/// # Arguments
///
/// * `adapter` – Mutable LoRA adapter whose weights are updated in place.
/// * `samples` – Slice of training samples; must be non-empty.
/// * `config`  – Hyper-parameters for the training run.
///
/// # Errors
///
/// * [`TuneError::Training`]         – `samples` is empty.
/// * [`TuneError::InvalidConfig`]    – `batch_size == 0` or `num_epochs == 0`.
/// * [`TuneError::Training`] – A sample references a `(layer_idx, module)` pair
///   that is not present in the adapter.
/// * [`TuneError::DimensionMismatch`] – A sample's `input` or `target_delta` has the
///   wrong length for its LoRA layer.
pub fn train_lora(
    adapter: &mut LoraAdapter,
    samples: &[TrainSample],
    config: &LoraTrainConfig,
) -> Result<TrainResult, TuneError> {
    if samples.is_empty() {
        return Err(TuneError::Training(
            "no training samples provided".to_string(),
        ));
    }
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(TuneError::InvalidConfig(
            "learning_rate must be finite and positive".to_string(),
        ));
    }
    if config.batch_size == 0 {
        return Err(TuneError::InvalidConfig(
            "batch_size must be > 0".to_string(),
        ));
    }
    if config.num_epochs == 0 {
        return Err(TuneError::InvalidConfig(
            "num_epochs must be > 0".to_string(),
        ));
    }

    let mut epoch_losses = Vec::with_capacity(config.num_epochs);

    for _ in 0..config.num_epochs {
        let mut epoch_loss_sum = 0.0f32;

        for sample in samples {
            let step_result = adapt_step(
                adapter,
                sample.layer_idx,
                &sample.module,
                &sample.input,
                &sample.target_delta,
                config.learning_rate,
            )?;
            epoch_loss_sum += step_result.loss;
        }

        let avg = epoch_loss_sum / samples.len() as f32;
        epoch_losses.push(avg);
    }

    let final_loss = epoch_losses
        .last()
        .copied()
        .ok_or_else(|| TuneError::InvalidConfig("num_epochs produced no losses".into()))?;
    let total_steps = config.num_epochs * samples.len();

    Ok(TrainResult {
        final_loss,
        epoch_losses,
        total_steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::{LoraAdapter, LoraConfig, LoraLayer};
    use std::collections::HashMap;

    /// rank=2, d_in=3, d_out=2, scale=1.0 (alpha=2.0, rank=2)
    fn make_small_adapter() -> LoraAdapter {
        let config = LoraConfig {
            rank: 2,
            alpha: 2.0,
            target_modules: vec!["q_proj".into()],
        };
        let mut layers = HashMap::new();
        layers.insert(
            (0, "q_proj".into()),
            LoraLayer {
                a: vec![
                    1.0, 0.5, 0.0, // row 0
                    0.0, 0.5, 1.0, // row 1
                ],
                b: vec![
                    1.0, 0.0, // row 0
                    0.0, 1.0, // row 1
                ],
                d_in: 3,
                d_out: 2,
                rank: 2,
            },
        );
        LoraAdapter::new(config, layers)
    }

    /// Build a synthetic sample whose target_delta is the initial LoRA output for the given
    /// input, slightly perturbed — this gives a reachable regression target without
    /// requiring a warm-up forward pass from the caller.
    fn make_samples(n: usize) -> Vec<TrainSample> {
        // Use varied inputs so samples span different directions in weight-space.
        (0..n)
            .map(|i| {
                let fi = i as f32;
                TrainSample {
                    layer_idx: 0,
                    module: "q_proj".to_string(),
                    // d_in = 3
                    input: vec![fi * 0.1 + 0.1, fi * 0.2 + 0.2, fi * 0.3 + 0.3],
                    // d_out = 2  — target is something the network can learn toward
                    target_delta: vec![fi * 0.05, fi * 0.05 + 0.1],
                }
            })
            .collect()
    }

    #[test]
    fn test_train_convergence() {
        let mut adapter = make_small_adapter();
        let samples = make_samples(5);

        let config = LoraTrainConfig {
            learning_rate: 0.01,
            num_epochs: 10,
            batch_size: 5,
        };

        let result = train_lora(&mut adapter, &samples, &config).expect("train_lora must succeed");

        assert_eq!(result.epoch_losses.len(), 10);
        assert_eq!(result.total_steps, 50);
        assert!(
            result.epoch_losses.last().unwrap() < result.epoch_losses.first().unwrap(),
            "loss must decrease over 10 epochs: first={:.6}, last={:.6}",
            result.epoch_losses.first().unwrap(),
            result.epoch_losses.last().unwrap(),
        );
    }

    #[test]
    fn test_train_empty_batch() {
        let mut adapter = make_small_adapter();
        let config = LoraTrainConfig {
            learning_rate: 0.01,
            num_epochs: 5,
            batch_size: 1,
        };

        let err =
            train_lora(&mut adapter, &[], &config).expect_err("empty samples must return Err");
        assert!(
            matches!(err, TuneError::Training(_)),
            "expected Training variant, got {err:?}"
        );
    }

    #[test]
    fn test_train_dimension_mismatch() {
        let mut adapter = make_small_adapter();
        // d_in for layer (0, "q_proj") is 3, but we pass a 2-element input.
        let bad_samples = vec![TrainSample {
            layer_idx: 0,
            module: "q_proj".to_string(),
            input: vec![1.0, 2.0], // wrong length (d_in=3)
            target_delta: vec![1.0, 1.0],
        }];
        let config = LoraTrainConfig {
            learning_rate: 0.01,
            num_epochs: 1,
            batch_size: 1,
        };

        let err = train_lora(&mut adapter, &bad_samples, &config)
            .expect_err("dimension mismatch must return Err");
        assert!(
            matches!(err, TuneError::DimensionMismatch { .. }),
            "expected DimensionMismatch variant, got {err:?}"
        );
    }

    #[test]
    fn test_train_metrics() {
        let mut adapter = make_small_adapter();
        let samples = make_samples(5);

        let config = LoraTrainConfig {
            learning_rate: 0.01,
            num_epochs: 3,
            batch_size: 5,
        };

        let result = train_lora(&mut adapter, &samples, &config).expect("train_lora must succeed");

        assert_eq!(result.total_steps, 15, "3 epochs × 5 samples = 15 steps");
        assert_eq!(result.epoch_losses.len(), 3, "one loss entry per epoch");
    }
}
