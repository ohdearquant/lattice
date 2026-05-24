//! Batch LoRA training loop (ADR-057, perf-opt-train-loop).
//!
//! Provides [`train_lora`] which iterates over a slice of [`TrainSample`]s for
//! multiple epochs, calling [`adapt_step`] for each sample and accumulating per-epoch
//! MSE loss statistics.
//!
//! When [`LoraTrainConfig::optimizer`] is `Some(Adam | AdamW)`, the loop uses
//! [`AdamState`](super::optimizer::AdamState) to apply adaptive gradient updates
//! instead of plain SGD.  When `optimizer` is `None`, the loop falls back to the
//! original [`adapt_step`] SGD path for full backward compatibility.

use super::{
    LoraAdapter,
    online::adapt_step,
    optimizer::{AdamState, compute_lora_gradients},
};
use crate::error::TuneError;
use crate::train::{LRSchedule, Optimizer, OptimizerConfig};

/// Configuration for a batch LoRA training run.
#[derive(Debug, Clone)]
pub struct LoraTrainConfig {
    /// SGD step size applied inside each [`adapt_step`] call (also used as
    /// the base learning rate when an Adam optimizer is active).
    pub learning_rate: f32,
    /// Number of full passes over the sample set.
    pub num_epochs: usize,
    /// Batch size (informational; does not affect the current sequential loop).
    pub batch_size: usize,
    /// Optional optimizer configuration.  When `None` the loop uses the
    /// backward-compatible SGD path ([`adapt_step`]).  When `Some`, an
    /// Adam or AdamW update is performed using the parameters from the
    /// [`OptimizerConfig`] (the `learning_rate` field in the config is used
    /// as the base LR, overriding this struct's `learning_rate` field).
    pub optimizer: Option<OptimizerConfig>,
    /// Optional learning-rate schedule.  Only used when `optimizer` is
    /// `Some`; ignored in the SGD path.
    pub lr_schedule: Option<LRSchedule>,
    /// Optional gradient clipping by global L2 norm.  Applied to each
    /// sample's gradients before the optimizer step when `optimizer` is `Some`.
    pub grad_clip_norm: Option<f32>,
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
/// deterministic behaviour).  The update rule is selected by
/// [`LoraTrainConfig::optimizer`]:
///
/// * `None` — backward-compatible SGD via [`adapt_step`].
/// * `Some(Adam)` — Adam adaptive gradient update.
/// * `Some(AdamW)` — AdamW (Adam + decoupled weight decay).
///
/// # Arguments
///
/// * `adapter` – Mutable LoRA adapter whose weights are updated in place.
/// * `samples` – Slice of training samples; must be non-empty.
/// * `config`  – Hyper-parameters for the training run.
///
/// # Errors
///
/// * [`TuneError::Training`]          – `samples` is empty.
/// * [`TuneError::InvalidConfig`]     – `batch_size == 0` or `num_epochs == 0`.
/// * [`TuneError::Training`]          – A sample references a `(layer_idx, module)` pair
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

    match &config.optimizer {
        None => train_lora_sgd(adapter, samples, config),
        Some(opt_cfg) => train_lora_adam(adapter, samples, config, opt_cfg),
    }
}

/// SGD path — delegates to `adapt_step` for backward compatibility.
fn train_lora_sgd(
    adapter: &mut LoraAdapter,
    samples: &[TrainSample],
    config: &LoraTrainConfig,
) -> Result<TrainResult, TuneError> {
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
        epoch_losses.push(epoch_loss_sum / samples.len() as f32);
    }

    let final_loss = *epoch_losses.last().expect("num_epochs > 0");
    Ok(TrainResult {
        final_loss,
        epoch_losses,
        total_steps: config.num_epochs * samples.len(),
    })
}

/// Adam / AdamW path.
fn train_lora_adam(
    adapter: &mut LoraAdapter,
    samples: &[TrainSample],
    config: &LoraTrainConfig,
    opt_cfg: &OptimizerConfig,
) -> Result<TrainResult, TuneError> {
    let decoupled = matches!(opt_cfg.optimizer, Optimizer::AdamW);
    let base_lr = opt_cfg.learning_rate;

    let mut adam = AdamState::new();
    let mut epoch_losses = Vec::with_capacity(config.num_epochs);
    let mut global_step: usize = 0;

    for epoch in 0..config.num_epochs {
        let mut epoch_loss_sum = 0.0f32;

        for sample in samples {
            // Effective LR from optional schedule.
            let lr = if let Some(sched) = &config.lr_schedule {
                sched.get_lr(base_lr, global_step, epoch)
            } else {
                base_lr
            };

            // Compute gradients without touching weights.
            let mut grads = compute_lora_gradients(
                adapter,
                sample.layer_idx,
                &sample.module,
                &sample.input,
                &sample.target_delta,
            )?;

            epoch_loss_sum += grads.loss;

            // Optional gradient clipping by global L2 norm.
            if let Some(max_norm) = config.grad_clip_norm {
                let sq_sum: f32 = grads
                    .grad_b
                    .iter()
                    .chain(grads.grad_a.iter())
                    .map(|g| g * g)
                    .sum();
                let norm = sq_sum.sqrt();
                if norm > max_norm {
                    let scale = max_norm / (norm + 1e-8);
                    for g in grads.grad_b.iter_mut().chain(grads.grad_a.iter_mut()) {
                        *g *= scale;
                    }
                }
            }

            // Apply Adam update to B and A parameters for this layer.
            let key = &(sample.layer_idx, sample.module.clone());
            let lora = adapter.layers.get_mut(key).ok_or_else(|| {
                TuneError::Training(format!(
                    "no LoRA layer for ({}, {})",
                    sample.layer_idx, sample.module
                ))
            })?;

            let key_b = format!("{}_{}_b", sample.layer_idx, sample.module);
            let key_a = format!("{}_{}_a", sample.layer_idx, sample.module);

            adam.step(
                &key_b,
                &mut lora.b,
                &grads.grad_b,
                lr,
                opt_cfg.beta1,
                opt_cfg.beta2,
                opt_cfg.epsilon,
                opt_cfg.weight_decay,
                decoupled,
            );
            // Decrement step counter so A and B share the same effective step t.
            adam.t -= 1;
            adam.step(
                &key_a,
                &mut lora.a,
                &grads.grad_a,
                lr,
                opt_cfg.beta1,
                opt_cfg.beta2,
                opt_cfg.epsilon,
                opt_cfg.weight_decay,
                decoupled,
            );

            global_step += 1;
        }

        epoch_losses.push(epoch_loss_sum / samples.len() as f32);
    }

    let final_loss = *epoch_losses.last().expect("num_epochs > 0");
    Ok(TrainResult {
        final_loss,
        epoch_losses,
        total_steps: config.num_epochs * samples.len(),
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

    fn sgd_config(learning_rate: f32, num_epochs: usize) -> LoraTrainConfig {
        LoraTrainConfig {
            learning_rate,
            num_epochs,
            batch_size: 5,
            optimizer: None,
            lr_schedule: None,
            grad_clip_norm: None,
        }
    }

    #[test]
    fn test_train_convergence() {
        let mut adapter = make_small_adapter();
        let samples = make_samples(5);

        let config = sgd_config(0.01, 10);

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
            optimizer: None,
            lr_schedule: None,
            grad_clip_norm: None,
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
            optimizer: None,
            lr_schedule: None,
            grad_clip_norm: None,
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

        let config = sgd_config(0.01, 3);

        let result = train_lora(&mut adapter, &samples, &config).expect("train_lora must succeed");

        assert_eq!(result.total_steps, 15, "3 epochs × 5 samples = 15 steps");
        assert_eq!(result.epoch_losses.len(), 3, "one loss entry per epoch");
    }
}
