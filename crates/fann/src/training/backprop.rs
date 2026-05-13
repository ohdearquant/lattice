//! Backpropagation trainer implementation
//!
//! Provides the `BackpropTrainer` for training neural networks using
//! backpropagation with momentum.

use crate::error::{FannError, FannResult};
use crate::network::Network;
use crate::training::gradient::{GradientGuardStrategy, check_gradients_valid, sanitize_gradients};
use crate::training::{Trainer, TrainingConfig, TrainingResult};

/// Basic backpropagation trainer with momentum
#[derive(Debug, Default)]
pub struct BackpropTrainer {
    /// Velocity buffers for momentum (one per layer)
    weight_velocities: Vec<Vec<f32>>,
    bias_velocities: Vec<Vec<f32>>,
}

impl BackpropTrainer {
    /// Create a new backpropagation trainer
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize velocity buffers for a network
    fn init_velocities(&mut self, network: &Network) {
        self.weight_velocities.clear();
        self.bias_velocities.clear();

        for layer in network.layers() {
            self.weight_velocities
                .push(vec![0.0; layer.weights().len()]);
            self.bias_velocities.push(vec![0.0; layer.biases().len()]);
        }
    }

    /// Compute gradients for a single training sample using backpropagation.
    ///
    /// After the forward pass, references weights and activations directly from
    /// the network (no per-sample cloning). The forward pass is the only mutation;
    /// all subsequent reads use immutable borrows.
    fn compute_gradients(
        &self,
        network: &mut Network,
        input: &[f32],
        target: &[f32],
        weight_grads: &mut [Vec<f32>],
        bias_grads: &mut [Vec<f32>],
    ) -> FannResult<f32> {
        let num_layers = network.num_layers();

        // Forward pass (populates activation buffers) — only mutation
        let output: Vec<f32> = network.forward(input)?.to_vec();
        let output_len = output.len();

        // After forward returns, reborrow network immutably.
        // Reference layers and activations directly — no cloning needed.
        let layers = network.layers();

        // Compute output error (MSE derivative)
        let mut error = 0.0;
        let mut deltas: Vec<Vec<f32>> = Vec::with_capacity(num_layers);

        // Output layer deltas
        let output_activation = layers[num_layers - 1].activation();
        let mut output_deltas = vec![0.0_f32; output_len];
        if matches!(output_activation, crate::activation::Activation::Softmax) {
            // Full Jacobian for softmax: J[i,j] = s[j]*(delta[i,j] - s[i])
            // Chain rule with MSE loss: delta[i] = sum_j((output[j]-target[j]) * J[j,i])
            for i in 0..output_len {
                let mut delta = 0.0_f32;
                for j in 0..output_len {
                    let diff_j = output[j] - target[j];
                    let jacobian_ji = if i == j {
                        output[j] * (1.0 - output[i])
                    } else {
                        -output[j] * output[i]
                    };
                    delta += diff_j * jacobian_ji;
                }
                output_deltas[i] = delta;
                let diff = output[i] - target[i];
                error += diff * diff;
            }
        } else {
            for i in 0..output_len {
                let diff = output[i] - target[i];
                error += diff * diff;
                output_deltas[i] = diff * output_activation.derivative(output[i]);
            }
        }
        deltas.push(output_deltas);

        // Backpropagate deltas through hidden layers
        for layer_idx in (0..num_layers - 1).rev() {
            let layer_activation = layers[layer_idx].activation();
            let layer_num_outputs = layers[layer_idx].num_outputs();
            let next_num_inputs = layers[layer_idx + 1].num_inputs();
            let next_num_outputs = layers[layer_idx + 1].num_outputs();
            let next_weights = layers[layer_idx + 1].weights(); // &[f32] — no clone
            let layer_activations = network.activations(layer_idx).ok_or_else(|| {
                FannError::TrainingError(format!("missing activations for layer {layer_idx}"))
            })?;

            let mut layer_deltas = vec![0.0; layer_num_outputs];

            let prev_deltas = deltas.last().ok_or_else(|| {
                FannError::TrainingError("empty deltas during backpropagation".to_string())
            })?;
            for i in 0..layer_num_outputs {
                let mut sum = 0.0;
                for j in 0..next_num_outputs {
                    let weight = next_weights[j * next_num_inputs + i];
                    sum += weight * prev_deltas[j];
                }
                let deriv = layer_activation.derivative(layer_activations[i]);
                layer_deltas[i] = sum * deriv;
            }

            deltas.push(layer_deltas);
        }

        // Reverse deltas (now in layer order)
        deltas.reverse();

        // Compute gradients — reference layer dimensions and activations directly
        for (layer_idx, delta) in deltas.iter().enumerate() {
            let num_inputs = layers[layer_idx].num_inputs();
            let num_outputs = layers[layer_idx].num_outputs();
            let layer_input: &[f32] = if layer_idx == 0 {
                input
            } else {
                network.activations(layer_idx - 1).ok_or_else(|| {
                    FannError::TrainingError(format!(
                        "missing activations for layer {}",
                        layer_idx - 1
                    ))
                })?
            };

            // Weight gradients: dW[i,j] = delta[i] * input[j]
            for (i, &d) in delta.iter().enumerate().take(num_outputs) {
                for (j, &inp) in layer_input.iter().enumerate().take(num_inputs) {
                    weight_grads[layer_idx][i * num_inputs + j] += d * inp;
                }
            }

            // Bias gradients: dB[i] = delta[i]
            for (i, &d) in delta.iter().enumerate().take(num_outputs) {
                bias_grads[layer_idx][i] += d;
            }
        }

        Ok(error / output_len as f32)
    }

    /// Apply gradients with momentum and weight decay
    fn apply_gradients(
        &mut self,
        network: &mut Network,
        weight_grads: &[Vec<f32>],
        bias_grads: &[Vec<f32>],
        config: &TrainingConfig,
        batch_size: usize,
    ) {
        let lr = config.learning_rate / batch_size as f32;
        let momentum = config.momentum;
        let weight_decay = config.weight_decay;

        for layer_idx in 0..network.num_layers() {
            let Some(layer) = network.layer_mut(layer_idx) else {
                continue;
            };

            // Update weights
            let weights = layer.weights_mut();
            for (i, w) in weights.iter_mut().enumerate() {
                // Momentum update
                self.weight_velocities[layer_idx][i] = momentum
                    * self.weight_velocities[layer_idx][i]
                    - lr * (weight_grads[layer_idx][i] + weight_decay * *w);
                *w += self.weight_velocities[layer_idx][i];
            }

            // Update biases
            let biases = layer.biases_mut();
            for (i, b) in biases.iter_mut().enumerate() {
                self.bias_velocities[layer_idx][i] =
                    momentum * self.bias_velocities[layer_idx][i] - lr * bias_grads[layer_idx][i];
                *b += self.bias_velocities[layer_idx][i];
            }
        }
    }
}

impl Trainer for BackpropTrainer {
    fn train(
        &mut self,
        network: &mut Network,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        config: &TrainingConfig,
    ) -> FannResult<TrainingResult> {
        if inputs.is_empty() {
            return Err(FannError::TrainingError("No training data".into()));
        }

        if config.batch_size == 0 {
            return Err(FannError::TrainingError(
                "batch_size must be greater than 0".to_string(),
            ));
        }

        if inputs.len() != targets.len() {
            return Err(FannError::TrainingError(format!(
                "Input count {} != target count {}",
                inputs.len(),
                targets.len()
            )));
        }

        // Validate dimensions
        for (i, (input, target)) in inputs.iter().zip(targets.iter()).enumerate() {
            if input.len() != network.num_inputs() {
                return Err(FannError::TrainingError(format!(
                    "Sample {}: input size {} != network input size {}",
                    i,
                    input.len(),
                    network.num_inputs()
                )));
            }
            if target.len() != network.num_outputs() {
                return Err(FannError::TrainingError(format!(
                    "Sample {}: target size {} != network output size {}",
                    i,
                    target.len(),
                    network.num_outputs()
                )));
            }
        }

        // Initialize velocity buffers
        self.init_velocities(network);

        // Allocate gradient buffers
        let mut weight_grads: Vec<Vec<f32>> = network
            .layers()
            .iter()
            .map(|l| vec![0.0; l.weights().len()])
            .collect();
        let mut bias_grads: Vec<Vec<f32>> = network
            .layers()
            .iter()
            .map(|l| vec![0.0; l.biases().len()])
            .collect();

        let mut error_history = Vec::with_capacity(config.max_epochs);
        let mut indices: Vec<usize> = (0..inputs.len()).collect();

        // Create RNG once before the epoch loop to avoid per-epoch allocation
        use rand::SeedableRng;
        let mut rng = match config.seed {
            Some(seed) => rand::rngs::SmallRng::seed_from_u64(seed),
            None => rand::rngs::SmallRng::from_entropy(),
        };

        for epoch in 0..config.max_epochs {
            // Shuffle if configured
            if config.shuffle {
                use rand::seq::SliceRandom;
                indices.shuffle(&mut rng);
            }

            let mut epoch_error = 0.0;
            let mut batch_count = 0;

            // Process batches
            let effective_batch_size = if config.batch_size == 0 {
                1
            } else {
                config.batch_size
            };
            for batch_start in (0..inputs.len()).step_by(effective_batch_size) {
                let batch_end = (batch_start + effective_batch_size).min(inputs.len());
                let actual_batch_size = batch_end - batch_start;

                // Zero gradients
                for grads in weight_grads.iter_mut() {
                    grads.fill(0.0);
                }
                for grads in bias_grads.iter_mut() {
                    grads.fill(0.0);
                }

                // Accumulate gradients over batch
                for &idx in &indices[batch_start..batch_end] {
                    let error = self.compute_gradients(
                        network,
                        &inputs[idx],
                        &targets[idx],
                        &mut weight_grads,
                        &mut bias_grads,
                    )?;
                    epoch_error += error;
                }

                // Check for NaN/Inf gradients before applying
                let gradient_check = check_gradients_valid(&weight_grads, &bias_grads);
                if let Err(location) = gradient_check {
                    match config.gradient_guard {
                        GradientGuardStrategy::Error => {
                            return Err(FannError::NumericInstability(format!(
                                "Epoch {epoch}: {location}"
                            )));
                        }
                        GradientGuardStrategy::Sanitize => {
                            // Replace NaN/Inf with zeros and continue
                            for grads in weight_grads.iter_mut() {
                                sanitize_gradients(grads);
                            }
                            for grads in bias_grads.iter_mut() {
                                sanitize_gradients(grads);
                            }
                            tracing::warn!(
                                "Epoch {}: sanitized NaN/Inf gradients ({})",
                                epoch,
                                location
                            );
                        }
                        GradientGuardStrategy::SkipBatch => {
                            // Skip this batch entirely
                            tracing::warn!(
                                "Epoch {}: skipping batch due to NaN/Inf gradients ({})",
                                epoch,
                                location
                            );
                            continue;
                        }
                    }
                }

                // Apply gradients
                self.apply_gradients(
                    network,
                    &weight_grads,
                    &bias_grads,
                    config,
                    actual_batch_size,
                );

                batch_count += actual_batch_size;
            }

            // Average error
            let avg_error = epoch_error / batch_count as f32;
            error_history.push(avg_error);

            // Check convergence
            if avg_error < config.target_error {
                return Ok(TrainingResult {
                    final_error: avg_error,
                    epochs_trained: epoch + 1,
                    error_history,
                    converged: true,
                });
            }

            // Check for NaN (numeric instability)
            if avg_error.is_nan() {
                return Err(FannError::NumericInstability(
                    "NaN error during training".into(),
                ));
            }
        }

        Ok(TrainingResult {
            final_error: *error_history.last().unwrap_or(&f32::MAX),
            epochs_trained: config.max_epochs,
            error_history,
            converged: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::network::NetworkBuilder;

    #[test]
    fn test_xor_training() {
        // XOR problem - classic test for neural network training
        let mut network = NetworkBuilder::new()
            .input(2)
            .hidden(4, Activation::Tanh)
            .output(1, Activation::Tanh)
            .build()
            .unwrap();

        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![
            vec![0.0], // 0 XOR 0 = 0
            vec![1.0], // 0 XOR 1 = 1
            vec![1.0], // 1 XOR 0 = 1
            vec![0.0], // 1 XOR 1 = 0
        ];

        let mut trainer = BackpropTrainer::new();
        let config = TrainingConfig::new()
            .learning_rate(0.5)
            .momentum(0.0)
            .max_epochs(5000)
            .target_error(0.01)
            .batch_size(4)
            .shuffle(false);

        let result = trainer.train(&mut network, &inputs, &targets, &config);

        // Training should complete without error
        // Note: XOR convergence is not guaranteed with random initialization
        assert!(result.is_ok());
    }

    #[test]
    fn test_training_empty_data() {
        let mut network = NetworkBuilder::new()
            .input(2)
            .output(1, Activation::Linear)
            .build()
            .unwrap();

        let mut trainer = BackpropTrainer::new();
        let config = TrainingConfig::default();

        let result = trainer.train(&mut network, &[], &[], &config);
        assert!(matches!(result, Err(FannError::TrainingError(_))));
    }

    #[test]
    fn test_training_dimension_mismatch() {
        let mut network = NetworkBuilder::new()
            .input(2)
            .output(1, Activation::Linear)
            .build()
            .unwrap();

        let inputs = vec![vec![1.0, 2.0, 3.0]]; // Wrong input size
        let targets = vec![vec![1.0]];

        let mut trainer = BackpropTrainer::new();
        let config = TrainingConfig::default();

        let result = trainer.train(&mut network, &inputs, &targets, &config);
        assert!(matches!(result, Err(FannError::TrainingError(_))));
    }

    #[test]
    fn test_training_count_mismatch() {
        let mut network = NetworkBuilder::new()
            .input(2)
            .output(1, Activation::Linear)
            .build()
            .unwrap();

        let inputs = vec![vec![1.0, 2.0]];
        let targets = vec![vec![1.0], vec![2.0]]; // Different count

        let mut trainer = BackpropTrainer::new();
        let config = TrainingConfig::default();

        let result = trainer.train(&mut network, &inputs, &targets, &config);
        assert!(matches!(result, Err(FannError::TrainingError(_))));
    }

    #[test]
    fn test_linear_regression() {
        // Simple linear regression: y = 2x + 1
        let mut network = NetworkBuilder::new()
            .input(1)
            .output(1, Activation::Linear)
            .build()
            .unwrap();

        let inputs: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32 / 10.0]).collect();
        let targets: Vec<Vec<f32>> = inputs.iter().map(|x| vec![2.0 * x[0] + 1.0]).collect();

        let mut trainer = BackpropTrainer::new();
        let config = TrainingConfig::new()
            .learning_rate(0.1)
            .momentum(0.0)
            .max_epochs(1000)
            .target_error(0.001)
            .batch_size(20)
            .shuffle(false);

        let result = trainer
            .train(&mut network, &inputs, &targets, &config)
            .unwrap();

        // Linear regression should converge easily
        assert!(
            result.final_error < 0.1,
            "Error {} too high",
            result.final_error
        );
    }
}
