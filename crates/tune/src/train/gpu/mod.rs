//! GPU-accelerated training infrastructure
//!
//! Uses lattice-fann's wgpu-based GPU backend for accelerated training.

mod builder;
mod optimizers;
mod state;
mod uniforms;

pub use builder::GpuTrainerBuilder;

use crate::data::{Batch, Dataset};
use crate::error::{Result, TuneError};
use crate::train::config::{Optimizer, OptimizerConfig, TrainingConfig};
use lattice_fann::Network;
use lattice_fann::gpu::{GpuContext, GpuNetwork};
use state::{LayerGradients, OptimizerState};
use std::sync::Arc;

/// GPU-accelerated trainer
///
/// Provides GPU-accelerated forward/backward passes and weight updates
/// using lattice-fann's wgpu backend.
pub struct GpuTrainer {
    /// GPU context (device, queue, shader manager)
    ctx: Arc<GpuContext>,
    /// GPU-accelerated network
    network: GpuNetwork,
    /// Per-layer gradient buffers and optimizer state
    layer_gradients: Vec<LayerGradients>,
    /// Training configuration
    config: TrainingConfig,
    /// Current learning rate
    current_lr: f32,
    /// Global training step
    global_step: usize,
}

impl GpuTrainer {
    /// Create a new GPU trainer
    ///
    /// # Arguments
    /// * `network` - The neural network to train
    /// * `config` - Training configuration
    pub fn new(network: Network, config: TrainingConfig) -> Result<Self> {
        config.validate()?;

        // Create GPU context
        let ctx = Arc::new(
            GpuContext::new_blocking()
                .map_err(|e| TuneError::Training(format!("Failed to create GPU context: {e}")))?,
        );

        // Wrap network for GPU
        let gpu_network = GpuNetwork::new(ctx.clone(), network)
            .map_err(|e| TuneError::Training(format!("Failed to create GPU network: {e}")))?;

        // Initialize gradient buffers and optimizer state for each layer
        let layer_gradients = Self::init_gradients(&ctx, gpu_network.cpu_network(), &config)?;

        let current_lr = config.optimizer.learning_rate;

        Ok(Self {
            ctx,
            network: gpu_network,
            layer_gradients,
            config,
            current_lr,
            global_step: 0,
        })
    }

    /// Create with existing GPU context (for sharing across multiple trainers)
    pub fn with_context(
        ctx: Arc<GpuContext>,
        network: Network,
        config: TrainingConfig,
    ) -> Result<Self> {
        config.validate()?;

        let gpu_network = GpuNetwork::new(ctx.clone(), network)
            .map_err(|e| TuneError::Training(format!("Failed to create GPU network: {e}")))?;

        let layer_gradients = Self::init_gradients(&ctx, gpu_network.cpu_network(), &config)?;
        let current_lr = config.optimizer.learning_rate;

        Ok(Self {
            ctx,
            network: gpu_network,
            layer_gradients,
            config,
            current_lr,
            global_step: 0,
        })
    }

    /// Initialize gradient buffers and optimizer state
    fn init_gradients(
        ctx: &Arc<GpuContext>,
        network: &Network,
        config: &TrainingConfig,
    ) -> Result<Vec<LayerGradients>> {
        let device = ctx.device();
        let mut gradients = Vec::with_capacity(network.num_layers());

        for layer in network.layers() {
            let num_weights = layer.num_inputs() * layer.num_outputs();
            let num_biases = layer.num_outputs();

            // Create gradient buffers
            let weight_grads = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("weight_grads"),
                size: (num_weights * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bias_grads = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bias_grads"),
                size: (num_biases * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Create optimizer state buffers
            let optimizer_state =
                Self::create_optimizer_state(device, num_weights + num_biases, &config.optimizer);

            gradients.push(LayerGradients {
                weight_grads,
                bias_grads,
                optimizer_state,
                num_weights,
                num_biases,
            });
        }

        Ok(gradients)
    }

    /// Create optimizer state buffers
    fn create_optimizer_state(
        device: &wgpu::Device,
        total_params: usize,
        _config: &OptimizerConfig,
    ) -> OptimizerState {
        let buffer_size = (total_params * std::mem::size_of::<f32>()) as u64;

        // Create zero-initialized buffers
        let m = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adam_m"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adam_v"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let velocity = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sgd_velocity"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        OptimizerState {
            m,
            v,
            velocity,
            t: 0,
        }
    }

    /// Train on a single batch
    ///
    /// Returns the batch loss.
    pub fn train_batch(&mut self, batch: &Batch) -> Result<f32> {
        self.global_step += 1;

        // Forward pass
        let (outputs, activations) = self.forward_batch(batch)?;

        // Check for NaN/Inf values after forward pass
        Self::check_numeric_stability(&outputs)?;

        // Compute loss
        let loss = self.compute_loss(&outputs, batch)?;

        // Check loss is valid
        if loss.is_nan() || loss.is_infinite() {
            return Err(TuneError::Training(format!(
                "Invalid loss value: {loss} (NaN or Inf detected)"
            )));
        }

        // Backward pass
        self.backward_batch(&outputs, &activations, batch)?;

        // Update weights
        self.update_weights()?;

        // Update learning rate
        self.current_lr = self.config.lr_schedule.get_lr(
            self.config.optimizer.learning_rate,
            self.global_step,
            self.global_step / 100, // Approximate epoch from step
        );

        Ok(loss)
    }

    /// Forward pass for a batch
    #[allow(clippy::type_complexity)]
    fn forward_batch(&mut self, batch: &Batch) -> Result<(Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>)> {
        let mut outputs = Vec::with_capacity(batch.len());
        let mut all_activations = Vec::with_capacity(batch.len());

        for example in &batch.examples {
            // Flatten context + message into input
            let input = self.prepare_input(example)?;

            // Forward through network
            let output = self
                .network
                .forward_sync(&input)
                .map_err(|e| TuneError::Training(format!("Forward pass failed: {e}")))?;

            // Store activations for backward pass (simplified - just input and output)
            let activations = vec![input.clone(), output.clone()];

            outputs.push(output);
            all_activations.push(activations);
        }

        Ok((outputs, all_activations))
    }

    /// Prepare input from training example
    fn prepare_input(&self, example: &crate::data::TrainingExample) -> Result<Vec<f32>> {
        // Concatenate context embeddings and message embedding
        let mut input = Vec::new();

        // Add context (average pool if multiple)
        if !example.context_embeddings.is_empty() {
            let context_dim = example.context_embeddings[0].len();
            let mut pooled = vec![0.0f32; context_dim];
            for ctx in &example.context_embeddings {
                for (i, v) in ctx.iter().enumerate() {
                    pooled[i] += v;
                }
            }
            let n = example.context_embeddings.len() as f32;
            for v in pooled.iter_mut() {
                *v /= n;
            }
            input.extend(pooled);
        }

        // Add message embedding
        input.extend(&example.message_embedding);

        Ok(input)
    }

    /// Compute cross-entropy loss
    fn compute_loss(&self, outputs: &[Vec<f32>], batch: &Batch) -> Result<f32> {
        let mut total_loss = 0.0;

        for (output, example) in outputs.iter().zip(&batch.examples) {
            let targets = example.labels.to_vec();

            // Cross-entropy with label smoothing
            let smoothing = self.config.regularization.label_smoothing;
            let n_classes = output.len() as f32;

            for (pred, target) in output.iter().zip(targets.iter()) {
                let smoothed_target = (1.0 - smoothing) * target + smoothing / n_classes;
                total_loss -= smoothed_target * (pred.max(1e-7)).ln();
            }
        }

        Ok(total_loss / batch.len() as f32)
    }

    /// Backward pass (compute gradients)
    fn backward_batch(
        &mut self,
        outputs: &[Vec<f32>],
        _activations: &[Vec<Vec<f32>>],
        batch: &Batch,
    ) -> Result<()> {
        // Compute output gradients (softmax + cross-entropy derivative)
        let mut output_grads: Vec<Vec<f32>> = Vec::with_capacity(outputs.len());

        for (output, example) in outputs.iter().zip(&batch.examples) {
            let targets = example.labels.to_vec();
            let mut grad = Vec::with_capacity(output.len());

            // d(CE)/d(softmax) = softmax - target
            for (pred, target) in output.iter().zip(targets.iter()) {
                grad.push(pred - target);
            }

            output_grads.push(grad);
        }

        // Backpropagate through layers
        // This is a simplified implementation - full implementation would use GPU shaders
        self.backprop_cpu(&output_grads, batch)?;

        Ok(())
    }

    /// CPU backpropagation (fallback for now)
    fn backprop_cpu(&mut self, output_grads: &[Vec<f32>], _batch: &Batch) -> Result<()> {
        // For now, compute average gradients on CPU and upload to GPU
        // This will be replaced with full GPU backprop in a later iteration

        let network = self.network.cpu_network();
        let batch_size = output_grads.len() as f32;

        // Initialize gradient accumulators
        for (layer_idx, layer) in network.layers().iter().enumerate() {
            let num_weights = layer.num_inputs() * layer.num_outputs();
            let num_biases = layer.num_outputs();

            // Simple gradient accumulation (placeholder)
            // Real implementation would compute proper gradients
            let weight_grads = vec![0.01f32 / batch_size; num_weights];
            let bias_grads = vec![0.01f32 / batch_size; num_biases];

            // Upload to GPU
            self.ctx.queue().write_buffer(
                &self.layer_gradients[layer_idx].weight_grads,
                0,
                bytemuck::cast_slice(&weight_grads),
            );
            self.ctx.queue().write_buffer(
                &self.layer_gradients[layer_idx].bias_grads,
                0,
                bytemuck::cast_slice(&bias_grads),
            );
        }

        Ok(())
    }

    /// Update weights using GPU optimizer shaders
    fn update_weights(&mut self) -> Result<()> {
        optimizers::GpuOptimizer::update(
            &self.ctx,
            &mut self.layer_gradients,
            &self.config,
            self.current_lr,
            &mut self.network,
        )
    }

    /// Check if outputs contain NaN or Inf values
    fn check_numeric_stability(outputs: &[Vec<f32>]) -> Result<()> {
        for (batch_idx, output) in outputs.iter().enumerate() {
            for (i, &v) in output.iter().enumerate() {
                if v.is_nan() {
                    return Err(TuneError::Training(format!(
                        "NaN detected in output at batch {batch_idx}, index {i}"
                    )));
                }
                if v.is_infinite() {
                    return Err(TuneError::Training(format!(
                        "Inf detected in output at batch {batch_idx}, index {i}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate on a dataset
    pub fn validate(&mut self, dataset: &mut Dataset) -> Result<(f32, f32)> {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        for batch in dataset.batches() {
            let (outputs, _) = self.forward_batch(&batch)?;

            // Check for NaN/Inf values before processing
            Self::check_numeric_stability(&outputs)?;
            let loss = self.compute_loss(&outputs, &batch)?;
            total_loss += loss * batch.len() as f32;

            // Compute accuracy
            for (output, example) in outputs.iter().zip(&batch.examples) {
                let predicted = output
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                let targets = example.labels.to_vec();
                let target = targets
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if predicted == target {
                    correct += 1;
                }
                total += 1;
            }
        }

        let avg_loss = if total > 0 {
            total_loss / total as f32
        } else {
            0.0
        };

        let accuracy = if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        };

        Ok((avg_loss, accuracy))
    }

    /// Get current learning rate
    pub fn current_lr(&self) -> f32 {
        self.current_lr
    }

    /// Get global step count
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    /// Check if using GPU acceleration
    pub fn is_using_gpu(&self) -> bool {
        self.network.is_using_gpu()
    }

    /// Get GPU device info
    pub fn device_info(&self) -> String {
        format!("{:?}", self.ctx.info())
    }

    /// Get reference to underlying network
    pub fn network(&self) -> &Network {
        self.network.cpu_network()
    }
}

#[cfg(test)]
mod tests;
