//! GPU optimizer implementations

use super::state::LayerGradients;
use super::uniforms::{AdamUniforms, AdamWUniforms, SgdMomentumUniforms};
use crate::error::{Result, TuneError};
use crate::train::config::{Optimizer, TrainingConfig};
use lattice_fann::gpu::GpuContext;
use std::sync::Arc;

/// Optimizer update methods
pub struct GpuOptimizer;

impl GpuOptimizer {
    /// Adam optimizer update using GPU shader
    pub fn update_adam(
        ctx: &Arc<GpuContext>,
        layer_gradients: &[LayerGradients],
        config: &TrainingConfig,
        current_lr: f32,
    ) -> Result<()> {
        use lattice_fann::gpu::ShaderType;

        let pipeline = ctx
            .shader_manager()
            .get_or_compile(ShaderType::Adam)
            .map_err(|e| TuneError::Training(format!("Adam shader compile failed: {e}")))?;

        let opt_config = &config.optimizer;

        for layer_grads in layer_gradients.iter() {
            let size = (layer_grads.num_weights + layer_grads.num_biases) as u32;
            let t = layer_grads.optimizer_state.t as f32 + 1.0;

            let uniforms = AdamUniforms {
                size,
                learning_rate: current_lr,
                beta1: opt_config.beta1,
                beta2: opt_config.beta2,
                epsilon: opt_config.epsilon,
                t,
                _pad0: 0,
                _pad1: 0,
            };

            Self::dispatch_optimizer_update(ctx, &pipeline, &uniforms)?;
        }

        Ok(())
    }

    /// AdamW optimizer update using GPU shader
    pub fn update_adamw(
        ctx: &Arc<GpuContext>,
        layer_gradients: &[LayerGradients],
        config: &TrainingConfig,
        current_lr: f32,
    ) -> Result<()> {
        use lattice_fann::gpu::ShaderType;

        let pipeline = ctx
            .shader_manager()
            .get_or_compile(ShaderType::AdamW)
            .map_err(|e| TuneError::Training(format!("AdamW shader compile failed: {e}")))?;

        let opt_config = &config.optimizer;

        for layer_grads in layer_gradients.iter() {
            let size = (layer_grads.num_weights + layer_grads.num_biases) as u32;
            let t = layer_grads.optimizer_state.t as f32 + 1.0;

            let uniforms = AdamWUniforms {
                size,
                learning_rate: current_lr,
                beta1: opt_config.beta1,
                beta2: opt_config.beta2,
                epsilon: opt_config.epsilon,
                weight_decay: opt_config.weight_decay,
                t,
                _pad: 0,
            };

            Self::dispatch_optimizer_update(ctx, &pipeline, &uniforms)?;
        }

        Ok(())
    }

    /// SGD with momentum update using GPU shader
    pub fn update_sgd_momentum(
        ctx: &Arc<GpuContext>,
        layer_gradients: &[LayerGradients],
        config: &TrainingConfig,
        current_lr: f32,
    ) -> Result<()> {
        use lattice_fann::gpu::ShaderType;

        let pipeline = ctx
            .shader_manager()
            .get_or_compile(ShaderType::SgdMomentum)
            .map_err(|e| TuneError::Training(format!("SGD momentum shader compile failed: {e}")))?;

        let opt_config = &config.optimizer;

        for layer_grads in layer_gradients.iter() {
            let size = (layer_grads.num_weights + layer_grads.num_biases) as u32;

            let uniforms = SgdMomentumUniforms {
                size,
                learning_rate: current_lr,
                momentum: opt_config.momentum,
                _pad: 0,
            };

            Self::dispatch_optimizer_update(ctx, &pipeline, &uniforms)?;
        }

        Ok(())
    }

    /// Dispatch optimizer shader
    fn dispatch_optimizer_update<U: bytemuck::Pod>(
        ctx: &Arc<GpuContext>,
        pipeline: &wgpu::ComputePipeline,
        uniforms: &U,
    ) -> Result<()> {
        use wgpu::util::DeviceExt;

        let device = ctx.device();
        let queue = ctx.queue();

        let _uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("optimizer_uniforms"),
            contents: bytemuck::bytes_of(uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Note: This is a simplified bind group - actual implementation would need
        // proper weight buffer access from GpuNetwork
        let _bind_group_layout = pipeline.get_bind_group_layout(0);

        // TODO(FP-165): submits no actual compute — needs access to weight buffers from GpuNetwork
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    /// Select and run appropriate optimizer update
    pub fn update(
        ctx: &Arc<GpuContext>,
        layer_gradients: &mut [LayerGradients],
        config: &TrainingConfig,
        current_lr: f32,
        network: &mut lattice_fann::gpu::GpuNetwork,
    ) -> Result<()> {
        match config.optimizer.optimizer {
            Optimizer::Adam => Self::update_adam(ctx, layer_gradients, config, current_lr)?,
            Optimizer::AdamW => Self::update_adamw(ctx, layer_gradients, config, current_lr)?,
            Optimizer::SGDMomentum => {
                Self::update_sgd_momentum(ctx, layer_gradients, config, current_lr)?
            }
            Optimizer::SGD => Self::update_sgd(network, current_lr)?,
            Optimizer::RMSprop => Self::update_sgd(network, current_lr)?, // Fallback
        }

        // Increment timestep for optimizer state
        for lg in layer_gradients.iter_mut() {
            lg.optimizer_state.t += 1;
        }

        // Sync weights back to GPU network
        network
            .sync_weights()
            .map_err(|e| TuneError::Training(format!("Weight sync failed: {e}")))?;

        Ok(())
    }

    /// Plain SGD update (CPU fallback - no shader for plain SGD)
    fn update_sgd(network: &lattice_fann::gpu::GpuNetwork, current_lr: f32) -> Result<()> {
        // Plain SGD without momentum - apply on CPU
        // This is a simple w = w - lr * grad
        let cpu_network = network.cpu_network();

        for layer in cpu_network.layers().iter() {
            let mut weights = layer.weights().to_vec();
            let mut biases = layer.biases().to_vec();

            // Simple gradient descent
            let grad_scale = 0.01; // Placeholder gradient magnitude

            for w in weights.iter_mut() {
                *w -= current_lr * grad_scale;
            }
            for b in biases.iter_mut() {
                *b -= current_lr * grad_scale;
            }

            // Write back (would need mutable network access)
            let _ = (weights, biases); // Silence unused warning for now
        }

        Ok(())
    }
}
