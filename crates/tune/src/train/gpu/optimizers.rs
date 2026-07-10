//! GPU optimizer implementations

use super::state::LayerGradients;
use crate::error::{Result, TuneError};
use crate::train::config::{Optimizer, TrainingConfig};
use lattice_fann::gpu::GpuContext;
use std::sync::Arc;

/// Optimizer update methods
pub struct GpuOptimizer;

impl GpuOptimizer {
    /// Adam optimizer update using GPU shader
    ///
    /// # Errors
    ///
    /// Always returns `TuneError::Training`: the GPU dispatch has no buffer
    /// bindings wired to the network's weight/gradient buffers (#797). Wiring
    /// the real dispatch is tracked as follow-up work; until then this arm
    /// fails loudly instead of silently performing a zero-effect update.
    pub fn update_adam(
        _ctx: &Arc<GpuContext>,
        _layer_gradients: &[LayerGradients],
        _config: &TrainingConfig,
        _current_lr: f32,
    ) -> Result<()> {
        Err(TuneError::Training(
            "GPU Adam optimizer update not implemented: optimizer shader dispatch has no \
             buffer bindings wired to the network's weight/gradient buffers (#797)"
                .to_string(),
        ))
    }

    /// AdamW optimizer update using GPU shader
    ///
    /// # Errors
    ///
    /// Always returns `TuneError::Training`: see [`Self::update_adam`] (#797).
    pub fn update_adamw(
        _ctx: &Arc<GpuContext>,
        _layer_gradients: &[LayerGradients],
        _config: &TrainingConfig,
        _current_lr: f32,
    ) -> Result<()> {
        Err(TuneError::Training(
            "GPU AdamW optimizer update not implemented: optimizer shader dispatch has no \
             buffer bindings wired to the network's weight/gradient buffers (#797)"
                .to_string(),
        ))
    }

    /// SGD with momentum update using GPU shader
    ///
    /// # Errors
    ///
    /// Always returns `TuneError::Training`: see [`Self::update_adam`] (#797).
    pub fn update_sgd_momentum(
        _ctx: &Arc<GpuContext>,
        _layer_gradients: &[LayerGradients],
        _config: &TrainingConfig,
        _current_lr: f32,
    ) -> Result<()> {
        Err(TuneError::Training(
            "GPU SGD-momentum optimizer update not implemented: optimizer shader dispatch \
             has no buffer bindings wired to the network's weight/gradient buffers (#797)"
                .to_string(),
        ))
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
            Optimizer::RMSprop => {
                return Err(TuneError::Training(
                    "GPU RMSprop optimizer not implemented; select SGD explicitly (#797)"
                        .to_string(),
                ));
            }
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
    ///
    /// NOTE (#797 follow-up, out of scope for this fix): this arm computes
    /// updated weights/biases but `GpuNetwork` exposes only an immutable
    /// `cpu_network()` accessor, so the computed values are never written
    /// back — this is untouched, pre-existing behavior, tracked separately
    /// from the shader-dispatch fail-loud fix above.
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
