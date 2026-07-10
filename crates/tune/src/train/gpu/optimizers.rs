//! GPU optimizer implementations
//!
//! Every `GpuOptimizer` arm (Adam, AdamW, SGD-momentum, plain SGD, RMSprop)
//! currently fails loudly with `TuneError::Training` rather than performing a
//! real weight update: the shader-based arms have no buffer bindings wired
//! to the network's weight/gradient buffers, and the CPU-side plain-SGD arm
//! has neither real gradient plumbing nor a mutable weight write-back path.
//! See #797. Wiring the real dispatch/write-back is tracked follow-up work.

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
                    "GPU RMSprop optimizer not implemented: this arm previously silently \
                     substituted plain SGD instead of running the requested algorithm; every \
                     GpuOptimizer arm fails loudly until real buffer/gradient wiring lands (#797)"
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
    /// # Errors
    ///
    /// Always returns `TuneError::Training`: this is not real SGD (#797).
    /// The previous body used a constant placeholder gradient magnitude
    /// (`grad_scale = 0.01`) instead of the actual per-layer gradients
    /// accumulated in `LayerGradients`, and had no mutable write-back path
    /// from `GpuNetwork` (only the immutable `cpu_network()` accessor
    /// exists) even if it had used real gradients — so the computed values
    /// were always discarded. A working plain-SGD arm needs both real
    /// gradient plumbing and a mutable weight-write path on `GpuNetwork`;
    /// until then it fails loudly like every other arm rather than
    /// reporting success for a step that changed nothing.
    fn update_sgd(_network: &lattice_fann::gpu::GpuNetwork, _current_lr: f32) -> Result<()> {
        Err(TuneError::Training(
            "GPU SGD optimizer update not implemented: no real gradient plumbing or mutable \
             weight write-back from GpuNetwork is wired (#797)"
                .to_string(),
        ))
    }
}
