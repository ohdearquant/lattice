//! GPU optimizer dispatch.
//!
//! All optimizer choices return `TuneError::Training` until real gradient
//! bindings and a mutable network weight write-back path exist. This is
//! intentional: a no-effect optimizer step must never report success. See
//! `docs/train.md` for the current backend contract.

use super::state::LayerGradients;
use crate::error::{Result, TuneError};
use crate::train::config::{Optimizer, TrainingConfig};
use lattice_fann::gpu::GpuContext;
use std::sync::Arc;

/// GPU optimizer dispatcher.
///
/// Every update arm currently returns an error rather than performing a
/// no-effect update. See `docs/train.md` for implementation status.
pub struct GpuOptimizer;

impl GpuOptimizer {
    /// Return an error because GPU Adam buffer bindings are not implemented.
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

    /// Return an error because GPU AdamW buffer bindings are not implemented.
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

    /// Return an error because GPU SGD-momentum bindings are not implemented.
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

    /// Return an error because plain SGD has no gradient or weight write-back path.
    fn update_sgd(_network: &lattice_fann::gpu::GpuNetwork, _current_lr: f32) -> Result<()> {
        Err(TuneError::Training(
            "GPU SGD optimizer update not implemented: no real gradient plumbing or mutable \
             weight write-back from GpuNetwork is wired (#797)"
                .to_string(),
        ))
    }
}
