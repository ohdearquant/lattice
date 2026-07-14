//! Training algorithms and their shared configuration.
//!
//! Provides momentum backpropagation, gradient-instability policies, and the
//! feature-gated EWC and RLOO tools for online routing.
//!
//! See `docs/training.md` for training flow, configuration, and algorithms.

mod backprop;
#[cfg(feature = "online-router")]
pub mod ewc;
mod gradient;
#[cfg(feature = "online-router")]
pub mod rloo;

pub use backprop::BackpropTrainer;
#[cfg(feature = "online-router")]
pub use ewc::DiagonalFisher;
pub use gradient::GradientGuardStrategy;
#[cfg(feature = "online-router")]
pub use rloo::{RlooConfig, RlooTrainer};

use crate::error::FannResult;
use crate::network::Network;

/// Training configuration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrainingConfig {
    /// Learning rate.
    pub learning_rate: f32,
    /// Momentum coefficient.
    pub momentum: f32,
    /// L2 weight-decay coefficient.
    pub weight_decay: f32,
    /// Maximum epochs.
    pub max_epochs: usize,
    /// Early-stop error threshold.
    pub target_error: f32,
    /// Batch size; `1` is stochastic gradient descent.
    pub batch_size: usize,
    /// Whether to shuffle each epoch.
    pub shuffle: bool,
    /// Response to non-finite gradients.
    pub gradient_guard: GradientGuardStrategy,
    /// Optional seed for reproducible shuffling.
    pub seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
            max_epochs: 1000,
            target_error: 0.001,
            batch_size: 32,
            shuffle: true,
            gradient_guard: GradientGuardStrategy::Error,
            seed: None,
        }
    }
}

impl TrainingConfig {
    /// Creates the default training configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Sets momentum.
    pub fn momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }

    /// Sets weight decay.
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Sets maximum epochs.
    pub fn max_epochs(mut self, epochs: usize) -> Self {
        self.max_epochs = epochs;
        self
    }

    /// Sets the target error.
    pub fn target_error(mut self, error: f32) -> Self {
        self.target_error = error;
        self
    }

    /// Sets the batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Sets epoch shuffling.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets the gradient guard strategy.
    pub fn gradient_guard(mut self, strategy: GradientGuardStrategy) -> Self {
        self.gradient_guard = strategy;
        self
    }

    /// Sets the shuffle seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Training metrics.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrainingResult {
    /// Final mean squared error.
    pub final_error: f32,
    /// Completed epochs.
    pub epochs_trained: usize,
    /// Error after each epoch.
    pub error_history: Vec<f32>,
    /// Whether the target error was reached.
    pub converged: bool,
}

/// Training algorithm.
pub trait Trainer {
    /// Trains `network` on input-target pairs.
    fn train(
        &mut self,
        network: &mut Network,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        config: &TrainingConfig,
    ) -> FannResult<TrainingResult>;
}

#[cfg(all(test, feature = "online-router"))]
mod online_router_reachability {
    use super::{DiagonalFisher, RlooConfig, RlooTrainer};

    #[test]
    fn online_router_types_are_reachable() {
        let _: Option<DiagonalFisher> = None;
        let _config = RlooConfig::default();
        let _trainer = RlooTrainer::new(RlooConfig::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.batch_size > 0);
    }

    #[test]
    fn test_training_config_builder() {
        let config = TrainingConfig::new()
            .learning_rate(0.001)
            .momentum(0.95)
            .batch_size(64)
            .max_epochs(500);

        assert!((config.learning_rate - 0.001).abs() < 1e-6);
        assert!((config.momentum - 0.95).abs() < 1e-6);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.max_epochs, 500);
    }

    #[test]
    fn test_gradient_guard_strategy_builder() {
        let config = TrainingConfig::new().gradient_guard(GradientGuardStrategy::Sanitize);
        assert_eq!(config.gradient_guard, GradientGuardStrategy::Sanitize);

        let config = TrainingConfig::new().gradient_guard(GradientGuardStrategy::SkipBatch);
        assert_eq!(config.gradient_guard, GradientGuardStrategy::SkipBatch);

        let default_config = TrainingConfig::default();
        assert_eq!(default_config.gradient_guard, GradientGuardStrategy::Error);
    }
}
