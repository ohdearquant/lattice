//! Training module for neural networks
//!
//! Provides basic backpropagation training with configurable learning rates
//! and optimization strategies.

mod backprop;
mod gradient;

pub use backprop::BackpropTrainer;
pub use gradient::GradientGuardStrategy;

use crate::error::FannResult;
use crate::network::Network;

/// Configuration for training
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum coefficient (0.0 = no momentum)
    pub momentum: f32,
    /// L2 regularization coefficient (weight decay)
    pub weight_decay: f32,
    /// Maximum number of epochs
    pub max_epochs: usize,
    /// Target error threshold for early stopping
    pub target_error: f32,
    /// Batch size (1 = stochastic gradient descent)
    pub batch_size: usize,
    /// Whether to shuffle training data each epoch
    pub shuffle: bool,
    /// Strategy for handling NaN/Inf gradients
    pub gradient_guard: GradientGuardStrategy,
    /// Optional RNG seed for reproducible shuffling (None = use entropy)
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
    /// Create a new training configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set momentum
    pub fn momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set maximum epochs
    pub fn max_epochs(mut self, epochs: usize) -> Self {
        self.max_epochs = epochs;
        self
    }

    /// Set target error
    pub fn target_error(mut self, error: f32) -> Self {
        self.target_error = error;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set shuffle flag
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set gradient guard strategy
    pub fn gradient_guard(mut self, strategy: GradientGuardStrategy) -> Self {
        self.gradient_guard = strategy;
        self
    }

    /// Set RNG seed for reproducible training
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Training result containing metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrainingResult {
    /// Final mean squared error
    pub final_error: f32,
    /// Number of epochs trained
    pub epochs_trained: usize,
    /// Error history per epoch
    pub error_history: Vec<f32>,
    /// Whether target error was reached
    pub converged: bool,
}

/// Trait for training algorithms
pub trait Trainer {
    /// Train the network on the given data
    ///
    /// # Arguments
    /// * `network` - The network to train
    /// * `inputs` - Training input vectors
    /// * `targets` - Target output vectors
    /// * `config` - Training configuration
    fn train(
        &mut self,
        network: &mut Network,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        config: &TrainingConfig,
    ) -> FannResult<TrainingResult>;
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

        // Default should be Error
        let default_config = TrainingConfig::default();
        assert_eq!(default_config.gradient_guard, GradientGuardStrategy::Error);
    }
}
