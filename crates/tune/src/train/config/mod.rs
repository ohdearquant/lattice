//! Training configuration
//!
//! This module provides configuration for training loops, optimizers,
//! learning rate schedules, and regularization.

mod early_stopping;
mod lr_schedule;
mod optimizer;
mod regularization;

pub use early_stopping::EarlyStopping;
pub use lr_schedule::LRSchedule;
pub use optimizer::{Optimizer, OptimizerConfig};
pub use regularization::RegularizationConfig;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Result, TuneError};

/// Maximum allowed batch size to prevent excessive memory allocation.
/// 8192 is large enough for most practical scenarios while preventing OOM.
pub const MAX_BATCH_SIZE: usize = 8192;

/// Training configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,

    /// Batch size
    pub batch_size: usize,

    /// Optimizer configuration
    pub optimizer: OptimizerConfig,

    /// Learning rate schedule
    pub lr_schedule: LRSchedule,

    /// Regularization configuration
    pub regularization: RegularizationConfig,

    /// Early stopping (None = train for all epochs)
    pub early_stopping: Option<EarlyStopping>,

    /// Validation split ratio (0.0 = no validation)
    pub val_split: f32,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Checkpoint directory
    pub checkpoint_dir: Option<String>,

    /// Save checkpoint every N epochs
    pub checkpoint_interval: usize,

    /// Log metrics every N steps
    pub log_interval: usize,

    /// Use gradient accumulation (effective batch size = batch_size * accumulation_steps)
    pub accumulation_steps: usize,

    /// Enable mixed precision training
    pub mixed_precision: bool,

    /// Memory budget in MB (None = no limit)
    ///
    /// When set, training will fail with `TuneError::MemoryBudgetExceeded`
    /// if estimated memory usage exceeds this budget.
    pub memory_budget_mb: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            optimizer: OptimizerConfig::default(),
            lr_schedule: LRSchedule::default(),
            regularization: RegularizationConfig::default(),
            early_stopping: Some(EarlyStopping::default()),
            val_split: 0.1,
            seed: Some(42),
            checkpoint_dir: None,
            checkpoint_interval: 10,
            log_interval: 100,
            accumulation_steps: 1,
            mixed_precision: false,
            memory_budget_mb: None,
        }
    }
}

impl TrainingConfig {
    /// Create a quick training config (fewer epochs, no validation)
    pub fn quick() -> Self {
        Self {
            epochs: 10,
            batch_size: 64,
            val_split: 0.0,
            early_stopping: None,
            checkpoint_interval: 5,
            ..Default::default()
        }
    }

    /// Create a thorough training config
    pub fn thorough() -> Self {
        Self {
            epochs: 200,
            batch_size: 32,
            optimizer: OptimizerConfig::adamw(0.0001, 0.01),
            lr_schedule: LRSchedule::CosineAnnealingWarmup {
                warmup_steps: 500,
                min_lr: 1e-7,
                t_max: 200,
            },
            regularization: RegularizationConfig::strong(),
            early_stopping: Some(EarlyStopping::val_loss(20)),
            val_split: 0.15,
            seed: Some(42),
            checkpoint_dir: None,
            checkpoint_interval: 5,
            log_interval: 50,
            accumulation_steps: 1,
            mixed_precision: true,
            memory_budget_mb: None,
        }
    }

    /// Set number of epochs
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.optimizer.learning_rate = lr;
        self
    }

    /// Set optimizer configuration
    pub fn optimizer(mut self, optimizer: OptimizerConfig) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set learning rate schedule
    pub fn lr_schedule(mut self, schedule: LRSchedule) -> Self {
        self.lr_schedule = schedule;
        self
    }

    /// Set regularization configuration
    pub fn regularization(mut self, reg: RegularizationConfig) -> Self {
        self.regularization = reg;
        self
    }

    /// Set early stopping configuration
    pub fn early_stopping(mut self, es: EarlyStopping) -> Self {
        self.early_stopping = Some(es);
        self
    }

    /// Disable early stopping
    pub fn no_early_stopping(mut self) -> Self {
        self.early_stopping = None;
        self
    }

    /// Set validation split ratio
    pub fn val_split(mut self, split: f32) -> Self {
        self.val_split = split;
        self
    }

    /// Set random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set checkpoint directory
    pub fn checkpoint_dir(mut self, dir: impl Into<String>) -> Self {
        self.checkpoint_dir = Some(dir.into());
        self
    }

    /// Set memory budget in MB
    ///
    /// Training will fail with `TuneError::MemoryBudgetExceeded` if
    /// estimated memory usage exceeds this budget.
    pub fn memory_budget_mb(mut self, budget_mb: usize) -> Self {
        self.memory_budget_mb = Some(budget_mb);
        self
    }

    /// Check if estimated memory usage exceeds the budget
    ///
    /// Returns `TuneError::MemoryBudgetExceeded` if budget is set and exceeded.
    ///
    /// # Arguments
    /// * `required_mb` - Estimated memory requirement in MB
    pub fn check_memory_budget(&self, required_mb: usize) -> Result<()> {
        if let Some(budget_mb) = self.memory_budget_mb {
            if required_mb > budget_mb {
                return Err(TuneError::MemoryBudgetExceeded {
                    required_mb,
                    budget_mb,
                });
            }
        }
        Ok(())
    }

    /// Estimate memory usage for training with given parameters
    ///
    /// Returns estimated memory in MB.
    ///
    /// # Arguments
    /// * `model_params` - Number of model parameters
    /// * `embedding_dim` - Embedding dimension
    pub fn estimate_memory_mb(&self, model_params: usize, embedding_dim: usize) -> Result<usize> {
        let model_size_bytes = model_params.checked_mul(4).ok_or_else(|| {
            TuneError::InvalidConfig("model_params overflow in memory estimate".into())
        })?;

        let gradient_size_bytes = model_size_bytes;

        let optimizer_state_bytes = match self.optimizer.optimizer {
            Optimizer::Adam | Optimizer::AdamW => model_params.checked_mul(8),
            Optimizer::RMSprop | Optimizer::SGDMomentum => model_params.checked_mul(4),
            Optimizer::SGD => Some(0),
        }
        .ok_or_else(|| {
            TuneError::InvalidConfig("optimizer state overflow in memory estimate".into())
        })?;

        let batch_data_bytes = self
            .batch_size
            .checked_mul(embedding_dim)
            .and_then(|v| v.checked_mul(8))
            .ok_or_else(|| {
                TuneError::InvalidConfig("batch data overflow in memory estimate".into())
            })?;

        let activation_bytes = batch_data_bytes.checked_mul(2).ok_or_else(|| {
            TuneError::InvalidConfig("activation overflow in memory estimate".into())
        })?;

        let total_bytes = model_size_bytes
            .checked_add(gradient_size_bytes)
            .and_then(|v| v.checked_add(optimizer_state_bytes))
            .and_then(|v| v.checked_add(batch_data_bytes))
            .and_then(|v| v.checked_add(activation_bytes))
            .ok_or_else(|| TuneError::InvalidConfig("total overflow in memory estimate".into()))?;

        Ok((total_bytes as f64 * 1.2 / (1024.0 * 1024.0)).ceil() as usize)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.epochs == 0 {
            return Err(TuneError::InvalidConfig("epochs must be > 0".to_string()));
        }
        if self.batch_size == 0 {
            return Err(TuneError::InvalidConfig(
                "batch_size must be > 0".to_string(),
            ));
        }
        if self.batch_size > MAX_BATCH_SIZE {
            return Err(TuneError::InvalidConfig(format!(
                "batch_size {} exceeds MAX_BATCH_SIZE {}",
                self.batch_size, MAX_BATCH_SIZE
            )));
        }
        if !(0.0..=1.0).contains(&self.val_split) {
            return Err(TuneError::InvalidConfig(format!(
                "val_split must be between 0 and 1, got {}",
                self.val_split
            )));
        }
        if self.accumulation_steps == 0 {
            return Err(TuneError::InvalidConfig(
                "accumulation_steps must be > 0".to_string(),
            ));
        }

        self.optimizer.validate()?;
        self.regularization.validate()?;
        self.lr_schedule.validate()?;

        Ok(())
    }

    /// Get effective batch size (accounting for accumulation)
    pub fn effective_batch_size(&self) -> usize {
        self.batch_size.saturating_mul(self.accumulation_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_defaults() {
        let config = TrainingConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_training_config_builder() {
        let config = TrainingConfig::default()
            .epochs(50)
            .batch_size(64)
            .learning_rate(0.0001)
            .val_split(0.2);

        assert_eq!(config.epochs, 50);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.optimizer.learning_rate, 0.0001);
        assert_eq!(config.val_split, 0.2);
    }

    #[test]
    fn test_training_config_validation() {
        let mut config = TrainingConfig::default();
        assert!(config.validate().is_ok());

        config.epochs = 0;
        assert!(config.validate().is_err());
        config.epochs = 100;

        config.val_split = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_presets() {
        let quick = TrainingConfig::quick();
        assert!(quick.validate().is_ok());
        assert!(quick.epochs < TrainingConfig::default().epochs);

        let thorough = TrainingConfig::thorough();
        assert!(thorough.validate().is_ok());
        assert!(thorough.epochs > TrainingConfig::default().epochs);
    }
}
