//! JIT (Just-In-Time) model adaptation
//!
//! Fast fine-tuning infrastructure for rapid model adaptation.
//! Enables 5-10 second adaptation cycles with GPU acceleration.

use crate::data::{Dataset, TrainingExample};
use crate::error::{Result, TuneError};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "gpu")]
use super::GpuTrainer;

/// JIT adaptation strategy
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum JitStrategy {
    /// Fine-tune only the last N layers
    LastNLayers(usize),
    /// Fine-tune only the classification head
    HeadOnly,
    /// Fine-tune all layers with lower learning rate for earlier layers
    GradualUnfreeze,
    /// Low-rank adaptation (LoRA-style)
    LowRank {
        /// Rank of the adaptation matrices
        rank: usize,
    },
    /// Full fine-tuning (all parameters)
    Full,
}

impl Default for JitStrategy {
    fn default() -> Self {
        Self::LastNLayers(2)
    }
}

/// Configuration for JIT adaptation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JitConfig {
    /// Adaptation strategy
    pub strategy: JitStrategy,

    /// Number of adaptation steps (iterations)
    pub steps: usize,

    /// Learning rate for adaptation
    pub learning_rate: f32,

    /// Learning rate multiplier for frozen layers (GradualUnfreeze)
    pub frozen_lr_multiplier: f32,

    /// Batch size for adaptation
    pub batch_size: usize,

    /// Whether to use GPU if available
    pub use_gpu: bool,

    /// Maximum adaptation time in seconds
    pub timeout_secs: f32,

    /// Early stop if loss doesn't improve for N steps
    pub patience: usize,

    /// Minimum improvement to reset patience
    pub min_delta: f32,

    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            strategy: JitStrategy::default(),
            steps: 100,
            learning_rate: 0.001,
            frozen_lr_multiplier: 0.1,
            batch_size: 16,
            use_gpu: true,
            timeout_secs: 10.0,
            patience: 10,
            min_delta: 1e-4,
            seed: Some(42),
        }
    }
}

impl JitConfig {
    /// Create a fast adaptation config (5 seconds target)
    pub fn fast() -> Self {
        Self {
            strategy: JitStrategy::HeadOnly,
            steps: 50,
            learning_rate: 0.01,
            batch_size: 32,
            timeout_secs: 5.0,
            patience: 5,
            ..Default::default()
        }
    }

    /// Create a thorough adaptation config (10 seconds target)
    pub fn thorough() -> Self {
        Self {
            strategy: JitStrategy::LastNLayers(3),
            steps: 200,
            learning_rate: 0.001,
            batch_size: 16,
            timeout_secs: 10.0,
            patience: 20,
            ..Default::default()
        }
    }

    /// Create config for very small datasets (<10 examples)
    pub fn few_shot() -> Self {
        Self {
            strategy: JitStrategy::HeadOnly,
            steps: 30,
            learning_rate: 0.005,
            batch_size: 4,
            timeout_secs: 3.0,
            patience: 5,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.steps == 0 {
            return Err(TuneError::InvalidConfig("steps must be > 0".into()));
        }
        if self.learning_rate <= 0.0 {
            return Err(TuneError::InvalidConfig("learning_rate must be > 0".into()));
        }
        if self.batch_size == 0 {
            return Err(TuneError::InvalidConfig("batch_size must be > 0".into()));
        }
        if self.timeout_secs <= 0.0 {
            return Err(TuneError::InvalidConfig("timeout_secs must be > 0".into()));
        }
        Ok(())
    }
}

/// Result of JIT adaptation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JitResult {
    /// Final loss after adaptation
    pub final_loss: f32,

    /// Final accuracy (if validation available)
    pub final_accuracy: Option<f32>,

    /// Number of steps completed
    pub steps_completed: usize,

    /// Actual adaptation time in seconds
    pub time_secs: f32,

    /// Whether adaptation was stopped early
    pub early_stopped: bool,

    /// Whether timeout was reached
    pub timed_out: bool,

    /// Loss history
    pub loss_history: Vec<f32>,
}

impl JitResult {
    /// Check if adaptation was successful
    pub fn is_success(&self) -> bool {
        !self.timed_out && self.final_loss.is_finite()
    }

    /// Get average loss from history
    pub fn avg_loss(&self) -> f32 {
        if self.loss_history.is_empty() {
            return 0.0;
        }
        self.loss_history.iter().sum::<f32>() / self.loss_history.len() as f32
    }
}

/// JIT model adapter for fast fine-tuning
///
/// Provides rapid adaptation of pre-trained models to new tasks/domains.
pub struct JitAdapter {
    /// Configuration
    config: JitConfig,
}

impl JitAdapter {
    /// Create a new JIT adapter with default config
    pub fn new() -> Self {
        Self {
            config: JitConfig::default(),
        }
    }

    /// Create with specific configuration
    pub fn with_config(config: JitConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Get the configuration
    pub fn config(&self) -> &JitConfig {
        &self.config
    }

    /// Adapt a model using CPU training
    pub fn adapt_cpu(&self, examples: Vec<TrainingExample>) -> Result<JitResult> {
        if examples.is_empty() {
            return Err(TuneError::Dataset("No examples provided".into()));
        }

        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs_f32(self.config.timeout_secs);

        let mut dataset = Dataset::from_examples(examples);
        dataset.set_config(
            crate::data::DatasetConfig::with_batch_size(self.config.batch_size)
                .shuffle(true)
                .seed(self.config.seed.unwrap_or(42)),
        )?;

        let mut loss_history = Vec::with_capacity(self.config.steps);
        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;
        let mut steps_completed = 0;
        let mut early_stopped = false;
        let mut timed_out = false;

        // Training loop
        for step in 0..self.config.steps {
            // Check timeout
            if start.elapsed() > timeout {
                timed_out = true;
                break;
            }

            // Simulate training step (placeholder)
            let loss = self.train_step_cpu(&mut dataset, step)?;
            loss_history.push(loss);
            steps_completed = step + 1;

            // Early stopping check
            if loss < best_loss - self.config.min_delta {
                best_loss = loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    early_stopped = true;
                    break;
                }
            }
        }

        let final_loss = loss_history.last().copied().unwrap_or(f32::INFINITY);

        Ok(JitResult {
            final_loss,
            final_accuracy: None,
            steps_completed,
            time_secs: start.elapsed().as_secs_f32(),
            early_stopped,
            timed_out,
            loss_history,
        })
    }

    /// Adapt a model using GPU training (when feature enabled)
    #[cfg(feature = "gpu")]
    pub fn adapt_gpu(
        &self,
        trainer: &mut GpuTrainer,
        examples: Vec<TrainingExample>,
    ) -> Result<JitResult> {
        if examples.is_empty() {
            return Err(TuneError::Dataset("No examples provided".into()));
        }

        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs_f32(self.config.timeout_secs);

        let mut dataset = Dataset::from_examples(examples);
        dataset.set_config(
            crate::data::DatasetConfig::with_batch_size(self.config.batch_size)
                .shuffle(true)
                .seed(self.config.seed.unwrap_or(42)),
        )?;

        let mut loss_history = Vec::with_capacity(self.config.steps);
        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;
        let mut steps_completed = 0;
        let mut early_stopped = false;
        let mut timed_out = false;

        // Training loop with GPU
        for step in 0..self.config.steps {
            // Check timeout
            if start.elapsed() > timeout {
                timed_out = true;
                break;
            }

            // Get next batch
            let batch = match dataset.next_batch() {
                Some(b) => b,
                None => {
                    dataset.reset_epoch();
                    dataset
                        .next_batch()
                        .ok_or_else(|| TuneError::Training("Dataset is empty".into()))?
                }
            };

            // GPU training step
            let loss = trainer.train_batch(&batch)?;
            loss_history.push(loss);
            steps_completed = step + 1;

            // Early stopping check
            if loss < best_loss - self.config.min_delta {
                best_loss = loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    early_stopped = true;
                    break;
                }
            }
        }

        let final_loss = loss_history.last().copied().unwrap_or(f32::INFINITY);

        // Optionally validate
        let final_accuracy = if !dataset.is_empty() {
            let (_, acc) = trainer.validate(&mut dataset)?;
            Some(acc)
        } else {
            None
        };

        Ok(JitResult {
            final_loss,
            final_accuracy,
            steps_completed,
            time_secs: start.elapsed().as_secs_f32(),
            early_stopped,
            timed_out,
            loss_history,
        })
    }

    /// Placeholder CPU training step
    fn train_step_cpu(&self, dataset: &mut Dataset, step: usize) -> Result<f32> {
        // Get batch (cycle through dataset)
        let _batch = match dataset.next_batch() {
            Some(b) => b,
            None => {
                dataset.reset_epoch();
                dataset
                    .next_batch()
                    .ok_or_else(|| TuneError::Training("Dataset is empty".into()))?
            }
        };

        // Simulate decreasing loss
        let base_loss = 1.0 / (1.0 + step as f32 * 0.1);
        let noise = (step as f32 * 0.1).sin() * 0.1;
        Ok(base_loss + noise.abs())
    }
}

impl Default for JitAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Layer freezing utilities
pub mod freeze {
    /// Determine which layers to freeze based on strategy
    pub fn get_frozen_layers(strategy: &super::JitStrategy, total_layers: usize) -> Vec<bool> {
        match strategy {
            super::JitStrategy::LastNLayers(n) => {
                let n = (*n).min(total_layers);
                let frozen_count = total_layers.saturating_sub(n);
                let mut frozen = vec![true; total_layers];
                for item in frozen.iter_mut().skip(frozen_count) {
                    *item = false;
                }
                frozen
            }
            super::JitStrategy::HeadOnly => {
                let mut frozen = vec![true; total_layers];
                if total_layers > 0 {
                    frozen[total_layers - 1] = false;
                }
                frozen
            }
            super::JitStrategy::GradualUnfreeze => {
                // Start with all layers trainable but different LRs
                vec![false; total_layers]
            }
            super::JitStrategy::LowRank { .. } => {
                // In LoRA, original weights are frozen
                vec![true; total_layers]
            }
            super::JitStrategy::Full => {
                vec![false; total_layers]
            }
        }
    }

    /// Get learning rate multiplier for each layer
    pub fn get_lr_multipliers(
        strategy: &super::JitStrategy,
        total_layers: usize,
        frozen_mult: f32,
    ) -> Vec<f32> {
        match strategy {
            super::JitStrategy::GradualUnfreeze => {
                // Linear increase from frozen_mult to 1.0
                (0..total_layers)
                    .map(|i| {
                        let progress = i as f32 / (total_layers - 1).max(1) as f32;
                        frozen_mult + (1.0 - frozen_mult) * progress
                    })
                    .collect()
            }
            _ => {
                let frozen = get_frozen_layers(strategy, total_layers);
                frozen
                    .iter()
                    .map(|&f| if f { frozen_mult } else { 1.0 })
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::IntentLabels;

    fn make_examples(count: usize) -> Vec<TrainingExample> {
        (0..count)
            .map(|_| {
                TrainingExample::new(
                    vec![vec![0.1, 0.2, 0.3]],
                    vec![0.4, 0.5, 0.6],
                    IntentLabels::continuation(0.8),
                )
            })
            .collect()
    }

    #[test]
    fn test_jit_config_defaults() {
        let config = JitConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.steps, 100);
    }

    #[test]
    fn test_jit_config_presets() {
        let fast = JitConfig::fast();
        assert!(fast.validate().is_ok());
        assert!(fast.timeout_secs <= 5.0);

        let thorough = JitConfig::thorough();
        assert!(thorough.validate().is_ok());
        assert!(thorough.steps > fast.steps);
    }

    #[test]
    fn test_jit_adapter_creation() {
        let adapter = JitAdapter::new();
        assert_eq!(adapter.config().steps, 100);

        let adapter = JitAdapter::with_config(JitConfig::fast());
        assert!(adapter.is_ok());
    }

    #[test]
    fn test_jit_adapt_cpu() {
        let adapter = JitAdapter::with_config(JitConfig {
            steps: 10,
            timeout_secs: 1.0,
            ..JitConfig::fast()
        })
        .unwrap();

        let examples = make_examples(20);
        let result = adapter.adapt_cpu(examples);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_success());
        assert!(result.steps_completed > 0);
        assert!(result.time_secs < 1.0);
    }

    #[test]
    fn test_jit_result_metrics() {
        let result = JitResult {
            final_loss: 0.5,
            final_accuracy: Some(0.85),
            steps_completed: 50,
            time_secs: 3.5,
            early_stopped: true,
            timed_out: false,
            loss_history: vec![1.0, 0.8, 0.6, 0.5],
        };

        assert!(result.is_success());
        assert!((result.avg_loss() - 0.725).abs() < 0.01);
    }

    #[test]
    fn test_freeze_last_n_layers() {
        let frozen = freeze::get_frozen_layers(&JitStrategy::LastNLayers(2), 5);
        assert_eq!(frozen, vec![true, true, true, false, false]);
    }

    #[test]
    fn test_freeze_head_only() {
        let frozen = freeze::get_frozen_layers(&JitStrategy::HeadOnly, 4);
        assert_eq!(frozen, vec![true, true, true, false]);
    }

    #[test]
    fn test_freeze_full() {
        let frozen = freeze::get_frozen_layers(&JitStrategy::Full, 3);
        assert_eq!(frozen, vec![false, false, false]);
    }

    #[test]
    fn test_lr_multipliers_gradual() {
        let multipliers = freeze::get_lr_multipliers(&JitStrategy::GradualUnfreeze, 4, 0.1);
        assert_eq!(multipliers.len(), 4);
        assert!((multipliers[0] - 0.1).abs() < 0.01); // First layer
        assert!((multipliers[3] - 1.0).abs() < 0.01); // Last layer
    }
}
