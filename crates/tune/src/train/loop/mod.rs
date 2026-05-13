//! Training loop implementation

mod callbacks;
mod checkpoint;
mod metrics;
mod state;

pub use callbacks::{LoggingCallback, NoOpCallback, TrainingCallback};
pub use checkpoint::Checkpoint;
pub use metrics::{EpochMetrics, TrainingMetrics};
pub use state::TrainingState;

use super::config::{EarlyStopping, TrainingConfig};
use crate::data::{Batch, Dataset};
use crate::error::{Result, TuneError};

/// Training loop orchestrator
///
/// This is a placeholder implementation. The actual neural network operations
/// would be implemented via lattice-fann when it's available.
pub struct TrainingLoop {
    /// Training configuration
    config: TrainingConfig,

    /// Current training state
    state: TrainingState,

    /// Training metrics
    metrics: TrainingMetrics,

    /// Callbacks
    callbacks: Vec<Box<dyn TrainingCallback>>,

    /// Early stopping tracker
    early_stopping: Option<EarlyStopping>,
}

impl TrainingLoop {
    /// Create a new training loop
    pub fn new(config: TrainingConfig) -> Result<Self> {
        config.validate()?;

        let state = TrainingState::new(config.optimizer.learning_rate);

        Ok(Self {
            early_stopping: config.early_stopping.clone(),
            config,
            state,
            metrics: TrainingMetrics::default(),
            callbacks: Vec::new(),
        })
    }

    /// Add a callback
    pub fn add_callback(&mut self, callback: Box<dyn TrainingCallback>) {
        self.callbacks.push(callback);
    }

    /// Get the current configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Get current training state
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// Get training metrics
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// Train on a dataset
    ///
    /// This is a placeholder that simulates training. Real implementation
    /// would use lattice-fann for actual neural network operations.
    pub fn train(&mut self, dataset: &mut Dataset) -> Result<TrainingMetrics> {
        if dataset.is_empty() {
            return Err(TuneError::Dataset("Dataset is empty".to_string()));
        }

        // Split into train/val if configured
        let (mut train_data, val_data) = if self.config.val_split > 0.0 {
            let (train, val) = dataset.split(1.0 - self.config.val_split)?;
            (train, Some(val))
        } else {
            (dataset.clone(), None)
        };

        // Configure dataset batching
        train_data.set_config(
            crate::data::DatasetConfig::with_batch_size(self.config.batch_size)
                .shuffle(true)
                .seed(self.config.seed.unwrap_or(42)),
        )?;

        // Notify callbacks
        for cb in &mut self.callbacks {
            cb.on_train_start(&self.config);
        }

        // Training loop
        for epoch in 0..self.config.epochs {
            let start_time = std::time::Instant::now();

            self.state.epoch = epoch;
            self.state.reset_epoch();

            for cb in &mut self.callbacks {
                cb.on_epoch_start(epoch);
            }

            // Train epoch
            self.train_epoch(&mut train_data)?;

            // Validation
            let (val_loss, val_accuracy) = if let Some(ref val) = val_data {
                let (loss, acc) = self.validate(val)?;
                (Some(loss), Some(acc))
            } else {
                (None, None)
            };

            // Create epoch metrics
            let epoch_metrics = EpochMetrics {
                epoch,
                train_loss: self.state.epoch_loss(),
                val_loss,
                train_accuracy: self.state.epoch_accuracy(),
                val_accuracy,
                learning_rate: self.state.learning_rate,
                duration_secs: start_time.elapsed().as_secs_f32(),
                examples_seen: self.state.running_total,
            };

            self.metrics.add_epoch(epoch_metrics.clone());

            // Notify callbacks
            for cb in &mut self.callbacks {
                cb.on_epoch_end(epoch, &epoch_metrics);
            }

            // Check early stopping
            if let Some(ref es) = self.early_stopping {
                if let Some(metric_val) = epoch_metrics.get_metric(&es.monitor) {
                    if es.is_improvement(metric_val, self.state.best_metric) {
                        self.state.best_metric = metric_val;
                        self.state.patience_counter = 0;
                    } else {
                        self.state.patience_counter += 1;
                        if self.state.patience_counter >= es.patience {
                            self.metrics.early_stopped = true;
                            break;
                        }
                    }
                }
            }

            // Update learning rate
            self.state.learning_rate = self.config.lr_schedule.get_lr(
                self.config.optimizer.learning_rate,
                self.state.global_step,
                epoch,
            );

            // Checkpoint
            if self.config.checkpoint_dir.is_some()
                && (epoch + 1) % self.config.checkpoint_interval == 0
            {
                let checkpoint =
                    Checkpoint::new(epoch, self.state.global_step, self.metrics.clone());
                for cb in &mut self.callbacks {
                    cb.on_checkpoint(&checkpoint);
                }
            }
        }

        // Notify callbacks
        for cb in &mut self.callbacks {
            cb.on_train_end(&self.metrics);
        }

        Ok(self.metrics.clone())
    }

    /// Train one epoch
    fn train_epoch(&mut self, dataset: &mut Dataset) -> Result<()> {
        for batch in dataset.batches() {
            for cb in &mut self.callbacks {
                cb.on_batch_start(batch.batch_idx);
            }

            let loss = self.train_batch(&batch)?;

            for cb in &mut self.callbacks {
                cb.on_batch_end(batch.batch_idx, loss);
            }
        }
        Ok(())
    }

    /// Train one batch
    ///
    /// Placeholder: returns simulated loss. Real implementation would:
    /// 1. Forward pass
    /// 2. Compute loss
    /// 3. Backward pass
    /// 4. Update weights
    fn train_batch(&mut self, batch: &Batch) -> Result<f32> {
        self.state.step += 1;
        self.state.global_step += 1;

        // Simulate training
        let batch_size = batch.len();
        self.state.running_total += batch_size;

        // Simulated loss (decreases over training)
        let loss = 0.5 * (-0.01 * self.state.global_step as f32).exp() + 0.1;
        self.state.running_loss += loss;

        // Simulated accuracy (increases over training)
        let accuracy_rate = 0.6 + 0.35 * (1.0 - (-0.005 * self.state.global_step as f32).exp());
        let correct = (batch_size as f32 * accuracy_rate) as usize;
        self.state.running_correct += correct;

        Ok(loss)
    }

    /// Validate on a dataset
    ///
    /// Placeholder: returns simulated validation metrics.
    fn validate(&self, _dataset: &Dataset) -> Result<(f32, f32)> {
        // Simulate validation
        let val_loss = self.state.epoch_loss() * 1.1; // Slightly higher than train
        let val_accuracy = self.state.epoch_accuracy() * 0.95; // Slightly lower than train

        Ok((val_loss, val_accuracy))
    }

    /// Create a checkpoint
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint::new(
            self.state.epoch,
            self.state.global_step,
            self.metrics.clone(),
        )
    }

    /// Resume from checkpoint
    pub fn resume_from(&mut self, checkpoint: &Checkpoint) {
        self.state.epoch = checkpoint.epoch;
        self.state.global_step = checkpoint.global_step;
        self.metrics = checkpoint.metrics.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{IntentLabels, TrainingExample};

    fn make_dataset(size: usize) -> Dataset {
        let examples: Vec<TrainingExample> = (0..size)
            .map(|_| {
                TrainingExample::new(
                    vec![vec![0.1, 0.2, 0.3]],
                    vec![0.4, 0.5, 0.6],
                    IntentLabels::continuation(0.8),
                )
            })
            .collect();
        Dataset::from_examples(examples)
    }

    #[test]
    fn test_training_loop_creation() {
        let config = TrainingConfig::default();
        let trainer = TrainingLoop::new(config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_training_loop_train() {
        let config = TrainingConfig::quick();
        let mut trainer = TrainingLoop::new(config).unwrap();
        let mut dataset = make_dataset(100);

        let metrics = trainer.train(&mut dataset);
        assert!(metrics.is_ok());

        let metrics = metrics.unwrap();
        assert!(metrics.epochs_completed > 0);
        assert!(metrics.final_train_loss > 0.0);
    }

    #[test]
    fn test_training_with_validation() {
        let config = TrainingConfig::quick().val_split(0.2);
        let mut trainer = TrainingLoop::new(config).unwrap();
        let mut dataset = make_dataset(100);

        let metrics = trainer.train(&mut dataset).unwrap();
        assert!(metrics.final_val_loss.is_some());
    }

    #[test]
    fn test_early_stopping_check() {
        let es = EarlyStopping::val_loss(5);

        // Lower is better for loss
        assert!(es.is_improvement(0.4, 0.5));
        assert!(!es.is_improvement(0.5, 0.4));
    }
}
