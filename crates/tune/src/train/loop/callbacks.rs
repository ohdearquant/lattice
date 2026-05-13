//! Training callbacks

use super::checkpoint::Checkpoint;
use super::metrics::{EpochMetrics, TrainingMetrics};
use crate::train::config::TrainingConfig;

/// Training callback trait
pub trait TrainingCallback: Send + Sync {
    /// Called at the start of training
    fn on_train_start(&mut self, _config: &TrainingConfig) {}

    /// Called at the end of training
    fn on_train_end(&mut self, _metrics: &TrainingMetrics) {}

    /// Called at the start of each epoch
    fn on_epoch_start(&mut self, _epoch: usize) {}

    /// Called at the end of each epoch
    fn on_epoch_end(&mut self, _epoch: usize, _metrics: &EpochMetrics) {}

    /// Called at the start of each batch
    fn on_batch_start(&mut self, _batch_idx: usize) {}

    /// Called at the end of each batch
    fn on_batch_end(&mut self, _batch_idx: usize, _loss: f32) {}

    /// Called when a checkpoint is saved
    fn on_checkpoint(&mut self, _checkpoint: &Checkpoint) {}
}

/// No-op callback implementation
#[derive(Default)]
pub struct NoOpCallback;

impl TrainingCallback for NoOpCallback {}

/// Logging callback that prints progress
pub struct LoggingCallback {
    log_interval: usize,
}

impl LoggingCallback {
    /// Create a new logging callback
    ///
    /// # Arguments
    /// * `log_interval` - Log progress every N batches
    pub fn new(log_interval: usize) -> Self {
        Self { log_interval }
    }
}

impl TrainingCallback for LoggingCallback {
    fn on_epoch_end(&mut self, epoch: usize, metrics: &EpochMetrics) {
        let val_info = metrics
            .val_loss
            .map(|v| format!(", val_loss: {v:.4}"))
            .unwrap_or_default();

        println!(
            "Epoch {}: train_loss: {:.4}, train_acc: {:.2}%{} [{:.1}s]",
            epoch,
            metrics.train_loss,
            metrics.train_accuracy * 100.0,
            val_info,
            metrics.duration_secs
        );
    }

    fn on_batch_end(&mut self, batch_idx: usize, loss: f32) {
        if batch_idx % self.log_interval == 0 {
            println!("  batch {batch_idx}: loss {loss:.4}");
        }
    }
}
