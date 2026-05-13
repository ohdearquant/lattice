//! Training metrics

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Metrics for a single epoch
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EpochMetrics {
    /// Epoch number (0-indexed)
    pub epoch: usize,

    /// Training loss
    pub train_loss: f32,

    /// Validation loss (if validation enabled)
    pub val_loss: Option<f32>,

    /// Training accuracy
    pub train_accuracy: f32,

    /// Validation accuracy (if validation enabled)
    pub val_accuracy: Option<f32>,

    /// Learning rate at end of epoch
    pub learning_rate: f32,

    /// Duration in seconds
    pub duration_secs: f32,

    /// Number of training examples seen
    pub examples_seen: usize,
}

impl EpochMetrics {
    /// Create new metrics
    pub fn new(epoch: usize) -> Self {
        Self {
            epoch,
            train_loss: 0.0,
            val_loss: None,
            train_accuracy: 0.0,
            val_accuracy: None,
            learning_rate: 0.0,
            duration_secs: 0.0,
            examples_seen: 0,
        }
    }

    /// Check if validation metrics are available
    pub fn has_validation(&self) -> bool {
        self.val_loss.is_some()
    }

    /// Get the primary metric for early stopping
    pub fn get_metric(&self, name: &str) -> Option<f32> {
        match name {
            "train_loss" => Some(self.train_loss),
            "val_loss" => self.val_loss,
            "train_accuracy" => Some(self.train_accuracy),
            "val_accuracy" => self.val_accuracy,
            _ => None,
        }
    }
}

/// Overall training metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainingMetrics {
    /// Per-epoch metrics
    pub history: Vec<EpochMetrics>,

    /// Best validation loss achieved
    pub best_val_loss: Option<f32>,

    /// Epoch with best validation loss
    pub best_epoch: Option<usize>,

    /// Final training loss
    pub final_train_loss: f32,

    /// Final validation loss
    pub final_val_loss: Option<f32>,

    /// Total training time in seconds
    pub total_time_secs: f32,

    /// Whether training was stopped early
    pub early_stopped: bool,

    /// Total epochs completed
    pub epochs_completed: usize,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            best_val_loss: None,
            best_epoch: None,
            final_train_loss: 0.0,
            final_val_loss: None,
            total_time_secs: 0.0,
            early_stopped: false,
            epochs_completed: 0,
        }
    }
}

impl TrainingMetrics {
    /// Add epoch metrics
    pub fn add_epoch(&mut self, metrics: EpochMetrics) {
        // Update best validation loss
        if let Some(val_loss) = metrics.val_loss {
            if self.best_val_loss.is_none_or(|best| val_loss < best) {
                self.best_val_loss = Some(val_loss);
                self.best_epoch = Some(metrics.epoch);
            }
        }

        self.final_train_loss = metrics.train_loss;
        self.final_val_loss = metrics.val_loss;
        self.total_time_secs += metrics.duration_secs;
        self.epochs_completed = metrics.epoch + 1;

        self.history.push(metrics);
    }

    /// Get training loss history
    pub fn train_loss_history(&self) -> Vec<f32> {
        self.history.iter().map(|m| m.train_loss).collect()
    }

    /// Get validation loss history
    pub fn val_loss_history(&self) -> Vec<f32> {
        self.history.iter().filter_map(|m| m.val_loss).collect()
    }

    /// Check if model is overfitting (val loss increasing while train loss decreasing)
    pub fn is_overfitting(&self, window: usize) -> bool {
        if self.history.len() < window * 2 {
            return false;
        }

        let recent = &self.history[self.history.len() - window..];
        let earlier = &self.history[self.history.len() - window * 2..self.history.len() - window];

        let recent_train: f32 = recent.iter().map(|m| m.train_loss).sum::<f32>() / window as f32;
        let earlier_train: f32 = earlier.iter().map(|m| m.train_loss).sum::<f32>() / window as f32;

        let recent_val: f32 = recent.iter().filter_map(|m| m.val_loss).sum::<f32>() / window as f32;
        let earlier_val: f32 =
            earlier.iter().filter_map(|m| m.val_loss).sum::<f32>() / window as f32;

        // Train loss decreasing but val loss increasing
        recent_train < earlier_train && recent_val > earlier_val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_metrics() {
        let mut metrics = EpochMetrics::new(0);
        metrics.train_loss = 0.5;
        metrics.val_loss = Some(0.6);
        metrics.train_accuracy = 0.8;

        assert!(metrics.has_validation());
        assert_eq!(metrics.get_metric("train_loss"), Some(0.5));
        assert_eq!(metrics.get_metric("val_loss"), Some(0.6));
    }

    #[test]
    fn test_training_metrics_history() {
        let mut metrics = TrainingMetrics::default();

        for i in 0..5 {
            let mut epoch_metrics = EpochMetrics::new(i);
            epoch_metrics.train_loss = 1.0 - 0.1 * i as f32;
            epoch_metrics.val_loss = Some(1.1 - 0.1 * i as f32);
            metrics.add_epoch(epoch_metrics);
        }

        assert_eq!(metrics.history.len(), 5);
        assert_eq!(metrics.best_epoch, Some(4));
        assert_eq!(metrics.epochs_completed, 5);
    }
}
