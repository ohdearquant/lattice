//! Early stopping configuration

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Early stopping configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EarlyStopping {
    /// Metric to monitor ("loss" or "accuracy")
    pub monitor: String,

    /// Number of epochs with no improvement before stopping
    pub patience: usize,

    /// Minimum change to qualify as improvement
    pub min_delta: f32,

    /// Whether higher values are better (for accuracy)
    pub mode_max: bool,
}

impl Default for EarlyStopping {
    fn default() -> Self {
        Self {
            monitor: "val_loss".to_string(),
            patience: 10,
            min_delta: 1e-4,
            mode_max: false,
        }
    }
}

impl EarlyStopping {
    /// Monitor validation loss (minimize)
    pub fn val_loss(patience: usize) -> Self {
        Self {
            monitor: "val_loss".to_string(),
            patience,
            min_delta: 1e-4,
            mode_max: false,
        }
    }

    /// Monitor validation accuracy (maximize)
    pub fn val_accuracy(patience: usize) -> Self {
        Self {
            monitor: "val_accuracy".to_string(),
            patience,
            min_delta: 1e-4,
            mode_max: true,
        }
    }

    /// Check if metric improved
    pub fn is_improvement(&self, current: f32, best: f32) -> bool {
        if self.mode_max {
            current > best + self.min_delta
        } else {
            current < best - self.min_delta
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping() {
        let es = EarlyStopping::val_loss(10);
        assert!(!es.mode_max);
        assert!(es.is_improvement(0.5, 0.6)); // Lower is better

        let es_acc = EarlyStopping::val_accuracy(10);
        assert!(es_acc.mode_max);
        assert!(es_acc.is_improvement(0.9, 0.8)); // Higher is better
    }
}
