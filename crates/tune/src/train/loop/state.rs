//! Training state

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Training state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainingState {
    /// Current epoch
    pub epoch: usize,

    /// Current step within epoch
    pub step: usize,

    /// Global step (across all epochs)
    pub global_step: usize,

    /// Current learning rate
    pub learning_rate: f32,

    /// Running loss for current epoch
    pub running_loss: f32,

    /// Running accuracy for current epoch
    pub running_correct: usize,

    /// Running total examples
    pub running_total: usize,

    /// Best metric value seen (for early stopping)
    pub best_metric: f32,

    /// Epochs without improvement (for early stopping)
    pub patience_counter: usize,
}

impl TrainingState {
    /// Create initial state
    pub fn new(base_lr: f32) -> Self {
        Self {
            epoch: 0,
            step: 0,
            global_step: 0,
            learning_rate: base_lr,
            running_loss: 0.0,
            running_correct: 0,
            running_total: 0,
            best_metric: f32::INFINITY,
            patience_counter: 0,
        }
    }

    /// Reset for new epoch
    pub fn reset_epoch(&mut self) {
        self.step = 0;
        self.running_loss = 0.0;
        self.running_correct = 0;
        self.running_total = 0;
    }

    /// Calculate current epoch accuracy
    pub fn epoch_accuracy(&self) -> f32 {
        if self.running_total == 0 {
            return 0.0;
        }
        self.running_correct as f32 / self.running_total as f32
    }

    /// Calculate current epoch loss
    pub fn epoch_loss(&self) -> f32 {
        if self.step == 0 {
            return 0.0;
        }
        self.running_loss / self.step as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_state() {
        let mut state = TrainingState::new(0.001);
        state.running_correct = 80;
        state.running_total = 100;
        state.running_loss = 0.5;
        state.step = 10;

        assert_eq!(state.epoch_accuracy(), 0.8);
        assert_eq!(state.epoch_loss(), 0.05);

        state.reset_epoch();
        assert_eq!(state.running_total, 0);
    }
}
