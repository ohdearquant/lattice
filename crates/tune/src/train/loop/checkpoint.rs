//! Checkpoint management

use super::metrics::TrainingMetrics;
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Checkpoint of training state.
///
/// # Serialization Format
///
/// When serialized with the `serde` feature, checkpoints use JSON format with the following structure:
///
/// ```json
/// {
///   "id": "550e8400-e29b-41d4-a716-446655440000",
///   "epoch": 10,
///   "global_step": 5000,
///   "metrics": {
///     "final_train_loss": 0.05,
///     "final_val_loss": 0.06,
///     "epochs_completed": 10,
///     "total_steps": 5000,
///     "best_epoch": 8,
///     "best_val_loss": 0.055
///   },
///   "created_at": "2024-01-15T10:30:00Z",
///   "weights": "<base64-encoded bytes>",
///   "optimizer_state": "<base64-encoded bytes>"
/// }
/// ```
///
/// ## Field Details
///
/// | Field | Type | Description |
/// |-------|------|-------------|
/// | `id` | UUID v4 | Unique identifier for this checkpoint |
/// | `epoch` | usize | Training epoch when created (0-indexed) |
/// | `global_step` | usize | Total batches processed |
/// | `metrics` | TrainingMetrics | Training statistics at checkpoint time |
/// | `created_at` | ISO 8601 | UTC timestamp of creation |
/// | `weights` | bytes | Serialized model parameters |
/// | `optimizer_state` | bytes | Serialized optimizer momentum/state |
///
/// ## Weight Serialization
///
/// The `weights` field contains model parameters serialized as:
/// 1. Little-endian f32 values concatenated
/// 2. Layer order: input-to-output (layer 0 weights, layer 0 biases, layer 1...)
/// 3. Weight matrices in row-major order
///
/// For a network with layers [4->8, 8->2], weights layout:
/// ```text
/// [layer0_weights: 32 floats][layer0_biases: 8 floats][layer1_weights: 16 floats][layer1_biases: 2 floats]
/// ```
///
/// ## Optimizer State Serialization
///
/// The `optimizer_state` field contains optimizer-specific state:
///
/// - **SGD with momentum**: Velocity vectors matching weight dimensions
/// - **Adam**: First moment (m) + second moment (v) for each parameter
///
/// ## Loading Checkpoints
///
/// ```ignore
/// use lattice_tune::Checkpoint;
///
/// // Load from JSON file
/// let json = std::fs::read_to_string("checkpoint_epoch_10.json")?;
/// let checkpoint: Checkpoint = serde_json::from_str(&json)?;
///
/// // Restore model weights
/// model.load_weights(&checkpoint.weights);
/// optimizer.load_state(&checkpoint.optimizer_state);
/// ```
///
/// ## Checkpoint Naming Convention
///
/// Recommended file naming: `checkpoint_epoch_{epoch:04d}_step_{step:08d}.json`
///
/// Example: `checkpoint_epoch_0010_step_00005000.json`
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Checkpoint {
    /// Unique checkpoint ID (UUID v4)
    pub id: Uuid,

    /// Epoch when checkpoint was created (0-indexed)
    pub epoch: usize,

    /// Global step (total batches processed)
    pub global_step: usize,

    /// Training metrics up to this point
    pub metrics: TrainingMetrics,

    /// Timestamp (UTC)
    pub created_at: DateTime<Utc>,

    /// Model weights (serialized as little-endian f32 bytes)
    pub weights: Vec<u8>,

    /// Optimizer state (serialized momentum/moment vectors)
    pub optimizer_state: Vec<u8>,
}

impl Checkpoint {
    /// Create a new checkpoint
    pub fn new(epoch: usize, global_step: usize, metrics: TrainingMetrics) -> Self {
        Self {
            id: Uuid::new_v4(),
            epoch,
            global_step,
            metrics,
            created_at: Utc::now(),
            weights: Vec::new(),
            optimizer_state: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint() {
        let metrics = TrainingMetrics::default();
        let checkpoint = Checkpoint::new(5, 500, metrics);

        assert_eq!(checkpoint.epoch, 5);
        assert_eq!(checkpoint.global_step, 500);
    }
}
