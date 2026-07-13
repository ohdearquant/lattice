//! Checkpoint management

use super::metrics::TrainingMetrics;
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Checkpoint of training state.
///
/// # Serialization format (load-bearing)
///
/// With the `serde` feature, checkpoints serialize to JSON. See
/// [`docs/design.md`](https://github.com/ohdearquant/lattice/blob/main/crates/tune/docs/design.md)
/// for the full JSON schema, a loading example, and the file naming convention. Two fields
/// carry a byte-level contract that is not visible from their `Vec<u8>` type:
///
/// - `weights`: little-endian `f32` values, concatenated layer-by-layer (each layer's weight
///   matrix in row-major order, then its biases; input-to-output order across layers).
/// - `optimizer_state`: optimizer-specific — SGD-with-momentum stores one velocity vector per
///   parameter; Adam stores first (`m`) and second (`v`) moment vectors per parameter.
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
