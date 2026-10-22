//! Bounded in-memory audit records for model rollbacks.
//!
//! Recording a rollback does not update registry status, swap a live model, or
//! persist the audit trail. Applications coordinate those operations explicitly.
//!
//! See `docs/registry.md` for rollback lifecycle and retention semantics.

use chrono::{DateTime, Utc};
use uuid::Uuid;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Record of a rollback operation.
///
/// Captures the complete context of a rollback for audit purposes:
/// what models were involved, why it happened, and who initiated it.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollbackRecord {
    /// Unique ID of this rollback operation
    pub id: Uuid,
    /// Model ID that was in production before rollback
    pub from_model_id: Uuid,
    /// Model ID that became production after rollback
    pub to_model_id: Uuid,
    /// Reason for the rollback (required for audit)
    pub reason: String,
    /// Who or what initiated the rollback
    pub initiated_by: Option<String>,
    /// When the rollback occurred
    pub timestamp: DateTime<Utc>,
}

impl RollbackRecord {
    /// Create a new rollback record.
    pub fn new(
        from_model_id: Uuid,
        to_model_id: Uuid,
        reason: impl Into<String>,
        initiated_by: Option<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            from_model_id,
            to_model_id,
            reason: reason.into(),
            initiated_by,
            timestamp: Utc::now(),
        }
    }
}

/// Bounded in-memory history of rollback operations.
///
/// The controller does not enact a rollback or persist records. Synchronize
/// mutable shared access externally with a `Mutex` or `RwLock`.
pub struct RollbackController {
    /// History of rollback operations (oldest first)
    history: Vec<RollbackRecord>,
    /// Maximum history entries to keep
    max_history: usize,
}

impl Default for RollbackController {
    fn default() -> Self {
        Self::new(100)
    }
}

impl RollbackController {
    /// Creates a controller retaining at most `max_history` rollback records.
    /// See [`docs/registry.md`](../../docs/registry.md#rollbackcontroller-retention) for eviction behavior.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history,
        }
    }

    /// Appends and returns a timestamped rollback audit record.
    /// Evicts the oldest record after exceeding the configured capacity.
    /// See [`docs/registry.md`](../../docs/registry.md#rollbackcontroller-retention) for lifecycle limits.
    pub fn record_rollback(
        &mut self,
        from_id: Uuid,
        to_id: Uuid,
        reason: impl Into<String>,
        initiated_by: Option<String>,
    ) -> RollbackRecord {
        let record = RollbackRecord::new(from_id, to_id, reason, initiated_by);

        self.history.push(record.clone());

        // Trim history if needed (remove oldest first)
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        record
    }

    /// Get the complete rollback history.
    ///
    /// Records are ordered oldest-first.
    pub fn history(&self) -> &[RollbackRecord] {
        &self.history
    }

    /// Get the most recent rollback record.
    pub fn last_rollback(&self) -> Option<&RollbackRecord> {
        self.history.last()
    }

    /// Get rollbacks involving a specific model (either as source or target).
    pub fn rollbacks_involving(&self, model_id: Uuid) -> Vec<&RollbackRecord> {
        self.history
            .iter()
            .filter(|r| r.from_model_id == model_id || r.to_model_id == model_id)
            .collect()
    }

    /// Get rollbacks where a specific model was the source (rolled back from).
    pub fn rollbacks_from(&self, model_id: Uuid) -> Vec<&RollbackRecord> {
        self.history
            .iter()
            .filter(|r| r.from_model_id == model_id)
            .collect()
    }

    /// Get rollbacks where a specific model was the target (rolled back to).
    pub fn rollbacks_to(&self, model_id: Uuid) -> Vec<&RollbackRecord> {
        self.history
            .iter()
            .filter(|r| r.to_model_id == model_id)
            .collect()
    }

    /// Get the number of rollback records.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if there are no rollback records.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Clear all rollback history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get the maximum history size.
    pub fn max_history(&self) -> usize {
        self.max_history
    }
}

#[cfg(test)]
mod tests;
