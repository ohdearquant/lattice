//! Rollback controller for model version management.
//!
//! Enables safe rollback to previous model versions with history tracking.
//! When a production model causes issues, operators can roll back to a
//! known-good version while maintaining an audit trail.
//!
//! # Example
//!
//! ```
//! use lattice_tune::registry::{RollbackController, RollbackRecord};
//! use uuid::Uuid;
//!
//! let mut controller = RollbackController::new(100);
//!
//! // Record a rollback operation
//! let from_id = Uuid::new_v4();
//! let to_id = Uuid::new_v4();
//! let record = controller.record_rollback(
//!     from_id,
//!     to_id,
//!     "Performance regression detected",
//!     Some("ops-team".to_string()),
//! );
//!
//! // Query history
//! assert_eq!(controller.history().len(), 1);
//! assert_eq!(controller.last_rollback().unwrap().reason, "Performance regression detected");
//! ```

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

/// Controller for model rollback operations.
///
/// Maintains a bounded history of rollback operations for audit and
/// debugging purposes. The history is stored in memory; for persistence,
/// serialize the records externally.
///
/// # Thread Safety
///
/// This type is NOT thread-safe. For concurrent access, wrap in a
/// `Mutex` or `RwLock`.
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
    /// Create a new rollback controller with specified history limit.
    ///
    /// # Arguments
    ///
    /// * `max_history` - Maximum number of rollback records to retain.
    ///   When exceeded, oldest records are removed first.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history,
        }
    }

    /// Record a rollback operation.
    ///
    /// Creates a new [`RollbackRecord`] with the current timestamp and
    /// adds it to the history. If history exceeds `max_history`, the
    /// oldest record is removed.
    ///
    /// # Arguments
    ///
    /// * `from_id` - The model ID being rolled back from (current production)
    /// * `to_id` - The model ID being rolled back to (new production)
    /// * `reason` - Human-readable reason for the rollback
    /// * `initiated_by` - Optional identifier for who/what initiated the rollback
    ///
    /// # Returns
    ///
    /// The created [`RollbackRecord`].
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
mod tests {
    use super::*;

    #[test]
    fn test_rollback_record_creation() {
        let from_id = Uuid::new_v4();
        let to_id = Uuid::new_v4();
        let record = RollbackRecord::new(from_id, to_id, "test reason", Some("tester".to_string()));

        assert_eq!(record.from_model_id, from_id);
        assert_eq!(record.to_model_id, to_id);
        assert_eq!(record.reason, "test reason");
        assert_eq!(record.initiated_by, Some("tester".to_string()));
    }

    #[test]
    fn test_controller_default() {
        let controller = RollbackController::default();
        assert_eq!(controller.max_history(), 100);
        assert!(controller.is_empty());
    }

    #[test]
    fn test_record_rollback() {
        let mut controller = RollbackController::new(10);
        let from_id = Uuid::new_v4();
        let to_id = Uuid::new_v4();

        let record =
            controller.record_rollback(from_id, to_id, "perf regression", Some("ops".to_string()));

        assert_eq!(controller.len(), 1);
        assert_eq!(record.from_model_id, from_id);
        assert_eq!(record.to_model_id, to_id);
        assert_eq!(record.reason, "perf regression");
    }

    #[test]
    fn test_history_trimming() {
        let mut controller = RollbackController::new(3);

        // Add 5 records
        for i in 0..5 {
            let from_id = Uuid::new_v4();
            let to_id = Uuid::new_v4();
            controller.record_rollback(from_id, to_id, format!("reason {i}"), None);
        }

        // Should only have 3 (the most recent)
        assert_eq!(controller.len(), 3);

        // Oldest should be "reason 2" (0 and 1 were trimmed)
        assert_eq!(controller.history()[0].reason, "reason 2");
        assert_eq!(controller.history()[2].reason, "reason 4");
    }

    #[test]
    fn test_last_rollback() {
        let mut controller = RollbackController::new(10);
        assert!(controller.last_rollback().is_none());

        let from_id = Uuid::new_v4();
        let to_id = Uuid::new_v4();
        controller.record_rollback(from_id, to_id, "first", None);

        let from_id2 = Uuid::new_v4();
        let to_id2 = Uuid::new_v4();
        controller.record_rollback(from_id2, to_id2, "second", None);

        assert_eq!(controller.last_rollback().unwrap().reason, "second");
    }

    #[test]
    fn test_rollbacks_involving() {
        let mut controller = RollbackController::new(10);

        let model_a = Uuid::new_v4();
        let model_b = Uuid::new_v4();
        let model_c = Uuid::new_v4();

        // A -> B
        controller.record_rollback(model_a, model_b, "rollback 1", None);
        // B -> C
        controller.record_rollback(model_b, model_c, "rollback 2", None);
        // C -> A
        controller.record_rollback(model_c, model_a, "rollback 3", None);

        // Model A involved in rollback 1 (from) and rollback 3 (to)
        let involving_a = controller.rollbacks_involving(model_a);
        assert_eq!(involving_a.len(), 2);

        // Model B involved in rollback 1 (to) and rollback 2 (from)
        let involving_b = controller.rollbacks_involving(model_b);
        assert_eq!(involving_b.len(), 2);

        // Model C involved in rollback 2 (to) and rollback 3 (from)
        let involving_c = controller.rollbacks_involving(model_c);
        assert_eq!(involving_c.len(), 2);
    }

    #[test]
    fn test_rollbacks_from_and_to() {
        let mut controller = RollbackController::new(10);

        let model_a = Uuid::new_v4();
        let model_b = Uuid::new_v4();
        let model_c = Uuid::new_v4();

        controller.record_rollback(model_a, model_b, "1", None);
        controller.record_rollback(model_a, model_c, "2", None);
        controller.record_rollback(model_b, model_c, "3", None);

        // Model A was source twice
        assert_eq!(controller.rollbacks_from(model_a).len(), 2);
        // Model A was never a target
        assert_eq!(controller.rollbacks_to(model_a).len(), 0);

        // Model C was target twice
        assert_eq!(controller.rollbacks_to(model_c).len(), 2);
        // Model C was never a source
        assert_eq!(controller.rollbacks_from(model_c).len(), 0);
    }

    #[test]
    fn test_clear_history() {
        let mut controller = RollbackController::new(10);

        controller.record_rollback(Uuid::new_v4(), Uuid::new_v4(), "test", None);
        controller.record_rollback(Uuid::new_v4(), Uuid::new_v4(), "test", None);

        assert_eq!(controller.len(), 2);

        controller.clear_history();

        assert!(controller.is_empty());
        assert!(controller.last_rollback().is_none());
    }
}
