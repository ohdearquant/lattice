//! Migration types: states, plans, progress snapshots, and errors.

use serde::{Deserialize, Serialize};

use crate::model::EmbeddingModel;

/// Reason why an embedding was skipped during migration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SkipReason {
    /// Content exceeds maximum size for embedding.
    ContentTooLarge {
        /// Actual content size in bytes.
        size: usize,
        /// Maximum allowed size in bytes.
        max: usize,
    },
    /// Content encoding is invalid or unsupported.
    InvalidEncoding(String),
    /// Content was deleted during migration.
    ContentDeleted,
    /// Embedding API returned a permanent (non-retryable) error.
    PermanentApiError(String),
    /// Manually skipped by operator.
    ManualSkip(String),
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkipReason::ContentTooLarge { size, max } => {
                write!(f, "content too large: {size} bytes (max {max})")
            }
            SkipReason::InvalidEncoding(enc) => write!(f, "invalid encoding: {enc}"),
            SkipReason::ContentDeleted => write!(f, "content deleted"),
            SkipReason::PermanentApiError(msg) => write!(f, "permanent API error: {msg}"),
            SkipReason::ManualSkip(reason) => write!(f, "manually skipped: {reason}"),
        }
    }
}

/// Migration state machine states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MigrationState {
    /// Migration is planned but has not started.
    // FP-036: alias allows deserializing data stored before rename_all = "snake_case" was applied.
    #[serde(alias = "Planned")]
    Planned,
    /// Migration is actively processing embeddings.
    #[serde(alias = "InProgress")]
    InProgress {
        /// Number of embeddings processed so far.
        processed: usize,
        /// Total number of embeddings to process.
        total: usize,
        /// Number of embeddings skipped.
        #[serde(default)]
        skipped: usize,
    },
    /// Migration is paused (can be resumed).
    #[serde(alias = "Paused")]
    Paused {
        /// Number of embeddings processed before pause.
        processed: usize,
        /// Total number of embeddings to process.
        total: usize,
        /// Number of embeddings skipped.
        #[serde(default)]
        skipped: usize,
        /// Reason the migration was paused.
        reason: String,
    },
    /// Migration completed successfully.
    #[serde(alias = "Completed")]
    Completed {
        /// Total number of embeddings processed.
        processed: usize,
        /// Number of embeddings skipped.
        #[serde(default)]
        skipped: usize,
        /// Wall-clock duration in seconds.
        duration_secs: f64,
    },
    /// Migration failed with an error.
    #[serde(alias = "Failed")]
    Failed {
        /// Number of embeddings processed before failure.
        processed: usize,
        /// Total number of embeddings to process.
        total: usize,
        /// Number of embeddings skipped.
        #[serde(default)]
        skipped: usize,
        /// Error message describing the failure.
        error: String,
    },
    /// Migration was cancelled by the operator.
    #[serde(alias = "Cancelled")]
    Cancelled {
        /// Number of embeddings processed before cancellation.
        processed: usize,
        /// Total number of embeddings to process.
        total: usize,
        /// Number of embeddings skipped.
        #[serde(default)]
        skipped: usize,
    },
}

impl MigrationState {
    /// Returns `true` if the migration can be resumed (paused or failed).
    #[inline]
    pub fn is_resumable(&self) -> bool {
        matches!(
            self,
            MigrationState::Paused { .. } | MigrationState::Failed { .. }
        )
    }

    /// Returns `true` if the migration has reached a terminal state (completed or cancelled).
    #[inline]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            MigrationState::Completed { .. } | MigrationState::Cancelled { .. }
        )
    }

    /// Returns `true` if the migration is currently in progress.
    #[inline]
    pub fn is_active(&self) -> bool {
        matches!(self, MigrationState::InProgress { .. })
    }

    /// Returns the progress as a fraction in [0.0, 1.0], or `None` if not applicable.
    pub fn progress(&self) -> Option<f64> {
        match self {
            MigrationState::Planned => Some(0.0),
            MigrationState::InProgress {
                processed, total, ..
            }
            | MigrationState::Paused {
                processed, total, ..
            }
            | MigrationState::Failed {
                processed, total, ..
            }
            | MigrationState::Cancelled {
                processed, total, ..
            } => {
                if *total == 0 {
                    Some(1.0)
                } else {
                    Some(*processed as f64 / *total as f64)
                }
            }
            MigrationState::Completed { .. } => Some(1.0),
        }
    }

    /// Returns the number of embeddings processed so far.
    pub fn processed(&self) -> usize {
        match self {
            MigrationState::Planned => 0,
            MigrationState::InProgress { processed, .. }
            | MigrationState::Paused { processed, .. }
            | MigrationState::Failed { processed, .. }
            | MigrationState::Cancelled { processed, .. }
            | MigrationState::Completed { processed, .. } => *processed,
        }
    }

    /// Returns the number of embeddings skipped.
    pub fn skipped(&self) -> usize {
        match self {
            MigrationState::Planned => 0,
            MigrationState::InProgress { skipped, .. }
            | MigrationState::Paused { skipped, .. }
            | MigrationState::Failed { skipped, .. }
            | MigrationState::Cancelled { skipped, .. }
            | MigrationState::Completed { skipped, .. } => *skipped,
        }
    }

    /// Returns the total number of embeddings to process.
    pub fn total(&self) -> usize {
        match self {
            MigrationState::Planned | MigrationState::Completed { .. } => 0,
            MigrationState::InProgress { total, .. }
            | MigrationState::Paused { total, .. }
            | MigrationState::Failed { total, .. }
            | MigrationState::Cancelled { total, .. } => *total,
        }
    }

    /// Returns total minus skipped -- the number of embeddings that actually need processing.
    pub fn effective_total(&self) -> usize {
        self.total().saturating_sub(self.skipped())
    }

    /// Returns the fraction of effective_total that has been processed (0.0 to 1.0).
    pub fn effective_coverage(&self) -> f64 {
        let eff = self.effective_total();
        if eff == 0 {
            1.0
        } else {
            self.processed() as f64 / eff as f64
        }
    }
}

/// Describes an embedding migration operation.
///
/// # Example
///
/// ```rust
/// use lattice_embed::migration::MigrationPlan;
/// use lattice_embed::EmbeddingModel;
///
/// let plan = MigrationPlan {
///     id: "mig-001".to_string(),
///     source_model: EmbeddingModel::BgeSmallEnV15,
///     target_model: EmbeddingModel::BgeBaseEnV15,
///     total_embeddings: 10_000,
///     batch_size: 256,
///     created_at: "2026-01-27T00:00:00Z".to_string(),
/// };
///
/// assert_eq!(plan.total_embeddings, 10_000);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// Unique migration identifier.
    pub id: String,
    /// Model to migrate from.
    pub source_model: EmbeddingModel,
    /// Model to migrate to.
    pub target_model: EmbeddingModel,
    /// Total number of embeddings to migrate.
    pub total_embeddings: usize,
    /// Number of embeddings processed per batch.
    pub batch_size: usize,
    /// ISO 8601 timestamp when the plan was created.
    pub created_at: String,
}

/// Progress report for an active migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationProgress {
    /// Identifier of the migration this progress belongs to.
    pub migration_id: String,
    /// Current state of the migration.
    pub state: MigrationState,
    /// Number of embeddings skipped so far.
    #[serde(default)]
    pub skipped: usize,
    /// Total minus skipped -- the number of embeddings that actually need processing.
    #[serde(default)]
    pub effective_total: usize,
    /// Fraction of effective_total that has been processed (0.0 to 1.0).
    #[serde(default)]
    pub effective_coverage: f64,
    /// Embeddings processed per second.
    pub throughput: f64,
    /// Estimated seconds remaining, if calculable.
    pub eta_secs: Option<f64>,
    /// Number of errors encountered during processing.
    pub error_count: usize,
}

/// Errors from migration operations.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MigrationError {
    /// Attempted an invalid state transition.
    InvalidTransition {
        /// State being transitioned from.
        from: String,
        /// State being transitioned to.
        to: String,
    },
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrationError::InvalidTransition { from, to } => {
                write!(f, "invalid migration transition from {from} to {to}")
            }
        }
    }
}

impl std::error::Error for MigrationError {}
