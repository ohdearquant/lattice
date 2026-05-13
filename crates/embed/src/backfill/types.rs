//! Backfill types: routing decisions, phases, and configuration.

use serde::{Deserialize, Serialize};

use crate::model::EmbeddingModel;

/// Routing decision for an embedding request during migration.
///
/// Determines which model should handle an embedding operation based on
/// the current migration state and the nature of the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingRoute {
    /// Use the current (legacy) model -- no migration active or migration
    /// incomplete for this operation type.
    Legacy,
    /// Use the new (target) model -- migration complete or sufficient
    /// progress for query routing.
    Target,
    /// Embed with both models (dual-write) -- during active migration
    /// for new documents to ensure both indexes stay current.
    DualWrite,
}

/// Phase of the migration lifecycle for routing decisions.
///
/// Distinguishes between normal operation, active migration, and the
/// post-cutover rollback window where dual-writing continues to enable
/// safe rollback without data loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingPhase {
    /// Normal operation, no migration active.
    Stable,
    /// Active migration in progress.
    Migrating,
    /// Post-cutover rollback window (still dual-writing).
    ///
    /// During this phase, queries use the target model but writes go to
    /// both models. This enables instant rollback without losing atoms
    /// created after cutover.
    RollbackWindow,
}

/// Configuration for embedding request routing.
///
/// Separates query routing from write routing to enable true rollback.
/// During the rollback window after cutover, queries use the new model
/// but writes continue to both models, ensuring atoms created after
/// cutover are present in both indexes.
///
/// # Example
///
/// ```rust
/// use lattice_embed::backfill::{EmbeddingRoutingConfig, RoutingPhase};
/// use lattice_embed::EmbeddingModel;
///
/// let config = EmbeddingRoutingConfig {
///     query_model: EmbeddingModel::BgeBaseEnV15,
///     write_models: vec![EmbeddingModel::BgeSmallEnV15, EmbeddingModel::BgeBaseEnV15],
///     phase: RoutingPhase::RollbackWindow,
///     migration_id: Some("mig-001".to_string()),
/// };
///
/// assert!(config.write_models.len() > 1); // dual-write active
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRoutingConfig {
    /// Which model to query against.
    pub query_model: EmbeddingModel,
    /// Which models to write to (may be multiple for dual-write).
    pub write_models: Vec<EmbeddingModel>,
    /// Current routing phase.
    pub phase: RoutingPhase,
    /// Migration ID if in migration/rollback phase.
    pub migration_id: Option<String>,
}

/// Configuration for the backfill coordinator.
///
/// Controls batch sizing, concurrency, dual-write behavior, and the
/// threshold at which queries switch to the target model.
///
/// # Example
///
/// ```rust
/// use lattice_embed::backfill::BackfillConfig;
///
/// let config = BackfillConfig {
///     batch_size: 200,
///     max_concurrent: 8,
///     dual_write: true,
///     target_query_threshold: 0.9,
///     rollback_window_secs: 3600, // 1 hour
/// };
///
/// assert_eq!(config.batch_size, 200);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillConfig {
    /// Batch size for backfill operations.
    pub batch_size: usize,
    /// Maximum concurrent backfill batches.
    pub max_concurrent: usize,
    /// Whether new documents should be dual-written during migration.
    pub dual_write: bool,
    /// Minimum progress fraction before allowing target-only queries `[0.0, 1.0]`.
    ///
    /// When the migration progress reaches this threshold, query routing
    /// switches from legacy to target embeddings.
    pub target_query_threshold: f64,
    /// Duration in seconds to continue dual-writing after cutover.
    ///
    /// During the rollback window, queries use the new target model but
    /// writes continue to both models. This ensures atoms created after
    /// cutover exist in both indexes, enabling instant rollback without
    /// data loss.
    ///
    /// Default: 86400 seconds (24 hours).
    pub rollback_window_secs: u64,
}

impl Default for BackfillConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            max_concurrent: 4,
            dual_write: true,
            target_query_threshold: 0.8,
            rollback_window_secs: 86400, // 24 hours
        }
    }
}
