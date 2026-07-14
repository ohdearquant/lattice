//! Defines migration-time routing decisions and backfill configuration.
//!
//! The routing configuration separates query and write model selection so a completed
//! cutover can retain source writes during its rollback window.
//!
//! See [docs/backfill.md](../../docs/backfill.md) for the routing contract and defaults.

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
/// Query and write selections are separate so a rollback window can preserve source writes.
/// See [docs/backfill.md](../../docs/backfill.md) for how each phase is represented.
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
/// Controls batch sizing, caller-managed concurrency, routing, and rollback timing.
/// See [docs/backfill.md](../../docs/backfill.md) for default values and policy semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillConfig {
    /// Batch size for backfill operations.
    pub batch_size: usize,
    /// Maximum concurrent backfill batches.
    pub max_concurrent: usize,
    /// Whether new documents should be dual-written during migration.
    pub dual_write: bool,
    /// Raw progress threshold at which query routing may choose the target model.
    pub target_query_threshold: f64,
    /// Seconds to retain both write models after a completed cutover.
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
