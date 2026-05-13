//! Backfill coordinator: orchestrates embedding migration with routing logic.
//!
//! # Overview
//!
//! When migrating from one embedding model to another, the system needs to:
//!
//! 1. Continue serving queries against existing (legacy) embeddings
//! 2. Route new document embeddings appropriately (dual-write or target-only)
//! 3. Backfill old documents with the new model's embeddings in batches
//! 4. Switch queries to the new model once sufficient coverage exists
//!
//! The [`BackfillCoordinator`] wraps a [`MigrationController`] and adds routing
//! logic and batch management on top of the state machine.
//!
//! # Example
//!
//! ```rust
//! use lattice_embed::backfill::{BackfillCoordinator, BackfillConfig, EmbeddingRoute};
//! use lattice_embed::migration::MigrationPlan;
//! use lattice_embed::EmbeddingModel;
//!
//! let plan = MigrationPlan {
//!     id: "mig-001".to_string(),
//!     source_model: EmbeddingModel::BgeSmallEnV15,
//!     target_model: EmbeddingModel::BgeBaseEnV15,
//!     total_embeddings: 1000,
//!     batch_size: 100,
//!     created_at: "2026-01-27T00:00:00Z".to_string(),
//! };
//!
//! let mut coord = BackfillCoordinator::with_defaults(plan);
//!
//! // Before starting, all requests go to legacy
//! assert_eq!(coord.route_request(true), EmbeddingRoute::Legacy);
//!
//! coord.start().unwrap();
//!
//! // During migration, new documents get dual-written
//! assert_eq!(coord.route_request(true), EmbeddingRoute::DualWrite);
//! // Existing documents still use legacy
//! assert_eq!(coord.route_request(false), EmbeddingRoute::Legacy);
//! ```

use std::time::{Duration, Instant};

use crate::migration::{
    MigrationController, MigrationError, MigrationPlan, MigrationProgress, MigrationState,
};
use crate::model::EmbeddingModel;

use super::types::{BackfillConfig, EmbeddingRoute, EmbeddingRoutingConfig, RoutingPhase};

/// Coordinates the backfill process during embedding migration.
///
/// Wraps a [`MigrationController`] and adds:
/// - **Request routing**: decides which model handles new embedding requests
/// - **Query routing**: decides which model's embeddings to search
/// - **Batch management**: tracks backfill progress and computes next batch sizes
///
/// # State Machine
///
/// The coordinator delegates all state transitions to [`MigrationController`]
/// and layers routing logic on top:
///
/// | State       | New Doc Route | Existing Doc Route | Query Route         |
/// |-------------|---------------|--------------------|---------------------|
/// | Planned     | Legacy        | Legacy             | Legacy              |
/// | InProgress  | DualWrite*    | Legacy             | Legacy or Target**  |
/// | Paused      | Legacy        | Legacy             | Legacy              |
/// | Completed   | Target        | Target             | Target              |
/// | Failed      | Legacy        | Legacy             | Legacy              |
/// | Cancelled   | Legacy        | Legacy             | Legacy              |
///
/// \* Only if `dual_write` is enabled in config.
/// \*\* Switches to Target when progress >= `target_query_threshold`.
#[derive(Debug)]
pub struct BackfillCoordinator {
    pub(super) config: BackfillConfig,
    pub(super) controller: MigrationController,
    /// Count of items successfully backfilled.
    backfilled_count: usize,
    /// Timestamp when cutover occurred (for rollback window tracking).
    pub(super) cutover_at: Option<Instant>,
    /// Duration of rollback window (computed from config).
    rollback_window: Duration,
}

impl BackfillCoordinator {
    /// Create a new backfill coordinator with the given plan and config.
    pub fn new(plan: MigrationPlan, config: BackfillConfig) -> Self {
        let rollback_window = Duration::from_secs(config.rollback_window_secs);
        Self {
            config,
            controller: MigrationController::new(plan),
            backfilled_count: 0,
            cutover_at: None,
            rollback_window,
        }
    }

    /// Create a backfill coordinator with default configuration.
    pub fn with_defaults(plan: MigrationPlan) -> Self {
        Self::new(plan, BackfillConfig::default())
    }

    /// Start the backfill process.
    ///
    /// Transitions the underlying migration from `Planned` to `InProgress`.
    ///
    /// # Errors
    ///
    /// Returns [`MigrationError::InvalidTransition`] if not in `Planned` state.
    pub fn start(&mut self) -> Result<(), MigrationError> {
        self.controller.start()
    }

    /// Route an embedding request based on current migration state.
    ///
    /// - `is_new_document`: `true` if this is a new document being indexed
    ///   for the first time, `false` if it is being re-embedded as part of
    ///   normal operations (not backfill).
    ///
    /// During active migration with `dual_write` enabled, new documents are
    /// embedded with both source and target models to keep both indexes current.
    pub fn route_request(&self, is_new_document: bool) -> EmbeddingRoute {
        match self.controller.state() {
            MigrationState::Planned => EmbeddingRoute::Legacy,
            MigrationState::InProgress { .. } => {
                if is_new_document && self.config.dual_write {
                    EmbeddingRoute::DualWrite
                } else {
                    EmbeddingRoute::Legacy
                }
            }
            MigrationState::Paused { .. } => EmbeddingRoute::Legacy,
            MigrationState::Completed { .. } => EmbeddingRoute::Target,
            MigrationState::Failed { .. } => EmbeddingRoute::Legacy,
            MigrationState::Cancelled { .. } => EmbeddingRoute::Legacy,
        }
    }

    /// Route a query request (which model's embeddings to search against).
    ///
    /// Returns [`EmbeddingRoute::Target`] once migration is complete, or
    /// once progress exceeds [`BackfillConfig::target_query_threshold`]
    /// during an active migration. Otherwise returns [`EmbeddingRoute::Legacy`].
    pub fn route_query(&self) -> EmbeddingRoute {
        match self.controller.state() {
            MigrationState::Completed { .. } => EmbeddingRoute::Target,
            MigrationState::InProgress {
                processed, total, ..
            } => {
                let progress = if *total == 0 {
                    1.0
                } else {
                    *processed as f64 / *total as f64
                };
                if progress >= self.config.target_query_threshold {
                    EmbeddingRoute::Target
                } else {
                    EmbeddingRoute::Legacy
                }
            }
            _ => EmbeddingRoute::Legacy,
        }
    }

    /// Record a completed backfill batch.
    ///
    /// Updates both the internal backfilled count and the underlying
    /// migration controller's progress. If this causes processed to
    /// reach total, the migration auto-completes and the cutover timestamp
    /// is recorded for rollback window tracking.
    ///
    /// # Errors
    ///
    /// Returns [`MigrationError::InvalidTransition`] if not in `InProgress` state.
    pub fn record_batch(&mut self, count: usize) -> Result<(), MigrationError> {
        let was_in_progress = matches!(self.controller.state(), MigrationState::InProgress { .. });
        self.backfilled_count += count;
        self.controller.record_progress(count)?;

        // Track cutover time when transitioning to Completed
        if was_in_progress && matches!(self.controller.state(), MigrationState::Completed { .. }) {
            self.cutover_at = Some(Instant::now());
        }

        Ok(())
    }

    /// Record a non-fatal error encountered during backfill processing.
    ///
    /// Increments the error counter without changing state. Useful for
    /// tracking transient failures (e.g., individual embedding retries).
    pub fn record_error(&mut self) {
        self.controller.record_error();
    }

    /// Pause the backfill process.
    ///
    /// # Errors
    ///
    /// Returns [`MigrationError::InvalidTransition`] if not in `InProgress` state.
    pub fn pause(&mut self, reason: impl Into<String>) -> Result<(), MigrationError> {
        self.controller.pause(reason)
    }

    /// Resume the backfill process.
    ///
    /// # Errors
    ///
    /// Returns [`MigrationError::InvalidTransition`] if not in `Paused` or `Failed` state.
    pub fn resume(&mut self) -> Result<(), MigrationError> {
        self.controller.resume()
    }

    /// Cancel the backfill process.
    ///
    /// # Errors
    ///
    /// Returns [`MigrationError::InvalidTransition`] if already in a terminal state.
    pub fn cancel(&mut self) -> Result<(), MigrationError> {
        self.controller.cancel()
    }

    /// Get the current migration state.
    #[inline]
    pub fn state(&self) -> &MigrationState {
        self.controller.state()
    }

    /// Get a snapshot of current migration progress.
    #[inline]
    pub fn progress(&self) -> MigrationProgress {
        self.controller.progress()
    }

    /// Get the source model being migrated from.
    #[inline]
    pub fn source_model(&self) -> EmbeddingModel {
        self.controller.plan().source_model
    }

    /// Get the target model being migrated to.
    #[inline]
    pub fn target_model(&self) -> EmbeddingModel {
        self.controller.plan().target_model
    }

    /// Check if we are currently in the post-cutover rollback window.
    ///
    /// Returns `true` if the migration has completed and we are still within
    /// the rollback window duration. During this period, queries use the
    /// target model but writes continue to both models.
    #[inline]
    pub fn in_rollback_window(&self) -> bool {
        self.cutover_at
            .map(|t| t.elapsed() < self.rollback_window)
            .unwrap_or(false)
    }

    /// Get the routing configuration for the current state.
    ///
    /// Returns an [`EmbeddingRoutingConfig`] that separates query routing
    /// from write routing. This enables true rollback during the post-cutover
    /// window by continuing to dual-write while queries use the new model.
    ///
    /// # Routing Logic
    ///
    /// | State               | Query Model | Write Models        | Phase         |
    /// |---------------------|-------------|---------------------|---------------|
    /// | Planned             | source      | `[source]`          | Stable        |
    /// | InProgress          | source      | `[source, target]`  | Migrating     |
    /// | Completed (window)  | target      | `[source, target]`  | RollbackWindow|
    /// | Completed (stable)  | target      | `[target]`          | Stable        |
    /// | Paused/Failed/etc   | source      | `[source]`          | Stable        |
    pub fn routing_config(&self) -> EmbeddingRoutingConfig {
        match self.controller.state() {
            MigrationState::Planned => EmbeddingRoutingConfig {
                query_model: self.source_model(),
                write_models: vec![self.source_model()],
                phase: RoutingPhase::Stable,
                migration_id: None,
            },
            MigrationState::InProgress { .. } => {
                let write_models = if self.config.dual_write {
                    vec![self.source_model(), self.target_model()]
                } else {
                    vec![self.source_model()]
                };
                EmbeddingRoutingConfig {
                    query_model: self.source_model(), // query legacy during migration
                    write_models,
                    phase: RoutingPhase::Migrating,
                    migration_id: Some(self.controller.plan().id.clone()),
                }
            }
            MigrationState::Completed { .. } => {
                // During rollback window, still dual-write
                if self.in_rollback_window() {
                    EmbeddingRoutingConfig {
                        query_model: self.target_model(), // query new
                        write_models: vec![self.source_model(), self.target_model()], // still dual
                        phase: RoutingPhase::RollbackWindow,
                        migration_id: Some(self.controller.plan().id.clone()),
                    }
                } else {
                    EmbeddingRoutingConfig {
                        query_model: self.target_model(),
                        write_models: vec![self.target_model()],
                        phase: RoutingPhase::Stable,
                        migration_id: None,
                    }
                }
            }
            // Paused, Failed, Cancelled: fall back to source-only
            _ => EmbeddingRoutingConfig {
                query_model: self.source_model(),
                write_models: vec![self.source_model()],
                phase: RoutingPhase::Stable,
                migration_id: None,
            },
        }
    }

    /// Get the backfill configuration.
    #[inline]
    pub fn config(&self) -> &BackfillConfig {
        &self.config
    }

    /// Get the total count of items successfully backfilled.
    #[inline]
    pub fn backfilled_count(&self) -> usize {
        self.backfilled_count
    }

    /// Compute the next batch size to process.
    ///
    /// Returns `0` if the migration is not in `InProgress` state.
    /// Otherwise returns the smaller of the configured batch size
    /// and the remaining items.
    pub fn next_batch_size(&self) -> usize {
        match self.controller.state() {
            MigrationState::InProgress {
                processed, total, ..
            } => {
                let remaining = total.saturating_sub(*processed);
                remaining.min(self.config.batch_size)
            }
            _ => 0,
        }
    }
}
