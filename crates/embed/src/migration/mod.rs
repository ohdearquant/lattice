//! Migration module: state machine for embedding model migration.
//!
//! Provides the [`MigrationController`] state machine, migration plan types,
//! and progress tracking for embedding model migrations.
//!
//! # Quick Start
//!
//! ```rust
//! use lattice_embed::migration::{MigrationController, MigrationPlan};
//! use lattice_embed::EmbeddingModel;
//!
//! let plan = MigrationPlan {
//!     id: "mig-001".to_string(),
//!     source_model: EmbeddingModel::BgeSmallEnV15,
//!     target_model: EmbeddingModel::BgeBaseEnV15,
//!     total_embeddings: 10_000,
//!     batch_size: 256,
//!     created_at: "2026-01-27T00:00:00Z".to_string(),
//! };
//!
//! let mut ctrl = MigrationController::new(plan);
//! ctrl.start().unwrap();
//! ctrl.record_progress(256).unwrap();
//!
//! let report = ctrl.progress();
//! assert!(report.state.is_active());
//! ```

mod controller;
mod types;

#[cfg(test)]
mod tests;

pub use controller::MigrationController;
pub use types::{MigrationError, MigrationPlan, MigrationProgress, MigrationState, SkipReason};
