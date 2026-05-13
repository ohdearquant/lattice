//! Backfill coordinator for embedding migration.
//!
//! Orchestrates the re-embedding of documents during model migration,
//! routing new requests to the target model while backfilling old embeddings.
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
//! The [`BackfillCoordinator`] wraps a [`crate::migration::MigrationController`] and adds
//! routing logic and batch management on top of the state machine.
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

mod coordinator;
mod types;

#[cfg(test)]
mod tests;

pub use coordinator::BackfillCoordinator;
pub use types::{BackfillConfig, EmbeddingRoute, EmbeddingRoutingConfig, RoutingPhase};
