//! Public API for model-migration backfills.
//!
//! The coordinator layers read/write routing and batch sizing over the migration
//! controller while callers execute the actual embedding and index updates.
//!
//! See [docs/backfill.md](../../docs/backfill.md) for the complete operational model.

mod coordinator;
mod types;

#[cfg(test)]
mod tests;

pub use coordinator::BackfillCoordinator;
pub use types::{BackfillConfig, EmbeddingRoute, EmbeddingRoutingConfig, RoutingPhase};
