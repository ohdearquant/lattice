//! Public API for the embedding-model migration state machine.
//!
//! It exposes plans, serializable lifecycle state, progress accounting, permanent skips,
//! and transition errors without choosing application routing policy.
//!
//! See [docs/migration.md](../../docs/migration.md) for the complete transition model.

mod controller;
mod types;

#[cfg(test)]
mod tests;

pub use controller::MigrationController;
pub use types::{MigrationError, MigrationPlan, MigrationProgress, MigrationState, SkipReason};
