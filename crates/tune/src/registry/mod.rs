//! Versioned model records, artifact storage, and deployment helpers.
//!
//! [`ModelRegistry`] provides lock-free record reads and serialized writes;
//! [`LiveModel`] atomically swaps a serving record. Shadow evaluation and
//! rollback recording are explicit workflow helpers, not automatic deployment.
//!
//! See `docs/registry.md` for the registry design, storage formats, and lifecycle.

mod live_model;
mod model;
mod rollback;
mod shadow;
mod storage;

#[cfg(feature = "safetensors")]
pub mod safetensors_io;

pub use live_model::LiveModel;
pub use model::{ModelMetadata, ModelStatus, RegisteredModel};
pub use rollback::{RollbackController, RollbackRecord};
pub use shadow::{ShadowComparison, ShadowConfig, ShadowSession, ShadowState};
pub use storage::{ModelQuery, ModelRegistry, StorageBackend, sha256_hash};
