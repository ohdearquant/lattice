//! Model registry module
//!
//! Provides versioned model storage and retrieval with deployment features.
//!
//! # Architecture
//!
//! ```text
//! Training → Model → Registry → Shadow → Deployment
//!                  ↓              ↓           ↓
//!              Storage     Evaluation    Rollback
//! ```
//!
//! # Concurrency
//!
//! [`ModelRegistry`] uses `ArcSwap` for lock-free reads during concurrent
//! updates. All read methods return owned snapshots; write methods are
//! serialized internally. See [`LiveModel`] for atomic model hot-swap.
//!
//! # Safe Deployment
//!
//! The registry includes two features for safe model deployment:
//!
//! - **Shadow Evaluation** ([`ShadowSession`]): Run candidate models in parallel
//!   with production to compare outputs before promotion.
//! - **Rollback** ([`RollbackController`]): Quickly revert to a previous model
//!   version when issues are detected, with full audit history.
//!
//! # Security
//!
//! When the `safetensors` feature is enabled, model weights can be serialized
//! using the safetensors format which prevents arbitrary code execution attacks.
//! This is **strongly recommended** for production deployments.
//!
//! # Example
//!
//! ```ignore
//! use lattice_tune::registry::{ModelRegistry, RegisteredModel, ModelMetadata};
//!
//! // Create a registry
//! let mut registry = ModelRegistry::new("/path/to/models");
//!
//! // Register a model
//! let model = RegisteredModel::new("intent_classifier", "1.0.0")
//!     .with_metadata(metadata);
//! registry.register(model)?;
//!
//! // Load a model
//! let model = registry.get("intent_classifier", "1.0.0")?;
//! ```

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
