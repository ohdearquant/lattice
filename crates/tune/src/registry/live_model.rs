//! Live model handle for atomic hot-swap during inference.
//!
//! [`LiveModel`] wraps a [`RegisteredModel`] in an [`ArcSwap`], enabling
//! lock-free reads and atomic swaps for model hot-reload scenarios.
//!
//! # Example
//!
//! ```
//! use lattice_tune::registry::{RegisteredModel, LiveModel};
//!
//! let model_v1 = RegisteredModel::new("classifier", "1.0.0");
//! let live = LiveModel::new(model_v1);
//!
//! // Read current model (lock-free)
//! let current = live.load();
//! assert_eq!(current.version, "1.0.0");
//!
//! // Hot-swap to new version
//! let model_v2 = RegisteredModel::new("classifier", "2.0.0");
//! let old = live.swap(model_v2);
//! assert_eq!(old.version, "1.0.0");
//!
//! // Readers now see v2
//! assert_eq!(live.version(), "2.0.0");
//! ```

use arc_swap::ArcSwap;
use std::sync::Arc;
use uuid::Uuid;

use super::model::RegisteredModel;

/// A handle to a live model that can be atomically swapped.
///
/// Readers get a consistent snapshot via [`load()`](LiveModel::load) (lock-free);
/// writers replace the model atomically via [`swap()`](LiveModel::swap).
///
/// This is designed for inference hot-reload: an inference server holds a
/// `LiveModel` and reads the current model on each request. A background
/// updater swaps in a new model version without blocking readers.
pub struct LiveModel {
    current: ArcSwap<RegisteredModel>,
}

impl LiveModel {
    /// Create a new `LiveModel` from an initial [`RegisteredModel`].
    pub fn new(model: RegisteredModel) -> Self {
        Self {
            current: ArcSwap::new(Arc::new(model)),
        }
    }

    /// Get a snapshot of the current model.
    ///
    /// This is a lock-free operation that returns an [`Arc`] pointing to
    /// the current model. The returned `Arc` remains valid even if the
    /// model is swapped after this call.
    pub fn load(&self) -> Arc<RegisteredModel> {
        self.current.load_full()
    }

    /// Atomically swap to a new model version, returning the previous one.
    pub fn swap(&self, new_model: RegisteredModel) -> Arc<RegisteredModel> {
        self.current.swap(Arc::new(new_model))
    }

    /// Get the current model's ID (lock-free snapshot).
    pub fn model_id(&self) -> Uuid {
        self.current.load().id
    }

    /// Get the current model's version string (lock-free snapshot).
    pub fn version(&self) -> String {
        self.current.load().version.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::model::RegisteredModel;

    #[test]
    fn test_live_model_load() {
        let model = RegisteredModel::new("classifier", "1.0.0");
        let expected_id = model.id;
        let live = LiveModel::new(model);

        let snap = live.load();
        assert_eq!(snap.name, "classifier");
        assert_eq!(snap.version, "1.0.0");
        assert_eq!(live.model_id(), expected_id);
        assert_eq!(live.version(), "1.0.0");
    }

    #[test]
    fn test_live_model_swap() {
        let model_v1 = RegisteredModel::new("classifier", "1.0.0");
        let live = LiveModel::new(model_v1);
        assert_eq!(live.version(), "1.0.0");

        let model_v2 = RegisteredModel::new("classifier", "2.0.0");
        let old = live.swap(model_v2);

        assert_eq!(old.version, "1.0.0");
        assert_eq!(live.version(), "2.0.0");
        assert_eq!(live.load().version, "2.0.0");
    }

    #[test]
    fn test_live_model_snapshot_stability() {
        // A snapshot taken before a swap should still be valid after the swap.
        let model_v1 = RegisteredModel::new("classifier", "1.0.0");
        let live = LiveModel::new(model_v1);

        let snap_before = live.load();
        assert_eq!(snap_before.version, "1.0.0");

        let model_v2 = RegisteredModel::new("classifier", "2.0.0");
        live.swap(model_v2);

        // Old snapshot still sees v1
        assert_eq!(snap_before.version, "1.0.0");
        // New load sees v2
        assert_eq!(live.load().version, "2.0.0");
    }

    #[test]
    fn test_live_model_concurrent_swap() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let model = RegisteredModel::new("classifier", "0.0.0");
        let live = StdArc::new(LiveModel::new(model));

        let mut handles = Vec::new();

        // Spawn writer threads that each swap in a new version
        for i in 1..=10 {
            let live_clone = StdArc::clone(&live);
            handles.push(thread::spawn(move || {
                let new_model = RegisteredModel::new("classifier", &format!("{i}.0.0"));
                live_clone.swap(new_model);
            }));
        }

        // Spawn reader threads that load concurrently
        for _ in 0..10 {
            let live_clone = StdArc::clone(&live);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let snap = live_clone.load();
                    // Should always be a valid model
                    assert_eq!(snap.name, "classifier");
                    assert!(!snap.version.is_empty());
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Final version should be one of the swapped-in versions
        let final_version = live.version();
        assert!(!final_version.is_empty());
    }
}
