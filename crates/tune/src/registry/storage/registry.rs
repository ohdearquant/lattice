//! Model registry for storing and retrieving versioned models.
//!
//! Uses `ArcSwap` for lock-free concurrent reads and a clone-modify-store
//! pattern for writes. Readers always see a consistent snapshot; writers
//! are serialized via `write_lock` to prevent lost updates.

use super::backends::InMemoryStorage;
use super::{FileSystemStorage, StorageBackend, sha256_hash};
use crate::error::{Result, TuneError};
use crate::registry::model::{ModelStatus, RegisteredModel};
use arc_swap::ArcSwap;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

/// Model registry for storing and retrieving versioned models.
///
/// All read operations (`get_by_id`, `get`, `list_*`, `len`, `is_empty`)
/// are lock-free via `ArcSwap` snapshots. Write operations (`register`,
/// `update_status`, `promote_to_production`, `delete`) use a
/// clone-modify-store pattern serialized by an internal write lock.
pub struct ModelRegistry {
    /// Storage backend (behind Mutex for `&mut self` methods)
    storage: parking_lot::Mutex<Box<dyn StorageBackend>>,

    /// In-memory index of registered models (lock-free reads via ArcSwap)
    models: ArcSwap<HashMap<Uuid, RegisteredModel>>,

    /// Name+version to ID mapping (lock-free reads via ArcSwap)
    name_index: ArcSwap<HashMap<String, Uuid>>,

    /// Serializes write operations to prevent lost updates
    write_lock: parking_lot::Mutex<()>,
}

impl ModelRegistry {
    /// Create a new registry with in-memory storage
    pub fn in_memory() -> Self {
        Self {
            storage: parking_lot::Mutex::new(Box::new(InMemoryStorage::new())),
            models: ArcSwap::new(Arc::new(HashMap::new())),
            name_index: ArcSwap::new(Arc::new(HashMap::new())),
            write_lock: parking_lot::Mutex::new(()),
        }
    }

    /// Create a new registry with filesystem storage
    pub fn with_path(path: impl Into<PathBuf>) -> Result<Self> {
        let storage = FileSystemStorage::new(path)?;
        Ok(Self {
            storage: parking_lot::Mutex::new(Box::new(storage)),
            models: ArcSwap::new(Arc::new(HashMap::new())),
            name_index: ArcSwap::new(Arc::new(HashMap::new())),
            write_lock: parking_lot::Mutex::new(()),
        })
    }

    /// Create a registry with custom storage backend
    pub fn with_storage(storage: Box<dyn StorageBackend>) -> Self {
        Self {
            storage: parking_lot::Mutex::new(storage),
            models: ArcSwap::new(Arc::new(HashMap::new())),
            name_index: ArcSwap::new(Arc::new(HashMap::new())),
            write_lock: parking_lot::Mutex::new(()),
        }
    }

    // ---------------------------------------------------------------
    // Write methods (&self, serialized by write_lock)
    // ---------------------------------------------------------------

    /// Register a model with weights
    pub fn register(&self, mut model: RegisteredModel, weights: &[u8]) -> Result<Uuid> {
        model.validate().map_err(TuneError::Validation)?;

        let key = model.full_name();

        let _wg = self.write_lock.lock();

        // Check for duplicates under the write lock
        if self.name_index.load().contains_key(&key) {
            return Err(TuneError::DuplicateModel {
                name: model.name.clone(),
                version: model.version.clone(),
            });
        }

        // Save weights to storage (may fail)
        let weights_path = self.storage.lock().save(&model, weights)?;
        let weights_size = weights.len();
        let weights_hash = sha256_hash(weights);

        // Update model with weights info
        model = model.with_weights(weights_path, weights_size, weights_hash);

        let id = model.id;

        // Clone-modify-store for models
        let current_models = self.models.load();
        let mut new_models = (**current_models).clone();
        new_models.insert(id, model);
        self.models.store(Arc::new(new_models));

        // Clone-modify-store for name_index
        let current_index = self.name_index.load();
        let mut new_index = (**current_index).clone();
        new_index.insert(key, id);
        self.name_index.store(Arc::new(new_index));

        Ok(id)
    }

    /// Register a model without weights (metadata only)
    pub fn register_metadata(&self, model: RegisteredModel) -> Result<Uuid> {
        model.validate().map_err(TuneError::Validation)?;

        let key = model.full_name();

        let _wg = self.write_lock.lock();

        if self.name_index.load().contains_key(&key) {
            return Err(TuneError::DuplicateModel {
                name: model.name.clone(),
                version: model.version.clone(),
            });
        }

        let id = model.id;

        let current_models = self.models.load();
        let mut new_models = (**current_models).clone();
        new_models.insert(id, model);
        self.models.store(Arc::new(new_models));

        let current_index = self.name_index.load();
        let mut new_index = (**current_index).clone();
        new_index.insert(key, id);
        self.name_index.store(Arc::new(new_index));

        Ok(id)
    }

    /// Update model status
    pub fn update_status(&self, id: &Uuid, status: ModelStatus) -> Result<()> {
        let _wg = self.write_lock.lock();

        let current = self.models.load();
        let mut new_models = (**current).clone();

        let model = new_models
            .get_mut(id)
            .ok_or_else(|| TuneError::ModelNotFound {
                name: id.to_string(),
                version: "".to_string(),
            })?;

        model.status = status;
        model.updated_at = chrono::Utc::now();

        self.models.store(Arc::new(new_models));
        Ok(())
    }

    /// Promote a model to production (demote current production)
    pub fn promote_to_production(&self, id: &Uuid) -> Result<()> {
        let _wg = self.write_lock.lock();

        let current = self.models.load();
        let mut new_models = (**current).clone();

        let name = new_models
            .get(id)
            .ok_or_else(|| TuneError::ModelNotFound {
                name: id.to_string(),
                version: "".to_string(),
            })?
            .name
            .clone();

        let now = chrono::Utc::now();

        // Demote current production models
        let current_production: Vec<Uuid> = new_models
            .values()
            .filter(|m| m.name == name && m.status == ModelStatus::Production)
            .map(|m| m.id)
            .collect();

        for prod_id in current_production {
            if let Some(m) = new_models.get_mut(&prod_id) {
                m.status = ModelStatus::Staged;
                m.updated_at = now;
            }
        }

        // Promote new model
        if let Some(m) = new_models.get_mut(id) {
            m.status = ModelStatus::Production;
            m.updated_at = now;
        }

        self.models.store(Arc::new(new_models));
        Ok(())
    }

    /// Delete a model
    pub fn delete(&self, id: &Uuid) -> Result<()> {
        let _wg = self.write_lock.lock();

        let current_models = self.models.load();
        let mut new_models = (**current_models).clone();

        let model = new_models
            .remove(id)
            .ok_or_else(|| TuneError::ModelNotFound {
                name: id.to_string(),
                version: "".to_string(),
            })?;

        // Delete weights from storage first (may fail; if it does, indexes stay intact)
        if let Some(path) = &model.weights_path {
            self.storage.lock().delete(path)?;
        }

        // Update indexes only after storage succeeds
        let key = model.full_name();
        let current_index = self.name_index.load();
        let mut new_index = (**current_index).clone();
        new_index.remove(&key);

        self.models.store(Arc::new(new_models));
        self.name_index.store(Arc::new(new_index));

        Ok(())
    }

    // ---------------------------------------------------------------
    // Read methods (&self, lock-free via ArcSwap snapshots)
    // ---------------------------------------------------------------

    /// Get a model by ID (lock-free snapshot)
    pub fn get_by_id(&self, id: &Uuid) -> Option<RegisteredModel> {
        self.models.load().get(id).cloned()
    }

    /// Get a model by name and version (lock-free snapshot)
    pub fn get(&self, name: &str, version: &str) -> Option<RegisteredModel> {
        let key = format!("{name}:{version}");
        let index = self.name_index.load();
        let models = self.models.load();
        index.get(&key).and_then(|id| models.get(id).cloned())
    }

    /// Get the latest version of a model (lock-free snapshot)
    pub fn get_latest(&self, name: &str) -> Option<RegisteredModel> {
        self.list_versions(name).into_iter().max_by(|a, b| {
            let a_ver = a.version_tuple().unwrap_or((0, 0, 0));
            let b_ver = b.version_tuple().unwrap_or((0, 0, 0));
            a_ver.cmp(&b_ver)
        })
    }

    /// Get the production model for a name (lock-free snapshot)
    pub fn get_production(&self, name: &str) -> Option<RegisteredModel> {
        self.list_versions(name)
            .into_iter()
            .find(|m| m.status == ModelStatus::Production)
    }

    /// List all versions of a model (lock-free snapshot)
    pub fn list_versions(&self, name: &str) -> Vec<RegisteredModel> {
        self.models
            .load()
            .values()
            .filter(|m| m.name == name)
            .cloned()
            .collect()
    }

    /// List all models (lock-free snapshot)
    pub fn list_all(&self) -> Vec<RegisteredModel> {
        self.models.load().values().cloned().collect()
    }

    /// List models by status (lock-free snapshot)
    pub fn list_by_status(&self, status: ModelStatus) -> Vec<RegisteredModel> {
        self.models
            .load()
            .values()
            .filter(|m| m.status == status)
            .cloned()
            .collect()
    }

    /// List model names (lock-free snapshot)
    pub fn list_names(&self) -> Vec<String> {
        let snap = self.models.load();
        let mut names: Vec<String> = snap.values().map(|m| m.name.clone()).collect();
        names.sort();
        names.dedup();
        names
    }

    /// Load model weights, verifying the recorded checksum when one is on
    /// record.
    ///
    /// #504 remaining slice 3: this used to be a pure raw load with **no**
    /// checksum check at all — a caller reaching for the "obvious" method
    /// name silently bypassed the verification `load_weights_verified`
    /// performed, even though every model registered via [`Self::register`]
    /// always carries a `weights_hash`. Fail-closed discipline now applies
    /// here too: a hash mismatch is a hard
    /// `TuneError::WeightIntegrityError`, not a silent pass-through.
    ///
    /// Verification is anchored to the **canonical registry record**
    /// (resolved by `model.id` from the registry's own snapshot), never to
    /// the caller-provided value: [`RegisteredModel`] is a plain DTO with
    /// public `weights_path`/`weights_hash` fields, so a mutated clone
    /// could otherwise redirect the load to a different path or "verify"
    /// tampered bytes against a recomputed hash. A caller argument that
    /// disagrees with the canonical record on either field is rejected
    /// outright. A model whose *canonical* record has no hash at all
    /// (e.g. [`Self::register_metadata`] with weights added out-of-band,
    /// or a legacy pre-hash record) has nothing to check against and loads
    /// unverified — that is the only remaining "raw" case, and it is
    /// explicit (a `None` hash on the canonical record), never a silent
    /// skip of a hash that *is* present.
    pub fn load_weights(&self, model: &RegisteredModel) -> Result<Vec<u8>> {
        let snap = self.models.load();
        let canonical = snap
            .get(&model.id)
            .ok_or_else(|| TuneError::ModelNotFound {
                name: model.name.clone(),
                version: model.version.clone(),
            })?;

        if model.weights_path != canonical.weights_path
            || model.weights_hash != canonical.weights_hash
        {
            return Err(TuneError::Storage(format!(
                "load_weights: model {} argument disagrees with the canonical registry \
                 record on weights_path/weights_hash; refusing to load from a mutated clone",
                model.id
            )));
        }

        let path = canonical
            .weights_path
            .as_ref()
            .ok_or_else(|| TuneError::Storage("No weights path".to_string()))?;

        let weights = self.storage.lock().load(path)?;

        if let Some(ref expected_hash) = canonical.weights_hash {
            let actual_hash = sha256_hash(&weights);
            if &actual_hash != expected_hash {
                return Err(TuneError::WeightIntegrityError {
                    expected: expected_hash.clone(),
                    actual: actual_hash,
                });
            }
        }

        Ok(weights)
    }

    /// Load model weights, **requiring** a recorded checksum to verify
    /// against.
    ///
    /// This is [`Self::load_weights`] (which already verifies whenever a
    /// hash is on record) plus one additional guarantee for callers that
    /// explicitly want verification: a model with **no** recorded
    /// `weights_hash` is rejected rather than silently loaded unverified.
    /// Use this entry point wherever the caller's contract requires "this
    /// load is verified", and [`Self::load_weights`] only when an
    /// unverified load of a hash-less (metadata-only-registered) model is
    /// an accepted, understood case.
    ///
    /// Returns `TuneError::WeightIntegrityError` if the checksum doesn't
    /// match, or `TuneError::Storage` if no checksum is on record at all.
    ///
    /// Like [`Self::load_weights`], the "is a hash on record" decision is
    /// made against the canonical registry record, not the caller's clone
    /// (whose public `weights_hash` field could have been set to `None` or
    /// to a recomputed hash of tampered bytes).
    pub fn load_weights_verified(&self, model: &RegisteredModel) -> Result<Vec<u8>> {
        let snap = self.models.load();
        let canonical = snap
            .get(&model.id)
            .ok_or_else(|| TuneError::ModelNotFound {
                name: model.name.clone(),
                version: model.version.clone(),
            })?;
        if canonical.weights_hash.is_none() {
            return Err(TuneError::Storage(format!(
                "load_weights_verified: model {} ({} v{}) has no recorded \
                 weights_hash to verify against",
                model.id, model.name, model.version
            )));
        }
        drop(snap);
        self.load_weights(model)
    }

    /// Get number of registered models (lock-free snapshot)
    pub fn len(&self) -> usize {
        self.models.load().len()
    }

    /// Check if registry is empty (lock-free snapshot)
    pub fn is_empty(&self) -> bool {
        self.models.load().is_empty()
    }
}
