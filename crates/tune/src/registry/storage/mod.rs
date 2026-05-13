//! Model storage backend

mod backends;
mod query;
mod registry;

pub use backends::FileSystemStorage;
#[cfg(feature = "sqlite")]
pub use backends::SqliteStorage;
pub use query::ModelQuery;
pub use registry::ModelRegistry;

use crate::error::{Result, TuneError};
use sha2::{Digest, Sha256};
use std::path::Path;

use super::model::RegisteredModel;

/// Storage backend trait
pub trait StorageBackend: Send + Sync {
    /// Save a model to storage
    fn save(&mut self, model: &RegisteredModel, weights: &[u8]) -> Result<String>;

    /// Load model weights from storage
    fn load(&self, path: &str) -> Result<Vec<u8>>;

    /// Delete model from storage
    fn delete(&mut self, path: &str) -> Result<()>;

    /// Check if model exists in storage
    fn exists(&self, path: &str) -> bool;

    /// List all model paths
    fn list(&self) -> Vec<String>;
}

/// Validate that a path is safe and doesn't escape the root directory.
/// Returns error if path contains `..`, is absolute, or contains null bytes.
pub(crate) fn validate_path(path: &str) -> Result<()> {
    // Check for null bytes (could be used to truncate path in some systems)
    if path.contains('\0') {
        return Err(TuneError::Storage("Path contains null byte".to_string()));
    }

    // Check for absolute path
    if Path::new(path).is_absolute() {
        return Err(TuneError::Storage(format!(
            "Path traversal attempt: absolute path not allowed: {path}"
        )));
    }

    // Check for path traversal using ..
    for component in Path::new(path).components() {
        if let std::path::Component::ParentDir = component {
            return Err(TuneError::Storage(format!(
                "Path traversal attempt: '..' not allowed in path: {path}"
            )));
        }
    }

    Ok(())
}

/// Validate model name and version don't contain path traversal characters
pub(crate) fn validate_model_identity(name: &str, version: &str) -> Result<()> {
    validate_path(name)?;
    validate_path(version)?;
    Ok(())
}

/// Calculate SHA256 hash of data and return as hex string
pub fn sha256_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    format!("{result:x}")
}

#[cfg(test)]
mod tests;
