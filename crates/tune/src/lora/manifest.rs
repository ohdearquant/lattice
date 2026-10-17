//! Versioned LoRA manifest schema and bounded JSON I/O.
//!
//! A manifest records adapter provenance, integrity data, and one authoritative
//! status. Only approved entries are admissible to the governed loader.
//!
//! See `docs/lora-core.md` for schema version 1 and fail-closed loading rules.

use crate::error::{Result, TuneError};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Maximum on-disk size of the manifest JSON file, in bytes (64 MiB).
///
/// This cap is enforced before parsing and again with a bounded read sentinel.
const MAX_MANIFEST_SIZE: u64 = 64 * 1024 * 1024;

/// Opaque adapter identifier (arbitrary string; a UUID is conventional).
pub type AdapterId = String;

/// Governance status of a LoRA adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AdapterStatus {
    /// Adapter may participate in a mixture.
    Approved,
    /// Adapter is suspended; loader rejects it with an explicit error.
    Quarantined,
    /// Adapter is permanently disabled; loader rejects it.
    Revoked,
}

impl AdapterStatus {
    /// Lowercase `snake_case` string form, matching the serde wire format
    /// (`"approved"` / `"quarantined"` / `"revoked"`). Used where a plain
    /// string is needed outside of JSON (de)serialization, e.g. embedding
    /// status into a safetensors metadata header.
    pub fn as_str(&self) -> &'static str {
        match self {
            AdapterStatus::Approved => "approved",
            AdapterStatus::Quarantined => "quarantined",
            AdapterStatus::Revoked => "revoked",
        }
    }
}

/// Metadata and governance state for one adapter in the manifest.
///
/// `status` is the sole approval authority; a separate `approved` boolean
/// would create conflicting sources of truth. See ADR-045 and
/// `docs/lora-core.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Opaque identifier; matched against the `adapter_id` field in the
    /// adapter's safetensors header when present.
    pub id: AdapterId,
    /// Human-readable name for logging and debugging.
    pub name: String,
    /// Owning team or individual responsible for this adapter (free-form,
    /// e.g. `"team-inference"` or an email). Provenance only; not enforced
    /// by the loader.
    pub owner: String,
    /// File path or URI for the safetensors file. Relative paths are resolved
    /// from the manifest file's parent directory.
    pub uri: String,
    /// SHA-256 hex digest of the file bytes (computed by `sha256_hash`).
    pub integrity_sha256: String,
    /// Git rev of the base model weights used during training, or `"none"`.
    pub base_model_rev: String,
    /// Git rev of the tokenizer used during training, or `"none"`.
    pub tokenizer_rev: String,
    /// LoRA rank (must match what the file contains; validated at load time).
    pub rank: usize,
    /// LoRA alpha.
    pub alpha: f32,
    /// Target modules this adapter covers (e.g. `["q_proj", "v_proj"]`).
    pub target_modules: Vec<String>,
    /// Tensor dtype the adapter was trained/saved in (free-form label, e.g.
    /// `"f32"`, `"f16"`, `"bf16"`). Provenance only; not cross-checked
    /// against the actual tensor bytes at load time (the safetensors parse
    /// in Check 5 validates real tensor shapes/dtypes independently).
    pub dtype: String,
    /// Governance status. Only `Approved` adapters may be loaded.
    pub status: AdapterStatus,
}

/// Governed manifest listing admissible adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraManifest {
    /// Manifest schema version. New manifests use 1; parsing preserves the
    /// supplied value without enforcing a supported-version set.
    pub version: u32,
    /// All adapter entries. The loader iterates this list; order is not significant.
    pub adapters: Vec<ManifestEntry>,
}

impl LoraManifest {
    /// Create a new empty manifest at schema version 1.
    pub fn new() -> Self {
        LoraManifest {
            version: 1,
            adapters: Vec::new(),
        }
    }

    /// Parse a manifest from a JSON string.
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| TuneError::Serialization(e.to_string()))
    }

    /// Serialize to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| TuneError::Serialization(e.to_string()))
    }

    /// Load from a file path.
    ///
    /// Reads the file with a hard cap of `MAX_MANIFEST_SIZE` bytes to prevent
    /// allocation-DoS from an attacker-supplied or symlinked giant path.
    pub fn load(path: &Path) -> Result<Self> {
        use std::io::Read;
        // Fast-path: stat before open (cheap; avoids opening a known-huge file).
        let file_size = std::fs::metadata(path).map_err(TuneError::Io)?.len();
        if file_size > MAX_MANIFEST_SIZE {
            return Err(TuneError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "manifest file is {file_size} bytes, exceeds maximum of {MAX_MANIFEST_SIZE} bytes"
                ),
            )));
        }
        // Bounded read: +1 sentinel detects a file that grew after the stat.
        let f = std::fs::File::open(path).map_err(TuneError::Io)?;
        let mut buf = Vec::new();
        f.take(MAX_MANIFEST_SIZE + 1)
            .read_to_end(&mut buf)
            .map_err(TuneError::Io)?;
        if buf.len() as u64 > MAX_MANIFEST_SIZE {
            return Err(TuneError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("manifest file exceeds maximum of {MAX_MANIFEST_SIZE} bytes at read time"),
            )));
        }
        let data = String::from_utf8(buf)
            .map_err(|e| TuneError::Validation(format!("manifest file is not valid UTF-8: {e}")))?;
        Self::from_json(&data)
    }

    /// Save to a file path (pretty-printed JSON, UTF-8).
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(TuneError::Io)
    }
}

impl Default for LoraManifest {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
