//! LoRA adapter manifest — governance and integrity tracking.
//!
//! The manifest is a human-reviewed JSON document that records every adapter
//! produced by a training run. Loaders consult it to enforce approval status
//! and integrity before materialising weights.

use crate::error::{Result, TuneError};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Opaque adapter identifier (arbitrary string; a UUID is conventional).
pub type AdapterId = String;

/// Governance status of a LoRA adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterStatus {
    /// Adapter may participate in a mixture.
    Approved,
    /// Adapter is suspended; loader rejects it with an explicit error.
    Quarantined,
    /// Adapter is permanently disabled; loader rejects it.
    Revoked,
}

/// Metadata and governance state for one adapter in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Opaque identifier; matched against the `adapter_id` field in the
    /// adapter's safetensors header when present.
    pub id: AdapterId,
    /// Human-readable name for logging and debugging.
    pub name: String,
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
    /// Governance status. Only `Approved` adapters may be loaded.
    pub status: AdapterStatus,
}

/// Governed manifest listing admissible adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraManifest {
    /// Manifest schema version. Current: 1.
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
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).map_err(TuneError::Io)?;
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
mod tests {
    use super::*;

    fn sample_entry(id: &str, status: AdapterStatus) -> ManifestEntry {
        ManifestEntry {
            id: id.to_string(),
            name: format!("{id} test adapter"),
            uri: format!("adapters/{id}.safetensors"),
            integrity_sha256: "abc123def456".to_string(),
            base_model_rev: "main".to_string(),
            tokenizer_rev: "main".to_string(),
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            status,
        }
    }

    #[test]
    fn roundtrip_json() {
        let mut manifest = LoraManifest::new();
        manifest
            .adapters
            .push(sample_entry("adapter-1", AdapterStatus::Approved));
        manifest
            .adapters
            .push(sample_entry("adapter-2", AdapterStatus::Quarantined));

        let json = manifest.to_json().unwrap();
        let parsed = LoraManifest::from_json(&json).unwrap();
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.adapters.len(), 2);
        assert_eq!(parsed.adapters[0].id, "adapter-1");
        assert_eq!(parsed.adapters[0].status, AdapterStatus::Approved);
        assert_eq!(parsed.adapters[1].id, "adapter-2");
        assert_eq!(parsed.adapters[1].status, AdapterStatus::Quarantined);
        assert_eq!(parsed.adapters[0].rank, 8);
        assert!((parsed.adapters[0].alpha - 16.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_missing_required_field() {
        // `id` is omitted — serde must reject this
        let json = r#"{
            "version": 1,
            "adapters": [{
                "name": "test", "uri": "test.safetensors",
                "integrity_sha256": "abc",
                "base_model_rev": "none", "tokenizer_rev": "none",
                "rank": 8, "alpha": 16.0, "target_modules": ["q_proj"],
                "status": "approved"
            }]
        }"#;
        assert!(LoraManifest::from_json(json).is_err());
    }

    #[test]
    fn status_snake_case_roundtrip() {
        let approved = serde_json::to_string(&AdapterStatus::Approved).unwrap();
        assert_eq!(approved, "\"approved\"");

        let quarantined = serde_json::to_string(&AdapterStatus::Quarantined).unwrap();
        assert_eq!(quarantined, "\"quarantined\"");

        let revoked = serde_json::to_string(&AdapterStatus::Revoked).unwrap();
        assert_eq!(revoked, "\"revoked\"");

        let back: AdapterStatus = serde_json::from_str("\"approved\"").unwrap();
        assert_eq!(back, AdapterStatus::Approved);

        let back: AdapterStatus = serde_json::from_str("\"revoked\"").unwrap();
        assert_eq!(back, AdapterStatus::Revoked);
    }

    #[test]
    fn rejects_unknown_status() {
        let json = r#"{
            "id": "x", "name": "n", "uri": "u",
            "integrity_sha256": "h",
            "base_model_rev": "none", "tokenizer_rev": "none",
            "rank": 4, "alpha": 4.0,
            "target_modules": ["q_proj"],
            "status": "provisional"
        }"#;
        assert!(serde_json::from_str::<ManifestEntry>(json).is_err());
    }

    #[test]
    fn load_save_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.json");
        let mut manifest = LoraManifest::new();
        manifest
            .adapters
            .push(sample_entry("rev-adapter", AdapterStatus::Revoked));
        manifest.save(&path).unwrap();

        let loaded = LoraManifest::load(&path).unwrap();
        assert_eq!(loaded.adapters.len(), 1);
        assert_eq!(loaded.adapters[0].status, AdapterStatus::Revoked);
        assert_eq!(loaded.adapters[0].id, "rev-adapter");
        assert_eq!(loaded.adapters[0].base_model_rev, "main");
        assert_eq!(loaded.adapters[0].tokenizer_rev, "main");
    }

    #[test]
    fn default_creates_empty_v1() {
        let m = LoraManifest::default();
        assert_eq!(m.version, 1);
        assert!(m.adapters.is_empty());
    }
}
