//! Fail-closed manifest-driven loader for LoRA adapters.
//!
//! `load_adapters_from_manifest` validates every approved adapter through ten
//! ordered checks, returning `Err` on the first anomaly encountered. There is
//! no partial success and no silent skip.

use super::LoraAdapter;
use super::manifest::{AdapterId, AdapterStatus, LoraManifest, ManifestEntry};
use crate::error::{Result, TuneError};
use crate::registry::sha256_hash;
use std::path::Path;

/// A successfully loaded and fully validated adapter.
#[derive(Debug)]
pub struct LoadedAdapter {
    /// Identifier from the manifest entry.
    pub id: AdapterId,
    /// The manifest entry this adapter was loaded from.
    pub entry: ManifestEntry,
    /// The loaded adapter weights.
    pub adapter: LoraAdapter,
}

/// Load and validate all approved adapters in a manifest.
///
/// Fails closed: returns `Err` on **any** anomaly. Returns
/// `Ok(Vec<LoadedAdapter>)` only when every adapter in the manifest passes all
/// checks. Quarantined and Revoked entries are rejected immediately — the
/// function does not even attempt to read their files.
///
/// # Validation checklist (all mandatory, in order)
///
/// For each `ManifestEntry` in `manifest.adapters`:
/// 1. `status == Approved` — else `Err` (quarantined/revoked message).
/// 2. Resolve `uri` against `base_dir` (relative) or use as-is (absolute).
/// 3. File exists — else `Err(TuneError::Io(...))`.
/// 4. File readable and SHA-256 of bytes == `integrity_sha256` — else `Err`.
/// 5. Bytes parse as a valid PEFT safetensors adapter — else `Err` (propagated).
/// 6. Loaded `config.rank == entry.rank` — else `Err(TuneError::Validation(...))`.
/// 7. Loaded `config.alpha` within 1e-4 of `entry.alpha` — else `Err`.
/// 8. Loaded `config.target_modules ⊆ entry.target_modules` — else `Err`.
/// 9. (When `inference-hook` feature active and `model_config` is `Some`)
///    `adapter.validate_against(model_config)` passes — else `Err`.
/// 10. If the adapter's safetensors header contains an `adapter_id` key, it
///     must equal `entry.id` — else `Err(TuneError::Validation(...))`.
///
/// # Arguments
///
/// * `manifest` — Parsed `LoraManifest`.
/// * `base_dir` — Directory for resolving relative URIs (typically the
///   manifest file's parent directory).
/// * `model_config` — (Only when `inference-hook` is active) Model config for
///   dimension validation; `None` skips step 9.
pub fn load_adapters_from_manifest(
    manifest: &LoraManifest,
    base_dir: &Path,
    #[cfg(feature = "inference-hook")] model_config: Option<
        &lattice_inference::model::qwen35_config::Qwen35Config,
    >,
) -> Result<Vec<LoadedAdapter>> {
    let mut out = Vec::with_capacity(manifest.adapters.len());

    for entry in &manifest.adapters {
        // Check 1: Status — Quarantined and Revoked are rejected without reading files.
        match entry.status {
            AdapterStatus::Quarantined => {
                return Err(TuneError::Validation(format!(
                    "adapter '{}' is quarantined — refusing to load",
                    entry.id
                )));
            }
            AdapterStatus::Revoked => {
                return Err(TuneError::Validation(format!(
                    "adapter '{}' is revoked — refusing to load",
                    entry.id
                )));
            }
            AdapterStatus::Approved => {}
        }

        // Check 2: Resolve URI against base_dir (relative) or use as-is (absolute).
        let full_path = {
            let p = Path::new(&entry.uri);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                base_dir.join(p)
            }
        };

        // Check 3: File existence.
        if !full_path.exists() {
            return Err(TuneError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "adapter '{}': file '{}' does not exist",
                    entry.id,
                    full_path.display()
                ),
            )));
        }

        // Check 4: Read bytes (size-bounded) and verify SHA-256 integrity. The
        // size bound is enforced by a metadata stat before the read, so a
        // manifest URI pointing at an oversized file is refused without
        // allocating for its contents — the path-based loader's guard, applied
        // here so this bytes path cannot bypass it.
        let bytes = crate::lora::safetensors::read_lora_file_bounded(
            &full_path,
            crate::lora::safetensors::MAX_LORA_SIZE,
        )
        .map_err(|e| TuneError::Validation(format!("adapter '{}': {e}", entry.id)))?;

        let computed_hash = sha256_hash(&bytes);
        if computed_hash != entry.integrity_sha256 {
            return Err(TuneError::Validation(format!(
                "integrity check failed for '{}': expected '{}', computed '{computed_hash}'",
                entry.id, entry.integrity_sha256
            )));
        }

        // Check 5: Parse safetensors bytes into a LoraAdapter.
        let adapter =
            crate::lora::safetensors::load_peft_safetensors_bytes(&bytes).map_err(|e| {
                TuneError::Validation(format!(
                    "adapter '{}': failed to parse safetensors: {e}",
                    entry.id
                ))
            })?;

        // Check 6: Rank must match the manifest entry.
        if adapter.config.rank != entry.rank {
            return Err(TuneError::Validation(format!(
                "rank mismatch for '{}': manifest says {}, file has {}",
                entry.id, entry.rank, adapter.config.rank
            )));
        }

        // Check 7: Alpha must match within f32 round-trip tolerance.
        if (adapter.config.alpha - entry.alpha).abs() > 1e-4 {
            return Err(TuneError::Validation(format!(
                "alpha mismatch for '{}': manifest says {}, file has {}",
                entry.id, entry.alpha, adapter.config.alpha
            )));
        }

        // Check 8: target_modules in adapter must be a subset of entry's list.
        for module in &adapter.config.target_modules {
            if !entry.target_modules.contains(module) {
                return Err(TuneError::Validation(format!(
                    "target_modules mismatch for '{}': adapter module '{}' \
                     not listed in manifest {:?}",
                    entry.id, module, entry.target_modules
                )));
            }
        }

        // Check 9: Dimension validation against model config (inference-hook only).
        #[cfg(feature = "inference-hook")]
        if let Some(cfg) = model_config {
            adapter.validate_against(cfg).map_err(|e| {
                TuneError::Validation(format!(
                    "adapter '{}': dimension validation failed: {e}",
                    entry.id
                ))
            })?;
        }

        // Check 10: If safetensors header carries `adapter_id`, it must equal entry.id.
        let header_id = crate::lora::safetensors::read_peft_header_adapter_id(&bytes);
        if let Some(ref hid) = header_id {
            if hid != &entry.id {
                return Err(TuneError::Validation(format!(
                    "adapter id mismatch: manifest id '{}' != safetensors header adapter_id '{hid}'",
                    entry.id
                )));
            }
        }

        out.push(LoadedAdapter {
            id: entry.id.clone(),
            entry: entry.clone(),
            adapter,
        });
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::manifest::LoraManifest;
    use crate::registry::sha256_hash;
    use tempfile::tempdir;

    /// Write a minimal valid PEFT safetensors file with controlled metadata.
    ///
    /// Produces a single-layer (layer 0) adapter with the given rank, alpha,
    /// and target modules. Optionally embeds `adapter_id` in the header.
    fn write_test_adapter(
        path: &Path,
        rank: usize,
        alpha: f32,
        target_modules: &[&str],
        adapter_id: Option<&str>,
    ) {
        use safetensors::Dtype;
        use safetensors::tensor::{TensorView, serialize};
        use std::collections::HashMap;

        let d_in = 4usize;
        let d_out = 4usize;

        // Build tensors; keep byte buffers alive.
        let mut byte_buffers: Vec<(String, Vec<u8>)> = Vec::new();
        for module in target_modules {
            let a_key = format!("base_model.model.model.layers.0.self_attn.{module}.lora_A.weight");
            let b_key = format!("base_model.model.model.layers.0.self_attn.{module}.lora_B.weight");
            byte_buffers.push((a_key, vec![0u8; rank * d_in * 4]));
            byte_buffers.push((b_key, vec![0u8; d_out * rank * 4]));
        }

        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (name, bytes) in &byte_buffers {
            let is_a = name.contains("lora_A");
            let shape = if is_a {
                vec![rank, d_in]
            } else {
                vec![d_out, rank]
            };
            let view = TensorView::new(Dtype::F32, shape, bytes).unwrap();
            tensors.insert(name.clone(), view);
        }

        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("rank".to_string(), rank.to_string());
        meta.insert("alpha".to_string(), alpha.to_string());
        meta.insert("target_modules".to_string(), target_modules.join(","));
        if let Some(id) = adapter_id {
            meta.insert("adapter_id".to_string(), id.to_string());
        }

        let file_bytes = serialize(&tensors, &Some(meta)).unwrap();
        std::fs::write(path, file_bytes).unwrap();
    }

    fn make_entry(
        id: &str,
        uri: &str,
        sha256: &str,
        rank: usize,
        alpha: f32,
        status: AdapterStatus,
        target_modules: Vec<String>,
    ) -> ManifestEntry {
        ManifestEntry {
            id: id.to_string(),
            name: format!("{id} test"),
            uri: uri.to_string(),
            integrity_sha256: sha256.to_string(),
            base_model_rev: "none".to_string(),
            tokenizer_rev: "none".to_string(),
            rank,
            alpha,
            target_modules,
            status,
        }
    }

    #[test]
    fn loader_rejects_quarantined() {
        let dir = tempdir().unwrap();
        let mut manifest = LoraManifest::new();
        manifest.adapters.push(make_entry(
            "quarantined-adapter",
            "q.safetensors",
            "dummy",
            8,
            16.0,
            AdapterStatus::Quarantined,
            vec!["q_proj".to_string()],
        ));
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("quarantined"),
            "expected 'quarantined' in: {msg}"
        );
    }

    #[test]
    fn loader_rejects_revoked() {
        let dir = tempdir().unwrap();
        let mut manifest = LoraManifest::new();
        manifest.adapters.push(make_entry(
            "revoked-adapter",
            "r.safetensors",
            "dummy",
            8,
            16.0,
            AdapterStatus::Revoked,
            vec!["q_proj".to_string()],
        ));
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("revoked"), "expected 'revoked' in: {msg}");
    }

    #[test]
    fn loader_rejects_missing_file() {
        let dir = tempdir().unwrap();
        let mut manifest = LoraManifest::new();
        manifest.adapters.push(make_entry(
            "missing-adapter",
            "does_not_exist.safetensors",
            "dummy",
            8,
            16.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        ));
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn loader_rejects_sha256_mismatch() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adapter.safetensors");
        write_test_adapter(&file_path, 4, 8.0, &["q_proj"], None);

        let mut manifest = LoraManifest::new();
        manifest.adapters.push(make_entry(
            "sha-mismatch-adapter",
            "adapter.safetensors",
            // deliberately wrong hash
            "0000000000000000000000000000000000000000000000000000000000000000",
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        ));
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("integrity check failed"),
            "expected integrity msg in: {msg}"
        );
    }

    #[test]
    fn loader_rejects_corrupt_safetensors() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("corrupt.safetensors");
        let garbage = b"this is not a safetensors file at all";
        std::fs::write(&file_path, garbage).unwrap();
        let sha = sha256_hash(garbage);

        let mut manifest = LoraManifest::new();
        manifest.adapters.push(make_entry(
            "corrupt-adapter",
            "corrupt.safetensors",
            &sha,
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        ));
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn loader_rejects_rank_mismatch() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adapter.safetensors");
        // File has rank=4
        write_test_adapter(&file_path, 4, 8.0, &["q_proj"], None);
        let bytes = std::fs::read(&file_path).unwrap();
        let sha = sha256_hash(&bytes);

        let mut manifest = LoraManifest::new();
        // Manifest says rank=8 — mismatch
        manifest.adapters.push(make_entry(
            "rank-mismatch-adapter",
            "adapter.safetensors",
            &sha,
            8,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        ));
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("rank mismatch"),
            "expected rank mismatch in: {msg}"
        );
    }

    #[test]
    fn loader_rejects_id_mismatch() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adapter.safetensors");
        // Write adapter with adapter_id = "file-id" in the safetensors header
        write_test_adapter(&file_path, 4, 8.0, &["q_proj"], Some("file-id"));
        let bytes = std::fs::read(&file_path).unwrap();
        let sha = sha256_hash(&bytes);

        let mut manifest = LoraManifest::new();
        // Manifest entry has a different id — check 10 must reject this
        manifest.adapters.push(make_entry(
            "manifest-id",
            "adapter.safetensors",
            &sha,
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        ));
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("mismatch"),
            "expected id mismatch message in: {msg}"
        );
    }

    #[test]
    fn loader_ok_approved_valid() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adapter.safetensors");
        write_test_adapter(&file_path, 4, 8.0, &["q_proj", "v_proj"], None);
        let bytes = std::fs::read(&file_path).unwrap();
        let sha = sha256_hash(&bytes);

        let mut manifest = LoraManifest::new();
        manifest.adapters.push(make_entry(
            "valid-adapter",
            "adapter.safetensors",
            &sha,
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string(), "v_proj".to_string()],
        ));
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
        );
        assert!(
            result.is_ok(),
            "expected Ok, got: {:?}",
            result.unwrap_err()
        );
        let loaded = result.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "valid-adapter");
        assert_eq!(loaded[0].adapter.config.rank, 4);
    }

    #[test]
    fn loader_ok_with_matching_header_id() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adapter.safetensors");
        // Header id matches entry id
        write_test_adapter(&file_path, 4, 8.0, &["q_proj"], Some("matching-id"));
        let bytes = std::fs::read(&file_path).unwrap();
        let sha = sha256_hash(&bytes);

        let mut manifest = LoraManifest::new();
        manifest.adapters.push(make_entry(
            "matching-id",
            "adapter.safetensors",
            &sha,
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        ));
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
        );
        assert!(
            result.is_ok(),
            "expected Ok, got: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn loader_empty_manifest_returns_empty_vec() {
        let dir = tempdir().unwrap();
        let manifest = LoraManifest::new();
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
