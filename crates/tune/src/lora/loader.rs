//! Fail-closed manifest-driven loading for LoRA adapters.
//!
//! Every manifest entry must be approved and pass path, size, integrity,
//! format, metadata, and optional serving-revision checks before any adapter
//! list is returned. The loader never silently skips a failed entry or returns
//! a partial result.
//! See docs/lora-io.md.

use super::LoraAdapter;
use super::manifest::{AdapterId, AdapterStatus, LoraManifest, ManifestEntry};
use crate::error::{Result, TuneError};
use crate::registry::sha256_hash;
use std::path::Path;

const ALPHA_TOLERANCE: f32 = 1e-4;

fn validate_manifest_alpha(adapter_alpha: f32, entry: &ManifestEntry) -> Result<()> {
    let difference = (adapter_alpha - entry.alpha).abs();
    if !adapter_alpha.is_finite()
        || !entry.alpha.is_finite()
        || !difference.is_finite()
        || difference > ALPHA_TOLERANCE
    {
        return Err(TuneError::Validation(format!(
            "alpha mismatch for '{}': manifest says {}, file has {}",
            entry.id, entry.alpha, adapter_alpha
        )));
    }
    Ok(())
}

/// A successfully loaded and fully validated adapter.
#[derive(Debug)]
pub struct LoadedAdapter {
    /// Identifier from the manifest entry.
    pub id: AdapterId,
    /// The manifest entry this adapter was loaded from.
    pub entry: ManifestEntry,
    /// The loaded adapter weights.
    pub adapter: LoraAdapter,
    /// Whether an allowed serving-revision mismatch bypassed the admission check.
    /// This is `false` when revisions matched or enforcement was not requested.
    /// See [`docs/lora-io.md`](../../docs/lora-io.md#runningrevisions) for serving implications.
    pub rev_mismatch_overridden: bool,
}

/// Revisions of the base model and tokenizer currently serving an adapter.
/// They are compared literally with each manifest entry when supplied to the loader.
/// See [`docs/lora-io.md`](../../docs/lora-io.md#runningrevisions) for strict and migration behavior.
#[derive(Debug, Clone, Copy)]
pub struct RunningRevisions<'a> {
    /// Revision of the base model weights currently loaded for serving.
    pub base_model_rev: &'a str,
    /// Revision of the tokenizer currently loaded for serving.
    pub tokenizer_rev: &'a str,
    /// When `true`, a revision mismatch is recorded via
    /// `LoadedAdapter::rev_mismatch_overridden` instead of failing the load.
    pub allow_mismatch: bool,
}

impl<'a> RunningRevisions<'a> {
    /// Strict (fail-closed, default-recommended) revision context: any
    /// mismatch on either field rejects the adapter.
    pub fn strict(base_model_rev: &'a str, tokenizer_rev: &'a str) -> Self {
        Self {
            base_model_rev,
            tokenizer_rev,
            allow_mismatch: false,
        }
    }

    /// Permit revision mismatches for controlled migrations or backfills.
    /// The loaded adapter records that this bypassed the usual rejection.
    pub fn permissive(base_model_rev: &'a str, tokenizer_rev: &'a str) -> Self {
        Self {
            base_model_rev,
            tokenizer_rev,
            allow_mismatch: true,
        }
    }
}

/// Load every approved manifest adapter or reject the entire manifest.
/// Validates confinement, integrity, format, manifest claims, and optional live-serving context.
/// See [`docs/lora-io.md`](../../docs/lora-io.md#load_adapters_from_manifest) for the ordered admission checks.
pub fn load_adapters_from_manifest(
    manifest: &LoraManifest,
    base_dir: &Path,
    #[cfg(feature = "inference-hook")] model_config: Option<
        &lattice_inference::model::qwen35_config::Qwen35Config,
    >,
    running: Option<&RunningRevisions<'_>>,
) -> Result<Vec<LoadedAdapter>> {
    // Reject unapproved entries before any adapter I/O.
    for entry in &manifest.adapters {
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
    }

    let mut out = Vec::with_capacity(manifest.adapters.len());

    for entry in &manifest.adapters {
        // Recheck status in case the manifest changes after the pre-scan.
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

        // Canonicalization must prove an untrusted URI remains in `base_dir`.
        let full_path = {
            let p = Path::new(&entry.uri);
            if p.is_absolute() {
                return Err(TuneError::Validation(format!(
                    "adapter '{}': URI '{}' is absolute — only relative paths are permitted",
                    entry.id, entry.uri
                )));
            }
            for component in p.components() {
                if component == std::path::Component::ParentDir {
                    return Err(TuneError::Validation(format!(
                        "adapter '{}': URI '{}' contains '..' — path traversal is not permitted",
                        entry.id, entry.uri
                    )));
                }
            }
            let joined = base_dir.join(p);
            let canonical_base = std::fs::canonicalize(base_dir).map_err(|e| {
                TuneError::Io(std::io::Error::new(
                    e.kind(),
                    format!(
                        "adapter '{}': failed to canonicalize base_dir '{}': {e}",
                        entry.id,
                        base_dir.display()
                    ),
                ))
            })?;
            match std::fs::canonicalize(&joined) {
                Ok(canonical_joined) => {
                    if !canonical_joined.starts_with(&canonical_base) {
                        return Err(TuneError::Validation(format!(
                            "adapter '{}': resolved path '{}' escapes base directory '{}'",
                            entry.id,
                            canonical_joined.display(),
                            canonical_base.display()
                        )));
                    }
                    canonical_joined
                }
                Err(e) => {
                    return Err(TuneError::Io(std::io::Error::new(
                        e.kind(),
                        format!(
                            "adapter '{}': file '{}' does not exist or is not accessible within base directory",
                            entry.id,
                            joined.display()
                        ),
                    )));
                }
            }
        };

        // Bound the verified-path read, then verify its checksum.
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

        let adapter =
            crate::lora::safetensors::load_peft_safetensors_bytes(&bytes).map_err(|e| {
                TuneError::Validation(format!(
                    "adapter '{}': failed to parse safetensors: {e}",
                    entry.id
                ))
            })?;

        if adapter.config().rank != entry.rank {
            return Err(TuneError::Validation(format!(
                "rank mismatch for '{}': manifest says {}, file has {}",
                entry.id,
                entry.rank,
                adapter.config().rank
            )));
        }

        // Synthesized alpha must also match the manifest claim.
        validate_manifest_alpha(adapter.config().alpha, entry)?;

        for module in &adapter.config().target_modules {
            if !entry.target_modules.contains(module) {
                return Err(TuneError::Validation(format!(
                    "target_modules mismatch for '{}': adapter module '{}' \
                     not listed in manifest {:?}",
                    entry.id, module, entry.target_modules
                )));
            }
        }

        #[cfg(feature = "inference-hook")]
        if let Some(cfg) = model_config {
            adapter.validate_against(cfg).map_err(|e| {
                TuneError::Validation(format!(
                    "adapter '{}': dimension validation failed: {e}",
                    entry.id
                ))
            })?;
        }

        let header_id = crate::lora::safetensors::read_peft_header_adapter_id(&bytes);
        if let Some(ref hid) = header_id
            && hid != &entry.id
        {
            return Err(TuneError::Validation(format!(
                "adapter id mismatch: manifest id '{}' != safetensors header adapter_id '{hid}'",
                entry.id
            )));
        }

        // Revision drift can silently degrade serving quality.
        let mut rev_mismatch_overridden = false;
        if let Some(running) = running {
            let mut mismatches = Vec::new();
            if entry.base_model_rev != running.base_model_rev {
                mismatches.push(format!(
                    "base_model_rev (manifest '{}' != running '{}')",
                    entry.base_model_rev, running.base_model_rev
                ));
            }
            if entry.tokenizer_rev != running.tokenizer_rev {
                mismatches.push(format!(
                    "tokenizer_rev (manifest '{}' != running '{}')",
                    entry.tokenizer_rev, running.tokenizer_rev
                ));
            }
            if !mismatches.is_empty() {
                if running.allow_mismatch {
                    rev_mismatch_overridden = true;
                } else {
                    return Err(TuneError::Validation(format!(
                        "revision mismatch for '{}': {}",
                        entry.id,
                        mismatches.join(", ")
                    )));
                }
            }
        }

        out.push(LoadedAdapter {
            id: entry.id.clone(),
            entry: entry.clone(),
            adapter,
            rev_mismatch_overridden,
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
            owner: "test-owner".to_string(),
            uri: uri.to_string(),
            integrity_sha256: sha256.to_string(),
            base_model_rev: "none".to_string(),
            tokenizer_rev: "none".to_string(),
            rank,
            alpha,
            target_modules,
            dtype: "f32".to_string(),
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
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn loader_rejects_rank_mismatch() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adapter.safetensors");
        write_test_adapter(&file_path, 4, 8.0, &["q_proj"], None);
        let bytes = std::fs::read(&file_path).unwrap();
        let sha = sha256_hash(&bytes);

        let mut manifest = LoraManifest::new();
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
        write_test_adapter(&file_path, 4, 8.0, &["q_proj"], Some("file-id"));
        let bytes = std::fs::read(&file_path).unwrap();
        let sha = sha256_hash(&bytes);

        let mut manifest = LoraManifest::new();
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
        assert_eq!(loaded[0].adapter.config().rank, 4);
    }

    #[test]
    fn loader_ok_with_matching_header_id() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adapter.safetensors");
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
            None,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    /// A later revoked entry must reject the manifest before earlier adapter I/O.
    #[test]
    fn loader_prescan_rejects_revoked_without_reading_approved_file() {
        let dir = tempdir().unwrap();
        let mut manifest = LoraManifest::new();

        // This missing file proves the pre-scan runs before adapter I/O.
        manifest.adapters.push(make_entry(
            "approved-a",
            "approved_a_does_not_exist.safetensors",
            "dummy",
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        ));
        manifest.adapters.push(make_entry(
            "revoked-b",
            "revoked_b.safetensors",
            "dummy",
            4,
            8.0,
            AdapterStatus::Revoked,
            vec!["q_proj".to_string()],
        ));

        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
            None,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("revoked"),
            "expected prescan 'revoked' rejection, not an IO error; got: {msg}"
        );
        assert!(
            !msg.contains("does not exist"),
            "prescan fired too late — loader touched approved-A's (nonexistent) file; got: {msg}"
        );
    }

    /// FIX-3: absolute URIs must be rejected.
    #[test]
    fn loader_rejects_absolute_uri() {
        let dir = tempdir().unwrap();
        let mut manifest = LoraManifest::new();
        manifest.adapters.push(make_entry(
            "abs-adapter",
            "/etc/passwd",
            "dummy",
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
            None,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("absolute"),
            "expected 'absolute' rejection; got: {msg}"
        );
    }

    /// FIX-3: URIs containing `..` must be rejected.
    #[test]
    fn loader_rejects_dotdot_uri() {
        let dir = tempdir().unwrap();
        let mut manifest = LoraManifest::new();
        manifest.adapters.push(make_entry(
            "traversal-adapter",
            "../escape/adapter.safetensors",
            "dummy",
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
            None,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("..") || msg.contains("traversal") || msg.contains("permitted"),
            "expected path-traversal rejection; got: {msg}"
        );
    }

    /// Manifest alpha must match the rank-derived fallback for omitted metadata.
    #[test]
    fn loader_alpha_synthesis_mismatch_is_rejected() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("no_alpha_meta.safetensors");

        // Omitted metadata makes the parser synthesize `alpha = rank`.
        use safetensors::Dtype;
        use safetensors::tensor::{TensorView, serialize};
        use std::collections::HashMap;

        let rank: usize = 16;
        let d_in: usize = 4;
        let d_out: usize = 4;
        let a_bytes = vec![0u8; rank * d_in * 4];
        let b_bytes = vec![0u8; d_out * rank * 4];
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            TensorView::new(Dtype::F32, vec![rank, d_in], &a_bytes).unwrap(),
        );
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            TensorView::new(Dtype::F32, vec![d_out, rank], &b_bytes).unwrap(),
        );
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("rank".to_string(), rank.to_string());
        meta.insert("target_modules".to_string(), "q_proj".to_string());
        let file_bytes = serialize(&tensors, &Some(meta)).unwrap();
        std::fs::write(&file_path, &file_bytes).unwrap();

        let sha = sha256_hash(&file_bytes);

        let mut manifest = LoraManifest::new();
        manifest.adapters.push(make_entry(
            "alpha-mismatch",
            "no_alpha_meta.safetensors",
            &sha,
            rank,
            64.0, // declared alpha != synthesised 16
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        ));

        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
            None,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("alpha mismatch"),
            "expected alpha mismatch rejection; got: {msg}"
        );
    }

    #[test]
    fn loader_rejects_nan_adapter_alpha_against_finite_manifest_alpha() {
        let entry = make_entry(
            "nan-alpha",
            "adapter.safetensors",
            "unused",
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        );

        let err = validate_manifest_alpha(f32::NAN, &entry).unwrap_err();
        assert!(err.to_string().contains("alpha mismatch"));
    }

    /// A symlink within `base_dir` must not escape its canonicalized boundary.
    #[cfg(unix)]
    #[test]
    fn loader_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;
        let base = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let secret = outside.path().join("secret.safetensors");
        write_test_adapter(&secret, 4, 8.0, &["q_proj"], None);
        let link = base.path().join("link.safetensors");
        symlink(&secret, &link).unwrap();

        let mut manifest = LoraManifest::new();
        manifest.adapters.push(make_entry(
            "escape",
            "link.safetensors",
            "dummy",
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        ));
        let result = load_adapters_from_manifest(
            &manifest,
            base.path(),
            #[cfg(feature = "inference-hook")]
            None,
            None,
        );
        assert!(
            result.is_err(),
            "symlink escaping base_dir must be rejected"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("escapes base directory"),
            "expected escape rejection; got: {msg}"
        );
    }

    /// A base-model revision mismatch must reject an otherwise valid adapter.
    #[test]
    fn loader_rejects_base_model_rev_mismatch() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adapter.safetensors");
        write_test_adapter(&file_path, 4, 8.0, &["q_proj"], None);
        let bytes = std::fs::read(&file_path).unwrap();
        let sha = sha256_hash(&bytes);

        let mut manifest = LoraManifest::new();
        let mut entry = make_entry(
            "rev-adapter",
            "adapter.safetensors",
            &sha,
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        );
        entry.base_model_rev = "trained-rev-a".to_string();
        entry.tokenizer_rev = "tok-rev-a".to_string();
        manifest.adapters.push(entry);

        let running = RunningRevisions::strict("running-rev-b", "tok-rev-a");
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
            Some(&running),
        );
        assert!(result.is_err(), "expected Err, got Ok");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("revision mismatch") && msg.contains("base_model_rev"),
            "expected revision-mismatch/base_model_rev in: {msg}"
        );
    }

    /// A tokenizer revision mismatch must also reject the adapter.
    #[test]
    fn loader_rejects_tokenizer_rev_mismatch() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adapter.safetensors");
        write_test_adapter(&file_path, 4, 8.0, &["q_proj"], None);
        let bytes = std::fs::read(&file_path).unwrap();
        let sha = sha256_hash(&bytes);

        let mut manifest = LoraManifest::new();
        let mut entry = make_entry(
            "rev-adapter",
            "adapter.safetensors",
            &sha,
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        );
        entry.base_model_rev = "trained-rev-a".to_string();
        entry.tokenizer_rev = "tok-rev-a".to_string();
        manifest.adapters.push(entry);

        let running = RunningRevisions::strict("trained-rev-a", "tok-rev-mismatch");
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
            Some(&running),
        );
        assert!(result.is_err(), "expected Err, got Ok");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("revision mismatch") && msg.contains("tokenizer_rev"),
            "expected revision-mismatch/tokenizer_rev in: {msg}"
        );
    }

    /// Matching revisions remain admissible.
    #[test]
    fn loader_accepts_matching_revs() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adapter.safetensors");
        write_test_adapter(&file_path, 4, 8.0, &["q_proj"], None);
        let bytes = std::fs::read(&file_path).unwrap();
        let sha = sha256_hash(&bytes);

        let mut manifest = LoraManifest::new();
        let mut entry = make_entry(
            "rev-adapter",
            "adapter.safetensors",
            &sha,
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        );
        entry.base_model_rev = "trained-rev-a".to_string();
        entry.tokenizer_rev = "tok-rev-a".to_string();
        manifest.adapters.push(entry);

        let running = RunningRevisions::strict("trained-rev-a", "tok-rev-a");
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
            Some(&running),
        );
        assert!(
            result.is_ok(),
            "expected Ok, got: {:?}",
            result.unwrap_err()
        );
        assert!(!result.unwrap()[0].rev_mismatch_overridden);
    }

    /// `RunningRevisions::permissive` lets a mismatch through but must record
    /// it on `LoadedAdapter::rev_mismatch_overridden` — the override is
    /// observable, not silent.
    #[test]
    fn loader_allows_rev_mismatch_with_override_flag() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adapter.safetensors");
        write_test_adapter(&file_path, 4, 8.0, &["q_proj"], None);
        let bytes = std::fs::read(&file_path).unwrap();
        let sha = sha256_hash(&bytes);

        let mut manifest = LoraManifest::new();
        let mut entry = make_entry(
            "rev-adapter",
            "adapter.safetensors",
            &sha,
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        );
        entry.base_model_rev = "trained-rev-a".to_string();
        entry.tokenizer_rev = "tok-rev-a".to_string();
        manifest.adapters.push(entry);

        let running = RunningRevisions::permissive("running-rev-b", "tok-rev-mismatch");
        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
            Some(&running),
        );
        assert!(
            result.is_ok(),
            "expected Ok (override), got: {:?}",
            result.unwrap_err()
        );
        assert!(result.unwrap()[0].rev_mismatch_overridden);
    }

    /// `running: None` must skip revision enforcement entirely — every other
    /// existing test in this module relies on this (they all pass `None`),
    /// this test makes the opt-in semantics explicit with revs that would
    /// otherwise mismatch.
    #[test]
    fn loader_skips_rev_enforcement_when_running_is_none() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("adapter.safetensors");
        write_test_adapter(&file_path, 4, 8.0, &["q_proj"], None);
        let bytes = std::fs::read(&file_path).unwrap();
        let sha = sha256_hash(&bytes);

        let mut manifest = LoraManifest::new();
        let mut entry = make_entry(
            "rev-adapter",
            "adapter.safetensors",
            &sha,
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        );
        entry.base_model_rev = "trained-rev-a".to_string();
        entry.tokenizer_rev = "tok-rev-a".to_string();
        manifest.adapters.push(entry);

        let result = load_adapters_from_manifest(
            &manifest,
            dir.path(),
            #[cfg(feature = "inference-hook")]
            None,
            None,
        );
        assert!(
            result.is_ok(),
            "expected Ok (no enforcement requested), got: {:?}",
            result.unwrap_err()
        );
        assert!(!result.unwrap()[0].rev_mismatch_overridden);
    }
}
