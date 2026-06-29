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
/// 2. Resolve `uri` under `base_dir`. Absolute URIs and `..` components are
///    rejected; the resolved path is canonicalized and must stay within a
///    canonicalized `base_dir` (symlink-escape proof).
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
    // Pre-scan: verify ALL entries are Approved before touching any file or
    // allocating the output buffer. A quarantined/revoked entry anywhere in the
    // manifest must prevent all file IO — not just its own entry — so an attacker
    // cannot hide a bad entry after good ones and have the good files read first.
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

    // Allocate output capacity only after the prescan confirms all entries are
    // Approved (and the manifest size is bounded by the FIX-1a manifest cap).
    let mut out = Vec::with_capacity(manifest.adapters.len());

    for entry in &manifest.adapters {
        // Check 1: Status — defence-in-depth after the prescan above; can only
        // fire if the manifest is mutated between the prescan and this loop.
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

        // Check 2: Resolve URI and enforce path confinement.
        // Absolute URIs are rejected because a manifest is an attacker-influenced
        // governance input; an absolute URI could reach any file on the filesystem.
        // `..` components escape base_dir even with a relative path.
        // Canonicalization catches symlinks that point outside base_dir.
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
            // Canonicalize to catch symlink escapes. If the target does not yet
            // exist, canonicalize fails; pass the non-canonical joined path so
            // Check 3 reports the expected "file not found" error.
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
                    // Fail closed. A real approved adapter always exists and
                    // canonicalizes (we are about to read and hash it). A
                    // canonicalize failure means the target is missing, a path
                    // component is unreadable, or a parent is an unresolved
                    // symlink. Falling back to the lexical `joined` would read
                    // through a path the confinement check never proved in-base,
                    // so report not-found directly instead.
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

        // Check 3 (existence) is folded into Check 4: `full_path` is the
        // canonicalized, in-base path proven above, and the size-bounded read
        // opens it and fails closed if it is missing or unreadable. A separate
        // `exists()` precheck would only widen the canonicalize-to-open window.
        // The residual window (the proven path's final component being swapped to
        // a symlink between canonicalization and open) is backstopped by the
        // SHA-256 integrity check in Check 4 and grants no capability without
        // write access to base_dir, which already defeats confinement.
        //
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
        // When an adapter file omits explicit alpha metadata, the parser
        // synthesises `alpha = rank` (scale = 1.0). That synthesised value is
        // still compared here against `entry.alpha`, so a manifest that declares
        // a non-rank alpha (e.g. alpha=64, rank=16) correctly rejects an adapter
        // whose header omits alpha — the synthesised 16 ≠ declared 64. The
        // synthesis fallback therefore cannot smuggle a mismatched alpha past the
        // governed loader.
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

    /// FIX-2 mutation-sensitive: a revoked entry that appears AFTER an approved
    /// entry must cause rejection WITHOUT reading the approved entry's file.
    /// The approved entry's URI points at a path that does not exist, so if the
    /// prescan were removed the loader would attempt to read it and return a
    /// different (IO/NotFound) error. The prescan fires first — the error must
    /// contain "revoked", not "not found" or "does not exist".
    #[test]
    fn loader_prescan_rejects_revoked_without_reading_approved_file() {
        let dir = tempdir().unwrap();
        let mut manifest = LoraManifest::new();

        // approved-A: file intentionally missing — if reached, the error would
        // say "does not exist", not "revoked".
        manifest.adapters.push(make_entry(
            "approved-a",
            "approved_a_does_not_exist.safetensors",
            "dummy",
            4,
            8.0,
            AdapterStatus::Approved,
            vec!["q_proj".to_string()],
        ));
        // revoked-B appears after approved-A; prescan must reject before IO.
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
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("..") || msg.contains("traversal") || msg.contains("permitted"),
            "expected path-traversal rejection; got: {msg}"
        );
    }

    /// D4 (alpha synthesis): an adapter whose header omits alpha metadata
    /// synthesises alpha = rank. The governed loader must reject it when the
    /// manifest declares a different alpha. This test proves the alpha-agreement
    /// check in Check 7 is not bypassed by the synthesis fallback.
    ///
    /// Mutation-sensitive: removing the Check 7 alpha comparison makes this test
    /// pass (wrong), so the test fails when the guard is absent.
    #[test]
    fn loader_alpha_synthesis_mismatch_is_rejected() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("no_alpha_meta.safetensors");

        // Write an adapter with rank=16 but NO `alpha` key in the header.
        // The loader's safetensors parser will synthesise alpha = rank = 16.
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
        // Deliberately omit `alpha` from metadata so synthesis fires.
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("rank".to_string(), rank.to_string());
        meta.insert("target_modules".to_string(), "q_proj".to_string());
        let file_bytes = serialize(&tensors, &Some(meta)).unwrap();
        std::fs::write(&file_path, &file_bytes).unwrap();

        let sha = sha256_hash(&file_bytes);

        // Manifest declares alpha=64, but the file synthesises alpha=16 — mismatch.
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
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("alpha mismatch"),
            "expected alpha mismatch rejection; got: {msg}"
        );
    }

    /// FIX-3 (round 2), mutation-sensitive: a symlink inside base_dir that points
    /// to a real file OUTSIDE base_dir must not let an adapter escape confinement.
    /// The canonicalized path resolves outside base, so the loader rejects it
    /// rather than reading through the symlink. Removing the `starts_with`
    /// confinement check makes this load the outside file (then fail later on the
    /// "dummy" hash) — a different error — so this test fails when the guard is
    /// absent.
    #[cfg(unix)]
    #[test]
    fn loader_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;
        let base = tempdir().unwrap();
        let outside = tempdir().unwrap();
        // A real adapter sits OUTSIDE base_dir.
        let secret = outside.path().join("secret.safetensors");
        write_test_adapter(&secret, 4, 8.0, &["q_proj"], None);
        // base_dir/link.safetensors -> outside/secret.safetensors
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
}
