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

    /// Permissive revision context: mismatches are recorded on the loaded
    /// adapter (`rev_mismatch_overridden`) but do not fail the load. Intended
    /// for controlled migrations/backfills only — this reopens the silent
    /// quality-loss gap Check 11 exists to close.
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
    // Reject a disallowed manifest before any adapter I/O — see docs/lora-io.md.
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

    // Allocate only after the admission pre-scan.
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

        // Check 2: canonicalization must prove an untrusted URI remains in `base_dir`.
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
            // Prove both endpoints before reading — see docs/lora-io.md.
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
                    // Never fall back to an unproven lexical path.
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

        // Checks 3–4: read the proven path once; bounded I/O and SHA-256 backstop races.
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
        if adapter.config().rank != entry.rank {
            return Err(TuneError::Validation(format!(
                "rank mismatch for '{}': manifest says {}, file has {}",
                entry.id,
                entry.rank,
                adapter.config().rank
            )));
        }

        // Check 7: synthesized alpha also must match; omission cannot bypass the claim.
        validate_manifest_alpha(adapter.config().alpha, entry)?;

        // Check 8: target_modules in adapter must be a subset of entry's list.
        for module in &adapter.config().target_modules {
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
        if let Some(ref hid) = header_id
            && hid != &entry.id
        {
            return Err(TuneError::Validation(format!(
                "adapter id mismatch: manifest id '{}' != safetensors header adapter_id '{hid}'",
                entry.id
            )));
        }

        // Check 11: reject silent serving-quality loss from revision drift — see docs/lora-io.md.
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
mod tests;
