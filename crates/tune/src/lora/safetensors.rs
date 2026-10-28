//! PEFT/MLX safetensors import and PEFT-compatible export for LoRA adapters.
//!
//! A factor pair is keyed by transformer layer and projection: A is
//! `(rank, d_in)` and B is `(d_out, rank)` after MLX normalization. Parsing
//! rejects incomplete, malformed, non-finite, or rank-inconsistent pairs.
//! File reads are bounded before parsing. The format code lives in
//! `lattice_lora::codec`; these wrappers keep `TuneError` and its messages.
//! See docs/lora-io.md.

use super::{LoraAdapter, LoraConfig};
use crate::error::TuneError;
use lattice_lora::LoraError;
use lattice_lora::codec;
use std::collections::HashMap;
use std::path::Path;

/// Map a codec error into the category this module has always returned.
fn from_codec(e: LoraError) -> TuneError {
    match e {
        LoraError::Io(e) => TuneError::Io(e),
        LoraError::Validation(msg) => TuneError::Validation(msg),
        other => TuneError::Serialization(other.message()),
    }
}

/// Parse raw PEFT or MLX safetensors bytes into a complete LoRA adapter.
/// The caller supplies file-path and integrity context when needed.
/// See [`docs/lora-io.md`](../../docs/lora-io.md#load_peft_safetensors_bytes) for parsing invariants.
pub(crate) fn load_peft_safetensors_bytes(bytes: &[u8]) -> Result<LoraAdapter, TuneError> {
    let decoded = codec::decode_peft_safetensors(bytes).map_err(from_codec)?;
    let config = LoraConfig {
        rank: decoded.rank,
        alpha: decoded.alpha,
        target_modules: decoded.target_modules,
        dtype: "f32".into(),
    };
    LoraAdapter::new(config, decoded.layers)
}

/// Read the `adapter_id` metadata key from a PEFT safetensors byte slice.
///
/// Returns `None` if the bytes cannot be parsed, the header has no metadata,
/// or the `adapter_id` key is absent. Used by the manifest loader to enforce
/// the id consistency check (loader check 10).
pub(crate) fn read_peft_header_adapter_id(data: &[u8]) -> Option<String> {
    codec::read_header_metadata_value(data, "adapter_id")
}

/// Provenance metadata embeddable in an adapter safetensors header.
/// Whole-file integrity remains manifest-owned because a file cannot embed its own digest.
/// See [`docs/lora-io.md`](../../docs/lora-io.md#adaptergovernance) for header and integrity boundaries.
#[derive(Debug, Clone)]
pub struct AdapterGovernance {
    /// Human-readable name for logging and debugging.
    pub name: String,
    /// Owning team or individual responsible for this adapter.
    pub owner: String,
    /// Git rev of the base model weights used during training.
    pub base_model_rev: String,
    /// Git rev of the tokenizer used during training.
    pub tokenizer_rev: String,
    /// Tensor dtype label (e.g. `"f32"`, `"f16"`, `"bf16"`).
    pub dtype: String,
    /// Lowercase governance status, kept as a string so this type does not require `serde`.
    pub status: String,
}

impl AdapterGovernance {
    /// Build governance metadata from a manifest entry (requires `serde`,
    /// since `ManifestEntry` lives in the `serde`-gated `manifest` module).
    /// This is the common way to construct this type when saving an adapter
    /// that already has a governed manifest entry.
    #[cfg(feature = "serde")]
    pub fn from_entry(entry: &crate::lora::manifest::ManifestEntry) -> Self {
        Self {
            name: entry.name.clone(),
            owner: entry.owner.clone(),
            base_model_rev: entry.base_model_rev.clone(),
            tokenizer_rev: entry.tokenizer_rev.clone(),
            dtype: entry.dtype.clone(),
            status: entry.status.as_str().to_string(),
        }
    }
}

/// Read optional, advisory governance metadata from a safetensors header.
/// `gov_name` is the presence sentinel for a writer-produced six-field set.
/// See [`docs/lora-io.md`](../../docs/lora-io.md#adaptergovernance) for advisory-read semantics.
#[allow(dead_code)]
pub(crate) fn read_peft_header_governance(data: &[u8]) -> Option<AdapterGovernance> {
    let m = codec::read_header_metadata(data)?;
    Some(AdapterGovernance {
        name: m.get("gov_name")?.clone(),
        owner: m.get("gov_owner").cloned().unwrap_or_default(),
        base_model_rev: m.get("gov_base_model_rev").cloned().unwrap_or_default(),
        tokenizer_rev: m.get("gov_tokenizer_rev").cloned().unwrap_or_default(),
        dtype: m.get("gov_dtype").cloned().unwrap_or_default(),
        status: m.get("gov_status").cloned().unwrap_or_default(),
    })
}

/// Maximum in-memory adapter file size (10 GiB).
/// See [`docs/lora-io.md`](../../docs/lora-io.md#read_lora_file_bounded) for the allocation bound.
pub(crate) const MAX_LORA_SIZE: u64 = codec::MAX_ADAPTER_FILE_BYTES;

/// Read an adapter file into memory without exceeding `max_bytes`.
/// Returns I/O errors for metadata, over-limit, open, or read failures.
/// See [`docs/lora-io.md`](../../docs/lora-io.md#read_lora_file_bounded) for the two-stage bound.
pub(crate) fn read_lora_file_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, TuneError> {
    codec::read_adapter_file_bounded(path, max_bytes).map_err(from_codec)
}

/// Load and validate a PEFT or MLX safetensors LoRA adapter from `path`.
/// Rejects unreadable, malformed, incomplete, or shape-inconsistent files.
/// See [`docs/lora-io.md`](../../docs/lora-io.md#load_peft_safetensors) for parser checks.
pub fn load_peft_safetensors(path: &Path) -> Result<LoraAdapter, TuneError> {
    let data = read_lora_file_bounded(path, MAX_LORA_SIZE)?;
    load_peft_safetensors_bytes(&data)
}

/// Save an adapter as PEFT safetensors, optionally embedding provenance metadata.
/// Returns errors for invalid tensor views or file writes.
/// See [`docs/lora-io.md`](../../docs/lora-io.md#save_peft_safetensors) for keys and header fields.
pub fn save_peft_safetensors(
    adapter: &LoraAdapter,
    path: &Path,
    governance: Option<&AdapterGovernance>,
) -> Result<(), TuneError> {
    adapter.config().validate()?;

    let mut metadata_map = HashMap::new();
    metadata_map.insert("rank".to_string(), adapter.config().rank.to_string());
    metadata_map.insert("alpha".to_string(), adapter.config().alpha.to_string());
    metadata_map.insert(
        "target_modules".to_string(),
        adapter.config().target_modules.join(","),
    );
    if let Some(governance) = governance {
        metadata_map.insert("gov_name".to_string(), governance.name.clone());
        metadata_map.insert("gov_owner".to_string(), governance.owner.clone());
        metadata_map.insert(
            "gov_base_model_rev".to_string(),
            governance.base_model_rev.clone(),
        );
        metadata_map.insert(
            "gov_tokenizer_rev".to_string(),
            governance.tokenizer_rev.clone(),
        );
        metadata_map.insert("gov_dtype".to_string(), governance.dtype.clone());
        metadata_map.insert("gov_status".to_string(), governance.status.clone());
    }
    let bytes = codec::encode_peft_safetensors(
        adapter
            .layers()
            .iter()
            .map(|((layer_idx, module), layer)| (*layer_idx, module.as_str(), layer)),
        metadata_map,
    )
    .map_err(from_codec)?;

    codec::write_adapter_file(path, &bytes).map_err(from_codec)?;

    Ok(())
}

#[cfg(test)]
mod tests;
