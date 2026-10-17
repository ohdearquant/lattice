//! PEFT/MLX safetensors import and PEFT-compatible export for LoRA adapters.
//!
//! A factor pair is keyed by transformer layer and projection: A is
//! `(rank, d_in)` and B is `(d_out, rank)` after MLX normalization. Parsing
//! rejects incomplete, malformed, non-finite, or rank-inconsistent pairs.
//! File reads are bounded before parsing.
//! See docs/lora-io.md.

use super::{LoraAdapter, LoraConfig, LoraLayer};
use crate::error::TuneError;
use safetensors::Dtype;
use safetensors::tensor::{SafeTensors, TensorView, serialize};
use std::collections::HashMap;
use std::path::Path;

/// A parsed PEFT tensor key.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PeftKey {
    /// Transformer layer index (0-based).
    layer_idx: usize,
    /// Module name: "q_proj", "v_proj", "gate_proj", etc.
    module: String,
    /// Whether this is the A or B matrix.
    matrix: LoraMatrix,
    /// Whether the tensor needs transposing (MLX format: A=(d_in, rank), B=(rank, d_out)).
    transposed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoraMatrix {
    A,
    B,
}

/// Parse a recognized PEFT or MLX LoRA tensor key, ignoring non-factor keys.
fn parse_peft_key(key: &str) -> Option<PeftKey> {
    // Strip the trailing `.weight` if present
    let key = key.strip_suffix(".weight").unwrap_or(key);

    // Determine A or B (handles both PEFT uppercase and MLX lowercase)
    let (key, matrix, is_mlx) = if let Some(k) = key.strip_suffix(".lora_A") {
        (k, LoraMatrix::A, false)
    } else if let Some(k) = key.strip_suffix(".lora_B") {
        (k, LoraMatrix::B, false)
    } else if let Some(k) = key.strip_suffix(".lora_a") {
        (k, LoraMatrix::A, true)
    } else if let Some(k) = key.strip_suffix(".lora_b") {
        (k, LoraMatrix::B, true)
    } else {
        return None;
    };

    // Find "layers.{i}" segment and extract what follows
    let layers_marker = ".layers.";
    let layers_pos = key.find(layers_marker)?;
    let after_layers = &key[layers_pos + layers_marker.len()..];

    // Split: "{i}.{rest}" where rest is like "self_attn.q_proj" or "mlp.gate_proj"
    let dot_pos = after_layers.find('.')?;
    let layer_idx: usize = after_layers[..dot_pos].parse().ok()?;
    let rest = &after_layers[dot_pos + 1..];

    // Extract the module name (last segment after the block qualifier).
    // "self_attn.q_proj" -> "q_proj"
    // "mlp.gate_proj" -> "gate_proj"
    let module = rest.rsplit('.').next()?.to_string();
    if module.is_empty() {
        return None;
    }

    Some(PeftKey {
        layer_idx,
        module,
        matrix,
        transposed: is_mlx,
    })
}

/// Read a tensor from a safetensors file as a Vec<f32>.
///
/// Handles f32 and f16 dtypes (converting f16 to f32).
fn read_tensor_f32(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), TuneError> {
    let tensor = tensors
        .tensor(name)
        .map_err(|e| TuneError::Serialization(format!("failed to read tensor '{name}': {e}")))?;

    let shape: Vec<usize> = tensor.shape().to_vec();
    let data = tensor.data();

    let values: Vec<f32> = match tensor.dtype() {
        Dtype::F32 => {
            if data.len() % 4 != 0 {
                return Err(TuneError::Serialization(format!(
                    "tensor '{name}' f32 data length {} not aligned to 4 bytes",
                    data.len()
                )));
            }
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        Dtype::F16 => {
            if data.len() % 2 != 0 {
                return Err(TuneError::Serialization(format!(
                    "tensor '{name}' f16 data length {} not aligned to 2 bytes",
                    data.len()
                )));
            }
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f16_to_f32(bits)
                })
                .collect()
        }
        Dtype::BF16 => {
            if data.len() % 2 != 0 {
                return Err(TuneError::Serialization(format!(
                    "tensor '{name}' bf16 data length {} not aligned to 2 bytes",
                    data.len()
                )));
            }
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    bf16_to_f32(bits)
                })
                .collect()
        }
        other => {
            return Err(TuneError::Serialization(format!(
                "tensor '{name}' has unsupported dtype {other:?}, expected F32/F16/BF16"
            )));
        }
    };

    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(TuneError::Serialization(format!(
            "tensor '{name}' contains non-finite value at index {idx}: {value}"
        )));
    }

    Ok((values, shape))
}

/// Convert IEEE 754 half-precision (f16) to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: value = (-1)^sign * 2^(-14) * (frac / 1024)
            let val = (frac as f32) / 1024.0 * (2.0f32).powi(-14);
            if sign == 1 { -val } else { val }
        }
    } else if exp == 31 {
        // Inf or NaN
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
        }
    } else {
        // Normalized: re-bias exponent from 15 to 127
        let f32_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
    }
}

/// Convert bfloat16 to f32 (simply shift left by 16 bits).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Parse raw PEFT or MLX safetensors bytes into a complete LoRA adapter.
/// The caller supplies file-path and integrity context when needed.
/// See [`docs/lora-io.md`](../../docs/lora-io.md#load_peft_safetensors_bytes) for parsing invariants.
pub(crate) fn load_peft_safetensors_bytes(bytes: &[u8]) -> Result<LoraAdapter, TuneError> {
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|e| TuneError::Serialization(format!("failed to parse safetensors: {e}")))?;

    // Collect all parsed LoRA keys
    let names: Vec<String> = tensors.names().into_iter().map(String::from).collect();
    let mut a_tensors: HashMap<(usize, String), (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut b_tensors: HashMap<(usize, String), (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut target_modules = std::collections::BTreeSet::new();

    for name in &names {
        if let Some(peft_key) = parse_peft_key(name) {
            let key = (peft_key.layer_idx, peft_key.module.clone());
            let (data, shape) = read_tensor_f32(&tensors, name)?;
            target_modules.insert(peft_key.module.clone());

            // MLX format stores transposed: A=(d_in, rank), B=(rank, d_out)
            // PEFT format (what we expect): A=(rank, d_in), B=(d_out, rank)
            let (data, shape) = if peft_key.transposed && shape.len() == 2 {
                let (rows, cols) = (shape[0], shape[1]);
                let expected = rows.checked_mul(cols).ok_or_else(|| {
                    TuneError::Serialization(format!(
                        "tensor '{name}' shape [{rows}, {cols}] overflows usize"
                    ))
                })?;
                if data.len() != expected {
                    return Err(TuneError::Serialization(format!(
                        "tensor '{name}' shape [{rows}, {cols}] implies {expected} elements but data has {}",
                        data.len()
                    )));
                }
                let mut transposed = vec![0.0f32; data.len()];
                for r in 0..rows {
                    for c in 0..cols {
                        transposed[c * rows + r] = data[r * cols + c];
                    }
                }
                (transposed, vec![cols, rows])
            } else {
                (data, shape)
            };

            match peft_key.matrix {
                LoraMatrix::A => {
                    a_tensors.insert(key, (data, shape));
                }
                LoraMatrix::B => {
                    b_tensors.insert(key, (data, shape));
                }
            }
        }
    }

    if a_tensors.is_empty() && b_tensors.is_empty() {
        return Err(TuneError::Serialization(
            "LoRA adapter contains no lora_A/lora_B tensors".to_string(),
        ));
    }

    // Pair A and B matrices
    let mut layers: HashMap<(usize, String), LoraLayer> = HashMap::new();
    let mut rank: Option<usize> = None;

    for (key, (a_data, a_shape)) in &a_tensors {
        let (b_data, b_shape) = b_tensors.get(key).ok_or_else(|| {
            TuneError::Serialization(format!(
                "LoRA adapter has lora_A for layer {} module '{}' but no matching lora_B",
                key.0, key.1
            ))
        })?;

        // A shape: (rank, d_in), B shape: (d_out, rank)
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TuneError::Serialization(format!(
                "LoRA matrices for layer {} module '{}' must be 2D, got A={:?} B={:?}",
                key.0, key.1, a_shape, b_shape
            )));
        }

        let a_rank = a_shape[0];
        let d_in = a_shape[1];
        let d_out = b_shape[0];
        let b_rank = b_shape[1];

        if a_rank != b_rank {
            return Err(TuneError::Serialization(format!(
                "LoRA rank mismatch for layer {} module '{}': A rank={}, B rank={}",
                key.0, key.1, a_rank, b_rank
            )));
        }

        // Verify all layers use the same rank
        match rank {
            None => rank = Some(a_rank),
            Some(r) if r != a_rank => {
                return Err(TuneError::Serialization(format!(
                    "inconsistent LoRA ranks: first seen rank={}, layer {} module '{}' has rank={}",
                    r, key.0, key.1, a_rank
                )));
            }
            _ => {}
        }

        // Verify data sizes match shapes
        let expected_a = a_rank.checked_mul(d_in).ok_or_else(|| {
            TuneError::Serialization(format!(
                "LoRA A shape overflow for layer {} module '{}': {} * {}",
                key.0, key.1, a_rank, d_in
            ))
        })?;
        if a_data.len() != expected_a {
            return Err(TuneError::Serialization(format!(
                "LoRA A data size mismatch for layer {} module '{}': expected {}, got {}",
                key.0,
                key.1,
                expected_a,
                a_data.len()
            )));
        }
        let expected_b = d_out.checked_mul(b_rank).ok_or_else(|| {
            TuneError::Serialization(format!(
                "LoRA B shape overflow for layer {} module '{}': {} * {}",
                key.0, key.1, d_out, b_rank
            ))
        })?;
        if b_data.len() != expected_b {
            return Err(TuneError::Serialization(format!(
                "LoRA B data size mismatch for layer {} module '{}': expected {}, got {}",
                key.0,
                key.1,
                expected_b,
                b_data.len()
            )));
        }

        layers.insert(
            key.clone(),
            LoraLayer {
                a: a_data.clone(),
                b: b_data.clone(),
                d_in,
                d_out,
                rank: a_rank,
            },
        );
    }

    // Check for orphaned B matrices
    for key in b_tensors.keys() {
        if !a_tensors.contains_key(key) {
            return Err(TuneError::Serialization(format!(
                "LoRA adapter has lora_B for layer {} module '{}' but no matching lora_A",
                key.0, key.1
            )));
        }
    }

    let rank = rank.unwrap_or(0);

    let alpha_metadata = SafeTensors::read_metadata(bytes)
        .ok()
        .and_then(|(_, meta)| {
            meta.metadata()
                .as_ref()
                .and_then(|m| m.get("alpha").cloned())
        });
    let alpha = match alpha_metadata {
        Some(value) => {
            let alpha = value.parse::<f32>().map_err(|e| {
                TuneError::Serialization(format!("invalid LoRA alpha metadata '{value}': {e}"))
            })?;
            if !alpha.is_finite() {
                return Err(TuneError::Serialization(format!(
                    "LoRA alpha metadata must be finite, got '{value}'"
                )));
            }
            alpha
        }
        None => rank as f32,
    };

    let config = LoraConfig {
        rank,
        alpha,
        target_modules: target_modules.into_iter().collect(),
        dtype: "f32".into(),
    };
    LoraAdapter::new(config, layers)
}

/// Read the `adapter_id` metadata key from a PEFT safetensors byte slice.
///
/// Returns `None` if the bytes cannot be parsed, the header has no metadata,
/// or the `adapter_id` key is absent. Used by the manifest loader to enforce
/// the id consistency check (loader check 10).
pub(crate) fn read_peft_header_adapter_id(data: &[u8]) -> Option<String> {
    SafeTensors::read_metadata(data).ok().and_then(|(_, meta)| {
        meta.metadata()
            .as_ref()
            .and_then(|m| m.get("adapter_id").cloned())
    })
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
    let (_, meta) = SafeTensors::read_metadata(data).ok()?;
    let m = meta.metadata().as_ref()?;
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
pub(crate) const MAX_LORA_SIZE: u64 = 10 * 1024 * 1024 * 1024;

/// Read an adapter file into memory without exceeding `max_bytes`.
/// Returns I/O errors for metadata, over-limit, open, or read failures.
/// See [`docs/lora-io.md`](../../docs/lora-io.md#read_lora_file_bounded) for the two-stage bound.
pub(crate) fn read_lora_file_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, TuneError> {
    // Reject a known-oversized file before allocation.
    let file_size = std::fs::metadata(path)
        .map_err(|e| {
            TuneError::Io(std::io::Error::new(
                e.kind(),
                format!("failed to read metadata for {}: {e}", path.display()),
            ))
        })?
        .len();
    if file_size > max_bytes {
        return Err(TuneError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "LoRA file {} is {file_size} bytes, exceeds maximum of {max_bytes} bytes",
                path.display()
            ),
        )));
    }
    // The one-byte sentinel catches growth after the metadata check.
    use std::io::Read;
    let f = std::fs::File::open(path).map_err(|e| {
        TuneError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to open LoRA adapter from {}: {e}", path.display()),
        ))
    })?;
    let mut buf = Vec::new();
    f.take(max_bytes + 1).read_to_end(&mut buf).map_err(|e| {
        TuneError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to read LoRA adapter from {}: {e}", path.display()),
        ))
    })?;
    if buf.len() as u64 > max_bytes {
        return Err(TuneError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "LoRA file {} exceeds maximum of {max_bytes} bytes at read time (grew after stat)",
                path.display()
            ),
        )));
    }
    Ok(buf)
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

    // Collect owned byte buffers first; TensorView borrows from these.
    let mut byte_data: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();

    for ((layer_idx, module), layer) in adapter.layers() {
        // Empty factors denote an untrained module and cannot form a tensor view.
        if layer.a.is_empty() || layer.b.is_empty() {
            continue;
        }

        let block = match module.as_str() {
            "q_proj" | "k_proj" | "v_proj" | "o_proj" => "self_attn",
            "gate_proj" | "up_proj" | "down_proj" => "mlp",
            _ => "self_attn",
        };

        let a_key =
            format!("base_model.model.model.layers.{layer_idx}.{block}.{module}.lora_A.weight");
        let b_key =
            format!("base_model.model.model.layers.{layer_idx}.{block}.{module}.lora_B.weight");

        let a_bytes: Vec<u8> = layer.a.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = layer.b.iter().flat_map(|f| f.to_le_bytes()).collect();

        byte_data.push((a_key, vec![layer.rank, layer.d_in], a_bytes));
        byte_data.push((b_key, vec![layer.d_out, layer.rank], b_bytes));
    }

    let mut tensor_views: HashMap<String, TensorView<'_>> = HashMap::new();
    for (name, shape, bytes) in &byte_data {
        let view = TensorView::new(Dtype::F32, shape.clone(), bytes).map_err(|e| {
            TuneError::Serialization(format!("failed to create tensor view for '{name}': {e}"))
        })?;
        tensor_views.insert(name.clone(), view);
    }

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
    let metadata = Some(metadata_map);

    let bytes = serialize(&tensor_views, &metadata)
        .map_err(|e| TuneError::Serialization(format!("failed to serialize LoRA adapter: {e}")))?;

    std::fs::write(path, bytes).map_err(|e| {
        TuneError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to write LoRA adapter to {}: {e}", path.display()),
        ))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests;
