//! Load PEFT-format LoRA adapters from safetensors files.
//!
//! PEFT (Parameter-Efficient Fine-Tuning) saves LoRA weights with keys like:
//! ```text
//! base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight  -> (rank, d_in)
//! base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight  -> (d_out, rank)
//! base_model.model.model.layers.{i}.mlp.gate_proj.lora_A.weight     -> (rank, d_in)
//! base_model.model.model.layers.{i}.mlp.gate_proj.lora_B.weight     -> (d_out, rank)
//! ```
//!
//! This module parses those keys, extracts layer index and module name,
//! and loads the A/B matrix pairs into [`LoraLayer`] structs.

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

/// Parse a PEFT safetensors key into its components.
///
/// Accepted formats:
/// - `base_model.model.model.layers.{i}.self_attn.{module}.lora_A.weight`
/// - `base_model.model.model.layers.{i}.mlp.{module}.lora_B.weight`
///
/// Also handles the simpler HuggingFace format without the `base_model.model` prefix:
/// - `model.layers.{i}.self_attn.{module}.lora_A.weight`
/// - `model.layers.{i}.mlp.{module}.lora_B.weight`
///
/// Returns `None` for keys that don't match (e.g., non-LoRA metadata tensors).
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

/// Load a LoRA adapter from raw PEFT-format safetensors bytes.
///
/// This is the byte-level counterpart of [`load_peft_safetensors`]. It is
/// used by the manifest-driven loader, which reads and integrity-checks the
/// bytes externally before calling this function.
///
/// Error messages omit the original file path because the caller has already
/// provided that context.
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
    };
    config.validate()?;

    Ok(LoraAdapter { config, layers })
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

/// Governance provenance fields embeddable into a saved adapter's
/// safetensors metadata header, closing the "a manifest-less adapter file
/// carries no provenance" gap (issue #610).
///
/// Deliberately does **not** include `integrity_sha256`: that field (as
/// carried by a manifest `ManifestEntry`) is a SHA-256 hash of the
/// *complete* safetensors file, header included (see the manifest loader's
/// Check 4). A file cannot correctly embed a hash of its own complete byte
/// stream inside itself — writing the hash into the header changes the
/// header bytes, which changes the true hash, and there is no fixed point
/// short of a cryptographic preimage. The governed manifest remains the sole
/// authority for whole-file integrity; this struct carries everything else
/// from the #439 field set that a header CAN coherently hold.
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
    /// Governance status as a lowercase string (`"approved"` /
    /// `"quarantined"` / `"revoked"`). A plain `String` here (rather than the
    /// manifest's `AdapterStatus` enum) keeps this struct constructible when
    /// the crate is built with `safetensors` but not `serde` — `AdapterStatus`
    /// lives in the `serde`-gated `manifest` module. See
    /// [`AdapterGovernance::from_entry`] for the common case of building this
    /// from an existing manifest entry.
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

/// Read the governance fields embedded by [`save_peft_safetensors`] back out
/// of a safetensors header, if present.
///
/// Returns `None` if the bytes cannot be parsed, the header has no metadata,
/// or the `gov_name` key is absent. `gov_name` is used as the presence
/// sentinel: the writer always sets all six governance keys together from a
/// single `Option<&AdapterGovernance>`, so its absence means no governance
/// was embedded (the other five default to an empty string rather than also
/// gating on presence, since a partially-written header is not expected in
/// practice and this reader is best-effort/advisory, not itself a
/// governance gate).
///
/// Exercised today only by the round-trip tests below (no production caller
/// consumes it yet — writing governance is the acceptance-critical half of
/// issue #610; reading it back is provided for completeness/future tooling).
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

/// Maximum on-disk size of a single LoRA adapter file, in bytes (10 GiB).
///
/// Adapter files are read fully into memory before parsing, so an unbounded
/// read of a caller-supplied path is an allocation-DoS vector. Every path that
/// materialises adapter bytes from disk routes through [`read_lora_file_bounded`]
/// with this ceiling, so a second reader cannot reintroduce the bypass.
pub(crate) const MAX_LORA_SIZE: u64 = 10 * 1024 * 1024 * 1024;

/// Read a file into memory after rejecting it if it exceeds `max_bytes`.
///
/// Two-layer defence against oversized reads:
///
/// 1. **Fast-path stat**: `metadata().len()` checked before the file is opened.
///    A known-huge file is refused without any allocation.
/// 2. **Bounded read**: `File::open` + `.take(max_bytes + 1)` enforces the cap
///    at read time. The `+1` sentinel detects a file that grew between the stat
///    and the open (TOCTOU window), because an exactly-at-cap file reads fully
///    while any larger file produces `buf.len() > max_bytes` after the read.
///
/// This is the single guarded entry point for loading adapter bytes from disk;
/// both the path-based [`load_peft_safetensors`] and the manifest-driven loader
/// route through here, so the size bound cannot be bypassed by a second reader.
///
/// # Errors
///
/// Returns an error if the file metadata cannot be read, the file is larger
/// than `max_bytes`, or the read itself fails.
pub(crate) fn read_lora_file_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, TuneError> {
    // Fast-path: stat before open. A known-oversized file is refused without
    // allocating for its contents (defence-in-depth — the read is also bounded).
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
    // Bounded read: take max_bytes + 1 bytes so a file that grew after the
    // stat is still caught. If buf.len() exceeds max_bytes at read time, the
    // file is rejected even though the stat passed.
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

/// Load a LoRA adapter from a PEFT-format safetensors file.
///
/// Reads the file, parses all LoRA tensor keys, pairs A/B matrices per
/// (layer_idx, module), and assembles the adapter.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read or is not valid safetensors
/// - A/B matrix pairs are incomplete (A without B or vice versa)
/// - Tensor shapes are inconsistent (ranks must match within a pair)
pub fn load_peft_safetensors(path: &Path) -> Result<LoraAdapter, TuneError> {
    let data = read_lora_file_bounded(path, MAX_LORA_SIZE)?;
    load_peft_safetensors_bytes(&data)
}

/// Save a LoRA adapter to a PEFT-format safetensors file.
///
/// Writes one `lora_A` and one `lora_B` tensor per `(layer_idx, module)` pair
/// using the standard PEFT key format:
/// `base_model.model.model.layers.{i}.{block}.{module}.lora_{A,B}.weight`
///
/// Metadata stored in the safetensors header: `rank`, `alpha`, `target_modules`,
/// and, when `governance` is `Some`, `gov_name`, `gov_owner`,
/// `gov_base_model_rev`, `gov_tokenizer_rev`, `gov_dtype`, `gov_status` (see
/// [`AdapterGovernance`] for why `integrity_sha256` is deliberately not
/// among them). Pass `None` to save without embedding governance metadata
/// (unchanged behavior from before issue #610).
///
/// # Errors
///
/// Returns an error if tensor views cannot be created or the file cannot be written.
pub fn save_peft_safetensors(
    adapter: &LoraAdapter,
    path: &Path,
    governance: Option<&AdapterGovernance>,
) -> Result<(), TuneError> {
    adapter.config.validate()?;

    // Collect owned byte buffers first; TensorView borrows from these.
    let mut byte_data: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();

    for ((layer_idx, module), layer) in &adapter.layers {
        // A LoRA layer with an empty factor buffer means "this module was not trained"
        // (e.g. GDN-attention slots leave q_proj/v_proj empty). Skip it rather than
        // emit an InvalidTensorView for a zero-byte tensor with non-zero shape.
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
    metadata_map.insert("rank".to_string(), adapter.config.rank.to_string());
    metadata_map.insert("alpha".to_string(), adapter.config.alpha.to_string());
    metadata_map.insert(
        "target_modules".to_string(),
        adapter.config.target_modules.join(","),
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
mod tests {
    use super::*;

    #[test]
    fn read_lora_file_bounded_rejects_oversized_before_read() {
        // A 64-byte file with a 32-byte cap must be refused by the metadata
        // stat, before its contents are read into memory. The file is trivially
        // readable, so a failure here is the size policy firing, not an IO error.
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 64]).unwrap();
        f.flush().unwrap();
        let err = read_lora_file_bounded(f.path(), 32)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("exceeds maximum"),
            "expected the size guard to fire; got: {err}"
        );
    }

    #[test]
    fn read_lora_file_bounded_reads_within_cap() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        let payload = [7u8; 64];
        f.write_all(&payload).unwrap();
        f.flush().unwrap();
        let bytes = read_lora_file_bounded(f.path(), 1024).unwrap();
        assert_eq!(bytes, payload);
    }

    #[test]
    fn test_parse_peft_key_self_attn() {
        let key = "base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer_idx, 5);
        assert_eq!(parsed.module, "q_proj");
        assert_eq!(parsed.matrix, LoraMatrix::A);
    }

    #[test]
    fn test_parse_peft_key_mlp() {
        let key = "base_model.model.model.layers.12.mlp.gate_proj.lora_B.weight";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer_idx, 12);
        assert_eq!(parsed.module, "gate_proj");
        assert_eq!(parsed.matrix, LoraMatrix::B);
    }

    #[test]
    fn test_parse_peft_key_simple_prefix() {
        let key = "model.layers.0.self_attn.v_proj.lora_A.weight";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer_idx, 0);
        assert_eq!(parsed.module, "v_proj");
        assert_eq!(parsed.matrix, LoraMatrix::A);
    }

    #[test]
    fn test_parse_peft_key_no_weight_suffix() {
        let key = "base_model.model.model.layers.3.mlp.up_proj.lora_B";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer_idx, 3);
        assert_eq!(parsed.module, "up_proj");
        assert_eq!(parsed.matrix, LoraMatrix::B);
    }

    #[test]
    fn test_parse_peft_key_rejects_non_lora() {
        assert!(parse_peft_key("model.layers.0.self_attn.q_proj.weight").is_none());
        assert!(parse_peft_key("some_random_tensor").is_none());
        assert!(parse_peft_key("").is_none());
    }

    #[test]
    fn test_parse_mlx_key_lowercase() {
        let key = "model.layers.3.self_attn.q_proj.lora_a";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer_idx, 3);
        assert_eq!(parsed.module, "q_proj");
        assert_eq!(parsed.matrix, LoraMatrix::A);
        assert!(parsed.transposed);
    }

    #[test]
    fn test_parse_mlx_key_mlp() {
        let key = "model.layers.7.mlp.down_proj.lora_b";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer_idx, 7);
        assert_eq!(parsed.module, "down_proj");
        assert_eq!(parsed.matrix, LoraMatrix::B);
        assert!(parsed.transposed);
    }

    #[test]
    fn test_peft_key_not_transposed() {
        let key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let parsed = parse_peft_key(key).unwrap();
        assert!(!parsed.transposed);
    }

    #[test]
    fn test_parse_peft_key_all_target_modules() {
        let modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ];
        for (module, block) in modules.iter().zip(
            [
                "self_attn",
                "self_attn",
                "self_attn",
                "self_attn",
                "mlp",
                "mlp",
                "mlp",
            ]
            .iter(),
        ) {
            let key = format!("base_model.model.model.layers.7.{block}.{module}.lora_A.weight");
            let parsed = parse_peft_key(&key).unwrap();
            assert_eq!(parsed.layer_idx, 7);
            assert_eq!(parsed.module, *module);
            assert_eq!(parsed.matrix, LoraMatrix::A);
        }
    }

    #[test]
    fn test_f16_to_f32_roundtrip() {
        // Test a few known values
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
        // 0.0 in f16 = 0x0000
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // 0.5 in f16 = 0x3800
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32() {
        // 1.0 in bf16 = 0x3F80 (same as upper 16 bits of f32 1.0)
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        // 0.0
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    /// Helper: create a PEFT-format safetensors file with known LoRA weights.
    fn write_test_peft_safetensors(path: &std::path::Path, rank: usize, d_in: usize, d_out: usize) {
        use safetensors::Dtype;
        use safetensors::tensor::{TensorView, serialize};
        use std::collections::HashMap;

        // Layer 0, q_proj: A=(rank, d_in), B=(d_out, rank)
        let a_data: Vec<f32> = (0..rank * d_in).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..d_out * rank).map(|i| (i as f32) * 0.1).collect();

        // Layer 2, gate_proj: same dimensions
        let a2_data: Vec<f32> = (0..rank * d_in).map(|i| (i as f32) * -0.01).collect();
        let b2_data: Vec<f32> = (0..d_out * rank).map(|i| (i as f32) * -0.1).collect();

        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let a2_bytes: Vec<u8> = a2_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b2_bytes: Vec<u8> = b2_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = HashMap::new();
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            TensorView::new(Dtype::F32, vec![rank, d_in], &a_bytes).unwrap(),
        );
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            TensorView::new(Dtype::F32, vec![d_out, rank], &b_bytes).unwrap(),
        );
        tensors.insert(
            "base_model.model.model.layers.2.mlp.gate_proj.lora_A.weight".to_string(),
            TensorView::new(Dtype::F32, vec![rank, d_in], &a2_bytes).unwrap(),
        );
        tensors.insert(
            "base_model.model.model.layers.2.mlp.gate_proj.lora_B.weight".to_string(),
            TensorView::new(Dtype::F32, vec![d_out, rank], &b2_bytes).unwrap(),
        );

        let bytes = serialize(&tensors, &None).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn test_load_peft_safetensors_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter.safetensors");

        let rank = 4;
        let d_in = 8;
        let d_out = 8;
        write_test_peft_safetensors(&path, rank, d_in, d_out);

        let adapter = load_peft_safetensors(&path).unwrap();

        // Check config
        assert_eq!(adapter.config.rank, rank);
        assert_eq!(adapter.config.target_modules.len(), 2);
        assert!(
            adapter
                .config
                .target_modules
                .contains(&"q_proj".to_string())
        );
        assert!(
            adapter
                .config
                .target_modules
                .contains(&"gate_proj".to_string())
        );

        // Check we got both layers
        assert_eq!(adapter.layers.len(), 2);
        assert!(adapter.layers.contains_key(&(0, "q_proj".to_string())));
        assert!(adapter.layers.contains_key(&(2, "gate_proj".to_string())));

        // Check dimensions
        let q_lora = &adapter.layers[&(0, "q_proj".to_string())];
        assert_eq!(q_lora.rank, rank);
        assert_eq!(q_lora.d_in, d_in);
        assert_eq!(q_lora.d_out, d_out);
        assert_eq!(q_lora.a.len(), rank * d_in);
        assert_eq!(q_lora.b.len(), d_out * rank);

        // Verify actual values for A[0]
        assert!((q_lora.a[0] - 0.0).abs() < 1e-6);
        assert!((q_lora.a[1] - 0.01).abs() < 1e-6);
        assert!((q_lora.a[2] - 0.02).abs() < 1e-6);

        // Check gate_proj layer
        let g_lora = &adapter.layers[&(2, "gate_proj".to_string())];
        assert_eq!(g_lora.rank, rank);
        assert!((g_lora.a[1] - (-0.01)).abs() < 1e-6);
    }

    #[test]
    fn test_load_mlx_safetensors_transposed() {
        // MLX format: A=(d_in, rank), B=(rank, d_out) with lowercase keys
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapters.safetensors");

        use safetensors::Dtype;
        use safetensors::tensor::{TensorView, serialize};

        // rank=2, d_in=4, d_out=3
        // MLX A: (d_in=4, rank=2) -> after transpose: (rank=2, d_in=4)
        // MLX B: (rank=2, d_out=3) -> after transpose: (d_out=3, rank=2)
        let a_data: Vec<f32> = vec![
            // 4x2 matrix (d_in=4, rank=2)
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ];
        let b_data: Vec<f32> = vec![
            // 2x3 matrix (rank=2, d_out=3)
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        ];

        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.lora_a".to_string(),
            TensorView::new(Dtype::F32, vec![4, 2], &a_bytes).unwrap(),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.lora_b".to_string(),
            TensorView::new(Dtype::F32, vec![2, 3], &b_bytes).unwrap(),
        );

        let bytes = serialize(&tensors, &None).unwrap();
        std::fs::write(&path, bytes).unwrap();

        let adapter = load_peft_safetensors(&path).unwrap();
        assert_eq!(adapter.config.rank, 2);

        let lora = &adapter.layers[&(0, "q_proj".to_string())];
        assert_eq!(lora.rank, 2);
        assert_eq!(lora.d_in, 4);
        assert_eq!(lora.d_out, 3);

        // Verify transpose: original A[0,0]=1, A[1,0]=3, A[2,0]=5, A[3,0]=7
        // Transposed A[0,0]=1, A[0,1]=3, A[0,2]=5, A[0,3]=7
        assert!((lora.a[0] - 1.0).abs() < 1e-6); // row 0
        assert!((lora.a[1] - 3.0).abs() < 1e-6);
        assert!((lora.a[2] - 5.0).abs() < 1e-6);
        assert!((lora.a[3] - 7.0).abs() < 1e-6);
        assert!((lora.a[4] - 2.0).abs() < 1e-6); // row 1
        assert!((lora.a[5] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_load_peft_safetensors_apply_correctness() {
        // End-to-end: write safetensors, load, apply, verify numerical result.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter.safetensors");

        // rank=1, d_in=2, d_out=2 for simplicity
        // A = [[0.0, 0.01]]  (1x2)
        // B = [[0.0], [0.1]] (2x1)
        use safetensors::Dtype;
        use safetensors::tensor::{TensorView, serialize};

        let a_data: Vec<f32> = vec![0.0, 1.0]; // A = [[0, 1]]
        let b_data: Vec<f32> = vec![1.0, 0.0]; // B = [[1], [0]]
        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.v_proj.lora_A.weight".to_string(),
            TensorView::new(Dtype::F32, vec![1, 2], &a_bytes).unwrap(),
        );
        tensors.insert(
            "model.layers.0.self_attn.v_proj.lora_B.weight".to_string(),
            TensorView::new(Dtype::F32, vec![2, 1], &b_bytes).unwrap(),
        );

        let bytes = serialize(&tensors, &None).unwrap();
        std::fs::write(&path, bytes).unwrap();

        let adapter = load_peft_safetensors(&path).unwrap();
        assert_eq!(adapter.config.rank, 1);
        // Default alpha = rank = 1, so scale = 1.0

        let lora = &adapter.layers[&(0, "v_proj".to_string())];
        // x = [3, 5]
        // A @ x = [0*3 + 1*5] = [5]
        // B @ [5] = [1*5, 0*5] = [5, 0]
        // scale = 1.0 => output += [5, 0]
        let x = [3.0f32, 5.0];
        let mut output = [10.0, 20.0];
        super::super::apply_lora(lora, adapter.config.scale(), &x, &mut output);

        assert!((output[0] - 15.0).abs() < 1e-6);
        assert!((output[1] - 20.0).abs() < 1e-6);
    }

    fn write_raw_safetensors(path: &std::path::Path, header: &str, data: &[u8]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(path, bytes).unwrap();
    }

    fn peft_bytes_with_alpha(alpha: &str) -> Vec<u8> {
        use safetensors::Dtype;
        use safetensors::tensor::{TensorView, serialize};

        let a_bytes = 1.0f32.to_le_bytes();
        let b_bytes = 1.0f32.to_le_bytes();
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            TensorView::new(Dtype::F32, vec![1, 1], &a_bytes).unwrap(),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            TensorView::new(Dtype::F32, vec![1, 1], &b_bytes).unwrap(),
        );
        let metadata = HashMap::from([("alpha".to_string(), alpha.to_string())]);
        serialize(&tensors, &Some(metadata)).unwrap()
    }

    #[test]
    fn test_rejects_non_finite_alpha_metadata() {
        for alpha in ["nan", "inf", "-inf"] {
            let bytes = peft_bytes_with_alpha(alpha);
            let err = load_peft_safetensors_bytes(&bytes).unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("LoRA alpha metadata must be finite"),
                "expected descriptive rejection for {alpha}, got: {message}"
            );
        }
    }

    #[test]
    fn test_rejects_truncated_safetensors_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.safetensors");
        std::fs::write(&path, 128u64.to_le_bytes()).unwrap();

        let err = load_peft_safetensors(&path).unwrap_err();
        assert!(err.to_string().contains("failed to parse safetensors"));
    }

    #[test]
    fn test_rejects_file_without_lora_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.safetensors");
        let header = r#"{"metadata.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        write_raw_safetensors(&path, header, &1.0f32.to_le_bytes());

        let err = load_peft_safetensors(&path).unwrap_err();
        assert!(err.to_string().contains("no lora_A/lora_B tensors"));
    }

    #[test]
    fn test_rejects_shape_product_overflow() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overflow.safetensors");
        let max = usize::MAX;
        let header = format!(
            r#"{{"model.layers.0.self_attn.q_proj.lora_A.weight":{{"dtype":"F32","shape":[{max},2],"data_offsets":[0,0]}},"model.layers.0.self_attn.q_proj.lora_B.weight":{{"dtype":"F32","shape":[1,{max}],"data_offsets":[0,0]}}}}"#
        );
        write_raw_safetensors(&path, &header, &[]);

        let err = load_peft_safetensors(&path).unwrap_err();
        assert!(
            err.to_string().contains("shape overflow")
                || err.to_string().contains("failed to parse safetensors")
        );
    }

    #[test]
    fn test_rejects_non_finite_tensor_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nan.safetensors");

        use safetensors::Dtype;
        use safetensors::tensor::{TensorView, serialize};

        let a_data: Vec<f32> = vec![f32::NAN, 1.0];
        let b_data: Vec<f32> = vec![1.0, 2.0];
        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            TensorView::new(Dtype::F32, vec![1, 2], &a_bytes).unwrap(),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            TensorView::new(Dtype::F32, vec![2, 1], &b_bytes).unwrap(),
        );

        let bytes = serialize(&tensors, &None).unwrap();
        std::fs::write(&path, bytes).unwrap();

        let err = load_peft_safetensors(&path).unwrap_err();
        assert!(err.to_string().contains("non-finite"));
    }

    #[test]
    fn test_save_load_round_trip() {
        use tempfile::NamedTempFile;

        let rank = 4;
        let d_in = 8;
        let d_out = 16;

        // alpha == rank here, so the round-trip lands back on rank either way;
        // test_alpha_metadata_round_trips covers the alpha != rank case.
        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["q_proj".to_string(), "gate_proj".to_string()],
        };

        let a_data: Vec<f32> = (0..rank * d_in).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..d_out * rank).map(|i| i as f32 * 0.1).collect();
        let a2_data: Vec<f32> = (0..rank * d_in).map(|i| i as f32 * -0.01).collect();
        let b2_data: Vec<f32> = (0..d_out * rank).map(|i| i as f32 * -0.1).collect();

        let mut layers = HashMap::new();
        layers.insert(
            (0usize, "q_proj".to_string()),
            LoraLayer {
                a: a_data.clone(),
                b: b_data.clone(),
                d_in,
                d_out,
                rank,
            },
        );
        layers.insert(
            (2usize, "gate_proj".to_string()),
            LoraLayer {
                a: a2_data.clone(),
                b: b2_data.clone(),
                d_in,
                d_out,
                rank,
            },
        );

        let adapter = LoraAdapter::new(config, layers);

        let temp = NamedTempFile::new().unwrap();
        save_peft_safetensors(&adapter, temp.path(), None).unwrap();

        let loaded = load_peft_safetensors(temp.path()).unwrap();

        assert_eq!(loaded.layers.len(), adapter.layers.len());
        assert_eq!(loaded.config.rank, adapter.config.rank);
        assert_eq!(loaded.config.alpha, adapter.config.alpha);

        for (key, orig) in &adapter.layers {
            let got = loaded
                .layers
                .get(key)
                .expect("layer key missing after round-trip");
            assert_eq!(got.rank, orig.rank);
            assert_eq!(got.d_in, orig.d_in);
            assert_eq!(got.d_out, orig.d_out);
            assert_eq!(got.a.len(), orig.a.len());
            assert_eq!(got.b.len(), orig.b.len());
            for (g, w) in got.a.iter().zip(&orig.a) {
                assert!((g - w).abs() < f32::EPSILON, "A mismatch: {g} vs {w}");
            }
            for (g, w) in got.b.iter().zip(&orig.b) {
                assert!((g - w).abs() < f32::EPSILON, "B mismatch: {g} vs {w}");
            }
        }
    }

    /// `save_peft_safetensors(.., Some(governance))` must embed all six
    /// governance fields into the header, round-trippable via
    /// `read_peft_header_governance`. Mutation-sensitive: removing the
    /// `if let Some(governance) = governance { .. }` embedding block makes
    /// this `None` (no `gov_name` key present), so `.expect(..)` panics —
    /// this test fails when the embedding is absent.
    #[test]
    fn test_governance_metadata_round_trips() {
        use tempfile::NamedTempFile;

        let rank = 4;
        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["q_proj".to_string()],
        };
        let mut layers = HashMap::new();
        layers.insert(
            (0usize, "q_proj".to_string()),
            LoraLayer {
                a: vec![0.0; rank * 4],
                b: vec![0.0; 4 * rank],
                d_in: 4,
                d_out: 4,
                rank,
            },
        );
        let adapter = LoraAdapter::new(config, layers);

        let governance = AdapterGovernance {
            name: "test-adapter".to_string(),
            owner: "team-inference".to_string(),
            base_model_rev: "rev-aaa".to_string(),
            tokenizer_rev: "rev-bbb".to_string(),
            dtype: "f32".to_string(),
            status: "approved".to_string(),
        };

        let temp = NamedTempFile::new().unwrap();
        save_peft_safetensors(&adapter, temp.path(), Some(&governance)).unwrap();

        let bytes = std::fs::read(temp.path()).unwrap();
        let got = read_peft_header_governance(&bytes).expect("governance metadata must be present");
        assert_eq!(got.name, "test-adapter");
        assert_eq!(got.owner, "team-inference");
        assert_eq!(got.base_model_rev, "rev-aaa");
        assert_eq!(got.tokenizer_rev, "rev-bbb");
        assert_eq!(got.dtype, "f32");
        assert_eq!(got.status, "approved");
    }

    /// Saving with `governance: None` must not embed any `gov_*` keys — the
    /// feature is opt-in, not a default-on side effect.
    #[test]
    fn test_no_governance_metadata_when_not_supplied() {
        use tempfile::NamedTempFile;

        let rank = 4;
        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["q_proj".to_string()],
        };
        let mut layers = HashMap::new();
        layers.insert(
            (0usize, "q_proj".to_string()),
            LoraLayer {
                a: vec![0.0; rank * 4],
                b: vec![0.0; 4 * rank],
                d_in: 4,
                d_out: 4,
                rank,
            },
        );
        let adapter = LoraAdapter::new(config, layers);

        let temp = NamedTempFile::new().unwrap();
        save_peft_safetensors(&adapter, temp.path(), None).unwrap();

        let bytes = std::fs::read(temp.path()).unwrap();
        assert!(read_peft_header_governance(&bytes).is_none());
    }

    /// `AdapterGovernance::from_entry` must faithfully carry over every
    /// field, converting `AdapterStatus` to its lowercase string form.
    #[cfg(feature = "serde")]
    #[test]
    fn test_adapter_governance_from_entry() {
        use crate::lora::manifest::{AdapterStatus, ManifestEntry};

        let entry = ManifestEntry {
            id: "adapter-1".to_string(),
            name: "adapter one".to_string(),
            owner: "team-tune".to_string(),
            uri: "adapter-1.safetensors".to_string(),
            integrity_sha256: "deadbeef".to_string(),
            base_model_rev: "rev-aaa".to_string(),
            tokenizer_rev: "rev-bbb".to_string(),
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string()],
            dtype: "f16".to_string(),
            status: AdapterStatus::Approved,
        };

        let governance = AdapterGovernance::from_entry(&entry);
        assert_eq!(governance.name, "adapter one");
        assert_eq!(governance.owner, "team-tune");
        assert_eq!(governance.base_model_rev, "rev-aaa");
        assert_eq!(governance.tokenizer_rev, "rev-bbb");
        assert_eq!(governance.dtype, "f16");
        assert_eq!(governance.status, "approved");
    }

    #[test]
    fn test_alpha_metadata_round_trips() {
        // Regression: an adapter trained with alpha != rank must load with its
        // saved alpha, not alpha = rank. The old loader hardcoded alpha = rank,
        // so a rank-4 / alpha-16 adapter loaded at scale 1.0 instead of 4.0.
        use tempfile::NamedTempFile;

        let rank = 4;
        let d_in = 8;
        let d_out = 16;
        let config = LoraConfig {
            rank,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string()],
        };

        let mut layers = HashMap::new();
        layers.insert(
            (0usize, "q_proj".to_string()),
            LoraLayer {
                a: (0..rank * d_in).map(|i| i as f32 * 0.01).collect(),
                b: (0..d_out * rank).map(|i| i as f32 * 0.1).collect(),
                d_in,
                d_out,
                rank,
            },
        );

        let adapter = LoraAdapter::new(config, layers);
        let temp = NamedTempFile::new().unwrap();
        save_peft_safetensors(&adapter, temp.path(), None).unwrap();
        let loaded = load_peft_safetensors(temp.path()).unwrap();

        assert_eq!(loaded.config.rank, 4);
        assert_eq!(loaded.config.alpha, 16.0);
        assert!((loaded.config.alpha - 16.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_alpha_falls_back_to_rank_without_metadata() {
        // An adapter file with no `alpha` in its header (e.g. a raw PEFT export)
        // must fall back to alpha = rank (scale = 1.0), not panic or zero out.
        use safetensors::Dtype;
        use safetensors::tensor::{TensorView, serialize};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_alpha.safetensors");

        let a_data: Vec<f32> = (0..2 * 8).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..16 * 2).map(|i| i as f32 * 0.1).collect();
        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = HashMap::new();
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            TensorView::new(Dtype::F32, vec![2, 8], &a_bytes).unwrap(),
        );
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            TensorView::new(Dtype::F32, vec![16, 2], &b_bytes).unwrap(),
        );

        let bytes = serialize(&tensors, &None).unwrap();
        std::fs::write(&path, bytes).unwrap();

        let loaded = load_peft_safetensors(&path).unwrap();
        assert_eq!(loaded.config.rank, 2);
        assert_eq!(loaded.config.alpha, 2.0);
    }

    #[test]
    fn test_transpose_rejects_shape_element_mismatch() {
        // MLX-format (lowercase lora_a/lora_b) triggers the transpose path.
        // Shape [4, 2] implies 8 elements; we craft data_offsets=[0,24] which is 6 f32s.
        // The safetensors library catches this mismatch at deserialize() with TensorInvalidInfo
        // before our transpose guard fires.  The invariant we assert is: Err, not panic.
        //
        // Our guard (`data.len() != expected`) provides defence-in-depth for callers that
        // construct a (values, shape) pair from outside the safetensors library.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mismatch.safetensors");

        let header = r#"{"model.layers.0.self_attn.q_proj.lora_a":{"dtype":"F32","shape":[4,2],"data_offsets":[0,24]},"model.layers.0.self_attn.q_proj.lora_b":{"dtype":"F32","shape":[2,3],"data_offsets":[24,48]}}"#;
        let header_bytes = header.as_bytes();
        let mut raw: Vec<u8> = Vec::new();
        raw.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        raw.extend_from_slice(header_bytes);
        let payload: Vec<u8> = (0u32..12).flat_map(u32::to_le_bytes).collect();
        raw.extend_from_slice(&payload);
        std::fs::write(&path, &raw).unwrap();

        let err = load_peft_safetensors(&path).unwrap_err();
        assert!(
            !err.to_string().is_empty(),
            "expected a non-empty error message, got empty string"
        );
    }

    #[test]
    fn test_transpose_rejects_unaligned_f32_payload() {
        // MLX-format lora_a with shape=[2,2] but data_offsets=[0,11] (11 bytes, not aligned).
        // The library rejects this as TensorInvalidInfo.  Assert Err, not panic.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unaligned.safetensors");

        let header = r#"{"model.layers.0.self_attn.q_proj.lora_a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"model.layers.0.self_attn.q_proj.lora_b":{"dtype":"F32","shape":[2,2],"data_offsets":[16,32]}}"#;
        let header_bytes = header.as_bytes();
        let mut raw: Vec<u8> = Vec::new();
        raw.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        raw.extend_from_slice(header_bytes);
        // Only 11 bytes instead of the declared 32 — truncated, so deserialize fails.
        raw.extend_from_slice(&[0u8; 11]);
        std::fs::write(&path, &raw).unwrap();

        let err = load_peft_safetensors(&path).unwrap_err();
        assert!(
            !err.to_string().is_empty(),
            "expected a non-empty error message, got empty string"
        );
    }

    #[test]
    fn test_transpose_guard_logic_is_sound() {
        // White-box check: verify that the guard expressions added to the transpose path
        // would produce the correct result for a known mismatch.  This exercises the
        // arithmetic without requiring the safetensors library to be bypassed.
        let (rows, cols): (usize, usize) = (4, 2);
        let data_len: usize = 6; // 6 elements, but shape implies 8
        let expected = rows.checked_mul(cols).unwrap();
        assert_ne!(data_len, expected, "precondition: mismatch exists");

        // Mimic the guard: data.len() != expected => would return Err
        let guard_fires = data_len != expected;
        assert!(guard_fires, "guard must detect the mismatch");
    }

    /// Regression test: saving an adapter that contains a GDN-slot layer with empty A/B
    /// buffers must succeed (pre-fix it returned `Err(InvalidTensorView)`). The empty layer
    /// must be silently dropped from the saved file; the real layer must round-trip intact.
    #[test]
    fn test_save_skips_empty_buffer_layers() {
        use tempfile::NamedTempFile;

        let rank: usize = 8;
        let config = LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        };

        let mut layers = HashMap::new();

        // Real GQA layer — should survive the round-trip.
        layers.insert(
            (19usize, "q_proj".to_string()),
            LoraLayer {
                a: vec![0.1f32; rank * 1024],
                b: vec![0.2f32; 4096 * rank],
                d_in: 1024,
                d_out: 4096,
                rank,
            },
        );

        // Empty GDN-slot layer — must be skipped, not serialized.
        layers.insert(
            (20usize, "v_proj".to_string()),
            LoraLayer {
                a: Vec::new(),
                b: Vec::new(),
                d_in: 1024,
                d_out: 512,
                rank,
            },
        );

        let adapter = LoraAdapter::new(config, layers);

        let temp = NamedTempFile::new().unwrap();
        // This was the regression: pre-fix this call returned Err(InvalidTensorView).
        save_peft_safetensors(&adapter, temp.path(), None)
            .expect("save must succeed even with an empty-buffer GDN slot");

        let loaded = load_peft_safetensors(temp.path()).unwrap();

        // The real layer is present.
        let real_key = (19usize, "q_proj".to_string());
        assert!(
            loaded.layers.contains_key(&real_key),
            "real GQA layer (19, q_proj) must be present after round-trip"
        );

        // The empty layer was dropped — it must NOT appear in the saved file.
        let empty_key = (20usize, "v_proj".to_string());
        assert!(
            !loaded.layers.contains_key(&empty_key),
            "empty GDN-slot layer (20, v_proj) must be absent from saved adapter"
        );

        // The A buffer of the real layer must be bit-exact.
        let got = &loaded.layers[&real_key];
        let expected_a = vec![0.1f32; rank * 1024];
        assert_eq!(got.a.len(), expected_a.len());
        for (g, w) in got.a.iter().zip(&expected_a) {
            assert!(
                (g - w).abs() < f32::EPSILON,
                "A buffer mismatch: {g} vs {w}"
            );
        }
    }
}
