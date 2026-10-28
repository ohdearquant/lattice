//! PEFT/MLX safetensors decoding and PEFT-compatible encoding of LoRA factors.
//!
//! A factor pair is keyed by transformer layer and projection: A is
//! `(rank, d_in)` and B is `(d_out, rank)` after MLX normalization. Decoding
//! rejects incomplete, malformed, non-finite, or rank-inconsistent pairs.
//! File reads are bounded before parsing.
//!
//! These functions return [`LoraError`]; adapter types in other crates wrap
//! them and map the error into their own type at their boundary.

use crate::{LoraError, LoraLayer};
use safetensors::Dtype;
use safetensors::tensor::{SafeTensors, TensorView, serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

/// Maximum in-memory adapter file size (10 GiB).
pub const MAX_ADAPTER_FILE_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// The factor pairs and identity decoded from one adapter file.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodedAdapter {
    /// Factor pairs keyed by `(layer_idx, module)`.
    pub layers: HashMap<(usize, String), LoraLayer>,
    /// The rank shared by every pair; zero when the file has no pairs.
    pub rank: usize,
    /// The header `alpha`, or `rank` when the header has none.
    pub alpha: f32,
    /// The projection modules present, sorted.
    pub target_modules: Vec<String>,
    /// Unknown-module factor tensor names skipped in sorted order.
    pub skipped: Vec<String>,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LoraMatrix {
    A,
    B,
}

/// Parse a recognized PEFT or MLX LoRA tensor key, ignoring non-factor keys.
fn parse_peft_key(key: &str) -> Option<PeftKey> {
    let (key, matrix, transposed) = if let Some(k) = key
        .strip_suffix(".lora_A.weight")
        .or_else(|| key.strip_suffix(".lora_A"))
    {
        (k, LoraMatrix::A, false)
    } else if let Some(k) = key
        .strip_suffix(".lora_B.weight")
        .or_else(|| key.strip_suffix(".lora_B"))
    {
        (k, LoraMatrix::B, false)
    } else if let Some(k) = key.strip_suffix(".lora_a") {
        (k, LoraMatrix::A, true)
    } else if let Some(k) = key.strip_suffix(".lora_b") {
        (k, LoraMatrix::B, true)
    } else {
        return None;
    };

    const TEXT_PREFIXES: &[&str] = &[
        "base_model.model.model.language_model.layers.",
        "base_model.model.model.layers.",
        "model.language_model.layers.",
        "model.layers.",
    ];
    let rest = TEXT_PREFIXES
        .iter()
        .find_map(|prefix| key.strip_prefix(prefix))?;
    let mut parts = rest.split('.');
    let layer_idx = parts.next()?.parse().ok()?;
    let block = parts.next()?;
    let module = parts.next()?;
    if !matches!(block, "self_attn" | "linear_attn" | "mlp")
        || module.is_empty()
        || parts.next().is_some()
    {
        return None;
    }
    Some(PeftKey {
        layer_idx,
        module: module.to_string(),
        matrix,
        transposed,
    })
}

/// Ignore unrelated tensors, but fail closed on malformed or out-of-scope factor keys.
fn classify_peft_key(key: &str) -> Result<Option<PeftKey>, LoraError> {
    if !key.split('.').any(|part| part.starts_with("lora_")) {
        return Ok(None);
    }
    parse_peft_key(key)
        .map(Some)
        .ok_or_else(|| LoraError::Format(format!("unsupported LoRA factor key '{key}'")))
}

/// Read a tensor from a safetensors file as a Vec<f32>.
///
/// Handles f32 and f16 dtypes (converting f16 to f32).
fn read_tensor_f32(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), LoraError> {
    let tensor = tensors
        .tensor(name)
        .map_err(|e| LoraError::Format(format!("failed to read tensor '{name}': {e}")))?;

    let shape: Vec<usize> = tensor.shape().to_vec();
    let data = tensor.data();

    let values: Vec<f32> = match tensor.dtype() {
        Dtype::F32 => {
            if data.len() % 4 != 0 {
                return Err(LoraError::Format(format!(
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
                return Err(LoraError::Format(format!(
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
                return Err(LoraError::Format(format!(
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
            return Err(LoraError::Format(format!(
                "tensor '{name}' has unsupported dtype {other:?}, expected F32/F16/BF16"
            )));
        }
    };

    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(LoraError::Format(format!(
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

/// Decode raw PEFT or MLX safetensors bytes into factor pairs and identity.
///
/// The caller supplies file-path and integrity context when needed, and
/// validates the result as an adapter descriptor.
///
/// # Errors
///
/// [`LoraError::Format`] when the bytes are not a safetensors file, hold no
/// factor tensors, hold an unpaired, non-2D, rank-inconsistent, mis-sized or
/// non-finite factor, or carry an unparsable or non-finite `alpha`.
pub fn decode_peft_safetensors(bytes: &[u8]) -> Result<DecodedAdapter, LoraError> {
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|e| LoraError::Format(format!("failed to parse safetensors: {e}")))?;

    // Collect all parsed LoRA keys. Sorted so that error messages name the
    // same keys on every run.
    let mut names: Vec<String> = tensors.names().into_iter().map(String::from).collect();
    names.sort_unstable();
    let mut a_tensors: HashMap<(usize, String), (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut b_tensors: HashMap<(usize, String), (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut target_modules = BTreeSet::new();
    let mut sources: HashMap<(usize, String, LoraMatrix), &str> = HashMap::new();
    let mut dialects: HashMap<(usize, String), (bool, &str)> = HashMap::new();
    let mut skipped = Vec::new();

    for name in &names {
        if let Some(peft_key) = classify_peft_key(name)? {
            if !crate::KNOWN_LORA_TARGET_MODULES.contains(&peft_key.module.as_str()) {
                skipped.push(name.clone());
                continue;
            }
            let key = (peft_key.layer_idx, peft_key.module.clone());
            let factor = match peft_key.matrix {
                LoraMatrix::A => "lora_A",
                LoraMatrix::B => "lora_B",
            };
            if let Some(first) = sources.insert((key.0, key.1.clone(), peft_key.matrix), name) {
                return Err(LoraError::Format(format!(
                    "LoRA adapter has two {factor} tensors for layer {} module '{}': '{first}' and '{name}'",
                    key.0, key.1
                )));
            }
            match dialects.get(&key) {
                Some(&(mlx, first)) if mlx != peft_key.transposed => {
                    return Err(LoraError::Format(format!(
                        "LoRA adapter mixes PEFT and MLX key forms for layer {} module '{}': '{first}' and '{name}'",
                        key.0, key.1
                    )));
                }
                Some(_) => {}
                None => {
                    dialects.insert(key.clone(), (peft_key.transposed, name));
                }
            }
            let (data, shape) = read_tensor_f32(&tensors, name)?;
            target_modules.insert(peft_key.module.clone());

            // MLX format stores transposed: A=(d_in, rank), B=(rank, d_out)
            // PEFT format (what we expect): A=(rank, d_in), B=(d_out, rank)
            let (data, shape) = if peft_key.transposed && shape.len() == 2 {
                let (rows, cols) = (shape[0], shape[1]);
                let expected = rows.checked_mul(cols).ok_or_else(|| {
                    LoraError::Format(format!(
                        "tensor '{name}' shape [{rows}, {cols}] overflows usize"
                    ))
                })?;
                if data.len() != expected {
                    return Err(LoraError::Format(format!(
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
        return Err(LoraError::Format(
            "LoRA adapter contains no lora_A/lora_B tensors".to_string(),
        ));
    }

    // Pair A and B matrices
    let mut layers: HashMap<(usize, String), LoraLayer> = HashMap::new();
    let mut rank: Option<usize> = None;

    for (key, (a_data, a_shape)) in &a_tensors {
        let (b_data, b_shape) = b_tensors.get(key).ok_or_else(|| {
            LoraError::Format(format!(
                "LoRA adapter has lora_A for layer {} module '{}' but no matching lora_B",
                key.0, key.1
            ))
        })?;

        // A shape: (rank, d_in), B shape: (d_out, rank)
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(LoraError::Format(format!(
                "LoRA matrices for layer {} module '{}' must be 2D, got A={:?} B={:?}",
                key.0, key.1, a_shape, b_shape
            )));
        }

        let a_rank = a_shape[0];
        let d_in = a_shape[1];
        let d_out = b_shape[0];
        let b_rank = b_shape[1];

        if a_rank != b_rank {
            return Err(LoraError::Format(format!(
                "LoRA rank mismatch for layer {} module '{}': A rank={}, B rank={}",
                key.0, key.1, a_rank, b_rank
            )));
        }

        // Verify all layers use the same rank
        match rank {
            None => rank = Some(a_rank),
            Some(r) if r != a_rank => {
                return Err(LoraError::Format(format!(
                    "inconsistent LoRA ranks: first seen rank={}, layer {} module '{}' has rank={}",
                    r, key.0, key.1, a_rank
                )));
            }
            _ => {}
        }

        // Verify data sizes match shapes
        let expected_a = a_rank.checked_mul(d_in).ok_or_else(|| {
            LoraError::Format(format!(
                "LoRA A shape overflow for layer {} module '{}': {} * {}",
                key.0, key.1, a_rank, d_in
            ))
        })?;
        if a_data.len() != expected_a {
            return Err(LoraError::Format(format!(
                "LoRA A data size mismatch for layer {} module '{}': expected {}, got {}",
                key.0,
                key.1,
                expected_a,
                a_data.len()
            )));
        }
        let expected_b = d_out.checked_mul(b_rank).ok_or_else(|| {
            LoraError::Format(format!(
                "LoRA B shape overflow for layer {} module '{}': {} * {}",
                key.0, key.1, d_out, b_rank
            ))
        })?;
        if b_data.len() != expected_b {
            return Err(LoraError::Format(format!(
                "LoRA B data size mismatch for layer {} module '{}': expected {}, got {}",
                key.0,
                key.1,
                expected_b,
                b_data.len()
            )));
        }
    }

    // Check for orphaned B matrices
    for key in b_tensors.keys() {
        if !a_tensors.contains_key(key) {
            return Err(LoraError::Format(format!(
                "LoRA adapter has lora_B for layer {} module '{}' but no matching lora_A",
                key.0, key.1
            )));
        }
    }

    // Keep the validation and orphan passes above in their original order;
    // only move buffers once every pair has passed those checks.
    for (key, (a_data, a_shape)) in a_tensors {
        let (b_data, b_shape) = b_tensors.remove(&key).ok_or_else(|| {
            LoraError::Format(format!(
                "LoRA adapter has lora_A for layer {} module '{}' but no matching lora_B",
                key.0, key.1
            ))
        })?;
        layers.insert(
            key,
            LoraLayer {
                a: a_data,
                b: b_data,
                d_in: a_shape[1],
                d_out: b_shape[0],
                rank: a_shape[0],
            },
        );
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
                LoraError::Format(format!("invalid LoRA alpha metadata '{value}': {e}"))
            })?;
            if !alpha.is_finite() {
                return Err(LoraError::Format(format!(
                    "LoRA alpha metadata must be finite, got '{value}'"
                )));
            }
            alpha
        }
        None => rank as f32,
    };

    Ok(DecodedAdapter {
        layers,
        rank,
        alpha,
        target_modules: target_modules.into_iter().collect(),
        skipped,
    })
}

/// Read one string value from a safetensors header's metadata.
///
/// Returns `None` if the bytes cannot be parsed, the header has no metadata,
/// or `key` is absent.
pub fn read_header_metadata_value(bytes: &[u8], key: &str) -> Option<String> {
    SafeTensors::read_metadata(bytes)
        .ok()
        .and_then(|(_, meta)| meta.metadata().as_ref().and_then(|m| m.get(key).cloned()))
}

/// Read a safetensors header's whole metadata map.
///
/// Returns `None` if the bytes cannot be parsed or the header has no metadata.
pub fn read_header_metadata(bytes: &[u8]) -> Option<HashMap<String, String>> {
    let (_, meta) = SafeTensors::read_metadata(bytes).ok()?;
    meta.metadata().clone()
}

/// Read an adapter file into memory without exceeding `max_bytes`.
///
/// # Errors
///
/// [`LoraError::Io`] for metadata, over-limit, open, or read failures. The
/// messages name `path`.
pub fn read_adapter_file_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, LoraError> {
    // Reject a known-oversized file before allocation.
    let file_size = std::fs::metadata(path)
        .map_err(|e| {
            LoraError::Io(std::io::Error::new(
                e.kind(),
                format!("failed to read metadata for {}: {e}", path.display()),
            ))
        })?
        .len();
    if file_size > max_bytes {
        return Err(LoraError::Io(std::io::Error::new(
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
        LoraError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to open LoRA adapter from {}: {e}", path.display()),
        ))
    })?;
    let mut buf = Vec::new();
    f.take(max_bytes + 1).read_to_end(&mut buf).map_err(|e| {
        LoraError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to read LoRA adapter from {}: {e}", path.display()),
        ))
    })?;
    if buf.len() as u64 > max_bytes {
        return Err(LoraError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "LoRA file {} exceeds maximum of {max_bytes} bytes at read time (grew after stat)",
                path.display()
            ),
        )));
    }
    Ok(buf)
}

/// Encode factor pairs as PEFT safetensors bytes with the given header metadata.
///
/// Each pair is written as `lora_A` `(rank, d_in)` and `lora_B` `(d_out, rank)`
/// f32 tensors under `base_model.model.model.layers.<i>.<block>.<module>`. A
/// pair with an empty `a` or `b` is an untrained module and is skipped.
///
/// # Errors
///
/// [`LoraError::Format`] when a pair's buffers do not form a tensor of its
/// declared shape, or serialization fails.
pub fn encode_peft_safetensors<'a, I>(
    layers: I,
    metadata: HashMap<String, String>,
) -> Result<Vec<u8>, LoraError>
where
    I: IntoIterator<Item = (usize, &'a str, &'a LoraLayer)>,
{
    // Collect owned byte buffers first; TensorView borrows from these.
    let mut byte_data: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();

    for (layer_idx, module, layer) in layers {
        // Empty factors denote an untrained module and cannot form a tensor view.
        if layer.a.is_empty() || layer.b.is_empty() {
            continue;
        }

        let block = match module {
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
            LoraError::Format(format!("failed to create tensor view for '{name}': {e}"))
        })?;
        tensor_views.insert(name.clone(), view);
    }

    serialize(&tensor_views, &Some(metadata))
        .map_err(|e| LoraError::Format(format!("failed to serialize LoRA adapter: {e}")))
}

/// Write encoded adapter bytes to `path`.
///
/// # Errors
///
/// [`LoraError::Io`] when the write fails; the message names `path`.
pub fn write_adapter_file(path: &Path, bytes: &[u8]) -> Result<(), LoraError> {
    std::fs::write(path, bytes).map_err(|e| {
        LoraError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to write LoRA adapter to {}: {e}", path.display()),
        ))
    })
}

#[cfg(test)]
mod tests;
