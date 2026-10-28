//! PEFT/MLX safetensors decoding and PEFT-compatible encoding of LoRA factors.
//!
//! A factor pair is keyed by transformer layer and projection: A is
//! `(rank, d_in)` and B is `(d_out, rank)` after MLX normalization. Decoding
//! rejects incomplete, malformed, non-finite, or rank-inconsistent pairs.
//! File reads are bounded before parsing.
//!
//! A PEFT adapter keeps its `lora_alpha` in an `adapter_config.json` beside the
//! weights rather than in the safetensors header. [`read_peft_sidecar`] reads
//! that file and [`apply_peft_sidecar`] checks it against the decoded factors
//! and takes its alpha.
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
    /// The header `alpha`, or `rank` when the header has none, until
    /// [`apply_peft_sidecar`] replaces it.
    pub alpha: f32,
    /// Where `alpha` came from.
    pub alpha_source: AlphaSource,
    /// Whether any factor uses the MLX key form.
    pub has_mlx_factors: bool,
    /// The PEFT module path of each decoded PEFT-form factor, sorted: the key
    /// without a leading `base_model.model.` and without its factor suffix,
    /// for example `model.layers.0.self_attn.q_proj`.
    pub peft_module_paths: Vec<String>,
    /// The projection modules present, sorted.
    pub target_modules: Vec<String>,
    /// Unknown-module factor tensor names skipped in sorted order.
    pub skipped: Vec<String>,
}

/// Where a decoded adapter's alpha came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AlphaSource {
    /// The safetensors header `alpha` metadata value.
    Header,
    /// The `lora_alpha` of a PEFT `adapter_config.json`.
    PeftSidecar,
    /// No source declared alpha, so it equals the rank.
    RankDefault,
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
/// non-finite factor, mix PEFT and MLX key forms anywhere in the file, or
/// carry an unparsable or non-finite `alpha`.
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
    let mut has_mlx_factors = false;
    let mut peft_module_paths = BTreeSet::new();

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
            if peft_key.transposed {
                has_mlx_factors = true;
            } else {
                peft_module_paths.insert(peft_module_path(name));
            }

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

    // A PEFT sidecar applies to the whole file or not at all, so one file must
    // use one key form even when each module is internally consistent.
    let first_of_form = |mlx: bool| {
        dialects
            .values()
            .filter(|(form, _)| *form == mlx)
            .map(|(_, name)| *name)
            .min()
    };
    if let (Some(peft), Some(mlx)) = (first_of_form(false), first_of_form(true)) {
        return Err(LoraError::Format(format!(
            "LoRA adapter mixes PEFT and MLX key forms across modules: '{peft}' and '{mlx}'"
        )));
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
    let alpha_source = if alpha_metadata.is_some() {
        AlphaSource::Header
    } else {
        AlphaSource::RankDefault
    };
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
        alpha_source,
        has_mlx_factors,
        peft_module_paths: peft_module_paths.into_iter().collect(),
        target_modules: target_modules.into_iter().collect(),
        skipped,
    })
}

/// The module path a PEFT factor key names: the key without a leading
/// `base_model.model.` and without its factor suffix.
fn peft_module_path(key: &str) -> String {
    let stem = [".lora_A.weight", ".lora_B.weight", ".lora_A", ".lora_B"]
        .iter()
        .find_map(|suffix| key.strip_suffix(suffix))
        .unwrap_or(key);
    stem.strip_prefix("base_model.model.")
        .unwrap_or(stem)
        .to_string()
}

/// File name of the PEFT adapter config that sits beside the weights.
pub const PEFT_SIDECAR_FILE_NAME: &str = "adapter_config.json";

/// Maximum PEFT adapter config size read (1 MiB).
pub const MAX_PEFT_SIDECAR_BYTES: u64 = 1024 * 1024;

/// PEFT `target_modules`: a list of module names, or one regular expression.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TargetModules {
    /// A module path matches when it equals a name or ends in `.` plus a name.
    Names(Vec<String>),
    /// A module path matches when the expression matches all of it.
    Pattern(String),
}

impl TargetModules {
    fn from_json(field: &str, value: &serde_json::Value) -> Result<Self, LoraError> {
        match value {
            serde_json::Value::String(pattern) => {
                if pattern == "all-linear" {
                    return Err(LoraError::Unsupported(format!(
                        "PEFT {field} \"all-linear\" is not supported; list the modules"
                    )));
                }
                full_match_regex(pattern).map_err(|e| {
                    LoraError::Unsupported(format!(
                        "PEFT {field} pattern '{pattern}' is not supported: {e}"
                    ))
                })?;
                Ok(Self::Pattern(pattern.clone()))
            }
            serde_json::Value::Array(items) => {
                let names = items
                    .iter()
                    .map(|item| item.as_str().map(String::from))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| LoraError::Format(format!("PEFT {field} must list strings")))?;
                if names.is_empty() {
                    return Err(LoraError::Format(format!("PEFT {field} is empty")));
                }
                Ok(Self::Names(names))
            }
            _ => Err(LoraError::Format(format!(
                "PEFT {field} must be a list of names or a pattern"
            ))),
        }
    }

    fn matcher(&self) -> Result<ModuleMatcher<'_>, LoraError> {
        Ok(match self {
            Self::Names(names) => ModuleMatcher::Names(names),
            Self::Pattern(pattern) => ModuleMatcher::Pattern(
                full_match_regex(pattern).map_err(|e| LoraError::Unsupported(e.to_string()))?,
            ),
        })
    }
}

enum ModuleMatcher<'a> {
    Names(&'a [String]),
    Pattern(regex::Regex),
}

impl ModuleMatcher<'_> {
    fn matches(&self, path: &str) -> bool {
        match self {
            Self::Names(names) => names.iter().any(|name| {
                path == name
                    || path
                        .strip_suffix(name.as_str())
                        .is_some_and(|head| head.ends_with('.'))
            }),
            Self::Pattern(re) => re.is_match(path),
        }
    }
}

/// PEFT matches a string `target_modules` against the whole module path.
///
/// The pattern is compiled alone first, so one that would close the wrapping
/// group early (`q_proj)|(?:.*`) is refused instead of matching every path.
fn full_match_regex(pattern: &str) -> Result<regex::Regex, regex::Error> {
    regex::Regex::new(pattern)?;
    regex::Regex::new(&format!("^(?:{pattern})$"))
}

/// The fields of a PEFT `adapter_config.json` that decide how an adapter loads.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PeftSidecar {
    /// `lora_alpha`, authoritative for a PEFT adapter.
    pub lora_alpha: f32,
    /// `r`, which must equal the factor rank.
    pub r: usize,
    /// `target_modules`, which must cover every decoded module path.
    pub target_modules: TargetModules,
    /// `exclude_modules`, which must match no decoded module path.
    pub exclude_modules: Option<TargetModules>,
}

/// Parse a PEFT `adapter_config.json`.
///
/// # Errors
///
/// [`LoraError::Format`] when the file is not a JSON object, `lora_alpha` is
/// missing, not a number or not finite, `r` is missing or not a positive
/// integer, or `target_modules` is missing or malformed.
/// [`LoraError::Unsupported`] for a `peft_type` other than `LORA`,
/// `use_rslora`, `use_dora`, a non-empty `rank_pattern` or `alpha_pattern`,
/// trained biases, `modules_to_save`, `trainable_token_indices`, or a
/// `target_modules` pattern this reader cannot evaluate. These are refused,
/// never approximated.
pub fn parse_peft_sidecar(json: &[u8]) -> Result<PeftSidecar, LoraError> {
    use serde_json::Value;
    let value: Value = serde_json::from_slice(json)
        .map_err(|e| LoraError::Format(format!("invalid PEFT adapter config JSON: {e}")))?;
    let config = value
        .as_object()
        .ok_or_else(|| LoraError::Format("PEFT adapter config must be a JSON object".into()))?;
    let field = |name: &str| config.get(name).filter(|v| !v.is_null());

    if let Some(peft_type) = field("peft_type")
        && peft_type.as_str() != Some("LORA")
    {
        return Err(LoraError::Unsupported(format!(
            "PEFT adapter type {peft_type} is not supported; only LORA is"
        )));
    }
    for flag in ["use_rslora", "use_dora", "lora_bias"] {
        if field(flag).is_some_and(|v| v.as_bool() != Some(false)) {
            return Err(LoraError::Unsupported(format!(
                "PEFT {flag} is not supported"
            )));
        }
    }
    for pattern in ["rank_pattern", "alpha_pattern"] {
        if field(pattern).is_some_and(|v| v.as_object().is_none_or(|m| !m.is_empty())) {
            return Err(LoraError::Unsupported(format!(
                "PEFT {pattern} is not supported"
            )));
        }
    }
    if field("bias").is_some_and(|v| v.as_str() != Some("none")) {
        return Err(LoraError::Unsupported(
            "PEFT trained biases are not supported; bias must be \"none\"".into(),
        ));
    }
    if field("modules_to_save").is_some_and(|v| v.as_array().is_none_or(|a| !a.is_empty())) {
        return Err(LoraError::Unsupported(
            "PEFT modules_to_save is not supported".into(),
        ));
    }
    if field("trainable_token_indices").is_some() {
        return Err(LoraError::Unsupported(
            "PEFT trainable_token_indices is not supported".into(),
        ));
    }

    let lora_alpha = field("lora_alpha")
        .ok_or_else(|| LoraError::Format("PEFT adapter config has no lora_alpha".into()))?
        .as_f64()
        .ok_or_else(|| LoraError::Format("PEFT lora_alpha must be a number".into()))?
        as f32;
    if !lora_alpha.is_finite() {
        return Err(LoraError::Format(format!(
            "PEFT lora_alpha must be finite, got {lora_alpha}"
        )));
    }
    let r = field("r")
        .ok_or_else(|| LoraError::Format("PEFT adapter config has no r".into()))?
        .as_u64()
        .and_then(|r| usize::try_from(r).ok())
        .filter(|&r| r > 0)
        .ok_or_else(|| LoraError::Format("PEFT r must be a positive integer".into()))?;
    let target_modules = TargetModules::from_json(
        "target_modules",
        field("target_modules")
            .ok_or_else(|| LoraError::Format("PEFT adapter config has no target_modules".into()))?,
    )?;
    let exclude_modules = field("exclude_modules")
        .map(|v| TargetModules::from_json("exclude_modules", v))
        .transpose()?;

    Ok(PeftSidecar {
        lora_alpha,
        r,
        target_modules,
        exclude_modules,
    })
}

/// Read the PEFT `adapter_config.json` beside an adapter file.
///
/// Returns `Ok(None)` only when no entry of that name exists. Any other state,
/// including an unreadable, oversized or malformed file, is an error, so a
/// broken config is never mistaken for an absent one.
///
/// # Errors
///
/// [`LoraError::Io`] for read failures, and the errors of
/// [`parse_peft_sidecar`] with the config's path in the message.
pub fn read_peft_sidecar(adapter_path: &Path) -> Result<Option<PeftSidecar>, LoraError> {
    let path = adapter_path.with_file_name(PEFT_SIDECAR_FILE_NAME);
    match std::fs::symlink_metadata(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(LoraError::Io(std::io::Error::new(
                e.kind(),
                format!("failed to read metadata for {}: {e}", path.display()),
            )));
        }
        Ok(_) => {}
    }
    let bytes = read_adapter_file_bounded(&path, MAX_PEFT_SIDECAR_BYTES)?;
    parse_peft_sidecar(&bytes).map(Some).map_err(|e| match e {
        LoraError::Format(msg) => LoraError::Format(format!("{}: {msg}", path.display())),
        LoraError::Unsupported(msg) => LoraError::Unsupported(format!("{}: {msg}", path.display())),
        other => other,
    })
}

/// Check a PEFT config against decoded factors and take its alpha.
///
/// The adapter is changed only when every check passes.
///
/// # Errors
///
/// [`LoraError::Validation`] when the adapter has MLX-form factors, the header
/// `alpha` differs from `lora_alpha` (both are named), `r` differs from the
/// factor rank, a decoded module path is not covered by `target_modules`, or
/// one matches `exclude_modules`.
pub fn apply_peft_sidecar(
    adapter: &mut DecodedAdapter,
    sidecar: &PeftSidecar,
) -> Result<(), LoraError> {
    if adapter.has_mlx_factors {
        return Err(LoraError::Validation(
            "a PEFT adapter config does not apply to MLX-form factors".into(),
        ));
    }
    if adapter.alpha_source == AlphaSource::Header && adapter.alpha != sidecar.lora_alpha {
        return Err(LoraError::Validation(format!(
            "LoRA alpha disagrees: safetensors header alpha {} but PEFT lora_alpha {}",
            adapter.alpha, sidecar.lora_alpha
        )));
    }
    if adapter.rank != sidecar.r {
        return Err(LoraError::Validation(format!(
            "LoRA rank disagrees: factors have rank {} but PEFT r is {}",
            adapter.rank, sidecar.r
        )));
    }
    let targets = sidecar.target_modules.matcher()?;
    if let Some(path) = adapter
        .peft_module_paths
        .iter()
        .find(|path| !targets.matches(path))
    {
        return Err(LoraError::Validation(format!(
            "LoRA module '{path}' is not covered by PEFT target_modules"
        )));
    }
    if let Some(exclude) = &sidecar.exclude_modules {
        let exclude = exclude.matcher()?;
        if let Some(path) = adapter
            .peft_module_paths
            .iter()
            .find(|path| exclude.matches(path))
        {
            return Err(LoraError::Validation(format!(
                "LoRA module '{path}' matches PEFT exclude_modules"
            )));
        }
    }
    adapter.alpha = sidecar.lora_alpha;
    adapter.alpha_source = AlphaSource::PeftSidecar;
    Ok(())
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
    f.take(max_bytes.saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|e| {
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
