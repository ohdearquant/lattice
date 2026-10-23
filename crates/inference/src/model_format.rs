//! Canonical model-directory format detector (ADR-080 amendment, #829).
//!
//! **Stability**: unstable, internal-binaries-only. This module exists so
//! internal binaries in `src/bin/` -- including the three serving/chat
//! binaries (`lattice`, `lattice_serve`, `chat_metal`) and the three
//! benchmark binaries (`bench_decode_ab`, `bench_decode_slopefit`,
//! `bench_logit_dump`) -- which are separate Cargo binary targets and
//! therefore cannot see one another's `pub(crate)` items -- can share a
//! single model-directory format decision instead of each carrying its own
//! copy. It is public but unstable and internal-use: `pub mod model_format` is
//! externally nameable, but its compatibility follows the crate's
//! Experimental stability policy rather than semver-relevant deprecation
//! cycles, and it may change shape at any time; downstream consumers of this
//! crate should go through [`crate::model`] and [`crate::forward`] instead.
//! See the crate-level `Stability tier: Experimental` note in
//! `lib.rs` and `docs/adr/ADR-080-consolidation-duplicated-contracts.md`
//! (implementation record amendment) for the full rationale.
//!
//! Before this module existed, the same detection logic (and the same
//! precedence rule -- a `model.safetensors`/`model.safetensors.index.json`
//! file always wins over a `.q4` tensor file, and an unreadable or
//! non-matching directory fails closed to [`ModelFormat::Unknown`](crate::model_format::ModelFormat::Unknown)) was
//! implemented independently in `bin/lattice.rs` (`backend::detect_format`,
//! enum-valued, with its own unit tests), `bin/lattice_serve/main.rs`
//! (`detect_q4`, bool-valued), and `bin/chat_metal.rs` (`is_q4_dir`,
//! bool-valued, plus a fourth, partial decision site that only re-checked
//! safetensors absence inside the non-Q4 branch). All three binaries now
//! match on this single [`ModelFormat`](crate::model_format::ModelFormat) enum; none re-checks sentinel files
//! after the match.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// The on-disk format of a model directory, decided before any tensor I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelFormat {
    /// `model.safetensors` or `model.safetensors.index.json` present, every
    /// listed header readable, and no quantized tensor or quantization
    /// declaration found.
    Safetensors,
    /// No safetensors file, but at least one `*.q4` tensor file present
    /// (the output of `quantize_q4`).
    Q4,
    /// Neither a safetensors file, a `.q4` file nor a `.gguf` file was found.
    Unknown,
    /// A safetensors checkpoint whose `config.json` carries an MLX
    /// `quantization` object and whose quantized modules all have a
    /// complete set of tensors. Lattice does not load this format yet.
    MlxQuantized,
    /// A single `*.gguf` file and no safetensors file. Lattice does not load
    /// this format yet.
    Gguf,
    /// A model directory that cannot be classified safely: an ambiguous file
    /// set, an unreadable header or index, a foreign quantization
    /// declaration, or a quantized tensor that nothing declares.
    /// [`inspect_format`] names the reason.
    Refused,
}

/// The format of a model directory together with the reason for a refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatDetection {
    format: ModelFormat,
    refusal: Option<String>,
}

impl FormatDetection {
    fn found(format: ModelFormat) -> Self {
        Self {
            format,
            refusal: None,
        }
    }

    fn refused(reason: String) -> Self {
        Self {
            format: ModelFormat::Refused,
            refusal: Some(reason),
        }
    }

    /// The detected format.
    pub fn format(&self) -> ModelFormat {
        self.format
    }

    /// Why the directory was refused, when [`Self::format`] is
    /// [`ModelFormat::Refused`].
    pub fn refusal(&self) -> Option<&str> {
        self.refusal.as_deref()
    }
}

/// Detect the format of a model directory. See [`inspect_format`] for the
/// rules; this drops the refusal reason.
pub fn detect_format(dir: &Path) -> ModelFormat {
    inspect_format(dir).format()
}

const SAFETENSORS_FILE: &str = "model.safetensors";
const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
/// Dtypes that only a quantized checkpoint stores as weights. A tensor of one
/// of these dtypes is admitted only as part of a declared quantized module.
const QUANTIZED_DTYPES: [&str; 6] = ["U32", "U8", "I8", "I32", "F8_E4M3", "F8_E8M0"];
const AFFINE_METADATA_DTYPES: [&str; 3] = ["F16", "BF16", "F32"];
const NON_AFFINE_SCALE_DTYPES: [&str; 3] = ["U8", "F8_E8M0", "F8_E4M3"];

/// Detect the format of a model directory, reading only directory entries,
/// `config.json`, `hf_quant_config.json`, the safetensors index and the
/// safetensors headers. No tensor payload is read.
///
/// The rules apply in order, and the first that matches decides:
///
/// 1. `Q4`: no `model.safetensors` and no `model.safetensors.index.json`, and
///    at least one `*.q4` file.
/// 2. `Gguf`: exactly one `*.gguf` file and no safetensors file. Two or more
///    `.gguf` files, or a `.gguf` file beside a safetensors file, refuse.
/// 3. `Unknown`: no `model.safetensors` and no index.
/// 4. The header set is every shard the index lists, or `model.safetensors`
///    when there is no index. An index must have a non-empty `weight_map`
///    whose entries match the shard headers exactly, and every `.safetensors`
///    file in the directory must be listed. Without an index,
///    `model.safetensors` must be the only `.safetensors` file. A header or
///    index that cannot be read refuses.
/// 5. A `config.json` that cannot be read or parsed declares nothing. A
///    quantization declaration other than an MLX `quantization` object
///    (`quantization_config` at the top level or in `text_config`, or an
///    `hf_quant_config.json` file) with no `quantization` object refuses.
/// 6. `MlxQuantized`: a `quantization` object at the top level of
///    `config.json` or in `text_config`. Every `U32` `<p>.weight` with
///    `<p>.scales` is claimed as a module: affine when the scales are F16,
///    BF16 or F32 and `<p>.biases` has the same dtype, non-affine when the
///    scales are `U8`, `F8_E8M0` or `F8_E4M3`, there are no biases, and any
///    `<p>.global_scale` is F32. A partial or mixed module refuses, and so
///    does any tensor of a quantized dtype, `scales`, `biases` or
///    `global_scale` tensor that no module claims, or a declaration with no
///    module.
/// 7. With no declaration at all, any tensor of a quantized dtype and any
///    `*.global_scale` tensor refuses.
/// 8. Otherwise `Safetensors`.
///
/// An unreadable directory resolves to `Unknown`.
pub fn inspect_format(dir: &Path) -> FormatDetection {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return FormatDetection::found(ModelFormat::Unknown);
    };
    let mut names = BTreeSet::new();
    for entry in entries {
        match entry {
            Ok(entry) => {
                names.insert(entry.file_name().to_string_lossy().into_owned());
            }
            Err(err) => {
                return FormatDetection::refused(format!(
                    "failed to list an entry of {}: {err}",
                    dir.display()
                ));
            }
        }
    }
    let has_single = dir.join(SAFETENSORS_FILE).exists();
    let has_index = dir.join(SAFETENSORS_INDEX).exists();
    let safetensors_files: Vec<&String> = names
        .iter()
        .filter(|n| n.ends_with(".safetensors"))
        .collect();

    if !has_single && !has_index && names.iter().any(|n| n.ends_with(".q4")) {
        return FormatDetection::found(ModelFormat::Q4);
    }

    let gguf_files: Vec<&String> = names.iter().filter(|n| n.ends_with(".gguf")).collect();
    if !gguf_files.is_empty() {
        if has_index || !safetensors_files.is_empty() {
            return FormatDetection::refused(format!(
                "{} holds both GGUF and safetensors files; keep one format per directory",
                dir.display()
            ));
        }
        if gguf_files.len() > 1 {
            return FormatDetection::refused(format!(
                "{} holds {} GGUF files ({}); keep exactly one",
                dir.display(),
                gguf_files.len(),
                join_names(gguf_files.iter().map(|s| s.as_str()))
            ));
        }
        return FormatDetection::found(ModelFormat::Gguf);
    }

    if !has_single && !has_index {
        return FormatDetection::found(ModelFormat::Unknown);
    }

    let tensors = match read_header_set(dir, has_index, &safetensors_files) {
        Ok(tensors) => tensors,
        Err(reason) => return FormatDetection::refused(reason),
    };
    let declarations = match read_declarations(dir) {
        Ok(declarations) => declarations,
        Err(reason) => return FormatDetection::refused(reason),
    };

    if declarations.native {
        return match claim_mlx_modules(&tensors) {
            Ok(()) => FormatDetection::found(ModelFormat::MlxQuantized),
            Err(reason) => FormatDetection::refused(format!("{}: {reason}", dir.display())),
        };
    }
    if !declarations.foreign.is_empty() {
        return FormatDetection::refused(format!(
            "{} declares a quantization lattice cannot read ({}) and no MLX \
             `quantization` object",
            dir.display(),
            declarations.foreign.join(", ")
        ));
    }
    let undeclared: Vec<String> = tensors
        .iter()
        .filter(|(name, dtype)| {
            QUANTIZED_DTYPES.contains(&dtype.as_str()) || name.ends_with(".global_scale")
        })
        .map(|(name, dtype)| format!("{name} ({dtype})"))
        .collect();
    if !undeclared.is_empty() {
        return FormatDetection::refused(format!(
            "{} holds {} quantized tensor(s) but no readable config.json declares a \
             quantization, e.g. {}",
            dir.display(),
            undeclared.len(),
            join_names(undeclared.iter().take(3).map(String::as_str))
        ));
    }
    FormatDetection::found(ModelFormat::Safetensors)
}

fn join_names<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names.collect::<Vec<_>>().join(", ")
}

/// Tensor name to dtype over the whole header set.
fn read_header_set(
    dir: &Path,
    has_index: bool,
    safetensors_files: &[&String],
) -> Result<BTreeMap<String, String>, String> {
    if !has_index {
        if safetensors_files.len() != 1 || safetensors_files[0] != SAFETENSORS_FILE {
            return Err(format!(
                "{} has no {SAFETENSORS_INDEX} and holds {} .safetensors files ({}); without an \
                 index, {SAFETENSORS_FILE} must be the only one",
                dir.display(),
                safetensors_files.len(),
                join_names(safetensors_files.iter().map(|s| s.as_str()))
            ));
        }
        return read_safetensors_header_dtypes(&dir.join(SAFETENSORS_FILE));
    }

    let index = crate::weights::parse_index(dir).map_err(|err| err.to_string())?;
    if index.weight_map.is_empty() {
        return Err(format!(
            "{} has an empty weight_map",
            dir.join(SAFETENSORS_INDEX).display()
        ));
    }
    let shards: BTreeSet<&str> = index.weight_map.values().map(String::as_str).collect();
    let listed_here: BTreeSet<&str> = shards
        .iter()
        .filter_map(|shard| {
            let mut parts = Path::new(shard)
                .components()
                .filter(|c| !matches!(c, std::path::Component::CurDir));
            match (parts.next(), parts.next()) {
                (Some(std::path::Component::Normal(name)), None) => name.to_str(),
                _ => None,
            }
        })
        .collect();
    // A symlink that resolves to a listed shard (a `model.safetensors` link
    // beside a one-shard index) holds no bytes the header set does not read.
    let listed_targets: BTreeSet<PathBuf> = listed_here
        .iter()
        .filter_map(|name| std::fs::canonicalize(dir.join(name)).ok())
        .collect();
    let unlisted: Vec<&str> = safetensors_files
        .iter()
        .map(|s| s.as_str())
        .filter(|name| !listed_here.contains(name))
        .filter(|name| {
            let path = dir.join(name);
            let is_link =
                std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink());
            !(is_link && std::fs::canonicalize(&path).is_ok_and(|p| listed_targets.contains(&p)))
        })
        .collect();
    if !unlisted.is_empty() {
        return Err(format!(
            "{} holds .safetensors file(s) its {SAFETENSORS_INDEX} does not list: {}",
            dir.display(),
            join_names(unlisted.into_iter())
        ));
    }
    let mut tensors = BTreeMap::new();
    let mut shard_of: HashMap<String, &str> = HashMap::new();
    for shard in &shards {
        let path =
            crate::weights::contained_shard_path(dir, shard).map_err(|err| err.to_string())?;
        for (name, dtype) in read_safetensors_header_dtypes(&path)? {
            if let Some(previous) = shard_of.insert(name.clone(), shard) {
                return Err(format!(
                    "tensor {name} appears in both {previous} and {shard} in {}",
                    dir.display()
                ));
            }
            tensors.insert(name, dtype);
        }
    }
    for (name, shard) in &index.weight_map {
        if shard_of.get(name.as_str()) != Some(&shard.as_str()) {
            return Err(format!(
                "{SAFETENSORS_INDEX} in {} maps {name} to {shard}, but that shard's header does \
                 not hold it",
                dir.display()
            ));
        }
    }
    if let Some(name) = shard_of
        .keys()
        .find(|name| !index.weight_map.contains_key(*name))
    {
        return Err(format!(
            "tensor {name} in shard {} is missing from {SAFETENSORS_INDEX} in {}",
            shard_of[name],
            dir.display()
        ));
    }
    Ok(tensors)
}

/// Read one safetensors header, bounded, and return tensor name to dtype.
fn read_safetensors_header_dtypes(path: &Path) -> Result<BTreeMap<String, String>, String> {
    let mut file = crate::weights::mmap_trust::open_regular_file_no_hang(path)
        .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|err| format!("failed to inspect {}: {err}", path.display()))?
        .len();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).map_err(|err| {
        format!(
            "{} is too short for a safetensors header: {err}",
            path.display()
        )
    })?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > crate::weights::f32_weights::MAX_SAFETENSORS_HEADER_BYTES as u64 {
        return Err(format!(
            "{} declares a {header_len}-byte header, over the {}-byte limit",
            path.display(),
            crate::weights::f32_weights::MAX_SAFETENSORS_HEADER_BYTES
        ));
    }
    if header_len > file_len.saturating_sub(8) {
        return Err(format!(
            "{} declares a {header_len}-byte header but holds {file_len} bytes",
            path.display()
        ));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|err| format!("failed to read the header of {}: {err}", path.display()))?;
    let parsed: serde_json::Map<String, Value> =
        serde_json::from_slice(&header).map_err(|err| {
            format!(
                "{} has a malformed safetensors header: {err}",
                path.display()
            )
        })?;
    let mut tensors = BTreeMap::new();
    for (name, entry) in parsed {
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{}: tensor {name} has no dtype", path.display()))?;
        tensors.insert(name, dtype.to_owned());
    }
    Ok(tensors)
}

struct Declarations {
    /// An MLX `quantization` object at the top level or in `text_config`.
    native: bool,
    /// Where any other quantization declaration was found.
    foreign: Vec<&'static str>,
}

fn read_declarations(dir: &Path) -> Result<Declarations, String> {
    let mut foreign = Vec::new();
    match std::fs::symlink_metadata(dir.join("hf_quant_config.json")) {
        Ok(_) => foreign.push("hf_quant_config.json"),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => {
            return Err(format!(
                "failed to inspect {}: {err}",
                dir.join("hf_quant_config.json").display()
            ));
        }
    }
    let config_path = dir.join("config.json");
    // A config.json that cannot be read declares nothing. That is safe because
    // every quantized layout stores a dtype the no-declaration rule refuses,
    // and reporting a broken config.json stays the loader's job.
    let config = match std::fs::symlink_metadata(&config_path) {
        Err(_) => None,
        Ok(_) => crate::model::config_file::read_config_json_bounded(&config_path, "config.json")
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .filter(Value::is_object),
    };
    let mut native = false;
    if let Some(config) = &config {
        let text_config = config.get("text_config");
        for (scope, holder) in [("top level", Some(config)), ("text_config", text_config)] {
            let Some(holder) = holder else { continue };
            if let Some(quantization) = holder.get("quantization") {
                if !quantization.is_object() {
                    return Err(format!(
                        "{}: `quantization` at the {scope} is not an object",
                        config_path.display()
                    ));
                }
                native = true;
            }
            if holder.get("quantization_config").is_some() {
                foreign.push(if scope == "top level" {
                    "config.json quantization_config"
                } else {
                    "text_config.quantization_config"
                });
            }
        }
    }
    Ok(Declarations { native, foreign })
}

/// Claim every MLX quantized module in the header set, refusing a partial or
/// mixed module and any quantized-looking tensor that no module claims.
fn claim_mlx_modules(tensors: &BTreeMap<String, String>) -> Result<(), String> {
    let mut claimed: BTreeSet<&str> = BTreeSet::new();
    let dtype_of = |name: &str| tensors.get(name).map(String::as_str);
    for (name, dtype) in tensors {
        let Some(prefix) = name.strip_suffix(".weight") else {
            continue;
        };
        if dtype != "U32" {
            continue;
        }
        let scales = format!("{prefix}.scales");
        let biases = format!("{prefix}.biases");
        let global = format!("{prefix}.global_scale");
        let Some(scale_dtype) = dtype_of(&scales) else {
            continue;
        };
        if AFFINE_METADATA_DTYPES.contains(&scale_dtype) {
            match dtype_of(&biases) {
                None => {
                    return Err(format!(
                        "module {prefix} has {scale_dtype} scales and no biases"
                    ));
                }
                Some(bias_dtype) if bias_dtype != scale_dtype => {
                    return Err(format!(
                        "module {prefix} has {scale_dtype} scales and {bias_dtype} biases"
                    ));
                }
                Some(_) => {}
            }
            if tensors.contains_key(&global) {
                return Err(format!("affine module {prefix} carries a global_scale"));
            }
            claimed.extend([
                name.as_str(),
                map_key(tensors, &scales),
                map_key(tensors, &biases),
            ]);
        } else if NON_AFFINE_SCALE_DTYPES.contains(&scale_dtype) {
            if tensors.contains_key(&biases) {
                return Err(format!(
                    "module {prefix} has {scale_dtype} scales beside biases"
                ));
            }
            if let Some(global_dtype) = dtype_of(&global) {
                if global_dtype != "F32" {
                    return Err(format!(
                        "module {prefix} has a {global_dtype} global_scale; only F32 is defined"
                    ));
                }
                claimed.insert(map_key(tensors, &global));
            }
            claimed.extend([name.as_str(), map_key(tensors, &scales)]);
        } else {
            return Err(format!(
                "module {prefix} has {scale_dtype} scales, which no quantization mode defines"
            ));
        }
    }
    if claimed.is_empty() {
        return Err(
            "config.json declares a `quantization` object but no tensor forms a quantized module"
                .to_owned(),
        );
    }
    for (name, dtype) in tensors {
        if claimed.contains(name.as_str()) {
            continue;
        }
        let quantized_dtype = QUANTIZED_DTYPES.contains(&dtype.as_str());
        let quantization_part = name.ends_with(".scales")
            || name.ends_with(".biases")
            || name.ends_with(".global_scale");
        if quantized_dtype || quantization_part {
            return Err(format!(
                "tensor {name} ({dtype}) is not part of a complete quantized module"
            ));
        }
    }
    Ok(())
}

/// The map's own key for `name`, so claimed names borrow from the map.
fn map_key<'a>(tensors: &'a BTreeMap<String, String>, name: &str) -> &'a str {
    tensors
        .get_key_value(name)
        .map(|(key, _)| key.as_str())
        .unwrap_or_default()
}

/// The model family a serving binary routes a checkpoint directory to.
///
/// Decided from the directory's `config.json`, not from tensor files: a Gemma
/// 4 checkpoint and a Qwen3.5 checkpoint are both `model.safetensors`
/// directories, so [`detect_format`] cannot tell them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelFamily {
    /// Qwen3.5. Also the family of every directory that is not positively
    /// identified as another one, which is what `lattice serve` has always
    /// done: the Qwen loader reports whatever is wrong with such a directory.
    Qwen35,
    /// Gemma 4, positively identified by a top-level `model_type` of
    /// `gemma4` in `config.json`. Whether the checkpoint is a supported
    /// Gemma 4 configuration is the Gemma loader's decision, not this one's.
    Gemma4,
}

impl ModelFamily {
    /// The spelling used in route markers and log lines.
    pub fn name(self) -> &'static str {
        match self {
            Self::Qwen35 => "qwen35",
            Self::Gemma4 => "gemma4",
        }
    }
}

/// Top-level `config.json` `model_type` that identifies a Gemma 4 checkpoint.
/// The nested `text_config`, `audio_config` and `vision_config` objects carry
/// their own `model_type` values, which are not read here.
const GEMMA4_MODEL_TYPE: &str = "gemma4";

/// Decide which family `dir` belongs to from its `config.json`.
///
/// Returns [`ModelFamily::Gemma4`] only when the file is readable JSON whose
/// top-level `model_type` is `gemma4`. A missing, unreadable, oversized or
/// malformed `config.json`, and any other `model_type`, resolve to
/// [`ModelFamily::Qwen35`], so a directory the Gemma route does not claim is
/// handled exactly as it was before this detector existed.
pub fn detect_family(dir: &Path) -> ModelFamily {
    let is_gemma4 = crate::model::config_file::read_config_json_bounded(
        &dir.join("config.json"),
        "config.json",
    )
    .ok()
    .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
    .and_then(|root| {
        root.get("model_type")
            .and_then(serde_json::Value::as_str)
            .map(|model_type| model_type == GEMMA4_MODEL_TYPE)
    })
    .unwrap_or(false);
    if is_gemma4 {
        ModelFamily::Gemma4
    } else {
        ModelFamily::Qwen35
    }
}

/// Error message shown when a Q4 directory is passed to a binary that was
/// built without the `metal-gpu` feature. Q4 inference only runs on the
/// Metal GPU forward pass; there is no CPU fallback for `.q4` tensors, so
/// this is a hard, fail-closed error rather than a silent degrade.
pub fn metal_gpu_required_message(dir: &Path) -> String {
    format!(
        "model directory '{}' is a native Q4 quantized checkpoint, which requires \
         the Metal GPU forward pass. This binary was built without the `metal-gpu` \
         feature. Rebuild with `--features \"f16 metal-gpu\"` (macOS only), or point \
         --model at a safetensors directory instead.",
        dir.display()
    )
}

/// Error message for a directory that is neither safetensors nor Q4.
pub fn unrecognized_format_message(dir: &Path) -> String {
    format!(
        "'{}' is not a recognized model directory: no model.safetensors, \
         model.safetensors.index.json, or *.q4 tensor files were found",
        dir.display()
    )
}

/// Error message for a directory this build cannot load as safetensors or
/// native Q4: names the detected format, or the reason the directory was
/// refused. Reads the directory again, so it is meant for error paths.
pub fn unloadable_format_message(dir: &Path) -> String {
    let detection = inspect_format(dir);
    match detection.format() {
        ModelFormat::MlxQuantized => format!(
            "'{}' is an MLX quantized checkpoint, which lattice cannot load yet",
            dir.display()
        ),
        ModelFormat::Gguf => format!(
            "'{}' is a GGUF checkpoint, which lattice cannot load yet",
            dir.display()
        ),
        ModelFormat::Refused => format!(
            "'{}' cannot be loaded: {}",
            dir.display(),
            detection
                .refusal()
                .unwrap_or("its format could not be determined")
        ),
        ModelFormat::Safetensors | ModelFormat::Q4 => format!(
            "'{}' is a {} checkpoint, which this build cannot load",
            dir.display(),
            if detection.format() == ModelFormat::Q4 {
                "native Q4"
            } else {
                "safetensors"
            }
        ),
        ModelFormat::Unknown => unrecognized_format_message(dir),
    }
}

#[cfg(test)]
mod tests;

/// ADR-100 D1 detection: real MLX headers, and one fixture per refusal.
#[cfg(test)]
mod detection_tests;
