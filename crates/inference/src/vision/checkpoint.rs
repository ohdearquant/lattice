//! Loader for the real Qwen3.5 vision-language checkpoint's 153 `model.visual.*`
//! tensors (ADR-069 stage S2).
//!
//! This is deliberately independent from [`super::vit::VisionWeights`] (the ADR-049
//! CPU ViT scaffold, which targets a hypothetical 7B model). Reconciling that
//! scaffold's forward-pass semantics to the real 12-layer/768-hidden geometry —
//! including its temporal-patch-folded patch embedding (`[hidden, in_channels,
//! temporal_patch_size, patch_size, patch_size]`, vs. the scaffold's
//! `[d_model, patch_size^2 * 3]` assumption with no temporal factor) and dropping
//! the scaffold's windowed-attention assumption (the real checkpoint has no
//! window-specific weights) — is ADR-069 stage S3a (CPU reference; the Metal
//! port is a separate S3b fast-follow gated against this CPU reference), not
//! S1/S2. This module only loads the real tensors as flat data; it does not
//! wire them into any forward pass.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::error::InferenceError;
use crate::model::qwen35_config::VisionModelConfig;
use crate::quant::q4_manifest;
use crate::weights::f32_weights::{
    SafetensorsFile, ShardedSafetensors, TensorSource, contained_shard_path,
    open_manifest_entry_once, parse_index,
};
use crate::weights::q4_weights::{
    F16LoadError, dequantize_q4_to_f32, load_f16_tensor_from_open_file,
    load_f16_tensor_from_open_file_expecting, load_q4_from_open_file,
};

/// Tensor name to its dequantized data paired with the shape it was declared with.
///
/// The shape travels alongside the data because a checkpoint's declared shape is what
/// preflight validates against, and it must stay available after materialization for
/// the sources that cannot report a shape without reading the tensor.
type NamedTensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// One ViT transformer block's real tensors (`model.visual.blocks.{i}.*`).
#[derive(Debug, Clone)]
pub struct VisualBlockWeights {
    /// `attn.qkv.weight` — fused Q/K/V projection, `[3 * hidden_size, hidden_size]`.
    pub qkv_weight: Vec<f32>,
    /// `attn.qkv.bias` — `[3 * hidden_size]`.
    pub qkv_bias: Vec<f32>,
    /// `attn.proj.weight` — output projection, `[hidden_size, hidden_size]`.
    pub proj_weight: Vec<f32>,
    /// `attn.proj.bias` — `[hidden_size]`.
    pub proj_bias: Vec<f32>,
    /// `mlp.linear_fc1.weight` — `[4 * hidden_size, hidden_size]`.
    pub fc1_weight: Vec<f32>,
    /// `mlp.linear_fc1.bias` — `[4 * hidden_size]`.
    pub fc1_bias: Vec<f32>,
    /// `mlp.linear_fc2.weight` — `[hidden_size, 4 * hidden_size]`.
    pub fc2_weight: Vec<f32>,
    /// `mlp.linear_fc2.bias` — `[hidden_size]`.
    pub fc2_bias: Vec<f32>,
    /// `norm1.weight` — `[hidden_size]`.
    pub norm1_weight: Vec<f32>,
    /// `norm1.bias` — `[hidden_size]`.
    pub norm1_bias: Vec<f32>,
    /// `norm2.weight` — `[hidden_size]`.
    pub norm2_weight: Vec<f32>,
    /// `norm2.bias` — `[hidden_size]`.
    pub norm2_bias: Vec<f32>,
}

/// The `model.visual.merger.*` tensors: spatial-merge MLP projecting ViT output into
/// the decoder's embedding space.
#[derive(Debug, Clone)]
pub struct VisualMergerWeights {
    /// `merger.linear_fc1.weight` — `[spatial_merge_size^2 * hidden_size, spatial_merge_size^2 * hidden_size]`.
    pub fc1_weight: Vec<f32>,
    /// `merger.linear_fc1.bias`.
    pub fc1_bias: Vec<f32>,
    /// `merger.linear_fc2.weight` — `[out_hidden_size, spatial_merge_size^2 * hidden_size]`.
    pub fc2_weight: Vec<f32>,
    /// `merger.linear_fc2.bias` — `[out_hidden_size]`.
    pub fc2_bias: Vec<f32>,
    /// `merger.norm.weight` — `[hidden_size]`.
    pub norm_weight: Vec<f32>,
    /// `merger.norm.bias` — `[hidden_size]`.
    pub norm_bias: Vec<f32>,
}

/// All 153 real `model.visual.*` tensors from the Qwen3.5 vision-language checkpoint
/// (ADR-069 S2): `patch_embed.proj.{weight,bias}` + `pos_embed.weight` + `depth` blocks
/// (12 tensors each) + 6 merger tensors. Distinct from [`super::vit::VisionWeights`] —
/// see the module docs for why.
#[derive(Debug, Clone)]
pub struct Qwen35VisionWeights {
    /// `patch_embed.proj.weight`, flattened. Raw shape is `[hidden_size, in_channels,
    /// temporal_patch_size, patch_size, patch_size]` (a Conv3d-shaped weight); this
    /// loader does not reinterpret it, only carries it plus `patch_embed_weight_shape`.
    pub patch_embed_weight: Vec<f32>,
    /// Raw shape of `patch_embed_weight`, as above.
    pub patch_embed_weight_shape: Vec<usize>,
    /// `patch_embed.proj.bias` — `[hidden_size]`.
    pub patch_embed_bias: Vec<f32>,
    /// `pos_embed.weight` — `[num_position_embeddings, hidden_size]`.
    pub pos_embed: Vec<f32>,
    /// Per-block weights, length == `vision_cfg.depth`.
    pub blocks: Vec<VisualBlockWeights>,
    /// The spatial-merge projection MLP.
    pub merger: VisualMergerWeights,
}

impl Qwen35VisionWeights {
    /// Total tensor count actually loaded: `2 (patch_embed) + 1 (pos_embed) + depth * 12
    /// (per-block) + 6 (merger)`. For the real 0.8B checkpoint (`depth == 12`) this is 153.
    pub fn tensor_count(&self) -> usize {
        2 + 1 + self.blocks.len() * 12 + 6
    }
}

/// Load the real `model.visual.*` tensors from a model directory. Manifest-backed
/// checkpoints retain their existing precedence: `quantize_index.json`, then
/// `model.safetensors.index.json`. When neither manifest exists, exactly one regular
/// `*.safetensors` file must be present and it must be named `model.safetensors`.
/// Its header is parsed and its exact visual tensor inventory is validated before any
/// tensor payload is materialized.
///
/// # Errors
///
/// Returns [`InferenceError::ModelNotFound`] if no supported checkpoint is present,
/// [`InferenceError::InvalidSafetensors`] when an unindexed layout is ambiguous or its
/// header is corrupt, and [`InferenceError::MissingTensor`] /
/// [`InferenceError::ShapeMismatch`] if an expected tensor (derived from `vision_cfg`)
/// is absent or has the wrong shape.
pub fn load_qwen35_vision_weights(
    model_dir: &Path,
    vision_cfg: &VisionModelConfig,
) -> Result<Qwen35VisionWeights, InferenceError> {
    let mut never_cancel = || false;
    match load_qwen35_vision_weights_with_cancel(model_dir, vision_cfg, &mut never_cancel)? {
        Some(weights) => Ok(weights),
        None => Err(InferenceError::Inference(
            "non-cancellable vision weight load was cancelled".to_string(),
        )),
    }
}

/// **Unstable**: the reader-based checkpoint-loading surface may evolve before 1.0.
///
/// Load Qwen3.5 `model.visual.*` tensors from an already-open safetensors
/// reader. This is the descriptor-binding entry point for callers that also
/// load decoder tensors from the same [`SafetensorsFile`]: both components
/// then come from one mmap-backed open file even if the directory entry is
/// replaced while a multi-gigabyte checkpoint is loading.
///
/// The reader may also contain decoder tensors, but its `model.visual.*`
/// inventory must exactly match `vision_cfg`. `checkpoint_path` is used only
/// for attributable error messages; it does not reopen the file.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidSafetensors`] for a corrupt header,
/// [`InferenceError::MissingTensor`] or [`InferenceError::ShapeMismatch`] for
/// an incompatible visual inventory, and an inference error when a tensor
/// cannot be materialized safely.
pub fn load_qwen35_vision_weights_from_safetensors(
    reader: &mut SafetensorsFile,
    checkpoint_path: &Path,
    vision_cfg: &VisionModelConfig,
) -> Result<Qwen35VisionWeights, InferenceError> {
    vision_cfg.validate()?;
    let mut never_cancel = || false;
    match load_from_safetensors_reader_with_cancel(
        reader,
        checkpoint_path,
        vision_cfg,
        &mut never_cancel,
    )? {
        Some(weights) => Ok(weights),
        None => Err(InferenceError::Inference(
            "non-cancellable vision weight load was cancelled".to_string(),
        )),
    }
}

pub(crate) fn load_qwen35_vision_weights_with_cancel(
    model_dir: &Path,
    vision_cfg: &VisionModelConfig,
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Option<Qwen35VisionWeights>, InferenceError> {
    if should_cancel() {
        return Ok(None);
    }
    // Callers can construct `VisionModelConfig` directly (it's a public struct), so this
    // boundary re-validates rather than trusting that every caller went through
    // `Qwen35Config::from_config_json_str`'s parse-time check.
    vision_cfg.validate()?;
    if checkpoint_entry_present(&model_dir.join("quantize_index.json"))? {
        load_from_q4_dir_with_cancel(model_dir, vision_cfg, should_cancel)
    } else if checkpoint_entry_present(&model_dir.join("model.safetensors.index.json"))? {
        load_from_fp16_dir_with_cancel(model_dir, vision_cfg, should_cancel)
    } else {
        load_from_single_file_with_cancel(model_dir, vision_cfg, should_cancel)
    }
}

/// Validate the exact vision tensor inventory without materializing tensor data.
///
/// This is the serving capability preflight: every expected `model.visual.*`
/// name must occur exactly once and every referenced shard/file must be
/// openable before HTTP request normalization advertises vision support. For
/// an unindexed `model.safetensors`, the bounded safetensors header is parsed
/// and every expected tensor shape is checked without decoding tensor values.
pub fn validate_qwen35_vision_weight_inventory(
    model_dir: &Path,
    vision_cfg: &VisionModelConfig,
) -> Result<(), InferenceError> {
    vision_cfg.validate()?;
    let expected: HashSet<String> = tensor_names(vision_cfg).into_iter().collect();
    if checkpoint_entry_present(&model_dir.join("quantize_index.json"))? {
        let manifest = q4_manifest::load_manifest(model_dir)
            .map_err(|err| {
                InferenceError::InvalidSafetensors(format!(
                    "failed to read quantize_index.json in {}: {err}",
                    model_dir.display()
                ))
            })?
            .ok_or_else(|| {
                InferenceError::ModelNotFound(format!(
                    "quantize_index.json missing in {}",
                    model_dir.display()
                ))
            })?;
        let mut actual = HashSet::with_capacity(expected.len());
        for entry in manifest
            .tensors
            .iter()
            .filter(|entry| entry.name.starts_with("model.visual."))
        {
            if !actual.insert(entry.name.clone()) {
                return Err(InferenceError::Inference(format!(
                    "vision checkpoint manifest in {} contains duplicate tensor {}",
                    model_dir.display(),
                    entry.name
                )));
            }
            let _ = open_manifest_entry_once(model_dir, &entry.file)?;
        }
        if actual != expected {
            return Err(InferenceError::Inference(format!(
                "vision checkpoint inventory mismatch in {}: found {} unique model.visual.* \
                 tensor(s), expected {} for depth {}",
                model_dir.display(),
                actual.len(),
                expected.len(),
                vision_cfg.depth
            )));
        }
        return Ok(());
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if checkpoint_entry_present(&index_path)? {
        let reader = ShardedSafetensors::open_index(&index_path)?;
        let actual: HashSet<String> = reader
            .index()
            .weight_map
            .keys()
            .filter(|name| name.starts_with("model.visual."))
            .cloned()
            .collect();
        if actual != expected {
            return Err(InferenceError::Inference(format!(
                "vision checkpoint inventory mismatch in {}: found {} unique model.visual.* \
                 tensor(s), expected {} for depth {}",
                index_path.display(),
                actual.len(),
                expected.len(),
                vision_cfg.depth
            )));
        }
        let shards: HashSet<&str> = expected
            .iter()
            .filter_map(|name| reader.index().weight_map.get(name).map(String::as_str))
            .collect();
        for shard in shards {
            let _ = open_manifest_entry_once(model_dir, shard)?;
        }
        return Ok(());
    }

    let single_path = resolve_qwen35_single_decoder_safetensors(model_dir)?;
    let reader = SafetensorsFile::open(&single_path)?;
    validate_single_file_inventory(&reader, &single_path, vision_cfg)
}

/// **Unstable**: checkpoint-layout resolution may evolve before 1.0.
///
/// Resolve the one safetensors file required by the pooled vision-embedding
/// decoder loader. An existing `model.safetensors.index.json` is authoritative,
/// even when a convenience `model.safetensors` also exists; its `weight_map`
/// must name exactly one contained shard. Without an index, the directory must
/// contain exactly one `*.safetensors` candidate whose resolved metadata is a
/// file, and that candidate must be named `model.safetensors`. Symlinks are
/// followed consistently with the checkpoint mmap trust policy.
///
/// This is the shared file-set policy used by `lattice-embed`'s
/// `VisionEmbeddingModel`: decoder and visual tensors cannot independently
/// select the plain and indexed layouts when both sentinels are present. The
/// embed constructor opens the returned path once and loads both components
/// through that reader; this resolver itself performs structural discovery
/// and does not compute a checkpoint content digest.
///
/// # Errors
///
/// Returns [`InferenceError::ModelNotFound`] when no supported single-file
/// decoder checkpoint exists, or [`InferenceError::InvalidSafetensors`] when
/// an index escapes the model directory or an unindexed layout is ambiguous.
pub fn resolve_qwen35_single_decoder_safetensors(
    model_dir: &Path,
) -> Result<PathBuf, InferenceError> {
    Ok(resolve_qwen35_single_checkpoint_file_set(model_dir)?.path)
}

/// **Unstable**: checkpoint-layout opening may evolve before 1.0.
///
/// Resolve and open the one safetensors file required by the pooled
/// vision-embedding decoder loader. The returned [`SafetensorsFile`] is the
/// only open of the selected checkpoint needed to materialize both visual and
/// decoder tensors. For an indexed layout, the authoritative `weight_map`
/// must exactly match the opened shard's header inventory; a sparse, stale, or
/// otherwise contradictory index fails before any tensor payload is decoded.
///
/// # Errors
///
/// Returns the same layout errors as
/// [`resolve_qwen35_single_decoder_safetensors`], an invalid-safetensors error
/// when an indexed `weight_map` and shard header disagree, or an open/header
/// error from [`SafetensorsFile`].
pub fn open_qwen35_single_decoder_safetensors(
    model_dir: &Path,
) -> Result<(SafetensorsFile, PathBuf), InferenceError> {
    let resolved = resolve_qwen35_single_checkpoint_file_set(model_dir)?;
    let reader = SafetensorsFile::open(&resolved.path)?;
    if let Some(indexed_weight_map) = &resolved.indexed_weight_map {
        validate_indexed_single_shard_inventory(&reader, &resolved.path, indexed_weight_map)?;
    }
    Ok((reader, resolved.path))
}

struct ResolvedQwen35SingleCheckpoint {
    path: PathBuf,
    indexed_weight_map: Option<HashMap<String, String>>,
}

fn resolve_qwen35_single_checkpoint_file_set(
    model_dir: &Path,
) -> Result<ResolvedQwen35SingleCheckpoint, InferenceError> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if checkpoint_entry_present(&index_path)? {
        let index = parse_index(model_dir)?;
        let mut shards: Vec<String> = index.weight_map.values().cloned().collect();
        shards.sort_unstable();
        shards.dedup();
        return match shards.as_slice() {
            [one] => Ok(ResolvedQwen35SingleCheckpoint {
                path: contained_shard_path(model_dir, one)?,
                indexed_weight_map: Some(index.weight_map),
            }),
            [] => Err(InferenceError::InvalidSafetensors(format!(
                "empty weight_map in {}",
                index_path.display()
            ))),
            _ => Err(InferenceError::InvalidSafetensors(format!(
                "checkpoint at {} is sharded across {} files; pooled vision embedding requires \
                 one decoder safetensors shard",
                model_dir.display(),
                shards.len()
            ))),
        };
    }

    Ok(ResolvedQwen35SingleCheckpoint {
        path: resolve_unindexed_safetensors(model_dir)?,
        indexed_weight_map: None,
    })
}

fn validate_indexed_single_shard_inventory(
    reader: &SafetensorsFile,
    checkpoint_path: &Path,
    indexed_weight_map: &HashMap<String, String>,
) -> Result<(), InferenceError> {
    let declared: HashSet<&str> = indexed_weight_map.keys().map(String::as_str).collect();
    let actual: HashSet<&str> = reader.tensor_names().into_iter().collect();
    if declared == actual {
        return Ok(());
    }

    let mut missing_from_shard: Vec<&str> = declared.difference(&actual).copied().collect();
    missing_from_shard.sort_unstable();
    let mut missing_from_index: Vec<&str> = actual.difference(&declared).copied().collect();
    missing_from_index.sort_unstable();
    Err(InferenceError::InvalidSafetensors(format!(
        "authoritative model.safetensors.index.json weight_map/header inventory mismatch for {}: \
         weight_map declares {} tensor(s), shard header contains {}; missing_from_shard={:?}, \
         missing_from_index={:?}",
        checkpoint_path.display(),
        declared.len(),
        actual.len(),
        missing_from_shard.into_iter().take(5).collect::<Vec<_>>(),
        missing_from_index.into_iter().take(5).collect::<Vec<_>>()
    )))
}

fn checkpoint_entry_present(path: &Path) -> Result<bool, InferenceError> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(InferenceError::InvalidSafetensors(format!(
            "failed to inspect checkpoint sentinel {}: {err}",
            path.display()
        ))),
    }
}

fn resolve_unindexed_safetensors(model_dir: &Path) -> Result<PathBuf, InferenceError> {
    let entries = std::fs::read_dir(model_dir).map_err(|err| {
        InferenceError::InvalidSafetensors(format!(
            "failed to inspect unindexed checkpoint directory {}: {err}",
            model_dir.display()
        ))
    })?;
    let mut candidates = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| {
            InferenceError::InvalidSafetensors(format!(
                "failed to inspect an entry in unindexed checkpoint directory {}: {err}",
                model_dir.display()
            ))
        })?;
        let path = entry.path();
        if path.extension() != Some(std::ffi::OsStr::new("safetensors")) {
            continue;
        }
        let metadata = path.metadata().map_err(|err| {
            InferenceError::InvalidSafetensors(format!(
                "failed to inspect safetensors candidate {}: {err}",
                path.display()
            ))
        })?;
        if metadata.is_file() {
            candidates.push(path);
        }
    }
    candidates.sort();

    match candidates.as_slice() {
        [only] if only.file_name() == Some(std::ffi::OsStr::new("model.safetensors")) => {
            Ok(only.clone())
        }
        [] => Err(InferenceError::ModelNotFound(format!(
            "no model.safetensors.index.json or model.safetensors in {}",
            model_dir.display()
        ))),
        [only] => Err(InferenceError::ModelNotFound(format!(
            "unindexed checkpoint in {} contains {}, but the sole safetensors file must be named \
             model.safetensors",
            model_dir.display(),
            only.display()
        ))),
        _ => Err(InferenceError::InvalidSafetensors(format!(
            "ambiguous unindexed checkpoint in {}: found {} safetensors files; provide exactly \
             one model.safetensors or a model.safetensors.index.json",
            model_dir.display(),
            candidates.len()
        ))),
    }
}

fn validate_single_file_inventory(
    reader: &SafetensorsFile,
    checkpoint_path: &Path,
    vision_cfg: &VisionModelConfig,
) -> Result<(), InferenceError> {
    let expected_names = tensor_names(vision_cfg);
    let expected: HashSet<&str> = expected_names.iter().map(String::as_str).collect();
    let actual: HashSet<String> = reader
        .tensor_names()
        .into_iter()
        .filter(|name| name.starts_with("model.visual."))
        .map(str::to_string)
        .collect();
    if actual.len() != expected.len() || actual.iter().any(|name| !expected.contains(name.as_str()))
    {
        let mut missing: Vec<&str> = expected
            .iter()
            .copied()
            .filter(|name| !actual.contains(*name))
            .collect();
        missing.sort_unstable();
        let mut unexpected: Vec<&str> = actual
            .iter()
            .map(String::as_str)
            .filter(|name| !expected.contains(name))
            .collect();
        unexpected.sort_unstable();
        return Err(InferenceError::Inference(format!(
            "vision checkpoint inventory mismatch in {}: found {} unique model.visual.* \
             tensor(s), expected {} for depth {}; missing={:?}, unexpected={:?}",
            checkpoint_path.display(),
            actual.len(),
            expected.len(),
            vision_cfg.depth,
            missing.into_iter().take(5).collect::<Vec<_>>(),
            unexpected.into_iter().take(5).collect::<Vec<_>>()
        )));
    }

    for name in &expected_names {
        if let Some(expected_shape) = expected_visual_tensor_shape(name, vision_cfg) {
            let actual_shape = reader
                .tensor_shape(name)
                .ok_or_else(|| InferenceError::MissingTensor(name.clone()))?;
            if actual_shape != expected_shape {
                return Err(InferenceError::ShapeMismatch {
                    name: name.clone(),
                    expected: expected_shape,
                    actual: actual_shape.to_vec(),
                });
            }
        }
    }
    Ok(())
}

/// The 153 real tensor names for a `depth`-block checkpoint (used to drive the fp16
/// sharded-safetensors fetch; the q4 path instead filters the manifest by prefix).
fn tensor_names(vision_cfg: &VisionModelConfig) -> Vec<String> {
    let mut names = vec![
        "model.visual.patch_embed.proj.weight".to_string(),
        "model.visual.patch_embed.proj.bias".to_string(),
        "model.visual.pos_embed.weight".to_string(),
        "model.visual.merger.linear_fc1.weight".to_string(),
        "model.visual.merger.linear_fc1.bias".to_string(),
        "model.visual.merger.linear_fc2.weight".to_string(),
        "model.visual.merger.linear_fc2.bias".to_string(),
        "model.visual.merger.norm.weight".to_string(),
        "model.visual.merger.norm.bias".to_string(),
    ];
    for i in 0..vision_cfg.depth {
        for suffix in [
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "mlp.linear_fc1.weight",
            "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight",
            "mlp.linear_fc2.bias",
            "norm1.weight",
            "norm1.bias",
            "norm2.weight",
            "norm2.bias",
        ] {
            names.push(format!("model.visual.blocks.{i}.{suffix}"));
        }
    }
    names
}

/// The shape `assemble` will require for a given `model.visual.*` tensor name, mirroring
/// its per-tensor `take(name, expected_shape)` calls. `None` means the name is not one of
/// the expected tensors (unreachable given `tensor_names()`, but the caller falls back to
/// the post-materialization check in that case rather than assuming a shape).
fn expected_visual_tensor_shape(name: &str, vision_cfg: &VisionModelConfig) -> Option<Vec<usize>> {
    let hidden = vision_cfg.hidden_size;
    let qkv_out = 3 * hidden;
    let mlp_intermediate = vision_cfg.intermediate_size.unwrap_or(4 * hidden);
    let merge_in = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size * hidden;
    let out_hidden = vision_cfg.out_hidden_size;

    if let Some(rest) = name.strip_prefix("model.visual.blocks.") {
        let suffix = rest.split_once('.').map(|(_, s)| s).unwrap_or(rest);
        return match suffix {
            "attn.qkv.weight" => Some(vec![qkv_out, hidden]),
            "attn.qkv.bias" => Some(vec![qkv_out]),
            "attn.proj.weight" => Some(vec![hidden, hidden]),
            "attn.proj.bias" => Some(vec![hidden]),
            "mlp.linear_fc1.weight" => Some(vec![mlp_intermediate, hidden]),
            "mlp.linear_fc1.bias" => Some(vec![mlp_intermediate]),
            "mlp.linear_fc2.weight" => Some(vec![hidden, mlp_intermediate]),
            "mlp.linear_fc2.bias" => Some(vec![hidden]),
            "norm1.weight" | "norm2.weight" | "norm1.bias" | "norm2.bias" => Some(vec![hidden]),
            _ => None,
        };
    }

    match name {
        "model.visual.patch_embed.proj.weight" => Some(vec![
            hidden,
            vision_cfg.in_channels,
            vision_cfg.temporal_patch_size,
            vision_cfg.patch_size,
            vision_cfg.patch_size,
        ]),
        "model.visual.patch_embed.proj.bias" => Some(vec![hidden]),
        "model.visual.pos_embed.weight" => Some(vec![vision_cfg.num_position_embeddings, hidden]),
        "model.visual.merger.linear_fc1.weight" => Some(vec![merge_in, merge_in]),
        "model.visual.merger.linear_fc1.bias" => Some(vec![merge_in]),
        "model.visual.merger.linear_fc2.weight" => Some(vec![out_hidden, merge_in]),
        "model.visual.merger.linear_fc2.bias" => Some(vec![out_hidden]),
        "model.visual.merger.norm.weight" | "model.visual.merger.norm.bias" => Some(vec![hidden]),
        _ => None,
    }
}

fn load_from_fp16_dir_with_cancel(
    model_dir: &Path,
    vision_cfg: &VisionModelConfig,
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Option<Qwen35VisionWeights>, InferenceError> {
    if should_cancel() {
        return Ok(None);
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    let mut reader = ShardedSafetensors::open_index(&index_path)?;
    if should_cancel() {
        return Ok(None);
    }

    let expected_names = tensor_names(vision_cfg);
    // Inventory-exactness: this path only ever fetches the names it asks for, so an
    // insufficient `depth` (or other geometry field) would silently ignore the rest of
    // the checkpoint's `model.visual.*` tensors rather than erroring. Compare the count
    // this vision_cfg expects against how many `model.visual.*` entries the index
    // actually has.
    let actual_visual_count = reader
        .index()
        .weight_map
        .keys()
        .filter(|name| name.starts_with("model.visual."))
        .count();
    if actual_visual_count != expected_names.len() {
        return Err(InferenceError::Inference(format!(
            "vision checkpoint inventory mismatch in {}: found {actual_visual_count} \
             model.visual.* tensor(s) but vision_config (depth={}) expects exactly {}",
            index_path.display(),
            vision_cfg.depth,
            expected_names.len(),
        )));
    }

    let Some(tensors) =
        fetch_expected_tensors_with_cancel(&mut reader, expected_names, vision_cfg, should_cancel)?
    else {
        return Ok(None);
    };
    if should_cancel() {
        return Ok(None);
    }
    assemble(tensors, vision_cfg).map(Some)
}

fn load_from_single_file_with_cancel(
    model_dir: &Path,
    vision_cfg: &VisionModelConfig,
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Option<Qwen35VisionWeights>, InferenceError> {
    if should_cancel() {
        return Ok(None);
    }
    let checkpoint_path = resolve_qwen35_single_decoder_safetensors(model_dir)?;
    let mut reader = SafetensorsFile::open(&checkpoint_path)?;
    load_from_safetensors_reader_with_cancel(
        &mut reader,
        &checkpoint_path,
        vision_cfg,
        should_cancel,
    )
}

fn load_from_safetensors_reader_with_cancel(
    reader: &mut SafetensorsFile,
    checkpoint_path: &Path,
    vision_cfg: &VisionModelConfig,
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Option<Qwen35VisionWeights>, InferenceError> {
    validate_single_file_inventory(reader, checkpoint_path, vision_cfg)?;
    if should_cancel() {
        return Ok(None);
    }
    let expected_names = tensor_names(vision_cfg);
    let Some(tensors) =
        fetch_expected_tensors_with_cancel(reader, expected_names, vision_cfg, should_cancel)?
    else {
        return Ok(None);
    };
    if should_cancel() {
        return Ok(None);
    }
    assemble(tensors, vision_cfg).map(Some)
}

/// Fetch every name in `names` from `source`, checking each one's header-declared shape
/// against `expected_visual_tensor_shape` before materializing it -- mirroring the
/// Qwen3.5 text-decoder loader's `load_owned_tensor_checked`. A shape mismatch is
/// rejected here instead of after a full owned read and allocation. Generic over
/// `TensorSource` (rather than inlined into `load_from_fp16_dir`) so tests can exercise
/// the preflight-before-materialize ordering with a mock source.
#[cfg(test)]
fn fetch_expected_tensors<T: TensorSource + ?Sized>(
    source: &mut T,
    names: Vec<String>,
    vision_cfg: &VisionModelConfig,
) -> Result<NamedTensors, InferenceError> {
    let mut never_cancel = || false;
    match fetch_expected_tensors_with_cancel(source, names, vision_cfg, &mut never_cancel)? {
        Some(tensors) => Ok(tensors),
        None => Err(InferenceError::Inference(
            "non-cancellable vision tensor fetch was cancelled".to_string(),
        )),
    }
}

fn fetch_expected_tensors_with_cancel<T: TensorSource + ?Sized>(
    source: &mut T,
    names: Vec<String>,
    vision_cfg: &VisionModelConfig,
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Option<NamedTensors>, InferenceError> {
    if should_cancel() {
        return Ok(None);
    }
    let mut tensors = HashMap::with_capacity(names.len());
    for name in names {
        if should_cancel() {
            return Ok(None);
        }
        if let Some(expected) = expected_visual_tensor_shape(&name, vision_cfg)
            && let Some(declared) = source.tensor_shape(&name)?
            && declared != expected
        {
            return Err(InferenceError::ShapeMismatch {
                name,
                expected,
                actual: declared,
            });
        }
        // `get_f32_tensor` (reached via `get_f32_tensor_owned`) fully decodes and copies
        // the tensor -- including FP16->F32 expansion -- before `assemble`'s `take` ever
        // sees it. Not every expected name has a config-derived shape to compare above,
        // so budget the *declared* header shape (a cheap lookup, no decode) against
        // `MAX_VISION_TENSOR_BYTES` as well, mirroring the budget-before-materialize
        // pattern used on the q4 side, so an oversized declared tensor is rejected before
        // its expansion rather than after.
        if let Some(declared_shape) = source.tensor_shape(&name)? {
            let declared_elems: u128 = declared_shape.iter().map(|&d| d as u128).product();
            let declared_bytes = declared_elems * 4;
            if declared_bytes > crate::model::qwen35_config::MAX_VISION_TENSOR_BYTES {
                return Err(InferenceError::Inference(format!(
                    "vision checkpoint tensor {name}: declared size ({declared_bytes} bytes) \
                     exceeds MAX_VISION_TENSOR_BYTES ({}) -- rejected before decoding",
                    crate::model::qwen35_config::MAX_VISION_TENSOR_BYTES
                )));
            }
        }
        let (data, shape) = source.get_f32_tensor_owned(&name)?;
        tensors.insert(name, (data, shape));
    }
    if should_cancel() {
        return Ok(None);
    }
    Ok(Some(tensors))
}

fn load_from_q4_dir_with_cancel(
    model_dir: &Path,
    vision_cfg: &VisionModelConfig,
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Option<Qwen35VisionWeights>, InferenceError> {
    if should_cancel() {
        return Ok(None);
    }
    let manifest = q4_manifest::load_manifest(model_dir)
        .map_err(|e| {
            InferenceError::InvalidSafetensors(format!(
                "failed to read quantize_index.json in {}: {e}",
                model_dir.display()
            ))
        })?
        .ok_or_else(|| {
            InferenceError::ModelNotFound(format!(
                "quantize_index.json missing in {}",
                model_dir.display()
            ))
        })?;
    if should_cancel() {
        return Ok(None);
    }

    // FIX 4: `expected_names` is the exact, bounded (<= 12 * MAX_VISION_DEPTH + 9) set of
    // `model.visual.*` tensors `vision_cfg` implies. Previously every visual-prefixed
    // manifest entry was dequantized/decoded up front, and only *afterward* did
    // `assemble`'s leftover check reject names it didn't expect -- so a manifest with
    // many extra `model.visual.*` entries drove unbounded aggregate dequant memory
    // despite each individual tensor's own `MAX_VISION_TENSOR_BYTES` cap. Reject an
    // unexpected name before touching its file at all, and track a running aggregate
    // budget across the (now-bounded) set of tensors this loop can ever process.
    let expected_names: std::collections::HashSet<String> =
        tensor_names(vision_cfg).into_iter().collect();
    let aggregate_budget_bytes: u128 =
        expected_names.len() as u128 * crate::model::qwen35_config::MAX_VISION_TENSOR_BYTES;
    let mut aggregate_bytes: u128 = 0;

    let mut tensors = HashMap::new();
    for entry in manifest
        .tensors
        .iter()
        .filter(|e| e.name.starts_with("model.visual."))
    {
        if should_cancel() {
            return Ok(None);
        }
        if !expected_names.contains(&entry.name) {
            return Err(InferenceError::Inference(format!(
                "vision checkpoint manifest in {}: unexpected tensor {} not accounted for \
                 by vision_config (depth={}) -- rejected before dequantizing/decoding",
                model_dir.display(),
                entry.name,
                vision_cfg.depth,
            )));
        }
        // `HashMap::insert` below would silently let a later duplicate-named entry
        // overwrite an earlier one, making `tensors.is_empty()` (the inventory-exactness
        // check in `assemble`) blind to a manifest that names the same tensor twice.
        // Reject the duplicate here, before reading its file, so a corrupted manifest
        // fails deterministically instead of depending on entry order.
        if tensors.contains_key(&entry.name) {
            return Err(InferenceError::Inference(format!(
                "vision checkpoint manifest in {}: duplicate entry for tensor {} -- \
                 each model.visual.* tensor name must appear exactly once",
                model_dir.display(),
                entry.name,
            )));
        }
        // Config-shape preflight, mirroring what `fetch_expected_tensors` does for the
        // fp16 path. The manifest's declared shape is compared against what `vision_cfg`
        // implies BEFORE the tensor's file is opened, so a checkpoint that disagrees with
        // the config is rejected without reading or allocating it. This sits ahead of the
        // branch below because BOTH arms materialize: the q4 arm through
        // `dequantize_q4_to_f32` and the f16 arm inside the `.f16` loader.
        if let Some(expected) = expected_visual_tensor_shape(&entry.name, vision_cfg)
            && let Some(declared) = &entry.shape
            && declared != &expected
        {
            return Err(InferenceError::ShapeMismatch {
                name: entry.name.clone(),
                expected,
                actual: declared.clone(),
            });
        }
        // PATH CONTAINMENT -- `entry.file` comes from `quantize_index.json`, part of the
        // untrusted checkpoint directory, so it is validated before the join. The file is
        // opened exactly once and read from that fd rather than reopened by path.
        let (file, real_path) = open_manifest_entry_once(model_dir, &entry.file)?;
        let (data, shape) = if entry.quantized.unwrap_or(false) {
            let q4 = load_q4_from_open_file(file, &real_path, None).map_err(|e| {
                InferenceError::InvalidSafetensors(format!(
                    "failed to load q4 tensor {} from {}: {e}",
                    entry.name,
                    real_path.display()
                ))
            })?;
            // The manifest's own recorded shape (when present) must agree with the
            // shape carried in the tensor's `.q4` header; a mismatch means the
            // manifest and the on-disk tensor have drifted apart.
            if let Some(manifest_shape) = &entry.shape
                && manifest_shape != &q4.shape
            {
                return Err(InferenceError::ShapeMismatch {
                    name: entry.name.clone(),
                    expected: q4.shape.clone(),
                    actual: manifest_shape.clone(),
                });
            }
            // `entry.shape` is optional, so when the manifest omits it the `.q4` header
            // is the only declared shape that exists and the preflight above had nothing
            // to check. Compare the header against the config here, while the tensor is
            // still its compressed self and before `dequantize_q4_to_f32` allocates the
            // full f32 buffer.
            if let Some(expected) = expected_visual_tensor_shape(&entry.name, vision_cfg)
                && q4.shape != expected
            {
                return Err(InferenceError::ShapeMismatch {
                    name: entry.name.clone(),
                    expected,
                    actual: q4.shape.clone(),
                });
            }
            // ORDERING: `dequantize_q4_to_f32` expands the packed q4
            // buffer into a full f32 `Vec` sized by `q4.shape`'s product -- a value read
            // from the on-disk tensor header, independent of `vision_cfg` and not yet
            // checked against any expected shape at this point. Budget that product
            // BEFORE dequantizing (materializing) rather than after, so a hostile
            // declared shape is rejected before the allocation, not once `assemble`'s
            // per-tensor shape check runs on the already-materialized buffer.
            let q4_elems: u128 = q4.shape.iter().map(|&d| d as u128).product();
            let q4_bytes = q4_elems * 4;
            if q4_bytes > crate::model::qwen35_config::MAX_VISION_TENSOR_BYTES {
                return Err(InferenceError::Inference(format!(
                    "vision checkpoint tensor {} in {}: dequantized size ({q4_bytes} bytes) \
                     exceeds MAX_VISION_TENSOR_BYTES ({}) -- rejected before dequantizing",
                    entry.name,
                    real_path.display(),
                    crate::model::qwen35_config::MAX_VISION_TENSOR_BYTES
                )));
            }
            aggregate_bytes += q4_bytes;
            if aggregate_bytes > aggregate_budget_bytes {
                return Err(InferenceError::Inference(format!(
                    "vision checkpoint manifest in {}: aggregate dequantized size \
                     ({aggregate_bytes} bytes) exceeds the aggregate budget \
                     ({aggregate_budget_bytes} bytes) for {} expected tensor(s)",
                    model_dir.display(),
                    expected_names.len(),
                )));
            }
            let shape = q4.shape.clone();
            (dequantize_q4_to_f32(&q4), shape)
        } else if let Some(expected) = expected_visual_tensor_shape(&entry.name, vision_cfg) {
            // Same reasoning as the `.q4` header check above, for the arm that reads an
            // `.f16` companion: when the manifest omits `shape`, the preflight before the
            // open had nothing to compare, and this arm would otherwise materialize the
            // whole tensor before `assemble` noticed the disagreement.
            //
            // The check and the payload read happen inside the ONE handle opened and
            // identity-verified above. Checking a shape through one open and
            // materializing through a second leaves nothing binding the validated header
            // to the bytes actually read, since the pathname can be replaced in between,
            // and the checkpoint directory is untrusted input.
            let display_path = real_path.display().to_string();
            load_f16_tensor_from_open_file_expecting(file, &display_path, &expected).map_err(
                |e| match e {
                    F16LoadError::ShapeMismatch { declared } => InferenceError::ShapeMismatch {
                        name: entry.name.clone(),
                        expected,
                        actual: declared,
                    },
                    F16LoadError::Other(e) => InferenceError::InvalidSafetensors(format!(
                        "failed to load f16 tensor {} from {display_path}: {e}",
                        entry.name,
                    )),
                },
            )?
        } else {
            load_f16_tensor_from_open_file(file, &real_path.display().to_string(), None).map_err(
                |e| {
                    InferenceError::InvalidSafetensors(format!(
                        "failed to load f16 tensor {} from {}: {e}",
                        entry.name,
                        real_path.display()
                    ))
                },
            )?
        };
        tensors.insert(entry.name.clone(), (data, shape));
    }
    if should_cancel() {
        return Ok(None);
    }
    assemble(tensors, vision_cfg).map(Some)
}

/// Pull the 153 expected tensors out of a name→(data, shape) map, validating each one's
/// full shape (not just its element count) against the shape derived from `vision_cfg`,
/// and assemble them into [`Qwen35VisionWeights`]. Shared by both the fp16 and q4 load
/// paths so shape validation only lives in one place.
fn assemble(
    mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
    vision_cfg: &VisionModelConfig,
) -> Result<Qwen35VisionWeights, InferenceError> {
    let hidden = vision_cfg.hidden_size;
    let qkv_out = 3 * hidden;
    // FIX 16: use the checkpoint's own `intermediate_size` when `config.json` carries one
    // (official Qwen3.5-VL: hidden_size=1152, intermediate_size=4304, NOT 4*1152=4608);
    // fall back to `4 * hidden_size` only when the field is absent, matching HF default
    // semantics for architectures that omit an explicit vision MLP width.
    let mlp_intermediate = vision_cfg.intermediate_size.unwrap_or(4 * hidden);
    let merge_in = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size * hidden;
    let out_hidden = vision_cfg.out_hidden_size;

    let mut take = |name: String, expected_shape: Vec<usize>| -> Result<Vec<f32>, InferenceError> {
        let (v, actual_shape) = tensors
            .remove(&name)
            .ok_or_else(|| InferenceError::MissingTensor(name.clone()))?;
        // A same-numel transposition (e.g. FC2 stored as [in, out] instead of
        // [out, in]) has an identical element count but a different shape vector —
        // comparing the full shape (not just `v.len()`) is what rejects it.
        if actual_shape != expected_shape {
            return Err(InferenceError::ShapeMismatch {
                name,
                expected: expected_shape,
                actual: actual_shape,
            });
        }
        Ok(v)
    };

    let patch_embed_weight_shape = vec![
        hidden,
        vision_cfg.in_channels,
        vision_cfg.temporal_patch_size,
        vision_cfg.patch_size,
        vision_cfg.patch_size,
    ];
    let patch_embed_weight = take(
        "model.visual.patch_embed.proj.weight".to_string(),
        patch_embed_weight_shape.clone(),
    )?;
    let patch_embed_bias = take(
        "model.visual.patch_embed.proj.bias".to_string(),
        vec![hidden],
    )?;
    let pos_embed = take(
        "model.visual.pos_embed.weight".to_string(),
        vec![vision_cfg.num_position_embeddings, hidden],
    )?;

    let mut blocks = Vec::with_capacity(vision_cfg.depth);
    for i in 0..vision_cfg.depth {
        let name = |suffix: &str| format!("model.visual.blocks.{i}.{suffix}");
        blocks.push(VisualBlockWeights {
            qkv_weight: take(name("attn.qkv.weight"), vec![qkv_out, hidden])?,
            qkv_bias: take(name("attn.qkv.bias"), vec![qkv_out])?,
            proj_weight: take(name("attn.proj.weight"), vec![hidden, hidden])?,
            proj_bias: take(name("attn.proj.bias"), vec![hidden])?,
            fc1_weight: take(
                name("mlp.linear_fc1.weight"),
                vec![mlp_intermediate, hidden],
            )?,
            fc1_bias: take(name("mlp.linear_fc1.bias"), vec![mlp_intermediate])?,
            fc2_weight: take(
                name("mlp.linear_fc2.weight"),
                vec![hidden, mlp_intermediate],
            )?,
            fc2_bias: take(name("mlp.linear_fc2.bias"), vec![hidden])?,
            norm1_weight: take(name("norm1.weight"), vec![hidden])?,
            norm1_bias: take(name("norm1.bias"), vec![hidden])?,
            norm2_weight: take(name("norm2.weight"), vec![hidden])?,
            norm2_bias: take(name("norm2.bias"), vec![hidden])?,
        });
    }

    let merger = VisualMergerWeights {
        fc1_weight: take(
            "model.visual.merger.linear_fc1.weight".to_string(),
            vec![merge_in, merge_in],
        )?,
        fc1_bias: take(
            "model.visual.merger.linear_fc1.bias".to_string(),
            vec![merge_in],
        )?,
        fc2_weight: take(
            "model.visual.merger.linear_fc2.weight".to_string(),
            vec![out_hidden, merge_in],
        )?,
        fc2_bias: take(
            "model.visual.merger.linear_fc2.bias".to_string(),
            vec![out_hidden],
        )?,
        norm_weight: take("model.visual.merger.norm.weight".to_string(), vec![hidden])?,
        norm_bias: take("model.visual.merger.norm.bias".to_string(), vec![hidden])?,
    };

    // Inventory-exactness: every tensor fetched must be accounted for by the expected
    // set above. The q4 path loads every `model.visual.*` entry in the manifest up
    // front (not just the names `vision_cfg` implies), so a checkpoint that carries
    // more real block tensors than `vision_cfg.depth` implies (e.g. a full 153-entry
    // checkpoint paired with `depth: 0`) would otherwise silently return a truncated
    // `Qwen35VisionWeights` instead of erroring.
    if !tensors.is_empty() {
        let mut leftover: Vec<&String> = tensors.keys().collect();
        leftover.sort();
        return Err(InferenceError::Inference(format!(
            "vision checkpoint has {} unconsumed model.visual.* tensor(s) not accounted for \
             by vision_config (depth={}): {:?}",
            tensors.len(),
            vision_cfg.depth,
            leftover.into_iter().take(5).collect::<Vec<_>>(),
        )));
    }

    Ok(Qwen35VisionWeights {
        patch_embed_weight,
        patch_embed_weight_shape,
        patch_embed_bias,
        pos_embed,
        blocks,
        merger,
    })
}

#[cfg(test)]
mod tests;
