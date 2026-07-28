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

use std::collections::HashMap;
use std::path::Path;

use crate::error::InferenceError;
use crate::model::qwen35_config::VisionModelConfig;
use crate::quant::q4_manifest;
use crate::weights::f32_weights::{ShardedSafetensors, TensorSource, open_manifest_entry_once};
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

/// Load the real `model.visual.*` tensors from a model directory, in either the fp16
/// sharded-safetensors form (`model.safetensors.index.json`) or the per-tensor q4 form
/// (`quantize_index.json`) — whichever manifest is present. Both forms are supported
/// because the on-disk q4 checkpoint (verified by inspection) already retains all 153
/// visual tensors alongside the text decoder's, using the same per-tensor `.q4`/`.f16`
/// convention; no fp16-fallback plumbing is needed for the q4 case.
///
/// # Errors
///
/// Returns [`InferenceError::ModelNotFound`] if neither manifest is present, and
/// [`InferenceError::MissingTensor`] / [`InferenceError::ShapeMismatch`] if any of the
/// expected tensors (derived from `vision_cfg`) is absent or the wrong size.
pub fn load_qwen35_vision_weights(
    model_dir: &Path,
    vision_cfg: &VisionModelConfig,
) -> Result<Qwen35VisionWeights, InferenceError> {
    // Callers can construct `VisionModelConfig` directly (it's a public struct), so this
    // boundary re-validates rather than trusting that every caller went through
    // `Qwen35Config::from_config_json_str`'s parse-time check.
    vision_cfg.validate()?;
    if model_dir.join("quantize_index.json").exists() {
        load_from_q4_dir(model_dir, vision_cfg)
    } else if model_dir.join("model.safetensors.index.json").exists() {
        load_from_fp16_dir(model_dir, vision_cfg)
    } else {
        Err(InferenceError::ModelNotFound(format!(
            "no model.safetensors.index.json or quantize_index.json in {} -- cannot load \
             model.visual.* vision tensors from this directory",
            model_dir.display()
        )))
    }
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
    let mlp_intermediate = 4 * hidden;
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

fn load_from_fp16_dir(
    model_dir: &Path,
    vision_cfg: &VisionModelConfig,
) -> Result<Qwen35VisionWeights, InferenceError> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let mut reader = ShardedSafetensors::open_index(&index_path)?;

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

    let tensors = fetch_expected_tensors(&mut reader, expected_names, vision_cfg)?;
    assemble(tensors, vision_cfg)
}

/// Fetch every name in `names` from `source`, checking each one's header-declared shape
/// against `expected_visual_tensor_shape` before materializing it -- mirroring the
/// Qwen3.5 text-decoder loader's `load_owned_tensor_checked`. A shape mismatch is
/// rejected here instead of after a full owned read and allocation. Generic over
/// `TensorSource` (rather than inlined into `load_from_fp16_dir`) so tests can exercise
/// the preflight-before-materialize ordering with a mock source.
fn fetch_expected_tensors<T: TensorSource + ?Sized>(
    source: &mut T,
    names: Vec<String>,
    vision_cfg: &VisionModelConfig,
) -> Result<NamedTensors, InferenceError> {
    let mut tensors = HashMap::with_capacity(names.len());
    for name in names {
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
    Ok(tensors)
}

fn load_from_q4_dir(
    model_dir: &Path,
    vision_cfg: &VisionModelConfig,
) -> Result<Qwen35VisionWeights, InferenceError> {
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
    assemble(tensors, vision_cfg)
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
mod tests {
    use super::*;
    use crate::model::qwen35_config::Qwen35Config;

    fn real_vision_cfg() -> VisionModelConfig {
        VisionModelConfig {
            depth: 12,
            hidden_size: 768,
            num_heads: 12,
            patch_size: 16,
            spatial_merge_size: 2,
            out_hidden_size: 1024,
            temporal_patch_size: 2,
            num_position_embeddings: 2304,
            in_channels: 3,
            deepstack_visual_indexes: vec![],
            intermediate_size: None,
        }
    }

    #[test]
    fn missing_manifest_is_a_descriptive_error_not_a_panic() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = real_vision_cfg();
        let err = load_qwen35_vision_weights(tmp.path(), &cfg)
            .expect_err("a directory with neither manifest must be a hard error");
        let msg = err.to_string();
        assert!(
            msg.contains("model.safetensors.index.json") || msg.contains("quantize_index.json"),
            "error must name the missing manifests: {msg}"
        );
    }

    #[test]
    fn tensor_names_count_matches_153_for_depth_12() {
        let cfg = real_vision_cfg();
        // 2 (patch_embed) + 1 (pos_embed) + 12 * 12 (blocks) + 6 (merger) = 153.
        assert_eq!(tensor_names(&cfg).len(), 153);
    }

    #[test]
    fn depth_zero_vision_config_rejected_at_loader_boundary() {
        // `load_qwen35_vision_weights` is public, so a caller can hand it a directly
        // constructed `VisionModelConfig` that never went through
        // `Qwen35Config::from_config_json_str`'s parse-time validation. depth: 0 must
        // still fail closed here, before the (nonexistent) directory is even touched.
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = real_vision_cfg();
        cfg.depth = 0;
        let err = load_qwen35_vision_weights(tmp.path(), &cfg)
            .expect_err("depth: 0 must be rejected at the public loader boundary");
        assert!(
            err.to_string().contains("depth"),
            "error must name depth: {err}"
        );
    }

    #[test]
    fn num_heads_zero_vision_config_rejected_at_loader_boundary() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = real_vision_cfg();
        cfg.num_heads = 0;
        let err = load_qwen35_vision_weights(tmp.path(), &cfg)
            .expect_err("num_heads: 0 must be rejected at the public loader boundary");
        assert!(
            err.to_string().contains("num_heads"),
            "error must name num_heads: {err}"
        );
    }

    #[test]
    fn q4_full_inventory_with_depth_zero_is_rejected() {
        // A checkpoint that genuinely carries the full 153-tensor real inventory, paired
        // with a (malformed) vision_config claiming depth: 0, must error rather than
        // silently returning a nine-tensor `Qwen35VisionWeights`.
        let tmp = tempfile::tempdir().unwrap();
        let full_cfg = real_vision_cfg();
        let entries: Vec<String> = tensor_names(&full_cfg)
            .into_iter()
            .map(|name| format!(r#"{{"name":"{name}","file":"missing.f16","quantized":false}}"#))
            .collect();
        std::fs::write(
            tmp.path().join("quantize_index.json"),
            format!("[{}]", entries.join(",")),
        )
        .expect("test setup: write manifest");

        let mut depth_zero_cfg = full_cfg;
        depth_zero_cfg.depth = 0;
        let err = load_qwen35_vision_weights(tmp.path(), &depth_zero_cfg)
            .expect_err("depth: 0 with a full 153-entry inventory present must still be rejected");
        assert!(
            err.to_string().contains("depth"),
            "error must name depth: {err}"
        );
    }

    fn tiny_vision_cfg() -> VisionModelConfig {
        VisionModelConfig {
            depth: 1,
            hidden_size: 4,
            num_heads: 2,
            patch_size: 2,
            spatial_merge_size: 1,
            out_hidden_size: 4,
            temporal_patch_size: 1,
            num_position_embeddings: 4,
            in_channels: 1,
            deepstack_visual_indexes: vec![],
            intermediate_size: None,
        }
    }

    /// The 21 expected (name, shape) pairs for [`tiny_vision_cfg`], computed
    /// independently of `assemble`'s derivation so a fixture bug can't cancel out a
    /// production bug.
    fn tiny_expected_shapes() -> Vec<(String, Vec<usize>)> {
        let hidden = 4;
        let qkv_out = 3 * hidden;
        let mlp_intermediate = 4 * hidden;
        let merge_in = hidden; // spatial_merge_size^2 (1) * hidden
        let out_hidden = 4;
        let mut v = vec![
            (
                "model.visual.patch_embed.proj.weight".to_string(),
                vec![hidden, 1, 1, 2, 2],
            ),
            (
                "model.visual.patch_embed.proj.bias".to_string(),
                vec![hidden],
            ),
            ("model.visual.pos_embed.weight".to_string(), vec![4, hidden]),
            (
                "model.visual.merger.linear_fc1.weight".to_string(),
                vec![merge_in, merge_in],
            ),
            (
                "model.visual.merger.linear_fc1.bias".to_string(),
                vec![merge_in],
            ),
            (
                "model.visual.merger.linear_fc2.weight".to_string(),
                vec![out_hidden, merge_in],
            ),
            (
                "model.visual.merger.linear_fc2.bias".to_string(),
                vec![out_hidden],
            ),
            ("model.visual.merger.norm.weight".to_string(), vec![hidden]),
            ("model.visual.merger.norm.bias".to_string(), vec![hidden]),
        ];
        for (suffix, shape) in [
            ("attn.qkv.weight", vec![qkv_out, hidden]),
            ("attn.qkv.bias", vec![qkv_out]),
            ("attn.proj.weight", vec![hidden, hidden]),
            ("attn.proj.bias", vec![hidden]),
            ("mlp.linear_fc1.weight", vec![mlp_intermediate, hidden]),
            ("mlp.linear_fc1.bias", vec![mlp_intermediate]),
            ("mlp.linear_fc2.weight", vec![hidden, mlp_intermediate]),
            ("mlp.linear_fc2.bias", vec![hidden]),
            ("norm1.weight", vec![hidden]),
            ("norm1.bias", vec![hidden]),
            ("norm2.weight", vec![hidden]),
            ("norm2.bias", vec![hidden]),
        ] {
            v.push((format!("model.visual.blocks.0.{suffix}"), shape));
        }
        v
    }

    /// A [`TensorSource`] that records every name passed to `get_f32_tensor_owned`, so a
    /// test can assert a specific tensor's shape mismatch is caught from the
    /// header-declared shape without that tensor's data ever being copied (other,
    /// correctly-shaped tensors ahead of it in iteration order are still fetched
    /// normally).
    struct CountingSource {
        tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
        materialized: std::cell::RefCell<std::collections::HashSet<String>>,
    }

    impl TensorSource for CountingSource {
        fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError> {
            Ok(self.tensors.contains_key(name))
        }
        fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
            Ok(self.tensors.get(name).map(|(_, s)| s.clone()))
        }
        fn get_f32_tensor_owned(
            &mut self,
            name: &str,
        ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
            self.materialized.borrow_mut().insert(name.to_string());
            self.tensors
                .get(name)
                .map(|(d, s)| (d.clone(), s.clone()))
                .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
        }
    }

    /// An undersized (but declared-shape-visible) block tensor must be rejected from
    /// its header-declared shape, before `get_f32_tensor_owned` ever copies its data.
    /// `assemble`'s own post-materialization check would also catch this shape
    /// mismatch, so `expect_err` alone would not be mutation-sensitive to the
    /// preflight; the "mutated name was never materialized" assertion is what pins the
    /// preflight-before-materialize ordering specifically.
    #[test]
    fn fetch_expected_tensors_rejects_undersized_tensor_before_materialization() {
        let cfg = tiny_vision_cfg();
        let names = tensor_names(&cfg);
        let mutated_name = "model.visual.blocks.0.mlp.linear_fc1.weight".to_string();
        let hidden = cfg.hidden_size;

        let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = tiny_expected_shapes()
            .into_iter()
            .map(|(name, shape)| {
                let numel: usize = shape.iter().product();
                (name, (vec![0.5f32; numel], shape))
            })
            .collect();
        // Declared shape undersized relative to [4*hidden, hidden], data consistent
        // with the (wrong) declared shape.
        tensors.insert(
            mutated_name.clone(),
            (
                vec![0.5f32; (4 * hidden - 1) * hidden],
                vec![4 * hidden - 1, hidden],
            ),
        );
        let mut source = CountingSource {
            tensors,
            materialized: std::cell::RefCell::new(std::collections::HashSet::new()),
        };

        let err = fetch_expected_tensors(&mut source, names, &cfg)
            .expect_err("undersized fc1 weight must be rejected");
        match err {
            InferenceError::ShapeMismatch { name, .. } => {
                assert_eq!(name, mutated_name, "error must name the mutated tensor");
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
        assert!(
            !source.materialized.borrow().contains(&mutated_name),
            "the mismatched tensor's data must never be copied"
        );
    }

    /// Corrupt the FC2 weight entry in `shapes` to a same-numel transposition
    /// (`[hidden, mlp_intermediate]` -> `[mlp_intermediate, hidden]`; both have 64
    /// elements for [`tiny_vision_cfg`]).
    fn transpose_fc2_weight(shapes: &mut [(String, Vec<usize>)]) {
        for (name, shape) in shapes.iter_mut() {
            if name == "model.visual.blocks.0.mlp.linear_fc2.weight" {
                assert_eq!(
                    *shape,
                    vec![4, 16],
                    "fixture assumption for tiny_vision_cfg"
                );
                *shape = vec![16, 4];
                return;
            }
        }
        panic!("fc2 weight entry not found in fixture");
    }

    fn write_multi_f32_tensor_shard(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let mut header_parts = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
            let end = data.len();
            let shape_str = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            header_parts.push(format!(
                r#""{name}":{{"dtype":"F32","shape":[{shape_str}],"data_offsets":[{start},{end}]}}"#
            ));
        }
        let header = format!("{{{}}}", header_parts.join(","));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        std::fs::write(path, &bytes).expect("test setup: write shard");
    }

    fn assert_fc2_shape_mismatch(result: Result<Qwen35VisionWeights, InferenceError>) {
        let err = result.expect_err("transposed FC2 (same numel, wrong shape) must be rejected");
        match err {
            InferenceError::ShapeMismatch { name, .. } => {
                assert!(
                    name.contains("fc2"),
                    "expected FC2 shape mismatch, got {name}"
                )
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn fp16_same_numel_transposed_fc2_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tiny_vision_cfg();
        let mut shapes = tiny_expected_shapes();
        transpose_fc2_weight(&mut shapes);

        let shard = tmp.path().join("model-00001-of-00001.safetensors");
        let tensors: Vec<(String, Vec<usize>, Vec<f32>)> = shapes
            .iter()
            .map(|(name, shape)| {
                let numel: usize = shape.iter().product();
                (name.clone(), shape.clone(), vec![0.5f32; numel])
            })
            .collect();
        write_multi_f32_tensor_shard(&shard, &tensors);

        let weight_map = shapes
            .iter()
            .map(|(name, _)| format!(r#""{name}":"model-00001-of-00001.safetensors""#))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            format!(r#"{{"weight_map":{{{weight_map}}}}}"#),
        )
        .expect("test setup: write index");

        assert_fc2_shape_mismatch(load_qwen35_vision_weights(tmp.path(), &cfg));
    }

    #[test]
    fn q4_same_numel_transposed_fc2_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tiny_vision_cfg();
        let mut shapes = tiny_expected_shapes();
        transpose_fc2_weight(&mut shapes);

        let mut manifest_entries = Vec::new();
        for (i, (name, shape)) in shapes.iter().enumerate() {
            let numel: usize = shape.iter().product();
            let data: Vec<f64> = vec![0.25_f64; numel];
            let q4 = crate::weights::q4_weights::quantize_f64_to_q4(&data, shape)
                .expect("quantize succeeds");
            let file_name = format!("t{i}.q4");
            crate::weights::q4_weights::save_q4_file(&tmp.path().join(&file_name), &q4)
                .expect("test setup: write q4 file");
            manifest_entries.push(format!(
                r#"{{"name":"{name}","file":"{file_name}","quantized":true}}"#
            ));
        }
        std::fs::write(
            tmp.path().join("quantize_index.json"),
            format!("[{}]", manifest_entries.join(",")),
        )
        .expect("test setup: write manifest");

        assert_fc2_shape_mismatch(load_qwen35_vision_weights(tmp.path(), &cfg));
    }

    /// #1069: quantize_index.json file entries are untrusted checkpoint
    /// content; an entry escaping the model directory must be rejected
    /// before its file is read.
    #[test]
    fn q4_manifest_entry_escaping_model_dir_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        std::fs::create_dir_all(&model_dir).expect("test setup");
        // A structurally valid q4 file OUTSIDE the model dir — containment,
        // not file validity, must be what rejects the entry.
        let data = vec![0.25_f64; 4];
        let q4 = crate::weights::q4_weights::quantize_f64_to_q4(&data, &[2, 2])
            .expect("quantize succeeds");
        crate::weights::q4_weights::save_q4_file(&tmp.path().join("evil.q4"), &q4)
            .expect("test setup: write q4 file");
        std::fs::write(
            model_dir.join("quantize_index.json"),
            r#"[{"name":"model.visual.patch_embed.proj.weight","file":"../evil.q4","quantized":true}]"#,
        )
        .expect("test setup: write manifest");

        let err = load_qwen35_vision_weights(&model_dir, &tiny_vision_cfg())
            .expect_err("escaping manifest entry must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("must stay within the model directory")
                || msg.contains("escapes model root"),
            "unexpected error: {err}"
        );
    }

    /// The manifest's declared shape must be checked against `vision_cfg` BEFORE the
    /// tensor's file is opened, matching what `fetch_expected_tensors` does on the fp16
    /// path.
    ///
    /// Pointing the entry at a file that does not exist is what makes this
    /// mutation-sensitive. With the preflight the loader never gets that far and returns
    /// ShapeMismatch; without it the loader tries to read `missing.q4` and reports a read
    /// failure instead. Asserting merely that *some* error came back would pass either
    /// way and guard nothing.
    #[test]
    fn q4_manifest_shape_disagreeing_with_config_is_rejected_before_the_file_is_read() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("quantize_index.json"),
            r#"[{"name":"model.visual.patch_embed.proj.bias","file":"missing.q4","quantized":true,"shape":[9999]}]"#,
        )
        .expect("test setup: write manifest");
        assert!(
            !tmp.path().join("missing.q4").exists(),
            "test setup: the tensor file must NOT exist, that absence is the assertion"
        );

        let err = load_qwen35_vision_weights(tmp.path(), &tiny_vision_cfg())
            .expect_err("a manifest shape contradicting vision_cfg must be rejected");
        match err {
            InferenceError::ShapeMismatch {
                name,
                expected,
                actual,
            } => {
                assert_eq!(name, "model.visual.patch_embed.proj.bias");
                assert_eq!(expected, vec![4]);
                assert_eq!(actual, vec![9999]);
            }
            other => panic!("expected ShapeMismatch before any file read, got: {other}"),
        }
    }

    /// When the manifest omits `shape`, the `.q4` header carries the only declared shape
    /// there is and the manifest-level preflight has nothing to compare. The header must
    /// then be checked against `vision_cfg` before `dequantize_q4_to_f32` allocates the
    /// full f32 buffer.
    ///
    /// Mutation-sensitivity needs care here, and it is exactly the trap the fp16-path
    /// test author already called out: `assemble` rejects a contradictory shape too, so
    /// `expect_err` alone still passes with the fix reverted. The discriminator is WHICH
    /// error comes back. This manifest carries exactly one tensor and it is deliberately
    /// not the one `assemble` takes first, so with the header check removed the loader
    /// dequantizes happily and then dies on MissingTensor for `patch_embed.proj.weight`,
    /// a different error about a different tensor.
    #[test]
    fn q4_header_shape_disagreeing_with_config_is_rejected_before_dequantization() {
        let tmp = tempfile::tempdir().unwrap();
        let data = vec![0.25_f64; 64];
        let q4 = crate::weights::q4_weights::quantize_f64_to_q4(&data, &[64])
            .expect("quantize succeeds");
        crate::weights::q4_weights::save_q4_file(&tmp.path().join("t0.q4"), &q4)
            .expect("test setup: write q4 file");
        std::fs::write(
            tmp.path().join("quantize_index.json"),
            r#"[{"name":"model.visual.patch_embed.proj.bias","file":"t0.q4","quantized":true}]"#,
        )
        .expect("test setup: write manifest");

        let err = load_qwen35_vision_weights(tmp.path(), &tiny_vision_cfg())
            .expect_err("a q4 header shape contradicting vision_cfg must be rejected");
        match err {
            InferenceError::ShapeMismatch {
                name,
                expected,
                actual,
            } => {
                assert_eq!(name, "model.visual.patch_embed.proj.bias");
                assert_eq!(expected, vec![4]);
                assert_eq!(actual, vec![64]);
            }
            other => panic!("expected ShapeMismatch before dequantization, got: {other}"),
        }
    }

    fn write_khf1_f16_file(path: &Path, shape: &[usize], values: &[f32]) {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHF1");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in shape {
            buf.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            buf.extend_from_slice(&crate::weights::half_bits::f32_to_f16_bits(*v).to_le_bytes());
        }
        std::fs::write(path, &buf).expect("test setup: write f16 file");
    }

    #[test]
    fn f16_companion_same_numel_transposed_fc2_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tiny_vision_cfg();
        let mut shapes = tiny_expected_shapes();
        transpose_fc2_weight(&mut shapes);

        let mut manifest_entries = Vec::new();
        for (i, (name, shape)) in shapes.iter().enumerate() {
            let numel: usize = shape.iter().product();
            let values = vec![0.5f32; numel];
            let file_name = format!("t{i}.f16");
            write_khf1_f16_file(&tmp.path().join(&file_name), shape, &values);
            manifest_entries.push(format!(
                r#"{{"name":"{name}","file":"{file_name}","quantized":false}}"#
            ));
        }
        std::fs::write(
            tmp.path().join("quantize_index.json"),
            format!("[{}]", manifest_entries.join(",")),
        )
        .expect("test setup: write manifest");

        assert_fc2_shape_mismatch(load_qwen35_vision_weights(tmp.path(), &cfg));
    }

    /// The `.f16` companion arm needs its own header check, and this is the case that
    /// proves it: a non-quantized entry whose manifest omits `shape` skips the preflight
    /// before `contained_shard_path` (there is no declared shape to compare) and does not
    /// reach the `.q4` header check (that arm is not taken). Without a header check here
    /// the tensor is fully read and converted before `assemble` notices.
    ///
    /// The single manifest entry is the bias rather than `patch_embed.proj.weight`, so
    /// without the guard the loader gets far enough for `assemble` to report the absent
    /// weight as `MissingTensor`. That makes the assertion below discriminating: it can
    /// only pass if the header was checked before materialization, not merely because
    /// loading failed somewhere.
    #[test]
    fn f16_header_shape_disagreeing_with_config_is_rejected_before_materialization() {
        let tmp = tempfile::tempdir().unwrap();
        write_khf1_f16_file(&tmp.path().join("t0.f16"), &[64], &vec![0.5f32; 64]);
        std::fs::write(
            tmp.path().join("quantize_index.json"),
            r#"[{"name":"model.visual.patch_embed.proj.bias","file":"t0.f16","quantized":false}]"#,
        )
        .expect("test setup: write manifest");

        let err = load_qwen35_vision_weights(tmp.path(), &tiny_vision_cfg())
            .expect_err("an f16 header shape contradicting vision_cfg must be rejected");
        match err {
            InferenceError::ShapeMismatch {
                name,
                expected,
                actual,
            } => {
                assert_eq!(name, "model.visual.patch_embed.proj.bias");
                assert_eq!(expected, vec![4]);
                assert_eq!(actual, vec![64]);
            }
            other => panic!("expected ShapeMismatch before materialization, got: {other}"),
        }
    }

    /// Proves the shape comparison happens BEFORE the payload is read, rather than merely
    /// before `assemble`.
    ///
    /// The fixture's header declares 64 elements but the file carries no payload bytes at
    /// all. If the loader read the payload first it would fail on the truncated read and
    /// surface `InvalidSafetensors`; only a loader that compares the header against the
    /// expected shape first can return `ShapeMismatch` for this input. The two outcomes
    /// are therefore distinguishable, which is the whole point of the fixture.
    ///
    /// What this does not prove is the same-handle property. That the header and payload
    /// come from one open cannot be shown without a filesystem seam to swap the file
    /// mid-load; it is structural, held by `load_f16_tensor_file_expecting` performing a
    /// single open, and this test would still pass if that were split back into two.
    #[test]
    fn f16_shape_is_compared_before_the_payload_is_read() {
        let tmp = tempfile::tempdir().unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"KHF1");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&64u64.to_le_bytes());
        buf.extend_from_slice(&64u64.to_le_bytes());
        // Deliberately no payload: 64 declared elements, zero bytes of data.
        std::fs::write(tmp.path().join("t0.f16"), &buf).expect("test setup: write f16 header");
        std::fs::write(
            tmp.path().join("quantize_index.json"),
            r#"[{"name":"model.visual.patch_embed.proj.bias","file":"t0.f16","quantized":false}]"#,
        )
        .expect("test setup: write manifest");

        let err = load_qwen35_vision_weights(tmp.path(), &tiny_vision_cfg())
            .expect_err("a header disagreeing with vision_cfg must be rejected");
        match err {
            InferenceError::ShapeMismatch { actual, .. } => assert_eq!(actual, vec![64]),
            other => panic!(
                "expected ShapeMismatch from the header check, which proves the payload \
                 read was never attempted; got: {other}"
            ),
        }
    }

    #[test]
    fn q4_manifest_duplicate_visual_tensor_name_is_rejected() {
        // A tiny valid q4 inventory (all 21 entries `tiny_vision_cfg` expects, each with
        // a correct shape) plus one repeated `model.visual.*` name must be rejected
        // before `assemble`'s inventory-exactness check ever runs -- otherwise
        // `HashMap::insert` would silently let the duplicate's file overwrite the
        // first's, and the loader would return `Ok` depending on manifest entry order.
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tiny_vision_cfg();
        let shapes = tiny_expected_shapes();

        let mut manifest_entries = Vec::new();
        for (i, (name, shape)) in shapes.iter().enumerate() {
            let numel: usize = shape.iter().product();
            let data: Vec<f64> = vec![0.25_f64; numel];
            let q4 = crate::weights::q4_weights::quantize_f64_to_q4(&data, shape)
                .expect("quantize succeeds");
            let file_name = format!("t{i}.q4");
            crate::weights::q4_weights::save_q4_file(&tmp.path().join(&file_name), &q4)
                .expect("test setup: write q4 file");
            manifest_entries.push(format!(
                r#"{{"name":"{name}","file":"{file_name}","quantized":true}}"#
            ));
        }
        // Duplicate: same name and shape as an existing entry, written to a distinct
        // file so a naive "does the file exist" check would not catch it.
        let (dup_name, dup_shape) = &shapes[0];
        let dup_numel: usize = dup_shape.iter().product();
        let dup_q4 =
            crate::weights::q4_weights::quantize_f64_to_q4(&vec![0.5_f64; dup_numel], dup_shape)
                .expect("quantize succeeds");
        let dup_file_name = "dup.q4";
        crate::weights::q4_weights::save_q4_file(&tmp.path().join(dup_file_name), &dup_q4)
            .expect("test setup: write duplicate q4 file");
        manifest_entries.push(format!(
            r#"{{"name":"{dup_name}","file":"{dup_file_name}","quantized":true}}"#
        ));

        std::fs::write(
            tmp.path().join("quantize_index.json"),
            format!("[{}]", manifest_entries.join(",")),
        )
        .expect("test setup: write manifest");

        let err = load_qwen35_vision_weights(tmp.path(), &cfg)
            .expect_err("a duplicate model.visual.* manifest entry must be rejected");
        match err {
            InferenceError::Inference(msg) => {
                assert!(
                    msg.contains("duplicate") && msg.contains(dup_name),
                    "expected a duplicate-entry error naming {dup_name}, got: {msg}"
                );
            }
            other => panic!("expected InferenceError::Inference, got {other:?}"),
        }
    }

    #[test]
    fn q4_manifest_unexpected_visual_tensor_rejected_before_dequantizing() {
        // FIX 4 regression: a manifest entry whose name is NOT in the set
        // `vision_cfg` expects must be rejected before its file is even opened -- not
        // decoded and only caught afterward by `assemble`'s "unconsumed leftover" check.
        // A valid tiny inventory plus one hostile extra entry proves the new pre-decode
        // name check fires; the assertion on the error text ("unexpected" + naming the
        // rejected tensor, not "unconsumed") distinguishes this from the old ordering.
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tiny_vision_cfg();
        let shapes = tiny_expected_shapes();

        let mut manifest_entries = Vec::new();
        for (i, (name, shape)) in shapes.iter().enumerate() {
            let numel: usize = shape.iter().product();
            let data: Vec<f64> = vec![0.25_f64; numel];
            let q4 = crate::weights::q4_weights::quantize_f64_to_q4(&data, shape)
                .expect("quantize succeeds");
            let file_name = format!("t{i}.q4");
            crate::weights::q4_weights::save_q4_file(&tmp.path().join(&file_name), &q4)
                .expect("test setup: write q4 file");
            manifest_entries.push(format!(
                r#"{{"name":"{name}","file":"{file_name}","quantized":true}}"#
            ));
        }
        // An extra, valid, individually in-budget q4 tensor whose name `vision_cfg`
        // (depth=1) never asks for. Deliberately does NOT contain the substring
        // "unexpected" -- the post-decode "unconsumed leftover" fallback check (see
        // `assemble`) also names the rejected tensor in its error text, so asserting on
        // "unexpected" + the raw name alone would pass via either code path and fail to
        // discriminate a pre-decode rejection from a post-decode one.
        let hostile_name = "model.visual.blocks.0.rogue_injected_tensor";
        let hostile_q4 = crate::weights::q4_weights::quantize_f64_to_q4(&[0.5_f64; 4], &[4])
            .expect("quantize succeeds");
        let hostile_file_name = "hostile.q4";
        crate::weights::q4_weights::save_q4_file(&tmp.path().join(hostile_file_name), &hostile_q4)
            .expect("test setup: write hostile q4 file");
        manifest_entries.push(format!(
            r#"{{"name":"{hostile_name}","file":"{hostile_file_name}","quantized":true}}"#
        ));

        std::fs::write(
            tmp.path().join("quantize_index.json"),
            format!("[{}]", manifest_entries.join(",")),
        )
        .expect("test setup: write manifest");

        let err = load_qwen35_vision_weights(tmp.path(), &cfg)
            .expect_err("an unexpected model.visual.* manifest entry must be rejected");
        match err {
            InferenceError::Inference(msg) => {
                assert!(
                    msg.contains("rejected before dequantizing/decoding")
                        && msg.contains(hostile_name),
                    "expected a pre-decode unexpected-tensor error naming {hostile_name}, \
                     got: {msg}"
                );
            }
            other => panic!("expected InferenceError::Inference, got {other:?}"),
        }
    }

    #[test]
    fn fp16_sharded_index_duplicate_raw_weight_map_key_is_rejected() {
        // A raw `weight_map` JSON object can name the same tensor twice, mapped to two
        // different shards. Ordinary map deserialization collapses that to one
        // `HashMap` entry (last member wins) *before* the inventory-count check ever
        // runs, so the count stays exact and the loader silently resolves the tensor
        // from whichever shard happened to be the last raw JSON member -- raw-member-
        // order-dependent, not a deterministic error. All 21 tiny names resolve to
        // shard A; one name (`dup_name`) is additionally repeated, resolving to shard B
        // with a different (but shape-correct) value, so a silent last-write-wins
        // collapse would succeed with B's value instead of failing.
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tiny_vision_cfg();
        let shapes = tiny_expected_shapes();

        let shard_a = tmp.path().join("model-00001-of-00002.safetensors");
        let tensors_a: Vec<(String, Vec<usize>, Vec<f32>)> = shapes
            .iter()
            .map(|(name, shape)| {
                let numel: usize = shape.iter().product();
                (name.clone(), shape.clone(), vec![0.5f32; numel])
            })
            .collect();
        write_multi_f32_tensor_shard(&shard_a, &tensors_a);

        let (dup_name, dup_shape) = shapes[0].clone();
        let dup_numel: usize = dup_shape.iter().product();
        let shard_b = tmp.path().join("model-00002-of-00002.safetensors");
        write_multi_f32_tensor_shard(
            &shard_b,
            &[(dup_name.clone(), dup_shape, vec![9.0f32; dup_numel])],
        );

        let mut weight_map_members: Vec<String> = shapes
            .iter()
            .map(|(name, _)| format!(r#""{name}":"model-00001-of-00002.safetensors""#))
            .collect();
        // Duplicate raw member for `dup_name`, pointing at the second shard.
        weight_map_members.push(format!(
            r#""{dup_name}":"model-00002-of-00002.safetensors""#
        ));
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            format!(r#"{{"weight_map":{{{}}}}}"#, weight_map_members.join(",")),
        )
        .expect("test setup: write index");

        let err = load_qwen35_vision_weights(tmp.path(), &cfg).expect_err(
            "a duplicate raw weight_map key routed to a second shard must be rejected, \
             not silently resolved by raw member order",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("duplicate") && msg.contains(&dup_name),
            "expected a duplicate-key error naming {dup_name}, got: {msg}"
        );
    }

    #[test]
    fn assemble_rejects_unconsumed_leftover_tensors_when_depth_understates_inventory() {
        // The q4 path loads every `model.visual.*` manifest entry up front, not just the
        // names `vision_cfg.depth` implies. If the checkpoint has more real block tensors
        // than `depth` accounts for, the leftovers must be a hard error, not silently
        // dropped. The 21 tensors `small_cfg` (depth=1) does expect are given correct
        // shapes so this test exercises the leftover check specifically, not the
        // per-tensor shape check above it.
        let full_cfg = real_vision_cfg(); // depth 12, 153 real tensors
        let mut small_cfg = full_cfg.clone();
        small_cfg.depth = 1;

        let hidden = full_cfg.hidden_size;
        let qkv_out = 3 * hidden;
        let mlp_intermediate = 4 * hidden;
        let merge_in = full_cfg.spatial_merge_size * full_cfg.spatial_merge_size * hidden;
        let out_hidden = full_cfg.out_hidden_size;
        let shape_for = |name: &str| -> Vec<usize> {
            match name {
                "model.visual.patch_embed.proj.weight" => vec![
                    hidden,
                    full_cfg.in_channels,
                    full_cfg.temporal_patch_size,
                    full_cfg.patch_size,
                    full_cfg.patch_size,
                ],
                "model.visual.pos_embed.weight" => vec![full_cfg.num_position_embeddings, hidden],
                "model.visual.merger.linear_fc1.weight" => vec![merge_in, merge_in],
                "model.visual.merger.linear_fc1.bias" => vec![merge_in],
                "model.visual.merger.linear_fc2.weight" => vec![out_hidden, merge_in],
                "model.visual.merger.linear_fc2.bias" => vec![out_hidden],
                n if n.ends_with("attn.qkv.weight") => vec![qkv_out, hidden],
                n if n.ends_with("attn.qkv.bias") => vec![qkv_out],
                n if n.ends_with("attn.proj.weight") => vec![hidden, hidden],
                n if n.ends_with("mlp.linear_fc1.weight") => vec![mlp_intermediate, hidden],
                n if n.ends_with("mlp.linear_fc1.bias") => vec![mlp_intermediate],
                n if n.ends_with("mlp.linear_fc2.weight") => vec![hidden, mlp_intermediate],
                _ => vec![hidden], // *.bias, norm1/2.*, merger.norm.*
            }
        };

        let expected_names: std::collections::HashSet<String> =
            tensor_names(&small_cfg).into_iter().collect();
        let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for name in tensor_names(&full_cfg) {
            if expected_names.contains(&name) {
                let shape = shape_for(&name);
                let numel: usize = shape.iter().product();
                tensors.insert(name, (vec![0.0_f32; numel], shape));
            } else {
                // A leftover tensor beyond small_cfg's depth=1 — its shape is irrelevant
                // because it must never be consumed by `take`.
                tensors.insert(name, (vec![0.0_f32], vec![1]));
            }
        }

        let err = assemble(tensors, &small_cfg)
            .expect_err("leftover model.visual.* tensors beyond depth=1 must be rejected");
        match err {
            InferenceError::Inference(msg) => {
                assert!(
                    msg.contains("unconsumed"),
                    "expected an unconsumed-tensor inventory error, got: {msg}"
                );
            }
            other => panic!("expected InferenceError::Inference, got {other:?}"),
        }
    }

    #[test]
    fn assemble_accepts_official_qwen35_vl_vision_mlp_dims() {
        // FIX 16 regression: official Qwen3.5-VL vision tower dims are hidden_size=1152,
        // intermediate_size=4304 (NOT 4*1152=4608). Before this fix, `assemble` hardcoded
        // `mlp_intermediate = 4 * hidden_size`, so a real checkpoint carrying a
        // [4304, 1152] `mlp.linear_fc1.weight` would be rejected as a shape mismatch
        // against the loader's wrongly-derived [4608, 1152] expectation.
        let mut cfg = tiny_vision_cfg();
        cfg.hidden_size = 1152;
        cfg.num_heads = 1; // must divide hidden_size; head geometry is irrelevant here
        cfg.intermediate_size = Some(4304);

        let hidden = cfg.hidden_size;
        let mlp_intermediate = cfg.intermediate_size.unwrap();
        let qkv_out = 3 * hidden;
        let merge_in = cfg.spatial_merge_size * cfg.spatial_merge_size * hidden;
        let out_hidden = cfg.out_hidden_size;
        let shape_for = |name: &str| -> Vec<usize> {
            match name {
                "model.visual.patch_embed.proj.weight" => vec![
                    hidden,
                    cfg.in_channels,
                    cfg.temporal_patch_size,
                    cfg.patch_size,
                    cfg.patch_size,
                ],
                "model.visual.pos_embed.weight" => vec![cfg.num_position_embeddings, hidden],
                "model.visual.merger.linear_fc1.weight" => vec![merge_in, merge_in],
                "model.visual.merger.linear_fc1.bias" => vec![merge_in],
                "model.visual.merger.linear_fc2.weight" => vec![out_hidden, merge_in],
                "model.visual.merger.linear_fc2.bias" => vec![out_hidden],
                n if n.ends_with("attn.qkv.weight") => vec![qkv_out, hidden],
                n if n.ends_with("attn.qkv.bias") => vec![qkv_out],
                n if n.ends_with("attn.proj.weight") => vec![hidden, hidden],
                n if n.ends_with("mlp.linear_fc1.weight") => vec![mlp_intermediate, hidden],
                n if n.ends_with("mlp.linear_fc1.bias") => vec![mlp_intermediate],
                n if n.ends_with("mlp.linear_fc2.weight") => vec![hidden, mlp_intermediate],
                _ => vec![hidden], // *.bias, norm1/2.*, merger.norm.*
            }
        };
        let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        for name in tensor_names(&cfg) {
            let shape = shape_for(&name);
            let numel: usize = shape.iter().product();
            tensors.insert(name, (vec![0.0_f32; numel], shape));
        }

        assemble(tensors, &cfg)
            .expect("official Qwen3.5-VL vision dims (1152/4304) must be accepted");
    }

    // Reading BF16/F16 safetensors tensors requires the `f16` feature (not default);
    // without it `get_f32_tensor` returns `InvalidSafetensors` for every such tensor.
    #[cfg(feature = "f16")]
    #[test]
    fn loads_real_fp16_checkpoint_with_correct_shapes() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let model_dir = std::path::PathBuf::from(format!("{home}/.lattice/models/qwen3.5-0.8b"));
        if !model_dir.join("config.json").exists() {
            return; // model not downloaded; skip
        }
        let cfg = Qwen35Config::from_model_dir(&model_dir).expect("0.8b config.json parses");
        let vision_cfg = cfg
            .vision_config
            .expect("released 0.8b checkpoint has a vision_config");

        let weights = load_qwen35_vision_weights(&model_dir, &vision_cfg)
            .expect("fp16 vision weights must load without error");

        assert_eq!(weights.tensor_count(), 153);
        assert_eq!(weights.blocks.len(), 12);
        assert_eq!(weights.patch_embed_weight.len(), 768 * 3 * 2 * 16 * 16);
        assert_eq!(weights.patch_embed_bias.len(), 768);
        assert_eq!(weights.pos_embed.len(), 2304 * 768);

        let block0 = &weights.blocks[0];
        assert_eq!(block0.qkv_weight.len(), 2304 * 768);
        assert_eq!(block0.qkv_bias.len(), 2304);
        assert_eq!(block0.proj_weight.len(), 768 * 768);
        assert_eq!(block0.proj_bias.len(), 768);
        assert_eq!(block0.fc1_weight.len(), 3072 * 768);
        assert_eq!(block0.fc1_bias.len(), 3072);
        assert_eq!(block0.fc2_weight.len(), 768 * 3072);
        assert_eq!(block0.fc2_bias.len(), 768);
        assert_eq!(block0.norm1_weight.len(), 768);
        assert_eq!(block0.norm2_weight.len(), 768);

        assert_eq!(weights.merger.fc1_weight.len(), 3072 * 3072);
        assert_eq!(weights.merger.fc2_weight.len(), 1024 * 3072);
        assert_eq!(weights.merger.norm_weight.len(), 768);
    }

    #[test]
    fn loads_real_q4_checkpoint_with_correct_shapes() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let model_dir = std::path::PathBuf::from(format!("{home}/.lattice/models/qwen3.5-0.8b-q4"));
        if !model_dir.join("config.json").exists() {
            return; // model not downloaded; skip
        }
        let cfg = Qwen35Config::from_model_dir(&model_dir).expect("q4 config.json parses");
        let vision_cfg = cfg
            .vision_config
            .expect("released q4 checkpoint has a vision_config");

        let weights = load_qwen35_vision_weights(&model_dir, &vision_cfg)
            .expect("q4 vision weights must load without error");

        assert_eq!(weights.tensor_count(), 153);
        assert_eq!(weights.blocks.len(), 12);
        assert_eq!(weights.patch_embed_weight.len(), 768 * 3 * 2 * 16 * 16);
        assert_eq!(weights.pos_embed.len(), 2304 * 768);

        let block0 = &weights.blocks[0];
        assert_eq!(block0.qkv_weight.len(), 2304 * 768);
        assert_eq!(block0.proj_weight.len(), 768 * 768);
        assert_eq!(block0.fc1_weight.len(), 3072 * 768);
        assert_eq!(block0.fc2_weight.len(), 768 * 3072);

        assert_eq!(weights.merger.fc1_weight.len(), 3072 * 3072);
        assert_eq!(weights.merger.fc2_weight.len(), 1024 * 3072);

        // Dequantized values must be finite (no NaN/Inf leaking through the q4 path).
        assert!(block0.qkv_weight.iter().all(|v| v.is_finite()));
        assert!(weights.merger.fc2_weight.iter().all(|v| v.is_finite()));
    }
}
