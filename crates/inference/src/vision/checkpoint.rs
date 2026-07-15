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
//! window-specific weights) — is ADR-069 stage S3 ("Metal ViT forward"), not S1/S2.
//! This module only loads the real tensors as flat data; it does not wire them into
//! any forward pass.

use std::collections::HashMap;
use std::path::Path;

use crate::error::InferenceError;
use crate::model::qwen35_config::VisionModelConfig;
use crate::quant::q4_manifest;
use crate::weights::f32_weights::{ShardedSafetensors, TensorSource};
use crate::weights::q4_weights::{dequantize_q4_to_f32, load_f16_tensor_file, load_q4_file};

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

    let mut tensors = HashMap::with_capacity(expected_names.len());
    for name in expected_names {
        let (data, shape) = reader.get_f32_tensor_owned(&name)?;
        tensors.insert(name, (data, shape));
    }
    assemble(tensors, vision_cfg)
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

    let mut tensors = HashMap::new();
    for entry in manifest
        .tensors
        .iter()
        .filter(|e| e.name.starts_with("model.visual."))
    {
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
        let file_path = model_dir.join(&entry.file);
        let (data, shape) = if entry.quantized.unwrap_or(false) {
            let q4 = load_q4_file(&file_path).map_err(|e| {
                InferenceError::InvalidSafetensors(format!(
                    "failed to load q4 tensor {} from {}: {e}",
                    entry.name,
                    file_path.display()
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
            let shape = q4.shape.clone();
            (dequantize_q4_to_f32(&q4), shape)
        } else {
            load_f16_tensor_file(&file_path).map_err(|e| {
                InferenceError::InvalidSafetensors(format!(
                    "failed to load f16 tensor {} from {}: {e}",
                    entry.name,
                    file_path.display()
                ))
            })?
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
    // The real checkpoint's MLP intermediate size is 4 * hidden_size (3072 == 4 * 768
    // for the 0.8B ViT) — vision_config.json also carries an explicit
    // `intermediate_size` field, but ADR-069 S1 intentionally does not parse it, so it
    // is derived here instead.
    let mlp_intermediate = 4 * hidden;
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
        // silently returning a nine-tensor `Qwen35VisionWeights` (the S1/S2 review's
        // exact failure scenario).
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
            num_position_embeddings: 2,
            in_channels: 1,
            deepstack_visual_indexes: vec![],
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
            ("model.visual.pos_embed.weight".to_string(), vec![2, hidden]),
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
