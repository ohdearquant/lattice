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

    let mut tensors = HashMap::with_capacity(tensor_names(vision_cfg).len());
    for name in tensor_names(vision_cfg) {
        let (data, _shape) = reader.get_f32_tensor_owned(&name)?;
        tensors.insert(name, data);
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
        let file_path = model_dir.join(&entry.file);
        let data = if entry.quantized.unwrap_or(false) {
            let q4 = load_q4_file(&file_path).map_err(|e| {
                InferenceError::InvalidSafetensors(format!(
                    "failed to load q4 tensor {} from {}: {e}",
                    entry.name,
                    file_path.display()
                ))
            })?;
            dequantize_q4_to_f32(&q4)
        } else {
            let (data, _shape) = load_f16_tensor_file(&file_path).map_err(|e| {
                InferenceError::InvalidSafetensors(format!(
                    "failed to load f16 tensor {} from {}: {e}",
                    entry.name,
                    file_path.display()
                ))
            })?;
            data
        };
        tensors.insert(entry.name.clone(), data);
    }
    assemble(tensors, vision_cfg)
}

/// Pull the 153 expected tensors out of a name→data map, validating each one's length
/// against the shape derived from `vision_cfg`, and assemble them into
/// [`Qwen35VisionWeights`]. Shared by both the fp16 and q4 load paths so shape
/// validation only lives in one place.
fn assemble(
    mut tensors: HashMap<String, Vec<f32>>,
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

    let mut take = |name: String, expected_len: usize| -> Result<Vec<f32>, InferenceError> {
        let v = tensors
            .remove(&name)
            .ok_or_else(|| InferenceError::MissingTensor(name.clone()))?;
        if v.len() != expected_len {
            return Err(InferenceError::ShapeMismatch {
                name,
                expected: vec![expected_len],
                actual: vec![v.len()],
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
    let patch_embed_weight_len: usize = patch_embed_weight_shape.iter().product();
    let patch_embed_weight = take(
        "model.visual.patch_embed.proj.weight".to_string(),
        patch_embed_weight_len,
    )?;
    let patch_embed_bias = take("model.visual.patch_embed.proj.bias".to_string(), hidden)?;
    let pos_embed = take(
        "model.visual.pos_embed.weight".to_string(),
        vision_cfg.num_position_embeddings * hidden,
    )?;

    let mut blocks = Vec::with_capacity(vision_cfg.depth);
    for i in 0..vision_cfg.depth {
        let name = |suffix: &str| format!("model.visual.blocks.{i}.{suffix}");
        blocks.push(VisualBlockWeights {
            qkv_weight: take(name("attn.qkv.weight"), qkv_out * hidden)?,
            qkv_bias: take(name("attn.qkv.bias"), qkv_out)?,
            proj_weight: take(name("attn.proj.weight"), hidden * hidden)?,
            proj_bias: take(name("attn.proj.bias"), hidden)?,
            fc1_weight: take(name("mlp.linear_fc1.weight"), mlp_intermediate * hidden)?,
            fc1_bias: take(name("mlp.linear_fc1.bias"), mlp_intermediate)?,
            fc2_weight: take(name("mlp.linear_fc2.weight"), hidden * mlp_intermediate)?,
            fc2_bias: take(name("mlp.linear_fc2.bias"), hidden)?,
            norm1_weight: take(name("norm1.weight"), hidden)?,
            norm1_bias: take(name("norm1.bias"), hidden)?,
            norm2_weight: take(name("norm2.weight"), hidden)?,
            norm2_bias: take(name("norm2.bias"), hidden)?,
        });
    }

    let merger = VisualMergerWeights {
        fc1_weight: take(
            "model.visual.merger.linear_fc1.weight".to_string(),
            merge_in * merge_in,
        )?,
        fc1_bias: take("model.visual.merger.linear_fc1.bias".to_string(), merge_in)?,
        fc2_weight: take(
            "model.visual.merger.linear_fc2.weight".to_string(),
            out_hidden * merge_in,
        )?,
        fc2_bias: take(
            "model.visual.merger.linear_fc2.bias".to_string(),
            out_hidden,
        )?,
        norm_weight: take("model.visual.merger.norm.weight".to_string(), hidden)?,
        norm_bias: take("model.visual.merger.norm.bias".to_string(), hidden)?,
    };

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
