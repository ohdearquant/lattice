//! Bounded synthetic Qwen3-VL single-image vision path; no checkpoint support.
use super::checkpoint::Qwen35VisionWeights;
use super::qwen35_merger::qwen35_merger_forward;
use super::qwen35_vit::{GridThw, build_pos_embed_and_rope_tables, qwen35_vit_forward};
use lattice_arch::qwen35_config::VisionModelConfig;
use lattice_compute::vision::batch_matvec;
use lattice_inference_core::InferenceError;

const SIDE: usize = 224;
const PATCH: usize = 16;
const GRID: usize = SIDE / PATCH;
const MERGE: usize = 2;
const PATCH_LEN: usize = 3 * 2 * PATCH * PATCH;
const COUNT: usize = GRID * GRID;
const HIDDEN: usize = 8;
const HEADS: usize = 2;
const FFN: usize = 16;
const OUTPUT: usize = 6;
const POSITIONS: usize = 48 * 48;

/// **Unstable**: intermediates from the fixed synthetic image experiment.
pub struct SyntheticVisionOutput {
    /// Flattened block-major patches `[196,1536]`.
    pub patches: Vec<f32>,
    /// Patch projection including bias `[196,8]`.
    pub patch_embedding: Vec<f32>,
    /// Interpolated learned positions `[196,8]`.
    pub positions: Vec<f32>,
    /// One ViT block output `[196,8]`.
    pub block: Vec<f32>,
    /// Main merger output `[49,6]`.
    pub merged: Vec<f32>,
}
fn invalid(s: &str) -> InferenceError {
    InferenceError::InvalidInput(s.to_owned())
}
fn config() -> VisionModelConfig {
    VisionModelConfig {
        depth: 1,
        hidden_size: HIDDEN,
        num_heads: HEADS,
        patch_size: PATCH,
        spatial_merge_size: MERGE,
        out_hidden_size: OUTPUT,
        temporal_patch_size: 2,
        num_position_embeddings: POSITIONS,
        in_channels: 3,
        deepstack_visual_indexes: vec![],
        intermediate_size: Some(FFN),
    }
}
fn validate(w: &Qwen35VisionWeights) -> Result<(), InferenceError> {
    if w.blocks.len() != 1 || w.patch_embed_weight_shape != [HIDDEN, 3, 2, PATCH, PATCH] {
        return Err(invalid(
            "synthetic Qwen3-VL requires one block and the fixed patch shape",
        ));
    }
    let b = &w.blocks[0];
    let m = &w.merger;
    for (values, expected) in [
        (w.patch_embed_weight.as_slice(), HIDDEN * PATCH_LEN),
        (w.patch_embed_bias.as_slice(), HIDDEN),
        (w.pos_embed.as_slice(), POSITIONS * HIDDEN),
        (b.qkv_weight.as_slice(), 3 * HIDDEN * HIDDEN),
        (b.qkv_bias.as_slice(), 3 * HIDDEN),
        (b.proj_weight.as_slice(), HIDDEN * HIDDEN),
        (b.proj_bias.as_slice(), HIDDEN),
        (b.fc1_weight.as_slice(), FFN * HIDDEN),
        (b.fc1_bias.as_slice(), FFN),
        (b.fc2_weight.as_slice(), HIDDEN * FFN),
        (b.fc2_bias.as_slice(), HIDDEN),
        (b.norm1_weight.as_slice(), HIDDEN),
        (b.norm1_bias.as_slice(), HIDDEN),
        (b.norm2_weight.as_slice(), HIDDEN),
        (b.norm2_bias.as_slice(), HIDDEN),
        (m.norm_weight.as_slice(), HIDDEN),
        (m.norm_bias.as_slice(), HIDDEN),
        (m.fc1_weight.as_slice(), 4 * HIDDEN * 4 * HIDDEN),
        (m.fc1_bias.as_slice(), 4 * HIDDEN),
        (m.fc2_weight.as_slice(), OUTPUT * 4 * HIDDEN),
        (m.fc2_bias.as_slice(), OUTPUT),
    ] {
        if values.len() != expected || values.iter().any(|v| !v.is_finite() || v.abs() > 16.0) {
            return Err(invalid("synthetic Qwen3-VL invalid weight shape or value"));
        }
    }
    Ok(())
}

/// Run the fixed 224x224 one-block synthetic vision path over normalized RGB.
///
/// The input is channel-major `[3,224,224]`, with finite values in [-1,1].
/// This opt-in experiment is not a checkpoint loader or a supported model.
///
/// # Errors
/// Rejects mismatched shapes, out-of-range or nonfinite values, allocation and computation errors.
pub fn run(
    image: &[f32],
    weights: &Qwen35VisionWeights,
) -> Result<SyntheticVisionOutput, InferenceError> {
    if image.len() != 3 * SIDE * SIDE || image.iter().any(|v| !v.is_finite() || v.abs() > 1.0) {
        return Err(invalid(
            "synthetic Qwen3-VL requires finite normalized 224x224 RGB",
        ));
    }
    validate(weights)?;
    let mut patches = Vec::new();
    patches
        .try_reserve_exact(COUNT * PATCH_LEN)
        .map_err(|_| invalid("synthetic Qwen3-VL patch allocation failed"))?;
    for br in 0..GRID / MERGE {
        for bc in 0..GRID / MERGE {
            for sr in 0..MERGE {
                for sc in 0..MERGE {
                    let y = (br * MERGE + sr) * PATCH;
                    let x = (bc * MERGE + sc) * PATCH;
                    for c in 0..3 {
                        for _ in 0..2 {
                            for dy in 0..PATCH {
                                for dx in 0..PATCH {
                                    patches.push(image[c * SIDE * SIDE + (y + dy) * SIDE + x + dx]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let cfg = config();
    let grid = GridThw {
        t: 1,
        h: GRID,
        w: GRID,
    };
    let mut patch_embedding = batch_matvec(
        &weights.patch_embed_weight,
        &patches,
        COUNT,
        HIDDEN,
        PATCH_LEN,
    );
    for row in patch_embedding.chunks_exact_mut(HIDDEN) {
        for (value, bias) in row.iter_mut().zip(&weights.patch_embed_bias) {
            *value += bias;
        }
    }
    let (positions, _, _) = build_pos_embed_and_rope_tables(weights, &cfg, grid);
    let block = qwen35_vit_forward(weights, &cfg, &patches, grid)
        .map_err(|e| invalid(&format!("synthetic Qwen3-VL ViT failed: {e}")))?;
    let merged = qwen35_merger_forward(&weights.merger, &cfg, &block)
        .map_err(|e| invalid(&format!("synthetic Qwen3-VL merger failed: {e}")))?;
    if [&patch_embedding, &positions, &block, &merged]
        .into_iter()
        .flatten()
        .any(|v| !v.is_finite())
    {
        return Err(invalid("synthetic Qwen3-VL produced nonfinite values"));
    }
    Ok(SyntheticVisionOutput {
        patches,
        patch_embedding,
        positions,
        block,
        merged,
    })
}

#[cfg(test)]
mod tests;
