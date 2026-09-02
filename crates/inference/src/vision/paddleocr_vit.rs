//! PaddleOCR-VL-1.6 vision encoder + projector, CPU reference forward.
//!
//! The checkpoint's `visual.*` tree is a SigLIP-shaped NaViT encoder:
//! patch-14 conv embedding, a learned 27x27 position table bilinearly
//! interpolated (`align_corners=False`) to each image's patch grid, 27
//! pre-LayerNorm blocks (separate biased Q/K/V/out projections, 2-axis
//! rotate-half RoPE over `head_dim / 2` from the `(row, col)` patch
//! coordinates, tanh-GELU MLP), and a final LayerNorm. Its `mlp_AR.*`
//! projector is `LayerNorm(1e-5) -> 2x2 raster merge -> Linear -> exact
//! GELU -> Linear` into the text hidden size. Patches stay in plain raster
//! order throughout (the reference permutes nothing), which is the one
//! structural difference from `qwen35_vit.rs`, whose preprocessing emits
//! spatial-merge-block-major order and whose position interpolation samples
//! with `align_corners=True` semantics. Shared arithmetic (rotate-half RoPE,
//! full attention, LayerNorm, both GELU variants) is reused from there.
//!
//! The 32768-row `packing_position_embedding` table and the attention
//! pooling head are not loaded: the reference's OCR path runs with
//! `interpolate_pos_encoding=True` and `return_pooler_output=False`, so
//! neither participates in the forward.

use std::path::Path;

use serde::Deserialize;

use super::qwen35_merger::gelu_exact;
use super::qwen35_vit::{apply_rope_inplace, multihead_attention_full};
use super::vit::{batch_matvec, gelu, layer_norm};
use crate::error::InferenceError;
use crate::weights::TensorSource;

/// Vision-tower and projector hyperparameters read from the checkpoint's
/// `config.json` (`vision_config` plus the text `hidden_size` the projector
/// targets).
#[derive(Debug, Clone, PartialEq)]
pub struct PaddleOcrVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub layer_norm_eps: f32,
    pub spatial_merge_size: usize,
    /// Text-decoder hidden size: the projector's output width.
    pub text_hidden_size: usize,
}

#[derive(Deserialize)]
struct RawTopLevel {
    hidden_size: usize,
    vision_config: RawVision,
}

#[derive(Deserialize)]
struct RawVision {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default = "default_channels")]
    num_channels: usize,
    patch_size: usize,
    image_size: usize,
    #[serde(default = "default_ln_eps")]
    layer_norm_eps: f32,
    #[serde(default = "default_merge")]
    spatial_merge_size: usize,
}

fn default_channels() -> usize {
    3
}
fn default_ln_eps() -> f32 {
    1e-6
}
fn default_merge() -> usize {
    2
}

/// The projector's own LayerNorm epsilon: hardcoded `1e-05` in the reference
/// `Projector.__init__`, independent of `vision_config.layer_norm_eps`.
pub const PROJECTOR_NORM_EPS: f32 = 1e-5;
/// Vision RoPE base, hardcoded in the reference `SigLIPRotaryEmbedding`.
pub const VISION_ROPE_THETA: f32 = 10_000.0;
/// Maximum number of transformer blocks accepted from a vision checkpoint.
///
/// The shipped checkpoint has 27 blocks; 64 leaves room for checkpoints more
/// than twice as deep without allowing unbounded layer-vector allocation.
pub const MAX_VISION_LAYERS: usize = 64;
/// Maximum vision hidden width accepted from a checkpoint.
///
/// The shipped width is 1152; 4096 is a generous multiple that keeps the
/// checked tensor products within practical CPU memory bounds.
pub const MAX_VISION_HIDDEN_SIZE: usize = 4096;
/// Maximum vision MLP intermediate width accepted from a checkpoint.
///
/// The shipped width is 4304; 16384 leaves room for wider checkpoints while
/// bounding the per-token MLP buffers.
pub const MAX_VISION_INTERMEDIATE_SIZE: usize = 16_384;
/// Maximum attention-head count accepted from a checkpoint.
///
/// The shipped count is 16; 64 accommodates wider attention layouts while
/// bounding per-head work.
pub const MAX_VISION_HEADS: usize = 64;
/// Maximum learned position-table side in patches.
///
/// The shipped table side is 27; 128 allows substantially larger position
/// tables while keeping their checked allocation products bounded.
pub const MAX_VISION_POSITION_SIDE: usize = 128;
/// Maximum text-decoder width emitted by the projector.
///
/// The shipped text width is 1024; 4096 supports wider decoder projections
/// while bounding projector output allocations.
pub const MAX_VISION_TEXT_HIDDEN_SIZE: usize = 4096;
/// Maximum number of image patches accepted by the PaddleOCR-VL forward.
///
/// The shipped preprocessing budget of 1,003,520 pixels at patch 14 yields at
/// most 5,120 tokens. At this cap the shared per-head score buffer is 256 MiB.
pub const MAX_VISION_TOKENS: usize = 8192;

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize, InferenceError> {
    a.checked_mul(b).ok_or_else(|| {
        InferenceError::Inference(format!(
            "vision dimension product {what} ({a} * {b}) overflows usize"
        ))
    })
}

fn reserve_exact<T>(buffer: &mut Vec<T>, len: usize, what: &str) -> Result<(), InferenceError> {
    buffer.try_reserve_exact(len).map_err(|error| {
        InferenceError::Inference(format!(
            "failed to allocate {what} buffer for {len} elements: {error}"
        ))
    })
}

fn try_zeroed_f32(len: usize, what: &str) -> Result<Vec<f32>, InferenceError> {
    let mut buffer = Vec::new();
    reserve_exact(&mut buffer, len, what)?;
    buffer.resize(len, 0.0);
    Ok(buffer)
}

fn try_clone_f32(input: &[f32], what: &str) -> Result<Vec<f32>, InferenceError> {
    let mut buffer = Vec::new();
    reserve_exact(&mut buffer, input.len(), what)?;
    buffer.extend_from_slice(input);
    Ok(buffer)
}

#[derive(Debug, Clone, Copy)]
struct CheckedConfigSizes {
    patch_len: usize,
    position_rows: usize,
    merged_width: usize,
}

fn checked_config_sizes(cfg: &PaddleOcrVisionConfig) -> Result<CheckedConfigSizes, InferenceError> {
    let channels_patch = checked_mul(
        cfg.num_channels,
        cfg.patch_size,
        "num_channels * patch_size",
    )?;
    let patch_len = checked_mul(channels_patch, cfg.patch_size, "patch length")?;
    let side = cfg.pos_table_side();
    let position_rows = checked_mul(side, side, "position table side squared")?;
    checked_mul(position_rows, cfg.hidden_size, "position table values")?;
    checked_mul(cfg.hidden_size, cfg.hidden_size, "hidden_size squared")?;
    checked_mul(
        cfg.intermediate_size,
        cfg.hidden_size,
        "intermediate_size * hidden_size",
    )?;
    checked_mul(
        cfg.hidden_size,
        cfg.intermediate_size,
        "hidden_size * intermediate_size",
    )?;
    let merge_width = checked_mul(
        cfg.hidden_size,
        cfg.spatial_merge_size,
        "hidden_size * spatial_merge_size",
    )?;
    let merged_width = checked_mul(merge_width, cfg.spatial_merge_size, "merged width")?;
    checked_mul(merged_width, merged_width, "merged width squared")?;
    checked_mul(
        cfg.text_hidden_size,
        merged_width,
        "text_hidden_size * merged width",
    )?;
    Ok(CheckedConfigSizes {
        patch_len,
        position_rows,
        merged_width,
    })
}

#[derive(Debug, Clone, Copy)]
struct CheckedForwardSizes {
    config: CheckedConfigSizes,
    hidden_values: usize,
    qkv_values: usize,
    intermediate_values: usize,
    packed_values: usize,
    projector_values: usize,
}

fn checked_forward_sizes(
    cfg: &PaddleOcrVisionConfig,
    n: usize,
    grid_h: usize,
    grid_w: usize,
) -> Result<CheckedForwardSizes, InferenceError> {
    let config = checked_config_sizes(cfg)?;
    let hidden_values = checked_mul(n, cfg.hidden_size, "n * hidden_size")?;
    let qkv_width = checked_mul(3, cfg.hidden_size, "3 * hidden_size")?;
    let qkv_values = checked_mul(n, qkv_width, "n * 3 * hidden_size")?;
    let intermediate_values = checked_mul(n, cfg.intermediate_size, "n * intermediate_size")?;
    let blocks_h = grid_h / cfg.spatial_merge_size;
    let blocks_w = grid_w / cfg.spatial_merge_size;
    let merged_tokens = checked_mul(blocks_h, blocks_w, "merged token count")?;
    let packed_values = checked_mul(
        merged_tokens,
        config.merged_width,
        "packed projector values",
    )?;
    let projector_values = checked_mul(
        merged_tokens,
        cfg.text_hidden_size,
        "projector output values",
    )?;
    Ok(CheckedForwardSizes {
        config,
        hidden_values,
        qkv_values,
        intermediate_values,
        packed_values,
        projector_values,
    })
}

impl PaddleOcrVisionConfig {
    /// Parse and validate a checkpoint `config.json`.
    ///
    /// # Errors
    ///
    /// [`InferenceError::Inference`] on unreadable/unparseable JSON or a
    /// configuration this reference does not implement (see [`Self::validate`]).
    pub fn from_config_json(path: &Path) -> Result<Self, InferenceError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| InferenceError::Inference(format!("read {}: {e}", path.display())))?;
        Self::from_config_json_str(&text)
    }

    /// [`Self::from_config_json`] over an in-memory document.
    ///
    /// # Errors
    ///
    /// As [`Self::from_config_json`].
    pub fn from_config_json_str(text: &str) -> Result<Self, InferenceError> {
        let raw: RawTopLevel = serde_json::from_str(text)
            .map_err(|e| InferenceError::Inference(format!("parse config.json: {e}")))?;
        let v = raw.vision_config;
        let cfg = Self {
            hidden_size: v.hidden_size,
            intermediate_size: v.intermediate_size,
            num_hidden_layers: v.num_hidden_layers,
            num_attention_heads: v.num_attention_heads,
            num_channels: v.num_channels,
            patch_size: v.patch_size,
            image_size: v.image_size,
            layer_norm_eps: v.layer_norm_eps,
            spatial_merge_size: v.spatial_merge_size,
            text_hidden_size: raw.hidden_size,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject shapes this reference cannot run: dimensions must stay within
    /// the documented checkpoint caps, heads must divide the hidden size,
    /// `head_dim` must be divisible by 4 (two RoPE axes, each with paired
    /// frequencies), the position table must be square in patches, and the
    /// projector's merge kernel is fixed at 2x2 by the reference.
    ///
    /// # Errors
    ///
    /// [`InferenceError::Inference`] naming the violated invariant.
    pub fn validate(&self) -> Result<(), InferenceError> {
        if !(1..=MAX_VISION_LAYERS).contains(&self.num_hidden_layers) {
            return Err(InferenceError::Inference(format!(
                "vision num_hidden_layers {} must be in 1..=MAX_VISION_LAYERS ({MAX_VISION_LAYERS})",
                self.num_hidden_layers
            )));
        }
        if !(1..=MAX_VISION_HIDDEN_SIZE).contains(&self.hidden_size) {
            return Err(InferenceError::Inference(format!(
                "vision hidden_size {} must be in 1..=MAX_VISION_HIDDEN_SIZE ({MAX_VISION_HIDDEN_SIZE})",
                self.hidden_size
            )));
        }
        if !(1..=MAX_VISION_INTERMEDIATE_SIZE).contains(&self.intermediate_size) {
            return Err(InferenceError::Inference(format!(
                "vision intermediate_size {} must be in 1..=MAX_VISION_INTERMEDIATE_SIZE ({MAX_VISION_INTERMEDIATE_SIZE})",
                self.intermediate_size
            )));
        }
        if !(1..=MAX_VISION_HEADS).contains(&self.num_attention_heads) {
            return Err(InferenceError::Inference(format!(
                "vision num_attention_heads {} must be in 1..=MAX_VISION_HEADS ({MAX_VISION_HEADS})",
                self.num_attention_heads
            )));
        }
        if self.num_channels != 3 {
            return Err(InferenceError::Inference(format!(
                "vision num_channels {} is unsupported; the reference requires 3",
                self.num_channels
            )));
        }
        if !(1..=MAX_VISION_TEXT_HIDDEN_SIZE).contains(&self.text_hidden_size) {
            return Err(InferenceError::Inference(format!(
                "vision text_hidden_size {} must be in 1..=MAX_VISION_TEXT_HIDDEN_SIZE ({MAX_VISION_TEXT_HIDDEN_SIZE})",
                self.text_hidden_size
            )));
        }
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return Err(InferenceError::Inference(format!(
                "vision hidden_size {} not divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if !self.head_dim().is_multiple_of(4) {
            return Err(InferenceError::Inference(format!(
                "vision head_dim {} must be a multiple of 4 for 2-axis RoPE",
                self.head_dim()
            )));
        }
        // The reference sizes its position table as `(image_size // patch_size)^2`
        // (384 // 14 = 27, discarding the 6-pixel remainder).
        if self.patch_size == 0 {
            return Err(InferenceError::Inference(format!(
                "vision patch_size {} must be greater than zero",
                self.patch_size
            )));
        }
        let side = self.image_size / self.patch_size;
        if !(1..=MAX_VISION_POSITION_SIDE).contains(&side) {
            return Err(InferenceError::Inference(format!(
                "vision position-table side {side} must be in 1..=MAX_VISION_POSITION_SIDE ({MAX_VISION_POSITION_SIDE})"
            )));
        }
        if self.spatial_merge_size != 2 {
            return Err(InferenceError::Inference(format!(
                "projector merge kernel is fixed at 2x2 by the reference; got {}",
                self.spatial_merge_size
            )));
        }
        checked_config_sizes(self)?;
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Flattened per-patch input length: `channels * patch * patch`.
    pub fn patch_len(&self) -> usize {
        self.num_channels * self.patch_size * self.patch_size
    }

    /// Side of the square learned position table, in patches (27 for the
    /// 384/14 checkpoint).
    pub fn pos_table_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Projector input width: `hidden * merge * merge`.
    pub fn merged_width(&self) -> usize {
        self.hidden_size * self.spatial_merge_size * self.spatial_merge_size
    }
}

/// One encoder block's tensors, all f32 row-major `[out, in]`.
#[derive(Debug, Clone)]
pub struct PaddleOcrVitLayerWeights {
    pub ln1_weight: Vec<f32>,
    pub ln1_bias: Vec<f32>,
    pub q_weight: Vec<f32>,
    pub q_bias: Vec<f32>,
    pub k_weight: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_weight: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub out_weight: Vec<f32>,
    pub out_bias: Vec<f32>,
    pub ln2_weight: Vec<f32>,
    pub ln2_bias: Vec<f32>,
    pub fc1_weight: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<f32>,
    pub fc2_bias: Vec<f32>,
}

/// Encoder + projector tensors (`visual.vision_model.*` minus the pooling
/// head and packing table, plus `mlp_AR.*`).
#[derive(Debug, Clone)]
pub struct PaddleOcrVisionWeights {
    /// `[hidden, channels * patch * patch]`, the conv kernel flattened in
    /// `(channel, row, col)` order — the same order a raster patch flattens to.
    pub patch_weight: Vec<f32>,
    pub patch_bias: Vec<f32>,
    /// `[side * side, hidden]`, row-major over the square table.
    pub pos_embed: Vec<f32>,
    pub layers: Vec<PaddleOcrVitLayerWeights>,
    pub post_ln_weight: Vec<f32>,
    pub post_ln_bias: Vec<f32>,
    pub proj_norm_weight: Vec<f32>,
    pub proj_norm_bias: Vec<f32>,
    /// `[merged, merged]`
    pub proj_l1_weight: Vec<f32>,
    pub proj_l1_bias: Vec<f32>,
    /// `[text_hidden, merged]`
    pub proj_l2_weight: Vec<f32>,
    pub proj_l2_bias: Vec<f32>,
}

fn load_tensor<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, InferenceError> {
    let (data, shape) = source.get_f32_tensor_owned(name)?;
    if shape != expected {
        return Err(InferenceError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: shape,
        });
    }
    Ok(data)
}

impl PaddleOcrVisionWeights {
    /// Load every tensor the forward reads, with fail-closed shape checks
    /// against `cfg`.
    ///
    /// # Errors
    ///
    /// Any source error, or [`InferenceError::ShapeMismatch`] on the first
    /// tensor whose header shape disagrees with `cfg`.
    pub fn load<T: TensorSource + ?Sized>(
        source: &mut T,
        cfg: &PaddleOcrVisionConfig,
    ) -> Result<Self, InferenceError> {
        cfg.validate()?;
        let sizes = checked_config_sizes(cfg)?;
        let h = cfg.hidden_size;
        let p = cfg.patch_size;
        let vm = "visual.vision_model.";
        let patch_weight = load_tensor(
            source,
            &format!("{vm}embeddings.patch_embedding.weight"),
            &[h, cfg.num_channels, p, p],
        )?;
        let patch_bias = load_tensor(
            source,
            &format!("{vm}embeddings.patch_embedding.bias"),
            &[h],
        )?;
        let pos_embed = load_tensor(
            source,
            &format!("{vm}embeddings.position_embedding.weight"),
            &[sizes.position_rows, h],
        )?;
        let mut layers = Vec::new();
        reserve_exact(&mut layers, cfg.num_hidden_layers, "vision layer")?;
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("{vm}encoder.layers.{i}.");
            let inter = cfg.intermediate_size;
            layers.push(PaddleOcrVitLayerWeights {
                ln1_weight: load_tensor(source, &format!("{lp}layer_norm1.weight"), &[h])?,
                ln1_bias: load_tensor(source, &format!("{lp}layer_norm1.bias"), &[h])?,
                q_weight: load_tensor(source, &format!("{lp}self_attn.q_proj.weight"), &[h, h])?,
                q_bias: load_tensor(source, &format!("{lp}self_attn.q_proj.bias"), &[h])?,
                k_weight: load_tensor(source, &format!("{lp}self_attn.k_proj.weight"), &[h, h])?,
                k_bias: load_tensor(source, &format!("{lp}self_attn.k_proj.bias"), &[h])?,
                v_weight: load_tensor(source, &format!("{lp}self_attn.v_proj.weight"), &[h, h])?,
                v_bias: load_tensor(source, &format!("{lp}self_attn.v_proj.bias"), &[h])?,
                out_weight: load_tensor(
                    source,
                    &format!("{lp}self_attn.out_proj.weight"),
                    &[h, h],
                )?,
                out_bias: load_tensor(source, &format!("{lp}self_attn.out_proj.bias"), &[h])?,
                ln2_weight: load_tensor(source, &format!("{lp}layer_norm2.weight"), &[h])?,
                ln2_bias: load_tensor(source, &format!("{lp}layer_norm2.bias"), &[h])?,
                fc1_weight: load_tensor(source, &format!("{lp}mlp.fc1.weight"), &[inter, h])?,
                fc1_bias: load_tensor(source, &format!("{lp}mlp.fc1.bias"), &[inter])?,
                fc2_weight: load_tensor(source, &format!("{lp}mlp.fc2.weight"), &[h, inter])?,
                fc2_bias: load_tensor(source, &format!("{lp}mlp.fc2.bias"), &[h])?,
            });
        }
        let post_ln_weight = load_tensor(source, &format!("{vm}post_layernorm.weight"), &[h])?;
        let post_ln_bias = load_tensor(source, &format!("{vm}post_layernorm.bias"), &[h])?;
        let merged = sizes.merged_width;
        let t = cfg.text_hidden_size;
        Ok(Self {
            patch_weight,
            patch_bias,
            pos_embed,
            layers,
            post_ln_weight,
            post_ln_bias,
            proj_norm_weight: load_tensor(source, "mlp_AR.pre_norm.weight", &[h])?,
            proj_norm_bias: load_tensor(source, "mlp_AR.pre_norm.bias", &[h])?,
            proj_l1_weight: load_tensor(source, "mlp_AR.linear_1.weight", &[merged, merged])?,
            proj_l1_bias: load_tensor(source, "mlp_AR.linear_1.bias", &[merged])?,
            proj_l2_weight: load_tensor(source, "mlp_AR.linear_2.weight", &[t, merged])?,
            proj_l2_bias: load_tensor(source, "mlp_AR.linear_2.bias", &[t])?,
        })
    }
}

/// Every intermediate the golden gate compares, all `[n, hidden]` row-major
/// in raster patch order except `projector` (`[n / 4, text_hidden]`).
#[derive(Debug, Clone)]
pub struct PaddleOcrVisionTrace {
    /// Patch embedding plus interpolated position embedding.
    pub embed: Vec<f32>,
    pub layer_outputs: Vec<Vec<f32>>,
    pub post_layernorm: Vec<f32>,
    pub projector: Vec<f32>,
}

/// Bilinear sample of the square position table at output cell
/// `(h_idx, w_idx)` of an `(grid_h, grid_w)` grid, PyTorch
/// `F.interpolate(mode="bilinear", align_corners=False)` semantics: source
/// coordinate `(dst + 0.5) * side / dst_len - 0.5`, clamped at zero, upper
/// neighbour clamped at `side - 1`. Accumulates into `out` (zeroed by the
/// caller).
#[allow(clippy::too_many_arguments)]
fn bilinear_pos_embed_half_pixel(
    pos_embed: &[f32],
    hidden: usize,
    side: usize,
    grid_h: usize,
    grid_w: usize,
    h_idx: usize,
    w_idx: usize,
    out: &mut [f32],
) {
    let src = |dst: usize, dst_len: usize| -> (usize, usize, f32) {
        let scale = side as f32 / dst_len as f32;
        let coord = ((dst as f32 + 0.5) * scale - 0.5).max(0.0);
        let lo = (coord.floor() as usize).min(side - 1);
        let hi = (lo + 1).min(side - 1);
        (lo, hi, coord - lo as f32)
    };
    let (h0, h1, hf) = src(h_idx, grid_h);
    let (w0, w1, wf) = src(w_idx, grid_w);
    let corners = [
        (h0, w0, (1.0 - hf) * (1.0 - wf)),
        (h0, w1, (1.0 - hf) * wf),
        (h1, w0, hf * (1.0 - wf)),
        (h1, w1, hf * wf),
    ];
    for (r, c, weight) in corners {
        if weight == 0.0 {
            continue;
        }
        let row = &pos_embed[(r * side + c) * hidden..(r * side + c + 1) * hidden];
        for (o, &v) in out.iter_mut().zip(row) {
            *o += v * weight;
        }
    }
}

/// Position-embedding contribution `[n * hidden]` and rotate-half RoPE
/// `cos`/`sin` tables `[n * head_dim]` for a raster `(grid_h, grid_w)` grid.
/// Per patch: `rotary = concat(row * inv_freq, col * inv_freq)` over
/// `head_dim / 2`, then `emb = concat(rotary, rotary)` — the reference's
/// `rope_emb_max_grid[pids].flatten(1).repeat(1, 2)`.
type PosEmbedAndRopeTables = (Vec<f32>, Vec<f32>, Vec<f32>);

fn build_pos_embed_and_rope_tables(
    weights: &PaddleOcrVisionWeights,
    cfg: &PaddleOcrVisionConfig,
    grid_h: usize,
    grid_w: usize,
) -> Result<PosEmbedAndRopeTables, InferenceError> {
    let hidden = cfg.hidden_size;
    let n = grid_h.checked_mul(grid_w).ok_or_else(|| {
        InferenceError::Inference(format!(
            "vision grid {grid_h}x{grid_w} overflows the patch count"
        ))
    })?;
    let sizes = checked_forward_sizes(cfg, n, grid_h, grid_w)?;
    let side = cfg.pos_table_side();
    let head_dim = cfg.head_dim();
    let rope_dim = head_dim / 2;
    let rope_half = rope_dim / 2;
    let mut inv_freq = Vec::new();
    reserve_exact(&mut inv_freq, rope_half, "vision RoPE frequency")?;
    for i in 0..rope_half {
        inv_freq.push(1.0 / VISION_ROPE_THETA.powf((2 * i) as f32 / rope_dim as f32));
    }

    let mut pos = try_zeroed_f32(sizes.hidden_values, "position embedding")?;
    let cos_values = checked_mul(n, head_dim, "n * head_dim for cosine table")?;
    let sin_values = checked_mul(n, head_dim, "n * head_dim for sine table")?;
    let mut cos_t = try_zeroed_f32(cos_values, "cosine table")?;
    let mut sin_t = try_zeroed_f32(sin_values, "sine table")?;
    for (idx, (pos_row, (cos_row, sin_row))) in pos
        .chunks_mut(hidden)
        .zip(cos_t.chunks_mut(head_dim).zip(sin_t.chunks_mut(head_dim)))
        .enumerate()
    {
        let h_idx = idx / grid_w;
        let w_idx = idx % grid_w;
        bilinear_pos_embed_half_pixel(
            &weights.pos_embed,
            hidden,
            side,
            grid_h,
            grid_w,
            h_idx,
            w_idx,
            pos_row,
        );
        for i in 0..rope_half {
            let (sh, ch) = (h_idx as f32 * inv_freq[i]).sin_cos();
            let (sw, cw) = (w_idx as f32 * inv_freq[i]).sin_cos();
            for base in [0, rope_dim] {
                cos_row[base + i] = ch;
                sin_row[base + i] = sh;
                cos_row[base + rope_half + i] = cw;
                sin_row[base + rope_half + i] = sw;
            }
        }
    }
    Ok((pos, cos_t, sin_t))
}

fn add_bias_rows(x: &mut [f32], bias: &[f32]) {
    for row in x.chunks_mut(bias.len()) {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
}

fn layer_norm_rows(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    for row in x.chunks_mut(weight.len()) {
        layer_norm(row, weight, bias, eps);
    }
}

fn check_grid(
    cfg: &PaddleOcrVisionConfig,
    grid_h: usize,
    grid_w: usize,
) -> Result<usize, InferenceError> {
    let n = grid_h.checked_mul(grid_w).ok_or_else(|| {
        InferenceError::Inference(format!(
            "vision grid {grid_h}x{grid_w} overflows the patch count"
        ))
    })?;
    if n > MAX_VISION_TOKENS {
        return Err(InferenceError::Inference(format!(
            "vision grid {grid_h}x{grid_w} has {n} tokens; maximum is MAX_VISION_TOKENS ({MAX_VISION_TOKENS})"
        )));
    }
    let m = cfg.spatial_merge_size;
    if m == 0
        || grid_h == 0
        || grid_w == 0
        || !grid_h.is_multiple_of(m)
        || !grid_w.is_multiple_of(m)
    {
        return Err(InferenceError::Inference(format!(
            "vision grid {grid_h}x{grid_w} must be non-empty and a multiple of the {m}x{m} merge kernel"
        )));
    }
    Ok(n)
}

/// Encoder + projector over one image's raster-ordered patches.
///
/// `pixel_values` is `[grid_h * grid_w, channels * patch * patch]`, each
/// patch flattened `(channel, row, col)` after the reference's
/// rescale-and-normalize preprocessing.
///
/// # Errors
///
/// [`InferenceError::Inference`] if the grid is empty, exceeds
/// [`MAX_VISION_TOKENS`], is not a multiple of the merge kernel on either
/// axis, overflows a dimension product, or `pixel_values` has the wrong
/// length.
pub fn paddleocr_vision_forward_trace(
    weights: &PaddleOcrVisionWeights,
    cfg: &PaddleOcrVisionConfig,
    pixel_values: &[f32],
    grid_h: usize,
    grid_w: usize,
) -> Result<PaddleOcrVisionTrace, InferenceError> {
    let n = check_grid(cfg, grid_h, grid_w)?;
    cfg.validate()?;
    let sizes = checked_forward_sizes(cfg, n, grid_h, grid_w)?;
    let hidden = cfg.hidden_size;
    let patch_len = sizes.config.patch_len;
    let pixel_values_len = checked_mul(n, patch_len, "n * patch_len")?;
    if pixel_values.len() != pixel_values_len {
        return Err(InferenceError::Inference(format!(
            "pixel_values has {} values; expected {n} patches x {patch_len}",
            pixel_values.len()
        )));
    }
    if weights.layers.len() != cfg.num_hidden_layers {
        return Err(InferenceError::Inference(format!(
            "weights carry {} layers; config says {}",
            weights.layers.len(),
            cfg.num_hidden_layers
        )));
    }

    let mut hidden_states = batch_matvec(&weights.patch_weight, pixel_values, n, hidden, patch_len);
    add_bias_rows(&mut hidden_states, &weights.patch_bias);
    let (pos, cos_t, sin_t) = build_pos_embed_and_rope_tables(weights, cfg, grid_h, grid_w)?;
    for (x, p) in hidden_states.iter_mut().zip(&pos) {
        *x += p;
    }
    let embed = try_clone_f32(&hidden_states, "embedding trace")?;

    let head_dim = cfg.head_dim();
    let n_heads = cfg.num_attention_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let eps = cfg.layer_norm_eps;
    let mut layer_outputs = Vec::new();
    reserve_exact(
        &mut layer_outputs,
        cfg.num_hidden_layers,
        "layer output trace",
    )?;

    for layer in &weights.layers {
        let mut normed = try_clone_f32(&hidden_states, "attention normalization")?;
        layer_norm_rows(&mut normed, &layer.ln1_weight, &layer.ln1_bias, eps);
        let mut q = batch_matvec(&layer.q_weight, &normed, n, hidden, hidden);
        let mut k = batch_matvec(&layer.k_weight, &normed, n, hidden, hidden);
        let mut v = batch_matvec(&layer.v_weight, &normed, n, hidden, hidden);
        add_bias_rows(&mut q, &layer.q_bias);
        add_bias_rows(&mut k, &layer.k_bias);
        add_bias_rows(&mut v, &layer.v_bias);

        // Fused `[n, Q | K | V]` layout for the shared attention kernel, RoPE
        // applied per head on Q and K while packing.
        let qkv_width = checked_mul(3, hidden, "3 * hidden_size")?;
        let mut qkv = try_zeroed_f32(sizes.qkv_values, "qkv")?;
        for (((q_row, k_row), v_row), (dst, (cos_row, sin_row))) in q
            .chunks(hidden)
            .zip(k.chunks(hidden))
            .zip(v.chunks(hidden))
            .zip(
                qkv.chunks_mut(qkv_width)
                    .zip(cos_t.chunks(head_dim).zip(sin_t.chunks(head_dim))),
            )
        {
            dst[..hidden].copy_from_slice(q_row);
            dst[hidden..2 * hidden].copy_from_slice(k_row);
            dst[2 * hidden..].copy_from_slice(v_row);
            for h in 0..n_heads {
                apply_rope_inplace(&mut dst[h * head_dim..(h + 1) * head_dim], cos_row, sin_row);
                let kb = hidden + h * head_dim;
                apply_rope_inplace(&mut dst[kb..kb + head_dim], cos_row, sin_row);
            }
        }
        let attn = multihead_attention_full(&qkv, n, hidden, n_heads, head_dim, scale);
        let mut proj = batch_matvec(&layer.out_weight, &attn, n, hidden, hidden);
        add_bias_rows(&mut proj, &layer.out_bias);
        for (x, p) in hidden_states.iter_mut().zip(&proj) {
            *x += p;
        }

        let mut normed = try_clone_f32(&hidden_states, "MLP normalization")?;
        layer_norm_rows(&mut normed, &layer.ln2_weight, &layer.ln2_bias, eps);
        let inter = cfg.intermediate_size;
        let mut fc1 = batch_matvec(&layer.fc1_weight, &normed, n, inter, hidden);
        debug_assert_eq!(fc1.len(), sizes.intermediate_values);
        for row in fc1.chunks_mut(inter) {
            for (x, b) in row.iter_mut().zip(&layer.fc1_bias) {
                *x = gelu(*x + b);
            }
        }
        let mut fc2 = batch_matvec(&layer.fc2_weight, &fc1, n, hidden, inter);
        add_bias_rows(&mut fc2, &layer.fc2_bias);
        for (x, p) in hidden_states.iter_mut().zip(&fc2) {
            *x += p;
        }
        layer_outputs.push(try_clone_f32(&hidden_states, "layer output trace")?);
    }

    let mut post = hidden_states;
    layer_norm_rows(
        &mut post,
        &weights.post_ln_weight,
        &weights.post_ln_bias,
        eps,
    );
    let projector = project_merged(weights, cfg, &post, grid_h, grid_w)?;

    Ok(PaddleOcrVisionTrace {
        embed,
        layer_outputs,
        post_layernorm: post,
        projector,
    })
}

/// `mlp_AR`: per-token LayerNorm, then each 2x2 raster block of tokens
/// `(2r + p1, 2c + p2)` concatenated in `(p1, p2)` order, then
/// `Linear -> exact GELU -> Linear`. Output `[n / 4, text_hidden]` in block
/// raster order.
fn project_merged(
    weights: &PaddleOcrVisionWeights,
    cfg: &PaddleOcrVisionConfig,
    features: &[f32],
    grid_h: usize,
    grid_w: usize,
) -> Result<Vec<f32>, InferenceError> {
    let n = checked_mul(grid_h, grid_w, "grid_h * grid_w")?;
    let sizes = checked_forward_sizes(cfg, n, grid_h, grid_w)?;
    let hidden = cfg.hidden_size;
    let m = cfg.spatial_merge_size;
    let merged = sizes.config.merged_width;
    let mut normed = try_clone_f32(features, "projector normalization")?;
    layer_norm_rows(
        &mut normed,
        &weights.proj_norm_weight,
        &weights.proj_norm_bias,
        PROJECTOR_NORM_EPS,
    );
    let blocks_h = grid_h / m;
    let blocks_w = grid_w / m;
    let nb = checked_mul(blocks_h, blocks_w, "merged token count")?;
    let mut packed = try_zeroed_f32(sizes.packed_values, "projector packed")?;
    for br in 0..blocks_h {
        for bc in 0..blocks_w {
            let dst = &mut packed[(br * blocks_w + bc) * merged..][..merged];
            for p1 in 0..m {
                for p2 in 0..m {
                    let tok = (br * m + p1) * grid_w + (bc * m + p2);
                    let slot = (p1 * m + p2) * hidden;
                    dst[slot..slot + hidden]
                        .copy_from_slice(&normed[tok * hidden..(tok + 1) * hidden]);
                }
            }
        }
    }
    let mut l1 = batch_matvec(&weights.proj_l1_weight, &packed, nb, merged, merged);
    debug_assert_eq!(l1.len(), sizes.packed_values);
    for row in l1.chunks_mut(merged) {
        for (x, b) in row.iter_mut().zip(&weights.proj_l1_bias) {
            *x = gelu_exact(*x + b);
        }
    }
    let t = cfg.text_hidden_size;
    let mut out = batch_matvec(&weights.proj_l2_weight, &l1, nb, t, merged);
    debug_assert_eq!(out.len(), sizes.projector_values);
    add_bias_rows(&mut out, &weights.proj_l2_bias);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: &str = r#"{"hidden_size": 1024, "vision_config": {"hidden_size": 1152,
        "intermediate_size": 4304, "num_hidden_layers": 27, "num_attention_heads": 16,
        "patch_size": 14, "image_size": 384, "layer_norm_eps": 1e-06, "spatial_merge_size": 2}}"#;

    fn valid_cfg() -> PaddleOcrVisionConfig {
        PaddleOcrVisionConfig::from_config_json_str(CFG).unwrap()
    }

    #[test]
    fn config_parses_pinned_shape() {
        let cfg = PaddleOcrVisionConfig::from_config_json_str(CFG).unwrap();
        assert_eq!(cfg.head_dim(), 72);
        assert_eq!(cfg.pos_table_side(), 27);
        assert_eq!(cfg.patch_len(), 588);
        assert_eq!(cfg.merged_width(), 4608);
        assert_eq!(cfg.text_hidden_size, 1024);
    }

    #[test]
    fn config_rejects_unsupported_merge_and_heads() {
        let bad_merge = CFG.replace("\"spatial_merge_size\": 2", "\"spatial_merge_size\": 3");
        assert!(PaddleOcrVisionConfig::from_config_json_str(&bad_merge).is_err());
        let bad_heads = CFG.replace("\"num_attention_heads\": 16", "\"num_attention_heads\": 7");
        assert!(PaddleOcrVisionConfig::from_config_json_str(&bad_heads).is_err());
    }

    #[test]
    fn config_rejects_layer_count_above_cap() {
        let mut cfg = valid_cfg();
        for layers in [0, MAX_VISION_LAYERS + 1, usize::MAX] {
            cfg.num_hidden_layers = layers;
            let error = cfg.validate().unwrap_err().to_string();
            assert!(error.contains("MAX_VISION_LAYERS"), "{error}");
        }

        cfg.num_hidden_layers = MAX_VISION_LAYERS;
        cfg.validate().unwrap();
        cfg.num_hidden_layers = 27;
        cfg.validate().unwrap();
    }

    #[test]
    fn config_enforces_dimension_caps_and_lower_bounds() {
        let mut cfg = valid_cfg();

        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
        cfg = valid_cfg();
        cfg.hidden_size = MAX_VISION_HIDDEN_SIZE;
        cfg.num_attention_heads = MAX_VISION_HEADS;
        cfg.validate().unwrap();
        cfg.hidden_size = MAX_VISION_HIDDEN_SIZE + 1;
        assert!(cfg.validate().is_err());

        cfg = valid_cfg();
        cfg.intermediate_size = 1;
        cfg.validate().unwrap();
        cfg.intermediate_size = MAX_VISION_INTERMEDIATE_SIZE;
        cfg.validate().unwrap();
        cfg.intermediate_size = MAX_VISION_INTERMEDIATE_SIZE + 1;
        assert!(cfg.validate().is_err());

        cfg = valid_cfg();
        cfg.hidden_size = 4;
        cfg.num_attention_heads = 1;
        cfg.validate().unwrap();
        cfg.num_attention_heads = MAX_VISION_HEADS;
        cfg.hidden_size = MAX_VISION_HIDDEN_SIZE;
        cfg.validate().unwrap();
        cfg.num_attention_heads = MAX_VISION_HEADS + 1;
        assert!(cfg.validate().is_err());

        cfg = valid_cfg();
        cfg.image_size = cfg.patch_size;
        cfg.validate().unwrap();
        cfg.image_size = MAX_VISION_POSITION_SIDE * cfg.patch_size;
        cfg.validate().unwrap();
        cfg.image_size = (MAX_VISION_POSITION_SIDE + 1) * cfg.patch_size;
        assert!(cfg.validate().is_err());

        cfg = valid_cfg();
        cfg.text_hidden_size = 1;
        cfg.validate().unwrap();
        cfg.text_hidden_size = MAX_VISION_TEXT_HIDDEN_SIZE;
        cfg.validate().unwrap();
        cfg.text_hidden_size = MAX_VISION_TEXT_HIDDEN_SIZE + 1;
        assert!(cfg.validate().is_err());

        cfg = valid_cfg();
        cfg.num_channels = 3;
        cfg.validate().unwrap();
        cfg.num_channels = 4;
        assert!(cfg.validate().is_err());
    }

    struct PanickingTensorSource;

    impl TensorSource for PanickingTensorSource {
        fn has_tensor(&mut self, _name: &str) -> Result<bool, InferenceError> {
            panic!("invalid configuration reached tensor source");
        }

        fn tensor_shape(&mut self, _name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
            panic!("invalid configuration reached tensor source");
        }

        fn get_f32_tensor_owned(
            &mut self,
            _name: &str,
        ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
            panic!("invalid configuration reached tensor source");
        }
    }

    #[test]
    fn loader_validates_before_reading_tensors() {
        let mut cfg = valid_cfg();
        cfg.num_hidden_layers = MAX_VISION_LAYERS + 1;
        let mut source = PanickingTensorSource;
        let error = PaddleOcrVisionWeights::load(&mut source, &cfg)
            .unwrap_err()
            .to_string();
        assert!(error.contains("MAX_VISION_LAYERS"), "{error}");
    }

    #[test]
    fn check_grid_rejects_quadratic_attention_blowup() {
        let cfg = valid_cfg();
        let error = check_grid(&cfg, 256, 256).unwrap_err().to_string();
        assert!(error.contains("MAX_VISION_TOKENS"));
    }

    #[test]
    fn check_grid_accepts_largest_legal_grid() {
        let cfg = valid_cfg();
        assert_eq!(
            check_grid(&cfg, 2, MAX_VISION_TOKENS / 2).unwrap(),
            MAX_VISION_TOKENS
        );
    }

    #[test]
    fn check_grid_rejects_overflow_and_unmerged_shapes() {
        let cfg = valid_cfg();
        assert!(check_grid(&cfg, usize::MAX, 2).is_err());
        assert!(check_grid(&cfg, 1, 2).is_err());
    }

    #[test]
    fn half_pixel_bilinear_identity_when_grid_matches_table() {
        // Sampling a 3x3 table onto a 3x3 grid must return the table rows exactly.
        let side = 3;
        let hidden = 2;
        let table: Vec<f32> = (0..side * side * hidden).map(|v| v as f32).collect();
        for h in 0..side {
            for w in 0..side {
                let mut out = vec![0.0f32; hidden];
                bilinear_pos_embed_half_pixel(&table, hidden, side, side, side, h, w, &mut out);
                let row = &table[(h * side + w) * hidden..(h * side + w + 1) * hidden];
                assert_eq!(out, row);
            }
        }
    }

    #[test]
    fn half_pixel_bilinear_upsample_matches_torch_convention() {
        // 2-wide table [0, 1] upsampled to 4: torch align_corners=False gives
        // [0, 0.25, 0.75, 1].
        let table = [0.0f32, 0.0, 1.0, 1.0]; // side 2, hidden 1, rows: (0,0)=0 (0,1)=0 (1,0)=1 (1,1)=1
        let expect = [0.0f32, 0.25, 0.75, 1.0];
        for (h, e) in expect.iter().enumerate() {
            let mut out = [0.0f32];
            bilinear_pos_embed_half_pixel(&table, 1, 2, 4, 1, h, 0, &mut out);
            assert!((out[0] - e).abs() < 1e-6, "row {h}: {} vs {e}", out[0]);
        }
    }
}
