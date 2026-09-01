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

    /// Reject shapes this reference cannot run: heads must divide the hidden
    /// size, `head_dim` must be divisible by 4 (two RoPE axes, each with
    /// paired frequencies), the position table must be square in patches,
    /// and the projector's merge kernel is fixed at 2x2 by the reference.
    ///
    /// # Errors
    ///
    /// [`InferenceError::Inference`] naming the violated invariant.
    pub fn validate(&self) -> Result<(), InferenceError> {
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
        // (384 // 14 = 27, discarding the 6-pixel remainder), so only a zero
        // side is invalid here.
        if self.patch_size == 0 || self.image_size / self.patch_size == 0 {
            return Err(InferenceError::Inference(format!(
                "vision image_size {} smaller than patch_size {}",
                self.image_size, self.patch_size
            )));
        }
        if self.spatial_merge_size != 2 {
            return Err(InferenceError::Inference(format!(
                "projector merge kernel is fixed at 2x2 by the reference; got {}",
                self.spatial_merge_size
            )));
        }
        if self.num_hidden_layers == 0 || self.num_channels == 0 || self.text_hidden_size == 0 {
            return Err(InferenceError::Inference(
                "vision config has a zero layer/channel/text-hidden count".into(),
            ));
        }
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
        let h = cfg.hidden_size;
        let p = cfg.patch_size;
        let side = cfg.pos_table_side();
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
            &[side * side, h],
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
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
        let merged = cfg.merged_width();
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
fn build_pos_embed_and_rope_tables(
    weights: &PaddleOcrVisionWeights,
    cfg: &PaddleOcrVisionConfig,
    grid_h: usize,
    grid_w: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = cfg.hidden_size;
    let n = grid_h * grid_w;
    let side = cfg.pos_table_side();
    let head_dim = cfg.head_dim();
    let rope_dim = head_dim / 2;
    let rope_half = rope_dim / 2;
    let inv_freq: Vec<f32> = (0..rope_half)
        .map(|i| 1.0 / VISION_ROPE_THETA.powf((2 * i) as f32 / rope_dim as f32))
        .collect();

    let mut pos = vec![0.0f32; n * hidden];
    let mut cos_t = vec![0.0f32; n * head_dim];
    let mut sin_t = vec![0.0f32; n * head_dim];
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
    (pos, cos_t, sin_t)
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

/// Encoder + projector over one image's raster-ordered patches.
///
/// `pixel_values` is `[grid_h * grid_w, channels * patch * patch]`, each
/// patch flattened `(channel, row, col)` after the reference's
/// rescale-and-normalize preprocessing.
///
/// # Errors
///
/// [`InferenceError::Inference`] if the grid is empty, not a multiple of the
/// merge kernel on either axis, or `pixel_values` has the wrong length.
pub fn paddleocr_vision_forward_trace(
    weights: &PaddleOcrVisionWeights,
    cfg: &PaddleOcrVisionConfig,
    pixel_values: &[f32],
    grid_h: usize,
    grid_w: usize,
) -> Result<PaddleOcrVisionTrace, InferenceError> {
    let m = cfg.spatial_merge_size;
    if grid_h == 0 || grid_w == 0 || !grid_h.is_multiple_of(m) || !grid_w.is_multiple_of(m) {
        return Err(InferenceError::Inference(format!(
            "vision grid {grid_h}x{grid_w} must be non-empty and a multiple of the {m}x{m} merge kernel"
        )));
    }
    let n = grid_h * grid_w;
    let hidden = cfg.hidden_size;
    let patch_len = cfg.patch_len();
    if pixel_values.len() != n * patch_len {
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
    let (pos, cos_t, sin_t) = build_pos_embed_and_rope_tables(weights, cfg, grid_h, grid_w);
    for (x, p) in hidden_states.iter_mut().zip(&pos) {
        *x += p;
    }
    let embed = hidden_states.clone();

    let head_dim = cfg.head_dim();
    let n_heads = cfg.num_attention_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let eps = cfg.layer_norm_eps;
    let mut layer_outputs = Vec::with_capacity(cfg.num_hidden_layers);

    for layer in &weights.layers {
        let mut normed = hidden_states.clone();
        layer_norm_rows(&mut normed, &layer.ln1_weight, &layer.ln1_bias, eps);
        let mut q = batch_matvec(&layer.q_weight, &normed, n, hidden, hidden);
        let mut k = batch_matvec(&layer.k_weight, &normed, n, hidden, hidden);
        let mut v = batch_matvec(&layer.v_weight, &normed, n, hidden, hidden);
        add_bias_rows(&mut q, &layer.q_bias);
        add_bias_rows(&mut k, &layer.k_bias);
        add_bias_rows(&mut v, &layer.v_bias);

        // Fused `[n, Q | K | V]` layout for the shared attention kernel, RoPE
        // applied per head on Q and K while packing.
        let mut qkv = vec![0.0f32; n * 3 * hidden];
        for i in 0..n {
            let cos_row = &cos_t[i * head_dim..(i + 1) * head_dim];
            let sin_row = &sin_t[i * head_dim..(i + 1) * head_dim];
            let dst = &mut qkv[i * 3 * hidden..(i + 1) * 3 * hidden];
            dst[..hidden].copy_from_slice(&q[i * hidden..(i + 1) * hidden]);
            dst[hidden..2 * hidden].copy_from_slice(&k[i * hidden..(i + 1) * hidden]);
            dst[2 * hidden..].copy_from_slice(&v[i * hidden..(i + 1) * hidden]);
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

        let mut normed = hidden_states.clone();
        layer_norm_rows(&mut normed, &layer.ln2_weight, &layer.ln2_bias, eps);
        let inter = cfg.intermediate_size;
        let mut fc1 = batch_matvec(&layer.fc1_weight, &normed, n, inter, hidden);
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
        layer_outputs.push(hidden_states.clone());
    }

    let mut post = hidden_states;
    layer_norm_rows(
        &mut post,
        &weights.post_ln_weight,
        &weights.post_ln_bias,
        eps,
    );
    let projector = project_merged(weights, cfg, &post, grid_h, grid_w);

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
) -> Vec<f32> {
    let hidden = cfg.hidden_size;
    let m = cfg.spatial_merge_size;
    let merged = cfg.merged_width();
    let mut normed = features.to_vec();
    layer_norm_rows(
        &mut normed,
        &weights.proj_norm_weight,
        &weights.proj_norm_bias,
        PROJECTOR_NORM_EPS,
    );
    let blocks_h = grid_h / m;
    let blocks_w = grid_w / m;
    let nb = blocks_h * blocks_w;
    let mut packed = vec![0.0f32; nb * merged];
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
    for row in l1.chunks_mut(merged) {
        for (x, b) in row.iter_mut().zip(&weights.proj_l1_bias) {
            *x = gelu_exact(*x + b);
        }
    }
    let t = cfg.text_hidden_size;
    let mut out = batch_matvec(&weights.proj_l2_weight, &l1, nb, t, merged);
    add_bias_rows(&mut out, &weights.proj_l2_bias);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: &str = r#"{"hidden_size": 1024, "vision_config": {"hidden_size": 1152,
        "intermediate_size": 4304, "num_hidden_layers": 27, "num_attention_heads": 16,
        "patch_size": 14, "image_size": 384, "layer_norm_eps": 1e-06, "spatial_merge_size": 2}}"#;

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
