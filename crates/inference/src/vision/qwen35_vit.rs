//! Real Qwen3.5-0.8B ViT forward pass (ADR-069 S3a — CPU reference; the
//! Metal port is a separate S3b fast-follow gated against this module):
//! depth-12 / hidden-768 encoder over the checkpoint's `model.visual.*`
//! weights ([`super::checkpoint::Qwen35VisionWeights`]), producing the
//! pre-merger hidden states `[num_patches, hidden_size]`.
//!
//! Deliberately independent from `vit.rs`'s ADR-049 7B-scaffold `ViT` (fused
//! Q/K/V is a *single* `qkv` projection here vs. three separate ones there;
//! no per-head Q/K RMSNorm here (the real checkpoint's vision attention has
//! none — only the text decoder does); `LayerNorm` with bias (not the
//! decoder's bias-free RMSNorm); no windowed attention (the 0.8B tower's
//! `cu_seqlens` is a single segment per image — full attention); and a
//! bilinear-interpolated learned position-embedding table plus a 2-axis
//! (h, w) vision RoPE instead of the scaffold's from-scratch 2D RoPE). See
//! `checkpoint.rs`'s module docs for why the two paths are kept separate
//! rather than reconciled into one.
//!
//! Verified bit-for-bit against the HF reference implementation
//! (`transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionModel`,
//! `transformers.vision_utils.{get_vision_bilinear_indices_and_weights,
//! get_vision_position_ids, get_vision_cu_seqlens}`) via a differential
//! script run locally against the real `Qwen/Qwen3.5-0.8B` checkpoint before
//! this module was written (ADR-069 S3a; CLAUDE.md "Differential Test First").
//! The committed gate is `tests/vision_s3_vit_forward_test.rs`.

use super::VisionError;
use super::checkpoint::Qwen35VisionWeights;
use super::vit::{batch_matvec, gelu, layer_norm, softmax_inplace};
use crate::model::qwen35_config::VisionModelConfig;
use image::{DynamicImage, ImageReader};
use std::io::Cursor;

/// Per-channel rescale-then-normalize constants for the real Qwen3.5-0.8B
/// image processor (`Qwen2VLImageProcessor` defaults as actually configured
/// for this checkpoint: `image_mean = image_std = [0.5, 0.5, 0.5]`,
/// `rescale_factor = 1/255`) — verified against the checkpoint's fetched
/// `preprocessor_config.json` via the differential script referenced above.
/// NOT the ImageNet statistics `preprocess.rs` uses for the unrelated
/// ADR-049 7B scaffold.
const QWEN35_IMAGE_MEAN: f32 = 0.5;
const QWEN35_IMAGE_STD: f32 = 0.5;

/// Maximum accepted image width/height in pixels, enforced by
/// [`preprocess_qwen35_image`] before any pixel data is decoded (ADR-069 S6 review round 1
/// blocker). Nothing in `VisionModelConfig` bounds this naturally: `num_position_embeddings`
/// only sizes the learned position-embedding table, which [`build_pos_embed_and_rope_tables`]
/// bilinearly *interpolates* to whatever patch grid the input image produces, so it places no
/// ceiling on `GridThw::num_patches()`. Left unbounded, the serve layer's compressed-byte
/// clamps (which bound the encoded stream, not decoded pixels) let a small, highly-compressible
/// image decode to an arbitrarily large pixel grid, driving an unbounded `num_patches *
/// patch_len` allocation below and unbounded O(n^2) full-attention work in
/// `multihead_attention_full`. Chosen as a conservative serving ceiling rather than derived from
/// the checkpoint: 2048x2048 is exactly `64 * (patch_size=16 * spatial_merge_size=2)` for the
/// real 0.8B checkpoint's factor, i.e. a 128x128 = 16,384-patch grid pre-merge. This cap alone
/// is NOT the resource bound -- the O(n^2) full-attention cost bound is
/// [`DEFAULT_MAX_VISION_PATCHES`] below (round-2 review); this dimension cap remains as the
/// outer decoder-level limit the
/// `image` crate can enforce during header parsing.
const MAX_IMAGE_DIMENSION_PIXELS: u32 = 2048;

/// Default maximum accepted pre-merge patch count for one image, enforced from the
/// header-reported dimensions before any pixel data is decoded (ADR-069 S6 review round 2
/// blocker). The dimension cap above is not sufficient on its own because the ViT's full
/// attention materializes a per-head `scores[n, n]` f32 matrix (`multihead_attention_full` via
/// `gemm_bt`): at the 2048x2048 dimension boundary with the real checkpoint's `patch_size=16`,
/// `n = 128^2 = 16,384` and that single allocation is `16,384^2 * 4 = 1 GiB`, before compute.
///
/// PR #1021 review round 3 major finding: the score-allocation bound above (originally
/// `n = 4,096`, a 1024x1024 image) does not bound *wall-clock serving time* on the single
/// serialized Metal worker -- a real, end-to-end timed `qwen35_vit_forward_metal` pass at
/// `n = 4,096` on this development machine (Apple M2 Max) measured **78.8 s**, monopolizing the
/// worker for the whole request. Re-derived from a direct measurement of the same real
/// checkpoint's ViT forward at several patch counts (`cargo test -p lattice-inference --features
/// f16,metal-gpu,test-utils --lib -- <ad hoc timing harness>`, GPU-flock-serialized): `n=64` ->
/// 330.7 ms, `n=256` -> 1,431.0 ms, `n=576` -> 4,811.6 ms, `n=4,096` -> 78,882.7 ms (cost grows
/// worse than linearly but sub-quadratically across this range -- consistent with a
/// dispatch-count-bound linear component plus an O(n^2) full-attention component). `n = 256`
/// (a 256x256 image, patch_size=16 -- the same natural grid boundary the module's committed
/// golden fixture already uses) is the largest measured grid whose wall time (1.43 s) stays
/// under the ~2 s worst-case serving budget, rounded down from the next larger measured point
/// (576 patches, 4.81 s, well over budget). Overridable per-machine via
/// [`LATTICE_VISION_MAX_PATCHES_ENV`] without a rebuild.
const DEFAULT_MAX_VISION_PATCHES: usize = 256;

/// Environment variable overriding [`DEFAULT_MAX_VISION_PATCHES`] (PR #1021 review round 3): the
/// measured default above is specific to the development machine it was measured on, so an
/// operator serving on different hardware (or accepting a different latency/image-size
/// tradeoff) can retune the cap without a rebuild. A missing, unparseable, or zero value falls
/// back to the default.
const LATTICE_VISION_MAX_PATCHES_ENV: &str = "LATTICE_VISION_MAX_PATCHES";

/// Current serving-budget patch cap: [`LATTICE_VISION_MAX_PATCHES_ENV`] if set to a valid
/// positive integer, otherwise [`DEFAULT_MAX_VISION_PATCHES`]. Read fresh on every call
/// (cheap -- this guard runs once per image request, not per patch) so a changed environment
/// takes effect without restarting the process mid-test; production callers only ever observe
/// the value fixed at process start, since nothing in this codebase mutates its own environment
/// after startup.
fn max_vision_patches() -> usize {
    std::env::var(LATTICE_VISION_MAX_PATCHES_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_VISION_PATCHES)
}

/// The temporal/height/width patch-grid shape for one image (`grid_thw` in
/// the HF reference). `t` is always 1 for a still image (video is out of
/// scope — ADR-069 Deferred list).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridThw {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl GridThw {
    pub fn num_patches(&self) -> usize {
        self.t * self.h * self.w
    }
}

/// Decode + preprocess raw image bytes into the checkpoint's flattened,
/// temporal-patch-folded pixel tensor: `[num_patches, in_channels *
/// temporal_patch_size * patch_size * patch_size]`, row-major over patches in
/// block-major spatial-merge order (matching [`GridThw`] / the HF processor).
///
/// Scope limitation (ADR-069 S3a, matches the ADR's own "dynamic-resolution
/// tiling" deferral): this does NOT implement the HF processor's
/// `smart_resize` — it requires the input image's pixel dimensions to
/// already be exact multiples of `patch_size * spatial_merge_size` (the
/// fixed 256x256 golden image satisfies this: 256 = 16 * 2 * 8). Images that
/// need resizing are rejected with [`VisionError::InvalidConfig`].
///
/// # Errors
///
/// [`VisionError::ImageDecode`] if the bytes cannot be decoded.
/// [`VisionError::InvalidConfig`] if the decoded image's dimensions are not
/// exact multiples of `patch_size * spatial_merge_size`.
pub fn preprocess_qwen35_image(
    image_bytes: &[u8],
    cfg: &VisionModelConfig,
) -> Result<(Vec<f32>, GridThw), VisionError> {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIMENSION_PIXELS);
    limits.max_image_height = Some(MAX_IMAGE_DIMENSION_PIXELS);

    // Cheap header-only dimension probe -- `into_dimensions()` parses just enough of the
    // container (e.g. the PNG IHDR chunk) to report width/height, and with `limits` set both
    // the PNG and JPEG decoders (the only formats this crate enables) enforce
    // `max_image_width`/`max_image_height` at that same header-parse step, before any pixel
    // data is read. Rejects an oversized image (ADR-069 S6 review round 1 blocker) without
    // paying for full decode or the pixel-tensor allocation below.
    let mut dim_reader = ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .map_err(|e| VisionError::ImageDecode(format!("format detection failed: {e}")))?;
    dim_reader.limits(limits.clone());
    let (header_w, header_h) = match dim_reader.into_dimensions() {
        Ok(dims) => dims,
        Err(image::ImageError::Limits(e)) => {
            return Err(VisionError::DimensionsExceeded(format!(
                "image exceeds the {MAX_IMAGE_DIMENSION_PIXELS}x{MAX_IMAGE_DIMENSION_PIXELS}-pixel \
                 serving limit: {e}"
            )));
        }
        Err(e) => {
            return Err(VisionError::ImageDecode(format!(
                "header/dimension read failed: {e}"
            )));
        }
    };

    // Attention-budget guard, still header-only (ADR-069 S6 review round 2 blocker): the
    // dimension cap alone admits grids whose per-head `scores[n, n]` full-attention allocation
    // reaches 1 GiB (see DEFAULT_MAX_VISION_PATCHES docs). Conservative ceil-division so unaligned
    // dimensions (rejected later anyway) cannot round the estimate below the true patch count.
    // A zero patch_size falls through to the InvalidConfig rejection below.
    if cfg.patch_size > 0 {
        let est_patches = (header_w as usize)
            .div_ceil(cfg.patch_size)
            .saturating_mul((header_h as usize).div_ceil(cfg.patch_size));
        let max_patches = max_vision_patches();
        if est_patches > max_patches {
            return Err(VisionError::DimensionsExceeded(format!(
                "image {header_w}x{header_h} yields {est_patches} patches at \
                 patch_size={}, exceeding the {max_patches}-patch serving \
                 budget (full-attention cost is quadratic in patch count)",
                cfg.patch_size
            )));
        }
    }

    // Independent defense-in-depth: re-apply the same strict limits on the actual decode call
    // (not just the header probe above), in case a future format-specific quirk lets dimensions
    // slip past `into_dimensions()`'s check without also being caught here.
    let mut reader = ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .map_err(|e| VisionError::ImageDecode(format!("format detection failed: {e}")))?;
    reader.limits(limits);
    let img: DynamicImage = reader.decode().map_err(|e| match e {
        image::ImageError::Limits(e) => VisionError::DimensionsExceeded(format!(
            "image exceeds the {MAX_IMAGE_DIMENSION_PIXELS}x{MAX_IMAGE_DIMENSION_PIXELS}-pixel \
             serving limit: {e}"
        )),
        e => VisionError::ImageDecode(format!("decode failed: {e}")),
    })?;
    let rgb = img.into_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    let patch_size = cfg.patch_size;
    let merge = cfg.spatial_merge_size;
    let factor = patch_size * merge;
    if patch_size == 0 || merge == 0 || factor == 0 {
        return Err(VisionError::InvalidConfig(
            "patch_size and spatial_merge_size must be > 0".into(),
        ));
    }
    if !height.is_multiple_of(factor) || !width.is_multiple_of(factor) {
        return Err(VisionError::InvalidConfig(format!(
            "image {width}x{height} is not a multiple of patch_size*spatial_merge_size={factor} \
             (dynamic resize is out of scope for ADR-069 S3a)"
        )));
    }

    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    let grid = GridThw {
        t: 1,
        h: grid_h,
        w: grid_w,
    };

    let in_channels = cfg.in_channels;
    let temporal = cfg.temporal_patch_size;
    let patch_len = in_channels * temporal * patch_size * patch_size;
    let num_patches = grid.num_patches();
    // Defense in depth alongside the dimension guard above: checked arithmetic on the
    // pixel-tensor allocation size itself, so a config/grid combination that somehow slips past
    // the width/height limit still cannot drive an unchecked, attacker-influenced allocation.
    let alloc_len = num_patches.checked_mul(patch_len).ok_or_else(|| {
        VisionError::InvalidConfig(format!(
            "image patch grid {grid:?} with patch_len={patch_len} overflows the pixel-tensor \
             allocation size"
        ))
    })?;
    let mut out = vec![0.0f32; alloc_len];

    let blocks_h = grid_h / merge;
    let blocks_w = grid_w / merge;
    let mut patch_idx = 0usize;
    for block_row in 0..blocks_h {
        for block_col in 0..blocks_w {
            for sub_row in 0..merge {
                for sub_col in 0..merge {
                    let h = block_row * merge + sub_row;
                    let w = block_col * merge + sub_col;
                    let py = h * patch_size;
                    let px = w * patch_size;

                    let row = &mut out[patch_idx * patch_len..(patch_idx + 1) * patch_len];
                    let mut k = 0usize;
                    for c in 0..in_channels {
                        for _t in 0..temporal {
                            for dy in 0..patch_size {
                                for dx in 0..patch_size {
                                    let pixel = rgb.get_pixel((px + dx) as u32, (py + dy) as u32);
                                    let raw = pixel[c] as f32 / 255.0;
                                    row[k] = (raw - QWEN35_IMAGE_MEAN) / QWEN35_IMAGE_STD;
                                    k += 1;
                                }
                            }
                        }
                    }
                    patch_idx += 1;
                }
            }
        }
    }

    Ok((out, grid))
}

/// Bilinear-interpolate the learned `pos_embed` table (`[num_position_embeddings,
/// hidden]`, square `num_grid_per_side x num_grid_per_side`) at grid position
/// `(h, w)` (a fractional coordinate `h/(grid_h-1) * (side-1)`, matching
/// `torch.linspace(0, side-1, grid_h)` in the HF reference), accumulating into
/// `out` (`[hidden]`, zeroed by the caller).
///
/// `grid_h`/`grid_w` are the *unmerged* patch-grid dimensions (16 for the
/// golden fixture); `h_idx`/`w_idx` are this patch's position within that
/// grid (0..grid_h, 0..grid_w) — NOT the post-spatial-merge block index.
#[allow(clippy::too_many_arguments)]
fn bilinear_pos_embed(
    pos_embed: &[f32],
    hidden: usize,
    side: usize,
    grid_h: usize,
    grid_w: usize,
    h_idx: usize,
    w_idx: usize,
    out: &mut [f32],
) {
    let h_frac_pos = if grid_h > 1 {
        h_idx as f32 * (side - 1) as f32 / (grid_h - 1) as f32
    } else {
        0.0
    };
    let w_frac_pos = if grid_w > 1 {
        w_idx as f32 * (side - 1) as f32 / (grid_w - 1) as f32
    } else {
        0.0
    };

    let h_floor = h_frac_pos.floor() as usize;
    let w_floor = w_frac_pos.floor() as usize;
    let h_ceil = (h_floor + 1).min(side - 1);
    let w_ceil = (w_floor + 1).min(side - 1);
    let h_frac = h_frac_pos - h_floor as f32;
    let w_frac = w_frac_pos - w_floor as f32;

    let corners = [
        (h_floor, w_floor, (1.0 - h_frac) * (1.0 - w_frac)),
        (h_floor, w_ceil, (1.0 - h_frac) * w_frac),
        (h_ceil, w_floor, h_frac * (1.0 - w_frac)),
        (h_ceil, w_ceil, h_frac * w_frac),
    ];
    for (ch, cw, weight) in corners {
        if weight == 0.0 {
            continue;
        }
        let row_idx = ch * side + cw;
        let row = &pos_embed[row_idx * hidden..(row_idx + 1) * hidden];
        for (o, &v) in out.iter_mut().zip(row.iter()) {
            *o += v * weight;
        }
    }
}

/// Apply rotate-half RoPE to one `[head_dim]` vector in place, given
/// `cos`/`sin` each `[head_dim]` (matches `apply_rotary_pos_emb_vision` in
/// the HF reference: `x_embed = x * cos + rotate_half(x) * sin`, where
/// `rotate_half(x) = cat(-x[half:], x[:half])`).
///
/// `pub(crate)` so the S3b Metal port (`qwen35_vit_metal.rs`) can reuse this
/// exact function for its own RoPE application rather than re-deriving the
/// rotate-half convention (ADR-069 S3b's "zero independent convention
/// decisions" contract).
pub(crate) fn apply_rope_inplace(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    let mut rotated = vec![0.0f32; x.len()];
    for i in 0..half {
        rotated[i] = -x[half + i];
        rotated[half + i] = x[i];
    }
    for i in 0..x.len() {
        x[i] = x[i] * cos[i] + rotated[i] * sin[i];
    }
}

/// Build the per-patch bilinear-interpolated position-embedding contribution
/// (`[n * hidden]`, to be added into the patch-embedded hidden states) and
/// the per-patch 2-axis vision RoPE `cos`/`sin` tables (`[n * head_dim]`
/// each), from the same block-major (spatial-merge-block-outer) patch order
/// `preprocess_qwen35_image` produces. Factored out of [`qwen35_vit_forward`]
/// so the S3b Metal port can reuse this exact CPU setup logic (cheap,
/// deterministic table construction — not the heavy compute the Metal port
/// targets) instead of re-deriving the interpolation/RoPE conventions.
pub(crate) fn build_pos_embed_and_rope_tables(
    weights: &Qwen35VisionWeights,
    cfg: &VisionModelConfig,
    grid: GridThw,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = cfg.hidden_size;
    let n = grid.num_patches();
    let side = (cfg.num_position_embeddings as f64).sqrt().round() as usize;
    let merge = cfg.spatial_merge_size;
    let head_dim = hidden / cfg.num_heads;
    let rope_dim = head_dim / 2; // matches Qwen3_5VisionRotaryEmbedding(head_dim // 2)
    let rope_half = rope_dim / 2; // inv_freq has rope_dim/2 entries (arange(0, rope_dim, 2))
    let theta = 10_000.0_f32;
    let inv_freq: Vec<f32> = (0..rope_half)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / rope_dim as f32))
        .collect();

    let mut pos_embed_contrib = vec![0.0f32; n * hidden];
    let mut cos_table = vec![0.0f32; n * head_dim];
    let mut sin_table = vec![0.0f32; n * head_dim];

    let blocks_h = grid.h / merge;
    let blocks_w = grid.w / merge;
    let mut patch_idx = 0usize;
    for block_row in 0..blocks_h {
        for block_col in 0..blocks_w {
            for sub_row in 0..merge {
                for sub_col in 0..merge {
                    let h_idx = block_row * merge + sub_row;
                    let w_idx = block_col * merge + sub_col;

                    let pos_slice =
                        &mut pos_embed_contrib[patch_idx * hidden..(patch_idx + 1) * hidden];
                    bilinear_pos_embed(
                        &weights.pos_embed,
                        hidden,
                        side,
                        grid.h,
                        grid.w,
                        h_idx,
                        w_idx,
                        pos_slice,
                    );

                    // rotary = concat(h*inv_freq, w*inv_freq)  [rope_dim]
                    // emb = concat(rotary, rotary)             [head_dim]
                    let mut rotary = vec![0.0f32; rope_dim];
                    for i in 0..rope_half {
                        rotary[i] = h_idx as f32 * inv_freq[i];
                        rotary[rope_half + i] = w_idx as f32 * inv_freq[i];
                    }
                    let cos_row = &mut cos_table[patch_idx * head_dim..(patch_idx + 1) * head_dim];
                    let sin_row = &mut sin_table[patch_idx * head_dim..(patch_idx + 1) * head_dim];
                    for i in 0..rope_dim {
                        let (s, c) = rotary[i].sin_cos();
                        cos_row[i] = c;
                        cos_row[rope_dim + i] = c;
                        sin_row[i] = s;
                        sin_row[rope_dim + i] = s;
                    }

                    patch_idx += 1;
                }
            }
        }
    }
    debug_assert_eq!(patch_idx, n);

    (pos_embed_contrib, cos_table, sin_table)
}

/// Run the real Qwen3.5 ViT forward pass over one image's preprocessed pixel
/// tensor, producing the pre-merger hidden states `[num_patches, hidden_size]`
/// (row-major flat), matching HF's `Qwen3_5VisionModel.forward(...).last_hidden_state`
/// exactly (no post-block normalization is applied — the real checkpoint has
/// none; the merger's own `LayerNorm` is a separate S4-scope step).
///
/// # Errors
///
/// [`VisionError::ShapeMismatch`] if `pixel_values.len()` doesn't match
/// `grid.num_patches() * (in_channels * temporal_patch_size * patch_size^2)`.
pub fn qwen35_vit_forward(
    weights: &Qwen35VisionWeights,
    cfg: &VisionModelConfig,
    pixel_values: &[f32],
    grid: GridThw,
) -> Result<Vec<f32>, VisionError> {
    let hidden = cfg.hidden_size;
    let n = grid.num_patches();
    let patch_len = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    if pixel_values.len() != n * patch_len {
        return Err(VisionError::ShapeMismatch {
            expected: n * patch_len,
            actual: pixel_values.len(),
            context: "qwen35_vit_forward: pixel_values length".into(),
        });
    }

    // ---- Patch embedding: linear-equivalent of the checkpoint's Conv3d
    // (kernel size == input size, stride == kernel size, so it degenerates
    // to a per-patch matvec over the flattened [hidden, patch_len] weight). ----
    let mut hidden_states = batch_matvec(
        &weights.patch_embed_weight,
        pixel_values,
        n,
        hidden,
        patch_len,
    );
    for i in 0..n {
        for j in 0..hidden {
            hidden_states[i * hidden + j] += weights.patch_embed_bias[j];
        }
    }

    // ---- Bilinear-interpolated learned position embedding + per-patch (h, w)
    // grid coordinates for the 2-axis vision RoPE — both computed from the
    // same block-major (spatial-merge-block-outer) patch order that
    // `preprocess_qwen35_image` already produced, so no separate reorder
    // permutation is needed here (unlike the HF reference, which computes
    // patches in plain raster order and permutes afterward). ----
    let head_dim = hidden / cfg.num_heads;
    let (pos_embed_contrib, cos_table, sin_table) =
        build_pos_embed_and_rope_tables(weights, cfg, grid);
    for i in 0..n * hidden {
        hidden_states[i] += pos_embed_contrib[i];
    }

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let n_heads = cfg.num_heads;

    // ---- Transformer blocks: full (unwindowed) self-attention over all n
    // patches — the 0.8B checkpoint's `cu_seqlens` for a single image is one
    // segment `[0, n]` (see module docs), so there is no window/segment
    // boundary to respect. ----
    for block in &weights.blocks {
        // -- Attention sub-layer --
        let residual = hidden_states.clone();
        let mut normed = hidden_states.clone();
        for i in 0..n {
            layer_norm(
                &mut normed[i * hidden..(i + 1) * hidden],
                &block.norm1_weight,
                &block.norm1_bias,
                1e-6,
            );
        }

        let mut qkv = batch_matvec(&block.qkv_weight, &normed, n, 3 * hidden, hidden);
        for i in 0..n {
            for j in 0..3 * hidden {
                qkv[i * 3 * hidden + j] += block.qkv_bias[j];
            }
        }

        // Apply RoPE to Q and K in place, per head.
        for i in 0..n {
            let base = i * 3 * hidden;
            let cos_row = &cos_table[i * head_dim..(i + 1) * head_dim];
            let sin_row = &sin_table[i * head_dim..(i + 1) * head_dim];
            for h in 0..n_heads {
                let q = &mut qkv[base + h * head_dim..base + (h + 1) * head_dim];
                apply_rope_inplace(q, cos_row, sin_row);
                let k_base = base + hidden;
                let k = &mut qkv[k_base + h * head_dim..k_base + (h + 1) * head_dim];
                apply_rope_inplace(k, cos_row, sin_row);
            }
        }

        let attn_out = multihead_attention_full(&qkv, n, hidden, n_heads, head_dim, scale);
        let proj_out = batch_matvec(&block.proj_weight, &attn_out, n, hidden, hidden);
        for i in 0..n * hidden {
            hidden_states[i] = residual[i] + proj_out[i] + block.proj_bias[i % hidden];
        }

        // -- MLP sub-layer --
        let residual = hidden_states.clone();
        let mut normed = hidden_states.clone();
        for i in 0..n {
            layer_norm(
                &mut normed[i * hidden..(i + 1) * hidden],
                &block.norm2_weight,
                &block.norm2_bias,
                1e-6,
            );
        }

        let mlp_dim = block.fc1_bias.len();
        let mut fc1_out = batch_matvec(&block.fc1_weight, &normed, n, mlp_dim, hidden);
        for i in 0..n {
            for j in 0..mlp_dim {
                let idx = i * mlp_dim + j;
                fc1_out[idx] = gelu(fc1_out[idx] + block.fc1_bias[j]);
            }
        }
        let fc2_out = batch_matvec(&block.fc2_weight, &fc1_out, n, hidden, mlp_dim);
        for i in 0..n * hidden {
            hidden_states[i] = residual[i] + fc2_out[i] + block.fc2_bias[i % hidden];
        }
    }

    Ok(hidden_states)
}

/// Standard full multi-head self-attention (no causal mask, no windowing —
/// the entire `[n, ...]` sequence is one attention segment).
/// `qkv`: `[n, 3 * hidden]` with per-row layout `[Q(hidden) | K(hidden) | V(hidden)]`,
/// each of Q/K/V split into `n_heads` contiguous `head_dim`-wide chunks.
fn multihead_attention_full(
    qkv: &[f32],
    n: usize,
    hidden: usize,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n * hidden];

    for h in 0..n_heads {
        let mut q_h = vec![0.0f32; n * head_dim];
        let mut k_h = vec![0.0f32; n * head_dim];
        let mut v_h = vec![0.0f32; n * head_dim];
        for i in 0..n {
            let base = i * 3 * hidden;
            q_h[i * head_dim..(i + 1) * head_dim]
                .copy_from_slice(&qkv[base + h * head_dim..base + (h + 1) * head_dim]);
            k_h[i * head_dim..(i + 1) * head_dim].copy_from_slice(
                &qkv[base + hidden + h * head_dim..base + hidden + (h + 1) * head_dim],
            );
            v_h[i * head_dim..(i + 1) * head_dim].copy_from_slice(
                &qkv[base + 2 * hidden + h * head_dim..base + 2 * hidden + (h + 1) * head_dim],
            );
        }

        let mut scores = vec![0.0f32; n * n];
        for i in 0..n {
            let qi = &q_h[i * head_dim..(i + 1) * head_dim];
            for j in 0..n {
                let kj = &k_h[j * head_dim..(j + 1) * head_dim];
                let dot: f32 = qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
                scores[i * n + j] = dot * scale;
            }
        }
        for i in 0..n {
            softmax_inplace(&mut scores[i * n..(i + 1) * n]);
        }
        for i in 0..n {
            let attn_row = &scores[i * n..(i + 1) * n];
            for j in 0..head_dim {
                let mut acc = 0.0f32;
                for k in 0..n {
                    acc += attn_row[k] * v_h[k * head_dim + j];
                }
                out[i * hidden + h * head_dim + j] = acc;
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> VisionModelConfig {
        VisionModelConfig {
            depth: 1,
            hidden_size: 8,
            num_heads: 2,
            patch_size: 2,
            spatial_merge_size: 2,
            out_hidden_size: 8,
            temporal_patch_size: 1,
            num_position_embeddings: 16, // side = 4
            in_channels: 1,
            deepstack_visual_indexes: vec![],
        }
    }

    fn make_test_png(w: u32, h: u32) -> Vec<u8> {
        use image::RgbImage;
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = ((x + y) % 256) as u8;
                img.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    /// Builds an all-black RGB PNG at `w`x`h` -- highly compressible (a real-world encoder
    /// collapses uniform rows to a few bytes each), so this fixture stays tiny on disk/wire
    /// while still decoding to the full `w * h` pixel grid. Mirrors the external review's
    /// concrete construction: a small, standards-conformant PNG whose *decoded* dimensions are
    /// the actual attack surface, distinct from `make_test_png`'s gradient fill (which this
    /// guard test does not need).
    fn make_black_test_png(w: u32, h: u32) -> Vec<u8> {
        use image::RgbImage;
        use image::codecs::png::{CompressionType, FilterType, PngEncoder};
        let img = RgbImage::new(w, h); // zero-initialized == solid black
        let mut buf = Vec::new();
        // Default compression leaves an all-black 4064x4064 fixture at ~245 KiB;
        // Best gets it under the 64 KiB content-part clamp, which is the point
        // of the construction (small wire size, huge decoded dimensions).
        let encoder = PngEncoder::new_with_quality(
            std::io::Cursor::new(&mut buf),
            CompressionType::Best,
            FilterType::Adaptive,
        );
        img.write_with_encoder(encoder).unwrap();
        buf
    }

    /// ADR-069 S6 review round 1 blocker: a compressed-byte clamp at the HTTP boundary does not
    /// bound decoded pixel dimensions. This reproduces the reviewer's concrete construction (an
    /// all-black 4064x4064 RGB PNG, well under every byte clamp, 32-divisible so it would
    /// otherwise pass the patch-alignment check too) and asserts `preprocess_qwen35_image`
    /// rejects it with the dedicated `DimensionsExceeded` variant *before* ever reaching the
    /// `num_patches * patch_len` pixel-tensor allocation or the ViT forward pass -- proven by
    /// the fact that this test completes near-instantly rather than allocating/processing a
    /// ~396 MB tensor for a 4064x4064 grid.
    ///
    /// Mutation-sensitive: temporarily commenting out the dimension-limit block in
    /// `preprocess_qwen35_image` turns this from an instant `DimensionsExceeded` rejection into
    /// either an `Ok` (full decode + huge allocation succeeds) or, at minimum, a decode that no
    /// longer returns this error variant -- verified locally by reverting the guard and watching
    /// this assertion fail, then restoring it (not committed).
    #[test]
    fn preprocess_rejects_oversized_decoded_dimensions_before_allocation() {
        let cfg = tiny_cfg(); // factor = patch_size(2) * merge(2) = 4; 4064 % 4 == 0
        let png = make_black_test_png(4064, 4064);
        assert!(
            png.len() < 65_536,
            "fixture must itself be a realistic small-upload size (got {} bytes) -- the whole \
             point is that byte-size clamps alone cannot catch this",
            png.len()
        );

        let err = preprocess_qwen35_image(&png, &cfg).unwrap_err();
        assert!(
            matches!(err, VisionError::DimensionsExceeded(_)),
            "expected DimensionsExceeded, got {err:?}"
        );
    }

    /// The real 0.8B checkpoint's patch geometry, which the patch-budget
    /// guard's boundary numbers are quoted in (round-2 review blocker).
    fn real_geometry_cfg() -> VisionModelConfig {
        VisionModelConfig {
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            in_channels: 3,
            ..tiny_cfg()
        }
    }

    #[test]
    fn preprocess_rejects_attention_budget_at_admitted_dimension_boundary() {
        // Pinned to the unset-env default (see `with_max_patches_env` docs): this test's
        // boundary numbers are quoted against DEFAULT_MAX_VISION_PATCHES specifically, so it
        // must not observe a concurrently-running env-override test's temporary value.
        with_max_patches_env(None, || {
            // 2048x2048 passes the MAX_IMAGE_DIMENSION_PIXELS cap exactly, but at the
            // real patch_size=16 it is a 128x128 = 16,384-patch grid whose per-head
            // full-attention scores allocation is 16,384^2 * 4 = 1 GiB. The patch
            // budget must reject it from the header alone (round-2 review blocker).
            let cfg = real_geometry_cfg();
            let png = make_black_test_png(2048, 2048);
            assert!(
                png.len() < 65_536,
                "fixture must stay a realistic small-upload size (got {} bytes)",
                png.len()
            );
            let err = preprocess_qwen35_image(&png, &cfg).unwrap_err();
            assert!(
                matches!(err, VisionError::DimensionsExceeded(_)),
                "expected DimensionsExceeded from the patch budget, got {err:?}"
            );
        });
    }

    #[test]
    fn preprocess_accepts_largest_grid_within_attention_budget() {
        // Pinned to the unset-env default -- see the sibling test above for why.
        with_max_patches_env(None, || {
            // 256x256 at patch_size=16 is exactly DEFAULT_MAX_VISION_PATCHES = 256 pre-merge
            // patches (PR #1021 review round 3: re-measured from a real, end-to-end timed Metal
            // ViT forward on this development machine -- see DEFAULT_MAX_VISION_PATCHES's doc
            // comment for the measured numbers) -- the largest square grid the default budget
            // admits. Must preprocess successfully (not be rejected by either guard), proving
            // the budget boundary sits where the docs say.
            let cfg = real_geometry_cfg();
            let png = make_black_test_png(256, 256);
            let (patches, grid) =
                preprocess_qwen35_image(&png, &cfg).expect("boundary grid accepted");
            assert_eq!(grid, GridThw { t: 1, h: 16, w: 16 });
            assert_eq!(grid.num_patches(), DEFAULT_MAX_VISION_PATCHES);
            let patch_len =
                cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
            assert_eq!(patches.len(), DEFAULT_MAX_VISION_PATCHES * patch_len);
        });
    }

    /// Serializes every test below that mutates the real process environment
    /// (`LATTICE_VISION_MAX_PATCHES` is process-global -- `cargo test` runs this file's tests on
    /// multiple threads within one process, so unguarded concurrent `set_var`/`remove_var` calls
    /// would race and flake). Same pattern as `moe_expert_cache.rs`'s `ENV_TEST_LOCK`.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Sets (or clears) `LATTICE_VISION_MAX_PATCHES` for the duration of `f`, holding
    /// `ENV_TEST_LOCK` and restoring the prior value (including "was unset") on the way out,
    /// even if `f` panics.
    fn with_max_patches_env<R>(value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let _guard = ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var(LATTICE_VISION_MAX_PATCHES_ENV).ok();
        // SAFETY: serialized by `ENV_TEST_LOCK` above -- no concurrent reader/writer of this
        // process-global variable.
        unsafe {
            match value {
                Some(v) => std::env::set_var(LATTICE_VISION_MAX_PATCHES_ENV, v),
                None => std::env::remove_var(LATTICE_VISION_MAX_PATCHES_ENV),
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        // SAFETY: see above.
        unsafe {
            match &prior {
                Some(v) => std::env::set_var(LATTICE_VISION_MAX_PATCHES_ENV, v),
                None => std::env::remove_var(LATTICE_VISION_MAX_PATCHES_ENV),
            }
        }
        match result {
            Ok(r) => r,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    #[test]
    fn max_vision_patches_unset_env_falls_back_to_default() {
        with_max_patches_env(None, || {
            assert_eq!(max_vision_patches(), DEFAULT_MAX_VISION_PATCHES);
        });
    }

    /// PR #1021 review round 3 major finding: the patch cap must be configurable per-machine
    /// without a rebuild. Setting the env var below the default must make a request that the
    /// default would admit get rejected instead -- proving the override actually reaches the
    /// admission guard, not just the standalone `max_vision_patches()` accessor.
    #[test]
    fn preprocess_honors_lower_env_override_below_default() {
        with_max_patches_env(Some("64"), || {
            assert_eq!(max_vision_patches(), 64);
            let cfg = real_geometry_cfg();
            // 256x256 = 256 patches, which the DEFAULT_MAX_VISION_PATCHES=256 budget admits
            // (proved above) but a 64-patch override must reject.
            let png = make_black_test_png(256, 256);
            let err = preprocess_qwen35_image(&png, &cfg).unwrap_err();
            assert!(
                matches!(err, VisionError::DimensionsExceeded(_)),
                "expected the lowered env override to reject a default-admitted grid, got {err:?}"
            );
        });
    }

    #[test]
    fn max_vision_patches_malformed_or_zero_env_falls_back_to_default() {
        for bad in ["not-a-number", "0", ""] {
            with_max_patches_env(Some(bad), || {
                assert_eq!(
                    max_vision_patches(),
                    DEFAULT_MAX_VISION_PATCHES,
                    "env value {bad:?} must fall back to the default, not panic or admit \
                     unbounded patches"
                );
            });
        }
    }

    #[test]
    fn preprocess_rejects_misaligned_image() {
        let cfg = tiny_cfg(); // factor = patch_size(2) * merge(2) = 4
        let png = make_test_png(6, 4); // 6 not a multiple of 4
        let err = preprocess_qwen35_image(&png, &cfg).unwrap_err();
        assert!(matches!(err, VisionError::InvalidConfig(_)));
    }

    #[test]
    fn preprocess_produces_expected_shape_and_grid() {
        let cfg = tiny_cfg();
        let png = make_test_png(8, 8); // grid 4x4 patches of size 2
        let (patches, grid) = preprocess_qwen35_image(&png, &cfg).expect("preprocess");
        assert_eq!(grid, GridThw { t: 1, h: 4, w: 4 });
        let patch_len = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        assert_eq!(patches.len(), grid.num_patches() * patch_len);
        assert!(patches.iter().all(|v| v.is_finite()));
    }

    fn make_test_weights(cfg: &VisionModelConfig) -> Qwen35VisionWeights {
        use crate::vision::checkpoint::{VisualBlockWeights, VisualMergerWeights};
        let hidden = cfg.hidden_size;
        let patch_len = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let mlp_dim = 2 * hidden;
        let merge_in = cfg.spatial_merge_size * cfg.spatial_merge_size * hidden;

        // Deterministic pseudo-random weights (not all-zero/identity) so the
        // forward pass actually exercises every op, via a tiny xorshift LCG.
        let mut state = 0x1234_5678_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 0.2 - 0.1
        };
        let mut v = |n: usize| (0..n).map(|_| next()).collect::<Vec<f32>>();

        let block = VisualBlockWeights {
            qkv_weight: v(3 * hidden * hidden),
            qkv_bias: v(3 * hidden),
            proj_weight: v(hidden * hidden),
            proj_bias: v(hidden),
            fc1_weight: v(mlp_dim * hidden),
            fc1_bias: v(mlp_dim),
            fc2_weight: v(hidden * mlp_dim),
            fc2_bias: v(hidden),
            norm1_weight: vec![1.0; hidden],
            norm1_bias: vec![0.0; hidden],
            norm2_weight: vec![1.0; hidden],
            norm2_bias: vec![0.0; hidden],
        };

        Qwen35VisionWeights {
            patch_embed_weight: v(hidden * patch_len),
            patch_embed_weight_shape: vec![
                hidden,
                cfg.in_channels,
                cfg.temporal_patch_size,
                cfg.patch_size,
                cfg.patch_size,
            ],
            patch_embed_bias: v(hidden),
            pos_embed: v(cfg.num_position_embeddings * hidden),
            blocks: vec![block],
            merger: VisualMergerWeights {
                fc1_weight: v(merge_in * merge_in),
                fc1_bias: v(merge_in),
                fc2_weight: v(cfg.out_hidden_size * merge_in),
                fc2_bias: v(cfg.out_hidden_size),
                norm_weight: vec![1.0; hidden],
                norm_bias: vec![0.0; hidden],
            },
        }
    }

    #[test]
    fn vit_forward_output_shape_and_finite() {
        let cfg = tiny_cfg();
        let weights = make_test_weights(&cfg);
        let png = make_test_png(8, 8);
        let (pixel_values, grid) = preprocess_qwen35_image(&png, &cfg).expect("preprocess");

        let out = qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("forward");
        assert_eq!(out.len(), grid.num_patches() * cfg.hidden_size);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn vit_forward_rejects_pixel_length_mismatch() {
        let cfg = tiny_cfg();
        let weights = make_test_weights(&cfg);
        let grid = GridThw { t: 1, h: 4, w: 4 };
        let bad_pixels = vec![0.0f32; 3]; // way too short
        let err = qwen35_vit_forward(&weights, &cfg, &bad_pixels, grid).unwrap_err();
        assert!(matches!(err, VisionError::ShapeMismatch { .. }));
    }

    #[test]
    fn vit_forward_is_deterministic() {
        let cfg = tiny_cfg();
        let weights = make_test_weights(&cfg);
        let png = make_test_png(8, 8);
        let (pixel_values, grid) = preprocess_qwen35_image(&png, &cfg).expect("preprocess");

        let out1 = qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("forward 1");
        let out2 = qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("forward 2");
        assert_eq!(out1, out2);
    }

    /// Perturbing a single loaded weight must change the output — proves this
    /// forward pass is actually sensitive to the weights it's handed (the
    /// same mutation-sensitivity property the committed golden gate in
    /// `tests/vision_s3_vit_forward_test.rs` relies on).
    #[test]
    fn vit_forward_is_sensitive_to_weight_mutation() {
        let cfg = tiny_cfg();
        let mut weights = make_test_weights(&cfg);
        let png = make_test_png(8, 8);
        let (pixel_values, grid) = preprocess_qwen35_image(&png, &cfg).expect("preprocess");

        let baseline = qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("forward");
        weights.blocks[0].qkv_weight[0] += 5.0;
        let mutated = qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("forward");

        assert_ne!(
            baseline, mutated,
            "weight mutation had no effect on ViT output"
        );
    }
}
