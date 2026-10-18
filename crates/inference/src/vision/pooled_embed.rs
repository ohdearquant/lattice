//! Raw image bytes -> pooled, L2-normalized embedding: wires the existing
//! real Qwen3.5-0.8B vision pipeline (`qwen35_vit::{preprocess_qwen35_image,
//! qwen35_vit_forward}`, `qwen35_merger::qwen35_merger_forward`) into the
//! decoder-side pooling in [`crate::forward::cpu_f16`]
//! (`prefill_hidden_states_f16`, `embed_image_f16`, `PoolingStrategy`).
//!
//! ## Scope
//!
//! The scaffold this assembles — `[vision_start] + [image_pad; N] +
//! [vision_end] + tokenized(prompt)` — is a minimal, documented
//! approximation of the real HF chat template (no system role, no
//! `<think></think>` block, no BOS). It is sufficient to exercise correct
//! decoder injection and pooling; callers that need exact chat-template
//! parity should assemble their own `input_ids` (matching
//! `tests/fixtures/vision/input_ids.json`'s layout) and call
//! [`crate::forward::cpu_f16::embed_image_f16`] directly.
//!
//! Retrieval quality against the base Qwen3.5-0.8B *instruct* checkpoint is
//! unvalidated — see [`crate::forward::cpu_f16::PoolingStrategy`]'s doc
//! comment.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use super::VisionError;
use super::checkpoint::Qwen35VisionWeights;
use super::multimodal::Qwen35VisionRequest;
use super::qwen35_merger::qwen35_merger_forward;
use super::qwen35_vit::{GridThw, preprocess_qwen35_image, qwen35_vit_forward};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use super::qwen35_vit_metal::qwen35_vit_forward_metal;
use crate::error::InferenceError;
use crate::forward::cpu_f16::{PoolingStrategy, embed_image_f16};
use crate::model::qwen35_config::{Qwen35Config, VisionModelConfig};
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer;
use crate::weights::f16_weights::F16ModelWeights;

/// **Unstable**: the pooled-embedding prompt scaffold and pooling contract
/// may evolve before 1.0.
///
/// Encode `image_bytes` through the real Qwen3.5-0.8B vision pipeline
/// (preprocess -> ViT -> merger), assemble a minimal vision-prompt scaffold
/// around `prompt`, run decoder prefill, and return a pooled, L2-normalized
/// `[cfg.hidden_size]` embedding vector (2048 for the 0.8B checkpoint).
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if `cfg` has no `vision_config`,
/// `image_token_id`, `vision_start_token_id`, or `vision_end_token_id` (a
/// non-vision-language checkpoint); if `image_bytes` cannot be decoded or
/// its dimensions are not an exact multiple of `patch_size *
/// spatial_merge_size` (see [`preprocess_qwen35_image`]'s scope note); or if
/// the assembled request fails [`Qwen35VisionRequest::validate`].
pub fn embed_image_from_bytes_f16(
    weights: &F16ModelWeights,
    cfg: &Qwen35Config,
    vision_weights: &Qwen35VisionWeights,
    tokenizer: &BpeTokenizer,
    image_bytes: &[u8],
    prompt: &str,
    pooling: PoolingStrategy,
) -> Result<Vec<f32>, InferenceError> {
    let (vision_cfg, image_token_id, vision_start, vision_end) = require_vision_ids(cfg)?;

    let (pixel_values, grid) = preprocess_qwen35_image(image_bytes, vision_cfg)
        .map_err(|e| InferenceError::InvalidInput(format!("image preprocessing failed: {e}")))?;
    let pre_merger = qwen35_vit_forward(vision_weights, vision_cfg, &pixel_values, grid)
        .map_err(|e| InferenceError::InvalidInput(format!("ViT forward failed: {e}")))?;

    pool_from_pre_merger_hidden_states(
        weights,
        cfg,
        vision_weights,
        vision_cfg,
        tokenizer,
        grid,
        pre_merger,
        image_token_id,
        vision_start,
        vision_end,
        prompt,
        pooling,
    )
}

/// **Unstable**: the Metal-dispatch surface (this cfg-split shape, and the
/// runtime-probe fail-closed error) may evolve before 1.0.
///
/// Metal-dispatching sibling of [`embed_image_from_bytes_f16`]: runs the ViT
/// forward pass on the Metal GPU ([`qwen35_vit_forward_metal`]) instead of
/// the CPU, mirroring the serving path's CPU/Metal split
/// (`crate::serve::metal_worker::build_vision_request` dispatches
/// `qwen35_vit_forward_metal_with_cancel` then the CPU
/// `qwen35_merger_forward`). The merger and the decoder-side pooling both
/// stay CPU here too — only the ViT block loop moves to the GPU.
///
/// Three distinct "no GPU" regimes exist, and this function's contract only
/// covers the first two — the third is not a failure at all:
///
/// 1. **Build-level unavailable** (non-macOS, or this crate's `metal-gpu`
///    feature is off): the sibling definition of this function below (same
///    name, opposite `cfg` gate) fails closed with
///    [`InferenceError::UnsupportedModel`] before any work — Metal code
///    simply is not compiled into that build.
/// 2. **Runtime device unavailable** (macOS + `metal-gpu` compiled in, but no
///    Metal device could be initialized on this machine, e.g. no GPU or
///    `MTLCreateSystemDefaultDevice` returned null): this function probes
///    [`crate::forward::metal_gemm::is_available`] up front and fails closed
///    with the same [`InferenceError::UnsupportedModel`] before touching the
///    ViT weights or doing any preprocessing.
/// 3. **Per-GEMM below `GPU_DISPATCH_THRESHOLD`** (an individual matrix in
///    the ViT forward is too small to be worth a GPU launch): this is NOT an
///    availability failure. `crate::forward::metal_gemm::metal_matmul`/
///    `metal_matmul_bt` silently run the equivalent CPU dot product for that
///    one GEMM call, by design (a dispatch-threshold optimization internal
///    to the Metal path, not a fallback this function's caller can observe
///    or needs to handle) — the ViT forward as a whole still ran via this
///    Metal entry point, it just used CPU math for its smallest matrices.
///
/// # Errors
///
/// Returns [`InferenceError::UnsupportedModel`] for either "no GPU" regime
/// above — this entry never falls back to the CPU ViT forward silently; a
/// caller that wants a fallback must catch this variant itself and call
/// [`embed_image_from_bytes_f16`]. Also maps a
/// [`VisionError::InvalidConfig`] surfaced from the underlying
/// [`qwen35_vit_forward_metal`] call to the same
/// [`InferenceError::UnsupportedModel`], as defense in depth — this should
/// not be reachable once the `is_available` probe above has already passed,
/// but is kept in case a future change to `qwen35_vit_forward_metal`
/// reintroduces an availability-shaped failure inside it; it is not itself
/// the availability guard. See [`embed_image_from_bytes_f16`]'s docs for the
/// remaining error conditions (missing vision config, undecodable or
/// misaligned image, invalid assembled request), which are identical here.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn embed_image_from_bytes_f16_metal(
    weights: &F16ModelWeights,
    cfg: &Qwen35Config,
    vision_weights: &Qwen35VisionWeights,
    tokenizer: &BpeTokenizer,
    image_bytes: &[u8],
    prompt: &str,
    pooling: PoolingStrategy,
) -> Result<Vec<f32>, InferenceError> {
    if !crate::forward::metal_gemm::is_available() {
        return Err(InferenceError::UnsupportedModel(
            "embed_image_from_bytes_f16_metal: this build supports metal-gpu, but no Metal \
             device could be initialized on this machine"
                .into(),
        ));
    }

    let (vision_cfg, image_token_id, vision_start, vision_end) = require_vision_ids(cfg)?;

    let (pixel_values, grid) = preprocess_qwen35_image(image_bytes, vision_cfg)
        .map_err(|e| InferenceError::InvalidInput(format!("image preprocessing failed: {e}")))?;
    let pre_merger = qwen35_vit_forward_metal(vision_weights, vision_cfg, &pixel_values, grid)
        .map_err(|e| match e {
            // Defense in depth, not the availability guard (see doc comment
            // above) -- the `is_available` probe already ran before this call.
            VisionError::InvalidConfig(msg) => {
                InferenceError::UnsupportedModel(format!("Metal ViT forward unavailable: {msg}"))
            }
            other => InferenceError::InvalidInput(format!("Metal ViT forward failed: {other}")),
        })?;

    pool_from_pre_merger_hidden_states(
        weights,
        cfg,
        vision_weights,
        vision_cfg,
        tokenizer,
        grid,
        pre_merger,
        image_token_id,
        vision_start,
        vision_end,
        prompt,
        pooling,
    )
}

/// Build-level-unavailable sibling of the `metal-gpu` macOS definition above
/// (see its doc comment for the full three-regime contract): this crate was
/// built without Metal support at all (non-macOS, or the `metal-gpu` feature
/// is off), so this fails closed immediately, before touching any argument.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub fn embed_image_from_bytes_f16_metal(
    _weights: &F16ModelWeights,
    _cfg: &Qwen35Config,
    _vision_weights: &Qwen35VisionWeights,
    _tokenizer: &BpeTokenizer,
    _image_bytes: &[u8],
    _prompt: &str,
    _pooling: PoolingStrategy,
) -> Result<Vec<f32>, InferenceError> {
    Err(InferenceError::UnsupportedModel(
        "embed_image_from_bytes_f16_metal requires the metal-gpu feature on macOS".into(),
    ))
}

/// Shared `cfg.vision_config`/token-id extraction for both
/// [`embed_image_from_bytes_f16`] and [`embed_image_from_bytes_f16_metal`].
fn require_vision_ids(
    cfg: &Qwen35Config,
) -> Result<(&VisionModelConfig, u32, u32, u32), InferenceError> {
    let vision_cfg = cfg.vision_config.as_ref().ok_or_else(|| {
        InferenceError::InvalidInput(
            "checkpoint has no vision_config; embed_image_from_bytes_f16 requires a \
             vision-language checkpoint"
                .to_string(),
        )
    })?;
    let image_token_id = cfg
        .image_token_id
        .ok_or_else(|| InferenceError::InvalidInput("checkpoint has no image_token_id".into()))?;
    let vision_start = cfg.vision_start_token_id.ok_or_else(|| {
        InferenceError::InvalidInput("checkpoint has no vision_start_token_id".into())
    })?;
    let vision_end = cfg.vision_end_token_id.ok_or_else(|| {
        InferenceError::InvalidInput("checkpoint has no vision_end_token_id".into())
    })?;
    Ok((vision_cfg, image_token_id, vision_start, vision_end))
}

/// Shared merger + decoder-prompt-scaffold + pooled-decoder-prefill tail,
/// common to the CPU and Metal ViT entry points above — the only thing that
/// differs between them is which forward produced `pre_merger`.
#[allow(clippy::too_many_arguments)]
fn pool_from_pre_merger_hidden_states(
    weights: &F16ModelWeights,
    cfg: &Qwen35Config,
    vision_weights: &Qwen35VisionWeights,
    vision_cfg: &VisionModelConfig,
    tokenizer: &BpeTokenizer,
    grid: GridThw,
    pre_merger: Vec<f32>,
    image_token_id: u32,
    vision_start: u32,
    vision_end: u32,
    prompt: &str,
    pooling: PoolingStrategy,
) -> Result<Vec<f32>, InferenceError> {
    let post_merger = qwen35_merger_forward(&vision_weights.merger, vision_cfg, &pre_merger)
        .map_err(|e| InferenceError::InvalidInput(format!("merger forward failed: {e}")))?;

    let merge_sq = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size;
    if merge_sq == 0 || !grid.num_patches().is_multiple_of(merge_sq) {
        return Err(InferenceError::InvalidInput(format!(
            "image grid {grid:?} patch count is not a multiple of spatial_merge_size^2"
        )));
    }
    let num_pads = grid.num_patches() / merge_sq;

    let text_ids = {
        let out = tokenizer.tokenize(prompt);
        out.input_ids[..out.real_length].to_vec()
    };

    let mut input_ids = Vec::with_capacity(2 + num_pads + text_ids.len());
    input_ids.push(vision_start);
    input_ids.extend(std::iter::repeat_n(image_token_id, num_pads));
    input_ids.push(vision_end);
    input_ids.extend(text_ids);

    let request = Qwen35VisionRequest {
        input_ids,
        image_grids: vec![grid],
        post_merger_rows: post_merger,
        image_token_id,
        spatial_merge_size: vision_cfg.spatial_merge_size,
        decoder_hidden_size: cfg.hidden_size,
    };

    embed_image_f16(weights, cfg, &request, pooling)
}

#[cfg(test)]
mod tests;
