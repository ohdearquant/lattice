//! Real Qwen3.5-0.8B patch-merger / projection stage (ADR-069 S4): maps the
//! S3a pre-merger hidden states `[num_patches, hidden_size]` to the
//! post-merger projected visual embeddings `[num_patches / spatial_merge_size^2,
//! out_hidden_size]` — the exact tensor injected into the decoder's residual
//! stream at image-token positions.
//!
//! Verified against the pinned HF reference implementation
//! (`transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionPatchMerger`,
//! `transformers==5.12.1`, same pin as S3a): `Qwen3_5VisionModel` always
//! constructs its merger with `use_postshuffle_norm=False`, so the op order
//! is `LayerNorm(hidden_size) -> fold merge_size^2 consecutive patches into
//! one row -> Linear(merge_in, merge_in) -> GELU(exact/erf) -> Linear(merge_in,
//! out_hidden_size)`. The fold is a plain reshape with no data movement
//! because `qwen35_vit_forward`'s input patches are already produced in
//! block-major (spatial-merge-block-outer) order by
//! `preprocess_qwen35_image` — the same convention this module's S3a sibling
//! relies on for its own position/RoPE indexing.
//!
//! The merger's activation is `nn.GELU()` (exact, erf-based) — NOT
//! `ACT2FN[config.hidden_act]` ("gelu_pytorch_tanh", the tanh approximation
//! `lattice_compute::vision::gelu` implements for the ViT block MLPs). This is a
//! hardcoded HF implementation detail independent of `hidden_act`; using the
//! tanh approximation here would be a convention mismatch, not just added
//! imprecision.
//!
//! The committed gate is `tests/vision_s4_merger_test.rs`.

use super::VisionError;
use super::checkpoint::VisualMergerWeights;
use lattice_arch::qwen35_config::VisionModelConfig;
use lattice_compute::vision::{gelu_exact, layer_norm};

/// Run the real Qwen3.5 patch-merger forward pass over one image's S3a
/// pre-merger hidden states, producing the post-merger projected visual
/// embeddings `[num_patches / spatial_merge_size^2, out_hidden_size]`
/// (row-major flat) — matching HF's `Qwen3_5VisionModel.forward(...).pooler_output`
/// exactly.
///
/// # Errors
///
/// [`VisionError::ShapeMismatch`] if `pre_merger_hidden.len()` is not a
/// multiple of `hidden_size`, or the patch count is not a multiple of
/// `spatial_merge_size^2`.
pub fn qwen35_merger_forward(
    weights: &VisualMergerWeights,
    cfg: &VisionModelConfig,
    pre_merger_hidden: &[f32],
) -> Result<Vec<f32>, VisionError> {
    let mut never_cancel = || false;
    match qwen35_merger_forward_with_cancel(weights, cfg, pre_merger_hidden, &mut never_cancel)? {
        Some(output) => Ok(output),
        None => Err(VisionError::InvalidConfig(
            "non-cancellable vision merger was cancelled".to_string(),
        )),
    }
}

fn batch_matvec_with_cancel(
    weights: &[f32],
    input: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Option<Vec<f32>>, VisionError> {
    let output_len = batch.checked_mul(rows).ok_or_else(|| {
        VisionError::InvalidConfig(
            "qwen35_merger_forward: matrix output shape overflow".to_string(),
        )
    })?;
    let mut output = vec![0.0f32; output_len];
    lattice_compute::forward::cpu::validate_gemm_bt(
        input.len(),
        weights.len(),
        output.len(),
        batch,
        cols,
        rows,
        "qwen35_merger_forward",
    );
    for batch_index in 0..batch {
        if should_cancel() {
            return Ok(None);
        }
        let input_row = &input[batch_index * cols..(batch_index + 1) * cols];
        let output_row = &mut output[batch_index * rows..(batch_index + 1) * rows];
        for row in 0..rows {
            if row.is_multiple_of(16) && should_cancel() {
                return Ok(None);
            }
            let weight_row = &weights[row * cols..(row + 1) * cols];
            let mut accumulator = 0.0f32;
            for (weight, value) in weight_row.iter().zip(input_row) {
                accumulator += weight * value;
            }
            output_row[row] = accumulator;
        }
    }
    Ok(Some(output))
}

/// **Unstable**: cancellation-aware merger execution may evolve before 1.0.
#[doc(hidden)]
pub fn qwen35_merger_forward_with_cancel(
    weights: &VisualMergerWeights,
    cfg: &VisionModelConfig,
    pre_merger_hidden: &[f32],
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Option<Vec<f32>>, VisionError> {
    if should_cancel() {
        return Ok(None);
    }
    let hidden = cfg.hidden_size;
    if hidden == 0 || !pre_merger_hidden.len().is_multiple_of(hidden) {
        return Err(VisionError::ShapeMismatch {
            expected: 0,
            actual: pre_merger_hidden.len(),
            context: "qwen35_merger_forward: pre_merger_hidden length must be a multiple of \
                      hidden_size"
                .into(),
        });
    }
    let n = pre_merger_hidden.len() / hidden;
    let merge_sq = cfg.spatial_merge_size * cfg.spatial_merge_size;
    if merge_sq == 0 || !n.is_multiple_of(merge_sq) {
        return Err(VisionError::ShapeMismatch {
            expected: 0,
            actual: n,
            context: "qwen35_merger_forward: patch count must be a multiple of \
                      spatial_merge_size^2"
                .into(),
        });
    }

    // ---- LayerNorm(hidden_size), applied per-patch BEFORE the spatial fold
    // (use_postshuffle_norm=False in the HF reference: the norm's own
    // normalized_shape is hidden_size, not merge_in). ----
    let mut normed = pre_merger_hidden.to_vec();
    for i in 0..n {
        if should_cancel() {
            return Ok(None);
        }
        layer_norm(
            &mut normed[i * hidden..(i + 1) * hidden],
            &weights.norm_weight,
            &weights.norm_bias,
            1e-6,
        );
    }

    // ---- Spatial fold: `x.view(-1, merge_in)` in the HF reference. Purely a
    // reinterpretation of the flat row-major buffer -- every merge_sq
    // consecutive patch rows already sit contiguously in block-major order
    // (see module docs), so no permutation is needed here. ----
    let merge_in = merge_sq * hidden;
    let num_visual_tokens = n / merge_sq;

    // ---- linear_fc1 (merge_in -> merge_in) + exact GELU + linear_fc2
    // (merge_in -> out_hidden_size). ----
    let Some(mut fc1_out) = batch_matvec_with_cancel(
        &weights.fc1_weight,
        &normed,
        num_visual_tokens,
        merge_in,
        merge_in,
        should_cancel,
    )?
    else {
        return Ok(None);
    };
    for i in 0..num_visual_tokens {
        if should_cancel() {
            return Ok(None);
        }
        for j in 0..merge_in {
            let idx = i * merge_in + j;
            fc1_out[idx] = gelu_exact(fc1_out[idx] + weights.fc1_bias[j]);
        }
    }

    let out_hidden = cfg.out_hidden_size;
    let Some(mut out) = batch_matvec_with_cancel(
        &weights.fc2_weight,
        &fc1_out,
        num_visual_tokens,
        out_hidden,
        merge_in,
        should_cancel,
    )?
    else {
        return Ok(None);
    };
    for i in 0..num_visual_tokens {
        if should_cancel() {
            return Ok(None);
        }
        for j in 0..out_hidden {
            out[i * out_hidden + j] += weights.fc2_bias[j];
        }
    }

    if should_cancel() {
        return Ok(None);
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests;
