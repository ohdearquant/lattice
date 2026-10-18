//! Multimodal input type and the `generate_multimodal` integration point.
//!
//! `MultimodalInput` carries:
//! - `patch_embeddings`: projected visual token vectors `[visual_tokens, d_model]`
//! - `text_tokens`: text token IDs that follow the image
//!
//! The integration with `MetalQwen35State` prepends the visual token embeddings
//! to the text token embeddings before calling the existing decode loop.
//! Position IDs for text tokens are offset by `visual_tokens` so that patch
//! positions `[0, visual_tokens)` are reserved for the image.
//!
//! [`Qwen35VisionRequest`] below is a separate, Qwen3.5-specific request type
//! (ADR-069 Stage 5b) for the case `MultimodalInput`'s visual-prefix layout
//! cannot represent: an expanded, original-order token stream of the form
//! `text + vision_start + image_pads + vision_end + text`, where visual rows
//! replace only the `<|image_pad|>` slots and the delimiters keep their
//! normal text embeddings (HF's `masked_scatter` contract, ADR-069 RECON §1).

/// A multimodal prompt: merged visual patch embeddings prepended to text tokens.
///
/// Constructed by callers who have a `VisionOutput` and a tokenized text prompt:
/// ```no_run
/// use lattice_inference::vision::{VisionOutput, MultimodalInput};
///
/// fn build_multimodal(vision_out: VisionOutput, text_ids: Vec<u32>) -> MultimodalInput {
///     MultimodalInput {
///         patch_embeddings: vision_out.patch_embeddings,
///         raw_patches: vision_out.raw_patches,
///         visual_tokens: vision_out.visual_tokens,
///         d_model: vision_out.d_model,
///         text_tokens: text_ids,
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MultimodalInput {
    /// Projected patch embeddings: `[visual_tokens * d_model]` flat f32, row-major.
    ///
    /// Index into this with `[token_idx * d_model .. (token_idx+1) * d_model]`.
    pub patch_embeddings: Vec<f32>,
    /// Number of raw ViT patches before spatial merge (e.g., 784 for 448/16).
    pub raw_patches: usize,
    /// Number of visual tokens after spatial merge (e.g., 196 for merge_size=2).
    pub visual_tokens: usize,
    /// Decoder hidden dimension (must match `MetalQwen35State.config.hidden_size`).
    pub d_model: usize,
    /// Text token IDs that follow the image in the prompt.
    pub text_tokens: Vec<u32>,
}

impl MultimodalInput {
    /// Validate that field dimensions are internally consistent.
    pub fn validate(&self) -> Result<(), crate::vision::VisionError> {
        if self.d_model == 0 {
            return Err(crate::vision::VisionError::InvalidConfig(
                "d_model must be > 0".into(),
            ));
        }
        let expected_len = self.visual_tokens * self.d_model;
        if self.patch_embeddings.len() != expected_len {
            return Err(crate::vision::VisionError::ShapeMismatch {
                expected: expected_len,
                actual: self.patch_embeddings.len(),
                context: "patch_embeddings length must equal visual_tokens * d_model".into(),
            });
        }
        if self.visual_tokens == 0 {
            return Err(crate::vision::VisionError::InvalidConfig(
                "visual_tokens must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Total sequence length: visual tokens + text tokens.
    pub fn total_sequence_len(&self) -> usize {
        self.visual_tokens + self.text_tokens.len()
    }
}

use crate::vision::qwen35_vit::GridThw;

/// A Qwen3.5 multimodal decode request carrying the expanded, original-order
/// token stream and the post-merger visual rows to inject at `<|image_pad|>`
/// slots (ADR-069 Stage 5b). Unlike [`MultimodalInput`], text and image
/// tokens are not split into separate fields: `input_ids` is the exact
/// physical sequence the decoder consumes, in HF processor order, so that
/// vision delimiters (`vision_start_token_id` / `vision_end_token_id`) and
/// interleaved text spans are representable.
#[derive(Debug, Clone)]
pub struct Qwen35VisionRequest {
    /// Expanded, original-order token ids: text + delimiters + one
    /// `image_token_id` entry per post-merger visual row, per image run.
    pub input_ids: Vec<u32>,
    /// Unmerged `(T, H, W)` patch-grid shape for each image run, in the
    /// order those runs appear in `input_ids`.
    pub image_grids: Vec<GridThw>,
    /// Post-merger visual rows for all images concatenated in image order,
    /// flat row-major `[total_post_merger_rows * decoder_hidden_size]`.
    pub post_merger_rows: Vec<f32>,
    /// The `<|image_pad|>` token id (checkpoint-specific, e.g. 248056).
    pub image_token_id: u32,
    /// Spatial merge size `m` (checkpoint vision config, e.g. 2).
    pub spatial_merge_size: usize,
    /// Decoder hidden size that each post-merger row must match (e.g. 1024).
    pub decoder_hidden_size: usize,
}

impl Qwen35VisionRequest {
    /// Number of `image_token_id` occurrences in `input_ids`.
    pub fn image_pad_count(&self) -> usize {
        self.input_ids
            .iter()
            .filter(|&&id| id == self.image_token_id)
            .count()
    }

    /// Validate every count/shape/finiteness invariant this request depends
    /// on before it reaches the decoder. Fails closed (never panics) on:
    /// zero dims, arithmetic overflow, non-finite floats, empty input, and
    /// any mismatch between the image-pad count, the grid list, and the
    /// post-merger row buffer.
    pub fn validate(&self) -> Result<(), crate::vision::VisionError> {
        use crate::vision::VisionError;

        if self.decoder_hidden_size == 0 {
            return Err(VisionError::InvalidConfig(
                "decoder_hidden_size must be > 0".into(),
            ));
        }
        if self.spatial_merge_size == 0 {
            return Err(VisionError::InvalidConfig(
                "spatial_merge_size must be > 0".into(),
            ));
        }
        if self.input_ids.is_empty() {
            return Err(VisionError::InvalidConfig(
                "input_ids must not be empty".into(),
            ));
        }

        let merge_sq = self
            .spatial_merge_size
            .checked_mul(self.spatial_merge_size)
            .ok_or_else(|| VisionError::InvalidConfig("spatial_merge_size^2 overflowed".into()))?;

        // Count of contiguous image-pad runs, so a grid-list length
        // mismatch (one run needs exactly one grid) is caught even when the
        // total pad count happens to still line up numerically.
        let mut image_runs = 0usize;
        let mut prev_was_pad = false;
        for &id in &self.input_ids {
            let is_pad = id == self.image_token_id;
            if is_pad && !prev_was_pad {
                image_runs += 1;
            }
            prev_was_pad = is_pad;
        }
        if image_runs != self.image_grids.len() {
            return Err(VisionError::ShapeMismatch {
                expected: self.image_grids.len(),
                actual: image_runs,
                context: "image_grids length must equal the number of image-pad runs in \
                          input_ids"
                    .into(),
            });
        }

        let mut expected_pad_count = 0usize;
        let mut expected_rows = 0usize;
        for grid in &self.image_grids {
            if grid.t == 0 || grid.h == 0 || grid.w == 0 {
                return Err(VisionError::InvalidConfig(format!(
                    "image grid has a zero dimension: {grid:?}"
                )));
            }
            if grid.h % self.spatial_merge_size != 0 || grid.w % self.spatial_merge_size != 0 {
                return Err(VisionError::InvalidConfig(format!(
                    "image grid {grid:?} is not divisible by spatial_merge_size={}",
                    self.spatial_merge_size
                )));
            }
            let num_patches: usize = grid
                .t
                .checked_mul(grid.h)
                .and_then(|v: usize| v.checked_mul(grid.w))
                .ok_or_else(|| {
                    VisionError::InvalidConfig(format!("image grid {grid:?} overflowed T*H*W"))
                })?;
            let rows = num_patches / merge_sq;
            expected_pad_count = expected_pad_count.checked_add(rows).ok_or_else(|| {
                VisionError::InvalidConfig("total post-merger row count overflowed".into())
            })?;
            expected_rows = expected_pad_count;
        }

        let actual_pad_count = self.image_pad_count();
        if actual_pad_count != expected_pad_count {
            return Err(VisionError::ShapeMismatch {
                expected: expected_pad_count,
                actual: actual_pad_count,
                context: "image-pad token count in input_ids must equal T*H*W/merge_size^2 \
                          summed over image_grids"
                    .into(),
            });
        }

        let expected_buf_len = expected_rows
            .checked_mul(self.decoder_hidden_size)
            .ok_or_else(|| {
                VisionError::InvalidConfig(
                    "post_merger_rows length (rows * decoder_hidden_size) overflowed".into(),
                )
            })?;
        if self.post_merger_rows.len() != expected_buf_len {
            return Err(VisionError::ShapeMismatch {
                expected: expected_buf_len,
                actual: self.post_merger_rows.len(),
                context: "post_merger_rows length must equal total post-merger rows * \
                          decoder_hidden_size"
                    .into(),
            });
        }

        if let Some(bad) = self.post_merger_rows.iter().find(|v| !v.is_finite()) {
            return Err(VisionError::InvalidConfig(format!(
                "post_merger_rows contains a non-finite value: {bad}"
            )));
        }

        Ok(())
    }

    /// Build the per-physical-token M-RoPE position ids and interleaved
    /// cos/sin tables for this request's `input_ids`, using the checkpoint's
    /// M-RoPE configuration (`mrope_section`, `rope_theta`,
    /// `partial_rotary_factor`, `head_dim`). This is the decoder-facing
    /// wiring entry point over [`crate::vision::qwen35_mrope`]'s pure
    /// builder: it resolves the request/config fields into that module's
    /// arguments and validates the config carries a vision M-RoPE section
    /// before calling it. Only the six GQA layers consume the returned
    /// tables; GDN layers must not call this.
    pub fn build_mrope_tables(
        &self,
        cfg: &crate::model::qwen35_config::Qwen35Config,
    ) -> Result<
        (
            crate::vision::qwen35_mrope::MRopePositions,
            crate::vision::qwen35_mrope::MRopeTables,
        ),
        crate::error::InferenceError,
    > {
        use crate::vision::qwen35_mrope::{build_cos_sin, build_position_ids};

        let (mrope_section, theta, partial_rotary_factor) = Self::resolve_mrope_params(cfg)?;

        let positions = build_position_ids(
            &self.input_ids,
            self.image_token_id,
            &self.image_grids,
            self.spatial_merge_size,
        )?;
        let tables = build_cos_sin(
            &positions,
            cfg.head_dim,
            partial_rotary_factor,
            theta,
            &mrope_section,
        )?;
        Ok((positions, tables))
    }

    /// Resolve the checkpoint's M-RoPE config fields (`mrope_section`,
    /// `rope_theta`, `partial_rotary_factor`), shared by [`Self::build_mrope_tables`]
    /// (the full prefill table) and [`Self::build_decode_cos_sin`] (a single
    /// decode-time row), so both call the same checkpoint-field resolution.
    fn resolve_mrope_params(
        cfg: &crate::model::qwen35_config::Qwen35Config,
    ) -> Result<(Vec<usize>, f32, f32), crate::error::InferenceError> {
        use crate::error::InferenceError;

        let rope_params = cfg.rope_parameters.as_ref().ok_or_else(|| {
            InferenceError::InvalidInput(
                "checkpoint has no rope_parameters; cannot build M-RoPE tables".to_string(),
            )
        })?;
        let mrope_section = rope_params.mrope_section.clone().ok_or_else(|| {
            InferenceError::InvalidInput(
                "checkpoint rope_parameters has no mrope_section; not a vision-language config"
                    .to_string(),
            )
        })?;
        let partial_rotary_factor = rope_params.partial_rotary_factor.unwrap_or(0.25);
        let theta = rope_params.rope_theta as f32;
        Ok((mrope_section, theta, partial_rotary_factor))
    }

    /// Build the M-RoPE cos/sin row for a single decode-time position (all
    /// three axes equal — the physical-cache-length + `rope_delta` coordinate
    /// from [`crate::vision::qwen35_mrope::decode_position`]), via the same
    /// pure `build_cos_sin` builder [`Self::build_mrope_tables`] uses for the
    /// prefill table.
    pub fn build_decode_cos_sin(
        &self,
        cfg: &crate::model::qwen35_config::Qwen35Config,
        position: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), crate::error::InferenceError> {
        use crate::vision::qwen35_mrope::{MRopePositions, build_cos_sin};

        let (mrope_section, theta, partial_rotary_factor) = Self::resolve_mrope_params(cfg)?;
        let positions = MRopePositions {
            positions: vec![(position, position, position)],
            rope_delta: 0,
        };
        let tables = build_cos_sin(
            &positions,
            cfg.head_dim,
            partial_rotary_factor,
            theta,
            &mrope_section,
        )?;
        Ok((tables.cos[0].clone(), tables.sin[0].clone()))
    }
}

#[cfg(test)]
mod qwen35_vision_request_tests;

#[cfg(test)]
mod tests;
