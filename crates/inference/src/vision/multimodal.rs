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
mod qwen35_vision_request_tests {
    use super::*;
    use crate::vision::qwen35_vit::GridThw;

    fn base_request() -> Qwen35VisionRequest {
        // One image run: grid (1,4,4), merge_size=2 -> 4 post-merger rows.
        let mut input_ids = vec![10u32, 11, 248_053];
        input_ids.extend(std::iter::repeat_n(248_056u32, 4));
        input_ids.push(248_054);
        input_ids.push(12);
        Qwen35VisionRequest {
            input_ids,
            image_grids: vec![GridThw { t: 1, h: 4, w: 4 }],
            post_merger_rows: vec![0.1f32; 4 * 8],
            image_token_id: 248_056,
            spatial_merge_size: 2,
            decoder_hidden_size: 8,
        }
    }

    #[test]
    fn valid_request_passes() {
        let req = base_request();
        assert_eq!(req.image_pad_count(), 4);
        req.validate().expect("valid request");
    }

    #[test]
    fn zero_decoder_hidden_size_rejected() {
        let mut req = base_request();
        req.decoder_hidden_size = 0;
        assert!(req.validate().is_err());
    }

    #[test]
    fn zero_spatial_merge_size_rejected() {
        let mut req = base_request();
        req.spatial_merge_size = 0;
        assert!(req.validate().is_err());
    }

    #[test]
    fn empty_input_ids_rejected() {
        let mut req = base_request();
        req.input_ids.clear();
        assert!(req.validate().is_err());
    }

    #[test]
    fn zero_dim_grid_rejected() {
        let mut req = base_request();
        req.image_grids[0] = GridThw { t: 0, h: 4, w: 4 };
        assert!(req.validate().is_err());
    }

    #[test]
    fn row_times_hidden_overflow_rejected() {
        let mut req = base_request();
        // Force the rows*hidden_size product past usize::MAX.
        req.decoder_hidden_size = usize::MAX / 2;
        assert!(req.validate().is_err());
    }

    #[test]
    fn grid_patch_count_overflow_rejected() {
        let mut req = base_request();
        req.image_grids[0] = GridThw {
            t: usize::MAX,
            h: usize::MAX,
            w: 2,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn non_finite_visual_scalar_rejected_nan() {
        let mut req = base_request();
        req.post_merger_rows[0] = f32::NAN;
        assert!(req.validate().is_err());
    }

    #[test]
    fn non_finite_visual_scalar_rejected_inf() {
        let mut req = base_request();
        req.post_merger_rows[3] = f32::INFINITY;
        assert!(req.validate().is_err());
    }

    #[test]
    fn empty_grid_list_with_no_images_is_valid() {
        let req = Qwen35VisionRequest {
            input_ids: vec![1, 2, 3],
            image_grids: vec![],
            post_merger_rows: vec![],
            image_token_id: 248_056,
            spatial_merge_size: 2,
            decoder_hidden_size: 8,
        };
        req.validate().expect("text-only request is valid");
    }

    #[test]
    fn grid_list_length_mismatched_against_image_runs_rejected() {
        let mut req = base_request();
        // Two grids supplied but only one contiguous image-pad run exists.
        req.image_grids.push(GridThw { t: 1, h: 2, w: 2 });
        assert!(req.validate().is_err());
    }

    #[test]
    fn mismatched_pad_count_vs_grid_rejected() {
        let mut req = base_request();
        // Drop one image-pad token so the run no longer matches T*H*W/m^2.
        let first_pad = req
            .input_ids
            .iter()
            .position(|&id| id == req.image_token_id)
            .expect("has a pad token");
        req.input_ids.remove(first_pad);
        assert!(req.validate().is_err());
    }

    #[test]
    fn post_merger_buffer_wrong_length_rejected() {
        let mut req = base_request();
        req.post_merger_rows.pop();
        assert!(req.validate().is_err());
    }

    #[test]
    fn grid_not_divisible_by_merge_size_rejected() {
        let mut req = base_request();
        req.image_grids[0] = GridThw { t: 1, h: 3, w: 4 };
        // 3 is not divisible by spatial_merge_size=2.
        assert!(req.validate().is_err());
    }

    /// HF-probed golden (probes/RESULTS.md, probe 2 + probe 3): an 82-token
    /// prefill with a text prefix of length 4, then an image with unmerged
    /// grid `(1,16,16)` / merge_size=2 (merged LLM grid `1,8,8`, 64
    /// post-merger rows). The first image-pad physical token lands at
    /// M-RoPE position `(4,4,4)`, and its 32-lane cos/sin row matches the
    /// checkpoint's `theta=1e7`, `partial_rotary_factor=0.25`,
    /// `mrope_section=[11,11,10]` config exactly (float32-level agreement).
    /// This test drives the request's `build_mrope_tables` wiring method,
    /// not the already-covered S5a unit functions directly.
    #[test]
    fn build_mrope_tables_matches_hf_probe_golden_through_wired_path() {
        use crate::model::qwen35_config::{Qwen35Config, RopeParams};

        let mut input_ids: Vec<u32> = vec![10, 11, 248_053, 12];
        input_ids.extend(std::iter::repeat_n(248_056u32, 64));
        input_ids.push(248_054);

        let req = Qwen35VisionRequest {
            input_ids,
            image_grids: vec![GridThw { t: 1, h: 16, w: 16 }],
            post_merger_rows: vec![0.0f32; 64 * 1024],
            image_token_id: 248_056,
            spatial_merge_size: 2,
            decoder_hidden_size: 1024,
        };
        req.validate().expect("probe-shaped request is valid");

        let cfg = Qwen35Config {
            head_dim: 256,
            rope_parameters: Some(RopeParams {
                rope_theta: 1.0e7,
                partial_rotary_factor: Some(0.25),
                mrope_section: Some(vec![11, 11, 10]),
                mrope_interleaved: Some(true),
            }),
            ..Qwen35Config::default()
        };

        let (positions, tables) = req.build_mrope_tables(&cfg).expect("builds tables");

        // Probe 2: first image-pad token (physical index 4) lands at (4,4,4).
        assert_eq!(positions.positions[4], (4, 4, 4));

        // Probe 3: 32-lane cos/sin row at that position, float32-level match.
        let cos = &tables.cos[4];
        let sin = &tables.sin[4];
        assert!((cos[0] - (-0.653_644)).abs() < 1e-4, "cos[0]={}", cos[0]);
        assert!((sin[0] - (-0.756_802)).abs() < 1e-4, "sin[0]={}", sin[0]);
        assert!((cos[1] - (-0.748_892)).abs() < 1e-4, "cos[1]={}", cos[1]);
        assert!((cos[2] - 0.109_877).abs() < 1e-4, "cos[2]={}", cos[2]);
        assert!((cos[3] - 0.635_073).abs() < 1e-4, "cos[3]={}", cos[3]);
    }

    /// Text-only non-regression, driven through `build_mrope_tables` (the
    /// new decoder-facing wiring), not just the already-covered S5a unit
    /// (`text_only_reduces_to_1d_table` in `qwen35_mrope`): with no image
    /// runs, every physical token has all three M-RoPE axes equal to its
    /// sequential position, so the interleaved axis-selection collapses to
    /// plain 1-D stride-half RoPE — `cos[lane] = cos(pos * theta^(-2*lane/64))`
    /// for every lane regardless of which axis it nominally reads from.
    #[test]
    fn build_mrope_tables_text_only_collapses_to_1d() {
        use crate::model::qwen35_config::{Qwen35Config, RopeParams};

        let req = Qwen35VisionRequest {
            input_ids: vec![5, 6, 7, 8, 9],
            image_grids: vec![],
            post_merger_rows: vec![],
            image_token_id: 248_056,
            spatial_merge_size: 2,
            decoder_hidden_size: 1024,
        };
        req.validate().expect("text-only request is valid");

        let cfg = Qwen35Config {
            head_dim: 256,
            rope_parameters: Some(RopeParams {
                rope_theta: 1.0e7,
                partial_rotary_factor: Some(0.25),
                mrope_section: Some(vec![11, 11, 10]),
                mrope_interleaved: Some(true),
            }),
            ..Qwen35Config::default()
        };

        let (positions, tables) = req.build_mrope_tables(&cfg).expect("builds tables");

        for (token_idx, &(t, h, w)) in positions.positions.iter().enumerate() {
            assert_eq!(t, token_idx as u32);
            assert_eq!(h, token_idx as u32);
            assert_eq!(w, token_idx as u32);

            let pos = f64::from(token_idx as u32);
            for lane in 0..32usize {
                let inv_freq = 1.0e7_f64.powf(-2.0 * lane as f64 / 64.0);
                let angle = pos * inv_freq;
                let expected_cos = angle.cos() as f32;
                let expected_sin = angle.sin() as f32;
                assert!(
                    (tables.cos[token_idx][lane] - expected_cos).abs() < 1e-4,
                    "token {token_idx} lane {lane}: cos {} vs expected {expected_cos}",
                    tables.cos[token_idx][lane]
                );
                assert!(
                    (tables.sin[token_idx][lane] - expected_sin).abs() < 1e-4,
                    "token {token_idx} lane {lane}: sin {} vs expected {expected_sin}",
                    tables.sin[token_idx][lane]
                );
            }
        }
    }

    #[test]
    fn build_mrope_tables_rejects_config_without_mrope_section() {
        use crate::model::qwen35_config::Qwen35Config;

        let req = base_request();
        let cfg = Qwen35Config::default(); // no rope_parameters -> no mrope_section
        assert!(req.build_mrope_tables(&cfg).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_input_valid() {
        let input = MultimodalInput {
            patch_embeddings: vec![0.0f32; 196 * 2048],
            raw_patches: 784,
            visual_tokens: 196,
            d_model: 2048,
            text_tokens: vec![1, 2, 3],
        };
        input.validate().expect("valid input");
        assert_eq!(input.total_sequence_len(), 199); // 196 + 3
    }

    #[test]
    fn multimodal_input_wrong_embed_len() {
        let input = MultimodalInput {
            patch_embeddings: vec![0.0f32; 100], // wrong: should be 196*2048
            raw_patches: 784,
            visual_tokens: 196,
            d_model: 2048,
            text_tokens: vec![],
        };
        assert!(input.validate().is_err());
    }

    #[test]
    fn multimodal_input_zero_d_model() {
        let input = MultimodalInput {
            patch_embeddings: vec![],
            raw_patches: 784,
            visual_tokens: 196,
            d_model: 0,
            text_tokens: vec![],
        };
        assert!(input.validate().is_err());
    }

    #[test]
    fn multimodal_input_zero_visual_tokens() {
        let input = MultimodalInput {
            patch_embeddings: vec![],
            raw_patches: 0,
            visual_tokens: 0,
            d_model: 2048,
            text_tokens: vec![1, 2],
        };
        assert!(input.validate().is_err());
    }

    #[test]
    fn total_sequence_len() {
        let input = MultimodalInput {
            patch_embeddings: vec![0.0f32; 196 * 512],
            raw_patches: 784,
            visual_tokens: 196,
            d_model: 512,
            text_tokens: vec![10, 20, 30, 40],
        };
        assert_eq!(input.total_sequence_len(), 200);
    }
}
