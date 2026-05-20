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
