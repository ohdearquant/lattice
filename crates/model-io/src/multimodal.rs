//! Shared multimodal prompt input.

/// A multimodal prompt: merged visual patch embeddings prepended to text tokens.
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
    pub fn validate(&self) -> Result<(), crate::image::VisionError> {
        if self.d_model == 0 {
            return Err(crate::image::VisionError::InvalidConfig(
                "d_model must be > 0".into(),
            ));
        }
        let expected_len = self.visual_tokens * self.d_model;
        if self.patch_embeddings.len() != expected_len {
            return Err(crate::image::VisionError::ShapeMismatch {
                expected: expected_len,
                actual: self.patch_embeddings.len(),
                context: "patch_embeddings length must equal visual_tokens * d_model".into(),
            });
        }
        if self.visual_tokens == 0 {
            return Err(crate::image::VisionError::InvalidConfig(
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
mod tests;
