//! Shared validation for encoder implementations.

use crate::encoder::{EncodedText, TokenUsage};
use lattice_inference_core::InferenceError;
use std::num::NonZeroUsize;

/// Rejects a zero output width with the model-specific error.
pub fn output_dimension(width: usize, model: &str) -> Result<NonZeroUsize, InferenceError> {
    NonZeroUsize::new(width)
        .ok_or_else(|| InferenceError::InvalidInput(format!("{model} has output dimension 0")))
}

/// Validates counts reported by one tokenization.
pub fn usage(before_truncation: usize, embedded: usize) -> Result<TokenUsage, InferenceError> {
    TokenUsage::new(before_truncation, embedded).ok_or_else(|| {
        InferenceError::Inference(format!(
            "token counts report {} embedded tokens of {}",
            embedded, before_truncation
        ))
    })
}

/// Pairs each vector with its input's counts, refusing the whole call when the model returned a
/// different number of vectors or a vector of another length.
pub fn pair(
    vectors: Vec<Vec<f32>>,
    usages: Vec<TokenUsage>,
    dimension: NonZeroUsize,
) -> Result<Vec<EncodedText>, InferenceError> {
    if vectors.len() != usages.len() {
        return Err(InferenceError::Inference(format!(
            "the model returned {} vectors for {} inputs",
            vectors.len(),
            usages.len()
        )));
    }
    vectors
        .into_iter()
        .zip(usages)
        .map(|(vector, usage)| {
            if vector.len() == dimension.get() {
                Ok(EncodedText::new(vector, usage))
            } else {
                Err(InferenceError::Inference(format!(
                    "the model returned a vector of length {} for output dimension {dimension}",
                    vector.len()
                )))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
