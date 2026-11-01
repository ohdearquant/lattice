//! Qwen3 text encoder adapter.

use crate::QwenModel;
use lattice_arch::__internal::encoder::{output_dimension, pair};
use lattice_arch::encoder::{
    EncodedText, Encoder, EncoderCapabilities, EncoderIdentity, Normalization, Pooling, TokenUsage,
};
use lattice_inference_core::InferenceError;
use lattice_tokenizer::common::EmbeddingTokenCounts;
use std::num::NonZeroUsize;
use std::sync::Arc;

fn usage(counts: EmbeddingTokenCounts) -> Result<TokenUsage, InferenceError> {
    lattice_arch::__internal::encoder::usage(counts.before_truncation, counts.embedded)
}

/// A Qwen3 embedding model behind [`Encoder`]: last-token pooling, L2 normalized, truncated to
/// the model's output dimension, one cached forward pass per input.
pub struct QwenEncoder {
    model: Arc<QwenModel>,
    capabilities: EncoderCapabilities,
    identity: EncoderIdentity,
}

impl QwenEncoder {
    /// Wraps a loaded model. The variant records the backend the model chose when it loaded:
    /// `metal` when it holds a Metal forward pass, `cpu f32` otherwise.
    ///
    /// # Errors
    ///
    /// [`InferenceError::InvalidInput`] for a zero output dimension.
    pub fn new(model: Arc<QwenModel>) -> Result<Self, InferenceError> {
        let dimension = output_dimension(model.dimensions(), "the Qwen model")?;
        let variant = if model.has_gpu() { "metal" } else { "cpu f32" };
        Ok(Self {
            capabilities: EncoderCapabilities::text(NonZeroUsize::new(
                model.tokenizer().max_seq_len(),
            )),
            identity: EncoderIdentity::new(Pooling::LastToken, Normalization::L2, dimension)
                .with_variant(variant),
            model,
        })
    }
}

impl Encoder for QwenEncoder {
    fn capabilities(&self) -> &EncoderCapabilities {
        &self.capabilities
    }

    fn output_dimension(&self) -> NonZeroUsize {
        self.identity.output_dimension()
    }

    fn identity(&self) -> &EncoderIdentity {
        &self.identity
    }

    fn encode_texts(&self, prepared: &[&str]) -> Result<Vec<EncodedText>, InferenceError> {
        let mut vectors = Vec::with_capacity(prepared.len());
        let mut usages = Vec::with_capacity(prepared.len());
        for text in prepared {
            usages.push(usage(self.model.embedding_token_counts(text))?);
            vectors.push(self.model.encode(text)?);
        }
        pair(vectors, usages, self.output_dimension())
    }
}
