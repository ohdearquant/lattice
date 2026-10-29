//! EmbeddingGemma 2 implementation of the shared text encoder contract.

use crate::model::EmbeddingGemma2Model;
use lattice_arch::__internal::encoder::{output_dimension, pair, usage};
use lattice_arch::encoder::{
    EncodedText, Encoder, EncoderCapabilities, EncoderIdentity, Normalization, Pooling,
};
use lattice_inference_core::InferenceError;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// An EmbeddingGemma 2 model behind [`Encoder`]: mean pooling, L2 normalized, truncated to the
/// chosen output dimension and normalized again, one tokenization per input whose ids are the
/// ones embedded.
pub struct EmbeddingGemma2Encoder {
    model: Arc<EmbeddingGemma2Model>,
    capabilities: EncoderCapabilities,
    identity: EncoderIdentity,
}

impl EmbeddingGemma2Encoder {
    /// Wraps a loaded model, embedding at `output_dim` leading dimensions, or at the model's
    /// native width when `None`.
    ///
    /// # Errors
    ///
    /// [`InferenceError::InvalidInput`] when `output_dim` is 0 or above the native width.
    pub fn new(
        model: Arc<EmbeddingGemma2Model>,
        output_dim: Option<usize>,
    ) -> Result<Self, InferenceError> {
        let width = output_dim.unwrap_or_else(|| model.dimensions());
        model.check_widths(&[width])?;
        let dimension = output_dimension(width, "the EmbeddingGemma 2 model")?;
        Ok(Self {
            capabilities: EncoderCapabilities::text(model.max_tokens().and_then(NonZeroUsize::new)),
            identity: EncoderIdentity::new(Pooling::Mean, Normalization::L2, dimension)
                .with_variant("cpu f32"),
            model,
        })
    }
}

impl Encoder for EmbeddingGemma2Encoder {
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
        let dimension = self.output_dimension();
        let mut vectors = Vec::with_capacity(prepared.len());
        let mut usages = Vec::with_capacity(prepared.len());
        for text in prepared {
            let (ids, counts) = self.model.tokenize_with_counts(text)?;
            usages.push(usage(counts.before_truncation, counts.embedded)?);
            vectors.push(self.model.encode_ids(&ids, Some(dimension.get()))?);
        }
        pair(vectors, usages, dimension)
    }
}
