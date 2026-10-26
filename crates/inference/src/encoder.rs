//! The text encoder contract of `lattice-arch` (ADR-103 Amendment 21 item 3) and its adapters
//! for the embedding models this crate loads: BERT-family, Qwen3 and EmbeddingGemma 2.
//!
//! **Unstable**: each adapter wraps a loaded model in an `Arc` and is shared as
//! `Arc<dyn Encoder>`. It reads the model's settings once, when it is built, and reports
//! [`Provenance::Unavailable`](crate::encoder::Provenance::Unavailable) for the checkpoint and
//! the tokenizer: these models record no revision or digest of their files.

pub use lattice_arch::encoder::{
    EncodedText, Encoder, EncoderCapabilities, EncoderIdentity, Normalization, Pooling, Provenance,
    TokenUsage,
};

use crate::error::InferenceError;
use crate::model::embeddinggemma2::EmbeddingGemma2Model;
use crate::model::{BertModel, QwenModel};
use crate::pool::BertPooling;
use crate::tokenizer::EmbeddingTokenCounts;
use std::num::NonZeroUsize;
use std::sync::Arc;

fn output_dimension(width: usize, model: &str) -> Result<NonZeroUsize, InferenceError> {
    NonZeroUsize::new(width)
        .ok_or_else(|| InferenceError::InvalidInput(format!("{model} has output dimension 0")))
}

fn usage(counts: EmbeddingTokenCounts) -> Result<TokenUsage, InferenceError> {
    TokenUsage::new(counts.before_truncation, counts.embedded).ok_or_else(|| {
        InferenceError::Inference(format!(
            "token counts report {} embedded tokens of {}",
            counts.embedded, counts.before_truncation
        ))
    })
}

/// Pairs each vector with its input's counts, refusing the whole call when the model returned a
/// different number of vectors or a vector of another length.
fn pair(
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

/// A BERT-family model behind [`Encoder`]: mean or first-token pooling as the model is set, L2
/// normalized, one batched forward pass per call.
pub struct BertEncoder {
    model: Arc<BertModel>,
    capabilities: EncoderCapabilities,
    identity: EncoderIdentity,
}

impl BertEncoder {
    /// Wraps a loaded model. Its pooling is read here, so [`BertModel::set_pooling`] comes first.
    ///
    /// # Errors
    ///
    /// [`InferenceError::InvalidInput`] for a zero output dimension or a pooling strategy this
    /// adapter cannot describe.
    pub fn new(model: Arc<BertModel>) -> Result<Self, InferenceError> {
        let dimension = output_dimension(model.dimensions(), "the BERT model")?;
        let pooling = match model.pooling() {
            BertPooling::Mean => Pooling::Mean,
            BertPooling::CLS => Pooling::FirstToken,
            other => {
                return Err(InferenceError::InvalidInput(format!(
                    "the BERT encoder adapter does not describe pooling {other:?}"
                )));
            }
        };
        let limit = model
            .tokenizer()
            .max_seq_len()
            .min(model.config().max_position_embeddings);
        Ok(Self {
            capabilities: EncoderCapabilities::text(NonZeroUsize::new(limit)),
            identity: EncoderIdentity::new(pooling, Normalization::L2, dimension)
                .with_variant("cpu f32"),
            model,
        })
    }
}

impl Encoder for BertEncoder {
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
        let usages = prepared
            .iter()
            .map(|text| usage(self.model.embedding_token_counts(text)))
            .collect::<Result<Vec<_>, _>>()?;
        pair(
            self.model.encode_batch(prepared)?,
            usages,
            self.output_dimension(),
        )
    }
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
            usages.push(usage(counts)?);
            vectors.push(self.model.encode_ids(&ids, Some(dimension.get()))?);
        }
        pair(vectors, usages, dimension)
    }
}

#[cfg(test)]
mod tests;
