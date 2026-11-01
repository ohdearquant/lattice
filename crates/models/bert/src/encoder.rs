//! BERT-family adaptation to the shared text encoder contract.

use crate::BertModel;
use lattice_arch::__internal::encoder::{output_dimension, pair, usage};
use lattice_arch::encoder::{
    EncodedText, Encoder, EncoderCapabilities, EncoderIdentity, Normalization, Pooling,
};
use lattice_compute::pool::BertPooling;
use lattice_inference_core::InferenceError;
use std::num::NonZeroUsize;
use std::sync::Arc;

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
            .map(|text| {
                let counts = self.model.embedding_token_counts(text);
                usage(counts.before_truncation, counts.embedded)
            })
            .collect::<Result<Vec<_>, _>>()?;
        pair(
            self.model.encode_batch(prepared)?,
            usages,
            self.output_dimension(),
        )
    }
}
