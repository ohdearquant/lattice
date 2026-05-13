//! Cross-encoder reranking model for BERT/MiniLM-style checkpoints.
//!
//! Wraps `BertModel` with a scalar classifier head to score (query, document)
//! pairs. Only supports `BertForSequenceClassification` checkpoints that have
//! `classifier.weight [1, hidden_size]` and `classifier.bias [1]` tensors.

use std::path::Path;

use crate::attention::AttentionBuffers;
use crate::error::InferenceError;
use crate::model::bert::BertModel;
use crate::pool::cls_pool;
use crate::weights::{CrossEncoderWeights, SafetensorsFile};

/// Cross-encoder reranking model.
///
/// Loads a `BertForSequenceClassification` checkpoint and scores
/// `(query, document)` pairs as sigmoid probabilities.
pub struct CrossEncoderModel {
    bert: BertModel,
    classifier: CrossEncoderWeights,
}

impl CrossEncoderModel {
    /// Load a cross-encoder from a model directory containing `model.safetensors`.
    ///
    /// Returns `Err(InferenceError::UnsupportedModel)` if the tokenizer does not
    /// support pair tokenization or if `type_vocab_size < 2`.
    pub fn from_directory(dir: &Path) -> Result<Self, InferenceError> {
        let bert = BertModel::from_directory(dir)?;

        if !bert.tokenizer().supports_pair_tokenization() {
            return Err(InferenceError::UnsupportedModel(
                "cross-encoder requires a tokenizer with BERT pair tokenization".to_string(),
            ));
        }
        if bert.config().type_vocab_size < 2 {
            return Err(InferenceError::UnsupportedModel(
                "BERT cross-encoder pair tokenization requires type_vocab_size >= 2".to_string(),
            ));
        }

        let safetensors = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        let classifier = safetensors.load_cross_encoder_weights(bert.config().hidden_size)?;

        Ok(Self { bert, classifier })
    }

    /// Score a single (query, document) pair; returns sigmoid probability in [0, 1].
    pub fn score(&self, query: &str, document: &str) -> f32 {
        let input = self.bert.tokenizer().tokenize_pair(query, document);
        let seq_len = input.real_length;
        if seq_len == 0 {
            return 0.5;
        }
        let hidden_size = self.bert.config().hidden_size;
        let mut buffers = AttentionBuffers::new(
            seq_len,
            hidden_size,
            self.bert.config().num_attention_heads,
            self.bert.config().intermediate_size,
        );
        let hidden = self.bert.forward_tokenized(&input, &mut buffers);
        let pooled = cls_pool(&hidden, seq_len, hidden_size);
        let logit = self.classifier.logit(&pooled);
        sigmoid(logit)
    }

    /// Score a query against a batch of documents; returns one sigmoid per document.
    pub fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<f32> {
        documents.iter().map(|doc| self.score(query, doc)).collect()
    }

    /// Access the underlying `BertModel` (for config and tokenizer inspection).
    pub fn bert(&self) -> &BertModel {
        &self.bert
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}
