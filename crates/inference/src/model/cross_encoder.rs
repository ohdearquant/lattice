//! Cross-encoder reranking model for BERT/MiniLM-style checkpoints.
//!
//! Wraps `BertModel` with a scalar classifier head to score (query, document)
//! pairs. Only supports `BertForSequenceClassification` checkpoints that have
//! `classifier.weight [1, hidden_size]` and `classifier.bias [1]` tensors.

use std::path::Path;

use crate::attention::AttentionBuffers;
use crate::error::InferenceError;
use crate::lora_hook::LoraHook;
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

    /// Score a single (query, document) pair with a LoRA hook applied during the forward pass.
    ///
    /// Geometry validation is *delegated* to the hook: this method calls
    /// [`LoraHook::validate_against_bert`] with the model's BERT dimensions
    /// before the forward pass runs, and maps any `Err` to
    /// [`InferenceError::InvalidInput`]. The trait's default implementation
    /// of that method returns `Ok(())` — it trusts the caller — so an
    /// adapter is checked here only to the extent that its own
    /// implementation checks itself. Adapters obtained through this
    /// workspace's own types (e.g. `lattice_tune::lora::LoraAdapter`)
    /// override it and are validated, so a mismatched one is rejected with a
    /// recoverable error rather than reaching the forward pass.
    pub fn score_with_hook(
        &self,
        query: &str,
        document: &str,
        lora: &dyn LoraHook,
    ) -> Result<f32, InferenceError> {
        self.validate_hook(lora)?;

        let input = self.bert.tokenizer().tokenize_pair(query, document);
        let seq_len = input.real_length;
        if seq_len == 0 {
            return Ok(0.5);
        }
        let hidden_size = self.bert.config().hidden_size;
        let mut buffers = AttentionBuffers::new(
            seq_len,
            hidden_size,
            self.bert.config().num_attention_heads,
            self.bert.config().intermediate_size,
        );
        let hidden = self
            .bert
            .forward_tokenized_with_hook(&input, &mut buffers, lora);
        let pooled = cls_pool(&hidden, seq_len, hidden_size);
        let logit = self.classifier.logit(&pooled);
        Ok(sigmoid(logit))
    }

    /// Score a query against a batch of documents with a LoRA hook applied during each forward pass.
    ///
    /// The hook is validated at the batch boundary, before any document is
    /// scored, and again by each per-document call this delegates to; the
    /// `validate_hook` helper documents why both calls are kept. The boundary call
    /// is what makes validation a property of the request: delegating it to
    /// the per-document method alone would
    /// tie it to the number of documents: an empty slice never enters the
    /// closure, so the request would answer `Ok(vec![])` without the hook ever
    /// having been asked about its geometry. A caller admitting an adapter on
    /// that answer would accept a malformed one and only discover it on a
    /// later nonempty request.
    pub fn score_batch_with_hook(
        &self,
        query: &str,
        documents: &[&str],
        lora: &dyn LoraHook,
    ) -> Result<Vec<f32>, InferenceError> {
        self.validate_hook(lora)?;

        documents
            .iter()
            .map(|doc| self.score_with_hook(query, doc, lora))
            .collect()
    }

    /// Ask a hook to check its own declared geometry against this model's BERT
    /// dimensions, mapping a rejection to [`InferenceError::InvalidInput`].
    ///
    /// Both hooked entry points route through here so that the check is a
    /// property of the request rather than of the work the request happens to
    /// perform. `score_with_hook` keeps its own call rather than relying on the
    /// batch boundary, because it is a public entry point in its own right and
    /// the geometry check is what makes it safe to reach the row loop; the
    /// resulting re-validation per batch document reads only declared
    /// dimensions and costs nothing measurable against a BERT forward pass.
    fn validate_hook(&self, lora: &dyn LoraHook) -> Result<(), InferenceError> {
        let config = self.bert.config();
        lora.validate_against_bert(
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
        )
        .map_err(InferenceError::InvalidInput)
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
