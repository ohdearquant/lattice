//! Cross-encoder reranking model for BERT/MiniLM-style checkpoints.
//!
//! Wraps `BertModel` with a scalar classifier head to score (query, document)
//! pairs. Requires `classifier.weight [1, hidden_size]` and `classifier.bias [1]`.
//! Complete `pooler.dense.weight` / `pooler.dense.bias` pairs apply BERT's learned
//! dense-plus-tanh pooler before classification. Poolerless checkpoints retain
//! direct CLS classification for compatibility; incomplete poolers are rejected.

use std::path::Path;

use crate::attention::AttentionBuffers;
use crate::error::InferenceError;
use crate::forward::cpu::matmul_bt;
use crate::lora_hook::LoraHook;
use crate::model::bert::BertModel;
use crate::pool::cls_pool;
use crate::weights::{CrossEncoderWeights, SafetensorsFile};

/// Cross-encoder reranking model.
///
/// Loads a BERT-style scalar classification checkpoint and scores
/// `(query, document)` pairs as sigmoid probabilities. Uses the learned BERT
/// pooler when present, or direct CLS classification when both pooler tensors
/// are absent.
pub struct CrossEncoderModel {
    bert: BertModel,
    classifier: CrossEncoderWeights,
}

impl CrossEncoderModel {
    /// Load a cross-encoder from a model directory containing `model.safetensors`.
    ///
    /// Returns `Err(InferenceError::UnsupportedModel)` if the tokenizer does not
    /// support pair tokenization, if `type_vocab_size < 2`, or if only one of
    /// `pooler.dense.weight` and `pooler.dense.bias` is present.
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

        let (pooler_weight, pooler_bias) = bert.pooler_parameters();
        if pooler_weight.is_empty() != pooler_bias.is_empty() {
            return Err(InferenceError::UnsupportedModel(
                "cross-encoder pooler requires both pooler.dense.weight and pooler.dense.bias"
                    .to_string(),
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
        let mut pooled = cls_pool(&hidden, seq_len, hidden_size);
        let logit = self.classifier_logit(&hidden[..hidden_size], &mut pooled);
        sigmoid(logit)
    }

    /// Score a query against a batch of documents; returns one sigmoid per document.
    ///
    /// Deterministic: each returned score is bit-identical to what
    /// [`score`](Self::score) would return for that document scored on its own, and
    /// an empty `documents` slice returns an empty vec without touching the model.
    /// Tokenization for a validated `CrossEncoderModel` cannot fail per document
    /// (see [`from_directory`](Self::from_directory)), so one document's input
    /// never poisons the scores of the others in the batch.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lattice_inference::CrossEncoderModel;
    /// # use std::path::Path;
    /// # fn demo() -> Result<(), Box<dyn std::error::Error>> {
    /// // A directory laid out like a Hugging Face `cross-encoder/ms-marco-MiniLM-L-6-v2`
    /// // checkout: `config.json`, `vocab.txt`, `model.safetensors`.
    /// let model = CrossEncoderModel::from_directory(Path::new("ms-marco-MiniLM-L-6-v2"))?;
    /// let scores = model.score_batch(
    ///     "how many calories in an egg",
    ///     &["A large egg has about 78 calories.", "Paris is the capital of France."],
    /// );
    /// assert_eq!(scores.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
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
        let mut pooled = cls_pool(&hidden, seq_len, hidden_size);
        let logit = self.classifier_logit(&hidden[..hidden_size], &mut pooled);
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

    fn classifier_logit(&self, cls: &[f32], pooled: &mut [f32]) -> f32 {
        let (weight, bias) = self.bert.pooler_parameters();
        if weight.is_empty() {
            return self.classifier.logit(cls);
        }

        let hidden_size = self.bert.config().hidden_size;
        matmul_bt(cls, weight, pooled, 1, hidden_size, hidden_size);
        for (value, &bias) in pooled.iter_mut().zip(bias) {
            *value = (*value + bias).tanh();
        }
        self.classifier.logit(pooled)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const HIDDEN: usize = 4;

    type Tensor = (String, Vec<usize>, Vec<f32>);

    fn tensor(name: impl Into<String>, shape: &[usize], values: &[f32]) -> Tensor {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        (name.into(), shape.to_vec(), values.to_vec())
    }

    /// A minimal, self-contained BERT-style checkpoint: no pooler tensors (direct
    /// CLS classification), zero-gamma final LayerNorm so the pooled row is a fixed
    /// bias vector independent of attention/input. Needs no files from the repo.
    fn checkpoint() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("vocab.txt"),
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nquery\nshort\nlong\ndocument\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            json!({
                "vocab_size": 9,
                "hidden_size": HIDDEN,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "intermediate_size": HIDDEN,
                "max_position_embeddings": 32,
                "type_vocab_size": 2,
                "layer_norm_eps": 1e-5
            })
            .to_string(),
        )
        .unwrap();

        let mut tensors = vec![
            tensor(
                "embeddings.word_embeddings.weight",
                &[9, HIDDEN],
                &[0.0; 36],
            ),
            tensor(
                "embeddings.position_embeddings.weight",
                &[32, HIDDEN],
                &[0.0; 128],
            ),
            tensor(
                "embeddings.token_type_embeddings.weight",
                &[2, HIDDEN],
                &[0.0; 8],
            ),
            tensor("embeddings.LayerNorm.weight", &[HIDDEN], &[1.0; HIDDEN]),
            tensor("embeddings.LayerNorm.bias", &[HIDDEN], &[0.0; HIDDEN]),
            tensor("classifier.weight", &[1, HIDDEN], &[0.8, -0.4, 0.6, -0.3]),
            tensor("classifier.bias", &[1], &[0.15]),
        ];
        for module in [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ] {
            tensors.push(tensor(
                format!("encoder.layer.0.{module}.weight"),
                &[HIDDEN, HIDDEN],
                &[0.0; HIDDEN * HIDDEN],
            ));
            tensors.push(tensor(
                format!("encoder.layer.0.{module}.bias"),
                &[HIDDEN],
                &[0.0; HIDDEN],
            ));
        }
        tensors.push(tensor(
            "encoder.layer.0.attention.output.LayerNorm.weight",
            &[HIDDEN],
            &[1.0; HIDDEN],
        ));
        tensors.push(tensor(
            "encoder.layer.0.attention.output.LayerNorm.bias",
            &[HIDDEN],
            &[0.0; HIDDEN],
        ));
        // Zero gamma: every final hidden row is exactly beta, so the pooled CLS
        // vector below is fixed regardless of tokenization/attention.
        tensors.push(tensor(
            "encoder.layer.0.output.LayerNorm.weight",
            &[HIDDEN],
            &[0.0; HIDDEN],
        ));
        tensors.push(tensor(
            "encoder.layer.0.output.LayerNorm.bias",
            &[HIDDEN],
            &[0.25, -0.5, 0.75, 1.0],
        ));

        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, shape, values) in tensors {
            let start = payload.len();
            payload.extend(values.iter().flat_map(|value| value.to_le_bytes()));
            header.insert(
                name,
                json!({"dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        std::fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
        dir
    }

    /// (a) score_batch must be bit-identical to N sequential score() calls.
    #[test]
    fn score_batch_is_bit_identical_to_sequential_score_calls() {
        let dir = checkpoint();
        let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
        let documents = ["short", "long document", "", "query document pair"];

        let batch = model.score_batch("query", &documents);
        let sequential: Vec<f32> = documents
            .iter()
            .map(|doc| model.score("query", doc))
            .collect();

        assert_eq!(batch.len(), sequential.len());
        for (index, (&b, &s)) in batch.iter().zip(&sequential).enumerate() {
            assert_eq!(
                b.to_bits(),
                s.to_bits(),
                "document {index}: score_batch={b} score()={s}"
            );
        }
    }

    /// (b) an empty document list returns an empty vec.
    #[test]
    fn score_batch_empty_documents_returns_empty_vec() {
        let dir = checkpoint();
        let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
        let documents: [&str; 0] = [];

        let scores = model.score_batch("query", &documents);

        assert!(scores.is_empty());
    }
}
