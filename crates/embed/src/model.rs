//! Embedding model selection, runtime dimensions, and load provenance.
//!
//! `EmbeddingModel` defines model-specific limits, prompting, and inference wiring.
//! `ModelConfig` validates optional Matryoshka output truncation for supported models.
//!
//! See docs/model.md for the model and cache design.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
///
/// Records the model source and metadata for a load event.
///
/// See [`docs/model.md`](../docs/model.md) (§ModelProvenance source behavior) for hash semantics and verification boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvenance {
    /// **Stable**: model variant that was loaded.
    pub model: EmbeddingModel,
    /// **Stable**: source identifier (HuggingFace ID, URL, or file path).
    pub model_id: String,
    /// **Stable**: metadata-derived BLAKE3 identifier for this load event, not a weight checksum.
    pub hash: String,
    /// **Stable**: when the model was loaded.
    pub loaded_at: SystemTime,
    /// **Stable**: formatted timestamp string for convenience.
    pub loaded_at_iso: String,
}

impl ModelProvenance {
    /// **Stable**: create new provenance information for a loaded model.
    pub fn new(model: EmbeddingModel, model_id: String) -> Self {
        let loaded_at = SystemTime::now();
        let loaded_at_iso = {
            let dt: chrono::DateTime<chrono::Utc> = loaded_at.into();
            dt.to_rfc3339()
        };

        // Create a lightweight hash from model metadata
        let hash_input = format!("{model_id}:{loaded_at_iso}:{model:?}");
        let hash = blake3::hash(hash_input.as_bytes()).to_hex().to_string();

        Self {
            model,
            model_id,
            hash,
            loaded_at,
            loaded_at_iso,
        }
    }

    /// **Stable**: get the model dimensions.
    pub fn dimensions(&self) -> usize {
        self.model.dimensions()
    }

    /// **Stable**: check if this provenance matches expected model.
    pub fn matches_model(&self, expected: EmbeddingModel) -> bool {
        self.model == expected
    }
}

/// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
///
/// Registry of supported local and remote embedding models.
///
/// See [`docs/model.md`](../docs/model.md) (§EmbeddingModel source behavior) for model capabilities and identity rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EmbeddingModel {
    /// BGE small English v1.5 (384 dimensions) - fast and efficient.
    #[default]
    #[serde(alias = "BgeSmallEnV15")]
    BgeSmallEnV15,

    /// BGE base English v1.5 (768 dimensions) - balanced quality/speed.
    #[serde(alias = "BgeBaseEnV15")]
    BgeBaseEnV15,

    /// BGE large English v1.5 (1024 dimensions) - highest quality local.
    #[serde(alias = "BgeLargeEnV15")]
    BgeLargeEnV15,

    /// Multilingual E5 small (384 dimensions) - multilingual, same arch as BGE.
    #[serde(alias = "MultilingualE5Small")]
    MultilingualE5Small,

    /// Multilingual E5 base (768 dimensions) - best multilingual quality/speed.
    #[serde(alias = "MultilingualE5Base")]
    MultilingualE5Base,

    /// Qwen3-Embedding-0.6B (1024 dimensions) - multilingual, decoder-only, GPU-accelerated.
    #[serde(alias = "Qwen3Embedding0_6B")]
    Qwen3Embedding0_6B,

    /// Qwen3-Embedding-4B (2560 dimensions, MRL-capable) - multilingual, decoder-only, GPU-accelerated.
    #[serde(alias = "Qwen3Embedding4B")]
    Qwen3Embedding4B,

    /// all-MiniLM-L6-v2 (384 dimensions) - BERT-class, WordPiece tokenizer, sentence-transformers.
    #[serde(alias = "AllMiniLmL6V2")]
    AllMiniLmL6V2,

    /// paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions) - multilingual, XLM-R base, sentence-transformers.
    #[serde(alias = "ParaphraseMultilingualMiniLmL12V2")]
    ParaphraseMultilingualMiniLmL12V2,

    /// OpenAI text-embedding-3-small (1536 dimensions) - remote API.
    #[serde(alias = "TextEmbedding3Small")]
    TextEmbedding3Small,
}

impl EmbeddingModel {
    /// **Stable**: get the native (full-resolution) output dimension of this model's embeddings.
    ///
    /// Returns the model's intrinsic dimension regardless of any MRL truncation.
    /// For MRL-capable models with a configured truncation, use `ModelConfig::dimensions()`.
    #[inline]
    pub const fn native_dimensions(&self) -> usize {
        match self {
            EmbeddingModel::BgeSmallEnV15
            | EmbeddingModel::MultilingualE5Small
            | EmbeddingModel::AllMiniLmL6V2
            | EmbeddingModel::ParaphraseMultilingualMiniLmL12V2 => 384,
            EmbeddingModel::BgeBaseEnV15 | EmbeddingModel::MultilingualE5Base => 768,
            EmbeddingModel::BgeLargeEnV15 | EmbeddingModel::Qwen3Embedding0_6B => 1024,
            EmbeddingModel::Qwen3Embedding4B => 2560,
            EmbeddingModel::TextEmbedding3Small => 1536,
        }
    }

    /// **Stable**: get this model's native output dimension.
    ///
    /// See [`docs/model.md`](../docs/model.md) (§EmbeddingModel source behavior) for active-dimension selection.
    #[inline]
    pub const fn dimensions(&self) -> usize {
        self.native_dimensions()
    }

    /// **Stable**: check if this model can run locally (via lattice-inference).
    #[inline]
    pub const fn is_local(&self) -> bool {
        matches!(
            self,
            EmbeddingModel::BgeSmallEnV15
                | EmbeddingModel::BgeBaseEnV15
                | EmbeddingModel::BgeLargeEnV15
                | EmbeddingModel::MultilingualE5Small
                | EmbeddingModel::MultilingualE5Base
                | EmbeddingModel::AllMiniLmL6V2
                | EmbeddingModel::ParaphraseMultilingualMiniLmL12V2
                | EmbeddingModel::Qwen3Embedding0_6B
                | EmbeddingModel::Qwen3Embedding4B
        )
    }

    /// **Stable**: check if this model requires a remote API.
    #[inline]
    pub const fn is_remote(&self) -> bool {
        matches!(self, EmbeddingModel::TextEmbedding3Small)
    }

    /// **Stable**: conservative maximum input tokens for chunking and truncation.
    ///
    /// See [`docs/model.md`](../docs/model.md) (§EmbeddingModel source behavior) for per-model limits.
    #[inline]
    pub const fn max_input_tokens(&self) -> usize {
        match self {
            EmbeddingModel::BgeSmallEnV15 => 512,
            EmbeddingModel::BgeBaseEnV15 => 512,
            EmbeddingModel::BgeLargeEnV15 => 512,
            EmbeddingModel::MultilingualE5Small => 512,
            EmbeddingModel::MultilingualE5Base => 512,
            EmbeddingModel::AllMiniLmL6V2 => 256,
            EmbeddingModel::ParaphraseMultilingualMiniLmL12V2 => 128,
            // Conservative cap; see docs/model.md.
            EmbeddingModel::Qwen3Embedding0_6B => 8192,
            EmbeddingModel::Qwen3Embedding4B => 8192,
            EmbeddingModel::TextEmbedding3Small => 8191,
        }
    }

    /// **Stable**: query instruction prefix for asymmetric retrieval, when required.
    ///
    /// See [`docs/model.md`](../docs/model.md) (§EmbeddingModel source behavior) for prompt policy and vector-space implications.
    #[inline]
    pub const fn query_instruction(&self) -> Option<&'static str> {
        match self {
            EmbeddingModel::MultilingualE5Small | EmbeddingModel::MultilingualE5Base => {
                Some("query: ")
            }
            EmbeddingModel::Qwen3Embedding0_6B | EmbeddingModel::Qwen3Embedding4B => Some(
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            ),
            EmbeddingModel::BgeSmallEnV15
            | EmbeddingModel::BgeBaseEnV15
            | EmbeddingModel::BgeLargeEnV15 => {
                Some("Represent this sentence for searching relevant passages: ")
            }
            _ => None,
        }
    }

    /// **Stable**: document instruction prefix for asymmetric retrieval, when required.
    ///
    /// See [`docs/model.md`](../docs/model.md) (§EmbeddingModel source behavior) for prompt policy and vector-space implications.
    #[inline]
    pub const fn document_instruction(&self) -> Option<&'static str> {
        match self {
            EmbeddingModel::MultilingualE5Small | EmbeddingModel::MultilingualE5Base => {
                Some("passage: ")
            }
            _ => None,
        }
    }

    /// **Stable**: get the model identifier (HuggingFace ID or provider/model).
    #[inline]
    pub const fn model_id(&self) -> &'static str {
        match self {
            EmbeddingModel::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            EmbeddingModel::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
            EmbeddingModel::BgeLargeEnV15 => "BAAI/bge-large-en-v1.5",
            EmbeddingModel::MultilingualE5Small => "intfloat/multilingual-e5-small",
            EmbeddingModel::MultilingualE5Base => "intfloat/multilingual-e5-base",
            EmbeddingModel::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            EmbeddingModel::ParaphraseMultilingualMiniLmL12V2 => {
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            }
            EmbeddingModel::Qwen3Embedding0_6B => "Qwen/Qwen3-Embedding-0.6B",
            EmbeddingModel::Qwen3Embedding4B => "Qwen/Qwen3-Embedding-4B",
            EmbeddingModel::TextEmbedding3Small => "text-embedding-3-small",
        }
    }

    /// **Stable**: whether this model supports configurable output dimensions (MRL/Matryoshka).
    #[inline]
    pub const fn supports_output_dim(&self) -> bool {
        matches!(
            self,
            EmbeddingModel::Qwen3Embedding0_6B | EmbeddingModel::Qwen3Embedding4B
        )
    }

    /// **Stable**: BERT pooling strategy for this model, or `None` for non-BERT paths.
    ///
    /// See [`docs/model.md`](../docs/model.md) (§EmbeddingModel source behavior) for pooling routing.
    #[cfg(feature = "native")]
    #[inline]
    pub const fn bert_pooling(&self) -> Option<lattice_inference::BertPooling> {
        match self {
            EmbeddingModel::BgeSmallEnV15
            | EmbeddingModel::BgeBaseEnV15
            | EmbeddingModel::BgeLargeEnV15 => Some(lattice_inference::BertPooling::CLS),
            EmbeddingModel::MultilingualE5Small | EmbeddingModel::MultilingualE5Base => {
                Some(lattice_inference::BertPooling::Mean)
            }
            EmbeddingModel::AllMiniLmL6V2 | EmbeddingModel::ParaphraseMultilingualMiniLmL12V2 => {
                Some(lattice_inference::BertPooling::Mean)
            }
            EmbeddingModel::Qwen3Embedding0_6B
            | EmbeddingModel::Qwen3Embedding4B
            | EmbeddingModel::TextEmbedding3Small => None,
        }
    }

    /// **Stable**: embedding key revision string for this model family.
    #[inline]
    pub const fn key_version(&self) -> &'static str {
        match self {
            EmbeddingModel::TextEmbedding3Small
            | EmbeddingModel::Qwen3Embedding0_6B
            | EmbeddingModel::Qwen3Embedding4B => "v3",
            EmbeddingModel::AllMiniLmL6V2 | EmbeddingModel::ParaphraseMultilingualMiniLmL12V2 => {
                "v2"
            }
            _ => "v1.5",
        }
    }
}

impl std::fmt::Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingModel::BgeSmallEnV15 => write!(f, "bge-small-en-v1.5"),
            EmbeddingModel::BgeBaseEnV15 => write!(f, "bge-base-en-v1.5"),
            EmbeddingModel::BgeLargeEnV15 => write!(f, "bge-large-en-v1.5"),
            EmbeddingModel::MultilingualE5Small => write!(f, "multilingual-e5-small"),
            EmbeddingModel::MultilingualE5Base => write!(f, "multilingual-e5-base"),
            EmbeddingModel::Qwen3Embedding0_6B => write!(f, "qwen3-embedding-0.6b"),
            EmbeddingModel::Qwen3Embedding4B => write!(f, "qwen3-embedding-4b"),
            EmbeddingModel::AllMiniLmL6V2 => write!(f, "all-minilm-l6-v2"),
            EmbeddingModel::ParaphraseMultilingualMiniLmL12V2 => {
                write!(f, "paraphrase-multilingual-minilm-l12-v2")
            }
            EmbeddingModel::TextEmbedding3Small => write!(f, "text-embedding-3-small"),
        }
    }
}

impl std::str::FromStr for EmbeddingModel {
    type Err = String;

    /// **Stable**: parse a normalized canonical name, alias, or supported provider identifier.
    ///
    /// See [`docs/model.md`](../docs/model.md) (§EmbeddingModel source behavior) for accepted forms and persistence guidance.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        let normalized = lower.trim().replace("_", "-").replace("baai/", "");

        match normalized.as_str() {
            "bge-small-en-v1.5" | "bge-small-en" | "bge-small" | "small" => {
                Ok(EmbeddingModel::BgeSmallEnV15)
            }
            "bge-base-en-v1.5" | "bge-base-en" | "bge-base" | "base" => {
                Ok(EmbeddingModel::BgeBaseEnV15)
            }
            "bge-large-en-v1.5" | "bge-large-en" | "bge-large" | "large" => {
                Ok(EmbeddingModel::BgeLargeEnV15)
            }
            "multilingual-e5-small" | "e5-small" | "intfloat/multilingual-e5-small" => {
                Ok(EmbeddingModel::MultilingualE5Small)
            }
            "multilingual-e5-base" | "e5-base" | "intfloat/multilingual-e5-base" => {
                Ok(EmbeddingModel::MultilingualE5Base)
            }
            "qwen3-embedding-0.6b" | "qwen3-embedding" | "qwen3" | "qwen/qwen3-embedding-0.6b" => {
                Ok(EmbeddingModel::Qwen3Embedding0_6B)
            }
            "qwen3-embedding-4b" | "qwen3-4b" | "qwen/qwen3-embedding-4b" => {
                Ok(EmbeddingModel::Qwen3Embedding4B)
            }
            "all-minilm-l6-v2"
            | "minilm"
            | "all-minilm"
            | "sentence-transformers/all-minilm-l6-v2" => Ok(EmbeddingModel::AllMiniLmL6V2),
            "paraphrase-multilingual-minilm-l12-v2"
            | "paraphrase-multilingual"
            | "multilingual-minilm"
            | "sentence-transformers/paraphrase-multilingual-minilm-l12-v2" => {
                Ok(EmbeddingModel::ParaphraseMultilingualMiniLmL12V2)
            }
            "text-embedding-3-small" | "openai-small" | "openai" => {
                Ok(EmbeddingModel::TextEmbedding3Small)
            }
            _ => Err(format!(
                "unknown embedding model: '{s}'. Valid: bge-small-en-v1.5, bge-base-en-v1.5, bge-large-en-v1.5, multilingual-e5-small, multilingual-e5-base, text-embedding-3-small"
            )),
        }
    }
}

// ============================================================================
// ModelConfig — runtime MRL dimension configuration
// ============================================================================

/// Minimum allowed MRL output dimension.
pub const MIN_MRL_OUTPUT_DIM: usize = 32;

/// Runtime model configuration with an optional MRL truncation dimension.
///
/// See [`docs/model.md`](../docs/model.md) (§ModelConfig source behavior) for validation and namespace requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelConfig {
    /// The underlying embedding model.
    pub model: EmbeddingModel,
    /// MRL truncation dimension. `None` uses the model's native dimension.
    #[serde(default)]
    pub output_dim: Option<usize>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::new(EmbeddingModel::default())
    }
}

impl ModelConfig {
    /// Create a config with no MRL truncation (native model dimension).
    pub const fn new(model: EmbeddingModel) -> Self {
        Self {
            model,
            output_dim: None,
        }
    }

    /// Create and validate a config with an optional MRL truncation dimension.
    pub fn try_new(
        model: EmbeddingModel,
        output_dim: Option<usize>,
    ) -> std::result::Result<Self, crate::error::EmbedError> {
        let config = Self { model, output_dim };
        config.validate()?;
        Ok(config)
    }

    /// Validate that the output dimension is consistent with the model.
    pub fn validate(&self) -> std::result::Result<(), crate::error::EmbedError> {
        let Some(dim) = self.output_dim else {
            return Ok(());
        };
        if !self.model.supports_output_dim() {
            return Err(crate::error::EmbedError::InvalidInput(format!(
                "{} does not support configurable embedding dimensions",
                self.model
            )));
        }
        if dim < MIN_MRL_OUTPUT_DIM {
            return Err(crate::error::EmbedError::InvalidInput(format!(
                "embedding output dimension {dim} is below minimum {MIN_MRL_OUTPUT_DIM}"
            )));
        }
        let native = self.model.native_dimensions();
        if dim > native {
            return Err(crate::error::EmbedError::InvalidInput(format!(
                "embedding output dimension {dim} exceeds native dimension {native} for {}",
                self.model
            )));
        }
        Ok(())
    }

    /// Active output dimension: configured truncation if set, otherwise the model's native dimension.
    pub fn dimensions(&self) -> usize {
        self.output_dim
            .unwrap_or_else(|| self.model.native_dimensions())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model() {
        let model = EmbeddingModel::default();
        assert_eq!(model, EmbeddingModel::BgeSmallEnV15);
    }

    #[test]
    fn test_model_provenance_new() {
        let provenance = ModelProvenance::new(
            EmbeddingModel::BgeSmallEnV15,
            "BAAI/bge-small-en-v1.5".into(),
        );

        assert_eq!(provenance.model, EmbeddingModel::BgeSmallEnV15);
        assert_eq!(provenance.model_id, "BAAI/bge-small-en-v1.5");
        assert!(!provenance.hash.is_empty());
        assert_eq!(provenance.hash.len(), 64); // blake3 hex is 64 chars
        assert!(!provenance.loaded_at_iso.is_empty());
    }

    #[test]
    fn test_model_provenance_unique_hash() {
        let p1 = ModelProvenance::new(EmbeddingModel::BgeSmallEnV15, "model1".into());
        std::thread::sleep(std::time::Duration::from_millis(10)); // Ensure different timestamp
        let p2 = ModelProvenance::new(EmbeddingModel::BgeSmallEnV15, "model1".into());

        // Different timestamps should produce different hashes
        assert_ne!(p1.hash, p2.hash);
    }

    #[test]
    fn test_model_provenance_dimensions() {
        let p1 = ModelProvenance::new(EmbeddingModel::BgeSmallEnV15, "small".into());
        assert_eq!(p1.dimensions(), 384);

        let p2 = ModelProvenance::new(EmbeddingModel::BgeBaseEnV15, "base".into());
        assert_eq!(p2.dimensions(), 768);

        let p3 = ModelProvenance::new(EmbeddingModel::BgeLargeEnV15, "large".into());
        assert_eq!(p3.dimensions(), 1024);
    }

    #[test]
    fn test_model_provenance_matches_model() {
        let provenance = ModelProvenance::new(EmbeddingModel::BgeSmallEnV15, "test".into());

        assert!(provenance.matches_model(EmbeddingModel::BgeSmallEnV15));
        assert!(!provenance.matches_model(EmbeddingModel::BgeBaseEnV15));
        assert!(!provenance.matches_model(EmbeddingModel::BgeLargeEnV15));
    }

    #[test]
    fn test_model_provenance_serialization() {
        let provenance = ModelProvenance::new(EmbeddingModel::BgeSmallEnV15, "test-model".into());

        let json = serde_json::to_string(&provenance).unwrap();
        // FP-037: EmbeddingModel has #[serde(rename_all = "snake_case")] so
        // BgeSmallEnV15 serializes as "bge_small_en_v15", not "BgeSmallEnV15".
        assert!(json.contains("bge_small_en_v15"), "json={json}");
        assert!(json.contains("test-model"));
        assert!(json.contains(&provenance.hash));

        let parsed: ModelProvenance = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model, provenance.model);
        assert_eq!(parsed.model_id, provenance.model_id);
        assert_eq!(parsed.hash, provenance.hash);
    }

    #[test]
    fn test_dimensions() {
        assert_eq!(EmbeddingModel::BgeSmallEnV15.dimensions(), 384);
        assert_eq!(EmbeddingModel::BgeBaseEnV15.dimensions(), 768);
        assert_eq!(EmbeddingModel::BgeLargeEnV15.dimensions(), 1024);
        assert_eq!(EmbeddingModel::Qwen3Embedding4B.dimensions(), 2560);
    }

    #[test]
    fn test_model_config_native_dims() {
        assert_eq!(
            ModelConfig::new(EmbeddingModel::Qwen3Embedding4B).dimensions(),
            2560
        );
        assert_eq!(
            ModelConfig::new(EmbeddingModel::Qwen3Embedding0_6B).dimensions(),
            1024
        );
        assert_eq!(
            ModelConfig::new(EmbeddingModel::BgeSmallEnV15).dimensions(),
            384
        );
    }

    #[test]
    fn test_model_config_configured_dim() {
        let cfg = ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, Some(1024)).unwrap();
        assert_eq!(cfg.dimensions(), 1024);

        let cfg = ModelConfig::try_new(EmbeddingModel::Qwen3Embedding0_6B, Some(512)).unwrap();
        assert_eq!(cfg.dimensions(), 512);
    }

    #[test]
    fn test_model_config_validation_below_min() {
        assert!(ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, Some(31)).is_err());
        assert!(ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, Some(0)).is_err());
    }

    #[test]
    fn test_model_config_validation_above_native() {
        assert!(ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, Some(2561)).is_err());
        assert!(ModelConfig::try_new(EmbeddingModel::Qwen3Embedding0_6B, Some(1025)).is_err());
    }

    #[test]
    fn test_model_config_validation_non_mrl_model() {
        assert!(ModelConfig::try_new(EmbeddingModel::BgeSmallEnV15, Some(128)).is_err());
        assert!(ModelConfig::try_new(EmbeddingModel::BgeBaseEnV15, Some(512)).is_err());
    }

    #[test]
    fn test_model_config_none_output_dim_ok_for_any_model() {
        assert!(ModelConfig::try_new(EmbeddingModel::BgeSmallEnV15, None).is_ok());
        assert!(ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, None).is_ok());
    }

    #[test]
    fn test_is_local() {
        assert!(EmbeddingModel::BgeSmallEnV15.is_local());
        assert!(EmbeddingModel::BgeBaseEnV15.is_local());
        assert!(EmbeddingModel::BgeLargeEnV15.is_local());
    }

    #[test]
    fn test_display() {
        assert_eq!(
            EmbeddingModel::BgeSmallEnV15.to_string(),
            "bge-small-en-v1.5"
        );
        assert_eq!(EmbeddingModel::BgeBaseEnV15.to_string(), "bge-base-en-v1.5");
        assert_eq!(
            EmbeddingModel::BgeLargeEnV15.to_string(),
            "bge-large-en-v1.5"
        );
    }

    #[test]
    fn test_serialization_roundtrip() {
        let model = EmbeddingModel::BgeSmallEnV15;
        let json = serde_json::to_string(&model).unwrap();
        let parsed: EmbeddingModel = serde_json::from_str(&json).unwrap();
        assert_eq!(model, parsed);
    }

    #[test]
    fn test_max_input_tokens() {
        assert_eq!(EmbeddingModel::BgeSmallEnV15.max_input_tokens(), 512);
        assert_eq!(EmbeddingModel::BgeBaseEnV15.max_input_tokens(), 512);
        assert_eq!(EmbeddingModel::BgeLargeEnV15.max_input_tokens(), 512);
    }

    #[test]
    fn test_from_str_display_names() {
        assert_eq!(
            "bge-small-en-v1.5".parse::<EmbeddingModel>().unwrap(),
            EmbeddingModel::BgeSmallEnV15
        );
        assert_eq!(
            "bge-base-en-v1.5".parse::<EmbeddingModel>().unwrap(),
            EmbeddingModel::BgeBaseEnV15
        );
        assert_eq!(
            "bge-large-en-v1.5".parse::<EmbeddingModel>().unwrap(),
            EmbeddingModel::BgeLargeEnV15
        );
    }

    #[test]
    fn test_from_str_short_names() {
        assert_eq!(
            "small".parse::<EmbeddingModel>().unwrap(),
            EmbeddingModel::BgeSmallEnV15
        );
        assert_eq!(
            "bge-base".parse::<EmbeddingModel>().unwrap(),
            EmbeddingModel::BgeBaseEnV15
        );
        assert_eq!(
            "LARGE".parse::<EmbeddingModel>().unwrap(), // case insensitive
            EmbeddingModel::BgeLargeEnV15
        );
    }

    #[test]
    fn test_from_str_huggingface_ids() {
        assert_eq!(
            "BAAI/bge-small-en-v1.5".parse::<EmbeddingModel>().unwrap(),
            EmbeddingModel::BgeSmallEnV15
        );
    }

    #[test]
    fn test_from_str_invalid() {
        let result = "unknown-model".parse::<EmbeddingModel>();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown embedding model"));
    }

    // -------------------------------------------------------------------------
    // bert_pooling() routing tests (P1-E3) — require `native` feature
    // -------------------------------------------------------------------------

    /// BGE small/base/large must use CLS pooling per their HF model cards.
    #[cfg(feature = "native")]
    #[test]
    fn test_bge_models_use_cls_pooling() {
        use lattice_inference::BertPooling;

        assert_eq!(
            EmbeddingModel::BgeSmallEnV15.bert_pooling(),
            Some(BertPooling::CLS),
            "BgeSmallEnV15 must use CLS pooling"
        );
        assert_eq!(
            EmbeddingModel::BgeBaseEnV15.bert_pooling(),
            Some(BertPooling::CLS),
            "BgeBaseEnV15 must use CLS pooling"
        );
        assert_eq!(
            EmbeddingModel::BgeLargeEnV15.bert_pooling(),
            Some(BertPooling::CLS),
            "BgeLargeEnV15 must use CLS pooling"
        );
    }

    /// E5 models must use mean pooling per their HF model cards.
    #[cfg(feature = "native")]
    #[test]
    fn test_e5_models_use_mean_pooling() {
        use lattice_inference::BertPooling;

        assert_eq!(
            EmbeddingModel::MultilingualE5Small.bert_pooling(),
            Some(BertPooling::Mean),
            "MultilingualE5Small must use mean pooling"
        );
        assert_eq!(
            EmbeddingModel::MultilingualE5Base.bert_pooling(),
            Some(BertPooling::Mean),
            "MultilingualE5Base must use mean pooling"
        );
    }

    /// MiniLM models must use mean pooling per sentence-transformers convention.
    #[cfg(feature = "native")]
    #[test]
    fn test_minilm_models_use_mean_pooling() {
        use lattice_inference::BertPooling;

        assert_eq!(
            EmbeddingModel::AllMiniLmL6V2.bert_pooling(),
            Some(BertPooling::Mean),
            "AllMiniLmL6V2 must use mean pooling"
        );
        assert_eq!(
            EmbeddingModel::ParaphraseMultilingualMiniLmL12V2.bert_pooling(),
            Some(BertPooling::Mean),
            "ParaphraseMultilingualMiniLmL12V2 must use mean pooling"
        );
    }

    /// Qwen and remote models return None — they have separate pooling paths.
    #[cfg(feature = "native")]
    #[test]
    fn test_non_bert_models_return_none_pooling() {
        assert_eq!(
            EmbeddingModel::Qwen3Embedding0_6B.bert_pooling(),
            None,
            "Qwen model must return None for bert_pooling()"
        );
        assert_eq!(
            EmbeddingModel::Qwen3Embedding4B.bert_pooling(),
            None,
            "Qwen model must return None for bert_pooling()"
        );
        assert_eq!(
            EmbeddingModel::TextEmbedding3Small.bert_pooling(),
            None,
            "Remote model must return None for bert_pooling()"
        );
    }

    /// BGE and E5 use DIFFERENT pooling strategies — this is the key correctness distinction.
    #[cfg(feature = "native")]
    #[test]
    fn test_bge_and_e5_use_different_pooling() {
        assert_ne!(
            EmbeddingModel::BgeSmallEnV15.bert_pooling(),
            EmbeddingModel::MultilingualE5Small.bert_pooling(),
            "BGE and E5 must use different pooling strategies"
        );
    }
}
