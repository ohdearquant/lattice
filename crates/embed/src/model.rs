//! Embedding model definitions.
//!
//! Provides `EmbeddingModel` enum for local model selection.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
///
/// Model provenance information for security audits.
///
/// Tracks metadata about when and how a model was loaded, including a hash
/// for verification that the model hasn't been tampered with.
///
/// # Example
///
/// ```rust
/// use lattice_embed::{EmbeddingModel, ModelProvenance};
///
/// // Created when a model is loaded
/// let provenance = ModelProvenance::new(
///     EmbeddingModel::BgeSmallEnV15,
///     "BAAI/bge-small-en-v1.5".to_string(),
/// );
///
/// assert!(provenance.model_id.contains("BAAI"));
/// assert!(!provenance.hash.is_empty());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvenance {
    /// **Stable**: model variant that was loaded.
    pub model: EmbeddingModel,
    /// **Stable**: source identifier (HuggingFace ID, URL, or file path).
    pub model_id: String,
    /// **Stable**: Blake3 hash of the model identifier + timestamp for uniqueness.
    ///
    /// Note: This is a lightweight hash based on metadata, not a full hash
    /// of model weights (which would be expensive). For full model verification,
    /// use the lattice-inference library's built-in checksum verification.
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
/// Supported embedding models.
///
/// This enum represents the embedding models available for text vectorization.
/// Models are categorized as either local (run on-device via lattice-inference) or
/// remote (require API calls).
///
/// # Example
///
/// ```rust
/// use lattice_embed::EmbeddingModel;
///
/// let model = EmbeddingModel::default();
/// assert_eq!(model.dimensions(), 384);
/// assert!(model.is_local());
/// ```
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

    /// **Stable**: get the output dimension of this model's embeddings.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lattice_embed::EmbeddingModel;
    ///
    /// assert_eq!(EmbeddingModel::BgeSmallEnV15.dimensions(), 384);
    /// assert_eq!(EmbeddingModel::BgeBaseEnV15.dimensions(), 768);
    /// assert_eq!(EmbeddingModel::BgeLargeEnV15.dimensions(), 1024);
    /// ```
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

    /// **Stable**: maximum input tokens supported by this model.
    ///
    /// Use this for chunking/truncation decisions. Values are conservative
    /// to leave room for special tokens.
    ///
    /// Reference limits:
    /// - BGE models: 512 tokens
    /// - OpenAI text-embedding-3: 8191 tokens
    /// - Gemini embedding-001: 20000 tokens
    #[inline]
    pub const fn max_input_tokens(&self) -> usize {
        match self {
            // BGE models have 512 token limit
            EmbeddingModel::BgeSmallEnV15 => 512,
            EmbeddingModel::BgeBaseEnV15 => 512,
            EmbeddingModel::BgeLargeEnV15 => 512,
            // E5 models have 512 token limit
            EmbeddingModel::MultilingualE5Small => 512,
            EmbeddingModel::MultilingualE5Base => 512,
            // MiniLM has a shorter context window
            EmbeddingModel::AllMiniLmL6V2 => 256,
            // paraphrase-multilingual-MiniLM max sequence length 128
            EmbeddingModel::ParaphraseMultilingualMiniLmL12V2 => 128,
            // Qwen3-Embedding supports 32K but we cap at 8192 for practical use
            EmbeddingModel::Qwen3Embedding0_6B => 8192,
            EmbeddingModel::Qwen3Embedding4B => 8192,
            // OpenAI text-embedding-3-small has 8191 token limit
            EmbeddingModel::TextEmbedding3Small => 8191,
        }
    }

    /// **Stable**: query instruction prefix for asymmetric retrieval.
    ///
    /// Some models require different text for queries vs documents (asymmetric retrieval).
    ///
    /// - **E5 models** (`MultilingualE5Small`, `MultilingualE5Base`): trained with
    ///   "query: " / "passage: " asymmetric prefixes. Omitting the prefix degrades
    ///   retrieval quality significantly — the model expects them during fine-tuning.
    ///
    /// - **Qwen3-Embedding** models: require an instruction prompt to align the
    ///   decoder embedding space for retrieval tasks.
    ///
    /// - **BGE / MiniLM** models: trained with contrastive objectives on raw text;
    ///   no prefix needed.
    ///
    /// Returns `Some(prefix)` if the query text should be wrapped as
    /// `"{prefix}{query}"` before embedding. Returns `None` for models that
    /// don't need instruction prompting.
    #[inline]
    pub const fn query_instruction(&self) -> Option<&'static str> {
        match self {
            EmbeddingModel::MultilingualE5Small | EmbeddingModel::MultilingualE5Base => {
                // E5 asymmetric retrieval: "query: " prefix for queries,
                // "passage: " prefix for documents (see document_instruction()).
                Some("query: ")
            }
            EmbeddingModel::Qwen3Embedding0_6B | EmbeddingModel::Qwen3Embedding4B => Some(
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            ),
            _ => None,
        }
    }

    /// **Stable**: document instruction prefix for asymmetric retrieval.
    ///
    /// Some models use different prompts for documents vs queries.
    /// Returns `Some(prefix)` if the document text should be wrapped as
    /// `"{prefix}{text}"` before embedding at storage time.
    #[inline]
    pub const fn document_instruction(&self) -> Option<&'static str> {
        None
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

    /// **Stable**: parse model from string (case-insensitive, flexible matching).
    ///
    /// Accepts:
    /// - Display names: "bge-small-en-v1.5"
    /// - Short names: "bge-small", "small"
    /// - HuggingFace IDs: "BAAI/bge-small-en-v1.5"
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

/// Runtime configuration pairing a model with an optional MRL truncation dimension.
///
/// Two `ModelConfig` values with different `output_dim` produce different embedding spaces
/// and must be stored in separate vector index namespaces.
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
}
