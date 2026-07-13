//! Async embedding-service contract and native implementations.
//!
//! The trait defines generic, query, and passage embedding; native builds additionally expose
//! a lazy local-inference service and an LRU caching wrapper. See `docs/service.md` for the
//! lifecycle, prompt handling, validation rules, and cache behavior.

#[cfg(feature = "native")]
mod cached;
#[cfg(feature = "native")]
mod native;

#[cfg(test)]
mod tests;

use crate::error::{EmbedError, Result};
use crate::model::{EmbeddingModel, ModelConfig};
use async_trait::async_trait;

// Re-exports
#[cfg(feature = "native")]
pub use cached::CachedEmbeddingService;
#[cfg(feature = "native")]
pub use native::NativeEmbeddingService;

/// **Stable**: default maximum batch size to prevent OOM.
///
/// This limit prevents accidentally passing huge batches that could exhaust memory.
/// Can be overridden by using chunked calls if larger batches are needed.
pub const DEFAULT_MAX_BATCH_SIZE: usize = 1000;

/// **Stable**: maximum allowed text length in characters.
///
/// This limit prevents OOM attacks via extremely large input texts.
/// 32KB is sufficient for most embedding use cases while preventing abuse.
pub const MAX_TEXT_CHARS: usize = 32768;

/// **Stable**: role of text in asymmetric retrieval.
///
/// Models trained with asymmetric objectives (E5, Qwen3-Embedding) use different
/// prompt prefixes for queries vs documents.  Providing the wrong role causes the
/// embedding to land in the wrong region of the model's retrieval space, degrading
/// retrieval quality.
///
/// Use [`EmbeddingService::embed_query`] / [`EmbeddingService::embed_passage`] to
/// apply the correct prefix automatically.  The role is also included in the cache
/// key so that `embed_query("hello")` and `embed_passage("hello")` are stored as
/// separate entries even when the raw text is identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingRole {
    /// Query / question text — may receive a query-side prompt prefix.
    Query,
    /// Document / passage text — may receive a passage-side prompt prefix.
    Passage,
    /// Generic text with no role-specific prefix (backwards-compatible default).
    Generic,
}

impl EmbeddingRole {
    /// Short ASCII tag included in the cache key hash.
    ///
    /// Distinct strings ensure that role changes affect the Blake3 hash even
    /// when the raw text and model config are identical.
    #[inline]
    pub(crate) const fn cache_tag(self) -> &'static str {
        match self {
            EmbeddingRole::Query => "role:query",
            EmbeddingRole::Passage => "role:passage",
            EmbeddingRole::Generic => "role:generic",
        }
    }
}

/// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
///
/// Trait for embedding generation services.
///
/// This trait defines the interface for services that can convert text
/// into vector embeddings. Implementations may use local models (native Rust)
/// or remote APIs.
///
/// # Example
///
/// ```rust,no_run
/// use lattice_embed::{EmbeddingService, EmbeddingModel, NativeEmbeddingService};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let service = NativeEmbeddingService::default();
///     let embedding = service.embed_one("Hello, world!", EmbeddingModel::default()).await?;
///     assert_eq!(embedding.len(), 384);
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    /// **Stable**: generate embeddings for multiple texts.
    ///
    /// Returns a vector of embeddings, one for each input text, in the same order.
    /// Applies no role-specific prompt prefix (equivalent to `Generic` role).
    /// Use [`EmbeddingService::embed_query`] / [`EmbeddingService::embed_passage`]
    /// for asymmetric retrieval models.
    async fn embed(&self, texts: &[String], model: EmbeddingModel) -> Result<Vec<Vec<f32>>>;

    /// **Stable**: generate an embedding for a single text.
    ///
    /// This is a convenience method that calls `embed` with a single-element slice.
    async fn embed_one(&self, text: &str, model: EmbeddingModel) -> Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let mut embeddings = self.embed(&texts, model).await?;
        embeddings
            .pop()
            .ok_or_else(|| EmbedError::Internal("no embedding generated".into()))
    }

    /// **Stable**: embed query texts with model-specific query prompt prefix applied.
    ///
    /// For models that use asymmetric prompts (BGE, E5, Qwen3-Embedding), this prepends the
    /// `query_instruction()` prefix before calling the model forward.  For models with
    /// no query prefix (MiniLM), this is equivalent to `embed()`.
    ///
    /// Cache keys produced by this method are distinct from those produced by
    /// `embed_passage()` and `embed()` even when the raw text is identical.
    async fn embed_query(&self, texts: &[String], model: EmbeddingModel) -> Result<Vec<Vec<f32>>> {
        let prefix = model.query_instruction();
        let prompted = apply_prefix(texts, prefix);
        self.embed(&prompted, model).await
    }

    /// **Stable**: embed document/passage texts with model-specific document prompt prefix applied.
    ///
    /// For models that use asymmetric prompts (E5), this prepends the
    /// `document_instruction()` prefix before calling the model forward.  For models with
    /// no document prefix (BGE, MiniLM, Qwen3), this is equivalent to `embed()`.
    ///
    /// Cache keys produced by this method are distinct from those produced by
    /// `embed_query()` and `embed()` even when the raw text is identical.
    async fn embed_passage(
        &self,
        texts: &[String],
        model: EmbeddingModel,
    ) -> Result<Vec<Vec<f32>>> {
        let prefix = model.document_instruction();
        let prompted = apply_prefix(texts, prefix);
        self.embed(&prompted, model).await
    }

    /// **Unstable**: returns the effective `ModelConfig` for a given model on this service.
    ///
    /// The default returns a config with no MRL truncation. `NativeEmbeddingService`
    /// overrides this to expose the configured output dimension so `CachedEmbeddingService`
    /// can include the actual dimension in cache keys.
    fn model_config(&self, model: EmbeddingModel) -> ModelConfig {
        ModelConfig::new(model)
    }

    /// **Stable**: check if the service supports a given model.
    fn supports_model(&self, model: EmbeddingModel) -> bool;

    /// **Stable**: get the name/identifier of this service.
    fn name(&self) -> &'static str;
}

/// Apply an optional prompt prefix to each text.
///
/// Returns a new `Vec<String>` with the prefix prepended where the prefix is
/// `Some`, or a cloned vec of the original texts when the prefix is `None`.
/// This is a free function (not a method) so it can be called from default
/// trait method bodies without going through `self`.
pub(crate) fn apply_prefix(texts: &[String], prefix: Option<&str>) -> Vec<String> {
    match prefix {
        None => texts.to_vec(),
        Some(p) => texts.iter().map(|t| format!("{p}{t}")).collect(),
    }
}
