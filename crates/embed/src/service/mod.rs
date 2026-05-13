//! Embedding service trait and implementations.

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
