//! Serving interfaces for pooled embeddings.

use std::sync::Arc;

use crate::error::InferenceError;
use crate::forward::cpu_f16::PoolingStrategy;

/// Identifies the loader and vector width used to produce an embedding.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingRepresentation {
    /// Stable identifier for the checkpoint loader.
    pub loader_format: &'static str,
    /// Number of values in each embedding vector.
    pub dimensions: usize,
}

/// Token lengths before truncation and after tokenization.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenCounts {
    /// Number of tokens before the tokenizer's sequence cap is applied.
    pub pre_truncation: usize,
    /// Number of tokens consumed by the embedding computation.
    pub embedded: usize,
}

pub(crate) trait PooledEmbeddingService: Send + Sync {
    fn representation(&self) -> EmbeddingRepresentation;
    fn max_context(&self) -> usize;
    fn text_counts(&self, text: &str) -> TokenCounts;
    fn image_scaffold_len(&self, image: &[u8]) -> Result<usize, InferenceError>;
    fn embed_text(&self, text: &str, pooling: PoolingStrategy) -> Result<Vec<f32>, InferenceError>;
    fn embed_image(
        &self,
        image: &[u8],
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError>;
}

/// Shared access to a loaded pooled embedding service.
#[doc(hidden)]
#[derive(Clone)]
pub struct PooledEmbeddingHandle {
    inner: Arc<dyn PooledEmbeddingService>,
}

impl PooledEmbeddingHandle {
    pub(crate) fn new(inner: Arc<dyn PooledEmbeddingService>) -> Self {
        Self { inner }
    }

    /// Returns the loader identifier and vector width.
    pub fn representation(&self) -> EmbeddingRepresentation {
        self.inner.representation()
    }

    /// Returns the maximum accepted pooled input length.
    pub fn max_context(&self) -> usize {
        self.inner.max_context()
    }

    /// Returns token lengths before truncation and as embedded.
    pub fn text_counts(&self, text: &str) -> TokenCounts {
        self.inner.text_counts(text)
    }

    /// Counts the image scaffold tokens used by the pooled service.
    pub fn image_scaffold_len(&self, image: &[u8]) -> Result<usize, InferenceError> {
        self.inner.image_scaffold_len(image)
    }

    /// Computes a pooled text embedding.
    pub fn embed_text(
        &self,
        text: &str,
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError> {
        self.inner.embed_text(text, pooling)
    }

    /// Computes a pooled image embedding.
    pub fn embed_image(
        &self,
        image: &[u8],
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError> {
        self.inner.embed_image(image, pooling)
    }
}

impl PooledEmbeddingService for PooledEmbeddingHandle {
    fn representation(&self) -> EmbeddingRepresentation {
        self.inner.representation()
    }

    fn max_context(&self) -> usize {
        self.inner.max_context()
    }

    fn text_counts(&self, text: &str) -> TokenCounts {
        self.inner.text_counts(text)
    }

    fn image_scaffold_len(&self, image: &[u8]) -> Result<usize, InferenceError> {
        self.inner.image_scaffold_len(image)
    }

    fn embed_text(&self, text: &str, pooling: PoolingStrategy) -> Result<Vec<f32>, InferenceError> {
        self.inner.embed_text(text, pooling)
    }

    fn embed_image(
        &self,
        image: &[u8],
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError> {
        self.inner.embed_image(image, pooling)
    }
}
