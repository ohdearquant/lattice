//! Caching wrapper for embedding services.

use super::{DEFAULT_MAX_BATCH_SIZE, EmbeddingService, MAX_TEXT_CHARS};
use crate::error::Result;
use crate::model::EmbeddingModel;
use async_trait::async_trait;
use std::sync::Arc;
use tracing::debug;

/// **Unstable**: caching strategy and constructor API may change; foundation-internal use only.
///
/// Caching wrapper around an embedding service.
///
/// Wraps any `EmbeddingService` implementation with LRU caching. Identical
/// texts (with the same model) will return cached embeddings instead of
/// recomputing.
///
/// # Example
///
/// ```rust,no_run
/// use lattice_embed::{
///     CachedEmbeddingService, NativeEmbeddingService, EmbeddingService,
///     EmbeddingModel, EmbeddingCache,
/// };
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let inner = Arc::new(NativeEmbeddingService::new());
///     let cached = CachedEmbeddingService::new(inner, 1000);
///
///     // First call - computes and caches
///     let emb1 = cached.embed_one("Hello", EmbeddingModel::default()).await?;
///
///     // Second call - returns from cache
///     let emb2 = cached.embed_one("Hello", EmbeddingModel::default()).await?;
///
///     assert_eq!(emb1, emb2);
///     Ok(())
/// }
/// ```
pub struct CachedEmbeddingService<S> {
    inner: Arc<S>,
    cache: crate::cache::EmbeddingCache,
}

impl<S: EmbeddingService> CachedEmbeddingService<S> {
    /// **Unstable**: constructor signature may change when cache config becomes a struct.
    ///
    /// # Arguments
    ///
    /// * `inner` - The underlying embedding service
    /// * `cache_capacity` - Maximum number of embeddings to cache
    pub fn new(inner: Arc<S>, cache_capacity: usize) -> Self {
        Self {
            inner,
            cache: crate::cache::EmbeddingCache::new(cache_capacity),
        }
    }

    /// **Unstable**: constructor signature may change when cache config becomes a struct.
    pub fn with_default_cache(inner: Arc<S>) -> Self {
        Self {
            inner,
            cache: crate::cache::EmbeddingCache::with_default_capacity(),
        }
    }

    /// **Unstable**: returns internal `CacheStats` type which is itself Unstable.
    pub fn cache_stats(&self) -> crate::cache::CacheStats {
        self.cache.stats()
    }

    /// **Unstable**: internal cache management; API subject to change.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

#[async_trait]
impl<S: EmbeddingService + 'static> EmbeddingService for CachedEmbeddingService<S> {
    async fn embed(&self, texts: &[String], model: EmbeddingModel) -> Result<Vec<Vec<f32>>> {
        use crate::error::EmbedError;

        // Validate inputs before any cache interaction so callers always get
        // consistent errors regardless of whether the result is fully cached.
        if texts.is_empty() {
            return Err(EmbedError::InvalidInput("no texts provided".into()));
        }
        if texts.len() > DEFAULT_MAX_BATCH_SIZE {
            return Err(EmbedError::InvalidInput(format!(
                "batch size {} exceeds maximum {}",
                texts.len(),
                DEFAULT_MAX_BATCH_SIZE
            )));
        }
        for text in texts {
            if text.len() > MAX_TEXT_CHARS {
                return Err(EmbedError::TextTooLong {
                    length: text.len(),
                    max: MAX_TEXT_CHARS,
                });
            }
        }

        // Fast path: bypass cache entirely when disabled (no key computation, no locking)
        if !self.cache.is_enabled() {
            return self.inner.embed(texts, model).await;
        }

        // Compute cache keys — include the active dimension (for MRL models).
        let model_config = self.inner.model_config(model);
        let keys: Vec<_> = texts
            .iter()
            .map(|t| self.cache.compute_key(t, model_config))
            .collect();

        // Check cache for all texts — returns Arc<[f32]> refs (O(1) per hit)
        let cached = self.cache.get_many(&keys);

        // Identify which texts need embedding
        let mut to_embed: Vec<(usize, &String)> = Vec::new();
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];

        for (i, (text, cached_emb)) in texts.iter().zip(cached.into_iter()).enumerate() {
            if let Some(arc) = cached_emb {
                results[i] = Some(arc.to_vec());
            } else {
                to_embed.push((i, text));
            }
        }

        // If all cached, return immediately
        if to_embed.is_empty() {
            debug!("all {} texts found in cache", texts.len());
            // SAFETY: All slots are Some because we only reach here when to_embed is empty,
            // meaning every text was found in cache and had results[i] = Some(...) assigned.
            return Ok(results.into_iter().flatten().collect());
        }

        debug!(
            "{} texts cached, {} need embedding",
            texts.len() - to_embed.len(),
            to_embed.len()
        );

        // Embed missing texts
        let texts_to_embed: Vec<String> = to_embed.iter().map(|(_, t)| (*t).clone()).collect();
        let new_embeddings = self.inner.embed(&texts_to_embed, model).await?;

        // FP-035: validate count before zipping — a count mismatch would silently
        // drop slots via zip() and return fewer embeddings than requested.
        if new_embeddings.len() != to_embed.len() {
            return Err(EmbedError::InferenceFailed(format!(
                "embedding service returned {} vectors for {} inputs",
                new_embeddings.len(),
                to_embed.len()
            )));
        }

        // Store in cache and populate results
        let mut cache_entries = Vec::with_capacity(to_embed.len());
        for ((i, _), embedding) in to_embed.into_iter().zip(new_embeddings.into_iter()) {
            cache_entries.push((keys[i], embedding.clone()));
            results[i] = Some(embedding);
        }
        self.cache.put_many(cache_entries);

        // Return all results
        // SAFETY: All slots are guaranteed to be Some at this point:
        // - Cached items were assigned via results[i] = Some(arc.to_vec())
        // - Non-cached items were assigned via results[i] = Some(embedding) in the loop above
        Ok(results.into_iter().flatten().collect())
    }

    fn supports_model(&self, model: EmbeddingModel) -> bool {
        self.inner.supports_model(model)
    }

    fn name(&self) -> &'static str {
        "cached-embedding"
    }
}

// Suppress dead code warnings for constants that are used by other modules
const _: () = {
    let _ = DEFAULT_MAX_BATCH_SIZE;
    let _ = MAX_TEXT_CHARS;
};
