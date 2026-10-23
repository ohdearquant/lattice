//! Native-only LRU wrapper for an [`EmbeddingService`].
//!
//! It preserves caller order across partial cache hits and uses role-aware keys for asymmetric
//! retrieval. See `docs/service.md` for the lookup and fill algorithm.

use super::{
    EmbeddingRole, EmbeddingService, EmbeddingsWithReport, TokenCount, ValidatedTextBatch,
};
use crate::error::Result;
use crate::model::EmbeddingModel;
use async_trait::async_trait;
use std::sync::Arc;
use tracing::debug;

/// **Unstable**: caching strategy and constructor API may change; foundation-internal use only.
///
/// LRU-caching wrapper around an embedding service.
///
/// It preserves input order while reusing embeddings with matching model configuration and role.
/// See [`docs/service.md`](../../docs/service.md#cachedembeddingservice-cache-hit-behavior) for the lookup and fill algorithm.
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
        // Generic has its own role tag — see docs/service.md.
        let texts = ValidatedTextBatch::new(texts)?;
        self.cache_and_embed(texts, model, EmbeddingRole::Generic)
            .await
    }

    /// Override: cache under the role key rather than prefixing here.
    ///
    /// `embed_query` and `embed_passage` reach this through their trait defaults,
    /// so all three role paths share one cache-aware implementation.
    async fn embed_with_role(
        &self,
        texts: &[String],
        model: EmbeddingModel,
        role: EmbeddingRole,
    ) -> Result<Vec<Vec<f32>>> {
        let texts = ValidatedTextBatch::new(texts)?;
        self.cache_and_embed(texts, model, role).await
    }

    async fn embed_with_role_prevalidated(
        &self,
        texts: ValidatedTextBatch<'_>,
        model: EmbeddingModel,
        role: EmbeddingRole,
    ) -> Result<Vec<Vec<f32>>> {
        self.cache_and_embed(texts, model, role).await
    }

    /// Forwarded to the wrapped service: token counts do not depend on the vector cache.
    async fn count_tokens(
        &self,
        texts: &[String],
        model: EmbeddingModel,
        role: EmbeddingRole,
    ) -> Result<Vec<TokenCount>> {
        self.inner.count_tokens(texts, model, role).await
    }

    /// Vectors come through the cache exactly as in `embed_with_role`; counts come from the
    /// wrapped service's `count_tokens`, which tokenizes the batch again. Counts are not
    /// cached, so even an all-hit request reaches the wrapped service for them.
    async fn embed_with_report(
        &self,
        texts: &[String],
        model: EmbeddingModel,
        role: EmbeddingRole,
    ) -> Result<EmbeddingsWithReport> {
        let token_counts = self.inner.count_tokens(texts, model, role).await?;
        let embeddings = self.embed_with_role(texts, model, role).await?;
        if token_counts.len() != embeddings.len() {
            return Err(crate::error::EmbedError::InferenceFailed(format!(
                "embedding service returned {} token counts for {} inputs",
                token_counts.len(),
                embeddings.len()
            )));
        }
        Ok(EmbeddingsWithReport::new(embeddings, token_counts))
    }

    fn supports_model(&self, model: EmbeddingModel) -> bool {
        self.inner.supports_model(model)
    }

    fn name(&self) -> &'static str {
        "cached-embedding"
    }
}

impl<S: EmbeddingService + 'static> CachedEmbeddingService<S> {
    /// Core cache-and-embed implementation shared by every entry point.
    ///
    /// `texts` is caller text. The retrieval instruction is applied by the wrapped
    /// service rather than here, so the published cap is checked against what the
    /// caller actually passed. `role` selects that instruction downstream and
    /// namespaces the cache key, so the same raw text under two roles never shares
    /// an entry; keying on caller text is equivalent to keying on prepared text
    /// because the instruction is a function of the role and model config already
    /// in the key.
    async fn cache_and_embed(
        &self,
        texts: ValidatedTextBatch<'_>,
        model: EmbeddingModel,
        role: EmbeddingRole,
    ) -> Result<Vec<Vec<f32>>> {
        use crate::error::EmbedError;

        // Fast path: bypass cache entirely when disabled (no key computation, no locking)
        if !self.cache.is_enabled() {
            return self
                .inner
                .embed_with_role_prevalidated(texts, model, role)
                .await;
        }

        // Compute cache keys — include the active dimension (for MRL models) and role.
        let model_config = self.inner.model_config(model);
        let keys: Vec<_> = (0..texts.len())
            .map(|index| self.cache.compute_key(texts.get(index), model_config, role))
            .collect();

        // Check cache for all texts — returns Arc<[f32]> refs (O(1) per hit)
        let cached = self.cache.get_many(&keys);

        let mut to_embed = Vec::new();
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];

        for (i, cached_emb) in cached.into_iter().enumerate() {
            if let Some(arc) = cached_emb {
                results[i] = Some(arc.to_vec());
            } else {
                to_embed.push(i);
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

        // Embed missing texts; the wrapped service applies the role instruction.
        let texts_to_embed: Vec<&str> = to_embed.iter().map(|&index| texts.get(index)).collect();
        let new_embeddings = self
            .inner
            .embed_with_role_prevalidated(texts.borrowed_subset(&texts_to_embed), model, role)
            .await?;

        // FP-035: validate count before zipping — a count mismatch would silently
        // drop slots via zip() and return fewer embeddings than requested.
        if new_embeddings.len() != to_embed.len() {
            return Err(EmbedError::InferenceFailed(format!(
                "embedding service returned {} vectors for {} inputs",
                new_embeddings.len(),
                to_embed.len()
            )));
        }

        let mut cache_entries = Vec::with_capacity(to_embed.len());
        for (i, embedding) in to_embed.into_iter().zip(new_embeddings.into_iter()) {
            cache_entries.push((keys[i], embedding.clone()));
            results[i] = Some(embedding);
        }
        self.cache.put_many(cache_entries);

        // SAFETY: All slots are guaranteed to be Some at this point:
        // - Cached items were assigned via results[i] = Some(arc.to_vec())
        // - Non-cached items were assigned via results[i] = Some(embedding) in the loop above
        Ok(results.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod tests;
