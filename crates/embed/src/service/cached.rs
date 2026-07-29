//! Native-only LRU wrapper for an [`EmbeddingService`].
//!
//! It preserves caller order across partial cache hits and uses role-aware keys for asymmetric
//! retrieval. See `docs/service.md` for the lookup and fill algorithm.

use super::{EmbeddingRole, EmbeddingService, ValidatedTextBatch};
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
mod tests {
    use super::*;
    use crate::error::EmbedError;
    use crate::service::{
        MAX_TEXT_BYTES, NativeEmbeddingService, reset_validate_texts_calls, validate_texts_calls,
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct ProbeService {
        calls: AtomicUsize,
        requests: Mutex<Vec<Vec<String>>>,
        fail: bool,
    }

    impl ProbeService {
        fn failing() -> Self {
            Self {
                fail: true,
                ..Self::default()
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }

        fn requests(&self) -> Vec<Vec<String>> {
            self.requests.lock().unwrap().clone()
        }

        fn record(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.requests.lock().unwrap().push(texts.to_vec());
            if self.fail {
                return Err(EmbedError::InferenceFailed("probe failure".into()));
            }
            Ok(texts.iter().map(|text| vec![text.len() as f32]).collect())
        }
    }

    #[async_trait]
    impl EmbeddingService for ProbeService {
        async fn embed(&self, texts: &[String], _model: EmbeddingModel) -> Result<Vec<Vec<f32>>> {
            self.record(texts)
        }

        async fn embed_with_role_prevalidated(
            &self,
            texts: ValidatedTextBatch<'_>,
            model: EmbeddingModel,
            role: EmbeddingRole,
        ) -> Result<Vec<Vec<f32>>> {
            let prepared = texts.to_owned_with_prefix(role.instruction(model));
            self.record(&prepared)
        }

        fn supports_model(&self, _model: EmbeddingModel) -> bool {
            true
        }

        fn name(&self) -> &'static str {
            "cache-probe"
        }
    }

    #[derive(Default)]
    struct BorrowProbe {
        saw_borrowed: AtomicBool,
    }

    #[async_trait]
    impl EmbeddingService for BorrowProbe {
        async fn embed(&self, _texts: &[String], _model: EmbeddingModel) -> Result<Vec<Vec<f32>>> {
            panic!("cache delegation must use the prevalidated hook")
        }

        async fn embed_with_role_prevalidated(
            &self,
            texts: ValidatedTextBatch<'_>,
            _model: EmbeddingModel,
            _role: EmbeddingRole,
        ) -> Result<Vec<Vec<f32>>> {
            self.saw_borrowed
                .store(texts.borrowed().is_some(), Ordering::Relaxed);
            Ok((0..texts.len())
                .map(|index| vec![texts.get(index).len() as f32])
                .collect())
        }

        fn supports_model(&self, _model: EmbeddingModel) -> bool {
            true
        }

        fn name(&self) -> &'static str {
            "borrow-probe"
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn disabled_delegate_validates_once_and_preserves_role_preparation() {
        let inner = Arc::new(ProbeService::default());
        let service = CachedEmbeddingService::new(inner.clone(), 0);
        let texts = vec!["hello".to_string()];
        let model = EmbeddingModel::BgeSmallEnV15;

        reset_validate_texts_calls();
        let result = service.embed_query(&texts, model).await.unwrap();

        assert_eq!(validate_texts_calls(), 1);
        assert_eq!(inner.calls(), 1);
        let requests = inner.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0][0],
            format!("{}hello", model.query_instruction().unwrap())
        );
        assert_eq!(result, vec![vec![requests[0][0].len() as f32]]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn miss_validates_once_and_delegates_only_uncached_texts() {
        let inner = Arc::new(ProbeService::default());
        let service = CachedEmbeddingService::new(inner.clone(), 128);
        let model = EmbeddingModel::AllMiniLmL6V2;

        reset_validate_texts_calls();
        service.embed(&["cached".to_string()], model).await.unwrap();
        assert_eq!(validate_texts_calls(), 1);

        let calls_before = inner.calls();
        reset_validate_texts_calls();
        let result = service
            .embed(&["cached".to_string(), "missing".to_string()], model)
            .await
            .unwrap();

        assert_eq!(validate_texts_calls(), 1);
        assert_eq!(inner.calls(), calls_before + 1);
        assert_eq!(
            inner.requests().last().unwrap(),
            &vec!["missing".to_string()]
        );
        assert_eq!(result, vec![vec![6.0], vec![7.0]]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn miss_delegates_borrowed_text_views_after_one_validation() {
        let inner = Arc::new(BorrowProbe::default());
        let service = CachedEmbeddingService::new(inner.clone(), 128);

        reset_validate_texts_calls();
        let result = service
            .embed(&["uncached".to_string()], EmbeddingModel::AllMiniLmL6V2)
            .await
            .unwrap();

        assert_eq!(validate_texts_calls(), 1);
        assert!(inner.saw_borrowed.load(Ordering::Relaxed));
        assert_eq!(result, vec![vec![8.0]]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn all_hit_validates_once_without_delegating() {
        let inner = Arc::new(ProbeService::default());
        let service = CachedEmbeddingService::new(inner.clone(), 128);
        let texts = vec!["cached".to_string()];
        let model = EmbeddingModel::AllMiniLmL6V2;

        service.embed(&texts, model).await.unwrap();
        let calls_before = inner.calls();

        reset_validate_texts_calls();
        let result = service.embed(&texts, model).await.unwrap();

        assert_eq!(validate_texts_calls(), 1);
        assert_eq!(inner.calls(), calls_before);
        assert_eq!(result, vec![vec![6.0]]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn invalid_input_validates_once_without_delegating() {
        let inner = Arc::new(ProbeService::default());
        let service = CachedEmbeddingService::new(inner.clone(), 128);
        let texts = vec!["x".repeat(MAX_TEXT_BYTES + 1)];

        reset_validate_texts_calls();
        let error = service
            .embed(&texts, EmbeddingModel::AllMiniLmL6V2)
            .await
            .unwrap_err();

        assert_eq!(validate_texts_calls(), 1);
        assert!(matches!(
            error,
            EmbedError::TextTooLong {
                max: MAX_TEXT_BYTES,
                ..
            }
        ));
        assert_eq!(inner.calls(), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn delegate_error_still_validates_once() {
        let inner = Arc::new(ProbeService::failing());
        let service = CachedEmbeddingService::new(inner.clone(), 0);

        reset_validate_texts_calls();
        let error = service
            .embed(&["valid".to_string()], EmbeddingModel::AllMiniLmL6V2)
            .await
            .unwrap_err();

        assert_eq!(validate_texts_calls(), 1);
        assert_eq!(inner.calls(), 1);
        assert!(matches!(error, EmbedError::InferenceFailed(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn nested_cache_delegate_preserves_single_validation() {
        let probe = Arc::new(ProbeService::default());
        let inner = Arc::new(CachedEmbeddingService::new(probe.clone(), 0));
        let service = CachedEmbeddingService::new(inner, 0);

        reset_validate_texts_calls();
        let result = service
            .embed_passage(&["nested".to_string()], EmbeddingModel::AllMiniLmL6V2)
            .await
            .unwrap();

        assert_eq!(validate_texts_calls(), 1);
        assert_eq!(probe.calls(), 1);
        assert_eq!(result, vec![vec![6.0]]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn native_delegate_preserves_model_validation_without_rescanning_caller_text() {
        let inner = Arc::new(NativeEmbeddingService::default());
        let service = CachedEmbeddingService::new(inner, 0);

        reset_validate_texts_calls();
        let error = service
            .embed(&["valid".to_string()], EmbeddingModel::BgeBaseEnV15)
            .await
            .unwrap_err();

        assert_eq!(validate_texts_calls(), 1);
        assert!(matches!(error, EmbedError::InvalidInput(_)));
    }
}
