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

/// Stands in for a native service: every byte of the prepared text is one token and the
/// model keeps the first `COUNT_PROBE_LIMIT`.
const COUNT_PROBE_LIMIT: usize = 8;

#[derive(Default)]
struct CountProbe {
    embed_calls: AtomicUsize,
    count_calls: AtomicUsize,
    drop_a_count: bool,
}

#[async_trait]
impl EmbeddingService for CountProbe {
    async fn embed(&self, texts: &[String], _model: EmbeddingModel) -> Result<Vec<Vec<f32>>> {
        self.embed_calls.fetch_add(1, Ordering::Relaxed);
        Ok(texts.iter().map(|text| vec![text.len() as f32]).collect())
    }

    async fn embed_with_role_prevalidated(
        &self,
        texts: ValidatedTextBatch<'_>,
        model: EmbeddingModel,
        role: EmbeddingRole,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_calls.fetch_add(1, Ordering::Relaxed);
        let prepared = texts.to_owned_with_prefix(role.instruction(model));
        Ok(prepared
            .iter()
            .map(|text| vec![text.len() as f32])
            .collect())
    }

    async fn count_tokens(
        &self,
        texts: &[String],
        model: EmbeddingModel,
        role: EmbeddingRole,
    ) -> Result<Vec<crate::service::TokenCount>> {
        self.count_calls.fetch_add(1, Ordering::Relaxed);
        let prefix = role.instruction(model).unwrap_or("");
        let mut counts: Vec<_> = texts
            .iter()
            .map(|text| {
                let visible = prefix.len() + text.len();
                crate::service::TokenCount::new(visible, visible.min(COUNT_PROBE_LIMIT))
            })
            .collect();
        if self.drop_a_count {
            counts.pop();
        }
        Ok(counts)
    }

    fn supports_model(&self, _model: EmbeddingModel) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "count-probe"
    }
}

#[tokio::test(flavor = "current_thread")]
async fn count_tokens_forwards_to_the_wrapped_service_and_counts_the_role_instruction() {
    let inner = Arc::new(CountProbe::default());
    let service = CachedEmbeddingService::new(inner.clone(), 16);
    let model = EmbeddingModel::MultilingualE5Small;
    let texts = vec!["ab".to_string(), "abcdefghij".to_string()];

    let generic = service
        .count_tokens(&texts, model, EmbeddingRole::Generic)
        .await
        .expect("the cached wrapper must not fall through to the default error");
    assert_eq!(
        generic,
        vec![
            crate::service::TokenCount::new(2, 2),
            crate::service::TokenCount::new(10, 8)
        ]
    );
    assert!(!generic[0].truncated() && generic[1].truncated());

    let query = service
        .count_tokens(&texts, model, EmbeddingRole::Query)
        .await
        .expect("count_tokens");
    let prefix = "query: ".len();
    assert_eq!(query[0].before_truncation, 2 + prefix);
    assert_eq!(query[0].embedded, COUNT_PROBE_LIMIT);
    assert_eq!(inner.count_calls.load(Ordering::Relaxed), 2);
    assert_eq!(inner.embed_calls.load(Ordering::Relaxed), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn embed_with_report_matches_embed_with_role_and_keeps_the_vector_cache() {
    let inner = Arc::new(CountProbe::default());
    let service = CachedEmbeddingService::new(inner.clone(), 16);
    let model = EmbeddingModel::MultilingualE5Small;
    let texts = vec!["ab".to_string(), "abcdefghij".to_string()];

    let report = service
        .embed_with_report(&texts, model, EmbeddingRole::Query)
        .await
        .expect("embed_with_report");
    let plain = service
        .embed_with_role(&texts, model, EmbeddingRole::Query)
        .await
        .expect("embed_with_role");
    assert_eq!(report.embeddings, plain);
    assert_eq!(report.token_counts.len(), texts.len());
    assert_eq!(
        report.token_counts[1].before_truncation,
        10 + "query: ".len()
    );
    assert!(report.token_counts[1].truncated());
    // Vectors were computed once and the second call was served from the cache; the
    // counts are not cached, so the wrapped service was asked for them each time it was.
    assert_eq!(inner.embed_calls.load(Ordering::Relaxed), 1);
    assert_eq!(inner.count_calls.load(Ordering::Relaxed), 1);

    let again = service
        .embed_with_report(&texts, model, EmbeddingRole::Query)
        .await
        .expect("embed_with_report again");
    assert_eq!(again.embeddings, plain);
    assert_eq!(again.token_counts, report.token_counts);
    assert_eq!(inner.embed_calls.load(Ordering::Relaxed), 1);
    assert_eq!(inner.count_calls.load(Ordering::Relaxed), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn embed_with_report_rejects_a_count_per_input_mismatch() {
    let inner = Arc::new(CountProbe {
        drop_a_count: true,
        ..CountProbe::default()
    });
    let service = CachedEmbeddingService::new(inner, 16);
    let texts = vec!["ab".to_string(), "cd".to_string()];
    let err = service
        .embed_with_report(
            &texts,
            EmbeddingModel::BgeSmallEnV15,
            EmbeddingRole::Generic,
        )
        .await
        .expect_err("a short count list must not be returned as a report");
    assert!(matches!(err, EmbedError::InferenceFailed(_)), "got: {err}");
}

#[tokio::test(flavor = "current_thread")]
async fn token_methods_over_a_service_without_overrides_stay_unsupported() {
    let service = CachedEmbeddingService::new(Arc::new(ProbeService::default()), 16);
    let texts = vec!["ab".to_string()];
    let model = EmbeddingModel::BgeSmallEnV15;
    let err = service
        .count_tokens(&texts, model, EmbeddingRole::Generic)
        .await
        .expect_err("the wrapped service has no counts to forward");
    assert!(
        matches!(&err, EmbedError::Unsupported(m) if m.contains("cache-probe")),
        "got: {err}"
    );
    let err = service
        .embed_with_report(&texts, model, EmbeddingRole::Generic)
        .await
        .expect_err("no counts, no report");
    assert!(matches!(err, EmbedError::Unsupported(_)), "got: {err}");
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
