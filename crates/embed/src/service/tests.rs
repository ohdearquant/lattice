//! Tests for embedding services.

use super::*;
use crate::model::EmbeddingModel;

#[test]
fn test_max_batch_size_constant() {
    assert_eq!(DEFAULT_MAX_BATCH_SIZE, 1000);
}

#[test]
fn test_max_text_bytes_constant() {
    assert_eq!(MAX_TEXT_BYTES, 32768);
}

#[test]
#[allow(deprecated)]
fn test_max_text_chars_deprecated_alias_matches() {
    assert_eq!(MAX_TEXT_CHARS, MAX_TEXT_BYTES);
}

#[cfg(feature = "native")]
mod native_tests {
    use super::*;
    use crate::{EmbeddingModel, ModelConfig};

    /// Default service loads BgeSmallEnV15 — only that model is supported.
    #[test]
    fn test_native_service_supports_only_loaded_model() {
        let service = NativeEmbeddingService::default();
        assert!(service.supports_model(EmbeddingModel::BgeSmallEnV15));
        // All other models must be rejected even if they are local.
        assert!(!service.supports_model(EmbeddingModel::BgeBaseEnV15));
        assert!(!service.supports_model(EmbeddingModel::BgeLargeEnV15));
        assert!(!service.supports_model(EmbeddingModel::MultilingualE5Small));
        assert!(!service.supports_model(EmbeddingModel::MultilingualE5Base));
        assert!(!service.supports_model(EmbeddingModel::Qwen3Embedding0_6B));
        assert!(!service.supports_model(EmbeddingModel::Qwen3Embedding4B));
        assert!(!service.supports_model(EmbeddingModel::TextEmbedding3Small));
    }

    /// with_model() constructor — only the selected model is supported.
    #[test]
    fn test_native_service_with_model_supports_only_that_model() {
        let service = NativeEmbeddingService::with_model(EmbeddingModel::MultilingualE5Small);
        assert!(service.supports_model(EmbeddingModel::MultilingualE5Small));
        assert!(!service.supports_model(EmbeddingModel::BgeSmallEnV15));
    }

    #[test]
    fn test_native_service_name() {
        let service = NativeEmbeddingService::default();
        assert_eq!(service.name(), "native-bert");
    }

    #[test]
    fn test_native_service_with_model_config_qwen_default() {
        use crate::model::ModelConfig;
        let cfg = ModelConfig::new(EmbeddingModel::Qwen3Embedding4B);
        let service = NativeEmbeddingService::with_model_config(cfg).unwrap();
        assert!(service.supports_model(EmbeddingModel::Qwen3Embedding4B));
        assert!(!service.supports_model(EmbeddingModel::Qwen3Embedding0_6B));
    }

    #[test]
    fn test_native_service_with_model_config_invalid_dim_rejected() {
        use crate::model::ModelConfig;
        // BgeSmallEnV15 does not support MRL — output_dim must be rejected at construction.
        let cfg = ModelConfig {
            model: EmbeddingModel::BgeSmallEnV15,
            output_dim: Some(128),
        };
        assert!(NativeEmbeddingService::with_model_config(cfg).is_err());
    }

    #[test]
    fn test_native_service_model_config_returns_configured_dim() {
        let cfg = ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, Some(1024)).unwrap();
        let service = NativeEmbeddingService::with_model_config(cfg).unwrap();
        let returned = service.model_config(EmbeddingModel::Qwen3Embedding4B);
        assert_eq!(returned.output_dim, Some(1024));
        assert_eq!(returned.dimensions(), 1024);
    }

    #[test]
    fn test_native_service_model_config_unknown_model_returns_native() {
        let service = NativeEmbeddingService::default(); // BgeSmallEnV15
        let returned = service.model_config(EmbeddingModel::BgeBaseEnV15);
        assert_eq!(returned.output_dim, None);
        assert_eq!(returned.dimensions(), 768);
    }

    // Mutex to serialize all tests that read or write LATTICE_EMBED_DIM.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// LATTICE_EMBED_DIM absent → with_model_from_env returns native dim (2560 for 4B).
    #[test]
    fn test_with_model_from_env_absent_returns_native_dim() {
        let _g = ENV_MUTEX.lock().unwrap();
        // SAFETY: ENV_MUTEX serialises all env-var mutations in this test binary.
        unsafe { std::env::remove_var("LATTICE_EMBED_DIM") };
        let svc =
            NativeEmbeddingService::with_model_from_env(EmbeddingModel::Qwen3Embedding4B).unwrap();
        assert_eq!(
            svc.model_config(EmbeddingModel::Qwen3Embedding4B)
                .output_dim,
            None
        );
        assert_eq!(
            svc.model_config(EmbeddingModel::Qwen3Embedding4B)
                .dimensions(),
            2560
        );
    }

    /// LATTICE_EMBED_DIM=1024 → with_model_from_env stores output_dim=Some(1024).
    #[test]
    fn test_with_model_from_env_dim_1024() {
        let _g = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialised by ENV_MUTEX.
        unsafe { std::env::set_var("LATTICE_EMBED_DIM", "1024") };
        let result = NativeEmbeddingService::with_model_from_env(EmbeddingModel::Qwen3Embedding4B);
        unsafe { std::env::remove_var("LATTICE_EMBED_DIM") };
        let svc = result.unwrap();
        let cfg = svc.model_config(EmbeddingModel::Qwen3Embedding4B);
        assert_eq!(cfg.output_dim, Some(1024));
        assert_eq!(cfg.dimensions(), 1024);
    }

    /// LATTICE_EMBED_DIM=512 → Qwen3Embedding0_6B works (512 < 1024 native).
    #[test]
    fn test_with_model_from_env_qwen_06b_dim_512() {
        let _g = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialised by ENV_MUTEX.
        unsafe { std::env::set_var("LATTICE_EMBED_DIM", "512") };
        let result =
            NativeEmbeddingService::with_model_from_env(EmbeddingModel::Qwen3Embedding0_6B);
        unsafe { std::env::remove_var("LATTICE_EMBED_DIM") };
        let svc = result.unwrap();
        let cfg = svc.model_config(EmbeddingModel::Qwen3Embedding0_6B);
        assert_eq!(cfg.output_dim, Some(512));
        assert_eq!(cfg.dimensions(), 512);
    }

    /// LATTICE_EMBED_DIM=not_a_number → error.
    #[test]
    fn test_with_model_from_env_invalid_value_returns_error() {
        let _g = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialised by ENV_MUTEX.
        unsafe { std::env::set_var("LATTICE_EMBED_DIM", "not_a_number") };
        let result = NativeEmbeddingService::with_model_from_env(EmbeddingModel::Qwen3Embedding4B);
        unsafe { std::env::remove_var("LATTICE_EMBED_DIM") };
        assert!(result.is_err(), "non-numeric LATTICE_EMBED_DIM must fail");
    }

    /// LATTICE_EMBED_DIM=16 → error because 16 < MIN_MRL_OUTPUT_DIM (32).
    #[test]
    fn test_with_model_from_env_dim_below_minimum_returns_error() {
        let _g = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialised by ENV_MUTEX.
        unsafe { std::env::set_var("LATTICE_EMBED_DIM", "16") };
        let result = NativeEmbeddingService::with_model_from_env(EmbeddingModel::Qwen3Embedding4B);
        unsafe { std::env::remove_var("LATTICE_EMBED_DIM") };
        assert!(result.is_err(), "dim < 32 must be rejected");
    }

    /// LATTICE_EMBED_DIM=9999 → error because 9999 > 2560 (native dim for 4B).
    #[test]
    fn test_with_model_from_env_dim_above_native_returns_error() {
        let _g = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialised by ENV_MUTEX.
        unsafe { std::env::set_var("LATTICE_EMBED_DIM", "9999") };
        let result = NativeEmbeddingService::with_model_from_env(EmbeddingModel::Qwen3Embedding4B);
        unsafe { std::env::remove_var("LATTICE_EMBED_DIM") };
        assert!(result.is_err(), "dim > native must be rejected");
    }

    /// LATTICE_EMBED_DIM="" (empty string) → treated as absent → native dim.
    #[test]
    fn test_with_model_from_env_empty_string_treated_as_absent() {
        let _g = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialised by ENV_MUTEX.
        unsafe { std::env::set_var("LATTICE_EMBED_DIM", "") };
        let result = NativeEmbeddingService::with_model_from_env(EmbeddingModel::Qwen3Embedding4B);
        unsafe { std::env::remove_var("LATTICE_EMBED_DIM") };
        let svc = result.unwrap();
        assert_eq!(
            svc.model_config(EmbeddingModel::Qwen3Embedding4B)
                .output_dim,
            None
        );
    }

    /// embed() must return an error when the requested model differs from the loaded one.
    #[tokio::test]
    async fn test_native_service_embed_wrong_model_returns_error() {
        let service = NativeEmbeddingService::default(); // loads BgeSmallEnV15
        let texts = vec!["hello".to_string()];
        // Requesting a different local model should fail before any IO.
        let result = service.embed(&texts, EmbeddingModel::BgeBaseEnV15).await;
        assert!(result.is_err(), "expected error for wrong model");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("BgeBaseEnV15") || err.contains("requested model"),
            "error should mention the model mismatch, got: {err}"
        );
    }

    /// #1010 repro shape 1: an all-ASCII text where char count == byte count.
    /// Rejected under either a byte or a char guard, so this alone would not
    /// have caught the bug — kept alongside the multibyte case for coverage.
    #[tokio::test]
    async fn test_native_service_ascii_text_too_long_reports_bytes() {
        let text = "a".repeat(36_556);
        assert_eq!(text.len(), 36_556);
        assert_eq!(text.chars().count(), 36_556);

        let service = NativeEmbeddingService::default();
        let texts = vec![text];
        let err = service
            .embed(&texts, EmbeddingModel::BgeSmallEnV15)
            .await
            .expect_err("text over MAX_TEXT_BYTES must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("36556"), "got: {msg}");
        assert!(msg.contains("bytes"), "got: {msg}");
    }

    /// #1010 repro shape 2 (discriminating case): 32,768 chars / 32,846 bytes —
    /// under char count, this is exactly at the limit and would be ADMITTED;
    /// under byte count (the guard's actual behavior), it exceeds the limit
    /// and must be REJECTED. If the guard is ever switched to `chars().count()`,
    /// this test fails.
    #[tokio::test]
    async fn test_native_service_multibyte_text_too_long_uses_byte_count() {
        // 78 two-byte chars + 32,690 one-byte chars = 32,768 chars / 32,846 bytes.
        let text = format!("{}{}", "é".repeat(78), "a".repeat(32_690));
        assert_eq!(text.chars().count(), 32_768);
        assert_eq!(text.len(), 32_846);

        let service = NativeEmbeddingService::default();
        let texts = vec![text];
        let err = service
            .embed(&texts, EmbeddingModel::BgeSmallEnV15)
            .await
            .expect_err("32846-byte text must be rejected under byte semantics");
        let msg = err.to_string();
        assert!(msg.contains("32846"), "got: {msg}");
        assert!(
            !msg.contains("32768 bytes exceeds") && !msg.contains("32768 chars exceeds"),
            "reported length must be the byte count (32846), not the char count: {msg}"
        );
    }

    /// Same discriminating shape, through the `CachedEmbeddingService` wrapper —
    /// a sibling invocation path with its own copy of the length guard.
    #[tokio::test]
    async fn test_cached_service_multibyte_text_too_long_uses_byte_count() {
        use crate::service::CachedEmbeddingService;
        use std::sync::Arc;

        let text = format!("{}{}", "é".repeat(78), "a".repeat(32_690));
        assert_eq!(text.chars().count(), 32_768);
        assert_eq!(text.len(), 32_846);

        let service =
            CachedEmbeddingService::with_default_cache(Arc::new(NativeEmbeddingService::default()));
        let texts = vec![text];
        let err = service
            .embed(&texts, EmbeddingModel::BgeSmallEnV15)
            .await
            .expect_err("32846-byte text must be rejected under byte semantics");
        let msg = err.to_string();
        assert!(msg.contains("32846"), "got: {msg}");
    }

    /// #1104: the published cap covers caller text, so text sitting exactly at
    /// the cap must clear validation on a role path even though the model's
    /// query instruction makes the prepared string longer than the cap.
    ///
    /// Model weights are not present in every environment, so this asserts the
    /// request got PAST length validation rather than asserting success. That is
    /// the precise thing that regressed: validating prepared text returned
    /// `TextTooLong` here and never reached model loading.
    #[tokio::test]
    async fn test_role_path_validates_caller_text_not_prepared_text() {
        let model = EmbeddingModel::BgeSmallEnV15;
        assert!(
            model.max_instruction_bytes() > 0,
            "test needs a model that prepends a query instruction"
        );

        let text = "a".repeat(MAX_TEXT_BYTES);
        assert_eq!(text.len(), MAX_TEXT_BYTES);

        let service = NativeEmbeddingService::default();
        if let Err(e) = service.embed_query(&[text], model).await {
            assert!(
                !matches!(e, crate::error::EmbedError::TextTooLong { .. }),
                "caller text at exactly the cap must not be rejected for length: {e}"
            );
        }
    }

    /// #1104 companion: the cap is still enforced, and the error still reports
    /// the published cap rather than the wider backstop used on prepared text.
    #[tokio::test]
    async fn test_role_path_still_rejects_caller_text_over_the_cap() {
        let text = "a".repeat(MAX_TEXT_BYTES + 1);
        let service = NativeEmbeddingService::default();
        let err = service
            .embed_query(&[text], EmbeddingModel::BgeSmallEnV15)
            .await
            .expect_err("caller text over the cap must be rejected");
        assert!(
            matches!(err, crate::error::EmbedError::TextTooLong { max, .. } if max == MAX_TEXT_BYTES),
            "must report the published cap, got: {err}"
        );
    }

    /// Sibling invocation path: the cached wrapper has its own entry point and
    /// must apply the same caller-text semantics.
    #[tokio::test]
    async fn test_cached_role_path_validates_caller_text() {
        use crate::service::CachedEmbeddingService;
        use std::sync::Arc;

        let text = "a".repeat(MAX_TEXT_BYTES);
        let service =
            CachedEmbeddingService::with_default_cache(Arc::new(NativeEmbeddingService::default()));

        if let Err(e) = service
            .embed_query(&[text], EmbeddingModel::BgeSmallEnV15)
            .await
        {
            assert!(
                !matches!(e, crate::error::EmbedError::TextTooLong { .. }),
                "caller text at exactly the cap must not be rejected for length: {e}"
            );
        }
    }
}

#[test]
fn test_max_instruction_bytes_matches_the_longest_instruction() {
    for model in [
        EmbeddingModel::BgeSmallEnV15,
        EmbeddingModel::MultilingualE5Small,
        EmbeddingModel::Qwen3Embedding0_6B,
        EmbeddingModel::AllMiniLmL6V2,
    ] {
        let q = model.query_instruction().map_or(0, str::len);
        let d = model.document_instruction().map_or(0, str::len);
        assert_eq!(
            model.max_instruction_bytes(),
            q.max(d),
            "{model:?} reports the wrong instruction bound"
        );
    }

    // Symmetric models take no instruction, so prepared text is caller text and
    // the backstop collapses onto the published cap exactly.
    assert_eq!(EmbeddingModel::AllMiniLmL6V2.max_instruction_bytes(), 0);
}

// ---------------------------------------------------------------------------
// Role-aware prompt tests (P0-E2)
// ---------------------------------------------------------------------------

/// E5 query_instruction returns "query: ", document_instruction returns "passage: ".
#[test]
fn test_e5_query_instruction() {
    assert_eq!(
        EmbeddingModel::MultilingualE5Small.query_instruction(),
        Some("query: "),
        "E5 small must return 'query: ' prefix"
    );
    assert_eq!(
        EmbeddingModel::MultilingualE5Base.query_instruction(),
        Some("query: "),
        "E5 base must return 'query: ' prefix"
    );
}

/// E5 document_instruction returns "passage: " (P0-E2 fix — was None before).
#[test]
fn test_e5_document_instruction() {
    assert_eq!(
        EmbeddingModel::MultilingualE5Small.document_instruction(),
        Some("passage: "),
        "E5 small must return 'passage: ' document prefix"
    );
    assert_eq!(
        EmbeddingModel::MultilingualE5Base.document_instruction(),
        Some("passage: "),
        "E5 base must return 'passage: ' document prefix"
    );
}

/// BGE-v1.5 query_instruction returns the documented retrieval instruction
/// (asymmetric retrieval — queries get the prefix, passages do not). Guards
/// against a future over-correction that adds the prefix to the passage side.
#[test]
fn test_bge_query_instruction() {
    assert_eq!(
        EmbeddingModel::BgeSmallEnV15.query_instruction(),
        Some("Represent this sentence for searching relevant passages: "),
        "BGE small must return the retrieval query instruction"
    );
    assert_eq!(
        EmbeddingModel::BgeBaseEnV15.query_instruction(),
        Some("Represent this sentence for searching relevant passages: "),
        "BGE base must return the retrieval query instruction"
    );
    assert_eq!(
        EmbeddingModel::BgeLargeEnV15.query_instruction(),
        Some("Represent this sentence for searching relevant passages: "),
        "BGE large must return the retrieval query instruction"
    );
    assert_eq!(
        EmbeddingModel::BgeSmallEnV15.document_instruction(),
        None,
        "BGE passages must stay unprefixed"
    );
}

/// BGE and MiniLM models must NOT have document_instruction (they use raw text).
#[test]
fn test_bge_minilm_no_document_instruction() {
    assert_eq!(
        EmbeddingModel::BgeSmallEnV15.document_instruction(),
        None,
        "BGE small must not have document prefix"
    );
    assert_eq!(EmbeddingModel::BgeBaseEnV15.document_instruction(), None);
    assert_eq!(EmbeddingModel::BgeLargeEnV15.document_instruction(), None);
    assert_eq!(EmbeddingModel::AllMiniLmL6V2.document_instruction(), None);
    assert_eq!(
        EmbeddingModel::ParaphraseMultilingualMiniLmL12V2.document_instruction(),
        None
    );
}

/// Qwen document_instruction returns None (raw passage, instruction only for queries).
#[test]
fn test_qwen_no_document_instruction() {
    assert_eq!(
        EmbeddingModel::Qwen3Embedding0_6B.document_instruction(),
        None,
        "Qwen document side uses raw text"
    );
}

/// apply_prefix prepends the prefix when Some, returns clone when None.
#[test]
fn test_apply_prefix_some() {
    let texts = vec!["hello".to_string(), "world".to_string()];
    let result = apply_prefix(&texts, Some("query: "));
    assert_eq!(result, vec!["query: hello", "query: world"]);
}

#[test]
fn test_apply_prefix_none() {
    let texts = vec!["hello".to_string()];
    let result = apply_prefix(&texts, None);
    assert_eq!(result, texts);
}

/// EmbeddingRole cache tags are distinct strings.
#[test]
fn test_embedding_role_cache_tags_distinct() {
    let q = EmbeddingRole::Query.cache_tag();
    let p = EmbeddingRole::Passage.cache_tag();
    let g = EmbeddingRole::Generic.cache_tag();
    assert_ne!(q, p);
    assert_ne!(q, g);
    assert_ne!(p, g);
}
