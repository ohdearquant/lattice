//! Tests for embedding services.

use super::*;

#[test]
fn test_max_batch_size_constant() {
    assert_eq!(DEFAULT_MAX_BATCH_SIZE, 1000);
}

#[test]
fn test_max_text_chars_constant() {
    assert_eq!(MAX_TEXT_CHARS, 32768);
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
}
