//! Errors returned by embedding, model configuration, and prepared SIMD-dispatch operations.
//!
//! See `docs/service.md` for error boundaries and caller recovery guidance.

use thiserror::Error;

/// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
///
/// Errors that can occur during embedding operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum EmbedError {
    /// Model not loaded (needs initialization).
    #[error("model not loaded: {0}")]
    ModelNotLoaded(String),

    /// Wrong model loaded (concurrent model switch in progress).
    ///
    /// This can happen when multiple tasks request different models concurrently.
    /// The caller should retry with backoff.
    #[error("wrong model loaded: expected {expected}, got {actual}")]
    WrongModelLoaded {
        /// Model that was expected.
        expected: String,
        /// Model that was actually loaded.
        actual: String,
    },

    /// Model initialization failed.
    #[error("model initialization failed: {0}")]
    ModelInitialization(String),

    /// Embedding inference failed.
    #[error("embedding inference failed: {0}")]
    InferenceFailed(String),

    /// Blocking task failed (panic or cancellation).
    ///
    /// The model cache may be lost; next call will reinitialize.
    #[error("task execution failed: {0}")]
    TaskFailed(String),

    /// Invalid input provided.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Input text exceeds maximum allowed length.
    #[error("text too long: {length} chars exceeds maximum {max} chars")]
    TextTooLong {
        /// Actual length in characters.
        length: usize,
        /// Maximum allowed length.
        max: usize,
    },

    /// Dimension mismatch between expected and actual.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },

    /// Model not supported by this service.
    #[error("model not supported: {0}")]
    UnsupportedModel(String),

    /// Internal logic error (count mismatch, unexpected state).
    #[error("internal error: {0}")]
    Internal(String),

    /// A prepared SIMD operation received data at a different quantization tier.
    /// See `docs/design.md` (§Prepared-dispatch errors) for recovery and failure semantics.
    #[error("tier mismatch in {op}: expected {expected:?}, got {actual:?}")]
    TierMismatch {
        /// Name of the operation where the mismatch was detected.
        op: &'static str,
        /// Tier the operation required.
        expected: crate::simd::QuantizationTier,
        /// Tier actually supplied.
        actual: crate::simd::QuantizationTier,
    },
}

/// **Stable**: result type alias for embedding operations.
pub type Result<T> = std::result::Result<T, EmbedError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = EmbedError::DimensionMismatch {
            expected: 384,
            actual: 768,
        };
        assert_eq!(err.to_string(), "dimension mismatch: expected 384, got 768");
    }

    #[test]
    fn test_error_variants() {
        let err = EmbedError::ModelNotLoaded("test".into());
        assert_eq!(err.to_string(), "model not loaded: test");

        let err = EmbedError::WrongModelLoaded {
            expected: "small".into(),
            actual: "large".into(),
        };
        assert!(err.to_string().contains("expected small"));

        let err = EmbedError::ModelInitialization("failed".into());
        assert!(err.to_string().contains("initialization"));

        let err = EmbedError::InferenceFailed("oom".into());
        assert!(err.to_string().contains("inference"));

        let err = EmbedError::TaskFailed("panic".into());
        assert!(err.to_string().contains("task"));

        let err = EmbedError::InvalidInput("empty".into());
        assert!(err.to_string().contains("invalid input"));

        let err = EmbedError::UnsupportedModel("gpt4".into());
        assert!(err.to_string().contains("not supported"));

        let err = EmbedError::Internal("bug".into());
        assert!(err.to_string().contains("internal"));

        let err = EmbedError::TextTooLong {
            length: 50000,
            max: 32768,
        };
        assert!(err.to_string().contains("50000"));
        assert!(err.to_string().contains("32768"));
    }
}
