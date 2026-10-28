//! Errors returned by prepared quantized-tier operations.

use crate::QuantizationTier;

/// **Unstable**: error returned by a prepared quantized-tier operation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SimdError {
    /// Internal logic error (count mismatch, unexpected state).
    #[error("internal error: {0}")]
    Internal(String),

    /// A prepared SIMD operation received data at a different quantization tier.
    /// See [`docs/simd.md`](../docs/simd.md#prepared-queries-and-tier-matching) for recovery and failure semantics.
    #[error("tier mismatch in {op}: expected {expected:?}, got {actual:?}")]
    TierMismatch {
        /// Name of the operation where the mismatch was detected.
        op: &'static str,
        /// Tier the operation required.
        expected: QuantizationTier,
        /// Tier actually supplied.
        actual: QuantizationTier,
    },
}

/// **Unstable**: result type alias for prepared quantized-tier operations.
pub type Result<T> = std::result::Result<T, SimdError>;
