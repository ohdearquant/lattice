//! ML-domain vector types local to lattice-embed.
//!
//! These types are defined here because the new lattice-types foundation crate
//! only contains identity/policy/capability primitives and does not include
//! vector configuration types. These are ML-domain concerns.

use serde::{Deserialize, Serialize};

// ============================================================================
// DistanceMetric
// ============================================================================

/// Distance metric used for vector similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
#[repr(u8)]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine distance).
    #[default]
    Cosine = 1,
    /// Dot product (inner product).
    Dot = 2,
    /// Euclidean (L2) distance.
    L2 = 3,
}

impl DistanceMetric {
    /// Return the wire byte for this variant.
    #[inline]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }

    /// Reconstruct from a wire byte. Returns `None` for unknown values.
    #[inline]
    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            1 => Some(Self::Cosine),
            2 => Some(Self::Dot),
            3 => Some(Self::L2),
            _ => None,
        }
    }
}

// ============================================================================
// VectorDType
// ============================================================================

/// Element data type for stored vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
#[repr(u8)]
pub enum VectorDType {
    /// 32-bit float.
    #[default]
    F32 = 1,
    /// 16-bit float (half precision).
    F16 = 2,
    /// 8-bit signed integer (quantized).
    I8 = 3,
}

impl VectorDType {
    /// Return the wire byte for this variant.
    #[inline]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }

    /// Size in bytes per element.
    #[inline]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
        }
    }
}

// ============================================================================
// VectorNorm
// ============================================================================

/// Normalization state of stored vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
#[repr(u8)]
pub enum VectorNorm {
    /// No normalization applied.
    #[default]
    None = 0,
    /// Normalized to unit length (L2 norm = 1).
    Unit = 1,
}

impl VectorNorm {
    /// Return the wire byte for this variant.
    #[inline]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }
}

// ============================================================================
// EmbeddingKey
// ============================================================================

/// Identifies an embedding space (model + revision + dims + metric + dtype + norm).
///
/// Used for selecting vector store collections, caching, and embedding migration routing.
/// `canonical_bytes()` produces a stable hash for deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbeddingKey {
    /// Provider/model name (e.g., "bge-small-en-v1.5").
    pub model: Box<str>,
    /// Provider-specific revision (semver, date tag, or commit hash).
    pub revision: Box<str>,
    /// Vector dimensionality.
    pub dims: u32,
    /// Distance metric for similarity.
    pub metric: DistanceMetric,
    /// Element data type.
    pub dtype: VectorDType,
    /// Normalization state.
    pub norm: VectorNorm,
}

impl EmbeddingKey {
    /// Create a new `EmbeddingKey`.
    pub fn new(
        model: impl Into<Box<str>>,
        revision: impl Into<Box<str>>,
        dims: u32,
        metric: DistanceMetric,
        dtype: VectorDType,
        norm: VectorNorm,
    ) -> Self {
        Self {
            model: model.into(),
            revision: revision.into(),
            dims,
            metric,
            dtype,
            norm,
        }
    }

    /// Returns canonical bytes for deterministic hashing.
    ///
    /// Format:
    /// - model (4-byte big-endian length prefix + UTF-8 bytes)
    /// - revision (4-byte big-endian length prefix + UTF-8 bytes)
    /// - dims (4 bytes, big-endian)
    /// - metric (1 byte)
    /// - dtype (1 byte)
    /// - norm (1 byte)
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        let model_bytes = self.model.as_bytes();
        buf.extend_from_slice(&(model_bytes.len() as u32).to_be_bytes());
        buf.extend_from_slice(model_bytes);

        let rev_bytes = self.revision.as_bytes();
        buf.extend_from_slice(&(rev_bytes.len() as u32).to_be_bytes());
        buf.extend_from_slice(rev_bytes);

        buf.extend_from_slice(&self.dims.to_be_bytes());
        buf.push(self.metric.as_byte());
        buf.push(self.dtype.as_byte());
        buf.push(self.norm.as_byte());

        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metric_byte_roundtrip() {
        for (metric, byte) in [
            (DistanceMetric::Cosine, 1u8),
            (DistanceMetric::Dot, 2u8),
            (DistanceMetric::L2, 3u8),
        ] {
            assert_eq!(metric.as_byte(), byte);
            assert_eq!(DistanceMetric::from_byte(byte), Some(metric));
        }
        assert_eq!(DistanceMetric::from_byte(99), None);
    }

    #[test]
    fn test_vector_dtype_size_bytes() {
        assert_eq!(VectorDType::F32.size_bytes(), 4);
        assert_eq!(VectorDType::F16.size_bytes(), 2);
        assert_eq!(VectorDType::I8.size_bytes(), 1);
    }

    #[test]
    fn test_vector_norm_defaults() {
        assert_eq!(VectorNorm::default(), VectorNorm::None);
    }

    #[test]
    fn test_embedding_key_canonical_bytes_deterministic() {
        let k1 = EmbeddingKey::new(
            "bge-small-en-v1.5",
            "v1.5",
            384,
            DistanceMetric::Cosine,
            VectorDType::F32,
            VectorNorm::Unit,
        );
        let k2 = EmbeddingKey::new(
            "bge-small-en-v1.5",
            "v1.5",
            384,
            DistanceMetric::Cosine,
            VectorDType::F32,
            VectorNorm::Unit,
        );
        assert_eq!(k1.canonical_bytes(), k2.canonical_bytes());
    }

    #[test]
    fn test_embedding_key_canonical_bytes_differs_by_field() {
        let k1 = EmbeddingKey::new(
            "model-a",
            "v1",
            384,
            DistanceMetric::Cosine,
            VectorDType::F32,
            VectorNorm::Unit,
        );
        let k2 = EmbeddingKey::new(
            "model-b",
            "v1",
            384,
            DistanceMetric::Cosine,
            VectorDType::F32,
            VectorNorm::Unit,
        );
        assert_ne!(k1.canonical_bytes(), k2.canonical_bytes());

        let k3 = EmbeddingKey::new(
            "model-a",
            "v1",
            768,
            DistanceMetric::Cosine,
            VectorDType::F32,
            VectorNorm::Unit,
        );
        assert_ne!(k1.canonical_bytes(), k3.canonical_bytes());
    }
}
