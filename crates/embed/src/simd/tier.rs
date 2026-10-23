//! Quantization tiers, prepared queries, and unified distance dispatch.
//!
//! Tiers trade storage for fidelity; prepared queries avoid repeated
//! quantization in homogeneous candidate searches.
//!
//! See docs/simd.md for tier selection and dispatch semantics.

use super::binary::BinaryVector;
use super::int4::Int4Vector;
use super::quantized::{QuantizedVector, cosine_similarity_i8_trusted, dot_product_i8_trusted};
use super::{cosine_similarity, dot_product};
use crate::error::{EmbedError, Result};

/// Caller assertion that a vector is L2-unit-normalized (norm ≈ 1).
///
/// When both query and stored vectors carry `UnitNorm`, cosine similarity equals
/// the dot product — the norm division can be skipped entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NormalizationHint {
    /// No guarantee — full cosine (with norm division) is required.
    Unknown,
    /// Caller asserts this vector is L2-unit-normalized (norm ≈ 1 within 1e-4).
    Unit,
}

/// **Unstable**: tier design is under active iteration; tier boundaries may change.
///
/// Quantization precision tier, ordered from highest to lowest fidelity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum QuantizationTier {
    /// Full f32 precision (4 bytes/dim, 1x baseline).
    Full,
    /// INT8 symmetric quantization (1 byte/dim, 4x compression).
    Int8,
    /// INT4 packed nibble quantization (0.5 bytes/dim, 8x compression).
    Int4,
    /// Binary sign-bit quantization (0.125 bytes/dim, 32x compression).
    Binary,
}

impl QuantizationTier {
    /// **Unstable**: bytes-per-dimension constant; may change with new tiers.
    pub fn bytes_per_dim(&self) -> f32 {
        match self {
            Self::Full => 4.0,
            Self::Int8 => 1.0,
            Self::Int4 => 0.5,
            Self::Binary => 0.125,
        }
    }

    /// **Unstable**: compression ratio; derived from `bytes_per_dim`, may be removed.
    pub fn compression_ratio(&self) -> f32 {
        4.0 / self.bytes_per_dim()
    }

    /// **Unstable**: storage byte computation; may change with new tiers.
    pub fn storage_bytes(&self, dims: usize) -> usize {
        match self {
            Self::Full => dims * 4,
            Self::Int8 => dims,
            Self::Int4 => dims.div_ceil(2),
            Self::Binary => dims.div_ceil(8),
        }
    }

    /// **Warning**: this is a placeholder storage policy, not evidence that older vectors
    /// tolerate lower precision. Callers should measure retrieval quality for their workload.
    ///
    /// **Unstable**: tier boundaries may be tuned.
    pub fn from_age_seconds(age_secs: u64) -> Self {
        const HOUR: u64 = 3600;
        const DAY: u64 = 86400;
        const WEEK: u64 = 604800;

        if age_secs < HOUR {
            Self::Full
        } else if age_secs < DAY {
            Self::Int8
        } else if age_secs < WEEK {
            Self::Int4
        } else {
            Self::Binary
        }
    }
}

/// **Unstable**: unified quantized data container; variants may change with tier redesign.
///
/// Wraps the tier-specific vector types into a single enum for
/// uniform storage and distance dispatch.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum QuantizedData {
    /// Full-precision f32 vector.
    Full(Vec<f32>),
    /// INT8 quantized vector.
    Int8(QuantizedVector),
    /// INT4 packed quantized vector.
    Int4(Int4Vector),
    /// Binary sign-bit vector.
    Binary(BinaryVector),
}

impl QuantizedData {
    /// **Unstable**: returns `QuantizationTier` which is itself Unstable.
    pub fn tier(&self) -> QuantizationTier {
        match self {
            Self::Full(_) => QuantizationTier::Full,
            Self::Int8(_) => QuantizationTier::Int8,
            Self::Int4(_) => QuantizationTier::Int4,
            Self::Binary(_) => QuantizationTier::Binary,
        }
    }

    /// **Unstable**: dimension accessor; may be removed if `QuantizedData` gains a dims field.
    pub fn dims(&self) -> usize {
        match self {
            Self::Full(v) => v.len(),
            Self::Int8(q) => q.len(),
            Self::Int4(q) => q.dims,
            Self::Binary(q) => q.dims,
        }
    }

    /// **Unstable**: storage byte count; may change with tier redesign.
    pub fn storage_bytes(&self) -> usize {
        match self {
            Self::Full(v) => v.len() * 4,
            Self::Int8(q) => q.len(),
            Self::Int4(q) => q.data.len(),
            Self::Binary(q) => q.data.len(),
        }
    }

    /// **Unstable**: quantization factory; tier dispatch logic may change.
    pub fn from_f32(vector: &[f32], tier: QuantizationTier) -> Self {
        match tier {
            QuantizationTier::Full => Self::Full(vector.to_vec()),
            QuantizationTier::Int8 => Self::Int8(QuantizedVector::from_f32(vector)),
            QuantizationTier::Int4 => Self::Int4(Int4Vector::from_f32(vector)),
            QuantizationTier::Binary => Self::Binary(BinaryVector::from_f32(vector)),
        }
    }

    /// **Unstable**: dequantization; output precision is tier-dependent.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Self::Full(v) => v.clone(),
            Self::Int8(q) => q.to_f32(),
            Self::Int4(q) => q.to_f32(),
            Self::Binary(q) => q.to_f32(),
        }
    }

    /// **Unstable**: re-quantizes through `f32`; lost information is not recovered.
    pub fn promote(&self, target: QuantizationTier) -> Self {
        let f32_data = self.to_f32();
        Self::from_f32(&f32_data, target)
    }

    /// **Unstable**: tier demotion; delegates to `promote`; may be removed.
    pub fn demote(&self, target: QuantizationTier) -> Self {
        self.promote(target) // Same operation, just going the other direction
    }
}

/// **Unstable**: pre-quantized query for repeated same-tier distance computation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PreparedQuery {
    /// Full f32 query.
    Full(Vec<f32>),
    /// INT8 quantized query.
    Int8(QuantizedVector),
    /// INT4 packed quantized query.
    Int4(Int4Vector),
    /// Binary sign-bit query.
    Binary(BinaryVector),
}

impl PreparedQuery {
    /// Quantize a query at the given tier for repeated distance calls.
    #[inline]
    pub fn from_f32(query_f32: &[f32], tier: QuantizationTier) -> Self {
        match tier {
            QuantizationTier::Full => Self::Full(query_f32.to_vec()),
            QuantizationTier::Int8 => Self::Int8(QuantizedVector::from_f32(query_f32)),
            QuantizationTier::Int4 => Self::Int4(Int4Vector::from_f32(query_f32)),
            QuantizationTier::Binary => Self::Binary(BinaryVector::from_f32(query_f32)),
        }
    }

    /// Returns the quantization tier of this prepared query.
    #[inline]
    pub fn tier(&self) -> QuantizationTier {
        match self {
            Self::Full(_) => QuantizationTier::Full,
            Self::Int8(_) => QuantizationTier::Int8,
            Self::Int4(_) => QuantizationTier::Int4,
            Self::Binary(_) => QuantizationTier::Binary,
        }
    }

    /// Returns the number of dimensions.
    #[inline]
    pub fn dims(&self) -> usize {
        match self {
            Self::Full(v) => v.len(),
            Self::Int8(q) => q.len(),
            Self::Int4(q) => q.dims,
            Self::Binary(q) => q.dims,
        }
    }
}

/// Prepare a query vector for repeated distance computation against a homogeneous tier.
#[inline]
pub fn prepare_query(query_f32: &[f32], tier: QuantizationTier) -> PreparedQuery {
    PreparedQuery::from_f32(query_f32, tier)
}

/// A prepared query with caller-provided normalization metadata.
#[derive(Debug, Clone)]
pub struct PreparedQueryWithMeta {
    /// The quantized query (owns the data).
    pub query: PreparedQuery,
    /// Caller assertion about the query vector's normalization state.
    pub norm: NormalizationHint,
}

impl PreparedQueryWithMeta {
    /// Create a prepared query from an f32 vector, asserting its normalization state.
    #[inline]
    pub fn from_f32(query_f32: &[f32], tier: QuantizationTier, norm: NormalizationHint) -> Self {
        Self {
            query: PreparedQuery::from_f32(query_f32, tier),
            norm,
        }
    }

    /// Returns the quantization tier.
    #[inline]
    pub fn tier(&self) -> QuantizationTier {
        self.query.tier()
    }

    /// Returns the number of dimensions.
    #[inline]
    pub fn dims(&self) -> usize {
        self.query.dims()
    }
}

/// Returns `true` when the squared norm of `v` is within 1e-4 of 1.0.
///
/// Uses the SIMD-dispatched [`dot_product`] for the self-dot rather than a plain
/// scalar reduction. This helper is no longer called on the cosine hot path
/// (`approximate_cosine_distance_prepared_with_meta` delegates to the fused
/// path instead of guarding a hint-selected shortcut), but any caller checking
/// norms per candidate gets the SIMD cost model, not a scalar one.
#[inline]
pub fn is_unit_norm(v: &[f32]) -> bool {
    let sq = dot_product(v, v);
    (sq - 1.0).abs() < 1e-4
}

/// Prepare a query annotated with the given normalization hint.
#[inline]
pub fn prepare_query_with_norm(
    query_f32: &[f32],
    tier: QuantizationTier,
    norm: NormalizationHint,
) -> PreparedQueryWithMeta {
    PreparedQueryWithMeta::from_f32(query_f32, tier, norm)
}

/// **Unstable**: computes prepared cosine distance in `[0, 2]` for matching tiers.
///
/// Returns [`EmbedError::TierMismatch`] for a different stored tier.
/// See [`docs/simd.md`](../../docs/simd.md#prepared-queries-and-tier-matching) for the per-tier paths.
#[inline]
pub fn approximate_cosine_distance_prepared(
    query: &PreparedQuery,
    stored: &QuantizedData,
) -> Result<f32> {
    match (query, stored) {
        (PreparedQuery::Full(q), QuantizedData::Full(s)) => Ok(1.0 - cosine_similarity(q, s)),
        (PreparedQuery::Int8(q), QuantizedData::Int8(s)) => {
            Ok(1.0 - cosine_similarity_i8_trusted(s, q))
        }
        (PreparedQuery::Int4(q), QuantizedData::Int4(s)) => Ok(s.cosine_distance(q)),
        (PreparedQuery::Binary(q), QuantizedData::Binary(s)) => Ok(s.cosine_distance_approx(q)),
        _ => Err(EmbedError::TierMismatch {
            op: "approximate_cosine_distance_prepared",
            expected: stored.tier(),
            actual: query.tier(),
        }),
    }
}

/// Alias for [`approximate_cosine_distance_prepared`] retained for compatibility.
#[inline]
pub fn try_approximate_cosine_distance_prepared(
    query: &PreparedQuery,
    stored: &QuantizedData,
) -> Result<f32> {
    approximate_cosine_distance_prepared(query, stored)
}

/// Alias for [`approximate_dot_product_prepared`] retained for compatibility.
#[inline]
pub fn try_approximate_dot_product_prepared(
    query: &PreparedQuery,
    stored: &QuantizedData,
) -> Result<f32> {
    approximate_dot_product_prepared(query, stored)
}

/// Computes prepared cosine distance; hints are accepted but do not select a
/// separate code path.
///
/// The former `Full` unit-norm "fast path" (skip norm division when both sides
/// assert unit norm) was measurably slower than the general path it guarded:
/// verifying the stored side's norm plus the query dot takes two O(d) passes,
/// while [`cosine_similarity`] computes the dot and both norms in one fused
/// pass. With the guard it was also a correctness risk, trusting release-time
/// hints. Delegating unconditionally is both the fastest and the safest shape.
///
/// Returns [`EmbedError::TierMismatch`] for a tier mismatch.
/// See [`docs/simd.md`](../../docs/simd.md#prepared-queries-and-tier-matching) for hint semantics.
#[inline]
pub fn approximate_cosine_distance_prepared_with_meta(
    meta: &PreparedQueryWithMeta,
    stored: &QuantizedData,
    _stored_norm: NormalizationHint,
) -> Result<f32> {
    approximate_cosine_distance_prepared(&meta.query, stored)
}

/// **Unstable**: computes a prepared dot product for matching non-binary tiers.
///
/// Returns [`EmbedError::TierMismatch`] for different tiers or [`EmbedError::Internal`] for binary.
/// See [`docs/simd.md`](../../docs/simd.md#prepared-queries-and-tier-matching) for supported paths.
#[inline]
pub fn approximate_dot_product_prepared(
    query: &PreparedQuery,
    stored: &QuantizedData,
) -> Result<f32> {
    match (query, stored) {
        (PreparedQuery::Full(q), QuantizedData::Full(s)) => Ok(dot_product(q, s)),
        (PreparedQuery::Int8(q), QuantizedData::Int8(s)) => Ok(dot_product_i8_trusted(q, s)),
        (PreparedQuery::Int4(q), QuantizedData::Int4(s)) => Ok(s.dot_product(q)),
        (PreparedQuery::Binary(_), QuantizedData::Binary(_)) => Err(EmbedError::Internal(
            "Binary has no prepared dot product; use approximate_cosine_distance_prepared".into(),
        )),
        _ => Err(EmbedError::TierMismatch {
            op: "approximate_dot_product_prepared",
            expected: stored.tier(),
            actual: query.tier(),
        }),
    }
}

/// Computes distances from one prepared query to all stored vectors.
///
/// Returns [`EmbedError::TierMismatch`] if any stored tier differs.
#[inline]
pub fn batch_approximate_cosine_distance_prepared(
    query: &PreparedQuery,
    stored: &[QuantizedData],
) -> Result<Vec<f32>> {
    stored
        .iter()
        .map(|item| approximate_cosine_distance_prepared(query, item))
        .collect()
}

/// Writes prepared-query distances into a reusable buffer, clearing it on error.
///
/// Returns [`EmbedError::TierMismatch`] if any stored tier differs.
/// See [`docs/simd.md`](../../docs/simd.md#prepared-queries-and-tier-matching) for buffer semantics.
#[inline]
pub fn batch_approximate_cosine_distance_prepared_into(
    query: &PreparedQuery,
    stored: &[QuantizedData],
    out: &mut Vec<f32>,
) -> Result<()> {
    out.clear();
    out.reserve(stored.len());
    for item in stored {
        match approximate_cosine_distance_prepared(query, item) {
            Ok(distance) => out.push(distance),
            Err(e) => {
                out.clear();
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Computes distances from one prepared INT8 query without re-quantizing it.
///
/// Returns [`EmbedError::TierMismatch`] unless the query is INT8.
#[inline]
pub fn approximate_int8_batch_prepared(
    query: &PreparedQuery,
    candidates: &[QuantizedVector],
) -> Result<Vec<f32>> {
    let PreparedQuery::Int8(q) = query else {
        return Err(EmbedError::TierMismatch {
            op: "approximate_int8_batch_prepared",
            expected: QuantizationTier::Int8,
            actual: query.tier(),
        });
    };
    Ok(candidates
        .iter()
        .map(|candidate| 1.0 - cosine_similarity_i8_trusted(candidate, q))
        .collect())
}

/// Writes prepared INT8 distances into a reusable buffer, clearing it on error.
///
/// Returns [`EmbedError::TierMismatch`] unless the query is INT8.
#[inline]
pub fn approximate_int8_batch_prepared_into(
    query: &PreparedQuery,
    candidates: &[QuantizedVector],
    out: &mut Vec<f32>,
) -> Result<()> {
    out.clear();
    let PreparedQuery::Int8(q) = query else {
        return Err(EmbedError::TierMismatch {
            op: "approximate_int8_batch_prepared_into",
            expected: QuantizationTier::Int8,
            actual: query.tier(),
        });
    };
    out.reserve(candidates.len());
    out.extend(
        candidates
            .iter()
            .map(|candidate| 1.0 - cosine_similarity_i8_trusted(candidate, q)),
    );
    Ok(())
}

/// Computes distances from one prepared INT4 query without re-quantizing it.
///
/// Returns [`EmbedError::TierMismatch`] unless the query is INT4.
#[inline]
pub fn approximate_int4_batch_prepared(
    query: &PreparedQuery,
    candidates: &[Int4Vector],
) -> Result<Vec<f32>> {
    let PreparedQuery::Int4(q) = query else {
        return Err(EmbedError::TierMismatch {
            op: "approximate_int4_batch_prepared",
            expected: QuantizationTier::Int4,
            actual: query.tier(),
        });
    };
    Ok(candidates
        .iter()
        .map(|candidate| candidate.cosine_distance(q))
        .collect())
}

/// Writes prepared INT4 distances into a reusable buffer, clearing it on error.
///
/// Returns [`EmbedError::TierMismatch`] unless the query is INT4.
#[inline]
pub fn approximate_int4_batch_prepared_into(
    query: &PreparedQuery,
    candidates: &[Int4Vector],
    out: &mut Vec<f32>,
) -> Result<()> {
    out.clear();
    let PreparedQuery::Int4(q) = query else {
        return Err(EmbedError::TierMismatch {
            op: "approximate_int4_batch_prepared_into",
            expected: QuantizationTier::Int4,
            actual: query.tier(),
        });
    };
    out.reserve(candidates.len());
    out.extend(
        candidates
            .iter()
            .map(|candidate| candidate.cosine_distance(q)),
    );
    Ok(())
}

/// **Unstable**: quantizes an `f32` query and computes tiered cosine distance.
///
/// `query_f32.len()` must match stored dimensionality.
/// See [`docs/simd.md`](../../docs/simd.md#prepared-queries-and-tier-matching) for hot-loop guidance.
pub fn approximate_cosine_distance(query_f32: &[f32], stored: &QuantizedData) -> f32 {
    debug_assert_eq!(
        query_f32.len(),
        stored.dims(),
        "approximate_cosine_distance: query length {} != stored dims {}",
        query_f32.len(),
        stored.dims(),
    );
    match stored {
        QuantizedData::Full(v) => {
            // Exact cosine distance
            1.0 - cosine_similarity(query_f32, v)
        }
        QuantizedData::Int8(q) => {
            let query_q = QuantizedVector::from_f32(query_f32);
            1.0 - q.cosine_similarity(&query_q)
        }
        QuantizedData::Int4(q) => {
            let query_q = Int4Vector::from_f32(query_f32);
            q.cosine_distance(&query_q)
        }
        QuantizedData::Binary(q) => {
            let query_q = BinaryVector::from_f32(query_f32);
            q.cosine_distance_approx(&query_q)
        }
    }
}

/// **Unstable**: approximate tiered dot-product dispatch.
pub fn approximate_dot_product(query_f32: &[f32], stored: &QuantizedData) -> f32 {
    match stored {
        QuantizedData::Full(v) => dot_product(query_f32, v),
        QuantizedData::Int8(q) => {
            let query_q = QuantizedVector::from_f32(query_f32);
            q.dot_product(&query_q)
        }
        QuantizedData::Int4(q) => {
            let query_q = Int4Vector::from_f32(query_f32);
            q.dot_product(&query_q)
        }
        QuantizedData::Binary(_q) => {
            // Binary doesn't have a meaningful dot product; fall back to dequantize
            let stored_f32 = _q.to_f32();
            dot_product(query_f32, &stored_f32)
        }
    }
}

#[cfg(test)]
mod tests;
