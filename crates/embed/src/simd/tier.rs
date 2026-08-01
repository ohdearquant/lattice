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
mod tests {
    use super::*;

    fn generate_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed ^ ((dim as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        (0..dim)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407)
                    .wrapping_add(i as u64);
                let unit = ((state >> 32) as u32) as f32 / u32::MAX as f32;
                unit * 2.0 - 1.0
            })
            .collect()
    }

    fn scalar_cosine_f64(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (&a, &b) in a.iter().zip(b) {
            let a = f64::from(a);
            let b = f64::from(b);
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 { 0.0 } else { dot / denom }
    }

    fn reference_ranking(query: &[f32], corpus: &[Vec<f32>]) -> Vec<usize> {
        let mut ranked: Vec<_> = corpus
            .iter()
            .enumerate()
            .map(|(index, candidate)| (index, scalar_cosine_f64(query, candidate)))
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.into_iter().map(|(index, _)| index).collect()
    }

    fn tier_ranking(query: &[f32], stored: &[QuantizedData], tier: QuantizationTier) -> Vec<usize> {
        let prepared = PreparedQuery::from_f32(query, tier);
        assert_eq!(prepared.tier(), tier);
        let mut ranked: Vec<_> = stored
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                (
                    index,
                    approximate_cosine_distance_prepared(&prepared, candidate).unwrap(),
                )
            })
            .collect();
        ranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        ranked.into_iter().map(|(index, _)| index).collect()
    }

    fn recall_hits_at(reference: &[usize], actual: &[usize], k: usize) -> usize {
        actual[..k]
            .iter()
            .filter(|candidate| reference[..k].contains(candidate))
            .count()
    }

    fn recall_at(reference: &[usize], actual: &[usize], k: usize) -> f64 {
        recall_hits_at(reference, actual, k) as f64 / k as f64
    }

    fn pairwise_ranking_agreements(reference: &[usize], actual: &[usize]) -> usize {
        assert_eq!(reference.len(), actual.len());
        let mut actual_position = vec![0usize; actual.len()];
        for (position, &candidate) in actual.iter().enumerate() {
            actual_position[candidate] = position;
        }
        let mut agreements = 0usize;
        for (position, &left) in reference.iter().enumerate() {
            for &right in &reference[position + 1..] {
                agreements += usize::from(actual_position[left] < actual_position[right]);
            }
        }
        agreements
    }

    fn pairwise_ranking_agreement(reference: &[usize], actual: &[usize]) -> f64 {
        let pairs = reference.len() * (reference.len() - 1) / 2;
        pairwise_ranking_agreements(reference, actual) as f64 / pairs as f64
    }

    fn retrieval_quality_counts(
        reference: &[usize],
        actual: &[usize],
        top_k: usize,
    ) -> (usize, usize) {
        (
            recall_hits_at(reference, actual, top_k),
            pairwise_ranking_agreements(reference, actual),
        )
    }

    fn index_order_surrogate_quality(
        corpus: &[Vec<f32>],
        queries: &[Vec<f32>],
        top_k: usize,
    ) -> (f64, f64) {
        let index_order: Vec<_> = (0..corpus.len()).collect();
        let mut recall = 0.0;
        let mut agreement = 0.0;
        for query in queries {
            let reference = reference_ranking(query, corpus);
            recall += recall_at(&reference, &index_order, top_k);
            agreement += pairwise_ranking_agreement(&reference, &index_order);
        }
        (
            recall / queries.len() as f64,
            agreement / queries.len() as f64,
        )
    }

    fn retrieval_quality_floor(tier: QuantizationTier) -> (f64, f64) {
        match tier {
            QuantizationTier::Full => (1.0, 0.999),
            QuantizationTier::Int8 => (0.98, 0.995),
            QuantizationTier::Int4 => (0.85, 0.95),
            QuantizationTier::Binary => (0.30, 0.70),
        }
    }

    /// A floor includes equality; one epsilon recovers equality lost while averaging.
    fn meets_retrieval_quality_floor(value: f64, minimum: f64) -> bool {
        value.is_finite() && value + f64::EPSILON >= minimum
    }

    /// Bounds each query so a collapsed subset cannot hide behind healthy queries.
    ///
    /// The fixed 16-query fixture produces these per-query ranges at Recall@10:
    ///
    /// | Tier | Recall@10 hits | Agreeing pairs |
    /// | --- | --- | --- |
    /// | Full | 10..=10 | 32,640..=32,640 |
    /// | Int8 | 10..=10 | 32,567..=32,604 |
    /// | Int4 | 8..=10 | 31,531..=31,773 |
    /// | Binary | 3..=7 | 24,423..=25,705 |
    ///
    /// Recall gets one additional miss beyond the observed minimum, exactly one
    /// Recall@10 step. Agreement gets one full observed min-to-max range below
    /// the observed minimum. This fixture-derived slack rejects whole-query
    /// collapse without making ordinary variation in the healthy fixture fatal.
    /// Full agreement has no observed spread and remains exact.
    fn retrieval_quality_per_query_floor(tier: QuantizationTier) -> (usize, usize) {
        match tier {
            QuantizationTier::Full => (9, 32_640),
            QuantizationTier::Int8 => (9, 32_530),
            QuantizationTier::Int4 => (7, 31_289),
            QuantizationTier::Binary => (2, 23_141),
        }
    }

    /// Pins known-good metrics as Recall@10 hits and agreeing pairs out of 32,640.
    ///
    /// Each metric may move by one observed min-to-max span in total across all
    /// 16 queries, or one sixteenth of that span under a uniform shift. Absolute
    /// movement prevents cross-query cancellation and makes a broader path change
    /// require deliberate recalibration. A zero-spread metric remains exact.
    /// Recall-hit and agreeing-pair counts can collide across different rankings. The
    /// exercised path breaks equal distances by candidate index, so this remains a
    /// metric limitation. A future top-k identity check would detect membership changes;
    /// a rank fingerprint would also detect position changes that preserve both counts.
    const HEALTHY_FULL_QUERY_QUALITY: [(usize, usize); 16] = [(10, 32_640); 16];
    const HEALTHY_INT8_QUERY_QUALITY: [(usize, usize); 16] = [
        (10, 32_588),
        (10, 32_575),
        (10, 32_597),
        (10, 32_590),
        (10, 32_595),
        (10, 32_592),
        (10, 32_580),
        (10, 32_604),
        (10, 32_587),
        (10, 32_573),
        (10, 32_580),
        (10, 32_579),
        (10, 32_567),
        (10, 32_578),
        (10, 32_576),
        (10, 32_588),
    ];
    const HEALTHY_INT4_QUERY_QUALITY: [(usize, usize); 16] = [
        (8, 31_582),
        (9, 31_531),
        (9, 31_642),
        (8, 31_656),
        (10, 31_773),
        (8, 31_638),
        (10, 31_744),
        (9, 31_758),
        (10, 31_661),
        (10, 31_651),
        (10, 31_662),
        (8, 31_727),
        (10, 31_531),
        (10, 31_625),
        (9, 31_655),
        (9, 31_653),
    ];
    const HEALTHY_BINARY_QUERY_QUALITY: [(usize, usize); 16] = [
        (7, 25_430),
        (3, 25_031),
        (6, 25_033),
        (5, 25_339),
        (4, 25_534),
        (3, 25_166),
        (3, 25_156),
        (6, 25_705),
        (5, 25_677),
        (4, 25_419),
        (4, 25_040),
        (3, 25_150),
        (4, 24_965),
        (5, 25_190),
        (5, 24_423),
        (4, 25_355),
    ];

    fn healthy_query_quality(tier: QuantizationTier) -> &'static [(usize, usize); 16] {
        match tier {
            QuantizationTier::Full => &HEALTHY_FULL_QUERY_QUALITY,
            QuantizationTier::Int8 => &HEALTHY_INT8_QUERY_QUALITY,
            QuantizationTier::Int4 => &HEALTHY_INT4_QUERY_QUALITY,
            QuantizationTier::Binary => &HEALTHY_BINARY_QUERY_QUALITY,
        }
    }

    fn retrieval_quality_movement_budget(healthy: &[(usize, usize)]) -> (usize, usize) {
        let minimum_recall_hits = healthy.iter().map(|quality| quality.0).min().unwrap();
        let maximum_recall_hits = healthy.iter().map(|quality| quality.0).max().unwrap();
        let minimum_agreements = healthy.iter().map(|quality| quality.1).min().unwrap();
        let maximum_agreements = healthy.iter().map(|quality| quality.1).max().unwrap();
        (
            maximum_recall_hits - minimum_recall_hits,
            maximum_agreements - minimum_agreements,
        )
    }

    /// Bounds concentration relative to each query's own healthy counts.
    ///
    /// Each cap is the ceiling of one sixteenth of the full healthy span, the smallest
    /// integer allowance that admits the total budget's uniform per-query share. Binary
    /// therefore permits one Recall@10 hit or 81 agreeing pairs to move on one query.
    /// This catches concentrated cliffs; the retained L1 budget catches broad movement.
    fn retrieval_quality_concentration_budget(healthy: &[(usize, usize)]) -> (usize, usize) {
        let movement_budget = retrieval_quality_movement_budget(healthy);
        (
            movement_budget.0.div_ceil(healthy.len()),
            movement_budget.1.div_ceil(healthy.len()),
        )
    }

    fn validate_tier_retrieval_quality(
        tier: QuantizationTier,
        query_quality: &[(usize, usize)],
        top_k: usize,
    ) -> std::result::Result<(f64, f64), String> {
        if query_quality.is_empty() {
            return Err(format!("{tier:?} retrieval quality has zero queries"));
        }

        let healthy = healthy_query_quality(tier);
        if query_quality.len() != healthy.len() {
            return Err(format!(
                "{tier:?} retrieval quality has {} queries, expected {}",
                query_quality.len(),
                healthy.len()
            ));
        }

        let (minimum_query_recall, minimum_query_agreement) =
            retrieval_quality_per_query_floor(tier);
        for (query_index, &(recall_hits, agreements)) in query_quality.iter().enumerate() {
            if recall_hits < minimum_query_recall || agreements < minimum_query_agreement {
                return Err(format!(
                    "{tier:?} query {query_index} fails the per-query floor: Recall@{top_k}=\
                     {:.6} (minimum {:.6}), pairwise ranking agreement=\
                     {:.6} (minimum {:.6})",
                    recall_hits as f64 / top_k as f64,
                    minimum_query_recall as f64 / top_k as f64,
                    agreements as f64 / 32_640.0,
                    minimum_query_agreement as f64 / 32_640.0,
                ));
            }
        }

        let recall_hits = query_quality.iter().map(|quality| quality.0).sum::<usize>();
        let agreements = query_quality.iter().map(|quality| quality.1).sum::<usize>();
        let recall = recall_hits as f64 / (query_quality.len() * top_k) as f64;
        let agreement = agreements as f64 / (query_quality.len() * 32_640) as f64;
        let (minimum_recall, minimum_agreement) = retrieval_quality_floor(tier);
        if !meets_retrieval_quality_floor(recall, minimum_recall) {
            return Err(format!(
                "{tier:?} Recall@{top_k} {recall:.6} is below the measured-data floor \
                 {minimum_recall:.3}"
            ));
        }
        if !meets_retrieval_quality_floor(agreement, minimum_agreement) {
            return Err(format!(
                "{tier:?} pairwise ranking agreement {agreement:.6} is below the measured-data \
                 floor {minimum_agreement:.3}"
            ));
        }

        let (recall_movement, agreement_movement) = query_quality.iter().zip(healthy).fold(
            (0usize, 0usize),
            |(recall_movement, agreement_movement), (actual, expected)| {
                (
                    recall_movement + actual.0.abs_diff(expected.0),
                    agreement_movement + actual.1.abs_diff(expected.1),
                )
            },
        );
        let (recall_budget, agreement_budget) = retrieval_quality_movement_budget(healthy);
        if recall_movement > recall_budget || agreement_movement > agreement_budget {
            return Err(format!(
                "{tier:?} retrieval quality exceeds the fixture-relative movement budget: \
                 total absolute Recall@{top_k} movement={recall_movement} hit(s) \
                 (maximum {recall_budget}), total absolute pairwise-agreement movement=\
                 {agreement_movement} pair(s) (maximum {agreement_budget})"
            ));
        }

        let (maximum_query_recall_movement, maximum_query_agreement_movement) =
            retrieval_quality_concentration_budget(healthy);
        for (query_index, (&(recall_hits, agreements), &(healthy_hits, healthy_agreements))) in
            query_quality.iter().zip(healthy).enumerate()
        {
            let recall_movement = recall_hits.abs_diff(healthy_hits);
            let agreement_movement = agreements.abs_diff(healthy_agreements);
            if recall_movement > maximum_query_recall_movement
                || agreement_movement > maximum_query_agreement_movement
            {
                return Err(format!(
                    "{tier:?} query {query_index} exceeds the fixture-relative concentration \
                     bound: Recall@{top_k} movement={recall_movement} hit(s) \
                     (maximum {maximum_query_recall_movement}), pairwise-agreement \
                     movement={agreement_movement} pair(s) \
                     (maximum {maximum_query_agreement_movement})"
                ));
            }
        }
        Ok((recall, agreement))
    }

    /// Refuses to let a retrieval-fidelity fixture be scored when it cannot
    /// distinguish quality tiers from each other.
    ///
    /// Only the independent f64 reference ranking is checked here — ties
    /// among *quantized* tier scores are the exact behaviour under test
    /// (e.g. Binary legitimately collapses many candidates to the same
    /// Hamming distance) and must never be rejected.
    fn validate_retrieval_fixture(
        corpus: &[Vec<f32>],
        queries: &[Vec<f32>],
        top_k: usize,
    ) -> std::result::Result<(), String> {
        if top_k == 0 {
            return Err("retrieval fixture has top_k=0".to_string());
        }
        if queries.is_empty() {
            return Err("retrieval fixture has zero queries".to_string());
        }
        if corpus.len() < top_k {
            return Err(format!(
                "retrieval fixture corpus size {} is smaller than top_k={top_k}",
                corpus.len()
            ));
        }
        for (query_index, query) in queries.iter().enumerate() {
            let mut ranked_scores = Vec::with_capacity(corpus.len());
            for (candidate_index, candidate) in corpus.iter().enumerate() {
                let score = scalar_cosine_f64(query, candidate);
                if !score.is_finite() {
                    return Err(format!(
                        "retrieval fixture query {query_index} candidate {candidate_index} has \
                         non-finite reference score {score}"
                    ));
                }
                ranked_scores.push((candidate_index, score));
            }
            ranked_scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            let mut distinct_scores: Vec<_> =
                ranked_scores.iter().map(|(_, score)| *score).collect();
            distinct_scores.sort_unstable_by(f64::total_cmp);
            distinct_scores.dedup();
            if distinct_scores.len() <= top_k {
                return Err(format!(
                    "retrieval fixture query {query_index} is non-discriminating: only \
                     {} distinct finite reference score(s) across {} candidates, need \
                     more than top_k={top_k}",
                    distinct_scores.len(),
                    corpus.len()
                ));
            }

            for (rank, boundary) in ranked_scores.windows(2).enumerate() {
                let higher_score = boundary[0].1;
                let lower_score = boundary[1].1;
                if higher_score as f32 <= lower_score as f32 {
                    return Err(format!(
                        "retrieval fixture query {query_index} has a near-tied ranking boundary \
                         at ranks {rank} and {}: reference scores {higher_score} and \
                         {lower_score} do not remain ordered at f32 precision",
                        rank + 1
                    ));
                }
            }
        }

        let (surrogate_recall, surrogate_agreement) =
            index_order_surrogate_quality(corpus, queries, top_k);
        for tier in [
            QuantizationTier::Full,
            QuantizationTier::Int8,
            QuantizationTier::Int4,
            QuantizationTier::Binary,
        ] {
            let (minimum_recall, minimum_agreement) = retrieval_quality_floor(tier);
            if meets_retrieval_quality_floor(surrogate_recall, minimum_recall)
                && meets_retrieval_quality_floor(surrogate_agreement, minimum_agreement)
            {
                return Err(format!(
                    "retrieval fixture is non-discriminating: an all-tied index-order surrogate \
                     passes the {tier:?} floor with Recall@{top_k}={surrogate_recall:.6} and \
                     pairwise ranking agreement={surrogate_agreement:.6}"
                ));
            }
        }
        Ok(())
    }

    fn fixed_retrieval_fixture() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        const DIMS: usize = 384;
        const CORPUS_SIZE: usize = 256;
        const QUERY_COUNT: usize = 16;
        let corpus = (0..CORPUS_SIZE)
            .map(|index| generate_vector(DIMS, 0xC0A5_0000 + index as u64))
            .collect();
        let queries = (0..QUERY_COUNT)
            .map(|index| generate_vector(DIMS, 0x0A11_0000 + index as u64))
            .collect();
        (corpus, queries)
    }

    #[test]
    fn test_tier_retrieval_quality_against_independent_f64_ranking() {
        const TOP_K: usize = 10;
        let (corpus, queries) = fixed_retrieval_fixture();
        validate_retrieval_fixture(&corpus, &queries, TOP_K)
            .expect("retrieval fixture must be discriminating before scoring tiers against it");

        for tier in [
            QuantizationTier::Full,
            QuantizationTier::Int8,
            QuantizationTier::Int4,
            QuantizationTier::Binary,
        ] {
            let stored: Vec<_> = corpus
                .iter()
                .map(|candidate| QuantizedData::from_f32(candidate, tier))
                .collect();
            assert!(
                stored.iter().all(|candidate| candidate.tier() == tier),
                "{tier:?} conversion was bypassed or routed to another tier"
            );
            assert!(
                stored
                    .iter()
                    .all(|candidate| candidate.storage_bytes()
                        == tier.storage_bytes(candidate.dims())),
                "{tier:?} conversion produced the wrong representation size"
            );

            if tier != QuantizationTier::Full {
                assert!(
                    stored.iter().zip(&corpus).any(|(quantized, original)| {
                        quantized
                            .to_f32()
                            .iter()
                            .zip(original)
                            .any(|(actual, expected)| (actual - expected).abs() > 1e-4)
                    }),
                    "{tier:?} conversion did not exercise a lossy representation"
                );
            }

            let mut query_quality = Vec::with_capacity(queries.len());
            let mut quantized_distance_witness = false;
            for query in &queries {
                let reference = reference_ranking(query, &corpus);
                let actual = tier_ranking(query, &stored, tier);
                query_quality.push(retrieval_quality_counts(&reference, &actual, TOP_K));

                if tier != QuantizationTier::Full {
                    let prepared = PreparedQuery::from_f32(query, tier);
                    quantized_distance_witness |=
                        stored.iter().zip(&corpus).any(|(quantized, original)| {
                            let actual =
                                approximate_cosine_distance_prepared(&prepared, quantized).unwrap();
                            let reference = 1.0 - scalar_cosine_f64(query, original) as f32;
                            (actual - reference).abs() > 1e-4
                        });
                }
            }
            let (recall, agreement) = validate_tier_retrieval_quality(tier, &query_quality, TOP_K)
                .unwrap_or_else(|error| panic!("{error}"));
            eprintln!(
                "{tier:?}: Recall@{TOP_K}={recall:.6}, pairwise ranking agreement={agreement:.6}"
            );

            if tier != QuantizationTier::Full {
                assert!(
                    quantized_distance_witness,
                    "{tier:?} distance path did not differ from the independent f32 reference"
                );
            }
        }
    }

    #[test]
    fn test_tier_retrieval_quality_rejects_concentrated_binary_query_collapse() {
        const TOP_K: usize = 10;
        const COLLAPSED_QUERY_COUNT: usize = 10;
        let (corpus, queries) = fixed_retrieval_fixture();
        let index_order: Vec<_> = (0..corpus.len()).collect();
        let query_quality: Vec<_> = queries
            .iter()
            .enumerate()
            .map(|(query_index, query)| {
                let reference = reference_ranking(query, &corpus);
                let actual = if query_index < COLLAPSED_QUERY_COUNT {
                    index_order.clone()
                } else {
                    reference.clone()
                };
                retrieval_quality_counts(&reference, &actual, TOP_K)
            })
            .collect();

        let mean_recall = query_quality.iter().map(|quality| quality.0).sum::<usize>() as f64
            / (query_quality.len() * TOP_K) as f64;
        let mean_agreement = query_quality.iter().map(|quality| quality.1).sum::<usize>() as f64
            / (query_quality.len() * 32_640) as f64;
        assert_eq!(format!("{mean_recall:.6}"), "0.381250");
        assert_eq!(format!("{mean_agreement:.6}"), "0.705356");

        let error = validate_tier_retrieval_quality(
            QuantizationTier::Binary,
            &query_quality,
            TOP_K,
        )
        .expect_err(
            "collapsing fixed fixture queries 0 through 9 must fail despite passing both means",
        );
        assert!(error.contains("query 0"), "unexpected error: {error}");
    }

    #[test]
    fn test_tier_retrieval_quality_rejects_single_binary_query_concentration() {
        const TOP_K: usize = 10;
        const SUFFIX_INVERSIONS: usize = 7_161;
        let (corpus, queries) = fixed_retrieval_fixture();
        let reference = reference_ranking(&queries[0], &corpus);
        let mut remaining = reference[3..10]
            .iter()
            .chain(&reference[17..])
            .copied()
            .collect::<Vec<_>>();
        let mut suffix = Vec::with_capacity(remaining.len());
        let mut inversions = SUFFIX_INVERSIONS;
        while !remaining.is_empty() {
            let index = inversions.min(remaining.len() - 1);
            inversions -= index;
            suffix.push(remaining.remove(index));
        }
        assert_eq!(inversions, 0);

        let mut actual = Vec::with_capacity(reference.len());
        actual.extend_from_slice(&reference[..3]);
        actual.extend_from_slice(&reference[10..17]);
        actual.extend(suffix);
        assert_eq!(actual.len(), reference.len());

        let collapsed_quality = retrieval_quality_counts(&reference, &actual, TOP_K);
        assert_eq!(collapsed_quality, (3, 25_430));

        let mut query_quality = healthy_query_quality(QuantizationTier::Binary).to_vec();
        query_quality[0] = collapsed_quality;

        let error =
            validate_tier_retrieval_quality(QuantizationTier::Binary, &query_quality, TOP_K)
                .expect_err("one Binary query must not spend the complete fixture movement budget");
        assert!(
            error.contains("query 0") && error.contains("concentration"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_tier_retrieval_quality_rejects_single_binary_query_agreement_concentration() {
        const TOP_K: usize = 10;
        let mut query_quality = healthy_query_quality(QuantizationTier::Binary).to_vec();
        query_quality[0].1 -= 1_281;

        let error =
            validate_tier_retrieval_quality(QuantizationTier::Binary, &query_quality, TOP_K)
                .expect_err("one Binary query must not spend nearly the complete pair budget");
        assert!(
            error.contains("query 0") && error.contains("concentration"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_tier_retrieval_quality_rejects_distributed_binary_query_collapse() {
        const TOP_K: usize = 10;
        let (corpus, queries) = fixed_retrieval_fixture();
        let query_quality: Vec<_> = queries
            .iter()
            .map(|query| {
                let reference = reference_ranking(query, &corpus);
                let mut actual = Vec::with_capacity(reference.len());
                actual.extend_from_slice(&reference[..3]);
                actual.extend_from_slice(&reference[10..17]);
                actual.extend_from_slice(&reference[3..10]);
                actual.extend(reference[17..154].iter().rev().copied());
                actual.extend_from_slice(&reference[154..]);
                assert_eq!(actual.len(), reference.len());
                retrieval_quality_counts(&reference, &actual, TOP_K)
            })
            .collect();

        assert!(
            query_quality
                .iter()
                .all(|&(recall, agreement)| { recall == 3 && agreement == 32_640 - 9_365 })
        );

        let error =
            validate_tier_retrieval_quality(QuantizationTier::Binary, &query_quality, TOP_K)
                .expect_err("a distributed 70% Recall@10 loss across every query must fail");
        assert!(
            error.contains("movement budget"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_tier_retrieval_quality_rejects_shallow_all_query_movement() {
        const TOP_K: usize = 10;
        let (corpus, queries) = fixed_retrieval_fixture();
        let query_quality: Vec<_> = queries
            .iter()
            .map(|query| {
                let reference = reference_ranking(query, &corpus);
                let mut actual = reference.clone();
                let last = actual.len() - 1;
                actual.swap(TOP_K - 1, last);
                retrieval_quality_counts(&reference, &actual, TOP_K)
            })
            .collect();

        assert!(
            query_quality
                .iter()
                .all(|&(recall, agreement)| { recall == 9 && agreement == 32_640 - 491 })
        );

        for (tier, result) in [
            (
                QuantizationTier::Int4,
                validate_tier_retrieval_quality(QuantizationTier::Int4, &query_quality, TOP_K),
            ),
            (
                QuantizationTier::Binary,
                validate_tier_retrieval_quality(QuantizationTier::Binary, &query_quality, TOP_K),
            ),
        ] {
            let error = result.unwrap_err();
            assert!(
                error.contains("movement budget"),
                "unexpected {tier:?} error: {error}"
            );
        }
    }

    #[test]
    fn test_tier_retrieval_quality_bounds_stated_binary_uniform_movement() {
        const TOP_K: usize = 10;
        let query_quality_with_pair_loss = |pair_loss: u16| {
            healthy_query_quality(QuantizationTier::Binary)
                .iter()
                .map(|&(recall_hits, agreements)| {
                    (recall_hits, agreements - usize::from(pair_loss))
                })
                .collect::<Vec<_>>()
        };

        validate_tier_retrieval_quality(
            QuantizationTier::Binary,
            &query_quality_with_pair_loss(80),
            TOP_K,
        )
        .expect("1,280 total agreement-pair changes are within the 1,282-pair budget");

        let error = validate_tier_retrieval_quality(
            QuantizationTier::Binary,
            &query_quality_with_pair_loss(81),
            TOP_K,
        )
        .expect_err("1,296 total agreement-pair changes must exceed the 1,282-pair budget");
        assert!(
            error.contains("movement budget"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_tier_retrieval_quality_accepts_non_uniform_exact_binary_movement_boundary() {
        const TOP_K: usize = 10;
        let query_quality = healthy_query_quality(QuantizationTier::Binary)
            .iter()
            .enumerate()
            .map(|(query_index, &(recall_hits, agreements))| {
                let pair_loss = if query_index < 14 { 81 } else { 74 };
                (recall_hits, agreements - pair_loss)
            })
            .collect::<Vec<_>>();

        validate_tier_retrieval_quality(QuantizationTier::Binary, &query_quality, TOP_K)
            .expect("the exact 1,282-pair non-uniform movement boundary must be accepted");
    }

    #[test]
    fn test_validate_retrieval_fixture_rejects_non_discriminating_corpus() {
        const TOP_K: usize = 10;
        const DIMS: usize = 384;
        const CORPUS_SIZE: usize = 256;
        const QUERY_COUNT: usize = 16;

        // Every candidate is the *same* lossy, non-constant vector: the
        // independent f64 reference score ties across the whole corpus, so
        // recall@10 and pairwise agreement would collapse to a meaningless
        // 1.0 for every tier if this fixture were ever scored.
        let repeated_vector = generate_vector(DIMS, 0xC0A5_0000);
        let corpus: Vec<Vec<f32>> = std::iter::repeat_n(repeated_vector, CORPUS_SIZE).collect();
        let queries: Vec<Vec<f32>> = (0..QUERY_COUNT)
            .map(|index| generate_vector(DIMS, 0x0A11_0000 + index as u64))
            .collect();

        let err = validate_retrieval_fixture(&corpus, &queries, TOP_K)
            .expect_err("a corpus of identical vectors ties every reference score; the guard must refuse rather than let it be scored");
        eprintln!("guard refused as expected: {err}");
    }

    #[test]
    fn test_validate_retrieval_fixture_rejects_zero_queries() {
        let (corpus, _queries) = fixed_retrieval_fixture();
        let err = validate_retrieval_fixture(&corpus, &[], 10)
            .expect_err("zero queries must be refused rather than panic or silently pass");
        eprintln!("guard refused as expected: {err}");
    }

    #[test]
    fn test_validate_retrieval_fixture_rejects_corpus_smaller_than_top_k() {
        let (corpus, queries) = fixed_retrieval_fixture();
        let small_corpus = corpus[..5].to_vec();
        let err = validate_retrieval_fixture(&small_corpus, &queries, 10).expect_err(
            "a corpus smaller than top_k must be refused rather than panic or return NaN",
        );
        eprintln!("guard refused as expected: {err}");
    }

    #[test]
    fn test_validate_retrieval_fixture_accepts_the_real_fixture() {
        let (corpus, queries) = fixed_retrieval_fixture();
        validate_retrieval_fixture(&corpus, &queries, 10)
            .expect("the real fixture is discriminating and must not be rejected");

        let (recall, agreement) = index_order_surrogate_quality(&corpus, &queries, 10);
        assert!((recall - 0.025).abs() < f64::EPSILON);
        assert!((agreement - 0.526_646_752_450_980_4).abs() < 1e-15);
        for (tier, minimum_recall, minimum_agreement) in [
            ("Full", 1.0, 0.999),
            ("Int8", 0.98, 0.995),
            ("Int4", 0.85, 0.95),
            ("Binary", 0.30, 0.70),
        ] {
            assert!(
                !meets_retrieval_quality_floor(recall, minimum_recall)
                    || !meets_retrieval_quality_floor(agreement, minimum_agreement),
                "index-order surrogate must fail the pinned {tier} floor"
            );
        }
    }

    #[test]
    fn test_validate_retrieval_fixture_rejects_binary_index_aligned_collapse() {
        const TOP_K: usize = 10;
        let query = vec![1.0, 0.0];
        let corpus: Vec<_> = (0..21)
            .map(|index| vec![21.0 - index as f32, 1.0])
            .collect();
        let queries = vec![query.clone()];

        let mut scores: Vec<_> = corpus
            .iter()
            .map(|candidate| scalar_cosine_f64(&query, candidate))
            .collect();
        scores.sort_unstable_by(f64::total_cmp);
        scores.dedup();
        assert_eq!(scores.len(), 21);

        let reference = reference_ranking(&query, &corpus);
        let binary: Vec<_> = corpus
            .iter()
            .map(|candidate| QuantizedData::from_f32(candidate, QuantizationTier::Binary))
            .collect();
        let prepared = PreparedQuery::from_f32(&query, QuantizationTier::Binary);
        let first_code = match &binary[0] {
            QuantizedData::Binary(value) => &value.data,
            _ => unreachable!("binary conversion must produce the Binary variant"),
        };
        assert!(binary.iter().all(|candidate| {
            let QuantizedData::Binary(value) = candidate else {
                return false;
            };
            value.data == *first_code
                && approximate_cosine_distance_prepared(&prepared, candidate).unwrap() == 0.0
        }));
        let actual = tier_ranking(&query, &binary, QuantizationTier::Binary);
        assert_eq!(reference, (0..21).collect::<Vec<_>>());
        assert_eq!(actual, reference);
        assert_eq!(recall_at(&reference, &actual, TOP_K), 1.0);
        assert_eq!(pairwise_ranking_agreement(&reference, &actual), 1.0);

        let err = validate_retrieval_fixture(&corpus, &queries, TOP_K).expect_err(
            "an index-aligned reference must not let a totally collapsed tier pass its floors",
        );
        assert!(
            err.contains("index-order surrogate"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_validate_retrieval_fixture_rejects_mixed_non_finite_scores() {
        for non_finite in [f32::NAN, f32::INFINITY] {
            let (mut corpus, queries) = fixed_retrieval_fixture();
            corpus[0][0] = non_finite;
            let err = validate_retrieval_fixture(&corpus, &queries, 10)
                .expect_err("the first non-finite reference score must invalidate the oracle");
            assert!(
                err.contains("query 0 candidate 0"),
                "unexpected error: {err}"
            );
            assert!(err.contains("non-finite"), "unexpected error: {err}");
        }
    }

    #[test]
    fn test_validate_retrieval_fixture_rejects_zero_top_k() {
        let (corpus, queries) = fixed_retrieval_fixture();
        let err = validate_retrieval_fixture(&corpus, &queries, 0)
            .expect_err("top_k=0 must be refused before recall divides by zero");
        assert!(err.contains("top_k=0"), "unexpected error: {err}");
    }

    #[test]
    fn test_validate_retrieval_fixture_rejects_near_tied_top_k_boundary() {
        let query = vec![1.0, 0.0];
        let corpus = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0001],
            vec![-1.0, 0.0],
        ];
        let best = scalar_cosine_f64(&query, &corpus[1]);
        let runner_up = scalar_cosine_f64(&query, &corpus[2]);
        let boundary_gap = best - runner_up;
        assert!(boundary_gap > 0.0 && boundary_gap < 1e-6);

        let err = validate_retrieval_fixture(&corpus, &[query], 1)
            .expect_err("an f64-only near-tie at the evaluated cutoff must be refused");
        assert!(err.contains("near-tied"), "unexpected error: {err}");
    }

    #[test]
    fn test_validate_retrieval_fixture_rejects_near_tied_pairwise_boundary() {
        let query = vec![1.0, 0.0];
        let corpus = vec![
            vec![-1.0, 0.0],
            vec![1.0, 0.0],
            vec![100.0, 1.0],
            vec![100.0, 1.0001],
        ];
        let second = scalar_cosine_f64(&query, &corpus[2]);
        let third = scalar_cosine_f64(&query, &corpus[3]);
        assert!(second > third);
        assert_eq!(second as f32, third as f32);

        let err = validate_retrieval_fixture(&corpus, &[query], 1)
            .expect_err("an f64-only near-tie evaluated by pairwise agreement must be refused");
        assert!(err.contains("near-tied"), "unexpected error: {err}");
    }

    #[test]
    fn test_tier_bytes_per_dim() {
        assert_eq!(QuantizationTier::Full.bytes_per_dim(), 4.0);
        assert_eq!(QuantizationTier::Int8.bytes_per_dim(), 1.0);
        assert_eq!(QuantizationTier::Int4.bytes_per_dim(), 0.5);
        assert_eq!(QuantizationTier::Binary.bytes_per_dim(), 0.125);
    }

    #[test]
    fn test_tier_compression_ratios() {
        assert_eq!(QuantizationTier::Full.compression_ratio(), 1.0);
        assert_eq!(QuantizationTier::Int8.compression_ratio(), 4.0);
        assert_eq!(QuantizationTier::Int4.compression_ratio(), 8.0);
        assert_eq!(QuantizationTier::Binary.compression_ratio(), 32.0);
    }

    #[test]
    fn test_tier_storage_bytes() {
        assert_eq!(QuantizationTier::Full.storage_bytes(384), 1536);
        assert_eq!(QuantizationTier::Int8.storage_bytes(384), 384);
        assert_eq!(QuantizationTier::Int4.storage_bytes(384), 192);
        assert_eq!(QuantizationTier::Binary.storage_bytes(384), 48);
    }

    #[test]
    fn test_tier_from_age() {
        assert_eq!(
            QuantizationTier::from_age_seconds(0),
            QuantizationTier::Full
        );
        assert_eq!(
            QuantizationTier::from_age_seconds(1800),
            QuantizationTier::Full
        ); // 30 min
        assert_eq!(
            QuantizationTier::from_age_seconds(7200),
            QuantizationTier::Int8
        ); // 2 hours
        assert_eq!(
            QuantizationTier::from_age_seconds(172800),
            QuantizationTier::Int4
        ); // 2 days
        assert_eq!(
            QuantizationTier::from_age_seconds(1_000_000),
            QuantizationTier::Binary
        ); // ~11 days
    }

    #[test]
    fn test_quantized_data_from_f32_all_tiers() {
        let v = generate_vector(384, 42);

        for tier in [
            QuantizationTier::Full,
            QuantizationTier::Int8,
            QuantizationTier::Int4,
            QuantizationTier::Binary,
        ] {
            let data = QuantizedData::from_f32(&v, tier);
            assert_eq!(data.tier(), tier, "tier mismatch for {tier:?}");
            assert_eq!(data.dims(), 384, "dims mismatch for {tier:?}");

            // Verify storage bytes match expected
            let expected_bytes = tier.storage_bytes(384);
            assert_eq!(
                data.storage_bytes(),
                expected_bytes,
                "storage bytes mismatch for {tier:?}"
            );
        }
    }

    #[test]
    fn test_approximate_cosine_distance_ordering() {
        // Vectors a and b should be "closer" than a and c.
        let a = generate_vector(384, 1);
        // b = a + small noise
        let b: Vec<f32> = a
            .iter()
            .enumerate()
            .map(|(i, &x)| x + 0.05 * (i as f32 * 0.3).sin())
            .collect();
        // c = random, uncorrelated
        let c = generate_vector(384, 999);

        for tier in [
            QuantizationTier::Full,
            QuantizationTier::Int8,
            QuantizationTier::Int4,
            QuantizationTier::Binary,
        ] {
            let stored_b = QuantizedData::from_f32(&b, tier);
            let stored_c = QuantizedData::from_f32(&c, tier);

            let dist_ab = approximate_cosine_distance(&a, &stored_b);
            let dist_ac = approximate_cosine_distance(&a, &stored_c);

            // a should be closer to b than to c at all tiers
            assert!(
                dist_ab < dist_ac,
                "{tier:?}: dist(a,b)={dist_ab} should be < dist(a,c)={dist_ac}"
            );
        }
    }

    #[test]
    fn test_promote_demote_roundtrip() {
        let v = generate_vector(384, 42);
        let binary = QuantizedData::from_f32(&v, QuantizationTier::Binary);

        // Promote Binary -> Int4 -> Int8 -> Full
        let int4 = binary.promote(QuantizationTier::Int4);
        assert_eq!(int4.tier(), QuantizationTier::Int4);

        let int8 = int4.promote(QuantizationTier::Int8);
        assert_eq!(int8.tier(), QuantizationTier::Int8);

        let full = int8.promote(QuantizationTier::Full);
        assert_eq!(full.tier(), QuantizationTier::Full);
        assert_eq!(full.dims(), 384);
    }

    #[test]
    fn test_int8_batch_prepared_matches_per_item_prepared() {
        let query = generate_vector(384, 42);
        let prepared = PreparedQuery::from_f32(&query, QuantizationTier::Int8);
        let candidates: Vec<QuantizedVector> = (0..32)
            .map(|i| QuantizedVector::from_f32(&generate_vector(384, i + 1)))
            .collect();
        let wrapped: Vec<QuantizedData> = candidates
            .iter()
            .cloned()
            .map(QuantizedData::Int8)
            .collect();

        let got = approximate_int8_batch_prepared(&prepared, &candidates).unwrap();
        for (i, item) in wrapped.iter().enumerate() {
            let expected = approximate_cosine_distance_prepared(&prepared, item).unwrap();
            assert!(
                (got[i] - expected).abs() < 1e-6,
                "int8 batch prepared mismatch at candidate {i}: got={}, expected={}",
                got[i],
                expected
            );
        }
    }

    #[test]
    fn test_int4_batch_prepared_matches_per_item_prepared() {
        let query = generate_vector(384, 42);
        let prepared = PreparedQuery::from_f32(&query, QuantizationTier::Int4);
        let candidates: Vec<Int4Vector> = (0..32)
            .map(|i| Int4Vector::from_f32(&generate_vector(384, i + 1)))
            .collect();
        let wrapped: Vec<QuantizedData> = candidates
            .iter()
            .cloned()
            .map(QuantizedData::Int4)
            .collect();

        let got = approximate_int4_batch_prepared(&prepared, &candidates).unwrap();
        for (i, item) in wrapped.iter().enumerate() {
            let expected = approximate_cosine_distance_prepared(&prepared, item).unwrap();
            assert!(
                (got[i] - expected).abs() < 1e-5,
                "int4 batch prepared mismatch at candidate {i}: got={}, expected={}",
                got[i],
                expected
            );
        }
    }

    #[test]
    fn test_int4_batch_prepared_api_dispatch_parity() {
        // Verify that approximate_int4_batch_prepared produces the same cosine distance
        // as approximate_cosine_distance_prepared for each candidate. On aarch64 both
        // sides dispatch to NEON; on other targets both use the packed scalar fallback.
        // For direct scalar-vs-NEON integer parity, see int4::tests::test_packed_scalar_matches_neon_exact.
        for dim in [1usize, 3, 31, 127, 383, 384] {
            let query = generate_vector(dim, 700 + dim as u64);
            let candidate = generate_vector(dim, 800 + dim as u64);
            let prepared = PreparedQuery::from_f32(&query, QuantizationTier::Int4);
            let q_cand = Int4Vector::from_f32(&candidate);
            let wrapped = QuantizedData::Int4(q_cand.clone());

            let batch_result = approximate_int4_batch_prepared(&prepared, &[q_cand]).unwrap();
            let per_item_result =
                approximate_cosine_distance_prepared(&prepared, &wrapped).unwrap();

            assert!(
                (batch_result[0] - per_item_result).abs() < 1e-5,
                "int4 batch prepared dispatch mismatch at dim={dim}: batch={}, per_item={}",
                batch_result[0],
                per_item_result
            );
        }
    }

    #[test]
    fn test_quantized_data_to_f32_roundtrip() {
        let v = generate_vector(384, 55);

        // Full tier should be lossless
        let full_data = QuantizedData::from_f32(&v, QuantizationTier::Full);
        let full_rt = full_data.to_f32();
        for (a, b) in v.iter().zip(full_rt.iter()) {
            assert!((a - b).abs() < 1e-10, "Full tier should be lossless");
        }
    }

    // ------------------------------------------------------------------
    // Regression tests for issue #210: tier-mismatch in prepared SIMD
    // dispatch must return a typed error, not panic.
    // ------------------------------------------------------------------

    #[test]
    fn test_cosine_distance_prepared_tier_mismatch_returns_typed_error() {
        let v = generate_vector(64, 1);
        let query = PreparedQuery::from_f32(&v, QuantizationTier::Int8);
        let stored = QuantizedData::from_f32(&v, QuantizationTier::Int4);

        let err = approximate_cosine_distance_prepared(&query, &stored).unwrap_err();
        match err {
            EmbedError::TierMismatch {
                op,
                expected,
                actual,
            } => {
                assert_eq!(op, "approximate_cosine_distance_prepared");
                assert_eq!(expected, QuantizationTier::Int4);
                assert_eq!(actual, QuantizationTier::Int8);
            }
            other => panic!("expected TierMismatch, got {other:?}"),
        }

        // try_ alias must agree.
        assert!(try_approximate_cosine_distance_prepared(&query, &stored).is_err());
    }

    #[test]
    fn test_dot_product_prepared_tier_mismatch_returns_typed_error() {
        let v = generate_vector(64, 2);
        let query = PreparedQuery::from_f32(&v, QuantizationTier::Full);
        let stored = QuantizedData::from_f32(&v, QuantizationTier::Int8);

        let err = approximate_dot_product_prepared(&query, &stored).unwrap_err();
        assert!(
            matches!(
                err,
                EmbedError::TierMismatch {
                    op: "approximate_dot_product_prepared",
                    ..
                }
            ),
            "unexpected error variant: {err:?}"
        );

        assert!(try_approximate_dot_product_prepared(&query, &stored).is_err());
    }

    #[test]
    fn test_dot_product_prepared_binary_returns_typed_error_not_panic() {
        let v = generate_vector(64, 3);
        let query = PreparedQuery::from_f32(&v, QuantizationTier::Binary);
        let stored = QuantizedData::from_f32(&v, QuantizationTier::Binary);

        let err = approximate_dot_product_prepared(&query, &stored).unwrap_err();
        assert!(
            matches!(err, EmbedError::Internal(_)),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn test_cosine_distance_prepared_with_meta_tier_mismatch_returns_typed_error() {
        let v = generate_vector(64, 4);
        let meta =
            PreparedQueryWithMeta::from_f32(&v, QuantizationTier::Full, NormalizationHint::Unknown);
        let stored = QuantizedData::from_f32(&v, QuantizationTier::Int8);

        let err = approximate_cosine_distance_prepared_with_meta(
            &meta,
            &stored,
            NormalizationHint::Unknown,
        )
        .unwrap_err();
        assert!(matches!(err, EmbedError::TierMismatch { .. }));
    }

    #[test]
    fn test_cosine_distance_prepared_with_meta_validates_stored_unit_norm() {
        let query = vec![std::f32::consts::FRAC_1_SQRT_2; 2];
        let meta = PreparedQueryWithMeta::from_f32(
            &query,
            QuantizationTier::Full,
            NormalizationHint::Unit,
        );
        let stored = QuantizedData::Full(vec![2.0, 0.0]);

        let got =
            approximate_cosine_distance_prepared_with_meta(&meta, &stored, NormalizationHint::Unit)
                .unwrap();
        let expected = approximate_cosine_distance_prepared(&meta.query, &stored).unwrap();

        assert!(
            (got - expected).abs() < 1e-6,
            "got={got}, expected={expected}"
        );
    }

    #[test]
    fn test_batch_cosine_distance_prepared_tier_mismatch_returns_typed_error() {
        let v = generate_vector(64, 5);
        let query = PreparedQuery::from_f32(&v, QuantizationTier::Int8);
        let stored = vec![
            QuantizedData::from_f32(&v, QuantizationTier::Int8),
            QuantizedData::from_f32(&v, QuantizationTier::Int4), // mismatched
        ];

        let err = batch_approximate_cosine_distance_prepared(&query, &stored).unwrap_err();
        assert!(matches!(err, EmbedError::TierMismatch { .. }));

        let mut out = vec![9.0, 9.0, 9.0]; // pre-populated, must be cleared even on error
        let err =
            batch_approximate_cosine_distance_prepared_into(&query, &stored, &mut out).unwrap_err();
        assert!(matches!(err, EmbedError::TierMismatch { .. }));
        assert!(
            out.is_empty(),
            "buffer must be cleared, not left with stale data"
        );
    }

    #[test]
    fn test_int8_batch_prepared_wrong_tier_returns_typed_error() {
        let v = generate_vector(64, 6);
        let query = PreparedQuery::from_f32(&v, QuantizationTier::Int4); // not Int8
        let candidates = vec![QuantizedVector::from_f32(&v)];

        let err = approximate_int8_batch_prepared(&query, &candidates).unwrap_err();
        match err {
            EmbedError::TierMismatch {
                op,
                expected,
                actual,
            } => {
                assert_eq!(op, "approximate_int8_batch_prepared");
                assert_eq!(expected, QuantizationTier::Int8);
                assert_eq!(actual, QuantizationTier::Int4);
            }
            other => panic!("expected TierMismatch, got {other:?}"),
        }

        let mut out = vec![9.0];
        let err = approximate_int8_batch_prepared_into(&query, &candidates, &mut out).unwrap_err();
        assert!(matches!(err, EmbedError::TierMismatch { .. }));
        assert!(
            out.is_empty(),
            "buffer must be cleared, not left with stale data"
        );
    }

    #[test]
    fn test_int4_batch_prepared_wrong_tier_returns_typed_error() {
        let v = generate_vector(64, 7);
        let query = PreparedQuery::from_f32(&v, QuantizationTier::Int8); // not Int4
        let candidates = vec![Int4Vector::from_f32(&v)];

        let err = approximate_int4_batch_prepared(&query, &candidates).unwrap_err();
        match err {
            EmbedError::TierMismatch {
                op,
                expected,
                actual,
            } => {
                assert_eq!(op, "approximate_int4_batch_prepared");
                assert_eq!(expected, QuantizationTier::Int4);
                assert_eq!(actual, QuantizationTier::Int8);
            }
            other => panic!("expected TierMismatch, got {other:?}"),
        }

        let mut out = vec![9.0];
        let err = approximate_int4_batch_prepared_into(&query, &candidates, &mut out).unwrap_err();
        assert!(matches!(err, EmbedError::TierMismatch { .. }));
        assert!(
            out.is_empty(),
            "buffer must be cleared, not left with stale data"
        );
    }
}
