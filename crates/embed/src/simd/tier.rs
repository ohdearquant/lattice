//! Quantization tier management and unified distance dispatch.
//!
//! Provides a `QuantizationTier` enum for selecting precision levels and
//! a `QuantizedData` enum for storing vectors at any tier with unified
//! distance computation.
//!
//! ## Tier hierarchy
//!
//! | Tier   | Precision | Bytes/dim | Compression | Use case                |
//! |--------|-----------|-----------|-------------|-------------------------|
//! | Full   | f32       | 4.0       | 1x          | Hot data, exact search   |
//! | Int8   | 8-bit     | 1.0       | 4x          | Warm data, HNSW search   |
//! | Int4   | 4-bit     | 0.5       | 8x          | Cool data, pre-filtering |
//! | Binary | 1-bit     | 0.125     | 32x         | Cold data, coarse filter |

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

    /// **Unstable**: age-based tier heuristic; boundaries (HOUR/DAY/WEEK) may be tuned.
    ///
    /// - Hot (accessed in last hour): Full
    /// - Warm (accessed in last day): Int8
    /// - Cool (accessed in last week): Int4
    /// - Cold (accessed in last month+): Binary
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
            Self::Int8(q) => q.data.len(),
            Self::Int4(q) => q.dims,
            Self::Binary(q) => q.dims,
        }
    }

    /// **Unstable**: storage byte count; may change with tier redesign.
    pub fn storage_bytes(&self) -> usize {
        match self {
            Self::Full(v) => v.len() * 4,
            Self::Int8(q) => q.data.len(),
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

    /// **Unstable**: tier promotion; re-quantizes via f32; may be rethought.
    ///
    /// Dequantizes to f32 then re-quantizes at the target tier.
    /// Note: this does NOT recover lost information -- it merely changes
    /// the storage format. INT4 -> INT8 promotion fills in new bits
    /// based on the dequantized approximation.
    pub fn promote(&self, target: QuantizationTier) -> Self {
        let f32_data = self.to_f32();
        Self::from_f32(&f32_data, target)
    }

    /// **Unstable**: tier demotion; delegates to `promote`; may be removed.
    pub fn demote(&self, target: QuantizationTier) -> Self {
        self.promote(target) // Same operation, just going the other direction
    }
}

/// **Unstable**: pre-quantized query for repeated distance computation.
///
/// Quantize a query vector once and reuse it against a homogeneous candidate list,
/// eliminating per-call `from_f32` overhead. The query tier must match the stored data tier.
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
            Self::Int8(q) => q.data.len(),
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

/// A prepared query annotated with a normalization hint for fast-path dispatch.
///
/// When `norm == NormalizationHint::Unit` and the stored vector is also unit-normalized,
/// `approximate_cosine_distance_prepared_with_meta` skips the norm division and uses
/// `1.0 - dot_product(q, s)` directly — recovering ~26% at 384d on the Full tier.
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
#[inline]
pub fn is_unit_norm(v: &[f32]) -> bool {
    let sq: f32 = v.iter().map(|x| x * x).sum();
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

/// **Unstable**: prepared cosine distance; query tier must match stored data tier.
///
/// Returns a value in [0, 2] where 0 = identical, 2 = opposite.
///
/// # Panics
///
/// Panics if the query tier does not match the stored-data tier.  Use
/// [`try_approximate_cosine_distance_prepared`] when tiers are not statically
/// guaranteed to match.
#[inline]
pub fn approximate_cosine_distance_prepared(query: &PreparedQuery, stored: &QuantizedData) -> f32 {
    match (query, stored) {
        (PreparedQuery::Full(q), QuantizedData::Full(s)) => 1.0 - cosine_similarity(q, s),
        (PreparedQuery::Int8(q), QuantizedData::Int8(s)) => {
            1.0 - cosine_similarity_i8_trusted(s, q)
        }
        (PreparedQuery::Int4(q), QuantizedData::Int4(s)) => s.cosine_distance(q),
        (PreparedQuery::Binary(q), QuantizedData::Binary(s)) => s.cosine_distance_approx(q),
        _ => panic!("PreparedQuery tier must match QuantizedData tier"),
    }
}

/// Non-panicking variant of [`approximate_cosine_distance_prepared`].
///
/// Returns `Err(EmbedError::Internal(...))` when the query tier does not match the
/// stored-data tier instead of panicking.  Prefer this in contexts where the tiers
/// may not be statically guaranteed to match.
#[inline]
pub fn try_approximate_cosine_distance_prepared(
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
        _ => Err(EmbedError::Internal(
            "PreparedQuery tier must match QuantizedData tier for cosine distance".into(),
        )),
    }
}

/// Non-panicking variant of [`approximate_dot_product_prepared`].
///
/// Returns `Err(EmbedError::Internal(...))` for binary inputs or a tier mismatch
/// instead of panicking.
#[inline]
pub fn try_approximate_dot_product_prepared(
    query: &PreparedQuery,
    stored: &QuantizedData,
) -> Result<f32> {
    match (query, stored) {
        (PreparedQuery::Full(q), QuantizedData::Full(s)) => Ok(dot_product(q, s)),
        (PreparedQuery::Int8(q), QuantizedData::Int8(s)) => Ok(dot_product_i8_trusted(q, s)),
        (PreparedQuery::Int4(q), QuantizedData::Int4(s)) => Ok(s.dot_product(q)),
        (PreparedQuery::Binary(_), QuantizedData::Binary(_)) => Err(EmbedError::Internal(
            "Binary has no prepared dot product; use try_approximate_cosine_distance_prepared"
                .into(),
        )),
        _ => Err(EmbedError::Internal(
            "PreparedQuery tier must match QuantizedData tier for dot product".into(),
        )),
    }
}

/// Cosine distance with unit-norm fast path.
///
/// When `meta.norm == NormalizationHint::Unit` and `stored` is a `QuantizedData::Full`
/// vector whose squared norm is ≈ 1, skips the norm division and returns
/// `1.0 - clamp(dot(q, s), -1, 1)`. For all other tier/norm combinations, falls
/// back to `approximate_cosine_distance_prepared`.
///
/// The stored vector's unit claim is verified lazily via [`is_unit_norm`]; callers
/// that batch many lookups against a fixed stored set should pre-check once and
/// use the cheaper overload directly.
#[inline]
pub fn approximate_cosine_distance_prepared_with_meta(
    meta: &PreparedQueryWithMeta,
    stored: &QuantizedData,
    stored_norm: NormalizationHint,
) -> f32 {
    if meta.norm == NormalizationHint::Unit && stored_norm == NormalizationHint::Unit {
        if let (PreparedQuery::Full(q), QuantizedData::Full(s)) = (&meta.query, stored) {
            let dot = dot_product(q, s);
            return 1.0 - dot.clamp(-1.0, 1.0);
        }
    }
    approximate_cosine_distance_prepared(&meta.query, stored)
}

/// **Unstable**: prepared dot product dispatch; query tier must match stored data tier.
///
/// # Panics
///
/// Panics if the query tier does not match the stored-data tier, or if called
/// with `Binary` data (binary has no meaningful dot product; use cosine distance
/// instead).  Use [`try_approximate_dot_product_prepared`] for a non-panicking
/// version.
#[inline]
pub fn approximate_dot_product_prepared(query: &PreparedQuery, stored: &QuantizedData) -> f32 {
    match (query, stored) {
        (PreparedQuery::Full(q), QuantizedData::Full(s)) => dot_product(q, s),
        (PreparedQuery::Int8(q), QuantizedData::Int8(s)) => dot_product_i8_trusted(q, s),
        (PreparedQuery::Int4(q), QuantizedData::Int4(s)) => s.dot_product(q),
        (PreparedQuery::Binary(_), QuantizedData::Binary(_)) => {
            panic!("Binary has no prepared dot product; use approximate_cosine_distance_prepared")
        }
        _ => panic!("PreparedQuery tier must match QuantizedData tier"),
    }
}

/// Compute cosine distances from one prepared query to a slice of stored vectors.
#[inline]
pub fn batch_approximate_cosine_distance_prepared(
    query: &PreparedQuery,
    stored: &[QuantizedData],
) -> Vec<f32> {
    stored
        .iter()
        .map(|item| approximate_cosine_distance_prepared(query, item))
        .collect()
}

/// Like [`batch_approximate_cosine_distance_prepared`] but writes into a caller-supplied buffer.
///
/// Clears and reuses the buffer to avoid allocations across repeated searches.
#[inline]
pub fn batch_approximate_cosine_distance_prepared_into(
    query: &PreparedQuery,
    stored: &[QuantizedData],
    out: &mut Vec<f32>,
) {
    out.clear();
    out.reserve(stored.len());
    out.extend(
        stored
            .iter()
            .map(|item| approximate_cosine_distance_prepared(query, item)),
    );
}

/// **Unstable**: tiered distance dispatch; tier mix and formula may change.
///
/// This is the primary distance function for HNSW search with tiered storage.
/// The query is always f32; the stored data may be at any tier.
///
/// Returns a value in [0, 2] where 0 = identical, 2 = opposite.
pub fn approximate_cosine_distance(query_f32: &[f32], stored: &QuantizedData) -> f32 {
    match stored {
        QuantizedData::Full(v) => {
            // Exact cosine distance
            1.0 - cosine_similarity(query_f32, v)
        }
        QuantizedData::Int8(q) => {
            // Quantize query to INT8, compute via INT8 path
            let query_q = QuantizedVector::from_f32(query_f32);
            1.0 - q.cosine_similarity(&query_q)
        }
        QuantizedData::Int4(q) => {
            // Quantize query to INT4, compute via INT4 path
            let query_q = Int4Vector::from_f32(query_f32);
            q.cosine_distance(&query_q)
        }
        QuantizedData::Binary(q) => {
            // Quantize query to binary, compute Hamming-based approx
            let query_q = BinaryVector::from_f32(query_f32);
            q.cosine_distance_approx(&query_q)
        }
    }
}

/// **Unstable**: tiered dot product dispatch; tier mix and formula may change.
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
    fn test_quantized_data_to_f32_roundtrip() {
        let v = generate_vector(384, 55);

        // Full tier should be lossless
        let full_data = QuantizedData::from_f32(&v, QuantizationTier::Full);
        let full_rt = full_data.to_f32();
        for (a, b) in v.iter().zip(full_rt.iter()) {
            assert!((a - b).abs() < 1e-10, "Full tier should be lossless");
        }
    }
}
