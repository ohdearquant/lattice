//! # lattice-embed
//!
//! Pure-Rust local embedding generation, caching, SIMD vector operations, and model migration.
//! Most of the crate, including model loading and SIMD dispatch, is **Unstable**; consumers
//! should normally use it through `lattice-engine`.
//!
//! The exception is the stable 0.4.x ANN contract in `simd`: squared/ordinary L2 distance,
//! dot product, and cosine similarity retain their `(&[f32], &[f32]) -> f32` APIs and
//! documented degenerate-input behavior. SIMD is not bit-identical to scalar, so near-tie
//! ordering is not guaranteed.
//!
//! See `docs/design.md` for the architecture, service lifecycle, cache identity, and migration
//! boundaries; use `docs/INDEX.md` to find subsystem references.

#![warn(missing_docs)]
#![allow(clippy::clone_on_copy)]

pub mod backfill;
mod cache;
mod error;
pub mod migration;
mod model;
pub mod service;
pub mod simd;
pub mod types;
// Keep this gate aligned with wasm-bindgen's wasm32-only dependency — see docs/design.md.
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub mod wasm;

pub use cache::{CacheStats, DEFAULT_CACHE_CAPACITY, EmbeddingCache, ShardStats};
pub use error::{EmbedError, Result};
pub use model::{EmbeddingModel, MIN_MRL_OUTPUT_DIM, ModelConfig, ModelProvenance};
pub use service::{DEFAULT_MAX_BATCH_SIZE, EmbeddingRole, EmbeddingService, MAX_TEXT_CHARS};
pub use simd::{SimdConfig, simd_config};

#[cfg(feature = "native")]
pub use service::{CachedEmbeddingService, NativeEmbeddingService};

/// Utility functions for vector operations.
///
/// All functions in this module are SIMD-accelerated when available (AVX2 on x86_64, NEON on aarch64).
/// Runtime feature detection ensures automatic fallback to scalar implementations
/// on systems without SIMD support.
pub mod utils {
    use crate::simd;

    /// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
    ///
    /// Computes cosine similarity through the SIMD dispatcher.
    /// Returns `0.0` for empty, unequal-length, or zero-norm vectors.
    /// See [`docs/design.md`] (§Vector utility facade) for dispatch and performance details.
    #[inline]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        simd::cosine_similarity(a, b)
    }

    /// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
    ///
    /// Computes the dot product of two vectors through the SIMD dispatcher.
    /// Returns `0.0` for unequal-length vectors; for unit vectors it equals cosine similarity.
    /// See [`docs/design.md`] (§Vector utility facade) for dispatch and performance details.
    #[inline]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        simd::dot_product(a, b)
    }

    /// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
    ///
    /// L2-normalizes a vector in place through the SIMD dispatcher.
    /// Leaves zero- or NaN-norm vectors unchanged.
    /// See [`docs/design.md`] (§Vector utility facade) for dispatch and examples.
    #[inline]
    pub fn normalize(vector: &mut [f32]) {
        simd::normalize(vector)
    }

    /// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
    ///
    /// Computes Euclidean distance between two vectors through the SIMD dispatcher.
    /// Returns `f32::MAX` for unequal-length vectors.
    /// See [`docs/design.md`] (§Vector utility facade) for dispatch and examples.
    #[inline]
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        simd::euclidean_distance(a, b)
    }

    /// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
    ///
    /// Computes cosine similarities for vector pairs in input order.
    /// See [`docs/design.md`] (§Vector utility facade) for batch-dispatch behavior.
    #[inline]
    pub fn batch_cosine_similarity(pairs: &[(&[f32], &[f32])]) -> Vec<f32> {
        simd::batch_cosine_similarity(pairs)
    }

    /// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
    ///
    /// Computes dot products for vector pairs in input order.
    /// See [`docs/design.md`] (§Vector utility facade) for batch-dispatch behavior.
    #[inline]
    pub fn batch_dot_product(pairs: &[(&[f32], &[f32])]) -> Vec<f32> {
        simd::batch_dot_product(pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = utils::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = utils::cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = utils::cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        utils::normalize(&mut v);
        let magnitude: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = utils::euclidean_distance(&a, &b);
        assert!((dist - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_model_default() {
        let model = EmbeddingModel::default();
        assert_eq!(model.dimensions(), 384);
    }
}
