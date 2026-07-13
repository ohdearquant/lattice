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
// Gated on BOTH the `wasm` feature and the wasm32 target: `wasm-bindgen` is
// only ever a dependency on the wasm32 target table (see Cargo.toml), so if
// this module were feature-gated alone, enabling `--features wasm` on a
// native build would fail to find the `wasm_bindgen` crate. Gating on both
// makes enabling `wasm` on a non-wasm target a no-op instead of a build error.
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
    /// Compute cosine similarity between two vectors.
    ///
    /// Uses SIMD acceleration (AVX2/NEON) when available, with automatic scalar fallback.
    ///
    /// Returns a value between -1.0 and 1.0, where 1.0 indicates
    /// identical direction and -1.0 indicates opposite direction.
    ///
    /// # Performance
    ///
    /// | Dimension | Scalar | SIMD |
    /// |-----------|--------|------|
    /// | 384 | ~650ns | ~90ns |
    /// | 768 | ~1300ns | ~180ns |
    /// | 1024 | ~1700ns | ~240ns |
    ///
    /// # Example
    ///
    /// ```rust
    /// use lattice_embed::utils::cosine_similarity;
    ///
    /// let a = vec![1.0, 0.0, 0.0];
    /// let b = vec![1.0, 0.0, 0.0];
    /// assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.0001);
    ///
    /// let c = vec![1.0, 0.0, 0.0];
    /// let d = vec![0.0, 1.0, 0.0];
    /// assert!((cosine_similarity(&c, &d) - 0.0).abs() < 0.0001);
    /// ```
    #[inline]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        simd::cosine_similarity(a, b)
    }

    /// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
    ///
    /// Compute dot product between two vectors.
    ///
    /// For normalized vectors (like embeddings), this equals cosine similarity.
    /// Using `dot_product` directly on pre-normalized vectors is faster than
    /// `cosine_similarity` since it skips norm computation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lattice_embed::utils::dot_product;
    ///
    /// let a = vec![1.0, 2.0, 3.0];
    /// let b = vec![4.0, 5.0, 6.0];
    /// assert!((dot_product(&a, &b) - 32.0).abs() < 0.0001);
    /// ```
    #[inline]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        simd::dot_product(a, b)
    }

    /// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
    ///
    /// Normalize a vector to unit length (L2 normalization).
    ///
    /// Modifies the vector in place. After normalization, the vector
    /// will have magnitude 1.0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lattice_embed::utils::normalize;
    ///
    /// let mut v = vec![3.0, 4.0];
    /// normalize(&mut v);
    ///
    /// let magnitude: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    /// assert!((magnitude - 1.0).abs() < 0.0001);
    /// ```
    #[inline]
    pub fn normalize(vector: &mut [f32]) {
        simd::normalize(vector)
    }

    /// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
    ///
    /// Compute Euclidean distance between two vectors.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lattice_embed::utils::euclidean_distance;
    ///
    /// let a = vec![0.0, 0.0];
    /// let b = vec![3.0, 4.0];
    /// assert!((euclidean_distance(&a, &b) - 5.0).abs() < 0.0001);
    /// ```
    #[inline]
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        simd::euclidean_distance(a, b)
    }

    /// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
    ///
    /// Compute cosine similarities for many vector pairs (batch operation).
    ///
    /// More efficient than calling `cosine_similarity` in a loop.
    #[inline]
    pub fn batch_cosine_similarity(pairs: &[(&[f32], &[f32])]) -> Vec<f32> {
        simd::batch_cosine_similarity(pairs)
    }

    /// **Stable**: external consumers may depend on this; breaking changes require a SemVer bump.
    ///
    /// Compute dot products for many vector pairs (batch operation).
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
