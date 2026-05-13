//! SIMD-accelerated vector operations for embedding similarity.
//!
//! Provides optimized implementations with automatic fallback:
//! - **x86_64 (float32)**: AVX-512F > AVX2 + FMA > scalar
//! - **x86_64 (int8)**: AVX-512 VNNI > AVX2 > scalar
//! - **aarch64**: ARM NEON with multiple accumulators and loop unrolling
//! - **Other**: Scalar fallback
//!
//! ## Optimizations
//!
//! - **Multiple accumulators**: 4 parallel accumulators to break dependency chains
//! - **Loop unrolling**: Process 16/32/64 elements per iteration depending on ISA
//! - **AVX-512F**: Wide float32 kernels for dot, cosine, normalize, and distance
//! - **AVX-512 VNNI**: Integer VNNI instructions when available (quantized int8 path)

mod binary;
mod cosine;
mod distance;
mod dot_product;
mod int4;
mod normalize;
mod quantized;
mod tier;

#[cfg(test)]
mod tests;

// Re-export public API
pub use binary::BinaryVector;
pub use cosine::{batch_cosine_similarity, cosine_similarity};
pub use distance::{euclidean_distance, squared_euclidean_distance};
pub use dot_product::{
    DotBatch4Kernel, DotKernel, batch_dot_product, dot_product, dot_product_batch4,
    resolved_dot_product_batch4_kernel, resolved_dot_product_kernel,
};
pub use int4::{Int4Params, Int4Vector};
pub use normalize::normalize;
pub use quantized::{
    I8DotKernel, QuantizationParams, QuantizedVector, cosine_similarity_i8, dot_product_i8,
    dot_product_i8_raw, resolved_i8_dot_kernel,
};
pub use tier::{
    NormalizationHint, PreparedQuery, PreparedQueryWithMeta, QuantizationTier, QuantizedData,
    approximate_cosine_distance, approximate_cosine_distance_prepared,
    approximate_cosine_distance_prepared_with_meta, approximate_dot_product,
    approximate_dot_product_prepared, batch_approximate_cosine_distance_prepared,
    batch_approximate_cosine_distance_prepared_into, is_unit_norm, prepare_query,
    prepare_query_with_norm, try_approximate_cosine_distance_prepared,
    try_approximate_dot_product_prepared,
};

use std::sync::OnceLock;

/// **Unstable**: SIMD dispatch internals; fields may be added as new ISAs are supported.
///
/// SIMD configuration with runtime feature detection.
#[derive(Debug, Clone, Copy)]
pub struct SimdConfig {
    /// **Unstable**: AVX-512F support available (x86_64).
    pub avx512f_enabled: bool,
    /// **Unstable**: AVX2 support available (x86_64).
    pub avx2_enabled: bool,
    /// **Unstable**: FMA (Fused Multiply-Add) support available (x86_64).
    pub fma_enabled: bool,
    /// **Unstable**: AVX-512F + AVX-512VNNI support available (x86_64).
    pub avx512vnni_enabled: bool,
    /// **Unstable**: NEON support available (aarch64/ARM64).
    pub neon_enabled: bool,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self::detect()
    }
}

impl SimdConfig {
    /// **Unstable**: feature detection details may change as ISA support expands.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let avx512f_enabled = is_x86_feature_detected!("avx512f");

            Self {
                avx512f_enabled,
                avx2_enabled: is_x86_feature_detected!("avx2"),
                fma_enabled: is_x86_feature_detected!("fma"),
                avx512vnni_enabled: avx512f_enabled
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512vnni"),
                neon_enabled: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on aarch64, always available.
            Self {
                avx512f_enabled: false,
                avx2_enabled: false,
                fma_enabled: false,
                avx512vnni_enabled: false,
                neon_enabled: true,
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                avx512f_enabled: false,
                avx2_enabled: false,
                fma_enabled: false,
                avx512vnni_enabled: false,
                neon_enabled: false,
            }
        }
    }

    /// **Unstable**: check if any SIMD is available; logic may expand with new ISAs.
    #[inline]
    pub fn simd_available(&self) -> bool {
        self.avx512f_enabled || self.avx512vnni_enabled || self.avx2_enabled || self.neon_enabled
    }

    /// Force scalar-only mode (useful for testing).
    #[cfg(test)]
    pub fn scalar_only() -> Self {
        Self {
            avx512f_enabled: false,
            avx2_enabled: false,
            fma_enabled: false,
            avx512vnni_enabled: false,
            neon_enabled: false,
        }
    }
}

// Process-wide SIMD configuration (detected once).
static SIMD_CONFIG: OnceLock<SimdConfig> = OnceLock::new();

/// **Unstable**: SIMD dispatch internal; shape may change as new backends are added.
///
/// The config is detected once per process and cached.
#[inline]
pub fn simd_config() -> SimdConfig {
    *SIMD_CONFIG.get_or_init(SimdConfig::detect)
}
