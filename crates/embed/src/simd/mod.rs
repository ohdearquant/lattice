//! SIMD vector operations for embedding similarity and compressed-vector search.
//!
//! Dispatch uses the best supported target kernel with scalar fallbacks; WASM
//! SIMD128 is selected at compile time. The public SIMD surface is unstable.
//!
//! See docs/simd.md for the kernel family, dispatch, and quantization design.

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
pub use cosine::{
    batch_cosine_one_vs_many, batch_cosine_similarity, cosine_similarity, cosine_similarity_fused,
};
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
    approximate_dot_product_prepared, approximate_int4_batch_prepared,
    approximate_int4_batch_prepared_into, approximate_int8_batch_prepared,
    approximate_int8_batch_prepared_into, batch_approximate_cosine_distance_prepared,
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
    /// **Unstable**: ARM FEAT_DotProd (SDOT/UDOT instructions) available (aarch64).
    ///
    /// Mandatory on Armv8.4+; optional on Armv8.2/v8.3. Always false on non-aarch64.
    /// SDOT kernels must only be dispatched when this is `true`.
    pub dotprod_enabled: bool,
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
                dotprod_enabled: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on aarch64, always available.
            // FEAT_DotProd (dotprod) is optional: required on Armv8.4+,
            // optional on Armv8.2/v8.3. Detect at runtime.
            Self {
                avx512f_enabled: false,
                avx2_enabled: false,
                fma_enabled: false,
                avx512vnni_enabled: false,
                neon_enabled: true,
                dotprod_enabled: std::arch::is_aarch64_feature_detected!("dotprod"),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            // No runtime detection on wasm32: `simd128` is either compiled in
            // for the whole module (via `-C target-feature=+simd128`) or not.
            // `cfg!` reads the same compile-time flag the `#[cfg(...)]` gates
            // on the SIMD kernel functions themselves key off, so this stays
            // consistent with which kernels actually exist in the binary.
            Self {
                avx512f_enabled: false,
                avx2_enabled: false,
                fma_enabled: false,
                avx512vnni_enabled: false,
                neon_enabled: false,
                dotprod_enabled: false,
            }
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )))]
        {
            Self {
                avx512f_enabled: false,
                avx2_enabled: false,
                fma_enabled: false,
                avx512vnni_enabled: false,
                neon_enabled: false,
                dotprod_enabled: false,
            }
        }
    }

    /// **Unstable**: reports compile-time wasm32 SIMD128 availability.
    ///
    /// This mirrors the build's `target-feature=+simd128` setting; no
    /// `SimdConfig` instance can override it at runtime. See docs/simd.md.
    #[inline]
    pub fn simd128_enabled(&self) -> bool {
        cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
    }

    /// **Unstable**: check if any SIMD is available; logic may expand with new ISAs.
    #[inline]
    pub fn simd_available(&self) -> bool {
        self.avx512f_enabled
            || self.avx512vnni_enabled
            || self.avx2_enabled
            || self.neon_enabled
            || self.simd128_enabled()
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
            dotprod_enabled: false,
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
