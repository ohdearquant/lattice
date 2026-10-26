//! Runtime SIMD configuration and accessor.
use std::sync::OnceLock;

/// CPU SIMD features detected at runtime, used to pick a kernel path.
///
/// Each field reports one instruction set. Fields for another architecture are always `false`. New fields may be
/// added as more instruction sets are detected, so the struct is `#[non_exhaustive]`: read it through
/// [`simd_config`] or [`SimdConfig::detect`] rather than building one.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SimdConfig {
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub avx2_enabled: bool,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub fma_enabled: bool,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub avx512f_enabled: bool,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub neon_enabled: bool,
}

impl SimdConfig {
    /// Detects the SIMD features of the running CPU. On aarch64 NEON is part of the baseline and always reported.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2_enabled: is_x86_feature_detected!("avx2"),
                fma_enabled: is_x86_feature_detected!("fma"),
                avx512f_enabled: is_x86_feature_detected!("avx512f"),
                neon_enabled: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx2_enabled: false,
                fma_enabled: false,
                avx512f_enabled: false,
                neon_enabled: true,
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                avx2_enabled: false,
                fma_enabled: false,
                avx512f_enabled: false,
                neon_enabled: false,
            }
        }
    }
}

static SIMD_CONFIG: OnceLock<SimdConfig> = OnceLock::new();

/// Returns the SIMD features of the running CPU, detected once per process and cached.
pub fn simd_config() -> SimdConfig {
    *SIMD_CONFIG.get_or_init(SimdConfig::detect)
}
