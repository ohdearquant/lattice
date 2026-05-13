use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub(crate) struct SimdConfig {
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

pub(crate) fn simd_config() -> SimdConfig {
    *SIMD_CONFIG.get_or_init(SimdConfig::detect)
}
