//! Dormant BERT CPU policy value contracts for sealed native preparation.

use super::simd::{SimdConfig, simd_config};

/// Frozen CPU capability facts captured for a future pinned BERT execution policy.
///
/// This value is metadata only. Possessing it does not prove that a
/// [`crate::model::BertModel`] was loaded or executed with pinned kernels. No model constructor
/// accepts this profile yet.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct BertCpuKernelProfile {
    target_architecture: &'static str,
    avx2_enabled: bool,
    fma_enabled: bool,
    avx512f_enabled: bool,
    neon_enabled: bool,
}

impl BertCpuKernelProfile {
    /// Captures the process-wide CPU capability set without reading environment variables.
    ///
    /// The captured facts are immutable and repeatable for the process lifetime. This method
    /// does not select kernels or make an existing model a pinned model.
    pub fn capture() -> Self {
        Self::from_detected(std::env::consts::ARCH, simd_config())
    }

    fn from_detected(target_architecture: &'static str, simd: SimdConfig) -> Self {
        Self {
            target_architecture,
            avx2_enabled: simd.avx2_enabled,
            fma_enabled: simd.fma_enabled,
            avx512f_enabled: simd.avx512f_enabled,
            neon_enabled: simd.neon_enabled,
        }
    }

    /// Returns the Rust target architecture captured by this profile.
    pub const fn target_architecture(&self) -> &'static str {
        self.target_architecture
    }

    /// Reports that the Lattice-owned scalar fallback is part of every capability set.
    pub const fn scalar_enabled(&self) -> bool {
        true
    }

    /// Reports the captured AVX2 capability.
    pub const fn avx2_enabled(&self) -> bool {
        self.avx2_enabled
    }

    /// Reports the captured FMA capability.
    pub const fn fma_enabled(&self) -> bool {
        self.fma_enabled
    }

    /// Reports the captured AVX-512F capability.
    pub const fn avx512f_enabled(&self) -> bool {
        self.avx512f_enabled
    }

    /// Reports the captured NEON capability.
    pub const fn neon_enabled(&self) -> bool {
        self.neon_enabled
    }
}

/// BERT CPU execution-policy request.
///
/// This is a dormant contract: existing BERT constructors and forward paths remain `Auto`, and
/// no constructor accepts `Pinned` yet. A later implementation must thread the frozen profile
/// through every output-affecting kernel before the policy can be treated as enforcement proof.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BertCpuKernelPolicy {
    /// Preserve the existing ambient platform dispatcher.
    Auto,
    /// Request the future pinned path with one immutable captured capability profile.
    Pinned(BertCpuKernelProfile),
}

impl BertCpuKernelPolicy {
    /// Borrows the captured profile only when the pinned policy was requested.
    pub const fn pinned_profile(&self) -> Option<&BertCpuKernelProfile> {
        match self {
            Self::Auto => None,
            Self::Pinned(profile) => Some(profile),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detected(
        avx2_enabled: bool,
        fma_enabled: bool,
        avx512f_enabled: bool,
        neon_enabled: bool,
    ) -> SimdConfig {
        SimdConfig {
            avx2_enabled,
            fma_enabled,
            avx512f_enabled,
            neon_enabled,
        }
    }

    #[test]
    fn synthetic_capability_rows_are_copied_without_aliasing_fields() {
        let rows = [
            ("scalar", detected(false, false, false, false)),
            ("x86-avx2", detected(true, false, false, false)),
            ("x86-avx2-fma", detected(true, true, false, false)),
            ("x86-avx512", detected(true, true, true, false)),
            ("aarch64-neon", detected(false, false, false, true)),
        ];

        for (target, capabilities) in rows {
            let profile = BertCpuKernelProfile::from_detected(target, capabilities);
            assert_eq!(profile.target_architecture(), target);
            assert!(profile.scalar_enabled());
            assert_eq!(profile.avx2_enabled(), capabilities.avx2_enabled);
            assert_eq!(profile.fma_enabled(), capabilities.fma_enabled);
            assert_eq!(profile.avx512f_enabled(), capabilities.avx512f_enabled);
            assert_eq!(profile.neon_enabled(), capabilities.neon_enabled);
        }
    }
}
