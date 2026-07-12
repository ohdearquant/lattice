//! Weight module index for f16, f32, q4, and q8 weights, with f32 weight re-exports.
pub mod f16_weights;
pub mod f32_weights;
pub(crate) mod half_bits;
pub(crate) mod ingress;
pub mod q4_weights;
pub mod q8_weights;

// Re-export from f32_weights (the primary/base weight types)
pub use self::f32_weights::*;

/// Bench-only, `bench-internals`-gated access to the production
/// scalar f16/bf16 decoders in [`half_bits`] (lattice#799): `f16_convert_bench`
/// previously carried its own copied scalar decoder instead of calling the
/// migrated production path, so its Criterion numbers said nothing about
/// this crate's actual conversion call sites. This module keeps the crate's
/// default public API unchanged — outside
/// `bench-internals` builds, `half_bits`'s functions stay `pub(crate)`,
/// reachable only from this crate's own load/quantization call sites, the
/// same visibility discipline `forward::metal_qwen35::bench_support` uses
/// for its own bench-only surface.
#[cfg(feature = "bench-internals")]
pub mod bench_support {
    /// Bench-only re-export of [`super::half_bits::f16_bits_to_f32`].
    #[inline]
    pub fn f16_bits_to_f32(bits: u16) -> f32 {
        super::half_bits::f16_bits_to_f32(bits)
    }

    /// Bench-only re-export of [`super::half_bits::f32_to_f16_bits`].
    #[inline]
    pub fn f32_to_f16_bits(v: f32) -> u16 {
        super::half_bits::f32_to_f16_bits(v)
    }

    /// Bench-only re-export of [`super::half_bits::bf16_bits_to_f32`].
    #[inline]
    pub fn bf16_bits_to_f32(bits: u16) -> f32 {
        super::half_bits::bf16_bits_to_f32(bits)
    }
}
