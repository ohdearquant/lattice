//! Single source of truth for scalar IEEE-754 half-precision (f16) and
//! bfloat16 (bf16) bit-pattern conversion.
//!
//! Before this module existed, the same hand-rolled sign/exponent/mantissa
//! bit-twiddling was independently reimplemented in five places across
//! `crates/inference/src` (`weights/f32_weights.rs`, `quant/quarot/io.rs`,
//! `weights/q4_weights.rs`, `weights/f16_weights.rs`, and
//! `forward/metal_qwen35.rs`). A precision-edge-case fix in one copy never
//! propagated to the others. This module is the one decoder every call site
//! now delegates to (lattice#799).
//!
//! Always compiled in every feature combination — QuaRot and Q4 quantization
//! need half-precision metadata inspection regardless of whether *runtime
//! F16/BF16 model loading* is gated behind the crate's `f16` feature. Only
//! the loading permission is feature-gated, never this bit conversion math.

pub(crate) use super::f16_encode::{f32_to_f16_bits, f32_to_finite_f16_bits};

/// Widen an IEEE-754 binary16 (f16) bit pattern to `f32`, exactly.
///
/// Handles signed zero, subnormals, infinities, and NaN (NaN payload is
/// widened losslessly into the f32 mantissa, quiet/signaling bit preserved).
#[inline]
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;

    let f32_bits = match (exp, frac) {
        // Zero (signed)
        (0, 0) => sign << 31,
        // Subnormal: normalize by shifting the leading 1 into bit 10, then
        // strip it and treat the remainder as the f32 mantissa.
        (0, _) => {
            let mut mant = frac;
            let mut e = -14i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (mant << 13)
        }
        // Infinity
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        // NaN
        (0x1f, _) => (sign << 31) | 0x7f80_0000 | (frac << 13),
        // Normal
        _ => (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 13),
    };

    f32::from_bits(f32_bits)
}

/// Widen a bfloat16 bit pattern to `f32`.
///
/// BF16 shares f32's sign+exponent layout truncated to a 7-bit mantissa, so
/// widening is a lossless zero-extend of the top 16 bits into an f32 word —
/// no rounding, no special-case branches needed.
#[inline]
pub(crate) fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Widen an OCP FP8 E4M3 (safetensors `F8_E4M3`, PyTorch `float8_e4m3fn`)
/// bit pattern to `f32`, exactly.
///
/// This is the "FN" (finite-only) OCP variant: 1 sign bit, 4 exponent bits
/// (bias 7), 3 mantissa bits, no infinities — the all-ones exponent field
/// is used for ordinary finite values (max magnitude 448) *except* the one
/// pattern where the mantissa is also all-ones, which is the sole NaN
/// encoding (both signs). This is deliberately distinct from `F8_E4M3FNUZ`
/// (AMD's variant: single unsigned NaN, no negative zero), which this
/// function does not decode (lattice#684).
#[cfg(feature = "f16")]
#[inline]
pub(crate) fn f8_e4m3_bits_to_f32(bits: u8) -> f32 {
    let sign = ((bits >> 7) & 0x1) as u32;
    let exp = ((bits >> 3) & 0x0f) as u32;
    let frac = (bits & 0x07) as u32;

    let f32_bits = match (exp, frac) {
        // Zero (signed)
        (0, 0) => sign << 31,
        // Subnormal: normalize by shifting the leading 1 into bit 3, then
        // strip it and treat the remainder as the f32 mantissa.
        (0, _) => {
            let mut mant = frac;
            let mut e = -6i32;
            while (mant & 0x08) == 0 {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x07;
            (sign << 31) | (((e + 127) as u32) << 23) | (mant << 20)
        }
        // The single E4M3FN NaN encoding: exponent and mantissa both all-ones.
        (0x0f, 0x07) => (sign << 31) | 0x7fc0_0000,
        // Normal (including exponent=0b1111 with a non-max mantissa, which
        // is a finite value in this "FN" variant, not infinity).
        _ => (sign << 31) | (((exp as i32 - 7 + 127) as u32) << 23) | (frac << 20),
    };

    f32::from_bits(f32_bits)
}

/// Widen an OCP FP8 E5M2 (safetensors `F8_E5M2`, PyTorch `float8_e5m2`) bit
/// pattern to `f32`, exactly.
///
/// Standard IEEE-754-style layout: 1 sign bit, 5 exponent bits (bias 15), 2
/// mantissa bits, with the all-ones exponent field reserved for infinity
/// (zero mantissa) and NaN (nonzero mantissa) — same shape as f16, just
/// narrower. Distinct from `F8_E5M2FNUZ`, which this function does not
/// decode (lattice#684).
#[cfg(feature = "f16")]
#[inline]
pub(crate) fn f8_e5m2_bits_to_f32(bits: u8) -> f32 {
    let sign = ((bits >> 7) & 0x1) as u32;
    let exp = ((bits >> 2) & 0x1f) as u32;
    let frac = (bits & 0x03) as u32;

    let f32_bits = match (exp, frac) {
        // Zero (signed)
        (0, 0) => sign << 31,
        // Subnormal: normalize by shifting the leading 1 into bit 2, then
        // strip it and treat the remainder as the f32 mantissa.
        (0, _) => {
            let mut mant = frac;
            let mut e = -14i32;
            while (mant & 0x04) == 0 {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03;
            (sign << 31) | (((e + 127) as u32) << 23) | (mant << 21)
        }
        // Infinity
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        // NaN
        (0x1f, _) => (sign << 31) | 0x7fc0_0000 | (frac << 21),
        // Normal
        _ => (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 21),
    };

    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests;
