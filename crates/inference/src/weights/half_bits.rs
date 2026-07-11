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

/// Round-to-nearest-even right shift used for mantissa truncation in the
/// f32-to-f16 direction.
#[inline]
fn round_shift_right_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    if shift >= 32 {
        return 0;
    }

    let base = value >> shift;
    let mask = (1u32 << shift) - 1;
    let remainder = value & mask;
    let half = 1u32 << (shift - 1);

    if remainder > half || (remainder == half && (base & 1) != 0) {
        base + 1
    } else {
        base
    }
}

/// Convert `f32` to an IEEE-754 binary16 (f16) bit pattern using
/// round-to-nearest-even.
///
/// Handles ±0, ±∞, NaN (payload preserved, quiet bit forced set, guaranteed
/// non-zero mantissa), subnormals, and overflow (rounds up to ±∞).
#[inline]
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) as u16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    // Inf or NaN
    if exp == 0xff {
        if frac == 0 {
            return sign | 0x7c00;
        }
        let mut payload = (frac >> 13) as u16;
        if payload == 0 {
            payload = 1;
        }
        payload |= 0x0200;
        return sign | 0x7c00 | (payload & 0x03ff);
    }

    // Zero or f32 subnormal (underflows to f16 zero)
    if exp == 0 {
        return sign;
    }

    let exp32 = exp - 127;

    // Overflow to infinity
    if exp32 > 15 {
        return sign | 0x7c00;
    }

    // Normal f16 range
    if exp32 >= -14 {
        let mut exp16 = (exp32 + 15) as u16;
        let mut frac16 = round_shift_right_even(frac, 13) as u16;

        if frac16 == 0x0400 {
            frac16 = 0;
            exp16 += 1;
            if exp16 >= 0x1f {
                return sign | 0x7c00;
            }
        }

        return sign | (exp16 << 10) | frac16;
    }

    // Subnormal f16 range
    let mant = frac | 0x0080_0000;
    let shift = (-exp32 - 1) as u32;
    if shift >= 32 {
        return sign;
    }

    let frac16 = round_shift_right_even(mant, shift) as u16;
    if frac16 == 0 {
        return sign;
    }
    if frac16 == 0x0400 {
        return sign | 0x0400;
    }

    sign | frac16
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference-quality tie-to-even f32->bf16 for tests only. Not used by
    /// production code (BF16 is a load-time source format in this crate,
    /// never an f32-to-bf16 encode target), but needed to build golden
    /// f32<->bf16 round-trip fixtures.
    fn f32_to_bf16_bits_reference(v: f32) -> u16 {
        let bits = v.to_bits();
        if v.is_nan() {
            return ((bits >> 16) as u16) | 0x0040;
        }
        let round_bit = (bits >> 16) & 1;
        let half = 0x7fff + round_bit;
        (bits.wrapping_add(half) >> 16) as u16
    }

    #[test]
    fn f16_bits_to_f32_exhaustive_roundtrip_matches_ieee754() {
        // Exhaustive check over all 65,536 f16 bit patterns: widening to f32
        // and narrowing back with our own encoder must reproduce the
        // original bits, except where IEEE-754 legitimately collapses
        // distinct encodings (only signaling-vs-quiet NaN payload variance,
        // which f32_to_f16_bits always normalizes to a canonical quiet NaN).
        for bits in 0u32..=0xffff {
            let bits = bits as u16;
            let widened = f16_bits_to_f32(bits);
            let exp = (bits >> 10) & 0x1f;
            let frac = bits & 0x03ff;
            if exp == 0x1f && frac != 0 {
                assert!(
                    widened.is_nan(),
                    "f16 NaN bits {bits:#06x} must widen to NaN"
                );
                continue;
            }
            let narrowed = f32_to_f16_bits(widened);
            assert_eq!(
                narrowed, bits,
                "roundtrip mismatch: bits={bits:#06x} widened={widened} narrowed={narrowed:#06x}"
            );
        }
    }

    #[test]
    fn f16_special_values() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0f32);
        assert!(f16_bits_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_bits_to_f32(0x8000), 0.0f32);
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0f32);
        assert_eq!(f16_bits_to_f32(0xbc00), -1.0f32);
        assert_eq!(f16_bits_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_bits_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_bits_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn f16_denormals_round_trip() {
        // Smallest subnormal f16 (2^-24) and a mid-range subnormal.
        for &bits in &[0x0001u16, 0x0200, 0x03ff] {
            let widened = f16_bits_to_f32(bits);
            assert!(widened.is_finite() && widened != 0.0);
            assert_eq!(f32_to_f16_bits(widened), bits);
        }
    }

    #[test]
    fn f32_to_f16_bits_signed_zero() {
        assert_eq!(f32_to_f16_bits(0.0f32), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0f32), 0x8000);
    }

    #[test]
    fn f32_to_f16_bits_overflow_to_infinity() {
        assert_eq!(f32_to_f16_bits(1.0e6), 0x7c00);
        assert_eq!(f32_to_f16_bits(-1.0e6), 0xfc00);
        assert_eq!(f32_to_f16_bits(f32::MAX), 0x7c00);
    }

    #[test]
    fn f32_to_f16_bits_nan_payload_preserved_and_quiet() {
        let bits = f32_to_f16_bits(f32::NAN);
        assert_eq!(bits & 0x7c00, 0x7c00, "exponent field must be all-ones");
        assert_ne!(bits & 0x03ff, 0, "mantissa must stay non-zero (quiet NaN)");
        assert_ne!(bits & 0x0200, 0, "quiet bit must be set");
    }

    #[test]
    fn f32_to_f16_bits_tie_to_even_rounding() {
        // Midpoint between 0x3c00 (1.0) and 0x3c01 rounds down to the even
        // mantissa (0x3c00). Midpoint between 0x3c01 and 0x3c02 rounds up to
        // the even mantissa on that side (0x3c02).
        let a = f16_bits_to_f32(0x3c00);
        let b = f16_bits_to_f32(0x3c01);
        let tie_low = (a + b) * 0.5;
        assert_eq!(f32_to_f16_bits(tie_low), 0x3c00);

        let c = f16_bits_to_f32(0x3c01);
        let d = f16_bits_to_f32(0x3c02);
        let tie_high = (c + d) * 0.5;
        assert_eq!(f32_to_f16_bits(tie_high), 0x3c02);
    }

    #[test]
    fn bf16_bits_to_f32_lossless_widen() {
        for bits in [0x0000u16, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc0] {
            let widened = bf16_bits_to_f32(bits);
            let renarrowed = f32_to_bf16_bits_reference(widened);
            assert_eq!(
                renarrowed, bits,
                "bf16 widen must be exactly reversible for bits={bits:#06x}"
            );
        }
    }

    #[test]
    fn bf16_bits_to_f32_special_values() {
        assert_eq!(bf16_bits_to_f32(0x0000), 0.0f32);
        assert!(bf16_bits_to_f32(0x8000).is_sign_negative());
        assert_eq!(bf16_bits_to_f32(0x3f80), 1.0f32);
        assert_eq!(bf16_bits_to_f32(0x7f80), f32::INFINITY);
        assert_eq!(bf16_bits_to_f32(0xff80), f32::NEG_INFINITY);
        assert!(bf16_bits_to_f32(0x7fc0).is_nan());
    }

    #[test]
    fn matches_f16_weights_original_impl_golden_values() {
        // Golden values previously asserted directly against
        // `weights::f16_weights::F16` before consolidation.
        let cases: &[(f32, u16)] = &[
            (0.0, 0x0000),
            (1.0, 0x3c00),
            (-1.0, 0xbc00),
            (2.0, 0x4000),
            (0.5, 0x3800),
            (65504.0, 0x7bff), // f16 MAX
        ];
        for &(f, bits) in cases {
            assert_eq!(f32_to_f16_bits(f), bits, "encode mismatch for {f}");
            assert_eq!(f16_bits_to_f32(bits), f, "decode mismatch for {bits:#06x}");
        }
    }
}
