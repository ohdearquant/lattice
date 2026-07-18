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

/// Test whether an IEEE-754 binary16 (f16) bit pattern encodes a finite
/// value (not infinity, not NaN), without widening to `f32`.
///
/// An f16 bit pattern is infinity or NaN exactly when its 5-bit exponent
/// field (bits 10..15) is all-ones (`0x1f`); every other exponent value is
/// finite (zero, subnormal, or normal). Checking the exponent field directly
/// avoids a widen-then-`is_finite()` round trip through `f32`.
#[inline]
pub(crate) fn f16_bits_is_finite(bits: u16) -> bool {
    ((bits >> 10) & 0x1f) != 0x1f
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

    /// A signaling-NaN f16 bit pattern: all-ones exponent, nonzero mantissa,
    /// mantissa MSB (the quiet bit, bit 9) clear.
    fn is_signaling_nan_bits(bits: u16) -> bool {
        let exp = (bits >> 10) & 0x1f;
        let frac = bits & 0x03ff;
        exp == 0x1f && frac != 0 && (frac & 0x0200) == 0
    }

    #[test]
    fn f16_widen_narrow_composition_round_trips_self_consistently() {
        // NOTE: this test only proves `f16_bits_to_f32` and `f32_to_f16_bits`
        // are mutually consistent (encode(decode(bits)) == bits) — it does
        // NOT prove either function matches IEEE-754 or an external decoder,
        // because both sides of the comparison come from this module. A
        // decode bug that is exactly undone by a matching encode bug (or
        // vice versa) passes this test silently. Independent verification
        // against a third-party decoder lives in
        // `f16_bits_to_f32_matches_independent_half_crate_oracle` and
        // `f16_bits_to_f32_signaling_nan_is_lossless_widen_independent_of_decoder`
        // below; do not treat this test alone as a correctness guarantee.
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

    /// Independent-oracle equivalence check (lattice#799):
    /// for every non-signaling-NaN f16 bit pattern (zero, subnormal, normal,
    /// infinity, quiet NaN), `f16_bits_to_f32` must produce the exact same
    /// f32 bits as the third-party `half` crate's `f16::from_bits().to_f32()`
    /// — a decoder this module shares no code with. `half` is already an
    /// unconditional workspace dependency (`half.workspace = true` in
    /// `crates/inference/Cargo.toml`, used directly elsewhere in this crate,
    /// e.g. `kv_cache/flat.rs`), so this adds no new dependency.
    ///
    /// Signaling NaN bit patterns are excluded here on purpose, not skipped
    /// out of laziness: on this machine `half::f16::to_f32()` dispatches to
    /// the AArch64 hardware `fcvt` instruction (via runtime
    /// `is_aarch64_feature_detected!("fp16")`), and ARM's FCVT forces the
    /// quiet bit on a signaling-NaN operand per the architecture's default
    /// NaN-propagation rule for conversions. That is correct, real hardware
    /// behavior for `half`, but it means `half` does NOT perform a pure
    /// lossless bit-widen for signaling NaNs on this platform — comparing
    /// against it here would fail on all 1,022 signaling-NaN patterns for a
    /// reason that has nothing to do with `half_bits`. This module's own
    /// documented contract (preserve the signaling/quiet distinction
    /// exactly, matching a pure software widen) is verified independently,
    /// without going through `half`, in the sibling test below.
    #[test]
    fn f16_bits_to_f32_matches_independent_half_crate_oracle() {
        let mut checked = 0u32;
        for bits in 0u32..=0xffff {
            let bits = bits as u16;
            if is_signaling_nan_bits(bits) {
                continue;
            }
            let ours = f16_bits_to_f32(bits).to_bits();
            let oracle = half::f16::from_bits(bits).to_f32().to_bits();
            assert_eq!(
                ours, oracle,
                "f16_bits_to_f32({bits:#06x}) diverges from the `half` crate oracle: \
                 ours={ours:#010x} oracle={oracle:#010x}"
            );
            checked += 1;
        }
        // 65,536 total patterns minus the 1,022 excluded signaling NaNs
        // (511 payloads x 2 signs) confirms the exclusion is exact, not an
        // accidentally-empty sweep.
        assert_eq!(
            checked,
            65536 - 1022,
            "expected exactly the non-signaling-NaN f16 bit space to be checked"
        );
    }

    /// Independent, hand-derived (not decoder-composed) check that signaling
    /// NaN widening is a pure lossless bit-widen: sign preserved, exponent
    /// field forced all-ones, and the f32 mantissa is exactly the f16
    /// mantissa left-shifted by 13 with the low 13 bits zero-filled — the
    /// textbook IEEE-754 widening formula, computed here directly from the
    /// bit pattern rather than by calling any function in this module. This
    /// is what makes the signaling/quiet distinction claim in
    /// `f16_bits_to_f32`'s doc comment independently verifiable even though
    /// the third-party oracle above cannot be used for these patterns (see
    /// its doc comment for why).
    #[test]
    fn f16_bits_to_f32_signaling_nan_is_lossless_widen_independent_of_decoder() {
        let mut checked = 0u32;
        for bits in 0u32..=0xffff {
            let bits = bits as u16;
            if !is_signaling_nan_bits(bits) {
                continue;
            }
            let sign = (bits >> 15) & 0x1;
            let frac = (bits & 0x03ff) as u32;
            let expected_bits = ((sign as u32) << 31) | 0x7f80_0000 | (frac << 13);

            let widened = f16_bits_to_f32(bits);
            assert!(
                widened.is_nan(),
                "signaling NaN bits {bits:#06x} must widen to NaN"
            );
            assert_eq!(
                widened.to_bits(),
                expected_bits,
                "signaling NaN {bits:#06x} did not widen losslessly: \
                 got={:#010x} expected={expected_bits:#010x}",
                widened.to_bits()
            );
            // The signaling bit (mantissa MSB, f32 bit 22) must stay clear —
            // a decoder that force-quiets NaNs (like this platform's `half`
            // hardware path) would set it and this assertion would catch it.
            assert_eq!(
                widened.to_bits() & 0x0040_0000,
                0,
                "signaling NaN {bits:#06x} must NOT be quieted by decode"
            );
            checked += 1;
        }
        assert_eq!(checked, 1022, "expected exactly 511 payloads x 2 signs");
    }

    /// Independent boundary check (lattice#799) at the
    /// subnormal/normal f16 encoding edge, computed from literal f32 values
    /// (not by calling `f16_bits_to_f32`) and verified against the `half`
    /// crate oracle. The smallest normal f16 is 2^-14 (bits `0x0400`); the
    /// largest subnormal is 2^-14 * (1023/1024) (bits `0x03ff`).
    #[test]
    fn f32_to_f16_bits_subnormal_normal_boundary_matches_independent_oracle() {
        let smallest_normal = 2f32.powi(-14);
        let largest_subnormal = 2f32.powi(-14) * (1023.0 / 1024.0);
        // One ULP below the subnormal/normal boundary on each side (f16
        // subnormal ULP is 2^-24), staying inside its own bin.
        let just_below_boundary = largest_subnormal - 2f32.powi(-25); // rounds down, stays subnormal
        let just_above_boundary = smallest_normal + 2f32.powi(-25); // rounds up, stays normal

        for v in [
            largest_subnormal,
            smallest_normal,
            just_below_boundary,
            just_above_boundary,
        ] {
            let ours = f32_to_f16_bits(v);
            let oracle = half::f16::from_f32(v).to_bits();
            assert_eq!(
                ours, oracle,
                "f32_to_f16_bits({v}) diverges from the `half` crate oracle at the \
                 subnormal/normal boundary: ours={ours:#06x} oracle={oracle:#06x}"
            );
        }
        assert_eq!(f32_to_f16_bits(largest_subnormal), 0x03ff);
        assert_eq!(f32_to_f16_bits(smallest_normal), 0x0400);
    }

    /// Independent boundary check (lattice#799) at the
    /// finite/infinity f16 encoding edge, computed from literal f32 values
    /// and verified against the `half` crate oracle. `65504.0` (bits
    /// `0x7bff`) is the largest finite f16; the f16 ULP at that exponent is
    /// 32, so `65504 + 16 = 65520` is the exact round-to-nearest-even
    /// midpoint between the largest finite value and overflow to infinity.
    #[test]
    fn f32_to_f16_bits_finite_infinity_boundary_matches_independent_oracle() {
        let f16_max = 65504.0f32;
        let midpoint = 65520.0f32;
        let just_below_midpoint = 65519.0f32; // rounds down, stays finite
        let just_above_midpoint = 65521.0f32; // rounds up, overflows to infinity

        for v in [f16_max, midpoint, just_below_midpoint, just_above_midpoint] {
            let ours = f32_to_f16_bits(v);
            let oracle = half::f16::from_f32(v).to_bits();
            assert_eq!(
                ours, oracle,
                "f32_to_f16_bits({v}) diverges from the `half` crate oracle at the \
                 finite/infinity boundary: ours={ours:#06x} oracle={oracle:#06x}"
            );
        }
        assert_eq!(f32_to_f16_bits(f16_max), 0x7bff);
        assert_eq!(f32_to_f16_bits(just_below_midpoint), 0x7bff);
        assert_eq!(f32_to_f16_bits(just_above_midpoint), 0x7c00);
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
    fn f16_bits_is_finite_matches_widen_is_finite_across_full_space() {
        for bits in 0u32..=0xffff {
            let bits = bits as u16;
            assert_eq!(
                f16_bits_is_finite(bits),
                f16_bits_to_f32(bits).is_finite(),
                "f16_bits_is_finite({bits:#06x}) diverges from widen-then-is_finite"
            );
        }
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
