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
        // vice versa) passes this test silently: a wrong infinity decode
        // constant and ties-away-from-zero rounding were both injected as
        // test mutations and left this test green. Independent
        // verification against a third-party decoder lives in
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

    /// Exhaustive independent-oracle table (lattice#684) for OCP FP8 E4M3
    /// (`torch.float8_e4m3fn`), covering all 256 bit patterns. Generated via:
    ///
    /// ```python
    /// import torch
    /// for b in range(256):
    ///     t = torch.tensor([b], dtype=torch.uint8).view(torch.float8_e4m3fn)
    ///     f = t.float().item()
    ///     # f != f (NaN) -> None; else f32 bit pattern of f -> Some(bits)
    /// ```
    ///
    /// `torch` shares no code with this module's hand-rolled bit math, so
    /// this is a genuine independent-decoder check, the same role the `half`
    /// crate plays for the f16/bf16 oracle tests above.
    #[cfg(feature = "f16")]
    const F8_E4M3_ORACLE: [(u8, Option<u32>); 256] = [
        (0x00, Some(0x00000000_u32)),
        (0x01, Some(0x3b000000_u32)),
        (0x02, Some(0x3b800000_u32)),
        (0x03, Some(0x3bc00000_u32)),
        (0x04, Some(0x3c000000_u32)),
        (0x05, Some(0x3c200000_u32)),
        (0x06, Some(0x3c400000_u32)),
        (0x07, Some(0x3c600000_u32)),
        (0x08, Some(0x3c800000_u32)),
        (0x09, Some(0x3c900000_u32)),
        (0x0a, Some(0x3ca00000_u32)),
        (0x0b, Some(0x3cb00000_u32)),
        (0x0c, Some(0x3cc00000_u32)),
        (0x0d, Some(0x3cd00000_u32)),
        (0x0e, Some(0x3ce00000_u32)),
        (0x0f, Some(0x3cf00000_u32)),
        (0x10, Some(0x3d000000_u32)),
        (0x11, Some(0x3d100000_u32)),
        (0x12, Some(0x3d200000_u32)),
        (0x13, Some(0x3d300000_u32)),
        (0x14, Some(0x3d400000_u32)),
        (0x15, Some(0x3d500000_u32)),
        (0x16, Some(0x3d600000_u32)),
        (0x17, Some(0x3d700000_u32)),
        (0x18, Some(0x3d800000_u32)),
        (0x19, Some(0x3d900000_u32)),
        (0x1a, Some(0x3da00000_u32)),
        (0x1b, Some(0x3db00000_u32)),
        (0x1c, Some(0x3dc00000_u32)),
        (0x1d, Some(0x3dd00000_u32)),
        (0x1e, Some(0x3de00000_u32)),
        (0x1f, Some(0x3df00000_u32)),
        (0x20, Some(0x3e000000_u32)),
        (0x21, Some(0x3e100000_u32)),
        (0x22, Some(0x3e200000_u32)),
        (0x23, Some(0x3e300000_u32)),
        (0x24, Some(0x3e400000_u32)),
        (0x25, Some(0x3e500000_u32)),
        (0x26, Some(0x3e600000_u32)),
        (0x27, Some(0x3e700000_u32)),
        (0x28, Some(0x3e800000_u32)),
        (0x29, Some(0x3e900000_u32)),
        (0x2a, Some(0x3ea00000_u32)),
        (0x2b, Some(0x3eb00000_u32)),
        (0x2c, Some(0x3ec00000_u32)),
        (0x2d, Some(0x3ed00000_u32)),
        (0x2e, Some(0x3ee00000_u32)),
        (0x2f, Some(0x3ef00000_u32)),
        (0x30, Some(0x3f000000_u32)),
        (0x31, Some(0x3f100000_u32)),
        (0x32, Some(0x3f200000_u32)),
        (0x33, Some(0x3f300000_u32)),
        (0x34, Some(0x3f400000_u32)),
        (0x35, Some(0x3f500000_u32)),
        (0x36, Some(0x3f600000_u32)),
        (0x37, Some(0x3f700000_u32)),
        (0x38, Some(0x3f800000_u32)),
        (0x39, Some(0x3f900000_u32)),
        (0x3a, Some(0x3fa00000_u32)),
        (0x3b, Some(0x3fb00000_u32)),
        (0x3c, Some(0x3fc00000_u32)),
        (0x3d, Some(0x3fd00000_u32)),
        (0x3e, Some(0x3fe00000_u32)),
        (0x3f, Some(0x3ff00000_u32)),
        (0x40, Some(0x40000000_u32)),
        (0x41, Some(0x40100000_u32)),
        (0x42, Some(0x40200000_u32)),
        (0x43, Some(0x40300000_u32)),
        (0x44, Some(0x40400000_u32)),
        (0x45, Some(0x40500000_u32)),
        (0x46, Some(0x40600000_u32)),
        (0x47, Some(0x40700000_u32)),
        (0x48, Some(0x40800000_u32)),
        (0x49, Some(0x40900000_u32)),
        (0x4a, Some(0x40a00000_u32)),
        (0x4b, Some(0x40b00000_u32)),
        (0x4c, Some(0x40c00000_u32)),
        (0x4d, Some(0x40d00000_u32)),
        (0x4e, Some(0x40e00000_u32)),
        (0x4f, Some(0x40f00000_u32)),
        (0x50, Some(0x41000000_u32)),
        (0x51, Some(0x41100000_u32)),
        (0x52, Some(0x41200000_u32)),
        (0x53, Some(0x41300000_u32)),
        (0x54, Some(0x41400000_u32)),
        (0x55, Some(0x41500000_u32)),
        (0x56, Some(0x41600000_u32)),
        (0x57, Some(0x41700000_u32)),
        (0x58, Some(0x41800000_u32)),
        (0x59, Some(0x41900000_u32)),
        (0x5a, Some(0x41a00000_u32)),
        (0x5b, Some(0x41b00000_u32)),
        (0x5c, Some(0x41c00000_u32)),
        (0x5d, Some(0x41d00000_u32)),
        (0x5e, Some(0x41e00000_u32)),
        (0x5f, Some(0x41f00000_u32)),
        (0x60, Some(0x42000000_u32)),
        (0x61, Some(0x42100000_u32)),
        (0x62, Some(0x42200000_u32)),
        (0x63, Some(0x42300000_u32)),
        (0x64, Some(0x42400000_u32)),
        (0x65, Some(0x42500000_u32)),
        (0x66, Some(0x42600000_u32)),
        (0x67, Some(0x42700000_u32)),
        (0x68, Some(0x42800000_u32)),
        (0x69, Some(0x42900000_u32)),
        (0x6a, Some(0x42a00000_u32)),
        (0x6b, Some(0x42b00000_u32)),
        (0x6c, Some(0x42c00000_u32)),
        (0x6d, Some(0x42d00000_u32)),
        (0x6e, Some(0x42e00000_u32)),
        (0x6f, Some(0x42f00000_u32)),
        (0x70, Some(0x43000000_u32)),
        (0x71, Some(0x43100000_u32)),
        (0x72, Some(0x43200000_u32)),
        (0x73, Some(0x43300000_u32)),
        (0x74, Some(0x43400000_u32)),
        (0x75, Some(0x43500000_u32)),
        (0x76, Some(0x43600000_u32)),
        (0x77, Some(0x43700000_u32)),
        (0x78, Some(0x43800000_u32)),
        (0x79, Some(0x43900000_u32)),
        (0x7a, Some(0x43a00000_u32)),
        (0x7b, Some(0x43b00000_u32)),
        (0x7c, Some(0x43c00000_u32)),
        (0x7d, Some(0x43d00000_u32)),
        (0x7e, Some(0x43e00000_u32)),
        (0x7f, None),
        (0x80, Some(0x80000000_u32)),
        (0x81, Some(0xbb000000_u32)),
        (0x82, Some(0xbb800000_u32)),
        (0x83, Some(0xbbc00000_u32)),
        (0x84, Some(0xbc000000_u32)),
        (0x85, Some(0xbc200000_u32)),
        (0x86, Some(0xbc400000_u32)),
        (0x87, Some(0xbc600000_u32)),
        (0x88, Some(0xbc800000_u32)),
        (0x89, Some(0xbc900000_u32)),
        (0x8a, Some(0xbca00000_u32)),
        (0x8b, Some(0xbcb00000_u32)),
        (0x8c, Some(0xbcc00000_u32)),
        (0x8d, Some(0xbcd00000_u32)),
        (0x8e, Some(0xbce00000_u32)),
        (0x8f, Some(0xbcf00000_u32)),
        (0x90, Some(0xbd000000_u32)),
        (0x91, Some(0xbd100000_u32)),
        (0x92, Some(0xbd200000_u32)),
        (0x93, Some(0xbd300000_u32)),
        (0x94, Some(0xbd400000_u32)),
        (0x95, Some(0xbd500000_u32)),
        (0x96, Some(0xbd600000_u32)),
        (0x97, Some(0xbd700000_u32)),
        (0x98, Some(0xbd800000_u32)),
        (0x99, Some(0xbd900000_u32)),
        (0x9a, Some(0xbda00000_u32)),
        (0x9b, Some(0xbdb00000_u32)),
        (0x9c, Some(0xbdc00000_u32)),
        (0x9d, Some(0xbdd00000_u32)),
        (0x9e, Some(0xbde00000_u32)),
        (0x9f, Some(0xbdf00000_u32)),
        (0xa0, Some(0xbe000000_u32)),
        (0xa1, Some(0xbe100000_u32)),
        (0xa2, Some(0xbe200000_u32)),
        (0xa3, Some(0xbe300000_u32)),
        (0xa4, Some(0xbe400000_u32)),
        (0xa5, Some(0xbe500000_u32)),
        (0xa6, Some(0xbe600000_u32)),
        (0xa7, Some(0xbe700000_u32)),
        (0xa8, Some(0xbe800000_u32)),
        (0xa9, Some(0xbe900000_u32)),
        (0xaa, Some(0xbea00000_u32)),
        (0xab, Some(0xbeb00000_u32)),
        (0xac, Some(0xbec00000_u32)),
        (0xad, Some(0xbed00000_u32)),
        (0xae, Some(0xbee00000_u32)),
        (0xaf, Some(0xbef00000_u32)),
        (0xb0, Some(0xbf000000_u32)),
        (0xb1, Some(0xbf100000_u32)),
        (0xb2, Some(0xbf200000_u32)),
        (0xb3, Some(0xbf300000_u32)),
        (0xb4, Some(0xbf400000_u32)),
        (0xb5, Some(0xbf500000_u32)),
        (0xb6, Some(0xbf600000_u32)),
        (0xb7, Some(0xbf700000_u32)),
        (0xb8, Some(0xbf800000_u32)),
        (0xb9, Some(0xbf900000_u32)),
        (0xba, Some(0xbfa00000_u32)),
        (0xbb, Some(0xbfb00000_u32)),
        (0xbc, Some(0xbfc00000_u32)),
        (0xbd, Some(0xbfd00000_u32)),
        (0xbe, Some(0xbfe00000_u32)),
        (0xbf, Some(0xbff00000_u32)),
        (0xc0, Some(0xc0000000_u32)),
        (0xc1, Some(0xc0100000_u32)),
        (0xc2, Some(0xc0200000_u32)),
        (0xc3, Some(0xc0300000_u32)),
        (0xc4, Some(0xc0400000_u32)),
        (0xc5, Some(0xc0500000_u32)),
        (0xc6, Some(0xc0600000_u32)),
        (0xc7, Some(0xc0700000_u32)),
        (0xc8, Some(0xc0800000_u32)),
        (0xc9, Some(0xc0900000_u32)),
        (0xca, Some(0xc0a00000_u32)),
        (0xcb, Some(0xc0b00000_u32)),
        (0xcc, Some(0xc0c00000_u32)),
        (0xcd, Some(0xc0d00000_u32)),
        (0xce, Some(0xc0e00000_u32)),
        (0xcf, Some(0xc0f00000_u32)),
        (0xd0, Some(0xc1000000_u32)),
        (0xd1, Some(0xc1100000_u32)),
        (0xd2, Some(0xc1200000_u32)),
        (0xd3, Some(0xc1300000_u32)),
        (0xd4, Some(0xc1400000_u32)),
        (0xd5, Some(0xc1500000_u32)),
        (0xd6, Some(0xc1600000_u32)),
        (0xd7, Some(0xc1700000_u32)),
        (0xd8, Some(0xc1800000_u32)),
        (0xd9, Some(0xc1900000_u32)),
        (0xda, Some(0xc1a00000_u32)),
        (0xdb, Some(0xc1b00000_u32)),
        (0xdc, Some(0xc1c00000_u32)),
        (0xdd, Some(0xc1d00000_u32)),
        (0xde, Some(0xc1e00000_u32)),
        (0xdf, Some(0xc1f00000_u32)),
        (0xe0, Some(0xc2000000_u32)),
        (0xe1, Some(0xc2100000_u32)),
        (0xe2, Some(0xc2200000_u32)),
        (0xe3, Some(0xc2300000_u32)),
        (0xe4, Some(0xc2400000_u32)),
        (0xe5, Some(0xc2500000_u32)),
        (0xe6, Some(0xc2600000_u32)),
        (0xe7, Some(0xc2700000_u32)),
        (0xe8, Some(0xc2800000_u32)),
        (0xe9, Some(0xc2900000_u32)),
        (0xea, Some(0xc2a00000_u32)),
        (0xeb, Some(0xc2b00000_u32)),
        (0xec, Some(0xc2c00000_u32)),
        (0xed, Some(0xc2d00000_u32)),
        (0xee, Some(0xc2e00000_u32)),
        (0xef, Some(0xc2f00000_u32)),
        (0xf0, Some(0xc3000000_u32)),
        (0xf1, Some(0xc3100000_u32)),
        (0xf2, Some(0xc3200000_u32)),
        (0xf3, Some(0xc3300000_u32)),
        (0xf4, Some(0xc3400000_u32)),
        (0xf5, Some(0xc3500000_u32)),
        (0xf6, Some(0xc3600000_u32)),
        (0xf7, Some(0xc3700000_u32)),
        (0xf8, Some(0xc3800000_u32)),
        (0xf9, Some(0xc3900000_u32)),
        (0xfa, Some(0xc3a00000_u32)),
        (0xfb, Some(0xc3b00000_u32)),
        (0xfc, Some(0xc3c00000_u32)),
        (0xfd, Some(0xc3d00000_u32)),
        (0xfe, Some(0xc3e00000_u32)),
        (0xff, None),
    ];

    #[cfg(feature = "f16")]
    #[test]
    fn f8_e4m3_bits_to_f32_matches_independent_torch_oracle_exhaustive() {
        let mut checked = 0u32;
        for &(bits, expected) in &F8_E4M3_ORACLE {
            let got = f8_e4m3_bits_to_f32(bits);
            match expected {
                Some(bits_expected) => {
                    assert_eq!(
                        got.to_bits(),
                        bits_expected,
                        "f8_e4m3_bits_to_f32({bits:#04x}) diverges from torch oracle: \
                         ours={:#010x} oracle={bits_expected:#010x}",
                        got.to_bits()
                    );
                }
                None => {
                    assert!(got.is_nan(), "bits {bits:#04x} must decode to NaN");
                }
            }
            checked += 1;
        }
        assert_eq!(
            checked, 256,
            "expected the full E4M3 8-bit space to be checked"
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f8_e4m3_special_values() {
        assert_eq!(f8_e4m3_bits_to_f32(0x00), 0.0f32);
        assert!(f8_e4m3_bits_to_f32(0x80).is_sign_negative());
        assert_eq!(f8_e4m3_bits_to_f32(0x80), 0.0f32);
        assert_eq!(f8_e4m3_bits_to_f32(0x38), 1.0f32); // 0_0111_000
        assert_eq!(f8_e4m3_bits_to_f32(0xb8), -1.0f32);
        assert_eq!(f8_e4m3_bits_to_f32(0x7e), 448.0f32); // largest finite E4M3
        assert!(f8_e4m3_bits_to_f32(0x7f).is_nan());
        assert!(f8_e4m3_bits_to_f32(0xff).is_nan());
        // No infinities in this "FN" variant — the top of the finite range
        // is a normal, non-special value.
        assert!(f8_e4m3_bits_to_f32(0x7e).is_finite());
    }

    /// Exhaustive independent-oracle table (lattice#684) for OCP FP8 E5M2
    /// (`torch.float8_e5m2`), covering all 256 bit patterns. Generated the
    /// same way as `F8_E4M3_ORACLE` above, substituting `torch.float8_e5m2`.
    #[cfg(feature = "f16")]
    const F8_E5M2_ORACLE: [(u8, Option<u32>); 256] = [
        (0x00, Some(0x00000000_u32)),
        (0x01, Some(0x37800000_u32)),
        (0x02, Some(0x38000000_u32)),
        (0x03, Some(0x38400000_u32)),
        (0x04, Some(0x38800000_u32)),
        (0x05, Some(0x38a00000_u32)),
        (0x06, Some(0x38c00000_u32)),
        (0x07, Some(0x38e00000_u32)),
        (0x08, Some(0x39000000_u32)),
        (0x09, Some(0x39200000_u32)),
        (0x0a, Some(0x39400000_u32)),
        (0x0b, Some(0x39600000_u32)),
        (0x0c, Some(0x39800000_u32)),
        (0x0d, Some(0x39a00000_u32)),
        (0x0e, Some(0x39c00000_u32)),
        (0x0f, Some(0x39e00000_u32)),
        (0x10, Some(0x3a000000_u32)),
        (0x11, Some(0x3a200000_u32)),
        (0x12, Some(0x3a400000_u32)),
        (0x13, Some(0x3a600000_u32)),
        (0x14, Some(0x3a800000_u32)),
        (0x15, Some(0x3aa00000_u32)),
        (0x16, Some(0x3ac00000_u32)),
        (0x17, Some(0x3ae00000_u32)),
        (0x18, Some(0x3b000000_u32)),
        (0x19, Some(0x3b200000_u32)),
        (0x1a, Some(0x3b400000_u32)),
        (0x1b, Some(0x3b600000_u32)),
        (0x1c, Some(0x3b800000_u32)),
        (0x1d, Some(0x3ba00000_u32)),
        (0x1e, Some(0x3bc00000_u32)),
        (0x1f, Some(0x3be00000_u32)),
        (0x20, Some(0x3c000000_u32)),
        (0x21, Some(0x3c200000_u32)),
        (0x22, Some(0x3c400000_u32)),
        (0x23, Some(0x3c600000_u32)),
        (0x24, Some(0x3c800000_u32)),
        (0x25, Some(0x3ca00000_u32)),
        (0x26, Some(0x3cc00000_u32)),
        (0x27, Some(0x3ce00000_u32)),
        (0x28, Some(0x3d000000_u32)),
        (0x29, Some(0x3d200000_u32)),
        (0x2a, Some(0x3d400000_u32)),
        (0x2b, Some(0x3d600000_u32)),
        (0x2c, Some(0x3d800000_u32)),
        (0x2d, Some(0x3da00000_u32)),
        (0x2e, Some(0x3dc00000_u32)),
        (0x2f, Some(0x3de00000_u32)),
        (0x30, Some(0x3e000000_u32)),
        (0x31, Some(0x3e200000_u32)),
        (0x32, Some(0x3e400000_u32)),
        (0x33, Some(0x3e600000_u32)),
        (0x34, Some(0x3e800000_u32)),
        (0x35, Some(0x3ea00000_u32)),
        (0x36, Some(0x3ec00000_u32)),
        (0x37, Some(0x3ee00000_u32)),
        (0x38, Some(0x3f000000_u32)),
        (0x39, Some(0x3f200000_u32)),
        (0x3a, Some(0x3f400000_u32)),
        (0x3b, Some(0x3f600000_u32)),
        (0x3c, Some(0x3f800000_u32)),
        (0x3d, Some(0x3fa00000_u32)),
        (0x3e, Some(0x3fc00000_u32)),
        (0x3f, Some(0x3fe00000_u32)),
        (0x40, Some(0x40000000_u32)),
        (0x41, Some(0x40200000_u32)),
        (0x42, Some(0x40400000_u32)),
        (0x43, Some(0x40600000_u32)),
        (0x44, Some(0x40800000_u32)),
        (0x45, Some(0x40a00000_u32)),
        (0x46, Some(0x40c00000_u32)),
        (0x47, Some(0x40e00000_u32)),
        (0x48, Some(0x41000000_u32)),
        (0x49, Some(0x41200000_u32)),
        (0x4a, Some(0x41400000_u32)),
        (0x4b, Some(0x41600000_u32)),
        (0x4c, Some(0x41800000_u32)),
        (0x4d, Some(0x41a00000_u32)),
        (0x4e, Some(0x41c00000_u32)),
        (0x4f, Some(0x41e00000_u32)),
        (0x50, Some(0x42000000_u32)),
        (0x51, Some(0x42200000_u32)),
        (0x52, Some(0x42400000_u32)),
        (0x53, Some(0x42600000_u32)),
        (0x54, Some(0x42800000_u32)),
        (0x55, Some(0x42a00000_u32)),
        (0x56, Some(0x42c00000_u32)),
        (0x57, Some(0x42e00000_u32)),
        (0x58, Some(0x43000000_u32)),
        (0x59, Some(0x43200000_u32)),
        (0x5a, Some(0x43400000_u32)),
        (0x5b, Some(0x43600000_u32)),
        (0x5c, Some(0x43800000_u32)),
        (0x5d, Some(0x43a00000_u32)),
        (0x5e, Some(0x43c00000_u32)),
        (0x5f, Some(0x43e00000_u32)),
        (0x60, Some(0x44000000_u32)),
        (0x61, Some(0x44200000_u32)),
        (0x62, Some(0x44400000_u32)),
        (0x63, Some(0x44600000_u32)),
        (0x64, Some(0x44800000_u32)),
        (0x65, Some(0x44a00000_u32)),
        (0x66, Some(0x44c00000_u32)),
        (0x67, Some(0x44e00000_u32)),
        (0x68, Some(0x45000000_u32)),
        (0x69, Some(0x45200000_u32)),
        (0x6a, Some(0x45400000_u32)),
        (0x6b, Some(0x45600000_u32)),
        (0x6c, Some(0x45800000_u32)),
        (0x6d, Some(0x45a00000_u32)),
        (0x6e, Some(0x45c00000_u32)),
        (0x6f, Some(0x45e00000_u32)),
        (0x70, Some(0x46000000_u32)),
        (0x71, Some(0x46200000_u32)),
        (0x72, Some(0x46400000_u32)),
        (0x73, Some(0x46600000_u32)),
        (0x74, Some(0x46800000_u32)),
        (0x75, Some(0x46a00000_u32)),
        (0x76, Some(0x46c00000_u32)),
        (0x77, Some(0x46e00000_u32)),
        (0x78, Some(0x47000000_u32)),
        (0x79, Some(0x47200000_u32)),
        (0x7a, Some(0x47400000_u32)),
        (0x7b, Some(0x47600000_u32)),
        (0x7c, Some(0x7f800000_u32)),
        (0x7d, None),
        (0x7e, None),
        (0x7f, None),
        (0x80, Some(0x80000000_u32)),
        (0x81, Some(0xb7800000_u32)),
        (0x82, Some(0xb8000000_u32)),
        (0x83, Some(0xb8400000_u32)),
        (0x84, Some(0xb8800000_u32)),
        (0x85, Some(0xb8a00000_u32)),
        (0x86, Some(0xb8c00000_u32)),
        (0x87, Some(0xb8e00000_u32)),
        (0x88, Some(0xb9000000_u32)),
        (0x89, Some(0xb9200000_u32)),
        (0x8a, Some(0xb9400000_u32)),
        (0x8b, Some(0xb9600000_u32)),
        (0x8c, Some(0xb9800000_u32)),
        (0x8d, Some(0xb9a00000_u32)),
        (0x8e, Some(0xb9c00000_u32)),
        (0x8f, Some(0xb9e00000_u32)),
        (0x90, Some(0xba000000_u32)),
        (0x91, Some(0xba200000_u32)),
        (0x92, Some(0xba400000_u32)),
        (0x93, Some(0xba600000_u32)),
        (0x94, Some(0xba800000_u32)),
        (0x95, Some(0xbaa00000_u32)),
        (0x96, Some(0xbac00000_u32)),
        (0x97, Some(0xbae00000_u32)),
        (0x98, Some(0xbb000000_u32)),
        (0x99, Some(0xbb200000_u32)),
        (0x9a, Some(0xbb400000_u32)),
        (0x9b, Some(0xbb600000_u32)),
        (0x9c, Some(0xbb800000_u32)),
        (0x9d, Some(0xbba00000_u32)),
        (0x9e, Some(0xbbc00000_u32)),
        (0x9f, Some(0xbbe00000_u32)),
        (0xa0, Some(0xbc000000_u32)),
        (0xa1, Some(0xbc200000_u32)),
        (0xa2, Some(0xbc400000_u32)),
        (0xa3, Some(0xbc600000_u32)),
        (0xa4, Some(0xbc800000_u32)),
        (0xa5, Some(0xbca00000_u32)),
        (0xa6, Some(0xbcc00000_u32)),
        (0xa7, Some(0xbce00000_u32)),
        (0xa8, Some(0xbd000000_u32)),
        (0xa9, Some(0xbd200000_u32)),
        (0xaa, Some(0xbd400000_u32)),
        (0xab, Some(0xbd600000_u32)),
        (0xac, Some(0xbd800000_u32)),
        (0xad, Some(0xbda00000_u32)),
        (0xae, Some(0xbdc00000_u32)),
        (0xaf, Some(0xbde00000_u32)),
        (0xb0, Some(0xbe000000_u32)),
        (0xb1, Some(0xbe200000_u32)),
        (0xb2, Some(0xbe400000_u32)),
        (0xb3, Some(0xbe600000_u32)),
        (0xb4, Some(0xbe800000_u32)),
        (0xb5, Some(0xbea00000_u32)),
        (0xb6, Some(0xbec00000_u32)),
        (0xb7, Some(0xbee00000_u32)),
        (0xb8, Some(0xbf000000_u32)),
        (0xb9, Some(0xbf200000_u32)),
        (0xba, Some(0xbf400000_u32)),
        (0xbb, Some(0xbf600000_u32)),
        (0xbc, Some(0xbf800000_u32)),
        (0xbd, Some(0xbfa00000_u32)),
        (0xbe, Some(0xbfc00000_u32)),
        (0xbf, Some(0xbfe00000_u32)),
        (0xc0, Some(0xc0000000_u32)),
        (0xc1, Some(0xc0200000_u32)),
        (0xc2, Some(0xc0400000_u32)),
        (0xc3, Some(0xc0600000_u32)),
        (0xc4, Some(0xc0800000_u32)),
        (0xc5, Some(0xc0a00000_u32)),
        (0xc6, Some(0xc0c00000_u32)),
        (0xc7, Some(0xc0e00000_u32)),
        (0xc8, Some(0xc1000000_u32)),
        (0xc9, Some(0xc1200000_u32)),
        (0xca, Some(0xc1400000_u32)),
        (0xcb, Some(0xc1600000_u32)),
        (0xcc, Some(0xc1800000_u32)),
        (0xcd, Some(0xc1a00000_u32)),
        (0xce, Some(0xc1c00000_u32)),
        (0xcf, Some(0xc1e00000_u32)),
        (0xd0, Some(0xc2000000_u32)),
        (0xd1, Some(0xc2200000_u32)),
        (0xd2, Some(0xc2400000_u32)),
        (0xd3, Some(0xc2600000_u32)),
        (0xd4, Some(0xc2800000_u32)),
        (0xd5, Some(0xc2a00000_u32)),
        (0xd6, Some(0xc2c00000_u32)),
        (0xd7, Some(0xc2e00000_u32)),
        (0xd8, Some(0xc3000000_u32)),
        (0xd9, Some(0xc3200000_u32)),
        (0xda, Some(0xc3400000_u32)),
        (0xdb, Some(0xc3600000_u32)),
        (0xdc, Some(0xc3800000_u32)),
        (0xdd, Some(0xc3a00000_u32)),
        (0xde, Some(0xc3c00000_u32)),
        (0xdf, Some(0xc3e00000_u32)),
        (0xe0, Some(0xc4000000_u32)),
        (0xe1, Some(0xc4200000_u32)),
        (0xe2, Some(0xc4400000_u32)),
        (0xe3, Some(0xc4600000_u32)),
        (0xe4, Some(0xc4800000_u32)),
        (0xe5, Some(0xc4a00000_u32)),
        (0xe6, Some(0xc4c00000_u32)),
        (0xe7, Some(0xc4e00000_u32)),
        (0xe8, Some(0xc5000000_u32)),
        (0xe9, Some(0xc5200000_u32)),
        (0xea, Some(0xc5400000_u32)),
        (0xeb, Some(0xc5600000_u32)),
        (0xec, Some(0xc5800000_u32)),
        (0xed, Some(0xc5a00000_u32)),
        (0xee, Some(0xc5c00000_u32)),
        (0xef, Some(0xc5e00000_u32)),
        (0xf0, Some(0xc6000000_u32)),
        (0xf1, Some(0xc6200000_u32)),
        (0xf2, Some(0xc6400000_u32)),
        (0xf3, Some(0xc6600000_u32)),
        (0xf4, Some(0xc6800000_u32)),
        (0xf5, Some(0xc6a00000_u32)),
        (0xf6, Some(0xc6c00000_u32)),
        (0xf7, Some(0xc6e00000_u32)),
        (0xf8, Some(0xc7000000_u32)),
        (0xf9, Some(0xc7200000_u32)),
        (0xfa, Some(0xc7400000_u32)),
        (0xfb, Some(0xc7600000_u32)),
        (0xfc, Some(0xff800000_u32)),
        (0xfd, None),
        (0xfe, None),
        (0xff, None),
    ];

    #[cfg(feature = "f16")]
    #[test]
    fn f8_e5m2_bits_to_f32_matches_independent_torch_oracle_exhaustive() {
        let mut checked = 0u32;
        for &(bits, expected) in &F8_E5M2_ORACLE {
            let got = f8_e5m2_bits_to_f32(bits);
            match expected {
                Some(bits_expected) => {
                    assert_eq!(
                        got.to_bits(),
                        bits_expected,
                        "f8_e5m2_bits_to_f32({bits:#04x}) diverges from torch oracle: \
                         ours={:#010x} oracle={bits_expected:#010x}",
                        got.to_bits()
                    );
                }
                None => {
                    assert!(got.is_nan(), "bits {bits:#04x} must decode to NaN");
                }
            }
            checked += 1;
        }
        assert_eq!(
            checked, 256,
            "expected the full E5M2 8-bit space to be checked"
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f8_e5m2_special_values() {
        assert_eq!(f8_e5m2_bits_to_f32(0x00), 0.0f32);
        assert!(f8_e5m2_bits_to_f32(0x80).is_sign_negative());
        assert_eq!(f8_e5m2_bits_to_f32(0x80), 0.0f32);
        assert_eq!(f8_e5m2_bits_to_f32(0x3c), 1.0f32); // 0_01111_00
        assert_eq!(f8_e5m2_bits_to_f32(0xbc), -1.0f32);
        assert_eq!(f8_e5m2_bits_to_f32(0x7c), f32::INFINITY);
        assert_eq!(f8_e5m2_bits_to_f32(0xfc), f32::NEG_INFINITY);
        assert!(f8_e5m2_bits_to_f32(0x7f).is_nan());
    }
}
