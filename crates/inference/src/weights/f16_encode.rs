//! Scalar IEEE-754 f32-to-f16 bit-pattern conversion shared by the library
//! and package-local offline quantizer binary.

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

/// Convert `f32` to an IEEE-754 binary16 bit pattern using
/// round-to-nearest-even.
#[inline]
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) as u16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

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

    if exp == 0 {
        return sign;
    }

    let exp32 = exp - 127;
    if exp32 > 15 {
        return sign | 0x7c00;
    }

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

/// Convert `f32` to finite f16 bits, returning the non-finite encoding on
/// NaN, infinity, or narrowing overflow.
#[inline]
pub(crate) fn f32_to_finite_f16_bits(v: f32) -> Result<u16, u16> {
    let bits = f32_to_f16_bits(v);
    if bits & 0x7c00 == 0x7c00 {
        Err(bits)
    } else {
        Ok(bits)
    }
}
