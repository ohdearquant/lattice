//! Metal Shading Language sources that several models compile into their own libraries.
//!
//! A model assembles one translation unit from these fragments and its own sources with
//! [`concat_msl!`](crate::concat_msl). Fragment order is part of each model's contract: Metal
//! requires a function to appear before its first call in the same unit.

/// The shared RMS-norm threadgroup reduction helper.
pub const RMS_REDUCE: &str = include_str!("../shaders/rms_reduce.metal");

/// Tiled GEMM, elementwise and flash-attention kernels, with `__FA_*__` and
/// `__FUSED_C_HEAD_DIM__` placeholders that each caller substitutes.
pub const FLASH_ATTENTION: &str = include_str!("../shaders/flash_attention.metal");

/// The rotary-embedding kernel over compact cosine and sine tables.
pub const ERNIE45_ROPE: &str = include_str!("../shaders/ernie45_rope.metal");

/// Total byte length of `parts`, for [`concat_msl!`](crate::concat_msl).
#[doc(hidden)]
pub const fn concat_len(parts: &[&str]) -> usize {
    let mut len = 0;
    let mut i = 0;
    while i < parts.len() {
        len += parts[i].len();
        i += 1;
    }
    len
}

/// The bytes of `parts` in order, for [`concat_msl!`](crate::concat_msl). `N` must equal
/// [`concat_len`] of the same parts; a mismatch fails constant evaluation.
#[doc(hidden)]
pub const fn concat_bytes<const N: usize>(parts: &[&str]) -> [u8; N] {
    let mut out = [0u8; N];
    let mut at = 0;
    let mut i = 0;
    while i < parts.len() {
        let bytes = parts[i].as_bytes();
        let mut j = 0;
        while j < bytes.len() {
            out[at] = bytes[j];
            at += 1;
            j += 1;
        }
        i += 1;
    }
    assert!(
        at == N,
        "concatenated length differs from the declared length"
    );
    out
}

/// Concatenate string constants into one `&'static str` at compile time.
///
/// Each argument is a constant `&str` expression: an `include_str!` of the caller's own file or a
/// source from [`msl`](crate::msl). The result is the same bytes `concat!` would produce from
/// literal fragments, so a model can mix shared and private sources in one constant.
#[macro_export]
macro_rules! concat_msl {
    ($($part:expr),+ $(,)?) => {{
        const PARTS: &[&str] = &[$($part),+];
        const LEN: usize = $crate::msl::concat_len(PARTS);
        const BYTES: [u8; LEN] = $crate::msl::concat_bytes::<LEN>(PARTS);
        match ::core::str::from_utf8(&BYTES) {
            Ok(source) => source,
            Err(_) => panic!("shader source is not UTF-8"),
        }
    }};
}

#[cfg(test)]
mod tests {
    const JOINED: &str = crate::concat_msl!("ab", "", "c\u{e9}", "d");
    const SHARED: &str = crate::concat_msl!(super::RMS_REDUCE, super::ERNIE45_ROPE);

    #[test]
    fn concatenates_in_order_with_empty_and_multibyte_parts() {
        assert_eq!(JOINED, "abc\u{e9}d");
        assert_eq!(JOINED.len(), 6);
    }

    #[test]
    fn shared_sources_concatenate_to_their_bytes() {
        assert_eq!(
            SHARED,
            [super::RMS_REDUCE, super::ERNIE45_ROPE].concat().as_str()
        );
        assert!(super::RMS_REDUCE.contains("rms_inv_from_local_sum("));
        assert!(super::ERNIE45_ROPE.contains("kernel void ernie45_rope("));
        assert!(super::FLASH_ATTENTION.contains("kernel void matmul_bt("));
    }
}
