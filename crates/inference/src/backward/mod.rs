//! Backward-pass module index for `attention_gqa`, `gradcheck`, `ops`, and `tape`.
pub mod attention_gqa;
pub mod gradcheck;
pub mod ops;
#[cfg(any(test, feature = "test-utils"))]
pub mod simd;
pub mod tape;
