//! Backward-pass module index for `attention_gqa`, `gradcheck`, `ops`, `dispatch`
//! (`test-utils` builds only), and `tape`.
//!
//! **Compatibility path.** Re-exported from `lattice-backward` so that code written against the
//! pre-split `lattice-inference` keeps compiling. Not a stable API; removed at 1.0. Import from
//! `lattice-backward` directly.
#[cfg(feature = "test-utils")]
pub use lattice_backward::dispatch;
/// Compatibility path for the renamed backward dispatch helpers.
///
/// ```
/// #[allow(deprecated)]
/// use lattice_inference::backward::simd::force_scalar;
/// let _ = force_scalar();
/// ```
///
/// ```compile_fail
/// #![deny(deprecated)]
/// use lattice_inference::backward::simd::force_scalar;
/// let _ = force_scalar();
/// ```
#[cfg(feature = "test-utils")]
#[deprecated(
    since = "0.15.0",
    note = "renamed to lattice_inference::backward::dispatch"
)]
pub mod simd {
    pub use lattice_backward::dispatch::*;
}
pub use lattice_backward::{attention_gqa, gradcheck, ops, tape};
