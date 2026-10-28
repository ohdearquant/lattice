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
/// #![allow(deprecated)]
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
    // Deprecation is carried by each item: rustc does not report a deprecated
    // module or re-export when a path merely passes through it.
    /// Renamed to [`dispatch::Backend`](lattice_backward::dispatch::Backend).
    #[deprecated(
        since = "0.15.0",
        note = "renamed to lattice_inference::backward::dispatch"
    )]
    pub type Backend = lattice_backward::dispatch::Backend;
    /// Renamed to [`dispatch::ForcedDispatchError`](lattice_backward::dispatch::ForcedDispatchError).
    #[deprecated(
        since = "0.15.0",
        note = "renamed to lattice_inference::backward::dispatch"
    )]
    pub type ForcedDispatchError = lattice_backward::dispatch::ForcedDispatchError;
    /// Renamed to [`dispatch::LoraGradients`](lattice_backward::dispatch::LoraGradients).
    #[deprecated(
        since = "0.15.0",
        note = "renamed to lattice_inference::backward::dispatch"
    )]
    pub type LoraGradients = lattice_backward::dispatch::LoraGradients;
    /// Renamed to [`dispatch::Operation`](lattice_backward::dispatch::Operation).
    #[deprecated(
        since = "0.15.0",
        note = "renamed to lattice_inference::backward::dispatch"
    )]
    pub type Operation = lattice_backward::dispatch::Operation;
    /// Renamed to [`dispatch::ForcedExecutionError`](lattice_backward::dispatch::ForcedExecutionError).
    #[deprecated(
        since = "0.15.0",
        note = "renamed to lattice_inference::backward::dispatch"
    )]
    pub type ForcedExecutionError = lattice_backward::dispatch::ForcedExecutionError;
    /// Renamed to [`dispatch::ForcedDispatch`](lattice_backward::dispatch::ForcedDispatch).
    #[deprecated(
        since = "0.15.0",
        note = "renamed to lattice_inference::backward::dispatch"
    )]
    pub type ForcedDispatch = lattice_backward::dispatch::ForcedDispatch;
    /// Renamed to [`dispatch::force_scalar`](lattice_backward::dispatch::force_scalar).
    #[deprecated(
        since = "0.15.0",
        note = "renamed to lattice_inference::backward::dispatch"
    )]
    #[inline]
    pub fn force_scalar() -> lattice_backward::dispatch::ForcedDispatch {
        lattice_backward::dispatch::force_scalar()
    }
    /// Renamed to [`dispatch::force_backend`](lattice_backward::dispatch::force_backend).
    #[deprecated(
        since = "0.15.0",
        note = "renamed to lattice_inference::backward::dispatch"
    )]
    #[inline]
    pub fn force_backend(
        backend: lattice_backward::dispatch::Backend,
    ) -> Result<
        lattice_backward::dispatch::ForcedDispatch,
        lattice_backward::dispatch::ForcedDispatchError,
    > {
        lattice_backward::dispatch::force_backend(backend)
    }
}
pub use lattice_backward::{attention_gqa, gradcheck, ops, tape};
