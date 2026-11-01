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
    // Every item is defined here, not re-exported: rustc applies the module's
    // deprecation to the items defined in it, but not to a `pub use` passing through.
    /// Renamed to [`dispatch::Backend`](lattice_backward::dispatch::Backend).
    pub type Backend = lattice_backward::dispatch::Backend;
    /// Renamed to [`dispatch::ForcedDispatchError`](lattice_backward::dispatch::ForcedDispatchError).
    pub type ForcedDispatchError = lattice_backward::dispatch::ForcedDispatchError;
    /// Renamed to [`dispatch::LoraGradients`](lattice_backward::dispatch::LoraGradients).
    pub type LoraGradients = lattice_backward::dispatch::LoraGradients;
    /// Renamed to [`dispatch::Operation`](lattice_backward::dispatch::Operation).
    pub type Operation = lattice_backward::dispatch::Operation;
    /// Renamed to [`dispatch::ForcedExecutionError`](lattice_backward::dispatch::ForcedExecutionError).
    pub type ForcedExecutionError = lattice_backward::dispatch::ForcedExecutionError;
    /// Renamed to [`dispatch::ForcedDispatch`](lattice_backward::dispatch::ForcedDispatch).
    pub type ForcedDispatch = lattice_backward::dispatch::ForcedDispatch;
    /// Renamed to [`dispatch::force_scalar`](lattice_backward::dispatch::force_scalar).
    #[inline]
    pub fn force_scalar() -> lattice_backward::dispatch::ForcedDispatch {
        lattice_backward::dispatch::force_scalar()
    }
    /// Renamed to [`dispatch::force_backend`](lattice_backward::dispatch::force_backend).
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
