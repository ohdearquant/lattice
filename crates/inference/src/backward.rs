//! Backward-pass module index for `attention_gqa`, `gradcheck`, `ops`, `dispatch`
//! (`test-utils` builds only), and `tape`.
//!
//! **Compatibility path.** Re-exported from `lattice-backward` so that code written against the
//! pre-split `lattice-inference` keeps compiling. Not a stable API; removed at 1.0. Import from
//! `lattice-backward` directly.
#[cfg(feature = "test-utils")]
pub use lattice_backward::dispatch;
pub use lattice_backward::{attention_gqa, gradcheck, ops, tape};
