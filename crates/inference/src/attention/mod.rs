//! Attention module index for `decode`, `differential`, `flash`, `flash_causal`, `gated`, `gdn`, `gdn_backward`, `gdn_fused`, `gqa`, `native_sparse`, and `standard`; re-exports the standard attention helpers.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-compute` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-compute` directly.
pub mod decode;
pub mod differential;
pub mod flash;
pub mod flash_causal;
pub mod gated;
pub mod gdn;
/// GatedDeltaNet backward. **Compatibility path.** Re-exported from `lattice-backward`
/// (`lattice_backward::gdn`) so that code written against the pre-split `lattice-inference` keeps
/// compiling. Not a stable API; removed at 1.0. Import from `lattice-backward` directly.
#[cfg(feature = "train-backward")]
pub use lattice_backward::gdn as gdn_backward;
pub mod gdn_fused;
pub mod gqa;
pub mod native_sparse;
pub mod softmax_row;
pub mod standard;

// Re-export from standard for backward compat
#[cfg(feature = "bench-internals")]
#[doc(no_inline)]
pub use self::standard::bench_support;
#[doc(no_inline)]
pub use self::standard::{AttentionBuffers, multi_head_attention};

#[doc(inline)]
pub use lattice_compute::attention::{AttentionKind, AttentionTag};
