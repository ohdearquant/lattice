//! Attention module index for `decode`, `differential`, `flash`, `flash_causal`, `gated`, `gdn`, `gdn_backward`, `gdn_fused`, `gqa`, `native_sparse`, and `standard`; re-exports the standard attention helpers.
pub mod decode;
pub mod differential;
pub mod flash;
pub mod flash_causal;
pub mod gated;
pub mod gdn;
#[cfg(feature = "train-backward")]
pub mod gdn_backward;
pub mod gdn_fused;
pub mod gqa;
pub mod native_sparse;
pub mod softmax_row;
pub mod standard;

// Re-export from standard for backward compat
#[cfg(feature = "bench-internals")]
#[doc(no_inline)]
pub use self::standard::bench_support;
#[cfg(test)]
pub(crate) use self::standard::multi_head_attention_batched_padded_reference;
#[doc(no_inline)]
pub use self::standard::{AttentionBuffers, multi_head_attention};

#[doc(inline)]
pub use lattice_compute::attention::{AttentionKind, AttentionTag};
