//! Tiled causal flash attention, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::flash_causal`.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-compute` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-compute` directly.

#[doc(inline)]
pub use lattice_compute::attention::flash_causal::{
    DEFAULT_BC, DEFAULT_BR, FlashAttentionConfig, flash_attention_causal,
    flash_attention_causal_with_config,
};
