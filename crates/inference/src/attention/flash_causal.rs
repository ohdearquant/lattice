//! Tiled causal flash attention, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::flash_causal`.

#[doc(inline)]
pub use lattice_compute::attention::flash_causal::{
    DEFAULT_BC, DEFAULT_BR, FlashAttentionConfig, flash_attention_causal,
    flash_attention_causal_with_config,
};
