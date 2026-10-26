//! Tiled CPU attention, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::flash`.

#[doc(inline)]
pub use lattice_compute::attention::flash::{
    TiledAttentionBuffers, TiledAttentionConfig, estimate_materialized_attention_buffer_bytes,
    tiled_attention_head, tiled_multi_head_attention,
};
