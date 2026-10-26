//! Tiled CPU attention over `lattice-compute`'s projection-view kernel.
//!
//! Named exports keep every existing path; `tiled_multi_head_attention` keeps its BERT-layer
//! signature here, and the kernel in `lattice_compute::attention::flash` takes the layer's
//! [`AttentionProjections`](crate::attention::standard::AttentionProjections).

#[doc(inline)]
pub use lattice_compute::attention::flash::{
    TiledAttentionBuffers, TiledAttentionConfig, estimate_materialized_attention_buffer_bytes,
    tiled_attention_head,
};

use crate::attention::standard::{AttentionBuffers, bert_attention_projections};
use crate::weights::TransformerLayerWeights;

/// **Unstable**: tiled multi-head attention using online-softmax; GQA support in progress.
///
/// Tiled multi-head attention. This mirrors the existing materialized path but
/// uses the online-softmax tiled kernel instead of materializing `scores`.
#[inline]
pub fn tiled_multi_head_attention(
    hidden_states: &[f32],
    layer_weights: &TransformerLayerWeights<'_>,
    attention_mask: &[u32],
    seq_len: usize,
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    buffers: &mut AttentionBuffers,
    tiled_buffers: &mut TiledAttentionBuffers,
    config: &TiledAttentionConfig,
) -> Vec<f32> {
    lattice_compute::attention::flash::tiled_multi_head_attention(
        hidden_states,
        &bert_attention_projections(layer_weights),
        attention_mask,
        seq_len,
        hidden_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        buffers,
        tiled_buffers,
        config,
    )
}
