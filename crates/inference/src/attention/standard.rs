//! Standard multi-head attention over `lattice-compute`'s projection-view kernels.
//!
//! `multi_head_attention` keeps its BERT-layer signature at this path; the kernel in
//! `lattice_compute::attention::standard` takes the layer's [`AttentionProjections`].

#[doc(inline)]
pub use lattice_compute::attention::standard::{AttentionBuffers, AttentionProjections};

use crate::lora_hook::LoraHook;
use crate::weights::TransformerLayerWeights;

#[cfg(test)]
mod padded_reference;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use self::padded_reference::multi_head_attention_batched_padded_reference;

/// The four attention projections of one BERT layer, as the view `lattice-compute`'s kernels take.
#[inline]
pub(crate) fn bert_attention_projections<'a>(
    layer: &TransformerLayerWeights<'a>,
) -> AttentionProjections<'a> {
    AttentionProjections {
        query_weight: layer.query_weight,
        query_bias: layer.query_bias,
        key_weight: layer.key_weight,
        key_bias: layer.key_bias,
        value_weight: layer.value_weight,
        value_bias: layer.value_bias,
        output_weight: layer.attn_output_weight,
        output_bias: layer.attn_output_bias,
    }
}

/// **Unstable**: compute multi-head self-attention and return the output projection.
#[inline]
pub fn multi_head_attention(
    hidden_states: &[f32],
    layer_weights: &TransformerLayerWeights<'_>,
    attention_mask: &[u32],
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    buffers: &mut AttentionBuffers,
    lora: &dyn LoraHook,
    layer_idx: usize,
) -> Vec<f32> {
    lattice_compute::attention::standard::multi_head_attention(
        hidden_states,
        &bert_attention_projections(layer_weights),
        attention_mask,
        seq_len,
        hidden_size,
        num_heads,
        head_dim,
        buffers,
        lora,
        layer_idx,
    )
}

/// Bench-only entry to the packed-batch attention kernel, taking a BERT layer. Only a
/// `--features bench-internals` build has this module, as with `forward::neon_forward::bench_support`.
#[cfg(feature = "bench-internals")]
pub mod bench_support {
    use super::{LoraHook, TransformerLayerWeights, bert_attention_projections};

    /// Packed-batch multi-head attention over `cu_seqlens`-indexed sequences.
    #[inline]
    pub fn multi_head_attention_batched(
        hidden_states: &[f32],
        layer_weights: &TransformerLayerWeights<'_>,
        fused_qkv_weight: &[f32],
        fused_qkv_bias: &[f32],
        cu_seqlens: &[usize],
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        qkv: &mut [f32],
        concat: &mut [f32],
        output: &mut [f32],
        lora: &dyn LoraHook,
        layer_idx: usize,
    ) {
        lattice_compute::attention::standard::bench_support::multi_head_attention_batched(
            hidden_states,
            &bert_attention_projections(layer_weights),
            fused_qkv_weight,
            fused_qkv_bias,
            cu_seqlens,
            hidden_size,
            num_heads,
            head_dim,
            q,
            k,
            v,
            qkv,
            concat,
            output,
            lora,
            layer_idx,
        )
    }
}
