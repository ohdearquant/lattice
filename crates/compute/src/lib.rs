//! Host-side compute for Lattice inference.
//!
//! This crate holds the computation that runs on the host for a decoder forward pass. It currently
//! contains rotary position embedding tables ([`rope::RopeTable`] and the Gemma 4 tables in
//! [`gemma_rope`]), flat/paged KV storage with prefix planning, LoRA adapter hooks
//! ([`lora_hook::LoraHook`]), the numerical diagnostics
//! ([`metrics::OnlineSoftmaxEntropy`], [`pruning::BlockInfluence`]), option-letter scoring
//! ([`option_scoring::OptionScores`]), shared attention row semantics
//! ([`attention::softmax_row`]), the CPU attention kernels and their variant taxonomy
//! ([`attention::AttentionKind`]), CPU forward kernels ([`forward::cpu`]), and BERT-family pooling
//! ([`pool`]); sampling moves here in a later release.
//!
//! `lattice-inference` re-exports the supported items at their original paths, so
//! `lattice_inference::rope::RopeTable` and `lattice_compute::rope::RopeTable` name the same type.

// ML inference kernels: many functions have >7 args by necessity (BLAS-style APIs
// where grouping into structs would require heap allocation in hot paths), and many
// loops use the index to access multiple arrays simultaneously so the
// needless_range_loop suggestion does not apply.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod attention;
/// Experimental CPU audio feature extraction profiles.
#[doc(hidden)]
pub mod audio;
pub mod batch;
pub mod forward;
pub mod gemma_ops;
pub mod gemma_rope;
pub mod kv_cache;
pub mod lora_hook;
pub mod metrics;
pub mod option_scoring;
pub mod pool;
pub mod pruning;
pub mod rope;
#[doc(hidden)]
pub mod vision;

/// Cache and attention glue used by the `lattice-inference` facade.
///
/// Not part of the supported API. The facade reaches these free functions under this module's
/// path and never re-exports them publicly.
#[doc(hidden)]
pub mod __internal {
    use crate::attention::standard::{AttentionBuffers, AttentionProjections};
    use crate::lora_hook::LoraHook;
    use lattice_inference_core::InferenceError;

    /// Checked count of floats in a paged cache allocation.
    pub fn paged_cache_try_floats_per_page(
        config: &crate::kv_cache::PagedKVCacheConfig,
    ) -> Result<usize, InferenceError> {
        config.try_floats_per_page()
    }

    /// Shape checks shared by the standard attention entry points.
    #[inline]
    pub fn assert_standard_no_overflow(
        seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        crate::attention::standard::assert_standard_no_overflow(
            seq_len,
            hidden_size,
            num_heads,
            head_dim,
        )
    }

    /// Single-sequence standard attention over a fused Q/K/V projection, writing into `buffers`.
    #[inline]
    pub fn multi_head_attention_in_place(
        hidden_states: &[f32],
        projections: &AttentionProjections<'_>,
        fused_qkv_weight: &[f32],
        fused_qkv_bias: &[f32],
        attention_mask: &[u32],
        seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        buffers: &mut AttentionBuffers,
        lora: &dyn LoraHook,
        layer_idx: usize,
    ) {
        crate::attention::standard::multi_head_attention_in_place(
            hidden_states,
            projections,
            fused_qkv_weight,
            fused_qkv_bias,
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

    /// Packed-batch standard attention over `cu_seqlens`-indexed sequences.
    #[inline]
    pub fn multi_head_attention_batched(
        hidden_states: &[f32],
        projections: &AttentionProjections<'_>,
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
        crate::attention::standard::multi_head_attention_batched(
            hidden_states,
            projections,
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
