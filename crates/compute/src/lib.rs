//! Host-side compute for Lattice inference.
//!
//! This crate holds the computation that runs on the host for a decoder forward pass. It currently
//! contains rotary position embedding tables ([`rope::RopeTable`] and the Gemma 4 tables in
//! [`gemma_rope`]), flat/paged KV storage with prefix planning, LoRA adapter hooks
//! ([`lora_hook::LoraHook`]), the numerical diagnostics
//! ([`metrics::OnlineSoftmaxEntropy`], [`pruning::BlockInfluence`]), option-letter scoring
//! ([`option_scoring::OptionScores`]), shared attention row semantics
//! ([`attention::softmax_row`]), and CPU forward kernels ([`forward::cpu`]); pooling and sampling
//! move here in later releases.
//!
//! `lattice-inference` re-exports the supported items at their original paths, so
//! `lattice_inference::rope::RopeTable` and `lattice_compute::rope::RopeTable` name the same type.

// ML inference kernels: many functions have >7 args by necessity (BLAS-style APIs
// where grouping into structs would require heap allocation in hot paths), and many
// loops use the index to access multiple arrays simultaneously so the
// needless_range_loop suggestion does not apply.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod attention;
pub mod forward;
pub mod gemma_rope;
pub mod kv_cache;
pub mod lora_hook;
pub mod metrics;
pub mod option_scoring;
pub mod pruning;
pub mod rope;

/// Cache glue used by the `lattice-inference` facade.
///
/// Not part of the supported API. The facade calls these free functions by their full path;
/// they are not re-exported by that facade.
#[doc(hidden)]
pub mod __internal {
    use lattice_inference_core::InferenceError;

    /// Checked count of floats in a paged cache allocation.
    pub fn paged_cache_try_floats_per_page(
        config: &crate::kv_cache::PagedKVCacheConfig,
    ) -> Result<usize, InferenceError> {
        config.try_floats_per_page()
    }
}
