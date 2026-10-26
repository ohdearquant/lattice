//! Host-side compute for Lattice inference.
//!
//! This crate holds the computation that runs on the host for a decoder forward pass. It currently
//! contains rotary position embedding tables ([`rope::RopeTable`]) and flat/paged KV storage with prefix planning; the
//! CPU kernels, attention, pooling and sampling move here in later releases.
//!
//! `lattice-inference` re-exports the supported items at their original paths, so
//! `lattice_inference::rope::RopeTable` and `lattice_compute::rope::RopeTable` name the same type.

#![allow(clippy::needless_range_loop)]

pub mod kv_cache;
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
