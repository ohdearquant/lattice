//! Standard multi-head attention, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::standard`.

#[doc(inline)]
pub use lattice_compute::attention::standard::{AttentionBuffers, multi_head_attention};

#[cfg(test)]
mod padded_reference;
#[cfg(test)]
pub(crate) use self::padded_reference::multi_head_attention_batched_padded_reference;

#[cfg(feature = "bench-internals")]
#[doc(inline)]
pub use lattice_compute::attention::standard::bench_support;
