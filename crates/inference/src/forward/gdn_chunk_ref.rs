//! **Compatibility path.** The items here are re-exported from `lattice-model-qwen35` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-qwen35` directly.

#![cfg(feature = "bench-internals")]

//! Qwen3.5 family compatibility exports.

#[doc(inline)]
pub use lattice_model_qwen35::forward::gdn_chunk_ref::l2_normalize_rows;

#[doc(inline)]
pub use lattice_model_qwen35::forward::gdn_chunk_ref::solve_unit_lower;

#[doc(inline)]
pub use lattice_model_qwen35::forward::gdn_chunk_ref::sequential_gdn;

#[doc(inline)]
pub use lattice_model_qwen35::forward::gdn_chunk_ref::chunkwise_gdn;
