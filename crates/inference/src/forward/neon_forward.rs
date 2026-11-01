//! Qwen3.5 family compatibility exports.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-qwen35` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-qwen35` directly.

#[doc(inline)]
pub use lattice_model_qwen35::forward::neon_forward::Q8NeonGdnWeights;

#[doc(inline)]
pub use lattice_model_qwen35::forward::neon_forward::Q8NeonFullAttnWeights;

#[doc(inline)]
pub use lattice_model_qwen35::forward::neon_forward::Q8NeonCommonWeights;

#[doc(inline)]
pub use lattice_model_qwen35::forward::neon_forward::Q8NeonAttentionWeights;

#[doc(inline)]
pub use lattice_model_qwen35::forward::neon_forward::Q8NeonModel;

#[doc(inline)]
pub use lattice_model_qwen35::forward::neon_forward::quantize_model;

#[doc(inline)]
pub use lattice_model_qwen35::forward::neon_forward::generate_q8_neon;

#[cfg(feature = "bench-internals")]
#[doc(inline)]
pub use lattice_model_qwen35::forward::neon_forward::bench_support;
