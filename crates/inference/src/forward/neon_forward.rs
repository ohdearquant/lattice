//! Qwen3.5 family compatibility exports.

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
