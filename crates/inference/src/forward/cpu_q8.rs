//! Qwen3.5 family compatibility exports.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-qwen35` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-qwen35` directly.

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_q8::quantize_from_model;

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_q8::gated_delta_net_step_fused_q8;

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_q8::generate_q8;
