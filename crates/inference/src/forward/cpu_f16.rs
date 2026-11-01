//! Qwen3.5 family compatibility exports.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-qwen35` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-qwen35` directly.

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_f16::gated_delta_net_step_fused_f16;

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_f16::generate_f16;

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_f16::generate_multimodal_f16;

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_f16::PoolingStrategy;

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_f16::prefill_hidden_states_f16;

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_f16::embed_image_f16;

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_f16::embed_text_vlm_f16;
