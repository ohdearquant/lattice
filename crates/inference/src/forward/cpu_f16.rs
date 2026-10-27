//! Qwen3.5 family compatibility exports.

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
