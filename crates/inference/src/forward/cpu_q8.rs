//! Qwen3.5 family compatibility exports.

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_q8::quantize_from_model;

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_q8::gated_delta_net_step_fused_q8;

#[doc(inline)]
pub use lattice_model_qwen35::forward::cpu_q8::generate_q8;
