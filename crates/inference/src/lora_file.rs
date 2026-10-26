//! Qwen3.5 family compatibility exports.

#[doc(inline)]
pub use lattice_model_qwen35::lora_file::resolve_lora_rank_alpha_scale;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(inline)]
pub use lattice_model_qwen35::lora_file::load_lora_safetensors;
