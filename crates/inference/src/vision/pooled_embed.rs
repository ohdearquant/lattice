//! Qwen3.5 family compatibility exports.

#[doc(inline)]
pub use lattice_model_qwen35::vision::pooled_embed::embed_image_from_bytes_f16;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(inline)]
pub use lattice_model_qwen35::vision::pooled_embed::embed_image_from_bytes_f16_metal;

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
#[doc(inline)]
pub use lattice_model_qwen35::vision::pooled_embed::embed_image_from_bytes_f16_metal;
