//! Qwen3.5 family compatibility exports.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-qwen35` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-qwen35` directly.

#[doc(inline)]
pub use lattice_model_qwen35::vision::pooled_embed::embed_image_from_bytes_f16;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(inline)]
pub use lattice_model_qwen35::vision::pooled_embed::embed_image_from_bytes_f16_metal;

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
#[doc(inline)]
pub use lattice_model_qwen35::vision::pooled_embed::embed_image_from_bytes_f16_metal;
