//! Qwen3.5 family compatibility exports.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-qwen35` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-qwen35` directly.

#[cfg(all(target_os = "macos", feature = "metal-gpu", feature = "test-utils"))]
#[doc(inline)]
pub use lattice_model_qwen35::vision::qwen35_vit_metal::{
    metal_dispatch_count, reset_metal_dispatch_count,
};

#[doc(inline)]
pub use lattice_model_qwen35::vision::qwen35_vit_metal::qwen35_vit_forward_metal;
