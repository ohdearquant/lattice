//! Qwen3.5 family compatibility exports.

#[cfg(all(target_os = "macos", feature = "metal-gpu", feature = "test-utils"))]
#[doc(inline)]
pub use lattice_model_qwen35::vision::qwen35_vit_metal::{
    metal_dispatch_count, reset_metal_dispatch_count,
};

#[doc(inline)]
pub use lattice_model_qwen35::vision::qwen35_vit_metal::qwen35_vit_forward_metal;
