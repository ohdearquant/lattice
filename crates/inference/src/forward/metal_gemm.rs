//! Qwen3.5 family compatibility exports.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_gemm::{is_available, metal_matmul, metal_matmul_bt};

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_gemm::metal_matmul_bt;

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_gemm::metal_matmul;
