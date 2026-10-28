//! CPU forward kernels, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementations live in
//! `lattice_compute::forward::cpu`.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-compute` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-compute` directly.

#[cfg(target_arch = "aarch64")]
#[doc(inline)]
pub use lattice_compute::forward::cpu::matmul_neon;
#[cfg(feature = "train-backward")]
pub(crate) use lattice_compute::forward::cpu::validate_gemm_bt;
#[doc(inline)]
pub use lattice_compute::forward::cpu::{
    BertCpuKernelPolicy, BertCpuKernelProfile, add_bias, add_bias_gelu, elementwise_mul, gelu,
    layer_norm, matmul, matmul_bt, matmul_into, matmul_scalar, residual_add_layer_norm, rms_norm,
    sgemm_bt_strided, sgemm_nn_ab, sgemm_nn_strided, silu_inplace, softmax_attention,
};
#[cfg(target_arch = "x86_64")]
#[doc(inline)]
pub use lattice_compute::forward::cpu::{matmul_avx2, matmul_avx512};
