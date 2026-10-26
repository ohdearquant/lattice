//! CPU forward kernels, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementations live in
//! `lattice_compute::forward::cpu`. The Q8 matrix kernel stays here.
pub(crate) mod q8_matmul;

#[cfg(target_arch = "aarch64")]
#[doc(inline)]
pub use lattice_compute::forward::cpu::matmul_neon;
#[doc(inline)]
pub use lattice_compute::forward::cpu::{
    BertCpuKernelPolicy, BertCpuKernelProfile, add_bias, add_bias_gelu, elementwise_mul, gelu,
    layer_norm, matmul, matmul_bt, matmul_into, matmul_scalar, residual_add_layer_norm, rms_norm,
    sgemm_bt_strided, sgemm_nn_ab, sgemm_nn_strided, silu_inplace, softmax_attention,
};
#[cfg(target_arch = "x86_64")]
#[doc(inline)]
pub use lattice_compute::forward::cpu::{matmul_avx2, matmul_avx512};
pub(crate) use lattice_compute::forward::cpu::{
    validate_gemm_bt, validate_gemm_nn, validate_ternary_matvec_args,
};
