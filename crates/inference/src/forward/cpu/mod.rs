mod activation;
mod arch_kernels;
mod blas;
mod elementwise;
mod matmul;
mod norm;
mod simd;
mod softmax;
mod tiled;
mod tiled_avx2;
mod tiled_neon;

#[cfg(test)]
mod tests;

pub use activation::{add_bias, add_bias_gelu, gelu};
#[cfg(target_arch = "aarch64")]
pub use arch_kernels::matmul_neon;
#[cfg(target_arch = "x86_64")]
pub use arch_kernels::{matmul_avx2, matmul_avx512};
pub use blas::{sgemm_bt_strided, sgemm_nn_ab, sgemm_nn_strided};
pub use elementwise::{elementwise_mul, rms_norm, silu_inplace};
pub use matmul::{matmul, matmul_bt, matmul_into, matmul_scalar};
pub use norm::layer_norm;
pub(crate) use simd::simd_config;
pub use softmax::softmax_attention;

#[cfg(test)]
pub use activation::{add_bias_gelu_scalar, fast_tanh, gelu_scalar};
#[cfg(test)]
pub use matmul::matmul_bt_scalar;
#[cfg(test)]
pub use norm::layer_norm_scalar;
#[cfg(test)]
pub use softmax::{fast_exp, softmax_attention_scalar};
