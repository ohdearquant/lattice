//! Forward-kernel module index for batch prefill, BitNet, CPU, f16 CPU, q8 CPU, GPU, Metal, and NEON paths.
#[cfg(feature = "qwen35")]
pub mod batch_prefill;
pub mod bitnet_kernel;
pub mod cpu;
#[cfg(feature = "qwen35")]
pub mod cpu_f16;
#[cfg(feature = "qwen35")]
pub mod cpu_q8;
// Host-side f32 chunkwise GDN parity oracle (issue #175). Test/bench-only —
// adds zero production surface to the default build.
#[cfg(any(test, feature = "bench-internals"))]
#[cfg(feature = "qwen35")]
pub mod gdn_chunk_ref;
pub mod gpu;
pub mod gpu_gemm;
pub mod metal;
#[cfg(feature = "embeddinggemma2")]
pub mod metal_embeddinggemma2;
#[cfg(feature = "ernie45")]
pub mod metal_ernie45;
#[cfg(feature = "qwen35")]
pub mod metal_gemm;
#[cfg(feature = "qwen35")]
pub mod metal_qwen35;
#[cfg(feature = "qwen35")]
pub mod moe_expert_cache;
#[cfg(feature = "qwen35")]
pub mod neon;
#[cfg(feature = "qwen35")]
pub mod neon_forward;
// Re-export the supported CPU names for backward compatibility (was `layers`).
#[cfg(target_arch = "aarch64")]
#[doc(no_inline)]
pub use self::cpu::matmul_neon;
#[doc(no_inline)]
pub use self::cpu::{
    BertCpuKernelPolicy, BertCpuKernelProfile, add_bias, add_bias_gelu, elementwise_mul, gelu,
    layer_norm, matmul, matmul_bt, matmul_into, matmul_scalar, residual_add_layer_norm, rms_norm,
    sgemm_bt_strided, sgemm_nn_ab, sgemm_nn_strided, silu_inplace, softmax_attention,
};
#[cfg(target_arch = "x86_64")]
#[doc(no_inline)]
pub use self::cpu::{matmul_avx2, matmul_avx512};
