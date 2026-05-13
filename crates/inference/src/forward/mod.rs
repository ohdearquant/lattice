pub mod batch_prefill;
pub mod bitnet_kernel;
pub mod cpu;
pub mod cpu_f16;
pub mod cpu_q8;
pub mod gpu;
pub mod gpu_gemm;
pub mod metal;
pub mod metal_gemm;
pub mod metal_qwen35;
pub mod neon;
pub mod neon_forward;

// Re-export from cpu for backward compat (was `layers`)
pub use self::cpu::*;
