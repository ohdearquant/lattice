//! Forward-kernel module index for batch prefill, BitNet, CPU, f16 CPU, q8 CPU, GPU, Metal, and NEON paths.
pub mod batch_prefill;
pub mod bitnet_kernel;
pub mod cpu;
pub mod cpu_f16;
pub mod cpu_q8;
// Host-side f32 chunkwise GDN parity oracle (issue #175). Test/bench-only —
// adds zero production surface to the default build.
#[cfg(any(test, feature = "bench-internals"))]
pub mod gdn_chunk_ref;
pub mod gpu;
pub mod gpu_gemm;
pub mod metal;
pub mod metal_gemm;
pub mod metal_qwen35;
pub mod moe_expert_cache;
pub mod neon;
pub mod neon_forward;
pub mod signpost;

// Re-export from cpu for backward compat (was `layers`)
pub use self::cpu::*;
