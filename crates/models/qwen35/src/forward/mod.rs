//! Qwen3.5 forward implementations.

pub mod batch_prefill;
pub mod cpu_f16;
pub mod cpu_q8;
#[cfg(any(test, feature = "bench-internals"))]
pub mod gdn_chunk_ref;
pub mod metal_gemm;
pub mod metal_qwen35;
pub mod moe_expert_cache;
pub mod neon_forward;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) mod signpost;
