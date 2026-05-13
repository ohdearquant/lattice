//! GPU compute backend for lattice-fann
//!
//! Production-grade GPU acceleration with:
//! - 3-tier buffer pooling (Small/Medium/Large)
//! - Circuit breaker for memory pressure
//! - Pipeline caching
//! - Intelligent GPU/CPU switching
//! - Apple Silicon optimizations
//!
//! # Architecture
//!
//! ```text
//! GpuContext (device/queue) ─┬─> BufferPool (3-tier, lifecycle tracking)
//!                            ├─> ShaderManager (compiled pipelines)
//!                            └─> CircuitBreaker (memory pressure)
//!
//! GpuNetwork (inference) ──> GpuContext
//! GpuTrainer (training) ───> GpuContext
//! ```
//!
//! # GPU/CPU Decision Heuristics
//!
//! | Operation | GPU Threshold | Rationale |
//! |-----------|---------------|-----------|
//! | Matrix-vector | >10K elements | GPU launch overhead dominates small ops |
//! | Batch matmul | >100 batch size | Amortize kernel launch |
//! | Activation | >1K elements | Element-wise is memory-bound |
//!
//! # Apple Silicon Specifics
//!
//! - 256-byte buffer alignment required
//! - 128MB max buffer size
//! - 2ms Metal watchdog (tile large dispatches)
//! - 32-lane SIMD workgroups

mod buffer;
mod circuit_breaker;
mod context;
mod error;
mod network;
mod shader_manager;
mod shaders;

pub use buffer::{BufferCategory, BufferPool, GpuBuffer};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerState, MemoryPressure};
pub use context::{GpuContext, MemoryPressureHandler, MemoryPressureLevel};
pub use error::GpuError;
pub use network::GpuNetwork;
pub use shader_manager::{ShaderManager, ShaderType};

/// GPU operation thresholds - below these, CPU is faster
pub mod thresholds {
    /// Minimum elements for GPU matrix-vector multiply benefit
    pub const MATMUL_MIN_ELEMENTS: usize = 10_000;
    /// Minimum batch size for GPU batch operations benefit
    pub const BATCH_MIN_SIZE: usize = 100;
    /// Minimum elements for GPU activation function benefit
    pub const ACTIVATION_MIN_ELEMENTS: usize = 1_000;
    /// Max elements per dispatch to avoid Metal watchdog (2ms limit)
    pub const MAX_ELEMENTS_PER_DISPATCH: usize = 100_000;
    /// Metal dispatch time headroom (stay under 2ms watchdog)
    pub const MAX_DISPATCH_TIME_MS: f32 = 1.5;
}

/// Apple Silicon specific constants
pub mod apple_silicon {
    /// Required buffer alignment on Apple Silicon
    pub const BUFFER_ALIGNMENT: usize = 256;
    /// Maximum buffer size (128MB)
    pub const MAX_BUFFER_SIZE: usize = 128 * 1024 * 1024;
    /// Optimal workgroup size (matches 32-lane SIMD)
    pub const WORKGROUP_SIZE: u32 = 32;
    /// Workgroup size for element-wise ops (maximize throughput)
    pub const ACTIVATION_WORKGROUP_SIZE: u32 = 256;
}

/// Check if GPU acceleration is available
pub fn is_gpu_available() -> bool {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .is_some()
    })
}

/// Get GPU device information
pub fn get_gpu_info() -> Option<GpuInfo> {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let info = adapter.get_info();
        let limits = adapter.limits();

        Some(GpuInfo {
            name: info.name,
            vendor: info.vendor,
            device_type: format!("{:?}", info.device_type),
            backend: format!("{:?}", info.backend),
            max_buffer_size: limits.max_buffer_size,
            max_workgroup_size: limits.max_compute_workgroup_size_x,
        })
    })
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Device name (e.g., "Apple M1 Pro")
    pub name: String,
    /// Vendor ID
    pub vendor: u32,
    /// Device type (e.g., "DiscreteGpu", "IntegratedGpu")
    pub device_type: String,
    /// Backend (e.g., "Metal", "Vulkan", "DX12")
    pub backend: String,
    /// Maximum buffer size in bytes
    pub max_buffer_size: u64,
    /// Maximum workgroup size
    pub max_workgroup_size: u32,
}

/// Determine if operation should use GPU based on problem size
#[inline]
pub fn should_use_gpu(elements: usize, batch_size: usize) -> bool {
    if batch_size > 1 {
        elements * batch_size >= thresholds::BATCH_MIN_SIZE * 100
    } else {
        elements >= thresholds::MATMUL_MIN_ELEMENTS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_availability() {
        // Just check it doesn't panic
        let _available = is_gpu_available();
    }

    #[test]
    fn test_should_use_gpu() {
        // Small operations -> CPU
        assert!(!should_use_gpu(100, 1));
        assert!(!should_use_gpu(1000, 1));

        // Large operations -> GPU
        assert!(should_use_gpu(50_000, 1));

        // Batch operations
        assert!(should_use_gpu(1000, 100));
        assert!(!should_use_gpu(10, 5));
    }
}
