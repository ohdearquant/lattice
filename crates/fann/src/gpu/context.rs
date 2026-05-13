//! GPU compute context - device, queue, and resource management

use super::buffer::BufferPool;
use super::circuit_breaker::MemoryPressure;
use super::error::{GpuError, GpuResult};
use super::shader_manager::ShaderManager;
use std::sync::{Arc, Mutex};

/// Re-export MemoryPressure as MemoryPressureLevel for API clarity
pub use super::circuit_breaker::MemoryPressure as MemoryPressureLevel;

/// Callback type for memory pressure events
///
/// The callback receives the current memory pressure level and the
/// estimated memory usage in bytes.
pub type MemoryPressureHandler = Box<dyn Fn(MemoryPressureLevel, u64) + Send + Sync>;

/// GPU compute context
///
/// Holds the wgpu device, queue, buffer pool, and shader manager.
/// This is the main entry point for GPU operations.
pub struct GpuContext {
    /// GPU device
    pub(crate) device: Arc<wgpu::Device>,
    /// Command queue
    pub(crate) queue: Arc<wgpu::Queue>,
    /// Buffer pool for memory management
    pub(crate) buffer_pool: BufferPool,
    /// Shader manager with pipeline caching
    pub(crate) shader_manager: ShaderManager,
    /// Device info
    info: GpuDeviceInfo,
    /// Memory pressure callback handler
    memory_pressure_handler: Mutex<Option<MemoryPressureHandler>>,
    /// Memory budget in bytes (None = no limit)
    memory_budget: Mutex<Option<u64>>,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub vendor: u32,
    pub device_type: wgpu::DeviceType,
    pub backend: wgpu::Backend,
    pub max_buffer_size: u64,
    pub max_workgroup_size: [u32; 3],
}

impl GpuContext {
    /// Create a new GPU context asynchronously
    pub async fn new() -> GpuResult<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| GpuError::NotAvailable("No GPU adapter found".into()))?;

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("lattice-fann-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let info = GpuDeviceInfo {
            name: adapter_info.name,
            vendor: adapter_info.vendor,
            device_type: adapter_info.device_type,
            backend: adapter_info.backend,
            max_buffer_size: limits.max_buffer_size,
            max_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
        };

        let buffer_pool = BufferPool::new(device.clone());
        let shader_manager = ShaderManager::with_warmup(device.clone())?;

        Ok(Self {
            device,
            queue,
            buffer_pool,
            shader_manager,
            info,
            memory_pressure_handler: Mutex::new(None),
            memory_budget: Mutex::new(None),
        })
    }

    /// Create a new GPU context (blocking)
    pub fn new_blocking() -> GpuResult<Self> {
        pollster::block_on(Self::new())
    }

    /// Get device information
    pub fn info(&self) -> &GpuDeviceInfo {
        &self.info
    }

    /// Get device reference
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get queue reference
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get buffer pool reference
    pub fn buffer_pool(&self) -> &BufferPool {
        &self.buffer_pool
    }

    /// Get shader manager reference
    pub fn shader_manager(&self) -> &ShaderManager {
        &self.shader_manager
    }

    /// Check if this is an Apple Silicon device
    pub fn is_apple_silicon(&self) -> bool {
        self.info.backend == wgpu::Backend::Metal
            && (self.info.name.contains("Apple")
                || self.info.name.contains("M1")
                || self.info.name.contains("M2")
                || self.info.name.contains("M3"))
    }

    /// Get optimal workgroup size for this device
    pub fn optimal_workgroup_size(&self, operation: &str) -> u32 {
        if self.is_apple_silicon() {
            match operation {
                "matmul" => 32, // Match 32-lane SIMD
                _ => 256,       // Maximize throughput for element-wise
            }
        } else {
            // Generic defaults
            match operation {
                "matmul" => 64,
                _ => 256,
            }
        }
    }

    /// Poll device to process pending work
    pub fn poll(&self) {
        self.device.poll(wgpu::Maintain::Poll);
    }

    /// Wait for all GPU work to complete
    pub fn wait(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Flush GPU memory and wait for pending operations
    ///
    /// This method:
    /// 1. Flushes all pooled buffers, releasing cached GPU memory
    /// 2. Waits for all pending GPU operations to complete
    ///
    /// Use this during long training loops to prevent OOM from async deallocation lag.
    /// Returns the number of bytes freed from the buffer pool.
    pub fn flush_memory(&self) -> u64 {
        let freed = self.buffer_pool.flush();
        self.wait();
        freed
    }

    /// Get current GPU memory usage from buffer pool
    pub fn memory_usage(&self) -> u64 {
        self.buffer_pool.memory_usage()
    }

    /// Set memory budget in bytes
    ///
    /// When set, memory pressure is calculated as usage/budget ratio.
    pub fn set_memory_budget(&self, budget_bytes: u64) {
        if let Ok(mut budget) = self.memory_budget.lock() {
            *budget = Some(budget_bytes);
        }
    }

    /// Set memory pressure handler callback
    ///
    /// The callback is invoked when memory pressure is detected or changes.
    /// Use this for graceful degradation under memory pressure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.set_memory_pressure_handler(|level, usage| {
    ///     match level {
    ///         MemoryPressureLevel::High | MemoryPressureLevel::Critical => {
    ///             eprintln!("GPU memory pressure: {:?}, usage: {} MB", level, usage / 1_000_000);
    ///             // Reduce batch size, flush caches, etc.
    ///         }
    ///         _ => {}
    ///     }
    /// });
    /// ```
    pub fn set_memory_pressure_handler<F>(&self, handler: F)
    where
        F: Fn(MemoryPressureLevel, u64) + Send + Sync + 'static,
    {
        if let Ok(mut h) = self.memory_pressure_handler.lock() {
            *h = Some(Box::new(handler));
        }
    }

    /// Clear the memory pressure handler
    pub fn clear_memory_pressure_handler(&self) {
        if let Ok(mut h) = self.memory_pressure_handler.lock() {
            *h = None;
        }
    }

    /// Check current memory pressure level
    ///
    /// Returns the pressure level based on current usage vs budget.
    /// If no budget is set, returns `None` (no pressure tracking).
    pub fn check_memory_pressure(&self) -> Option<MemoryPressureLevel> {
        let budget = self.memory_budget.lock().ok()?.as_ref().copied()?;
        let usage = self.memory_usage();
        let ratio = usage as f32 / budget as f32;
        Some(MemoryPressure::from_ratio(ratio))
    }

    /// Notify memory pressure handler if registered
    ///
    /// Call this after significant allocations to trigger graceful degradation.
    pub fn notify_memory_pressure(&self) {
        if let Some(level) = self.check_memory_pressure() {
            let usage = self.memory_usage();
            if let Ok(handler) = self.memory_pressure_handler.lock() {
                if let Some(ref callback) = *handler {
                    callback(level, usage);
                }
            }
        }
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device", &self.info.name)
            .field("backend", &format!("{:?}", self.info.backend))
            .field("max_buffer_size", &self.info.max_buffer_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires GPU hardware")]
    fn test_context_creation() {
        // Skip if no GPU
        if !super::super::is_gpu_available() {
            println!("Skipping GPU test - no GPU available");
            return;
        }

        let ctx = GpuContext::new_blocking();
        assert!(ctx.is_ok());

        let ctx = ctx.unwrap();
        println!("GPU: {} ({:?})", ctx.info.name, ctx.info.backend);
        assert!(!ctx.info.name.is_empty());
    }
}
