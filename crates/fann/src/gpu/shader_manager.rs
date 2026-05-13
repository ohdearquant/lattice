//! Shader compilation and pipeline caching
//!
//! Precompiles and caches compute pipelines to avoid
//! compilation overhead during inference.

use super::error::GpuResult;
use super::shaders;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use wgpu::ComputePipeline;

/// Types of shader operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderType {
    /// Matrix-vector multiplication (y = Wx + b)
    MatrixVectorMultiply,
    /// Fused matrix-vector multiply with ReLU activation
    MatrixVectorMultiplyRelu,

    /// ReLU activation function: max(0, x)
    ReLU,
    /// Leaky ReLU: max(alpha * x, x)
    LeakyReLU,
    /// Sigmoid activation: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh activation
    Tanh,

    /// Softmax pass 1: find maximum value
    SoftmaxMax,
    /// Softmax pass 2: compute exp(x - max) and sum
    SoftmaxExpSum,
    /// Softmax pass 3: normalize by sum
    SoftmaxNorm,

    /// SGD optimizer with momentum
    SgdMomentum,
    /// Adam optimizer
    Adam,
    /// AdamW optimizer with weight decay
    AdamW,
}

impl ShaderType {
    /// Get WGSL source code for this shader type
    pub fn source(&self) -> &'static str {
        match self {
            ShaderType::MatrixVectorMultiply => shaders::MATMUL_SHADER,
            ShaderType::MatrixVectorMultiplyRelu => shaders::MATMUL_RELU_SHADER,
            ShaderType::ReLU => shaders::RELU_SHADER,
            ShaderType::LeakyReLU => shaders::LEAKY_RELU_SHADER,
            ShaderType::Sigmoid => shaders::SIGMOID_SHADER,
            ShaderType::Tanh => shaders::TANH_SHADER,
            ShaderType::SoftmaxMax => shaders::SOFTMAX_MAX_SHADER,
            ShaderType::SoftmaxExpSum => shaders::SOFTMAX_EXP_SUM_SHADER,
            ShaderType::SoftmaxNorm => shaders::SOFTMAX_NORM_SHADER,
            ShaderType::SgdMomentum => shaders::SGD_MOMENTUM_SHADER,
            ShaderType::Adam => shaders::ADAM_SHADER,
            ShaderType::AdamW => shaders::ADAMW_SHADER,
        }
    }

    /// Get workgroup size for this shader type
    pub fn workgroup_size(&self) -> [u32; 3] {
        match self {
            // Matrix operations use smaller workgroups for Apple Silicon
            ShaderType::MatrixVectorMultiply | ShaderType::MatrixVectorMultiplyRelu => [32, 1, 1],
            // Everything else uses 256 for throughput
            _ => [256, 1, 1],
        }
    }

    /// Label for debugging
    pub fn label(&self) -> &'static str {
        match self {
            ShaderType::MatrixVectorMultiply => "matmul",
            ShaderType::MatrixVectorMultiplyRelu => "matmul_relu",
            ShaderType::ReLU => "relu",
            ShaderType::LeakyReLU => "leaky_relu",
            ShaderType::Sigmoid => "sigmoid",
            ShaderType::Tanh => "tanh",
            ShaderType::SoftmaxMax => "softmax_max",
            ShaderType::SoftmaxExpSum => "softmax_exp_sum",
            ShaderType::SoftmaxNorm => "softmax_norm",
            ShaderType::SgdMomentum => "sgd_momentum",
            ShaderType::Adam => "adam",
            ShaderType::AdamW => "adamw",
        }
    }
}

/// Statistics for pipeline cache
#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub compilations: u64,
    pub total_compile_time_ms: f64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        }
    }

    pub fn avg_compile_time_ms(&self) -> f64 {
        if self.compilations > 0 {
            self.total_compile_time_ms / self.compilations as f64
        } else {
            0.0
        }
    }
}

/// Manages shader compilation and pipeline caching
pub struct ShaderManager {
    device: Arc<wgpu::Device>,
    /// Cached compiled pipelines
    pipelines: RwLock<HashMap<ShaderType, Arc<ComputePipeline>>>,
    /// Cache statistics
    stats: RwLock<CacheStats>,
}

impl ShaderManager {
    /// Create a new shader manager
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            device,
            pipelines: RwLock::new(HashMap::new()),
            stats: RwLock::new(CacheStats::default()),
        }
    }

    /// Create with pipeline warmup for common shaders
    pub fn with_warmup(device: Arc<wgpu::Device>) -> GpuResult<Self> {
        let manager = Self::new(device);

        // Precompile common inference shaders
        let warmup_shaders = [
            ShaderType::MatrixVectorMultiply,
            ShaderType::MatrixVectorMultiplyRelu,
            ShaderType::ReLU,
            ShaderType::Sigmoid,
            ShaderType::Tanh,
        ];

        for shader_type in warmup_shaders {
            manager.get_or_compile(shader_type)?;
        }

        Ok(manager)
    }

    /// Get a compiled pipeline, compiling if necessary
    pub fn get_or_compile(&self, shader_type: ShaderType) -> GpuResult<Arc<ComputePipeline>> {
        use super::error::GpuError;

        // Try cache first (read lock)
        {
            let pipelines = self
                .pipelines
                .read()
                .map_err(|e| GpuError::LockPoisoned(format!("shader pipeline cache read: {e}")))?;
            if let Some(pipeline) = pipelines.get(&shader_type) {
                if let Ok(mut stats) = self.stats.write() {
                    stats.hits += 1;
                }
                return Ok(pipeline.clone());
            }
        }

        // Cache miss - compile (write lock)
        if let Ok(mut stats) = self.stats.write() {
            stats.misses += 1;
        }

        let start = std::time::Instant::now();
        let pipeline = self.compile(shader_type)?;
        let compile_time = start.elapsed().as_secs_f64() * 1000.0;

        let pipeline = Arc::new(pipeline);

        // Update cache and stats
        {
            let mut pipelines = self
                .pipelines
                .write()
                .map_err(|e| GpuError::LockPoisoned(format!("shader pipeline cache write: {e}")))?;
            pipelines.insert(shader_type, pipeline.clone());
        }
        {
            if let Ok(mut stats) = self.stats.write() {
                stats.compilations += 1;
                stats.total_compile_time_ms += compile_time;
            }
        }

        Ok(pipeline)
    }

    /// Compile a shader
    fn compile(&self, shader_type: ShaderType) -> GpuResult<ComputePipeline> {
        let source = shader_type.source();
        let label = shader_type.label();

        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        Ok(self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // Auto layout
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }))
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats
            .read()
            .map(|s| CacheStats {
                hits: s.hits,
                misses: s.misses,
                compilations: s.compilations,
                total_compile_time_ms: s.total_compile_time_ms,
            })
            .unwrap_or_default()
    }

    /// Clear the pipeline cache
    pub fn clear_cache(&self) {
        if let Ok(mut pipelines) = self.pipelines.write() {
            pipelines.clear();
        }
    }

    /// Number of cached pipelines
    pub fn cached_count(&self) -> usize {
        self.pipelines.read().map(|p| p.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_type_source() {
        // Just verify sources are non-empty and contain expected patterns
        assert!(
            ShaderType::MatrixVectorMultiply
                .source()
                .contains("@compute")
        );
        assert!(ShaderType::Sigmoid.source().contains("clamp"));
        assert!(ShaderType::Adam.source().contains("beta1"));
    }

    #[test]
    fn test_shader_type_workgroup_size() {
        assert_eq!(
            ShaderType::MatrixVectorMultiply.workgroup_size(),
            [32, 1, 1]
        );
        assert_eq!(ShaderType::ReLU.workgroup_size(), [256, 1, 1]);
    }

    #[test]
    fn test_cache_stats() {
        let stats = CacheStats {
            hits: 90,
            misses: 10,
            compilations: 10,
            total_compile_time_ms: 100.0,
        };

        assert!((stats.hit_rate() - 0.9).abs() < 0.001);
        assert!((stats.avg_compile_time_ms() - 10.0).abs() < 0.001);
    }
}
