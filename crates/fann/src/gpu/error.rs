//! GPU error types with proper categorization

use thiserror::Error;

/// GPU operation errors
#[derive(Debug, Error)]
pub enum GpuError {
    /// GPU device not available
    #[error("GPU not available: {0}")]
    NotAvailable(String),

    /// Device creation failed
    #[error("Failed to create GPU device: {0}")]
    DeviceCreation(String),

    /// Buffer allocation failed
    #[error("Buffer allocation failed: requested {requested} bytes, available {available}")]
    Allocation {
        /// Requested allocation size in bytes
        requested: usize,
        /// Available memory in bytes
        available: usize,
    },

    /// Buffer pool exhausted
    #[error("Buffer pool exhausted for category {category:?}")]
    PoolExhausted {
        /// Buffer category that was exhausted
        category: super::BufferCategory,
    },

    /// Invalid dimensions for operation
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Shader compilation error
    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// Pipeline creation error
    #[error("Pipeline creation failed: {0}")]
    PipelineCreation(String),

    /// GPU execution error
    #[error("GPU execution failed: {0}")]
    Execution(String),

    /// Buffer mapping/read error
    #[error("Buffer read failed: {0}")]
    BufferRead(String),

    /// Memory pressure - circuit breaker tripped
    #[error("Memory pressure: circuit breaker {state:?}, retry after {retry_after_secs}s")]
    MemoryPressure {
        /// Circuit breaker state
        state: super::CircuitBreakerState,
        /// Seconds until retry is allowed
        retry_after_secs: u64,
    },

    /// Operation would exceed Metal watchdog timeout
    #[error("Operation too large: {elements} elements would exceed {max_ms}ms limit")]
    WatchdogRisk {
        /// Number of elements in operation
        elements: usize,
        /// Maximum time in milliseconds
        max_ms: f32,
    },

    /// Network architecture error
    #[error("Network error: {0}")]
    Network(String),

    /// Internal lock poisoned (concurrent panic)
    #[error("Internal lock poisoned: {0}")]
    LockPoisoned(String),
}

impl GpuError {
    /// Whether this error is recoverable by retry
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            GpuError::MemoryPressure { .. }
                | GpuError::PoolExhausted { .. }
                | GpuError::Execution(_)
        )
    }

    /// Whether this error indicates GPU should be avoided (use CPU fallback)
    pub fn should_fallback_to_cpu(&self) -> bool {
        matches!(
            self,
            GpuError::NotAvailable(_)
                | GpuError::DeviceCreation(_)
                | GpuError::MemoryPressure { .. }
                | GpuError::WatchdogRisk { .. }
        )
    }
}

/// Result type for GPU operations
pub type GpuResult<T> = Result<T, GpuError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_recovery() {
        let err = GpuError::MemoryPressure {
            state: super::super::CircuitBreakerState::Open,
            retry_after_secs: 30,
        };
        assert!(err.is_recoverable());
        assert!(err.should_fallback_to_cpu());

        let err = GpuError::InvalidDimensions("test".into());
        assert!(!err.is_recoverable());
        assert!(!err.should_fallback_to_cpu());
    }
}
