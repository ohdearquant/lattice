//! GPU Buffer Pool - 3-tier memory management with lifecycle tracking
//!
//! # Memory Management Strategy
//!
//! This module implements a production-grade GPU buffer pooling system designed
//! to address key challenges in GPU memory management:
//!
//! ## Problem
//!
//! 1. **Allocation Overhead**: GPU buffer allocation is expensive (~100-500μs per call)
//! 2. **Fragmentation**: Frequent alloc/dealloc leads to memory fragmentation
//! 3. **Async Deallocation**: Rust's `Drop` is synchronous but GPU dealloc is async,
//!    causing OOM during training loops despite "freeing" memory
//!
//! ## Solution: 3-Tier Buffer Pool
//!
//! Buffers are categorized by size and managed in separate pools:
//!
//! | Tier   | Size Range | Max Pooled | Max Age | Use Case |
//! |--------|------------|------------|---------|----------|
//! | Small  | < 1MB      | 256        | 5 min   | Biases, small activations |
//! | Medium | 1-10MB     | 64         | 3 min   | Layer weights |
//! | Large  | > 10MB     | 16         | 1 min   | Batch data, large layers |
//!
//! ## Lifecycle Management
//!
//! Each buffer tracks:
//! - **Creation time**: For age-based eviction
//! - **Last used time**: For idle-based cleanup
//! - **Use count**: For reuse efficiency metrics
//!
//! ## Memory Pressure Handling
//!
//! The pool integrates with [`CircuitBreaker`] to handle memory pressure:
//!
//! - **Normal**: Regular pooling and reuse
//! - **Low/Medium**: Aggressive cleanup of idle buffers
//! - **High/Critical**: Block new allocations, force CPU fallback
//!
//! ## Explicit Flush
//!
//! For training loops, call [`BufferPool::flush`] periodically to:
//! 1. Release all cached buffers immediately
//! 2. Prevent OOM from async deallocation lag
//!
//! ## Example
//!
//! ```ignore
//! // During training loop
//! for epoch in 0..epochs {
//!     for batch in dataset.batches() {
//!         train_step(&mut network, batch);
//!     }
//!     // Flush every epoch to prevent memory buildup
//!     ctx.buffer_pool().flush();
//!     ctx.poll(); // Process pending deallocations
//! }
//! ```

use super::apple_silicon::BUFFER_ALIGNMENT;
use super::circuit_breaker::{CircuitBreaker, MemoryPressure};
use super::error::{GpuError, GpuResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use wgpu::BufferUsages;

/// Buffer size categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferCategory {
    /// < 1MB - biases, small activations
    Small,
    /// 1-10MB - typical layer weights
    Medium,
    /// > 10MB - large layers, batch data
    Large,
}

impl BufferCategory {
    /// Categorize buffer by size
    pub fn from_size(bytes: usize) -> Self {
        const MB: usize = 1024 * 1024;
        if bytes < MB {
            BufferCategory::Small
        } else if bytes < 10 * MB {
            BufferCategory::Medium
        } else {
            BufferCategory::Large
        }
    }

    /// Maximum buffers to keep in pool per category
    pub fn max_pooled(&self) -> usize {
        match self {
            BufferCategory::Small => 256,
            BufferCategory::Medium => 64,
            BufferCategory::Large => 16,
        }
    }

    /// Maximum age before eviction (seconds)
    pub fn max_age_secs(&self) -> u64 {
        match self {
            BufferCategory::Small => 300,  // 5 min
            BufferCategory::Medium => 180, // 3 min
            BufferCategory::Large => 60,   // 1 min
        }
    }
}

/// A GPU buffer with lifecycle tracking
pub struct GpuBuffer {
    /// The wgpu buffer
    pub buffer: wgpu::Buffer,
    /// Buffer size in bytes
    pub size: u64,
    /// Buffer usage flags
    pub usage: BufferUsages,
    /// Category for pooling
    pub category: BufferCategory,
    /// When buffer was created
    created_at: Instant,
    /// When buffer was last used
    last_used: Instant,
    /// How many times buffer has been reused
    use_count: AtomicU64,
    /// Unique allocation ID for debugging and tracing buffer lifecycle.
    /// Not read in normal operation but useful for debugging memory issues.
    #[allow(dead_code)]
    allocation_id: u64,
}

impl GpuBuffer {
    /// Create a new GPU buffer
    pub fn new(buffer: wgpu::Buffer, size: u64, usage: BufferUsages, allocation_id: u64) -> Self {
        let category = BufferCategory::from_size(size as usize);
        let now = Instant::now();
        Self {
            buffer,
            size,
            usage,
            category,
            created_at: now,
            last_used: now,
            use_count: AtomicU64::new(1),
            allocation_id,
        }
    }

    /// Mark buffer as used
    pub fn mark_used(&mut self) {
        self.last_used = Instant::now();
        self.use_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get buffer age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get time since last use
    pub fn idle_time(&self) -> Duration {
        self.last_used.elapsed()
    }

    /// Get use count
    pub fn use_count(&self) -> u64 {
        self.use_count.load(Ordering::Relaxed)
    }

    /// Calculate reuse efficiency (uses per hour)
    pub fn reuse_efficiency(&self) -> f64 {
        let hours = self.age().as_secs_f64() / 3600.0;
        if hours > 0.0 {
            self.use_count() as f64 / hours
        } else {
            self.use_count() as f64
        }
    }

    /// Should this buffer be retained in pool?
    pub fn should_retain(&self) -> bool {
        let max_age = Duration::from_secs(self.category.max_age_secs());
        if self.age() > max_age {
            return false;
        }

        // Retain if used recently or has good reuse efficiency
        self.idle_time() < Duration::from_secs(60) || self.reuse_efficiency() > 1.0
    }
}

/// Configuration for a pool tier
#[derive(Debug, Clone)]
pub struct TierConfig {
    pub max_buffers: usize,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self { max_buffers: 64 }
    }
}

/// Pool statistics
#[derive(Debug, Default)]
pub struct PoolStats {
    pub total_allocations: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub evictions: AtomicU64,
    pub current_memory_bytes: AtomicU64,
}

impl PoolStats {
    pub fn hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// 3-tier buffer pool with lifecycle tracking
pub struct BufferPool {
    device: Arc<wgpu::Device>,
    /// Pooled buffers by category, keyed by (size, usage)
    pools: RwLock<HashMap<BufferCategory, Vec<GpuBuffer>>>,
    /// Per-tier configuration
    tier_configs: HashMap<BufferCategory, TierConfig>,
    /// Circuit breaker for memory pressure
    circuit_breaker: CircuitBreaker,
    /// Pool statistics
    stats: Arc<PoolStats>,
    /// Next allocation ID
    next_allocation_id: AtomicU64,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        let mut tier_configs = HashMap::new();
        tier_configs.insert(BufferCategory::Small, TierConfig { max_buffers: 256 });
        tier_configs.insert(BufferCategory::Medium, TierConfig { max_buffers: 64 });
        tier_configs.insert(BufferCategory::Large, TierConfig { max_buffers: 16 });

        Self {
            device,
            pools: RwLock::new(HashMap::new()),
            tier_configs,
            circuit_breaker: CircuitBreaker::new(5, Duration::from_secs(30)),
            stats: Arc::new(PoolStats::default()),
            next_allocation_id: AtomicU64::new(1),
        }
    }

    /// Allocate or reuse a buffer
    pub fn allocate(
        &self,
        size: u64,
        usage: BufferUsages,
        label: Option<&str>,
    ) -> GpuResult<GpuBuffer> {
        // Check circuit breaker
        if !self.circuit_breaker.allow_request() {
            return Err(GpuError::MemoryPressure {
                state: self.circuit_breaker.state(),
                retry_after_secs: 30,
            });
        }

        let aligned_size = align_size(size as usize, BUFFER_ALIGNMENT) as u64;
        let category = BufferCategory::from_size(aligned_size as usize);

        // Try to reuse from pool
        if let Some(mut buffer) = self.try_reuse(aligned_size, usage, category) {
            buffer.mark_used();
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(buffer);
        }

        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);

        // Allocate new buffer
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: aligned_size,
            usage,
            mapped_at_creation: false,
        });

        let allocation_id = self.next_allocation_id.fetch_add(1, Ordering::Relaxed);
        self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.stats
            .current_memory_bytes
            .fetch_add(aligned_size, Ordering::Relaxed);

        Ok(GpuBuffer::new(buffer, aligned_size, usage, allocation_id))
    }

    /// Try to reuse a buffer from the pool
    fn try_reuse(
        &self,
        size: u64,
        usage: BufferUsages,
        category: BufferCategory,
    ) -> Option<GpuBuffer> {
        let mut pools = self
            .pools
            .write()
            .map_err(|e| {
                tracing::warn!("Buffer pool lock poisoned in try_reuse: {}", e);
                e
            })
            .ok()?;
        let pool = pools.get_mut(&category)?;

        // Find a suitable buffer (same usage, size >= requested)
        let idx = pool.iter().position(|b| {
            b.usage == usage && b.size >= size && b.size <= size * 2 // Don't waste >2x memory
        })?;

        Some(pool.swap_remove(idx))
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&self, buffer: GpuBuffer) {
        // Check if we should retain this buffer
        if !buffer.should_retain() {
            self.stats
                .current_memory_bytes
                .fetch_sub(buffer.size, Ordering::Relaxed);
            return; // Drop buffer
        }

        let config = self
            .tier_configs
            .get(&buffer.category)
            .cloned()
            .unwrap_or_default();

        if let Ok(mut pools) = self.pools.write() {
            let pool = pools.entry(buffer.category).or_insert_with(Vec::new);

            // Check if pool is full
            if pool.len() >= config.max_buffers {
                // Evict oldest buffer
                if let Some(oldest_idx) = pool
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, b)| b.use_count())
                    .map(|(i, _)| i)
                {
                    let evicted = pool.swap_remove(oldest_idx);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                    self.stats
                        .current_memory_bytes
                        .fetch_sub(evicted.size, Ordering::Relaxed);
                }
            }

            pool.push(buffer);
        }
    }

    /// Run cleanup on all pools
    pub fn cleanup(&self, pressure: MemoryPressure) {
        let aggressiveness = pressure.cleanup_aggressiveness();

        if let Ok(mut pools) = self.pools.write() {
            for (category, pool) in pools.iter_mut() {
                let config = self.tier_configs.get(category).cloned().unwrap_or_default();

                // Calculate target size based on pressure
                let target = ((1.0 - aggressiveness) * config.max_buffers as f32) as usize;

                // Remove buffers that shouldn't be retained, prioritizing oldest
                pool.sort_by_key(|b| std::cmp::Reverse(b.use_count()));

                while pool.len() > target {
                    if let Some(buffer) = pool.pop() {
                        self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                        self.stats
                            .current_memory_bytes
                            .fetch_sub(buffer.size, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    /// Record allocation failure (for circuit breaker)
    pub fn record_failure(&self) {
        self.circuit_breaker.record_failure();
    }

    /// Record allocation success (for circuit breaker)
    pub fn record_success(&self) {
        self.circuit_breaker.record_success();
    }

    /// Get pool statistics
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> u64 {
        self.stats.current_memory_bytes.load(Ordering::Relaxed)
    }

    /// Flush all pooled buffers, releasing GPU memory
    ///
    /// This method drops all cached buffers immediately, freeing their GPU memory.
    /// Use this during long training loops or when memory pressure is detected.
    ///
    /// Returns the number of bytes freed.
    pub fn flush(&self) -> u64 {
        let mut total_freed = 0u64;

        if let Ok(mut pools) = self.pools.write() {
            for (_, pool) in pools.iter_mut() {
                for buffer in pool.drain(..) {
                    total_freed += buffer.size;
                    self.stats
                        .current_memory_bytes
                        .fetch_sub(buffer.size, Ordering::Relaxed);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        total_freed
    }

    /// Flush buffers of a specific category
    ///
    /// Returns the number of bytes freed.
    pub fn flush_category(&self, category: BufferCategory) -> u64 {
        let mut total_freed = 0u64;

        if let Ok(mut pools) = self.pools.write() {
            if let Some(pool) = pools.get_mut(&category) {
                for buffer in pool.drain(..) {
                    total_freed += buffer.size;
                    self.stats
                        .current_memory_bytes
                        .fetch_sub(buffer.size, Ordering::Relaxed);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        total_freed
    }

    /// Get number of pooled buffers
    pub fn pooled_count(&self) -> usize {
        if let Ok(pools) = self.pools.read() {
            pools.values().map(|p| p.len()).sum()
        } else {
            0
        }
    }
}

/// Align size to boundary
#[inline]
fn align_size(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_category() {
        assert_eq!(BufferCategory::from_size(1000), BufferCategory::Small);
        assert_eq!(
            BufferCategory::from_size(5 * 1024 * 1024),
            BufferCategory::Medium
        );
        assert_eq!(
            BufferCategory::from_size(50 * 1024 * 1024),
            BufferCategory::Large
        );
    }

    #[test]
    fn test_align_size() {
        assert_eq!(align_size(100, 256), 256);
        assert_eq!(align_size(256, 256), 256);
        assert_eq!(align_size(257, 256), 512);
        assert_eq!(align_size(1000, 256), 1024);
    }
}
