//! In-memory embedding cache with sharded LRU eviction.
//!
//! A key selects one of 16 independent shards. Cache hits refresh LRU order and therefore
//! take that shard's write lock; hit/miss counters stay local to the shard. A zero-capacity
//! cache bypasses storage operations and locking. The power-of-two shard count is required
//! by the masking-based shard selection.
//!
//! See docs/model.md for key construction, capacity semantics, and lifecycle details.

use crate::model::ModelConfig;
use crate::service::EmbeddingRole;
use lru::LruCache;
use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::debug;

/// **Unstable**: internal implementation detail; type alias may change with cache redesign.
pub type CacheKey = [u8; 32];

/// **Unstable**: tuning constant; value may change as memory models evolve.
///
/// Default cache capacity (number of embeddings). ~6MB for 384-dim vectors at 4000 entries.
pub const DEFAULT_CACHE_CAPACITY: usize = 4000;

/// Number of cache shards. Must be a power of 2 for fast modulo (bitwise AND).
/// 16 shards on 8-core M4 Pro gives 2x oversubscription, keeping contention low.
const NUM_SHARDS: usize = 16;

/// Mask for shard index computation: `key[0] as usize & SHARD_MASK`.
const SHARD_MASK: usize = NUM_SHARDS - 1;

// Compile-time assertion that NUM_SHARDS is a power of 2.
const _: () = assert!(
    NUM_SHARDS.is_power_of_two(),
    "NUM_SHARDS must be a power of 2"
);

/// A single cache shard with its own LRU cache and hit/miss counters.
struct CacheShard {
    lru: RwLock<LruCache<CacheKey, Arc<[f32]>>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CacheShard {
    fn new(capacity: NonZeroUsize) -> Self {
        Self {
            lru: RwLock::new(LruCache::new(capacity)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    #[inline]
    fn get(&self, key: &CacheKey) -> Option<Arc<[f32]>> {
        let mut lru = self.lru.write();
        let result = lru.get(key).cloned();
        if result.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    #[inline]
    fn put(&self, key: CacheKey, embedding: Arc<[f32]>) {
        let mut lru = self.lru.write();
        lru.put(key, embedding);
    }

    fn len(&self) -> usize {
        self.lru.read().len()
    }

    fn clear(&self) {
        self.lru.write().clear();
    }

    fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
}

/// **Unstable**: the sharding and eviction implementation may change.
///
/// Thread-safe, sharded LRU cache for computed embeddings.
/// A capacity of zero disables storage operations.
/// See [`docs/design.md`](../docs/design.md#embeddingcache) for key identity, locking, and capacity semantics.
pub struct EmbeddingCache {
    shards: Vec<CacheShard>,
    enabled: bool,
    capacity: usize,
}

/// Select shard index from a cache key. Uses first byte masked to shard count.
/// SHA-256 output is uniformly distributed, so this gives balanced load.
#[inline(always)]
fn shard_index(key: &CacheKey) -> usize {
    key[0] as usize & SHARD_MASK
}

impl EmbeddingCache {
    /// **Unstable**: constructor signature may change when shard count becomes configurable.
    ///
    /// Creates a cache with the requested capacity; zero disables storage.
    /// Nonzero capacity is rounded up independently across its fixed shards.
    /// See [`docs/design.md`](../docs/design.md#embeddingcache) for sharding and eviction behavior.
    pub fn new(capacity: usize) -> Self {
        let enabled = capacity != 0;

        // Round up per shard so the actual aggregate capacity meets the request.
        let per_shard = if enabled {
            let base = capacity.div_ceil(NUM_SHARDS);
            if base == 0 { 1 } else { base }
        } else {
            1 // Disabled caches still need a valid LRU capacity.
        };

        #[expect(
            clippy::expect_used,
            reason = "per_shard is clamped to at least 1 on the lines above"
        )]
        let per_shard_nz = NonZeroUsize::new(per_shard).expect("per_shard is always >= 1");

        let shards = (0..NUM_SHARDS)
            .map(|_| CacheShard::new(per_shard_nz))
            .collect();

        Self {
            shards,
            enabled,
            capacity,
        }
    }

    /// **Unstable**: convenience constructor; subject to change with cache redesign.
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_CACHE_CAPACITY)
    }

    /// **Unstable**: the key scheme may change; do not persist keys across sessions.
    ///
    /// Hashes text, model identity, active dimension, and retrieval role into a cache key.
    /// See [`docs/design.md`](../docs/design.md#embeddingcache) for identity and collision-isolation details.
    pub fn compute_key(
        &self,
        text: &str,
        model_config: ModelConfig,
        role: EmbeddingRole,
    ) -> CacheKey {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        // Deterministic model/role namespace for this cache key.
        let model_key = format!(
            "{}:{}:{}:{}",
            model_config.model,
            model_config.model.key_version(),
            model_config.dimensions(),
            role.cache_tag(),
        );
        hasher.update(model_key.as_bytes());
        hasher.finalize().into()
    }

    /// **Unstable**: return type (`Arc<[f32]>`) may change to a newtype; internal cache API.
    ///
    /// Returns `Some(Arc<[f32]>)` if found (cheap refcount bump), `None` otherwise.
    /// Updates per-shard hit/miss counters for metrics.
    pub fn get(&self, key: &CacheKey) -> Option<Arc<[f32]>> {
        if !self.enabled {
            return None;
        }

        let idx = shard_index(key);
        let result = self.shards[idx].get(key);

        if result.is_some() {
            debug!("cache hit for key {:?}", &key[..8]);
        }

        result
    }

    /// **Unstable**: internal cache storage method; interface may change.
    ///
    /// Converts the Vec into `Arc<[f32]>` for shared-ownership storage.
    /// If the shard is at capacity, its least recently used entry is evicted.
    pub fn put(&self, key: CacheKey, embedding: Vec<f32>) {
        if !self.enabled {
            return;
        }

        let idx = shard_index(&key);
        self.shards[idx].put(key, Arc::from(embedding));
        debug!("cached embedding for key {:?}", &key[..8]);
    }

    /// **Unstable**: batch cache access; return type may change with cache redesign.
    ///
    /// Returns a vector of `Option<Arc<[f32]>>` for each key, in the same order.
    /// Each hit is an O(1) refcount bump (no data copy).
    pub fn get_many(&self, keys: &[CacheKey]) -> Vec<Option<Arc<[f32]>>> {
        if !self.enabled {
            return vec![None; keys.len()];
        }

        keys.iter()
            .map(|key| {
                let idx = shard_index(key);
                self.shards[idx].get(key)
            })
            .collect()
    }

    /// **Unstable**: batch cache storage; interface may change with cache redesign.
    ///
    /// Converts each Vec into `Arc<[f32]>` for shared-ownership storage.
    pub fn put_many(&self, entries: Vec<(CacheKey, Vec<f32>)>) {
        if !self.enabled {
            return;
        }

        for (key, embedding) in entries {
            let idx = shard_index(&key);
            self.shards[idx].put(key, Arc::from(embedding));
        }
    }

    /// **Unstable**: returns `CacheStats` which is itself Unstable; metrics shape may evolve.
    ///
    /// Aggregates per-shard counters. The `size` field is the sum of all shard sizes.
    pub fn stats(&self) -> CacheStats {
        if !self.enabled {
            let (hits, misses) = self.aggregate_counters();
            return CacheStats {
                size: 0,
                capacity: 0,
                hits,
                misses,
            };
        }

        let size: usize = self.shards.iter().map(CacheShard::len).sum();
        let (hits, misses) = self.aggregate_counters();

        CacheStats {
            size,
            capacity: self.capacity,
            hits,
            misses,
        }
    }

    /// **Unstable**: internal monitoring hook; shard count and `ShardStats` shape may change.
    ///
    /// Returns a vector of `(size, hits, misses)` tuples, one per shard.
    pub fn per_shard_stats(&self) -> Vec<ShardStats> {
        self.shards
            .iter()
            .enumerate()
            .map(|(i, s)| ShardStats {
                shard_id: i,
                size: s.len(),
                hits: s.hits(),
                misses: s.misses(),
            })
            .collect()
    }

    /// **Unstable**: internal cache management; may be removed in favor of capacity-based eviction.
    pub fn clear(&self) {
        if !self.enabled {
            return;
        }

        for shard in &self.shards {
            shard.clear();
        }
        debug!("cache cleared");
    }

    /// **Unstable**: internal state query; may be removed when zero-capacity is the only disable path.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Aggregate hit/miss counters across all shards.
    fn aggregate_counters(&self) -> (u64, u64) {
        let hits: u64 = self.shards.iter().map(CacheShard::hits).sum();
        let misses: u64 = self.shards.iter().map(CacheShard::misses).sum();
        (hits, misses)
    }
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

/// **Unstable**: metrics fields may be added/removed as monitoring needs evolve.
///
/// Cache statistics (aggregated across all shards).
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Current number of cached entries (sum across all shards).
    pub size: usize,
    /// Maximum total cache capacity.
    pub capacity: usize,
    /// Number of cache hits (sum across all shards).
    pub hits: u64,
    /// Number of cache misses (sum across all shards).
    pub misses: u64,
}

impl CacheStats {
    /// **Unstable**: convenience metric; may move to a separate stats helper.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// **Unstable**: shard count is an internal implementation detail; this struct may be removed.
///
/// Per-shard statistics for detailed monitoring.
#[derive(Debug, Clone, Copy)]
pub struct ShardStats {
    /// Shard index (0 to NUM_SHARDS-1).
    pub shard_id: usize,
    /// Current number of entries in this shard.
    pub size: usize,
    /// Number of cache hits in this shard.
    pub hits: u64,
    /// Number of cache misses in this shard.
    pub misses: u64,
}

impl ShardStats {
    /// **Unstable**: per-shard metric; may be removed with `ShardStats`.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests;
