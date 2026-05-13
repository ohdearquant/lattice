//! Sharded embedding cache with LRU eviction.
//!
//! Caches embeddings to avoid re-computing for identical texts. Uses 16 independent
//! shards to reduce write-lock contention — each `get()` on an LRU cache requires a
//! write lock (to update access order), so a single `RwLock<LruCache>` serializes all
//! reads under high QPS. Sharding reduces contention by a factor of `NUM_SHARDS`.
//!
//! # Design
//!
//! - **Shard selection**: First byte of the Blake3 cache key, masked to `NUM_SHARDS - 1`.
//!   Blake3 output is uniformly distributed, so shard load is balanced.
//! - **Per-shard capacity**: `total_capacity / NUM_SHARDS`. Each shard independently
//!   evicts its own LRU entries.
//! - **Per-shard statistics**: Hit/miss counters are per-shard `AtomicU64`s, aggregated
//!   in `stats()`. This eliminates cross-shard atomic contention on the hot path.
//! - **Zero-capacity**: `capacity=0` disables caching entirely — all operations become
//!   no-ops with no locking or hashing work.

use crate::model::ModelConfig;
use lru::LruCache;
use parking_lot::RwLock;
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

/// **Unstable**: internal LRU caching mechanism; shard count and eviction policy may change.
///
/// Embedding cache with sharded LRU eviction policy.
///
/// Thread-safe cache for storing computed embeddings. Uses Blake3 hashing
/// for fast, collision-resistant cache keys. Internally sharded into 16
/// independent LRU caches to reduce write-lock contention.
///
/// # Disabling
///
/// Pass `capacity=0` to disable caching. All cache operations become no-ops
/// (no locking, no hashing work beyond key construction).
///
/// # Example
///
/// ```rust
/// use lattice_embed::{EmbeddingCache, EmbeddingModel, ModelConfig};
///
/// let cache = EmbeddingCache::new(1000);
///
/// // Cache miss - no embedding stored yet
/// let key = cache.compute_key("Hello, world!", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));
/// assert!(cache.get(&key).is_none());
///
/// // Store embedding
/// let embedding = vec![0.1, 0.2, 0.3];
/// cache.put(key, embedding.clone());
///
/// // Cache hit — returns Arc<[f32]>
/// let cached = cache.get(&key).unwrap();
/// assert_eq!(&*cached, &embedding[..]);
/// ```
pub struct EmbeddingCache {
    shards: Vec<CacheShard>,
    enabled: bool,
    capacity: usize,
}

/// Select shard index from a cache key. Uses first byte masked to shard count.
/// Blake3 output is uniformly distributed, so this gives balanced load.
#[inline(always)]
fn shard_index(key: &CacheKey) -> usize {
    key[0] as usize & SHARD_MASK
}

impl EmbeddingCache {
    /// **Unstable**: constructor signature may change when shard count becomes configurable.
    ///
    /// The capacity is divided equally across 16 internal shards. Each shard
    /// independently manages its own LRU eviction.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum total number of embeddings to cache. Use 0 to disable caching.
    pub fn new(capacity: usize) -> Self {
        let enabled = capacity != 0;

        // Per-shard capacity: ceiling division ensures total actual capacity >= requested.
        // E.g., capacity=4000, NUM_SHARDS=16 → 250/shard (exact).
        // E.g., capacity=10, NUM_SHARDS=16 → 1/shard (at least 1).
        let per_shard = if enabled {
            // Ceiling division: (capacity + NUM_SHARDS - 1) / NUM_SHARDS, minimum 1.
            let base = capacity.div_ceil(NUM_SHARDS);
            if base == 0 { 1 } else { base }
        } else {
            1 // Dummy capacity for disabled cache
        };

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

    /// **Unstable**: key scheme (Blake3 + EmbeddingKey canonical bytes) may change; don't store keys across sessions.
    ///
    /// Uses Blake3 hashing for fast, collision-resistant keys. The key includes the model
    /// name, revision, and active dimension from the `ModelConfig`, so different MRL truncations
    /// produce different cache keys.
    pub fn compute_key(&self, text: &str, model_config: ModelConfig) -> CacheKey {
        let mut hasher = blake3::Hasher::new();
        hasher.update(text.as_bytes());
        // Unique identifier for the model config: "model_name:version:dims"
        let model_key = format!(
            "{}:{}:{}",
            model_config.model,
            model_config.model.key_version(),
            model_config.dimensions(),
        );
        hasher.update(model_key.as_bytes());
        *hasher.finalize().as_bytes()
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
mod tests {
    use super::*;
    use crate::model::EmbeddingModel;

    #[test]
    fn test_cache_basic_operations() {
        let cache = EmbeddingCache::new(100);
        let key = cache.compute_key("hello", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));

        // Miss
        assert!(cache.get(&key).is_none());

        // Put
        let embedding = vec![0.1, 0.2, 0.3];
        cache.put(key, embedding.clone());

        // Hit — returns Arc<[f32]>
        let cached = cache.get(&key).unwrap();
        assert_eq!(&*cached, &embedding[..]);
    }

    #[test]
    fn test_cache_eviction() {
        // With 16 shards, a capacity of 16 gives 1 entry per shard.
        // To test eviction, we need keys that hash to the same shard.
        // Use a larger capacity and fill it up.
        let cache = EmbeddingCache::new(16);

        // Insert 32 entries — each shard has capacity 1, so each shard
        // can only hold 1 entry. Inserting 2 entries to the same shard
        // will evict the first.
        let mut keys = Vec::new();
        for i in 0..32u32 {
            let text = format!("text_{}", i);
            let key = cache.compute_key(&text, ModelConfig::new(EmbeddingModel::BgeSmallEnV15));
            keys.push(key);
            cache.put(key, vec![i as f32]);
        }

        // Total size should not exceed capacity (16)
        let stats = cache.stats();
        assert!(stats.size <= 16, "size {} exceeds capacity 16", stats.size);
    }

    #[test]
    fn test_cache_lru_eviction_within_shard() {
        // Create cache with capacity 32 (2 per shard).
        let cache = EmbeddingCache::new(32);

        // Find 3 keys that land in the same shard.
        let mut same_shard_keys = Vec::new();
        let mut i = 0u32;
        let target_shard;

        // Find the first key's shard and collect 3 keys for it.
        let first_key =
            cache.compute_key("probe_0", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));
        target_shard = shard_index(&first_key);

        loop {
            let key = cache.compute_key(
                &format!("lru_test_{}", i),
                ModelConfig::new(EmbeddingModel::BgeSmallEnV15),
            );
            if shard_index(&key) == target_shard {
                same_shard_keys.push((key, i));
            }
            if same_shard_keys.len() == 3 {
                break;
            }
            i += 1;
        }

        let (k1, v1) = same_shard_keys[0];
        let (k2, v2) = same_shard_keys[1];
        let (k3, v3) = same_shard_keys[2];

        // Insert k1 and k2 (shard capacity is 2).
        cache.put(k1, vec![v1 as f32]);
        cache.put(k2, vec![v2 as f32]);

        // Access k1 to make it recently used.
        assert!(cache.get(&k1).is_some());

        // Insert k3 — should evict k2 (least recently used in this shard).
        cache.put(k3, vec![v3 as f32]);

        assert!(
            cache.get(&k1).is_some(),
            "k1 should survive (recently accessed)"
        );
        assert!(cache.get(&k2).is_none(), "k2 should be evicted (LRU)");
        assert!(cache.get(&k3).is_some(), "k3 should exist (just inserted)");
    }

    #[test]
    fn test_cache_different_models_different_keys() {
        let cache = EmbeddingCache::new(100);

        let key_small = cache.compute_key("text", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));
        let key_base = cache.compute_key("text", ModelConfig::new(EmbeddingModel::BgeBaseEnV15));

        // Same text, different models = different keys
        assert_ne!(key_small, key_base);
    }

    #[test]
    fn test_cache_stats() {
        let cache = EmbeddingCache::new(100);
        let key = cache.compute_key("hello", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));

        cache.get(&key); // Miss
        cache.put(key, vec![0.1]);
        cache.get(&key); // Hit

        let stats = cache.stats();
        assert_eq!(stats.size, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_cache_get_many() {
        // Use capacity large enough that no shard evicts.
        let cache = EmbeddingCache::new(100);

        let key1 = cache.compute_key("one", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));
        let key2 = cache.compute_key("two", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));
        let key3 = cache.compute_key("three", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));

        cache.put(key1, vec![1.0]);
        cache.put(key3, vec![3.0]);

        let results = cache.get_many(&[key1, key2, key3]);
        assert_eq!(results.len(), 3);
        assert_eq!(&**results[0].as_ref().unwrap(), &[1.0f32]);
        assert!(results[1].is_none());
        assert_eq!(&**results[2].as_ref().unwrap(), &[3.0f32]);
    }

    #[test]
    fn test_cache_put_many() {
        // Use capacity large enough that no shard evicts (ceil(100/16) = 7 per shard).
        let cache = EmbeddingCache::new(100);

        let key1 = cache.compute_key("one", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));
        let key2 = cache.compute_key("two", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));

        cache.put_many(vec![(key1, vec![1.0]), (key2, vec![2.0])]);

        let v1 = cache.get(&key1).unwrap();
        assert_eq!(&*v1, [1.0f32].as_slice());
        let v2 = cache.get(&key2).unwrap();
        assert_eq!(&*v2, [2.0f32].as_slice());
    }

    #[test]
    fn test_cache_clear() {
        let cache = EmbeddingCache::new(100);
        let key = cache.compute_key("hello", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));

        cache.put(key, vec![0.1]);
        assert!(cache.get(&key).is_some());

        cache.clear();
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().size, 0);
    }

    #[test]
    fn test_cache_default_capacity() {
        let cache = EmbeddingCache::with_default_capacity();
        assert_eq!(cache.stats().capacity, DEFAULT_CACHE_CAPACITY);
    }

    #[test]
    fn test_cache_disabled_is_noop() {
        let cache = EmbeddingCache::new(0);
        assert!(!cache.is_enabled());

        let key = cache.compute_key("hello", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));
        cache.put(key, vec![0.1]);
        assert!(cache.get(&key).is_none());

        let stats = cache.stats();
        assert_eq!(stats.capacity, 0);
        assert_eq!(stats.size, 0);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        // Use large capacity so no eviction occurs (800 entries across 16 shards).
        // Per-shard capacity = ceil(4000/16) = 250, so 800 entries fit easily.
        let cache = Arc::new(EmbeddingCache::new(4000));
        let mut handles = Vec::new();

        // Spawn 8 threads, each doing 100 put+get operations.
        for t in 0..8 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    // Each thread uses unique keys to avoid contention on same entry.
                    let text = format!("thread_{}_item_{}", t, i);
                    let key =
                        cache.compute_key(&text, ModelConfig::new(EmbeddingModel::BgeSmallEnV15));
                    let embedding = vec![t as f32; 384];
                    cache.put(key, embedding.clone());

                    let result = cache.get(&key);
                    assert!(result.is_some(), "put followed by get must succeed");
                    assert_eq!(result.unwrap().len(), 384);
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        // All 800 entries should be in cache (capacity 4000 >> 800).
        let stats = cache.stats();
        assert_eq!(stats.size, 800);
        assert!(stats.hits >= 800, "at least 800 hits expected");
    }

    #[test]
    fn test_shard_distribution() {
        // Use generous capacity to avoid eviction from uneven distribution.
        // 4000 total → 250/shard. We insert 800 entries → ~50/shard on average.
        let cache = EmbeddingCache::new(4000);

        let n = 800;
        for i in 0..n {
            let key = cache.compute_key(
                &format!("item_{}", i),
                ModelConfig::new(EmbeddingModel::BgeSmallEnV15),
            );
            cache.put(key, vec![i as f32]);
        }

        let shard_stats = cache.per_shard_stats();
        assert_eq!(shard_stats.len(), NUM_SHARDS);

        // Each shard should have entries. With uniform hash, each shard gets ~50.
        for ss in &shard_stats {
            assert!(
                ss.size > 0,
                "shard {} is empty — distribution is pathological",
                ss.shard_id
            );
        }

        // No eviction since 800 << 4000. Total should be exactly 800.
        let total: usize = shard_stats.iter().map(|s| s.size).sum();
        assert_eq!(total, n);

        // Check distribution is reasonably uniform: no shard has >3x the average.
        let avg = n / NUM_SHARDS; // 50
        for ss in &shard_stats {
            assert!(
                ss.size <= avg * 3,
                "shard {} has {} entries (avg {}), distribution too skewed",
                ss.shard_id,
                ss.size,
                avg
            );
        }
    }

    #[test]
    fn test_per_shard_stats_hit_tracking() {
        let cache = EmbeddingCache::new(100);

        // Insert a few entries and access them.
        let key1 = cache.compute_key("hello", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));
        let key2 = cache.compute_key("world", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));

        cache.put(key1, vec![1.0]);
        cache.put(key2, vec![2.0]);

        // Access key1 three times, key2 once.
        cache.get(&key1);
        cache.get(&key1);
        cache.get(&key1);
        cache.get(&key2);

        let shard_stats = cache.per_shard_stats();
        let total_hits: u64 = shard_stats.iter().map(|s| s.hits).sum();
        assert_eq!(total_hits, 4, "total hits should be 4");

        let stats = cache.stats();
        assert_eq!(stats.hits, 4);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_small_capacity_rounds_up() {
        // Capacity smaller than NUM_SHARDS: each shard gets at least 1.
        let cache = EmbeddingCache::new(3);
        assert!(cache.is_enabled());

        let key = cache.compute_key("x", ModelConfig::new(EmbeddingModel::BgeSmallEnV15));
        cache.put(key, vec![42.0]);
        assert!(cache.get(&key).is_some());
    }
}
