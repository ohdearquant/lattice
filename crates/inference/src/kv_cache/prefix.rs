//! Prefix-sharing layer for paged KV cache reuse.
//!
//! A prefix entry owns immutable page buffers through `Arc<[f32]>`. Active
//! `PagedKVCache` instances copy those buffers into their owned `PagePool`
//! before decode continues, so shared pages are never mutated.

use indexmap::IndexMap;
use rustc_hash::FxHasher;
use std::hash::{BuildHasher, BuildHasherDefault, Hash};
use std::sync::Arc;

type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

/// Adapter ID for LoRA-aware prefix namespacing.
///
/// `AdapterId(0)` is the base model with no adapter. Non-zero values are the
/// adapter content hashes supplied by the adapter-loading path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AdapterId(pub u64);

impl AdapterId {
    /// Base model / no adapter namespace.
    pub const BASE: Self = Self(0);

    /// Create an adapter namespace from a caller-supplied content hash.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
}

/// Compound key enforcing adapter namespace separation at the data structure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefixKey {
    /// Adapter namespace. `AdapterId::BASE` means no adapter.
    pub adapter_id: AdapterId,
    /// FxHash of the token ID slice used as the reusable prefix.
    pub token_hash: u64,
}

impl PrefixKey {
    /// Build a prefix key from adapter namespace plus token content.
    pub fn from_token_ids(adapter_id: AdapterId, token_ids: &[u32]) -> Self {
        Self {
            adapter_id,
            token_hash: Self::hash_token_ids(token_ids),
        }
    }

    /// Hash token IDs with FxHasher, matching ADR-047.
    pub fn hash_token_ids(token_ids: &[u32]) -> u64 {
        BuildHasherDefault::<FxHasher>::default().hash_one(token_ids)
    }
}

/// Read-only shared reference to one prefix page.
///
/// The slice layout is `[num_layers, 2, prefix_page_size, kv_dim]`.
#[derive(Debug, Clone)]
pub struct SharedPageRef {
    page: Arc<[f32]>,
}

impl SharedPageRef {
    /// Convert an owned page buffer into an immutable shared page.
    pub fn from_vec(page: Vec<f32>) -> Self {
        Self {
            page: Arc::from(page.into_boxed_slice()),
        }
    }

    /// Borrow the raw page data.
    pub fn as_slice(&self) -> &[f32] {
        self.page.as_ref()
    }

    /// Number of floats in this page.
    pub fn len(&self) -> usize {
        self.page.len()
    }

    /// Whether this page has no floats.
    pub fn is_empty(&self) -> bool {
        self.page.is_empty()
    }

    /// Current strong reference count, used only for cache accounting/tests.
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.page)
    }
}

/// Configuration for a bounded prefix page cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixPageCacheConfig {
    /// Maximum retained prefix entries across all adapter namespaces.
    pub capacity: usize,
    /// Tokens per shared prefix page. ADR-047 default is 64.
    pub prefix_page_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Dimension per KV head.
    pub head_dim: usize,
}

impl PrefixPageCacheConfig {
    /// ADR-047 default prefix cache capacity.
    pub const DEFAULT_CAPACITY: usize = 128;

    /// ADR-047 default prefix page size.
    pub const DEFAULT_PREFIX_PAGE_SIZE: usize = 64;

    /// KV dimension per token.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Floats in one shared prefix page.
    pub fn floats_per_prefix_page(&self) -> usize {
        self.num_layers * 2 * self.prefix_page_size * self.kv_dim()
    }
}

/// Prefix cache entry with immutable shared pages and LRU metadata.
#[derive(Debug, Clone)]
pub struct PrefixEntry {
    /// Number of valid tokens represented by this entry.
    pub prefix_len: usize,
    /// Tokens per page for `pages`.
    pub prefix_page_size: usize,
    /// Shared prefix pages, each laid out as `[num_layers, 2, prefix_page_size, kv_dim]`.
    pub pages: Vec<SharedPageRef>,
    /// Monotonic use counter; larger values are more recently used.
    pub last_used: u64,
}

impl PrefixEntry {
    /// Build a cache entry. `PrefixPageCache::insert` stamps `last_used`.
    pub fn new(
        prefix_len: usize,
        prefix_page_size: usize,
        pages: Vec<SharedPageRef>,
        last_used: u64,
    ) -> Self {
        Self {
            prefix_len,
            prefix_page_size,
            pages,
            last_used,
        }
    }

    /// Number of pages required to hold `token_count` tokens.
    pub fn pages_for_tokens(token_count: usize, page_size: usize) -> usize {
        assert!(page_size > 0, "page_size must be non-zero");
        if token_count == 0 {
            0
        } else {
            ((token_count - 1) / page_size) + 1
        }
    }
}

/// A hash-keyed LRU cache of immutable prefix pages.
///
/// LRU ordering is maintained via `IndexMap` insertion order: lookups
/// re-insert the entry at the back, eviction removes from the front.
/// This bounds memory to O(capacity) with no auxiliary deque.
#[derive(Debug)]
pub struct PrefixPageCache {
    config: PrefixPageCacheConfig,
    entries: FxIndexMap<PrefixKey, PrefixEntry>,
    clock: u64,
}

impl PrefixPageCache {
    /// Create an empty bounded prefix cache.
    ///
    /// `capacity` is an entry count, not a byte or page count. Capacity zero is
    /// valid and disables retention: inserts immediately evict themselves.
    pub fn new(config: PrefixPageCacheConfig) -> Self {
        assert!(
            config.prefix_page_size > 0,
            "prefix_page_size must be non-zero"
        );
        Self {
            config,
            entries: IndexMap::with_capacity_and_hasher(
                config.capacity,
                BuildHasherDefault::<FxHasher>::default(),
            ),
            clock: 0,
        }
    }

    /// Return immutable cache configuration.
    pub fn config(&self) -> PrefixPageCacheConfig {
        self.config
    }

    /// Number of retained prefix entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache currently has no retained entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return `true` when `key` is retained.
    pub fn contains_key(&self, key: &PrefixKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Look up a prefix entry and update LRU metadata on hit.
    ///
    /// Returns a cloned `PrefixEntry`; cloning only increments page `Arc`
    /// counts. This lets callers release any external mutex before copying page
    /// data into a live `PagePool`.
    pub fn lookup(&mut self, key: &PrefixKey) -> Option<PrefixEntry> {
        let last_used = self.next_clock();
        let mut entry = self.entries.swap_remove(key)?;
        entry.last_used = last_used;
        let cloned = entry.clone();
        self.entries.insert(*key, entry);
        Some(cloned)
    }

    /// Insert prefix pages for `key`.
    ///
    /// The page count must match `prefix_len` and `config.prefix_page_size`.
    /// Replaces and returns any existing entry for the same key. LRU eviction
    /// runs after insertion until `len() <= capacity`.
    pub fn insert(
        &mut self,
        key: PrefixKey,
        prefix_len: usize,
        pages: Vec<SharedPageRef>,
    ) -> Option<PrefixEntry> {
        let entry = PrefixEntry::new(prefix_len, self.config.prefix_page_size, pages, 0);
        self.insert_entry(key, entry)
    }

    /// Insert a fully constructed entry for tests or advanced callers.
    ///
    /// `entry.last_used` is overwritten by this cache's monotonic counter.
    pub fn insert_entry(&mut self, key: PrefixKey, mut entry: PrefixEntry) -> Option<PrefixEntry> {
        self.validate_entry(&entry);
        let last_used = self.next_clock();
        entry.last_used = last_used;
        let replaced = self.entries.swap_remove(&key);
        self.entries.insert(key, entry);
        self.evict_until_within_capacity();
        replaced
    }

    /// Evict one least-recently-used entry.
    ///
    /// Returns the number of pages whose `Arc` strong count was one before
    /// eviction, which is the number of page buffers actually reclaimed by this
    /// eviction. Entries with external clones are still removed from the cache,
    /// but their memory is released only after those clones drop.
    pub fn evict_lru(&mut self) -> usize {
        if self.entries.is_empty() {
            return 0;
        }
        let (_key, entry) = self.entries.shift_remove_index(0).expect("non-empty");
        entry
            .pages
            .iter()
            .filter(|page| page.strong_count() == 1)
            .count()
    }

    /// Remove every retained prefix entry.
    ///
    /// Use this on model swap or model reload. Existing active clones remain
    /// alive until their `Arc` references drop.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.clock = 0;
    }

    fn next_clock(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    fn evict_until_within_capacity(&mut self) {
        while self.entries.len() > self.config.capacity {
            let before = self.entries.len();
            let _ = self.evict_lru();
            if self.entries.len() == before {
                break;
            }
        }
    }

    fn validate_entry(&self, entry: &PrefixEntry) {
        assert_eq!(
            entry.prefix_page_size, self.config.prefix_page_size,
            "prefix entry page size must match prefix cache config"
        );

        let expected_pages =
            PrefixEntry::pages_for_tokens(entry.prefix_len, entry.prefix_page_size);
        assert_eq!(
            entry.pages.len(),
            expected_pages,
            "prefix entry page count does not match prefix length"
        );

        let expected_len = self.config.floats_per_prefix_page();
        for page in &entry.pages {
            assert_eq!(
                page.len(),
                expected_len,
                "prefix page length does not match prefix cache geometry"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(capacity: usize) -> PrefixPageCacheConfig {
        PrefixPageCacheConfig {
            capacity,
            prefix_page_size: 4,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
        }
    }

    fn make_page(config: PrefixPageCacheConfig, marker: f32) -> SharedPageRef {
        SharedPageRef::from_vec(vec![marker; config.floats_per_prefix_page()])
    }

    fn make_pages(
        config: PrefixPageCacheConfig,
        prefix_len: usize,
        marker: f32,
    ) -> Vec<SharedPageRef> {
        let count = PrefixEntry::pages_for_tokens(prefix_len, config.prefix_page_size);
        (0..count)
            .map(|idx| make_page(config, marker + idx as f32))
            .collect()
    }

    #[test]
    fn test_empty_cache_lookup() {
        let config = make_config(4);
        let mut cache = PrefixPageCache::new(config);
        let key = PrefixKey::from_token_ids(AdapterId::BASE, &[1, 2, 3]);

        assert!(cache.lookup(&key).is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_prefix_lookup_hit() {
        let config = make_config(4);
        let mut cache = PrefixPageCache::new(config);
        let key = PrefixKey::from_token_ids(AdapterId::BASE, &[1, 2, 3, 4]);
        let pages = make_pages(config, 4, 7.0);

        cache.insert(key, 4, pages);
        let entry = cache.lookup(&key).expect("prefix should hit");

        assert_eq!(entry.prefix_len, 4);
        assert_eq!(entry.pages.len(), 1);
        assert_eq!(entry.pages[0].as_slice()[0], 7.0);
        assert!(entry.last_used > 0);
    }

    #[test]
    fn test_prefix_cache_miss() {
        let config = make_config(4);
        let mut cache = PrefixPageCache::new(config);
        let key = PrefixKey::from_token_ids(AdapterId::BASE, &[99, 100]);

        assert!(cache.lookup(&key).is_none());
    }

    #[test]
    fn test_evict_lru_reclaims_unreferenced() {
        let config = make_config(2);
        let mut cache = PrefixPageCache::new(config);
        let key_a = PrefixKey::from_token_ids(AdapterId::BASE, &[1, 2, 3, 4]);
        let key_b = PrefixKey::from_token_ids(AdapterId::BASE, &[5, 6, 7, 8]);

        cache.insert(key_a, 4, make_pages(config, 4, 1.0));
        cache.insert(key_b, 4, make_pages(config, 4, 2.0));

        let _ = cache
            .lookup(&key_a)
            .expect("key_a should hit and become MRU");
        let pages_freed = cache.evict_lru();

        assert_eq!(pages_freed, 1);
        assert!(cache.contains_key(&key_a));
        assert!(!cache.contains_key(&key_b));
    }

    #[test]
    fn test_adapter_keying_separates_entries() {
        let config = make_config(4);
        let mut cache = PrefixPageCache::new(config);
        let tokens = [1, 2, 3, 4];
        let base_key = PrefixKey::from_token_ids(AdapterId::BASE, &tokens);
        let adapter_key = PrefixKey::from_token_ids(AdapterId::new(42), &tokens);

        cache.insert(base_key, 4, make_pages(config, 4, 1.0));

        assert!(cache.lookup(&base_key).is_some());
        assert!(cache.lookup(&adapter_key).is_none());
    }

    #[test]
    fn test_insert_at_capacity() {
        let config = make_config(1);
        let mut cache = PrefixPageCache::new(config);
        let key_a = PrefixKey::from_token_ids(AdapterId::BASE, &[1, 2, 3, 4]);
        let key_b = PrefixKey::from_token_ids(AdapterId::BASE, &[9, 8, 7, 6]);

        cache.insert(key_a, 4, make_pages(config, 4, 1.0));
        cache.insert(key_b, 4, make_pages(config, 4, 2.0));

        assert_eq!(cache.len(), 1);
        assert!(!cache.contains_key(&key_a));
        assert!(cache.contains_key(&key_b));
    }
}
