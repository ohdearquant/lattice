//! Paged KV cache for multi-model serving.
//!
//! Inspired by vLLM PagedAttention (Kwon et al., 2023), simplified for
//! Apple Silicon unified memory (no GPU-CPU page copies needed).
//!
//! Architecture:
//! - `PagePool`: Fixed-size slab allocator for pages. Each page holds
//!   `page_size` tokens worth of K and V data for ALL layers.
//! - `PageTable`: Logical token position -> (page_index, offset) mapping.
//! - `PagedKVCache`: Combines pool + table with eviction policy.
//!
//! Page layout per page:
//!   `[num_layers, 2 (K+V), page_size, kv_dim]`
//!   Stored as a flat `Vec<f32>` of size `num_layers * 2 * page_size * kv_dim`.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// **Unstable**: paged KV cache configuration; vLLM-inspired, under active design.
#[derive(Debug, Clone)]
pub struct PagedKVCacheConfig {
    /// Tokens per page.
    pub page_size: usize,
    /// Maximum number of pages in the pool.
    pub max_pages: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Eviction policy.
    pub eviction: EvictionPolicy,
}

impl PagedKVCacheConfig {
    /// **Unstable**: KV dimension per token.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Floats per page: stores K and V for all layers across page_size tokens.
    #[inline]
    fn floats_per_page(&self) -> usize {
        self.num_layers * 2 * self.page_size * self.kv_dim()
    }

    /// **Unstable**: memory footprint of a single page.
    pub fn bytes_per_page(&self) -> usize {
        self.floats_per_page() * std::mem::size_of::<f32>()
    }

    /// **Unstable**: total memory budget across all pages.
    pub fn total_bytes(&self) -> usize {
        self.max_pages * self.bytes_per_page()
    }

    /// **Unstable**: maximum token capacity.
    pub fn max_tokens(&self) -> usize {
        self.max_pages * self.page_size
    }
}

/// **Unstable**: eviction strategy for the paged KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Panic when out of pages (fail-fast for debugging).
    None,
    /// Evict least-recently-used page.
    Lru,
}

// ---------------------------------------------------------------------------
// Page Pool
// ---------------------------------------------------------------------------

/// **Unstable**: fixed-size slab allocator for KV cache pages.
///
/// All pages are pre-allocated in a single contiguous buffer.
/// Free pages are tracked via a free list (LIFO for cache locality).
#[derive(Debug)]
pub struct PagePool {
    /// Flat storage: `[max_pages * floats_per_page]`.
    data: Vec<f32>,
    /// Free page indices (LIFO stack).
    free_list: Vec<usize>,
    /// Number of pages.
    max_pages: usize,
    /// Floats per page.
    floats_per_page: usize,
}

impl PagePool {
    /// **Unstable**: create a pre-allocated page pool.
    pub fn new(max_pages: usize, floats_per_page: usize) -> Self {
        let data = vec![0.0f32; max_pages * floats_per_page];
        let free_list: Vec<usize> = (0..max_pages).rev().collect();
        Self {
            data,
            free_list,
            max_pages,
            floats_per_page,
        }
    }

    /// **Unstable**: allocate a page index; returns `None` when exhausted.
    pub fn alloc(&mut self) -> Option<usize> {
        self.free_list.pop()
    }

    /// **Unstable**: return a page to the free list; caller must zero if needed.
    pub fn free(&mut self, page_idx: usize) {
        debug_assert!(page_idx < self.max_pages);
        self.free_list.push(page_idx);
    }

    /// **Unstable**: number of currently free pages.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// **Unstable**: number of currently allocated pages.
    pub fn allocated_count(&self) -> usize {
        self.max_pages - self.free_list.len()
    }

    /// **Unstable**: read raw page data by physical index.
    #[inline]
    pub fn page_data(&self, page_idx: usize) -> &[f32] {
        let start = page_idx * self.floats_per_page;
        &self.data[start..start + self.floats_per_page]
    }

    /// **Unstable**: mutable raw page data by physical index.
    #[inline]
    pub fn page_data_mut(&mut self, page_idx: usize) -> &mut [f32] {
        let start = page_idx * self.floats_per_page;
        &mut self.data[start..start + self.floats_per_page]
    }
}

// ---------------------------------------------------------------------------
// Page Table
// ---------------------------------------------------------------------------

/// **Unstable**: logical-to-physical page mapping for one sequence.
///
/// A sequence's tokens are organized as: logical page 0 covers tokens [0, page_size),
/// logical page 1 covers [page_size, 2*page_size), etc.
#[derive(Debug, Clone)]
pub struct PageTable {
    /// Mapping from logical page index to physical page index.
    entries: Vec<usize>,
    /// Number of valid tokens across all pages.
    seq_len: usize,
    /// Tokens per page.
    page_size: usize,
}

impl PageTable {
    /// **Unstable**: create an empty page table.
    pub fn new(page_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            seq_len: 0,
            page_size,
        }
    }

    /// **Unstable**: number of logical pages allocated.
    pub fn num_pages(&self) -> usize {
        self.entries.len()
    }

    /// **Unstable**: current sequence length tracked by this table.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// **Unstable**: resolve a token position to (physical_page, offset).
    #[inline]
    pub fn resolve(&self, token_pos: usize) -> (usize, usize) {
        let logical = token_pos / self.page_size;
        let offset = token_pos % self.page_size;
        debug_assert!(logical < self.entries.len(), "token_pos out of range");
        (self.entries[logical], offset)
    }

    /// **Unstable**: append a physical page mapping.
    pub fn push_page(&mut self, physical_idx: usize) {
        self.entries.push(physical_idx);
    }

    /// **Unstable**: remove and return the last physical page index.
    pub fn pop_page(&mut self) -> Option<usize> {
        let phys = self.entries.pop()?;
        // Clamp seq_len to the remaining capacity.
        let max = self.entries.len() * self.page_size;
        if self.seq_len > max {
            self.seq_len = max;
        }
        Some(phys)
    }

    /// **Unstable**: remove and return the FIRST (oldest) physical page index.
    ///
    /// Used by LRU eviction: the oldest physical page always corresponds to the
    /// first logical page for a single-sequence cache.
    pub fn pop_front_page(&mut self) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }
        let phys = self.entries.remove(0);
        let max = self.entries.len() * self.page_size;
        if self.seq_len > max {
            self.seq_len = max;
        }
        Some(phys)
    }

    /// **Unstable**: set sequence length counter.
    pub fn set_seq_len(&mut self, len: usize) {
        debug_assert!(len <= self.entries.len() * self.page_size);
        self.seq_len = len;
    }

    /// **Unstable**: physical page indices slice; used by `PagedKVCache::reset`.
    pub fn physical_pages(&self) -> &[usize] {
        &self.entries
    }

    /// **Unstable**: clear all page mappings; does NOT free pages from pool.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.seq_len = 0;
    }
}

// ---------------------------------------------------------------------------
// Paged KV Cache
// ---------------------------------------------------------------------------

/// **Unstable**: paged KV cache with on-demand page allocation and optional LRU eviction.
///
/// The cache owns a `PagePool` and manages one `PageTable` per sequence.
/// For single-sequence use, there is one page table. Multi-sequence support
/// can be added by managing multiple page tables externally.
#[derive(Debug)]
pub struct PagedKVCache {
    pool: PagePool,
    table: PageTable,
    config: PagedKVCacheConfig,
    /// LRU order: front = least recently used, back = most recently used.
    /// Contains physical page indices.
    lru_order: VecDeque<usize>,
}

impl PagedKVCache {
    /// **Unstable**: create a new paged KV cache.
    pub fn new(config: PagedKVCacheConfig) -> Self {
        let fpp = config.floats_per_page();
        let pool = PagePool::new(config.max_pages, fpp);
        let table = PageTable::new(config.page_size);
        Self {
            pool,
            table,
            config,
            lru_order: VecDeque::new(),
        }
    }

    /// **Unstable**: current sequence length.
    pub fn seq_len(&self) -> usize {
        self.table.seq_len()
    }

    /// **Unstable**: maximum tokens this cache can hold.
    pub fn max_tokens(&self) -> usize {
        self.config.max_tokens()
    }

    /// **Unstable**: number of pages currently allocated for this sequence.
    pub fn num_pages(&self) -> usize {
        self.table.num_pages()
    }

    /// **Unstable**: number of free pages in the pool.
    pub fn free_pages(&self) -> usize {
        self.pool.free_count()
    }

    /// **Unstable**: append a single token's K and V for a specific layer.
    ///
    /// Automatically allocates new pages as needed.
    pub fn append_kv_layer(&mut self, layer: usize, k_token: &[f32], v_token: &[f32]) {
        let kv_dim = self.config.kv_dim();
        assert_eq!(k_token.len(), kv_dim);
        assert_eq!(v_token.len(), kv_dim);
        assert!(layer < self.config.num_layers);

        let pos = self.table.seq_len();
        let page_size = self.config.page_size;

        // Check if we need a new page.
        let needed_pages = (pos / page_size) + 1;
        while self.table.num_pages() < needed_pages {
            let phys = self.alloc_page();
            self.table.push_page(phys);
        }

        let (phys_page, offset) = self.table.resolve(pos);

        // Update LRU.
        self.touch_page(phys_page);

        // Compute offset into page data.
        // Page layout: [num_layers, 2, page_size, kv_dim]
        let page_data = self.pool.page_data_mut(phys_page);
        let layer_stride = 2 * page_size * kv_dim;
        let k_base = layer * layer_stride + offset * kv_dim;
        let v_base = layer * layer_stride + page_size * kv_dim + offset * kv_dim;

        page_data[k_base..k_base + kv_dim].copy_from_slice(k_token);
        page_data[v_base..v_base + kv_dim].copy_from_slice(v_token);
    }

    /// **Unstable**: advance sequence length by 1 (call after appending to all layers).
    pub fn advance(&mut self) {
        let new_len = self.table.seq_len() + 1;
        self.table.set_seq_len(new_len);
    }

    /// **Unstable**: read K values for a given layer across the full sequence.
    ///
    /// Writes into `dst` which must have length `seq_len * kv_dim`.
    pub fn gather_k(&self, layer: usize, dst: &mut [f32]) {
        let seq_len = self.table.seq_len();
        let kv_dim = self.config.kv_dim();
        let page_size = self.config.page_size;
        assert_eq!(dst.len(), seq_len * kv_dim);

        let layer_stride = 2 * page_size * kv_dim;
        let mut pos = 0usize;
        while pos < seq_len {
            let (phys_page, offset) = self.table.resolve(pos);
            let run_len = (page_size - offset).min(seq_len - pos);
            let len = run_len * kv_dim;
            let page_data = self.pool.page_data(phys_page);
            let src_base = layer * layer_stride + offset * kv_dim;
            let dst_base = pos * kv_dim;
            dst[dst_base..dst_base + len].copy_from_slice(&page_data[src_base..src_base + len]);
            pos += run_len;
        }
    }

    /// **Unstable**: read V values for a given layer across the full sequence.
    pub fn gather_v(&self, layer: usize, dst: &mut [f32]) {
        let seq_len = self.table.seq_len();
        let kv_dim = self.config.kv_dim();
        let page_size = self.config.page_size;
        assert_eq!(dst.len(), seq_len * kv_dim);

        let layer_stride = 2 * page_size * kv_dim;
        let mut pos = 0usize;
        while pos < seq_len {
            let (phys_page, offset) = self.table.resolve(pos);
            let run_len = (page_size - offset).min(seq_len - pos);
            let len = run_len * kv_dim;
            let page_data = self.pool.page_data(phys_page);
            let src_base = layer * layer_stride + page_size * kv_dim + offset * kv_dim;
            let dst_base = pos * kv_dim;
            dst[dst_base..dst_base + len].copy_from_slice(&page_data[src_base..src_base + len]);
            pos += run_len;
        }
    }

    /// **Unstable**: free all pages and clear the page table.
    pub fn reset(&mut self) {
        for &phys in self.table.physical_pages() {
            self.pool.free(phys);
        }
        self.table.clear();
        self.lru_order.clear();
    }

    /// **Unstable**: total memory usage in bytes (pool capacity, not just allocated).
    pub fn total_memory_bytes(&self) -> usize {
        self.config.total_bytes()
    }

    /// **Unstable**: memory used by allocated pages in bytes.
    pub fn used_memory_bytes(&self) -> usize {
        self.pool.allocated_count() * self.config.bytes_per_page()
    }

    // --- Internal ---

    fn alloc_page(&mut self) -> usize {
        if let Some(phys) = self.pool.alloc() {
            return phys;
        }

        match self.config.eviction {
            EvictionPolicy::None => {
                panic!(
                    "PagePool exhausted ({} pages allocated, eviction=None)",
                    self.pool.allocated_count()
                );
            }
            EvictionPolicy::Lru => self.evict_lru(),
        }
    }

    fn evict_lru(&mut self) -> usize {
        // FP-049: evict the LRU (oldest) physical page. For single-sequence caches
        // the LRU front always corresponds to logical page 0 (entries[0]), so
        // pop_front_page() removes the correct entry from the page table.
        let evicted = self
            .lru_order
            .pop_front()
            .expect("LRU order empty but pool exhausted");

        let removed = self
            .table
            .pop_front_page()
            .expect("invariant: page table has an LRU page when pool is exhausted");
        debug_assert_eq!(removed, evicted);

        // Zero the page data before reuse for safety.
        let fpp = self.config.floats_per_page();
        self.pool.page_data_mut(evicted)[..fpp].fill(0.0);
        evicted
    }

    fn touch_page(&mut self, phys: usize) {
        // Move to back of LRU order.
        if let Some(pos) = self.lru_order.iter().position(|&p| p == phys) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_back(phys);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(max_pages: usize) -> PagedKVCacheConfig {
        PagedKVCacheConfig {
            page_size: 4, // Small pages for testing.
            max_pages,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            eviction: EvictionPolicy::None,
        }
    }

    #[test]
    fn paged_append_gather_roundtrip() {
        let config = make_config(4);
        let kv_dim = config.kv_dim(); // 2 * 4 = 8
        let mut cache = PagedKVCache::new(config);

        // Append 3 tokens.
        for step in 0..3u32 {
            for layer in 0..2 {
                let marker = (step * 10 + layer as u32) as f32;
                let k = vec![marker; kv_dim];
                let v = vec![marker + 0.5; kv_dim];
                cache.append_kv_layer(layer, &k, &v);
            }
            cache.advance();
        }
        assert_eq!(cache.seq_len(), 3);

        // Gather and verify.
        let mut k_buf = vec![0.0f32; 3 * kv_dim];
        let mut v_buf = vec![0.0f32; 3 * kv_dim];

        cache.gather_k(0, &mut k_buf);
        cache.gather_v(0, &mut v_buf);

        // Token 0, layer 0: marker = 0.0
        assert_eq!(k_buf[0], 0.0);
        assert_eq!(v_buf[0], 0.5);
        // Token 1, layer 0: marker = 10.0
        assert_eq!(k_buf[kv_dim], 10.0);
        assert_eq!(v_buf[kv_dim], 10.5);
        // Token 2, layer 0: marker = 20.0
        assert_eq!(k_buf[2 * kv_dim], 20.0);
        assert_eq!(v_buf[2 * kv_dim], 20.5);

        // Check layer 1.
        cache.gather_k(1, &mut k_buf);
        cache.gather_v(1, &mut v_buf);
        assert_eq!(k_buf[0], 1.0); // marker = 0*10 + 1 = 1
        assert_eq!(v_buf[0], 1.5);
    }

    #[test]
    fn paged_page_allocation() {
        let config = make_config(8);
        let kv_dim = config.kv_dim();
        let mut cache = PagedKVCache::new(config);

        // page_size=4, so first page covers tokens 0-3.
        assert_eq!(cache.num_pages(), 0);
        assert_eq!(cache.free_pages(), 8);

        // Append 1 token -> allocates 1 page.
        let k = vec![1.0; kv_dim];
        let v = vec![2.0; kv_dim];
        for layer in 0..2 {
            cache.append_kv_layer(layer, &k, &v);
        }
        cache.advance();
        assert_eq!(cache.num_pages(), 1);
        assert_eq!(cache.free_pages(), 7);

        // Append 3 more -> still 1 page.
        for _ in 0..3 {
            for layer in 0..2 {
                cache.append_kv_layer(layer, &k, &v);
            }
            cache.advance();
        }
        assert_eq!(cache.num_pages(), 1);
        assert_eq!(cache.seq_len(), 4);

        // Append 1 more -> needs page 2.
        for layer in 0..2 {
            cache.append_kv_layer(layer, &k, &v);
        }
        cache.advance();
        assert_eq!(cache.num_pages(), 2);
        assert_eq!(cache.free_pages(), 6);
    }

    #[test]
    fn paged_reset() {
        let config = make_config(4);
        let kv_dim = config.kv_dim();
        let mut cache = PagedKVCache::new(config);

        let k = vec![1.0; kv_dim];
        let v = vec![2.0; kv_dim];
        for _ in 0..6 {
            for layer in 0..2 {
                cache.append_kv_layer(layer, &k, &v);
            }
            cache.advance();
        }
        assert!(cache.num_pages() > 0);

        cache.reset();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.num_pages(), 0);
        assert_eq!(cache.free_pages(), 4);
    }

    #[test]
    fn paged_cross_page_boundary() {
        // Verify data integrity when tokens span multiple pages.
        let config = make_config(4);
        let kv_dim = config.kv_dim(); // 8
        let mut cache = PagedKVCache::new(config);

        // Append 6 tokens (spans 2 pages at page_size=4).
        for step in 0..6u32 {
            for layer in 0..2 {
                let k: Vec<f32> = (0..kv_dim)
                    .map(|i| step as f32 * 100.0 + i as f32)
                    .collect();
                let v: Vec<f32> = (0..kv_dim)
                    .map(|i| step as f32 * 100.0 + i as f32 + 0.5)
                    .collect();
                cache.append_kv_layer(layer, &k, &v);
            }
            cache.advance();
        }
        assert_eq!(cache.num_pages(), 2);

        // Verify all 6 tokens K and V, layer 0.
        let mut k_buf = vec![0.0f32; 6 * kv_dim];
        let mut v_buf = vec![0.0f32; 6 * kv_dim];
        cache.gather_k(0, &mut k_buf);
        cache.gather_v(0, &mut v_buf);
        for step in 0..6u32 {
            for i in 0..kv_dim {
                let k_expected = step as f32 * 100.0 + i as f32;
                let v_expected = k_expected + 0.5;
                let k_got = k_buf[step as usize * kv_dim + i];
                let v_got = v_buf[step as usize * kv_dim + i];
                assert!(
                    (k_got - k_expected).abs() < 1e-6,
                    "K step={step}, i={i}: expected {k_expected}, got {k_got}"
                );
                assert!(
                    (v_got - v_expected).abs() < 1e-6,
                    "V step={step}, i={i}: expected {v_expected}, got {v_got}"
                );
            }
        }
    }

    #[test]
    fn paged_non_multiple_gather_kv() {
        // seq_len = page_size * 2 + 3: exercises partial final page for both K and V.
        let config = make_config(8);
        let kv_dim = config.kv_dim(); // 8
        let page_size = config.page_size; // 4
        let seq_len = page_size * 2 + 3; // 11
        let mut cache = PagedKVCache::new(config);

        for step in 0..seq_len as u32 {
            for layer in 0..2 {
                let k: Vec<f32> = (0..kv_dim)
                    .map(|i| step as f32 * 10.0 + layer as f32 + i as f32 * 0.1)
                    .collect();
                let v: Vec<f32> = k.iter().map(|&x| x + 0.5).collect();
                cache.append_kv_layer(layer, &k, &v);
            }
            cache.advance();
        }
        assert_eq!(cache.seq_len(), seq_len);

        let mut k_buf = vec![0.0f32; seq_len * kv_dim];
        let mut v_buf = vec![0.0f32; seq_len * kv_dim];
        cache.gather_k(0, &mut k_buf);
        cache.gather_v(0, &mut v_buf);

        for step in 0..seq_len as u32 {
            for i in 0..kv_dim {
                let k_expected = step as f32 * 10.0 + 0.0 + i as f32 * 0.1;
                let v_expected = k_expected + 0.5;
                let k_got = k_buf[step as usize * kv_dim + i];
                let v_got = v_buf[step as usize * kv_dim + i];
                assert!(
                    (k_got - k_expected).abs() < 1e-5,
                    "K step={step}, i={i}: expected {k_expected}, got {k_got}"
                );
                assert!(
                    (v_got - v_expected).abs() < 1e-5,
                    "V step={step}, i={i}: expected {v_expected}, got {v_got}"
                );
            }
        }
    }

    #[test]
    fn paged_memory_accounting() {
        let config = PagedKVCacheConfig {
            page_size: 256,
            max_pages: 16,
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            eviction: EvictionPolicy::None,
        };

        // Each page: 28 layers * 2 * 256 tokens * 1024 kv_dim * 4 bytes
        let expected_per_page = 28 * 2 * 256 * 1024 * 4;
        assert_eq!(config.bytes_per_page(), expected_per_page);

        let cache = PagedKVCache::new(config);
        assert_eq!(cache.total_memory_bytes(), 16 * expected_per_page);
        assert_eq!(cache.used_memory_bytes(), 0);
    }

    #[test]
    #[should_panic(expected = "PagePool exhausted")]
    fn paged_no_eviction_panics_on_exhaustion() {
        let config = make_config(1); // Only 1 page, page_size=4.
        let kv_dim = config.kv_dim();
        let mut cache = PagedKVCache::new(config);

        let k = vec![1.0; kv_dim];
        let v = vec![2.0; kv_dim];

        // Fill the single page (4 tokens).
        for _ in 0..4 {
            for layer in 0..2 {
                cache.append_kv_layer(layer, &k, &v);
            }
            cache.advance();
        }

        // 5th token needs a new page -> should panic.
        for layer in 0..2 {
            cache.append_kv_layer(layer, &k, &v);
        }
    }

    #[test]
    fn page_pool_alloc_free_cycle() {
        let mut pool = PagePool::new(4, 16);
        assert_eq!(pool.free_count(), 4);

        let p0 = pool.alloc().unwrap();
        let _p1 = pool.alloc().unwrap();
        assert_eq!(pool.free_count(), 2);
        assert_eq!(pool.allocated_count(), 2);

        pool.free(p0);
        assert_eq!(pool.free_count(), 3);

        // Re-allocate should give back p0 (LIFO).
        let p2 = pool.alloc().unwrap();
        assert_eq!(p2, p0);
    }

    #[test]
    fn page_table_resolve() {
        let mut table = PageTable::new(4);
        table.push_page(10); // logical 0 -> physical 10
        table.push_page(5); // logical 1 -> physical 5
        table.set_seq_len(6);

        // Token 0 -> page 10, offset 0
        assert_eq!(table.resolve(0), (10, 0));
        // Token 3 -> page 10, offset 3
        assert_eq!(table.resolve(3), (10, 3));
        // Token 4 -> page 5, offset 0
        assert_eq!(table.resolve(4), (5, 0));
        // Token 5 -> page 5, offset 1
        assert_eq!(table.resolve(5), (5, 1));
    }
}
