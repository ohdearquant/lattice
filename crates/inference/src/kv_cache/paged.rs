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
//!   Stored as either f32 values or symmetric Q8 values with one scale per
//!   K/V token vector.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[cfg(test)]
use super::prefix::PrefixPageCacheConfig;
use super::prefix::{AdapterId, PrefixEntry, PrefixKey, PrefixPageCache, SharedPageRef};
use crate::error::InferenceError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// **Unstable**: element format used by paged KV storage.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CacheType {
    /// Store every KV element as an f32 value.
    #[default]
    F32,
    /// Store each KV token vector as symmetric int8 values with one f32 scale.
    /// The scale is `max(abs(vector)) / 127`, falling back to `max(abs(vector))`
    /// when that quotient underflows to zero. Zero vectors use scale `1`.
    Q8,
}

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

    pub(crate) fn try_kv_dim(&self) -> Result<usize, InferenceError> {
        self.num_kv_heads.checked_mul(self.head_dim).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "num_kv_heads ({}) * head_dim ({}) overflows usize",
                self.num_kv_heads, self.head_dim
            ))
        })
    }

    /// Floats per page: stores K and V for all layers across page_size tokens.
    #[inline]
    fn floats_per_page(&self) -> usize {
        self.num_layers * 2 * self.page_size * self.kv_dim()
    }

    pub(crate) fn try_floats_per_page(&self) -> Result<usize, InferenceError> {
        let kv_dim = self.try_kv_dim()?;
        self.num_layers
            .checked_mul(2)
            .and_then(|n| n.checked_mul(self.page_size))
            .and_then(|n| n.checked_mul(kv_dim))
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "num_layers ({}) * 2 * page_size ({}) * kv_dim ({kv_dim}) overflows usize",
                    self.num_layers, self.page_size
                ))
            })
    }

    #[inline]
    fn q8_scales_per_page(&self) -> usize {
        self.num_layers * 2 * self.page_size
    }

    fn try_q8_scales_per_page(&self) -> Result<usize, InferenceError> {
        self.num_layers
            .checked_mul(2)
            .and_then(|n| n.checked_mul(self.page_size))
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "num_layers ({}) * 2 * page_size ({}) overflows usize",
                    self.num_layers, self.page_size
                ))
            })
    }

    /// **Unstable**: memory footprint of a single page.
    pub fn bytes_per_page(&self) -> usize {
        self.floats_per_page() * std::mem::size_of::<f32>()
    }

    pub fn try_bytes_per_page(&self) -> Result<usize, InferenceError> {
        let floats_per_page = self.try_floats_per_page()?;
        floats_per_page
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "floats_per_page ({floats_per_page}) * size_of::<f32>() ({}) overflows usize",
                    std::mem::size_of::<f32>()
                ))
            })
    }

    /// **Unstable**: total memory budget across all pages.
    pub fn total_bytes(&self) -> usize {
        self.max_pages * self.bytes_per_page()
    }

    pub fn try_total_bytes(&self) -> Result<usize, InferenceError> {
        let bytes_per_page = self.try_bytes_per_page()?;
        self.max_pages.checked_mul(bytes_per_page).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "max_pages ({}) * bytes_per_page ({bytes_per_page}) overflows usize",
                self.max_pages
            ))
        })
    }

    /// **Unstable**: maximum token capacity.
    pub fn max_tokens(&self) -> usize {
        self.max_pages * self.page_size
    }

    pub fn try_max_tokens(&self) -> Result<usize, InferenceError> {
        self.max_pages.checked_mul(self.page_size).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "max_pages ({}) * page_size ({}) overflows usize",
                self.max_pages, self.page_size
            ))
        })
    }
}

/// **Unstable**: eviction strategy for the paged KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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

    /// **Unstable**: create a pre-allocated page pool, returning an error on
    /// overflow rather than panicking or producing a silently-wrong capacity.
    pub fn try_new(max_pages: usize, floats_per_page: usize) -> Result<Self, InferenceError> {
        let len = max_pages.checked_mul(floats_per_page).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "max_pages ({max_pages}) * floats_per_page ({floats_per_page}) overflows usize"
            ))
        })?;
        // `vec![0.0f32; len]` allocates `len * size_of::<f32>()` bytes; a length
        // that fits usize can still overflow the byte layout. Guard it here so
        // the fallible path fails closed instead of panicking ("capacity
        // overflow") inside `vec!`.
        let byte_len = len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "page pool byte size ({len} * {}) overflows usize",
                std::mem::size_of::<f32>()
            ))
        })?;
        // Rust Vec panics when the byte allocation exceeds `isize::MAX`; a usize
        // product that fits can still cross this tighter bound.
        if byte_len > isize::MAX as usize {
            return Err(InferenceError::InvalidInput(format!(
                "page pool byte size ({byte_len}) exceeds isize::MAX — allocation would panic"
            )));
        }
        // `free_list` is a sibling allocation drawn from the same `max_pages`
        // cardinality as `data`, but sized in `usize` elements (8 bytes each)
        // rather than floats. When `floats_per_page` is 0, `len`/`byte_len`
        // above are 0 and pass trivially even for a huge `max_pages`, so this
        // allocation needs its own byte-layout guard to fail closed instead of
        // panicking ("capacity overflow") inside `.collect()`.
        let free_list_bytes = max_pages
            .checked_mul(std::mem::size_of::<usize>())
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "max_pages ({max_pages}) * size_of::<usize>() overflows usize"
                ))
            })?;
        if free_list_bytes > isize::MAX as usize {
            return Err(InferenceError::InvalidInput(format!(
                "free-list allocation ({free_list_bytes} bytes) exceeds isize::MAX — allocation would panic"
            )));
        }
        let data = vec![0.0f32; len];
        let free_list: Vec<usize> = (0..max_pages).rev().collect();
        Ok(Self {
            data,
            free_list,
            max_pages,
            floats_per_page,
        })
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

#[derive(Debug)]
struct Q8PagePool {
    data: Vec<i8>,
    scales: Vec<f32>,
    free_list: Vec<usize>,
    max_pages: usize,
    values_per_page: usize,
    scales_per_page: usize,
}

impl Q8PagePool {
    fn new(max_pages: usize, values_per_page: usize, scales_per_page: usize) -> Self {
        Self {
            data: vec![0; max_pages * values_per_page],
            scales: vec![1.0; max_pages * scales_per_page],
            free_list: (0..max_pages).rev().collect(),
            max_pages,
            values_per_page,
            scales_per_page,
        }
    }

    fn try_new(
        max_pages: usize,
        values_per_page: usize,
        scales_per_page: usize,
    ) -> Result<Self, InferenceError> {
        let data_len = max_pages.checked_mul(values_per_page).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "max_pages ({max_pages}) * Q8 values_per_page ({values_per_page}) overflows usize"
            ))
        })?;
        if data_len > isize::MAX as usize {
            return Err(InferenceError::InvalidInput(format!(
                "Q8 page data allocation ({data_len} bytes) exceeds isize::MAX"
            )));
        }

        let scales_len = max_pages.checked_mul(scales_per_page).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "max_pages ({max_pages}) * Q8 scales_per_page ({scales_per_page}) overflows usize"
            ))
        })?;
        let scales_bytes = scales_len
            .checked_mul(std::mem::size_of::<f32>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "Q8 page scale allocation ({scales_len} * {}) exceeds isize::MAX",
                    std::mem::size_of::<f32>()
                ))
            })?;
        let _ = scales_bytes;

        let free_list_bytes = max_pages
            .checked_mul(std::mem::size_of::<usize>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "Q8 free-list allocation ({max_pages} * {}) exceeds isize::MAX",
                    std::mem::size_of::<usize>()
                ))
            })?;
        let _ = free_list_bytes;

        // The two allocation checks above both pass when max_pages is zero, so
        // neither constrains the per-page geometry. `bytes_per_page` and
        // `total_memory_bytes` still evaluate that geometry after construction,
        // and this is the fallible constructor whose documented contract is to
        // return InvalidInput rather than let arithmetic overflow. The F32 path
        // gets the equivalent guarantee from `try_total_bytes`.
        let scale_bytes_per_page = scales_per_page
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "Q8 scales_per_page ({scales_per_page}) * {} overflows usize",
                    std::mem::size_of::<f32>()
                ))
            })?;
        let bytes_per_page = values_per_page
            .checked_add(scale_bytes_per_page)
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "Q8 values_per_page ({values_per_page}) + scale bytes \
                     ({scale_bytes_per_page}) overflows usize"
                ))
            })?;
        max_pages.checked_mul(bytes_per_page).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "max_pages ({max_pages}) * Q8 bytes_per_page ({bytes_per_page}) overflows usize"
            ))
        })?;

        Ok(Self {
            data: vec![0; data_len],
            scales: vec![1.0; scales_len],
            free_list: (0..max_pages).rev().collect(),
            max_pages,
            values_per_page,
            scales_per_page,
        })
    }

    fn alloc(&mut self) -> Option<usize> {
        self.free_list.pop()
    }

    fn free(&mut self, page_idx: usize) {
        debug_assert!(page_idx < self.max_pages);
        self.free_list.push(page_idx);
    }

    fn free_count(&self) -> usize {
        self.free_list.len()
    }

    fn allocated_count(&self) -> usize {
        self.max_pages - self.free_list.len()
    }

    fn store_vector(
        &mut self,
        page_idx: usize,
        value_offset: usize,
        scale_offset: usize,
        values: &[f32],
    ) {
        let abs_max = values.iter().fold(0.0f32, |max, &value| {
            assert!(value.is_finite(), "Q8 KV input must be finite");
            max.max(value.abs())
        });
        // `abs_max / 127.0` rounds to zero once abs_max falls below roughly
        // 1.8e-43, which is representable but 127 times smaller than the
        // smallest representable scale. Dividing by that zero yields infinity,
        // clamps to 127, and reads the whole vector back as zero, so a nonzero
        // input silently vanishes. Falling back to abs_max keeps the scale
        // representable; the vector then quantizes onto {-1, 0, 1} with error
        // bounded by abs_max / 2, which is the same relative accuracy the
        // normal range gets.
        let scale = if abs_max == 0.0 {
            1.0
        } else {
            let scale = abs_max / 127.0;
            if scale == 0.0 { abs_max } else { scale }
        };
        let value_base = page_idx * self.values_per_page + value_offset;
        let scale_idx = page_idx * self.scales_per_page + scale_offset;
        self.scales[scale_idx] = scale;
        for (&value, quantized) in values
            .iter()
            .zip(&mut self.data[value_base..value_base + values.len()])
        {
            *quantized = (value / scale).round().clamp(-127.0, 127.0) as i8;
        }
    }

    fn gather_vector(
        &self,
        page_idx: usize,
        value_offset: usize,
        scale_offset: usize,
        dst: &mut [f32],
    ) {
        let value_base = page_idx * self.values_per_page + value_offset;
        let scale = self.scales[page_idx * self.scales_per_page + scale_offset];
        for (&quantized, value) in self.data[value_base..value_base + dst.len()]
            .iter()
            .zip(dst)
        {
            *value = f32::from(quantized) * scale;
        }
    }

    fn clear_page(&mut self, page_idx: usize) {
        let value_base = page_idx * self.values_per_page;
        self.data[value_base..value_base + self.values_per_page].fill(0);
        let scale_base = page_idx * self.scales_per_page;
        self.scales[scale_base..scale_base + self.scales_per_page].fill(1.0);
    }

    fn bytes_per_page(&self) -> usize {
        self.values_per_page + self.scales_per_page * std::mem::size_of::<f32>()
    }
}

#[derive(Debug)]
enum PagedPagePool {
    F32(PagePool),
    Q8(Q8PagePool),
}

impl PagedPagePool {
    fn cache_type(&self) -> CacheType {
        match self {
            Self::F32(_) => CacheType::F32,
            Self::Q8(_) => CacheType::Q8,
        }
    }

    fn alloc(&mut self) -> Option<usize> {
        match self {
            Self::F32(pool) => pool.alloc(),
            Self::Q8(pool) => pool.alloc(),
        }
    }

    fn free(&mut self, page_idx: usize) {
        match self {
            Self::F32(pool) => pool.free(page_idx),
            Self::Q8(pool) => pool.free(page_idx),
        }
    }

    fn free_count(&self) -> usize {
        match self {
            Self::F32(pool) => pool.free_count(),
            Self::Q8(pool) => pool.free_count(),
        }
    }

    fn allocated_count(&self) -> usize {
        match self {
            Self::F32(pool) => pool.allocated_count(),
            Self::Q8(pool) => pool.allocated_count(),
        }
    }

    fn f32_page(&self, page_idx: usize) -> Result<&[f32], InferenceError> {
        match self {
            Self::F32(pool) => Ok(pool.page_data(page_idx)),
            Self::Q8(_) => Err(InferenceError::PrefixCache(
                "prefix sharing requires f32 paged KV storage".into(),
            )),
        }
    }

    fn f32_page_mut(&mut self, page_idx: usize) -> Result<&mut [f32], InferenceError> {
        match self {
            Self::F32(pool) => Ok(pool.page_data_mut(page_idx)),
            Self::Q8(_) => Err(InferenceError::PrefixCache(
                "prefix sharing requires f32 paged KV storage".into(),
            )),
        }
    }

    fn clear_page(&mut self, page_idx: usize) {
        match self {
            Self::F32(pool) => pool.page_data_mut(page_idx).fill(0.0),
            Self::Q8(pool) => pool.clear_page(page_idx),
        }
    }

    fn bytes_per_page(&self) -> usize {
        match self {
            Self::F32(pool) => pool.floats_per_page * std::mem::size_of::<f32>(),
            Self::Q8(pool) => pool.bytes_per_page(),
        }
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
    ///
    /// # Panics
    ///
    /// Panics if `page_size == 0`: it is the divisor in [`PageTable::resolve`],
    /// so a zero would otherwise surface as a cryptic divide-by-zero. This is
    /// the authoritative guard for every `PageTable` (the `PagedKVCache`
    /// constructors and `SequenceManager::add` both route through here).
    pub fn new(page_size: usize) -> Self {
        assert!(page_size > 0, "PageTable page_size must be non-zero");
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
    pool: PagedPagePool,
    table: PageTable,
    config: PagedKVCacheConfig,
    /// Optional shared prefix cache injected by the caller/session.
    prefix_cache: Option<Arc<Mutex<PrefixPageCache>>>,
    /// LRU order: front = least recently used, back = most recently used.
    /// Contains physical page indices.
    lru_order: VecDeque<usize>,
}

impl PagedKVCache {
    /// **Unstable**: create a new paged KV cache without prefix sharing.
    pub fn new(config: PagedKVCacheConfig) -> Self {
        Self::with_cache_type(config, CacheType::F32)
    }

    /// **Unstable**: fallible constructor — returns `InvalidInput` on overflow
    /// instead of panicking or silently allocating a wrong-sized pool.
    pub fn try_new(config: PagedKVCacheConfig) -> Result<Self, InferenceError> {
        Self::try_with_cache_type(config, CacheType::F32)
    }

    /// **Unstable**: create a paged KV cache using the selected storage format.
    ///
    /// Q8 uses symmetric per-token-vector quantization and does not enable
    /// prefix sharing. Use [`PagedKVCache::with_prefix_cache`] for an f32 cache
    /// with prefix reuse.
    ///
    /// # Panics
    ///
    /// Panics if page geometry overflows or `page_size` is zero. Use
    /// [`PagedKVCache::try_with_cache_type`] for checked geometry.
    pub fn with_cache_type(config: PagedKVCacheConfig, cache_type: CacheType) -> Self {
        match cache_type {
            CacheType::F32 => Self::with_prefix_cache(config, None),
            CacheType::Q8 => {
                assert!(
                    config.page_size > 0,
                    "PagedKVCacheConfig.page_size must be non-zero"
                );
                let values_per_page = config.floats_per_page();
                let scales_per_page = config.q8_scales_per_page();
                let pool = PagedPagePool::Q8(Q8PagePool::new(
                    config.max_pages,
                    values_per_page,
                    scales_per_page,
                ));
                let table = PageTable::new(config.page_size);
                Self {
                    pool,
                    table,
                    config,
                    prefix_cache: None,
                    lru_order: VecDeque::new(),
                }
            }
        }
    }

    /// **Unstable**: fallible storage-format constructor.
    pub fn try_with_cache_type(
        config: PagedKVCacheConfig,
        cache_type: CacheType,
    ) -> Result<Self, InferenceError> {
        match cache_type {
            CacheType::F32 => Self::try_with_prefix_cache(config, None),
            CacheType::Q8 => {
                if config.page_size == 0 {
                    return Err(InferenceError::InvalidInput(
                        "PagedKVCacheConfig.page_size must be non-zero".into(),
                    ));
                }
                let values_per_page = config.try_floats_per_page()?;
                let scales_per_page = config.try_q8_scales_per_page()?;
                let pool = PagedPagePool::Q8(Q8PagePool::try_new(
                    config.max_pages,
                    values_per_page,
                    scales_per_page,
                )?);
                let table = PageTable::new(config.page_size);
                Ok(Self {
                    pool,
                    table,
                    config,
                    prefix_cache: None,
                    lru_order: VecDeque::new(),
                })
            }
        }
    }

    /// **Unstable**: fallible constructor with optional prefix sharing.
    pub fn try_with_prefix_cache(
        config: PagedKVCacheConfig,
        prefix_cache: Option<Arc<Mutex<PrefixPageCache>>>,
    ) -> Result<Self, InferenceError> {
        if config.page_size == 0 {
            return Err(InferenceError::InvalidInput(
                "PagedKVCacheConfig.page_size must be non-zero".into(),
            ));
        }
        let fpp = config.try_floats_per_page()?;
        let _total_bytes = config.try_total_bytes()?;

        // When a prefix cache is present, validate that its page geometry cannot
        // produce a panicking allocation on the hot promote/restore path. Run
        // this before any allocation (including the PagePool below) so an
        // oversized prefix config is rejected before the constructor commits
        // to any allocation at all.
        if let Some(ref cache_arc) = prefix_cache {
            let guard = cache_arc
                .lock()
                .map_err(|_| InferenceError::PrefixCache("prefix cache lock poisoned".into()))?;
            let pc = guard.config();
            let kv_dim = config
                .num_kv_heads
                .checked_mul(config.head_dim)
                .ok_or_else(|| {
                    InferenceError::InvalidInput(format!(
                        "num_kv_heads ({}) * head_dim ({}) overflows usize",
                        config.num_kv_heads, config.head_dim
                    ))
                })?;
            let floats = config
                .num_layers
                .checked_mul(2)
                .and_then(|n| n.checked_mul(pc.prefix_page_size))
                .and_then(|n| n.checked_mul(kv_dim))
                .ok_or_else(|| {
                    InferenceError::InvalidInput(format!(
                        "prefix page float count (num_layers={} * 2 * prefix_page_size={} * kv_dim={kv_dim}) overflows usize",
                        config.num_layers, pc.prefix_page_size
                    ))
                })?;
            // Catch the isize::MAX bound that Rust Vec enforces at allocation.
            let prefix_page_bytes = floats
                .checked_mul(std::mem::size_of::<f32>())
                .filter(|&b| b <= isize::MAX as usize)
                .ok_or_else(|| {
                    InferenceError::InvalidInput(format!(
                        "prefix page byte size ({floats} * {}) exceeds isize::MAX",
                        std::mem::size_of::<f32>()
                    ))
                })?;
            let _ = prefix_page_bytes;
        }

        let pool = PagedPagePool::F32(PagePool::try_new(config.max_pages, fpp)?);
        let table = PageTable::new(config.page_size);
        Ok(Self {
            pool,
            table,
            config,
            prefix_cache,
            lru_order: VecDeque::new(),
        })
    }

    /// **Unstable**: create a new paged KV cache with optional prefix sharing.
    ///
    /// Passing `None` preserves existing behavior. Passing `Some(cache)` enables
    /// `restore_prefix` and `promote_to_prefix` without changing the hot append
    /// and gather paths.
    ///
    /// # Panics
    ///
    /// Panics if `config.page_size == 0`. `page_size` is the divisor in the
    /// logical-page math (`PageTable::resolve`), so a zero would otherwise
    /// surface later as a cryptic divide-by-zero in the hot path. This mirrors
    /// the prefix path, which already rejects a zero `prefix_page_size`.
    pub fn with_prefix_cache(
        config: PagedKVCacheConfig,
        prefix_cache: Option<Arc<Mutex<PrefixPageCache>>>,
    ) -> Self {
        assert!(
            config.page_size > 0,
            "PagedKVCacheConfig.page_size must be non-zero"
        );
        let fpp = config.floats_per_page();
        let pool = PagedPagePool::F32(PagePool::new(config.max_pages, fpp));
        let table = PageTable::new(config.page_size);
        Self {
            pool,
            table,
            config,
            prefix_cache,
            lru_order: VecDeque::new(),
        }
    }

    /// **Unstable**: restore a cached prefix into owned `PagePool` pages.
    ///
    /// Returns `Ok(Some(prefix_len))` on hit and `Ok(None)` when prefix sharing
    /// is disabled or the key is absent. The cache must be pristine
    /// (`seq_len == 0` and `num_pages == 0`) because restore fast-forwards the
    /// sequence before the first append.
    pub fn restore_prefix(
        &mut self,
        adapter_id: AdapterId,
        token_ids: &[u32],
    ) -> Result<Option<usize>, InferenceError> {
        if self.seq_len() != 0 || self.num_pages() != 0 {
            return Err(InferenceError::PrefixCache(
                "restore_prefix requires an empty PagedKVCache".into(),
            ));
        }

        let key = PrefixKey::from_token_ids(adapter_id, token_ids);
        let entry = match &self.prefix_cache {
            Some(prefix_cache) => {
                let mut guard = prefix_cache.lock().map_err(|_| {
                    InferenceError::PrefixCache("prefix cache lock poisoned".into())
                })?;
                guard.lookup(&key)
            }
            None => None,
        };

        let Some(entry) = entry else {
            return Ok(None);
        };

        if entry.prefix_len != token_ids.len() {
            return Err(InferenceError::PrefixCache(format!(
                "prefix hash collision or invalid entry length: key length {}, entry length {}",
                token_ids.len(),
                entry.prefix_len
            )));
        }

        let restored = self.restore_prefix_entry(&entry)?;
        Ok(Some(restored))
    }

    /// **Unstable**: promote the current owned pages into the shared prefix cache.
    ///
    /// Returns `Ok(Some(page_count))` when inserted, `Ok(None)` when prefix
    /// sharing is disabled or the sequence is empty. `token_ids` must exactly
    /// match the current sequence length so the hash key corresponds to the
    /// copied KV pages.
    pub fn promote_to_prefix(
        &mut self,
        adapter_id: AdapterId,
        token_ids: &[u32],
    ) -> Result<Option<usize>, InferenceError> {
        let prefix_len = self.seq_len();
        if prefix_len == 0 {
            return Ok(None);
        }
        if token_ids.len() != prefix_len {
            return Err(InferenceError::PrefixCache(format!(
                "promote_to_prefix token length {} does not match seq_len {}",
                token_ids.len(),
                prefix_len
            )));
        }

        let Some(prefix_cache) = self.prefix_cache.as_ref().cloned() else {
            return Ok(None);
        };

        let prefix_page_size = {
            let guard = prefix_cache
                .lock()
                .map_err(|_| InferenceError::PrefixCache("prefix cache lock poisoned".into()))?;
            guard.config().prefix_page_size
        };

        let pages = self.copy_owned_pages_to_shared(prefix_page_size)?;
        let page_count = pages.len();
        let key = PrefixKey::from_token_ids(adapter_id, token_ids);

        let mut guard = prefix_cache
            .lock()
            .map_err(|_| InferenceError::PrefixCache("prefix cache lock poisoned".into()))?;
        guard.insert(key, prefix_len, pages);
        Ok(Some(page_count))
    }

    fn restore_prefix_entry(&mut self, entry: &PrefixEntry) -> Result<usize, InferenceError> {
        self.validate_prefix_entry(entry)?;

        let live_page_count =
            PrefixEntry::pages_for_tokens(entry.prefix_len, self.config.page_size);
        let mut owned_pages = Vec::with_capacity(live_page_count);

        for _ in 0..live_page_count {
            let Some(phys) = self.pool.alloc() else {
                for allocated in owned_pages {
                    self.pool.free(allocated);
                }
                return Err(InferenceError::PrefixCache(format!(
                    "not enough free pages to restore prefix: needed {}, free {}",
                    live_page_count,
                    self.pool.free_count()
                )));
            };
            self.pool.f32_page_mut(phys)?.fill(0.0);
            owned_pages.push(phys);
        }

        for token_pos in 0..entry.prefix_len {
            let src_page_idx = token_pos / entry.prefix_page_size;
            let src_offset = token_pos % entry.prefix_page_size;
            let dst_page_idx = token_pos / self.config.page_size;
            let dst_offset = token_pos % self.config.page_size;

            let src_page = entry.pages[src_page_idx].as_slice();
            let dst_page = self.pool.f32_page_mut(owned_pages[dst_page_idx])?;
            Self::copy_token_between_page_layouts(
                src_page,
                entry.prefix_page_size,
                src_offset,
                dst_page,
                self.config.page_size,
                dst_offset,
                self.config.num_layers,
                self.config.kv_dim(),
            );
        }

        for phys in owned_pages.iter().copied() {
            self.table.push_page(phys);
            self.touch_page(phys);
        }
        self.table.set_seq_len(entry.prefix_len);
        Ok(entry.prefix_len)
    }

    fn copy_owned_pages_to_shared(
        &self,
        prefix_page_size: usize,
    ) -> Result<Vec<SharedPageRef>, InferenceError> {
        if prefix_page_size == 0 {
            return Err(InferenceError::PrefixCache(
                "prefix_page_size must be non-zero".into(),
            ));
        }

        let prefix_len = self.seq_len();
        let prefix_page_count = PrefixEntry::pages_for_tokens(prefix_len, prefix_page_size);
        let kv_dim = self.config.kv_dim();
        let floats_per_prefix_page = self
            .config
            .num_layers
            .checked_mul(2)
            .and_then(|n| n.checked_mul(prefix_page_size))
            .and_then(|n| n.checked_mul(kv_dim))
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "prefix page float count (num_layers={} * 2 * prefix_page_size={prefix_page_size} * kv_dim={kv_dim}) overflows usize",
                    self.config.num_layers
                ))
            })?;
        // Guard the byte size; a usize-valid count can still exceed isize::MAX
        // and panic inside `vec!`.
        let prefix_page_bytes = floats_per_prefix_page
            .checked_mul(std::mem::size_of::<f32>())
            .filter(|&b| b <= isize::MAX as usize)
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "prefix page byte size ({floats_per_prefix_page} * {}) exceeds isize::MAX",
                    std::mem::size_of::<f32>()
                ))
            })?;
        let _ = prefix_page_bytes; // size validated; the vec! uses float count below
        let mut pages = Vec::with_capacity(prefix_page_count);

        for prefix_page_idx in 0..prefix_page_count {
            let mut page = vec![0.0f32; floats_per_prefix_page];
            let start = prefix_page_idx * prefix_page_size;
            let end = (start + prefix_page_size).min(prefix_len);

            for token_pos in start..end {
                let (src_phys, src_offset) = self.table.resolve(token_pos);
                let src_page = self.pool.f32_page(src_phys)?;
                let dst_offset = token_pos - start;
                Self::copy_token_between_page_layouts(
                    src_page,
                    self.config.page_size,
                    src_offset,
                    &mut page,
                    prefix_page_size,
                    dst_offset,
                    self.config.num_layers,
                    kv_dim,
                );
            }

            pages.push(SharedPageRef::from_vec(page));
        }

        Ok(pages)
    }

    fn validate_prefix_entry(&self, entry: &PrefixEntry) -> Result<(), InferenceError> {
        if entry.prefix_page_size == 0 {
            return Err(InferenceError::PrefixCache(
                "prefix entry page size must be non-zero".into(),
            ));
        }

        let expected_pages =
            PrefixEntry::pages_for_tokens(entry.prefix_len, entry.prefix_page_size);
        if entry.pages.len() != expected_pages {
            return Err(InferenceError::PrefixCache(format!(
                "prefix entry page count {} does not match expected {}",
                entry.pages.len(),
                expected_pages
            )));
        }

        let expected_page_len =
            self.config.num_layers * 2 * entry.prefix_page_size * self.config.kv_dim();
        for page in &entry.pages {
            if page.len() != expected_page_len {
                return Err(InferenceError::PrefixCache(format!(
                    "prefix page has {} floats, expected {}",
                    page.len(),
                    expected_page_len
                )));
            }
        }

        Ok(())
    }

    fn copy_token_between_page_layouts(
        src_page: &[f32],
        src_page_size: usize,
        src_offset: usize,
        dst_page: &mut [f32],
        dst_page_size: usize,
        dst_offset: usize,
        num_layers: usize,
        kv_dim: usize,
    ) {
        let src_layer_stride = 2 * src_page_size * kv_dim;
        let dst_layer_stride = 2 * dst_page_size * kv_dim;

        for layer in 0..num_layers {
            let src_k_base = layer * src_layer_stride + src_offset * kv_dim;
            let src_v_base =
                layer * src_layer_stride + src_page_size * kv_dim + src_offset * kv_dim;
            let dst_k_base = layer * dst_layer_stride + dst_offset * kv_dim;
            let dst_v_base =
                layer * dst_layer_stride + dst_page_size * kv_dim + dst_offset * kv_dim;

            dst_page[dst_k_base..dst_k_base + kv_dim]
                .copy_from_slice(&src_page[src_k_base..src_k_base + kv_dim]);
            dst_page[dst_v_base..dst_v_base + kv_dim]
                .copy_from_slice(&src_page[src_v_base..src_v_base + kv_dim]);
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

    /// **Unstable**: element format used by this cache's page storage.
    pub fn cache_type(&self) -> CacheType {
        self.pool.cache_type()
    }

    /// **Unstable**: append a single token's K and V for a specific layer.
    ///
    /// Automatically allocates new pages as needed.
    ///
    /// # Panics
    ///
    /// Panics if `layer`/slice-length preconditions are violated, Q8 input
    /// contains a non-finite value, or `EvictionPolicy::Lru` would need to grow
    /// the page table past `max_pages` (i.e. the sequence has exceeded
    /// `max_tokens()`). Sliding
    /// the window past capacity is not yet supported (see issue #337):
    /// evicting the oldest page and re-pushing it is net-zero on
    /// `num_pages()`, so without this guard the growth loop below would
    /// never converge and would instead drain `lru_order` and panic on an
    /// unrelated internal invariant a few iterations later.
    pub fn append_kv_layer(&mut self, layer: usize, k_token: &[f32], v_token: &[f32]) {
        let kv_dim = self.config.kv_dim();
        assert_eq!(k_token.len(), kv_dim);
        assert_eq!(v_token.len(), kv_dim);
        assert!(layer < self.config.num_layers);

        let pos = self.table.seq_len();
        let page_size = self.config.page_size;

        // Check if we need a new page.
        let needed_pages = (pos / page_size) + 1;
        if self.config.eviction == EvictionPolicy::Lru {
            assert!(
                needed_pages <= self.config.max_pages,
                "PagedKVCache::append_kv_layer: position {pos} needs {needed_pages} pages but \
                 max_pages is {}; EvictionPolicy::Lru does not yet support sequences beyond \
                 max_tokens ({}) (see issue #337)",
                self.config.max_pages,
                self.config.max_tokens(),
            );
        }
        while self.table.num_pages() < needed_pages {
            let phys = self.alloc_page();
            self.table.push_page(phys);
        }

        let (phys_page, offset) = self.table.resolve(pos);

        // Update LRU.
        self.touch_page(phys_page);

        // Page layout: [num_layers, 2, page_size, kv_dim].
        let layer_stride = 2 * page_size * kv_dim;
        let k_base = layer * layer_stride + offset * kv_dim;
        let v_base = layer * layer_stride + page_size * kv_dim + offset * kv_dim;
        match &mut self.pool {
            PagedPagePool::F32(pool) => {
                let page_data = pool.page_data_mut(phys_page);
                page_data[k_base..k_base + kv_dim].copy_from_slice(k_token);
                page_data[v_base..v_base + kv_dim].copy_from_slice(v_token);
            }
            PagedPagePool::Q8(pool) => {
                let scale_layer_stride = 2 * page_size;
                let k_scale = layer * scale_layer_stride + offset;
                let v_scale = layer * scale_layer_stride + page_size + offset;
                pool.store_vector(phys_page, k_base, k_scale, k_token);
                pool.store_vector(phys_page, v_base, v_scale, v_token);
            }
        }
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
        self.gather(layer, false, dst);
    }

    /// **Unstable**: read V values for a given layer across the full sequence.
    pub fn gather_v(&self, layer: usize, dst: &mut [f32]) {
        self.gather(layer, true, dst);
    }

    fn gather(&self, layer: usize, value_cache: bool, dst: &mut [f32]) {
        let seq_len = self.table.seq_len();
        let kv_dim = self.config.kv_dim();
        let page_size = self.config.page_size;
        assert_eq!(dst.len(), seq_len * kv_dim);
        let value_offset = usize::from(value_cache) * page_size;

        match &self.pool {
            PagedPagePool::F32(pool) => {
                let layer_stride = 2 * page_size * kv_dim;
                let mut pos = 0usize;
                while pos < seq_len {
                    let (phys_page, offset) = self.table.resolve(pos);
                    let run_len = (page_size - offset).min(seq_len - pos);
                    let len = run_len * kv_dim;
                    let page_data = pool.page_data(phys_page);
                    let src_base = layer * layer_stride + (value_offset + offset) * kv_dim;
                    let dst_base = pos * kv_dim;
                    dst[dst_base..dst_base + len]
                        .copy_from_slice(&page_data[src_base..src_base + len]);
                    pos += run_len;
                }
            }
            PagedPagePool::Q8(pool) => {
                let layer_stride = 2 * page_size * kv_dim;
                let scale_layer_stride = 2 * page_size;
                for pos in 0..seq_len {
                    let (phys_page, offset) = self.table.resolve(pos);
                    let src_base = layer * layer_stride + (value_offset + offset) * kv_dim;
                    let scale_offset = layer * scale_layer_stride + value_offset + offset;
                    let dst_base = pos * kv_dim;
                    pool.gather_vector(
                        phys_page,
                        src_base,
                        scale_offset,
                        &mut dst[dst_base..dst_base + kv_dim],
                    );
                }
            }
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
        self.config.max_pages * self.pool.bytes_per_page()
    }

    /// **Unstable**: memory used by allocated pages in bytes.
    pub fn used_memory_bytes(&self) -> usize {
        self.pool.allocated_count() * self.pool.bytes_per_page()
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
        #[expect(
            clippy::expect_used,
            reason = "the pool is exhausted, so the LRU order holds at least one page"
        )]
        let evicted = self
            .lru_order
            .pop_front()
            .expect("LRU order empty but pool exhausted");

        #[expect(
            clippy::expect_used,
            reason = "the page table holds an LRU page whenever the pool is exhausted"
        )]
        let removed = self
            .table
            .pop_front_page()
            .expect("invariant: page table has an LRU page when pool is exhausted");
        debug_assert_eq!(removed, evicted);

        self.pool.clear_page(evicted);
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
mod tests;
