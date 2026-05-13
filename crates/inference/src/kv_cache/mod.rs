//! KV cache for autoregressive transformer decoding.
//!
//! Two implementations:
//! - [`FlatKVCache`]: Simple contiguous cache for immediate use. Pre-allocates
//!   per-layer K and V buffers up to `max_seq_len`. O(1) append and lookup.
//! - [`PagedKVCache`]: Page-based cache with on-demand 256-token page allocation,
//!   LRU eviction, and memory budgeting for multi-model serving.

pub(crate) mod flat;
pub(crate) mod paged;

pub use flat::{FlatKVCache, FlatKVCacheConfig};
pub use paged::{EvictionPolicy, PagePool, PageTable, PagedKVCache, PagedKVCacheConfig};
