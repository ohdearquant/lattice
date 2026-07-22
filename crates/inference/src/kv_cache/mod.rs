//! KV cache for autoregressive transformer decoding.
//!
//! Two implementations:
//! - `FlatKVCache`: Simple contiguous cache for immediate use. Pre-allocates
//!   per-layer K and V buffers up to `max_seq_len`. O(1) append and lookup.
//! - `PagedKVCache`: Page-based cache with on-demand 256-token page allocation,
//!   LRU eviction, and memory budgeting for multi-model serving.

pub(crate) mod cross_turn;
pub(crate) mod flat;
pub(crate) mod paged;
pub(crate) mod prefix;

pub use cross_turn::{
    CrossTurnPrefixEntry, CrossTurnPrefixMetadata, CrossTurnSlotId, KvPrefixHandle,
    PrefixRestorePlan, PrefixReuseMode, checkpoint_survives_save, longest_common_token_prefix,
    plan_prefix_reuse,
};
pub use flat::{FlatKVCache, FlatKVCacheConfig};
pub use paged::{CacheType, EvictionPolicy, PagePool, PageTable, PagedKVCache, PagedKVCacheConfig};
pub use prefix::{
    AdapterId, PrefixEntry, PrefixKey, PrefixPageCache, PrefixPageCacheConfig, SharedPageRef,
};
