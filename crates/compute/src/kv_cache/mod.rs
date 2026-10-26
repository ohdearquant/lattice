//! KV cache for autoregressive transformer decoding.
//!
//! Two implementations:
//! - `FlatKVCache`: Simple contiguous cache for immediate use. Pre-allocates
//!   per-layer K and V buffers up to `max_seq_len`. O(1) append and lookup.
//! - `PagedKVCache`: Page-based cache with on-demand 256-token page allocation,
//!   LRU eviction, and memory budgeting for multi-model serving.

pub mod cross_turn;
pub mod flat;
pub mod paged;
pub mod prefix;

pub use cross_turn::{
    CrossTurnPrefixEntry, CrossTurnPrefixMetadata, CrossTurnSlotId, KvPrefixHandle,
    PrefixRestorePlan, PrefixReuseMode, checkpoint_survives_save, longest_common_token_prefix,
    plan_prefix_reuse, plan_prefix_reuse_for_slot,
};
pub use flat::{FlatKVCache, FlatKVCacheConfig};
pub use paged::{CacheType, EvictionPolicy, PagePool, PageTable, PagedKVCache, PagedKVCacheConfig};
pub use prefix::{
    AdapterId, PrefixEntry, PrefixKey, PrefixPageCache, PrefixPageCacheConfig, SharedPageRef,
};
