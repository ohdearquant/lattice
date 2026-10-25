//! KV cache storage and prefix reuse for autoregressive decoding.

pub mod cross_turn;
pub mod flat;
pub mod prefix;

pub use cross_turn::{
    CrossTurnPrefixEntry, CrossTurnPrefixMetadata, CrossTurnSlotId, KvPrefixHandle,
    PrefixRestorePlan, PrefixReuseMode, checkpoint_survives_save, longest_common_token_prefix,
    plan_prefix_reuse,
};
pub use prefix::{
    AdapterId, PrefixEntry, PrefixKey, PrefixPageCache, PrefixPageCacheConfig, SharedPageRef,
};
pub use flat::{FlatKVCache, FlatKVCacheConfig};
