//! Cache definitions re-exported from `lattice-compute`.

#[doc(inline)]
pub use lattice_compute::kv_cache::cross_turn::{
    CrossTurnPrefixEntry, CrossTurnPrefixMetadata, CrossTurnSlotId, KvPrefixHandle,
    PrefixRestorePlan, PrefixReuseMode, checkpoint_survives_save, longest_common_token_prefix,
    plan_prefix_reuse, plan_prefix_reuse_for_slot,
};
