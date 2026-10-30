//! Cache definitions re-exported from `lattice-compute`.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-compute` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-compute` directly.

#[doc(inline)]
pub use lattice_compute::kv_cache::cross_turn::{
    CrossTurnPrefixEntry, CrossTurnPrefixMetadata, CrossTurnSlotId, KvPrefixHandle,
    PrefixRestorePlan, PrefixReuseMode, checkpoint_survives_save, longest_common_token_prefix,
    plan_prefix_reuse_for_slot,
};
// Re-exported for compatibility; callers of the deprecated path still see the deprecation.
#[allow(deprecated)]
#[doc(inline)]
pub use lattice_compute::kv_cache::cross_turn::plan_prefix_reuse;
