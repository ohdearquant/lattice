//! Qwen3.5 family compatibility exports.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-qwen35` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-qwen35` directly.

#[doc(inline)]
pub use lattice_model_qwen35::forward::moe_expert_cache::MOE_EXPERT_CACHE_SLOTS_ENV;

#[doc(inline)]
pub use lattice_model_qwen35::forward::moe_expert_cache::MoeExpertCacheConfig;

#[doc(inline)]
pub use lattice_model_qwen35::forward::moe_expert_cache::moe_expert_cache_num_slots;
