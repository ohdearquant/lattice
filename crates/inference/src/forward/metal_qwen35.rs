//! Qwen3.5 family compatibility exports.

#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::ChatRole;

#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::ChatMessage;

#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::ChatImage;

#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::format_chat_template;

#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::mtp_missing_weights_warning;

#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::MtpHiddenTap;

#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::MtpRoundOutcome;

#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::mtp_greedy_round;

#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::GdnStateTrafficScope;

#[cfg(feature = "gdn-state-counters")]
#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::GdnStateCopyKind;

#[cfg(feature = "gdn-state-counters")]
#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::GdnStateTrafficShape;

#[cfg(feature = "gdn-state-counters")]
#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::GdnStateTrafficBucket;

#[cfg(feature = "gdn-state-counters")]
#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::GdnStateTrafficReport;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::{
    ChatCompletionOutput, HiddenReadbackPathProofSnapshot, LayerImportanceScore, LayerPruningPlan,
    LogitReadbackPathProofSnapshot, LoraLayerData, MetalQwen35State, MoeRoutingTraceRecord,
    PathProofSnapshot, arm_moe_routing_trace, blend_lora_layer_data, dump_moe_routing_trace_jsonl,
    take_moe_routing_trace,
};

#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
))]
#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::bench_support;

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::LoraLayerData;

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::blend_lora_layer_data;

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
#[doc(inline)]
pub use lattice_model_qwen35::forward::metal_qwen35::MetalQwen35State;

/// lattice#1630 (widening lattice#1584): no production Metal dispatch anywhere under
/// `crates/inference/src` may create a command buffer outside an autorelease pool.
///
/// `metal-rs`'s `new_command_buffer` returns a borrowed reference to an autoreleased,
/// unretained object, so a command buffer created with no enclosing pool survives until
/// the thread's outermost pool drains, which on the long-lived serving worker never
/// happens during a session. The measured cost on the Qwen3.5 decode funnel was 1.3-1.5
/// KB per step (lattice#1584).
///
/// lattice#1584's guard scanned this file's population alone, so the GEMM dispatch in
/// `metal_gemm.rs` and the Qwen3-Embedding forward in `metal.rs` were never swept: same
/// unpooled `new_command_buffer` call, different file. Rather than hand-listing those two
/// files as a second known-sites population (the exact gap that let them go unswept the
/// first time), this guard walks `crates/inference/src` itself and scans every file it
/// finds, so a future production module that dispatches Metal work is discovered
/// automatically. Every function in a discovered file's production region that creates a
/// command buffer must be a `*_dispatch` body reached through a pooling wrapper of the
/// same base name, or appear in `EXEMPT` with its reason.
#[cfg(test)]
mod metal_command_buffer_pool_tests;
#[cfg(test)]
mod public_scheduling_entry_point_tests;

#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "serving-support"
))]
pub(crate) use lattice_model_qwen35::forward::metal_qwen35::CachedGenerateOutput;
