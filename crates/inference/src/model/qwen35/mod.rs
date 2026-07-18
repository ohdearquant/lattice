//! Qwen3.5-2B text generation model.
//!
//! Hybrid architecture: 18 GatedDeltaNet (linear attention) + 6 standard GQA layers.
//! Pattern: [linear, linear, linear, full] x 6 = 24 layers.
//!
//! Key features:
//! - Autoregressive text generation with temperature/top-k/top-p sampling
//! - Tied embeddings (lm_head = embed_tokens^T)
//! - Partial RoPE (25% of head_dim) on full-attention layers
//! - GatedDeltaNet recurrent state for linear layers, KV cache for full layers

mod cache;
mod debug;
pub(crate) mod detokenize;
mod eval;
mod forward;
mod generation;
mod loading;
mod model;
mod moe;
mod norm;
mod rerank;
mod sampling;
pub(crate) mod stop_strings;
mod weights;

pub use eval::{PerplexityConfig, PerplexityReport};
/// Re-exported for the Metal Q4 perplexity harness in
/// [`crate::forward::metal_qwen35`]; the CPU forward path consumes them
/// directly inside [`eval`].
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use eval::{log_softmax_nll, run_strided_perplexity};

#[cfg(test)]
mod tests;

/// Test-only tiny zero-weight model construction (ADR-080 C2), gated behind
/// the `test-utils` Cargo feature (for `crates/inference/src/bin/lattice.rs`'s
/// separate compilation unit) OR `cfg(test)` (for this crate's own library
/// tests, e.g. `generation.rs`'s `StopReason` tests) so it never ships in a
/// normal build.
#[cfg(any(test, feature = "test-utils"))]
pub mod test_support;

pub use model::Qwen35Model;
pub use weights::ModelWeights;

pub(crate) use cache::{ForwardScratch, KvCache, resize};
pub(crate) use detokenize::decode_tokens;
// Re-exported so that quantized CPU generate helpers (cpu_q8, cpu_f16,
// neon_forward) can share the same typed grammar-not-set guard without
// duplicating the predicate or the error message (#397/#398).
pub(crate) use generation::check_grammar_not_set;
// Sibling guard for `logprobs` on the same unwired paths (#585).
pub(crate) use generation::check_logprobs_not_set;
// Sibling guards for `stop_strings` / `reasoning_budget` on the same unwired
// paths (ADR-080 C3, #783).
pub(crate) use generation::{check_reasoning_budget_not_set, check_stop_strings_not_set};
// Shared empty-prompt preflight (#856): every CPU forward path (cpu_q8,
// cpu_f16, neon_forward) and every Metal generation entry point
// (forward::metal_qwen35) calls this instead of its own inline
// `if prompt_len == 0` copy, unifying the CPU/Metal empty-prompt contract.
pub(crate) use generation::check_prompt_not_empty;
// Shared total-context admission bound (#922): every Metal generation entry
// point (forward::metal_qwen35) calls this after check_prompt_not_empty to
// mirror the CPU `generate`/`generate_streaming` total bound
// (prompt_len + decode budget <= max_context), instead of only bounding the
// prompt alone. Only the Metal (`mod inner`, gated identically) consumer
// needs the re-export; the CPU forward paths (cpu_f16, cpu_q8, neon_forward)
// already enforce this same bound with their own inline check and
// `generation.rs` itself uses `check_context_budget` directly within its own
// module.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use generation::check_context_budget;
// Shared backend-neutral decode-policy struct (reasoning-budget accounting +
// logprobs formatting), consumed by the Metal streaming loops in
// `crate::forward::metal_qwen35` so the same bookkeeping isn't re-duplicated
// across the CPU/Metal boundary (ADR-080 C3). Only the Metal (`mod inner`,
// gated identically) consumer needs the re-export; `generation.rs` itself
// uses `DecodePolicy` directly within its own module.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use generation::{
    DecodePolicy, REASONING_CLOSE_MARKER, StepOutcome, StopCheckOutcome,
    resolve_reasoning_close_token,
};
// Sibling guard for `enable_mtp` on the cross-turn prefix-cache path, which
// has no MTP draft/verify wiring (PR #787). Only
// that Metal-only path needs it, same gate as `DecodePolicy`/`StepOutcome`.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use generation::check_mtp_not_requested;
pub(crate) use norm::qwen35_rms_norm;
pub(crate) use sampling::sample_token;
pub(crate) use weights::{
    AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
    FullAttentionLayerWeights,
};

#[cfg(test)]
pub(crate) use weights::{MoeLayerWeights, MoeRouter, RoutedExperts, SharedExpert};

#[cfg(test)]
pub use detokenize::bytes_to_unicode;
// Needed by all generate paths (cpu_q8, cpu_f16, neon_forward, batch_prefill)
// and by tests. `pub(crate)` keeps it out of the public API surface.
pub(crate) use generation::should_stop_token;
// Public raw generation-lifecycle observer event, consumed by
// `--emit-phase-events` in `qwen35_generate.rs` via
// `Qwen35Model::generate_streaming_with_observer`.
pub use generation::RawGenEvent;

/// Exposed for consumers that need to drive a per-layer coverage check
/// over a Qwen3.5 checkpoint without going through the model loader —
/// e.g., the QuaRot offline converter (ADR-044 step 3c) iterating
/// rotation rules against an actual safetensors file. Originally
/// `#[cfg(test)]`-only; promoted in step 3b.
pub use loading::qwen_required_tensor_names;
