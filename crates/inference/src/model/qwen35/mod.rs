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
mod sampling;
mod weights;

pub use eval::{PerplexityConfig, PerplexityReport};
/// Re-exported for the Metal Q4 perplexity harness in
/// [`crate::forward::metal_qwen35`]; the CPU forward path consumes them
/// directly inside [`eval`].
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use eval::{log_softmax_nll, run_strided_perplexity};

#[cfg(test)]
mod tests;

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

/// Exposed for consumers that need to drive a per-layer coverage check
/// over a Qwen3.5 checkpoint without going through the model loader —
/// e.g., the QuaRot offline converter (ADR-044 step 3c) iterating
/// rotation rules against an actual safetensors file. Originally
/// `#[cfg(test)]`-only; promoted in step 3b.
pub use loading::qwen_required_tensor_names;
