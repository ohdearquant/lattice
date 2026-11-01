//! **Stability tier**: Experimental
//!
//! This is a pure ML inference kernel with high churn, 295 `unsafe` blocks and 59
//! `dead_code_allows` as of this commit. Both counts exclude comment lines, so the
//! sentence you are reading does not count itself; recompute with
//! `grep -rn 'unsafe {' crates/inference/src --include='*.rs' | grep -vE ':\s*//' | wc -l`
//! and the same pipeline over `allow(dead_code` rather than trusting these numbers verbatim. It is NOT intended for direct use by platform or feature crates.
//! Consumers should go through `lattice-embed`.
//!
//! **Compatibility paths are removed at 1.0.** Since the split, the code lives in the crates
//! that own it: `lattice-compute`, `lattice-arch`, `lattice-weights`, `lattice-metal-runtime`
//! and the `lattice-model-*` crates, among others. Most module paths this crate still exposes
//! are re-exports kept so that code written against the pre-split `lattice-inference` keeps
//! compiling. They are not a stable API, and these low-level re-exports will be removed at 1.0.
//! New code should import from the owning crate. Each compatibility module says so in its own
//! documentation, and the removal list is tracked in
//! <https://github.com/ohdearquant/lattice/issues/2333>.
//!
// ML inference kernels: many functions have >7 args by necessity (BLAS-style APIs
// where grouping into structs would require heap allocation in hot paths), and many
// loops use the index to access multiple arrays simultaneously so the
// needless_range_loop suggestion does not apply.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
//! lattice-inference: pure Rust transformer inference for embedding models.
//!
//! Supports two architectures:
//! - **BERT/BGE** (encoder-only): bidirectional attention, mean pooling
//! - **Qwen3** (decoder-only): causal GQA with RoPE, SwiGLU, last-token pooling
//!
//! ## Module Organization
//!
//! - [`model`] — Model configs and loaders (BERT, Qwen, Qwen3.5, BitNet)
//! - [`tokenizer`] — Tokenizers (WordPiece, SentencePiece, BPE)
//! - [`weights`] — Weight storage formats (f32, f16, Q8)
//! - [`attention`] — Attention mechanisms (standard, GQA, flash, GDN)
//! - [`forward`] — Compute backends (CPU, NEON, Metal GPU, batched prefill)

// Grouped modules
/// Attention kernel variants (standard, GQA, flash, GDN, sparse, differential) and the
/// [`attention::AttentionTag`] used to dispatch between them. Called from [`forward`] and [`model`].
pub mod attention;
/// Compute backends: scalar CPU, NEON, Metal GPU, WGPU, Q8/f16 kernels, and batched prefill.
/// Consumes kernels from [`attention`] and tensors from [`weights`].
pub mod forward;
/// Model configs and loaders (BERT, Qwen, Qwen3.5, BitNet). Each submodule owns its
/// safetensors load path and forward-pass dispatch; see [`weights`], [`tokenizer`], and [`forward`].
pub mod model;
/// Canonical model-directory format detector (`ModelFormat`/`detect_format`) shared by
/// the `chat_metal` binary and the `lattice` and `lattice_serve` binaries of `lattice-serve`
/// (ADR-080 amendment, #829).
/// **Unstable, internal-binaries-only** -- see the module's own doc comment.
pub mod model_format;
/// Tokenizer implementations (`WordPiece`, `SentencePiece`, byte-level BPE) behind the
/// [`Tokenizer`] trait, plus the [`load_tokenizer`] auto-detect helper. See [`model`].
pub mod tokenizer;
/// Vision encoder paths: the Qwen3.5 vision tower and the PaddleOCR-VL encoder.
/// See [`model`] and [`weights`].
pub mod vision;
/// Safetensors-backed tensor storage and weight formats (f32, f16, Q8, Q4). See [`model`]
/// and [`forward`].
pub mod weights;

// Standalone modules
/// Continuous batching and scheduler support for multi-sequence inference. See [`kv_cache`]
/// and [`model`].
pub mod batch;
#[cfg(test)]
mod bounded_config;
#[cfg(test)]
mod bounded_read;
/// Registered held-out evidence for router candidate admission.
///
/// **Compatibility path.** Re-exported from `lattice-router` (`lattice_router::admission`) so that code
/// written against the pre-split `lattice-inference` keeps compiling. Not a stable API; removed at
/// 1.0. Import from `lattice-router` directly.
#[cfg(feature = "router-learning")]
pub use lattice_router::admission as router_admission;
/// Recorded routing episodes and the feedback that arrived for them.
///
/// **Compatibility path.** Re-exported from `lattice-router` (`lattice_router::episode`) so that code
/// written against the pre-split `lattice-inference` keeps compiling. Not a stable API; removed at
/// 1.0. Import from `lattice-router` directly.
pub use lattice_router::episode as router_episode;
/// Provisioning a router state directory with an untrained bootstrap
/// (ADR-096 decision 2).
///
/// **Compatibility path.** Re-exported from `lattice-router` (`lattice_router::init`) so that code
/// written against the pre-split `lattice-inference` keeps compiling. Not a stable API; removed at
/// 1.0. Import from `lattice-router` directly.
pub use lattice_router::init as router_init;
/// Bounded construction of router candidates from recorded feedback.
///
/// **Compatibility path.** Re-exported from `lattice-router` (`lattice_router::refit`) so that code
/// written against the pre-split `lattice-inference` keeps compiling. Not a stable API; removed at
/// 1.0. Import from `lattice-router` directly.
#[cfg(feature = "router-learning")]
pub use lattice_router::refit as router_refit;
/// The on-disk router gate artifact: version, trained-adapter names, payload.
/// Deliberately free of the `mixture` gate — see the module doc comment.
///
/// **Compatibility path.** Re-exported from `lattice-router` (`lattice_router::state`) so that code
/// written against the pre-split `lattice-inference` keeps compiling. Not a stable API; removed at
/// 1.0. Import from `lattice-router` directly.
pub use lattice_router::state as router_state;
// ADR-090 row C routed `model::qwen35::generation`'s `generate()` (by way of
// `generate_with_trace()`) through `QwenCpuSession` and `decoder::driver::run`;
// row R03 removed the `gen_cfg.grammar.is_none() && gen_cfg.logprobs.is_none()`
// gate, so every canonical Qwen3.5 CPU generate/stream request -- grammar and
// logprobs included -- now routes through the driver unconditionally, with no
// remaining inline fallback. This is the first PRODUCTION caller, so this
// module is reachable under the plain lib build and the blanket dead-code
// allow above it is no longer warranted.
pub(crate) mod decoder;
/// Measurement-only seams for the per-token allocation instrument (ADR-090 D7, row R01).
/// Compiled only with `bench-internals`; not part of the supported API.
#[cfg(all(feature = "qwen35", feature = "bench-internals"))]
#[doc(hidden)]
pub use decoder::bench_support as decoder_bench_support;
/// Synthetic checkpoint support for repository regression tests.
#[cfg(all(feature = "bert", feature = "test-utils"))]
#[doc(hidden)]
pub use lattice_model_bert::bert_pooling_fixture;
/// Model-file cache and conditional download helpers. See [`model`] and [`weights`].
pub mod download;
/// The text encoder contract and its adapters for the BERT-family, Qwen3 and EmbeddingGemma 2
/// embedding models. See [`encoder::Encoder`].
pub mod encoder;
/// Crate error taxonomy; see [`InferenceError`].
pub mod error;
/// The family contract: static descriptors naming each family's loaders, a registry built from
/// them, and the Qwen3.5 CPU decoder they load. See [`family::registry`].
pub mod family;
/// Time-series forecasting: context preparation and native forecasting models. See
/// [`forecast::Reverso`].
pub mod forecast;
/// Neutral generation types shared by every decoder path: request configuration and
/// result struct, independent of model family. See [`GenerateConfig`] and
/// [`GenerateOutput`].
pub mod generation;
/// Grammar-constrained decoding and logit masking. See [`model`] and [`sampling`].
pub mod grammar;
/// Flat and paged key/value cache implementations. See [`model`] and [`forward`].
pub mod kv_cache;
/// PEFT/MLX LoRA safetensors loading: a path in, the `(layers, descriptor)` pair the
/// Metal engine takes out. Every item in it is Metal-only, so the module itself is
/// gated rather than left as an empty shell on other builds; it also reaches
/// `lattice-fann` for the shared effective-scale helper, and that dependency exists
/// only under `metal-gpu` (or `mixture`). See [`lora_hook`]; `lattice-serve` is its serving caller.
#[cfg(all(feature = "qwen35", target_os = "macos", feature = "metal-gpu"))]
pub mod lora_file;
/// LoRA adapter hook called from inference forward paths. See [`model`] and [`forward`].
pub mod lora_hook;
/// Repository-internal guards shared by Metal tests and measurement targets.
///
/// This module is hidden from generated documentation and is not a supported
/// production API. It is exported because Cargo builds repository integration
/// tests, benches, examples, and binaries as separate crates.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub mod measurement {
    pub use lattice_metal_runtime::measurement::gpu_test_lock;
}
/// Inference metrics and entropy accumulation. See [`model`].
pub mod metrics;
/// Adapter routing and mixture support built on top of [`lora_hook`] and [`sampling`].
/// Requires the `mixture` feature.
///
/// **Compatibility path.** Re-exported from `lattice-router` (`lattice_router::mixture`) so that code
/// written against the pre-split `lattice-inference` keeps compiling. Not a stable API; removed at
/// 1.0. Import from `lattice-router` directly.
#[cfg(feature = "mixture")]
pub use lattice_router::mixture;
/// Offline MoE expert-cache admission-policy simulator (issue #682 Stage 3):
/// replays a JSONL routing trace against [`forward::moe_expert_cache`]'s
/// shipped LRU policy plus challenger policies (ARC, sequence-local
/// frequency admission) to measure hit-rate deltas before any engine
/// eviction-policy change. See [`moe_admission`]'s module doc comment.
#[cfg(feature = "qwen35")]
pub mod moe_admission;
/// Option-letter scoring: one prefill, with the answer read from the next-token logits at
/// the option letters. Holds [`option_scoring::OptionScores`] and the letter helpers shared by
/// the CPU and Metal Qwen3.5 scorers. See [`model`] and [`tokenizer`].
pub mod option_scoring;
/// Embedding pooling helpers (mean, CLS, last-token) including [`BertPooling`]. Used by
/// `model::BertModel` and `model::QwenModel`.
pub mod pool;
/// ShortGPT-style block influence scoring. See [`model`].
pub mod pruning;
/// Quantization and pre-transform primitives. See [`weights`] and [`forward`].
pub mod quant;
/// Synthetic Qwen3.5 checkpoint support for repository regression tests.
#[cfg(feature = "test-utils")]
#[doc(hidden)]
#[cfg(feature = "qwen35")]
#[doc(inline)]
pub use lattice_model_qwen35::qwen35_checkpoint_fixture;
/// Rotary position embedding tables and application helpers. See [`model`] and [`forward`].
pub mod rope;
/// Sampling configuration and token selection helpers. See [`model`] and `speculative`.
pub mod sampling;
/// Unsupported, version-coupled interface for `lattice-serve` (ADR-103
/// Amendment 4); not public API.
#[cfg(feature = "serving-support")]
#[doc(hidden)]
pub mod serving_support;
/// N-gram prompt lookup speculative decoding. See [`sampling`] and [`model`].
#[cfg(feature = "qwen35")]
pub mod speculative;
/// Generation stop reason taxonomy; see [`StopReason`] and [`model`].
pub mod stop_reason;

/// Test-support helper for the thirteen checkpoint-gated unit tests in
/// `model::bert`, [`model::qwen`], and `model::qwen35::generation` (a
/// private submodule, so not linked) (issue #1664): resolves and panics
/// naming an unset, relative, or missing
/// model-directory env var, so an explicitly requested `--ignored` run
/// without a checkpoint fails loudly instead of returning silently.
#[cfg(test)]
mod test_support;

/// Backward-pass support for training and LoRA workflows, built on [`lora_hook`] and
/// [`model`]. Requires the `train-backward` feature.
#[cfg(feature = "train-backward")]
pub mod backward;

// Re-exports for public API backward compatibility
/// Root error type for inference, tokenizer, model loading, and runtime failures. See [`error`].
pub use crate::error::InferenceError;
/// Dormant BERT CPU policy contract for sealed native preparation.
#[doc(hidden)]
pub use crate::forward::cpu::BertCpuKernelPolicy;
/// Frozen CPU capability facts for the dormant BERT pinned-policy contract.
#[doc(hidden)]
pub use crate::forward::cpu::BertCpuKernelProfile;
/// Sampling and stop configuration for a generation request. Canonical path under ADR-092;
/// `model::qwen35_config::GenerateConfig` remains as a deprecated alias. See [`generation`].
#[doc(no_inline)]
pub use crate::generation::GenerateConfig;
/// Result of a generation request: text, token ids, counts and stop reason. Canonical path
/// under ADR-092. See [`generation`] and [`StopReason`].
#[doc(no_inline)]
pub use crate::generation::GenerateOutput;
/// Per-token log-probability data for one generated token. See [`GenerateConfig::logprobs`].
#[doc(no_inline)]
pub use crate::generation::TokenLogprob;
/// One alternative token and its log-probability at a single step. See [`TokenLogprob`].
#[doc(no_inline)]
pub use crate::generation::TopLogprob;
/// Grammar-constrained decoding engine, reachable from the root because
/// [`GenerateConfig::grammar`] carries one. See [`grammar`].
#[doc(no_inline)]
pub use crate::grammar::GrammarEngine;
/// BERT encoder configuration. See `BertModel` and [`model`].
#[cfg(feature = "bert")]
pub use crate::model::BertConfig;
/// BERT/BGE encoder model. See `BertConfig`, [`Tokenizer`], and [`BertPooling`].
#[cfg(feature = "bert")]
pub use crate::model::BertModel;
/// BERT-style cross-encoder/reranker model. See `BertModel` and [`model`].
#[cfg(feature = "bert")]
pub use crate::model::CrossEncoderModel;
/// Per-layer profiling data collected during Qwen embedding inference. See [`ProfileTimings`]
/// and [`QwenModel`].
#[cfg(feature = "qwen3")]
pub use crate::model::LayerTimings;
/// Aggregate profiling report for Qwen inference. See [`LayerTimings`] and [`QwenModel`].
#[cfg(feature = "qwen3")]
pub use crate::model::ProfileTimings;
/// Qwen embedding model configuration. See `QwenModel` and [`weights`].
pub use crate::model::QwenConfig;
/// Qwen embedding model exposing `encode` for producing embeddings. See [`QwenConfig`],
/// [`Tokenizer`], and [`weights`].
#[cfg(feature = "qwen3")]
pub use crate::model::QwenModel;
/// BERT pooling strategy selector (mean or CLS). See [`pool`] and `BertModel`.
#[doc(no_inline)]
pub use crate::pool::BertPooling;
/// Reason a generation request stopped (e.g. EOS, max tokens). See [`stop_reason`] and
/// [`model`].
#[doc(no_inline)]
pub use crate::stop_reason::StopReason;
/// Byte-level BPE tokenizer used by Qwen-family models. See [`Tokenizer`] and [`TokenizedInput`].
pub use crate::tokenizer::BpeTokenizer;
/// Token counts of one embedding input before and after the model's length limit. See
/// `BertModel::embedding_token_counts` and `QwenModel::embedding_token_counts`.
pub use crate::tokenizer::EmbeddingTokenCounts;
/// Additive Gemma-family BPE tokenizer (literal-space `Split` + `▁` metaspace normalizer),
/// explicitly selected — never reached via [`load_tokenizer`]'s model-type sniffing. See
/// [`Tokenizer`] and ADR-082 G17.
pub use crate::tokenizer::GemmaBpeTokenizer;
/// `SentencePiece` tokenizer implementation. See [`Tokenizer`] and [`TokenizedInput`].
pub use crate::tokenizer::SentencePieceTokenizer;
/// Padded token IDs and the real (unpadded) sequence length returned by tokenizers. See
/// [`Tokenizer`] and [`tokenizer`].
pub use crate::tokenizer::TokenizedInput;
/// Object-safe tokenizer trait implemented by every tokenizer in [`tokenizer`]. See
/// [`load_tokenizer`].
pub use crate::tokenizer::Tokenizer;
/// `WordPiece` tokenizer used by BERT-family models. See [`Tokenizer`] and `BertModel`.
pub use crate::tokenizer::WordPieceTokenizer;
/// Model-directory tokenizer auto-loader. See [`Tokenizer`] and [`tokenizer`].
pub use crate::tokenizer::load_tokenizer;
/// `tokenizer.json`-text tokenizer loader (no filesystem access). See
/// [`Tokenizer`], [`tokenizer`], and `BertModel::from_bytes`.
pub use crate::tokenizer::tokenizer_from_json_str;
/// Stage-1 marker-expansion arithmetic (ADR-082 G11/G15/G17): `<|image|>`/`<|audio|>`
/// placeholder-to-soft-token-count contract, independent of the in-sequence scatter itself.
pub use crate::tokenizer::{
    GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES, GEMMA4_AUDIO_HOP_LENGTH_SAMPLES,
    GEMMA4_AUDIO_MAX_SOFT_TOKENS, GEMMA4_AUDIO_MS_PER_SOFT_TOKEN, GEMMA4_AUDIO_SAMPLING_RATE_HZ,
    GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE, audio_marker_expansion_tokens,
    audio_marker_expansion_tokens_from_samples, image_marker_expansion_tokens,
    total_audio_marker_expansion_tokens,
};
