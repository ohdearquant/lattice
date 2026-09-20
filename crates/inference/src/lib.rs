//! **Stability tier**: Experimental
//!
//! This is a pure ML inference kernel with high churn, 295 `unsafe` blocks and 59
//! `dead_code_allows` as of this commit. Both counts exclude comment lines, so the
//! sentence you are reading does not count itself; recompute with
//! `grep -rn 'unsafe {' crates/inference/src --include='*.rs' | grep -vE ':\s*//' | wc -l`
//! and the same pipeline over `allow(dead_code` rather than trusting these numbers verbatim. It is NOT intended for direct use by platform or feature crates.
//! Consumers should go through `lattice-embed`. The unsafe blocks are documented in
//! `foundation/STABILITY.md §Tech Debt`. Tracking issue: #1306.
//! See `foundation/STABILITY.md` for the full policy.
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
/// the `lattice`, `lattice_serve`, and `chat_metal` binaries (ADR-080 amendment, #829).
/// **Unstable, internal-binaries-only** -- see the module's own doc comment.
pub mod model_format;
/// Tokenizer implementations (`WordPiece`, `SentencePiece`, byte-level BPE) behind the
/// [`Tokenizer`] trait, plus the [`load_tokenizer`] auto-detect helper. See [`model`].
pub mod tokenizer;
/// Qwen3-VL vision encoder path: patch preprocessing, ViT forward pass, and MLP merger.
/// See [`model`] and [`weights`].
pub mod vision;
/// Safetensors-backed tensor storage and weight formats (f32, f16, Q8, Q4). See [`model`]
/// and [`forward`].
pub mod weights;

// Standalone modules
/// Continuous batching and scheduler support for multi-sequence inference. See [`kv_cache`]
/// and [`model`].
pub mod batch;
pub(crate) mod bounded_read;
#[allow(dead_code)] // removed by the row that adds the first DecoderSession implementation
pub(crate) mod decoder;
/// Model-file cache and conditional download helpers. See [`model`] and [`weights`].
pub mod download;
/// Crate error taxonomy; see [`InferenceError`].
pub mod error;
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
/// only under `metal-gpu` (or `mixture`). See [`lora_hook`] and [`serve`].
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
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
pub mod measurement;
/// Inference metrics and entropy accumulation. See [`model`].
pub mod metrics;
/// Adapter routing and mixture support built on top of [`lora_hook`] and [`sampling`].
/// Requires the `mixture` feature.
#[cfg(feature = "mixture")]
pub mod mixture;
/// Offline MoE expert-cache admission-policy simulator (issue #682 Stage 3):
/// replays a JSONL routing trace against [`forward::moe_expert_cache`]'s
/// shipped LRU policy plus challenger policies (ARC, sequence-local
/// frequency admission) to measure hit-rate deltas before any engine
/// eviction-policy change. See [`moe_admission`]'s module doc comment.
pub mod moe_admission;
/// Embedding pooling helpers (mean, CLS, last-token) including [`BertPooling`]. Used by
/// [`model::BertModel`] and [`model::QwenModel`].
pub mod pool;
/// ShortGPT-style block influence scoring. See [`model`].
pub mod pruning;
/// Quantization and pre-transform primitives. See [`weights`] and [`forward`].
pub mod quant;
/// Rotary position embedding tables and application helpers. See [`model`] and [`forward`].
pub mod rope;
/// Sampling configuration and token selection helpers. See [`model`] and [`speculative`].
pub mod sampling;
/// Shared HTTP serving contract (error envelope, `finish_reason`, `max_tokens`
/// zero-rejection, `/v1/models` body) consumed by both the `lattice` unified
/// server and the `lattice_serve` daemon binaries (ADR-080 cluster C2).
/// Requires the `serve` feature (axum/tokio/futures).
#[cfg(feature = "serve")]
pub mod serve;
/// N-gram prompt lookup speculative decoding. See [`sampling`] and [`model`].
pub mod speculative;
/// Generation stop reason taxonomy; see [`StopReason`] and [`model`].
pub mod stop_reason;

/// Cross-path sweep (#613): every CPU-family `generate*` entry point agrees on
/// the stop-token contract (excluded from `token_ids`/`text`). The Metal-family
/// entry points are covered in `forward::metal_qwen35`'s own test module; see
/// this module's doc comment for the full manifest and rationale.
#[cfg(test)]
mod stop_token_contract;

/// Backward-pass support for training and LoRA workflows, built on [`lora_hook`] and
/// [`model`]. Requires the `train-backward` feature.
#[cfg(feature = "train-backward")]
pub mod backward;

use std::path::PathBuf;

/// Default model cache directory.
pub(crate) fn default_cache_dir() -> Result<PathBuf, error::InferenceError> {
    if let Ok(path) = std::env::var("LATTICE_MODEL_CACHE") {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var("HOME").map_err(|_| {
        error::InferenceError::ModelNotFound(
            "unable to determine cache directory; HOME and LATTICE_MODEL_CACHE are unset".into(),
        )
    })?;
    Ok(PathBuf::from(home).join(".lattice").join("models"))
}

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
pub use crate::generation::GenerateConfig;
/// Result of a generation request: text, token ids, counts and stop reason. Canonical path
/// under ADR-092. See [`generation`] and [`StopReason`].
pub use crate::generation::GenerateOutput;
/// Per-token log-probability data for one generated token. See [`GenerateConfig::logprobs`].
pub use crate::generation::TokenLogprob;
/// One alternative token and its log-probability at a single step. See [`TokenLogprob`].
pub use crate::generation::TopLogprob;
/// Grammar-constrained decoding engine, reachable from the root because
/// [`GenerateConfig::grammar`] carries one. See [`grammar`].
pub use crate::grammar::GrammarEngine;
/// BERT encoder configuration. See [`BertModel`] and [`model`].
pub use crate::model::BertConfig;
/// BERT/BGE encoder model. See [`BertConfig`], [`Tokenizer`], and [`BertPooling`].
pub use crate::model::BertModel;
/// BERT-style cross-encoder/reranker model. See [`BertModel`] and [`model`].
pub use crate::model::CrossEncoderModel;
/// Per-layer profiling data collected during Qwen embedding inference. See [`ProfileTimings`]
/// and [`QwenModel`].
pub use crate::model::LayerTimings;
/// Aggregate profiling report for Qwen inference. See [`LayerTimings`] and [`QwenModel`].
pub use crate::model::ProfileTimings;
/// Qwen embedding model configuration. See [`QwenModel`] and [`weights`].
pub use crate::model::QwenConfig;
/// Qwen embedding model exposing `encode` for producing embeddings. See [`QwenConfig`],
/// [`Tokenizer`], and [`weights`].
pub use crate::model::QwenModel;
/// BERT pooling strategy selector (mean or CLS). See [`pool`] and [`BertModel`].
pub use crate::pool::BertPooling;
/// Reason a generation request stopped (e.g. EOS, max tokens). See [`stop_reason`] and
/// [`model`].
pub use crate::stop_reason::StopReason;
/// Byte-level BPE tokenizer used by Qwen-family models. See [`Tokenizer`] and [`TokenizedInput`].
pub use crate::tokenizer::BpeTokenizer;
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
/// `WordPiece` tokenizer used by BERT-family models. See [`Tokenizer`] and [`BertModel`].
pub use crate::tokenizer::WordPieceTokenizer;
/// Model-directory tokenizer auto-loader. See [`Tokenizer`] and [`tokenizer`].
pub use crate::tokenizer::load_tokenizer;
/// `tokenizer.json`-text tokenizer loader (no filesystem access). See
/// [`Tokenizer`], [`tokenizer`], and [`BertModel::from_bytes`].
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

/// Reads a boolean engine switch from the environment, by VALUE.
///
/// These switches used to be selected by presence alone, so `LATTICE_MTP_BATCH=0`
/// turned the batch verifier on while every document and comment spells the flag
/// `=1`. An operator writing `=0` to mean "run the default arm" measured the
/// non-default arm twice and saw a clean-looking negative.
///
/// A value spelling "off" now disables the switch: empty, `0`, `false`, `no` or
/// `off`, in any case, with surrounding whitespace ignored. Every other value
/// enables it, so `=1`, `=true` and `=yes` all keep working, and so does any value
/// a caller was already passing to mean "on". An absent variable is off, and a
/// value that is not valid Unicode cannot be one of the off spellings, so it
/// enables, which is what presence-checking did.
///
/// One `var_os` lookup and a few ASCII comparisons against short literals: no
/// allocation, because one of these sits inside a per-round decode loop.
///
/// Gated like its callers (see `check_mtp_not_requested`) so a non-metal-gpu build
/// does not carry an unused function.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(crate) fn env_switch_enabled(name: &str) -> bool {
    match std::env::var_os(name) {
        None => false,
        Some(raw) => match raw.to_str() {
            Some(text) => switch_value_enabled(Some(text)),
            None => true,
        },
    }
}

/// The value half of [`env_switch_enabled`], separated so it is testable without
/// mutating the process environment, which no test can do without racing every
/// other test in the binary.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(crate) fn switch_value_enabled(value: Option<&str>) -> bool {
    const OFF: [&str; 4] = ["0", "false", "no", "off"];
    match value {
        None => false,
        Some(raw) => {
            let text = raw.trim();
            !(text.is_empty() || OFF.iter().any(|off| text.eq_ignore_ascii_case(off)))
        }
    }
}

#[cfg(test)]
mod env_switch_tests {
    use super::switch_value_enabled;

    #[test]
    fn absent_is_off() {
        assert!(!switch_value_enabled(None));
    }

    #[test]
    fn documented_on_spellings_enable() {
        for raw in ["1", "true", "TRUE", "yes", "on", " 1 ", "2", "batch"] {
            assert!(switch_value_enabled(Some(raw)), "{raw:?} should enable");
        }
    }

    #[test]
    fn off_spellings_disable_which_presence_checking_could_not() {
        for raw in ["0", "false", "FALSE", "no", "off", "", "  ", " 0 "] {
            assert!(!switch_value_enabled(Some(raw)), "{raw:?} should disable");
        }
    }
}
