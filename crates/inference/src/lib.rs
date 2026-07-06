//! **Stability tier**: Experimental
//!
//! This is a pure ML inference kernel with high churn, 153 `unsafe` blocks, and 22
//! `dead_code_allows`. It is NOT intended for direct use by platform or feature crates.
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
/// and [`generate`].
pub mod batch;
/// Model-file cache and conditional download helpers. See [`model`] and [`weights`].
pub mod download;
/// Crate error taxonomy; see [`InferenceError`].
pub mod error;
/// Generic text generation loop and cache-backed forward path. See [`sampling`],
/// [`kv_cache`], and [`grammar`].
pub mod generate;
/// Grammar-constrained decoding and logit masking. See [`generate`] and [`sampling`].
pub mod grammar;
/// Flat and paged key/value cache implementations. See [`generate`] and [`forward`].
pub mod kv_cache;
/// LoRA adapter hook called from inference forward paths. See [`model`] and [`forward`].
pub mod lora_hook;
/// Inference metrics and entropy accumulation. See [`model`].
pub mod metrics;
/// Adapter routing and mixture support built on top of [`lora_hook`] and [`sampling`].
/// Requires the `mixture` feature.
#[cfg(feature = "mixture")]
pub mod mixture;
/// Embedding pooling helpers (mean, CLS, last-token) including [`BertPooling`]. Used by
/// [`model::BertModel`] and [`model::QwenModel`].
pub mod pool;
/// ShortGPT-style block influence scoring. See [`model`].
pub mod pruning;
/// Quantization and pre-transform primitives. See [`weights`] and [`forward`].
pub mod quant;
/// Rotary position embedding tables and application helpers. See [`model`] and [`forward`].
pub mod rope;
/// Sampling configuration and token selection helpers. See [`generate`] and [`speculative`].
pub mod sampling;
/// N-gram prompt lookup speculative decoding. See [`sampling`] and [`generate`].
pub mod speculative;
/// Generation stop reason taxonomy; see [`StopReason`] and [`generate`].
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
/// [`generate`].
pub use crate::stop_reason::StopReason;
/// Byte-level BPE tokenizer used by Qwen-family models. See [`Tokenizer`] and [`TokenizedInput`].
pub use crate::tokenizer::BpeTokenizer;
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
