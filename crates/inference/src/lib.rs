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
pub mod attention;
pub mod forward;
pub mod model;
pub mod tokenizer;
pub mod weights;

// Standalone modules
pub mod download;
pub mod error;
pub mod generate;
pub mod kv_cache;
pub mod lora_hook;
pub mod pool;
pub mod quant;
pub mod rope;
pub mod sampling;
pub mod speculative;

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
pub use crate::error::InferenceError;
pub use crate::model::{
    BertConfig, BertModel, CrossEncoderModel, LayerTimings, ProfileTimings, QwenConfig, QwenModel,
};
pub use crate::tokenizer::{
    BpeTokenizer, SentencePieceTokenizer, TokenizedInput, Tokenizer, WordPieceTokenizer,
    load_tokenizer,
};
