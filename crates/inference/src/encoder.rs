//! The text encoder contract of `lattice-arch` (ADR-103 Amendment 21 item 3) and its adapters
//! for the embedding models this crate loads: BERT-family, Qwen3 and EmbeddingGemma 2.
//!
//! **Unstable**: each adapter wraps a loaded model in an `Arc` and is shared as
//! `Arc<dyn Encoder>`. It reads the model's settings once, when it is built, and reports
//! [`Provenance::Unavailable`](crate::encoder::Provenance::Unavailable) for the checkpoint and
//! the tokenizer: these models record no revision or digest of their files.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-arch`,
//! `lattice-model-bert`, `lattice-model-embeddinggemma2` and
//! `lattice-model-qwen3-embedding` so that code written against the pre-split
//! `lattice-inference` keeps compiling. These low-level re-exports are not a stable API and
//! will be removed at 1.0; import from `lattice-arch`, `lattice-model-bert`,
//! `lattice-model-embeddinggemma2` and `lattice-model-qwen3-embedding` directly.

pub use lattice_arch::encoder::{
    EncodedText, Encoder, EncoderCapabilities, EncoderIdentity, Normalization, Pooling, Provenance,
    TokenUsage,
};

/// BERT-family encoder adapter.
#[cfg(feature = "bert")]
#[doc(inline)]
pub use lattice_model_bert::BertEncoder;

/// Qwen3 encoder adapter.
#[cfg(feature = "qwen3")]
#[doc(inline)]
pub use lattice_model_qwen3_embedding::QwenEncoder;

#[cfg(feature = "embeddinggemma2")]
#[doc(inline)]
pub use lattice_model_embeddinggemma2::encoder::EmbeddingGemma2Encoder;
