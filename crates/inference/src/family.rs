//! The family contract of `lattice-arch` (ADR-103 Amendment 21 item 3) and the descriptors for
//! the model families this crate loads.
//!
//! **Unstable**: `QWEN35` (when `qwen35` is enabled) offers a Qwen3.5 decoder on the CPU through
//! `Qwen35CpuDecoder`; `BERT` (when `bert` is enabled) and
//! `EMBEDDING_GEMMA2` (when `embeddinggemma2` is enabled) offer the CPU encoders of
//! [`crate::encoder`]. [`registry`](crate::family::registry) builds their available descriptors. The
//! Qwen3 embedding model has no descriptor yet: its loader chooses the Metal backend itself, so a
//! [`LoadRequest`](crate::family::LoadRequest) cannot fix its backend.

pub use lattice_arch::decoder::{DecoderModel, SessionRequest, TextDecoder, TextStream};
pub use lattice_arch::family::{
    Backend, DecoderLoader, EncoderLoader, FamilyDescriptor, LoadRequest, ModelKind, Registry,
    RegistryError, ResolveKey,
};

#[cfg(all(test, feature = "qwen35"))]
use crate::error::InferenceError;

#[cfg(feature = "qwen35")]
#[doc(inline)]
pub use lattice_model_qwen35::family::{QWEN35, Qwen35CpuDecoder};

/// BERT-family text embeddings on the CPU.
#[cfg(feature = "bert")]
#[doc(inline)]
pub use lattice_model_bert::family::BERT;

#[cfg(feature = "embeddinggemma2")]
#[doc(inline)]
pub use lattice_model_embeddinggemma2::family::EMBEDDING_GEMMA2;

/// A registry of the enabled decoder and encoder families.
///
/// # Errors
///
/// None in practice: the available descriptors share no name and no claim. The `Result` is
/// [`Registry::new`]'s.
pub fn registry() -> Result<Registry, RegistryError> {
    Registry::new(&[
        #[cfg(feature = "qwen35")]
        &QWEN35,
        #[cfg(feature = "bert")]
        &BERT,
        #[cfg(feature = "embeddinggemma2")]
        &EMBEDDING_GEMMA2,
    ])
}

#[cfg(test)]
mod tests;
