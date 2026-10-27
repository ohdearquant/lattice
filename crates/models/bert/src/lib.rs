//! CPU BERT encoders and cross-encoder rerankers.
//!
//! The family owns its model configuration, loading, composition and encoder
//! descriptor. The 0.15 weight records remain in `lattice-weights` for compatibility.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

pub mod bert;
pub mod cross_encoder;
pub mod encoder;
pub mod family;
pub mod lora;

pub use bert::{BertConfig, BertModel};
pub use cross_encoder::CrossEncoderModel;
pub use encoder::BertEncoder;

/// Synthetic checkpoint support for repository regression tests.
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub mod bert_pooling_fixture;
