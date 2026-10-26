//! EmbeddingGemma 2 configuration facade over `lattice-arch`.
//!
//! Every public item keeps the path it had before the configuration moved into its own crate. The
//! re-exports are named rather than globbed so the facade never widens the supported API.

#[doc(inline)]
pub use lattice_arch::embeddinggemma2_config::{
    EmbeddingGemma2Config, EmbeddingGemma2LayerKind, EmbeddingGemma2LayerShape,
};
