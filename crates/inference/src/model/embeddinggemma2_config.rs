//! EmbeddingGemma 2 configuration facade over `lattice-arch`.
//!
//! Every public item keeps the path it had before the configuration moved into its own crate. The
//! re-exports are named rather than globbed so the facade never widens the supported API.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-arch` so that code written
//! against the pre-split `lattice-inference` keeps compiling. These low-level re-exports are not a
//! stable API and will be removed at 1.0; import from `lattice-arch` directly.

#[doc(inline)]
pub use lattice_arch::embeddinggemma2_config::{
    EmbeddingGemma2Config, EmbeddingGemma2LayerKind, EmbeddingGemma2LayerShape,
};
