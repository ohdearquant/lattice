//! EmbeddingGemma 2 RGB input compatibility export.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-embeddinggemma2` so
//! that code written against the pre-split `lattice-inference` keeps compiling. These low-level
//! re-exports are not a stable API and will be removed at 1.0; import from
//! `lattice-model-embeddinggemma2` directly.

#[doc(inline)]
pub use lattice_model_embeddinggemma2::image::EmbeddingGemma2Rgb8Image;
