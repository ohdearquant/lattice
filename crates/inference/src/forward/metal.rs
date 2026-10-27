//! Qwen3 Metal backend, re-exported from its family crate.
//!
//! **Compatibility path.** The items here are re-exported from
//! `lattice-model-qwen3-embedding` so that code written against the pre-split
//! `lattice-inference` keeps compiling. These low-level re-exports are not a stable API and
//! will be removed at 1.0; import from `lattice-model-qwen3-embedding` directly.

#[cfg(feature = "qwen3")]
#[doc(inline)]
pub use lattice_model_qwen3_embedding::metal::MetalForwardPass;
