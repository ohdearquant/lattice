//! Qwen3 Metal backend, re-exported from its family crate.

#[cfg(feature = "qwen3")]
#[doc(inline)]
pub use lattice_model_qwen3_embedding::metal::MetalForwardPass;
