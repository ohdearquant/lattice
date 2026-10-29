//! Gemma4 CPU model, re-exported from its family crate.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-gemma4` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-gemma4` directly.

#[doc(inline)]
pub use lattice_model_gemma4::gemma4_model::Gemma4Model;
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub use lattice_model_gemma4::gemma4_model::tiny_zero_model_with_vocab;
