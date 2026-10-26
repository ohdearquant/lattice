//! Gemma4 CPU model, re-exported from its family crate.

#[doc(inline)]
pub use lattice_model_gemma4::gemma4_model::Gemma4Model;
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub use lattice_model_gemma4::gemma4_model::tiny_zero_model_with_vocab;
