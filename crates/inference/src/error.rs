//! Inference error taxonomy.
//!
//! [`InferenceError`] is defined by `lattice-inference-core` and shared by the
//! facade and the extracted inference crates.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-inference-core` so that
//! code written against the pre-split `lattice-inference` keeps compiling. These low-level
//! re-exports are not a stable API and will be removed at 1.0; import from `lattice-inference-core`
//! directly.

#[doc(inline)]
pub use lattice_inference_core::InferenceError;
