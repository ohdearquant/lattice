//! Model-file ensure/download flow, canonical model names, checksum verification, and download helper.
//!
//! The implementation lives in `lattice-model-io` (ADR-103 Amendment 20); this module keeps the
//! public path.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-io` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-io` directly.

#[doc(inline)]
pub use lattice_model_io::download::ensure_model_files;
