//! Model-directory format detection, re-exported from `lattice-weights`.
//!
//! **Stability**: unstable, internal-binaries-only; see `lattice_weights::model_format` for the full
//! rationale. The items keep the paths they had before the module moved.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-weights` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-weights` directly.

#[doc(inline)]
pub use lattice_weights::model_format::{
    FormatDetection, ModelFamily, ModelFormat, detect_family, detect_format, inspect_format,
    metal_gpu_required_message, unloadable_format_message, unrecognized_format_message,
};

#[cfg(test)]
mod tests;
