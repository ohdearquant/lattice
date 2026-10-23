//! Model-directory format detection, re-exported from `lattice-weights`.
//!
//! **Stability**: unstable, internal-binaries-only; see `lattice_weights::model_format` for the full
//! rationale. The items keep the paths they had before the module moved.

#[doc(inline)]
pub use lattice_weights::model_format::{
    FormatDetection, ModelFamily, ModelFormat, detect_family, detect_format, inspect_format,
    metal_gpu_required_message, unloadable_format_message, unrecognized_format_message,
};

#[cfg(all(test, not(feature = "metal-gpu")))]
mod tests;
