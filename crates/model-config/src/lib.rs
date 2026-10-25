//! Model configurations for Lattice inference: the parsed and validated `config.json` of the Qwen3.5
//! hybrid models, Gemma 4, EmbeddingGemma 2 and BitNet b1.58, with the validation bounds those
//! parsers enforce.
//!
//! `lattice-inference` re-exports the supported items at their original
//! `lattice_inference::model::{qwen35_config, gemma4_config, embeddinggemma2_config, bitnet_config}`
//! paths.

mod error {
    pub use lattice_inference_core::InferenceError;
}

pub mod bitnet_config;
pub mod embeddinggemma2_config;
pub mod gemma4_config;
pub mod qwen35_config;

/// Configuration glue used by the `lattice-inference` facade.
///
/// Not part of the supported API. The facade calls these free functions by their full path; they are
/// not re-exported by that facade.
#[doc(hidden)]
pub mod __internal {
    use lattice_inference_core::InferenceError;

    /// Checked doubling of a config-derived dimension, with the Qwen config error on overflow.
    pub fn checked_double(value: usize, what: &str) -> Result<usize, InferenceError> {
        crate::qwen35_config::checked_double(value, what)
    }
}
