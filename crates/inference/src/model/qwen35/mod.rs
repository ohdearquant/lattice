//! Qwen3.5 family compatibility exports.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-qwen35` and
//! `lattice-tokenizer` so that code written against the pre-split `lattice-inference` keeps
//! compiling. These low-level re-exports are not a stable API and will be removed at 1.0; import
//! from `lattice-model-qwen35` and `lattice-tokenizer` directly.

#[doc(inline)]
pub use lattice_model_qwen35::model::qwen35::HiddenPooling;

#[doc(inline)]
pub use lattice_model_qwen35::model::qwen35::{PerplexityConfig, PerplexityReport, PrefillInput};

#[cfg(any(test, feature = "test-utils"))]
#[doc(inline)]
pub use lattice_model_qwen35::model::qwen35::test_support;

#[doc(inline)]
pub use lattice_model_qwen35::model::qwen35::Qwen35Model;

#[doc(inline)]
pub use lattice_model_qwen35::model::qwen35::ModelWeights;

#[cfg(test)]
#[doc(inline)]
pub use lattice_tokenizer::__internal::bytes_to_unicode;

#[doc(inline)]
pub use lattice_model_qwen35::model::qwen35::RawGenEvent;

#[doc(inline)]
pub use lattice_model_qwen35::model::qwen35::{
    qwen_layer_tensor_prefix, qwen_required_tensor_names,
};
