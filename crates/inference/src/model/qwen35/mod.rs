//! Qwen3.5 family compatibility exports.

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
