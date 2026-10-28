//! BERT model and weight-loading paths retained by the inference facade.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-bert` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-bert` directly.

#[doc(inline)]
pub use lattice_model_bert::bert::{BertConfig, BertModel, weight_loading};
