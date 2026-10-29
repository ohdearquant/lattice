//! ERNIE4.5 CPU items, owned by the ERNIE4.5 family.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-ernie45` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-ernie45` directly.

#[doc(inline)]
pub use lattice_model_ernie45::model::ernie45::{
    Ernie45Config, Ernie45KvCache, Ernie45LayerWeights, Ernie45Model, Ernie45RopeScaling,
    Ernie45Trace, Ernie45Weights, MAX_HEAD_DIM, MAX_HEADS, MAX_HIDDEN_SIZE, MAX_INTERMEDIATE_SIZE,
    MAX_KV_HEADS, MAX_LAYERS, MAX_SEQ_LEN, MAX_VOCAB_SIZE,
};
