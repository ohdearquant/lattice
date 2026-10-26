//! ERNIE4.5 CPU items, owned by the ERNIE4.5 family.

#[doc(inline)]
pub use lattice_model_ernie45::model::ernie45::{
    Ernie45Config, Ernie45KvCache, Ernie45LayerWeights, Ernie45Model, Ernie45RopeScaling,
    Ernie45Trace, Ernie45Weights, MAX_HEAD_DIM, MAX_HEADS, MAX_HIDDEN_SIZE, MAX_INTERMEDIATE_SIZE,
    MAX_KV_HEADS, MAX_LAYERS, MAX_SEQ_LEN, MAX_VOCAB_SIZE,
};
