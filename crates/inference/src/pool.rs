//! BERT-family pooling, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in `lattice_compute::pool`.

#[doc(inline)]
pub use lattice_compute::pool::{
    BertPooling, cls_pool, default_bert_pooling_for_model_name, l2_normalize, last_token_pool,
    mean_pool,
};
