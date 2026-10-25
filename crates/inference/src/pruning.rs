//! ShortGPT-style block influence scoring, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in `lattice_compute::pruning`.

#[doc(inline)]
pub use lattice_compute::pruning::{
    BlockInfluence, BlockInfluenceAccumulator, cosine_similarity, pruning_rank,
    score_from_hidden_states,
};
