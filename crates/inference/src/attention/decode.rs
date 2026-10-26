//! Single-token decode attention, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::decode`.

#[doc(inline)]
pub use lattice_compute::attention::decode::{decode_attention, decode_attention_scores};
