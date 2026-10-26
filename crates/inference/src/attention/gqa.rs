//! Grouped-query attention, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::gqa`.

#[doc(inline)]
pub use lattice_compute::attention::gqa::{GqaConfig, GqaScratch, apply_gqa_attention};
