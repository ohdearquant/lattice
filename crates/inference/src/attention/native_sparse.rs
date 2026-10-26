//! Native sparse attention, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::native_sparse`.

#[doc(inline)]
pub use lattice_compute::attention::native_sparse::{
    NsaConfig, NsaScratch, NsaWeights, apply_native_sparse_attention,
};
