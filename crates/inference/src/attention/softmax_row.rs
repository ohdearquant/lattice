//! Fail-closed softmax row contract, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::softmax_row`.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-compute` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-compute` directly.

#[doc(inline)]
pub use lattice_compute::attention::softmax_row::{
    finalize_row, is_masked_neg_inf, row_fails_closed_pre_exp, row_max_and_any_nan,
};
