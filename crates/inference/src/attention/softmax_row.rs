//! Fail-closed softmax row contract, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::softmax_row`.

#[doc(inline)]
pub use lattice_compute::attention::softmax_row::{
    finalize_row, is_masked_neg_inf, row_fails_closed_pre_exp, row_max_and_any_nan,
};
