//! Differential attention, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::differential`.

#[doc(inline)]
pub use lattice_compute::attention::differential::{
    DiffAttnConfig, DiffAttnScratch, DiffLambdaParams, apply_differential_attention,
    compute_lambda_full,
};
