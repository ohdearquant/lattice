//! Walsh-Hadamard transform and randomized Hadamard generators, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.

#[doc(inline)]
pub use lattice_quant::quarot::hadamard::{
    BlockHadamard, RandomizedHadamard, walsh_hadamard_f64_in_place, walsh_hadamard_in_place,
    walsh_hadamard_orthonormal_f64_in_place, walsh_hadamard_orthonormal_in_place,
};
