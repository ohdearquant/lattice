//! Walsh-Hadamard transform and randomized Hadamard generators, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-quant` so that code written
//! against the pre-split `lattice-inference` keeps compiling. These low-level re-exports are not a
//! stable API and will be removed at 1.0; import from `lattice-quant` directly.

#[doc(inline)]
pub use lattice_quant::quarot::hadamard::{
    BlockHadamard, RandomizedHadamard, walsh_hadamard_f64_in_place, walsh_hadamard_in_place,
    walsh_hadamard_orthonormal_f64_in_place, walsh_hadamard_orthonormal_in_place,
};
