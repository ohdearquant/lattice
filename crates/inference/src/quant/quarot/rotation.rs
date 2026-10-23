//! Absorption of a [`RandomizedHadamard`](super::hadamard::RandomizedHadamard) rotation into a
//! linear-layer weight matrix, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.

#[doc(inline)]
pub use lattice_quant::quarot::rotation::{
    absorb_input_rotation, absorb_input_rotation_f64, absorb_output_rotation,
    absorb_output_rotation_f64,
};
