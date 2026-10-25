//! The QuaRot rotation pipeline over f64 tensors, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.

#[doc(inline)]
pub use lattice_quant::quarot::pipeline::{
    TensorEntry, absorb_rotations, fuse_rmsnorms, load_tensors_f64,
};
