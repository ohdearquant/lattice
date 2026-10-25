//! The f64 forward-equivalence check between an original and a QuaRot-rotated Qwen3.5 checkpoint, re-exported from
//! `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.

#[doc(inline)]
pub use lattice_quant::quarot::forward_equivalence::{
    ForwardEquivalenceConfig, ForwardEquivalenceReport, assert_forward_equivalence_qwen35,
};
