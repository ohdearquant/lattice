//! Rotation of LoRA adapters into a QuaRot-rotated basis, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-quant` so that code written
//! against the pre-split `lattice-inference` keeps compiling. These low-level re-exports are not a
//! stable API and will be removed at 1.0; import from `lattice-quant` directly.

#[doc(inline)]
pub use lattice_quant::quarot::lora::{
    LoraLayerMut, LoraRotationApplied, RotationReport, counter_rotate_a, needs_b_rotation,
    needs_counter_rotation, rotate_adapter_for_quarot, rotate_b_output_side,
};
