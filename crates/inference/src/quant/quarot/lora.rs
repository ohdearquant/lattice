//! Rotation of LoRA adapters into a QuaRot-rotated basis, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.

#[doc(inline)]
pub use lattice_quant::quarot::lora::{
    LoraLayerMut, LoraRotationApplied, RotationReport, counter_rotate_a, needs_b_rotation,
    needs_counter_rotation, rotate_adapter_for_quarot, rotate_b_output_side,
};
