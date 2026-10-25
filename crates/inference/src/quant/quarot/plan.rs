//! The QuaRot rotation plan: which rotation each tensor receives and on which side, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.

#[doc(inline)]
pub use lattice_quant::quarot::plan::{
    AbsorptionSide, CoverageReport, OnlineRotationSpec, OnlineTransformSite, RotationId,
    RotationPlan, RuleRequirement, TensorRotation, apply_tensor_rotation,
    apply_tensor_rotation_f64, qwen3_required_tensor_names,
};
