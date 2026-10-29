//! The QuaRot rotation plan: which rotation each tensor receives and on which side, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-quant` so that code written
//! against the pre-split `lattice-inference` keeps compiling. These low-level re-exports are not a
//! stable API and will be removed at 1.0; import from `lattice-quant` directly.

#[doc(inline)]
pub use lattice_quant::quarot::plan::{
    AbsorptionSide, CoverageReport, OnlineRotationSpec, OnlineTransformSite, RotationId,
    RotationPlan, RotationTarget, RuleRequirement, TensorRotation, apply_tensor_rotation,
    apply_tensor_rotation_f64, qwen3_required_tensor_names, qwen35_residual_stream_linear_layers,
    qwen35_rotation_target,
};
