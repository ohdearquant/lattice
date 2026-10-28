//! Fusion of RMSNorm scales into the following linear layers for QuaRot, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-quant` so that code written
//! against the pre-split `lattice-inference` keeps compiling. These low-level re-exports are not a
//! stable API and will be removed at 1.0; import from `lattice-quant` directly.

#[doc(inline)]
pub use lattice_quant::quarot::rmsnorm_fusion::{
    RmsNormFusionTarget, fuse_shifted_rmsnorm_into_next_layer_f64, neutralize_rmsnorm_gamma_f64,
    qwen35_per_layer_fusion_plan,
};
