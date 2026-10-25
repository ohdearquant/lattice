//! Fusion of RMSNorm scales into the following linear layers for QuaRot, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.

#[doc(inline)]
pub use lattice_quant::quarot::rmsnorm_fusion::{
    RmsNormFusionTarget, fuse_shifted_rmsnorm_into_next_layer_f64, neutralize_rmsnorm_gamma_f64,
    qwen35_per_layer_fusion_plan,
};
