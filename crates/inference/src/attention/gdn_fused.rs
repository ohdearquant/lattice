//! Fused GatedDeltaNet decode step, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path. The fused kernel lives in
//! `lattice_compute::attention::gdn_fused`; the step that builds its geometry from a Qwen3.5
//! configuration lives in `lattice_model_qwen35::attention::gdn_fused`.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-compute` and
//! `lattice-model-qwen35` so that code written against the pre-split `lattice-inference` keeps
//! compiling. These low-level re-exports are not a stable API and will be removed at 1.0; import
//! from `lattice-compute` and `lattice-model-qwen35` directly.

#[doc(inline)]
pub use lattice_compute::attention::gdn_fused::{
    GatedDeltaNetFusedScratch, conv1d_silu_fused, simd_decay_and_rank1_update, simd_gated_rms_norm,
    simd_l2_normalize, simd_matvec_transpose,
};

#[cfg(feature = "qwen35")]
#[doc(inline)]
pub use lattice_model_qwen35::attention::gdn_fused::gated_delta_net_step_fused;
