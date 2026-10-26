//! Fused GatedDeltaNet decode step, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::gdn_fused`.

#[doc(inline)]
pub use lattice_compute::attention::gdn_fused::{
    GatedDeltaNetFusedScratch, conv1d_silu_fused, gated_delta_net_step_fused,
    simd_decay_and_rank1_update, simd_gated_rms_norm, simd_l2_normalize, simd_matvec_transpose,
};
