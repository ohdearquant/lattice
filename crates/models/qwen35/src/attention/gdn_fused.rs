//! Fused GatedDeltaNet decode step, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::gdn_fused`. The fused step there takes a validated
//! [`GdnGeometry`](crate::attention::gdn::GdnGeometry); this module keeps the
//! configuration-taking step under its existing name.

use crate::attention::gdn::{
    GatedDeltaNetState, GatedDeltaNetWeights, require_qwen35_gdn_geometry,
};
use lattice_arch::qwen35_config::Qwen35Config;
use lattice_compute::lora_hook::LoraHook;

#[doc(inline)]
pub use lattice_compute::attention::gdn_fused::{
    GatedDeltaNetFusedScratch, conv1d_silu_fused, simd_decay_and_rank1_update, simd_gated_rms_norm,
    simd_l2_normalize, simd_matvec_transpose,
};

/// **Unstable**: fused + SIMD GatedDeltaNet step for a Qwen3.5 layer; primary hot path.
///
/// Builds the layer geometry from `cfg` and runs
/// `lattice_compute::attention::gdn_fused::gated_delta_net_step_fused` with it.
///
/// `input`: hidden state `[hidden_size]`
/// `state`: mutable recurrent state for this layer
/// `weights`: layer weights
/// `cfg`: model config
/// `scratch`: reusable fused scratch buffers
/// `output`: output buffer `[hidden_size]`, written in-place
///
/// # Panics
///
/// Panics if `cfg` does not describe a valid GatedDeltaNet geometry;
/// [`qwen35_gdn_geometry`](crate::attention::gdn::qwen35_gdn_geometry) reports the same
/// condition as an error.
#[inline]
pub fn gated_delta_net_step_fused(
    input: &[f32],
    state: &mut GatedDeltaNetState,
    weights: &GatedDeltaNetWeights,
    cfg: &Qwen35Config,
    scratch: &mut GatedDeltaNetFusedScratch,
    output: &mut [f32],
    lora: &dyn LoraHook,
    layer_idx: usize,
) {
    lattice_compute::attention::gdn_fused::gated_delta_net_step_fused(
        input,
        state,
        weights,
        &require_qwen35_gdn_geometry(cfg),
        scratch,
        output,
        lora,
        layer_idx,
    );
}
