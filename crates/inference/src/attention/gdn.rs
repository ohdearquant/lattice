//! GatedDeltaNet linear attention, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::gdn`. The kernels there take a validated [`GdnGeometry`];
//! this module builds one from a Qwen3.5 configuration and keeps the configuration-taking
//! step under its existing name.

use crate::error::InferenceError;
use crate::model::qwen35_config::Qwen35Config;

#[doc(inline)]
pub use lattice_compute::attention::gdn::{
    GatedDeltaNetScratch, GatedDeltaNetState, GatedDeltaNetWeights, GdnDimensions, GdnGeometry,
    GdnLayerSnapshot, GdnSnapshot, gated_rms_norm, l2_normalize_vec, sigmoid, softplus,
};

/// **Unstable**: the GatedDeltaNet layer geometry a Qwen3.5 configuration describes.
///
/// Reads `hidden_size`, the `linear_*` head counts, head widths and conv kernel length, and
/// `rms_norm_eps`. Returns the error [`GdnGeometry::new`] reports when they do not form a valid
/// geometry.
pub fn qwen35_gdn_geometry(cfg: &Qwen35Config) -> Result<GdnGeometry, InferenceError> {
    GdnGeometry::new(GdnDimensions {
        hidden_size: cfg.hidden_size,
        num_key_heads: cfg.linear_num_key_heads,
        num_value_heads: cfg.linear_num_value_heads(),
        key_head_dim: cfg.linear_key_head_dim,
        value_head_dim: cfg.linear_value_head_dim,
        conv_kernel_dim: cfg.linear_conv_kernel_dim,
        rms_norm_eps: cfg.rms_norm_eps,
    })
}

/// [`qwen35_gdn_geometry`] for callers that have no error path.
///
/// # Panics
///
/// Panics with the geometry error if `cfg` does not describe a valid GatedDeltaNet geometry.
pub(crate) fn require_qwen35_gdn_geometry(cfg: &Qwen35Config) -> GdnGeometry {
    match qwen35_gdn_geometry(cfg) {
        Ok(geometry) => geometry,
        Err(err) => panic!("{err}"),
    }
}

/// **Unstable**: scalar GatedDeltaNet single-token step for a Qwen3.5 layer.
///
/// Builds the layer geometry from `cfg` and runs
/// `lattice_compute::attention::gdn::gated_delta_net_step` with it.
///
/// `input`: hidden state `[hidden_size]`
/// `state`: mutable recurrent state for this layer
/// `weights`: layer weights
/// `cfg`: model config
/// `scratch`: reusable scratch buffers
/// `output`: output buffer `[hidden_size]`, written in-place
///
/// # Panics
///
/// Panics if `cfg` does not describe a valid GatedDeltaNet geometry; [`qwen35_gdn_geometry`]
/// reports the same condition as an error.
#[inline]
pub fn gated_delta_net_step(
    input: &[f32],
    state: &mut GatedDeltaNetState,
    weights: &GatedDeltaNetWeights,
    cfg: &Qwen35Config,
    scratch: &mut GatedDeltaNetScratch,
    output: &mut [f32],
) {
    lattice_compute::attention::gdn::gated_delta_net_step(
        input,
        state,
        weights,
        &require_qwen35_gdn_geometry(cfg),
        scratch,
        output,
    );
}

#[cfg(test)]
mod tests;
