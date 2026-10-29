//! GatedDeltaNet linear attention, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path. The kernels live in
//! `lattice_compute::attention::gdn` and take a validated [`GdnGeometry`]; the steps that
//! build one from a Qwen3.5 configuration live in `lattice_model_qwen35::attention::gdn`.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-compute` and
//! `lattice-model-qwen35` so that code written against the pre-split `lattice-inference` keeps
//! compiling. These low-level re-exports are not a stable API and will be removed at 1.0; import
//! from `lattice-compute` and `lattice-model-qwen35` directly.

#[doc(inline)]
pub use lattice_compute::attention::gdn::{
    GatedDeltaNetScratch, GatedDeltaNetState, GatedDeltaNetWeights, GdnDimensions, GdnGeometry,
    GdnLayerSnapshot, GdnSnapshot, gated_rms_norm, l2_normalize_vec, sigmoid, softplus,
};

#[cfg(feature = "qwen35")]
#[doc(inline)]
pub use lattice_model_qwen35::attention::gdn::{gated_delta_net_step, qwen35_gdn_geometry};
