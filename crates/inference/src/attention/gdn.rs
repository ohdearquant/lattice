//! GatedDeltaNet linear attention, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::gdn`.

#[doc(inline)]
pub use lattice_compute::attention::gdn::{
    GatedDeltaNetScratch, GatedDeltaNetState, GatedDeltaNetWeights, GdnLayerSnapshot, GdnSnapshot,
    gated_delta_net_step, gated_rms_norm, l2_normalize_vec, sigmoid, softplus,
};
