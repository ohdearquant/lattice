//! Inference metrics and entropy accumulation, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in `lattice_compute::metrics`.

#[doc(inline)]
pub use lattice_compute::metrics::{
    ForwardMetrics, LayerMetrics, MetricsMode, OnlineSoftmaxEntropy, l2_norm,
};
