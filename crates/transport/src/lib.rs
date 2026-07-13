//! Sinkhorn optimal transport for embedding distribution drift detection.
//!
//! **Stability tier: Unstable.** The Sinkhorn API is functional but currently
//! consumed by only one crate; its shape may still change.
//!
//! This crate implements entropy-regularized optimal transport (the Sinkhorn-Knopp
//! algorithm) in the log domain, for quantifying how much embedding geometry shifts
//! between model versions (for example, BGE-small to mE5-small). The load-bearing
//! numerical invariant is that the solvers stay in log space throughout and never
//! materialize the Gibbs kernel `exp(-C/epsilon)`, which prevents underflow at small
//! regularization epsilon. The crate is a dependency-free leaf: pure-Rust math with
//! no BLAS/LAPACK, and [`SinkhornWorkspace`] preallocates the inner-loop buffers so
//! callers allocate once and reuse.
//!
//! The layers run from the numerical foundation ([`logsumexp`]) up through the
//! [`cost`] abstraction, the core solvers ([`sinkhorn`], [`sinkhorn_log`],
//! [`unbalanced`]), post-processing ([`transport_plan`], [`divergence`]), and the
//! high-level [`drift`] and [`barycenter`] APIs. See
//! [`docs/design.md`](https://github.com/ohdearquant/lattice/blob/main/crates/transport/docs/design.md)
//! for the architecture and design rationale in full.
//!
//! # Example
//!
//! ```rust
//! use lattice_transport::{
//!     SinkhornSolver, SinkhornConfig, SinkhornWorkspace,
//!     DenseCostMatrix, uniform_weights,
//! };
//!
//! let cost = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
//! let source = uniform_weights(2);
//! let target = uniform_weights(2);
//! let solver = SinkhornSolver::default();
//! let mut workspace = SinkhornWorkspace::new(2, 2);
//! let result = solver.solve(&cost, &source, &target, &mut workspace).unwrap();
//! assert!(result.converged);
//! ```

#![warn(missing_docs)]
#![allow(clippy::uninlined_format_args)]

pub mod barycenter;
pub mod cost;
pub mod divergence;
pub mod drift;
pub mod logsumexp;
mod math;
pub mod online_drift;
pub mod sinkhorn;
pub mod sinkhorn_log;
pub mod transport_plan;
pub mod unbalanced;

#[cfg(test)]
mod bench;
#[cfg(test)]
mod tests;

// Core solver types
pub use sinkhorn::{
    BatchSolveResult, MarginalPair, ProgressObserver, ProgressState, SinkhornConfig, SinkhornError,
    SinkhornResult, SinkhornSolver, SinkhornWorkspace, normalize_weights, uniform_weights,
};

// Cost abstractions
pub use cost::{
    ClosureCostMatrix, ContiguousPoints, CosineDistance, CostError, CostMatrix, DenseCostMatrix,
    PairwiseCostMatrix, PointMetric, PointSet, SquaredEuclidean,
};

// Log-domain solver with epsilon scaling
pub use sinkhorn_log::{
    EpsilonScalingSchedule, EpsilonStageSummary, LogDomainSinkhornConfig, LogDomainSinkhornResult,
    LogDomainSinkhornSolver,
};

// Transport plan extraction
pub use transport_plan::{
    SparseTransportEntry, SparseTransportPlan, extract_sparse_plan, log_gamma,
};

// Sinkhorn divergence (debiased)
pub use divergence::{SinkhornDivergence, point_set_sinkhorn_divergence, sinkhorn_divergence};

// Unbalanced OT
pub use unbalanced::{UnbalancedConfig, UnbalancedResult, UnbalancedSinkhornSolver};

// Barycenter computation
pub use barycenter::{
    BarycenterConfig, BarycenterProblem, FixedSupportBarycenter, FixedSupportBarycenterWorkspace,
    FreeSupportBarycenter, FreeSupportConfig, OwnedPointMeasure, fixed_support_barycenter,
    free_support_barycenter,
};

// Drift detection API
pub use drift::{
    DriftConfig, DriftMetricKind, DriftReport, DriftSolverMode, DriftSummary, DriftWeighting,
    EmbeddingRecord, MemoryLike, PerEntryDisplacement, detect_drift_memories, detect_drift_records,
};

// Online streaming drift detection
pub use online_drift::{OnlineDriftConfig, OnlineDriftDetector, OnlineDriftSignal};
