//! Log-domain Sinkhorn optimal transport for embedding distribution drift.
//!
//! The solver never materializes `exp(-C/epsilon)` and reuses
//! [`SinkhornWorkspace`] buffers. Extended background: [design] and [algorithms].
//!
//! [design]: https://github.com/ohdearquant/lattice/blob/main/crates/transport/docs/design.md
//! [algorithms]: https://github.com/ohdearquant/lattice/blob/main/crates/transport/docs/algorithms.md
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
mod reference_cases;
#[cfg(test)]
mod tests;

pub use sinkhorn::{
    BatchSolveResult, MarginalPair, ProgressObserver, ProgressState, SinkhornConfig, SinkhornError,
    SinkhornResult, SinkhornSolver, SinkhornWorkspace, normalize_weights, uniform_weights,
};

pub use cost::{
    ClosureCostMatrix, ContiguousPoints, CosineDistance, CostError, CostMatrix, DenseCostMatrix,
    PairwiseCostMatrix, PointMetric, PointSet, SquaredEuclidean,
};

pub use sinkhorn_log::{
    EpsilonScalingSchedule, EpsilonStageSummary, LogDomainSinkhornConfig, LogDomainSinkhornResult,
    LogDomainSinkhornSolver,
};

pub use transport_plan::{
    SparseTransportEntry, SparseTransportPlan, extract_sparse_plan, log_gamma,
};

pub use divergence::{SinkhornDivergence, point_set_sinkhorn_divergence, sinkhorn_divergence};

pub use unbalanced::{UnbalancedConfig, UnbalancedResult, UnbalancedSinkhornSolver};

pub use barycenter::{
    BarycenterConfig, BarycenterProblem, FixedSupportBarycenter, FixedSupportBarycenterWorkspace,
    FreeSupportBarycenter, FreeSupportConfig, OwnedPointMeasure, fixed_support_barycenter,
    free_support_barycenter,
};

pub use drift::{
    DriftConfig, DriftMetricKind, DriftReport, DriftSolverMode, DriftSummary, DriftWeighting,
    EmbeddingRecord, MemoryLike, PerEntryDisplacement, detect_drift_memories, detect_drift_records,
};

pub use online_drift::{OnlineDriftConfig, OnlineDriftDetector, OnlineDriftSignal};
