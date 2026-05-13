//! **Stability tier**: Unstable
//!
//! Extracted from `foundation/score` in Wave4. The Sinkhorn API is functional but
//! not yet consumed by more than one crate. API shape may change as Brain Phase 7
//! Objective synthesis adopts it. Promote to Stable after second consumer lands.
//! See `foundation/STABILITY.md` for the full policy.
//!
//! Sinkhorn Optimal Transport for embedding distribution drift detection.
//!
//! This crate implements entropy-regularized optimal transport (Sinkhorn algorithm)
//! in log-domain for numerical stability. It is designed for quantifying how much
//! embedding geometry shifts between model versions (e.g., BGE-small to mE5-small).
//!
//! Extracted from `foundation/score` as a peer crate so that Brain Phase 7
//! Objective synthesis can consume optimal-transport primitives independently
//! of scoring infrastructure.
//!
//! # Architecture
//!
//! The crate is organized into layers:
//!
//! - **Numerical foundation**: [`math`], [`logsumexp`] -- stable log-domain arithmetic
//! - **Cost abstraction**: [`cost`] -- cost matrix traits and implementations
//! - **Core solvers**: [`sinkhorn`] (balanced), [`sinkhorn_log`] (epsilon-scaling),
//!   [`unbalanced`] (KL-relaxed marginals)
//! - **Post-processing**: [`transport_plan`] (sparse plan extraction),
//!   [`divergence`] (debiased Sinkhorn divergence)
//! - **High-level API**: [`drift`] (embedding drift detection),
//!   [`barycenter`] (Wasserstein barycenters)
//!
//! # Key Design Choices
//!
//! - **Log-domain throughout**: Never materializes the Gibbs kernel `exp(-C/epsilon)`
//!   directly. All scaling vectors stay in log space.
//! - **Pre-allocated buffers**: [`SinkhornWorkspace`] avoids heap allocation during
//!   the inner iteration loop.
//! - **No external dependencies**: Pure Rust math, no BLAS/LAPACK requirement.
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
#![warn(clippy::all)]
#![allow(clippy::uninlined_format_args)]

pub mod barycenter;
pub mod cost;
pub mod divergence;
pub mod drift;
pub mod logsumexp;
mod math;
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
