//! Debiased Sinkhorn divergence.
//!
//! Removes the entropy-induced self-cost of regularized transport. Extended
//! background: <https://github.com/ohdearquant/lattice/blob/main/crates/transport/docs/algorithms.md>.

use super::cost::{
    CostError, CostMatrix, DenseCostMatrix, PairwiseCostMatrix, PointMetric, PointSet,
};
use super::sinkhorn::{SinkhornError, SinkhornResult, SinkhornSolver, SinkhornWorkspace};

/// Combined limit for materialized cost matrices.
const DENSE_COST_LIMIT_BYTES: usize = 16 * 1024 * 1024;

fn cost_error_to_sinkhorn(e: CostError) -> SinkhornError {
    match e {
        CostError::NonFinitePointValue { value, .. } => SinkhornError::NonFiniteCost {
            row: 0,
            col: 0,
            value,
        },
        _ => SinkhornError::EmptyProblem,
    }
}

/// Result of a Sinkhorn divergence computation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SinkhornDivergence {
    /// Debiased divergence value.
    pub value: f32,
    /// Cross-term result: W_eps(source, target).
    pub cross: SinkhornResult,
    /// Self-term for source: W_eps(source, source).
    pub self_source: SinkhornResult,
    /// Self-term for target: W_eps(target, target).
    pub self_target: SinkhornResult,
}

/// Compute Sinkhorn divergence from three pre-built cost matrices.
#[allow(clippy::too_many_arguments)]
pub fn sinkhorn_divergence<Cxy, Cxx, Cyy>(
    solver: &SinkhornSolver,
    cost_xy: &Cxy,
    cost_xx: &Cxx,
    cost_yy: &Cyy,
    source: &[f32],
    target: &[f32],
    workspace_xy: &mut SinkhornWorkspace,
    workspace_xx: &mut SinkhornWorkspace,
    workspace_yy: &mut SinkhornWorkspace,
) -> Result<SinkhornDivergence, SinkhornError>
where
    Cxy: CostMatrix + ?Sized,
    Cxx: CostMatrix + ?Sized,
    Cyy: CostMatrix + ?Sized,
{
    let cross = solver.solve(cost_xy, source, target, workspace_xy)?;
    let self_source = solver.solve(cost_xx, source, source, workspace_xx)?;
    let self_target = solver.solve(cost_yy, target, target, workspace_yy)?;
    // Use regularized cost: raw primal cost breaks divergence symmetry.
    let value = cross.regularized_cost
        - 0.5 * self_source.regularized_cost
        - 0.5 * self_target.regularized_cost;
    Ok(SinkhornDivergence {
        value,
        cross,
        self_source,
        self_target,
    })
}

/// Compute Sinkhorn divergence from point sets, materializing costs below 16 MiB.
#[allow(clippy::too_many_arguments)]
pub fn point_set_sinkhorn_divergence<X, Y, M>(
    solver: &SinkhornSolver,
    source_points: &X,
    target_points: &Y,
    source: &[f32],
    target: &[f32],
    metric: M,
    workspace_xy: &mut SinkhornWorkspace,
    workspace_xx: &mut SinkhornWorkspace,
    workspace_yy: &mut SinkhornWorkspace,
) -> Result<SinkhornDivergence, SinkhornError>
where
    X: PointSet + ?Sized,
    Y: PointSet + ?Sized,
    M: PointMetric,
{
    let n_src = source_points.len();
    let n_tgt = target_points.len();
    let total_bytes = (n_src.saturating_mul(n_tgt))
        .saturating_add(n_src.saturating_mul(n_src))
        .saturating_add(n_tgt.saturating_mul(n_tgt))
        .saturating_mul(4); // f32 = 4 bytes

    if total_bytes <= DENSE_COST_LIMIT_BYTES {
        let cost_xy = DenseCostMatrix::from_point_sets(source_points, target_points, metric)
            .map_err(cost_error_to_sinkhorn)?;
        let cost_xx = DenseCostMatrix::from_point_sets(source_points, source_points, metric)
            .map_err(cost_error_to_sinkhorn)?;
        let cost_yy = DenseCostMatrix::from_point_sets(target_points, target_points, metric)
            .map_err(cost_error_to_sinkhorn)?;
        sinkhorn_divergence(
            solver,
            &cost_xy,
            &cost_xx,
            &cost_yy,
            source,
            target,
            workspace_xy,
            workspace_xx,
            workspace_yy,
        )
    } else {
        let cost_xy = PairwiseCostMatrix::new(source_points, target_points, metric);
        let cost_xx = PairwiseCostMatrix::new(source_points, source_points, metric);
        let cost_yy = PairwiseCostMatrix::new(target_points, target_points, metric);
        sinkhorn_divergence(
            solver,
            &cost_xy,
            &cost_xx,
            &cost_yy,
            source,
            target,
            workspace_xy,
            workspace_xx,
            workspace_yy,
        )
    }
}
