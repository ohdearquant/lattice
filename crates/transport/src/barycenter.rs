//! Wasserstein barycenters.
//!
//! The fixed-support solver is the workhorse for model-drift analysis across
//! multiple embedding versions. The support points are fixed, only the
//! barycenter weights are optimized. The optional free-support routine
//! alternates between fixed-support weight updates and Euclidean support
//! relocation.
//!
//! See: Cuturi & Doucet, "Fast Computation of Wasserstein Barycenters", ICML 2014.

use super::cost::{CostMatrix, DenseCostMatrix, SquaredEuclidean};
use super::logsumexp::{OnlineLogSumExp, max_abs_diff, normalize_log_weights, safe_exp};
use super::math::ln;
use super::sinkhorn::{
    SinkhornConfig, SinkhornError, SinkhornResult, SinkhornSolver, SinkhornWorkspace,
    normalize_weights, uniform_weights,
};

/// Configuration for barycenter computation.
///
/// **Unstable**: barycenter API is less mature; inner/outer iteration structure may change.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BarycenterConfig {
    /// Inner Sinkhorn solver configuration.
    pub sinkhorn: SinkhornConfig,
    /// Maximum outer iterations for barycenter convergence.
    pub max_iterations: usize,
    /// Convergence threshold for barycenter weights.
    pub convergence_threshold: f32,
}

impl Default for BarycenterConfig {
    fn default() -> Self {
        Self {
            sinkhorn: SinkhornConfig::default(),
            max_iterations: 50,
            convergence_threshold: 1e-5,
        }
    }
}

/// A single source measure and the cost to the common barycenter support.
///
/// **Unstable**: input type for `fixed_support_barycenter`; may grow fields as multi-source API matures.
pub struct BarycenterProblem<'a> {
    /// Cost matrix from this source to the barycenter support.
    pub cost: &'a dyn CostMatrix,
    /// Weights of this source distribution.
    pub weights: &'a [f32],
}

/// Workspace for fixed-support barycenter computation.
///
/// **Unstable**: internal buffer management type; field layout may change if warm-starting is added.
#[derive(Debug, Clone)]
pub struct FixedSupportBarycenterWorkspace {
    source_workspaces: Vec<SinkhornWorkspace>,
    log_barycenter: Vec<f32>,
    next_log_barycenter: Vec<f32>,
    barycenter_weights: Vec<f32>,
    ku_logs: Vec<Vec<f32>>,
}

impl FixedSupportBarycenterWorkspace {
    /// Create workspace for the given problems and support size.
    ///
    /// **Unstable**: constructor matches struct stability.
    pub fn new(problems: &[BarycenterProblem<'_>], support_size: usize) -> Self {
        let mut source_workspaces = Vec::with_capacity(problems.len());
        let mut ku_logs = Vec::with_capacity(problems.len());
        for problem in problems {
            source_workspaces.push(SinkhornWorkspace::new(problem.cost.rows(), support_size));
            ku_logs.push(vec![0.0; support_size]);
        }
        Self {
            source_workspaces,
            log_barycenter: vec![0.0; support_size],
            next_log_barycenter: vec![0.0; support_size],
            barycenter_weights: vec![0.0; support_size],
            ku_logs,
        }
    }
}

/// Result of a fixed-support barycenter computation.
///
/// **Unstable**: output type for `fixed_support_barycenter`; `source_results` field may be made optional for memory efficiency.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FixedSupportBarycenter {
    /// Barycenter weights on the support.
    pub weights: Vec<f32>,
    /// Log-domain barycenter weights.
    pub log_weights: Vec<f32>,
    /// Outer iterations performed.
    pub iterations: usize,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Final max absolute change in log-weights.
    pub last_error: f32,
    /// Weighted sum of regularized OT costs.
    pub objective: f32,
    /// Per-source Sinkhorn results from the final iteration.
    pub source_results: Vec<SinkhornResult>,
}

/// Compute a fixed-support Wasserstein barycenter.
///
/// **Unstable**: core barycenter algorithm; `initial_weights` parameter may become a typed enum.
pub fn fixed_support_barycenter(
    problems: &[BarycenterProblem<'_>],
    combination_weights: &[f32],
    initial_weights: Option<&[f32]>,
    config: &BarycenterConfig,
    workspace: &mut FixedSupportBarycenterWorkspace,
) -> Result<FixedSupportBarycenter, SinkhornError> {
    if problems.is_empty() {
        return Err(SinkhornError::EmptyProblem);
    }
    let support_size = problems[0].cost.cols();
    if support_size == 0 {
        return Err(SinkhornError::EmptyProblem);
    }
    if combination_weights.len() != problems.len() {
        return Err(SinkhornError::DimensionMismatch {
            expected_rows: problems.len(),
            got_rows: combination_weights.len(),
            expected_cols: support_size,
            got_cols: support_size,
        });
    }
    for problem in problems.iter().skip(1) {
        if problem.cost.cols() != support_size {
            return Err(SinkhornError::DimensionMismatch {
                expected_rows: problem.cost.rows(),
                got_rows: problem.cost.rows(),
                expected_cols: support_size,
                got_cols: problem.cost.cols(),
            });
        }
    }

    if workspace.source_workspaces.len() != problems.len()
        || workspace.log_barycenter.len() != support_size
    {
        *workspace = FixedSupportBarycenterWorkspace::new(problems, support_size);
    }

    let lambda = normalize_weights(combination_weights, config.sinkhorn.min_marginal)?;
    let bary_init = match initial_weights {
        Some(weights) => normalize_weights(weights, config.sinkhorn.min_marginal)?,
        None => uniform_weights(support_size),
    };
    for (slot, &weight) in workspace.log_barycenter.iter_mut().zip(bary_init.iter()) {
        *slot = ln(weight);
    }
    normalize_log_weights(&mut workspace.log_barycenter);
    for (slot, &log_weight) in workspace
        .barycenter_weights
        .iter_mut()
        .zip(workspace.log_barycenter.iter())
    {
        *slot = safe_exp(log_weight);
    }

    let solver = SinkhornSolver::new(config.sinkhorn.clone());
    let mut source_results = Vec::with_capacity(problems.len());
    let mut last_error = f32::INFINITY;
    let mut converged = false;
    let mut iterations = 0usize;
    let mut objective = 0.0f32;

    for iteration in 0..config.max_iterations {
        source_results.clear();
        objective = 0.0;
        let mut inner_converged = true;

        for (problem_index, problem) in problems.iter().enumerate() {
            let result = solver.solve_warm_start(
                problem.cost,
                problem.weights,
                &workspace.barycenter_weights,
                &mut workspace.source_workspaces[problem_index],
                None,
            )?;
            inner_converged &= result.converged;
            objective += lambda[problem_index] * result.regularized_cost;

            let inv_eps = 1.0 / result.epsilon;
            for col in 0..support_size {
                let mut acc = OnlineLogSumExp::new();
                for (row, &log_u_row) in result.log_u.iter().enumerate() {
                    let cost_value = problem.cost.cost(row, col);
                    if !cost_value.is_finite() {
                        return Err(SinkhornError::NonFiniteCost {
                            row,
                            col,
                            value: cost_value,
                        });
                    }
                    acc.push(log_u_row - cost_value * inv_eps);
                }
                workspace.ku_logs[problem_index][col] = acc.finish();
            }
            source_results.push(result);
        }

        for (col, next_log_bary) in workspace.next_log_barycenter.iter_mut().enumerate() {
            *next_log_bary = lambda
                .iter()
                .zip(workspace.ku_logs.iter())
                .map(|(&lam, ku_log)| lam * ku_log[col])
                .sum();
        }
        normalize_log_weights(&mut workspace.next_log_barycenter);
        last_error = max_abs_diff(&workspace.log_barycenter, &workspace.next_log_barycenter);
        core::mem::swap(
            &mut workspace.log_barycenter,
            &mut workspace.next_log_barycenter,
        );
        for (slot, &log_weight) in workspace
            .barycenter_weights
            .iter_mut()
            .zip(workspace.log_barycenter.iter())
        {
            *slot = safe_exp(log_weight);
        }
        iterations = iteration + 1;
        if last_error <= config.convergence_threshold && inner_converged {
            converged = true;
            break;
        }
    }

    Ok(FixedSupportBarycenter {
        weights: workspace.barycenter_weights.clone(),
        log_weights: workspace.log_barycenter.clone(),
        iterations,
        converged,
        last_error,
        objective,
        source_results,
    })
}

/// Owned measure used by the free-support routine.
///
/// **Unstable**: input type for `free_support_barycenter`; may be replaced by a borrowed view when lifetimes are cleaner.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwnedPointMeasure {
    /// Points in the measure.
    pub points: Vec<Vec<f32>>,
    /// Weights of each point.
    pub weights: Vec<f32>,
}

/// Configuration for free-support barycenter.
///
/// **Unstable**: heuristic free-support iteration config; support-update strategy is exploratory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FreeSupportConfig {
    /// Inner fixed-support barycenter configuration.
    pub barycenter: BarycenterConfig,
    /// Maximum support relocation steps.
    pub max_support_updates: usize,
    /// Convergence threshold for support point movement.
    pub support_tolerance: f32,
}

impl Default for FreeSupportConfig {
    fn default() -> Self {
        Self {
            barycenter: BarycenterConfig::default(),
            max_support_updates: 10,
            support_tolerance: 1e-4,
        }
    }
}

/// Result of a free-support barycenter computation.
///
/// **Unstable**: output of the heuristic free-support routine; shape tied to current relocation algorithm.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FreeSupportBarycenter {
    /// Barycenter support points.
    pub support: Vec<Vec<f32>>,
    /// Barycenter weights.
    pub weights: Vec<f32>,
    /// Support update iterations performed.
    pub iterations: usize,
    /// Whether support convergence was achieved.
    pub converged: bool,
    /// Maximum squared shift in the last update.
    pub last_shift: f32,
}

/// Heuristic free-support barycenter for squared Euclidean geometry.
///
/// Alternates between fixed-support weight updates and Euclidean support
/// relocation. Useful for exploratory analysis; production drift scoring
/// usually only needs fixed-support barycenters or pairwise OT.
///
/// **Unstable**: heuristic algorithm; convergence guarantees and parameter defaults are subject to tuning.
pub fn free_support_barycenter(
    measures: &[OwnedPointMeasure],
    combination_weights: &[f32],
    initial_support: Vec<Vec<f32>>,
    initial_weights: Option<Vec<f32>>,
    config: &FreeSupportConfig,
) -> Result<FreeSupportBarycenter, SinkhornError> {
    if measures.is_empty() || initial_support.is_empty() {
        return Err(SinkhornError::EmptyProblem);
    }
    let support_size = initial_support.len();
    let dim = initial_support[0].len();
    let mut support = initial_support;
    let mut weights = initial_weights.unwrap_or_else(|| uniform_weights(support_size));
    let lambda = normalize_weights(combination_weights, config.barycenter.sinkhorn.min_marginal)?;
    let mut last_shift = f32::INFINITY;
    let mut converged = false;
    let mut iterations = 0usize;

    for update_index in 0..config.max_support_updates {
        let costs: Vec<DenseCostMatrix> = measures
            .iter()
            .map(|measure| {
                DenseCostMatrix::from_point_sets(
                    &measure.points[..],
                    &support[..],
                    SquaredEuclidean,
                )
                .map_err(|_| SinkhornError::EmptyProblem)
            })
            .collect::<Result<_, _>>()?;
        let problems: Vec<BarycenterProblem<'_>> = costs
            .iter()
            .zip(measures.iter())
            .map(|(cost, measure)| BarycenterProblem {
                cost,
                weights: &measure.weights,
            })
            .collect();
        let mut workspace = FixedSupportBarycenterWorkspace::new(&problems, support_size);
        let fixed = fixed_support_barycenter(
            &problems,
            &lambda,
            Some(&weights),
            &config.barycenter,
            &mut workspace,
        )?;
        weights = fixed.weights.clone();

        let mut new_support = vec![vec![0.0f32; dim]; support_size];
        let mut denom = vec![0.0f32; support_size];
        for (measure_index, measure) in measures.iter().enumerate() {
            let result = &fixed.source_results[measure_index];
            let cost = &costs[measure_index];
            for row in 0..measure.points.len() {
                for col in 0..support_size {
                    let mass = safe_exp(
                        result.log_u[row] - cost.cost(row, col) / result.epsilon
                            + result.log_v[col],
                    );
                    if mass == 0.0 {
                        continue;
                    }
                    let weighted_mass = lambda[measure_index] * mass;
                    denom[col] += weighted_mass;
                    for (ns_d, &pt_d) in new_support[col].iter_mut().zip(measure.points[row].iter())
                    {
                        *ns_d += weighted_mass * pt_d;
                    }
                }
            }
        }

        last_shift = 0.0;
        for col in 0..support_size {
            if denom[col] > 0.0 {
                let denom_val = denom[col];
                for ns_d in new_support[col].iter_mut() {
                    *ns_d /= denom_val;
                }
            } else {
                new_support[col] = support[col].clone();
            }
            let mut shift = 0.0f32;
            for d in 0..dim {
                let delta = new_support[col][d] - support[col][d];
                shift += delta * delta;
            }
            if shift > last_shift {
                last_shift = shift;
            }
        }

        support = new_support;
        iterations = update_index + 1;
        if last_shift <= config.support_tolerance {
            converged = true;
            break;
        }
    }

    Ok(FreeSupportBarycenter {
        support,
        weights,
        iterations,
        converged,
        last_shift,
    })
}
