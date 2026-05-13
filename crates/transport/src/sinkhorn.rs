//! Core balanced Sinkhorn solver.
//!
//! Implements a numerically stable, log-domain Sinkhorn iteration with
//! preallocated work buffers, configurable convergence checks, progress
//! callbacks, and batch solving against a shared cost matrix.
//!
//! See: Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport", NeurIPS 2013.

use core::fmt;
use core::mem;

use rayon::prelude::*;

use super::cost::{CostMatrix, DenseCostMatrix};
use super::logsumexp::{OnlineLogSumExp, kl_term, safe_exp, safe_ln, sum};
use super::math::abs;

const PARALLEL_SINKHORN_MIN_N: usize = 500;
const PARALLEL_MIN_ITEMS_PER_JOB: usize = 16;
const PARALLEL_COL_BLOCK: usize = 16;

#[inline]
fn use_parallel_sinkhorn(rows: usize, cols: usize) -> bool {
    rows >= PARALLEL_SINKHORN_MIN_N
        && cols >= PARALLEL_SINKHORN_MIN_N
        && rows.saturating_mul(cols) >= 250_000
}

/// Solver configuration for balanced entropic OT.
///
/// **Stable** (provisional): all fields are semantically stable; a new field would be additive.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SinkhornConfig {
    /// Entropy regularization strength.
    ///
    /// Smaller values better approximate exact OT but require more iterations.
    /// Recommend scaling the cost matrix first and choosing epsilon relative
    /// to a characteristic cost scale.
    pub epsilon: f32,
    /// Hard iteration cap.
    pub max_iterations: usize,
    /// Convergence threshold on the max L1 marginal residual (row and column).
    pub convergence_threshold: f32,
    /// Frequency of convergence checks and progress callbacks.
    pub check_convergence_every: usize,
    /// Floor applied before taking `ln` of marginal entries.
    pub min_marginal: f32,
    /// If true, return an error when the cap is reached without convergence.
    /// If false, return the best iterate with `converged = false`.
    pub error_on_non_convergence: bool,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            epsilon: 5e-2,
            max_iterations: 500,
            convergence_threshold: 1e-5,
            check_convergence_every: 10,
            min_marginal: 1e-12,
            error_on_non_convergence: false,
        }
    }
}

/// Work buffers reused across solves to avoid heap allocation in the hot loop.
///
/// **Stable** (provisional): allocate once, solve many; the allocation pattern is part of the documented usage contract.
#[derive(Debug, Clone)]
pub struct SinkhornWorkspace {
    pub(crate) log_u: Vec<f32>,
    pub(crate) log_v: Vec<f32>,
    pub(crate) next_log_u: Vec<f32>,
    pub(crate) next_log_v: Vec<f32>,
    initialized: bool,
}

impl SinkhornWorkspace {
    /// Create workspace for a problem of the given dimensions.
    ///
    /// **Stable** (provisional): constructor matches struct stability.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            log_u: vec![0.0; rows],
            log_v: vec![0.0; cols],
            next_log_u: vec![0.0; rows],
            next_log_v: vec![0.0; cols],
            initialized: false,
        }
    }

    /// Resize buffers, resetting initialization state.
    ///
    /// **Stable** (provisional): needed when problem size changes between batch solves.
    pub fn resize(&mut self, rows: usize, cols: usize) {
        self.log_u.resize(rows, 0.0);
        self.log_v.resize(cols, 0.0);
        self.next_log_u.resize(rows, 0.0);
        self.next_log_v.resize(cols, 0.0);
        self.initialized = false;
    }

    /// Zero all buffers and mark as initialized (for warm-start).
    ///
    /// **Stable** (provisional): warm-start support is a documented feature.
    pub fn reset(&mut self) {
        self.log_u.fill(0.0);
        self.log_v.fill(0.0);
        self.next_log_u.fill(0.0);
        self.next_log_v.fill(0.0);
        self.initialized = true;
    }

    pub(crate) fn prepare(&mut self, rows: usize, cols: usize, warm_start: bool) {
        if self.log_u.len() != rows || self.log_v.len() != cols {
            self.resize(rows, cols);
        }
        if !warm_start || !self.initialized {
            self.reset();
        }
    }

    /// Number of source entries.
    ///
    /// **Stable** (provisional): dimension accessor.
    #[inline]
    pub fn rows(&self) -> usize {
        self.log_u.len()
    }

    /// Number of target entries.
    ///
    /// **Stable** (provisional): dimension accessor.
    #[inline]
    pub fn cols(&self) -> usize {
        self.log_v.len()
    }
}

/// Errors produced by the solver.
///
/// **Stable** (provisional): error variants reflect the documented failure modes; new variants would be additive.
#[derive(Debug, Clone, PartialEq)]
pub enum SinkhornError {
    /// The problem has zero rows or columns.
    EmptyProblem,
    /// Epsilon must be strictly positive and finite.
    NonPositiveEpsilon(f32),
    /// Marginal lengths do not match cost matrix dimensions.
    DimensionMismatch {
        /// Expected number of rows.
        expected_rows: usize,
        /// Actual number of rows received.
        got_rows: usize,
        /// Expected number of columns.
        expected_cols: usize,
        /// Actual number of columns received.
        got_cols: usize,
    },
    /// A marginal entry is non-finite or negative.
    InvalidMarginal {
        /// Which axis: `"source"` or `"target"`.
        axis: &'static str,
        /// Position of the invalid entry.
        index: usize,
        /// The invalid value.
        value: f32,
    },
    /// Source and target total mass differ beyond tolerance (balanced OT only).
    MarginalMassMismatch {
        /// Total source mass.
        source_sum: f32,
        /// Total target mass.
        target_sum: f32,
        /// Tolerance used for the check.
        tolerance: f32,
    },
    /// Cost matrix contains a non-finite entry.
    NonFiniteCost {
        /// Row index of the non-finite entry.
        row: usize,
        /// Column index of the non-finite entry.
        col: usize,
        /// The non-finite value.
        value: f32,
    },
    /// Iteration cap reached without convergence.
    NonConvergence {
        /// Number of iterations performed.
        iterations: usize,
        /// Last recorded error value.
        last_error: f32,
    },
    /// Epsilon-scaling schedule is invalid.
    InvalidEpsilonSchedule {
        /// Invalid schedule field.
        field: &'static str,
        /// Invalid value.
        value: f32,
    },
    /// Solve was cancelled by the progress observer.
    Cancelled,
}

impl fmt::Display for SinkhornError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyProblem => write!(f, "empty optimal transport problem"),
            Self::NonPositiveEpsilon(value) => {
                write!(f, "epsilon must be positive, got {value}")
            }
            Self::DimensionMismatch {
                expected_rows,
                got_rows,
                expected_cols,
                got_cols,
            } => write!(
                f,
                "marginal lengths ({got_rows}, {got_cols}) do not match cost matrix dimensions ({expected_rows}, {expected_cols})"
            ),
            Self::InvalidMarginal { axis, index, value } => write!(
                f,
                "{axis} marginal at index {index} is invalid: expected finite non-negative mass, got {value}"
            ),
            Self::MarginalMassMismatch {
                source_sum,
                target_sum,
                tolerance,
            } => write!(
                f,
                "balanced OT requires equal total mass within tolerance {tolerance}, got source={source_sum}, target={target_sum}"
            ),
            Self::NonFiniteCost { row, col, value } => write!(
                f,
                "cost matrix contains non-finite value at ({row}, {col}): {value}"
            ),
            Self::NonConvergence {
                iterations,
                last_error,
            } => write!(
                f,
                "Sinkhorn did not converge after {iterations} iterations (last error {last_error})"
            ),
            Self::InvalidEpsilonSchedule { field, value } => {
                write!(f, "invalid epsilon schedule {field}: {value}")
            }
            Self::Cancelled => write!(f, "Sinkhorn solve cancelled by progress observer"),
        }
    }
}

/// Lightweight progress snapshot emitted every `check_convergence_every` iterations.
///
/// **Stable** (provisional): all fields are pure read-only diagnostics; new fields would be additive.
#[derive(Debug, Clone, Copy)]
pub struct ProgressState {
    /// Index within a batch solve (0 for single solves).
    pub batch_index: usize,
    /// Total number of problems in the batch.
    pub batch_len: usize,
    /// Current iteration number (1-based).
    pub iteration: usize,
    /// Max absolute dual-variable update.
    pub last_error: f32,
    /// Whether convergence threshold has been met.
    pub converged: bool,
    /// Current epsilon value.
    pub epsilon: f32,
}

/// Observer interface used by long-running solves.
///
/// **Stable** (provisional): single-method trait; the `FnMut` blanket impl means closures work without a named type.
pub trait ProgressObserver {
    /// Called periodically during iteration. Return `false` to cancel the solve.
    ///
    /// **Stable** (provisional): signature matches struct stability.
    fn on_progress(&mut self, state: ProgressState) -> bool;
}

impl<F> ProgressObserver for F
where
    F: FnMut(ProgressState) -> bool,
{
    fn on_progress(&mut self, state: ProgressState) -> bool {
        self(state)
    }
}

/// Input pair for batch solving.
///
/// **Stable** (provisional): two-field view into source/target marginals; used by `solve_batch`.
#[derive(Debug, Clone, Copy)]
pub struct MarginalPair<'a> {
    /// Source marginal weights (sum to 1).
    pub source: &'a [f32],
    /// Target marginal weights (sum to 1).
    pub target: &'a [f32],
}

/// Returned by `solve_batch`.
///
/// **Stable** (provisional): simple wrapper over `Vec<SinkhornResult>`; shape stable.
#[derive(Debug, Clone)]
pub struct BatchSolveResult {
    /// One result per problem in the batch.
    pub results: Vec<SinkhornResult>,
}

/// Solver output containing dual variables and primal statistics.
///
/// **Stable** (provisional): primary result type; all fields are documented outputs of the solve.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SinkhornResult {
    /// Epsilon used for this solve.
    pub epsilon: f32,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the convergence threshold was met.
    pub converged: bool,
    /// Final max absolute dual-variable update.
    pub last_error: f32,
    /// Primal transport cost: `sum gamma[i,j] * C[i,j]`.
    pub transport_cost: f32,
    /// Transport cost minus entropy regularization.
    pub regularized_cost: f32,
    /// Entropy of the coupling: `-sum gamma[i,j] * (log gamma[i,j] - 1)`.
    pub entropy: f32,
    /// Total mass transported: `sum gamma[i,j]`.
    pub transported_mass: f32,
    /// L1 row marginal residual.
    pub row_residual_l1: f32,
    /// L1 column marginal residual.
    pub col_residual_l1: f32,
    /// Log-domain source dual variable.
    pub log_u: Vec<f32>,
    /// Log-domain target dual variable.
    pub log_v: Vec<f32>,
}

impl SinkhornResult {
    /// Returns the log-mass of a single coupling entry `gamma[row, col]`.
    ///
    /// **Stable** (provisional): used by `extract_sparse_plan` and audit tooling.
    #[inline]
    pub fn log_gamma<C>(&self, cost: &C, row: usize, col: usize) -> Result<f32, SinkhornError>
    where
        C: CostMatrix + ?Sized,
    {
        let value = cost.cost(row, col);
        if !value.is_finite() {
            return Err(SinkhornError::NonFiniteCost { row, col, value });
        }
        Ok(self.log_u[row] - value / self.epsilon + self.log_v[col])
    }
}

/// Balanced Sinkhorn solver.
///
/// **Stable** (provisional): the main solver type; `Default` produces production-ready configuration.
#[derive(Debug, Clone, Default)]
pub struct SinkhornSolver {
    /// Solver configuration.
    pub config: SinkhornConfig,
}

impl SinkhornSolver {
    /// Create a solver with the given configuration.
    ///
    /// **Stable** (provisional): constructor matches struct stability.
    pub fn new(config: SinkhornConfig) -> Self {
        Self { config }
    }

    /// Solve a single balanced OT problem.
    ///
    /// **Stable** (provisional): primary entry point; signature stable.
    pub fn solve<C>(
        &self,
        cost: &C,
        source: &[f32],
        target: &[f32],
        workspace: &mut SinkhornWorkspace,
    ) -> Result<SinkhornResult, SinkhornError>
    where
        C: CostMatrix + ?Sized,
    {
        self.solve_internal(cost, source, target, workspace, None, false, 0, 1)
    }

    /// Solve with a progress observer for cancellation and monitoring.
    ///
    /// **Stable** (provisional): progress/cancellation support is a documented feature.
    pub fn solve_with_progress<C>(
        &self,
        cost: &C,
        source: &[f32],
        target: &[f32],
        workspace: &mut SinkhornWorkspace,
        progress: Option<&mut dyn ProgressObserver>,
    ) -> Result<SinkhornResult, SinkhornError>
    where
        C: CostMatrix + ?Sized,
    {
        self.solve_internal(cost, source, target, workspace, progress, false, 0, 1)
    }

    /// Solve with warm-starting from previous dual variables.
    ///
    /// Mainly intended for epsilon-scaling and repeated solves with slowly
    /// changing marginals.
    ///
    /// **Stable** (provisional): warm-start is a documented performance feature.
    pub fn solve_warm_start<C>(
        &self,
        cost: &C,
        source: &[f32],
        target: &[f32],
        workspace: &mut SinkhornWorkspace,
        progress: Option<&mut dyn ProgressObserver>,
    ) -> Result<SinkhornResult, SinkhornError>
    where
        C: CostMatrix + ?Sized,
    {
        self.solve_internal(cost, source, target, workspace, progress, true, 0, 1)
    }

    /// Solve multiple problems sharing the same cost matrix structure.
    ///
    /// **Stable** (provisional): batch API reuses a single workspace across problems for efficiency.
    pub fn solve_batch<C>(
        &self,
        cost: &C,
        problems: &[MarginalPair<'_>],
        workspace: &mut SinkhornWorkspace,
        mut progress: Option<&mut dyn ProgressObserver>,
    ) -> Result<BatchSolveResult, SinkhornError>
    where
        C: CostMatrix + ?Sized,
    {
        let mut results = Vec::with_capacity(problems.len());
        for (batch_index, problem) in problems.iter().enumerate() {
            let result = if let Some(observer) = progress.as_deref_mut() {
                self.solve_internal(
                    cost,
                    problem.source,
                    problem.target,
                    workspace,
                    Some(observer),
                    false,
                    batch_index,
                    problems.len(),
                )?
            } else {
                self.solve_internal(
                    cost,
                    problem.source,
                    problem.target,
                    workspace,
                    None,
                    false,
                    batch_index,
                    problems.len(),
                )?
            };
            results.push(result);
        }
        Ok(BatchSolveResult { results })
    }

    /// Dense-matrix solve with gated Rayon-parallel row/column updates for n ≥ 500.
    ///
    /// Same semantics as `solve`; concrete `DenseCostMatrix` type enables internal parallelism.
    ///
    /// **Stable** (provisional): matches `solve` stability.
    pub fn solve_dense(
        &self,
        cost: &DenseCostMatrix,
        source: &[f32],
        target: &[f32],
        workspace: &mut SinkhornWorkspace,
    ) -> Result<SinkhornResult, SinkhornError> {
        self.solve_internal_par(cost, source, target, workspace, None, false, 0, 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_internal_par<C>(
        &self,
        cost: &C,
        source: &[f32],
        target: &[f32],
        workspace: &mut SinkhornWorkspace,
        mut progress: Option<&mut dyn ProgressObserver>,
        warm_start: bool,
        batch_index: usize,
        batch_len: usize,
    ) -> Result<SinkhornResult, SinkhornError>
    where
        C: CostMatrix + Sync + ?Sized,
    {
        validate_problem(cost, source, target, &self.config)?;

        let rows = cost.rows();
        let cols = cost.cols();
        workspace.prepare(rows, cols, warm_start);

        let epsilon = self.config.epsilon;
        let inv_epsilon = 1.0 / epsilon;
        let min_marginal = self.config.min_marginal;
        let check_every = self.config.check_convergence_every.max(1);
        let mut converged = false;
        let mut last_error = f32::INFINITY;
        let mut iterations = 0usize;
        let use_par = use_parallel_sinkhorn(rows, cols);

        for iteration in 0..self.config.max_iterations {
            // Update log_u (row scaling)
            if use_par {
                let log_v = workspace.log_v.as_slice();
                source
                    .par_iter()
                    .zip(workspace.next_log_u.par_iter_mut())
                    .enumerate()
                    .with_min_len(PARALLEL_MIN_ITEMS_PER_JOB)
                    .try_for_each(|(row, (&src, next_u))| -> Result<(), SinkhornError> {
                        let mut acc = OnlineLogSumExp::new();
                        #[allow(clippy::needless_range_loop)]
                        for col in 0..cols {
                            let cv = cost.cost(row, col);
                            if !cv.is_finite() {
                                return Err(SinkhornError::NonFiniteCost {
                                    row,
                                    col,
                                    value: cv,
                                });
                            }
                            acc.push(log_v[col] - cv * inv_epsilon);
                        }
                        *next_u = safe_ln(src, min_marginal) - acc.finish();
                        Ok(())
                    })?;
            } else {
                for (row, (&src, next_u)) in source
                    .iter()
                    .zip(workspace.next_log_u.iter_mut())
                    .enumerate()
                {
                    let mut acc = OnlineLogSumExp::new();
                    #[allow(clippy::needless_range_loop)]
                    for col in 0..cols {
                        let cost_value = cost.cost(row, col);
                        if !cost_value.is_finite() {
                            return Err(SinkhornError::NonFiniteCost {
                                row,
                                col,
                                value: cost_value,
                            });
                        }
                        acc.push(workspace.log_v[col] - cost_value * inv_epsilon);
                    }
                    *next_u = safe_ln(src, min_marginal) - acc.finish();
                }
            }
            mem::swap(&mut workspace.log_u, &mut workspace.next_log_u);

            // Update log_v (column scaling)
            if use_par {
                let log_u = workspace.log_u.as_slice();
                workspace
                    .next_log_v
                    .par_chunks_mut(PARALLEL_COL_BLOCK)
                    .enumerate()
                    .with_min_len(PARALLEL_MIN_ITEMS_PER_JOB)
                    .try_for_each(|(block_idx, next_block)| -> Result<(), SinkhornError> {
                        let col0 = block_idx * PARALLEL_COL_BLOCK;
                        for (offset, next_v) in next_block.iter_mut().enumerate() {
                            let col = col0 + offset;
                            let mut acc = OnlineLogSumExp::new();
                            #[allow(clippy::needless_range_loop)]
                            for row in 0..rows {
                                let cv = cost.cost(row, col);
                                if !cv.is_finite() {
                                    return Err(SinkhornError::NonFiniteCost {
                                        row,
                                        col,
                                        value: cv,
                                    });
                                }
                                acc.push(log_u[row] - cv * inv_epsilon);
                            }
                            *next_v = safe_ln(target[col], min_marginal) - acc.finish();
                        }
                        Ok(())
                    })?;
            } else {
                for (col, (&tgt, next_v)) in target
                    .iter()
                    .zip(workspace.next_log_v.iter_mut())
                    .enumerate()
                {
                    let mut acc = OnlineLogSumExp::new();
                    #[allow(clippy::needless_range_loop)]
                    for row in 0..rows {
                        let cost_value = cost.cost(row, col);
                        if !cost_value.is_finite() {
                            return Err(SinkhornError::NonFiniteCost {
                                row,
                                col,
                                value: cost_value,
                            });
                        }
                        acc.push(workspace.log_u[row] - cost_value * inv_epsilon);
                    }
                    *next_v = safe_ln(tgt, min_marginal) - acc.finish();
                }
            }
            mem::swap(&mut workspace.log_v, &mut workspace.next_log_v);

            iterations = iteration + 1;
            if iterations % check_every == 0 || iterations == self.config.max_iterations {
                let (row_res, col_res) = marginal_residuals(
                    cost,
                    epsilon,
                    source,
                    target,
                    &workspace.log_u,
                    &workspace.log_v,
                )?;
                last_error = row_res.max(col_res);
                if let Some(observer) = progress.as_deref_mut() {
                    let keep_going = observer.on_progress(ProgressState {
                        batch_index,
                        batch_len,
                        iteration: iterations,
                        last_error,
                        converged: last_error <= self.config.convergence_threshold,
                        epsilon,
                    });
                    if !keep_going {
                        return Err(SinkhornError::Cancelled);
                    }
                }
                if last_error <= self.config.convergence_threshold {
                    converged = true;
                    break;
                }
            }
        }

        if !converged && self.config.error_on_non_convergence {
            return Err(SinkhornError::NonConvergence {
                iterations,
                last_error,
            });
        }

        workspace.initialized = true;

        let (transport_cost, entropy, transported_mass) =
            compute_primal_statistics(cost, epsilon, &workspace.log_u, &workspace.log_v)?;
        let regularized_cost = transport_cost - epsilon * entropy;
        let (row_residual_l1, col_residual_l1) = marginal_residuals(
            cost,
            epsilon,
            source,
            target,
            &workspace.log_u,
            &workspace.log_v,
        )?;

        Ok(SinkhornResult {
            epsilon,
            iterations,
            converged,
            last_error,
            transport_cost,
            regularized_cost,
            entropy,
            transported_mass,
            row_residual_l1,
            col_residual_l1,
            log_u: workspace.log_u.clone(),
            log_v: workspace.log_v.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_internal<C>(
        &self,
        cost: &C,
        source: &[f32],
        target: &[f32],
        workspace: &mut SinkhornWorkspace,
        mut progress: Option<&mut dyn ProgressObserver>,
        warm_start: bool,
        batch_index: usize,
        batch_len: usize,
    ) -> Result<SinkhornResult, SinkhornError>
    where
        C: CostMatrix + ?Sized,
    {
        validate_problem(cost, source, target, &self.config)?;

        let rows = cost.rows();
        let cols = cost.cols();
        workspace.prepare(rows, cols, warm_start);

        let epsilon = self.config.epsilon;
        let inv_epsilon = 1.0 / epsilon;
        let min_marginal = self.config.min_marginal;
        let check_every = self.config.check_convergence_every.max(1);
        let mut converged = false;
        let mut last_error = f32::INFINITY;
        let mut iterations = 0usize;

        for iteration in 0..self.config.max_iterations {
            // Update log_u (row scaling)
            let mut max_du = 0.0f32;
            for (row, (&src, next_u)) in source
                .iter()
                .zip(workspace.next_log_u.iter_mut())
                .enumerate()
            {
                let mut acc = OnlineLogSumExp::new();
                for col in 0..cols {
                    let cost_value = cost.cost(row, col);
                    if !cost_value.is_finite() {
                        return Err(SinkhornError::NonFiniteCost {
                            row,
                            col,
                            value: cost_value,
                        });
                    }
                    acc.push(workspace.log_v[col] - cost_value * inv_epsilon);
                }
                let updated = safe_ln(src, min_marginal) - acc.finish();
                let delta = abs(updated - workspace.log_u[row]);
                if delta > max_du {
                    max_du = delta;
                }
                *next_u = updated;
            }
            mem::swap(&mut workspace.log_u, &mut workspace.next_log_u);

            // Update log_v (column scaling)
            let mut max_dv = 0.0f32;
            for (col, (&tgt, next_v)) in target
                .iter()
                .zip(workspace.next_log_v.iter_mut())
                .enumerate()
            {
                let mut acc = OnlineLogSumExp::new();
                for row in 0..rows {
                    let cost_value = cost.cost(row, col);
                    if !cost_value.is_finite() {
                        return Err(SinkhornError::NonFiniteCost {
                            row,
                            col,
                            value: cost_value,
                        });
                    }
                    acc.push(workspace.log_u[row] - cost_value * inv_epsilon);
                }
                let updated = safe_ln(tgt, min_marginal) - acc.finish();
                let delta = abs(updated - workspace.log_v[col]);
                if delta > max_dv {
                    max_dv = delta;
                }
                *next_v = updated;
            }
            mem::swap(&mut workspace.log_v, &mut workspace.next_log_v);

            iterations = iteration + 1;
            if iterations % check_every == 0 || iterations == self.config.max_iterations {
                // FP-026: convergence criterion is marginal constraint satisfaction
                // (L1 residuals), not dual-variable update size. Dual updates are a
                // proxy that can declare convergence while marginal constraints are
                // still substantially violated.
                let (row_res, col_res) = marginal_residuals(
                    cost,
                    epsilon,
                    source,
                    target,
                    &workspace.log_u,
                    &workspace.log_v,
                )?;
                last_error = row_res.max(col_res);
                if let Some(observer) = progress.as_deref_mut() {
                    let keep_going = observer.on_progress(ProgressState {
                        batch_index,
                        batch_len,
                        iteration: iterations,
                        last_error,
                        converged: last_error <= self.config.convergence_threshold,
                        epsilon,
                    });
                    if !keep_going {
                        return Err(SinkhornError::Cancelled);
                    }
                }
                if last_error <= self.config.convergence_threshold {
                    converged = true;
                    break;
                }
            }
        }

        if !converged && self.config.error_on_non_convergence {
            return Err(SinkhornError::NonConvergence {
                iterations,
                last_error,
            });
        }

        workspace.initialized = true;

        let (transport_cost, entropy, transported_mass) =
            compute_primal_statistics(cost, epsilon, &workspace.log_u, &workspace.log_v)?;
        let regularized_cost = transport_cost - epsilon * entropy;
        let (row_residual_l1, col_residual_l1) = marginal_residuals(
            cost,
            epsilon,
            source,
            target,
            &workspace.log_u,
            &workspace.log_v,
        )?;

        Ok(SinkhornResult {
            epsilon,
            iterations,
            converged,
            last_error,
            transport_cost,
            regularized_cost,
            entropy,
            transported_mass,
            row_residual_l1,
            col_residual_l1,
            log_u: workspace.log_u.clone(),
            log_v: workspace.log_v.clone(),
        })
    }
}

/// Creates a uniform probability vector.
///
/// **Stable** (provisional): convenience helper; commonly used in tests and examples.
pub fn uniform_weights(len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    let mass = 1.0 / len as f32;
    vec![mass; len]
}

/// Normalizes a non-negative weight vector to sum to one.
///
/// **Stable** (provisional): utility needed by `detect_drift_records` and barycenter code.
pub fn normalize_weights(weights: &[f32], floor: f32) -> Result<Vec<f32>, SinkhornError> {
    if weights.is_empty() {
        return Err(SinkhornError::EmptyProblem);
    }
    let mut normalized = Vec::with_capacity(weights.len());
    let mut total = 0.0f32;
    for (index, &weight) in weights.iter().enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(SinkhornError::InvalidMarginal {
                axis: "weight",
                index,
                value: weight,
            });
        }
        let clamped = weight.max(floor);
        normalized.push(clamped);
        total += clamped;
    }
    if total <= 0.0 || !total.is_finite() {
        return Err(SinkhornError::InvalidMarginal {
            axis: "weight",
            index: 0,
            value: total,
        });
    }
    for value in &mut normalized {
        *value /= total;
    }
    Ok(normalized)
}

fn validate_problem<C>(
    cost: &C,
    source: &[f32],
    target: &[f32],
    config: &SinkhornConfig,
) -> Result<(), SinkhornError>
where
    C: CostMatrix + ?Sized,
{
    if config.epsilon <= 0.0 || !config.epsilon.is_finite() {
        return Err(SinkhornError::NonPositiveEpsilon(config.epsilon));
    }
    if cost.rows() == 0 || cost.cols() == 0 {
        return Err(SinkhornError::EmptyProblem);
    }
    if source.len() != cost.rows() || target.len() != cost.cols() {
        return Err(SinkhornError::DimensionMismatch {
            expected_rows: cost.rows(),
            got_rows: source.len(),
            expected_cols: cost.cols(),
            got_cols: target.len(),
        });
    }
    // FP-024: reject zero-mass and non-positive entries explicitly. Previously
    // safe_ln would silently clamp zeros to min_marginal, injecting phantom mass
    // and producing incorrect transport plans.
    for (index, &value) in source.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(SinkhornError::InvalidMarginal {
                axis: "source",
                index,
                value,
            });
        }
    }
    for (index, &value) in target.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(SinkhornError::InvalidMarginal {
                axis: "target",
                index,
                value,
            });
        }
    }
    let source_sum = sum(source);
    let target_sum = sum(target);
    // FP-025: explicitly reject zero-total distributions. If both sums are zero
    // they satisfy the mass-balance check below but the problem is degenerate.
    if source_sum <= 0.0 || !source_sum.is_finite() {
        return Err(SinkhornError::InvalidMarginal {
            axis: "source",
            index: 0,
            value: source_sum,
        });
    }
    if target_sum <= 0.0 || !target_sum.is_finite() {
        return Err(SinkhornError::InvalidMarginal {
            axis: "target",
            index: 0,
            value: target_sum,
        });
    }
    let tolerance = 1e-4f32.max(config.min_marginal * source.len().max(target.len()) as f32);
    if abs(source_sum - target_sum) > tolerance {
        return Err(SinkhornError::MarginalMassMismatch {
            source_sum,
            target_sum,
            tolerance,
        });
    }
    Ok(())
}

pub(crate) fn compute_primal_statistics<C>(
    cost: &C,
    epsilon: f32,
    log_u: &[f32],
    log_v: &[f32],
) -> Result<(f32, f32, f32), SinkhornError>
where
    C: CostMatrix + ?Sized,
{
    let mut transport_cost = 0.0f32;
    let mut entropy = 0.0f32;
    let mut transported_mass = 0.0f32;
    let inv_eps = 1.0 / epsilon;

    for (row, &log_u_row) in log_u.iter().enumerate() {
        for (col, &log_v_col) in log_v.iter().enumerate() {
            let cost_value = cost.cost(row, col);
            if !cost_value.is_finite() {
                return Err(SinkhornError::NonFiniteCost {
                    row,
                    col,
                    value: cost_value,
                });
            }
            let log_gamma = log_u_row - cost_value * inv_eps + log_v_col;
            let gamma = safe_exp(log_gamma);
            if gamma == 0.0 {
                continue;
            }
            transport_cost += gamma * cost_value;
            entropy += -gamma * (log_gamma - 1.0);
            transported_mass += gamma;
        }
    }

    Ok((transport_cost, entropy, transported_mass))
}

pub(crate) fn marginal_residuals<C>(
    cost: &C,
    epsilon: f32,
    source: &[f32],
    target: &[f32],
    log_u: &[f32],
    log_v: &[f32],
) -> Result<(f32, f32), SinkhornError>
where
    C: CostMatrix + ?Sized,
{
    let inv_eps = 1.0 / epsilon;
    let mut row_residual = 0.0f32;
    for (row, (&log_u_row, &src)) in log_u.iter().zip(source.iter()).enumerate() {
        let mut acc = OnlineLogSumExp::new();
        for (col, &log_v_col) in log_v.iter().enumerate() {
            let cost_value = cost.cost(row, col);
            if !cost_value.is_finite() {
                return Err(SinkhornError::NonFiniteCost {
                    row,
                    col,
                    value: cost_value,
                });
            }
            acc.push(log_v_col - cost_value * inv_eps);
        }
        let recovered = safe_exp(log_u_row + acc.finish());
        row_residual += abs(recovered - src);
    }

    let mut col_residual = 0.0f32;
    for (col, (&log_v_col, &tgt)) in log_v.iter().zip(target.iter()).enumerate() {
        let mut acc = OnlineLogSumExp::new();
        for (row, &log_u_row) in log_u.iter().enumerate() {
            let cost_value = cost.cost(row, col);
            if !cost_value.is_finite() {
                return Err(SinkhornError::NonFiniteCost {
                    row,
                    col,
                    value: cost_value,
                });
            }
            acc.push(log_u_row - cost_value * inv_eps);
        }
        let recovered = safe_exp(log_v_col + acc.finish());
        col_residual += abs(recovered - tgt);
    }

    Ok((row_residual, col_residual))
}

pub(crate) fn marginal_kl_penalties(
    row_sums: &[f32],
    col_sums: &[f32],
    source: &[f32],
    target: &[f32],
    floor: f32,
) -> (f32, f32) {
    let row_kl = row_sums
        .iter()
        .zip(source.iter())
        .map(|(&row_sum, &mass)| kl_term(row_sum, mass, floor))
        .sum();
    let col_kl = col_sums
        .iter()
        .zip(target.iter())
        .map(|(&col_sum, &mass)| kl_term(col_sum, mass, floor))
        .sum();
    (row_kl, col_kl)
}
