//! Unbalanced Sinkhorn with KL-relaxed marginals.
//!
//! See: Chizat et al., "Scaling Algorithms for Unbalanced Optimal Transport Problems", Math. Comp. 2018.
//!
//! This variant is preferable when the two embedding corpora do not represent
//! exactly the same total mass: entries may have been added, deleted, or had
//! their importance reweighted. Instead of forcing equality with dummy nodes,
//! the KL penalty softly charges mass creation and destruction.

use core::mem;

use rayon::prelude::*;

use super::cost::{CostMatrix, DenseCostMatrix};
use super::logsumexp::{OnlineLogSumExp, safe_exp, safe_ln};
use super::math::abs;
use super::sinkhorn::{
    ProgressObserver, ProgressState, SinkhornError, SinkhornWorkspace, compute_primal_statistics,
    marginal_kl_penalties,
};

const PARALLEL_SINKHORN_MIN_N: usize = 500;
const PARALLEL_MIN_ITEMS_PER_JOB: usize = 16;
const PARALLEL_COL_BLOCK: usize = 16;

#[inline]
fn use_parallel_sinkhorn(rows: usize, cols: usize) -> bool {
    rows >= PARALLEL_SINKHORN_MIN_N
        && cols >= PARALLEL_SINKHORN_MIN_N
        && rows.saturating_mul(cols) >= 250_000
}

/// Configuration for KL-relaxed Sinkhorn.
///
/// **Unstable**: unbalanced OT API is secondary to the balanced solver; tau parameters may be restructured.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UnbalancedConfig {
    /// Entropy regularization strength.
    pub epsilon: f32,
    /// KL penalty weight for source marginal relaxation.
    pub tau_source: f32,
    /// KL penalty weight for target marginal relaxation.
    pub tau_target: f32,
    /// Hard iteration cap.
    pub max_iterations: usize,
    /// Convergence threshold.
    pub convergence_threshold: f32,
    /// Frequency of convergence checks.
    pub check_convergence_every: usize,
    /// Floor for log-marginal computation.
    pub min_marginal: f32,
}

impl Default for UnbalancedConfig {
    fn default() -> Self {
        Self {
            epsilon: 5e-2,
            tau_source: 1.0,
            tau_target: 1.0,
            max_iterations: 500,
            convergence_threshold: 1e-5,
            check_convergence_every: 10,
            min_marginal: 1e-12,
        }
    }
}

/// Result of an unbalanced Sinkhorn solve.
///
/// **Unstable**: field set mirrors `SinkhornResult` but adds KL penalty fields; may be unified in a future refactor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UnbalancedResult {
    /// Epsilon used.
    pub epsilon: f32,
    /// Source KL penalty weight.
    pub tau_source: f32,
    /// Target KL penalty weight.
    pub tau_target: f32,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Final error.
    pub last_error: f32,
    /// Primal transport cost.
    pub transport_cost: f32,
    /// Regularized cost including KL penalties.
    pub regularized_cost: f32,
    /// Coupling entropy.
    pub entropy: f32,
    /// Total transported mass.
    pub transported_mass: f32,
    /// KL divergence of row marginals from source.
    pub row_kl: f32,
    /// KL divergence of column marginals from target.
    pub col_kl: f32,
    /// Log-domain source dual variable.
    pub log_u: Vec<f32>,
    /// Log-domain target dual variable.
    pub log_v: Vec<f32>,
}

/// Unbalanced Sinkhorn solver with KL-relaxed marginal constraints.
///
/// **Unstable**: solver for mass-imbalanced distributions; API may be unified with `SinkhornSolver` via a mode enum.
#[derive(Debug, Clone, Default)]
pub struct UnbalancedSinkhornSolver {
    /// Solver configuration.
    pub config: UnbalancedConfig,
}

impl UnbalancedSinkhornSolver {
    /// Create a solver with the given configuration.
    ///
    /// **Unstable**: constructor matches struct stability.
    pub fn new(config: UnbalancedConfig) -> Self {
        Self { config }
    }

    /// Dense-matrix solve with gated Rayon-parallel row/column updates for n ≥ 500.
    ///
    /// Same semantics as `solve`; concrete `DenseCostMatrix` type enables internal parallelism.
    ///
    /// **Unstable**: matches `solve` stability.
    pub fn solve_dense(
        &self,
        cost: &DenseCostMatrix,
        source: &[f32],
        target: &[f32],
        workspace: &mut SinkhornWorkspace,
        progress: Option<&mut dyn ProgressObserver>,
    ) -> Result<UnbalancedResult, SinkhornError> {
        self.solve_internal_par(cost, source, target, workspace, progress)
    }

    fn solve_internal_par<C>(
        &self,
        cost: &C,
        source: &[f32],
        target: &[f32],
        workspace: &mut SinkhornWorkspace,
        mut progress: Option<&mut dyn ProgressObserver>,
    ) -> Result<UnbalancedResult, SinkhornError>
    where
        C: CostMatrix + Sync + ?Sized,
    {
        if self.config.epsilon <= 0.0 || !self.config.epsilon.is_finite() {
            return Err(SinkhornError::NonPositiveEpsilon(self.config.epsilon));
        }
        if self.config.tau_source <= 0.0
            || self.config.tau_target <= 0.0
            || !self.config.tau_source.is_finite()
            || !self.config.tau_target.is_finite()
        {
            return Err(SinkhornError::NonPositiveEpsilon(
                self.config.tau_source.min(self.config.tau_target),
            ));
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
        let mut source_sum = 0.0f32;
        for (index, &value) in source.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(SinkhornError::InvalidMarginal {
                    axis: "source",
                    index,
                    value,
                });
            }
            source_sum += value;
        }
        if source_sum <= 0.0 || !source_sum.is_finite() {
            return Err(SinkhornError::InvalidMarginal {
                axis: "source",
                index: 0,
                value: source_sum,
            });
        }
        let mut target_sum = 0.0f32;
        for (index, &value) in target.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(SinkhornError::InvalidMarginal {
                    axis: "target",
                    index,
                    value,
                });
            }
            target_sum += value;
        }
        if target_sum <= 0.0 || !target_sum.is_finite() {
            return Err(SinkhornError::InvalidMarginal {
                axis: "target",
                index: 0,
                value: target_sum,
            });
        }

        let rows = cost.rows();
        let cols = cost.cols();
        workspace.prepare(rows, cols, false);

        let epsilon = self.config.epsilon;
        let inv_epsilon = 1.0 / epsilon;
        let min_marginal = self.config.min_marginal;
        let alpha = self.config.tau_source / (self.config.tau_source + epsilon);
        let beta = self.config.tau_target / (self.config.tau_target + epsilon);
        let check_every = self.config.check_convergence_every.max(1);
        let mut converged = false;
        let mut last_error = f32::INFINITY;
        let mut iterations = 0usize;
        let use_par = use_parallel_sinkhorn(rows, cols);

        for iteration in 0..self.config.max_iterations {
            // Row update (log_u)
            let max_du = if use_par {
                let log_v = workspace.log_v.as_slice();
                let old_log_u = workspace.log_u.as_slice();
                source
                    .par_iter()
                    .zip(workspace.next_log_u.par_iter_mut())
                    .enumerate()
                    .with_min_len(PARALLEL_MIN_ITEMS_PER_JOB)
                    .map(|(row, (&src_w, next_u))| -> Result<f32, SinkhornError> {
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
                        let updated = alpha * (safe_ln(src_w, min_marginal) - acc.finish());
                        let delta = abs(updated - old_log_u[row]);
                        *next_u = updated;
                        Ok(delta)
                    })
                    .try_reduce(|| 0.0f32, |a, b| Ok(a.max(b)))?
            } else {
                let mut max_du = 0.0f32;
                for (row, (&src_w, next_u)) in source
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
                    let updated = alpha * (safe_ln(src_w, min_marginal) - acc.finish());
                    let delta = abs(updated - workspace.log_u[row]);
                    if delta > max_du {
                        max_du = delta;
                    }
                    *next_u = updated;
                }
                max_du
            };
            mem::swap(&mut workspace.log_u, &mut workspace.next_log_u);

            // Column update (log_v)
            let max_dv = if use_par {
                let log_u = workspace.log_u.as_slice();
                let old_log_v = workspace.log_v.as_slice();
                workspace
                    .next_log_v
                    .par_chunks_mut(PARALLEL_COL_BLOCK)
                    .enumerate()
                    .with_min_len(PARALLEL_MIN_ITEMS_PER_JOB)
                    .map(|(block_idx, next_block)| -> Result<f32, SinkhornError> {
                        let mut block_max = 0.0f32;
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
                            let updated =
                                beta * (safe_ln(target[col], min_marginal) - acc.finish());
                            block_max = block_max.max(abs(updated - old_log_v[col]));
                            *next_v = updated;
                        }
                        Ok(block_max)
                    })
                    .try_reduce(|| 0.0f32, |a, b| Ok(a.max(b)))?
            } else {
                let mut max_dv = 0.0f32;
                for (col, (&tgt_w, next_v)) in target
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
                    let updated = beta * (safe_ln(tgt_w, min_marginal) - acc.finish());
                    let delta = abs(updated - workspace.log_v[col]);
                    if delta > max_dv {
                        max_dv = delta;
                    }
                    *next_v = updated;
                }
                max_dv
            };
            mem::swap(&mut workspace.log_v, &mut workspace.next_log_v);

            iterations = iteration + 1;
            if iterations % check_every == 0 || iterations == self.config.max_iterations {
                last_error = max_du.max(max_dv);
                if let Some(observer) = progress.as_deref_mut() {
                    let keep_going = observer.on_progress(ProgressState {
                        batch_index: 0,
                        batch_len: 1,
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

        let (transport_cost, entropy, transported_mass) =
            compute_primal_statistics(cost, epsilon, &workspace.log_u, &workspace.log_v)?;

        let mut row_sums = vec![0.0f32; rows];
        let mut col_sums = vec![0.0f32; cols];
        for (row, (log_u_row, row_sum)) in
            workspace.log_u.iter().zip(row_sums.iter_mut()).enumerate()
        {
            for (col, (log_v_col, col_sum)) in
                workspace.log_v.iter().zip(col_sums.iter_mut()).enumerate()
            {
                let cost_value = cost.cost(row, col);
                if !cost_value.is_finite() {
                    return Err(SinkhornError::NonFiniteCost {
                        row,
                        col,
                        value: cost_value,
                    });
                }
                let gamma = safe_exp(*log_u_row - cost_value * inv_epsilon + *log_v_col);
                *row_sum += gamma;
                *col_sum += gamma;
            }
        }
        let (row_kl, col_kl) = marginal_kl_penalties(
            &row_sums,
            &col_sums,
            source,
            target,
            self.config.min_marginal,
        );
        let regularized_cost = transport_cost - epsilon * entropy
            + self.config.tau_source * row_kl
            + self.config.tau_target * col_kl;

        Ok(UnbalancedResult {
            epsilon,
            tau_source: self.config.tau_source,
            tau_target: self.config.tau_target,
            iterations,
            converged,
            last_error,
            transport_cost,
            regularized_cost,
            entropy,
            transported_mass,
            row_kl,
            col_kl,
            log_u: workspace.log_u.clone(),
            log_v: workspace.log_v.clone(),
        })
    }

    /// Solve an unbalanced OT problem.
    ///
    /// **Unstable**: primary entry point; signature may change when unified with balanced solver.
    pub fn solve<C>(
        &self,
        cost: &C,
        source: &[f32],
        target: &[f32],
        workspace: &mut SinkhornWorkspace,
        mut progress: Option<&mut dyn ProgressObserver>,
    ) -> Result<UnbalancedResult, SinkhornError>
    where
        C: CostMatrix + ?Sized,
    {
        if self.config.epsilon <= 0.0 || !self.config.epsilon.is_finite() {
            return Err(SinkhornError::NonPositiveEpsilon(self.config.epsilon));
        }
        if self.config.tau_source <= 0.0
            || self.config.tau_target <= 0.0
            || !self.config.tau_source.is_finite()
            || !self.config.tau_target.is_finite()
        {
            return Err(SinkhornError::NonPositiveEpsilon(
                self.config.tau_source.min(self.config.tau_target),
            ));
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
        let mut source_sum = 0.0f32;
        for (index, &value) in source.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(SinkhornError::InvalidMarginal {
                    axis: "source",
                    index,
                    value,
                });
            }
            source_sum += value;
        }
        if source_sum <= 0.0 || !source_sum.is_finite() {
            return Err(SinkhornError::InvalidMarginal {
                axis: "source",
                index: 0,
                value: source_sum,
            });
        }
        let mut target_sum = 0.0f32;
        for (index, &value) in target.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(SinkhornError::InvalidMarginal {
                    axis: "target",
                    index,
                    value,
                });
            }
            target_sum += value;
        }
        if target_sum <= 0.0 || !target_sum.is_finite() {
            return Err(SinkhornError::InvalidMarginal {
                axis: "target",
                index: 0,
                value: target_sum,
            });
        }

        let rows = cost.rows();
        let cols = cost.cols();
        workspace.prepare(rows, cols, false);

        let epsilon = self.config.epsilon;
        let inv_epsilon = 1.0 / epsilon;
        let min_marginal = self.config.min_marginal;
        let alpha = self.config.tau_source / (self.config.tau_source + epsilon);
        let beta = self.config.tau_target / (self.config.tau_target + epsilon);
        let check_every = self.config.check_convergence_every.max(1);
        let mut converged = false;
        let mut last_error = f32::INFINITY;
        let mut iterations = 0usize;

        for iteration in 0..self.config.max_iterations {
            let mut max_du = 0.0f32;
            for (row, (&src_w, next_u)) in source
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
                let updated = alpha * (safe_ln(src_w, min_marginal) - acc.finish());
                let delta = abs(updated - workspace.log_u[row]);
                if delta > max_du {
                    max_du = delta;
                }
                *next_u = updated;
            }
            mem::swap(&mut workspace.log_u, &mut workspace.next_log_u);

            let mut max_dv = 0.0f32;
            for (col, (&tgt_w, next_v)) in target
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
                let updated = beta * (safe_ln(tgt_w, min_marginal) - acc.finish());
                let delta = abs(updated - workspace.log_v[col]);
                if delta > max_dv {
                    max_dv = delta;
                }
                *next_v = updated;
            }
            mem::swap(&mut workspace.log_v, &mut workspace.next_log_v);

            iterations = iteration + 1;
            if iterations % check_every == 0 || iterations == self.config.max_iterations {
                last_error = max_du.max(max_dv);
                if let Some(observer) = progress.as_deref_mut() {
                    let keep_going = observer.on_progress(ProgressState {
                        batch_index: 0,
                        batch_len: 1,
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

        let (transport_cost, entropy, transported_mass) =
            compute_primal_statistics(cost, epsilon, &workspace.log_u, &workspace.log_v)?;

        let mut row_sums = vec![0.0f32; rows];
        let mut col_sums = vec![0.0f32; cols];
        for (row, (log_u_row, row_sum)) in
            workspace.log_u.iter().zip(row_sums.iter_mut()).enumerate()
        {
            for (col, (log_v_col, col_sum)) in
                workspace.log_v.iter().zip(col_sums.iter_mut()).enumerate()
            {
                let cost_value = cost.cost(row, col);
                if !cost_value.is_finite() {
                    return Err(SinkhornError::NonFiniteCost {
                        row,
                        col,
                        value: cost_value,
                    });
                }
                let gamma = safe_exp(*log_u_row - cost_value * inv_epsilon + *log_v_col);
                *row_sum += gamma;
                *col_sum += gamma;
            }
        }
        let (row_kl, col_kl) = marginal_kl_penalties(
            &row_sums,
            &col_sums,
            source,
            target,
            self.config.min_marginal,
        );
        let regularized_cost = transport_cost - epsilon * entropy
            + self.config.tau_source * row_kl
            + self.config.tau_target * col_kl;

        Ok(UnbalancedResult {
            epsilon,
            tau_source: self.config.tau_source,
            tau_target: self.config.tau_target,
            iterations,
            converged,
            last_error,
            transport_cost,
            regularized_cost,
            entropy,
            transported_mass,
            row_kl,
            col_kl,
            log_u: workspace.log_u.clone(),
            log_v: workspace.log_v.clone(),
        })
    }
}
