//! Log-domain stabilized variants beyond the plain fixed-epsilon solver.
//!
//! The base solver in `sinkhorn.rs` is already fully log-domain. This module
//! adds epsilon-scaling acceleration (Schmitzer-style): solve a sequence of
//! easier problems with larger `epsilon`, warm-starting the dual variables
//! at each stage.
//!
//! See: Schmitzer, "Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems", SIAM J. Sci. Comput. 2019.

use super::cost::CostMatrix;
use super::math::abs;
use super::sinkhorn::{
    ProgressObserver, SinkhornConfig, SinkhornError, SinkhornResult, SinkhornSolver,
    SinkhornWorkspace,
};

const MAX_EPSILON_STAGES: usize = 10_000;

/// Geometric epsilon schedule from `start` down to `target`.
///
/// **Unstable**: epsilon-scaling acceleration; schedule representation may change to support non-geometric sequences.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EpsilonScalingSchedule {
    /// Initial (large) epsilon.
    pub start: f32,
    /// Final (target) epsilon.
    pub target: f32,
    /// Multiplicative factor in `(0, 1)`. Example: `0.5` halves epsilon each stage.
    pub factor: f32,
    /// Inner iterations per stage.
    pub iterations_per_stage: usize,
}

impl EpsilonScalingSchedule {
    /// Generate the geometric sequence of epsilon values.
    ///
    /// **Unstable**: schedule generation detail; return type may become an iterator.
    pub fn geometric_values(&self) -> Vec<f32> {
        self.try_geometric_values().unwrap_or_default()
    }

    fn validate(&self) -> Result<(), SinkhornError> {
        if self.start <= 0.0 || !self.start.is_finite() {
            return Err(SinkhornError::InvalidEpsilonSchedule {
                field: "start",
                value: self.start,
            });
        }
        if self.target <= 0.0 || !self.target.is_finite() {
            return Err(SinkhornError::InvalidEpsilonSchedule {
                field: "target",
                value: self.target,
            });
        }
        if !(0.0..1.0).contains(&self.factor) || !self.factor.is_finite() {
            return Err(SinkhornError::InvalidEpsilonSchedule {
                field: "factor",
                value: self.factor,
            });
        }
        if self.start < self.target {
            return Err(SinkhornError::InvalidEpsilonSchedule {
                field: "start",
                value: self.start,
            });
        }
        if self.iterations_per_stage == 0 {
            return Err(SinkhornError::InvalidEpsilonSchedule {
                field: "iterations_per_stage",
                value: 0.0,
            });
        }
        Ok(())
    }

    fn try_geometric_values(&self) -> Result<Vec<f32>, SinkhornError> {
        self.validate()?;
        let mut values = Vec::new();
        let mut epsilon = self.start;
        values.push(epsilon);
        while epsilon > self.target {
            if values.len() >= MAX_EPSILON_STAGES {
                return Err(SinkhornError::InvalidEpsilonSchedule {
                    field: "factor",
                    value: self.factor,
                });
            }
            let next = (epsilon * self.factor).max(self.target);
            if !next.is_finite() || next >= epsilon || abs(next - epsilon) < f32::EPSILON {
                return Err(SinkhornError::InvalidEpsilonSchedule {
                    field: "factor",
                    value: self.factor,
                });
            }
            epsilon = next;
            values.push(epsilon);
            if epsilon <= self.target {
                break;
            }
        }
        if values.last().copied() != Some(self.target) {
            values.push(self.target);
        }
        Ok(values)
    }
}

/// Full configuration for the staged solver.
///
/// **Unstable**: wraps `SinkhornConfig` with an optional scaling schedule; field set may grow.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct LogDomainSinkhornConfig {
    /// Base solver configuration (epsilon here is used only when no scaling schedule is set).
    pub base: SinkhornConfig,
    /// Optional epsilon annealing schedule. When `None`, uses a single stage.
    pub epsilon_scaling: Option<EpsilonScalingSchedule>,
}

/// Summary of a single epsilon-scaling stage.
///
/// **Unstable**: per-stage diagnostic; field set mirrors `SinkhornResult` subset and may change.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EpsilonStageSummary {
    /// Stage index (0-based).
    pub stage_index: usize,
    /// Epsilon used in this stage.
    pub epsilon: f32,
    /// Iterations performed.
    pub iterations: usize,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Final error for this stage.
    pub last_error: f32,
    /// Transport cost at this stage.
    pub transport_cost: f32,
}

/// Result of a multi-stage log-domain solve.
///
/// **Unstable**: aggregates per-stage summaries with the final `SinkhornResult`; structure may be simplified.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogDomainSinkhornResult {
    /// Result from the final stage.
    pub final_result: SinkhornResult,
    /// Per-stage summaries.
    pub stages: Vec<EpsilonStageSummary>,
    /// Total iterations across all stages.
    pub total_iterations: usize,
}

/// Warm-started log-domain Sinkhorn with optional epsilon scaling.
///
/// **Unstable**: acceleration layer on top of `SinkhornSolver`; may be absorbed into the main solver via a schedule option.
#[derive(Debug, Clone, Default)]
pub struct LogDomainSinkhornSolver {
    /// Solver configuration.
    pub config: LogDomainSinkhornConfig,
}

impl LogDomainSinkhornSolver {
    /// Create a solver with the given configuration.
    ///
    /// **Unstable**: constructor matches struct stability.
    pub fn new(config: LogDomainSinkhornConfig) -> Self {
        Self { config }
    }

    /// Solve with optional epsilon scaling.
    ///
    /// **Unstable**: primary entry point; rejects invalid epsilon schedules.
    pub fn solve<C>(
        &self,
        cost: &C,
        source: &[f32],
        target: &[f32],
        workspace: &mut SinkhornWorkspace,
        mut progress: Option<&mut dyn ProgressObserver>,
    ) -> Result<LogDomainSinkhornResult, SinkhornError>
    where
        C: CostMatrix + ?Sized,
    {
        match &self.config.epsilon_scaling {
            None => {
                let solver = SinkhornSolver::new(self.config.base.clone());
                let result =
                    solver.solve_with_progress(cost, source, target, workspace, progress)?;
                let total_iterations = result.iterations;
                Ok(LogDomainSinkhornResult {
                    final_result: result.clone(),
                    stages: vec![EpsilonStageSummary {
                        stage_index: 0,
                        epsilon: result.epsilon,
                        iterations: result.iterations,
                        converged: result.converged,
                        last_error: result.last_error,
                        transport_cost: result.transport_cost,
                    }],
                    total_iterations,
                })
            }
            Some(schedule) => {
                let mut stages = Vec::new();
                let mut last_result = None;
                let mut total_iterations = 0usize;
                workspace.reset();

                for (stage_index, epsilon) in
                    schedule.try_geometric_values()?.into_iter().enumerate()
                {
                    let mut stage_config = self.config.base.clone();
                    stage_config.epsilon = epsilon;
                    stage_config.max_iterations = schedule.iterations_per_stage;
                    let solver = SinkhornSolver::new(stage_config);
                    let result = if let Some(observer) = progress.as_deref_mut() {
                        solver.solve_warm_start(cost, source, target, workspace, Some(observer))?
                    } else {
                        solver.solve_warm_start(cost, source, target, workspace, None)?
                    };
                    total_iterations += result.iterations;
                    stages.push(EpsilonStageSummary {
                        stage_index,
                        epsilon,
                        iterations: result.iterations,
                        converged: result.converged,
                        last_error: result.last_error,
                        transport_cost: result.transport_cost,
                    });
                    last_result = Some(result);
                }

                let final_result =
                    last_result.expect("epsilon schedule always has at least one stage");
                Ok(LogDomainSinkhornResult {
                    final_result,
                    stages,
                    total_iterations,
                })
            }
        }
    }
}
