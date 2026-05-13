//! High-level embedding drift API.
//!
//! Translates raw OT outputs into the quantities a memory subsystem actually
//! needs: a scalar drift magnitude, sparse correspondences, expected per-entry
//! displacement, and summary statistics suitable for thresholding.

use std::borrow::ToOwned;

use super::cost::{CosineDistance, PairwiseCostMatrix, PointMetric, SquaredEuclidean};
use super::divergence::{SinkhornDivergence, point_set_sinkhorn_divergence};
use super::logsumexp::{mean_std, median, safe_exp, sqrt};
use super::sinkhorn::{
    SinkhornConfig, SinkhornError, SinkhornSolver, SinkhornWorkspace, normalize_weights,
    uniform_weights,
};
use super::transport_plan::{SparseTransportPlan, extract_sparse_plan};
use super::unbalanced::{UnbalancedConfig, UnbalancedSinkhornSolver};

/// Lightweight embedding record accepted by the drift API.
///
/// **Stable** (provisional): input record for `detect_drift_records`; field set is minimal by design.
#[derive(Debug, Clone)]
pub struct EmbeddingRecord<'a, Id> {
    /// Identifier for this record.
    pub id: Id,
    /// The embedding vector.
    pub embedding: &'a [f32],
    /// Weight (importance) of this record.
    pub weight: f32,
    /// Optional model identifier.
    pub model: Option<&'a str>,
}

impl<'a, Id> EmbeddingRecord<'a, Id> {
    /// Create a record with uniform weight.
    ///
    /// **Stable** (provisional): convenience constructor; equivalent to setting `weight = 1.0` and `model = None`.
    pub fn uniform(id: Id, embedding: &'a [f32]) -> Self {
        Self {
            id,
            embedding,
            weight: 1.0,
            model: None,
        }
    }
}

/// Adapter trait for application-level memory structs.
///
/// **Stable** (provisional): bridge between application memory types and the OT drift pipeline; four-method contract is stable.
pub trait MemoryLike {
    /// Identifier type for memories.
    type Id: Clone;

    /// Get the memory's identifier.
    ///
    /// **Stable** (provisional): required method; part of the trait contract.
    fn memory_id(&self) -> Self::Id;
    /// Get the embedding, if present.
    ///
    /// **Stable** (provisional): required method; `None` records are filtered before the OT solve.
    fn embedding(&self) -> Option<&[f32]>;
    /// Get the importance weight (defaults to 1.0).
    ///
    /// **Stable** (provisional): optional method with provided default.
    fn importance(&self) -> f32 {
        1.0
    }
    /// Get the embedding model identifier.
    ///
    /// **Stable** (provisional): optional method with provided default; used for model-change audit logging.
    fn embed_model(&self) -> Option<&str> {
        None
    }
}

/// How to weight entries in drift computation.
///
/// **Stable** (provisional): two-variant enum; a `Custom` variant may be added but existing variants are stable.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftWeighting {
    /// Equal weight for all entries.
    Uniform,
    /// Use the record's weight/importance.
    RecordWeights,
}

/// Distance metric for drift computation.
///
/// **Stable** (provisional): two standard metrics; new entries (e.g., L1) would be additive.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftMetricKind {
    /// Squared Euclidean distance (Wasserstein-2).
    SquaredEuclidean,
    /// Cosine distance.
    Cosine,
}

/// Solver variant for drift computation.
///
/// **Stable** (provisional): `Balanced` and `Unbalanced` cover the primary use cases; existing variants are stable.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftSolverMode {
    /// Balanced OT (requires equal total mass).
    Balanced,
    /// Unbalanced OT with KL relaxation.
    Unbalanced {
        /// Source marginal relaxation weight.
        tau_source: f32,
        /// Target marginal relaxation weight.
        tau_target: f32,
    },
}

/// Configuration for drift detection.
///
/// **Stable** (provisional): aggregate config for the high-level drift API; new fields would be additive with `Default`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DriftConfig {
    /// Weighting strategy.
    pub weighting: DriftWeighting,
    /// Distance metric.
    pub metric: DriftMetricKind,
    /// Solver variant.
    pub solver_mode: DriftSolverMode,
    /// Inner Sinkhorn configuration.
    pub sinkhorn: SinkhornConfig,
    /// Threshold for sparse transport plan extraction.
    pub plan_threshold: f32,
    /// Whether to compute the debiased Sinkhorn divergence.
    pub compute_divergence: bool,
    /// Number of standard deviations for outlier classification.
    pub outlier_std_threshold: f32,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            weighting: DriftWeighting::Uniform,
            metric: DriftMetricKind::SquaredEuclidean,
            solver_mode: DriftSolverMode::Balanced,
            sinkhorn: SinkhornConfig::default(),
            plan_threshold: 1e-6,
            compute_divergence: true,
            outlier_std_threshold: 2.0,
        }
    }
}

/// Per-entry displacement in the transport plan.
///
/// **Stable** (provisional): per-source-entry diagnostic included in `DriftReport`; field set is stable.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerEntryDisplacement<Id> {
    /// Record identifier.
    pub id: Id,
    /// Source index.
    pub source_index: usize,
    /// Total mass matched for this entry.
    pub matched_mass: f32,
    /// Expected cost (mass-weighted average).
    pub expected_cost: f32,
    /// Expected distance (square root of cost for squared metrics).
    pub expected_distance: f32,
    /// Index of the target with the most mass.
    pub strongest_target: Option<usize>,
    /// Mass to the strongest target.
    pub strongest_target_mass: f32,
}

/// Summary statistics for drift displacements.
///
/// **Stable** (provisional): five scalar statistics; new statistics would be additive.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DriftSummary {
    /// Mean displacement across all source entries.
    pub mean_displacement: f32,
    /// Median displacement.
    pub median_displacement: f32,
    /// Maximum displacement.
    pub max_displacement: f32,
    /// Standard deviation of displacements.
    pub std_displacement: f32,
    /// Number of entries exceeding mean + outlier_std_threshold * std.
    pub outlier_count: usize,
}

/// Complete drift analysis report.
///
/// **Stable** (provisional): primary output type of the drift API; all fields are documented outputs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DriftReport<Id> {
    /// Wasserstein distance (scalar drift magnitude).
    pub wasserstein_distance: f32,
    /// Raw transport cost.
    pub transport_cost: f32,
    /// Regularized cost.
    pub regularized_cost: f32,
    /// Debiased Sinkhorn divergence (if computed).
    pub sinkhorn_divergence: Option<f32>,
    /// Sparse transport plan.
    pub transport_plan: SparseTransportPlan,
    /// Per-entry displacement analysis.
    pub per_entry: Vec<PerEntryDisplacement<Id>>,
    /// Summary statistics.
    pub summary: DriftSummary,
    /// Source embedding model.
    pub source_model: Option<String>,
    /// Target embedding model.
    pub target_model: Option<String>,
}

/// Detect drift from application-level memory objects.
///
/// **Stable** (provisional): primary entry point for `MemoryLike`-typed corpora; filters None embeddings internally.
pub fn detect_drift_memories<M>(
    source: &[M],
    target: &[M],
    config: &DriftConfig,
) -> Result<DriftReport<M::Id>, SinkhornError>
where
    M: MemoryLike,
    M::Id: Clone,
{
    let source_records: Vec<_> = source
        .iter()
        .filter_map(|memory| {
            memory.embedding().map(|embedding| EmbeddingRecord {
                id: memory.memory_id(),
                embedding,
                weight: match config.weighting {
                    DriftWeighting::Uniform => 1.0,
                    DriftWeighting::RecordWeights => memory.importance(),
                },
                model: memory.embed_model(),
            })
        })
        .collect();
    let target_records: Vec<_> = target
        .iter()
        .filter_map(|memory| {
            memory.embedding().map(|embedding| EmbeddingRecord {
                id: memory.memory_id(),
                embedding,
                weight: match config.weighting {
                    DriftWeighting::Uniform => 1.0,
                    DriftWeighting::RecordWeights => memory.importance(),
                },
                model: memory.embed_model(),
            })
        })
        .collect();

    detect_drift_records(&source_records, &target_records, config)
}

/// Detect drift from embedding records.
///
/// **Stable** (provisional): lower-level entry point; caller constructs `EmbeddingRecord` slices directly.
pub fn detect_drift_records<Id>(
    source: &[EmbeddingRecord<'_, Id>],
    target: &[EmbeddingRecord<'_, Id>],
    config: &DriftConfig,
) -> Result<DriftReport<Id>, SinkhornError>
where
    Id: Clone,
{
    if source.is_empty() || target.is_empty() {
        return Err(SinkhornError::EmptyProblem);
    }

    let source_points: Vec<&[f32]> = source.iter().map(|record| record.embedding).collect();
    let target_points: Vec<&[f32]> = target.iter().map(|record| record.embedding).collect();
    let source_weights = match config.weighting {
        DriftWeighting::Uniform => uniform_weights(source.len()),
        DriftWeighting::RecordWeights => normalize_weights(
            &source
                .iter()
                .map(|record| record.weight)
                .collect::<Vec<_>>(),
            config.sinkhorn.min_marginal,
        )?,
    };
    let target_weights = match config.weighting {
        DriftWeighting::Uniform => uniform_weights(target.len()),
        DriftWeighting::RecordWeights => normalize_weights(
            &target
                .iter()
                .map(|record| record.weight)
                .collect::<Vec<_>>(),
            config.sinkhorn.min_marginal,
        )?,
    };

    match (config.metric, config.solver_mode) {
        (DriftMetricKind::SquaredEuclidean, DriftSolverMode::Balanced) => detect_balanced(
            source,
            target,
            &source_points,
            &target_points,
            &source_weights,
            &target_weights,
            config,
            SquaredEuclidean,
            true,
        ),
        (DriftMetricKind::Cosine, DriftSolverMode::Balanced) => detect_balanced(
            source,
            target,
            &source_points,
            &target_points,
            &source_weights,
            &target_weights,
            config,
            CosineDistance::default(),
            false,
        ),
        (
            DriftMetricKind::SquaredEuclidean,
            DriftSolverMode::Unbalanced {
                tau_source,
                tau_target,
            },
        ) => detect_unbalanced(
            source,
            target,
            &source_points,
            &target_points,
            &source_weights,
            &target_weights,
            config,
            SquaredEuclidean,
            true,
            tau_source,
            tau_target,
        ),
        (
            DriftMetricKind::Cosine,
            DriftSolverMode::Unbalanced {
                tau_source,
                tau_target,
            },
        ) => detect_unbalanced(
            source,
            target,
            &source_points,
            &target_points,
            &source_weights,
            &target_weights,
            config,
            CosineDistance::default(),
            false,
            tau_source,
            tau_target,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn detect_balanced<Id, M>(
    source: &[EmbeddingRecord<'_, Id>],
    target: &[EmbeddingRecord<'_, Id>],
    source_points: &[&[f32]],
    target_points: &[&[f32]],
    source_weights: &[f32],
    target_weights: &[f32],
    config: &DriftConfig,
    metric: M,
    squared_metric: bool,
) -> Result<DriftReport<Id>, SinkhornError>
where
    Id: Clone,
    M: PointMetric,
{
    let cost_xy = PairwiseCostMatrix::new(source_points, target_points, metric);
    let solver = SinkhornSolver::new(config.sinkhorn.clone());
    let mut workspace_xy = SinkhornWorkspace::new(source.len(), target.len());
    let result = solver.solve(&cost_xy, source_weights, target_weights, &mut workspace_xy)?;
    let divergence = if config.compute_divergence {
        let mut workspace_xx = SinkhornWorkspace::new(source.len(), source.len());
        let mut workspace_yy = SinkhornWorkspace::new(target.len(), target.len());
        Some(point_set_sinkhorn_divergence(
            &solver,
            source_points,
            target_points,
            source_weights,
            target_weights,
            metric,
            &mut workspace_xy,
            &mut workspace_xx,
            &mut workspace_yy,
        )?)
    } else {
        None
    };
    build_report(
        source,
        target,
        source_points,
        target_points,
        config,
        metric,
        squared_metric,
        result.transport_cost,
        result.regularized_cost,
        divergence.as_ref(),
        &result.log_u,
        &result.log_v,
        result.epsilon,
        result.transport_cost,
    )
}

#[allow(clippy::too_many_arguments)]
fn detect_unbalanced<Id, M>(
    source: &[EmbeddingRecord<'_, Id>],
    target: &[EmbeddingRecord<'_, Id>],
    source_points: &[&[f32]],
    target_points: &[&[f32]],
    source_weights: &[f32],
    target_weights: &[f32],
    config: &DriftConfig,
    metric: M,
    squared_metric: bool,
    tau_source: f32,
    tau_target: f32,
) -> Result<DriftReport<Id>, SinkhornError>
where
    Id: Clone,
    M: PointMetric,
{
    let cost_xy = PairwiseCostMatrix::new(source_points, target_points, metric);
    let solver = UnbalancedSinkhornSolver::new(UnbalancedConfig {
        epsilon: config.sinkhorn.epsilon,
        tau_source,
        tau_target,
        max_iterations: config.sinkhorn.max_iterations,
        convergence_threshold: config.sinkhorn.convergence_threshold,
        check_convergence_every: config.sinkhorn.check_convergence_every,
        min_marginal: config.sinkhorn.min_marginal,
    });
    let mut workspace = SinkhornWorkspace::new(source.len(), target.len());
    let result = solver.solve(
        &cost_xy,
        source_weights,
        target_weights,
        &mut workspace,
        None,
    )?;
    build_report(
        source,
        target,
        source_points,
        target_points,
        config,
        metric,
        squared_metric,
        result.transport_cost,
        result.regularized_cost,
        None,
        &result.log_u,
        &result.log_v,
        result.epsilon,
        result.transport_cost,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_report<Id, M>(
    source: &[EmbeddingRecord<'_, Id>],
    target: &[EmbeddingRecord<'_, Id>],
    source_points: &[&[f32]],
    target_points: &[&[f32]],
    config: &DriftConfig,
    metric: M,
    squared_metric: bool,
    transport_cost: f32,
    regularized_cost: f32,
    divergence: Option<&SinkhornDivergence>,
    log_u: &[f32],
    log_v: &[f32],
    epsilon: f32,
    distance_cost: f32,
) -> Result<DriftReport<Id>, SinkhornError>
where
    Id: Clone,
    M: PointMetric,
{
    let cost_xy = PairwiseCostMatrix::new(source_points, target_points, metric);
    let pseudo_result = super::sinkhorn::SinkhornResult {
        epsilon,
        iterations: 0,
        converged: true,
        last_error: 0.0,
        transport_cost,
        regularized_cost,
        entropy: 0.0,
        transported_mass: 0.0,
        row_residual_l1: 0.0,
        col_residual_l1: 0.0,
        log_u: log_u.to_vec(),
        log_v: log_v.to_vec(),
    };
    let transport_plan = extract_sparse_plan(&pseudo_result, &cost_xy, config.plan_threshold)?;
    let per_entry = per_entry_displacements(
        source,
        source_points,
        target_points,
        metric,
        squared_metric,
        log_u,
        log_v,
        epsilon,
    );
    let distances: Vec<f32> = per_entry
        .iter()
        .map(|entry| entry.expected_distance)
        .collect();
    let (mean_displacement, std_displacement) = mean_std(&distances);
    let median_displacement = median(&distances);
    let max_displacement = distances.iter().copied().fold(0.0f32, f32::max);
    let cutoff = mean_displacement + config.outlier_std_threshold * std_displacement;
    let outlier_count = distances.iter().filter(|&&value| value > cutoff).count();
    let wasserstein_distance = if squared_metric {
        sqrt(distance_cost.max(0.0))
    } else {
        distance_cost.max(0.0)
    };

    Ok(DriftReport {
        wasserstein_distance,
        transport_cost,
        regularized_cost,
        sinkhorn_divergence: divergence.map(|value| value.value),
        transport_plan,
        per_entry,
        summary: DriftSummary {
            mean_displacement,
            median_displacement,
            max_displacement,
            std_displacement,
            outlier_count,
        },
        source_model: source
            .first()
            .and_then(|record| record.model)
            .map(ToOwned::to_owned),
        target_model: target
            .first()
            .and_then(|record| record.model)
            .map(ToOwned::to_owned),
    })
}

#[allow(clippy::too_many_arguments)]
fn per_entry_displacements<Id, M>(
    source: &[EmbeddingRecord<'_, Id>],
    source_points: &[&[f32]],
    target_points: &[&[f32]],
    metric: M,
    squared_metric: bool,
    log_u: &[f32],
    log_v: &[f32],
    epsilon: f32,
) -> Vec<PerEntryDisplacement<Id>>
where
    Id: Clone,
    M: PointMetric,
{
    let mut out = Vec::with_capacity(source.len());
    for row in 0..source.len() {
        let mut matched_mass = 0.0f32;
        let mut expected_cost = 0.0f32;
        let mut strongest_target = None;
        let mut strongest_target_mass = 0.0f32;
        for col in 0..target_points.len() {
            let cost = metric.distance(source_points[row], target_points[col]);
            let mass = safe_exp(log_u[row] - cost / epsilon + log_v[col]);
            matched_mass += mass;
            expected_cost += mass * cost;
            if mass > strongest_target_mass {
                strongest_target_mass = mass;
                strongest_target = Some(col);
            }
        }
        let normalized_cost = if matched_mass > 0.0 {
            expected_cost / matched_mass
        } else {
            0.0
        };
        let expected_distance = if squared_metric {
            sqrt(normalized_cost.max(0.0))
        } else {
            normalized_cost.max(0.0)
        };
        out.push(PerEntryDisplacement {
            id: source[row].id.clone(),
            source_index: row,
            matched_mass,
            expected_cost: normalized_cost,
            expected_distance,
            strongest_target,
            strongest_target_mass,
        });
    }
    out
}
