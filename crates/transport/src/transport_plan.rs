//! Sparse transport plan extraction.
//!
//! The solver itself keeps only dual variables because storing the dense coupling
//! is infeasible for large problems. This module reconstructs only the entries
//! above a user-specified threshold, which is typically what downstream drift
//! analysis and audit tooling needs.

use super::cost::CostMatrix;
use super::logsumexp::{safe_exp, safe_ln};
use super::sinkhorn::{SinkhornError, SinkhornResult};

/// A single entry in the sparse transport plan.
///
/// **Stable** (provisional): output record type; all fields represent documented plan attributes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SparseTransportEntry {
    /// Source index.
    pub source: usize,
    /// Target index.
    pub target: usize,
    /// Transport mass.
    pub mass: f32,
    /// Log of transport mass.
    pub log_mass: f32,
    /// Cost of this (source, target) pair.
    pub cost: f32,
}

/// Sparse representation of the optimal coupling.
///
/// **Stable** (provisional): output from `extract_sparse_plan`; embedded in `DriftReport`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SparseTransportPlan {
    /// Number of source entries.
    pub rows: usize,
    /// Number of target entries.
    pub cols: usize,
    /// Mass threshold used for sparsification.
    pub threshold: f32,
    /// Total mass in retained entries.
    pub retained_mass: f32,
    /// Total mass in dropped entries.
    pub dropped_mass: f32,
    /// Entries above the threshold.
    pub entries: Vec<SparseTransportEntry>,
}

/// Compute log-gamma for a single (row, col) entry.
///
/// **Stable** (provisional): thin delegation to `SinkhornResult::log_gamma`; re-exported for convenience.
#[inline]
pub fn log_gamma<C>(
    result: &SinkhornResult,
    cost: &C,
    row: usize,
    col: usize,
) -> Result<f32, SinkhornError>
where
    C: CostMatrix + ?Sized,
{
    result.log_gamma(cost, row, col)
}

/// Extract entries of the transport plan above a mass threshold.
///
/// **Stable** (provisional): primary plan extraction function used by `DriftReport` construction.
pub fn extract_sparse_plan<C>(
    result: &SinkhornResult,
    cost: &C,
    threshold: f32,
) -> Result<SparseTransportPlan, SinkhornError>
where
    C: CostMatrix + ?Sized,
{
    let log_threshold = if threshold <= 0.0 {
        f32::NEG_INFINITY
    } else {
        safe_ln(threshold, threshold)
    };

    let mut entries = Vec::new();
    let mut retained_mass = 0.0f32;
    let mut dropped_mass = 0.0f32;

    for row in 0..cost.rows() {
        for col in 0..cost.cols() {
            let cost_value = cost.cost(row, col);
            if !cost_value.is_finite() {
                return Err(SinkhornError::NonFiniteCost {
                    row,
                    col,
                    value: cost_value,
                });
            }
            let log_mass = result.log_u[row] - cost_value / result.epsilon + result.log_v[col];
            let mass = safe_exp(log_mass);
            if mass == 0.0 {
                continue;
            }
            if log_mass >= log_threshold {
                retained_mass += mass;
                entries.push(SparseTransportEntry {
                    source: row,
                    target: col,
                    mass,
                    log_mass,
                    cost: cost_value,
                });
            } else {
                dropped_mass += mass;
            }
        }
    }

    Ok(SparseTransportPlan {
        rows: cost.rows(),
        cols: cost.cols(),
        threshold,
        retained_mass,
        dropped_mass,
        entries,
    })
}
