//! Reference datasets and comparison helpers (test-only).
//!
//! Provides deterministic toy cases for correctness checks and a structured
//! place for cross-library comparisons.

use super::cost::{CostMatrix, DenseCostMatrix};
use super::math::abs;
use super::sinkhorn::{SinkhornError, SinkhornResult, SinkhornSolver, SinkhornWorkspace};

#[derive(Debug, Clone)]
pub struct ReferenceCase {
    pub name: &'static str,
    pub cost: DenseCostMatrix,
    pub source: Vec<f32>,
    pub target: Vec<f32>,
    pub expected_exact_cost: f32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ReferenceComparison {
    pub name: &'static str,
    pub expected_exact_cost: f32,
    pub sinkhorn_cost: f32,
    pub absolute_error: f32,
    pub relative_error: f32,
    pub result: SinkhornResult,
}

pub fn reference_cases() -> Vec<ReferenceCase> {
    vec![
        ReferenceCase {
            name: "identity_2x2",
            cost: DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]),
            source: vec![0.5, 0.5],
            target: vec![0.5, 0.5],
            expected_exact_cost: 0.0,
        },
        ReferenceCase {
            name: "one_to_one_scalar",
            cost: DenseCostMatrix::new(1, 1, vec![4.0]),
            source: vec![1.0],
            target: vec![1.0],
            expected_exact_cost: 4.0,
        },
        ReferenceCase {
            name: "off_diagonal_2x2",
            cost: DenseCostMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            source: vec![0.5, 0.5],
            target: vec![0.5, 0.5],
            expected_exact_cost: 0.0,
        },
    ]
}

pub fn compare_against_reference(
    solver: &SinkhornSolver,
) -> Result<Vec<ReferenceComparison>, SinkhornError> {
    let mut comparisons = Vec::new();
    for case in reference_cases() {
        let mut workspace = SinkhornWorkspace::new(case.cost.rows(), case.cost.cols());
        let result = solver.solve(&case.cost, &case.source, &case.target, &mut workspace)?;
        let absolute_error = abs(result.transport_cost - case.expected_exact_cost);
        let relative_error = if case.expected_exact_cost == 0.0 {
            absolute_error
        } else {
            absolute_error / abs(case.expected_exact_cost)
        };
        comparisons.push(ReferenceComparison {
            name: case.name,
            expected_exact_cost: case.expected_exact_cost,
            sinkhorn_cost: result.transport_cost,
            absolute_error,
            relative_error,
            result,
        });
    }
    Ok(comparisons)
}
