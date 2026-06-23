//! Regression tests for input-validation hardening.
//!
//! `SinkhornResult` and `DenseCostMatrix` expose public fields and derive
//! `Deserialize`, so a caller can construct values whose dual-variable lengths
//! or dimensions do not match. These tests pin that such values produce an
//! error (or a clear panic for the constructor contract) instead of an
//! out-of-bounds read.

use lattice_transport::cost::DenseCostMatrix;
use lattice_transport::sinkhorn::{SinkhornError, SinkhornResult};
use lattice_transport::transport_plan::extract_sparse_plan;

fn result_with_duals(rows: usize, cols: usize) -> SinkhornResult {
    SinkhornResult {
        epsilon: 0.1,
        iterations: 0,
        converged: true,
        last_error: 0.0,
        transport_cost: 0.0,
        regularized_cost: 0.0,
        entropy: 0.0,
        transported_mass: 0.0,
        row_residual_l1: 0.0,
        col_residual_l1: 0.0,
        log_u: vec![0.0; rows],
        log_v: vec![0.0; cols],
    }
}

#[test]
fn extract_sparse_plan_rejects_mismatched_result_and_cost() {
    // Dual variables sized for a 2×3 problem ...
    let result = result_with_duals(2, 3);
    // ... but the cost matrix claims 4×4: indexing log_u[row] in the loop would
    // read past its length without the up-front guard.
    let cost = DenseCostMatrix::new(4, 4, vec![1.0; 16]);
    let err = extract_sparse_plan(&result, &cost, 0.0).unwrap_err();
    assert!(
        matches!(err, SinkhornError::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
}

#[test]
fn log_gamma_rejects_out_of_range_indices() {
    let result = result_with_duals(2, 3);
    let cost = DenseCostMatrix::new(2, 3, vec![1.0; 6]);

    // row 5 is out of range for log_u (len 2).
    let err = result.log_gamma(&cost, 5, 0).unwrap_err();
    assert!(matches!(err, SinkhornError::DimensionMismatch { .. }));

    // col 9 is out of range for log_v (len 3).
    let err = result.log_gamma(&cost, 0, 9).unwrap_err();
    assert!(matches!(err, SinkhornError::DimensionMismatch { .. }));

    // in-range indices still succeed.
    assert!(result.log_gamma(&cost, 1, 2).is_ok());
}

#[test]
#[should_panic(expected = "overflow")]
fn dense_cost_matrix_rejects_dimension_overflow() {
    // rows × cols overflows usize; the constructor must panic clearly rather
    // than let the wrapped product pass the element-count check.
    let _ = DenseCostMatrix::new(usize::MAX, 2, Vec::new());
}
