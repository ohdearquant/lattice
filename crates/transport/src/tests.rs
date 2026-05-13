//! Tests for the Sinkhorn optimal transport module.

use super::*;

/// Test 1: Uniform marginals on a symmetric cost matrix.
/// The optimal transport for identical uniform distributions on a diagonal cost
/// is to not move any mass. Transport cost should be near zero.
#[test]
fn uniform_marginals_identity_cost() {
    let cost = DenseCostMatrix::new(3, 3, vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0]);
    let weights = uniform_weights(3);
    let solver = SinkhornSolver::new(SinkhornConfig {
        epsilon: 0.1,
        max_iterations: 500,
        convergence_threshold: 1e-6,
        ..SinkhornConfig::default()
    });
    let mut workspace = SinkhornWorkspace::new(3, 3);
    let result = solver
        .solve(&cost, &weights, &weights, &mut workspace)
        .unwrap();

    assert!(result.converged, "Sinkhorn should converge");
    // With identical source/target, exact OT cost is 0. Regularized version
    // will be slightly positive due to entropy, but transport_cost should be
    // very small relative to epsilon.
    assert!(
        result.transport_cost < 0.1,
        "transport cost should be small for identical distributions, got {}",
        result.transport_cost
    );
    assert!(
        result.transported_mass > 0.9,
        "should transport nearly all mass, got {}",
        result.transported_mass
    );
}

/// Test 2: Epsilon convergence -- smaller epsilon should give tighter approximation
/// to exact OT.
#[test]
fn epsilon_convergence() {
    // 1x1 problem where exact OT cost = 5.0.
    let cost = DenseCostMatrix::new(1, 1, vec![5.0]);
    let source = vec![1.0];
    let target = vec![1.0];

    let mut costs_at_eps = Vec::new();
    for &eps in &[1.0, 0.1, 0.01] {
        let solver = SinkhornSolver::new(SinkhornConfig {
            epsilon: eps,
            max_iterations: 1000,
            convergence_threshold: 1e-8,
            ..SinkhornConfig::default()
        });
        let mut workspace = SinkhornWorkspace::new(1, 1);
        let result = solver
            .solve(&cost, &source, &target, &mut workspace)
            .unwrap();
        costs_at_eps.push((eps, result.transport_cost));
    }

    // All should be close to 5.0 (for 1x1 problem, exact solution is trivial)
    for &(eps, cost_val) in &costs_at_eps {
        assert!(
            (cost_val - 5.0).abs() < eps + 0.01,
            "At epsilon={}, transport cost {} should be near 5.0",
            eps,
            cost_val
        );
    }
}

/// Test 3: Sinkhorn divergence symmetry.
/// S(a, b) and S(b, a) should be approximately equal (divergence is symmetric).
#[test]
fn sinkhorn_divergence_symmetry() {
    let points_a: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
    let points_b: Vec<Vec<f32>> = vec![vec![0.5, 0.5], vec![1.5, 0.5]];
    let weights_a = uniform_weights(2);
    let weights_b = uniform_weights(2);
    let solver = SinkhornSolver::new(SinkhornConfig {
        epsilon: 0.1,
        max_iterations: 500,
        convergence_threshold: 1e-6,
        ..SinkhornConfig::default()
    });

    // S(a, b)
    let mut ws_ab = SinkhornWorkspace::new(2, 2);
    let mut ws_aa = SinkhornWorkspace::new(2, 2);
    let mut ws_bb = SinkhornWorkspace::new(2, 2);
    let div_ab = point_set_sinkhorn_divergence(
        &solver,
        &points_a[..],
        &points_b[..],
        &weights_a,
        &weights_b,
        SquaredEuclidean,
        &mut ws_ab,
        &mut ws_aa,
        &mut ws_bb,
    )
    .unwrap();

    // S(b, a)
    let mut ws_ba = SinkhornWorkspace::new(2, 2);
    let mut ws_bb2 = SinkhornWorkspace::new(2, 2);
    let mut ws_aa2 = SinkhornWorkspace::new(2, 2);
    let div_ba = point_set_sinkhorn_divergence(
        &solver,
        &points_b[..],
        &points_a[..],
        &weights_b,
        &weights_a,
        SquaredEuclidean,
        &mut ws_ba,
        &mut ws_bb2,
        &mut ws_aa2,
    )
    .unwrap();

    let diff = (div_ab.value - div_ba.value).abs();
    assert!(
        diff < 1e-4,
        "Sinkhorn divergence should be symmetric: S(a,b)={}, S(b,a)={}, diff={}",
        div_ab.value,
        div_ba.value,
        diff
    );
}

/// Test 4: Transport plan validity.
/// The sparse transport plan entries should have non-negative mass and their total
/// should be close to 1.0 for uniform distributions.
#[test]
fn transport_plan_validity() {
    let cost = DenseCostMatrix::new(3, 3, vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0]);
    let weights = uniform_weights(3);
    let solver = SinkhornSolver::default();
    let mut workspace = SinkhornWorkspace::new(3, 3);
    let result = solver
        .solve(&cost, &weights, &weights, &mut workspace)
        .unwrap();

    let plan = extract_sparse_plan(&result, &cost, 1e-8).unwrap();

    // All entries should have non-negative mass
    for entry in &plan.entries {
        assert!(
            entry.mass >= 0.0,
            "Transport mass should be non-negative, got {}",
            entry.mass
        );
        assert!(
            entry.source < 3 && entry.target < 3,
            "Indices should be in range"
        );
    }

    // Total retained + dropped mass should be close to 1.0
    let total = plan.retained_mass + plan.dropped_mass;
    assert!(
        (total - 1.0).abs() < 0.05,
        "Total transport mass should be near 1.0, got {}",
        total
    );
}

/// Test 5: Drift metric with known displacement.
/// Two point clouds that are a known distance apart should produce a corresponding
/// Wasserstein distance.
#[test]
fn drift_metric_known_displacement() {
    // Source: two points at (0, 0) and (1, 0)
    // Target: two points at (2, 0) and (3, 0) -- shifted right by 2 units
    let source_emb: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
    let target_emb: Vec<Vec<f32>> = vec![vec![2.0, 0.0], vec![3.0, 0.0]];

    let source_records: Vec<EmbeddingRecord<'_, usize>> = source_emb
        .iter()
        .enumerate()
        .map(|(i, emb)| EmbeddingRecord::uniform(i, emb))
        .collect();
    let target_records: Vec<EmbeddingRecord<'_, usize>> = target_emb
        .iter()
        .enumerate()
        .map(|(i, emb)| EmbeddingRecord::uniform(i, emb))
        .collect();

    let config = DriftConfig {
        compute_divergence: false,
        sinkhorn: SinkhornConfig {
            epsilon: 0.01,
            max_iterations: 1000,
            convergence_threshold: 1e-7,
            ..SinkhornConfig::default()
        },
        ..DriftConfig::default()
    };

    let report = detect_drift_records(&source_records, &target_records, &config).unwrap();

    // Each point moves by 2 units, so squared distance = 4 per point.
    // Wasserstein-2 distance = sqrt(average squared cost) = sqrt(4) = 2.
    assert!(
        (report.wasserstein_distance - 2.0).abs() < 0.2,
        "Wasserstein distance should be near 2.0, got {}",
        report.wasserstein_distance
    );

    // Per-entry displacements should each be near 2.0
    for entry in &report.per_entry {
        assert!(
            (entry.expected_distance - 2.0).abs() < 0.5,
            "Per-entry displacement should be near 2.0, got {}",
            entry.expected_distance
        );
    }
}

/// Test 6: Unbalanced solver allows mass mismatch.
#[test]
fn unbalanced_solver_mass_mismatch() {
    let cost = DenseCostMatrix::new(2, 3, vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0]);
    // Different sizes, different total mass
    let source = vec![0.5, 0.5];
    let target = vec![0.3, 0.3, 0.4];

    let solver = UnbalancedSinkhornSolver::new(UnbalancedConfig {
        epsilon: 0.1,
        tau_source: 1.0,
        tau_target: 1.0,
        max_iterations: 500,
        convergence_threshold: 1e-5,
        ..UnbalancedConfig::default()
    });
    let mut workspace = SinkhornWorkspace::new(2, 3);
    let result = solver
        .solve(&cost, &source, &target, &mut workspace, None)
        .unwrap();

    assert!(result.converged, "Unbalanced solver should converge");
    assert!(
        result.transport_cost >= 0.0,
        "Transport cost should be non-negative"
    );
    assert!(result.transported_mass > 0.0, "Should transport some mass");
}

/// Test 7: Reference case correctness.
#[test]
fn reference_cases_pass() {
    let solver = SinkhornSolver::new(SinkhornConfig {
        epsilon: 0.01,
        max_iterations: 1000,
        convergence_threshold: 1e-7,
        ..SinkhornConfig::default()
    });
    let comparisons = super::bench::compare_against_reference(&solver).unwrap();

    for comp in &comparisons {
        assert!(
            comp.absolute_error < 0.1,
            "Reference case '{}': absolute error {} is too large (expected ~{}, got {})",
            comp.name,
            comp.absolute_error,
            comp.expected_exact_cost,
            comp.sinkhorn_cost,
        );
    }
}

/// Test 8: Epsilon scaling solver converges.
#[test]
fn epsilon_scaling_converges() {
    let cost = DenseCostMatrix::new(3, 3, vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0]);
    let weights = uniform_weights(3);
    let solver = LogDomainSinkhornSolver::new(LogDomainSinkhornConfig {
        base: SinkhornConfig {
            epsilon: 0.01,
            max_iterations: 200,
            convergence_threshold: 1e-5,
            ..SinkhornConfig::default()
        },
        epsilon_scaling: Some(EpsilonScalingSchedule {
            start: 1.0,
            target: 0.01,
            factor: 0.5,
            iterations_per_stage: 100,
        }),
    });
    let mut workspace = SinkhornWorkspace::new(3, 3);
    let result = solver
        .solve(&cost, &weights, &weights, &mut workspace, None)
        .unwrap();

    assert!(
        result.stages.len() >= 2,
        "Should have multiple stages, got {}",
        result.stages.len()
    );
    assert!(
        result.total_iterations > 0,
        "Should perform some iterations"
    );
}

/// Test 9: Cost matrix from point sets.
#[test]
fn cost_matrix_from_points() {
    let points_a = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
    let points_b = vec![vec![0.0, 1.0], vec![1.0, 1.0]];

    let cost =
        DenseCostMatrix::from_point_sets(&points_a[..], &points_b[..], SquaredEuclidean).unwrap();

    assert_eq!(cost.rows(), 2);
    assert_eq!(cost.cols(), 2);

    // (0,0) -> (0,1): squared distance = 1.0
    assert!((cost.cost(0, 0) - 1.0).abs() < 1e-6);
    // (0,0) -> (1,1): squared distance = 2.0
    assert!((cost.cost(0, 1) - 2.0).abs() < 1e-6);
    // (1,0) -> (0,1): squared distance = 2.0
    assert!((cost.cost(1, 0) - 2.0).abs() < 1e-6);
    // (1,0) -> (1,1): squared distance = 1.0
    assert!((cost.cost(1, 1) - 1.0).abs() < 1e-6);
}

/// Test 10: Cosine distance metric.
#[test]
fn cosine_distance_metric() {
    let metric = CosineDistance::default();

    // Same direction: cosine distance = 0
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![2.0, 0.0, 0.0];
    let d = metric.distance(&a, &b);
    assert!(
        d.abs() < 1e-6,
        "Same direction should have distance ~0, got {}",
        d
    );

    // Orthogonal: cosine distance = 1
    let c = vec![0.0, 1.0, 0.0];
    let d_val = metric.distance(&a, &c);
    assert!(
        (d_val - 1.0).abs() < 1e-6,
        "Orthogonal should have distance ~1.0, got {}",
        d_val
    );
}

/// Test 11: Contiguous points layout.
#[test]
fn contiguous_points_access() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let points = ContiguousPoints::new(&data, 3).unwrap();

    assert_eq!(points.len(), 2);
    assert_eq!(points.dim(), 3);
    assert_eq!(points.point(0), &[1.0, 2.0, 3.0]);
    assert_eq!(points.point(1), &[4.0, 5.0, 6.0]);
}

/// Test 12: Error on empty problem.
#[test]
fn error_on_empty_problem() {
    let cost = DenseCostMatrix::new(0, 0, vec![]);
    let solver = SinkhornSolver::default();
    let mut workspace = SinkhornWorkspace::new(0, 0);
    let result = solver.solve(&cost, &[], &[], &mut workspace);
    assert!(matches!(result, Err(SinkhornError::EmptyProblem)));
}

#[test]
fn balanced_solver_rejects_zero_marginal_entries() {
    let cost = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
    let solver = SinkhornSolver::default();
    let mut workspace = SinkhornWorkspace::new(2, 2);
    let result = solver.solve(&cost, &[1.0, 0.0], &[0.5, 0.5], &mut workspace);

    assert!(matches!(
        result,
        Err(SinkhornError::InvalidMarginal {
            axis: "source",
            index: 1,
            ..
        })
    ));
}

/// Test 13: Error on non-positive epsilon.
#[test]
fn error_on_non_positive_epsilon() {
    let cost = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
    let solver = SinkhornSolver::new(SinkhornConfig {
        epsilon: -1.0,
        ..SinkhornConfig::default()
    });
    let weights = uniform_weights(2);
    let mut workspace = SinkhornWorkspace::new(2, 2);
    let result = solver.solve(&cost, &weights, &weights, &mut workspace);
    assert!(matches!(result, Err(SinkhornError::NonPositiveEpsilon(_))));
}

#[test]
fn invalid_epsilon_scaling_schedule_is_rejected() {
    let cost = DenseCostMatrix::new(1, 1, vec![0.0]);
    let weights = vec![1.0];
    let solver = LogDomainSinkhornSolver::new(LogDomainSinkhornConfig {
        base: SinkhornConfig::default(),
        epsilon_scaling: Some(EpsilonScalingSchedule {
            start: 1.0,
            target: 0.1,
            factor: 1.1,
            iterations_per_stage: 10,
        }),
    });
    let mut workspace = SinkhornWorkspace::new(1, 1);
    let result = solver.solve(&cost, &weights, &weights, &mut workspace, None);

    assert!(matches!(
        result,
        Err(SinkhornError::InvalidEpsilonSchedule {
            field: "factor",
            ..
        })
    ));
}

#[test]
fn reversed_epsilon_scaling_schedule_is_rejected() {
    let cost = DenseCostMatrix::new(1, 1, vec![0.0]);
    let weights = vec![1.0];
    let solver = LogDomainSinkhornSolver::new(LogDomainSinkhornConfig {
        base: SinkhornConfig::default(),
        epsilon_scaling: Some(EpsilonScalingSchedule {
            start: 0.1,
            target: 1.0,
            factor: 0.5,
            iterations_per_stage: 10,
        }),
    });
    let mut workspace = SinkhornWorkspace::new(1, 1);
    let result = solver.solve(&cost, &weights, &weights, &mut workspace, None);

    assert!(matches!(
        result,
        Err(SinkhornError::InvalidEpsilonSchedule { field: "start", .. })
    ));
}

#[test]
fn unbalanced_solver_rejects_invalid_marginals() {
    let cost = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
    let solver = UnbalancedSinkhornSolver::default();
    let mut workspace = SinkhornWorkspace::new(2, 2);

    let negative = solver.solve(&cost, &[0.5, -0.1], &[0.5, 0.5], &mut workspace, None);
    assert!(matches!(
        negative,
        Err(SinkhornError::InvalidMarginal {
            axis: "source",
            index: 1,
            ..
        })
    ));

    let non_finite = solver.solve(&cost, &[0.5, f32::NAN], &[0.5, 0.5], &mut workspace, None);
    assert!(matches!(
        non_finite,
        Err(SinkhornError::InvalidMarginal {
            axis: "source",
            index: 1,
            ..
        })
    ));

    let zero_total = solver.solve(&cost, &[0.0, 0.0], &[0.5, 0.5], &mut workspace, None);
    assert!(matches!(
        zero_total,
        Err(SinkhornError::InvalidMarginal { axis: "source", .. })
    ));
}

#[test]
fn unbalanced_solver_rejects_empty_problem() {
    let cost = DenseCostMatrix::new(0, 0, vec![]);
    let solver = UnbalancedSinkhornSolver::default();
    let mut workspace = SinkhornWorkspace::new(0, 0);
    let result = solver.solve(&cost, &[], &[], &mut workspace, None);

    assert!(matches!(result, Err(SinkhornError::EmptyProblem)));
}

#[test]
fn fixed_support_barycenter_requires_inner_convergence() {
    let cost = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
    let weights = vec![0.5, 0.5];
    let problems = [BarycenterProblem {
        cost: &cost,
        weights: &weights,
    }];
    let combination_weights = vec![1.0];
    let config = BarycenterConfig {
        sinkhorn: SinkhornConfig {
            max_iterations: 0,
            ..SinkhornConfig::default()
        },
        max_iterations: 1,
        convergence_threshold: 1e6,
    };
    let mut workspace = FixedSupportBarycenterWorkspace::new(&problems, 2);
    let result = fixed_support_barycenter(
        &problems,
        &combination_weights,
        None,
        &config,
        &mut workspace,
    )
    .unwrap();

    assert!(!result.converged);
    assert!(!result.source_results[0].converged);
}

#[test]
fn safe_exp_preserves_representable_positive_values() {
    assert!(logsumexp::safe_exp(-87.0) > 0.0);
    assert_eq!(logsumexp::safe_exp(-104.0), 0.0);
}

/// Test 14: logsumexp numerical stability.
#[test]
fn logsumexp_stability() {
    use super::logsumexp::logsumexp;

    // Large values that would overflow naive exp
    let values = vec![100.0, 101.0, 102.0];
    let result = logsumexp(&values);
    // Should be close to 102 + ln(1 + exp(-1) + exp(-2))
    assert!(result.is_finite(), "logsumexp should not overflow");
    assert!(
        (result - 102.408).abs() < 0.01,
        "logsumexp should be approximately 102.408, got {}",
        result
    );

    // Very negative values
    let neg_values = vec![-100.0, -101.0, -102.0];
    let neg_result = logsumexp(&neg_values);
    assert!(
        neg_result.is_finite(),
        "logsumexp should handle large negatives"
    );
}
