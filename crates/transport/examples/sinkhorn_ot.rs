//! Optimal transport examples using lattice-transport.
//!
//! Run with:
//!   cargo run -p lattice-transport --example sinkhorn_ot
//!
//! Covers:
//! - Basic balanced Sinkhorn solve
//! - Custom SinkhornConfig
//! - Debiased Sinkhorn divergence
//! - Workspace reuse across multiple solves

use lattice_transport::{
    DenseCostMatrix, SinkhornConfig, SinkhornSolver, SinkhornWorkspace, normalize_weights,
    sinkhorn_divergence, uniform_weights,
};

fn main() {
    // -------------------------------------------------------------------------
    // 1. Minimal 2-point example
    // -------------------------------------------------------------------------
    // Source and target each have two atoms with equal mass.
    // Cost: moving within the same atom is free; moving between atoms costs 1.
    println!("=== 2-point balanced Sinkhorn ===");

    let cost = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
    let source = uniform_weights(2); // [0.5, 0.5]
    let target = uniform_weights(2); // [0.5, 0.5]

    let solver = SinkhornSolver::default();
    let mut workspace = SinkhornWorkspace::new(2, 2);

    let result = solver
        .solve(&cost, &source, &target, &mut workspace)
        .expect("2-point solve should not fail");

    println!("Converged:        {}", result.converged);
    println!("Iterations:       {}", result.iterations);
    println!("Transport cost:   {:.6}", result.transport_cost);
    println!("Regularized cost: {:.6}", result.regularized_cost);
    println!();

    // -------------------------------------------------------------------------
    // 2. 3-point problem with explicit configuration
    // -------------------------------------------------------------------------
    println!("=== 3-point problem with custom config ===");

    // Atoms on a line at positions 0, 1, 2:
    // cost[i,j] = (i - j)^2
    let cost_3x3 = DenseCostMatrix::new(3, 3, vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0]);

    // Tighter epsilon = closer to exact OT but more iterations needed
    let tight_config = SinkhornConfig {
        epsilon: 0.01,
        max_iterations: 1000,
        convergence_threshold: 1e-6,
        check_convergence_every: 20,
        min_marginal: 1e-12,
        error_on_non_convergence: false,
    };
    let tight_solver = SinkhornSolver {
        config: tight_config,
    };

    let source3 = uniform_weights(3);
    let target3 = uniform_weights(3);
    let mut ws3 = SinkhornWorkspace::new(3, 3);

    let result3 = tight_solver
        .solve(&cost_3x3, &source3, &target3, &mut ws3)
        .expect("3-point solve should not fail");

    println!("Converged:        {}", result3.converged);
    println!("Iterations:       {}", result3.iterations);
    println!("Transport cost:   {:.6}", result3.transport_cost);
    println!();

    // -------------------------------------------------------------------------
    // 3. Concentrated (near-point-mass) marginals
    // -------------------------------------------------------------------------
    // The solver requires strictly positive weights per FP-024 — exact zeros are
    // rejected to prevent phantom-mass injection. Use normalize_weights() with a
    // small floor to handle near-degenerate distributions safely.
    println!("=== Concentrated marginals (near-point-mass) ===");

    let cost_2x2 = DenseCostMatrix::new(2, 2, vec![0.0, 2.5, 2.5, 0.0]);

    // 999 units at atom 0, 1 unit at atom 1  =>  almost all mass at atom 0
    let raw_source = vec![999.0_f32, 1.0];
    // 1 unit at atom 0, 999 units at atom 1  =>  almost all mass at atom 1
    let raw_target = vec![1.0_f32, 999.0];

    // normalize_weights floors zeros and normalizes to sum=1
    let conc_source = normalize_weights(&raw_source, 1e-12).expect("valid weights");
    let conc_target = normalize_weights(&raw_target, 1e-12).expect("valid weights");

    let mut ws_pm = SinkhornWorkspace::new(2, 2);
    let result_pm = solver
        .solve(&cost_2x2, &conc_source, &conc_target, &mut ws_pm)
        .expect("concentrated-mass solve should not fail");

    println!(
        "Source weights: [{:.4}, {:.4}]",
        conc_source[0], conc_source[1]
    );
    println!(
        "Target weights: [{:.4}, {:.4}]",
        conc_target[0], conc_target[1]
    );
    println!(
        "Cost(0->1) = 2.5; transport cost: {:.4}",
        result_pm.transport_cost
    );
    // For nearly-point-mass marginals the transport cost approaches 2.5
    println!("Regularized cost:  {:.4}", result_pm.regularized_cost);
    println!();

    // -------------------------------------------------------------------------
    // 4. Workspace reuse across multiple solves
    // -------------------------------------------------------------------------
    // Re-using the workspace avoids heap allocation in the hot path.
    // Call workspace.reset() between solves with the same problem size,
    // or workspace.resize() when dimensions change.
    println!("=== Workspace reuse ===");

    let mut ws_shared = SinkhornWorkspace::new(2, 2);
    let simple_cost = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
    let w2 = uniform_weights(2);

    let mut transport_costs = Vec::new();
    for _ in 0..5 {
        let r = solver
            .solve(&simple_cost, &w2, &w2, &mut ws_shared)
            .expect("repeated solve");
        transport_costs.push(r.transport_cost);
        // Workspace is internally reset by the solver on each call (cold start)
    }

    // All solves of the same problem should produce the same result
    let max_spread = transport_costs.iter().copied().fold(0.0_f32, f32::max)
        - transport_costs.iter().copied().fold(f32::MAX, f32::min);
    println!(
        "5 identical solves, transport cost spread: {:.2e}  (should be ~0)",
        max_spread
    );
    assert!(max_spread < 1e-5);
    println!();

    // -------------------------------------------------------------------------
    // 5. Debiased Sinkhorn divergence
    // -------------------------------------------------------------------------
    // The raw Sinkhorn cost is positive even for identical distributions.
    // sinkhorn_divergence() removes this self-interaction bias.
    println!("=== Debiased Sinkhorn divergence ===");

    let cost_xy = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
    let cost_xx = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
    let cost_yy = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);

    let src = uniform_weights(2);
    let tgt = uniform_weights(2);

    let mut ws_xy = SinkhornWorkspace::new(2, 2);
    let mut ws_xx = SinkhornWorkspace::new(2, 2);
    let mut ws_yy = SinkhornWorkspace::new(2, 2);

    let divergence = sinkhorn_divergence(
        &solver, &cost_xy, &cost_xx, &cost_yy, &src, &tgt, &mut ws_xy, &mut ws_xx, &mut ws_yy,
    )
    .expect("divergence computation should not fail");

    println!("Divergence S(P, Q) = {:.6}", divergence.value);
    println!(
        "Cross term W(P,Q)  = {:.6}",
        divergence.cross.regularized_cost
    );
    println!(
        "Self P: W(P,P)     = {:.6}",
        divergence.self_source.regularized_cost
    );
    println!(
        "Self Q: W(Q,Q)     = {:.6}",
        divergence.self_target.regularized_cost
    );
    // For identical P=Q the debiased divergence is 0 by construction:
    //   S = W(P,Q) - 0.5*W(P,P) - 0.5*W(Q,Q)
    println!(
        "For identical P=Q, divergence should be ~0: {:.2e}",
        divergence.value.abs()
    );
    println!();

    println!("Done.");
}
