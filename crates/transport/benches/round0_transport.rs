//! Round 0 baseline benchmarks for lattice-transport.
//!
//! Groups (aligned with benchmark_plan.md):
//!   transport/sinkhorn_balanced   — B1: DenseCostMatrix solve, sizes 10–500, eps 0.05/0.01
//!   transport/divergence_point_set — B2: point_set_sinkhorn_divergence, n in [20,50,100], dim=384
//!   transport/cost_construction   — B3: DenseCostMatrix::from_point_sets, dim [64,384], n [20..500]
//!   transport/barycenter_fixed    — B4: fixed_support_barycenter, 3/5 measures, sizes [20,50,100]
//!   transport/sinkhorn_unbalanced — B5: UnbalancedSinkhornSolver::solve, sizes 20–500
//!
//! All inputs are generated deterministically (hash-based); no external rand crate.
//! Correctness snapshots (iterations, residual, cost) are printed once per group
//! before the timed loops so they appear in `cargo bench` stdout.

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use lattice_transport::{
    BarycenterConfig, BarycenterProblem, ContiguousPoints, CosineDistance, CostMatrix,
    DenseCostMatrix, EpsilonScalingSchedule, FixedSupportBarycenterWorkspace,
    LogDomainSinkhornConfig, LogDomainSinkhornSolver, SinkhornConfig, SinkhornSolver,
    SinkhornWorkspace, SquaredEuclidean, UnbalancedConfig, UnbalancedSinkhornSolver,
    fixed_support_barycenter, point_set_sinkhorn_divergence, uniform_weights,
};

// ─── Deterministic input helpers ─────────────────────────────────────────────

fn hash_f32(seed: u64, idx: u64) -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    (seed, idx).hash(&mut h);
    h.finish() as f32 / u64::MAX as f32
}

/// Flat row-major point cloud n×dim in (0.05, 0.95).
fn make_flat_points(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    (0..(n * dim))
        .map(|i| hash_f32(seed, i as u64) * 0.9 + 0.05)
        .collect()
}

/// Dense cost matrix n×n from deterministic values in (0.01, 1.0).
fn make_dense_cost(n: usize, seed: u64) -> DenseCostMatrix {
    DenseCostMatrix::from_fn(n, n, |row, col| {
        if row == col {
            0.0
        } else {
            hash_f32(seed, (row * n + col) as u64) * 0.99 + 0.01
        }
    })
}

/// Uniform source and target marginals of length n.
fn make_marginals(n: usize) -> (Vec<f32>, Vec<f32>) {
    (uniform_weights(n), uniform_weights(n))
}

/// Slightly non-uniform positive marginals summing to 1.0.
fn make_unbalanced_marginals(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let raw: Vec<f32> = (0..n)
        .map(|i| hash_f32(seed, i as u64) * 0.9 + 0.1)
        .collect();
    let total: f32 = raw.iter().sum();
    let src: Vec<f32> = raw.iter().map(|v| v / total).collect();
    let raw2: Vec<f32> = (0..n)
        .map(|i| hash_f32(seed + 1, i as u64) * 0.9 + 0.1)
        .collect();
    let total2: f32 = raw2.iter().sum();
    let tgt: Vec<f32> = raw2.iter().map(|v| v / total2).collect();
    (src, tgt)
}

// ─── Correctness snapshot helper ─────────────────────────────────────────────

/// Run one timed solve and print correctness metadata without polluting the
/// timed loops. Called once per benchmark variant before the criterion loop.
fn print_correctness_sinkhorn(label: &str, cost: &DenseCostMatrix, epsilon: f32) {
    let (src, tgt) = make_marginals(cost.rows());
    let solver = SinkhornSolver::new(SinkhornConfig {
        epsilon,
        ..Default::default()
    });
    let mut ws = SinkhornWorkspace::new(cost.rows(), cost.cols());
    match solver.solve_dense(cost, &src, &tgt, &mut ws) {
        Ok(r) => eprintln!(
            "[correctness] {label}: converged={} iters={} residual={:.2e} transport_cost={:.6}",
            r.converged, r.iterations, r.last_error, r.transport_cost
        ),
        Err(e) => eprintln!("[correctness] {label}: ERROR {e}"),
    }
}

// ─── B1: Balanced Sinkhorn Core ───────────────────────────────────────────────

fn bench_sinkhorn_balanced(c: &mut Criterion) {
    let sizes: &[usize] = &[10, 20, 50, 100, 500];
    let epsilons: &[f32] = &[0.05, 0.01];

    let mut group = c.benchmark_group("transport/sinkhorn_balanced");

    for &n in sizes {
        let cells = (n * n) as u64;
        for &eps in epsilons {
            let id = BenchmarkId::new(format!("n{n}_eps{eps}"), n);
            group.throughput(Throughput::Elements(cells));

            // Correctness snapshot (printed once, not timed)
            let cost_snap = make_dense_cost(n, 42);
            print_correctness_sinkhorn(&format!("balanced n={n} eps={eps}"), &cost_snap, eps);

            let config = SinkhornConfig {
                epsilon: eps,
                check_convergence_every: 10,
                ..Default::default()
            };
            let solver = SinkhornSolver::new(config);

            group.bench_with_input(id, &n, |b, &n| {
                let cost = make_dense_cost(n, 42);
                let (src, tgt) = make_marginals(n);
                let mut ws = SinkhornWorkspace::new(n, n);
                b.iter(|| {
                    ws.reset();
                    let result = solver
                        .solve_dense(black_box(&cost), black_box(&src), black_box(&tgt), &mut ws)
                        .expect("sinkhorn_balanced solve failed");
                    black_box(result.transport_cost)
                });
            });
        }
    }
    group.finish();
}

// ─── B2: Point-Set Divergence ─────────────────────────────────────────────────

fn bench_divergence_point_set(c: &mut Criterion) {
    let sizes: &[usize] = &[20, 50, 100];
    const DIM: usize = 384;

    let mut group = c.benchmark_group("transport/divergence_point_set");
    group.sample_size(20);

    for &n in sizes {
        group.throughput(Throughput::Elements((n * DIM) as u64));
        let id = BenchmarkId::new(format!("n{n}_dim{DIM}"), n);

        group.bench_with_input(id, &n, |b, &n| {
            let src_flat = make_flat_points(n, DIM, 1);
            let tgt_flat = make_flat_points(n, DIM, 2);
            let src_pts = ContiguousPoints::new(&src_flat, DIM).unwrap();
            let tgt_pts = ContiguousPoints::new(&tgt_flat, DIM).unwrap();
            let (src_w, tgt_w) = make_marginals(n);
            let solver = SinkhornSolver::default();
            let metric = SquaredEuclidean;

            // Correctness snapshot
            let mut wxy = SinkhornWorkspace::new(n, n);
            let mut wxx = SinkhornWorkspace::new(n, n);
            let mut wyy = SinkhornWorkspace::new(n, n);
            match point_set_sinkhorn_divergence(
                &solver,
                &src_pts,
                &tgt_pts,
                &src_w,
                &tgt_w,
                metric,
                &mut wxy,
                &mut wxx,
                &mut wyy,
            ) {
                Ok(div) => eprintln!(
                    "[correctness] divergence n={n} dim={DIM}: value={:.6} cross.iters={} cross.converged={}",
                    div.value, div.cross.iterations, div.cross.converged
                ),
                Err(e) => eprintln!("[correctness] divergence n={n}: ERROR {e}"),
            }

            b.iter_batched(
                || {
                    (
                        SinkhornWorkspace::new(n, n),
                        SinkhornWorkspace::new(n, n),
                        SinkhornWorkspace::new(n, n),
                    )
                },
                |(mut wxy, mut wxx, mut wyy)| {
                    let result = point_set_sinkhorn_divergence(
                        black_box(&solver),
                        black_box(&src_pts),
                        black_box(&tgt_pts),
                        black_box(&src_w),
                        black_box(&tgt_w),
                        black_box(metric),
                        &mut wxy,
                        &mut wxx,
                        &mut wyy,
                    )
                    .expect("divergence solve failed");
                    black_box(result.value)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ─── B3: Cost Construction ────────────────────────────────────────────────────

fn bench_cost_construction(c: &mut Criterion) {
    let sizes: &[usize] = &[20, 50, 100, 500];
    let dims: &[usize] = &[64, 384];

    let mut group = c.benchmark_group("transport/cost_construction");

    for &dim in dims {
        for &n in sizes {
            let cells = (n * n) as u64;

            // Squared Euclidean
            {
                let id = BenchmarkId::new(format!("sq_eucl_n{n}_dim{dim}"), n);
                group.throughput(Throughput::Elements(cells));
                group.bench_with_input(id, &(n, dim), |b, &(n, dim)| {
                    let src_flat = make_flat_points(n, dim, 10);
                    let tgt_flat = make_flat_points(n, dim, 11);
                    let src_pts = ContiguousPoints::new(&src_flat, dim).unwrap();
                    let tgt_pts = ContiguousPoints::new(&tgt_flat, dim).unwrap();
                    b.iter(|| {
                        let mat = DenseCostMatrix::from_point_sets(
                            black_box(&src_pts),
                            black_box(&tgt_pts),
                            black_box(SquaredEuclidean),
                        )
                        .expect("cost construction failed");
                        black_box(mat.as_slice().len())
                    });
                });
            }

            // Cosine (non-unit-norm)
            {
                let id = BenchmarkId::new(format!("cosine_n{n}_dim{dim}"), n);
                group.throughput(Throughput::Elements(cells));
                group.bench_with_input(id, &(n, dim), |b, &(n, dim)| {
                    let src_flat = make_flat_points(n, dim, 12);
                    let tgt_flat = make_flat_points(n, dim, 13);
                    let src_pts = ContiguousPoints::new(&src_flat, dim).unwrap();
                    let tgt_pts = ContiguousPoints::new(&tgt_flat, dim).unwrap();
                    let metric = CosineDistance {
                        assume_unit_norm: false,
                        norm_floor: 1e-12,
                    };
                    b.iter(|| {
                        let mat = DenseCostMatrix::from_point_sets(
                            black_box(&src_pts),
                            black_box(&tgt_pts),
                            black_box(metric),
                        )
                        .expect("cosine construction failed");
                        black_box(mat.as_slice().len())
                    });
                });
            }
        }
    }
    group.finish();
}

// ─── B4: Fixed-Support Barycenter ─────────────────────────────────────────────

fn bench_barycenter_fixed(c: &mut Criterion) {
    let configs: &[(usize, usize)] = &[(3, 20), (3, 50), (5, 20), (5, 50), (3, 100)];

    let mut group = c.benchmark_group("transport/barycenter_fixed");
    group.sample_size(10);

    for &(n_measures, n) in configs {
        let id = BenchmarkId::new(format!("m{n_measures}_n{n}"), n);
        group.throughput(Throughput::Elements((n_measures * n * n) as u64));

        group.bench_with_input(id, &(n_measures, n), |b, &(n_measures, n)| {
            // Build cost matrices: each source measure to the shared support
            let costs: Vec<DenseCostMatrix> = (0..n_measures)
                .map(|i| make_dense_cost(n, 100 + i as u64))
                .collect();
            let src_weights: Vec<Vec<f32>> = (0..n_measures)
                .map(|i| {
                    let (src, _) = make_unbalanced_marginals(n, 200 + i as u64);
                    src
                })
                .collect();
            let combination_weights = uniform_weights(n_measures);
            let config = BarycenterConfig::default();

            // Correctness snapshot
            {
                let problems: Vec<BarycenterProblem<'_>> = costs
                    .iter()
                    .zip(src_weights.iter())
                    .map(|(c, w)| BarycenterProblem { cost: c, weights: w })
                    .collect();
                let mut ws = FixedSupportBarycenterWorkspace::new(&problems, n);
                match fixed_support_barycenter(&problems, &combination_weights, None, &config, &mut ws) {
                    Ok(r) => eprintln!(
                        "[correctness] barycenter m={n_measures} n={n}: converged={} iters={} objective={:.6}",
                        r.converged, r.iterations, r.objective
                    ),
                    Err(e) => eprintln!("[correctness] barycenter m={n_measures} n={n}: ERROR {e}"),
                }
            }

            b.iter_batched(
                || {
                    let problems: Vec<BarycenterProblem<'_>> = costs
                        .iter()
                        .zip(src_weights.iter())
                        .map(|(c, w)| BarycenterProblem { cost: c, weights: w })
                        .collect();
                    let ws = FixedSupportBarycenterWorkspace::new(&problems, n);
                    (problems, ws)
                },
                |(problems, mut ws)| {
                    let result = fixed_support_barycenter(
                        black_box(&problems),
                        black_box(&combination_weights),
                        None,
                        black_box(&config),
                        &mut ws,
                    )
                    .expect("barycenter failed");
                    black_box(result.objective)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ─── B5: Unbalanced Sinkhorn ──────────────────────────────────────────────────

fn bench_sinkhorn_unbalanced(c: &mut Criterion) {
    let sizes: &[usize] = &[20, 50, 100, 500];
    let epsilons: &[f32] = &[0.05, 0.01];

    let mut group = c.benchmark_group("transport/sinkhorn_unbalanced");

    for &n in sizes {
        let cells = (n * n) as u64;
        for &eps in epsilons {
            let id = BenchmarkId::new(format!("n{n}_eps{eps}"), n);
            group.throughput(Throughput::Elements(cells));

            let config = UnbalancedConfig {
                epsilon: eps,
                tau_source: 1.0,
                tau_target: 1.0,
                check_convergence_every: 10,
                ..Default::default()
            };
            let solver = UnbalancedSinkhornSolver::new(config);

            // Correctness snapshot
            {
                let cost = make_dense_cost(n, 77);
                let (src, tgt) = make_unbalanced_marginals(n, 88);
                let mut ws = SinkhornWorkspace::new(n, n);
                match solver.solve_dense(&cost, &src, &tgt, &mut ws, None) {
                    Ok(r) => eprintln!(
                        "[correctness] unbalanced n={n} eps={eps}: converged={} iters={} cost={:.6}",
                        r.converged, r.iterations, r.transport_cost
                    ),
                    Err(e) => eprintln!("[correctness] unbalanced n={n}: ERROR {e}"),
                }
            }

            group.bench_with_input(id, &n, |b, &n| {
                let cost = make_dense_cost(n, 77);
                let (src, tgt) = make_unbalanced_marginals(n, 88);
                let mut ws = SinkhornWorkspace::new(n, n);
                b.iter(|| {
                    ws.reset();
                    let result = solver
                        .solve_dense(
                            black_box(&cost),
                            black_box(&src),
                            black_box(&tgt),
                            &mut ws,
                            None,
                        )
                        .expect("unbalanced solve failed");
                    black_box(result.transport_cost)
                });
            });
        }
    }
    group.finish();
}

// ─── B6: Epsilon-Scaling Sinkhorn (H7) ───────────────────────────────────────
//
// Validates H7: for dim=384 squared-Euclidean costs (scale ≈ 52), starting at
// epsilon ≫ 1 and geometrically annealing reaches convergence with fewer total
// Sinkhorn iterations than a direct single-epsilon solve at the target epsilon.
//
// Schedule naming: ES{n}_s{start}_t{target}_f{factor*10}
//   ES1  start=50,  target=5,   factor=0.5 — 4-5 stages, mild anneal
//   ES2  start=50,  target=1,   factor=0.5 — 6-7 stages, medium anneal
//   ES3  start=100, target=5,   factor=0.7 — 7-8 stages, gentler decay
fn bench_sinkhorn_eps_scaling(c: &mut Criterion) {
    let sizes: &[usize] = &[20, 50, 100];
    const DIM: usize = 384;

    // (label, start, target, factor)
    let sched_specs: &[(&str, f32, f32, f32)] = &[
        ("ES1_s50t5f05", 50.0, 5.0, 0.5),
        ("ES2_s50t1f05", 50.0, 1.0, 0.5),
        ("ES3_s100t5f07", 100.0, 5.0, 0.7),
    ];

    let mut group = c.benchmark_group("transport/sinkhorn_eps_scaling");
    group.sample_size(20);

    for &n in sizes {
        group.throughput(Throughput::Elements((n * n) as u64));

        let src_flat = make_flat_points(n, DIM, 1);
        let tgt_flat = make_flat_points(n, DIM, 2);
        let src_pts = ContiguousPoints::new(&src_flat, DIM).unwrap();
        let tgt_pts = ContiguousPoints::new(&tgt_flat, DIM).unwrap();
        let cost = DenseCostMatrix::from_point_sets(&src_pts, &tgt_pts, SquaredEuclidean)
            .expect("cost construction failed");
        let (src_w, tgt_w) = make_marginals(n);

        for &(label, start, target, factor) in sched_specs {
            let id = BenchmarkId::new(format!("n{n}_{label}"), n);

            let schedule = EpsilonScalingSchedule {
                start,
                target,
                factor,
                iterations_per_stage: 20,
            };
            let config = LogDomainSinkhornConfig {
                base: SinkhornConfig {
                    epsilon: schedule.target,
                    max_iterations: 500,
                    ..Default::default()
                },
                epsilon_scaling: Some(schedule),
            };
            let solver = LogDomainSinkhornSolver::new(config);

            // Correctness snapshot
            {
                let mut ws = SinkhornWorkspace::new(n, n);
                match solver.solve(&cost, &src_w, &tgt_w, &mut ws, None) {
                    Ok(r) => eprintln!(
                        "[correctness] eps_scaling {label} n={n}: converged={} total_iters={} stages={} cost={:.6}",
                        r.final_result.converged,
                        r.total_iterations,
                        r.stages.len(),
                        r.final_result.transport_cost
                    ),
                    Err(e) => eprintln!("[correctness] eps_scaling {label} n={n}: ERROR {e}"),
                }
            }

            group.bench_with_input(id, &n, |b, &_n| {
                let mut ws = SinkhornWorkspace::new(n, n);
                b.iter(|| {
                    ws.reset();
                    let result = solver
                        .solve(
                            black_box(&cost),
                            black_box(&src_w),
                            black_box(&tgt_w),
                            &mut ws,
                            None,
                        )
                        .expect("eps_scaling solve failed");
                    black_box(result.final_result.transport_cost)
                });
            });
        }
    }
    group.finish();
}

// ─── Registration ─────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_sinkhorn_balanced,
    bench_divergence_point_set,
    bench_cost_construction,
    bench_barycenter_fixed,
    bench_sinkhorn_unbalanced,
    bench_sinkhorn_eps_scaling,
);
criterion_main!(benches);
