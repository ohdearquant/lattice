# ADR-035: Sinkhorn-Knopp Balanced OT Solver

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-transport

## Context

Wasserstein distances between embedding distributions are needed for distribution drift
detection: determining whether the embedding space has shifted enough to require
re-indexing or model re-training. Computing exact Wasserstein distances is O(n³) via
the network simplex algorithm — too slow for the n=50-500 distributions typical in
embedding drift diagnostics.

Entropic regularization (Cuturi 2013) reduces this to O(n²) iterations via the
Sinkhorn-Knopp algorithm, at the cost of biased distances that must be corrected via
Sinkhorn divergence (see ADR-005).

## Decision

The primary solver is `SinkhornSolver`, a balanced (equal total mass) entropic OT solver
operating entirely in log domain. A secondary `LogDomainSinkhornSolver` adds epsilon-scaling
(Schmitzer 2019) for faster convergence to small-epsilon targets.

### Key Design Choices

**Fully log-domain iteration**

The algorithm maintains dual variables `log_u` (rows) and `log_v` (columns) throughout.
The Gibbs kernel `K = exp(-C / ε)` is never materialized as a matrix. Each update computes:

```
log_u[i] = ln(source[i]) - LSE_j(log_v[j] - C[i,j] / ε)
log_v[j] = ln(target[j]) - LSE_i(log_u[i] - C[i,j] / ε)
```

where `LSE` is log-sum-exp. This is numerically stable because the large negative values
in `exp(-C / ε)` for small ε remain as large negative log-values rather than underflowing
to 0.0.

**`OnlineLogSumExp` for inner loop efficiency**

The inner LSE accumulation uses `OnlineLogSumExp` (tracking max and running sum) rather
than chaining `logaddexp(a, b) = max + log1p(exp(min - max))`. The online variant requires
only one `ln` call per column/row (at `finish()`), versus one `log1p` call per element
when chaining. For a 100-column inner loop, this reduces transcendental function calls
from ~100 to ~1 per row.

**Convergence criterion: marginal L1 residuals (not dual-variable updates)**

Early versions used `max |log_u_new - log_u_old|` as the convergence criterion. This
was changed (FP-026) to marginal L1 residuals: `Σ |recovered_row_i - source_i|` and
`Σ |recovered_col_j - target_j|`. The dual-update proxy can declare convergence while
marginal constraints are still substantially violated. The residual-based criterion
measures what we actually care about: that the coupling's marginals match the specified
distributions. `last_error = max(row_residual, col_residual)`.

Convergence check frequency is configurable (`check_convergence_every`, default 10) to
amortize the O(n²) residual computation across multiple iterations.

**Preallocated `SinkhornWorkspace`**

`log_u`, `log_v`, `next_log_u`, `next_log_v` are preallocated `Vec<f32>` in a
`SinkhornWorkspace` struct. After each half-step, the active and next buffers are
swapped via `mem::swap` — no allocation in the hot loop. The workspace supports
warm-starting: if `initialized = true`, the previous dual variables carry over as
the starting point for the next solve. This is used by the epsilon-scaling solver.

**Rayon-parallel rows/columns for n ≥ 500**

`solve_dense` dispatches to `solve_internal_par` when `rows ≥ 500 AND cols ≥ 500 AND rows*cols ≥ 250_000`. Row updates use `par_iter()` with `with_min_len(16)`. Column updates use `par_chunks_mut(16)` (block size 16 columns per Rayon job). The sequential path `solve_internal` is used for smaller problems where parallelism overhead would dominate. `solve` (the generic `CostMatrix` path) is always sequential because `CostMatrix` is not `Sync`-bounded.

**Validation at entry**

`validate_problem` checks: ε > 0 and finite, non-empty cost matrix, marginal lengths match matrix dimensions, all marginal entries finite and positive (not just non-negative — zero mass entries are rejected at FP-024 to prevent phantom mass injection via `safe_ln` clamping), total masses equal within a tolerance of `max(1e-4, min_marginal × max(n, m))`.

**`SinkhornConfig` defaults**

```
epsilon: 5e-2
max_iterations: 500
convergence_threshold: 1e-5
check_convergence_every: 10
min_marginal: 1e-12
error_on_non_convergence: false
```

`error_on_non_convergence = false` means the solver returns the best iterate with
`converged = false` rather than erroring. Callers can inspect this flag and decide
whether to retry with more iterations or a larger epsilon.

### Alternatives Considered

| Alternative                                     | Pros                                    | Cons                                                           | Why Not                                                                |
| ----------------------------------------------- | --------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Network simplex (exact OT)                      | Exact distances, no regularization bias | O(n³) — infeasible for n > 100                                 | Too slow for drift diagnostics at n=50-500                             |
| Primal Sinkhorn (K-matrix explicit)             | Simpler code                            | exp(-C/ε) underflows to zero for small ε; numerically unstable | Numerical failures at small ε required for accurate distances          |
| Dual-update convergence criterion               | Cheaper per check                       | Can declare convergence with violated marginals (FP-026)       | Replaced by marginal residuals                                         |
| Fixed-iteration Sinkhorn (no convergence check) | Predictable runtime                     | Wastes iterations when already converged; no early stop        | Convergence check overhead is low (every 10 iters); early stop matters |

## Consequences

### Positive

- The log-domain formulation handles ε as small as 1e-3 on typical embedding cost matrices without numerical underflow.
- `solve_batch` reuses a single workspace across multiple (source, target) pairs sharing the same cost structure, saving repeated allocation.
- `ProgressObserver` trait allows cancellation and monitoring (reporting `ProgressState` with batch index, iteration, error, epsilon every `check_convergence_every` iterations).

### Negative

- The convergence check recomputes O(n²) marginal residuals every 10 iterations. For large n (e.g., 500×500 = 250K entries), this is a significant fraction of iteration cost.
- The sequential solver allocates no temporary buffers beyond the workspace, but `SinkhornResult` clones `log_u` and `log_v` into the returned value. For n=1000, this is 8 KB per result.
- `epsilon = 5e-2` is a reasonable default for unit-normalized cost matrices, but callers using unnormalized costs (e.g., raw pixel distances) must scale epsilon appropriately or face very slow convergence.

### Risks

- Marginal mass mismatch tolerance is `max(1e-4, min_marginal × max(n, m))`. For n=1000 with `min_marginal=1e-12`, this is `1e-4`. Float summation of 1000 values can accumulate ~1e-5 error; the tolerance leaves a reasonable margin.
- Warm-starting across different problems (when `workspace.initialized = true` and a new problem is solved without calling `reset()`) may slow convergence if the dual variables from the previous problem are far from the new optimum. The API requires explicit `warm_start=true` to activate this.

## References

- [`crates/transport/src/sinkhorn.rs`](/Users/lion/projects/lattice/crates/transport/src/sinkhorn.rs) — full solver implementation
- [`crates/transport/src/logsumexp.rs`](/Users/lion/projects/lattice/crates/transport/src/logsumexp.rs) — `OnlineLogSumExp`, `safe_ln`, `safe_exp`, `logaddexp`
- Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport", NeurIPS 2013
