# ADR-038: Wasserstein Barycenters

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-transport

## Context

Embedding drift detection requires comparing a current embedding distribution against a
reference. Rather than comparing against a single reference snapshot, it is useful to
compare against a _summary_ of multiple past snapshots: the Wasserstein barycenter, which
is the distribution that minimizes the sum of (weighted) Wasserstein distances to all
inputs. For fixed support (shared discrete positions), this is a tractable inner loop
built on top of the Sinkhorn solver.

The free-support variant allows the support positions themselves to move, producing a more
accurate barycenter but at higher computational cost. It is provided for exploratory analysis,
not production drift scoring.

## Decision

Two barycenter algorithms are implemented: `fixed_support_barycenter` (workhorse, used
for drift detection) and `free_support_barycenter` (heuristic, exploratory only). Both
use `SinkhornSolver` internally.

### Key Design Choices

**Fixed-support barycenter: outer iteration over inner Sinkhorn**

The algorithm (Cuturi & Doucet 2014) has an outer loop that alternates:

1. For each source distribution `a_k`, solve `W_ε(a_k, p)` where `p` is the current barycenter estimate.
2. Update `p` as the geometric mean (in log domain) of the `K^T u_k` column vectors:
   ```
   log_p[j] = Σ_k λ_k · log(Σ_i exp(log_u_k[i] - C_k[i,j] / ε))
   ```
   Equivalently: `log_p[j] = Σ_k λ_k · LSE_i(log_u_k[i] - C_k[i,j] / ε)`

This is computed entirely in log domain. `OnlineLogSumExp` accumulates each column's
contribution from each source's dual variable, then the per-source contributions are
combined as a weighted sum of log values.

**`FixedSupportBarycenterWorkspace` preallocates all buffers**

The workspace holds:

- `source_workspaces: Vec<SinkhornWorkspace>` — one per source distribution
- `log_barycenter`, `next_log_barycenter` — current and next log-domain barycenter weights
- `barycenter_weights` — exponentiated current weights (for Sinkhorn `target` input)
- `ku_logs: Vec<Vec<f32>>` — per-source column accumulations `LSE_i(log_u_k[i] - C_k[i,j] / ε)`

Buffer sizes are `[support_size]` for the barycenter and `[problems.len() × support_size]`
for `ku_logs`. If the workspace dimensions mismatch the problem, it is rebuilt in place.

**Warm-starting inner Sinkhorn calls**

The fixed-support loop calls `solver.solve_warm_start(...)` for each source at each outer
iteration. This reuses the dual variables from the previous outer iteration as warm starts,
exploiting the fact that consecutive barycenter estimates differ by small updates.

**Convergence: `max_abs_diff` on log-barycenter weights**

Outer convergence is declared when `max(|log_p_new[j] - log_p_old[j]|) ≤ convergence_threshold`
AND all inner Sinkhorn solves converged. Using log-domain comparison means the threshold
is scale-invariant: a change of `1e-5` in log probability corresponds to a ~1e-5 relative
change in probability mass.

`max_abs_diff` is a utility in `logsumexp.rs` used by both the barycenter and drift detection.

**Initial weights: uniform or caller-specified**

`fixed_support_barycenter` accepts `initial_weights: Option<&[f32]>`. If `None`,
uniform weights are used. If `Some`, the provided weights are normalized to sum to one
and converted to log domain. This allows warm-starting the outer loop from a previous
barycenter estimate.

**Free-support barycenter: alternating fixed-support + support relocation**

`free_support_barycenter` alternates:

1. `fixed_support_barycenter` with the current support positions.
2. Support relocation: the new support position `s_j` is the weighted average of all
   source points, weighted by the transport mass `γ_k[i,j]` from each source to that
   support point:
   ```
   s_j_new = Σ_{k,i} λ_k · γ_k[i,j] · x_k[i] / Σ_{k,i} λ_k · γ_k[i,j]
   ```

The transport mass is recovered from the dual variables:
`γ_k[i,j] = safe_exp(log_u_k[i] - C_k[i,j] / ε + log_v_k[j])`.

Convergence is measured by the maximum squared Euclidean shift of any support point
across the last relocation step: `last_shift = max_j ||s_j_new - s_j||²`.

**`FreeSupportConfig` marks the heuristic as exploratory**

The struct comment states: "Useful for exploratory analysis; production drift scoring
usually only needs fixed-support barycenters or pairwise OT." This is because the
free-support routine does not have a convergence guarantee — it is a coordinate-descent
heuristic that can cycle. Default `max_support_updates = 10` limits the outer iterations.

`OwnedPointMeasure` holds `points: Vec<Vec<f32>>` and `weights: Vec<f32>`. It is `Clone`
and serializable. The free-support routine rebuilds cost matrices from scratch on every
outer iteration (points move), so only owned data structures are appropriate.

### Alternatives Considered

| Alternative                             | Pros                 | Cons                                                                                               | Why Not                                                                          |
| --------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Drift detection via centroid comparison | Fast, O(n)           | Centroid is not a Wasserstein barycenter; insensitive to distribution shape changes                | Shape changes (mode splitting, variance increase) undetected                     |
| Frechet mean in embedding space         | Simpler math         | Requires Euclidean mean, which is not meaningful for probability distributions on discrete support | Wrong notion of mean for probability measures                                    |
| Free-support only (no fixed variant)    | More accurate        | 10x slower; requires support relocation; non-convergent heuristic                                  | Fixed-support sufficient for drift scoring; free-support too slow for production |
| Bregman projections without log-domain  | Well-known algorithm | Same numerical instability as primal Sinkhorn                                                      | Log-domain necessary for stability (see ADR-002)                                 |

## Consequences

### Positive

- Fixed-support barycenter re-uses the full `SinkhornSolver` machinery including Rayon parallelism (for large support sizes) and convergence checking.
- The workspace pattern eliminates allocations in the outer loop after the first iteration.
- `FixedSupportBarycenter.source_results` returns the per-source Sinkhorn results from the final outer iteration, enabling diagnosis of which source was furthest from the barycenter.

### Negative

- `fixed_support_barycenter` requires all source distributions to share the same support (the barycenter support). If sources have different discrete supports, the caller must first embed them into a shared support (e.g., by treating support positions as reference embeddings and computing an embedding-space cost matrix to each source point).
- The `ku_logs` intermediate buffer requires `problems.len() × support_size` f32 values per outer iteration. For 10 sources with support size 100, this is 4 KB — negligible. For 100 sources with support size 1000, this is 400 KB per outer iteration.
- `free_support_barycenter` rebuilds `DenseCostMatrix` on every support-relocation step. For large inputs, this is the dominant cost.

### Risks

- The inner Sinkhorn calls use `solve_warm_start` which skips `reset()` if `workspace.initialized = true`. On the first outer iteration, `initialized = false` so a full reset happens. On subsequent iterations, the warm start may carry stale dual variables if the barycenter changed substantially. In practice, the outer convergence criterion (1e-5 log-domain change) ensures the barycenter doesn't jump discontinuously.
- `BarycenterConfig.convergence_threshold = 1e-5` applies to log-domain weight differences. Users comparing this to `SinkhornConfig.convergence_threshold = 1e-5` (which is L1 marginal residuals) must understand these are different metrics — the barycenter threshold is tighter in the sense that log(1 + 1e-5) ≈ 1e-5 change in weight, while the Sinkhorn threshold is total-variation-like.

## References

- [`crates/transport/src/barycenter.rs`](/Users/lion/projects/lattice/crates/transport/src/barycenter.rs) — both barycenter algorithms
- [`crates/transport/src/logsumexp.rs`](/Users/lion/projects/lattice/crates/transport/src/logsumexp.rs) — `OnlineLogSumExp`, `max_abs_diff`, `normalize_log_weights`
- [`crates/transport/src/sinkhorn.rs`](/Users/lion/projects/lattice/crates/transport/src/sinkhorn.rs) — `SinkhornSolver` used as inner loop
- Cuturi & Doucet, "Fast Computation of Wasserstein Barycenters", ICML 2014
