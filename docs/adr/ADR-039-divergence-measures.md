# ADR-039: Sinkhorn Divergence

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-transport

## Context

Regularized optimal transport cost `W_ε(a, b)` is not a proper divergence: it does not
equal zero when `a = b`, because the entropy term introduces a positive self-cost. This
means raw `W_ε` cannot be used to measure whether two distributions are "the same" — the
baseline is non-zero and varies with distribution shape.

For embedding drift detection, the primary use case is: "is the current distribution
significantly different from the reference?" A baseline-corrected divergence that equals
zero on identical distributions is necessary.

## Decision

The Sinkhorn divergence (Genevay et al. 2018) removes the bias by subtracting half of each
self-interaction term:

```
S(a, b) = W_ε(a, b) - 0.5 * W_ε(a, a) - 0.5 * W_ε(b, b)
```

This requires three separate Sinkhorn solves: the cross-term and two self-terms.

### Key Design Choices

**Formula uses regularized OT cost, not raw transport cost (FP-023)**

The divergence formula operates on `regularized_cost = transport_cost - ε * entropy`,
not `transport_cost` alone. Using raw transport cost breaks both symmetry and positive
semi-definiteness of the divergence. This was a fixed bug (FP-023):

```rust
let value = cross.regularized_cost
    - 0.5 * self_source.regularized_cost
    - 0.5 * self_target.regularized_cost;
```

The `SinkhornResult` exposes both `transport_cost` (the raw primal cost `Σ γ[i,j] C[i,j]`)
and `regularized_cost = transport_cost - ε * entropy`. Using `regularized_cost` ensures
`S(a, a) ≈ 0` for all `a`, not just for specific distribution shapes.

**Three separate workspaces**

`sinkhorn_divergence` accepts three `SinkhornWorkspace` parameters (`workspace_xy`,
`workspace_xx`, `workspace_yy`). Reusing a single workspace would require resizing it
between the cross-term solve (n×m) and self-term solves (n×n and m×m). Three workspaces
preserve independently sized buffers across repeated divergence computations. Each term
uses `SinkhornSolver::solve`, which resets its workspace's dual variables before solving;
the current divergence API reuses buffer capacity but does not warm-start between calls.

**Three separate cost matrices**

The three solves use:

- `cost_xy`: from source points to target points (n×m)
- `cost_xx`: from source points to themselves (n×n, symmetric)
- `cost_yy`: from target points to themselves (m×m, symmetric)

All three must use the same metric for the formula to be valid. The `point_set_sinkhorn_divergence`
convenience function enforces this by accepting a single `metric: M` and building all three
cost matrices from it.

**Dense vs lazy selection at 16 MB**

```rust
const DENSE_COST_LIMIT_BYTES: usize = 16 * 1024 * 1024;

let total_bytes = (n_src * n_tgt + n_src * n_src + n_tgt * n_tgt) * 4;
if total_bytes <= DENSE_COST_LIMIT_BYTES {
    // Pre-materialize DenseCostMatrix for all three
} else {
    // Use PairwiseCostMatrix (lazy) for all three
}
```

Materializing all three dense cost matrices ahead of time avoids recomputing distances
on every solver iteration (500 iterations × 3 solves for each entry). For n_src = n_tgt = 50,
the three matrices are 10 KB + 10 KB + 10 KB = 30 KB — trivially materializable.
For n_src = n_tgt = 500, the combined size is 500×500×4 + 500×500×4 + 500×500×4 = 3 MB —
still within the 16 MB budget. The threshold is hit at roughly n_src = n_tgt ≈ 1150.

The 16 MB constant is a heuristic: it fits comfortably in L3 cache on an M4 Pro (12 MB L3)
for small problems, or fits within a reasonable per-request memory budget for larger ones.

**Lower-level `sinkhorn_divergence` API**

`sinkhorn_divergence` takes three pre-built `CostMatrix` references — useful when the
caller already has materialized cost matrices (e.g., from a cached computation). The
higher-level `point_set_sinkhorn_divergence` builds all three matrices internally.

**`SinkhornDivergence` result type**

```rust
pub struct SinkhornDivergence {
    pub value: f32,
    pub cross: SinkhornResult,
    pub self_source: SinkhornResult,
    pub self_target: SinkhornResult,
}
```

All three `SinkhornResult` values are returned, not just the divergence scalar.
This lets callers inspect per-solve convergence status, iteration counts, and residuals —
important for debugging cases where one of the three solves failed to converge.

### Alternatives Considered

| Alternative                              | Pros                            | Cons                                                                                          | Why Not                                                                     |
| ---------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Raw `W_ε(a, b)` without debiasing        | One solve instead of three      | Non-zero on identical distributions; can't serve as a drift threshold                         | Baseline varies — no stable threshold for "significant change"              |
| MMD (Maximum Mean Discrepancy)           | Closed form, O(n²)              | Kernel choice matters; no geometric interpretation; hard to relate to Wasserstein distances   | Less interpretable for embedding drift; no optimal transport interpretation |
| KL divergence on empirical distributions | O(n log n)                      | Requires binning continuous embeddings; sensitive to bin size; undefined when supports differ | Continuous embedding spaces don't have natural bins                         |
| Symmetric KL (Jensen-Shannon)            | Zero on identical distributions | Not a metric; doesn't capture geometry of embedding space                                     | No Wasserstein geometry                                                     |
| W₁ (unregularized, ε→0)                  | Exact geometric distance        | O(n³) computation; no closed form for continuous distributions                                | Too slow; Sinkhorn divergence is the tractable approximation                |

## Consequences

### Positive

- `S(a, a) ≈ 0` for any distribution `a` when the correct `regularized_cost` formula is used (FP-023 fix). This enables a single threshold parameter for drift detection regardless of distribution shape.
- The three-workspace design reuses independently sized buffers across repeated drift checks,
  avoiding allocation when the cross and self problem dimensions remain unchanged.
- The lower-level API `sinkhorn_divergence` is testable with synthetic `ClosureCostMatrix` inputs without needing real embedding data.

### Negative

- Three Sinkhorn solves per divergence computation. For n=100, each solve takes ~50 iterations × 100×100 operations × 1 ns/op ≈ 0.5 ms. Total ≈ 1.5 ms. For production drift detection over 1000 embeddings, this is ~15 ms per check — acceptable for background monitoring.
- The 16 MB dense-vs-lazy threshold is a heuristic constant, not configurable at the call site. Callers with specific memory budgets cannot override it without using the lower-level `sinkhorn_divergence` API.
- `self_source` and `self_target` solves are symmetric problems (`cost_xx` is symmetric). The solver does not exploit this symmetry — it runs full iterations on the symmetric problem. A symmetric Sinkhorn variant could halve the iteration cost for self-terms.

### Risks

- If `source = target` (same slice pointer), `sinkhorn_divergence` will compute `W_ε(a, a)` three times (as cross, self_source, and self_target). The result will be `0.0` (correct) but at 3x the cost of a single self-solve. `point_set_sinkhorn_divergence` does not detect this case.
- The debiasing formula `W_ε(a, b) - 0.5 W_ε(a, a) - 0.5 W_ε(b, b)` can be negative for large ε or when the three solves use different convergence states (e.g., one didn't converge). The `value` field in `SinkhornDivergence` is not clamped to `[0, ∞)`. Callers should check `cross.converged && self_source.converged && self_target.converged` before trusting the divergence value.

## References

- [`crates/transport/src/divergence.rs`](../../crates/transport/src/divergence.rs) — full implementation
- [`crates/transport/src/sinkhorn.rs`](../../crates/transport/src/sinkhorn.rs) — `SinkhornResult.regularized_cost` field
- [`crates/transport/src/cost.rs`](../../crates/transport/src/cost.rs) — `DenseCostMatrix`, `PairwiseCostMatrix` used for selection
- Genevay, Peyré, Cuturi, "Learning Generative Models with Sinkhorn Divergences", AISTATS 2018
