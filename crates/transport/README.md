# lattice-transport

**Stability tier: Unstable.** The Sinkhorn API is functional but currently consumed by only one
crate, so its shape may still change.

---

## What This Crate Is

`lattice-transport` implements entropy-regularized optimal transport (Sinkhorn-Knopp algorithm)
in log-domain for numerical stability. Its primary use case is quantifying how much embedding
geometry shifts between model versions — for example, when `lattice-embed` swaps from
BGE-small to mE5-small and needs to detect distribution drift across stored embeddings.

For the architecture layers and the design rationale behind the log-domain formulation, see
[`docs/design.md`](docs/design.md).

---

## Boundary Constraints

**f32-only.** All tensor inputs and computation use `f32`. No f16, bf16, or integer paths exist.

**No inference dependency.** `lattice-transport` has zero intra-workspace dependencies. Its
`Cargo.toml` lists only `rayon` and `serde` as runtime dependencies. It does not depend on
`lattice-inference`, `lattice-embed`, or any other lattice crate. This isolation is intentional:
the crate provides pure math primitives that any consumer can take without pulling in the full
transformer stack.

**No BLAS/LAPACK.** All arithmetic is pure Rust. Log-domain operations prevent numerical
underflow without external linear algebra libraries.

**Pre-allocated workspaces.** `SinkhornWorkspace` avoids heap allocation during the inner
iteration loop. Callers create a workspace once and reuse it across calls.

---

## Module Map

| Module | Responsibility |
|--------|---------------|
| `sinkhorn` | Core balanced Sinkhorn solver. `SinkhornSolver`, `SinkhornConfig`, `SinkhornWorkspace`, `SinkhornResult`. Also `uniform_weights`, `normalize_weights`. |
| `sinkhorn_log` | Log-domain solver with epsilon-scaling schedule. `LogDomainSinkhornSolver`, `LogDomainSinkhornConfig`, `EpsilonScalingSchedule`. More numerically stable for small epsilon. |
| `unbalanced` | KL-relaxed marginal variant. `UnbalancedSinkhornSolver`, `UnbalancedConfig`, `UnbalancedResult`. Useful when source/target marginals are imprecise. |
| `cost` | Cost matrix abstraction. `CostMatrix` trait, `DenseCostMatrix`, `PairwiseCostMatrix`, `SquaredEuclidean`, `CosineDistance`. |
| `divergence` | Debiased Sinkhorn divergence. `SinkhornDivergence`, `sinkhorn_divergence`, `point_set_sinkhorn_divergence`. Removes the self-transport bias from raw OT cost. |
| `transport_plan` | Sparse transport plan extraction. `SparseTransportPlan`, `extract_sparse_plan`, `log_gamma`. |
| `barycenter` | Wasserstein barycenter computation. `FixedSupportBarycenter`, `FreeSupportBarycenter`, `BarycenterConfig`. |
| `drift` | High-level embedding drift API. `detect_drift_records`, `detect_drift_memories`, `DriftReport`, `DriftSummary`, `DriftMetricKind`, `PerEntryDisplacement`. |
| `online_drift` | Streaming (sliding-window) drift detection. `OnlineDriftDetector`, `OnlineDriftConfig`, `OnlineDriftSignal`. Calls `point_set_sinkhorn_divergence` every `check_interval` observations. |
| `logsumexp` | Stable log-sum-exp arithmetic used throughout the solvers. Internal numeric foundation. |
| `math` | Internal numeric helpers (private). |

---

## Public Types (re-exported at crate root)

**Sinkhorn solver:**
`SinkhornSolver`, `SinkhornConfig`, `SinkhornWorkspace`, `SinkhornResult`,
`BatchSolveResult`, `MarginalPair`, `ProgressObserver`, `ProgressState`,
`SinkhornError`, `normalize_weights`, `uniform_weights`

**Cost abstractions:**
`CostMatrix`, `DenseCostMatrix`, `PairwiseCostMatrix`, `ClosureCostMatrix`,
`ContiguousPoints`, `SquaredEuclidean`, `CosineDistance`, `PointMetric`, `PointSet`, `CostError`

**Log-domain solver:**
`LogDomainSinkhornSolver`, `LogDomainSinkhornConfig`, `LogDomainSinkhornResult`,
`EpsilonScalingSchedule`, `EpsilonStageSummary`

**Transport plan:**
`SparseTransportPlan`, `SparseTransportEntry`, `extract_sparse_plan`, `log_gamma`

**Divergence:**
`SinkhornDivergence`, `sinkhorn_divergence`, `point_set_sinkhorn_divergence`

**Unbalanced:**
`UnbalancedSinkhornSolver`, `UnbalancedConfig`, `UnbalancedResult`

**Barycenter:**
`FixedSupportBarycenter`, `FreeSupportBarycenter`, `BarycenterConfig`, `BarycenterProblem`,
`FixedSupportBarycenterWorkspace`, `FreeSupportConfig`, `OwnedPointMeasure`,
`fixed_support_barycenter`, `free_support_barycenter`

**Drift:**
`detect_drift_records`, `detect_drift_memories`, `DriftReport`, `DriftSummary`,
`DriftConfig`, `DriftMetricKind`, `DriftSolverMode`, `DriftWeighting`,
`EmbeddingRecord`, `MemoryLike`, `PerEntryDisplacement`

**Online drift:**
`OnlineDriftDetector`, `OnlineDriftConfig`, `OnlineDriftSignal`

---

## Inference-Side Bridge (Not Yet Built)

`OnlineDriftDetector` lives in this crate. Per ADR-055, the inference-side `DriftSampler`
bridge that would feed live KV-cache statistics into the detector is not yet built. The two
pieces exist at opposite ends of the dependency graph: `lattice-transport` is a leaf crate
with no inference dependency, while the bridge would need to observe `lattice-inference`
internals. The integration point is tracked in ADR-055.

---

## Quick Example

```rust
use lattice_transport::{
    SinkhornSolver, SinkhornConfig, SinkhornWorkspace,
    DenseCostMatrix, uniform_weights,
};

let cost = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
let source = uniform_weights(2);
let target = uniform_weights(2);
let solver = SinkhornSolver::default();
let mut workspace = SinkhornWorkspace::new(2, 2);
let result = solver.solve(&cost, &source, &target, &mut workspace).unwrap();
assert!(result.converged);
```

For debiased divergence, non-uniform marginals, and embedding-drift detection examples, see
the `lattice-transport` section of [`docs/examples.md`](../../docs/examples.md).
