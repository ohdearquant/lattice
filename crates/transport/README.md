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

| Module           | Responsibility                                                                                                                                                                          |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sinkhorn`       | Core balanced Sinkhorn solver. `SinkhornSolver`, `SinkhornConfig`, `SinkhornWorkspace`, `SinkhornResult`. Also `uniform_weights`, `normalize_weights`.                                  |
| `sinkhorn_log`   | Log-domain solver with epsilon-scaling schedule. `LogDomainSinkhornSolver`, `LogDomainSinkhornConfig`, `EpsilonScalingSchedule`. More numerically stable for small epsilon.             |
| `unbalanced`     | KL-relaxed marginal variant. `UnbalancedSinkhornSolver`, `UnbalancedConfig`, `UnbalancedResult`. Useful when source/target marginals are imprecise.                                     |
| `cost`           | Cost matrix abstraction. `CostMatrix` trait, `DenseCostMatrix`, `PairwiseCostMatrix`, `SquaredEuclidean`, `CosineDistance`.                                                             |
| `divergence`     | Debiased Sinkhorn divergence. `SinkhornDivergence`, `sinkhorn_divergence`, `point_set_sinkhorn_divergence`. Removes the self-transport bias from raw OT cost.                           |
| `transport_plan` | Sparse transport plan extraction. `SparseTransportPlan`, `extract_sparse_plan`, `log_gamma`.                                                                                            |
| `barycenter`     | Wasserstein barycenter computation. `FixedSupportBarycenter`, `FreeSupportBarycenter`, `BarycenterConfig`.                                                                              |
| `drift`          | High-level embedding drift API. `detect_drift_records`, `detect_drift_memories`, `DriftReport`, `DriftSummary`, `DriftMetricKind`, `PerEntryDisplacement`.                              |
| `online_drift`   | Streaming (sliding-window) drift detection. `OnlineDriftDetector`, `OnlineDriftConfig`, `OnlineDriftSignal`. Calls `point_set_sinkhorn_divergence` every `check_interval` observations. |
| `logsumexp`      | Stable log-sum-exp arithmetic used throughout the solvers. Internal numeric foundation.                                                                                                 |
| `math`           | Internal numeric helpers (private).                                                                                                                                                     |

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

## Crate reference

The sections below are generated by `scripts/crate-readme.py` from the source tree and are refreshed after pushes to `main`. Text between the `generated` markers is overwritten; edit the source or the script instead.

<!-- generated:facts:begin -->

### Facts

- Version: `0.15.0`
- Minimum supported Rust version: `1.93`
- Edition: `2024`
- License: `MIT OR Apache-2.0`
- crates.io: <https://crates.io/crates/lattice-transport>
- API documentation: <https://docs.rs/lattice-transport>

<!-- generated:facts:end -->

<!-- generated:layout:begin -->

### Source layout

The tracked files under `src/`; each Rust file shows the first sentence of its `//!` module documentation.

<details>
<summary><code>src/</code>: 15 files</summary>

- `barycenter.rs`: Wasserstein barycenters.
- `cost.rs`: Cost matrix abstractions.
- `divergence.rs`: Debiased Sinkhorn divergence.
- `drift.rs`: High-level embedding drift API.
- `lib.rs`: Sinkhorn optimal transport for embedding distribution drift detection.
- `logsumexp.rs`: Numerically stable helpers for log-domain Sinkhorn updates.
- `math.rs`: Small math shim for transport module.
- `online_drift.rs`: Streaming online drift detection via Sinkhorn divergence.
- `online_drift/`
  - `tests.rs`
- `reference_cases.rs`: Reference datasets and comparison helpers (test-only).
- `sinkhorn.rs`: Core balanced Sinkhorn solver.
- `sinkhorn_log.rs`: Log-domain stabilized variants beyond the plain fixed-epsilon solver.
- `tests.rs`: Tests for the Sinkhorn optimal transport module.
- `transport_plan.rs`: Sparse transport plan extraction.
- `unbalanced.rs`: Unbalanced Sinkhorn with KL-relaxed marginals.

</details>

Targets declared by the manifest, including the ones Cargo discovers by directory:

- lib (1): `lattice_transport`
- example (2): `drift_compare`, `sinkhorn_ot`
- test (1): `input_validation`
- bench (1): `round0_transport`

<!-- generated:layout:end -->

<!-- generated:stats:begin -->

### Statistics

Rust lines by kind (code, comment, blank; doc comments count as comment), test functions (`#[test]` and `#[<runtime>::test]` attributes), items declared `pub` (not `pub(crate)`; reachability through private modules is not resolved) and occurrences of the `unsafe` keyword outside comments and strings.

| Area        | Files | Code | Comment | Blank | Tests | `pub` items | `unsafe` |
| ----------- | ----- | ---- | ------- | ----- | ----- | ----------- | -------- |
| `src/`      | 15    | 4409 | 924     | 390   | 34    | 140         | 0        |
| `tests/`    | 1     | 44   | 15      | 8     | 3     | 0           | 0        |
| `benches/`  | 1     | 404  | 45      | 54    | 0     | 0           | 0        |
| `examples/` | 2     | 212  | 69      | 50    | 4     | 0           | 0        |
| total       | 19    | 5069 | 1053    | 502   | 41    | 140         | 0        |

Other tracked files (the README excluded):

| Extension | Files | Size    |
| --------- | ----- | ------- |
| `.md`     | 3     | 4.4 KiB |
| `.toml`   | 1     | 814 B   |

<!-- generated:stats:end -->

<!-- generated:docs:begin -->

### In-crate documentation

- [Changelog](CHANGELOG.md) (`CHANGELOG.md`)
- [lattice-transport — extended docs](docs/INDEX.md) (`docs/INDEX.md`)
- [lattice-transport design](docs/design.md) (`docs/design.md`)

<!-- generated:docs:end -->

<!-- generated:features:begin -->

### Features

The manifest declares no features.

<!-- generated:features:end -->

<!-- generated:dependencies:begin -->

### Workspace dependencies

- Depends on: no other crate in this workspace
- Used by: no other crate in this workspace
- Used in tests by: `lattice-embed`

<!-- generated:dependencies:end -->

<!-- generated:adrs:begin -->

### Architecture decisions

ADRs under `docs/adr/` that name `lattice-transport` or `crates/transport/`:

- [ADR-014: Embedding Service](../../docs/adr/ADR-014-embedding-service.md)
- [ADR-035: Sinkhorn-Knopp Balanced OT Solver](../../docs/adr/ADR-035-sinkhorn-solver.md)
- [ADR-036: Log-Domain Numerical Stability](../../docs/adr/ADR-036-log-domain-stability.md)
- [ADR-037: Cost Matrix Abstractions](../../docs/adr/ADR-037-cost-matrices.md)
- [ADR-038: Wasserstein Barycenters](../../docs/adr/ADR-038-barycenters.md)
- [ADR-039: Sinkhorn Divergence](../../docs/adr/ADR-039-divergence-measures.md)
- [ADR-055: Online Distribution Drift Detection via Sinkhorn Divergence](../../docs/adr/ADR-055-online-drift-detection.md)
- [ADR-103: Splitting lattice-inference into focused crates behind a facade](../../docs/adr/ADR-103-inference-crate-decomposition.md)
- [ADR-ALIGNMENT-AUDIT: ADR Set Classification as of 2026-06-30](../../docs/adr/ADR-ALIGNMENT-AUDIT.md)
- [Architecture Decision Records](../../docs/adr/INDEX.md)

<!-- generated:adrs:end -->

<!-- generated:benchmarks:begin -->

### Benchmarks

No CI benchmark data for this crate: `bench-update.yml` records only `lattice-embed` `simd`, `lattice-inference` `elementwise_cpu_bench`.

<!-- generated:benchmarks:end -->
