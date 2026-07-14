# lattice-transport design

`lattice-transport` implements entropy-regularized optimal transport (the
Sinkhorn-Knopp algorithm) in the log domain. Its primary use case is quantifying
how much embedding geometry shifts between model versions — for example, when
`lattice-embed` swaps from BGE-small to mE5-small and needs to detect distribution
drift across stored embeddings.

## Architecture

The crate is organized into layers, leaf-most first:

- **Numerical foundation** — `math`, `logsumexp`: stable log-domain arithmetic.
- **Cost abstraction** — `cost`: the `CostMatrix` trait with its `DenseCostMatrix`
  and `PairwiseCostMatrix` implementations, plus the `PointMetric` distance types
  `SquaredEuclidean` and `CosineDistance`.
- **Core solvers** — `sinkhorn` (balanced), `sinkhorn_log` (epsilon-scaling
  schedule, more stable at small epsilon), `unbalanced` (KL-relaxed marginals for
  imprecise source/target masses).
- **Post-processing** — `transport_plan` (sparse plan extraction), `divergence`
  (debiased Sinkhorn divergence, which removes the self-transport bias from raw
  optimal-transport cost).
- **High-level API** — `drift` (embedding drift detection), `online_drift`
  (streaming sliding-window detection), `barycenter` (Wasserstein barycenters).

## Key design choices

**Log-domain throughout.** The solvers never materialize the Gibbs kernel
`exp(-C/epsilon)` directly; all scaling vectors stay in log space and combine
through `logsumexp`. This is the crate's load-bearing numerical invariant: it
prevents underflow at small regularization epsilon, where the kernel entries would
otherwise round to zero in `f32`.

**Pre-allocated workspaces.** `SinkhornWorkspace` holds the scratch buffers the
inner iteration needs, so a caller allocates once and reuses the workspace across
many solves rather than allocating on the heap each call.

**No external linear algebra.** All arithmetic is pure Rust with no BLAS/LAPACK
requirement. The log-domain formulation is what makes this practical without a
tuned numerical library underneath.

**f32-only.** All tensor inputs and computation use `f32`; there are no f16, bf16,
or integer paths.

**Leaf crate.** `lattice-transport` has no intra-workspace dependencies. It pulls
in only `rayon` and `serde` at runtime and depends on no other lattice crate, so a
consumer can take these optimal-transport primitives without pulling in the
transformer stack. The inference-side bridge that would feed live statistics into
`OnlineDriftDetector` sits at the opposite end of the dependency graph and is
tracked in ADR-055.
