# ADR-037: Cost Matrix Abstractions

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-transport

## Context

The Sinkhorn solver needs a cost matrix `C[i,j]` for every (row, col) pair. The natural
representation is a dense `Vec<f32>` of size `n×m`. However:

1. For large n×m, materializing the full matrix requires `n×m×4` bytes of RAM upfront.
   For n=500, m=500, this is 1 MB — acceptable. For n=5000, m=5000, it's 100 MB.
2. For the Sinkhorn divergence computation (ADR-005), three cost matrices are needed
   simultaneously: `C_xy`, `C_xx`, `C_yy`. When they fit within 16 MB combined,
   pre-materializing avoids recomputing distances on every solver iteration.
3. Custom cost functions (non-Euclidean penalties, symbolic costs, graph distances) must
   be expressible without wrapping them in a dense matrix.

## Decision

Cost access is abstracted through a `CostMatrix` trait. Three concrete implementations
cover the common cases: `DenseCostMatrix` (pre-computed), `PairwiseCostMatrix` (lazy),
and `ClosureCostMatrix` (arbitrary callback). A `PointSet` trait and `PointMetric` trait
allow point-cloud costs to be built in a generic, type-safe way.

### Key Design Choices

**`CostMatrix` trait: three methods only**

```rust
pub trait CostMatrix {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn cost(&self, row: usize, col: usize) -> f32;
}
```

The solver calls `cost(row, col)` in its inner loop. The trait has no `unsafe` methods,
no batching API, and no access-pattern hints. This keeps the abstraction minimal; batching
optimization belongs in solver internals (see Rayon parallel paths in ADR-001), not in
the cost trait.

**`DenseCostMatrix`: row-major pre-computed**

The dense matrix stores `rows × cols` f32 values in row-major order in a `Vec<f32>`.
`cost(row, col)` is `data[row * cols + col]` — a single bounds-checked indexing operation.
The constructor `DenseCostMatrix::new(rows, cols, data)` asserts `rows * cols == data.len()`
in both debug and release builds (FP-030): a shape mismatch corrupts every subsequent
lookup, so it must be caught at construction rather than producing silent wrong results.

`from_point_sets(source, target, metric)` validates both point sets (non-empty, uniform
dimension, no NaN/Inf) then fills the matrix via `from_fn`. `from_fn(rows, cols, f)` calls
`f(row, col)` for every entry in row-major order.

**`PairwiseCostMatrix`: lazy distance computation**

`PairwiseCostMatrix<'a, X, Y, M>` holds references to two point sets and a metric.
`cost(row, col)` calls `metric.distance(source.point(row), target.point(col))` on demand.
This avoids materializing the full matrix at the cost of recomputing each distance on
every solver iteration call. For n=100 Sinkhorn iterations over a 200×200 matrix, this
recomputes 200×200 = 40K distances 100 times = 4M distance evaluations instead of
40K evaluations + 40K lookups.

The choice between dense and lazy is made by the caller based on problem size. The
`point_set_sinkhorn_divergence` convenience function makes this choice automatically at
a 16 MB combined threshold (ADR-005).

**`ClosureCostMatrix`: arbitrary callback**

`ClosureCostMatrix<F>` stores `rows`, `cols`, and `f: F` where `F: Fn(usize, usize) -> f32`.
This enables non-Euclidean costs (graph adjacency penalties, hybrid symbolic/neural costs,
time-varying costs) without requiring the caller to pre-materialize a dense matrix.

**Two distance metrics**

`SquaredEuclidean` computes `||x - y||²` — appropriate for Wasserstein-2 (W₂) distances.
`CosineDistance` computes `1 - cos(θ)`, with an `assume_unit_norm` flag: when true, the
norm computation is skipped and cosine is computed as a pure dot product. The result is
clamped to `[0, 2]` via `.clamp(0.0, 2.0)` to prevent negative values from floating-point
cancellation near identical vectors. `norm_floor = 1e-12` prevents division by zero for
zero vectors.

**`PointSet` trait with two built-in implementations**

`PointSet` abstracts over storage layouts: `[Vec<f32>]` (heap-allocated points) and
`[&[f32]]` (borrowed slices). `ContiguousPoints<'a>` handles flat buffers (`&[f32]` of
length `n × dim`) which are common when embeddings come from a BLAS operation or database
page. `ContiguousPoints::new` validates that `data.len()` is divisible by `dim`.

**`validate_point_set` for input sanitization**

Before building any cost matrix from point sets, `validate_point_set` checks: non-empty,
`dim() > 0`, uniform dimension across all points, no NaN/Inf values. This runs at
`DenseCostMatrix::from_point_sets` call time, not inside the solver inner loop.

### Alternatives Considered

| Alternative                                        | Pros                                  | Cons                                                                | Why Not                                                   |
| -------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------- |
| Single `DenseCostMatrix` type everywhere           | Simple; direct indexing in solver     | Forces materialization for all inputs; 100 MB for 5K×5K             | Lazy path needed for large inputs and Sinkhorn divergence |
| BLAS `dgemm` for distance matrix                   | Very fast materialization             | Requires BLAS dependency; only works for squared Euclidean          | Dependency-free pure Rust is required                     |
| Sparse cost matrix                                 | Enables subquadratic Sinkhorn         | Only valid for sparse support problems; adds significant complexity | Current problem sizes don't require sparse support        |
| Trait object (`Box<dyn CostMatrix>`) over generics | Simpler ergonomics in some call sites | Virtual dispatch in inner loop; measurable overhead per cell        | Monomorphized generics avoid vtable in inner loop         |

## Consequences

### Positive

- The solver is oblivious to whether the cost matrix is pre-computed or lazy. Tests can use `ClosureCostMatrix` to inject synthetic costs without constructing point arrays.
- `DenseCostMatrix` is serializable via `serde` for caching/debugging materialized cost matrices.
- `PairwiseCostMatrix` is `Copy` (holds only shared references and a metric value) — it can be shared across Rayon threads without `Arc`.

### Negative

- `CostMatrix` is not `Sync`-bounded by default. The Rayon parallel path requires `CostMatrix + Sync`. For `DenseCostMatrix` this is automatic (shared reference to `Vec<f32>`). For `ClosureCostMatrix<F>`, the closure must be `Sync`. For thread-unsafe cost functions (e.g., with `Cell` interior mutability), the parallel path is unavailable.
- `PairwiseCostMatrix` recomputes each `(row, col)` distance on every solver iteration. For a 500-iteration solve over a 200×200 matrix with cosine distance (each call ~384 multiplications), this is 500 × 200 × 200 × 384 ≈ 7.7 billion multiplications. Pre-materializing the dense matrix reduces this to 200 × 200 × 384 = 15.4 million multiplications plus 500 × 200 × 200 = 20 million lookups.

### Risks

- `DenseCostMatrix::cost(row, col)` does not bounds-check in the `data[idx]` access — `idx = row * cols + col` and the bounds check is on the `Vec` access. If `rows`/`cols` are correct (guaranteed by the constructor assert), this is safe. If the `data` field were made public and modified after construction, bounds checks would fail at runtime rather than at construction.
- `CosineDistance::assume_unit_norm = true` is a caller assertion, not a checked invariant. A caller that passes non-normalized vectors with this flag set will get incorrect cosine values silently.

## References

- [`crates/transport/src/cost.rs`](/Users/lion/projects/lattice/crates/transport/src/cost.rs) — all cost matrix types and metrics
- [`crates/transport/src/divergence.rs`](/Users/lion/projects/lattice/crates/transport/src/divergence.rs) — 16 MB dense/lazy selection logic
