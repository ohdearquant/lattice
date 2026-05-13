# ADR-036: Log-Domain Numerical Stability

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-transport

## Context

The Sinkhorn algorithm iterates on a transport coupling `γ = diag(u) K diag(v)` where
`K[i,j] = exp(-C[i,j] / ε)`. For small ε (needed to approximate exact optimal transport),
`C[i,j] / ε` can be large (e.g., C=2.0, ε=0.01 gives K=exp(-200) ≈ 0, underflowing to zero
in f32). When K entries underflow, the row/column sums become zero, and the scaling update
divides by zero — producing NaN that propagates through the entire solve.

The `LogDomainSinkhornSolver` in `sinkhorn_log.rs` adds epsilon-scaling acceleration
(Schmitzer 2019) on top of the already log-domain base solver. Together they eliminate
numerical instability at small ε.

## Decision

All internal computations work in log space. The Gibbs kernel is never materialized.
A small set of numerical primitives in `logsumexp.rs` centralize all transitions between
log and linear probability domains.

### Key Design Choices

**Never materialize the Gibbs kernel**

The core design invariant (stated in the `logsumexp.rs` module docstring) is:

> never materialize the Gibbs kernel `K = exp(-C / epsilon)` directly.

Instead, the log-domain update is:

```
log_u[i] = ln(a[i]) - LSE_j(log_v[j] - C[i,j] / ε)
```

where LSE is log-sum-exp. The value `log_v[j] - C[i,j] / ε` stays in log space throughout;
`exp()` is called only at the very end when recovering transport mass for statistics.

**`safe_ln(value, floor)` — clamp marginals away from zero**

```rust
pub fn safe_ln(value: f32, floor: f32) -> f32 {
    let clamped = if value.is_finite() { value.max(floor) } else { floor };
    ln(clamped)
}
```

In production, users pass marginals with zeros (after filtering or importance truncation).
Zero mass entries would produce `ln(0) = -inf` which then propagates as NaN via `inf - inf`.
Clamping to `floor = 1e-12` (the default `min_marginal`) makes the algorithm robust while
keeping the perturbation bounded and explicit. The magnitude of perturbation is documented:
for a distribution with 100 entries each contributing `1e-12`, total phantom mass is `1e-10`,
negligible compared to typical marginal values.

Zero-mass entries are now also rejected at input validation (FP-024): `validate_problem`
requires `value > 0` for all marginals. `safe_ln` thus sees only positive values in the
normal path; `min_marginal` clamping handles the residual edge cases.

**`safe_exp(log_value)` — guard against f32 underflow**

```rust
const EXP_UNDERFLOW_CUTOFF: f32 = -103.97208;

pub fn safe_exp(log_value: f32) -> f32 {
    if log_value < EXP_UNDERFLOW_CUTOFF { 0.0 } else { exp(log_value) }
}
```

`exp(-104)` ≈ `1.2e-45` which is below f32's smallest subnormal (`~1.4e-45`), producing
0.0 in round-to-nearest mode. Subnormal numbers incur a performance penalty on some CPUs
(x87 "denormal assist"). The cutoff is set conservatively at `-103.97208` (≈ `ln(f32::MIN_POSITIVE)`)
to ensure clean zero rather than a subnormal. `safe_exp` is only called when recovering
transport mass for statistics — it is not in the inner Sinkhorn loop.

**`logaddexp(a, b)` — stable log(exp(a) + exp(b))**

```rust
pub fn logaddexp(a: f32, b: f32) -> f32 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + log1p(exp(lo - hi))
}
```

Uses the standard max-subtraction trick. Special cases for `a == LOG_ZERO` (which is
`f32::NEG_INFINITY`) return `b` directly, avoiding `NEG_INFINITY + log1p(...)` which
would produce NaN.

**`OnlineLogSumExp` — one-pass stable accumulator**

The inner Sinkhorn loop computes `LSE_j(log_v[j] - C[i,j] / ε)` for each row `i`.
`OnlineLogSumExp` tracks `(max, sum_of_rescaled_exp)` and finalizes as `max + ln(sum)`.
This requires only one `ln` call per row, versus one `log1p` call per column when
chaining `logaddexp`. For n=100 columns, this reduces from 100 transcendental calls
to 1 per row. The online formula is:

```
when x > max:  sum = sum * exp(max - x) + 1.0; max = x
otherwise:     sum += exp(x - max)
finish:        max + ln(sum)
```

Edge case: `sum = 0.0` at finish returns `f32::NEG_INFINITY` (representing log(0)),
not NaN.

**Epsilon scaling in `LogDomainSinkhornSolver`**

For small target ε, a single-stage Sinkhorn solve with that ε may converge slowly because
the problem landscape is highly non-smooth. Epsilon-scaling (Schmitzer 2019) solves a
geometric sequence of easier problems first:

`ε_0 > ε_1 > ... > ε_k = ε_target` (e.g., `1.0, 0.5, 0.25, ..., 0.01`)

Each stage warm-starts from the dual variables of the previous stage. Each stage runs
`iterations_per_stage` inner Sinkhorn iterations. The final stage uses the target ε.

`EpsilonScalingSchedule` validates: `start > target > 0`, `factor ∈ (0, 1)`,
`iterations_per_stage ≥ 1`, and enforces a `MAX_EPSILON_STAGES = 10_000` cap on the
geometric sequence to catch near-unit factors that would loop forever.

**`LogDomainSinkhornConfig.base` vs schedule interaction**

When no schedule is set, `base.epsilon` is used directly (single stage). When a schedule
is set, the base config's epsilon is _ignored_ for the solve — only `schedule.target`
determines the final epsilon. The `base.max_iterations` is also overridden per stage
by `iterations_per_stage`. Only `base.convergence_threshold`, `base.min_marginal`, etc.
carry over. This is a subtle API contract documented in the type.

### Alternatives Considered

| Alternative                                          | Pros                                    | Cons                                                                                 | Why Not                                                           |
| ---------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| f64 arithmetic                                       | More headroom before underflow/overflow | 2x memory; SIMD throughput halved; doesn't fix the fundamental issue at very small ε | f64 just shifts the underflow threshold, doesn't eliminate it     |
| Primal scaling (standard Sinkhorn on K matrix)       | Simpler code                            | Numerically unstable for ε < 0.1; underflow to zero corrupts row/col sums            | Instability is the root cause this ADR addresses                  |
| Soft-min / log-sum-exp with periodic renormalization | Used in some implementations            | Requires careful renormalization frequency tuning                                    | Log-domain from the start eliminates the need to renormalize      |
| Skip the floor clamp in `safe_ln`                    | Strict correctness (no phantom mass)    | NaN propagation on any zero-mass entry                                               | Defensive clamping with documented magnitude is better than crash |

## Consequences

### Positive

- The solver handles ε as small as 0.001 on unit-normalized cost matrices without numerical failure.
- `EXP_UNDERFLOW_CUTOFF` is a named constant with a detailed comment explaining its derivation — future readers can verify the value rather than treating it as magic.
- `OnlineLogSumExp` is a reusable primitive used by both the Sinkhorn solver and the barycenter computation.

### Negative

- The log-domain formulation requires `exp()` in the inner loop for `OnlineLogSumExp.push()`. This is unavoidable — the alternative would require two passes over the data (one for max, one for sum). The online one-pass variant is the correct tradeoff.
- `safe_ln` silently clamps zero-mass marginals. If a caller passes marginals with many zeros (e.g., a sparse histogram), the result will be subtly wrong rather than erroring. `validate_problem` now rejects zero entries, but the clamp remains as a defense-in-depth layer.

### Risks

- The `EXP_UNDERFLOW_CUTOFF = -103.97208` is architecture-specific: on x87 hardware with 80-bit extended precision, `exp(-104)` may not underflow to zero. Rust's `f32::exp` uses the native 32-bit instruction path on modern targets, so this concern is academic.
- Epsilon-scaling with `factor` very close to 1.0 (e.g., 0.999) would generate thousands of stages before reaching `target`. The `MAX_EPSILON_STAGES = 10_000` guard catches this, returning an `InvalidEpsilonSchedule` error.

## References

- [`crates/transport/src/logsumexp.rs`](/Users/lion/projects/lattice/crates/transport/src/logsumexp.rs) — all numerical primitives
- [`crates/transport/src/sinkhorn_log.rs`](/Users/lion/projects/lattice/crates/transport/src/sinkhorn_log.rs) — `LogDomainSinkhornSolver`, `EpsilonScalingSchedule`
- [`crates/transport/src/sinkhorn.rs`](/Users/lion/projects/lattice/crates/transport/src/sinkhorn.rs) — base solver using these primitives
- Schmitzer, "Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems", SIAM J. Sci. Comput. 2019
