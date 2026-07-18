# ADR-083: Backward SIMD Dispatch Contract

**Status**: Accepted
**Date**: 2026-07-17
**Crate**: lattice-inference, lattice-tune

## Context

The full-depth LoRA training tape spans `lattice-inference` backward primitives and
`lattice-tune::lora::train_core`. Its arithmetic surface is not one homogeneous loop. Current
`main` contains this inventory:

| Surface                  | Current execution                                                                          | Correctness coverage                                         |
| ------------------------ | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| `cross_entropy_backward` | Scalar max, exponential reduction, and row update                                          | Central difference; masked rows                              |
| `linear_vjp`             | `matmul_into`; Accelerate on macOS, scalar on other targets                                | Scalar parity and central difference                         |
| `lora_vjp`               | Scalar outer products; `matmul_into` for `B^T g` and `A^T(B^T g)`                          | Scalar parity and central difference                         |
| `rmsnorm_backward`       | Scalar reduction and elementwise update                                                    | Central difference                                           |
| `rope_backward`          | Scalar paired rotation                                                                     | Central difference                                           |
| `swiglu_backward`        | `matmul_into` for three linear reductions; scalar nonlinear update                         | Scalar parity and central difference                         |
| `gqa_backward`           | Scalar attention/reduction loops; calls dispatched `linear_vjp` for projection VJPs        | GQA LoRA central difference and fail-closed non-finite tests |
| `gdn_backward`           | Scalar recurrence and inner reductions; calls `lora_vjp` for adapter projections           | Central difference for `dx` and all ten LoRA arrays          |
| `train_core` work        | `lm_head_logits` uses `matmul_bt`; `rmsnorm_seq` and per-position cross entropy are scalar | LM-head scalar parity plus assembled trainer checks          |

The Stage 1 GEMM reuse predates this ADR. It improves important matrix-shaped work, but it does not
provide a dedicated backward dispatch layer, portable SIMD for `matmul_into`, forced backend
selection, or a common contract for the remaining scalar primitives. `train_grad_full` and
`train_micro_lora` both call the same `train_core`; there is no longer a duplicate tape to optimize
independently.

The remaining work must be profile-led. Vectorizing every loop would add unsafe code and multiple
accumulation orders without proving that dispatch overhead, low LoRA rank, recurrence dependencies,
or transcendental work leave a useful arithmetic win.

## Decision

Future dedicated backward CPU kernels live under `crates/inference/src/backward/simd/`. The module
owns dispatch and arithmetic implementations; the public backward operations remain the semantic
facade.

### Scalar references and allocation boundary

Each optimized primitive starts with a named scalar reference implementation. The dispatchable core
uses an `_into` entry point that writes caller-provided buffers. Existing public `Vec`-returning
wrappers may allocate and delegate during migration.

Moving the complete training tape to reusable scratch buffers is separate work. A SIMD PR must not
combine arithmetic vectorization with a broad allocation-lifetime rewrite, because the two changes
need different benchmarks and failure isolation.

### Capability detection and backend selection

CPU capability detection moves from the forward-private `forward::cpu::simd` module to an
inference-internal neutral CPU module and remains cached in one `OnceLock`. Forward and backward
dispatch reuse that single result; they do not maintain duplicate detectors.

The backward backend order is:

1. NEON on AArch64 when the recorded capability is available.
2. AVX2 plus FMA on x86_64 only when both runtime checks succeed.
3. Scalar everywhere else.

AVX-512 is outside the initial backward contract. Metal and WGPU are not fallback choices for this
layer; GPU training requires a separate end-to-end design rather than per-primitive CPU dispatch.

Production entry points use detected capabilities. Tests also receive internal forced-scalar and
forced-backend entry points. A forced backend must either run and emit its explicit backend marker or
fail as unavailable; it must never silently fall back and report success.

### Safety and tail contract

Every SIMD entry point validates its complete slice and shape contract before entering an unsafe
kernel. Each unsafe block documents the runtime feature proof, pointer validity, and initialized
output range. Public unsafe functions, if any are required, include a `# Safety` section.

Vector loops operate only on complete lanes. A scalar tail handles the remainder without reading or
writing past the declared range. Tests cover zero, sub-vector, exact-vector, odd, and multi-vector
lengths. Non-finite and masked-row behavior must remain identical to the scalar facade unless a
separate numeric-contract decision explicitly changes it.

### Dependency direction

`lattice-inference` owns capability detection, scalar oracles, dispatch, and optimized backward
kernels. `lattice-tune` continues to depend downward on `lattice-inference` and may call its
backward facade. `lattice-inference` must not depend on `lattice-tune`.

### Profiling and verification order

Work proceeds in this order, stopping when representative measurements no longer justify the added
complexity:

| Stage | Primitive or work unit                                                                    | Required gate before promotion                                                            |
| ----: | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
|     0 | Tiny and production-shape Criterion baselines; forced dispatch markers                    | Before/after distributions and proof that the intended backend ran                        |
|     1 | `linear_vjp` and `train_core::lm_head_logits` on a shared GEMV/matmul-transpose substrate | Scalar differential, central difference, and accumulation-lane mutation                   |
|     2 | `swiglu_backward` and the tape's `swiglu_forward`                                         | Scalar differential across ordinary and saturated inputs; separate gate/up/down mutations |
|     3 | `gqa_backward`, including score/context reductions and RoPE backward                      | Causal odd-length/GQA fixtures, all q/v LoRA gradients, and non-finite sibling-head gate  |
|     4 | `gdn_backward` inner GEMVs and reductions                                                 | Full recurrence parity plus `dx` and ten-array LoRA central differences                   |
|     5 | RMSNorm, RoPE, and cross-entropy reductions and elementwise loops                         | Tail/mask invariants and unchanged non-finite behavior                                    |
|     6 | `lora_vjp`                                                                                | Rank-sensitive benchmarks plus separate `A`, `B`, and `x` gradient mutations              |

Stage numbering describes evaluation order, not a commitment to implement every stage. Stage 1's
existing forward-GEMM reuse is the baseline for further work, not evidence that the dedicated
backward dispatch contract is complete. In particular, low-rank LoRA kernels do not land unless
benchmarks show that SIMD setup costs are recovered.

Every stage retains a SIMD-versus-scalar differential over sparse, dense, zero, odd, and tail-heavy
inputs as appropriate. Floating-point comparisons use a documented scale-aware tolerance. Existing
central-difference gates keep `max_rel < 1e-2`; a deliberate mutation must make the relevant test
fail before the change is accepted. The assembled trainer gradient check and real-model differential
run after each primitive stage because self-consistent local tests cannot detect a shared wrong
model convention.

The GDN reverse-time recurrence is never vectorized across dependent timesteps. SIMD may accelerate
independent heads or inner GEMV/reduction dimensions while preserving recurrence order and the
f64-sensitive oracle behavior.

## Alternatives Considered

| Alternative                                           | Pros                             | Cons                                                                                  | Why Not                                        |
| ----------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Let each primitive choose its own dispatch            | Small local PRs                  | Duplicate detection, safety rules, and test controls drift                            | One shared contract is easier to verify        |
| Reuse `lattice-embed` SIMD kernels                    | Existing public vector APIs      | Different operation shapes; would reverse the dependency edge                         | ADR-013 keeps transformer kernels in inference |
| Replace wrappers and allocations in the first SIMD PR | Immediate zero-allocation facade | Mixes arithmetic and lifetime changes; obscures attribution                           | Buffer migration is measured separately        |
| Add Metal or WGPU primitive fallbacks                 | Potential GPU throughput         | Transfer and command overhead dominate primitive calls; no end-to-end training design | CPU dispatch remains CPU-only                  |
| Vectorize every listed primitive                      | Uniform implementation           | Complexity may exceed measured benefit, especially at low rank                        | Profile and stop after each stage              |

## Consequences

### Positive

- One capability detector and backend policy serves forward and backward CPU kernels.
- Scalar implementations remain executable correctness references and portable fallbacks.
- Forced backend markers make a skipped SIMD path fail closed in tests.
- `_into` kernels permit later scratch-buffer reuse without coupling it to intrinsic work.

### Negative

- Multiple accumulation orders require explicit numeric tolerances and mutation checks.
- Named scalar references add code that must be maintained beside optimized kernels.
- Stage 0 profiling and cross-architecture validation add work before each optimization.

### Risks

- A backend test can give false confidence if its marker is not asserted.
- SIMD reassociation can hide convention errors when both the implementation and local oracle share
  the same wrong model layout; assembled trainer and real-model checks remain mandatory.
- Vectorizing across GDN timesteps would change the recurrence and is prohibited.

The streaming/cache-residency and parallel sample-accumulation levers tracked in issue #737 remain
separate and unresolved. This ADR governs backward arithmetic dispatch only.

## References

- ADR-002 — inference runtime SIMD dispatch
- ADR-013 — embedding-vector SIMD scope boundary
- ADR-058 — CPU performance measurement and regression policy
- `crates/inference/src/backward/ops.rs`
- `crates/inference/src/backward/attention_gqa.rs`
- `crates/inference/src/attention/gdn_backward.rs`
- `crates/inference/src/forward/cpu/{simd.rs,matmul.rs}`
- `crates/tune/src/lora/train_core.rs`
- Issue #737 — corpus-scaling limits and independent streaming/parallelism work
