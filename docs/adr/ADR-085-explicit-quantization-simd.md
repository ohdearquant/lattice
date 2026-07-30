# ADR-085: Explicit Quantization SIMD Dispatch

**Status**: Accepted
**Date**: 2026-07-28
**Crate**: lattice-embed

## Context

The finite-value reduction in `QuantizationParams::from_vector` previously
depended on LLVM auto-vectorizing a guarded loop. An unrelated codegen-unit
change demoted that loop to scalar instructions and regressed per-call INT8
quantization by 47.6%. The repair in #1062 added an explicit NEON reduction, but
the guarded conversion loops in INT8, INT4, and binary quantization retained the
same compiler sensitivity. The x86_64 reduction also remained scalar and
unmeasured.

The L2 norm loops in these constructors are different. Their scalar accumulation
order is part of the numerical behavior; vectorizing them would reassociate
floating-point addition.

## Decision

INT8, INT4, and binary constructors dispatch their guarded conversion loops to
explicit NEON kernels on aarch64 and AVX2 kernels on x86_64. INT8 finite min/max
and INT4 finite max-absolute reductions use the same architecture coverage.
Targets without the selected feature retain scalar fallbacks.

The kernels preserve the existing format:

- non-finite source lanes are replaced with zero before conversion;
- INT8 and INT4 use round-half-away-from-zero and their existing clamps;
- INT4 remains high-nibble-first and binary remains most-significant-bit-first;
- vector chunks use bounded unaligned loads, with safe scalar tails;
- L2 norm accumulation remains scalar-ordered.

Architecture-specific tests compare every SIMD result with the scalar reference
across boundary lengths, non-finite values, rounding boundaries, and binary
thresholds. Test-only thread-local counters prove each constructor executes its
applicable SIMD reduction and conversion kernels on capable hardware.

### Alternatives Considered

| Alternative                                 | Why not                                                                              |
| ------------------------------------------- | ------------------------------------------------------------------------------------ |
| Continue relying on LLVM auto-vectorization | Prior codegen-unit changes caused a silent 47.6% regression without a source change. |
| Add only disassembly checks                 | They detect demotion but leave throughput dependent on compiler heuristics.          |
| Vectorize L2 norm accumulation too          | Reassociation changes floating-point results and is outside this decision.           |
| Require AVX2 or NEON                        | CPU-first support requires a scalar fallback for older x86_64 and other targets.     |

## Consequences

Quantization no longer depends on guarded-loop auto-vectorization on supported
desktop architectures, and AVX2 closes the previously silent x86_64 gap. The
implementation adds unsafe intrinsic code that requires manual review, scalar
equivalence tests, and architecture-specific compilation.

The draft must receive uncontended A/B measurements before merge. Correctness
validation alone does not establish that the extra packing work around each
vector chunk improves end-to-end constructor throughput.

## References

- [Issue #1063](https://github.com/ohdearquant/lattice/issues/1063)
- [Issue #1062](https://github.com/ohdearquant/lattice/issues/1062)
- [ADR-058](ADR-058-cpu-perf-regression-ci.md)
