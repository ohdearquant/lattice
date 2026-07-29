# ADR-084: Fallible Validation for Unstable Inference Helpers

**Status**: Accepted
**Date**: 2026-07-28
**Crate**: lattice-inference

## Context

Two public helpers in the unstable inference surface could not report invalid caller input
without panicking or returning unusable numeric state:

- `GrammarEngine::mask_logits` asserted when the logits slice was shorter than the engine
  vocabulary.
- `pack_ternary` had an infallible return type even though a non-finite weight makes its row
  scale non-finite.

Both conditions can originate at model or tokenizer integration boundaries. Treating them as
process invariants makes ordinary input disagreement fatal and prevents callers from attaching
the failure to the request or checkpoint that caused it.

## Decision

The helpers return typed errors:

- `GrammarEngine::mask_logits` and its public simulation sibling return
  `Result<(), GrammarError>`. Generation entry points propagate that result through
  `InferenceError::InvalidInput`.
- `pack_ternary` returns `Result<(Vec<u8>, Vec<f32>), InferenceError>` and rejects non-finite
  weights and invalid matrix geometry before packing.

The success path for grammar masking retains the existing length comparison. Error allocation
and formatting occur only when validation fails.

### Alternatives Considered

| Alternative                                                        | Pros                 | Cons                                                                   | Why Not                                              |
| ------------------------------------------------------------------ | -------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------- |
| Keep panics as programmer errors                                   | No signature changes | A model/tokenizer mismatch terminates the process                      | These are caller-visible integration boundaries      |
| Add new `try_*` helpers and retain the infallible methods          | Source compatible    | Leaves the unsafe entry points available and easy to call accidentally | Validation must be enforced at the existing boundary |
| Return empty packed buffers or mask every token on malformed input | Infallible API       | Converts invalid input into ambiguous downstream behavior              | Callers need an explicit, attributable failure       |

## Consequences

### Positive

- Malformed grammar and BitNet inputs fail through typed error channels.
- Generation callers can preserve service availability and classify the request failure.
- Non-finite weight rows cannot seed downstream kernels with non-finite scales.

### Negative

- Direct callers of these unstable helpers must handle a `Result`.
- Grammar generation call sites carry an additional fallible return edge.

### Risks

- A future direct grammar-mask caller could discard the `Result`; linting and
  `Result`'s `must_use` annotation make that visible.
- Lower-level masking primitives still rely on the validated `GrammarEngine` entry point and
  must not be exposed as request boundaries without equivalent validation.

## References

- [Issue #1085](https://github.com/ohdearquant/lattice/issues/1085)
- [ADR-046: Structured Output](ADR-046-structured-output.md)
