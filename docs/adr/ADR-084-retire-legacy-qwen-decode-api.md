# ADR-084: Retire the Legacy Generic Qwen Decode API

**Status**: Accepted
**Date**: 2026-07-28
**Crate**: lattice-inference

## Context

Lattice carried two public text-generation stacks. The older stack was built on the Qwen
embedding model, had no repository caller, and duplicated its own decode loop, grammar handling,
cache policy, output types, and stop-token behavior. The Qwen3.5 stack is the maintained path used
by the binaries and serving surfaces.

The legacy API's deprecation annotation identified version 0.5.1, but that annotation and its
field-by-field migration guidance first shipped in 0.6.0. The 0.6.0 release notes deferred removal
to 0.7.0, the next pre-1.0 minor release available for the announced breaking change.

The legacy module also owned a feature-gated attention benchmark. Its wrapper called a private
kernel whose only production caller was the decode loop being retired, so preserving or moving the
benchmark would measure an algorithm no live dispatch path uses.

## Decision

Remove the legacy public module, its free generation function, its module-local configuration and
output types, and its private decode implementation in 0.7.0. Delete the associated attention
benchmark rather than transplanting its dead kernel.

`Qwen35Model::generate` and `Qwen35Model::generate_streaming` are the supported text-generation
entry points. Their `model::GenerateConfig` and `model::qwen35_config::GenerateOutput` types are
the canonical request and response contracts. `QwenModel::encode` remains supported and unchanged
for Qwen embedding consumers.

Keep the shared `sampling`, `grammar`, `kv_cache`, attention, and speculative-decoding modules.
Their surviving callers and tests remain independent of the removed decode loop.

### Alternatives Considered

| Alternative | Pros | Cons | Why Not |
| ----------- | ---- | ---- | ------- |
| Keep the deprecated stack indefinitely | No downstream source break | Maintains a second uncalled decode loop and invites fixes to drift between paths | The announced compatibility window has elapsed |
| Move the old attention helper to a benchmark-only module | Preserves historical Criterion names | Measures no production dispatch path and can create misleading performance evidence | Benchmarks must represent live behavior |
| Adapt the old function to delegate to Qwen3.5 | Keeps the symbol | The model, configuration, cache, and output types are not drop-in compatible | A wrapper would either change semantics silently or retain the duplicate types |

## Consequences

### Positive

- Generation fixes have one maintained CPU contract instead of two divergent loops.
- Public documentation and stop-token manifests describe only callable entry points.
- Benchmark inventory no longer includes a dead implementation.
- The Qwen embedding model remains independent of text-generation lifecycle decisions.

### Negative

- Downstream callers still using the deprecated symbols must migrate when adopting 0.7.0.
- Callers must handle the canonical differences documented in the 0.7.0 release notes,
  particularly prompt inclusion, per-request cache capacity, and EOS-specific stop reporting.

### Risks

- A downstream user may have ignored the deprecation warning; the release note remains the
  required migration source.
- A stale reference could reintroduce the removed module or its benchmark, so CI pins their
  absence and the canonical replacement's presence.

## References

- Issues #807, #809, and #1152
- `docs/releases/v0.6.0.md`
- `docs/releases/v0.7.0.md`
