# ADR-014: Embedding Service

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-embed

## Context

Verbs like `recall`, `search`, and `suggest` need vector representations of
text for semantic similarity matching. The embedding service generates these
vectors using local transformer models via `ml/inference`.

## Decision

### EmbeddingService trait + NativeEmbeddingService

```rust
pub trait EmbeddingService: Send + Sync {
    async fn embed(&self, texts: &[String], model: EmbeddingModel) -> Result<Vec<Vec<f32>>>;
    async fn embed_one(&self, text: &str, model: EmbeddingModel) -> Result<Vec<f32>>;
}
```

`NativeEmbeddingService` calls `ml/inference` for local model execution.
`CachedEmbeddingService` wraps any `EmbeddingService` with an LRU cache
(sharded by text hash, configurable capacity).

### Compositional validation capability (2026-07-28)

`CachedEmbeddingService` validates caller text before any cache access. Delegating a disabled-cache
request or a miss through the public role method would make a built-in inner service repeat the
same caller-bound scan. The trait therefore has a doc-hidden `embed_with_role_prevalidated` hook
whose opaque `ValidatedTextBatch` argument has no public constructor.

The compatibility default calls the implementor's existing public `embed_with_role` method. This
preserves arbitrary out-of-tree role overrides, even though an unknown implementation may validate
again. `NativeEmbeddingService` consumes the proof directly while retaining its model check and
prepared-text memory backstop. `CachedEmbeddingService` also consumes it directly, so nested cache
wrappers validate the caller once. A cache miss is represented as borrowed views into the validated
batch for those built-in paths rather than a cloned `Vec<String>`.

### Supported models

| Model               | Dims | Use case                        |
| ------------------- | ---- | ------------------------------- |
| BgeSmallEnV15       | 384  | Default — fast, English-focused |
| MultilingualE5Small | 384  | CJK + multilingual              |
| Qwen3Embedding0_6B  | 1024 | High quality, larger            |

Model selection is config-driven via application configuration. The `EmbeddingModel`
enum lives in `lattice-embed` — it's a type, not ML code.

### Transport submodule — drift detection

```
embed/
├── src/
│   ├── service.rs      # EmbeddingService trait + impls
│   ├── cache.rs        # LRU cache (sharded, lock-free reads)
│   └── transport/      # Sinkhorn optimal transport
│       ├── sinkhorn.rs    # Balanced OT solver
│       ├── divergence.rs  # Sinkhorn divergence
│       ├── barycenter.rs  # Wasserstein barycenters
│       └── drift.rs       # Distribution drift detection
```

Transport answers: "are my embeddings drifting?" by computing Sinkhorn
divergence between current and reference embedding distributions. Used for:

- Detecting when a model update changes the embedding space
- Monitoring embedding quality degradation over time
- Triggering re-indexing when drift exceeds a threshold

Not yet wired into production — the algorithms are proven and tested,
awaiting integration into the embedding pipeline.

## Implementation status (2026-06-24)

The `EmbeddingService` trait, `NativeEmbeddingService`, and `CachedEmbeddingService` are
implemented at `crates/embed/src/service/` (native.rs, cached.rs). The transport/drift submodule
shown in §Decision as `embed/src/service/transport/` was not placed there. Optimal transport
and drift detection shipped as a standalone crate: `crates/transport/src/` (sinkhorn.rs,
divergence.rs, barycenter.rs, online_drift.rs, etc.). The embedding crate (`crates/embed/`) does
not import `crates/transport/`; wiring drift detection into the embedding pipeline remains
a future step.

## Consequences

- `ml/embed` is the only crate that imports `ml/inference`.
- Verbs and the DB layer call `embed.embed_one()` — they never touch inference directly.
- The cache prevents redundant model calls for repeated text.
- Built-in cache delegation preserves validation-before-lookup while avoiding a repeated
  caller-bound scan; unknown trait implementations retain their existing role-method behavior.
- Drift detection is available but opt-in — no cost when unused.
