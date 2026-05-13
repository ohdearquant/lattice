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

## Consequences

- `ml/embed` is the only crate that imports `ml/inference`.
- Verbs and the DB layer call `embed.embed_one()` — they never touch inference directly.
- The cache prevents redundant model calls for repeated text.
- Drift detection is available but opt-in — no cost when unused.
