# ADR-017: NativeEmbeddingService Orchestration

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-embed

## Context

The embedding service must:

1. Load a transformer model once per process (model weights are 100-400 MB; reloading is too slow).
2. Be cancellation-safe: if an async client disconnects during model loading (e.g., MCP timeout), the loading must not be abandoned and then restarted on the next request.
3. Support both BERT encoder and Qwen3 decoder architectures with a single code path.
4. Allow the output dimension to be configured via environment variable for MRL models.

The Tokio async runtime cannot block on I/O inside an async task. Model loading is CPU- and
I/O-bound (reading safetensors, matrix multiplications during warm-up) and must run on the
blocking thread pool.

## Decision

`NativeEmbeddingService` uses `std::sync::OnceLock<Result<LoadedModel, String>>` wrapped in
`Arc` to guarantee once-and-only-once model loading that is safe under async cancellation.

### Key Design Choices

**`OnceLock` over `tokio::sync::OnceCell` for cancellation safety**

`tokio::sync::OnceCell::get_or_try_init` resets its state when the calling future is
dropped (e.g., client disconnect, MCP timeout). If the model was mid-load and the future
was cancelled, the next request would restart the load from scratch — potentially causing
continuous load attempts under flaky connectivity.

`std::sync::OnceLock` runs its initializer to completion inside `spawn_blocking`. Even if
the outer async future is cancelled and dropped, the `spawn_blocking` task continues
running. The model is set in the `OnceLock` exactly once regardless of how many callers
cancelled while waiting. The `OnceLock` is in an `Arc` so the `spawn_blocking` closure
can own a reference to it and store the result after the original `NativeEmbeddingService`
reference may have been dropped.

**`LoadedModel` enum for architecture dispatch**

```rust
enum LoadedModel {
    Bert(Arc<BertModel>),
    Qwen(Arc<QwenModel>),
}
```

Both `BertModel` and `QwenModel` are wrapped in `Arc` so they can be shared across
concurrent `embed()` calls without cloning the model weights. The `encode_batch` dispatch
has one asymmetry: `BertModel` supports true parallel batching (`encode_batch(&[&str])`),
while `QwenModel` exposes `encode(&str)` per-item. The `LoadedModel::encode_batch` wrapper
calls the Qwen path in a sequential loop — this is because the decoder model has internal
KV-cache state managed per call.

**`LATTICE_EMBED_DIM` environment variable for MRL**

At construction time, `with_model_from_env()` reads `LATTICE_EMBED_DIM` and passes it as
`ModelConfig::try_new(model, Some(dim))`. Validation (min 32, max = native dimension,
MRL-capable model only) runs at construction — not at first inference. An invalid value
causes construction to fail, not a delayed runtime error.

**Request flow: `embed()` → `ensure_model()` → `encode_batch()`**

1. Caller invokes `embed(texts, model)`.
2. Validation: non-empty, batch size ≤ `DEFAULT_MAX_BATCH_SIZE`, each text ≤ `MAX_TEXT_CHARS`, model matches loaded model.
3. `ensure_model()` fast-path: if `OnceLock` is already set, return the reference immediately (no blocking, no contention).
4. `ensure_model()` slow path: `spawn_blocking` runs `load_model_sync(model_config)` which calls either `BertModel::from_pretrained(name)` or `load_qwen_model(config)`.
5. `BertModel::from_pretrained` resolves by name through the lattice-inference crate.
6. `load_qwen_model` resolves the directory via `qwen_model_dir()`, loads from safetensors, sets output dim via `model.set_output_dim()`, and attempts to warm-start from a persistent embedding cache at `~/.lattice/cache/embed_{model}_{dim}d.bin`.
7. On success, `LoadedModel` is stored in the `OnceLock`. On failure, the error string is stored — subsequent calls will re-surface the same error without retrying.
8. Back in the async task: `encode_batch` is called with text references, returning `Vec<Vec<f32>>`.

**Persistent embedding cache for Qwen3**

Qwen3 models have an internal embedding cache (`QwenModel::cache_*` API). This cache is
persisted to `~/.lattice/cache/embed_{model}_{dim}d.bin` on `save_cache()` and loaded on
model initialization. The path includes both the model slug and the active dimension so
different MRL configurations never share a cache file. This is separate from the in-process
`EmbeddingCache` used by `CachedEmbeddingService`.

**Model mismatch is a hard error**

`NativeEmbeddingService` is single-model: it loads exactly one model at construction time.
`embed()` rejects requests for a different model with `EmbedError::InvalidInput`. Callers
that need multiple models should instantiate one `NativeEmbeddingService` per model and
multiplex through a higher-level router.

### Alternatives Considered

| Alternative                         | Pros                                      | Cons                                                                               | Why Not                                                           |
| ----------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `tokio::sync::OnceCell`             | Native async; no `spawn_blocking`         | Resets on future cancellation — model reloads on client disconnect                 | Cancellation resets break the once-only guarantee                 |
| `Mutex<Option<LoadedModel>>`        | Simple                                    | Holds lock during model load (minutes on first call); blocks all other async tasks | Blocks entire async runtime; not acceptable                       |
| Lazy static / `std::sync::LazyLock` | Process-global, zero constructor overhead | One model per process globally; can't run tests with different models              | Tests and multi-tenant services need model isolation per instance |
| Load model at startup (not lazy)    | No first-call latency spike               | Slows process startup even when embedding is unused; complicates integration tests | Lazy loading is strictly better for startup time                  |
| Multiple models per service         | Fewer service instances to manage         | Requires model registry, routing by request; cache becomes more complex            | Model-per-service is simpler and sufficient for current usage     |

## Consequences

### Positive

- The `OnceLock + spawn_blocking` pattern means model loading is race-free: if 10 tasks call `embed()` simultaneously before the model is loaded, only one `spawn_blocking` task actually loads — the other 9 either wait in `get_or_init` or observe the already-set value on the fast path.
- `save_cache()` / `cache_size()` expose Qwen3's persistent embedding cache for diagnostic monitoring and graceful shutdown hooks.
- The service compiles and runs without any C++ FFI: lattice-inference is pure Rust with safetensors loading.

### Negative

- First-call latency is high (~1-10 seconds for Qwen3-4B). Callers that need guaranteed sub-second response time must warm the service before accepting traffic.
- Error from model loading is permanent within a process lifetime: once `OnceLock` stores `Err(...)`, subsequent calls always return that error without retry. Process restart is required to attempt a fresh load.
- `NativeEmbeddingService` does not implement `Clone` — only `Arc<NativeEmbeddingService>` can be shared.

### Risks

- The `LATTICE_EMBED_DIM` env var is read once at construction. Changes after construction have no effect. If the env var is set to an invalid value (non-numeric, out of range, or set for a non-MRL model), `with_model_from_env()` returns `Err` and the service fails to construct — there is no fallback to the native dimension.
- Sequential per-item encoding for Qwen3 (vs batch encoding for BERT) means that large batches to Qwen3 take O(N) calls to the model instead of O(1). For the current workload (individual semantic queries), this is acceptable. High-throughput bulk embedding with Qwen3 would require batching at the model level.

## References

- [`crates/embed/src/service/native.rs`](/Users/lion/projects/lattice/crates/embed/src/service/native.rs) — full implementation
- [`crates/embed/src/service/cached.rs`](/Users/lion/projects/lattice/crates/embed/src/service/cached.rs) — `CachedEmbeddingService` wrapper
- [`crates/embed/src/service/mod.rs`](/Users/lion/projects/lattice/crates/embed/src/service/mod.rs) — `EmbeddingService` trait definition
