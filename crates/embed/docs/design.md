# lattice-embed architecture

`lattice-embed` turns text into local vector embeddings and provides the supporting pieces
needed to use those vectors safely: model selection, role-aware caching, SIMD similarity
operations, and staged migration to a new embedding space. The native path is pure Rust and
uses `lattice-inference`; it does not depend on an external ML runtime.

The crate is mostly an unstable implementation layer intended to sit below a higher-level
consumer. One deliberately narrow exception is the 0.4.x ANN consumer contract: the public
SIMD squared-L2/L2 distance, dot-product, and cosine-similarity functions retain their slice
signatures and documented behavior for degenerate input. Implementations may use different
instruction sets, so callers must not treat near-equal SIMD results as bit-identical scalar
results or depend on an exact near-tie ordering.

## System map

```text
application text
       |
       v
EmbeddingService  -- query/passage prefix selection --> EmbeddingModel / ModelConfig
       |                                                    |
       | native builds                                      v
       +--> CachedEmbeddingService --> EmbeddingCache --> cache key identity
       |            |                                      (text + model revision + dim + role)
       |            v
       +------> NativeEmbeddingService --> lattice-inference --> normalized vectors
                                                            |
                                                            v
                                                simd vector comparison / ANN index

stored vectors <--> migration state machine <--> backfill routing and batch coordination
```

The components have intentionally separate responsibilities. The service produces vectors;
the cache avoids repeated production; SIMD compares vectors after they exist; and migration
and backfill coordinate a change from one vector space to another. Neither the cache nor a
service itself decides when an index may cut over to a different model.

| Area | Module | Responsibility | Detailed reference |
| --- | --- | --- | --- |
| Model identity and configuration | `model` | Supported variants, provenance, native and MRL output dimensions, model prompts, pooling selection | [model.md](model.md) |
| In-memory reuse | `cache` | Sharded LRU and cache-key construction | This document and [service.md](service.md) |
| Vector arithmetic | `simd` | Runtime/compile-time dispatch and exact public operation contracts | [simd.md](simd.md) |
| Embedding production | `service` | Async API, prompt application, native model lifecycle, and cache wrapper | [service.md](service.md) |
| Migration state | `migration` | Model-swap state machine and progress tracking | [migration.md](migration.md) |
| Migration execution policy | `backfill` | Request/query routing, dual writes, and backfill batches | [backfill.md](backfill.md) |
| Browser boundary | `wasm` | In-memory BERT bindings and wasm SIMD utility exports | [WebAssembly bindings](#webassembly-bindings) |
| Shared values | `types`, `error` | Embedding metadata and typed operational failures | [service.md#error-boundaries](service.md#error-boundaries) |

## Embedding flow and identity

An embedding is meaningful only in the space that produced it. In practice, that space is more
specific than an `EmbeddingModel` enum variant: it also includes the model's key revision and
the active output dimension. A configurable Qwen MRL output dimension changes the space even
when the base model is the same. Indexes and cache entries must therefore not mix vectors from
different `ModelConfig` values.

The normal native flow is:

1. Construct a `NativeEmbeddingService` for one `EmbeddingModel` or `ModelConfig`. The native
   service accepts requests only for that configured model.
2. Choose the semantic operation: `embed` for generic text, `embed_query` for a retrieval
   query, or `embed_passage` for stored document text. The latter two apply the selected
   model's instruction prefix before inference.
3. Optionally place a `CachedEmbeddingService` around the native service. It validates the
   request, returns any cached vectors, embeds only misses, and restores the input order.
4. The native service loads its model lazily, off the async runtime, then encodes the input.
   BERT-family models use batch encoding; Qwen currently encodes the missing inputs one at a
   time so its model-local cache can participate.
5. Store or compare the returned vectors only with vectors produced under the same model,
   active dimension, and retrieval role policy.

For cache identity, the wrapper hashes the already-prompted text with the model identity,
model key version, active dimension, and `EmbeddingRole`. Role is deliberately separate from
the prompted text: a query and a passage can otherwise be confused for models whose text is
unchanged on one side. `embed` retains the `Generic` role for compatibility with existing
generic cache entries. See [service.md](service.md#cachedembeddingservice) for the exact
lookup-and-fill sequence.

## Model and inference boundary

`EmbeddingModel` is the portable selection value. `ModelConfig` pairs it with an optional MRL
truncation dimension and validates that a model supports the requested output dimension. The
native service owns one such configuration for its lifetime; it rejects a request selecting a
different model rather than switching models under active callers.

The native implementation routes encoder families and decoder families differently:

- BGE, multilingual E5, and the MiniLM variants load as BERT-family encoders. BGE is configured
  for CLS pooling; E5 and MiniLM use masked mean pooling.
- Qwen3 embedding variants load from a model directory, receive the configured output
  dimension, and can load/save their model-local embedding cache.
- A variant that is not locally supported fails through `EmbedError` instead of selecting a
  remote service implicitly.

Instruction prompts are a service concern because they encode query/document semantics, while
pooling and output dimensions are inference configuration. The complete prompt and model table
is in [model.md](model.md); service-level behavior is in [service.md](service.md#retrieval-roles-and-prompts).

## Caching layers

The crate has two different caches that should not be conflated:

1. **`EmbeddingCache`** is the process-local, sharded LRU used by
   `CachedEmbeddingService`. Its capacity is measured in vectors; a capacity of zero disables
   it and causes the wrapper to call the inner service without hashing or locking. It is the
   layer that preserves an application's request order across a mix of hits and misses.
2. **The Qwen model cache** belongs to `QwenModel`, not to `EmbeddingCache`. When a Qwen model
   loads, the native service attempts to load a persistent cache whose file name includes the
   model identifier and current vector dimension. `save_cache` persists that cache only for a
   loaded Qwen model. A failed integrity check is logged and ignored so future inference can
   regenerate entries; it is not treated as a successful cache hit.

The distinction matters during MRL use and model migration. The LRU key includes the active
dimension, and Qwen persistence uses it in the file name, so 1,024-dimensional and native-size
vectors do not share cached values. Those values still need separate vector-index namespaces
when a migration is in progress.

## SIMD and index boundary

`simd` contains the computation kernels used to compare or normalize embedding vectors. It
selects AVX-512/AVX2/FMA on supported x86_64 hosts, NEON on aarch64, SIMD128 when it was
compiled into a wasm artifact, and scalar fallback elsewhere. The runtime dispatch configuration
is initialized once and exposed as `SimdConfig`.

The service layer does not require a particular SIMD capability: every supported path has a
scalar fallback. SIMD remains independent of model loading, cache policy, and migration state;
this keeps it usable by ANN consumers that already have vectors. Numeric rules such as
zero/NaN handling, quantization tiers, and the squared-L2 ordering invariant are part of the
operation contract and live in [simd.md](simd.md) and the public Rust API docs.

## Migration and backfill boundary

Changing an embedding model is a data migration, not a service hot swap. Existing vectors and
new vectors occupy different spaces, even where their dimensions happen to match. The crate
separates the state of that migration from the policy used to serve traffic:

- `MigrationController` represents one `MigrationPlan` and records lifecycle transitions and
  progress without deciding which model should serve any request.
- `BackfillCoordinator` wraps that state machine and derives document-write, existing-document,
  and query routes. It also controls backfill batch progression, dual writes, the target query
  threshold, and the rollback window.

The intended operational progression is to keep searching the legacy index, begin dual-writing
new documents where configured, backfill old documents into the target index, and move query
traffic only when the configured coverage condition is met. The target remains a separate index
until the application considers cutover complete. The state-machine details and routing tables
are in [migration.md](migration.md) and [backfill.md](backfill.md).

## WebAssembly bindings

The `wasm` module is compiled only when both the `wasm` feature and the `wasm32` target are
active. The dual gate is intentional: the binding dependencies live in the wasm32 target
dependency table, so enabling the feature for a native target must leave the module absent
rather than trying to resolve `wasm-bindgen` there.

`wasm32-unknown-unknown` does not offer the filesystem/memory-map path used by native model
loading. In particular, a `memmap2` wasm fallback may compile but memory-map calls fail at
runtime. The browser API consequently accepts raw `model.safetensors`, `config.json`, and
`tokenizer.json` bytes in `LatticeEmbedder::new`; the JavaScript host is responsible for fetching
and retaining those bytes. Loading, embedding, and error conversion run synchronously in memory.
There is no native download, disk cache, or `spawn_blocking` lifecycle in this path.

The browser embedder supports BERT-family models. It starts with mean pooling, which is suitable
for E5 and MiniLM; a BGE caller must invoke `useClsPooling` before embedding. Invalid UTF-8 in
the JSON inputs, model parsing failures, and unsupported BERT configuration become JavaScript
exceptions through `wasm-bindgen`.

The module also exports dot product, squared Euclidean distance, cosine similarity, and
in-place normalization. These call the Rust SIMD API rather than a separate JavaScript loop.
SIMD128 is a compile-time property on wasm, not a runtime CPU probe. The
`simdSimd128Dispatch` binding deliberately reports the same `SimdConfig` decision used by the
kernels, allowing callers to verify that a SIMD128 build is exercising the intended path rather
than silently measuring scalar fallback.
