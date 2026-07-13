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

## Vector utility facade

The stable `utils` functions are deliberately thin forwarding wrappers around `simd`. They keep
the 0.4.x slice-based ANN contract independent from a particular instruction set: a runtime
dispatcher selects an available native SIMD kernel and every operation has a scalar fallback.
Implementations can therefore differ in floating-point accumulation order. Consumers may use the
functions for ranking, but must not rely on bit-identical scalar results to break near ties.

The wrappers preserve the underlying invalid-input sentinels. Cosine similarity returns `0.0` for
empty or unequal-length inputs and for a zero norm; its normal result is the familiar
`dot(a, b) / (|a| * |b|)`. Dot product also returns `0.0` for unequal lengths and is the cheaper
choice once embeddings have been L2-normalized, because it then equals cosine similarity without
recomputing norms. Euclidean distance returns `f32::MAX` for unequal lengths. Normalization is
in-place and leaves a zero- or NaN-norm vector unchanged rather than introducing an invalid scale.

The batch variants retain one output per input pair and preserve pair order. `batch_dot_product`
can select a four-candidate kernel when four adjacent pairs share the same query; other pairs use
the single-pair kernel. Each pair still follows the scalar API's mismatch sentinel, so batching
does not relax the input contract.

The historical benchmark snapshot that accompanied the facade illustrates why this boundary
exists; actual timing depends on the host ISA and vector shape.

| Dimension | Scalar | SIMD |
| --- | --- | --- |
| 384 | ~650ns | ~90ns |
| 768 | ~1300ns | ~180ns |
| 1024 | ~1700ns | ~240ns |

```rust
use lattice_embed::utils::{cosine_similarity, dot_product, euclidean_distance, normalize};

assert!((cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]) - 1.0).abs() < 0.0001);
assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 0.0001);
assert!((dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 0.0001);
assert!((euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 0.0001);

let mut vector = [3.0, 4.0];
normalize(&mut vector);
assert!((vector.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() < 0.0001);
```

## Vector-space identity wire format

`EmbeddingKey` names a vector space rather than merely a model. Compatibility requires all six
components to agree: provider/model name, revision, dimensionality, distance metric, element
type, and normalization state. This is suitable for choosing a vector-store collection,
deduplicating metadata, and identifying an embedding space during migration. Matching dimensions
alone are insufficient because a different revision, metric, representation, or normalization can
make two vectors incomparable.

`canonical_bytes` serializes the identity in this exact order:

1. model as a four-byte big-endian byte length followed by its UTF-8 bytes;
2. revision in the same length-prefixed form;
3. `dims` as a four-byte big-endian integer;
4. one byte each for `metric`, `dtype`, and `norm`.

The length prefixes make adjacent variable-length fields unambiguous. The enum wire values are
`DistanceMetric::{Cosine, Dot, L2}` = `1`, `2`, `3`; `VectorDType::{F32, F16, I8}` = `1`, `2`,
`3`; and `VectorNorm::{None, Unit}` = `0`, `1`. Unknown distance-metric bytes are rejected by
`DistanceMetric::from_byte`. Consumers that hash an `EmbeddingKey` should hash these canonical
bytes, not a display string or a serializer-specific representation.

## EmbeddingCache

`EmbeddingCache` is a process-local LRU for already computed vectors. It does not choose a model,
prepare prompts, or migrate index contents; its job is only to reuse the vector that belongs to a
specific request identity. Values are stored as `Arc<[f32]>`, so a hit returns a cheap reference
count increment rather than copying a vector. Batch lookup returns one optional value per supplied
key in the same order.

The cache computes a session-local 32-byte BLAKE3 key from the raw text followed by the formatted
model identity `model_name:key_version:active_dimension:role_tag`. Model key version and active
dimension prevent revision and MRL truncation collisions. The role tag isolates query, passage,
and backwards-compatible generic calls even if their raw text is identical. The key scheme is an
internal implementation detail and must not be stored for use across process versions or sessions.

There are 16 independent LRU shards. The first BLAKE3 output byte selects a shard with
`key[0] & 15`; this mask is valid only because the shard count is a power of two, which is checked
at compile time. Each shard owns an `RwLock<LruCache<...>>`; a lookup takes the write lock because
an LRU hit promotes its entry. Hit and miss counters are shard-local relaxed atomics and aggregate
into `CacheStats` or `ShardStats` without a global cache lock. Sharding reduces unrelated write
contention, while allowing eviction and recency to remain local to a shard.

For a nonzero requested capacity, every shard receives `ceil(capacity / 16)` entries. This makes
the actual aggregate storage capacity at least the requested number and can round it up when the
request is not divisible by 16. An insertion evicts the least-recently-used entry only from the
selected shard. A capacity of zero disables `get`, `put`, batch operations, and `clear` without
locking; the cache still permits callers to compute keys. Clearing removes values but deliberately
does not reset the diagnostic hit/miss counters.

```rust
use lattice_embed::{EmbeddingCache, EmbeddingModel, ModelConfig};
use lattice_embed::service::EmbeddingRole;

let cache = EmbeddingCache::new(1000);
let key = cache.compute_key(
    "Hello, world!",
    ModelConfig::new(EmbeddingModel::BgeSmallEnV15),
    EmbeddingRole::Generic,
);
assert!(cache.get(&key).is_none());

let embedding = vec![0.1, 0.2, 0.3];
cache.put(key, embedding.clone());
assert_eq!(&*cache.get(&key).unwrap(), &embedding);
```

## Prepared-dispatch errors

Prepared SIMD query paths bind a query to a quantization tier before comparing it with stored
quantized data. A prepared operation can only run when the query, stored data, and any
tier-specific batch entry point agree on that tier. A mismatch is returned as
`EmbedError::TierMismatch` instead of attempting a lossy reinterpretation or panicking. Its `op`,
`expected`, and `actual` fields identify the failing operation and both tiers so callers can
re-prepare the query or select the matching index partition. This is an operational input error;
the caller must not treat a partial or stale result as a valid similarity score.

## WebAssembly API details

The browser entry point is intentionally in-memory and synchronous. `LatticeEmbedder::new` takes
the raw `model.safetensors` bytes plus UTF-8 `config.json` and `tokenizer.json` bytes, then creates
a supported BERT-family model. Invalid JSON text, parse failures, and unsupported configurations
are converted to JavaScript exceptions by `wasm-bindgen`. `initPanicHook` is separate from normal
error conversion: it is optional and idempotent, and changes an otherwise opaque wasm panic trap
into a diagnostic sent to `console.error` rather than the usual `"unreachable executed"` message.
It is safe to omit; call it before constructing the embedder when browser panic diagnostics are
wanted.

Pooling is part of a model-family contract. The embedder starts with attention-mask-weighted mean
pooling for E5 and MiniLM. A BGE v1.5 caller must call `useClsPooling` immediately after
construction, before the first `embed`, to use the hidden state at position zero instead. `embed`
returns the model's L2-normalized vector and `dimensions` reports the loaded model's output width.

The four `simd*` exports are separate from `LatticeEmbedder`: they expose the stable ANN vector
contract directly to JavaScript so a wasm consumer uses Rust's scalar-or-SIMD implementation
instead of a hand-rolled JavaScript loop. The artifact selects SIMD128 only when it was compiled
with `-C target-feature=+simd128`; this is not a browser runtime capability probe.
`simdSimd128Dispatch` reads the exact `SimdConfig` value read by the vector kernels rather than
reconstructing the condition in the binding. A two-build parity harness can therefore distinguish
a true SIMD128 artifact from a stale or misconfigured build that silently used the scalar path.
The same exports are the A/B measurement entry points in `scripts/bench_wasm_simd.mjs`; the wasm
parity test is `crates/embed/tests/wasm/simd128_parity_wasm.mjs`.
