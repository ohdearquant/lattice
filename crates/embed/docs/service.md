# Embedding services

`EmbeddingService` is the async boundary between an application and an embedding producer. It
defines how batches of text become batches of `Vec<f32>` while allowing callers to choose a
generic, query, or document/passage semantic role. Native builds provide two implementations:
`NativeEmbeddingService`, which owns a lazily loaded local model, and
`CachedEmbeddingService`, which wraps any service with a process-local LRU.

The trait is the stable integration surface. Constructors and cache-management APIs on the
native implementations are marked unstable and may change independently of the trait.

## Service topology

```text
                         +------------------------+
texts + model ---------->| EmbeddingService        |
                         | embed / query / passage |
                         +-----------+------------+
                                     |
                      optional       | native
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
  CachedEmbeddingService                      NativeEmbeddingService
  validate -> lookup -> fill                 validate -> lazy load -> encode
                 |                                       |
                 v                                       v
       EmbeddingCache (sharded LRU)          BERT batch or Qwen item encoding
```

Both implementations satisfy `Send + Sync`. The trait is declared with `async_trait`, so
callers can use a concrete service, an `Arc`, or a suitable trait object without deciding how
the producer performs its work internally.

## Trait contract

### Generic batches and the single-item helper

`embed(&[String], EmbeddingModel)` is the fundamental method. A successful result contains one
embedding for each input text in the same order. It represents the `Generic` role: no
role-specific instruction is prepended. Implementations are responsible for returning a typed
error rather than an incomplete batch.

`embed_one(&str, EmbeddingModel)` is a default helper that allocates a one-element string batch,
calls `embed`, and returns its only vector. If an implementation incorrectly returns no vectors,
the helper reports `EmbedError::Internal` instead of inventing an empty embedding.

Use `embed_one` for generic text only. Query and passage operations accept batches so the caller
can select their semantic role explicitly.

```rust,no_run
use lattice_embed::{EmbeddingModel, EmbeddingService, NativeEmbeddingService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service = NativeEmbeddingService::with_model(EmbeddingModel::MultilingualE5Small);
    let queries = vec!["what is a vector database?".to_owned()];
    let vectors = service
        .embed_query(&queries, EmbeddingModel::MultilingualE5Small)
        .await?;
    assert_eq!(vectors.len(), queries.len());
    Ok(())
}
```

`supports_model` allows a caller to ask whether a service can handle a model before submitting a
request, and `name` provides a static implementation label. `model_config` defaults to the
native-size configuration for a model. An implementation with a configured output dimension can
override it so cache keys use the actual vector space.

### Retrieval roles and prompts

`embed_query` and `embed_passage` are default trait methods. Each gets the corresponding prefix
from `EmbeddingModel`, prepends it to every supplied text when present, then delegates to
`embed`. They are not interchangeable with generic `embed`: a model trained for asymmetric
retrieval expects its query and document vectors to be placed correctly in the shared space.

| Model family    | `embed_query` prefix                                       | `embed_passage` prefix | Pooling on native BERT path |
| --------------- | ---------------------------------------------------------- | ---------------------- | --------------------------- |
| BGE v1.5        | `Represent this sentence for searching relevant passages:` | none                   | CLS                         |
| Multilingual E5 | `query:`                                                   | `passage:`             | masked mean                 |
| Qwen3-Embedding | retrieval instruction followed by `Query:`                 | none                   | not a BERT path             |
| MiniLM variants | none                                                       | none                   | masked mean                 |

The Qwen query instruction currently reads: `Instruct: Given a web search query, retrieve
relevant passages that answer the query\nQuery:`. The library's model API owns these values; use
the role methods rather than duplicating prompt strings in an application.

The caching wrapper overrides the two role methods instead of relying only on the trait defaults.
It applies the same prefix but records `Query` or `Passage` in the cache key. This prevents
`embed_query("text")`, `embed_passage("text")`, and `embed("text")` from sharing a cache entry
solely because their raw input happens to match.

## Request validation

The cache wrapper validates empty batches, batch size, and text length before any lookup. The
native service enforces those same limits when it is called and additionally verifies that the
requested model matches its configured model:

| Rule                                                                                                        | Failure                    |
| ----------------------------------------------------------------------------------------------------------- | -------------------------- |
| Batch must contain at least one text                                                                        | `EmbedError::InvalidInput` |
| Batch may contain at most `DEFAULT_MAX_BATCH_SIZE` (1,000) texts                                            | `EmbedError::InvalidInput` |
| Each text must be at most `MAX_TEXT_BYTES` (32,768) according to the implementation's `String::len()` check | `EmbedError::TextTooLong`  |
| Native request model must equal that service's configured model                                             | `EmbedError::InvalidInput` |

`String::len()` measures UTF-8 bytes, so the present implementation can reject fewer than
32,768 non-ASCII characters. Prefixes are applied before validation in role-specific calls, so
the prefix bytes count too. A caller that needs a character-based product limit should enforce
that separately before calling the service.

The wrapper performs its own three request-bound checks even for an all-hit request. It does
not call `inner.embed` merely to validate a cache hit, so an inner service's model-specific
checks are not part of the all-hit path. In particular, `NativeEmbeddingService` compares the
requested model with its configured model only when the cache is disabled or the wrapper has at
least one miss to delegate. A fully cached request therefore bypasses that native model check;
a mixed request reaches it only for its missing subset.

## NativeEmbeddingService

`NativeEmbeddingService` is available with the `native` feature, which is enabled by default. It
uses `lattice-inference` directly, with SIMD-capable local kernels and safetensors model loading;
there is no ONNX runtime, C++ FFI, or external embedding process.

### Construction and configuration

| Constructor           | Configuration                                                                     |
| --------------------- | --------------------------------------------------------------------------------- |
| `new` / `default`     | The crate's default embedding model at its native dimension                       |
| `with_model`          | One selected `EmbeddingModel` at its native dimension                             |
| `with_model_config`   | A model plus validated optional MRL output dimension                              |
| `with_model_from_env` | A selected model with `LATTICE_EMBED_DIM` parsed as its optional output dimension |

An absent or blank `LATTICE_EMBED_DIM` means native dimension. A nonnumeric environment value or
a dimension rejected by `ModelConfig::validate` becomes `EmbedError::InvalidInput`. Configuring
MRL does not make an arbitrary model dimension-adjustable; only models that support it can use a
non-native output dimension. See [model.md](model.md) for that capability and its index-space
implications.

One native service is intentionally single-model. `embed` first compares the requested model
with the service configuration and rejects a mismatch. Applications that serve several models
should construct separate services (and separate compatible vector-index namespaces) rather
than relying on an implicit concurrent model switch. `supports_model` reports this same
single-model condition.

### Lazy, cancellation-safe loading

Construction does not load model weights. The first `embed` or `ensure_loaded` request follows
this sequence:

1. Check the `OnceLock` for a completed load result.
2. When absent, clone the shared lock and start the synchronous loader with
   `tokio::task::spawn_blocking`.
3. The blocking worker runs `OnceLock::get_or_init`; concurrent callers wait for the same
   initializer instead of starting independent loads.
4. Read the stored success or initialization failure from the lock and proceed or return the
   corresponding `EmbedError`.

The use of `std::sync::OnceLock` is deliberate. A caller may drop its async future while waiting
for a download or load, but that does not cancel the blocking initializer or clear the result.
The model load continues to completion and the service will reuse the stored outcome for its
remaining lifetime. Conversely, a load failure is also stored in the lock; retrying the same
service returns that initialization failure rather than silently starting a fresh load.

`ensure_loaded` is the explicit preload hook. It performs the same lazy initialization as the
first embedding request without an encoding pass. For BERT-family variants, the loader delegates
to the inference crate's pretrained-model path. Qwen variants require a local model directory:
`LATTICE_QWEN_MODEL_DIR` takes precedence, otherwise the service looks below
`$HOME/.lattice/models` using the variant's Qwen directory name. Missing Qwen files produce
`EmbedError::ModelInitialization` with download guidance.

### Architecture-specific encode paths

The loader routes supported BGE, E5, and MiniLM variants to `BertModel`. It applies the selected
pooling configuration before sharing that model through an `Arc`; BGE uses CLS pooling while E5
and MiniLM use masked mean pooling.

Qwen3-Embedding variants load as `QwenModel`, validate any requested output dimension, and set
that dimension before use. The service calls BERT's `encode_batch` once for a BERT batch. Qwen
instead calls `encode` for each item in order so Qwen's model-local cache participates. Both
paths return embeddings in the validated input order.

The native `name` is currently the static string `"native-bert"` for all its supported local
models, including Qwen. It is an implementation label, not a reliable model-family identifier;
the configured `ModelConfig` is the source of that identity.

### Qwen persistent cache

`save_cache` and `cache_size` concern only the cache held by a loaded `QwenModel`:

- Before a successful model load, `save_cache` returns `0` and `cache_size` is `0`.
- BERT-family models do not use this persistent cache through the service, so both operations
  likewise report `0`.
- Qwen persistence normally uses a file named `embed_{model}_{dimension}d.bin` below
  `$HOME/.lattice/cache`; the cache-path helper falls back to `/tmp` when `HOME` is absent.
  Including model and active dimension prevents different output spaces from sharing a file.
- A Qwen load attempts to restore that cache. An integrity-check failure is logged and ignored;
  inference proceeds and a later save can regenerate the cache.

This is distinct from `CachedEmbeddingService`'s in-memory LRU. The two can be used together:
the outer LRU avoids service calls for repeated application requests, while Qwen can reuse
model-local work for calls that do reach the native service.

## CachedEmbeddingService

`CachedEmbeddingService<S>` wraps an `Arc<S>` where `S: EmbeddingService`. It owns an
`EmbeddingCache` with a chosen capacity, or the cache module's default capacity through
`with_default_cache`. `cache_stats` observes its LRU and `clear_cache` removes its entries; both
are management APIs rather than a promise of distributed or persistent caching.

### Lookup and fill algorithm

For every generic, query, or passage request, the wrapper does the following:

1. Keep the caller's text as-is and record the request's `EmbeddingRole` (`Generic` for a plain
   `embed` call). The wrapper does not apply the role instruction; the wrapped service does.
2. Validate only the wrapper-owned empty-batch, batch-size, and text-length bounds, against the
   caller's text, before lookup. Checking caller text keeps the published cap describing what the
   caller passed rather than what the instruction adds.
3. If capacity is zero, bypass hash computation and locks and delegate the caller text to the
   inner service through `embed_with_role`, which applies the instruction for the role.
4. Ask the inner service for the effective `ModelConfig`, then hash each caller text with the
   model identifier, its key revision, active dimension, and role tag.
5. Look up all keys in the sharded LRU. Hits are returned from cache storage as shared slices and
   copied into the caller-owned result vectors; misses are remembered with their original input
   positions.
6. If misses exist, send only those caller texts to the inner service through `embed_with_role`.
   It must return exactly one vector for each miss. A count mismatch becomes
   `EmbedError::InferenceFailed` rather than allowing a `zip` operation to drop positions silently.
7. Insert new vectors, fill the missing result positions, and return every vector in the order
   supplied by the caller.

The wrapper avoids mixing role-specific data by keying caller text together with the
`EmbeddingRole` tag, which keeps generic, query, and passage requests in distinct cache namespaces
as prompt policies evolve. Keying caller text is equivalent to keying prepared text because the
instruction is a function of the role and model configuration already in the key, so no two
distinct prepared strings can collide under one key.

The cache is a reuse optimization, not a single-flight coordinator. Concurrent cache misses for
the same text may each call the inner service before either request inserts its result. Callers
that need cross-request work de-duplication must provide it outside this wrapper.

## Error boundaries

All service methods use `crate::Result<T>`, an alias for `Result<T, EmbedError>`. The error enum
also serves model configuration and prepared SIMD APIs, so not every variant originates in a
native embedding call.

| Error                 | Meaning at this boundary                                                            | Typical caller action                                                            |
| --------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `ModelNotLoaded`      | A service requires initialization before use                                        | Initialize/preload the service or surface the configuration issue                |
| `WrongModelLoaded`    | The loaded model differs from the expected model during a concurrent switch         | Retry with backoff after coordinating the model selection                        |
| `ModelInitialization` | Model discovery, download/load, or blocking initialization failed                   | Inspect model configuration/files; construct a fresh service after correcting it |
| `InferenceFailed`     | The inference backend failed, including a cache-wrapper result-count violation      | Surface the backend failure; do not accept a partial batch                       |
| `TaskFailed`          | A background task was cancelled or panicked                                         | Treat the current call as failed and check service state before retrying         |
| `InvalidInput`        | Empty/oversized batch, wrong model, invalid configuration, or other invalid request | Correct the request without retrying unchanged input                             |
| `TextTooLong`         | A text exceeded the service's length check                                          | Chunk or shorten the prompted input                                              |
| `DimensionMismatch`   | An operation received vectors of different expected and actual dimensions           | Keep model/config/index namespaces consistent                                    |
| `UnsupportedModel`    | The selected service cannot provide that model                                      | Select a capable service or model                                                |
| `Internal`            | An invariant failed, such as a missing single-item result                           | Treat as a bug report; do not synthesize a vector                                |
| `TierMismatch`        | Prepared SIMD quantization data used a different tier than the operation expects    | Reprepare/query with matching quantization metadata                              |

Errors are descriptive but not a transaction protocol: if a batch fails after an inner service
has done work, a caller should assume no usable batch result was returned and decide whether its
own retry policy is safe. In particular, cache insertion happens only after the wrapper has
received the expected number of new vectors.

## Trait API details

`EmbeddingService` is the stable async contract. `embed` produces one vector per input in input
order and represents the `Generic` role. `embed_one` is a one-item convenience call and reports
an internal error if an implementation violates that cardinality contract rather than fabricating
an empty vector.

`embed_with_role` is the single role entry point: it validates the caller's text against the
published cap first, then prepends the model instruction, then delegates to `embed`. Validating
before preparation is deliberate. The cap describes what the caller may submit, so charging the
caller for the instruction bytes the service itself adds would reject text that is within the
documented limit. `embed_query` and `embed_passage` are thin wrappers that call `embed_with_role`
with the `Query` or `Passage` role, so all three role paths share one ordering decision instead of
repeating it.

That default cannot preserve the caller-text cap for an external implementation that enforces the
exact cap inside its own `embed`: the prepared string reaches that `embed` already lengthened.
Keeping `embed` the sole abstract method is a backward-compatibility choice on a published crate;
an implementor that caps text should size that guard for the prepared length or override
`embed_with_role` to reach its backend directly. Both in-crate implementors override it. The
default also cannot create role-separated cache entries by itself: that behavior comes from
`CachedEmbeddingService` supplying its role tag on the key. `EmbeddingRole::Generic` also has its
own tag, so all three role paths receive distinct cache keys even when their resulting text is
identical.

`model_config` defaults to the model's native dimension. A service that has selected an MRL
dimension overrides it so a caching wrapper hashes the actual output space, not merely the model
variant. `supports_model` is a capability query and `name` is a static implementation label;
neither replaces a caller's own model/index compatibility checks.

## NativeEmbeddingService implementation notes

The native service contains an `Arc<OnceLock<Result<LoadedModel, String>>>` together with one
`ModelConfig`. It delegates BERT-family and Qwen inference to `lattice-inference`, using local
SIMD-capable kernels and safetensors rather than an external runtime. The wrapped model types
already satisfy the service's `Send + Sync` requirements, so the wrapper adds no manual unsafe
thread-safety implementation.

The first preload or embedding request checks the lock and otherwise schedules
`OnceLock::get_or_init` on Tokio's blocking pool. The cloned lock remains owned by the blocking
worker if the awaiting future is dropped, so cancellation cannot reset an in-progress load.
Concurrent callers share the same initializer; both a successful load and a loading failure are
memoized for the service lifetime. `ensure_loaded` exercises that sequence without encoding text,
which makes it suitable for an explicit download/load warm-up.

The BERT path calls `encode_batch` once after selecting CLS pooling for BGE or masked mean pooling
for E5 and MiniLM. The Qwen path calls `encode` once per text, in input order, so Qwen's
model-local cache participates. The persistent Qwen cache described above is separate from this
wrapper's LRU.

## CachedEmbeddingService cache-hit behavior

The wrapper keeps the caller's text, performs its local request-bound checks against that text,
and then either bypasses a disabled cache or computes keys from the caller text, effective model
configuration, and role. It gathers every LRU hit before deciding whether to call the inner
service. A full hit returns immediately and never invokes the inner service.

Consequently, the wrapper cannot enforce an inner service's model selection, authorization, or
other service-specific validation on a full hit. With at least one miss it delegates only the
misses through `inner.embed_with_role`; the wrapped service applies the role instruction and
performs its configured-model check before loading or encoding them. The wrapper rejects a
mismatched number of returned vectors before insertion, which prevents a partial `zip` from losing
original result positions.
