# Embedding models and cache

`lattice-embed` separates the choice of an embedding model from the runtime
configuration of its output space. That distinction is important: changing a
model, its revision family, its retrieval role, or its active output dimension
changes the meaning of an embedding. Such vectors must not be mixed in one
index or cache entry.

This document covers the model registry, prompt and pooling wiring, output
dimension configuration, provenance, vector-space identifiers, and the
in-memory embedding cache.

## Model wiring

`EmbeddingModel` is the stable, non-exhaustive registry of supported model
variants. It is serialised with snake-case variant names and accepts the
corresponding PascalCase names as Serde aliases. Its default is
`BgeSmallEnV15`.

The registry provides the facts that the rest of the embedding stack needs:

- the provider model identifier used to locate the model;
- native output width and conservative input limit;
- whether inference runs locally or through a remote API;
- the query and document prefixes required for retrieval;
- the BERT pooling mode when native BERT-family inference is enabled; and
- the embedding-key revision used to distinguish incompatible model families.

The usual relationship is:

```text
EmbeddingModel
    │ model-specific limits, prompts, pooling, key version
    ▼
ModelConfig
    │ active output dimension
    ├──────────────► embedding invocation
    └──────────────► cache key and vector-index namespace
```

`EmbeddingModel::dimensions()` always returns the native width. It does not
account for an optional Matryoshka Representation Learning (MRL) truncation;
use `ModelConfig::dimensions()` when allocating output buffers, defining an
index, or constructing an embedding cache key.

### Supported variants

| Variant | Model identifier | Native dimensions | Conservative maximum input tokens | Execution | Native BERT pooling |
| --- | --- | ---: | ---: | --- | --- |
| `BgeSmallEnV15` | `BAAI/bge-small-en-v1.5` | 384 | 512 | Local | CLS |
| `BgeBaseEnV15` | `BAAI/bge-base-en-v1.5` | 768 | 512 | Local | CLS |
| `BgeLargeEnV15` | `BAAI/bge-large-en-v1.5` | 1,024 | 512 | Local | CLS |
| `MultilingualE5Small` | `intfloat/multilingual-e5-small` | 384 | 512 | Local | Mean |
| `MultilingualE5Base` | `intfloat/multilingual-e5-base` | 768 | 512 | Local | Mean |
| `AllMiniLmL6V2` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | Local | Mean |
| `ParaphraseMultilingualMiniLmL12V2` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 128 | Local | Mean |
| `Qwen3Embedding0_6B` | `Qwen/Qwen3-Embedding-0.6B` | 1,024 | 8,192 | Local | Not a BERT path |
| `Qwen3Embedding4B` | `Qwen/Qwen3-Embedding-4B` | 2,560 | 8,192 | Local | Not a BERT path |
| `TextEmbedding3Small` | `text-embedding-3-small` | 1,536 | 8,191 | Remote | Not a BERT path |

The limits are intended for chunking and truncation. They leave room for
special tokens. In particular, Qwen3-Embedding is capped at 8,192 tokens for
practical use even though the underlying model supports a longer context.

`is_local()` is true for every variant except `TextEmbedding3Small`;
`is_remote()` is true only for `TextEmbedding3Small`. Native BERT pooling is
available only with the `native` feature. BGE v1.5 uses the first-token (CLS)
representation, whereas E5 and MiniLM variants use masked mean pooling. Qwen3
and the remote model follow their own embedding paths and therefore return no
BERT pooling choice.

### Retrieval prompting is part of the embedding space

Asymmetric retrieval models are trained to receive different input forms for
queries and documents. Apply the prefix returned by `query_instruction()` or
`document_instruction()` immediately before embedding the original text. The
prefix is not display-only metadata: omitting it changes retrieval quality and
creates vectors that are not comparable to correctly prepared vectors.

| Family | Query input | Document input | Why it matters |
| --- | --- | --- | --- |
| E5 | `query: {query}` | `passage: {document}` | E5 was trained with these asymmetric prefixes. |
| BGE v1.5 | `Represent this sentence for searching relevant passages: {query}` | Raw document text | The query instruction is required; BGE passages remain unprefixed. |
| Qwen3-Embedding | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}` | Raw document text | The task instruction aligns query embeddings with passages. |
| MiniLM variants | Raw query text | Raw document text | These models use symmetric raw-text inputs. |
| Remote text-embedding model | Raw query text | Raw document text | The registry supplies no instruction prefix. |

For example, a retrieval caller can make the role explicit before asking an
embedding service to embed the text:

```rust
let model = EmbeddingModel::MultilingualE5Small;
let query = format!("{}{}", model.query_instruction().unwrap_or(""), "rust lru cache");
let passage = format!(
    "{}{}",
    model.document_instruction().unwrap_or(""),
    "An LRU cache evicts the least recently used entry."
);
```

Do not use a query vector and a document vector as if their roles were
interchangeable for an asymmetric model. The cache also records the embedding
role in its key so that identical raw text embedded through query, passage, or
generic entry points cannot be reused across roles.

### Parsing and naming

`Display` produces a lower-case canonical model name, such as
`bge-small-en-v1.5` or `qwen3-embedding-4b`. `FromStr` accepts a case-insensitive
input after trimming whitespace and converting underscores to hyphens. It also
accepts common short names and selected full provider identifiers. For example,
`small`, `bge-small`, and `BAAI/bge-small-en-v1.5` select BGE small; `e5-base`
selects multilingual E5 base; `qwen3` selects the 0.6B Qwen3 variant; and
`openai` selects `TextEmbedding3Small`.

Parsing is deliberately convenient for configuration and command-line input,
but persisted model identity should use the registry value and its key version,
not a user-provided alias.

## Runtime output dimensions

`ModelConfig` pairs an `EmbeddingModel` with an optional output dimension. With
`output_dim: None`, its active dimension is the model's native width. An
explicit value requests MRL truncation and is valid only for the two
Qwen3-Embedding variants.

`ModelConfig::try_new(model, output_dim)` validates the requested space:

1. A model without MRL support rejects any explicit output dimension.
2. The requested dimension must be at least `MIN_MRL_OUTPUT_DIM` (32).
3. It must not exceed the model's native dimension.

`ModelConfig::new(model)` creates the no-truncation configuration, and
`validate()` is available when a configuration was deserialised or otherwise
assembled before use. The fields are public, so callers that construct a
configuration directly should validate it before selecting an output space.

Different active dimensions are different embedding spaces, even when they
come from the same Qwen model. Keep them in separate vector-index namespaces.
The embedding cache preserves this boundary by including the active dimension
in every key.

## Load provenance

`ModelProvenance` records which registry variant was loaded, the source
identifier, the loading time, and a convenience RFC 3339 timestamp. Its `hash`
is a 64-character BLAKE3 hexadecimal digest of:

```text
{model_id}:{loaded_at_iso}:{model_debug_representation}
```

This gives each load event a lightweight metadata identifier. It is not a hash
of model weights and must not be treated as a model-integrity verification.
Weight verification belongs to the inference layer's checksum facilities.
`dimensions()` forwards to the loaded model's native dimensions, and
`matches_model()` compares the recorded variant with an expected one.

## Embedding-space identity

The types in `types.rs` describe a vector space independently of the
in-memory cache. They are appropriate for selecting vector-store collections,
routing migrations, and producing deterministic identifiers for a complete
embedding-space description.

### Component encodings

The enum discriminants are part of `EmbeddingKey::canonical_bytes()`:

| Descriptor | Variant | Byte | Additional meaning |
| --- | --- | ---: | --- |
| `DistanceMetric` | `Cosine` | 1 | Cosine similarity / distance convention |
|  | `Dot` | 2 | Inner product |
|  | `L2` | 3 | Euclidean distance |
| `VectorDType` | `F32` | 1 | Four bytes per element |
|  | `F16` | 2 | Two bytes per element |
|  | `I8` | 3 | One byte per element |
| `VectorNorm` | `None` | 0 | No normalisation applied |
|  | `Unit` | 1 | L2 norm equals one |

`DistanceMetric::from_byte()` returns `None` for an unknown discriminant.
`VectorDType::size_bytes()` reports the storage width. All three types use
snake-case Serde representation and are non-exhaustive, so consumers should
avoid assuming that their current variants are exhaustive.

### Canonical `EmbeddingKey` format

An `EmbeddingKey` contains the provider/model name, its revision, dimensions,
distance metric, storage data type, and normalisation state. Its
`canonical_bytes()` method emits this exact deterministic sequence:

```text
model UTF-8 byte length       4-byte unsigned big-endian integer
model                         UTF-8 bytes
revision UTF-8 byte length    4-byte unsigned big-endian integer
revision                      UTF-8 bytes
dims                          4-byte unsigned big-endian integer
metric                        1 byte
dtype                         1 byte
norm                          1 byte
```

Length prefixes make the variable-length strings unambiguous. The resulting
bytes are suitable as the input to a deterministic hash for deduplication.
They should not be confused with `CacheKey`: the latter is a 32-byte in-memory
BLAKE3 digest with a cache-specific construction described below.

## Embedding cache

`EmbeddingCache` is an in-memory, thread-safe LRU cache for already computed
embeddings. It avoids repeating inference for equivalent inputs while keeping
the returned vector cheap to share: stored `Vec<f32>` values become
`Arc<[f32]>`, and a cache hit clones only the reference-counted handle rather
than the vector contents.

The cache is an internal, unstable mechanism. Its key scheme, shard count,
return type, and metric shapes may change. Do not persist `CacheKey` values or
assume they remain valid between process versions or sessions.

### Key construction

`CacheKey` is a 32-byte array: the BLAKE3 digest of the following byte stream:

```text
UTF-8 bytes of text
UTF-8 bytes of "{model_display}:{model_key_version}:{active_dimensions}:{role_cache_tag}"
```

`model_display` is the canonical `Display` name, `model_key_version` comes
from `EmbeddingModel::key_version()`, `active_dimensions` comes from
`ModelConfig::dimensions()`, and `role_cache_tag` is supplied by
`EmbeddingRole`. The role is included even for identical raw text so that a
query embedding cannot satisfy a passage or generic embedding lookup.

The model version values currently divide the registry into these families:

| Key version | Models |
| --- | --- |
| `v1.5` | BGE and multilingual E5 variants |
| `v2` | Both MiniLM variants |
| `v3` | Both Qwen3-Embedding variants and `TextEmbedding3Small` |

Including the model, version, active dimensions, and role prevents the cache
from crossing known embedding-space boundaries. `compute_key()` is public and
always performs this hash when called; disabling the cache bypasses lookup and
storage, not explicit key construction by its caller.

### Sharding and locking

The cache has 16 fixed shards. The shard is selected from the first byte of a
BLAKE3 key:

```text
shard = key[0] & 15
```

BLAKE3 output is uniformly distributed for this purpose, so the mask spreads
keys across the shards. The shard count must remain a power of two because the
implementation uses a bit mask rather than a general modulo operation.

Each shard owns an `LruCache` behind its own `RwLock` plus its own atomic
hit/miss counters. A lookup takes the write lock, not a read lock, because a
hit changes LRU recency. This is why sharding is important: independent keys
usually contend only with work that lands in the same shard rather than with
all cache traffic.

Counters use relaxed atomics on the hot path and are summed when statistics
are requested. Under concurrent traffic, aggregate statistics are monitoring
data rather than a single globally synchronised snapshot.

### Capacity and eviction

`EmbeddingCache::new(capacity)` accepts a requested total entry capacity. The
default is 4,000 entries, approximately 6 MB for 384-dimensional `f32`
embeddings before allocator and cache overhead.

For a nonzero requested capacity `C`, every shard receives:

```text
per_shard_capacity = ceil(C / 16), with a minimum of 1
```

This has two consequences worth accounting for when setting a memory budget:

1. Each shard evicts independently. A full shard may evict an old entry while
   another shard still has unused slots.
2. The physical maximum is `16 * ceil(C / 16)`, which can exceed the requested
   capacity. For example, a capacity of 10 creates 16 shards with one slot
   each, so up to 16 entries can be retained. `CacheStats::capacity` still
   reports the requested value (10), not the rounded physical maximum.

Insertion into a full shard evicts that shard's least-recently-used entry.
Successful `get()` calls refresh recency, so a recently accessed entry survives
an insertion ahead of an older entry in the same shard. There is no global LRU
order across shards.

Passing a capacity of zero disables the cache. `get()` returns `None`; `put()`,
`put_many()`, and `clear()` do nothing; and `get_many()` returns `None` for
each requested key. These paths avoid shard locking and do not change the
per-shard counters. The object still contains dummy shard allocations, but no
embedding is stored.

### Operations and lifecycle

- `get(key)` returns `Option<Arc<[f32]>>`, records one hit or miss when enabled,
  and refreshes recency on a hit.
- `put(key, embedding)` converts its vector to `Arc<[f32]>` and inserts it into
  the selected shard.
- `get_many(keys)` preserves input order and performs the same lookup semantics
  for every key. Each hit shares its vector without copying it.
- `put_many(entries)` applies the same storage rule to every `(key, vector)`
  pair.
- `stats()` sums all shard lengths and counters. With a disabled cache, it
  reports a size and capacity of zero.
- `per_shard_stats()` exposes the size, hits, and misses for each of the 16
  shards, which is useful when diagnosing uneven key distribution.
- `clear()` drops cached entries from every shard. It does not reset hit or miss
  counters, so cache lifetime metrics survive a manual eviction.

Both `CacheStats` and `ShardStats` calculate `hit_rate` as `hits / (hits +
misses)`. When no lookups have occurred, the result is `0.0` rather than an
undefined value.

### Choosing a safe cache boundary

Use one cache only for work that shares the same semantic configuration. In
particular, keep model changes and output-dimension changes isolated, and
always distinguish retrieval roles. `compute_key()` encodes those dimensions
for the built-in cache, but application-level collection or index naming must
enforce the same boundary for persisted vectors.

The cache stores only embedding values and never owns the model or its weights.
Clearing it releases the entries, not the model that produced them. A process
restart creates an empty cache; cache keys and contents are not an on-disk
format.

## EmbeddingModel source behavior

`EmbeddingModel` is the source of the model facts consumed by services: native dimensions,
conservative input-token limits, local-versus-remote execution, prompt policy, BERT pooling, and
the cache-key revision. `native_dimensions()` and `dimensions()` both report the registry's full
width; a service-specific active dimension belongs to `ModelConfig` instead. This distinction
keeps callers from allocating or indexing for a Qwen MRL space as though it were the full model
width.

`max_input_tokens()` is a conservative chunking limit rather than a tokenizer operation. It
leaves room for special tokens and currently reports 512 for BGE and E5, 256 for all-MiniLM,
128 for paraphrase multilingual MiniLM, 8,192 for Qwen3-Embedding, and 8,191 for the remote
OpenAI variant.

The query and document instruction accessors return the literal prefix that must be prepended to
the original text, or `None` for raw text. E5 uses paired `query:` and `passage:` prefixes, each
including a trailing space; BGE and Qwen use only a query instruction; MiniLM and the remote model
use neither. Prompted and unprompted text occupy different retrieval semantics, which is why
callers should select a role instead of copying these strings ad hoc.

With the `native` feature, BGE selects CLS pooling and E5 and MiniLM select masked mean pooling.
Qwen and the remote model return no BERT pooling choice because their inference paths own that
operation. `key_version()` groups BGE/E5 as `v1.5`, MiniLM as `v2`, and Qwen/remote as `v3` for
cache identity; it is not a substitute for the provider model identifier.

`Display` emits the canonical lowercase model name. `FromStr` lowercases, trims, converts
underscores to hyphens, removes a BAAI prefix, and accepts selected short aliases and provider
identifiers. That convenience is suitable for configuration input, whereas persisted identity
should use the selected registry variant and cache-key revision.

## ModelConfig source behavior

`ModelConfig` pairs an `EmbeddingModel` with an optional output dimension. `try_new` and
`validate` allow an explicit dimension only for Qwen3-Embedding, require at least 32 dimensions,
and reject a value above the model's native width. `new` leaves truncation unset and
`dimensions()` then returns the native width.

Its fields are public for serialization and configuration, so directly constructed values should
be validated before use. A changed active dimension is a changed embedding space: it must use a
separate vector-index namespace and cache key, even when the underlying Qwen model is unchanged.

## ModelProvenance source behavior

`ModelProvenance::new` records the selected model, caller-provided source identifier, current
load time, and an RFC 3339 rendering of that time. Its 64-character BLAKE3 hexadecimal `hash`
is calculated from `{model_id}:{loaded_at_iso}:{model_debug_representation}`. It identifies a
metadata load event, not the contents or integrity of model weights; weight verification belongs
to the inference layer's checksum facilities. `dimensions()` reports the model's native width and
`matches_model()` compares the recorded variant with an expected one.
