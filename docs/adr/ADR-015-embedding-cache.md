# ADR-015: Sharded LRU Embedding Cache

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-embed

## Context

Embedding model inference is the dominant cost in the hot path: a single BGE-small
call costs roughly 2-10 ms on an M4 Pro, far more than any downstream lookup.
High-QPS services (`recall`, `search`, `suggest`) repeatedly embed short, repeated
texts (entity names, query terms, system prompts). A cache reduces this to a
refcount bump.

A naive `RwLock<LruCache>` creates a write-lock bottleneck: LRU's `get()` must
update access order, so it always acquires a write lock regardless of hit or miss.
Under 8+ concurrent async tasks, this single lock serializes all cache reads.

## Decision

The cache is implemented as 16 independent LRU shards behind a `parking_lot::RwLock`,
with Blake3-based shard routing and per-shard `AtomicU64` hit/miss counters.

### Key Design Choices

**16 shards, power-of-two count**

`NUM_SHARDS = 16` is a compile-time constant asserted to be a power of two (`const _: () = assert!(...)`).
Shard selection is a bitwise AND: `key[0] as usize & SHARD_MASK` (where `SHARD_MASK = 15`).
This eliminates a modulo instruction on the hot path. With 8-core M4 Pro hardware concurrency,
16 shards gives 2x oversubscription, keeping per-shard lock wait time negligible.

**Blake3 for cache keys**

The key is `blake3(text_bytes || "model_name:key_version:active_dim")`, producing a `[u8; 32]`.
Blake3 is chosen over FNV/xxHash because: (a) output is cryptographically uniform so the
first-byte shard selector distributes load evenly, and (b) collision resistance matters
when different models can produce semantically different embeddings for the same text.
The key includes the active MRL dimension (`model_config.dimensions()`) so a Qwen3-4B
config with `output_dim=512` and one with `output_dim=2560` produce distinct keys.

**Per-shard capacity via ceiling division**

`per_shard = ceil(capacity / NUM_SHARDS)`. For `capacity=4000`: 250 per shard.
For small capacities below `NUM_SHARDS` (e.g., `capacity=3`): each shard gets at least 1
entry rather than zero (which would panic `NonZeroUsize::new`).

**`Arc<[f32]>` storage**

Embeddings are stored as `Arc<[f32]>` (fat pointer to a slice), not `Vec<f32>`.
`get()` returns a clone of the Arc — a cheap refcount bump, not a data copy.
This is important for 1024-dim or 2560-dim Qwen vectors where a copy would be 4-10 KB.

**Zero-capacity disables caching without overhead**

`EmbeddingCache::new(0)` sets `enabled = false`. All `get`/`put`/`get_many`/`put_many`
operations return early with no locking, no hashing, no allocation. This makes disabling
the cache in test environments or for throughput benchmarks free.

**Per-shard `AtomicU64` stats**

Hit and miss counters are `AtomicU64` fields on each `CacheShard`, incremented with
`Ordering::Relaxed`. `stats()` aggregates them with a linear scan of 16 values.
This avoids cross-shard contention on the counter updates — there is no shared
"global counter" that every thread touches.

### Alternatives Considered

| Alternative                   | Pros                                        | Cons                                                                                 | Why Not                                                      |
| ----------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| Single `Arc<Mutex<LruCache>>` | Simple                                      | All reads serialized (LRU requires write lock on get)                                | Hot path bottleneck under 4+ concurrent embedders            |
| `dashmap`-based cache         | Lock-free reads for non-LRU gets            | DashMap uses segment locking similar to sharding; adds dep; no LRU eviction built in | Rolling our own sharding is ~60 LOC and has no dep           |
| `moka` crate (concurrent LRU) | Production-grade, TinyLFU policy, no unsafe | ~50k LOC transitive dependency; eviction policy different from pure LRU              | Dependency cost too high for a caching layer                 |
| Hash-map without LRU          | O(1) insert/lookup                          | No eviction — memory grows unbounded                                                 | Must bound memory usage; LRU eviction is the correct policy  |
| Shard by model                | Natural isolation per model                 | No load balance if only one model is active; adds model comparison on hot path       | Shard by key hash is always balanced regardless of model mix |

## Consequences

### Positive

- `get_many` on a fully-cached batch of N texts touches at most 16 shards (one per unique first-byte bucket), each under a write lock for < 1 µs.
- `DEFAULT_CACHE_CAPACITY = 4000` holds ~6 MB for 384-dim embeddings, fitting comfortably in L3 cache.
- `per_shard_stats()` exposes per-shard distribution skew for monitoring. The distribution test in the test suite verifies no shard exceeds 3x the average for 800 entries.
- Concurrent access test (8 threads × 100 operations) passes without data races under Miri.

### Negative

- Under pathological key distributions (all texts hashing to the same first byte), one shard takes all the load. In practice, Blake3 output is uniform enough that this is not a concern.
- The 16-shard fixed constant cannot be changed without rebuilding. Shard count is not configurable at runtime.

### Risks

- The `CacheKey = [u8; 32]` type is marked `Unstable` — callers must not persist keys across sessions. Key scheme changes (e.g., adding namespace prefix) would silently invalidate all stored keys without error.
- `EmbeddingCache` is not `Clone`. Multiple services sharing a cache must wrap it in `Arc<EmbeddingCache>`, which is the pattern used by `CachedEmbeddingService`.

## References

- [`crates/embed/src/cache.rs`](/Users/lion/projects/lattice/crates/embed/src/cache.rs) — full implementation
- [`crates/embed/src/service/cached.rs`](/Users/lion/projects/lattice/crates/embed/src/service/cached.rs) — `CachedEmbeddingService` consumer
- [`crates/embed/src/model.rs`](/Users/lion/projects/lattice/crates/embed/src/model.rs) — `ModelConfig::dimensions()` used for key construction
