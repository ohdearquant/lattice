# Design: lattice-embed

## Current Design

**Multi-model embedding service.** 8 model variants in `EmbeddingModel` enum (`model.rs:102-135`), `#[non_exhaustive]` for future extension:

| Variant                        | Architecture  | Native Dims | Max Tokens | MRL | Notes                                   |
| ------------------------------ | ------------- | ----------- | ---------- | --- | --------------------------------------- |
| `BgeSmallEnV15` (enum default) | BERT encoder  | 384         | 512        | no  | English-only                            |
| `BgeBaseEnV15`                 | BERT encoder  | 768         | 512        | no  | English-only                            |
| `BgeLargeEnV15`                | BERT encoder  | 1024        | 512        | no  | English-only                            |
| `MultilingualE5Small`          | BERT encoder  | 384         | 512        | no  | Recommended production default          |
| `MultilingualE5Base`           | BERT encoder  | 768         | 512        | no  | Multilingual                            |
| `Qwen3Embedding0_6B`           | Qwen3 decoder | 1024        | 8192       | yes | Last-token pooling, instruction prefix  |
| `Qwen3Embedding4B`             | Qwen3 decoder | 2560        | 8192       | yes | Sharded safetensors, instruction prefix |
| `TextEmbedding3Small`          | OpenAI API    | 1536        | 8191       | no  | Remote only, no local weights           |

`ModelConfig` (`model.rs:381`) pairs a model variant with optional `output_dim` for MRL truncation. `validate()` enforces bounds. `to_embedding_key()` produces a unique key per (model, dimension) pair -- different dimensions = different embedding spaces = separate HNSW indexes and cache files. Production default is `multilingual-e5-small` via application config, not the enum default (`BgeSmallEnV15`). The enum default serves code paths without application config (tests, standalone usage). Qwen3 models require instruction prefix for queries (`model.rs:227-234`): `"Instruct: Given a web search query, retrieve relevant passages...\nQuery: "`. Documents get no prefix (`document_instruction()` returns `None`).

**MRL truncation.** Matryoshka Representation Learning allows Qwen3-Embedding models to produce native-dimension vectors (1024d for 0.6B, 2560d for 4B) that are truncated to smaller dimensions while preserving quality. Only Qwen3 models support this (`supports_output_dim()` at `model.rs:263-268`); non-MRL models reject any `output_dim` value (`model.rs:420-425`). Truncation to 1024d or 1536d retains >95% quality (per Qwen3 paper) at 40-60% storage reduction. Implementation: `NativeEmbeddingService` passes `output_dim` to `QwenModel::set_output_dim()` at `native.rs:230`. Truncation happens inside the inference crate after forward pass + pooling, before L2 normalization -- the embed crate only passes the config. Min dimension: `MIN_MRL_OUTPUT_DIM = 32` (`model.rs:375`). Max: model's `native_dimensions()`. `LATTICE_EMBED_DIM` env var overrides `output_dim` for Qwen models (`native.rs:74-93`). Different `output_dim` values produce different `EmbeddingKey` (`model.rs:447-456`), ensuring separate HNSW indexes per dimension -- mixing dimensionalities in one index would produce garbage similarity scores.

**Two-layer cache.** Layer 1 (embed crate): in-memory sharded LRU. `EmbeddingCache` at `cache.rs:131` -- 16 shards, each `RwLock<LruCache<CacheKey, Arc<[f32]>>>` with per-shard `AtomicU64` hit/miss counters. Shard selection: `key[0] & 0xF`. Default capacity 4000 entries (250/shard). Blake3-based cache key: `text_bytes + model_config.to_embedding_key().canonical_bytes()` -- different MRL dims produce different keys. Eviction: LRU per-shard. Layer 2 (inference crate): `QwenModel` persistent disk cache at `~/.lattice/cache/embed_{model}_{dim}d.bin` (`native.rs:219-248`). Loaded at model init, saved explicitly. Uses `HashMap<u64, Vec<f32>>` with flush-all eviction at 10K cap (see `crates/inference/DESIGN.md` KC-3). BERT models have no persistent cache. The two layers use different key formats: Layer 1 uses blake3 of text + model config (embed crate doesn't have token IDs), Layer 2 uses hash of token ID sequence (inference crate operates post-tokenization). This avoids coupling between crates. `CachedEmbeddingService` (`cached.rs:42`) wraps any `EmbeddingService`: compute blake3 keys -> batch `get_many()` -> embed misses only -> `put_many()` -> merge preserving order. FP-035 fix at `cached.rs:153-159` validates embedding count matches input count.

**Batch processing.** Entry: `NativeEmbeddingService::embed()` at `native.rs:326-357`. Input validation: model mismatch -> error, empty input -> error, batch <= 1000 (`DEFAULT_MAX_BATCH_SIZE`), per-text <= 32768 chars (`MAX_TEXT_CHARS`). Model loads exactly once via `OnceLock` on a blocking thread (`native.rs:161-189`) -- cancellation-safe, survives async drops. BERT models: native batch processing via `BertModel::encode_batch()`. Qwen models: sequential per-item `QwenModel::encode()` (checks internal cache per item) -- sequential because `QwenModel` holds a `Mutex<ForwardBuffers>` preventing parallel forward passes. Batch parallelism would require per-item buffer allocation, adding memory pressure for marginal gain (GPU forward is already fast). Output: `Vec<Vec<f32>>` with order preserved. Core trait: `EmbeddingService` (`service/mod.rs:55-86`) with `embed()`, `embed_one()`, `model_config()`, `supports_model()`, `name()`.

## Alternatives Rejected

| Alternative                                             | Rejected Because                                                                                                                                                                                                        | Date |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| Single global LRU cache (no sharding)                   | Write contention under concurrent requests. 16-shard design gives ~16x write throughput at cost of slightly less optimal global LRU ordering. Blake3 key distribution ensures balanced sharding.                        | --   |
| Runtime dimension selection (no ModelConfig validation) | Would allow invalid dimensions (0, negative, above native) to reach inference. Validation at config construction catches errors early with clear messages.                                                              | --   |
| Unified cache key format across both layers             | Layer 1 needs text-level keys (embed crate doesn't have token IDs). Layer 2 needs token-level keys (inference crate operates post-tokenization). Forcing one format adds unnecessary coupling between crates.           | --   |
| Async model loading (non-blocking)                      | Model load involves mmap + GPU init -- CPU-bound, not I/O-bound. `spawn_blocking` + `OnceLock` is simpler and correct. Async model load would require `async fn init()` on the trait, complicating all implementations. | --   |

## Invariants

**INV-1.** Different `output_dim` values produce different `EmbeddingKey` (`model.rs:447-456`). If broken: embeddings of different dimensionality mixed in same HNSW index, producing garbage similarity scores.
**INV-2.** MRL only for models where `supports_output_dim()` returns true (Qwen3 only, `model.rs:263-268`). If broken: truncation of non-MRL model produces semantically invalid embeddings (BERT embeddings are not trained for truncation).
**INV-3.** MRL dimension in `[MIN_MRL_OUTPUT_DIM(32), native_dimensions()]` (`model.rs:426-437`). If broken: dimension below training minimum degrades quality severely; above native is padding with zeros.
**INV-4.** Model loads exactly once per `NativeEmbeddingService` lifetime (`OnceLock` at `native.rs:63-64`). If broken: double model load wastes memory, or concurrent load attempts race.
**INV-5.** Cache key includes `model_config.to_embedding_key().canonical_bytes()` (`cache.rs:194-198`). If broken: same text with different model/dimension shares cache entry, returning wrong-dimension embeddings.
**INV-6.** `CachedEmbeddingService` validates `embeddings.len() == texts_to_embed.len()` (FP-035, `cached.rs:153-159`). If broken: cache merge produces misaligned results (embedding N assigned to text M).
**INV-7.** Batch size <= 1000, text length <= 32768 chars. If broken: OOM on large batches or excessive tokenization time. Validated at both `native.rs` and `cached.rs` entry points.
**INV-8.** Requested model must equal loaded model (`native.rs:327-332`). If broken: model produces embeddings in wrong dimension/space.

## Known Concerns Acknowledged

**KC-1.** Enum default (`BgeSmallEnV15`) differs from production default (`multilingual-e5-small`). Intentional: enum default serves code paths without application config (tests, standalone usage). Production routing goes through application config. Documenting both prevents confusion.

**KC-2.** Qwen models process batch items sequentially, not in parallel (`native.rs:29-32`). Intentional: `QwenModel`'s internal cache check is per-item, and the model holds a `Mutex<ForwardBuffers>` preventing parallel forward passes. Batch parallelism would require per-item buffer allocation, adding memory pressure for marginal gain (GPU forward is already fast). (ref: `native.rs:29-32`)

**KC-3.** `EmbeddingCache` marked Unstable in public API (`lib.rs:87-94`). Intentional: cache internals (shard count, key format, LRU parameters) may change. The stable surface is `EmbeddingService` trait + `ModelConfig`.

**KC-4.** 36 unsafe blocks in the `simd/` module (`distance.rs`, `normalize.rs`, `cosine.rs`, `dot_product.rs`, `quantized.rs`, `binary.rs`, `int4.rs`). Intentional: NEON/AVX2 SIMD intrinsics for vectorized distance computation and normalization in performance-critical embedding operations. TCB-justified -- safe abstractions would impose measurable overhead on hot-path distance calculations.

## Baseline Metrics

| Dimension | Metric                                       | Value                   | Measured   | Threshold | Command                                                           |
| --------- | -------------------------------------------- | ----------------------- | ---------- | --------- | ----------------------------------------------------------------- |
| perf      | Embed latency (mE5-small, 100 texts, cached) | pending                 | --         | +20%      | `cargo bench -p lattice-embed --bench embed`                      |
| perf      | Embed latency (Qwen3-0.6B, 100 texts, cold)  | pending                 | --         | +20%      | `cargo bench -p lattice-embed --bench embed`                      |
| perf      | Cache hit rate (steady state)                | pending                 | --         | -10%      | (logged in cache stats)                                           |
| quality   | MRL 1024d vs native 2560d cosine similarity  | >0.95 (per Qwen3 paper) | 2026-04-29 | <0.93     | pending eval harness                                              |
| security  | unsafe block count                           | 36                      | 2026-04-29 | +5        | `grep -rw "unsafe {" crates/embed/src/ --include="*.rs" \| wc -l` |
| quality   | Model variant count                          | 8                       | 2026-04-29 | -1        | count `EmbeddingModel` enum variants                              |

## Change Protocol

1. Read this DESIGN.md before modifying any file in `crates/embed/`.
2. Check Baseline Metrics -- run the measurement commands and compare against thresholds.
3. Check Known Concerns -- ensure your change doesn't re-introduce a concern already acknowledged.
4. Re-measure baselines after your change and update the table if values shift.
