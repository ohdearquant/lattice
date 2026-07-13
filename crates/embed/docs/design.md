# lattice-embed design

`lattice-embed` generates vector embeddings for the lattice-runtime substrate: pure-Rust
local inference (no ONNX, no Python) over the BGE and multilingual E5/Qwen3 model
families, plus an async `EmbeddingService` API and SIMD-accelerated vector ops
(`utils::cosine_similarity`, `dot_product`, `normalize`, `euclidean_distance`) consumed
directly by khive's ANN indexes.

For model variants, MRL truncation, the two-layer cache, and batch-processing internals,
see `DESIGN.md` at the crate root (line-numbered design notes with invariants and
rejected alternatives). This file covers the model-migration/backfill workflow, which is
large enough to warrant its own narrative.

## Architecture

The crate is organized leaf-most first:

- **Model & config** — `model`: the `EmbeddingModel` enum and `ModelConfig` (MRL output
  dimension, validation).
- **Cache** — `cache`: sharded in-memory LRU (`EmbeddingCache`), Blake3-keyed.
- **Services** — `service`: the `EmbeddingService` trait, `NativeEmbeddingService` (local
  inference), `CachedEmbeddingService` (LRU wrapper).
- **SIMD** — `simd`: dot product, cosine similarity, normalization, and int8/int4/binary
  quantized distance kernels with per-platform dispatch (AVX-512/AVX2/NEON/wasm32
  SIMD128). See the module-level docs inside `simd/` for the numeric and dispatch
  invariants — those stay inline because they are load-bearing correctness notes, not
  background.
- **Migration** — `migration`: the `MigrationController` state machine that tracks a
  single embedding-model migration from `Planned` through `Completed`.
- **Backfill** — `backfill`: the `BackfillCoordinator`, which wraps a
  `MigrationController` and adds request/query routing plus batch-size management on top
  of the state machine.
- **wasm** (optional, `wasm` feature + `wasm32` target only) — `wasm-bindgen` bindings
  over the SIMD utility functions.

## Model migration and backfill

Swapping embedding models (for example BGE-small to multilingual-E5-small) requires
re-embedding every stored document without an all-at-once cutover. The migration and
backfill modules split this into two layers:

1. **`MigrationController`** (in `migration`) is a plain state machine over a
   `MigrationPlan`: `Planned -> InProgress -> Completed`, with `Paused`, `Failed`, and
   `Cancelled` as side states. It has no opinion about routing — it only tracks progress
   (`record_progress`) and exposes the current `MigrationState`.
2. **`BackfillCoordinator`** (in `backfill`) wraps a `MigrationController` and answers the
   question every request needs answered: *which model handles this?* It layers routing
   decisions on top of the same state machine:

   | State       | New Doc Route | Existing Doc Route | Query Route         |
   |-------------|---------------|---------------------|----------------------|
   | Planned     | Legacy        | Legacy              | Legacy               |
   | InProgress  | DualWrite*    | Legacy               | Legacy or Target**  |
   | Paused      | Legacy        | Legacy              | Legacy               |
   | Completed   | Target        | Target              | Target               |
   | Failed      | Legacy        | Legacy              | Legacy               |
   | Cancelled   | Legacy        | Legacy              | Legacy               |

   \* Only when `dual_write` is enabled in `BackfillConfig`.
   \*\* Switches to `Target` once backfill progress reaches `target_query_threshold`.

The overall migration sequence a caller drives is:

1. Continue serving queries against the existing (legacy) embeddings.
2. Route new document embeddings appropriately (dual-write during `InProgress`, or
   target-only once cutover is safe).
3. Backfill old documents with the new model's embeddings in batches
   (`BackfillCoordinator` computes batch sizes from progress).
4. Switch queries to the new model once coverage crosses `target_query_threshold`.

This separation keeps the state machine (`migration`) reusable for any two-phase model
swap, while the routing policy (`backfill`) stays specific to the embedding read/write
paths.
