# lattice-embed

Vector embedding generation with SIMD-accelerated operations for semantic search and
similarity matching.

## Features

- **Native Embeddings**: Generate embeddings locally using pure Rust inference (no ONNX, no Python)
- **SIMD Acceleration**: AVX2/AVX-512/NEON optimized vector operations (7x speedup)
- **LRU Cache**: SHA-256-based caching to avoid recomputation
- **Async API**: Full async/await support with tokio

## Models

| Model                               | Dimensions | Use Case                                    | Model ID                                                    |
| ----------------------------------- | ---------- | ------------------------------------------- | ----------------------------------------------------------- |
| `BgeSmallEnV15`                     | 384        | Fast, general purpose (default)             | BAAI/bge-small-en-v1.5                                      |
| `BgeBaseEnV15`                      | 768        | Balanced quality/speed                      | BAAI/bge-base-en-v1.5                                       |
| `BgeLargeEnV15`                     | 1024       | Highest quality local                       | BAAI/bge-large-en-v1.5                                      |
| `MultilingualE5Small`               | 384        | Multilingual, same arch as BGE              | intfloat/multilingual-e5-small                              |
| `MultilingualE5Base`                | 768        | Best multilingual quality/speed             | intfloat/multilingual-e5-base                               |
| `AllMiniLmL6V2`                     | 384        | BERT-class, WordPiece tokenizer             | sentence-transformers/all-MiniLM-L6-v2                      |
| `ParaphraseMultilingualMiniLmL12V2` | 384        | Multilingual, XLM-R base                    | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| `Qwen3Embedding0_6B`                | 1024       | Multilingual, decoder-only, GPU-accelerated | Qwen/Qwen3-Embedding-0.6B                                   |
| `Qwen3Embedding4B`                  | 2560       | Multilingual, decoder-only, MRL-capable     | Qwen/Qwen3-Embedding-4B                                     |
| `EmbeddingGemma2`                   | 768        | Multilingual, bidirectional, MRL-capable    | google/embeddinggemma-2                                     |
| `TextEmbedding3Small`               | 1536       | Remote API (OpenAI, scaffold-only)          | text-embedding-3-small                                      |

`TextEmbedding3Small`'s ID is an OpenAI model id, not a HuggingFace one; `EmbeddingModel::is_remote()`
returns `true` for it and the native embedding service rejects it (see
[supported-model reference](../../docs/models.md)).

Dimensions and input-token limits vary per model. See the
[supported-model reference](../../docs/models.md) for every variant's limit, pooling strategy,
and shipped/partial/scaffold-only availability.

## Services

### NativeEmbeddingService

Single-model service with lazy initialization. Inference is serialized (one call at a
time). Concurrency is handled internally.

```rust
use lattice_embed::{EmbeddingService, EmbeddingModel, NativeEmbeddingService};

let service = NativeEmbeddingService::default();
let embedding = service.embed_one("Hello, world!", EmbeddingModel::default()).await?;
assert_eq!(embedding.len(), 384);
```

### CachedEmbeddingService

Wraps any `EmbeddingService` with LRU caching. Identical texts return cached embeddings.

```rust
use lattice_embed::{CachedEmbeddingService, NativeEmbeddingService, EmbeddingService};
use std::sync::Arc;

let inner = Arc::new(NativeEmbeddingService::new());
let cached = CachedEmbeddingService::new(inner, 1000); // 1000 entry cache

let emb1 = cached.embed_one("Hello", EmbeddingModel::default()).await?;
let emb2 = cached.embed_one("Hello", EmbeddingModel::default()).await?; // cache hit
```

## SIMD Vector Operations

Dispatch is resolved per platform. x86_64 uses runtime feature detection with automatic
scalar fallback. On aarch64 the f32 kernels always use NEON (mandatory on that
architecture, no runtime check), while the int8 kernel runtime-detects SDOT. On wasm32
the SIMD choice is fixed at compile time and covers the f32 kernels only (see below).

| Platform       | Instructions                                      |
| -------------- | ------------------------------------------------- |
| x86_64 (f32)   | AVX-512F > AVX2 + FMA > scalar                    |
| x86_64 (int8)  | AVX-512 VNNI (runtime-detected) > AVX2 > scalar   |
| aarch64 (f32)  | ARM NEON (mandatory, always on)                   |
| aarch64 (int8) | NEON SDOT (runtime-detected) > scalar             |
| wasm32         | SIMD128 (compile-time, f32 kernels only) > scalar |
| Other          | Scalar fallback                                   |

### wasm32 (SIMD128)

The SIMD128 route exists for the four f32 kernels: `dot_product`, `cosine_similarity`,
squared Euclidean distance, and `normalize`. The int8 path (`dot_product_i8`) has no
SIMD128 branch and always dispatches to the scalar kernel on wasm32.

Unlike the x86_64/aarch64 rows above, wasm32 has no runtime CPU-feature detection: a given
`.wasm` binary either was or wasn't compiled with `-C target-feature=+simd128`, and that
choice is fixed for the whole artifact. `SimdConfig::simd128_enabled()` mirrors this
directly (`cfg!(target_feature = "simd128")`) rather than reading a field, so it is not
something a config value can override -- a force-scalar-style `SimdConfig` cannot re-route a
simd128-compiled wasm build back to the scalar kernels. To get scalar dispatch on wasm32,
build without the `simd128` target feature; there is no runtime override for a build that
has it.

### Performance (384-dim vectors)

Measured by `crates/embed/benches/simd_bench.rs`; run `cargo bench -p lattice-embed` to
reproduce on your own hardware — these are illustrative, not a portability guarantee.

| Operation         | Scalar | SIMD  | Speedup |
| ----------------- | ------ | ----- | ------- |
| cosine_similarity | ~650ns | ~90ns | 7x      |
| dot_product       | ~230ns | ~35ns | 6.5x    |
| normalize         | ~400ns | ~60ns | 6.5x    |
| dot_product_i8    | ~300ns | ~25ns | 12x     |

### Usage

```rust
use lattice_embed::utils::{cosine_similarity, dot_product, normalize, euclidean_distance};

let a = vec![1.0, 0.0, 0.0];
let b = vec![1.0, 0.0, 0.0];

let sim = cosine_similarity(&a, &b);  // 1.0 (identical direction)
let dot = dot_product(&a, &b);        // 1.0
let dist = euclidean_distance(&a, &b); // 0.0

let mut v = vec![3.0, 4.0];
normalize(&mut v); // v = [0.6, 0.8], magnitude = 1.0
```

### Int8 Quantization

4x memory reduction with ~99% accuracy:

```rust
use lattice_embed::simd::QuantizedVector;

let v = vec![0.1, 0.2, 0.3, /* ... */];
let q = QuantizedVector::from_f32(&v);

// Compare quantized vectors directly
let sim = q.cosine_similarity(&other_quantized);
```

## Cache

SHA-256-based hashing with LRU eviction. Default capacity: 4000 entries (~6MB for 384-dim
vectors).

```rust
use lattice_embed::{EmbeddingCache, EmbeddingModel, EmbeddingRole, ModelConfig};

let cache = EmbeddingCache::new(1000); // custom capacity; use `with_default_capacity()` for 4000

let key = cache.compute_key(
    "text",
    ModelConfig::new(EmbeddingModel::BgeSmallEnV15),
    EmbeddingRole::Generic,
);
cache.put(key, vec![0.1, 0.2, 0.3]);

let stats = cache.stats();
println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);
```

## Feature Flags

| Feature     | Default | Description                                                                                                                                                                                                                         |
| ----------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `native`    | Yes     | Enable local embedding via pure Rust inference                                                                                                                                                                                      |
| `image`     | Yes     | PNG/JPEG decoding for vision embeddings (implies `native`)                                                                                                                                                                          |
| `download`  | Yes     | First-use model downloads (implies `native`)                                                                                                                                                                                        |
| `wasm`      | No      | wasm-bindgen JS bindings for a browser-loadable `.wasm` module                                                                                                                                                                      |
| `metal-gpu` | No      | Enables `lattice-inference/metal-gpu`                                                                                                                                                                                               |
| `avx512`    | No      | Deprecated no-op — the AVX-512 VNNI int8 kernel is compiled unconditionally on stable x86_64 and is selected at runtime only when AVX-512F/BW/VNNI are all detected; otherwise AVX2 or scalar runs; kept for manifest compatibility |
| `local`     | No      | Deprecated no-op — gated a bench module for a service removed long ago; kept for manifest compatibility                                                                                                                             |

```toml
[dependencies]
lattice-embed = { version = "0.12", default-features = false }  # SIMD only
lattice-embed = { version = "0.12" }  # Full (native + SIMD)
```

## Batch Processing

```rust
let texts = vec![
    "First document".to_string(),
    "Second document".to_string(),
    "Third document".to_string(),
];

let embeddings = service.embed(&texts, EmbeddingModel::BgeSmallEnV15).await?;
assert_eq!(embeddings.len(), 3);
```

Maximum batch size: 1000 texts (to prevent OOM).

## API Reference

### EmbeddingService Trait

```rust
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    async fn embed(&self, texts: &[String], model: EmbeddingModel) -> Result<Vec<Vec<f32>>>;
    async fn embed_one(&self, text: &str, model: EmbeddingModel) -> Result<Vec<f32>>;
    fn supports_model(&self, model: EmbeddingModel) -> bool;
    fn name(&self) -> &'static str;
}
```

### SIMD Config

```rust
use lattice_embed::simd_config;

let config = simd_config();
println!("AVX2: {}, NEON: {}", config.avx2_enabled, config.neon_enabled);
```

## Crate reference

The sections below are generated by `scripts/crate-readme.py` from the source tree and are refreshed after pushes to `main`. Text between the `generated` markers is overwritten; edit the source or the script instead.

<!-- generated:facts:begin -->

### Facts

- Version: `0.15.0`
- Minimum supported Rust version: `1.93`
- Edition: `2024`
- License: `MIT OR Apache-2.0`
- crates.io: <https://crates.io/crates/lattice-embed>
- API documentation: <https://docs.rs/lattice-embed>

<!-- generated:facts:end -->

<!-- generated:layout:begin -->

### Source layout

The tracked files under `src/`; each Rust file shows the first sentence of its `//!` module documentation.

<details>
<summary><code>src/</code>: 49 files</summary>

- `backfill/`
  - `coordinator.rs`: Coordinates an embedding backfill above the migration state machine.
  - `mod.rs`: Public API for model-migration backfills.
  - `tests.rs`: Tests for the backfill module.
  - `types.rs`: Defines migration-time routing decisions and backfill configuration.
- `bin/`
  - `embed-drift.rs`: Command-line embedding drift check and frozen-baseline updater.
  - `embed.rs`: CLI tool for generating text embeddings using lattice-embed.
- `cache.rs`: In-memory embedding cache with sharded LRU eviction.
- `cache/`
  - `tests.rs`
- `drift.rs`: Frozen-baseline embedding drift comparison.
- `error.rs`: Errors returned by embedding, model configuration, and prepared SIMD-dispatch operations.
- `lib.rs`: lattice-embed
- `migration/`
  - `controller.rs`: Executes lifecycle transitions and progress accounting for one migration.
  - `mod.rs`: Public API for the embedding-model migration state machine.
  - `tests.rs`: Tests for the migration module.
  - `types.rs`: Defines migration plans, states, progress snapshots, skip reasons, and errors.
- `model.rs`: Embedding model selection, runtime dimensions, and load provenance.
- `model/`
  - `tests.rs`
- `service/`
  - `cached.rs`: Native-only LRU wrapper for an `EmbeddingService`.
  - `cached/`
    - `tests.rs`
  - `mod.rs`: Async embedding-service contract and native implementations.
  - `native.rs`: Native, pure-Rust embedding service backed by the lattice model-family crates.
  - `native_pooling_tests.rs`
  - `prepared.rs`: Sealed native embedding attestation contracts.
  - `prepared/`
    - `limits.rs`: Dormant checked limits and accounting for sealed native preparation.
    - `limits/`
      - `tests.rs`
    - `resource.rs`: Dormant private admission and drain substrate for ADR-088.
    - `resource/`
      - `tests.rs`
    - `tensor_inventory.rs`: Canonical tensor-inventory identity framing for sealed native preparation.
    - `tensor_inventory/`
      - `tests.rs`
  - `tests.rs`: Tests for embedding services.
- `simd/`
  - `binary.rs`: Binary sign quantization and Hamming-distance operations.
  - `binary/`
    - `tests.rs`
  - `cosine.rs`: SIMD cosine-similarity kernels and batch variants.
  - `distance.rs`: SIMD Euclidean and squared-Euclidean distance kernels.
  - `dot_product.rs`: SIMD float dot-product kernels, including a one-query/four-candidate path.
  - `int4.rs`: INT4 packed-vector quantization and approximate dot products.
  - `int4/`
    - `tests.rs`
  - `manhattan.rs`: SIMD Manhattan (L1) distance kernels.
  - `mod.rs`: SIMD vector operations for embedding similarity and compressed-vector search.
  - `normalize.rs`: SIMD in-place L2 normalization kernels.
  - `quantized.rs`: INT8 vector quantization and approximate similarity kernels.
  - `quantized/`
    - `simd_parity_tests.rs`
  - `tests.rs`: Tests for SIMD operations.
  - `tier.rs`: Quantization tiers, prepared queries, and unified distance dispatch.
  - `tier/`
    - `tests.rs`
- `types.rs`: Vector-space identity and storage-format descriptors for `lattice-embed`.
- `vision.rs`: Image (and image+text) embedding through the Qwen3.5 vision-language pooled-embedding pipeline (ADR-069 S5, #1007).
- `vision/`
  - `tests.rs`
- `wasm.rs`: Browser `wasm-bindgen` bindings for in-memory BERT embeddings and vector utilities.

</details>

Targets declared by the manifest, including the ones Cargo discovers by directory:

- rlib (1): `lattice_embed`
- cdylib (1): `lattice_embed`
- bin (2): `embed-drift` (requires `native`), `embed` (requires `native`)
- example (4): `basic_embed` (requires `native`), `dump_parity_embeddings` (requires `native`), `prompt_router_vectors`, `similarity`
- test (16): `drift_api`, `embed_cli_image_args`, `embed_drift_baseline`, `embed_drift_cli`, `embed_parity_vs_hf`, `encoder_consumer`, `family_consumer`, `inference_dot_product_parity`, `integration`, `prepared_attestation_api`, `prepared_resource_budget_api`, `simd_edge_cases`, `simd_precision`, `token_report`, `tokenizer_parity_e2e`, `vision_single_file_real_model`
- bench (5): `embeddings`, `simd_bench`, `simd_opt_bench`, `simd`, `simsimd_comparison`

<!-- generated:layout:end -->

<!-- generated:stats:begin -->

### Statistics

Rust lines by kind (code, comment, blank; doc comments count as comment), test functions (`#[test]` and `#[<runtime>::test]` attributes), items declared `pub` (not `pub(crate)`; reachability through private modules is not resolved) and occurrences of the `unsafe` keyword outside comments and strings.

| Area        | Files | Code  | Comment | Blank | Tests | `pub` items | `unsafe` |
| ----------- | ----- | ----- | ------- | ----- | ----- | ----------- | -------- |
| `src/`      | 49    | 17640 | 2517    | 2421  | 418   | 303         | 119      |
| `tests/`    | 16    | 2896  | 498     | 487   | 111   | 0           | 1        |
| `benches/`  | 5     | 1616  | 286     | 366   | 0     | 0           | 0        |
| `examples/` | 4     | 342   | 79      | 64    | 0     | 0           | 0        |
| total       | 74    | 22494 | 3380    | 3338  | 529   | 303         | 120      |

Other tracked files (the README excluded):

| Extension | Files | Size      |
| --------- | ----- | --------- |
| `.json`   | 10    | 455.4 KiB |
| `.md`     | 11    | 169.4 KiB |
| `.mjs`    | 2     | 25.1 KiB  |
| `.toml`   | 2     | 11.9 KiB  |
| `.txt`    | 1     | 386 B     |

<!-- generated:stats:end -->

<!-- generated:docs:begin -->

### In-crate documentation

- [Changelog](CHANGELOG.md) (`CHANGELOG.md`)
- [lattice-embed — CONFIG.md](CONFIG.md) (`CONFIG.md`)
- [Design: lattice-embed](DESIGN.md) (`DESIGN.md`)
- [lattice-embed Benchmarks](benches/README.md) (`benches/README.md`)
- [lattice-embed extended documentation](docs/INDEX.md) (`docs/INDEX.md`)
- [Embedding backfill](docs/backfill.md) (`docs/backfill.md`)
- [lattice-embed architecture](docs/design.md) (`docs/design.md`)
- [Embedding migration controller](docs/migration.md) (`docs/migration.md`)
- [Embedding models and cache](docs/model.md) (`docs/model.md`)
- [Embedding services](docs/service.md) (`docs/service.md`)
- [SIMD kernels and quantized-vector paths](docs/simd.md) (`docs/simd.md`)

<!-- generated:docs:end -->

<!-- generated:features:begin -->

### Features

Default set: `download`, `image`, `native`. Each description is the first sentence of the comment above the feature in `Cargo.toml`.

- `avx512`: Deprecated no-op: the AVX-512 VNNI int8 kernel this used to gate is now compiled unconditionally on x86_64 (runtime-dispatched via `SimdConfig::avx512vnni_enabled`, same pattern as the ungated f32 AVX-512F kernel in dot_product.rs) and no longer needs an opt-in Cargo feature. Enables nothing.
- `download` (default): First-use model downloads for the native `embed` binary (its `--download-only` flag and the implicit fetch inside `BertModel::from_pretrained`). Enables `native`, `lattice-model-bert/download`, `lattice-model-qwen3/download`.
- `image` (default): Feature forwards. Enables `native`, `lattice-model-qwen35/image`, `lattice-model-embeddinggemma2/image`.
- `local`: Deprecated no-op: gated a bench module for a service removed long ago, but the feature is part of the published 0.4.x surface (removing it would break manifests that list it). Enables nothing.
- `metal-gpu`: Enables `lattice-model-qwen3/metal-gpu`, `lattice-model-qwen35/metal-gpu`, `lattice-model-embeddinggemma2/metal-gpu`.
- `native` (default): Enables `dep:lattice-model-bert`, `dep:lattice-model-qwen3`, `dep:lattice-model-qwen35`, `dep:lattice-model-embeddinggemma2`, `dep:lattice-compute`, `dep:lattice-inference-core`, `dep:lattice-arch`, `dep:lattice-tokenizer`, `dep:lattice-weights`.
- `wasm`: wasm-bindgen JS bindings (`LatticeEmbedder`) for building a browser-loadable `.wasm` module. Enables `native`, `dep:wasm-bindgen`, `dep:console_error_panic_hook`.
- `x86-gemm`: Enables `lattice-compute/x86-gemm`, `lattice-model-bert/x86-gemm`, `lattice-model-qwen3/x86-gemm`.

<!-- generated:features:end -->

<!-- generated:dependencies:begin -->

### Workspace dependencies

- Depends on: `lattice-arch` (optional, enabled by `native`), `lattice-compute` (optional, enabled by `native`, `x86-gemm`), `lattice-inference-core` (optional, enabled by `native`), `lattice-tokenizer` (optional, enabled by `native`), `lattice-weights` (optional, enabled by `native`; features `f16`)
- Dev-depends on: `lattice-inference` (no default features; features `bert`, `qwen3`, `std`, `test-utils`, `qwen35`), `lattice-tokenizer`, `lattice-transport`
- Used by: no other crate in this workspace

<!-- generated:dependencies:end -->

<!-- generated:adrs:begin -->

### Architecture decisions

ADRs under `docs/adr/` that name `lattice-embed` or `crates/embed/`:

<details>
<summary>23 ADRs</summary>

- [ADR-012: Runtime SIMD Detection Strategy](../../docs/adr/ADR-012-simd-strategy.md)
- [ADR-013: lattice-embed as SIMD Foundation Layer](../../docs/adr/ADR-013-simd-foundation.md)
- [ADR-014: Embedding Service](../../docs/adr/ADR-014-embedding-service.md)
- [ADR-015: Sharded LRU Embedding Cache](../../docs/adr/ADR-015-embedding-cache.md)
- [ADR-016: Embedding Model Variants](../../docs/adr/ADR-016-model-variants.md)
- [ADR-017: NativeEmbeddingService Orchestration](../../docs/adr/ADR-017-native-embedding-service.md)
- [ADR-018: Quantized Vector Tiers](../../docs/adr/ADR-018-quantized-vectors.md)
- [ADR-019: Backfill and Migration Pipeline](../../docs/adr/ADR-019-backfill-pipeline.md)
- [ADR-047: Paged KV Cache with Prefix Reuse](../../docs/adr/ADR-047-paged-kv-cache.md)
- [ADR-058: CPU Performance Regression Tracking in CI](../../docs/adr/ADR-058-cpu-perf-regression-ci.md)
- [ADR-064: CI Gate Taxonomy and Promotion Policy](../../docs/adr/ADR-064-ci-gate-taxonomy.md)
- [ADR-066: Output-Correctness Gate Architecture](../../docs/adr/ADR-066-output-correctness-gate-architecture.md)
- [ADR-070: CPU Embedding Performance: Fused Batched Encode and the ONNX Parity Gate](../../docs/adr/ADR-070-cpu-embedding-batched-encode.md)
- [ADR-080: Consolidation of Duplicated Numeric-Contract Helpers (Softmax, HTTP Serving, Decode Policy, GEMM Validation)](../../docs/adr/ADR-080-consolidation-duplicated-contracts.md)
- [ADR-083: Backward SIMD Dispatch Contract](../../docs/adr/ADR-083-backward-simd-dispatch.md)
- [ADR-085: Explicit Quantization SIMD Dispatch](../../docs/adr/ADR-085-explicit-quantization-simd.md)
- [ADR-087 Amendment 1 — open pull request population snapshot](../../docs/adr/ADR-087-amendment-1-population-snapshot.md)
- [ADR-087: bench-compare gate calibration, coverage, and admissible structural proof](../../docs/adr/ADR-087-bench-compare-gate-calibration-and-coverage.md)
- [ADR-088: Sealed Native Embedding Preparation and Attestation](../../docs/adr/ADR-088-sealed-native-embedding-attestation.md)
- [ADR-099: Linux CPU Embedding GEMM, Parallel Execution, and Resource Budgets](../../docs/adr/ADR-099-linux-cpu-embedding-gemm-threading-budget.md)
- [ADR-103: Splitting lattice-inference into focused crates behind a facade](../../docs/adr/ADR-103-inference-crate-decomposition.md)
- [ADR-ALIGNMENT-AUDIT: ADR Set Classification as of 2026-06-30](../../docs/adr/ADR-ALIGNMENT-AUDIT.md)
- [Architecture Decision Records](../../docs/adr/INDEX.md)

</details>

<!-- generated:adrs:end -->

<!-- generated:benchmarks:begin -->

### Benchmarks

The newest Criterion point estimates `bench-update.yml` stored on the `perf-baselines` branch, from the `simd` target. They come from GitHub-hosted runners and serve as trend data between commits, not as a performance claim; a claim needs a paired A/B (`make bench-compare`).

- `aarch64-linux`: commit `0be6e2313fd1`, recorded 2026-10-27T07:07:28+00:00
- `x86_64-linux`: commit `0be6e2313fd1`, recorded 2026-10-27T07:09:00+00:00

<details>
<summary>255 benchmarks</summary>

| Benchmark                                                         | `aarch64-linux` | `x86_64-linux` |
| ----------------------------------------------------------------- | --------------- | -------------- |
| `binary_cosine_distance/binary/1024`                              | 11.0 ns         | 107.5 ns       |
| `binary_cosine_distance/binary/1536`                              | 15.8 ns         | 155.1 ns       |
| `binary_cosine_distance/binary/384`                               | 5.5 ns          | 48.7 ns        |
| `binary_cosine_distance/binary/768`                               | 9.2 ns          | 84.0 ns        |
| `binary_cosine_distance/float32_simd/1024`                        | 129.0 ns        | 68.3 ns        |
| `binary_cosine_distance/float32_simd/1536`                        | 186.7 ns        | 81.5 ns        |
| `binary_cosine_distance/float32_simd/384`                         | 59.9 ns         | 30.8 ns        |
| `binary_cosine_distance/float32_simd/768`                         | 101.2 ns        | 55.7 ns        |
| `cached_embedding_delegate/native_wrong_model_disabled_64x4096`   | 236.1 ns        | 224.1 ns       |
| `int4_cosine_distance/float32_simd/1024`                          | 128.8 ns        | 68.8 ns        |
| `int4_cosine_distance/float32_simd/1536`                          | 186.8 ns        | 81.3 ns        |
| `int4_cosine_distance/float32_simd/384`                           | 59.9 ns         | 28.3 ns        |
| `int4_cosine_distance/float32_simd/768`                           | 101.1 ns        | 55.5 ns        |
| `int4_cosine_distance/int4/1024`                                  | 124.3 ns        | 353.4 ns       |
| `int4_cosine_distance/int4/1536`                                  | 181.1 ns        | 528.2 ns       |
| `int4_cosine_distance/int4/384`                                   | 53.4 ns         | 137.9 ns       |
| `int4_cosine_distance/int4/768`                                   | 96.1 ns         | 268.2 ns       |
| `int8_batch_cosine/float32_simd/10`                               | 576.6 ns        | 280.7 ns       |
| `int8_batch_cosine/float32_simd/100`                              | 6.89 µs         | 3.93 µs        |
| `int8_batch_cosine/float32_simd/1000`                             | 92.75 µs        | 105.19 µs      |
| `int8_batch_cosine/int8_loop/10`                                  | 164.9 ns        | 190.4 ns       |
| `int8_batch_cosine/int8_loop/100`                                 | 1.70 µs         | 1.85 µs        |
| `int8_batch_cosine/int8_loop/1000`                                | 19.87 µs        | 18.36 µs       |
| `int8_prepared_dot_product/per_call/1024`                         | 2.45 µs         | 1.50 µs        |
| `int8_prepared_dot_product/per_call/127`                          | 359.5 ns        | 281.8 ns       |
| `int8_prepared_dot_product/per_call/128`                          | 348.7 ns        | 224.5 ns       |
| `int8_prepared_dot_product/per_call/129`                          | 355.1 ns        | 248.9 ns       |
| `int8_prepared_dot_product/per_call/384`                          | 948.8 ns        | 574.3 ns       |
| `int8_prepared_dot_product/per_call/768`                          | 1.85 µs         | 1.13 µs        |
| `int8_prepared_dot_product/prepared/1024`                         | 31.5 ns         | 25.6 ns        |
| `int8_prepared_dot_product/prepared/127`                          | 12.4 ns         | 16.2 ns        |
| `int8_prepared_dot_product/prepared/128`                          | 8.4 ns          | 13.1 ns        |
| `int8_prepared_dot_product/prepared/129`                          | 9.1 ns          | 13.1 ns        |
| `int8_prepared_dot_product/prepared/384`                          | 14.8 ns         | 17.7 ns        |
| `int8_prepared_dot_product/prepared/768`                          | 24.8 ns         | 20.2 ns        |
| `int8_quantization/quantize/1024`                                 | 2.41 µs         | 1.47 µs        |
| `int8_quantization/quantize/1536`                                 | 3.60 µs         | 2.17 µs        |
| `int8_quantization/quantize/384`                                  | 949.8 ns        | 594.1 ns       |
| `int8_quantization/quantize/768`                                  | 1.83 µs         | 1.10 µs        |
| `int8_raw_dot_product/dot_product_i8/1024`                        | 30.7 ns         | 23.8 ns        |
| `int8_raw_dot_product/dot_product_i8/127`                         | 11.8 ns         | 16.2 ns        |
| `int8_raw_dot_product/dot_product_i8/128`                         | 8.2 ns          | 12.9 ns        |
| `int8_raw_dot_product/dot_product_i8/129`                         | 8.6 ns          | 12.9 ns        |
| `int8_raw_dot_product/dot_product_i8/384`                         | 14.0 ns         | 17.6 ns        |
| `int8_raw_dot_product/dot_product_i8/768`                         | 24.5 ns         | 18.9 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/1024`                    | 28.1 ns         | 23.7 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/127`                     | 10.0 ns         | 14.8 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/128`                     | 6.5 ns          | 12.5 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/129`                     | 7.2 ns          | 12.4 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/384`                     | 12.4 ns         | 16.8 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/768`                     | 21.8 ns         | 18.2 ns        |
| `int8_vs_float32_cosine/float32_simd/1024`                        | 129.0 ns        | 67.8 ns        |
| `int8_vs_float32_cosine/float32_simd/1536`                        | 185.1 ns        | 95.5 ns        |
| `int8_vs_float32_cosine/float32_simd/384`                         | 58.3 ns         | 27.9 ns        |
| `int8_vs_float32_cosine/float32_simd/768`                         | 99.5 ns         | 51.0 ns        |
| `int8_vs_float32_cosine/int8/1024`                                | 33.0 ns         | 24.4 ns        |
| `int8_vs_float32_cosine/int8/1536`                                | 46.1 ns         | 34.9 ns        |
| `int8_vs_float32_cosine/int8/384`                                 | 16.4 ns         | 19.6 ns        |
| `int8_vs_float32_cosine/int8/768`                                 | 27.3 ns         | 20.4 ns        |
| `memory_size/search_1000_float32`                                 | 57.40 µs        | 32.87 µs       |
| `memory_size/search_1000_int8`                                    | 14.92 µs        | 18.64 µs       |
| `simd_batch_cosine/scalar_loop/10`                                | 6.40 µs         | 6.44 µs        |
| `simd_batch_cosine/scalar_loop/100`                               | 64.17 µs        | 63.95 µs       |
| `simd_batch_cosine/scalar_loop/1000`                              | 645.63 µs       | 661.57 µs      |
| `simd_batch_cosine/simd_batch/10`                                 | 577.3 ns        | 304.3 ns       |
| `simd_batch_cosine/simd_batch/100`                                | 6.91 µs         | 3.96 µs        |
| `simd_batch_cosine/simd_batch/1000`                               | 86.46 µs        | 103.78 µs      |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_1000c`    | 146.52 µs       | 145.83 µs      |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_16c`      | 2.08 µs         | 1.32 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_256c`     | 35.62 µs        | 21.28 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_4c`       | 520.4 ns        | 260.5 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_64c`      | 8.27 µs         | 5.30 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_1000c`     | 58.10 µs        | 31.85 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_16c`       | 922.0 ns        | 452.3 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_256c`      | 14.79 µs        | 7.83 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_4c`        | 237.3 ns        | 124.2 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_64c`       | 3.70 µs         | 1.96 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_1000c`     | 106.90 µs       | 104.18 µs      |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_16c`       | 1.61 µs         | 910.5 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_256c`      | 27.73 µs        | 14.30 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_4c`        | 407.1 ns        | 208.8 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_64c`       | 6.43 µs         | 3.57 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_1000c`   | 146.50 µs       | 144.27 µs      |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_16c`     | 2.07 µs         | 1.30 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_256c`    | 35.55 µs        | 20.90 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_4c`      | 518.4 ns        | 259.0 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_64c`     | 8.24 µs         | 5.21 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_1000c`    | 57.70 µs        | 30.59 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_16c`      | 918.0 ns        | 440.2 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_256c`     | 14.69 µs        | 7.49 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_4c`       | 235.6 ns        | 122.2 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_64c`      | 3.69 µs         | 1.89 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_1000c`    | 107.13 µs       | 104.84 µs      |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_16c`      | 1.62 µs         | 898.3 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_256c`     | 27.70 µs        | 14.14 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_4c`       | 405.4 ns        | 207.3 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_64c`      | 6.42 µs         | 3.52 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_1000c` | 146.51 µs       | 141.76 µs      |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_16c`   | 2.08 µs         | 1.12 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_256c`  | 35.63 µs        | 18.06 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_4c`    | 519.3 ns        | 245.5 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_64c`   | 8.22 µs         | 4.47 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_1000c`  | 57.98 µs        | 35.09 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_16c`    | 920.5 ns        | 470.1 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_256c`   | 15.50 µs        | 8.62 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_4c`     | 236.8 ns        | 119.3 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_64c`    | 3.71 µs         | 2.13 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_1000c`  | 106.57 µs       | 101.43 µs      |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_16c`    | 1.61 µs         | 1.03 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_256c`   | 28.63 µs        | 16.49 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_4c`     | 405.3 ns        | 214.8 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_64c`    | 6.41 µs         | 4.11 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_1000c`    | 97.12 µs        | 141.24 µs      |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_16c`      | 1.38 µs         | 981.4 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_256c`     | 27.39 µs        | 16.20 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_4c`       | 296.7 ns        | 164.1 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_64c`      | 6.06 µs         | 3.99 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_1000c`     | 38.14 µs        | 30.58 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_16c`       | 495.8 ns        | 367.5 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_256c`      | 10.55 µs        | 7.67 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_4c`        | 131.9 ns        | 92.1 ns        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_64c`       | 2.43 µs         | 1.91 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_1000c`     | 73.74 µs        | 96.70 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_16c`       | 964.6 ns        | 901.9 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_256c`      | 22.09 µs        | 14.71 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_4c`        | 230.7 ns        | 177.8 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_64c`       | 4.59 µs         | 3.64 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_1000c`       | 146.52 µs       | 149.88 µs      |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_16c`         | 2.07 µs         | 1.10 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_256c`        | 35.55 µs        | 17.90 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_4c`          | 516.5 ns        | 244.7 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_64c`         | 8.21 µs         | 4.46 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/384d_1000c`        | 57.51 µs        | 33.19 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_16c`          | 915.6 ns        | 445.8 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_256c`         | 15.38 µs        | 8.18 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/384d_4c`           | 234.7 ns        | 118.2 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_64c`          | 3.68 µs         | 2.06 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_1000c`        | 107.30 µs       | 102.71 µs      |
| `simd_batch_cosine_normalized_query/simd_batch/768d_16c`          | 1.60 µs         | 1.00 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_256c`         | 28.87 µs        | 16.08 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_4c`           | 403.9 ns        | 213.1 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_64c`          | 6.38 µs         | 4.01 µs        |
| `simd_batch_dot_product/scalar_loop/10`                           | 2.15 µs         | 2.17 µs        |
| `simd_batch_dot_product/scalar_loop/100`                          | 21.53 µs        | 22.13 µs       |
| `simd_batch_dot_product/scalar_loop/1000`                         | 223.11 µs       | 248.48 µs      |
| `simd_batch_dot_product/simd_batch/10`                            | 336.3 ns        | 206.6 ns       |
| `simd_batch_dot_product/simd_batch/100`                           | 5.69 µs         | 3.69 µs        |
| `simd_batch_dot_product/simd_batch/1000`                          | 92.64 µs        | 100.25 µs      |
| `simd_cosine_similarity/scalar/1024`                              | 1.77 µs         | 1.94 µs        |
| `simd_cosine_similarity/scalar/1536`                              | 2.68 µs         | 2.98 µs        |
| `simd_cosine_similarity/scalar/384`                               | 644.5 ns        | 643.6 ns       |
| `simd_cosine_similarity/scalar/768`                               | 1.32 µs         | 1.42 µs        |
| `simd_cosine_similarity/simd/1024`                                | 130.1 ns        | 60.8 ns        |
| `simd_cosine_similarity/simd/1536`                                | 185.5 ns        | 80.0 ns        |
| `simd_cosine_similarity/simd/384`                                 | 58.0 ns         | 29.7 ns        |
| `simd_cosine_similarity/simd/768`                                 | 100.5 ns        | 54.7 ns        |
| `simd_dot_product/scalar/1024`                                    | 592.2 ns        | 646.3 ns       |
| `simd_dot_product/scalar/1536`                                    | 893.8 ns        | 991.4 ns       |
| `simd_dot_product/scalar/384`                                     | 214.5 ns        | 213.2 ns       |
| `simd_dot_product/scalar/768`                                     | 440.4 ns        | 470.9 ns       |
| `simd_dot_product/simd/1024`                                      | 73.6 ns         | 38.7 ns        |
| `simd_dot_product/simd/1536`                                      | 106.8 ns        | 57.0 ns        |
| `simd_dot_product/simd/384`                                       | 31.9 ns         | 15.5 ns        |
| `simd_dot_product/simd/768`                                       | 57.4 ns         | 29.1 ns        |
| `simd_euclidean_distance/scalar/1024`                             | 596.1 ns        | 663.9 ns       |
| `simd_euclidean_distance/scalar/1536`                             | 896.7 ns        | 1.01 µs        |
| `simd_euclidean_distance/scalar/384`                              | 217.2 ns        | 231.1 ns       |
| `simd_euclidean_distance/scalar/768`                              | 444.1 ns        | 488.8 ns       |
| `simd_euclidean_distance/simd/1024`                               | 116.8 ns        | 46.5 ns        |
| `simd_euclidean_distance/simd/1536`                               | 171.7 ns        | 64.6 ns        |
| `simd_euclidean_distance/simd/384`                                | 49.1 ns         | 19.2 ns        |
| `simd_euclidean_distance/simd/768`                                | 90.0 ns         | 36.9 ns        |
| `simd_manhattan_distance/scalar/1024`                             | 593.7 ns        | 663.6 ns       |
| `simd_manhattan_distance/scalar/1536`                             | 896.2 ns        | 1.01 µs        |
| `simd_manhattan_distance/scalar/384`                              | 212.5 ns        | 227.7 ns       |
| `simd_manhattan_distance/scalar/768`                              | 440.1 ns        | 491.4 ns       |
| `simd_manhattan_distance/simd/1024`                               | 85.3 ns         | 59.9 ns        |
| `simd_manhattan_distance/simd/1536`                               | 126.3 ns        | 69.5 ns        |
| `simd_manhattan_distance/simd/384`                                | 36.8 ns         | 23.0 ns        |
| `simd_manhattan_distance/simd/768`                                | 66.8 ns         | 44.5 ns        |
| `simd_normalize/scalar/1024`                                      | 686.0 ns        | 783.2 ns       |
| `simd_normalize/scalar/1536`                                      | 1.02 µs         | 1.17 µs        |
| `simd_normalize/scalar/384`                                       | 268.8 ns        | 294.9 ns       |
| `simd_normalize/scalar/768`                                       | 515.9 ns        | 583.2 ns       |
| `simd_normalize/simd/1024`                                        | 167.5 ns        | 190.5 ns       |
| `simd_normalize/simd/1536`                                        | 246.0 ns        | 281.8 ns       |
| `simd_normalize/simd/384`                                         | 75.7 ns         | 65.9 ns        |
| `simd_normalize/simd/768`                                         | 129.2 ns        | 152.6 ns       |
| `simd_normalized_cosine_fast_path/cosine_full/1024`               | 129.6 ns        | 60.6 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/384`                | 58.2 ns         | 29.9 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/768`                | 99.6 ns         | 55.4 ns        |
| `simd_normalized_cosine_fast_path/dot_product/1024`               | 77.3 ns         | 38.8 ns        |
| `simd_normalized_cosine_fast_path/dot_product/384`                | 32.6 ns         | 22.0 ns        |
| `simd_normalized_cosine_fast_path/dot_product/768`                | 56.8 ns         | 42.6 ns        |
| `simd_prepared_query_normalized_cosine/dot_product_loop/1024`     | 96.32 µs        | 138.96 µs      |
| `simd_prepared_query_normalized_cosine/dot_product_loop/384`      | 38.40 µs        | 31.79 µs       |
| `simd_prepared_query_normalized_cosine/dot_product_loop/768`      | 72.65 µs        | 98.88 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/1024` | 145.66 µs       | 150.04 µs      |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/384`  | 60.92 µs        | 36.32 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/768`  | 106.71 µs       | 103.80 µs      |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/1024`   | 147.48 µs       | 143.46 µs      |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/384`    | 60.75 µs        | 36.51 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/768`    | 108.08 µs       | 103.61 µs      |
| `simd_query_batch_dot_product/pair_loop/128d_16c`                 | 218.3 ns        | 128.5 ns       |
| `simd_query_batch_dot_product/pair_loop/128d_256c`                | 3.89 µs         | 2.63 µs        |
| `simd_query_batch_dot_product/pair_loop/128d_4c`                  | 67.4 ns         | 42.0 ns        |
| `simd_query_batch_dot_product/pair_loop/128d_64c`                 | 865.6 ns        | 494.0 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_16c`                 | 495.8 ns        | 372.5 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_256c`                | 9.88 µs         | 7.69 µs        |
| `simd_query_batch_dot_product/pair_loop/384d_4c`                  | 135.2 ns        | 99.8 ns        |
| `simd_query_batch_dot_product/pair_loop/384d_64c`                 | 2.41 µs         | 1.92 µs        |
| `simd_query_batch_dot_product/pair_loop/768d_16c`                 | 971.6 ns        | 912.9 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_256c`                | 22.08 µs        | 14.64 µs       |
| `simd_query_batch_dot_product/pair_loop/768d_4c`                  | 234.5 ns        | 176.5 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_64c`                 | 4.58 µs         | 3.64 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_16c`                | 138.3 ns        | 101.9 ns       |
| `simd_query_batch_dot_product/simd_batch/128d_256c`               | 3.03 µs         | 2.09 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_4c`                 | 46.2 ns         | 36.4 ns        |
| `simd_query_batch_dot_product/simd_batch/128d_64c`                | 532.2 ns        | 388.5 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_16c`                | 326.2 ns        | 212.6 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_256c`               | 9.02 µs         | 5.75 µs        |
| `simd_query_batch_dot_product/simd_batch/384d_4c`                 | 93.5 ns         | 61.9 ns        |
| `simd_query_batch_dot_product/simd_batch/384d_64c`                | 2.09 µs         | 1.43 µs        |
| `simd_query_batch_dot_product/simd_batch/768d_16c`                | 678.1 ns        | 622.2 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_256c`               | 18.53 µs        | 11.22 µs       |
| `simd_query_batch_dot_product/simd_batch/768d_4c`                 | 164.9 ns        | 112.2 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_64c`                | 3.85 µs         | 2.81 µs        |
| `simd_squared_euclidean_fast_path/euclidean_full/1024`            | 116.7 ns        | 45.5 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/384`             | 49.1 ns         | 18.9 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/768`             | 89.8 ns         | 43.1 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/1024`         | 113.6 ns        | 41.7 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/384`          | 45.8 ns         | 17.4 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/768`          | 86.6 ns         | 42.5 ns        |
| `simd_throughput_384/cosine_similarity`                           | 58.3 ns         | 27.7 ns        |
| `simd_throughput_384/dot_product`                                 | 31.8 ns         | 15.7 ns        |
| `simd_throughput_384/euclidean_distance`                          | 49.1 ns         | 18.8 ns        |
| `simd_throughput_384/normalize`                                   | 108.3 ns        | 124.0 ns       |
| `tier_prepared_batch_sizes/int4_batch_prepared/10`                | 516.2 ns        | 1.40 µs        |
| `tier_prepared_batch_sizes/int4_batch_prepared/100`               | 5.02 µs         | 13.78 µs       |
| `tier_prepared_batch_sizes/int4_batch_prepared/1000`              | 50.23 µs        | 137.63 µs      |
| `tier_prepared_batch_sizes/int4_query_per_call/10`                | 9.34 µs         | 7.91 µs        |
| `tier_prepared_batch_sizes/int4_query_per_call/100`               | 93.18 µs        | 78.93 µs       |
| `tier_prepared_batch_sizes/int4_query_per_call/1000`              | 931.99 µs       | 787.65 µs      |
| `tier_prepared_batch_sizes/int8_batch_prepared/10`                | 172.0 ns        | 212.8 ns       |
| `tier_prepared_batch_sizes/int8_batch_prepared/100`               | 1.54 µs         | 1.95 µs        |
| `tier_prepared_batch_sizes/int8_batch_prepared/1000`              | 15.40 µs        | 19.28 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/10`                | 9.75 µs         | 5.97 µs        |
| `tier_prepared_batch_sizes/int8_query_per_call/100`               | 97.14 µs        | 59.76 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/1000`              | 972.62 µs       | 597.26 µs      |
| `tier_prepared_query/binary_query_once_1000`                      | 8.01 µs         | 48.30 µs       |
| `tier_prepared_query/binary_query_per_call_1000`                  | 1.09 ms         | 379.85 µs      |
| `tier_prepared_query/int4_query_once_1000`                        | 52.17 µs        | 138.94 µs      |
| `tier_prepared_query/int4_query_per_call_1000`                    | 943.55 µs       | 785.73 µs      |
| `tier_prepared_query/int8_query_once_1000`                        | 18.30 µs        | 19.75 µs       |
| `tier_prepared_query/int8_query_per_call_1000`                    | 950.49 µs       | 568.32 µs      |

</details>

<!-- generated:benchmarks:end -->
