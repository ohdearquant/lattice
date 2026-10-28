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
  - `native.rs`: Native, pure-Rust embedding service backed by `lattice-inference`.
  - `native_pooling_tests.rs`
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
- test (14): `drift_api`, `embed_cli_image_args`, `embed_drift_baseline`, `embed_drift_cli`, `embed_parity_vs_hf`, `encoder_consumer`, `family_consumer`, `inference_dot_product_parity`, `integration`, `simd_edge_cases`, `simd_precision`, `token_report`, `tokenizer_parity_e2e`, `vision_single_file_real_model`
- bench (5): `embeddings`, `simd_bench`, `simd_opt_bench`, `simd`, `simsimd_comparison`

<!-- generated:layout:end -->

<!-- generated:stats:begin -->

### Statistics

Rust lines by kind (code, comment, blank; doc comments count as comment), test functions (`#[test]` and `#[<runtime>::test]` attributes), items declared `pub` (not `pub(crate)`; reachability through private modules is not resolved) and occurrences of the `unsafe` keyword outside comments and strings.

| Area        | Files | Code  | Comment | Blank | Tests | `pub` items | `unsafe` |
| ----------- | ----- | ----- | ------- | ----- | ----- | ----------- | -------- |
| `src/`      | 49    | 17637 | 2517    | 2421  | 418   | 303         | 119      |
| `tests/`    | 16    | 2896  | 498     | 487   | 111   | 0           | 1        |
| `benches/`  | 5     | 1616  | 286     | 366   | 0     | 0           | 0        |
| `examples/` | 4     | 342   | 79      | 64    | 0     | 0           | 0        |
| total       | 74    | 22491 | 3380    | 3338  | 529   | 303         | 120      |

Other tracked files (the README excluded):

| Extension | Files | Size      |
| --------- | ----- | --------- |
| `.json`   | 10    | 455.4 KiB |
| `.md`     | 11    | 169.4 KiB |
| `.mjs`    | 2     | 25.1 KiB  |
| `.toml`   | 2     | 10.5 KiB  |
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
- `download` (default): First-use model downloads for the native `embed` binary (its `--download-only` flag and the implicit fetch inside `BertModel::from_pretrained`). Enables `native`, `lattice-inference/download`.
- `image` (default): Enables `native`, `lattice-inference/image`.
- `local`: Deprecated no-op: gated a bench module for a service removed long ago, but the feature is part of the published 0.4.x surface (removing it would break manifests that list it). Enables nothing.
- `metal-gpu`: Enables `lattice-inference/metal-gpu`.
- `native` (default): Enables `dep:lattice-inference`, `dep:lattice-compute`, `dep:lattice-inference-core`, `dep:lattice-arch`, `dep:lattice-tokenizer`, `dep:lattice-weights`.
- `wasm`: wasm-bindgen JS bindings (`LatticeEmbedder`) for building a browser-loadable `.wasm` module. Enables `native`, `dep:wasm-bindgen`, `dep:console_error_panic_hook`.
- `x86-gemm`: Enables `lattice-inference/x86-gemm`.

<!-- generated:features:end -->

<!-- generated:dependencies:begin -->

### Workspace dependencies

- Depends on: `lattice-arch` (optional, enabled by `native`), `lattice-compute` (optional, enabled by `native`), `lattice-inference` (optional, enabled by `download`, `image`, `metal-gpu`, `native`, `x86-gemm`; no default features; features `std`, `f16`), `lattice-inference-core` (optional, enabled by `native`), `lattice-tokenizer` (optional, enabled by `native`), `lattice-weights` (optional, enabled by `native`; features `f16`)
- Dev-depends on: `lattice-inference` (no default features; features `std`, `test-utils`), `lattice-tokenizer`, `lattice-transport`
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

- `aarch64-linux`: commit `9487ae701021`, recorded 2026-10-26T14:29:30+00:00
- `x86_64-linux`: commit `9487ae701021`, recorded 2026-10-26T14:30:15+00:00

<details>
<summary>255 benchmarks</summary>

| Benchmark                                                         | `aarch64-linux` | `x86_64-linux` |
| ----------------------------------------------------------------- | --------------- | -------------- |
| `binary_cosine_distance/binary/1024`                              | 10.9 ns         | 97.4 ns        |
| `binary_cosine_distance/binary/1536`                              | 15.7 ns         | 140.7 ns       |
| `binary_cosine_distance/binary/384`                               | 5.5 ns          | 43.8 ns        |
| `binary_cosine_distance/binary/768`                               | 9.2 ns          | 74.7 ns        |
| `binary_cosine_distance/float32_simd/1024`                        | 129.2 ns        | 69.9 ns        |
| `binary_cosine_distance/float32_simd/1536`                        | 186.7 ns        | 103.2 ns       |
| `binary_cosine_distance/float32_simd/384`                         | 59.7 ns         | 39.7 ns        |
| `binary_cosine_distance/float32_simd/768`                         | 101.1 ns        | 55.2 ns        |
| `cached_embedding_delegate/native_wrong_model_disabled_64x4096`   | 233.8 ns        | 206.0 ns       |
| `int4_cosine_distance/float32_simd/1024`                          | 129.1 ns        | 74.9 ns        |
| `int4_cosine_distance/float32_simd/1536`                          | 186.8 ns        | 101.1 ns       |
| `int4_cosine_distance/float32_simd/384`                           | 59.8 ns         | 40.6 ns        |
| `int4_cosine_distance/float32_simd/768`                           | 101.1 ns        | 66.7 ns        |
| `int4_cosine_distance/int4/1024`                                  | 124.1 ns        | 283.4 ns       |
| `int4_cosine_distance/int4/1536`                                  | 181.2 ns        | 407.7 ns       |
| `int4_cosine_distance/int4/384`                                   | 53.3 ns         | 115.6 ns       |
| `int4_cosine_distance/int4/768`                                   | 96.3 ns         | 217.2 ns       |
| `int8_batch_cosine/float32_simd/10`                               | 573.5 ns        | 340.1 ns       |
| `int8_batch_cosine/float32_simd/100`                              | 6.87 µs         | 3.64 µs        |
| `int8_batch_cosine/float32_simd/1000`                             | 86.06 µs        | 49.23 µs       |
| `int8_batch_cosine/int8_loop/10`                                  | 162.8 ns        | 159.6 ns       |
| `int8_batch_cosine/int8_loop/100`                                 | 1.72 µs         | 1.59 µs        |
| `int8_batch_cosine/int8_loop/1000`                                | 20.59 µs        | 16.99 µs       |
| `int8_prepared_dot_product/per_call/1024`                         | 2.45 µs         | 1.64 µs        |
| `int8_prepared_dot_product/per_call/127`                          | 360.2 ns        | 267.9 ns       |
| `int8_prepared_dot_product/per_call/128`                          | 337.7 ns        | 251.0 ns       |
| `int8_prepared_dot_product/per_call/129`                          | 344.2 ns        | 247.8 ns       |
| `int8_prepared_dot_product/per_call/384`                          | 960.1 ns        | 641.9 ns       |
| `int8_prepared_dot_product/per_call/768`                          | 1.85 µs         | 1.22 µs        |
| `int8_prepared_dot_product/prepared/1024`                         | 30.9 ns         | 25.5 ns        |
| `int8_prepared_dot_product/prepared/127`                          | 12.5 ns         | 15.7 ns        |
| `int8_prepared_dot_product/prepared/128`                          | 8.6 ns          | 11.2 ns        |
| `int8_prepared_dot_product/prepared/129`                          | 9.1 ns          | 11.8 ns        |
| `int8_prepared_dot_product/prepared/384`                          | 14.3 ns         | 15.7 ns        |
| `int8_prepared_dot_product/prepared/768`                          | 24.7 ns         | 20.9 ns        |
| `int8_quantization/quantize/1024`                                 | 2.41 µs         | 1.61 µs        |
| `int8_quantization/quantize/1536`                                 | 3.60 µs         | 2.43 µs        |
| `int8_quantization/quantize/384`                                  | 935.2 ns        | 629.5 ns       |
| `int8_quantization/quantize/768`                                  | 1.83 µs         | 1.20 µs        |
| `int8_raw_dot_product/dot_product_i8/1024`                        | 30.4 ns         | 26.2 ns        |
| `int8_raw_dot_product/dot_product_i8/127`                         | 12.0 ns         | 14.6 ns        |
| `int8_raw_dot_product/dot_product_i8/128`                         | 8.1 ns          | 11.0 ns        |
| `int8_raw_dot_product/dot_product_i8/129`                         | 8.5 ns          | 11.6 ns        |
| `int8_raw_dot_product/dot_product_i8/384`                         | 14.0 ns         | 16.3 ns        |
| `int8_raw_dot_product/dot_product_i8/768`                         | 24.1 ns         | 19.3 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/1024`                    | 28.0 ns         | 23.3 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/127`                     | 10.1 ns         | 13.4 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/128`                     | 6.4 ns          | 9.7 ns         |
| `int8_raw_dot_product/dot_product_i8_raw/129`                     | 7.1 ns          | 9.5 ns         |
| `int8_raw_dot_product/dot_product_i8_raw/384`                     | 12.4 ns         | 15.0 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/768`                     | 21.9 ns         | 17.8 ns        |
| `int8_vs_float32_cosine/float32_simd/1024`                        | 129.6 ns        | 77.2 ns        |
| `int8_vs_float32_cosine/float32_simd/1536`                        | 185.8 ns        | 96.5 ns        |
| `int8_vs_float32_cosine/float32_simd/384`                         | 58.9 ns         | 41.5 ns        |
| `int8_vs_float32_cosine/float32_simd/768`                         | 100.3 ns        | 59.4 ns        |
| `int8_vs_float32_cosine/int8/1024`                                | 33.1 ns         | 23.1 ns        |
| `int8_vs_float32_cosine/int8/1536`                                | 45.9 ns         | 36.9 ns        |
| `int8_vs_float32_cosine/int8/384`                                 | 15.7 ns         | 17.0 ns        |
| `int8_vs_float32_cosine/int8/768`                                 | 27.0 ns         | 19.3 ns        |
| `memory_size/search_1000_float32`                                 | 57.25 µs        | 37.60 µs       |
| `memory_size/search_1000_int8`                                    | 14.93 µs        | 16.45 µs       |
| `simd_batch_cosine/scalar_loop/10`                                | 6.38 µs         | 9.48 µs        |
| `simd_batch_cosine/scalar_loop/100`                               | 63.90 µs        | 95.89 µs       |
| `simd_batch_cosine/scalar_loop/1000`                              | 645.07 µs       | 968.85 µs      |
| `simd_batch_cosine/simd_batch/10`                                 | 575.8 ns        | 371.3 ns       |
| `simd_batch_cosine/simd_batch/100`                                | 6.92 µs         | 4.10 µs        |
| `simd_batch_cosine/simd_batch/1000`                               | 86.06 µs        | 53.22 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_1000c`    | 145.57 µs       | 71.79 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_16c`      | 2.09 µs         | 1.09 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_256c`     | 35.27 µs        | 18.38 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_4c`       | 522.2 ns        | 277.9 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_64c`      | 8.27 µs         | 4.33 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_1000c`     | 58.70 µs        | 34.43 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_16c`       | 922.3 ns        | 531.1 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_256c`      | 15.28 µs        | 8.47 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_4c`        | 237.7 ns        | 142.9 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_64c`       | 3.73 µs         | 2.13 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_1000c`     | 106.99 µs       | 52.16 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_16c`       | 1.62 µs         | 800.9 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_256c`      | 28.31 µs        | 13.43 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_4c`        | 407.8 ns        | 206.4 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_64c`       | 6.46 µs         | 3.17 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_1000c`   | 146.65 µs       | 69.17 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_16c`     | 2.09 µs         | 1.07 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_256c`    | 35.22 µs        | 17.70 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_4c`      | 519.5 ns        | 270.8 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_64c`     | 8.24 µs         | 4.23 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_1000c`    | 57.61 µs        | 31.88 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_16c`      | 916.0 ns        | 512.6 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_256c`     | 15.24 µs        | 7.87 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_4c`       | 235.5 ns        | 134.6 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_64c`      | 3.70 µs         | 1.98 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_1000c`    | 107.25 µs       | 50.65 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_16c`      | 1.61 µs         | 784.4 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_256c`     | 28.52 µs        | 13.01 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_4c`       | 405.1 ns        | 203.6 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_64c`      | 6.43 µs         | 3.10 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_1000c` | 145.31 µs       | 67.14 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_16c`   | 2.08 µs         | 1.04 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_256c`  | 35.59 µs        | 17.09 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_4c`    | 519.8 ns        | 284.7 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_64c`   | 8.23 µs         | 4.07 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_1000c`  | 58.10 µs        | 39.63 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_16c`    | 920.5 ns        | 585.6 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_256c`   | 15.35 µs        | 9.34 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_4c`     | 237.5 ns        | 153.4 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_64c`    | 3.69 µs         | 2.43 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_1000c`  | 106.28 µs       | 62.45 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_16c`    | 1.60 µs         | 944.9 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_256c`   | 28.20 µs        | 15.66 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_4c`     | 405.8 ns        | 245.9 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_64c`    | 6.40 µs         | 3.78 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_1000c`    | 96.04 µs        | 58.20 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_16c`      | 1.37 µs         | 869.7 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_256c`     | 26.52 µs        | 14.88 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_4c`       | 297.5 ns        | 234.5 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_64c`      | 6.17 µs         | 3.39 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_1000c`     | 37.98 µs        | 32.18 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_16c`       | 500.7 ns        | 470.2 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_256c`      | 10.31 µs        | 7.48 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_4c`        | 132.6 ns        | 131.0 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_64c`       | 2.42 µs         | 1.92 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_1000c`     | 72.58 µs        | 56.85 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_16c`       | 965.5 ns        | 883.7 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_256c`      | 21.39 µs        | 14.32 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_4c`        | 232.3 ns        | 214.8 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_64c`       | 4.60 µs         | 3.40 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_1000c`       | 145.49 µs       | 65.56 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_16c`         | 2.07 µs         | 1.02 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_256c`        | 35.47 µs        | 16.63 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_4c`          | 516.6 ns        | 270.5 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_64c`         | 8.19 µs         | 3.99 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/384d_1000c`        | 57.61 µs        | 37.00 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_16c`          | 914.9 ns        | 540.3 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_256c`         | 15.22 µs        | 8.85 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/384d_4c`           | 235.3 ns        | 144.1 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_64c`          | 3.66 µs         | 2.25 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_1000c`        | 106.72 µs       | 59.92 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_16c`          | 1.60 µs         | 905.3 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_256c`         | 28.53 µs        | 14.93 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_4c`           | 403.7 ns        | 240.5 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_64c`          | 6.37 µs         | 3.68 µs        |
| `simd_batch_dot_product/scalar_loop/10`                           | 2.14 µs         | 3.04 µs        |
| `simd_batch_dot_product/scalar_loop/100`                          | 21.36 µs        | 31.33 µs       |
| `simd_batch_dot_product/scalar_loop/1000`                         | 218.29 µs       | 316.20 µs      |
| `simd_batch_dot_product/simd_batch/10`                            | 333.7 ns        | 260.7 ns       |
| `simd_batch_dot_product/simd_batch/100`                           | 5.71 µs         | 3.26 µs        |
| `simd_batch_dot_product/simd_batch/1000`                          | 88.32 µs        | 43.76 µs       |
| `simd_cosine_similarity/scalar/1024`                              | 1.78 µs         | 2.74 µs        |
| `simd_cosine_similarity/scalar/1536`                              | 2.69 µs         | 4.06 µs        |
| `simd_cosine_similarity/scalar/384`                               | 647.8 ns        | 933.0 ns       |
| `simd_cosine_similarity/scalar/768`                               | 1.33 µs         | 1.97 µs        |
| `simd_cosine_similarity/simd/1024`                                | 129.6 ns        | 75.5 ns        |
| `simd_cosine_similarity/simd/1536`                                | 186.0 ns        | 106.8 ns       |
| `simd_cosine_similarity/simd/384`                                 | 58.7 ns         | 40.0 ns        |
| `simd_cosine_similarity/simd/768`                                 | 101.2 ns        | 62.8 ns        |
| `simd_dot_product/scalar/1024`                                    | 593.1 ns        | 882.5 ns       |
| `simd_dot_product/scalar/1536`                                    | 892.3 ns        | 1.39 µs        |
| `simd_dot_product/scalar/384`                                     | 213.2 ns        | 319.3 ns       |
| `simd_dot_product/scalar/768`                                     | 439.6 ns        | 646.6 ns       |
| `simd_dot_product/simd/1024`                                      | 73.4 ns         | 48.6 ns        |
| `simd_dot_product/simd/1536`                                      | 106.8 ns        | 84.1 ns        |
| `simd_dot_product/simd/384`                                       | 31.5 ns         | 21.5 ns        |
| `simd_dot_product/simd/768`                                       | 55.4 ns         | 35.9 ns        |
| `simd_euclidean_distance/scalar/1024`                             | 595.5 ns        | 871.5 ns       |
| `simd_euclidean_distance/scalar/1536`                             | 895.9 ns        | 1.29 µs        |
| `simd_euclidean_distance/scalar/384`                              | 216.8 ns        | 319.7 ns       |
| `simd_euclidean_distance/scalar/768`                              | 443.4 ns        | 638.2 ns       |
| `simd_euclidean_distance/simd/1024`                               | 116.5 ns        | 65.5 ns        |
| `simd_euclidean_distance/simd/1536`                               | 170.9 ns        | 87.3 ns        |
| `simd_euclidean_distance/simd/384`                                | 49.1 ns         | 32.1 ns        |
| `simd_euclidean_distance/simd/768`                                | 89.7 ns         | 45.5 ns        |
| `simd_manhattan_distance/scalar/1024`                             | 592.2 ns        | 880.3 ns       |
| `simd_manhattan_distance/scalar/1536`                             | 891.4 ns        | 1.36 µs        |
| `simd_manhattan_distance/scalar/384`                              | 212.5 ns        | 309.0 ns       |
| `simd_manhattan_distance/scalar/768`                              | 438.7 ns        | 644.4 ns       |
| `simd_manhattan_distance/simd/1024`                               | 88.1 ns         | 55.2 ns        |
| `simd_manhattan_distance/simd/1536`                               | 127.3 ns        | 87.8 ns        |
| `simd_manhattan_distance/simd/384`                                | 36.7 ns         | 26.4 ns        |
| `simd_manhattan_distance/simd/768`                                | 66.6 ns         | 47.6 ns        |
| `simd_normalize/scalar/1024`                                      | 688.4 ns        | 1.04 µs        |
| `simd_normalize/scalar/1536`                                      | 1.03 µs         | 1.53 µs        |
| `simd_normalize/scalar/384`                                       | 266.4 ns        | 396.9 ns       |
| `simd_normalize/scalar/768`                                       | 519.6 ns        | 781.8 ns       |
| `simd_normalize/simd/1024`                                        | 158.4 ns        | 149.0 ns       |
| `simd_normalize/simd/1536`                                        | 231.0 ns        | 208.5 ns       |
| `simd_normalize/simd/384`                                         | 74.5 ns         | 71.9 ns        |
| `simd_normalize/simd/768`                                         | 126.5 ns        | 115.6 ns       |
| `simd_normalized_cosine_fast_path/cosine_full/1024`               | 128.8 ns        | 74.1 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/384`                | 58.8 ns         | 38.4 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/768`                | 100.0 ns        | 57.8 ns        |
| `simd_normalized_cosine_fast_path/dot_product/1024`               | 69.9 ns         | 55.2 ns        |
| `simd_normalized_cosine_fast_path/dot_product/384`                | 31.5 ns         | 22.3 ns        |
| `simd_normalized_cosine_fast_path/dot_product/768`                | 55.5 ns         | 42.5 ns        |
| `simd_prepared_query_normalized_cosine/dot_product_loop/1024`     | 95.83 µs        | 63.57 µs       |
| `simd_prepared_query_normalized_cosine/dot_product_loop/384`      | 40.05 µs        | 31.67 µs       |
| `simd_prepared_query_normalized_cosine/dot_product_loop/768`      | 72.50 µs        | 53.76 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/1024` | 146.35 µs       | 80.29 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/384`  | 60.69 µs        | 41.54 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/768`  | 107.70 µs       | 59.26 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/1024`   | 147.40 µs       | 82.02 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/384`    | 60.79 µs        | 38.68 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/768`    | 108.68 µs       | 61.12 µs       |
| `simd_query_batch_dot_product/pair_loop/128d_16c`                 | 221.1 ns        | 207.6 ns       |
| `simd_query_batch_dot_product/pair_loop/128d_256c`                | 3.94 µs         | 3.05 µs        |
| `simd_query_batch_dot_product/pair_loop/128d_4c`                  | 66.8 ns         | 68.2 ns        |
| `simd_query_batch_dot_product/pair_loop/128d_64c`                 | 866.4 ns        | 789.6 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_16c`                 | 502.1 ns        | 416.5 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_256c`                | 10.06 µs        | 6.91 µs        |
| `simd_query_batch_dot_product/pair_loop/384d_4c`                  | 133.9 ns        | 114.3 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_64c`                 | 2.42 µs         | 1.68 µs        |
| `simd_query_batch_dot_product/pair_loop/768d_16c`                 | 957.5 ns        | 869.3 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_256c`                | 21.52 µs        | 14.19 µs       |
| `simd_query_batch_dot_product/pair_loop/768d_4c`                  | 231.8 ns        | 226.5 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_64c`                 | 4.59 µs         | 3.41 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_16c`                | 136.6 ns        | 115.5 ns       |
| `simd_query_batch_dot_product/simd_batch/128d_256c`               | 3.04 µs         | 1.83 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_4c`                 | 45.9 ns         | 40.0 ns        |
| `simd_query_batch_dot_product/simd_batch/128d_64c`                | 534.2 ns        | 428.1 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_16c`                | 325.8 ns        | 226.5 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_256c`               | 9.11 µs         | 4.25 µs        |
| `simd_query_batch_dot_product/simd_batch/384d_4c`                 | 93.2 ns         | 67.9 ns        |
| `simd_query_batch_dot_product/simd_batch/384d_64c`                | 2.17 µs         | 1.04 µs        |
| `simd_query_batch_dot_product/simd_batch/768d_16c`                | 626.5 ns        | 507.1 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_256c`               | 17.76 µs        | 8.33 µs        |
| `simd_query_batch_dot_product/simd_batch/768d_4c`                 | 164.0 ns        | 132.6 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_64c`                | 3.85 µs         | 2.00 µs        |
| `simd_squared_euclidean_fast_path/euclidean_full/1024`            | 116.7 ns        | 60.4 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/384`             | 49.0 ns         | 24.6 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/768`             | 89.5 ns         | 48.0 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/1024`         | 113.5 ns        | 57.5 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/384`          | 45.9 ns         | 22.8 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/768`          | 86.4 ns         | 44.3 ns        |
| `simd_throughput_384/cosine_similarity`                           | 58.3 ns         | 42.8 ns        |
| `simd_throughput_384/dot_product`                                 | 31.5 ns         | 30.9 ns        |
| `simd_throughput_384/euclidean_distance`                          | 49.0 ns         | 36.0 ns        |
| `simd_throughput_384/normalize`                                   | 120.4 ns        | 109.2 ns       |
| `tier_prepared_batch_sizes/int4_batch_prepared/10`                | 518.4 ns        | 1.05 µs        |
| `tier_prepared_batch_sizes/int4_batch_prepared/100`               | 5.14 µs         | 10.29 µs       |
| `tier_prepared_batch_sizes/int4_batch_prepared/1000`              | 50.08 µs        | 102.75 µs      |
| `tier_prepared_batch_sizes/int4_query_per_call/10`                | 9.37 µs         | 6.29 µs        |
| `tier_prepared_batch_sizes/int4_query_per_call/100`               | 93.42 µs        | 62.97 µs       |
| `tier_prepared_batch_sizes/int4_query_per_call/1000`              | 934.33 µs       | 628.65 µs      |
| `tier_prepared_batch_sizes/int8_batch_prepared/10`                | 168.2 ns        | 188.8 ns       |
| `tier_prepared_batch_sizes/int8_batch_prepared/100`               | 1.50 µs         | 1.52 µs        |
| `tier_prepared_batch_sizes/int8_batch_prepared/1000`              | 15.21 µs        | 14.98 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/10`                | 9.72 µs         | 5.84 µs        |
| `tier_prepared_batch_sizes/int8_query_per_call/100`               | 96.74 µs        | 58.69 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/1000`              | 968.77 µs       | 583.11 µs      |
| `tier_prepared_query/binary_query_once_1000`                      | 8.01 µs         | 44.22 µs       |
| `tier_prepared_query/binary_query_per_call_1000`                  | 1.10 ms         | 447.68 µs      |
| `tier_prepared_query/int4_query_once_1000`                        | 52.44 µs        | 116.42 µs      |
| `tier_prepared_query/int4_query_per_call_1000`                    | 946.10 µs       | 695.19 µs      |
| `tier_prepared_query/int8_query_once_1000`                        | 18.29 µs        | 17.88 µs       |
| `tier_prepared_query/int8_query_per_call_1000`                    | 947.80 µs       | 591.63 µs      |

</details>

<!-- generated:benchmarks:end -->
