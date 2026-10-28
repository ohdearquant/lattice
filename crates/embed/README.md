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
<summary><code>src/</code>: 44 files</summary>

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
  - `native/`
    - `identity.rs`: Declared BGE-small native output identity bound to the successful load.
    - `identity/`
      - `tests.rs`
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
| `src/`      | 44    | 15619 | 2435    | 2096  | 371   | 294         | 119      |
| `tests/`    | 14    | 2773  | 498     | 469   | 107   | 0           | 1        |
| `benches/`  | 5     | 1616  | 286     | 366   | 0     | 0           | 0        |
| `examples/` | 4     | 342   | 79      | 64    | 0     | 0           | 0        |
| total       | 67    | 20350 | 3298    | 2995  | 478   | 294         | 120      |

Other tracked files (the README excluded):

| Extension | Files | Size      |
| --------- | ----- | --------- |
| `.json`   | 10    | 455.4 KiB |
| `.md`     | 11    | 168.6 KiB |
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
- `download` (default): First-use model downloads for the native `embed` binary (its `--download-only` flag and the implicit fetch inside `BertModel::from_pretrained`). Enables `native`, `lattice-model-bert/download`, `lattice-model-qwen3-embedding/download`.
- `image` (default): Feature forwards. Enables `native`, `lattice-model-qwen35/image`, `lattice-model-embeddinggemma2/image`.
- `local`: Deprecated no-op: gated a bench module for a service removed long ago, but the feature is part of the published 0.4.x surface (removing it would break manifests that list it). Enables nothing.
- `metal-gpu`: Enables `lattice-model-qwen3-embedding/metal-gpu`, `lattice-model-qwen35/metal-gpu`, `lattice-model-embeddinggemma2/metal-gpu`.
- `native` (default): Enables `dep:lattice-model-bert`, `dep:lattice-model-qwen3-embedding`, `dep:lattice-model-qwen35`, `dep:lattice-model-embeddinggemma2`, `dep:lattice-compute`, `dep:lattice-inference-core`, `dep:lattice-arch`, `dep:lattice-tokenizer`, `dep:lattice-weights`.
- `wasm`: wasm-bindgen JS bindings (`LatticeEmbedder`) for building a browser-loadable `.wasm` module. Enables `native`, `dep:wasm-bindgen`, `dep:console_error_panic_hook`.
- `x86-gemm`: Enables `lattice-compute/x86-gemm`, `lattice-model-bert/x86-gemm`, `lattice-model-qwen3-embedding/x86-gemm`.

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
<summary>25 ADRs</summary>

- [ADR-012: Runtime SIMD Detection Strategy](../../docs/adr/ADR-012-simd-strategy.md)
- [ADR-013: lattice-embed as SIMD Foundation Layer](../../docs/adr/ADR-013-simd-foundation.md)
- [ADR-014: Embedding Service](../../docs/adr/ADR-014-embedding-service.md)
- [ADR-015: Sharded LRU Embedding Cache](../../docs/adr/ADR-015-embedding-cache.md)
- [ADR-016: Embedding Model Variants](../../docs/adr/ADR-016-model-variants.md)
- [ADR-017: NativeEmbeddingService Orchestration](../../docs/adr/ADR-017-native-embedding-service.md)
- [ADR-018: Quantized Vector Tiers](../../docs/adr/ADR-018-quantized-vectors.md)
- [ADR-019: Backfill and Migration Pipeline](../../docs/adr/ADR-019-backfill-pipeline.md)
- [ADR-047: Paged KV Cache with Prefix Reuse](../../docs/adr/ADR-047-paged-kv-cache.md)
- [ADR-049: Vision Encoder Integration (Qwen-VL Path)](../../docs/adr/ADR-049-vision-encoder.md)
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
- [ADR-104: Embedding Vector-Space Identity](../../docs/adr/ADR-104-embedding-vector-space-identity.md)
- [ADR-ALIGNMENT-AUDIT: ADR Set Classification as of 2026-06-30](../../docs/adr/ADR-ALIGNMENT-AUDIT.md)
- [Architecture Decision Records](../../docs/adr/INDEX.md)

</details>

<!-- generated:adrs:end -->

<!-- generated:benchmarks:begin -->

### Benchmarks

The newest Criterion point estimates `bench-update.yml` stored on the `perf-baselines` branch, from the `simd` target. They come from GitHub-hosted runners and serve as trend data between commits, not as a performance claim; a claim needs a paired A/B (`make bench-compare`).

- `aarch64-linux`: commit `b740f8325b5c`, recorded 2026-10-28T08:12:18+00:00
- `x86_64-linux`: commit `b740f8325b5c`, recorded 2026-10-28T08:14:23+00:00

<details>
<summary>255 benchmarks</summary>

| Benchmark                                                         | `aarch64-linux` | `x86_64-linux` |
| ----------------------------------------------------------------- | --------------- | -------------- |
| `binary_cosine_distance/binary/1024`                              | 10.9 ns         | 109.2 ns       |
| `binary_cosine_distance/binary/1536`                              | 15.8 ns         | 160.0 ns       |
| `binary_cosine_distance/binary/384`                               | 5.5 ns          | 45.7 ns        |
| `binary_cosine_distance/binary/768`                               | 9.2 ns          | 83.6 ns        |
| `binary_cosine_distance/float32_simd/1024`                        | 129.3 ns        | 67.6 ns        |
| `binary_cosine_distance/float32_simd/1536`                        | 186.8 ns        | 99.4 ns        |
| `binary_cosine_distance/float32_simd/384`                         | 59.7 ns         | 35.6 ns        |
| `binary_cosine_distance/float32_simd/768`                         | 101.2 ns        | 59.2 ns        |
| `cached_embedding_delegate/native_wrong_model_disabled_64x4096`   | 236.3 ns        | 233.1 ns       |
| `int4_cosine_distance/float32_simd/1024`                          | 129.2 ns        | 67.7 ns        |
| `int4_cosine_distance/float32_simd/1536`                          | 186.7 ns        | 98.6 ns        |
| `int4_cosine_distance/float32_simd/384`                           | 59.5 ns         | 41.8 ns        |
| `int4_cosine_distance/float32_simd/768`                           | 101.2 ns        | 57.7 ns        |
| `int4_cosine_distance/int4/1024`                                  | 123.8 ns        | 333.4 ns       |
| `int4_cosine_distance/int4/1536`                                  | 181.4 ns        | 491.0 ns       |
| `int4_cosine_distance/int4/384`                                   | 53.7 ns         | 134.8 ns       |
| `int4_cosine_distance/int4/768`                                   | 96.6 ns         | 253.7 ns       |
| `int8_batch_cosine/float32_simd/10`                               | 575.5 ns        | 363.2 ns       |
| `int8_batch_cosine/float32_simd/100`                              | 6.91 µs         | 5.23 µs        |
| `int8_batch_cosine/float32_simd/1000`                             | 88.96 µs        | 110.43 µs      |
| `int8_batch_cosine/int8_loop/10`                                  | 163.0 ns        | 195.4 ns       |
| `int8_batch_cosine/int8_loop/100`                                 | 1.74 µs         | 1.88 µs        |
| `int8_batch_cosine/int8_loop/1000`                                | 20.54 µs        | 18.59 µs       |
| `int8_prepared_dot_product/per_call/1024`                         | 2.45 µs         | 2.00 µs        |
| `int8_prepared_dot_product/per_call/127`                          | 356.0 ns        | 339.5 ns       |
| `int8_prepared_dot_product/per_call/128`                          | 338.6 ns        | 317.3 ns       |
| `int8_prepared_dot_product/per_call/129`                          | 352.4 ns        | 310.4 ns       |
| `int8_prepared_dot_product/per_call/384`                          | 948.0 ns        | 775.0 ns       |
| `int8_prepared_dot_product/per_call/768`                          | 1.85 µs         | 1.53 µs        |
| `int8_prepared_dot_product/prepared/1024`                         | 31.1 ns         | 20.5 ns        |
| `int8_prepared_dot_product/prepared/127`                          | 12.4 ns         | 17.8 ns        |
| `int8_prepared_dot_product/prepared/128`                          | 8.6 ns          | 12.6 ns        |
| `int8_prepared_dot_product/prepared/129`                          | 9.1 ns          | 13.0 ns        |
| `int8_prepared_dot_product/prepared/384`                          | 14.4 ns         | 17.0 ns        |
| `int8_prepared_dot_product/prepared/768`                          | 25.0 ns         | 19.0 ns        |
| `int8_quantization/quantize/1024`                                 | 2.41 µs         | 1.91 µs        |
| `int8_quantization/quantize/1536`                                 | 3.60 µs         | 2.83 µs        |
| `int8_quantization/quantize/384`                                  | 942.9 ns        | 760.2 ns       |
| `int8_quantization/quantize/768`                                  | 1.84 µs         | 1.46 µs        |
| `int8_raw_dot_product/dot_product_i8/1024`                        | 30.9 ns         | 20.9 ns        |
| `int8_raw_dot_product/dot_product_i8/127`                         | 11.8 ns         | 18.1 ns        |
| `int8_raw_dot_product/dot_product_i8/128`                         | 8.1 ns          | 12.7 ns        |
| `int8_raw_dot_product/dot_product_i8/129`                         | 8.5 ns          | 13.2 ns        |
| `int8_raw_dot_product/dot_product_i8/384`                         | 14.0 ns         | 16.9 ns        |
| `int8_raw_dot_product/dot_product_i8/768`                         | 24.7 ns         | 16.9 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/1024`                    | 28.2 ns         | 20.0 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/127`                     | 9.9 ns          | 15.4 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/128`                     | 6.5 ns          | 10.9 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/129`                     | 7.1 ns          | 11.0 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/384`                     | 12.4 ns         | 16.0 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/768`                     | 21.9 ns         | 15.6 ns        |
| `int8_vs_float32_cosine/float32_simd/1024`                        | 129.6 ns        | 70.4 ns        |
| `int8_vs_float32_cosine/float32_simd/1536`                        | 184.9 ns        | 100.6 ns       |
| `int8_vs_float32_cosine/float32_simd/384`                         | 58.0 ns         | 34.9 ns        |
| `int8_vs_float32_cosine/float32_simd/768`                         | 99.5 ns         | 73.2 ns        |
| `int8_vs_float32_cosine/int8/1024`                                | 33.0 ns         | 25.4 ns        |
| `int8_vs_float32_cosine/int8/1536`                                | 45.9 ns         | 34.2 ns        |
| `int8_vs_float32_cosine/int8/384`                                 | 16.0 ns         | 18.1 ns        |
| `int8_vs_float32_cosine/int8/768`                                 | 27.0 ns         | 18.9 ns        |
| `memory_size/search_1000_float32`                                 | 57.06 µs        | 48.36 µs       |
| `memory_size/search_1000_int8`                                    | 14.69 µs        | 18.46 µs       |
| `simd_batch_cosine/scalar_loop/10`                                | 6.39 µs         | 11.56 µs       |
| `simd_batch_cosine/scalar_loop/100`                               | 64.01 µs        | 115.42 µs      |
| `simd_batch_cosine/scalar_loop/1000`                              | 646.71 µs       | 1.18 ms        |
| `simd_batch_cosine/simd_batch/10`                                 | 577.5 ns        | 366.1 ns       |
| `simd_batch_cosine/simd_batch/100`                                | 6.91 µs         | 5.28 µs        |
| `simd_batch_cosine/simd_batch/1000`                               | 88.03 µs        | 109.02 µs      |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_1000c`    | 145.35 µs       | 151.80 µs      |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_16c`      | 2.10 µs         | 1.84 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_256c`     | 36.85 µs        | 30.44 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_4c`       | 522.3 ns        | 367.2 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_64c`      | 8.30 µs         | 7.31 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_1000c`     | 58.16 µs        | 53.71 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_16c`       | 922.6 ns        | 649.8 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_256c`      | 14.76 µs        | 11.79 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_4c`        | 237.9 ns        | 184.0 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_64c`       | 3.72 µs         | 2.94 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_1000c`     | 106.48 µs       | 116.16 µs      |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_16c`       | 1.61 µs         | 1.36 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_256c`      | 28.68 µs        | 22.28 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_4c`        | 407.9 ns        | 277.7 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_64c`       | 6.45 µs         | 5.54 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_1000c`   | 144.91 µs       | 150.56 µs      |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_16c`     | 2.11 µs         | 1.80 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_256c`    | 37.08 µs        | 29.88 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_4c`      | 520.0 ns        | 359.7 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_64c`     | 8.26 µs         | 7.22 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_1000c`    | 57.47 µs        | 52.27 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_16c`      | 915.3 ns        | 601.0 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_256c`     | 14.61 µs        | 11.72 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_4c`       | 234.9 ns        | 170.3 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_64c`      | 3.69 µs         | 2.92 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_1000c`    | 107.01 µs       | 114.03 µs      |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_16c`      | 1.60 µs         | 1.31 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_256c`     | 28.13 µs        | 21.68 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_4c`       | 404.9 ns        | 270.6 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_64c`      | 6.41 µs         | 5.38 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_1000c` | 147.70 µs       | 151.78 µs      |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_16c`   | 2.07 µs         | 1.84 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_256c`  | 38.62 µs        | 30.52 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_4c`    | 519.8 ns        | 340.1 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_64c`   | 8.31 µs         | 7.30 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_1000c`  | 57.90 µs        | 55.26 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_16c`    | 922.1 ns        | 659.1 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_256c`   | 14.72 µs        | 11.69 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_4c`     | 237.6 ns        | 184.2 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_64c`    | 3.71 µs         | 2.93 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_1000c`  | 105.56 µs       | 112.29 µs      |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_16c`    | 1.61 µs         | 1.35 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_256c`   | 27.88 µs        | 21.91 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_4c`     | 406.9 ns        | 280.1 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_64c`    | 6.42 µs         | 5.47 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_1000c`    | 98.61 µs        | 134.13 µs      |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_16c`      | 1.37 µs         | 1.52 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_256c`     | 29.36 µs        | 25.16 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_4c`       | 298.9 ns        | 276.2 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_64c`      | 6.12 µs         | 6.05 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_1000c`     | 38.27 µs        | 45.66 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_16c`       | 493.7 ns        | 494.8 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_256c`      | 9.60 µs         | 9.51 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_4c`        | 133.5 ns        | 134.4 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_64c`       | 2.45 µs         | 2.37 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_1000c`     | 73.21 µs        | 103.22 µs      |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_16c`       | 975.4 ns        | 1.13 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_256c`      | 20.95 µs        | 18.43 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_4c`        | 234.8 ns        | 228.2 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_64c`       | 4.64 µs         | 4.60 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_1000c`       | 146.41 µs       | 151.00 µs      |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_16c`         | 2.07 µs         | 1.80 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_256c`        | 38.18 µs        | 29.83 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_4c`          | 515.9 ns        | 335.9 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_64c`         | 8.19 µs         | 7.12 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/384d_1000c`        | 57.21 µs        | 52.18 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_16c`          | 914.0 ns        | 596.4 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_256c`         | 14.55 µs        | 11.67 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_4c`           | 234.8 ns        | 183.5 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_64c`          | 3.67 µs         | 2.91 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_1000c`        | 105.72 µs       | 107.97 µs      |
| `simd_batch_cosine_normalized_query/simd_batch/768d_16c`          | 1.60 µs         | 1.31 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_256c`         | 27.79 µs        | 21.31 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_4c`           | 403.6 ns        | 271.2 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_64c`          | 6.38 µs         | 5.37 µs        |
| `simd_batch_dot_product/scalar_loop/10`                           | 2.14 µs         | 3.86 µs        |
| `simd_batch_dot_product/scalar_loop/100`                          | 21.49 µs        | 38.64 µs       |
| `simd_batch_dot_product/scalar_loop/1000`                         | 220.27 µs       | 409.06 µs      |
| `simd_batch_dot_product/simd_batch/10`                            | 342.0 ns        | 288.3 ns       |
| `simd_batch_dot_product/simd_batch/100`                           | 5.58 µs         | 4.49 µs        |
| `simd_batch_dot_product/simd_batch/1000`                          | 84.69 µs        | 116.56 µs      |
| `simd_cosine_similarity/scalar/1024`                              | 1.77 µs         | 3.38 µs        |
| `simd_cosine_similarity/scalar/1536`                              | 2.68 µs         | 5.15 µs        |
| `simd_cosine_similarity/scalar/384`                               | 641.7 ns        | 1.16 µs        |
| `simd_cosine_similarity/scalar/768`                               | 1.32 µs         | 2.48 µs        |
| `simd_cosine_similarity/simd/1024`                                | 130.4 ns        | 72.0 ns        |
| `simd_cosine_similarity/simd/1536`                                | 185.2 ns        | 102.4 ns       |
| `simd_cosine_similarity/simd/384`                                 | 57.9 ns         | 38.6 ns        |
| `simd_cosine_similarity/simd/768`                                 | 99.9 ns         | 59.2 ns        |
| `simd_dot_product/scalar/1024`                                    | 593.0 ns        | 1.13 µs        |
| `simd_dot_product/scalar/1536`                                    | 892.7 ns        | 1.72 µs        |
| `simd_dot_product/scalar/384`                                     | 213.1 ns        | 390.1 ns       |
| `simd_dot_product/scalar/768`                                     | 439.6 ns        | 833.2 ns       |
| `simd_dot_product/simd/1024`                                      | 73.6 ns         | 72.9 ns        |
| `simd_dot_product/simd/1536`                                      | 107.0 ns        | 110.3 ns       |
| `simd_dot_product/simd/384`                                       | 31.6 ns         | 24.7 ns        |
| `simd_dot_product/simd/768`                                       | 58.2 ns         | 56.6 ns        |
| `simd_euclidean_distance/scalar/1024`                             | 595.6 ns        | 1.15 µs        |
| `simd_euclidean_distance/scalar/1536`                             | 895.9 ns        | 1.74 µs        |
| `simd_euclidean_distance/scalar/384`                              | 217.3 ns        | 414.3 ns       |
| `simd_euclidean_distance/scalar/768`                              | 445.5 ns        | 849.0 ns       |
| `simd_euclidean_distance/simd/1024`                               | 117.0 ns        | 70.0 ns        |
| `simd_euclidean_distance/simd/1536`                               | 171.1 ns        | 98.9 ns        |
| `simd_euclidean_distance/simd/384`                                | 49.1 ns         | 32.9 ns        |
| `simd_euclidean_distance/simd/768`                                | 89.7 ns         | 71.7 ns        |
| `simd_manhattan_distance/scalar/1024`                             | 591.3 ns        | 1.20 µs        |
| `simd_manhattan_distance/scalar/1536`                             | 892.1 ns        | 1.82 µs        |
| `simd_manhattan_distance/scalar/384`                              | 213.2 ns        | 420.3 ns       |
| `simd_manhattan_distance/scalar/768`                              | 439.3 ns        | 891.2 ns       |
| `simd_manhattan_distance/simd/1024`                               | 85.0 ns         | 89.4 ns        |
| `simd_manhattan_distance/simd/1536`                               | 126.1 ns        | 101.5 ns       |
| `simd_manhattan_distance/simd/384`                                | 36.7 ns         | 29.7 ns        |
| `simd_manhattan_distance/simd/768`                                | 66.7 ns         | 60.1 ns        |
| `simd_normalize/scalar/1024`                                      | 681.5 ns        | 1.28 µs        |
| `simd_normalize/scalar/1536`                                      | 1.02 µs         | 1.91 µs        |
| `simd_normalize/scalar/384`                                       | 267.4 ns        | 483.2 ns       |
| `simd_normalize/scalar/768`                                       | 513.9 ns        | 957.0 ns       |
| `simd_normalize/simd/1024`                                        | 155.7 ns        | 274.8 ns       |
| `simd_normalize/simd/1536`                                        | 236.2 ns        | 403.8 ns       |
| `simd_normalize/simd/384`                                         | 73.1 ns         | 91.1 ns        |
| `simd_normalize/simd/768`                                         | 124.6 ns        | 218.5 ns       |
| `simd_normalized_cosine_fast_path/cosine_full/1024`               | 129.6 ns        | 69.7 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/384`                | 57.8 ns         | 35.1 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/768`                | 99.6 ns         | 60.1 ns        |
| `simd_normalized_cosine_fast_path/dot_product/1024`               | 72.1 ns         | 55.4 ns        |
| `simd_normalized_cosine_fast_path/dot_product/384`                | 31.5 ns         | 17.9 ns        |
| `simd_normalized_cosine_fast_path/dot_product/768`                | 54.9 ns         | 44.3 ns        |
| `simd_prepared_query_normalized_cosine/dot_product_loop/1024`     | 95.98 µs        | 137.34 µs      |
| `simd_prepared_query_normalized_cosine/dot_product_loop/384`      | 38.27 µs        | 43.51 µs       |
| `simd_prepared_query_normalized_cosine/dot_product_loop/768`      | 72.58 µs        | 104.33 µs      |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/1024` | 147.62 µs       | 146.51 µs      |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/384`  | 60.71 µs        | 56.16 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/768`  | 108.32 µs       | 116.26 µs      |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/1024`   | 146.60 µs       | 152.63 µs      |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/384`    | 60.72 µs        | 56.33 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/768`    | 106.99 µs       | 115.20 µs      |
| `simd_query_batch_dot_product/pair_loop/128d_16c`                 | 218.0 ns        | 166.9 ns       |
| `simd_query_batch_dot_product/pair_loop/128d_256c`                | 3.79 µs         | 3.19 µs        |
| `simd_query_batch_dot_product/pair_loop/128d_4c`                  | 66.8 ns         | 53.6 ns        |
| `simd_query_batch_dot_product/pair_loop/128d_64c`                 | 825.7 ns        | 613.3 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_16c`                 | 488.7 ns        | 491.4 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_256c`                | 9.55 µs         | 9.58 µs        |
| `simd_query_batch_dot_product/pair_loop/384d_4c`                  | 134.0 ns        | 134.1 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_64c`                 | 2.46 µs         | 2.38 µs        |
| `simd_query_batch_dot_product/pair_loop/768d_16c`                 | 974.3 ns        | 1.11 µs        |
| `simd_query_batch_dot_product/pair_loop/768d_256c`                | 21.06 µs        | 18.32 µs       |
| `simd_query_batch_dot_product/pair_loop/768d_4c`                  | 232.6 ns        | 228.5 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_64c`                 | 4.60 µs         | 4.59 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_16c`                | 138.5 ns        | 111.3 ns       |
| `simd_query_batch_dot_product/simd_batch/128d_256c`               | 3.07 µs         | 2.19 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_4c`                 | 46.6 ns         | 37.5 ns        |
| `simd_query_batch_dot_product/simd_batch/128d_64c`                | 509.6 ns        | 413.4 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_16c`                | 326.5 ns        | 236.1 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_256c`               | 8.48 µs         | 5.74 µs        |
| `simd_query_batch_dot_product/simd_batch/384d_4c`                 | 94.0 ns         | 69.0 ns        |
| `simd_query_batch_dot_product/simd_batch/384d_64c`                | 2.14 µs         | 1.44 µs        |
| `simd_query_batch_dot_product/simd_batch/768d_16c`                | 627.2 ns        | 617.7 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_256c`               | 17.80 µs        | 11.61 µs       |
| `simd_query_batch_dot_product/simd_batch/768d_4c`                 | 165.0 ns        | 123.8 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_64c`                | 3.89 µs         | 2.91 µs        |
| `simd_squared_euclidean_fast_path/euclidean_full/1024`            | 116.8 ns        | 86.7 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/384`             | 49.0 ns         | 38.7 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/768`             | 89.5 ns         | 66.4 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/1024`         | 113.7 ns        | 79.1 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/384`          | 45.8 ns         | 38.6 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/768`          | 86.5 ns         | 59.8 ns        |
| `simd_throughput_384/cosine_similarity`                           | 58.2 ns         | 36.2 ns        |
| `simd_throughput_384/dot_product`                                 | 30.4 ns         | 25.5 ns        |
| `simd_throughput_384/euclidean_distance`                          | 48.9 ns         | 31.4 ns        |
| `simd_throughput_384/normalize`                                   | 119.3 ns        | 128.1 ns       |
| `tier_prepared_batch_sizes/int4_batch_prepared/10`                | 520.3 ns        | 1.34 µs        |
| `tier_prepared_batch_sizes/int4_batch_prepared/100`               | 5.03 µs         | 13.25 µs       |
| `tier_prepared_batch_sizes/int4_batch_prepared/1000`              | 50.26 µs        | 132.40 µs      |
| `tier_prepared_batch_sizes/int4_query_per_call/10`                | 9.35 µs         | 9.78 µs        |
| `tier_prepared_batch_sizes/int4_query_per_call/100`               | 93.31 µs        | 97.49 µs       |
| `tier_prepared_batch_sizes/int4_query_per_call/1000`              | 933.19 µs       | 974.72 µs      |
| `tier_prepared_batch_sizes/int8_batch_prepared/10`                | 170.4 ns        | 212.3 ns       |
| `tier_prepared_batch_sizes/int8_batch_prepared/100`               | 1.53 µs         | 1.93 µs        |
| `tier_prepared_batch_sizes/int8_batch_prepared/1000`              | 15.17 µs        | 19.07 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/10`                | 9.74 µs         | 8.13 µs        |
| `tier_prepared_batch_sizes/int8_query_per_call/100`               | 97.16 µs        | 80.77 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/1000`              | 972.36 µs       | 819.22 µs      |
| `tier_prepared_query/binary_query_once_1000`                      | 8.02 µs         | 48.44 µs       |
| `tier_prepared_query/binary_query_per_call_1000`                  | 1.09 ms         | 576.05 µs      |
| `tier_prepared_query/int4_query_once_1000`                        | 52.25 µs        | 134.80 µs      |
| `tier_prepared_query/int4_query_per_call_1000`                    | 944.35 µs       | 973.14 µs      |
| `tier_prepared_query/int8_query_once_1000`                        | 18.14 µs        | 19.73 µs       |
| `tier_prepared_query/int8_query_per_call_1000`                    | 947.29 µs       | 772.44 µs      |

</details>

<!-- generated:benchmarks:end -->
