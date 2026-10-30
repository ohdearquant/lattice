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
<summary><code>src/</code>: 29 files</summary>

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
| `src/`      | 29    | 9360  | 1395    | 1006  | 245   | 192         | 13       |
| `tests/`    | 14    | 2880  | 500     | 468   | 109   | 0           | 1        |
| `benches/`  | 5     | 1616  | 286     | 366   | 0     | 0           | 0        |
| `examples/` | 4     | 342   | 79      | 64    | 0     | 0           | 0        |
| total       | 52    | 14198 | 2260    | 1904  | 354   | 192         | 14       |

Other tracked files (the README excluded):

| Extension | Files | Size      |
| --------- | ----- | --------- |
| `.json`   | 10    | 455.4 KiB |
| `.md`     | 10    | 142.6 KiB |
| `.mjs`    | 2     | 25.1 KiB  |
| `.toml`   | 2     | 12.1 KiB  |
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

- Depends on: `lattice-arch` (optional, enabled by `native`), `lattice-compute` (optional, enabled by `native`, `x86-gemm`), `lattice-inference-core` (optional, enabled by `native`), `lattice-simd`, `lattice-tokenizer` (optional, enabled by `native`), `lattice-weights` (optional, enabled by `native`; features `f16`)
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

- `aarch64-linux`: commit `11ad6ea1872f`, recorded 2026-10-30T07:45:52+00:00
- `x86_64-linux`: commit `11ad6ea1872f`, recorded 2026-10-30T07:48:12+00:00

<details>
<summary>255 benchmarks</summary>

| Benchmark                                                         | `aarch64-linux` | `x86_64-linux` |
| ----------------------------------------------------------------- | --------------- | -------------- |
| `binary_cosine_distance/binary/1024`                              | 10.9 ns         | 117.6 ns       |
| `binary_cosine_distance/binary/1536`                              | 15.8 ns         | 167.4 ns       |
| `binary_cosine_distance/binary/384`                               | 5.5 ns          | 53.5 ns        |
| `binary_cosine_distance/binary/768`                               | 9.2 ns          | 91.3 ns        |
| `binary_cosine_distance/float32_simd/1024`                        | 129.3 ns        | 82.9 ns        |
| `binary_cosine_distance/float32_simd/1536`                        | 186.8 ns        | 114.1 ns       |
| `binary_cosine_distance/float32_simd/384`                         | 59.5 ns         | 45.1 ns        |
| `binary_cosine_distance/float32_simd/768`                         | 101.1 ns        | 68.3 ns        |
| `cached_embedding_delegate/native_wrong_model_disabled_64x4096`   | 233.8 ns        | 271.4 ns       |
| `int4_cosine_distance/float32_simd/1024`                          | 129.3 ns        | 82.7 ns        |
| `int4_cosine_distance/float32_simd/1536`                          | 186.9 ns        | 114.7 ns       |
| `int4_cosine_distance/float32_simd/384`                           | 59.8 ns         | 45.2 ns        |
| `int4_cosine_distance/float32_simd/768`                           | 101.2 ns        | 67.3 ns        |
| `int4_cosine_distance/int4/1024`                                  | 123.8 ns        | 304.7 ns       |
| `int4_cosine_distance/int4/1536`                                  | 180.9 ns        | 443.3 ns       |
| `int4_cosine_distance/int4/384`                                   | 53.9 ns         | 127.1 ns       |
| `int4_cosine_distance/int4/768`                                   | 96.3 ns         | 232.6 ns       |
| `int8_batch_cosine/float32_simd/10`                               | 574.6 ns        | 427.2 ns       |
| `int8_batch_cosine/float32_simd/100`                              | 6.88 µs         | 4.38 µs        |
| `int8_batch_cosine/float32_simd/1000`                             | 88.83 µs        | 56.68 µs       |
| `int8_batch_cosine/int8_loop/10`                                  | 166.9 ns        | 177.1 ns       |
| `int8_batch_cosine/int8_loop/100`                                 | 1.74 µs         | 1.75 µs        |
| `int8_batch_cosine/int8_loop/1000`                                | 21.44 µs        | 18.78 µs       |
| `int8_prepared_dot_product/per_call/1024`                         | 2.45 µs         | 1.60 µs        |
| `int8_prepared_dot_product/per_call/127`                          | 363.6 ns        | 269.3 ns       |
| `int8_prepared_dot_product/per_call/128`                          | 340.2 ns        | 249.6 ns       |
| `int8_prepared_dot_product/per_call/129`                          | 356.6 ns        | 251.3 ns       |
| `int8_prepared_dot_product/per_call/384`                          | 951.2 ns        | 641.3 ns       |
| `int8_prepared_dot_product/per_call/768`                          | 1.85 µs         | 1.22 µs        |
| `int8_prepared_dot_product/prepared/1024`                         | 31.2 ns         | 34.5 ns        |
| `int8_prepared_dot_product/prepared/127`                          | 12.3 ns         | 17.1 ns        |
| `int8_prepared_dot_product/prepared/128`                          | 8.4 ns          | 10.1 ns        |
| `int8_prepared_dot_product/prepared/129`                          | 9.0 ns          | 10.7 ns        |
| `int8_prepared_dot_product/prepared/384`                          | 14.5 ns         | 16.0 ns        |
| `int8_prepared_dot_product/prepared/768`                          | 24.5 ns         | 28.8 ns        |
| `int8_quantization/quantize/1024`                                 | 2.42 µs         | 1.57 µs        |
| `int8_quantization/quantize/1536`                                 | 3.60 µs         | 2.32 µs        |
| `int8_quantization/quantize/384`                                  | 938.5 ns        | 624.9 ns       |
| `int8_quantization/quantize/768`                                  | 1.83 µs         | 1.19 µs        |
| `int8_raw_dot_product/dot_product_i8/1024`                        | 30.6 ns         | 35.5 ns        |
| `int8_raw_dot_product/dot_product_i8/127`                         | 11.8 ns         | 17.0 ns        |
| `int8_raw_dot_product/dot_product_i8/128`                         | 8.2 ns          | 10.0 ns        |
| `int8_raw_dot_product/dot_product_i8/129`                         | 8.5 ns          | 10.6 ns        |
| `int8_raw_dot_product/dot_product_i8/384`                         | 14.2 ns         | 15.6 ns        |
| `int8_raw_dot_product/dot_product_i8/768`                         | 24.3 ns         | 29.0 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/1024`                    | 28.1 ns         | 31.5 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/127`                     | 10.1 ns         | 13.6 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/128`                     | 6.5 ns          | 7.4 ns         |
| `int8_raw_dot_product/dot_product_i8_raw/129`                     | 7.1 ns          | 7.7 ns         |
| `int8_raw_dot_product/dot_product_i8_raw/384`                     | 12.4 ns         | 13.3 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/768`                     | 21.8 ns         | 25.1 ns        |
| `int8_vs_float32_cosine/float32_simd/1024`                        | 129.4 ns        | 80.8 ns        |
| `int8_vs_float32_cosine/float32_simd/1536`                        | 185.9 ns        | 112.8 ns       |
| `int8_vs_float32_cosine/float32_simd/384`                         | 58.6 ns         | 44.7 ns        |
| `int8_vs_float32_cosine/float32_simd/768`                         | 100.0 ns        | 66.9 ns        |
| `int8_vs_float32_cosine/int8/1024`                                | 32.9 ns         | 38.4 ns        |
| `int8_vs_float32_cosine/int8/1536`                                | 46.1 ns         | 48.7 ns        |
| `int8_vs_float32_cosine/int8/384`                                 | 16.4 ns         | 17.8 ns        |
| `int8_vs_float32_cosine/int8/768`                                 | 27.2 ns         | 32.4 ns        |
| `memory_size/search_1000_float32`                                 | 57.53 µs        | 43.04 µs       |
| `memory_size/search_1000_int8`                                    | 15.03 µs        | 16.76 µs       |
| `simd_batch_cosine/scalar_loop/10`                                | 6.38 µs         | 9.88 µs        |
| `simd_batch_cosine/scalar_loop/100`                               | 63.90 µs        | 98.60 µs       |
| `simd_batch_cosine/scalar_loop/1000`                              | 646.35 µs       | 990.68 µs      |
| `simd_batch_cosine/simd_batch/10`                                 | 576.7 ns        | 428.7 ns       |
| `simd_batch_cosine/simd_batch/100`                                | 6.92 µs         | 4.43 µs        |
| `simd_batch_cosine/simd_batch/1000`                               | 93.08 µs        | 56.94 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_1000c`    | 145.73 µs       | 88.87 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_16c`      | 2.08 µs         | 1.31 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_256c`     | 36.43 µs        | 22.37 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_4c`       | 521.8 ns        | 331.5 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_64c`      | 8.29 µs         | 5.18 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_1000c`     | 58.62 µs        | 43.96 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_16c`       | 924.1 ns        | 686.6 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_256c`      | 15.02 µs        | 11.33 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_4c`        | 237.8 ns        | 181.1 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_64c`       | 3.73 µs         | 2.73 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_1000c`     | 106.62 µs       | 72.51 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_16c`       | 1.62 µs         | 1.09 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_256c`      | 30.64 µs        | 18.96 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_4c`        | 407.9 ns        | 275.4 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_64c`       | 6.46 µs         | 4.30 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_1000c`   | 145.66 µs       | 87.07 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_16c`     | 2.07 µs         | 1.28 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_256c`    | 36.05 µs        | 22.11 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_4c`      | 518.2 ns        | 323.6 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_64c`     | 8.25 µs         | 5.15 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_1000c`    | 58.03 µs        | 43.62 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_16c`      | 915.4 ns        | 665.7 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_256c`     | 14.68 µs        | 11.16 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_4c`       | 235.1 ns        | 174.1 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_64c`      | 3.70 µs         | 2.66 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_1000c`    | 106.88 µs       | 71.71 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_16c`      | 1.61 µs         | 1.07 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_256c`     | 30.87 µs        | 18.56 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_4c`       | 404.9 ns        | 268.8 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_64c`      | 6.43 µs         | 4.25 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_1000c` | 145.60 µs       | 92.13 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_16c`   | 2.07 µs         | 1.34 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_256c`  | 35.84 µs        | 23.70 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_4c`    | 520.2 ns        | 338.1 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_64c`   | 8.26 µs         | 5.42 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_1000c`  | 58.60 µs        | 43.95 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_16c`    | 919.8 ns        | 683.5 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_256c`   | 14.74 µs        | 11.37 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_4c`     | 237.4 ns        | 180.5 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_64c`    | 3.71 µs         | 2.74 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_1000c`  | 106.73 µs       | 72.64 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_16c`    | 1.60 µs         | 1.09 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_256c`   | 28.81 µs        | 18.36 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_4c`     | 406.3 ns        | 277.8 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_64c`    | 6.41 µs         | 4.32 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_1000c`    | 96.18 µs        | 78.25 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_16c`      | 1.39 µs         | 1.10 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_256c`     | 26.74 µs        | 20.11 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_4c`       | 297.7 ns        | 280.4 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_64c`      | 6.07 µs         | 4.45 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_1000c`     | 39.15 µs        | 34.07 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_16c`       | 497.2 ns        | 476.5 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_256c`      | 9.56 µs         | 7.67 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_4c`        | 133.8 ns        | 130.4 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_64c`       | 2.43 µs         | 1.90 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_1000c`     | 72.61 µs        | 61.08 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_16c`       | 969.7 ns        | 877.1 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_256c`      | 22.13 µs        | 15.11 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_4c`        | 233.2 ns        | 224.7 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_64c`       | 4.59 µs         | 3.45 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_1000c`       | 145.43 µs       | 90.79 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_16c`         | 2.06 µs         | 1.33 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_256c`        | 35.83 µs        | 23.40 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_4c`          | 516.5 ns        | 330.5 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_64c`         | 8.21 µs         | 5.36 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/384d_1000c`        | 57.90 µs        | 43.06 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_16c`          | 914.1 ns        | 669.0 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_256c`         | 14.59 µs        | 11.11 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_4c`           | 234.7 ns        | 174.7 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_64c`          | 3.67 µs         | 2.67 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_1000c`        | 107.04 µs       | 71.58 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_16c`          | 1.60 µs         | 1.08 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_256c`         | 29.29 µs        | 18.34 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_4c`           | 403.3 ns        | 270.8 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_64c`          | 6.37 µs         | 4.28 µs        |
| `simd_batch_dot_product/scalar_loop/10`                           | 2.14 µs         | 3.30 µs        |
| `simd_batch_dot_product/scalar_loop/100`                          | 21.38 µs        | 33.05 µs       |
| `simd_batch_dot_product/scalar_loop/1000`                         | 219.39 µs       | 337.72 µs      |
| `simd_batch_dot_product/simd_batch/10`                            | 315.2 ns        | 281.9 ns       |
| `simd_batch_dot_product/simd_batch/100`                           | 5.61 µs         | 3.38 µs        |
| `simd_batch_dot_product/simd_batch/1000`                          | 86.73 µs        | 55.69 µs       |
| `simd_cosine_similarity/scalar/1024`                              | 1.77 µs         | 2.80 µs        |
| `simd_cosine_similarity/scalar/1536`                              | 2.68 µs         | 4.23 µs        |
| `simd_cosine_similarity/scalar/384`                               | 642.0 ns        | 996.0 ns       |
| `simd_cosine_similarity/scalar/768`                               | 1.32 µs         | 2.08 µs        |
| `simd_cosine_similarity/simd/1024`                                | 130.3 ns        | 82.1 ns        |
| `simd_cosine_similarity/simd/1536`                                | 185.8 ns        | 112.9 ns       |
| `simd_cosine_similarity/simd/384`                                 | 58.4 ns         | 44.7 ns        |
| `simd_cosine_similarity/simd/768`                                 | 101.0 ns        | 70.1 ns        |
| `simd_dot_product/scalar/1024`                                    | 593.6 ns        | 936.3 ns       |
| `simd_dot_product/scalar/1536`                                    | 892.6 ns        | 1.41 µs        |
| `simd_dot_product/scalar/384`                                     | 213.2 ns        | 337.8 ns       |
| `simd_dot_product/scalar/768`                                     | 439.8 ns        | 696.7 ns       |
| `simd_dot_product/simd/1024`                                      | 73.6 ns         | 62.7 ns        |
| `simd_dot_product/simd/1536`                                      | 106.8 ns        | 89.2 ns        |
| `simd_dot_product/simd/384`                                       | 31.7 ns         | 28.6 ns        |
| `simd_dot_product/simd/768`                                       | 55.0 ns         | 49.6 ns        |
| `simd_euclidean_distance/scalar/1024`                             | 595.9 ns        | 943.4 ns       |
| `simd_euclidean_distance/scalar/1536`                             | 896.1 ns        | 1.42 µs        |
| `simd_euclidean_distance/scalar/384`                              | 217.0 ns        | 345.2 ns       |
| `simd_euclidean_distance/scalar/768`                              | 443.9 ns        | 703.3 ns       |
| `simd_euclidean_distance/simd/1024`                               | 117.0 ns        | 55.9 ns        |
| `simd_euclidean_distance/simd/1536`                               | 171.0 ns        | 94.1 ns        |
| `simd_euclidean_distance/simd/384`                                | 48.6 ns         | 34.9 ns        |
| `simd_euclidean_distance/simd/768`                                | 89.4 ns         | 54.4 ns        |
| `simd_manhattan_distance/scalar/1024`                             | 592.8 ns        | 939.9 ns       |
| `simd_manhattan_distance/scalar/1536`                             | 891.5 ns        | 1.42 µs        |
| `simd_manhattan_distance/scalar/384`                              | 212.5 ns        | 341.3 ns       |
| `simd_manhattan_distance/scalar/768`                              | 438.8 ns        | 700.2 ns       |
| `simd_manhattan_distance/simd/1024`                               | 85.5 ns         | 74.8 ns        |
| `simd_manhattan_distance/simd/1536`                               | 127.3 ns        | 89.4 ns        |
| `simd_manhattan_distance/simd/384`                                | 36.7 ns         | 30.2 ns        |
| `simd_manhattan_distance/simd/768`                                | 66.7 ns         | 49.8 ns        |
| `simd_normalize/scalar/1024`                                      | 691.8 ns        | 1.06 µs        |
| `simd_normalize/scalar/1536`                                      | 1.04 µs         | 1.57 µs        |
| `simd_normalize/scalar/384`                                       | 266.1 ns        | 396.6 ns       |
| `simd_normalize/scalar/768`                                       | 521.8 ns        | 789.2 ns       |
| `simd_normalize/simd/1024`                                        | 159.7 ns        | 148.8 ns       |
| `simd_normalize/simd/1536`                                        | 233.8 ns        | 209.3 ns       |
| `simd_normalize/simd/384`                                         | 74.4 ns         | 69.3 ns        |
| `simd_normalize/simd/768`                                         | 123.8 ns        | 119.8 ns       |
| `simd_normalized_cosine_fast_path/cosine_full/1024`               | 129.0 ns        | 86.6 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/384`                | 59.7 ns         | 44.1 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/768`                | 99.9 ns         | 66.6 ns        |
| `simd_normalized_cosine_fast_path/dot_product/1024`               | 73.6 ns         | 75.1 ns        |
| `simd_normalized_cosine_fast_path/dot_product/384`                | 33.3 ns         | 28.5 ns        |
| `simd_normalized_cosine_fast_path/dot_product/768`                | 56.8 ns         | 49.6 ns        |
| `simd_prepared_query_normalized_cosine/dot_product_loop/1024`     | 96.44 µs        | 75.90 µs       |
| `simd_prepared_query_normalized_cosine/dot_product_loop/384`      | 38.80 µs        | 32.60 µs       |
| `simd_prepared_query_normalized_cosine/dot_product_loop/768`      | 73.62 µs        | 61.12 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/1024` | 146.94 µs       | 95.70 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/384`  | 61.19 µs        | 47.86 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/768`  | 107.69 µs       | 74.16 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/1024`   | 145.41 µs       | 97.09 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/384`    | 60.95 µs        | 48.01 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/768`    | 108.78 µs       | 75.02 µs       |
| `simd_query_batch_dot_product/pair_loop/128d_16c`                 | 221.4 ns        | 210.4 ns       |
| `simd_query_batch_dot_product/pair_loop/128d_256c`                | 3.81 µs         | 3.08 µs        |
| `simd_query_batch_dot_product/pair_loop/128d_4c`                  | 67.1 ns         | 68.2 ns        |
| `simd_query_batch_dot_product/pair_loop/128d_64c`                 | 827.1 ns        | 795.5 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_16c`                 | 490.2 ns        | 476.0 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_256c`                | 9.55 µs         | 7.67 µs        |
| `simd_query_batch_dot_product/pair_loop/384d_4c`                  | 133.6 ns        | 125.8 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_64c`                 | 2.42 µs         | 1.89 µs        |
| `simd_query_batch_dot_product/pair_loop/768d_16c`                 | 950.9 ns        | 859.3 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_256c`                | 23.58 µs        | 15.65 µs       |
| `simd_query_batch_dot_product/pair_loop/768d_4c`                  | 233.6 ns        | 222.2 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_64c`                 | 4.59 µs         | 3.40 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_16c`                | 138.1 ns        | 127.5 ns       |
| `simd_query_batch_dot_product/simd_batch/128d_256c`               | 3.08 µs         | 1.96 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_4c`                 | 46.2 ns         | 44.2 ns        |
| `simd_query_batch_dot_product/simd_batch/128d_64c`                | 520.9 ns        | 478.6 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_16c`                | 326.9 ns        | 259.0 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_256c`               | 8.39 µs         | 5.24 µs        |
| `simd_query_batch_dot_product/simd_batch/384d_4c`                 | 93.8 ns         | 74.7 ns        |
| `simd_query_batch_dot_product/simd_batch/384d_64c`                | 2.13 µs         | 1.14 µs        |
| `simd_query_batch_dot_product/simd_batch/768d_16c`                | 636.8 ns        | 552.7 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_256c`               | 18.73 µs        | 12.89 µs       |
| `simd_query_batch_dot_product/simd_batch/768d_4c`                 | 165.2 ns        | 131.4 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_64c`                | 3.83 µs         | 2.19 µs        |
| `simd_squared_euclidean_fast_path/euclidean_full/1024`            | 117.2 ns        | 79.7 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/384`             | 48.3 ns         | 34.7 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/768`             | 89.3 ns         | 54.4 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/1024`         | 114.0 ns        | 74.9 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/384`          | 45.9 ns         | 30.3 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/768`          | 86.8 ns         | 49.9 ns        |
| `simd_throughput_384/cosine_similarity`                           | 59.0 ns         | 43.8 ns        |
| `simd_throughput_384/dot_product`                                 | 31.5 ns         | 28.6 ns        |
| `simd_throughput_384/euclidean_distance`                          | 48.6 ns         | 34.8 ns        |
| `simd_throughput_384/normalize`                                   | 122.7 ns        | 114.3 ns       |
| `tier_prepared_batch_sizes/int4_batch_prepared/10`                | 519.0 ns        | 1.24 µs        |
| `tier_prepared_batch_sizes/int4_batch_prepared/100`               | 5.01 µs         | 12.19 µs       |
| `tier_prepared_batch_sizes/int4_batch_prepared/1000`              | 49.84 µs        | 121.89 µs      |
| `tier_prepared_batch_sizes/int4_query_per_call/10`                | 9.36 µs         | 7.38 µs        |
| `tier_prepared_batch_sizes/int4_query_per_call/100`               | 93.34 µs        | 73.74 µs       |
| `tier_prepared_batch_sizes/int4_query_per_call/1000`              | 933.46 µs       | 737.99 µs      |
| `tier_prepared_batch_sizes/int8_batch_prepared/10`                | 174.3 ns        | 190.4 ns       |
| `tier_prepared_batch_sizes/int8_batch_prepared/100`               | 1.53 µs         | 1.71 µs        |
| `tier_prepared_batch_sizes/int8_batch_prepared/1000`              | 15.87 µs        | 17.19 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/10`                | 9.72 µs         | 6.62 µs        |
| `tier_prepared_batch_sizes/int8_query_per_call/100`               | 96.85 µs        | 65.93 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/1000`              | 969.38 µs       | 660.88 µs      |
| `tier_prepared_query/binary_query_once_1000`                      | 8.15 µs         | 52.47 µs       |
| `tier_prepared_query/binary_query_per_call_1000`                  | 1.11 ms         | 474.96 µs      |
| `tier_prepared_query/int4_query_once_1000`                        | 52.81 µs        | 125.41 µs      |
| `tier_prepared_query/int4_query_per_call_1000`                    | 938.24 µs       | 736.19 µs      |
| `tier_prepared_query/int8_query_once_1000`                        | 18.55 µs        | 19.95 µs       |
| `tier_prepared_query/int8_query_per_call_1000`                    | 948.29 µs       | 632.34 µs      |

</details>

<!-- generated:benchmarks:end -->
