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

- Version: `0.16.0`
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
<summary><code>src/</code>: 34 files</summary>

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
    - `identity.rs`: Declared native output identities bound to the successful load.
    - `identity/`
      - `gemma.rs`
      - `gemma_fixture.rs`: Tiny synthetic EmbeddingGemma checkpoint for native identity tests.
      - `gemma_tests.rs`
      - `qwen.rs`
      - `qwen_tests.rs`
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
- bench (5): `embeddings`, `simd_bench`, `simd_opt_bench`, `simd` (requires `native`), `simsimd_comparison`

<!-- generated:layout:end -->

<!-- generated:stats:begin -->

### Statistics

Rust lines by kind (code, comment, blank; doc comments count as comment), test functions (`#[test]` and `#[<runtime>::test]` attributes), items declared `pub` (not `pub(crate)`; reachability through private modules is not resolved) and occurrences of the `unsafe` keyword outside comments and strings.

| Area        | Files | Code  | Comment | Blank | Tests | `pub` items | `unsafe` |
| ----------- | ----- | ----- | ------- | ----- | ----- | ----------- | -------- |
| `src/`      | 34    | 12049 | 1434    | 1113  | 284   | 193         | 13       |
| `tests/`    | 14    | 2883  | 500     | 468   | 109   | 0           | 1        |
| `benches/`  | 5     | 1616  | 286     | 366   | 0     | 0           | 0        |
| `examples/` | 4     | 342   | 79      | 64    | 0     | 0           | 0        |
| total       | 57    | 16890 | 2299    | 2011  | 393   | 193         | 14       |

Other tracked files (the README excluded):

| Extension | Files | Size      |
| --------- | ----- | --------- |
| `.json`   | 10    | 455.4 KiB |
| `.md`     | 10    | 143.5 KiB |
| `.mjs`    | 2     | 25.1 KiB  |
| `.toml`   | 2     | 12.2 KiB  |
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

- Depends on: `lattice-arch` (optional, enabled by `native`), `lattice-compute` (optional, enabled by `native`, `x86-gemm`), `lattice-inference-core` (optional, enabled by `native`), `lattice-model-bert` (optional, enabled by `download`, `native`, `x86-gemm`; no default features; features `std`, `f16`), `lattice-model-embeddinggemma2` (optional, enabled by `image`, `metal-gpu`, `native`), `lattice-model-qwen3-embedding` (optional, enabled by `download`, `metal-gpu`, `native`, `x86-gemm`; no default features), `lattice-model-qwen35` (optional, enabled by `image`, `metal-gpu`, `native`; features `f16`), `lattice-simd`, `lattice-tokenizer` (optional, enabled by `native`), `lattice-weights` (optional, enabled by `native`; features `f16`)
- Dev-depends on: `lattice-inference` (no default features; features `bert`, `qwen3`, `std`, `test-utils`, `qwen35`), `lattice-metal-runtime` (features `metal-gpu`; only on `cfg(target_os = "macos")`), `lattice-tokenizer`, `lattice-transport`
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

- `aarch64-linux`: commit `4abf98a66dcc`, recorded 2026-11-13T13:20:55+00:00
- `x86_64-linux`: commit `4abf98a66dcc`, recorded 2026-11-13T13:24:31+00:00

<details>
<summary>255 benchmarks</summary>

| Benchmark                                                         | `aarch64-linux` | `x86_64-linux` |
| ----------------------------------------------------------------- | --------------- | -------------- |
| `binary_cosine_distance/binary/1024`                              | 10.9 ns         | 117.5 ns       |
| `binary_cosine_distance/binary/1536`                              | 15.7 ns         | 168.2 ns       |
| `binary_cosine_distance/binary/384`                               | 5.5 ns          | 53.4 ns        |
| `binary_cosine_distance/binary/768`                               | 9.2 ns          | 91.6 ns        |
| `binary_cosine_distance/float32_simd/1024`                        | 129.0 ns        | 87.3 ns        |
| `binary_cosine_distance/float32_simd/1536`                        | 186.1 ns        | 113.6 ns       |
| `binary_cosine_distance/float32_simd/384`                         | 58.3 ns         | 44.9 ns        |
| `binary_cosine_distance/float32_simd/768`                         | 100.0 ns        | 67.8 ns        |
| `cached_embedding_delegate/native_wrong_model_disabled_64x4096`   | 236.0 ns        | 249.5 ns       |
| `int4_cosine_distance/float32_simd/1024`                          | 129.0 ns        | 83.3 ns        |
| `int4_cosine_distance/float32_simd/1536`                          | 186.5 ns        | 114.3 ns       |
| `int4_cosine_distance/float32_simd/384`                           | 59.3 ns         | 43.8 ns        |
| `int4_cosine_distance/float32_simd/768`                           | 100.2 ns        | 67.7 ns        |
| `int4_cosine_distance/int4/1024`                                  | 123.9 ns        | 302.4 ns       |
| `int4_cosine_distance/int4/1536`                                  | 181.2 ns        | 443.9 ns       |
| `int4_cosine_distance/int4/384`                                   | 53.8 ns         | 126.8 ns       |
| `int4_cosine_distance/int4/768`                                   | 96.8 ns         | 232.9 ns       |
| `int8_batch_cosine/float32_simd/10`                               | 573.8 ns        | 429.8 ns       |
| `int8_batch_cosine/float32_simd/100`                              | 6.91 µs         | 4.51 µs        |
| `int8_batch_cosine/float32_simd/1000`                             | 86.46 µs        | 58.63 µs       |
| `int8_batch_cosine/int8_loop/10`                                  | 161.6 ns        | 177.3 ns       |
| `int8_batch_cosine/int8_loop/100`                                 | 1.68 µs         | 1.83 µs        |
| `int8_batch_cosine/int8_loop/1000`                                | 19.91 µs        | 19.45 µs       |
| `int8_prepared_dot_product/per_call/1024`                         | 2.45 µs         | 1.61 µs        |
| `int8_prepared_dot_product/per_call/127`                          | 364.0 ns        | 278.1 ns       |
| `int8_prepared_dot_product/per_call/128`                          | 348.5 ns        | 256.1 ns       |
| `int8_prepared_dot_product/per_call/129`                          | 354.9 ns        | 260.6 ns       |
| `int8_prepared_dot_product/per_call/384`                          | 952.0 ns        | 640.0 ns       |
| `int8_prepared_dot_product/per_call/768`                          | 1.85 µs         | 1.22 µs        |
| `int8_prepared_dot_product/prepared/1024`                         | 30.7 ns         | 35.3 ns        |
| `int8_prepared_dot_product/prepared/127`                          | 12.2 ns         | 17.1 ns        |
| `int8_prepared_dot_product/prepared/128`                          | 8.5 ns          | 10.8 ns        |
| `int8_prepared_dot_product/prepared/129`                          | 9.0 ns          | 11.3 ns        |
| `int8_prepared_dot_product/prepared/384`                          | 14.3 ns         | 16.0 ns        |
| `int8_prepared_dot_product/prepared/768`                          | 24.6 ns         | 29.2 ns        |
| `int8_quantization/quantize/1024`                                 | 2.42 µs         | 1.56 µs        |
| `int8_quantization/quantize/1536`                                 | 3.60 µs         | 2.31 µs        |
| `int8_quantization/quantize/384`                                  | 940.8 ns        | 637.0 ns       |
| `int8_quantization/quantize/768`                                  | 1.83 µs         | 1.20 µs        |
| `int8_raw_dot_product/dot_product_i8/1024`                        | 30.5 ns         | 34.7 ns        |
| `int8_raw_dot_product/dot_product_i8/127`                         | 11.8 ns         | 17.1 ns        |
| `int8_raw_dot_product/dot_product_i8/128`                         | 8.1 ns          | 10.0 ns        |
| `int8_raw_dot_product/dot_product_i8/129`                         | 8.5 ns          | 10.4 ns        |
| `int8_raw_dot_product/dot_product_i8/384`                         | 13.9 ns         | 15.7 ns        |
| `int8_raw_dot_product/dot_product_i8/768`                         | 23.7 ns         | 28.7 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/1024`                    | 28.1 ns         | 30.5 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/127`                     | 10.0 ns         | 13.6 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/128`                     | 6.5 ns          | 7.3 ns         |
| `int8_raw_dot_product/dot_product_i8_raw/129`                     | 7.2 ns          | 7.6 ns         |
| `int8_raw_dot_product/dot_product_i8_raw/384`                     | 12.4 ns         | 13.3 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/768`                     | 21.9 ns         | 24.6 ns        |
| `int8_vs_float32_cosine/float32_simd/1024`                        | 129.4 ns        | 82.2 ns        |
| `int8_vs_float32_cosine/float32_simd/1536`                        | 186.1 ns        | 113.4 ns       |
| `int8_vs_float32_cosine/float32_simd/384`                         | 58.6 ns         | 44.4 ns        |
| `int8_vs_float32_cosine/float32_simd/768`                         | 100.2 ns        | 67.1 ns        |
| `int8_vs_float32_cosine/int8/1024`                                | 33.0 ns         | 38.0 ns        |
| `int8_vs_float32_cosine/int8/1536`                                | 45.7 ns         | 48.4 ns        |
| `int8_vs_float32_cosine/int8/384`                                 | 15.8 ns         | 18.9 ns        |
| `int8_vs_float32_cosine/int8/768`                                 | 27.0 ns         | 32.7 ns        |
| `memory_size/search_1000_float32`                                 | 57.16 µs        | 42.66 µs       |
| `memory_size/search_1000_int8`                                    | 14.70 µs        | 16.87 µs       |
| `simd_batch_cosine/scalar_loop/10`                                | 6.39 µs         | 9.94 µs        |
| `simd_batch_cosine/scalar_loop/100`                               | 63.99 µs        | 98.64 µs       |
| `simd_batch_cosine/scalar_loop/1000`                              | 647.26 µs       | 989.64 µs      |
| `simd_batch_cosine/simd_batch/10`                                 | 576.1 ns        | 432.2 ns       |
| `simd_batch_cosine/simd_batch/100`                                | 6.92 µs         | 4.56 µs        |
| `simd_batch_cosine/simd_batch/1000`                               | 90.14 µs        | 54.54 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_1000c`    | 144.66 µs       | 118.51 µs      |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_16c`      | 2.10 µs         | 1.78 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_256c`     | 34.81 µs        | 30.29 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_4c`       | 521.6 ns        | 436.2 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_64c`      | 8.28 µs         | 7.69 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_1000c`     | 58.30 µs        | 69.63 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_16c`       | 923.0 ns        | 1.07 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_256c`      | 14.80 µs        | 17.70 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_4c`        | 238.3 ns        | 272.1 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_64c`       | 3.73 µs         | 4.34 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_1000c`     | 105.60 µs       | 100.02 µs      |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_16c`       | 1.64 µs         | 1.52 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_256c`      | 28.38 µs        | 25.79 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_4c`        | 407.8 ns        | 372.7 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_64c`       | 6.46 µs         | 6.04 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_1000c`   | 144.52 µs       | 90.71 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_16c`     | 2.10 µs         | 1.35 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_256c`    | 35.58 µs        | 23.23 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_4c`      | 518.8 ns        | 338.8 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_64c`     | 8.25 µs         | 5.77 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_1000c`    | 57.72 µs        | 43.33 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_16c`      | 916.3 ns        | 664.3 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_256c`     | 14.65 µs        | 10.64 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_4c`       | 235.8 ns        | 176.6 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_64c`      | 3.73 µs         | 2.63 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_1000c`    | 105.95 µs       | 70.96 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_16c`      | 1.64 µs         | 1.09 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_256c`     | 28.46 µs        | 18.55 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_4c`       | 405.4 ns        | 277.9 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_64c`      | 6.43 µs         | 4.34 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_1000c` | 144.29 µs       | 116.77 µs      |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_16c`   | 2.08 µs         | 1.79 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_256c`  | 34.53 µs        | 30.46 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_4c`    | 519.8 ns        | 433.8 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_64c`   | 8.24 µs         | 7.59 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_1000c`  | 57.84 µs        | 71.06 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_16c`    | 921.7 ns        | 1.09 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_256c`   | 14.73 µs        | 17.92 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_4c`     | 237.9 ns        | 274.4 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_64c`    | 3.71 µs         | 4.37 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_1000c`  | 105.87 µs       | 100.41 µs      |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_16c`    | 1.62 µs         | 1.49 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_256c`   | 28.80 µs        | 25.79 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_4c`     | 406.2 ns        | 370.1 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_64c`    | 6.43 µs         | 6.03 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_1000c`    | 96.12 µs        | 75.17 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_16c`      | 1.46 µs         | 1.12 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_256c`     | 26.00 µs        | 19.82 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_4c`       | 296.1 ns        | 280.3 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_64c`      | 6.05 µs         | 4.90 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_1000c`     | 38.11 µs        | 32.46 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_16c`       | 491.7 ns        | 475.9 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_256c`      | 9.59 µs         | 8.22 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_4c`        | 134.7 ns        | 125.1 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_64c`       | 2.45 µs         | 1.92 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_1000c`     | 74.23 µs        | 58.75 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_16c`       | 1.03 µs         | 849.6 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_256c`      | 21.69 µs        | 15.59 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_4c`        | 232.1 ns        | 210.0 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_64c`       | 4.66 µs         | 3.43 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_1000c`       | 144.67 µs       | 89.05 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_16c`         | 2.07 µs         | 1.35 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_256c`        | 35.46 µs        | 22.96 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_4c`          | 516.3 ns        | 341.3 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_64c`         | 8.21 µs         | 5.57 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/384d_1000c`        | 57.36 µs        | 44.56 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_16c`          | 914.6 ns        | 674.6 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_256c`         | 14.60 µs        | 10.84 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_4c`           | 235.1 ns        | 173.0 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_64c`          | 3.68 µs         | 2.69 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_1000c`        | 106.66 µs       | 73.11 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_16c`          | 1.63 µs         | 1.09 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_256c`         | 28.77 µs        | 18.64 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_4c`           | 403.6 ns        | 266.5 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_64c`          | 6.39 µs         | 4.28 µs        |
| `simd_batch_dot_product/scalar_loop/10`                           | 2.14 µs         | 3.29 µs        |
| `simd_batch_dot_product/scalar_loop/100`                          | 21.37 µs        | 33.09 µs       |
| `simd_batch_dot_product/scalar_loop/1000`                         | 221.04 µs       | 337.33 µs      |
| `simd_batch_dot_product/simd_batch/10`                            | 334.8 ns        | 284.8 ns       |
| `simd_batch_dot_product/simd_batch/100`                           | 5.71 µs         | 3.45 µs        |
| `simd_batch_dot_product/simd_batch/1000`                          | 89.81 µs        | 45.48 µs       |
| `simd_cosine_similarity/scalar/1024`                              | 1.77 µs         | 2.80 µs        |
| `simd_cosine_similarity/scalar/1536`                              | 2.68 µs         | 4.24 µs        |
| `simd_cosine_similarity/scalar/384`                               | 642.9 ns        | 995.8 ns       |
| `simd_cosine_similarity/scalar/768`                               | 1.32 µs         | 2.08 µs        |
| `simd_cosine_similarity/simd/1024`                                | 130.1 ns        | 82.2 ns        |
| `simd_cosine_similarity/simd/1536`                                | 186.0 ns        | 113.4 ns       |
| `simd_cosine_similarity/simd/384`                                 | 58.2 ns         | 43.9 ns        |
| `simd_cosine_similarity/simd/768`                                 | 101.7 ns        | 66.9 ns        |
| `simd_dot_product/scalar/1024`                                    | 594.3 ns        | 936.3 ns       |
| `simd_dot_product/scalar/1536`                                    | 892.4 ns        | 1.42 µs        |
| `simd_dot_product/scalar/384`                                     | 213.2 ns        | 337.7 ns       |
| `simd_dot_product/scalar/768`                                     | 439.8 ns        | 696.8 ns       |
| `simd_dot_product/simd/1024`                                      | 73.7 ns         | 62.5 ns        |
| `simd_dot_product/simd/1536`                                      | 106.5 ns        | 89.5 ns        |
| `simd_dot_product/simd/384`                                       | 31.6 ns         | 28.6 ns        |
| `simd_dot_product/simd/768`                                       | 54.7 ns         | 58.8 ns        |
| `simd_euclidean_distance/scalar/1024`                             | 600.5 ns        | 942.5 ns       |
| `simd_euclidean_distance/scalar/1536`                             | 897.1 ns        | 1.42 µs        |
| `simd_euclidean_distance/scalar/384`                              | 218.7 ns        | 344.2 ns       |
| `simd_euclidean_distance/scalar/768`                              | 444.6 ns        | 703.3 ns       |
| `simd_euclidean_distance/simd/1024`                               | 117.1 ns        | 67.5 ns        |
| `simd_euclidean_distance/simd/1536`                               | 171.1 ns        | 112.4 ns       |
| `simd_euclidean_distance/simd/384`                                | 48.6 ns         | 35.2 ns        |
| `simd_euclidean_distance/simd/768`                                | 89.2 ns         | 54.5 ns        |
| `simd_manhattan_distance/scalar/1024`                             | 592.2 ns        | 938.8 ns       |
| `simd_manhattan_distance/scalar/1536`                             | 891.4 ns        | 1.42 µs        |
| `simd_manhattan_distance/scalar/384`                              | 212.4 ns        | 341.5 ns       |
| `simd_manhattan_distance/scalar/768`                              | 438.7 ns        | 700.1 ns       |
| `simd_manhattan_distance/simd/1024`                               | 84.8 ns         | 63.2 ns        |
| `simd_manhattan_distance/simd/1536`                               | 126.5 ns        | 89.5 ns        |
| `simd_manhattan_distance/simd/384`                                | 36.6 ns         | 30.2 ns        |
| `simd_manhattan_distance/simd/768`                                | 66.7 ns         | 49.9 ns        |
| `simd_normalize/scalar/1024`                                      | 692.3 ns        | 1.05 µs        |
| `simd_normalize/scalar/1536`                                      | 1.03 µs         | 1.58 µs        |
| `simd_normalize/scalar/384`                                       | 267.6 ns        | 396.5 ns       |
| `simd_normalize/scalar/768`                                       | 524.3 ns        | 788.7 ns       |
| `simd_normalize/simd/1024`                                        | 161.5 ns        | 157.9 ns       |
| `simd_normalize/simd/1536`                                        | 239.4 ns        | 250.9 ns       |
| `simd_normalize/simd/384`                                         | 76.2 ns         | 78.0 ns        |
| `simd_normalize/simd/768`                                         | 126.5 ns        | 130.3 ns       |
| `simd_normalized_cosine_fast_path/cosine_full/1024`               | 129.2 ns        | 82.0 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/384`                | 58.6 ns         | 42.9 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/768`                | 100.3 ns        | 66.7 ns        |
| `simd_normalized_cosine_fast_path/dot_product/1024`               | 73.7 ns         | 64.0 ns        |
| `simd_normalized_cosine_fast_path/dot_product/384`                | 31.0 ns         | 25.2 ns        |
| `simd_normalized_cosine_fast_path/dot_product/768`                | 55.7 ns         | 49.7 ns        |
| `simd_prepared_query_normalized_cosine/dot_product_loop/1024`     | 96.13 µs        | 76.85 µs       |
| `simd_prepared_query_normalized_cosine/dot_product_loop/384`      | 38.27 µs        | 29.54 µs       |
| `simd_prepared_query_normalized_cosine/dot_product_loop/768`      | 72.56 µs        | 61.24 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/1024` | 147.04 µs       | 96.09 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/384`  | 60.73 µs        | 46.48 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/768`  | 106.52 µs       | 72.82 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/1024`   | 145.62 µs       | 93.38 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/384`    | 60.68 µs        | 46.62 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/768`    | 107.88 µs       | 73.46 µs       |
| `simd_query_batch_dot_product/pair_loop/128d_16c`                 | 218.6 ns        | 208.5 ns       |
| `simd_query_batch_dot_product/pair_loop/128d_256c`                | 3.79 µs         | 3.10 µs        |
| `simd_query_batch_dot_product/pair_loop/128d_4c`                  | 65.8 ns         | 64.8 ns        |
| `simd_query_batch_dot_product/pair_loop/128d_64c`                 | 814.9 ns        | 792.0 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_16c`                 | 498.2 ns        | 476.7 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_256c`                | 9.55 µs         | 8.07 µs        |
| `simd_query_batch_dot_product/pair_loop/384d_4c`                  | 134.2 ns        | 128.4 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_64c`                 | 2.46 µs         | 1.92 µs        |
| `simd_query_batch_dot_product/pair_loop/768d_16c`                 | 977.2 ns        | 874.3 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_256c`                | 21.12 µs        | 14.68 µs       |
| `simd_query_batch_dot_product/pair_loop/768d_4c`                  | 235.9 ns        | 220.2 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_64c`                 | 4.76 µs         | 3.52 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_16c`                | 135.0 ns        | 126.3 ns       |
| `simd_query_batch_dot_product/simd_batch/128d_256c`               | 3.06 µs         | 1.96 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_4c`                 | 45.9 ns         | 42.5 ns        |
| `simd_query_batch_dot_product/simd_batch/128d_64c`                | 509.3 ns        | 485.7 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_16c`                | 325.2 ns        | 260.2 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_256c`               | 8.35 µs         | 6.07 µs        |
| `simd_query_batch_dot_product/simd_batch/384d_4c`                 | 93.1 ns         | 76.4 ns        |
| `simd_query_batch_dot_product/simd_batch/384d_64c`                | 2.13 µs         | 1.15 µs        |
| `simd_query_batch_dot_product/simd_batch/768d_16c`                | 723.7 ns        | 545.6 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_256c`               | 17.90 µs        | 12.84 µs       |
| `simd_query_batch_dot_product/simd_batch/768d_4c`                 | 164.1 ns        | 131.3 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_64c`                | 3.90 µs         | 2.29 µs        |
| `simd_squared_euclidean_fast_path/euclidean_full/1024`            | 117.2 ns        | 67.4 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/384`             | 48.5 ns         | 35.2 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/768`             | 89.0 ns         | 54.6 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/1024`         | 114.0 ns        | 62.9 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/384`          | 45.8 ns         | 31.6 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/768`          | 86.4 ns         | 49.9 ns        |
| `simd_throughput_384/cosine_similarity`                           | 59.1 ns         | 45.0 ns        |
| `simd_throughput_384/dot_product`                                 | 32.3 ns         | 30.0 ns        |
| `simd_throughput_384/euclidean_distance`                          | 49.0 ns         | 37.9 ns        |
| `simd_throughput_384/normalize`                                   | 118.9 ns        | 103.9 ns       |
| `tier_prepared_batch_sizes/int4_batch_prepared/10`                | 518.3 ns        | 1.24 µs        |
| `tier_prepared_batch_sizes/int4_batch_prepared/100`               | 5.02 µs         | 12.19 µs       |
| `tier_prepared_batch_sizes/int4_batch_prepared/1000`              | 50.39 µs        | 122.57 µs      |
| `tier_prepared_batch_sizes/int4_query_per_call/10`                | 9.34 µs         | 7.55 µs        |
| `tier_prepared_batch_sizes/int4_query_per_call/100`               | 93.16 µs        | 75.56 µs       |
| `tier_prepared_batch_sizes/int4_query_per_call/1000`              | 932.13 µs       | 757.48 µs      |
| `tier_prepared_batch_sizes/int8_batch_prepared/10`                | 169.5 ns        | 195.7 ns       |
| `tier_prepared_batch_sizes/int8_batch_prepared/100`               | 1.51 µs         | 1.73 µs        |
| `tier_prepared_batch_sizes/int8_batch_prepared/1000`              | 15.12 µs        | 17.44 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/10`                | 9.71 µs         | 6.72 µs        |
| `tier_prepared_batch_sizes/int8_query_per_call/100`               | 96.64 µs        | 67.42 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/1000`              | 966.90 µs       | 673.17 µs      |
| `tier_prepared_query/binary_query_once_1000`                      | 7.91 µs         | 52.51 µs       |
| `tier_prepared_query/binary_query_per_call_1000`                  | 1.09 ms         | 477.47 µs      |
| `tier_prepared_query/int4_query_once_1000`                        | 52.59 µs        | 125.67 µs      |
| `tier_prepared_query/int4_query_per_call_1000`                    | 944.59 µs       | 755.78 µs      |
| `tier_prepared_query/int8_query_once_1000`                        | 18.06 µs        | 20.60 µs       |
| `tier_prepared_query/int8_query_per_call_1000`                    | 946.07 µs       | 635.96 µs      |

</details>

<!-- generated:benchmarks:end -->
