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
- bench (5): `embeddings`, `simd_bench`, `simd_opt_bench`, `simd`, `simsimd_comparison`

<!-- generated:layout:end -->

<!-- generated:stats:begin -->

### Statistics

Rust lines by kind (code, comment, blank; doc comments count as comment), test functions (`#[test]` and `#[<runtime>::test]` attributes), items declared `pub` (not `pub(crate)`; reachability through private modules is not resolved) and occurrences of the `unsafe` keyword outside comments and strings.

| Area        | Files | Code  | Comment | Blank | Tests | `pub` items | `unsafe` |
| ----------- | ----- | ----- | ------- | ----- | ----- | ----------- | -------- |
| `src/`      | 34    | 11631 | 1408    | 1085  | 274   | 193         | 13       |
| `tests/`    | 14    | 2880  | 500     | 468   | 109   | 0           | 1        |
| `benches/`  | 5     | 1616  | 286     | 366   | 0     | 0           | 0        |
| `examples/` | 4     | 342   | 79      | 64    | 0     | 0           | 0        |
| total       | 57    | 16469 | 2273    | 1983  | 383   | 193         | 14       |

Other tracked files (the README excluded):

| Extension | Files | Size      |
| --------- | ----- | --------- |
| `.json`   | 10    | 455.4 KiB |
| `.md`     | 10    | 142.6 KiB |
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

- Depends on: `lattice-arch` (optional, enabled by `native`), `lattice-compute` (optional, enabled by `native`, `x86-gemm`), `lattice-inference-core` (optional, enabled by `native`), `lattice-simd`, `lattice-tokenizer` (optional, enabled by `native`), `lattice-weights` (optional, enabled by `native`; features `f16`)
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

- `aarch64-linux`: commit `0fcbb8bf787f`, recorded 2026-10-30T08:50:24+00:00
- `x86_64-linux`: commit `0fcbb8bf787f`, recorded 2026-10-30T08:50:17+00:00

<details>
<summary>255 benchmarks</summary>

| Benchmark                                                         | `aarch64-linux` | `x86_64-linux` |
| ----------------------------------------------------------------- | --------------- | -------------- |
| `binary_cosine_distance/binary/1024`                              | 10.9 ns         | 63.2 ns        |
| `binary_cosine_distance/binary/1536`                              | 15.7 ns         | 89.6 ns        |
| `binary_cosine_distance/binary/384`                               | 5.5 ns          | 29.2 ns        |
| `binary_cosine_distance/binary/768`                               | 9.2 ns          | 49.2 ns        |
| `binary_cosine_distance/float32_simd/1024`                        | 129.3 ns        | 31.3 ns        |
| `binary_cosine_distance/float32_simd/1536`                        | 186.7 ns        | 44.8 ns        |
| `binary_cosine_distance/float32_simd/384`                         | 59.7 ns         | 14.7 ns        |
| `binary_cosine_distance/float32_simd/768`                         | 101.3 ns        | 24.7 ns        |
| `cached_embedding_delegate/native_wrong_model_disabled_64x4096`   | 233.5 ns        | 157.6 ns       |
| `int4_cosine_distance/float32_simd/1024`                          | 129.1 ns        | 31.3 ns        |
| `int4_cosine_distance/float32_simd/1536`                          | 186.8 ns        | 44.9 ns        |
| `int4_cosine_distance/float32_simd/384`                           | 59.6 ns         | 14.7 ns        |
| `int4_cosine_distance/float32_simd/768`                           | 101.5 ns        | 24.3 ns        |
| `int4_cosine_distance/int4/1024`                                  | 123.8 ns        | 189.2 ns       |
| `int4_cosine_distance/int4/1536`                                  | 180.9 ns        | 284.1 ns       |
| `int4_cosine_distance/int4/384`                                   | 53.8 ns         | 75.2 ns        |
| `int4_cosine_distance/int4/768`                                   | 96.4 ns         | 144.1 ns       |
| `int8_batch_cosine/float32_simd/10`                               | 574.2 ns        | 142.5 ns       |
| `int8_batch_cosine/float32_simd/100`                              | 6.94 µs         | 2.31 µs        |
| `int8_batch_cosine/float32_simd/1000`                             | 92.02 µs        | 32.15 µs       |
| `int8_batch_cosine/int8_loop/10`                                  | 166.6 ns        | 83.8 ns        |
| `int8_batch_cosine/int8_loop/100`                                 | 1.74 µs         | 836.3 ns       |
| `int8_batch_cosine/int8_loop/1000`                                | 20.38 µs        | 9.09 µs        |
| `int8_prepared_dot_product/per_call/1024`                         | 2.46 µs         | 838.6 ns       |
| `int8_prepared_dot_product/per_call/127`                          | 356.9 ns        | 139.8 ns       |
| `int8_prepared_dot_product/per_call/128`                          | 347.6 ns        | 126.5 ns       |
| `int8_prepared_dot_product/per_call/129`                          | 357.0 ns        | 133.5 ns       |
| `int8_prepared_dot_product/per_call/384`                          | 940.8 ns        | 327.8 ns       |
| `int8_prepared_dot_product/per_call/768`                          | 1.85 µs         | 640.4 ns       |
| `int8_prepared_dot_product/prepared/1024`                         | 31.3 ns         | 14.4 ns        |
| `int8_prepared_dot_product/prepared/127`                          | 12.5 ns         | 9.3 ns         |
| `int8_prepared_dot_product/prepared/128`                          | 8.4 ns          | 5.6 ns         |
| `int8_prepared_dot_product/prepared/129`                          | 9.0 ns          | 5.8 ns         |
| `int8_prepared_dot_product/prepared/384`                          | 14.6 ns         | 7.6 ns         |
| `int8_prepared_dot_product/prepared/768`                          | 24.7 ns         | 12.3 ns        |
| `int8_quantization/quantize/1024`                                 | 2.42 µs         | 821.2 ns       |
| `int8_quantization/quantize/1536`                                 | 3.60 µs         | 1.23 µs        |
| `int8_quantization/quantize/384`                                  | 941.0 ns        | 334.1 ns       |
| `int8_quantization/quantize/768`                                  | 1.83 µs         | 636.4 ns       |
| `int8_raw_dot_product/dot_product_i8/1024`                        | 30.7 ns         | 14.2 ns        |
| `int8_raw_dot_product/dot_product_i8/127`                         | 11.7 ns         | 9.1 ns         |
| `int8_raw_dot_product/dot_product_i8/128`                         | 8.2 ns          | 5.4 ns         |
| `int8_raw_dot_product/dot_product_i8/129`                         | 8.6 ns          | 5.7 ns         |
| `int8_raw_dot_product/dot_product_i8/384`                         | 14.2 ns         | 7.5 ns         |
| `int8_raw_dot_product/dot_product_i8/768`                         | 24.4 ns         | 12.0 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/1024`                    | 28.1 ns         | 13.2 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/127`                     | 10.1 ns         | 7.7 ns         |
| `int8_raw_dot_product/dot_product_i8_raw/128`                     | 6.5 ns          | 4.2 ns         |
| `int8_raw_dot_product/dot_product_i8_raw/129`                     | 7.1 ns          | 4.5 ns         |
| `int8_raw_dot_product/dot_product_i8_raw/384`                     | 12.5 ns         | 6.4 ns         |
| `int8_raw_dot_product/dot_product_i8_raw/768`                     | 21.9 ns         | 10.8 ns        |
| `int8_vs_float32_cosine/float32_simd/1024`                        | 129.0 ns        | 31.6 ns        |
| `int8_vs_float32_cosine/float32_simd/1536`                        | 186.2 ns        | 44.7 ns        |
| `int8_vs_float32_cosine/float32_simd/384`                         | 59.0 ns         | 13.9 ns        |
| `int8_vs_float32_cosine/float32_simd/768`                         | 99.9 ns         | 21.9 ns        |
| `int8_vs_float32_cosine/int8/1024`                                | 32.9 ns         | 14.7 ns        |
| `int8_vs_float32_cosine/int8/1536`                                | 46.1 ns         | 26.4 ns        |
| `int8_vs_float32_cosine/int8/384`                                 | 16.4 ns         | 8.1 ns         |
| `int8_vs_float32_cosine/int8/768`                                 | 27.2 ns         | 12.6 ns        |
| `memory_size/search_1000_float32`                                 | 57.51 µs        | 16.71 µs       |
| `memory_size/search_1000_int8`                                    | 16.11 µs        | 7.51 µs        |
| `simd_batch_cosine/scalar_loop/10`                                | 6.38 µs         | 4.35 µs        |
| `simd_batch_cosine/scalar_loop/100`                               | 63.98 µs        | 43.36 µs       |
| `simd_batch_cosine/scalar_loop/1000`                              | 647.61 µs       | 437.37 µs      |
| `simd_batch_cosine/simd_batch/10`                                 | 576.6 ns        | 139.7 ns       |
| `simd_batch_cosine/simd_batch/100`                                | 6.92 µs         | 2.30 µs        |
| `simd_batch_cosine/simd_batch/1000`                               | 91.36 µs        | 31.92 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_1000c`    | 145.29 µs       | 43.08 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_16c`      | 2.10 µs         | 590.6 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_256c`     | 35.89 µs        | 10.49 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_4c`       | 521.5 ns        | 153.8 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_64c`      | 8.29 µs         | 2.28 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_1000c`     | 58.52 µs        | 20.37 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_16c`       | 922.4 ns        | 271.1 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_256c`      | 15.16 µs        | 4.57 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_4c`        | 237.9 ns        | 78.7 ns        |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_64c`       | 3.74 µs         | 1.18 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_1000c`     | 106.32 µs       | 33.77 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_16c`       | 1.61 µs         | 468.4 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_256c`      | 29.73 µs        | 7.40 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_4c`        | 407.7 ns        | 116.7 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_64c`       | 6.46 µs         | 1.84 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_1000c`   | 145.20 µs       | 42.45 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_16c`     | 2.11 µs         | 566.8 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_256c`    | 38.09 µs        | 10.15 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_4c`      | 518.6 ns        | 139.3 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_64c`     | 8.26 µs         | 2.22 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_1000c`    | 57.96 µs        | 17.71 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_16c`      | 916.1 ns        | 230.6 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_256c`     | 15.02 µs        | 3.87 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_4c`       | 235.3 ns        | 69.3 ns        |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_64c`      | 3.71 µs         | 997.9 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_1000c`    | 106.77 µs       | 32.58 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_16c`      | 1.61 µs         | 422.8 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_256c`     | 30.05 µs        | 7.02 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_4c`       | 405.1 ns        | 106.1 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_64c`      | 6.44 µs         | 1.72 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_1000c` | 144.56 µs       | 43.05 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_16c`   | 2.09 µs         | 582.5 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_256c`  | 36.40 µs        | 10.50 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_4c`    | 519.7 ns        | 142.8 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_64c`   | 8.25 µs         | 2.28 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_1000c`  | 58.18 µs        | 20.60 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_16c`    | 920.1 ns        | 269.5 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_256c`   | 14.79 µs        | 4.55 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_4c`     | 237.7 ns        | 72.5 ns        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_64c`    | 3.72 µs         | 1.15 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_1000c`  | 106.48 µs       | 33.62 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_16c`    | 1.61 µs         | 460.2 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_256c`   | 29.37 µs        | 7.32 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_4c`     | 406.5 ns        | 117.0 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_64c`    | 6.42 µs         | 1.85 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_1000c`    | 95.93 µs        | 44.97 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_16c`      | 1.42 µs         | 559.7 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_256c`     | 26.76 µs        | 10.27 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_4c`       | 299.6 ns        | 133.9 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_64c`      | 6.05 µs         | 2.19 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_1000c`     | 38.55 µs        | 19.02 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_16c`       | 485.0 ns        | 232.6 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_256c`      | 9.69 µs         | 4.01 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_4c`        | 132.4 ns        | 62.3 ns        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_64c`       | 2.47 µs         | 1.02 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_1000c`     | 73.63 µs        | 34.40 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_16c`       | 991.3 ns        | 430.9 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_256c`      | 22.44 µs        | 6.92 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_4c`        | 231.2 ns        | 107.0 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_64c`       | 4.73 µs         | 1.72 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_1000c`       | 144.80 µs       | 41.77 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_16c`         | 2.08 µs         | 554.2 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_256c`        | 37.43 µs        | 9.87 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_4c`          | 516.2 ns        | 132.6 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_64c`         | 8.21 µs         | 2.18 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/384d_1000c`        | 57.72 µs        | 17.84 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_16c`          | 915.3 ns        | 230.6 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_256c`         | 14.61 µs        | 3.86 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/384d_4c`           | 235.3 ns        | 68.2 ns        |
| `simd_batch_cosine_normalized_query/simd_batch/384d_64c`          | 3.69 µs         | 1.00 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_1000c`        | 106.84 µs       | 32.18 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_16c`          | 1.61 µs         | 428.6 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_256c`         | 29.13 µs        | 6.88 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_4c`           | 403.9 ns        | 110.2 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_64c`          | 6.38 µs         | 1.70 µs        |
| `simd_batch_dot_product/scalar_loop/10`                           | 2.14 µs         | 1.45 µs        |
| `simd_batch_dot_product/scalar_loop/100`                          | 21.39 µs        | 14.43 µs       |
| `simd_batch_dot_product/scalar_loop/1000`                         | 221.07 µs       | 145.53 µs      |
| `simd_batch_dot_product/simd_batch/10`                            | 331.0 ns        | 129.2 ns       |
| `simd_batch_dot_product/simd_batch/100`                           | 5.72 µs         | 2.22 µs        |
| `simd_batch_dot_product/simd_batch/1000`                          | 90.73 µs        | 30.16 µs       |
| `simd_cosine_similarity/scalar/1024`                              | 1.77 µs         | 1.30 µs        |
| `simd_cosine_similarity/scalar/1536`                              | 2.68 µs         | 1.97 µs        |
| `simd_cosine_similarity/scalar/384`                               | 643.5 ns        | 433.9 ns       |
| `simd_cosine_similarity/scalar/768`                               | 1.32 µs         | 946.4 ns       |
| `simd_cosine_similarity/simd/1024`                                | 130.1 ns        | 27.9 ns        |
| `simd_cosine_similarity/simd/1536`                                | 186.0 ns        | 40.7 ns        |
| `simd_cosine_similarity/simd/384`                                 | 58.6 ns         | 17.7 ns        |
| `simd_cosine_similarity/simd/768`                                 | 101.0 ns        | 26.4 ns        |
| `simd_dot_product/scalar/1024`                                    | 593.5 ns        | 429.1 ns       |
| `simd_dot_product/scalar/1536`                                    | 892.4 ns        | 656.7 ns       |
| `simd_dot_product/scalar/384`                                     | 213.1 ns        | 144.6 ns       |
| `simd_dot_product/scalar/768`                                     | 439.7 ns        | 315.4 ns       |
| `simd_dot_product/simd/1024`                                      | 74.3 ns         | 30.1 ns        |
| `simd_dot_product/simd/1536`                                      | 107.4 ns        | 44.1 ns        |
| `simd_dot_product/simd/384`                                       | 31.9 ns         | 9.3 ns         |
| `simd_dot_product/simd/768`                                       | 55.3 ns         | 23.1 ns        |
| `simd_euclidean_distance/scalar/1024`                             | 595.5 ns        | 433.2 ns       |
| `simd_euclidean_distance/scalar/1536`                             | 895.9 ns        | 661.3 ns       |
| `simd_euclidean_distance/scalar/384`                              | 216.9 ns        | 148.8 ns       |
| `simd_euclidean_distance/scalar/768`                              | 443.8 ns        | 319.6 ns       |
| `simd_euclidean_distance/simd/1024`                               | 117.3 ns        | 30.5 ns        |
| `simd_euclidean_distance/simd/1536`                               | 170.7 ns        | 34.9 ns        |
| `simd_euclidean_distance/simd/384`                                | 48.5 ns         | 12.7 ns        |
| `simd_euclidean_distance/simd/768`                                | 89.3 ns         | 23.3 ns        |
| `simd_manhattan_distance/scalar/1024`                             | 591.9 ns        | 435.4 ns       |
| `simd_manhattan_distance/scalar/1536`                             | 891.4 ns        | 661.1 ns       |
| `simd_manhattan_distance/scalar/384`                              | 212.4 ns        | 148.9 ns       |
| `simd_manhattan_distance/scalar/768`                              | 438.8 ns        | 329.4 ns       |
| `simd_manhattan_distance/simd/1024`                               | 85.1 ns         | 24.7 ns        |
| `simd_manhattan_distance/simd/1536`                               | 127.2 ns        | 34.0 ns        |
| `simd_manhattan_distance/simd/384`                                | 36.6 ns         | 12.3 ns        |
| `simd_manhattan_distance/simd/768`                                | 66.7 ns         | 24.6 ns        |
| `simd_normalize/scalar/1024`                                      | 692.4 ns        | 504.5 ns       |
| `simd_normalize/scalar/1536`                                      | 1.03 µs         | 760.6 ns       |
| `simd_normalize/scalar/384`                                       | 266.7 ns        | 193.5 ns       |
| `simd_normalize/scalar/768`                                       | 522.7 ns        | 378.4 ns       |
| `simd_normalize/simd/1024`                                        | 158.8 ns        | 75.4 ns        |
| `simd_normalize/simd/1536`                                        | 233.5 ns        | 107.1 ns       |
| `simd_normalize/simd/384`                                         | 76.7 ns         | 28.8 ns        |
| `simd_normalize/simd/768`                                         | 125.5 ns        | 63.1 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/1024`               | 129.2 ns        | 31.7 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/384`                | 58.6 ns         | 14.0 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/768`                | 99.9 ns         | 24.4 ns        |
| `simd_normalized_cosine_fast_path/dot_product/1024`               | 71.5 ns         | 30.3 ns        |
| `simd_normalized_cosine_fast_path/dot_product/384`                | 30.5 ns         | 9.5 ns         |
| `simd_normalized_cosine_fast_path/dot_product/768`                | 55.1 ns         | 22.7 ns        |
| `simd_prepared_query_normalized_cosine/dot_product_loop/1024`     | 96.33 µs        | 44.32 µs       |
| `simd_prepared_query_normalized_cosine/dot_product_loop/384`      | 38.89 µs        | 17.71 µs       |
| `simd_prepared_query_normalized_cosine/dot_product_loop/768`      | 72.61 µs        | 34.38 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/1024` | 146.99 µs       | 42.14 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/384`  | 61.04 µs        | 20.81 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/768`  | 108.06 µs       | 36.38 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/1024`   | 146.04 µs       | 46.73 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/384`    | 61.07 µs        | 20.70 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/768`    | 107.02 µs       | 32.65 µs       |
| `simd_query_batch_dot_product/pair_loop/128d_16c`                 | 219.0 ns        | 106.6 ns       |
| `simd_query_batch_dot_product/pair_loop/128d_256c`                | 3.79 µs         | 1.75 µs        |
| `simd_query_batch_dot_product/pair_loop/128d_4c`                  | 65.7 ns         | 32.2 ns        |
| `simd_query_batch_dot_product/pair_loop/128d_64c`                 | 823.4 ns        | 424.7 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_16c`                 | 483.4 ns        | 242.4 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_256c`                | 9.62 µs         | 4.07 µs        |
| `simd_query_batch_dot_product/pair_loop/384d_4c`                  | 132.1 ns        | 66.1 ns        |
| `simd_query_batch_dot_product/pair_loop/384d_64c`                 | 2.47 µs         | 1.03 µs        |
| `simd_query_batch_dot_product/pair_loop/768d_16c`                 | 960.1 ns        | 435.4 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_256c`                | 22.89 µs        | 6.93 µs        |
| `simd_query_batch_dot_product/pair_loop/768d_4c`                  | 231.1 ns        | 105.2 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_64c`                 | 4.70 µs         | 1.72 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_16c`                | 137.1 ns        | 69.1 ns        |
| `simd_query_batch_dot_product/simd_batch/128d_256c`               | 3.05 µs         | 1.18 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_4c`                 | 46.0 ns         | 25.3 ns        |
| `simd_query_batch_dot_product/simd_batch/128d_64c`                | 511.8 ns        | 261.0 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_16c`                | 326.8 ns        | 154.9 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_256c`               | 8.43 µs         | 3.05 µs        |
| `simd_query_batch_dot_product/simd_batch/384d_4c`                 | 94.3 ns         | 47.8 ns        |
| `simd_query_batch_dot_product/simd_batch/384d_64c`                | 2.15 µs         | 769.9 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_16c`                | 649.6 ns        | 317.7 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_256c`               | 18.28 µs        | 5.73 µs        |
| `simd_query_batch_dot_product/simd_batch/768d_4c`                 | 164.4 ns        | 79.2 ns        |
| `simd_query_batch_dot_product/simd_batch/768d_64c`                | 3.82 µs         | 1.43 µs        |
| `simd_squared_euclidean_fast_path/euclidean_full/1024`            | 116.9 ns        | 30.5 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/384`             | 48.3 ns         | 12.2 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/768`             | 89.2 ns         | 23.4 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/1024`         | 113.7 ns        | 30.5 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/384`          | 45.9 ns         | 12.2 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/768`          | 86.7 ns         | 23.3 ns        |
| `simd_throughput_384/cosine_similarity`                           | 58.8 ns         | 13.9 ns        |
| `simd_throughput_384/dot_product`                                 | 32.8 ns         | 9.4 ns         |
| `simd_throughput_384/euclidean_distance`                          | 48.4 ns         | 9.7 ns         |
| `simd_throughput_384/normalize`                                   | 118.9 ns        | 57.0 ns        |
| `tier_prepared_batch_sizes/int4_batch_prepared/10`                | 517.8 ns        | 741.8 ns       |
| `tier_prepared_batch_sizes/int4_batch_prepared/100`               | 5.02 µs         | 7.26 µs        |
| `tier_prepared_batch_sizes/int4_batch_prepared/1000`              | 50.04 µs        | 72.38 µs       |
| `tier_prepared_batch_sizes/int4_query_per_call/10`                | 9.36 µs         | 3.91 µs        |
| `tier_prepared_batch_sizes/int4_query_per_call/100`               | 93.32 µs        | 39.17 µs       |
| `tier_prepared_batch_sizes/int4_query_per_call/1000`              | 933.43 µs       | 391.49 µs      |
| `tier_prepared_batch_sizes/int8_batch_prepared/10`                | 173.8 ns        | 102.3 ns       |
| `tier_prepared_batch_sizes/int8_batch_prepared/100`               | 1.54 µs         | 802.0 ns       |
| `tier_prepared_batch_sizes/int8_batch_prepared/1000`              | 17.56 µs        | 7.90 µs        |
| `tier_prepared_batch_sizes/int8_query_per_call/10`                | 9.72 µs         | 3.45 µs        |
| `tier_prepared_batch_sizes/int8_query_per_call/100`               | 97.07 µs        | 34.90 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/1000`              | 969.82 µs       | 343.43 µs      |
| `tier_prepared_query/binary_query_once_1000`                      | 8.17 µs         | 28.57 µs       |
| `tier_prepared_query/binary_query_per_call_1000`                  | 1.11 ms         | 243.29 µs      |
| `tier_prepared_query/int4_query_once_1000`                        | 52.68 µs        | 74.17 µs       |
| `tier_prepared_query/int4_query_per_call_1000`                    | 947.11 µs       | 392.28 µs      |
| `tier_prepared_query/int8_query_once_1000`                        | 18.75 µs        | 9.37 µs        |
| `tier_prepared_query/int8_query_per_call_1000`                    | 947.63 µs       | 330.48 µs      |

</details>

<!-- generated:benchmarks:end -->
