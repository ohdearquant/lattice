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
- bench (5): `embeddings`, `simd_bench`, `simd_opt_bench`, `simd` (requires `native`), `simsimd_comparison`

<!-- generated:layout:end -->

<!-- generated:stats:begin -->

### Statistics

Rust lines by kind (code, comment, blank; doc comments count as comment), test functions (`#[test]` and `#[<runtime>::test]` attributes), items declared `pub` (not `pub(crate)`; reachability through private modules is not resolved) and occurrences of the `unsafe` keyword outside comments and strings.

| Area        | Files | Code  | Comment | Blank | Tests | `pub` items | `unsafe` |
| ----------- | ----- | ----- | ------- | ----- | ----- | ----------- | -------- |
| `src/`      | 34    | 11818 | 1415    | 1093  | 278   | 193         | 13       |
| `tests/`    | 14    | 2883  | 500     | 468   | 109   | 0           | 1        |
| `benches/`  | 5     | 1616  | 286     | 366   | 0     | 0           | 0        |
| `examples/` | 4     | 342   | 79      | 64    | 0     | 0           | 0        |
| total       | 57    | 16659 | 2280    | 1991  | 387   | 193         | 14       |

Other tracked files (the README excluded):

| Extension | Files | Size      |
| --------- | ----- | --------- |
| `.json`   | 10    | 455.4 KiB |
| `.md`     | 10    | 143.3 KiB |
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

- `aarch64-linux`: commit `163b257e1733`, recorded 2026-11-01T19:03:53+00:00
- `x86_64-linux`: commit `163b257e1733`, recorded 2026-11-01T19:05:56+00:00

<details>
<summary>255 benchmarks</summary>

| Benchmark                                                         | `aarch64-linux` | `x86_64-linux` |
| ----------------------------------------------------------------- | --------------- | -------------- |
| `binary_cosine_distance/binary/1024`                              | 10.9 ns         | 118.8 ns       |
| `binary_cosine_distance/binary/1536`                              | 15.7 ns         | 169.6 ns       |
| `binary_cosine_distance/binary/384`                               | 5.5 ns          | 54.4 ns        |
| `binary_cosine_distance/binary/768`                               | 9.2 ns          | 92.8 ns        |
| `binary_cosine_distance/float32_simd/1024`                        | 129.3 ns        | 92.0 ns        |
| `binary_cosine_distance/float32_simd/1536`                        | 186.7 ns        | 120.6 ns       |
| `binary_cosine_distance/float32_simd/384`                         | 59.5 ns         | 45.1 ns        |
| `binary_cosine_distance/float32_simd/768`                         | 101.2 ns        | 70.1 ns        |
| `cached_embedding_delegate/native_wrong_model_disabled_64x4096`   | 236.2 ns        | 263.8 ns       |
| `int4_cosine_distance/float32_simd/1024`                          | 129.3 ns        | 86.8 ns        |
| `int4_cosine_distance/float32_simd/1536`                          | 186.7 ns        | 120.9 ns       |
| `int4_cosine_distance/float32_simd/384`                           | 59.6 ns         | 44.7 ns        |
| `int4_cosine_distance/float32_simd/768`                           | 101.2 ns        | 70.0 ns        |
| `int4_cosine_distance/int4/1024`                                  | 123.3 ns        | 328.6 ns       |
| `int4_cosine_distance/int4/1536`                                  | 181.3 ns        | 485.1 ns       |
| `int4_cosine_distance/int4/384`                                   | 53.6 ns         | 135.8 ns       |
| `int4_cosine_distance/int4/768`                                   | 96.7 ns         | 252.8 ns       |
| `int8_batch_cosine/float32_simd/10`                               | 573.3 ns        | 406.0 ns       |
| `int8_batch_cosine/float32_simd/100`                              | 6.91 µs         | 4.33 µs        |
| `int8_batch_cosine/float32_simd/1000`                             | 90.38 µs        | 58.69 µs       |
| `int8_batch_cosine/int8_loop/10`                                  | 165.7 ns        | 168.1 ns       |
| `int8_batch_cosine/int8_loop/100`                                 | 1.71 µs         | 1.73 µs        |
| `int8_batch_cosine/int8_loop/1000`                                | 20.03 µs        | 18.78 µs       |
| `int8_prepared_dot_product/per_call/1024`                         | 2.45 µs         | 1.87 µs        |
| `int8_prepared_dot_product/per_call/127`                          | 362.8 ns        | 300.8 ns       |
| `int8_prepared_dot_product/per_call/128`                          | 347.1 ns        | 281.3 ns       |
| `int8_prepared_dot_product/per_call/129`                          | 353.4 ns        | 284.5 ns       |
| `int8_prepared_dot_product/per_call/384`                          | 949.9 ns        | 744.1 ns       |
| `int8_prepared_dot_product/per_call/768`                          | 1.85 µs         | 1.42 µs        |
| `int8_prepared_dot_product/prepared/1024`                         | 31.5 ns         | 35.5 ns        |
| `int8_prepared_dot_product/prepared/127`                          | 12.2 ns         | 16.3 ns        |
| `int8_prepared_dot_product/prepared/128`                          | 8.6 ns          | 9.0 ns         |
| `int8_prepared_dot_product/prepared/129`                          | 9.1 ns          | 9.5 ns         |
| `int8_prepared_dot_product/prepared/384`                          | 14.8 ns         | 15.8 ns        |
| `int8_prepared_dot_product/prepared/768`                          | 25.0 ns         | 27.0 ns        |
| `int8_quantization/quantize/1024`                                 | 2.42 µs         | 1.86 µs        |
| `int8_quantization/quantize/1536`                                 | 3.60 µs         | 2.74 µs        |
| `int8_quantization/quantize/384`                                  | 935.4 ns        | 723.5 ns       |
| `int8_quantization/quantize/768`                                  | 1.83 µs         | 1.39 µs        |
| `int8_raw_dot_product/dot_product_i8/1024`                        | 30.7 ns         | 35.1 ns        |
| `int8_raw_dot_product/dot_product_i8/127`                         | 11.8 ns         | 16.3 ns        |
| `int8_raw_dot_product/dot_product_i8/128`                         | 8.0 ns          | 9.2 ns         |
| `int8_raw_dot_product/dot_product_i8/129`                         | 8.5 ns          | 9.6 ns         |
| `int8_raw_dot_product/dot_product_i8/384`                         | 14.2 ns         | 15.7 ns        |
| `int8_raw_dot_product/dot_product_i8/768`                         | 24.0 ns         | 26.9 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/1024`                    | 28.1 ns         | 30.8 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/127`                     | 10.0 ns         | 14.3 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/128`                     | 6.5 ns          | 6.7 ns         |
| `int8_raw_dot_product/dot_product_i8_raw/129`                     | 7.2 ns          | 7.2 ns         |
| `int8_raw_dot_product/dot_product_i8_raw/384`                     | 12.4 ns         | 13.6 ns        |
| `int8_raw_dot_product/dot_product_i8_raw/768`                     | 21.9 ns         | 24.2 ns        |
| `int8_vs_float32_cosine/float32_simd/1024`                        | 129.0 ns        | 85.7 ns        |
| `int8_vs_float32_cosine/float32_simd/1536`                        | 185.8 ns        | 119.4 ns       |
| `int8_vs_float32_cosine/float32_simd/384`                         | 58.6 ns         | 43.8 ns        |
| `int8_vs_float32_cosine/float32_simd/768`                         | 99.9 ns         | 68.9 ns        |
| `int8_vs_float32_cosine/int8/1024`                                | 32.8 ns         | 38.3 ns        |
| `int8_vs_float32_cosine/int8/1536`                                | 46.3 ns         | 50.4 ns        |
| `int8_vs_float32_cosine/int8/384`                                 | 17.0 ns         | 17.9 ns        |
| `int8_vs_float32_cosine/int8/768`                                 | 27.2 ns         | 29.5 ns        |
| `memory_size/search_1000_float32`                                 | 57.27 µs        | 41.61 µs       |
| `memory_size/search_1000_int8`                                    | 14.99 µs        | 15.56 µs       |
| `simd_batch_cosine/scalar_loop/10`                                | 6.39 µs         | 10.76 µs       |
| `simd_batch_cosine/scalar_loop/100`                               | 63.95 µs        | 107.42 µs      |
| `simd_batch_cosine/scalar_loop/1000`                              | 646.27 µs       | 1.08 ms        |
| `simd_batch_cosine/simd_batch/10`                                 | 575.4 ns        | 406.5 ns       |
| `simd_batch_cosine/simd_batch/100`                                | 6.96 µs         | 4.34 µs        |
| `simd_batch_cosine/simd_batch/1000`                               | 88.50 µs        | 58.13 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_1000c`    | 143.92 µs       | 85.88 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_16c`      | 2.08 µs         | 1.33 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_256c`     | 36.68 µs        | 23.15 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_4c`       | 521.3 ns        | 337.1 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/1024d_64c`      | 8.25 µs         | 5.27 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_1000c`     | 58.30 µs        | 41.40 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_16c`       | 921.9 ns        | 643.5 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_256c`      | 14.84 µs        | 10.04 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_4c`        | 238.0 ns        | 176.0 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/384d_64c`       | 3.72 µs         | 2.53 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_1000c`     | 105.54 µs       | 67.97 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_16c`       | 1.62 µs         | 1.06 µs        |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_256c`      | 28.73 µs        | 17.54 µs       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_4c`        | 407.8 ns        | 265.6 ns       |
| `simd_batch_cosine_non_normalized_query/pair_loop/768d_64c`       | 6.44 µs         | 4.17 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_1000c`   | 143.94 µs       | 84.96 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_16c`     | 2.08 µs         | 1.31 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_256c`    | 37.13 µs        | 22.78 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_4c`      | 518.8 ns        | 333.8 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/1024d_64c`     | 8.22 µs         | 5.22 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_1000c`    | 57.56 µs        | 40.68 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_16c`      | 915.8 ns        | 630.0 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_256c`     | 14.64 µs        | 9.87 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_4c`       | 235.8 ns        | 170.4 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/384d_64c`      | 3.69 µs         | 2.48 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_1000c`    | 105.88 µs       | 67.13 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_16c`      | 1.61 µs         | 1.04 µs        |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_256c`     | 28.75 µs        | 17.29 µs       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_4c`       | 405.3 ns        | 264.2 ns       |
| `simd_batch_cosine_non_normalized_query/simd_batch/768d_64c`      | 6.42 µs         | 4.13 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_1000c` | 143.81 µs       | 85.85 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_16c`   | 2.07 µs         | 1.32 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_256c`  | 36.13 µs        | 23.06 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_4c`    | 519.5 ns        | 334.3 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/1024d_64c`   | 8.23 µs         | 5.26 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_1000c`  | 57.80 µs        | 41.35 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_16c`    | 920.6 ns        | 640.9 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_256c`   | 14.73 µs        | 10.06 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_4c`     | 237.8 ns        | 174.5 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/384d_64c`    | 3.70 µs         | 2.54 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_1000c`  | 106.06 µs       | 67.82 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_16c`    | 1.60 µs         | 1.05 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_256c`   | 28.29 µs        | 17.97 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_4c`     | 406.2 ns        | 267.4 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_cosine/768d_64c`    | 6.39 µs         | 4.18 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_1000c`    | 96.01 µs        | 69.21 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_16c`      | 1.38 µs         | 1.05 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_256c`     | 27.16 µs        | 18.78 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_4c`       | 296.6 ns        | 268.7 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/1024d_64c`      | 6.21 µs         | 4.19 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_1000c`     | 37.77 µs        | 31.78 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_16c`       | 498.6 ns        | 476.4 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_256c`      | 9.52 µs         | 7.64 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_4c`        | 132.2 ns        | 128.3 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/384d_64c`       | 2.43 µs         | 1.91 µs        |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_1000c`     | 72.96 µs        | 53.93 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_16c`       | 970.8 ns        | 845.6 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_256c`      | 21.21 µs        | 14.24 µs       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_4c`        | 232.0 ns        | 220.3 ns       |
| `simd_batch_cosine_normalized_query/pair_loop_dot/768d_64c`       | 4.59 µs         | 3.27 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_1000c`       | 143.75 µs       | 84.87 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_16c`         | 2.08 µs         | 1.31 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_256c`        | 35.40 µs        | 22.78 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_4c`          | 516.6 ns        | 331.8 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/1024d_64c`         | 8.19 µs         | 5.22 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/384d_1000c`        | 57.41 µs        | 41.06 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_16c`          | 914.0 ns        | 628.7 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_256c`         | 14.60 µs        | 9.89 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/384d_4c`           | 235.3 ns        | 170.2 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/384d_64c`          | 3.67 µs         | 2.48 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_1000c`        | 106.67 µs       | 66.94 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_16c`          | 1.60 µs         | 1.04 µs        |
| `simd_batch_cosine_normalized_query/simd_batch/768d_256c`         | 29.45 µs        | 17.80 µs       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_4c`           | 404.0 ns        | 264.7 ns       |
| `simd_batch_cosine_normalized_query/simd_batch/768d_64c`          | 6.36 µs         | 4.13 µs        |
| `simd_batch_dot_product/scalar_loop/10`                           | 2.14 µs         | 3.61 µs        |
| `simd_batch_dot_product/scalar_loop/100`                          | 21.40 µs        | 36.27 µs       |
| `simd_batch_dot_product/scalar_loop/1000`                         | 220.35 µs       | 375.03 µs      |
| `simd_batch_dot_product/simd_batch/10`                            | 342.6 ns        | 310.5 ns       |
| `simd_batch_dot_product/simd_batch/100`                           | 5.59 µs         | 3.62 µs        |
| `simd_batch_dot_product/simd_batch/1000`                          | 92.24 µs        | 49.70 µs       |
| `simd_cosine_similarity/scalar/1024`                              | 1.77 µs         | 3.11 µs        |
| `simd_cosine_similarity/scalar/1536`                              | 2.68 µs         | 4.73 µs        |
| `simd_cosine_similarity/scalar/384`                               | 643.3 ns        | 1.09 µs        |
| `simd_cosine_similarity/scalar/768`                               | 1.32 µs         | 2.31 µs        |
| `simd_cosine_similarity/simd/1024`                                | 130.2 ns        | 85.8 ns        |
| `simd_cosine_similarity/simd/1536`                                | 185.9 ns        | 119.7 ns       |
| `simd_cosine_similarity/simd/384`                                 | 58.5 ns         | 44.1 ns        |
| `simd_cosine_similarity/simd/768`                                 | 101.0 ns        | 68.8 ns        |
| `simd_dot_product/scalar/1024`                                    | 593.3 ns        | 1.05 µs        |
| `simd_dot_product/scalar/1536`                                    | 892.3 ns        | 1.59 µs        |
| `simd_dot_product/scalar/384`                                     | 213.2 ns        | 370.2 ns       |
| `simd_dot_product/scalar/768`                                     | 439.9 ns        | 775.7 ns       |
| `simd_dot_product/simd/1024`                                      | 73.5 ns         | 66.0 ns        |
| `simd_dot_product/simd/1536`                                      | 107.4 ns        | 96.8 ns        |
| `simd_dot_product/simd/384`                                       | 31.8 ns         | 27.2 ns        |
| `simd_dot_product/simd/768`                                       | 55.5 ns         | 60.7 ns        |
| `simd_euclidean_distance/scalar/1024`                             | 601.4 ns        | 1.06 µs        |
| `simd_euclidean_distance/scalar/1536`                             | 897.7 ns        | 1.60 µs        |
| `simd_euclidean_distance/scalar/384`                              | 218.5 ns        | 380.9 ns       |
| `simd_euclidean_distance/scalar/768`                              | 444.7 ns        | 785.9 ns       |
| `simd_euclidean_distance/simd/1024`                               | 117.0 ns        | 59.7 ns        |
| `simd_euclidean_distance/simd/1536`                               | 171.0 ns        | 83.6 ns        |
| `simd_euclidean_distance/simd/384`                                | 49.1 ns         | 34.2 ns        |
| `simd_euclidean_distance/simd/768`                                | 89.7 ns         | 56.7 ns        |
| `simd_manhattan_distance/scalar/1024`                             | 591.8 ns        | 1.05 µs        |
| `simd_manhattan_distance/scalar/1536`                             | 891.6 ns        | 1.59 µs        |
| `simd_manhattan_distance/scalar/384`                              | 212.4 ns        | 375.5 ns       |
| `simd_manhattan_distance/scalar/768`                              | 438.6 ns        | 780.1 ns       |
| `simd_manhattan_distance/simd/1024`                               | 84.9 ns         | 67.3 ns        |
| `simd_manhattan_distance/simd/1536`                               | 126.1 ns        | 98.0 ns        |
| `simd_manhattan_distance/simd/384`                                | 36.8 ns         | 28.8 ns        |
| `simd_manhattan_distance/simd/768`                                | 66.6 ns         | 52.2 ns        |
| `simd_normalize/scalar/1024`                                      | 687.2 ns        | 1.18 µs        |
| `simd_normalize/scalar/1536`                                      | 1.02 µs         | 1.77 µs        |
| `simd_normalize/scalar/384`                                       | 265.8 ns        | 445.5 ns       |
| `simd_normalize/scalar/768`                                       | 516.2 ns        | 886.3 ns       |
| `simd_normalize/simd/1024`                                        | 152.8 ns        | 148.2 ns       |
| `simd_normalize/simd/1536`                                        | 231.6 ns        | 197.6 ns       |
| `simd_normalize/simd/384`                                         | 70.1 ns         | 70.7 ns        |
| `simd_normalize/simd/768`                                         | 119.1 ns        | 111.5 ns       |
| `simd_normalized_cosine_fast_path/cosine_full/1024`               | 129.3 ns        | 85.9 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/384`                | 59.1 ns         | 44.9 ns        |
| `simd_normalized_cosine_fast_path/cosine_full/768`                | 100.0 ns        | 68.9 ns        |
| `simd_normalized_cosine_fast_path/dot_product/1024`               | 73.9 ns         | 66.6 ns        |
| `simd_normalized_cosine_fast_path/dot_product/384`                | 31.6 ns         | 32.3 ns        |
| `simd_normalized_cosine_fast_path/dot_product/768`                | 56.9 ns         | 51.0 ns        |
| `simd_prepared_query_normalized_cosine/dot_product_loop/1024`     | 96.22 µs        | 69.53 µs       |
| `simd_prepared_query_normalized_cosine/dot_product_loop/384`      | 38.33 µs        | 35.80 µs       |
| `simd_prepared_query_normalized_cosine/dot_product_loop/768`      | 72.95 µs        | 54.19 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/1024` | 145.81 µs       | 93.40 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/384`  | 60.73 µs        | 46.61 µs       |
| `simd_prepared_query_normalized_cosine/prepared_full_cosine/768`  | 108.68 µs       | 74.21 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/1024`   | 147.22 µs       | 93.90 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/384`    | 60.71 µs        | 46.59 µs       |
| `simd_prepared_query_normalized_cosine/prepared_meta_unit/768`    | 107.66 µs       | 74.53 µs       |
| `simd_query_batch_dot_product/pair_loop/128d_16c`                 | 218.0 ns        | 212.0 ns       |
| `simd_query_batch_dot_product/pair_loop/128d_256c`                | 3.80 µs         | 3.28 µs        |
| `simd_query_batch_dot_product/pair_loop/128d_4c`                  | 65.8 ns         | 68.6 ns        |
| `simd_query_batch_dot_product/pair_loop/128d_64c`                 | 825.5 ns        | 838.3 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_16c`                 | 492.1 ns        | 473.8 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_256c`                | 9.55 µs         | 7.63 µs        |
| `simd_query_batch_dot_product/pair_loop/384d_4c`                  | 134.8 ns        | 127.0 ns       |
| `simd_query_batch_dot_product/pair_loop/384d_64c`                 | 2.42 µs         | 1.89 µs        |
| `simd_query_batch_dot_product/pair_loop/768d_16c`                 | 975.8 ns        | 838.2 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_256c`                | 21.39 µs        | 14.06 µs       |
| `simd_query_batch_dot_product/pair_loop/768d_4c`                  | 232.1 ns        | 221.1 ns       |
| `simd_query_batch_dot_product/pair_loop/768d_64c`                 | 4.61 µs         | 3.27 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_16c`                | 137.2 ns        | 122.7 ns       |
| `simd_query_batch_dot_product/simd_batch/128d_256c`               | 3.05 µs         | 2.06 µs        |
| `simd_query_batch_dot_product/simd_batch/128d_4c`                 | 46.0 ns         | 45.9 ns        |
| `simd_query_batch_dot_product/simd_batch/128d_64c`                | 518.5 ns        | 485.1 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_16c`                | 326.5 ns        | 258.8 ns       |
| `simd_query_batch_dot_product/simd_batch/384d_256c`               | 8.42 µs         | 5.01 µs        |
| `simd_query_batch_dot_product/simd_batch/384d_4c`                 | 92.9 ns         | 78.8 ns        |
| `simd_query_batch_dot_product/simd_batch/384d_64c`                | 2.15 µs         | 1.23 µs        |
| `simd_query_batch_dot_product/simd_batch/768d_16c`                | 629.2 ns        | 595.5 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_256c`               | 17.86 µs        | 10.48 µs       |
| `simd_query_batch_dot_product/simd_batch/768d_4c`                 | 163.7 ns        | 148.6 ns       |
| `simd_query_batch_dot_product/simd_batch/768d_64c`                | 3.85 µs         | 2.37 µs        |
| `simd_squared_euclidean_fast_path/euclidean_full/1024`            | 116.9 ns        | 71.7 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/384`             | 48.9 ns         | 34.0 ns        |
| `simd_squared_euclidean_fast_path/euclidean_full/768`             | 89.7 ns         | 56.6 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/1024`         | 113.6 ns        | 67.2 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/384`          | 46.0 ns         | 29.1 ns        |
| `simd_squared_euclidean_fast_path/squared_euclidean/768`          | 86.8 ns         | 51.9 ns        |
| `simd_throughput_384/cosine_similarity`                           | 58.6 ns         | 44.2 ns        |
| `simd_throughput_384/dot_product`                                 | 31.5 ns         | 27.6 ns        |
| `simd_throughput_384/euclidean_distance`                          | 49.0 ns         | 34.3 ns        |
| `simd_throughput_384/normalize`                                   | 120.8 ns        | 120.0 ns       |
| `tier_prepared_batch_sizes/int4_batch_prepared/10`                | 513.7 ns        | 1.34 µs        |
| `tier_prepared_batch_sizes/int4_batch_prepared/100`               | 4.95 µs         | 13.26 µs       |
| `tier_prepared_batch_sizes/int4_batch_prepared/1000`              | 49.98 µs        | 132.03 µs      |
| `tier_prepared_batch_sizes/int4_query_per_call/10`                | 9.36 µs         | 8.04 µs        |
| `tier_prepared_batch_sizes/int4_query_per_call/100`               | 93.40 µs        | 80.26 µs       |
| `tier_prepared_batch_sizes/int4_query_per_call/1000`              | 934.09 µs       | 829.02 µs      |
| `tier_prepared_batch_sizes/int8_batch_prepared/10`                | 171.9 ns        | 200.2 ns       |
| `tier_prepared_batch_sizes/int8_batch_prepared/100`               | 1.53 µs         | 1.66 µs        |
| `tier_prepared_batch_sizes/int8_batch_prepared/1000`              | 15.61 µs        | 16.17 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/10`                | 9.71 µs         | 7.51 µs        |
| `tier_prepared_batch_sizes/int8_query_per_call/100`               | 96.58 µs        | 75.08 µs       |
| `tier_prepared_batch_sizes/int8_query_per_call/1000`              | 965.89 µs       | 751.36 µs      |
| `tier_prepared_query/binary_query_once_1000`                      | 7.95 µs         | 53.21 µs       |
| `tier_prepared_query/binary_query_per_call_1000`                  | 1.11 ms         | 523.72 µs      |
| `tier_prepared_query/int4_query_once_1000`                        | 52.56 µs        | 134.91 µs      |
| `tier_prepared_query/int4_query_per_call_1000`                    | 937.65 µs       | 826.76 µs      |
| `tier_prepared_query/int8_query_once_1000`                        | 18.29 µs        | 18.51 µs       |
| `tier_prepared_query/int8_query_per_call_1000`                    | 944.95 µs       | 731.38 µs      |

</details>

<!-- generated:benchmarks:end -->
