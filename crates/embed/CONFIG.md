# lattice-embed — CONFIG.md

Complete configuration reference for the lattice-embed embedding service — model selection, MRL truncation, caching, and SIMD dispatch.

---

## ModelConfig

**Source**: `foundation/embed/src/model.rs:382`

Runtime configuration pairing a model with an optional MRL truncation dimension. Two `ModelConfig` values with different `output_dim` produce **different embedding spaces** and must be stored in separate vector index namespaces.

### Fields

| Field        | Type             | Default         | `serde`             | Source of Default                                            |
| ------------ | ---------------- | --------------- | ------------------- | ------------------------------------------------------------ |
| `model`      | `EmbeddingModel` | `BgeSmallEnV15` | —                   | `EmbeddingModel::default()` (`#[default]` at `model.rs:104`) |
| `output_dim` | `Option<usize>`  | `None`          | `#[serde(default)]` | `None` = use model's native dimension                        |

**Default impl** at `model.rs:390`: `ModelConfig::new(EmbeddingModel::default())` — which sets `output_dim: None`.

### Field Details

#### `model` — default `BgeSmallEnV15`

**What it does**: Selects which embedding model to load and run.

**Why `BgeSmallEnV15` as the code default**: It's the lightest local model (384 dims, 33M params), suitable for fast testing and development. In production, the actual model is typically set via application configuration to `multilingual-e5-small` (same dimensions but multilingual). The enum default is rarely used directly — `EmbeddedConfig::memory()` and `EmbeddedConfig::file()` constructors set the model from config.

**Important**: The **production default is `multilingual-e5-small`, NOT `bge-small-en-v1.5`**. Configure the active model via your application's config. The `EmbeddingModel` enum `#[default]` is for code that doesn't go through an application config path.

#### `output_dim` — default `None`

**What it does**: When `Some(dim)`, truncates the model's native output to `dim` dimensions via Matryoshka Representation Learning (MRL). When `None`, uses the model's full native dimension.

**Why `None`**: Most models are used at their native resolution. MRL truncation is an optimization for storage/search speed that trades off quality — it should be an explicit decision, not a default.

**Validation** (`model.rs:416` `ModelConfig::validate()`):

- `output_dim` must be `>= MIN_MRL_OUTPUT_DIM` (32)
- `output_dim` must be `<= native_dimensions()`
- Non-MRL models (`supports_output_dim() == false`) reject any `output_dim`

**Where used**:

- `dimensions()` method returns `output_dim.unwrap_or(native_dim)` — always reflects actual output
- `to_embedding_key()` uses dimensions in the `EmbeddingKey` hash — different dims = different space
- Cache file path includes dim: `embed_{model}_{dim}d.bin`
- HNSW index stores one index per `EmbeddingKey` (per model+dim combo)

---

## EmbeddingModel Enum

**Source**: `foundation/embed/src/model.rs:102`

`#[derive(Default)]`, `#[non_exhaustive]`, `#[serde(rename_all = "snake_case")]`

### Supported Models

| Variant                        | Display Name             | HuggingFace ID                   | Native Dims | Max Tokens | Local?       | MRL? | Architecture  |
| ------------------------------ | ------------------------ | -------------------------------- | ----------- | ---------- | ------------ | ---- | ------------- |
| `BgeSmallEnV15` (code default) | `bge-small-en-v1.5`      | `BAAI/bge-small-en-v1.5`         | 384         | 512        | yes          | no   | BERT encoder  |
| `BgeBaseEnV15`                 | `bge-base-en-v1.5`       | `BAAI/bge-base-en-v1.5`          | 768         | 512        | yes          | no   | BERT encoder  |
| `BgeLargeEnV15`                | `bge-large-en-v1.5`      | `BAAI/bge-large-en-v1.5`         | 1024        | 512        | yes          | no   | BERT encoder  |
| `MultilingualE5Small`          | `multilingual-e5-small`  | `intfloat/multilingual-e5-small` | 384         | 512        | yes          | no   | BERT encoder  |
| `MultilingualE5Base`           | `multilingual-e5-base`   | `intfloat/multilingual-e5-base`  | 768         | 512        | yes          | no   | BERT encoder  |
| `Qwen3Embedding0_6B`           | `qwen3-embedding-0.6b`   | `Qwen/Qwen3-Embedding-0.6B`      | 1024        | 8192       | yes          | yes  | Qwen3 decoder |
| `Qwen3Embedding4B`             | `qwen3-embedding-4b`     | `Qwen/Qwen3-Embedding-4B`        | 2560        | 8192       | yes          | yes  | Qwen3 decoder |
| `TextEmbedding3Small`          | `text-embedding-3-small` | `text-embedding-3-small`         | 1536        | 8191       | **no** (API) | —    | OpenAI remote |

### Key Methods

| Method                   | Returns        | Notes                                                                    |
| ------------------------ | -------------- | ------------------------------------------------------------------------ |
| `native_dimensions()`    | `usize`        | Full output dimension (ignoring MRL) — `model.rs:143`                    |
| `dimensions()`           | `usize`        | Same as `native_dimensions()` (MRL is on `ModelConfig`) — `model.rs:165` |
| `is_local()`             | `bool`         | All except `TextEmbedding3Small` — `model.rs:171`                        |
| `is_remote()`            | `bool`         | Only `TextEmbedding3Small` — `model.rs:186`                              |
| `max_input_tokens()`     | `usize`        | Token limit for chunking/truncation — `model.rs:200`                     |
| `supports_output_dim()`  | `bool`         | Only Qwen3 variants — `model.rs:263`                                     |
| `query_instruction()`    | `Option<&str>` | Qwen3 requires instruction prefix — `model.rs:227`                       |
| `document_instruction()` | `Option<&str>` | Currently `None` for all — `model.rs:242`                                |
| `model_id()`             | `&str`         | HuggingFace ID — `model.rs:248`                                          |
| `key_version()`          | `&str`         | `"v3"` for Qwen3/OpenAI, `"v1.5"` for BGE/E5 — `model.rs:272`            |

### String Parsing

`FromStr` impl at `model.rs:306` accepts flexible input:

- Display names: `"bge-small-en-v1.5"`, `"multilingual-e5-small"`
- Short names: `"bge-small"`, `"small"`, `"qwen3"`, `"e5-small"`
- HuggingFace IDs: `"BAAI/bge-small-en-v1.5"`
- Case-insensitive, underscores normalized to hyphens

---

## MRL (Matryoshka Representation Learning)

MRL allows decoder-only models to produce truncated embeddings that are still semantically meaningful at lower dimensions. Only Qwen3 variants support this.

### Configuration

```rust
// Use 4B model at 1024 dimensions (instead of native 2560)
let config = ModelConfig {
    model: EmbeddingModel::Qwen3Embedding4B,
    output_dim: Some(1024),
};
assert_eq!(config.dimensions(), 1024);
```

### Constraints

- `MIN_MRL_OUTPUT_DIM = 32` (`model.rs:375`)
- `output_dim` must be `<= native_dimensions()`
- Non-MRL models (all BGE, E5, OpenAI) reject any `output_dim` value

### Behavior

- Different `output_dim` values produce **different `EmbeddingKey`** via `to_embedding_key()` — they are separate embedding spaces
- Different `output_dim` values produce **different cache paths**: `embed_{model}_{dim}d.bin`
- The HNSW index stores embeddings per `EmbeddingKey` — mixing dimensions would corrupt search

### Environment Override

`LATTICE_EMBED_DIM` overrides `output_dim` for all Qwen models loaded by `NativeEmbeddingService`:

```bash
LATTICE_EMBED_DIM=1024  # truncate 4B's 2560-dim output to 1024
```

**Validation** (`service/native.rs:77`):

- Non-numeric → error
- `< MIN_MRL_OUTPUT_DIM (32)` → error
- `> native_dim` → error
- Empty string → treated as absent (native dim)

---

## Embedding Cache

**Source**: `foundation/embed/src/cache.rs:131`

Sharded LRU cache for computed embeddings. Keys are BLAKE3 hashes of (model + input text). Avoids redundant inference for repeated inputs.

### Configuration

| Parameter          | Default               | Source                                    | Notes                              |
| ------------------ | --------------------- | ----------------------------------------- | ---------------------------------- |
| `capacity`         | `4000`                | `DEFAULT_CACHE_CAPACITY` at `cache.rs:33` | Total entries across all 16 shards |
| `shard_count`      | `16`                  | `NUM_SHARDS` (hardcoded)                  | Fixed at compile time              |
| Per-shard capacity | `ceil(capacity / 16)` | Computed at `cache.rs:161`                | Minimum 1 per shard                |

**Why 4000**: At 384 dimensions × 4 bytes = 1.5 KB per embedding, 4000 entries ≈ 6 MB — fits comfortably in memory while covering a typical session's repeated queries. For bulk import workloads, increase via `EmbeddingCache::new(capacity)`.

**Disable**: Pass `capacity = 0` to `EmbeddingCache::new()` — sets `enabled = false`, all `get()` calls return `None`.

### CacheStats

**Source**: `cache.rs:342`

| Field      | Type    | Notes                               |
| ---------- | ------- | ----------------------------------- |
| `size`     | `usize` | Current entries (sum across shards) |
| `capacity` | `usize` | Total capacity                      |
| `hits`     | `u64`   | Cache hit count                     |
| `misses`   | `u64`   | Cache miss count                    |

`hit_rate()` method: `hits / (hits + misses)`, returns 0.0 when total is 0.

---

## Service Constants

**Source**: `foundation/embed/src/service/mod.rs`

| Constant                 | Value   | Source              | Why                                                                                                                                                      |
| ------------------------ | ------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DEFAULT_MAX_BATCH_SIZE` | `1000`  | `service/mod.rs:25` | Prevents OOM from unbounded batch requests. 1000 × 512 tokens × 384 dims ≈ 750 MB peak memory — acceptable on modern hardware                            |
| `MAX_TEXT_CHARS`         | `32768` | `service/mod.rs:31` | ~8K tokens at 4 chars/token. Inputs beyond this are rejected (return `EmbedError`). Covers Qwen3's 8192-token limit with margin for instruction prefixes |

Both are enforced in `NativeEmbeddingService::embed()` and `CachedEmbeddingService::embed()`.

---

## SIMD Configuration

**Source**: `foundation/embed/src/lib.rs:91` and `foundation/embed/src/simd/`

Runtime SIMD feature detection determines which vector kernel is used for dot product, cosine similarity, and L2 distance operations.

### Detection

`simd_config()` returns a `SimdConfig` with detected features:

| Feature | Platform | Width  | Notes                                                  |
| ------- | -------- | ------ | ------------------------------------------------------ |
| AVX2    | x86_64   | 8×f32  | Runtime checked via `is_x86_feature_detected!("avx2")` |
| NEON    | aarch64  | 4×f32  | Always available on Apple Silicon                      |
| AVX512  | x86_64   | 16×f32 | Requires `avx512` feature flag + nightly Rust          |
| Scalar  | all      | 1×f32  | Fallback when no SIMD available                        |

**Key function**: `simd::resolved_dot_product_kernel()` returns a function pointer to the best available dot product implementation. Suitable for use by any downstream crate needing cosine similarity scoring.

---

## Production Model Selection

The **production default is `multilingual-e5-small`**, configured at the application level.

### How to Change the Active Model

1. Set via `EmbeddedConfig` in code:
   ```rust
   let config = EmbeddedConfig::file("/path/to/db")
       .with_embedding_model("qwen3-embedding-0.6b", 1024);
   ```

2. After changing models, existing embeddings are **incompatible** — check a model hash on startup and block if it mismatches. Re-embed existing data or start with a fresh database.

---

## Environment Variables

| Variable                 | Type    | Default                                  | Used In                                 | Source                      |
| ------------------------ | ------- | ---------------------------------------- | --------------------------------------- | --------------------------- |
| `LATTICE_EMBED_DIM`      | integer | — (absent = native)                      | MRL dimension override for Qwen models  | `service/native.rs:74`      |
| `LATTICE_QWEN_MODEL_DIR` | path    | `~/.lattice/models/Qwen3-Embedding-0.6B` | Qwen model directory                    | `service/native.rs:253`     |
| `HOME`                   | path    | —                                        | Fallback for model directory resolution | `service/native.rs:243,267` |

**Note**: `LATTICE_NO_GPU` and `LATTICE_INFERENCE_MODEL_DIR` are inference-crate env vars but affect embedding behavior indirectly (through `NativeEmbeddingService` → `lattice-inference` model loading).

---

## Example Configurations

### Minimal (defaults — BgeSmall, 384 dims)

```rust
let config = ModelConfig::default();
// model: BgeSmallEnV15, output_dim: None, dimensions(): 384
```

### Production (mE5-small via application config)

```toml
# Application config
[embedding]
model = "multilingual-e5-small"
dim = 384
```

### Qwen3 with MRL truncation

```rust
let config = ModelConfig {
    model: EmbeddingModel::Qwen3Embedding4B,
    output_dim: Some(1024), // truncate 2560 → 1024
};
```

Or via env var:

```bash
LATTICE_EMBED_DIM=1024 your-app --embed-query "query"
```

### Remote (OpenAI)

```rust
let config = ModelConfig::new(EmbeddingModel::TextEmbedding3Small);
// Requires OPENAI_API_KEY env var
// 1536 dimensions, no MRL support
```

---

## Decoder-Only Model Query Instructions

Qwen3 models require instruction-prefixed queries for asymmetric retrieval. The `query_instruction()` method (`model.rs:227`) returns the prefix:

```
"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
```

This is applied automatically by `NativeEmbeddingService`. BERT/BGE/E5 models return `None` (no prefix needed — trained with contrastive objectives on raw text).

`document_instruction()` returns `None` for all models — documents are embedded as-is.
