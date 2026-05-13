# ADR-016: Embedding Model Variants

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-embed

## Context

The system must support a heterogeneous set of embedding models to satisfy
different quality/speed/language tradeoffs: fast English-only search, CJK/multilingual
search, high-quality large models, and a remote OpenAI fallback. These models differ
in architecture (encoder vs decoder), token limits, required instruction prefixes,
output dimensions, and deployment location (local vs API).

All model selection must be config-driven so consuming crates (`verbs`, `db layer`)
never import inference code directly — they only name a model variant.

## Decision

Models are encoded as a single `#[non_exhaustive]` enum `EmbeddingModel` in `src/model.rs`.
All per-variant properties are encoded as `const fn` methods on the enum, not as a
separate registry table. A separate `ModelConfig` struct pairs a model variant with an
optional MRL truncation dimension.

### Key Design Choices

**10 variants across three families**

| Variant                             | HF ID                                                         | Dims | Max Tokens | Architecture | Location   |
| ----------------------------------- | ------------------------------------------------------------- | ---- | ---------- | ------------ | ---------- |
| `BgeSmallEnV15` (default)           | `BAAI/bge-small-en-v1.5`                                      | 384  | 512        | BERT encoder | Local      |
| `BgeBaseEnV15`                      | `BAAI/bge-base-en-v1.5`                                       | 768  | 512        | BERT encoder | Local      |
| `BgeLargeEnV15`                     | `BAAI/bge-large-en-v1.5`                                      | 1024 | 512        | BERT encoder | Local      |
| `MultilingualE5Small`               | `intfloat/multilingual-e5-small`                              | 384  | 512        | BERT encoder | Local      |
| `MultilingualE5Base`                | `intfloat/multilingual-e5-base`                               | 768  | 512        | BERT encoder | Local      |
| `AllMiniLmL6V2`                     | `sentence-transformers/all-MiniLM-L6-v2`                      | 384  | 256        | BERT encoder | Local      |
| `ParaphraseMultilingualMiniLmL12V2` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384  | 128        | BERT encoder | Local      |
| `Qwen3Embedding0_6B`                | `Qwen/Qwen3-Embedding-0.6B`                                   | 1024 | 8192       | Decoder      | Local      |
| `Qwen3Embedding4B`                  | `Qwen/Qwen3-Embedding-4B`                                     | 2560 | 8192       | Decoder      | Local      |
| `TextEmbedding3Small`               | `text-embedding-3-small`                                      | 1536 | 8191       | —            | Remote API |

**`const fn` properties instead of a registry**

Every model property (`dimensions()`, `max_input_tokens()`, `is_local()`, `model_id()`,
`query_instruction()`, `key_version()`, `supports_output_dim()`) is a `const fn` on
`EmbeddingModel`. This means property access is a zero-cost match arm at compile time
with no `HashMap` lookup, no `OnceLock`, no indirection through a registration table.
The trade-off is that adding a new model requires touching the source.

**Asymmetric retrieval instruction prefixes**

Two model families require instruction prefixes that must be prepended at inference time:

- E5 models (`MultilingualE5Small`, `MultilingualE5Base`): `query_instruction()` returns `Some("query: ")`.
  These models were fine-tuned with `"query: "` / `"passage: "` asymmetric prefixes; omitting them
  degrades retrieval quality substantially. `document_instruction()` returns `None` — documents
  are embedded with no prefix (by design in the original fine-tuning).
- Qwen3 models: `query_instruction()` returns the long instruction prompt `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "`.
  Decoder-based embedding models require an explicit task description to orient the pooled representation.
- BGE and MiniLM families: both return `None` — trained on raw contrastive pairs without prefixes.

**MRL (Matryoshka Representation Learning) for Qwen3 only**

`supports_output_dim()` returns `true` only for `Qwen3Embedding0_6B` and `Qwen3Embedding4B`.
`ModelConfig` accepts an `output_dim: Option<usize>` validated against the model's native
dimension and a minimum of 32. For non-MRL models, setting `output_dim` returns a
`EmbedError::InvalidInput` at construction time, not silently at inference time.

The `ModelConfig::dimensions()` method returns the active dimension (truncated or native),
and this is used as the third component of the Blake3 cache key — so different MRL
truncations never collide in the cache.

**`key_version()` for cache invalidation across model families**

Each model family has a version string embedded in cache keys to prevent cross-family
cache collisions after model updates:

- BGE/E5: `"v1.5"`
- MiniLM: `"v2"`
- Qwen3/OpenAI: `"v3"`

**`#[non_exhaustive]` and `#[serde(rename_all = "snake_case")]`**

The enum is marked `#[non_exhaustive]` so downstream match arms require a wildcard,
insulating them from new variants. Serde uses snake_case (`bge_small_en_v15`), with
`#[serde(alias = "BgeSmallEnV15")]` aliases for the PascalCase forms used in older configs.

**`FromStr` for config-driven selection**

`from_str` accepts flexible forms: display names (`"bge-small-en-v1.5"`), short names
(`"small"`, `"base"`), and HuggingFace IDs (`"BAAI/bge-small-en-v1.5"`). This allows
YAML/TOML config files to use human-friendly names.

**Model discovery via `model_id()` / `LATTICE_QWEN_MODEL_DIR`**

Local BERT models are loaded by name string passed to `BertModel::from_pretrained()`.
Qwen3 models require a directory path resolved as:

1. `LATTICE_QWEN_MODEL_DIR` env override (for testing/staging)
2. `~/.lattice/models/{slug}/` (e.g., `~/.lattice/models/qwen3-embedding-0.6b/`)

The presence check looks for `model.safetensors` or `model.safetensors.index.json`
(sharded weights) before returning the path; absent models produce a clear error
with the download URL.

### Alternatives Considered

| Alternative                                   | Pros                                     | Cons                                                                                | Why Not                                                                     |
| --------------------------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Registry `HashMap<EmbeddingModel, ModelSpec>` | Easier to add models at runtime          | Runtime lookup; `OnceLock` initialization; harder to make const                     | const fn matching is zero-cost and ensures all properties compile-checked   |
| External YAML model config files              | Users can add models without recompiling | Need config loading, validation, error handling; model properties are not type-safe | Model selection is a code-level concern, not a deployment config concern    |
| One struct per model                          | Fully typed, no match arms               | 10 structs with identical interfaces; no enum for switch dispatch                   | Enum unifies the dispatch surface; struct-per-model adds no expressiveness  |
| Separate `InstructionTemplate` enum           | Cleaner separation of instruction logic  | More types; instruction is always 1:1 with model                                    | `const fn` returning `Option<&'static str>` is simpler and covers all cases |

## Consequences

### Positive

- Adding a new model requires only: new variant in the enum and new arms in each `const fn`. The compiler will error at any match arm that forgets the new variant (once `#[non_exhaustive]` is lifted internally).
- `ModelConfig::validate()` catches MRL misconfiguration at construction time (below min 32, above native dimension, non-MRL model requesting truncation).
- `ModelProvenance` records model identity, load timestamp, and a Blake3 metadata hash for lightweight audit logging — without loading model weights.

### Negative

- Adding a model requires recompilation of all crates that pattern-match on `EmbeddingModel`. `#[non_exhaustive]` prevents downstream exhaustive matches outside the crate.
- The Qwen3 model directory resolution is synchronous and filesystem-blocking. If `~/.lattice/models/` is on a slow NFS mount, the first `NativeEmbeddingService` construction will block.

### Risks

- `MRL truncation + caching`: Two `ModelConfig` values with different `output_dim` produce different cache keys, but if a caller constructs a `NativeEmbeddingService` with one dimension and queries the cache with a different dimension, there will be a cache miss that doesn't error — just a silent inefficiency.
- The `TextEmbedding3Small` variant is remote-only and `is_local()` returns `false`. `NativeEmbeddingService::load_model_sync` returns `Err("unsupported model: TextEmbedding3Small")` if instantiated with this variant. The remote path is not yet implemented; the variant exists as a type-level placeholder.

## References

- [`crates/embed/src/model.rs`](/Users/lion/projects/lattice/crates/embed/src/model.rs) — `EmbeddingModel`, `ModelConfig`, `ModelProvenance`
- [`crates/embed/src/service/native.rs`](/Users/lion/projects/lattice/crates/embed/src/service/native.rs) — model loading, `qwen_model_dir`, `LATTICE_QWEN_MODEL_DIR`
