# Supported Models, Attention Variants & Features

Authoritative reference for what Lattice supports **today** in the current codebase.
Every entry is verified against source code. Status labels:

- **shipped** — callable path exists now for the stated scope.
- **partial** — callable path exists, but an important integration or model family is limited.
- **experimental** — code exists but is unstable, gated, or not a general serving API.
- **scaffold-only** — type/config placeholder exists; no usable runtime path.
- **not shipped** — absent in the current branch.
- **P1 metadata only** — PR #132 metadata enum; not a production dispatch layer.

---

## 1. Supported Models

Lattice has local embedding loaders for seven BERT-family embedding variants, local-directory
Qwen3 embedding loaders, a local Qwen3.5/Qwen3.6 generation loader, a local BERT-style
cross-encoder loader, a remote embedding enum placeholder, and a BitNet config scaffold.

Automatic HuggingFace download is available **only for the seven BERT-family embedding variants**.
Qwen3 embedding and Qwen3.5/Qwen3.6 generation paths require local files.

### Embedding models (`EmbeddingModel`)

| Variant                             | HuggingFace ID                                                | Dims  | Max tokens |  Pooling   |    Download    |    Status     |
| ----------------------------------- | ------------------------------------------------------------- | :---: | :--------: | :--------: | :------------: | :-----------: |
| `BgeSmallEnV15`                     | `BAAI/bge-small-en-v1.5`                                      |  384  |    512     |    CLS     |       ✓        |    shipped    |
| `BgeBaseEnV15`                      | `BAAI/bge-base-en-v1.5`                                       |  768  |    512     |    CLS     |       ✓        |    shipped    |
| `BgeLargeEnV15`                     | `BAAI/bge-large-en-v1.5`                                      | 1024  |    512     |    CLS     |       ✓        |    shipped    |
| `MultilingualE5Small`               | `intfloat/multilingual-e5-small`                              |  384  |    512     |    mean    |       ✓        |    shipped    |
| `MultilingualE5Base`                | `intfloat/multilingual-e5-base`                               |  768  |    512     |    mean    |       ✓        |    shipped    |
| `AllMiniLmL6V2`                     | `sentence-transformers/all-MiniLM-L6-v2`                      |  384  |    256     |    mean    |       ✓        |    shipped    |
| `ParaphraseMultilingualMiniLmL12V2` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |  384  |    128     |    mean    |       ✓        |    shipped    |
| `Qwen3Embedding0_6B`                | `Qwen/Qwen3-Embedding-0.6B`                                   | 1024  |   2048†    | last-token | local dir only |    partial    |
| `Qwen3Embedding4B`                  | `Qwen/Qwen3-Embedding-4B`                                     | 2560‡ |   2048†    | last-token | local dir only |    partial    |
| `TextEmbedding3Small`               | `text-embedding-3-small` (OpenAI)                             | 1536  |    8191    |     —      |       —        | scaffold-only |

†Model config advertises 32 768 max positions; `EmbeddingModel::max_input_tokens()` reports 2 048,
the tokenizer's sequence limit, and the native service truncates longer input.

‡`Qwen3Embedding4B` and `Qwen3Embedding0_6B` both support MRL (Matryoshka Representation
Learning): the output dimension can be truncated to any value ≥ 32 via
`ModelConfig::try_new(model, Some(dim))`.

**Notes on BGE pooling**: BGE v1.5 models use **CLS** pooling; E5 and MiniLM variants use
**mean** pooling. They do not all use mean pooling.

**`TextEmbedding3Small`**: `is_remote()` returns `true` and the native service rejects it.
No remote service implementation exists. Do not treat it as a working fallback.

**E5 prefixes**: `query_instruction()` returns `"query: "` and `document_instruction()` returns
`"passage: "` for both E5 variants. `embed_passage()` applies the prefix automatically — callers
do not need to add it manually.

### Other loader surfaces

| Surface                                                                 |                    Status                     | Notes                                                                                                                                                                                                                                                                        |
| ----------------------------------------------------------------------- | :-------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BERT/BGE local loader (`BertModel::from_directory` / `from_pretrained`) |                    shipped                    | Loads `model.safetensors` + auto-detected tokenizer. `from_pretrained` downloads only cataloged BERT-family IDs. Sharded BERT safetensors are **not** supported.                                                                                                             |
| Cross-encoder reranker (`CrossEncoderModel::from_directory`)            |                    shipped                    | BERT-style scalar query-document scoring with pair tokenization and `type_vocab_size >= 2`. Complete pooler tensors apply dense+tanh before classification; both absent retain direct CLS, and partial poolers are rejected. Tensor names use the unprefixed BERT layout.    |
| Qwen3 embedding local loader (`QwenModel::from_directory`)              | shipped for local files, partial for download | Loads single-file or sharded Qwen3 embedding checkpoints. Sharded loading is handled by `ShardedSafetensors::open_index` when `model.safetensors.index.json` is present — no `backfill` feature required. `from_pretrained` exists but the download catalog has no Qwen IDs. |
| Qwen3.5/Qwen3.6 generation loader (`Qwen35Model::from_safetensors`)     |                    partial                    | Loads local single-file or sharded Qwen3.5/Qwen3.6 generation checkpoints. MTP fields are config/runtime-gated (experimental).                                                                                                                                               |
| `Qwen35Config::qwen35_2b` preset                                        |                    shipped                    | 24 layers, 2 048 hidden, dense FFN, tied embeddings, 262 144 max positions.                                                                                                                                                                                                  |
| `Qwen35Config::qwen35_0_8b` preset                                      |                    partial                    | 24 layers, 1 024 hidden, dense FFN, 1 MTP layer. Base decode wired; MTP experimental.                                                                                                                                                                                        |
| `Qwen35Config::qwen36_35b_a3b` preset                                   |                    partial                    | 40 layers, MoE 256 experts top-8 plus shared expert. Local text loaders and per-token decode exist; Metal prompt prefill has restrictions. See the Qwen3.6 matrix below.                                                                                                     |
| `Qwen35Config::qwen36_27b` preset                                       |               shipped for text                | 64 layers, 5 120 hidden, dense FFN. Local safetensors CPU text path and native Q4 Metal text path; see the Qwen3.6 matrix below for limits and acceptance status.                                                                                                            |
| BitNet b1.58 2B4T (`BitNetConfig::bitnet_2b4t`)                         |                 scaffold-only                 | Config preset exists; no `BitNetModel` loader is re-exported from `model/mod.rs`.                                                                                                                                                                                            |

### Qwen3.6 generation support by variant

Here **supported** means a callable path exists for the stated scope. **Degraded / limited**
identifies a fallback or incomplete integration; **rejected** identifies an explicit refusal.
These labels do not certify real-checkpoint parity. The deterministic 27B/35B acceptance work
remains tracked in [#617](https://github.com/ohdearquant/lattice/issues/617).

Native Q4 text generation requires macOS and a build with `f16,metal-gpu`. In `lattice chat`
and `lattice serve`, safetensors text generation uses the CPU path and needs enough memory for
its materialized weights; standalone `lattice_serve` also has a Metal safetensors route. Use the
checkpoint's `config.json` and tokenizer; the Rust presets alone do not establish compatibility.

| Variant                   | Supported paths                                                                                          | Degraded / limited                                                                                                                                                                               | Rejected paths                                                                                                                                                                                                                                         |
| ------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Qwen3.6-27B (dense)**   | Local safetensors CPU text generation; native Q4 Metal text generation through `lattice chat` / `serve`. | Metal GDN prefill uses serial recurrence: the chunked-scan path only accepts the 0.8B geometry, not this model's 5 120 hidden width and 16 key / 48 value heads.                                 | QuaRot v0 conversion rejects the non-power-of-two hidden width.                                                                                                                                                                                        |
| **Qwen3.6-35B-A3B (MoE)** | Local safetensors CPU text generation; native Q4 conversion, Metal loading and per-token `forward_step`. | Native Q4 routed experts use a bounded LRU cache. Loading and per-token decode do not provide general Metal chat/serve support: ordinary prompt prefill hits the restriction in the next column. | Multi-token Metal prefill without an active LoRA adapter returns `UnsupportedModel`. Q8 quantization rejects MoE; QuaRot v0 conversion also rejects MoE. Native Q4 loading refuses a memory budget too small for the minimum top-k expert-cache slots. |

The MoE prefill refusal applies to the fallible request-facing paths. A single token or an active
LoRA adapter can select per-token forward steps instead; these exceptions do not establish
full-checkpoint parity. The legacy infallible `forward_prefill` wrapper can still panic on an
unsupported input. Backend restrictions are implemented in the
[Metal runtime](../crates/inference/src/forward/metal_qwen35.rs),
[Q8 conversion](../crates/inference/src/weights/q8_weights.rs), and
[QuaRot conversion](../crates/inference/src/quant/quarot/convert.rs).

**Image input for chat completions is a separate capability.** CPU HTTP chat routes reject
image requests. Metal HTTP chat routes require vision geometry, distinct valid image/start/end
token IDs and compatible vision
weights in the actual checkpoint. Both Qwen3.6 Rust presets are text-only. The chat image path
accepts one inline PNG/JPEG per request and rejects streaming image requests; it does not imply
27B/35B image parity. See [vision admission](../crates/inference/src/serve/metal_worker.rs) and
[request validation](../crates/inference/src/serve/contract.rs).

### Model loading and cache

```text
# BERT-family (auto-download available)
~/.lattice/models/<model-name>/model.safetensors
~/.lattice/models/<model-name>/vocab.txt            (WordPiece models)
~/.lattice/models/<model-name>/tokenizer.json       (BPE/SentencePiece models)

# Qwen3 embedding (local directory required)
LATTICE_QWEN_MODEL_DIR or ~/.lattice/models/qwen3-embedding-<variant>/

# Qwen3.5/Qwen3.6 generation (local directory required)
<path>/model.safetensors  (single-file or sharded)
<path>/tokenizer.json     (BPE, required)
```

Environment variables:

- `LATTICE_MODEL_CACHE` — overrides the BERT-family download cache directory (default
  `~/.lattice/models`).
- `LATTICE_QWEN_MODEL_DIR` — overrides the Qwen3 embedding local directory.

Public loader functions are `from_pretrained()` (download-backed, BERT-family only) and
`from_directory()` (local path). The old names `BertModel::load` / `QwenModel::load` are **not**
the current public API.

### Weight formats

`f32` safetensors are the primary and universal path. f16, Q8, Q4, and QuaRot Q4 code exists
but support is path-specific, not universal across all models:

- **f16** — half-precision helpers; not a shipped f16 KV-cache path (see §3).
- **Q8** — standalone quantization is dense-only; MoE layers return `UnsupportedModel`.
- **Q4** — available via the `quantize_q4` binary for supported paths.
- **QuaRot Q4** — offline converter (`quantize_quarot` binary) + Metal direct-load path. Converter
  refuses non-power-of-two hidden dims and MoE configs.

---

## 2. Attention Variants

The current `docs/supported-models-features` branch exports attention modules but does **not**
contain the `AttentionKind` enum. The 10-row table below is taken from PR #132's hardened
ADR-059 Phase-1 metadata enum on branch `show/seed-impl/attention-dispatch`.

**`AttentionKind` is P1 metadata only.** The enum does not implement `AttentionOp`, allocate
state or scratch buffers, or call kernels. Production callers via a dispatch layer are future
P2+ work. The "production caller" column describes whether the _backing implementation_ (kernel,
forward pass) is used today outside the enum.

The enum is exhaustive: PR #132 tests assert exactly 10 variants.

| Variant               | `is_causal` | `supports_kv_cache` | Production caller (backing impl)                                                                                                      |
| --------------------- | :---------: | :-----------------: | ------------------------------------------------------------------------------------------------------------------------------------- |
| `Mha`                 |    false    |        false        | Backing MHA is production-used by BERT and CrossEncoder. `AttentionKind`: P1 metadata only.                                           |
| `Gqa(gqa::GqaConfig)` |    true     |        true         | Backing cache-backed GQA is production-used by Qwen generation/decode paths. `AttentionKind`: P1 metadata only.                       |
| `Flash`               |    false    |        false        | Backing tiled CPU Flash code exists; no verified production caller. `AttentionKind`: P1 metadata only.                                |
| `FlashCausal`         |    true     |        false        | Backing causal Flash module exists; no verified production caller. `AttentionKind`: P1 metadata only.                                 |
| `Gdn`                 |    true     |        false        | Backing recurrent GDN step exists; production Qwen3.5 uses the fused variant instead. `AttentionKind`: P1 metadata only.              |
| `GdnFused`            |    true     |        false        | Backing fused GDN has a production caller in the Qwen3.5 forward path for linear-attention layers. `AttentionKind`: P1 metadata only. |
| `GatedGqa`            |    true     |        true         | Backing behavior is production-used by Qwen3.5 full-attention layers (K/V append + sigmoid gate). `AttentionKind`: P1 metadata only.  |
| `Differential`        |    true     |        false        | Backing standalone kernel exists; no verified production caller. `AttentionKind`: P1 metadata only.                                   |
| `NativeSparse`        |    true     |        false        | Backing standalone NSA kernel exists; no verified production caller. `AttentionKind`: P1 metadata only.                               |
| `Decode`              |    true     |        true         | Backing decode path has a Metal production caller. `AttentionKind`: P1 metadata only.                                                 |

**Recurrent vs KV-cache**: GDN and GdnFused use recurrent state (`GatedDeltaNetState`), not a
KV cache — hence `supports_kv_cache = false`.

---

## 3. Inference Features

| Feature                     |           Status           | Notes                                                                                                                                                                                                                                                  |
| --------------------------- | :------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| f16 KV cache                |      **not shipped**       | Current flat, paged, and Qwen3.5 ad hoc KV caches store `f32` and accept `&[f32]` append/read buffers. The `f16` feature flag covers weight helpers only.                                                                                              |
| LoRA inference hook         |      partial/shipped       | CPU inference hook for projection deltas; `Qwen35Model::set_lora` exposed. `lattice-tune` loads PEFT safetensors behind `inference-hook`. Metal single-active-adapter runtime path exists. Multi-adapter continuous-batching grouping is not complete. |
| LoRA + QuaRot composition   |      partial/shipped       | Rotation-aware LoRA transformation exists; Metal loader applies it when `quarot_seed` is supplied. Unknown targets fail closed. Scope limited to supported projection targets.                                                                         |
| QuaRot INT4                 |      partial/shipped       | QuaRot module family, offline converter (`quantize_quarot` binary), and Metal direct Q4/F16 directory loader exist. Converter refuses non-power-of-two hidden dims and MoE configs.                                                                    |
| Speculative decoding        |        experimental        | N-gram has a model-agnostic wrapper plus a state-owned Metal entry point (admission + prefill). Metal MTP/self-spec paths are greedy-only and disabled with grammar/compact sampling. Not a stable serving API.                                        |
| Continuous batching         |   partial — library only   | `batch` module exports Phase-1 scheduler/worker structures with chunked prefill and FIFO policy. HTTP/gRPC serving integration is **not** in scope for the current branch (ADR-048 excludes it; ADR-063 defers serving to v0.4+).                      |
| Serving / CLI               |      partial/shipped       | Unified `lattice chat` and `lattice serve` exist. The HTTP server exposes OpenAI-compatible chat completions, embeddings and model-list routes; accepted options depend on the model and backend. See [API capabilities](capability-matrix.md).        |
| Grammar / structured output | partial/shipped (unstable) | Grammar-constrained decoding supports a JSON Schema subset and GBNF escape hatch, compiled into `GrammarEngine`, masks logits, and is wired into the Metal generation loop. Grammar disables compact top-k and speculative paths.                      |
| Metrics / profiling         |      partial/proposed      | Current observability is profiling structs, `encode_profiled()`, and env-gated Metal step profiling. ADR-061 `MetricsMode`/`ForwardCtx`/`MetricSink`/SQLite/JSONL infrastructure is proposed/design-only.                                              |

---

## 4. Tokenizers

Lattice has three dependency-free tokenizer implementations behind a common `Tokenizer` trait
and an auto-loader. The loader probes files in this order: `tokenizer.json` → BPE vocab/merges →
`vocab.txt` → `tokenizer.model`. Unsupported `tokenizer.json` model types return an error.

`Tokenizer` trait methods: `tokenize`, `tokenize_batch`, `vocab_size`, `max_seq_len`,
pair-tokenization checks, and `decode`.

| Tokenizer     | Status  | Notes                                                                                                                                                                                                                                |
| ------------- | :-----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| BPE           | shipped | Byte-level GPT/Qwen-style BPE. Supports `tokenizer.json`, `vocab.json` + `merges.txt`, and `vocab.txt` + `merges.txt`. Qwen3.5/Qwen3.6 generation **requires** BPE `tokenizer.json`. Qwen3 embedding also typically resolves to BPE. |
| SentencePiece | shipped | Unigram model. Supports HF `tokenizer.json` types `Unigram` / `SentencePieceUnigram` and native `tokenizer.model`. Available to BERT-family loaders through `load_tokenizer` when the model directory provides compatible files.     |
| WordPiece     | shipped | Supports `tokenizer.json` model type `WordPiece` and `vocab.txt`. Implements BERT-style pair tokenization. BGE/MiniLM download paths use `vocab.txt`. CrossEncoder requires pair tokenization.                                       |

**Tokenizer assignment is auto-detected, not hard-coded per model.** Calling E5 "SentencePiece"
or BGE/BERT "WordPiece/BPE" as fixed assignments is incorrect — the loader auto-detects the
format from the files present in the model directory.
