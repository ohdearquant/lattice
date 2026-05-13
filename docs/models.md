# Models

## Supported Models

| Variant                             | HuggingFace ID                                                | Architecture  | Dimensions | Max Tokens | Languages    | Use Case                        |
| ----------------------------------- | ------------------------------------------------------------- | ------------- | ---------- | ---------- | ------------ | ------------------------------- |
| `BgeSmallEnV15`                     | `BAAI/bge-small-en-v1.5`                                      | BERT encoder  | 384        | 512        | English      | Default — fast, small footprint |
| `BgeBaseEnV15`                      | `BAAI/bge-base-en-v1.5`                                       | BERT encoder  | 768        | 512        | English      | Balanced quality/speed          |
| `BgeLargeEnV15`                     | `BAAI/bge-large-en-v1.5`                                      | BERT encoder  | 1024       | 512        | English      | Highest quality English         |
| `MultilingualE5Small`               | `intfloat/multilingual-e5-small`                              | BERT encoder  | 384        | 512        | 100+         | Multilingual, fast              |
| `MultilingualE5Base`                | `intfloat/multilingual-e5-base`                               | BERT encoder  | 768        | 512        | 100+         | Multilingual, balanced          |
| `AllMiniLmL6V2`                     | `sentence-transformers/all-MiniLM-L6-v2`                      | BERT encoder  | 384        | 256        | English      | Sentence similarity             |
| `ParaphraseMultilingualMiniLmL12V2` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | BERT encoder  | 384        | 128        | 50+          | Paraphrase detection            |
| `Qwen3Embedding0_6B`                | `Qwen/Qwen3-Embedding-0.6B`                                   | Decoder (GQA) | 1024       | 8192       | Multilingual | Long context, MRL-capable       |
| `Qwen3Embedding4B`                  | `Qwen/Qwen3-Embedding-4B`                                     | Decoder (GQA) | 2560       | 8192       | Multilingual | Highest quality, MRL-capable    |
| `TextEmbedding3Small`               | `text-embedding-3-small` (OpenAI)                             | Remote API    | 1536       | 8191       | Multilingual | Remote fallback                 |

`TextEmbedding3Small` is the only remote model; all others run locally via `lattice-inference`.

## Model Architecture Details

### BERT/BGE Family (Encoder-Only)

Config sourced from `lattice_inference::BertConfig`:

| Variant   | Hidden size | Layers | Attention heads | Intermediate | Parameters |
| --------- | ----------- | ------ | --------------- | ------------ | ---------- |
| BGE-small | 384         | 12     | 12              | 1536         | ~33M       |
| BGE-base  | 768         | 12     | 12              | 3072         | ~110M      |
| BGE-large | 1024        | 24     | 16              | 4096         | ~335M      |

- Tokenizer: WordPiece (BGE, all-MiniLM); SentencePiece/BPE (mE5)
- Vocab: 30,522 tokens
- Pooling: mean pooling over non-padding tokens, followed by L2 normalization
- Layer norm eps: 1e-12

### Qwen3-Embedding (Decoder-Only)

Architecture: causal attention with GQA (Grouped-Query Attention), RoPE positional encoding,
RMSNorm, SwiGLU FFN activation, last-token pooling.

Key difference from BERT: decoder models produce the embedding from the last non-padding token
rather than mean-pooling the full sequence. They handle long context (8192 tokens) natively
because of RoPE but have higher memory and compute requirements.

MRL (Matryoshka Representation Learning) is supported: the 4B model's 2560-dimensional output
can be truncated to any dimension ≥ 32 with `ModelConfig::try_new(model, Some(dim))`. This
allows trading off retrieval quality for storage and compute cost.

## How Models Are Loaded

All local models are loaded from safetensors files via memory-mapped I/O:

```
~/.lattice/models/<model-name>/model.safetensors   ← mmap'd (zero-copy)
~/.lattice/models/<model-name>/vocab.txt            ← WordPiece models
~/.lattice/models/<model-name>/tokenizer.json       ← SentencePiece/BPE models
```

The `SafetensorsFile` type in `lattice-inference` wraps the mmap and provides typed tensor
views (`Tensor1D`, `Tensor2D`). Weights are never copied to a separate heap buffer unless the
`backfill` feature is enabled for sharded Qwen checkpoints.

Weight formats available (controlled by `lattice-inference` feature flags):

- `f32` — default, highest precision
- `f16` — half precision, enabled by the `f16` feature
- `Q8` — 8-bit quantization
- `Q4` — 4-bit quantization

## Download Mechanism

The `download` feature in `lattice-inference` (enabled by default in `lattice-embed`) fetches
model files from HuggingFace on first use:

```
https://huggingface.co/{hf_model_id}/resolve/main/model.safetensors
https://huggingface.co/{hf_model_id}/resolve/main/vocab.txt       (BGE/MiniLM)
https://huggingface.co/{hf_model_id}/resolve/main/tokenizer.json  (E5/Qwen3)
```

The download is skipped when both `model.safetensors` and the tokenizer file are already
present in the cache directory. The SHA-2 hash of downloaded files is verified.

Disable auto-download:

```toml
lattice-inference = { git = "...", default-features = false, features = ["std"] }
```

With download disabled, `InferenceError::ModelNotFound` is returned if weights are absent.

## Cache Directory

Default cache: `~/.lattice/models/`. This is exposed as:

```rust
// lattice-inference
pub const DEFAULT_CACHE_DIR: &str = "~/.lattice/models";
```

Override by setting the `LATTICE_CACHE_DIR` environment variable or passing a custom path
to `BertModel::load` / `QwenModel::load`.

## Adding a New Model

To add support for a new embedding model:

1. **If BERT-family (encoder-only)**: Add a `BertConfig` factory method in
   `crates/inference/src/model/bert.rs` with the correct `hidden_size`, `num_hidden_layers`,
   `num_attention_heads`, `intermediate_size`, and `max_position_embeddings`. Wire the HuggingFace
   ID into the `canonical_model_name` mapping in `crates/inference/src/download.rs`.

2. **If decoder-only**: Add a `QwenConfig` in `crates/inference/src/model/qwen.rs` with
   the correct GQA head counts, RoPE theta, and SwiGLU expansion ratio. Verify that the
   `last_token_pool` path handles the model's attention mask convention.

3. **Add a variant to `EmbeddingModel`** in `crates/embed/src/model.rs`. Implement the match
   arms for `dimensions()`, `max_input_tokens()`, `query_instruction()`, `model_id()`, and
   `is_local()`. Add `FromStr` patterns.

4. **Wire into `NativeEmbeddingService`** in `crates/embed/src/service/native.rs` — the
   `LoadedModel` enum dispatches on the model variant to choose the BERT or Qwen path.

5. Add a test asserting the expected embedding dimensions and a round-trip `FromStr`/`Display`
   check.
