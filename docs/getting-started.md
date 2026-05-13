# Getting Started

## Add as a Dependency

```toml
[dependencies]
lattice-embed = { git = "https://github.com/ohdearquant/lattice" }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

The default feature set (`native`) pulls in `lattice-inference` and enables the download
feature, so the first call will fetch the model weights from HuggingFace if they are not
already cached.

## Generate an Embedding

```rust
use lattice_embed::{EmbeddingService, EmbeddingModel, NativeEmbeddingService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service = NativeEmbeddingService::default();

    let embedding = service
        .embed_one("The quick brown fox", EmbeddingModel::default())
        .await?;

    // Default model is BGE-small-en-v1.5: 384 dimensions
    println!("dimensions: {}", embedding.len());

    Ok(())
}
```

`NativeEmbeddingService::default()` selects `EmbeddingModel::BgeSmallEnV15` (384 dimensions,
~33M parameters, English). On first run the model weights are downloaded to
`~/.lattice/models/bge-small-en-v1.5/` and cached there for all subsequent calls.

## Cosine Similarity

```rust
use lattice_embed::{
    EmbeddingService, EmbeddingModel, NativeEmbeddingService,
    utils::cosine_similarity,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service = NativeEmbeddingService::default();
    let model = EmbeddingModel::default();

    let a = service.embed_one("Rust is a systems programming language", model).await?;
    let b = service.embed_one("Rust targets performance and memory safety", model).await?;
    let c = service.embed_one("The weather in Paris today", model).await?;

    println!("related:   {:.4}", cosine_similarity(&a, &b)); // ~0.85+
    println!("unrelated: {:.4}", cosine_similarity(&a, &c)); // ~0.3–0.5

    Ok(())
}
```

`cosine_similarity` is SIMD-accelerated and dispatches at runtime to AVX2 or NEON where
available.

## Batch Embedding

```rust
let texts = vec![
    "first document",
    "second document",
    "third document",
];

let embeddings = service
    .embed_batch(&texts, EmbeddingModel::default())
    .await?;

assert_eq!(embeddings.len(), 3);
```

## Choosing a Model

```rust
use lattice_embed::EmbeddingModel;

// Fast English model — good default
let _ = EmbeddingModel::BgeSmallEnV15;    // 384d, 512 tokens

// Higher quality English
let _ = EmbeddingModel::BgeBaseEnV15;    // 768d, 512 tokens
let _ = EmbeddingModel::BgeLargeEnV15;  // 1024d, 512 tokens

// Multilingual (100+ languages)
let _ = EmbeddingModel::MultilingualE5Small;  // 384d — prefix "query: " / "passage: "
let _ = EmbeddingModel::MultilingualE5Base;   // 768d — prefix "query: " / "passage: "

// Decoder-based, long context (8192 tokens)
let _ = EmbeddingModel::Qwen3Embedding0_6B;  // 1024d, MRL-capable
let _ = EmbeddingModel::Qwen3Embedding4B;    // 2560d, MRL-capable

// Sentence Transformers family
let _ = EmbeddingModel::AllMiniLmL6V2;                      // 384d, 256 tokens
let _ = EmbeddingModel::ParaphraseMultilingualMiniLmL12V2; // 384d, 128 tokens
```

Parse from string (flexible, case-insensitive):

```rust
let model: EmbeddingModel = "bge-base".parse()?;
let model: EmbeddingModel = "BAAI/bge-large-en-v1.5".parse()?;
```

## Asymmetric Retrieval (E5 and Qwen3 Models)

E5 and Qwen3 models are trained with different prefixes for queries vs. documents. The
`EmbeddingModel` API exposes these:

```rust
let model = EmbeddingModel::MultilingualE5Small;
// model.query_instruction() == Some("query: ")
// model.document_instruction() == None (apply "passage: " to documents manually)

// Correct asymmetric usage:
let query_text = format!("query: {}", user_query);
let doc_text = format!("passage: {}", document);
```

## MRL / Matryoshka Dimension Truncation

Qwen3 models support configurable output dimensions (Matryoshka Representation Learning):

```rust
use lattice_embed::{EmbeddingModel, ModelConfig};

// Truncate Qwen3-4B to 512 dimensions for cheaper storage
let config = ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, Some(512))?;
// config.dimensions() == 512
```

Minimum truncation dimension: 32. Maximum: the model's native dimension.

Two `ModelConfig` values with different `output_dim` produce different embedding spaces
and must be stored in separate namespaces.

## Feature Flags

| Feature     | Default | Description                                             |
| ----------- | ------- | ------------------------------------------------------- |
| `native`    | yes     | Enable `NativeEmbeddingService` via `lattice-inference` |
| `metal-gpu` | no      | Metal GPU backend for Apple Silicon                     |
| `avx512`    | no      | AVX-512 kernels (requires nightly Rust)                 |

```toml
# Apple Silicon with GPU acceleration
lattice-embed = { git = "...", features = ["metal-gpu"] }
```

## Model Cache Directory

Models are cached at `~/.lattice/models/<model-name>/`. Each model directory contains:

```
~/.lattice/models/bge-small-en-v1.5/
    model.safetensors   # weight file (mmap'd at inference time)
    vocab.txt           # WordPiece vocabulary (BGE/MiniLM)
    tokenizer.json      # SentencePiece config (E5/Qwen3)
```

The `download` feature in `lattice-inference` (enabled by default) fetches from
`https://huggingface.co/{model_id}/resolve/main/` on first use. Subsequent calls
skip the download.

To use pre-downloaded weights, place them in the expected directory before calling
`embed_one`. The service checks for the files before attempting any download.
