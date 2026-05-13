# lattice-embed

Vector embedding generation with SIMD-accelerated operations for semantic search and
similarity matching.

## Features

- **Local Embeddings**: Generate embeddings locally using BGE models via fastembed
- **SIMD Acceleration**: AVX2/AVX-512/NEON optimized vector operations (7x speedup)
- **LRU Cache**: Blake3-based caching to avoid recomputation
- **Async API**: Full async/await support with tokio

## Models

| Model           | Dimensions | Use Case                        | HuggingFace ID         |
| --------------- | ---------- | ------------------------------- | ---------------------- |
| `BgeSmallEnV15` | 384        | Fast, general purpose (default) | BAAI/bge-small-en-v1.5 |
| `BgeBaseEnV15`  | 768        | Balanced quality/speed          | BAAI/bge-base-en-v1.5  |
| `BgeLargeEnV15` | 1024       | Highest quality                 | BAAI/bge-large-en-v1.5 |

All models have a 512 token input limit.

## Services

### LocalEmbeddingService

Single-model service with lazy initialization. Inference is serialized (one call at a
time).

```rust
use lattice_embed::{EmbeddingService, EmbeddingModel, LocalEmbeddingService};

let service = LocalEmbeddingService::new();
let embedding = service.embed_one("Hello, world!", EmbeddingModel::default()).await?;
assert_eq!(embedding.len(), 384);
```

### CachedEmbeddingService

Wraps any `EmbeddingService` with LRU caching. Identical texts return cached embeddings.

```rust
use lattice_embed::{CachedEmbeddingService, LocalEmbeddingService, EmbeddingService};
use std::sync::Arc;

let inner = Arc::new(LocalEmbeddingService::new());
let cached = CachedEmbeddingService::new(inner, 1000); // 1000 entry cache

let emb1 = cached.embed_one("Hello", EmbeddingModel::default()).await?;
let emb2 = cached.embed_one("Hello", EmbeddingModel::default()).await?; // cache hit
```

### PooledEmbeddingService

Maintains N model instances for parallel inference. Memory scales linearly (~100-300MB
per instance).

```rust
use lattice_embed::{PooledEmbeddingService, EmbeddingService};

let service = PooledEmbeddingService::new(4); // 4 concurrent inference slots
```

## SIMD Vector Operations

All operations use runtime feature detection with automatic scalar fallback.

| Platform | Instructions                       |
| -------- | ---------------------------------- |
| x86_64   | AVX-512 VNNI > AVX2 + FMA > scalar |
| aarch64  | ARM NEON (mandatory)               |
| Other    | Scalar fallback                    |

### Performance (384-dim vectors)

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

Blake3-based hashing with LRU eviction. Default capacity: 4000 entries (~6MB for 384-dim
vectors).

```rust
use lattice_embed::{EmbeddingCache, EmbeddingModel};

let cache = EmbeddingCache::new(1000);

let key = cache.compute_key("text", EmbeddingModel::BgeSmallEnV15);
cache.put(key, vec![0.1, 0.2, 0.3]);

let stats = cache.stats();
println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);
```

## Feature Flags

| Feature | Default | Description                          |
| ------- | ------- | ------------------------------------ |
| `local` | Yes     | Enable local embedding via fastembed |

```toml
[dependencies]
lattice-embed = { version = "0.1", default-features = false }  # SIMD only
lattice-embed = { version = "0.1" }  # Full (local + SIMD)
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
