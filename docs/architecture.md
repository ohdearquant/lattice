# Architecture

Lattice is a pure Rust inference engine for transformer-based embedding models. It has no ONNX
dependency, no Python runtime, and no C++ FFI. All compute paths — matmul, attention, tokenization,
optimal transport — are written in Rust with hand-written SIMD kernels.

## Crate Dependency Graph

```
lattice-inference   (standalone — transformer forward pass, weights, tokenizers)
lattice-fann        (standalone — fast tiny neural nets, <5ms inference)
lattice-transport   (standalone — optimal transport math, Sinkhorn-Knopp)
       │
       ▼
lattice-embed   ←── lattice-inference   (embedding service, SIMD distance ops)
lattice-tune    ←── lattice-fann
                ←── lattice-inference   (training, LoRA, knowledge distillation)
```

The three leaf crates (`inference`, `fann`, `transport`) have zero intra-workspace dependencies
and can be used standalone. Consumer crates (`embed`, `tune`) compose them.

## Design Philosophy

**No external ML runtime.** ONNX, TensorRT, and similar runtimes add ABI complexity, version
pinning, and build surface. Lattice owns its entire compute graph from weight loading to
final pooling, which keeps builds reproducible and portable.

**SIMD-first.** Hot paths dispatch at runtime to AVX2, AVX-512 (nightly), or NEON. SIMD
intrinsics are the only reason `unsafe` exists in this codebase; every unsafe block is a
direct intrinsic call in a SIMD kernel.

**Safetensors native.** Model weights are loaded from HuggingFace safetensors format via
memory-mapped files. No conversion step. No custom format.

**Async at the service layer only.** `lattice-inference` and `lattice-fann` are synchronous.
Async lives in `lattice-embed`'s service layer where I/O (model download, cache lookup)
justifies it.

## Crate Descriptions

### lattice-inference

The transformer kernel. Stability tier: Experimental (high churn).

Contains two architecture paths:

- **BERT/BGE (encoder-only)**: bidirectional attention, mean pooling, WordPiece/BPE tokenizers.
  Covers BGE-small/base/large, mE5-small/base, all-MiniLM, paraphrase-multilingual-MiniLM.
- **Qwen3 (decoder-only)**: causal GQA, RoPE, RMSNorm, SwiGLU FFN, last-token pooling,
  BPE tokenizer. Covers Qwen3-Embedding-0.6B and 4B.

Modules: `model` (configs + loaders), `tokenizer`, `weights` (f32/f16/Q8/Q4), `attention`
(standard, GQA, Flash, GDN), `forward` (CPU/NEON/Metal/WGPU backends), `pool`, `rope`,
`kv_cache`, `lora_hook`, `download`, `speculative`, `sampling`.

**Not for direct use by application code.** Consumers should go through `lattice-embed`.

### lattice-embed

The public embedding service. Stability tier: Unstable (API still evolving).

Wraps `lattice-inference` with:

- `EmbeddingService` trait (async, tokio)
- `NativeEmbeddingService` — pure Rust inference, model loaded on first call
- `CachedEmbeddingService` — LRU cache over `NativeEmbeddingService`
- SIMD-accelerated distance operations: cosine similarity, dot product, euclidean distance,
  L2 normalization
- `EmbeddingModel` enum with all supported variants and metadata (dimensions, token limits,
  query/document instruction prefixes, MRL support)
- Backfill coordinator for re-embedding existing stores after model changes
- Migration controller for zero-downtime model swaps

This is the entry point for most applications.

### lattice-fann

Fast neural network primitives for tiny models. Stability tier: Stable.

`NetworkBuilder` → `Network` → `[Layer, ...]` with pre-allocated buffers. No heap
allocation during the forward pass. Target: <5ms inference for small classifiers.

Supports: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, Linear activations. Backpropagation
with momentum. Optional rayon parallelism for batch inference. Optional wgpu GPU training.

### lattice-tune

Training infrastructure. Stability tier: Unstable.

Four concerns cleanly separated:

- `data` — `TrainingExample`, `Dataset`, `Batch`
- `distill` — teacher model API (Claude/GPT/Gemini), `DistillationPipeline`
- `train` — `TrainingLoop`, `Optimizer`, `LRSchedule`, `EarlyStopping`, `Checkpoint`,
  JIT-compiled adapters, GPU trainer
- `registry` — versioned model store, `ModelRegistry`, `RollbackController`,
  shadow-comparison sessions
- `lora` — `LoraAdapter`, `LoraConfig`, `LoraLayer`

### lattice-transport

Optimal transport math. Stability tier: Unstable (API may change as second consumer lands).

Implements entropy-regularized optimal transport (Sinkhorn-Knopp) in log-domain for
numerical stability. Designed for quantifying embedding geometry drift between model versions.

Modules: `sinkhorn` (balanced), `sinkhorn_log` (epsilon-scaling), `unbalanced` (KL-relaxed),
`barycenter` (Wasserstein), `drift` (embedding drift detection), `transport_plan`, `divergence`,
`cost`, `math`, `logsumexp`.

Key design: log-domain throughout (never materializes the Gibbs kernel), pre-allocated
`SinkhornWorkspace`, no BLAS/LAPACK.

## Layer Diagram

```
Application
    │
    ▼
lattice-embed           ← public API for embedding generation
    │   SIMD distance ops (cosine, dot, euclidean)
    │   LRU cache, backfill, migration
    ▼
lattice-inference        ← transformer forward pass
    │   BERT/BGE encoder path (WordPiece/BPE tokenizers, mean pooling)
    │   Qwen3 decoder path (BPE, GQA, RoPE, SwiGLU, last-token pooling)
    │   Weight formats: f32, f16, Q8, Q4
    ▼
CPU backends             Metal (macOS)    WGPU (cross-platform GPU)
AVX2 / NEON kernels      Metal MSL kernels WGSL compute shaders


lattice-fann             ← tiny model inference + training (independent)
lattice-transport        ← optimal transport math (independent)
lattice-tune             ← training pipeline (depends on fann + inference)
```

## When to Use Which Crate

| Use case                                 | Crate                          |
| ---------------------------------------- | ------------------------------ |
| Generate text embeddings in an app       | `lattice-embed`                |
| Run a tiny classifier (<5ms)             | `lattice-fann`                 |
| Build a training loop with distillation  | `lattice-tune`                 |
| Measure embedding distribution drift     | `lattice-transport`            |
| Write a new model architecture or kernel | `lattice-inference` (internal) |
