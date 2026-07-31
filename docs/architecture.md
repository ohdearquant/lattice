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

## Forward Pass Pipeline

This section traces the canonical Qwen3.5 generation path end to end. All references are to
`crates/inference/src/` unless a full path is given. The removed generic Qwen text-decode
stack is not part of the current architecture; `QwenModel::encode` remains the live Qwen3
embedding path used by `lattice-embed`.

### Qwen3.5 Generation Path

The Qwen3.5 architecture (`crate::model::qwen35`) is the sole supported text-generation
stack. Its path is verified against `crates/inference/src/model/qwen35/` and
`crates/inference/src/forward/`:

```text
local safetensors directory -> Qwen35Model::from_safetensors
  -> required tensor validation (validate_required_tensor_names)
  -> load_weights                                    (model/qwen35/loading.rs)
  -> BpeTokenizer::from_tokenizer_json                (tokenizer.json)
  -> Qwen35Model::generate / generate_streaming       (model/qwen35/generation.rs)
  -> Tokenizer::tokenize(prompt)                      (tokenizer/common.rs)
  -> prefill_tokens_batched_for_generate              (forward/batch_prefill.rs)
       -> prefill_prompt                              (forward/batch_prefill.rs)
  -> sample_token (first generated token)             (model/qwen35/sampling.rs)
  -> forward_step (decode loop, one call per token)   (model/qwen35/forward.rs)
  -> sample_token (each subsequent generated token)
```

`Qwen35Model::from_safetensors` (`model/qwen35/model.rs`) resolves either `model.safetensors` or
a sharded `model.safetensors.index.json`, validates the required tensor names against the
config, then calls `load_weights` to materialize embedding, layer, and norm weights, and loads
the tokenizer from `tokenizer.json`. Generation itself is the canonical `Qwen35Model::generate` /
`Qwen35Model::generate_streaming` (`model/qwen35/generation.rs`): both tokenize the prompt with
`Tokenizer::tokenize`, then delegate their prefill phase to
`Qwen35Model::prefill_tokens_batched_for_generate` (`forward/batch_prefill.rs`), which runs a
single batched `prefill_prompt` pass over the whole prompt (PR #680) and returns the final
prompt position's logits. Both then sample the first token with `sample_token`, and enter a
decode loop that calls the single-token `forward_step` (`model/qwen35/forward.rs`) and
`sample_token` once per generated token.

Note: the tokenizer trait method is `tokenize`, not `encode` — `encode` is a method on the
separate `QwenModel` embedding path (`model/qwen.rs`), not on the `Tokenizer` trait used here.

For a deeper, function-by-function walkthrough of this path (including the GDN/full-attention
dispatch inside each layer), see [`docs/forward-pass.md`](forward-pass.md). For library usage of
either path, see [`docs/inference-usage.md`](inference-usage.md).
