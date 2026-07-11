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

This section traces the decoder (Qwen3) generate path end to end. All references are to
`crates/inference/src/` unless a full path is given. The encoder (BERT/BGE) path is
structurally similar but uses bidirectional attention and skips the KV cache and decode loop.

**Deprecated (since 0.5.1, removal targeted for 0.6.0, issue #807):** `crate::generate` (the
path traced below) is repository-dead — it has no production caller — and duplicates the
canonical, actively maintained Qwen3.5 decode loop. Text-generation users should use
`Qwen35Model::generate` / `Qwen35Model::generate_streaming` instead; see the
[Qwen3.5 Generation Path](#qwen35-generation-path) section below. This trace is kept for anyone
still debugging the deprecated path during its deprecation window; embedding users of
`QwenModel` (via `lattice-embed`) are unaffected either way.

### 1. Model Load

`QwenModel::load` (`model/qwen.rs`) opens the model directory and calls
`SafetensorsFile::open` (`weights/f32_weights.rs`) to memory-map `model.safetensors`. Tensors
are accessed as zero-copy `Tensor2D<'a>` slices backed by the mmap. Sharded checkpoints go
through `ShardedQwenBacking`, which owns heap-allocated copies of each shard. The model also
constructs a `RopeTable` (`rope.rs`) at load time by precomputing sin/cos tables up to
`max_position_embeddings`.

### 2. Entry Point

```
generate(model, prompt, config)   // generate.rs:276
```

`generate` is the single public entry point for text generation. It accepts a `&QwenModel`, a
prompt string, and a `GenerateConfig` (`generate.rs:27`), then orchestrates the five steps
below. `GenerateConfig` carries `max_new_tokens`, `SamplingConfig`, an optional EOS token ID,
an optional `GrammarEngine`, and an optional `kv_cache_capacity` cap.

### 3. Tokenize

```
tokenizer.tokenize(prompt)   // tokenizer/common.rs:37
```

`QwenModel::tokenizer()` returns a `Box<dyn Tokenizer>`. For Qwen3 this is a `BpeTokenizer`
(`tokenizer/bpe.rs`) doing GPT-2-style byte-level BPE encoding. `tokenize` returns a
`TokenizedInput` with `input_ids` (a `Vec<u32>`) and `real_length` (number of non-padding
tokens). The generate loop uses `tokenized.real_length` as `prompt_len`.

### 4. Allocate KV Cache and Scratch

`generate.rs` allocates a `FlatKVCache` (`kv_cache/flat.rs`) sized to
`prompt_len + max_new_tokens` (or the caller's `kv_cache_capacity` cap). The cache stores K
and V tensors in f16 (2 bytes per element) across all layers in two contiguous per-layer
buffers, totalling `2 * num_layers * capacity * kv_dim * 2` bytes.

A `ForwardScratch` struct (`generate.rs:103`) holds pre-allocated f32 buffers for all
intermediate activations: `hidden`, `residual`, `qkv_buf`, `q_buf`, `k_buf`, `v_buf`,
`attn_out`, `gate_up_buf`, `gate_buf`, `up_buf`, `ffn_out`, `scores`, `logits`, and two
dequantization buffers (`cached_k_f32`, `cached_v_f32`). `ensure_capacity` grows these
buffers on the first call (prefill size) and never shrinks them, so decode steps reuse the
same allocation.

### 5. Prefill

```
forward_with_cache(model, &prompt_ids[..prompt_len], &mut cache, 0, &mut scratch, max_seq)
```

`forward_with_cache` (`generate.rs:451`) runs the full prompt through the transformer in a
single pass (`start_pos = 0`, `seq_len = prompt_len`). Inside:

1. **Embedding lookup** (`generate.rs:475`). Each token ID indexes into
   `weights.embed_tokens.data` (the `[vocab_size, hidden_size]` embedding matrix) and copies
   one row into `scratch.hidden`.

2. **Transformer layers** (`generate.rs:494`). For each of `num_hidden_layers` layers:

   a. **Pre-attention RMS norm** (`forward/cpu.rs: rms_norm`). Normalizes
   `scratch.hidden` in place using the layer's `input_layernorm_weight`.

   b. **Fused QKV projection** (`matmul_bt`). A single `[seq_len, hidden_size] ×
      [qkv_dim, hidden_size]^T` matmul produces `[seq_len, qkv_dim]` in `scratch.qkv_buf`.
   The output is then scattered into separate `scratch.q_buf`, `scratch.k_buf`,
   `scratch.v_buf` slices.

   c. **Per-head QK RMS norm**. Each Q head and each KV head is independently RMS-normed
   using `lw.q_norm_weight` and `lw.k_norm_weight`.

   d. **RoPE** (`rope.rs: RopeTable::apply`). `RopeTable::apply` rotates each Q and K head
   in place using the precomputed sin/cos table at position `start_pos + token_index`.

   e. **KV cache write** (`generate.rs:563`). K and V are converted f32→f16 and appended to
   `cache.k_buffer_mut(layer_idx)` and `cache.v_buffer_mut(layer_idx)`.

   f. **Attention** (`generate.rs:620, compute_attention`). The cached K and V buffers are
   dequantized f16→f32 into `scratch.cached_k_f32` / `scratch.cached_v_f32`. Then
   `compute_attention` (`generate.rs:875`) runs GQA with causal masking: Q @ K^T scaled
   by `1/sqrt(head_dim)`, softmax, weighted sum of V. During prefill (`q_seq_len > 1`)
   the generic path applies a causal mask (`ki > start_pos + qi → NEG_INFINITY`). During
   decode (`q_seq_len == 1`) a faster path loads each KV row once and dots it against all
   Q heads in its GQA group, then uses an online safe-softmax (running `(m, l)`
   accumulation) before accumulating V via NEON on AArch64 for `head_dim = 128`.

   g. **Output projection** (`matmul_bt`). `[seq_len, q_dim] × [hidden_size, q_dim]^T →
      [seq_len, hidden_size]` writes the attention output back to `scratch.hidden`.

   h. **Residual add**. `scratch.hidden[i] += scratch.residual[i]` for each element.

   i. **Post-attention RMS norm** using `lw.post_attention_layernorm_weight`.

   j. **SwiGLU FFN**. A fused gate+up projection (`[seq_len, hidden] × [2*inter, hidden]^T`)
   fills `scratch.gate_up_buf`. Gate and up halves are scattered, then `silu_inplace` is
   applied to the gate half, followed by `elementwise_mul(gate, up)` to produce the
   activated hidden state. The down projection (`[seq_len, inter] × [hidden_size, inter]^T`)
   writes the FFN output. A second residual add completes the layer.

3. **KV cache advance** (`cache.advance_by(seq_len)`, `generate.rs:702`). Increments the
   logical sequence length by the number of tokens just processed.

4. **Final RMS norm** (`generate.rs:706`). Applied to the last token's hidden state only
   (`hidden[last_start..]`).

5. **Logits projection** (`generate.rs:716`). Qwen3 ties `lm_head` with `embed_tokens`:
   `logits = hidden_last @ embed_tokens^T`. This is a single `matmul_bt` call producing a
   `[1, vocab_size]` vector written into `scratch.logits`.

### 6. Sample First Token

```
sampler.sample(&scratch.logits[..cfg.vocab_size])   // sampling.rs
```

`Sampler::sample` applies temperature scaling, top-k filtering, top-p nucleus filtering, and
repetition penalty, then draws a token ID. If a `GrammarEngine` is active, `mask_logits` is
called on `scratch.logits` before sampling and `advance` is called after (`grammar/mod.rs`).

### 7. Decode Loop

```
for step in 0..max_new_tokens.saturating_sub(1) {
    forward_with_cache(model, &[last_token], &mut cache, pos-1, &mut scratch, max_seq)?;
    token = sampler.sample(&scratch.logits[..vocab_size]);
    generated_ids.push(token);
}
```

Each decode step calls `forward_with_cache` with a single-token input. The KV cache grows by
one position per step. The loop stops on EOS, when the cache is full (optional
`kv_cache_capacity` cap), or when `max_new_tokens` is exhausted.

### 8. Detokenize

```
tokenizer.decode(&generated_ids)   // tokenizer/common.rs:44
```

`BpeTokenizer::decode` converts generated token IDs back to a UTF-8 string using GPT-2
byte-level decoding. If `GenerateConfig::include_prompt` is set, the prompt string is
prepended to the result.

### Setting a Breakpoint

To inspect the forward pass at runtime, set a breakpoint on `forward_with_cache`
(`generate.rs:451`). The prefill call has `seq_len = prompt_len`; each decode call has
`seq_len = 1`. The logits for the sampled token are always in `scratch.logits[0..vocab_size]`
after the function returns.

### Qwen3.5 Generation Path

The trace above covers the older Qwen3 (`crate::generate`/`crate::sampling`) decoder path. The
newer Qwen3.5 architecture (`crate::model::qwen35`) has a separate, structurally similar trace,
verified against `crates/inference/src/model/qwen35/` and `crates/inference/src/forward/`:

```text
local safetensors directory -> Qwen35Model::from_safetensors
  -> required tensor validation (validate_required_tensor_names)
  -> load_weights                                    (model/qwen35/loading.rs)
  -> BpeTokenizer::from_tokenizer_json                (tokenizer.json)
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

Note: `Qwen35Model::generate_with_batch_prefill` (`forward/batch_prefill.rs`) is a separate,
**deprecated** (since 0.5.1, issue #807) standalone decode loop that predates the delegation
above — it calls the same batched-prefill core directly but is not the canonical entry point.
New callers should use `Qwen35Model::generate` / `generate_streaming`.

Note: the tokenizer trait method is `tokenize`, not `encode` — `encode` is a method on the
separate `QwenModel` embedding path (`model/qwen.rs`), not on the `Tokenizer` trait used here.

For a deeper, function-by-function walkthrough of this path (including the GDN/full-attention
dispatch inside each layer), see [`docs/forward-pass.md`](forward-pass.md). For library usage of
either path, see [`docs/inference-usage.md`](inference-usage.md).
