# lattice-inference — Contributor Map

**Stability tier: Experimental.** This crate has high churn, 153 `unsafe` blocks, and 22
`dead_code_allows`. It is NOT intended for direct use by platform or feature crates.
Consumers should go through `lattice-embed`. The unsafe blocks are documented in
`foundation/STABILITY.md §Tech Debt`. Tracking issue: #1306.
See `foundation/STABILITY.md` for the full stability policy.

If you are trying to generate embeddings in an application, start with `lattice-embed`, not
this crate. The README you are reading is a contributor's map for people opening PRs on the
inference engine itself.

---

## Module Map

### Grouped modules

These modules contain submodules (directories with their own `mod.rs`).

| Module | Responsibility |
|--------|---------------|
| `attention` | Attention kernel variants: standard MHA, GQA, flash (CPU tiled), flash-causal (decoder prefill), GDN (gated decay network), GDN-fused, native sparse, and differential. Exposes `AttentionTag` for typed dispatch. |
| `forward` | Compute backends: scalar CPU, NEON (AArch64), Metal (macOS MSL), WGPU, Q8/f16 CPU kernels, tiled GEMM (`metal_gemm`, `gpu_gemm`), batched prefill, and BitNet kernel. Exports `matmul_bt`, `rms_norm`, `silu_inplace`, `elementwise_mul` from the CPU submodule. |
| `model` | Model configs and loaders: `BertModel`/`BertConfig`, `QwenModel`/`QwenConfig`, `Qwen35Model`, `CrossEncoderModel`, `BitNet` config. Each submodule owns its safetensors load path and forward-pass dispatch. |
| `tokenizer` | Tokenizer implementations: `WordPieceTokenizer` (BERT), `BpeTokenizer` (Qwen3/GPT-2 byte-level), `SentencePieceTokenizer`. Shared `Tokenizer` trait and `load_tokenizer` auto-detect helper live in `tokenizer/common.rs`. |
| `vision` | Qwen3-VL vision encoder (ADR-049): ViT forward pass, patch preprocessing, MLP merger, `MultimodalInput` type. CPU-only; Metal GPU path is deferred. |
| `weights` | Weight storage formats (`Tensor2D`), safetensors memory-mapped loading, f32/f16/Q8/Q4 weight structs, sharded-checkpoint support. |

### Standalone modules

| Module | Responsibility |
|--------|---------------|
| `batch` | Continuous batching engine (ADR-048): `BatchWorker`, `FifoScheduler`, `Sequence`, `SequenceManager`, chunked prefill interleaved with decode steps. |
| `download` | Model file cache and conditional download (`ensure_model_files`). Disabled unless the `download` feature is enabled. |
| `error` | `InferenceError` — the crate's single error type, re-exported at the crate root. Stable; adding variants is backward-compatible. |
| `generate` | Text generation loop for Qwen3: `generate()` entry point, `GenerateConfig`, `GenerateOutput`, `forward_with_cache` inner loop, `compute_attention` GQA kernel. |
| `grammar` | Grammar-constrained decoding (ADR-046): `GrammarEngine` compiles a JSON Schema or GBNF spec to a byte-level PDA, then masks logits via `mask_logits` and advances state via `advance` each decode step. |
| `kv_cache` | Two KV cache implementations: `FlatKVCache` (contiguous pre-allocated, single request) and `PagedKVCache` (256-token pages, LRU eviction, multi-model serving). Includes `PrefixPageCache` for prefix reuse. |
| `lora_hook` | `LoraHook` trait: defines the `apply(layer_idx, module, x, output)` contract that the forward pass calls after each linear projection. `NoopLoraHook` is the zero-overhead default. |
| `metrics` | Inference metrics (ADR-061): `MetricsMode` enum and `LayerMetrics` schema. Zero overhead when `MetricsMode::Off`. |
| `pool` | Pooling strategies for producing a single embedding vector: `BertPooling` (Mean or CLS), `mean_pool`, `cls_pool`, `last_token_pool` (Qwen3), `l2_normalize`. |
| `pruning` | ShortGPT block-influence scorer (ADR-060 D2 seed): `BlockInfluenceAccumulator` and `BlockInfluence`. Standalone math; the calibration pipeline hooks are future work. |
| `quant` | Quantization pre-transforms: `quarot` (Walsh-Hadamard rotation primitives, ADR-044). Not yet wired into the Q4 weight path. |
| `rope` | Rotary position embeddings: `RopeTable` precomputes sin/cos tables and applies them in-place to Q and K slices before attention. |
| `sampling` | Token sampling for generation: `Sampler`, `SamplingConfig` (temperature, top-k, top-p, repetition penalty). |
| `speculative` | N-gram prompt-lookup speculative decoding: `NgramSpeculator::speculate` + `verify_draft` (low-level) and `generate_with_speculation` (high-level wrapper). |

### Feature-gated modules

| Module | Feature flag | Responsibility |
|--------|-------------|----------------|
| `mixture` | `mixture` | Adapter routing and mixture-of-LoRA support layered on top of `lora_hook`. |
| `backward` | `train-backward` | Reverse-mode autodiff for LoRA training: `tape`, `ops`, `attention_gqa`, `gradcheck`. Submodule of `lattice-tune`'s training loop. |

---

## Run an Example

Qwen3.5 text generation demo (`src/bin/qwen35_generate.rs`):

```sh
cargo run --release -p lattice-inference --bin qwen35_generate -- \
  --model-dir PATH --prompt "Hello" --max-tokens 64 --seed 42
```

Qwen3 embedding benchmark (`examples/bench_embedding.rs`, registered in `Cargo.toml` as an
`[[example]]` target — despite an outdated `--bin` reference in the file's own header comment,
run it with `--example`):

```sh
cargo run --release -p lattice-inference --example bench_embedding --features f16
```

For the concise forward-pass trace (safetensors load → tokenize → prefill → decode-loop
sampling), see [`docs/architecture.md` § Forward Pass Pipeline](../../docs/architecture.md#forward-pass-pipeline).
A deeper Qwen3.5-specific walkthrough lives in `docs/forward-pass.md`.

---

## Supported Architectures

| Architecture | Submodule | Notes |
|---|---|---|
| BERT / BGE (encoder-only) | `model/bert.rs` | Bidirectional attention, WordPiece tokenizer, mean or CLS pooling. Covers BGE-small/base/large, E5-small/base, all-MiniLM, paraphrase-multilingual-MiniLM. |
| Qwen3 / Qwen3.5 (decoder-only) | `model/qwen.rs`, `model/qwen35/` | Causal GQA, RoPE, RMSNorm, SwiGLU FFN, BPE tokenizer, last-token pooling. Covers Qwen3-Embedding-0.6B and 4B. |
| BitNet | `model/bitnet_config.rs` | 1.58-bit quantized weight architecture. |
| Qwen3-VL (vision) | `vision/` | ViT patch encoder + MLP merger for multimodal input; CPU-only (ADR-049 v0 scope). |
| Cross-encoder | `model/cross_encoder.rs` | BERT-style reranker with a classification head on top of the CLS token. |

---

## Where to Start for Common PR Types

**New model architecture.** Add a submodule under `model/` following the pattern in
`model/qwen.rs` (config struct, weights struct, `load` function, `encode`/`forward` method).
Wire the new architecture into `lattice-embed`'s `EmbeddingModel` enum when ready.

**Attention kernel change.** Work in `attention/` and `forward/`. A new kernel goes in its
own file under `attention/` (e.g. `attention/flash_causal.rs`), registers a variant on
`AttentionTag`, and dispatches from the relevant `forward/` backend. See
`docs/architecture.md` for the full forward-pass walkthrough before touching this path.

**Sampling or generation.** `sampling.rs` owns the `Sampler` and `SamplingConfig` structs.
The generate loop in `generate.rs` is where sampling is called; grammar-constrained paths
also live there.

**LoRA adapter injection.** The `LoraHook` trait in `lora_hook.rs` is the injection point.
The training-side implementation lives in `crates/tune`. Backward-pass gradients are behind
the `train-backward` feature in `backward/`.

**Quantization.** Q4 and Q8 weight structs are in `weights/q4_weights.rs` and
`weights/q8_weights.rs`. Pre-transform math (QuaRot) is in the `quant/quarot/` module
(`hadamard.rs`, `convert.rs`, `pipeline.rs`). Wiring QuaRot into the Q4 load path is tracked
in ADR-044.

---

## Cross-References

- `docs/architecture.md` — crate dependency graph, design philosophy, and forward-pass pipeline.
- `AGENTS.md` — contribution policy, SIMD verification rules, agent attribution requirements.
- `docs/adr/INDEX.md` — full list of architecture decision records by crate.
