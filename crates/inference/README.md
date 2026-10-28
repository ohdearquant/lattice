# lattice-inference — Contributor Map

**Stability tier: Experimental.** This crate has high churn, 295 `unsafe` blocks and 59
`dead_code_allows` as of this commit. Both counts exclude comment lines, so a doc mentioning
the pattern does not inflate them; recompute with
`grep -rn 'unsafe {' crates/inference/src --include='*.rs' | grep -vE ':\s*//' | wc -l`
and the same pipeline over `allow(dead_code` rather than trusting these numbers verbatim. It is NOT intended for direct use by platform or feature crates.
Consumers should go through `lattice-embed`.

If you are trying to generate embeddings in an application, start with `lattice-embed`, not
this crate. The README you are reading is a contributor's map for people opening PRs on the
inference engine itself.

---

## Module Map

### Grouped modules

These modules contain submodules (directories with their own `mod.rs`).

| Module      | Responsibility                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `attention` | Attention kernel variants: standard MHA, GQA, flash (CPU tiled), flash-causal (decoder prefill), GDN (gated decay network), GDN-fused, native sparse, and differential. Exposes `AttentionTag` for typed dispatch.                                                                                                                                                                                                                                   |
| `forward`   | Compute backends: scalar CPU, NEON (AArch64), Metal (macOS MSL), WGPU, Q8/f16 CPU kernels, tiled GEMM (`metal_gemm`, `gpu_gemm`), batched prefill, and BitNet kernel. Exports `matmul_bt`, `rms_norm`, `silu_inplace`, `elementwise_mul` from the CPU submodule.                                                                                                                                                                                     |
| `model`     | Model configs and loaders: `BertModel`/`BertConfig`, `QwenModel`/`QwenConfig`, `Qwen35Model` (config in the standalone `qwen35_config.rs`, separate from the `qwen35/` submodule directory), the `gemma4_*` family (`gemma4_config`/`gemma4_model`/`gemma4_loading`/`gemma4_ops`/`gemma4_preflight`/`gemma4_weights`/`gemma4_cache`), `CrossEncoderModel`, `BitNet` config. Each submodule owns its safetensors load path and forward-pass dispatch. |
| `tokenizer` | Tokenizer implementations: `WordPieceTokenizer` (BERT), `BpeTokenizer` (Qwen3/GPT-2 byte-level), `SentencePieceTokenizer`. Re-exported from `lattice-tokenizer`; the shared `Tokenizer` trait and `load_tokenizer` auto-detect helper live in `crates/tokenizer/src/common.rs`.                                                                                                                                                                      |
| `vision`    | Vision encoder paths (ADR-049): the Qwen3.5 vision tower and the PaddleOCR-VL encoder, plus the `MultimodalInput` type.                                                                                                                                                                                                                                                                                                                              |
| `weights`   | Weight storage formats (`Tensor2D`), safetensors memory-mapped loading, f32/f16/Q8/Q4 weight structs, sharded-checkpoint support.                                                                                                                                                                                                                                                                                                                    |

### Standalone modules

| Module        | Responsibility                                                                                                                                                                                                                                                                                                                                                                           |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `batch`       | Continuous batching engine (ADR-048): `BatchWorker`, `FifoScheduler`, `Sequence`, `SequenceManager`, chunked prefill interleaved with decode steps.                                                                                                                                                                                                                                      |
| `download`    | Model file cache and conditional download (`ensure_model_files`), re-exported from `lattice-model-io`. Disabled unless the `download` feature is enabled.                                                                                                                                                                                                                                |
| `error`       | `InferenceError` — the crate's single error type, re-exported at the crate root. Stable; adding variants is backward-compatible.                                                                                                                                                                                                                                                         |
| `grammar`     | Grammar-constrained decoding (ADR-046): `GrammarEngine` compiles a JSON Schema or GBNF spec to a byte-level PDA, then masks logits via `mask_logits` and advances state via `advance` each decode step.                                                                                                                                                                                  |
| `kv_cache`    | Two KV cache implementations: `FlatKVCache` (contiguous pre-allocated, single request) and `PagedKVCache` (256-token pages, LRU eviction, multi-model serving). Includes `PrefixPageCache` for prefix reuse.                                                                                                                                                                             |
| `lora_hook`   | `LoraHook` trait: defines the `apply(layer_idx, module, x, output)` contract that the forward pass calls after each linear projection. `NoopLoraHook` is the zero-overhead default.                                                                                                                                                                                                      |
| `metrics`     | Inference metrics (ADR-061): `MetricsMode` enum and `LayerMetrics` schema. Zero overhead when `MetricsMode::Off`.                                                                                                                                                                                                                                                                        |
| `pool`        | Pooling strategies for producing a single embedding vector: `BertPooling` (Mean or CLS), `mean_pool`, `cls_pool`, `last_token_pool` (Qwen3), `l2_normalize`.                                                                                                                                                                                                                             |
| `pruning`     | ShortGPT block-influence scorer (ADR-060 D2 seed): `BlockInfluenceAccumulator` and `BlockInfluence`. Standalone math; the calibration pipeline hooks are future work. `lattice prune-score`'s own layer-importance scorer is a last-token variant of the same idea, not a caller of this module, and its output artifact names itself accordingly (`method: "last_token_block_change"`). |
| `quant`       | Quantization pre-transforms: `quarot` (Walsh-Hadamard rotation primitives, ADR-044). Not yet wired into the Q4 weight path.                                                                                                                                                                                                                                                              |
| `rope`        | Rotary position embeddings: `RopeTable` precomputes sin/cos tables and applies them in-place to Q and K slices before attention.                                                                                                                                                                                                                                                         |
| `sampling`    | Token sampling for generation: `Sampler`, `SamplingConfig` (temperature, top-k, top-p, repetition penalty).                                                                                                                                                                                                                                                                              |
| `speculative` | N-gram prompt lookup: low-level `NgramSpeculator::speculate` + `verify_draft`, the model-agnostic `generate_with_speculation`, and a state-owned Metal method of the same name.                                                                                                                                                                                                          |

### Feature-gated modules

The `image` feature enables PNG/JPEG decoding and is enabled by default and by `lattice-serve`.
Without it, image decoding APIs remain available but return an error naming the required feature.

| Module     | Feature flag     | Responsibility                                                                                                                     |
| ---------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `mixture`  | `mixture`        | Adapter routing and mixture-of-LoRA support layered on top of `lora_hook`.                                                         |
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

| Architecture                   | Submodule                                                     | Notes                                                                                                                                                                                                                                                  |
| ------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| BERT / BGE (encoder-only)      | `model/bert.rs`                                               | Bidirectional attention, WordPiece tokenizer, mean or CLS pooling. Covers BGE-small/base/large, E5-small/base, all-MiniLM, paraphrase-multilingual-MiniLM.                                                                                             |
| Qwen3 / Qwen3.5 (decoder-only) | `model/qwen.rs`, `model/qwen35/`                              | Causal GQA, RoPE, RMSNorm, SwiGLU FFN, BPE tokenizer, last-token pooling. Covers Qwen3-Embedding-0.6B and 4B.                                                                                                                                          |
| BitNet                         | `model/bitnet_config.rs`                                      | 1.58-bit quantized weight architecture.                                                                                                                                                                                                                |
| Qwen3.5 vision                 | `vision/`                                                     | Qwen3.5 vision tower and merger for multimodal input (ADR-049).                                                                                                                                                                                        |
| Cross-encoder                  | `model/cross_encoder.rs`                                      | BERT-style reranker with a classification head on top of the CLS token.                                                                                                                                                                                |
| EmbeddingGemma 2               | `model/embeddinggemma2.rs`, `model/embeddinggemma2_vision.rs` | 24-layer bidirectional Gemma-family encoder, mean pooling to a 768-dimension vector; callers take leading widths, each L2-normalized. Text and image parts on the CPU in f32; an explicit f32 Metal text encoder (`forward/metal_embeddinggemma2.rs`). |
| Gemma 4 E2B (decoder-only)     | `model/gemma4_model.rs`                                       | Text-only greedy generation on the CPU in f32: sliding-window local layers interleaved with full-attention layers, shared K/V across the final layers, attention without score scaling.                                                                |
| PaddleOCR-VL / ERNIE-4.5       | `model/paddleocr_vl.rs`, `model/ernie45.rs`                   | Image + prompt to greedy text: SigLIP-shaped vision encoder and projector, ERNIE-4.5 dense decoder with sectioned multimodal RoPE. CPU f32, plus an explicit Metal decoder (`forward/metal_ernie45.rs`).                                               |

---

## Where to Start for Common PR Types

**New model architecture.** Add a submodule under `model/` following the pattern in
`model/qwen.rs` (config struct, weights struct, `load` function, `encode`/`forward` method).
Wire the new architecture into `lattice-embed`'s `EmbeddingModel` enum when ready.

**Attention kernel change.** The CPU attention kernels (standard, tiled, grouped-query,
decode and the other variants), the GatedDeltaNet kernels and the `AttentionKind`/`AttentionTag`
taxonomy live in `lattice-compute` (`crates/compute/src/attention/`, e.g. `flash_causal.rs`) and
keep their `lattice_inference::attention` paths through named re-exports. GDN backward stays in
`attention/` here. A new kernel goes in its own file in compute, registers a variant on
`AttentionTag`, and dispatches from the relevant `forward/` backend. See
`docs/architecture.md` for the full forward-pass walkthrough before touching this path.

**Sampling or generation.** `sampling.rs` owns the shared `Sampler` and `SamplingConfig`
primitives. Canonical Qwen3.5 generation lives in `model/qwen35/generation.rs`, with
backend-specific entry points under `forward/`; grammar-constrained decoding uses the
shared `grammar` module.

**LoRA adapter injection.** The `LoraHook` trait in `lora_hook.rs` is the injection point.
The training-side implementation lives in `crates/tune`. Backward-pass gradients are behind
the `train-backward` feature in `backward/`.

**Quantization.** Q4 and Q8 weight structs are in `crates/weights/src/q4_weights.rs` and
`crates/weights/src/q8_weights.rs`. QuaRot Hadamard transforms and rotation absorption live
in `crates/quant/src/quarot/hadamard.rs` and `crates/quant/src/quarot/rotation.rs`, and the
conversion orchestration in `crates/quant/src/quarot/convert.rs` and `pipeline.rs`.
The inference facade retains the existing `quant::quarot` module paths. Wiring QuaRot into the
Q4 load path is tracked in ADR-044.

---

## Cross-References

- `docs/architecture.md` — crate dependency graph, design philosophy, and forward-pass pipeline.
- `AGENTS.md` — contribution policy, SIMD verification rules, agent attribution requirements.
- `docs/adr/INDEX.md` — full list of architecture decision records by crate.

## Crate reference

The sections below are generated by `scripts/crate-readme.py` from the source tree and are refreshed after pushes to `main`. Text between the `generated` markers is overwritten; edit the source or the script instead.

<!-- generated:facts:begin -->

### Facts

- Version: `0.15.0`
- Minimum supported Rust version: `1.93`
- Edition: `2024`
- License: `MIT OR Apache-2.0`
- crates.io: <https://crates.io/crates/lattice-inference>
- API documentation: <https://docs.rs/lattice-inference>

<!-- generated:facts:end -->

<!-- generated:layout:begin -->

### Source layout

The tracked files under `src/`; each Rust file shows the first sentence of its `//!` module documentation.

<details>
<summary><code>src/</code>: 412 files</summary>

- `attention/`
  - `decode.rs`: Single-token decode attention, re-exported from `lattice-compute`.
  - `differential.rs`: Differential attention, re-exported from `lattice-compute`.
  - `flash.rs`: Tiled CPU attention over `lattice-compute`'s projection-view kernel.
  - `flash_causal.rs`: Tiled causal flash attention, re-exported from `lattice-compute`.
  - `gated.rs`: Sigmoid output gating for gated attention, re-exported from `lattice-compute`.
  - `gdn.rs`: GatedDeltaNet linear attention, re-exported from `lattice-compute`.
  - `gdn_backward.rs`: Reverse-mode backward (VJP) for a single GatedDeltaNet token sequence.
  - `gdn_fused.rs`: Fused GatedDeltaNet decode step, re-exported from `lattice-compute`.
  - `gqa.rs`: Grouped-query attention, re-exported from `lattice-compute`.
  - `mod.rs`: Attention module index for `decode`, `differential`, `flash`, `flash_causal`, `gated`, `gdn`, `gdn_backward`, `gdn_fused`, `gqa`, `native_sparse`, and `standard`; re-exports the standard attention helpers.
  - `native_sparse.rs`: Native sparse attention, re-exported from `lattice-compute`.
  - `softmax_row.rs`: Fail-closed softmax row contract, re-exported from `lattice-compute`.
  - `standard.rs`: Standard multi-head attention over `lattice-compute`'s projection-view kernels.
  - `standard/`
    - `padded_reference.rs`: Test-only padded batched attention reference for the BERT packed-batch parity test.
    - `tests.rs`: The facade's BERT-layer attention entry points against `lattice-compute`'s projection-view kernels.
- `backward/`
  - `attention_gqa.rs`: Materialized GQA attention backward data and helpers, including `AttnCache`, `AttnGrads`, `gqa_backward`, and `gqa_forward_with_cache`.
  - `gradcheck.rs`: Tiny-model gradient-checking utilities, including tiny weights, tiny forward/loss helpers, and `gradcheck_lora`.
  - `mod.rs`: Backward-pass module index for `attention_gqa`, `gradcheck`, `ops`, `simd` (test and `test-utils` builds only), and `tape`.
  - `ops.rs`: Backward primitives for cross entropy, linear VJP, LoRA VJP, RMSNorm backward, RoPE backward, and SwiGLU backward.
  - `simd/`
    - `mod.rs`: Test-only forced-dispatch controls for the staged backward SIMD workstream.
  - `tape.rs`: Forward-cache structs and helper forwards for RMSNorm and SwiGLU.
  - `tape/`
    - `tests.rs`
- `batch/`
  - `admission_tests.rs`
  - `config.rs`: Configuration for the continuous batching engine.
  - `mod.rs`: Continuous batching engine for multi-sequence inference (ADR-048).
  - `scheduler.rs`: Iteration-level batch scheduler for continuous batching (ADR-048).
  - `scheduler/`
    - `tests.rs`
  - `sequence.rs`: Per-sequence state tracking for the continuous batching engine.
  - `sequence/`
    - `tests.rs`
  - `worker.rs`: Continuous batching loop and GDN state pool (ADR-048).
  - `worker/`
    - `tests.rs`
- `bert_pooling_fixture.rs`: Deterministic, synthetic BERT checkpoints for pooling integration controls.
- `bert_pooling_fixture/`
  - `tests.rs`
- `bin/`
  - `backfill_qwen3.rs`: Backfill embeddings from BGE-Small-384d to Qwen3-1024d.
  - `bench_decode_ab.rs`: Apples-to-apple decode benchmark (A/B slope method).
  - `bench_decode_slopefit.rs`: ADR-064 Phase-0: decode slope/intercept fit harness.
  - `bench_gdn_prefill_ab.rs`: GDN-recurrence-isolating prefill A/B bench harness (issue #175, Phase B).
  - `bench_gemma4_driver_pair.rs`: Issue #1597 measurement cell: "PAIR on comparable output semantics, setup/prefill/decode separately." Replays the stage-5 golden prompt through Gemma 4's fixed-count greedy path (`Gemma4Model::generate_greedy`) and its shared-driver path (`Gemma4Model::generate_streaming_with_cancel`) at the SAME token count and asserts the two produce identical token ids.
  - `bench_logit_dump.rs`: Logit dump binary for divergence analysis vs MLX.
  - `bench_lora_mixture.rs`: LoRA mixture blend latency benchmark.
  - `bench_vision_prefill_ab.rs`: Issue #1336 multimodal-prefill A/B measurement harness.
  - `chat_metal.rs`: Metal GPU generation — JSON event mode for Lattice Studio app, plus interactive REPL.
  - `dump_quarot_q4_golden.rs`: QuaRot+Q4 composed-path golden dumper (issue #320).
  - `eval_perplexity.rs`: Perplexity evaluator for Qwen3.5 models on a UTF-8 text corpus.
  - `gramperf_profile.rs`: Profiling harness for issue #734 (grammar-constrained decode ~4.5x slower per token when a schema exceeds `MAX_GRAMMAR_STATES`).
  - `gramtime_profile.rs`: Measurement harness for issue #1342 (GrammarEngine::new doc-comment timing figures).
  - `ppl_metal.rs`: Metal Qwen3.5 perplexity binary entry point.
  - `qwen35_debug.rs`: Debug diagnostic for Qwen3.5-2B — checks each stage of the forward pass.
  - `qwen35_generate.rs`: Qwen3.5 text generation demo.
- `bounded_config.rs`
- `bounded_read.rs`
- `decoder.rs`: Decoder sessions (ADR-090 D1/D2).
- `decoder/`
  - `bench_support.rs`: Measurement-only seams for the per-token allocation instrument (ADR-090 D7, row R01).
  - `gemma_cpu.rs`: `GemmaCpuSession`: the Gemma 4 E2B text CPU concrete `super::DecoderSession` (ADR-090 row R04), the Gemma sibling of `super::qwen_cpu::QwenCpuSession`.
  - `gemma_cpu_bench.rs`: Manual real-checkpoint measurement of the production Gemma CPU session.
  - `qwen_cpu.rs`: `QwenCpuSession`: the first concrete `super::DecoderSession` (ADR-090 row B).
  - `qwen_cpu/`
    - `tests.rs`
  - `qwen_metal.rs`: `QwenMetalSession`: the ordinary Qwen3.5 Metal generation path behind the object-safe `super::DecoderSession` boundary (ADR-090 D1).
  - `standalone_cpu.rs`: `StandaloneCpuSession`: one `super::DecoderSession` for the three standalone CPU generation wrappers (`generate_f16`, `generate_q8`, `generate_q8_neon`).
  - `standalone_cpu/`
    - `parity.rs`
    - `tests.rs`
  - `standalone_cpu_q8_scratch_checks.rs`
- `download.rs`: Model-file ensure/download flow, canonical model names, checksum verification, and download helper.
- `encoder.rs`: The text encoder contract of `lattice-arch` (ADR-103 Amendment 21 item 3) and its adapters for the embedding models this crate loads: BERT-family, Qwen3 and EmbeddingGemma 2.
- `encoder/`
  - `tests.rs`
- `error.rs`: Inference error taxonomy.
- `family.rs`: The family contract of `lattice-arch` (ADR-103 Amendment 21 item 3) and the descriptors for the model families this crate loads.
- `family/`
  - `tests.rs`
- `forecast/`
  - `chronos2.rs`: Chronos-2: an encoder-only forecasting model over patches of a scaled series, with attention along time within each series and along the batch within each group of series (Ansari et al., "Chronos-2", 2025; reference implementation `amazon-science/chronos-forecasting`).
  - `fft.rs`: Complex radix-2 FFT in f64, and the circular convolution that long-convolution layers use.
  - `mod.rs`: Time-series forecasting: shared context preparation and native forecasting models.
  - `prep.rs`: Context preparation shared by forecasting models: missing-value imputation, padding to a fixed window, striding, and resampling a forecast back to its requested length.
  - `reverso.rs`: Reverso: a univariate forecasting model that alternates long circular convolutions with delta-rule linear attention, then reads a fixed block of future steps out through one cross-attention head (Fu et al., "Reverso", 2025; reference implementation `shinfxh/reverso`).
  - `toto2.rs`: Toto 2.0: a decoder-only forecasting model over patches of a causally scaled series, with attention along time within each variate and across the variates of one series (Khwaja et al., "Toto 2.0: Time Series Forecasting Enters the Scaling Era", 2026; reference implementation `DataDog/toto`, `toto2/toto2/model.py`).
- `forward/`
  - `batch_prefill.rs`: Shared batched prompt-prefill core for Qwen3.5 hybrid attention.
  - `batch_prefill_scratch_tests.rs`
  - `bitnet_kernel.rs`: Ternary matmul kernels for BitNet b1.58 inference.
  - `bitnet_kernel/`
    - `tests.rs`
  - `cpu/`
    - `mod.rs`: CPU forward kernels, re-exported from `lattice-compute`.
    - `q8_matmul.rs`: CPU matrix multiplication for per-row symmetric INT8 weights.
    - `q8_matmul/`
      - `scratch_tests.rs`
      - `tests.rs`
  - `cpu_f16.rs`: F16-weight forward pass for Qwen3.5-2B.
  - `cpu_f16/`
    - `tests.rs`
  - `cpu_q8.rs`: Q8-weight forward pass for Qwen3.5-2B.
  - `cpu_q8_scratch_bench.rs`
  - `gdn_chunk_ref.rs`: Host-side f32 reference oracle for the chunkwise WY/DPLR Gated-DeltaNet (GDN) prefill recurrence.
  - `gdn_chunk_ref/`
    - `tests.rs`
  - `gpu/`
    - `inner.rs`: Inner WGPU forward module index and API-type re-exports.
    - `inner/`
      - `api.rs`: GPU result/error types and Qwen3 GPU configuration and weight structs.
      - `bind_groups.rs`: WGPU bind-group structs and constructors for layer and global bindings.
      - `buffers.rs`: GPU weight and activation buffer structs plus upload and activation-buffer creation helpers.
      - `dims.rs`: GPU forward dimensions, configuration validation, and runtime feasibility validation.
      - `dispatch.rs`: WGPU dispatch helpers for matmul, RMSNorm, copy, add, SiLU, multiply, RoPE, attention scores, softmax, and context.
      - `params.rs`: GPU shader/workgroup constants and parameter-slot constants.
      - `pipelines.rs`: WGPU bind-group-layout and compute-pipeline creation helpers.
      - `shaders.rs`: WGSL shader source constants for matmul, RMSNorm, copy, add, and related GPU kernels.
      - `state.rs`: GPU model state, forward entry, and embedding helper.
      - `tests.rs`: GPU-vs-CPU tiny-model tests and CPU reference helpers.
      - `util.rs`: GPU utility helpers for RoPE table building, byte counts, `usize` to `u32` conversion, and ceiling division.
    - `mod.rs`: GPU forward path for a Qwen3-style decoder using `wgpu` compute pipelines.
  - `gpu_gemm.rs`: WebGPU-accelerated GEMM via `wgpu`.
  - `metal.rs`: Direct Metal compute shader forward pass for Qwen3-Embedding-0.6B.
  - `metal_embeddinggemma2.rs`: EmbeddingGemma 2 text encoder, f32 forward pass on Metal.
  - `metal_embeddinggemma2/`
    - `state.rs`: Persistent f32 EmbeddingGemma 2 forward on Metal.
    - `state/`
      - `tests.rs`: Kernel-level tests for the EmbeddingGemma 2 Metal state.
      - `vision.rs`: EmbeddingGemma 2 vision tower, f32 forward pass on Metal.
      - `vision/`
        - `tests.rs`: Kernel-level tests for the EmbeddingGemma 2 vision tower on Metal.
  - `metal_ernie45.rs`: ERNIE-4.5 f32 prefill and cached decode on Metal.
  - `metal_ernie45/`
    - `state.rs`: Persistent f32 ERNIE forwards using shared Metal primitives.
    - `state/`
      - `cached.rs`: Explicit cache ownership and embedding-driven forwards.
    - `tests.rs`
    - `tests/`
      - `cache_guards.rs`
      - `real/`
        - `mrope.rs`
  - `metal_gemm.rs`: Metal GPU-accelerated GEMM for Apple Silicon.
  - `metal_qwen35.rs`: Metal GPU forward pass for Qwen3.5-2B hybrid attention model.
  - `metal_qwen35/`
    - `gdn_state_traffic_tests.rs`
    - `has_finite_logit_tests.rs`
    - `inner/`
      - `bench_support.rs`
      - `blend_tests.rs`
      - `buffers.rs`
      - `diagnostics.rs`
      - `dispatch.rs`
      - `embedding_table.rs`: The `embed_tokens` source for the CPU row lookup, split out of `inner`.
      - `engine.rs`
      - `gdn_state.rs`
      - `generation.rs`
      - `layers/`
        - `gdn.rs`
        - `gqa.rs`
        - `mod.rs`
        - `moe_device_route.rs`: ADR-053 D3': MoE routing on the device for Eager expert storage.
      - `lora.rs`
      - `min_p_compact_sampling_tests.rs`
      - `mtp.rs`
      - `pipelines.rs`
      - `prefill.rs`
      - `prefill_chunk.rs`
      - `prefix_cache.rs`
      - `prefix_generation.rs`
      - `sampling.rs`
      - `spec_types.rs`
      - `speculative.rs`
      - `step.rs`
      - `tests.rs`
      - `tests/`
        - `checkpoint_refusals.rs`: `from_q4_dir` refusals on misconfigured checkpoints.
        - `chunked_prefill.rs`: Chunked batched prefill parity.
        - `cross_turn_kv_cache.rs`: Cross-turn KV prefix cache on a Metal state.
        - `decode_attention.rs`: Decode-attention kernels against the CPU reference.
        - `dispatch.rs`
        - `gdn_chunked_scan.rs`: GDN chunked-scan algebra and its Metal kernels.
        - `grammar.rs`: Grammar-constrained decoding: fail-closed admission, step-zero terminals and the decode loop.
        - `layers.rs`
        - `lm_head.rs`: lm_head block top-k, argmax and real-checkpoint agreement.
        - `lora.rs`: LoRA mixture admission and batched prefill with a loaded adapter.
        - `moe.rs`: MoE Q4 checkpoints: the FFN branch, routed-expert LRU, prefetch and mmap dequant validation.
        - `moe_device_route.rs`: MoE routing on the device for Eager expert storage: the router and shared-gate kernels read the completed current-layer norm, the published trace comes from the route buffers, and the router kernel's ordering and weights follow the documented rule for ties and non-finite logits.
        - `mtp_hidden_tap.rs`: The MTP head's hidden-state tap and the post-final-norm rows it reads.
        - `mtp_verify.rs`: MTP draft and verify: hidden-row verification and the generic loop on a real checkpoint.
        - `mtp_weight_load.rs`: MTP Q4 weight loading on a Metal device.
        - `multimodal_vision.rs`: Multimodal vision generation: cancellation, injected embeddings and M-RoPE.
        - `path_proof_bytes.rs`: `PathProofCounters::reset()` coverage, asserted directly.
        - `perplexity.rs`: Perplexity harness.
        - `pooled_embeddings.rs`: Pooled hidden-state embeddings, CPU against Metal.
        - `prefix_cache_disposition.rs`: What the cross-turn slot holds after each of the streaming exits the tests below drive; it is not a map of every return in `generate_streaming_with_prefix_cache_and_cancel`.
        - `prefix_cache_route.rs`: What the prefix-cache entry leaves behind after each request it routes through the shared decoder driver, over a sweep of scenario x config x cut.
        - `prefix_cache_slots.rs`: Prefix-cache slot isolation and eviction regressions.
        - `q3.rs`: Q3 MLP-only weights: loader and kernel parity.
        - `q4_signed_scale.rs`: A Q4 block's scale may be negative (a block repacked from another format keeps its source scale).
        - `q8_nan_guard.rs`: Q8_0 non-finite input guard.
        - `self_spec.rs`: GDN-first self-speculative decoding.
        - `speculative_route.rs`: Output and end-state characterization of the two speculative decode routes (default MTP with the sequential verifier, and GDN-first self-speculation).
        - `stop_token_sweep.rs`: Cross-path stop-token contract sweep.
        - `suffix_preflight.rs`: Suffix-content preflight in `plan_prefix_request`.
      - `verify.rs`
      - `weights.rs`
    - `metal_command_buffer_pool_tests.rs`
    - `mtp_greedy_round_tests.rs`
    - `mtp_hidden_tap_tests.rs`
    - `mtp_load_warning_tests.rs`
    - `mtp_weights.rs`
    - `multimodal_preflight_tests.rs`
    - `public_scheduling_entry_point_tests.rs`
    - `q4_embed_lookup.rs`: CPU row lookup for a Q4 embedding table read in place from its file.
    - `route_predicate_tests.rs`
    - `self_spec_eos_tests.rs`
    - `topk_boundary_tie_tests.rs`
  - `metal_runtime.rs`
  - `mod.rs`: Forward-kernel module index for batch prefill, BitNet, CPU, f16 CPU, q8 CPU, GPU, Metal, and NEON paths.
  - `moe_expert_cache.rs`: Bounded LRU cache of dequantized MoE routed-expert weight buffers, populated lazily from mmap'd `.q4` files on cache miss (issue #682, Stage 1: "dequant-on-demand routed experts with a bounded LRU cache").
  - `moe_expert_cache/`
    - `tests.rs`
  - `neon.rs`: Native Q8_0 NEON dotprod matvec kernel.
  - `neon/`
    - `tests.rs`
  - `neon_forward.rs`: Q8 NEON forward pass for Qwen3.5-2B CPU inference.
  - `neon_forward/`
    - `tests.rs`
  - `shaders/`
    - `attention_sg_hd64.metal`
    - `decode_reference.metal`
    - `embeddinggemma2.metal`
    - `embeddinggemma2_attention.metal`
    - `embeddinggemma2_vision.metal`
    - `ernie45_decode.metal`
    - `ernie45_embed.metal`
    - `gemm.metal`
    - `gemm_f32_sg.metal`
    - `gemm_q3_tiled.metal`
    - `gemm_q4_tiled.metal`
    - `gemm_q8_tiled.metal`
    - `qwen35.metal`
    - `qwen35_prefill.metal`
  - `signpost.rs`: `os_signpost` instrumentation for the Metal decode path (feature `signpost`, default OFF).
  - `signpost/`
    - `recorder.rs`: Emission-recording seam for tests: records begin/end pairing without touching the real macOS FFI path, so pairing can be asserted on every platform/feature combination `cargo test` runs under.
    - `tests.rs`
  - `wgsl/`
    - `add.wgsl`
    - `attention_context.wgsl`
    - `attention_scores.wgsl`
    - `attention_softmax.wgsl`
    - `copy.wgsl`
    - `gemm.wgsl`
    - `matmul_bt.wgsl`
    - `mul.wgsl`
    - `rms_norm.wgsl`
    - `rope.wgsl`
    - `silu.wgsl`
- `generation.rs`: Generation facade over `lattice-arch`.
- `grammar.rs`: Grammar facade over `lattice-grammar`.
- `kv_cache/`
  - `cross_turn.rs`: Cache definitions re-exported from `lattice-compute`.
  - `flat.rs`: Flat KV cache storage re-exported from `lattice-compute`.
  - `mod.rs`: KV cache for autoregressive transformer decoding.
  - `paged.rs`: Paged KV cache storage re-exported from `lattice-compute`.
  - `prefix.rs`: Cache definitions re-exported from `lattice-compute`.
- `lib.rs`: **Stability tier**: Experimental
- `lora_file.rs`: Load a PEFT or MLX LoRA adapter from a safetensors file.
- `lora_hook.rs`: LoRA adapter hooks, re-exported from `lattice-compute`.
- `measurement.rs`: Internal support for tests and measurement targets.
- `measurement/`
  - `lock_file_refresh_tests.rs`
  - `tests.rs`
- `metrics.rs`: Inference metrics and entropy accumulation, re-exported from `lattice-compute`.
- `mixture.rs`: Lightweight adapter routing: select and weight a subset of available adapters for a single inference request.
- `mixture/`
  - `tests.rs`
- `model/`
  - `bert.rs`: BERT/BGE model loading and inference.
  - `bert/`
    - `drop_order_tests.rs`
    - `tests.rs`
  - `bitnet_config.rs`: BitNet configuration facade over `lattice-arch`.
  - `cross_encoder.rs`: Cross-encoder reranking model for BERT/MiniLM-style checkpoints.
  - `cross_encoder/`
    - `bert_namespace_tests.rs`
    - `tests.rs`
  - `embeddinggemma2.rs`: EmbeddingGemma 2 text encoder, CPU f32 path.
  - `embeddinggemma2/`
    - `tests.rs`: Tests for the EmbeddingGemma 2 text encoder.
    - `tests/`
      - `metal_parity.rs`: Whole-model parity of the Metal f32 path with the CPU path on synthetic checkpoints.
  - `embeddinggemma2_config.rs`: EmbeddingGemma 2 configuration facade over `lattice-arch`.
  - `embeddinggemma2_image.rs`: Validated RGB input and image patch geometry for EmbeddingGemma 2.
  - `embeddinggemma2_image/`
    - `reference_tests.rs`: Processor digests observed on x86_64 CPU with PyTorch 2.14.1 and Torchvision 0.29.1.
    - `resize.rs`
    - `tests.rs`
  - `embeddinggemma2_vision.rs`: Deferred EmbeddingGemma 2 vision checkpoint loading.
  - `embeddinggemma2_vision/`
    - `forward.rs`: EmbeddingGemma 2 vision tower forward, CPU f32 over the valid patches.
    - `tests.rs`
  - `ernie45.rs`: ERNIE-4.5 dense text decoder (PaddleOCR-VL family): CPU f32 reference forward.
  - `gemma4_cache.rs`: Gemma 4 shared-KV cache: donor-slot indirection (ADR-082 Amendment 1, stage 4).
  - `gemma4_config.rs`: Gemma 4 configuration facade over `lattice-arch`.
  - `gemma4_loading.rs`: Gemma 4 E2B safetensors -> `Gemma4Weights` loader (ADR-082 stage 5).
  - `gemma4_loading/`
    - `tests.rs`
  - `gemma4_model.rs`: Gemma 4 E2B text-only CPU forward + greedy generate (ADR-082 stage 5).
  - `gemma4_ops.rs`: Gemma 4 E2B text math kernels (ADR-082 stage 3, G6-G10).
  - `gemma4_ops/`
    - `tests.rs`
  - `gemma4_prefill.rs`: Bounded Gemma prompt chunks with scalar, per-query attention reductions.
  - `gemma4_prefill_tests.rs`
  - `gemma4_preflight.rs`: Gemma 4 E2B safetensors loader preflight (ADR-082 stage 2).
  - `gemma4_preflight/`
    - `tests.rs`
  - `gemma4_weights.rs`: Gemma 4 E2B weight storage (ADR-082 stage 5).
  - `mod.rs`: Model module index for BERT, BitNet config, cross-encoder, Qwen, Qwen3.5, and Qwen3.5 config, with BERT/CrossEncoder/Qwen/Qwen3.5 type re-exports.
  - `paddleocr_vl.rs`: PaddleOCR-VL-1.6 end-to-end CPU forward: image + prompt text -> greedy text.
  - `qwen.rs`: Qwen3-Embedding model: decoder-only transformer inference.
  - `qwen/`
    - `tests.rs`
  - `qwen35/`
    - `cache.rs`: Qwen3.5 KV cache, forward scratch, and scratch resize helper.
    - `debug.rs`: Qwen3.5 debug methods and debug statistics.
    - `embed.rs`: Pooled hidden-state embeddings from the loaded generative model.
    - `eval.rs`: Language-model evaluation primitives for Qwen3.5: per-token NLLs and strided sliding-window perplexity.
    - `eval/`
      - `tests.rs`
      - `tests/`
        - `readout_tests.rs`: Numerical and input-validation tests for requested prefill readouts.
    - `f16_loading.rs`: Qwen3.5 checkpoint loading into F16 weight containers.
    - `f16_loading/`
      - `tests.rs`
    - `forward.rs`: Qwen3.5 forward implementation, including `forward_step`, attention dispatch, full attention, and dense FFN paths.
    - `generation.rs`: Qwen3.5 generation, streaming generation, prefill/decode loops, stop-streamer utilities, and stop-token helpers.
    - `generation/`
      - `tests.rs`
      - `tests/`
        - `streaming.rs`: Streaming generation with cancellation and observers, and token parity of the public prefill delegation.
    - `generation_setup.rs`
    - `generation_setup/`
      - `tests.rs`
    - `loading.rs`: Qwen3.5 shape-checked owned tensor loading, required-tensor validation, layer-specific weight loaders, and `load_weights`.
    - `loading/`
      - `tests.rs`
    - `mod.rs`: Qwen3.5-2B text generation model.
    - `model.rs`: Qwen3.5 model type, safetensors loader, accessors, LoRA setter, train-backward weight accessors, and layer weight statistics.
    - `moe.rs`: Qwen3.5 MoE FFN step, router probabilities, top-k selection, selected-weight renormalization, routed expert accumulation, and shared expert application.
    - `moe/`
      - `tests.rs`
    - `norm.rs`: Shifted RMSNorm implementation used by the Qwen3.5 forward path.
    - `q8_conversion.rs`: Qwen3.5 model conversion to per-row symmetric INT8 weights.
    - `q8_conversion/`
      - `tests.rs`
    - `rerank.rs`: Query-likelihood reranking over the Qwen3.5 causal language-model scoring path.
    - `rerank/`
      - `tests.rs`
    - `sampling.rs`: Qwen3.5 token sampling, repetition penalty, greedy fallback, top-n-sigma filtering, softmax probability build, min-p/top-p filtering, distribution draw, and RNG helper.
    - `sampling/`
      - `tests.rs`
    - `test_support.rs`: Test-only tiny zero-weight `Qwen35Model` construction, gated behind `cfg(any(test, feature = "test-utils"))` (ADR-080 C2, #816).
    - `tests.rs`: Qwen3.5 unit tests for partial RoPE, sampling, repetition penalty, detokenization, scratch temporaries, required tensor names, stop tokens, and MoE router construction.
    - `weights.rs`: Qwen3.5 full-attention, common, dense, attention, model, and MoE weight structs plus `ModelWeights::logits_weight`.
  - `qwen35_config.rs`: Qwen3.5 configuration facade over `lattice-arch`.
- `model_format.rs`: Model-directory format detection, re-exported from `lattice-weights`.
- `model_format/`
  - `tests.rs`
- `moe_admission.rs`: Offline MoE expert-cache admission-policy simulator (issue #682, Stage 3: "simulate admission policies against routing traces before touching the engine's cache eviction logic").
- `moe_admission/`
  - `tests.rs`
- `option_scoring.rs`: Option-letter scoring, re-exported from `lattice-compute`.
- `pool.rs`: BERT-family pooling, re-exported from `lattice-compute`.
- `pruning.rs`: ShortGPT-style block influence scoring, re-exported from `lattice-compute`.
- `quant/`
  - `mod.rs`: Quantization, re-exported from `lattice-quant`.
  - `q4_manifest.rs`: Bounded reader for `quantize_index.json`, the per-tensor manifest written next to a Q4 checkpoint's `.q4`/`.f16` files, re-exported from `lattice-quant`.
  - `quarot/`
    - `convert.rs`: QuaRot conversion of a Qwen3.5 checkpoint into a rotated Q4 checkpoint, and its promotion records, re-exported from `lattice-quant`.
    - `forward_equivalence.rs`: The f64 forward-equivalence check between an original and a QuaRot-rotated Qwen3.5 checkpoint, re-exported from `lattice-quant`.
    - `hadamard.rs`: Walsh-Hadamard transform and randomized Hadamard generators, re-exported from `lattice-quant`.
    - `io.rs`: Reading source tensors for QuaRot conversion and the online-rotation artifact descriptor, re-exported from `lattice-quant`.
    - `lm_head.rs`: LM-head materialization and untying for QuaRot conversion, re-exported from `lattice-quant`.
    - `lora.rs`: Rotation of LoRA adapters into a QuaRot-rotated basis, re-exported from `lattice-quant`.
    - `mod.rs`: QuaRot: Hadamard-rotated 4-bit quantization (Ashkboos et al., NeurIPS 2024), re-exported from `lattice-quant`.
    - `pipeline.rs`: The QuaRot rotation pipeline over f64 tensors, re-exported from `lattice-quant`.
    - `plan.rs`: The QuaRot rotation plan: which rotation each tensor receives and on which side, re-exported from `lattice-quant`.
    - `rmsnorm_fusion.rs`: Fusion of RMSNorm scales into the following linear layers for QuaRot, re-exported from `lattice-quant`.
    - `rotation.rs`: Absorption of a `RandomizedHadamard` rotation into a linear-layer weight matrix, re-exported from `lattice-quant`.
- `qwen35_checkpoint_fixture.rs`: Deterministic, synthetic Qwen3.5 checkpoint for serving startup tests.
- `rope.rs`: Rotary position embedding tables, re-exported from `lattice-compute`.
- `router_admission.rs`: Registered, held-out router admission evidence (ADR-096 decision 5).
- `router_admission/`
  - `reference.rs`: Model-derived reference evidence for admission; no caller-supplied NLL values.
  - `statistics.rs`: Deterministic numerical evidence for ADR-096 D5; this module grants no authority.
  - `statistics/`
    - `tests.rs`
  - `tests.rs`
- `router_episode.rs`: What a routed completion recorded about itself, and the judgements against it (ADR-096 decision 3).
- `router_episode/`
  - `tests.rs`
- `router_init.rs`: Provisioning a router state directory (ADR-096 decision 2).
- `router_init/`
  - `tests.rs`
- `router_refit.rs`: Bounded construction of router candidates from explicit feedback.
- `router_refit/`
  - `tests.rs`
- `router_state.rs`: The on-disk router gate artifact: what a learned gate is when it is not in memory (ADR-095 decision 3, ADR-094 decision 2 as amended 2026-09-21).
- `router_state/`
  - `storage.rs`: Cooperating readers and publishers share a persistent lock inode.
  - `storage/`
    - `sync_probe.rs`
    - `tests.rs`
  - `tests.rs`
- `sampling.rs`: Sampling facade over `lattice-arch`.
- `serving_support.rs`: Unsupported, version-coupled interface for `lattice-serve` (ADR-103 Amendment 4).
- `signed_scale_fixture.rs`
- `signed_scale_fixture/`
  - `fixture.rs`
- `speculative.rs`: N-gram prompt lookup speculative decoding.
- `speculative/`
  - `tests.rs`
  - `tests/`
    - `mtp_rejection.rs`: MTP rejection-sampling integration and the MtpVerifier f16 KV cache rollback regression.
- `stop_reason.rs`: Stop-reason facade over `lattice-arch`.
- `stop_token_contract.rs`: Cross-path stop-token contract sweep (#613).
- `stop_token_contract/`
  - `tests.rs`
- `test_support.rs`: Test support for the checkpoint-gated unit tests in `crate::model`.
- `test_support/`
  - `tests.rs`
- `tokenizer/`
  - `bpe/`
    - `tests.rs`
  - `mod.rs`: Tokenizer facade over `lattice-tokenizer`.
- `vision/`
  - `checkpoint.rs`: Loader for the real Qwen3.5 vision-language checkpoint's 153 `model.visual.*` tensors (ADR-069 stage S2).
  - `checkpoint/`
    - `tests.rs`
  - `config.rs`: Vision encoder configuration for Qwen3-VL.
  - `merger.rs`: MLP merger: spatial pooling + projection from ViT dim → decoder dim.
  - `merger/`
    - `tests.rs`
  - `mod.rs`: Vision encoder module.
  - `multimodal.rs`: Multimodal input type and the `generate_multimodal` integration point.
  - `multimodal/`
    - `qwen35_vision_request_tests.rs`
    - `tests.rs`
  - `paddleocr_preprocess.rs`: Image preprocessing for PaddleOCR-VL-1.6 vision input.
  - `paddleocr_preprocess/`
    - `tests.rs`
  - `paddleocr_vit.rs`: PaddleOCR-VL-1.6 vision encoder + projector, CPU reference forward.
  - `paddleocr_vit/`
    - `tests.rs`
  - `pooled_embed.rs`: Raw image bytes -> pooled, L2-normalized embedding: wires the existing real Qwen3.5-0.8B vision pipeline (`qwen35_vit::{preprocess_qwen35_image, qwen35_vit_forward}`, `qwen35_merger::qwen35_merger_forward`) into the decoder-side pooling in `crate::forward::cpu_f16` (`prefill_hidden_states_f16`, `embed_image_f16`, `PoolingStrategy`).
  - `pooled_embed/`
    - `tests.rs`
  - `preprocess.rs`: CPU-side image preprocessing for the Qwen3-VL vision encoder.
  - `preprocess/`
    - `tests.rs`
  - `qwen35_merger.rs`: Real Qwen3.5-0.8B patch-merger / projection stage (ADR-069 S4): maps the S3a pre-merger hidden states `\[num_patches, hidden_size\]` to the post-merger projected visual embeddings `\[num_patches / spatial_merge_size^2, out_hidden_size\]` — the exact tensor injected into the decoder's residual stream at image-token positions.
  - `qwen35_merger/`
    - `tests.rs`
  - `qwen35_mrope.rs`: Pure-Rust M-RoPE position-id and cos/sin table builder for Qwen3.5 vision-language decoding (ADR-069 Stage 5a).
  - `qwen35_mrope/`
    - `tests.rs`
  - `qwen35_vit.rs`: Real Qwen3.5-0.8B ViT forward pass (ADR-069 S3a — CPU reference; the Metal port is a separate S3b fast-follow gated against this module): depth-12 / hidden-768 encoder over the checkpoint's `model.visual.*` weights (`super::checkpoint::Qwen35VisionWeights`), producing the pre-merger hidden states `\[num_patches, hidden_size\]`.
  - `qwen35_vit/`
    - `tests.rs`
  - `qwen35_vit_metal.rs`: Metal port of the Qwen3.5-0.8B ViT forward pass (ADR-069 S3b), gated against the S3a CPU reference (`super::qwen35_vit::qwen35_vit_forward`).
  - `qwen35_vit_metal/`
    - `tests.rs`
  - `tests.rs`
  - `vit.rs`: Vision Transformer (ViT) encoder for Qwen3-VL.
  - `vit/`
    - `tests.rs`
- `weights.rs`: Weight storage facade over `lattice-weights`.

</details>

Targets declared by the manifest, including the ones Cargo discovers by directory:

- lib (1): `lattice_inference`
- bin (16): `backfill_qwen3` (requires `backfill`), `bench_decode_ab` (requires `f16`, `metal-gpu`), `bench_decode_slopefit`, `bench_gdn_prefill_ab` (requires `f16`, `metal-gpu`, `bench-internals`), `bench_gemma4_driver_pair` (requires `f16`), `bench_logit_dump` (requires `f16`, `metal-gpu`), `bench_lora_mixture` (requires `f16`, `metal-gpu`), `bench_vision_prefill_ab` (requires `f16`, `metal-gpu`, `bench-internals`), `chat_metal` (requires `f16`, `metal-gpu`), `dump_quarot_q4_golden` (requires `f16`, `metal-gpu`), `eval_perplexity`, `gramperf_profile`, `gramtime_profile`, `ppl_metal` (requires `f16`, `metal-gpu`), `qwen35_debug`, `qwen35_generate`
- example (35): `bench_concurrent` (requires `f16`, `metal-gpu`), `bench_decode_streaming` (requires `f16`, `metal-gpu`), `bench_dispatch2`, `bench_dispatch`, `bench_embed_quality`, `bench_embedding` (requires `f16`), `bench_gdn_decode` (requires `f16`, `metal-gpu`), `bench_gdn_prefill_ab`, `bench_gdn_state` (requires `f16`, `metal-gpu`, `gdn-state-counters`), `bench_gdn`, `bench_gemm_shapes` (requires `f16`, `metal-gpu`), `bench_gpu` (requires `f16`, `wgpu-gpu`), `bench_metal_forward_allocation` (requires `metal-gpu`, `f16`), `bench_metal` (requires `f16`, `metal-gpu`), `bench_mps_gemm`, `bench_persistent_state` (requires `f16`, `metal-gpu`), `bench_profile` (requires `f16`), `bench_pruning`, `bench_q4_prefill` (requires `f16`, `metal-gpu`), `bench_q8_prefill` (requires `f16`, `metal-gpu`), `bench_quality`, `bench_simdgroup`, `bench_stability` (requires `f16`, `metal-gpu`), `bench_suite` (requires `f16`), `decode_profile` (requires `f16`, `metal-gpu`), `diff_attn_layer23` (requires `train-backward`, `f16`), `diff_gdn_layer` (requires `train-backward`, `f16`), `dump_embeddinggemma2_vision`, `dump_embeddinggemma2`, `ernie45_trace_dump`, `layer_sweep` (requires `f16`), `load_cross_encoder`, `profile_metal_decode` (requires `f16`, `metal-gpu`), `profile_metal` (requires `f16`, `metal-gpu`), `rss_autorelease_probe`
- test (48): `audit_tokenizer_parity`, `bert_cpu_kernel_policy_api`, `bert_cross_encoder_pooler`, `bert_from_bytes_max_tokens`, `bert_lora_bounds_regression`, `cpu_pre_migration_greedy_golden`, `decoder_allocation_arm`, `dependency_direction_contract`, `differential_attention_test`, `env_var_contract`, `examples_no_preset_fallback`, `facade_api_parity`, `forecast_chronos2_goldens`, `forecast_reverso_goldens`, `forecast_toto2_goldens`, `gated_attention_test`, `gemma4_admission_test`, `gemma4_e2e_forward_test`, `gemma4_tensor_manifest_test`, `gemma4_tokenizer_goldens_test`, `grammar_json_schema_regression`, `grammar_shared_first_byte_alternatives`, `grammar_tbv_probe`, `inference_perf_allocation_gate` (requires `bench-internals`), `legacy_generate_removed`, `metal_generation_goldens`, `metal_measurement_lock_contract`, `metal_qwen35_population_contract`, `metal_qwen35_source_contract`, `metal_qwen35_source_population_controls`, `metal_runtime_buffers`, `mmap_trust_boundary_contract`, `native_sparse_attention_test`, `paddleocr_vl_adhoc_ocr`, `paddleocr_vl_decoder_goldens_test`, `paddleocr_vl_e2e_goldens_test`, `paddleocr_vl_preprocess_goldens_test`, `paddleocr_vl_tokenizer_goldens_test`, `paddleocr_vl_vision_goldens_test`, `pipeline_boundary_contract`, `quarot_q4_composed_golden`, `vision_goldens_fixture_test`, `vision_s3_vit_forward_test`, `vision_s3b_vit_metal_gate_test`, `vision_s4_merger_test`, `vision_s5b_e2e_gate_test`, `wgsl_shader_validation`, `x86_gemm_allocations`
- bench (25): `attention_dispatch_bench`, `attn_opt_bench`, `backward_stage0` (requires `train-backward`), `cross_turn_prefix_cache_bench` (requires `metal-gpu`, `f16`), `decode_attn_bench` (requires `f16`), `differential_attention_bench`, `e2e_bench`, `elementwise_bench`, `elementwise_cpu_bench`, `f16_convert_bench` (requires `bench-internals`), `gated_attention_bench`, `grammar_mask_bench`, `inference_bench`, `inference_perf`, `kv_cache_f16_bench`, `kv_cache_layout_bench` (requires `bench-internals`), `lm_head_bench`, `metal_decode_bench` (requires `metal-gpu`, `f16`), `metrics_bench`, `mtp_decode` (requires `metal-gpu`, `f16`), `native_sparse_attention_bench`, `pruning_bench`, `quarot_hadamard_bench`, `tokenizer_bench`, `topk_readback`

<!-- generated:layout:end -->

<!-- generated:stats:begin -->

### Statistics

Rust lines by kind (code, comment, blank; doc comments count as comment), test functions (`#[test]` and `#[<runtime>::test]` attributes), items declared `pub` (not `pub(crate)`; reachability through private modules is not resolved) and occurrences of the `unsafe` keyword outside comments and strings.

| Area        | Files | Code   | Comment | Blank | Tests | `pub` items | `unsafe` |
| ----------- | ----- | ------ | ------- | ----- | ----- | ----------- | -------- |
| `src/`      | 387   | 161987 | 31435   | 14178 | 2407  | 1281        | 235      |
| `tests/`    | 92    | 30282  | 3472    | 2218  | 498   | 86          | 17       |
| `benches/`  | 27    | 10323  | 1353    | 1082  | 0     | 1           | 28       |
| `examples/` | 35    | 8381   | 1041    | 958   | 14    | 0           | 26       |
| total       | 541   | 210973 | 37301   | 18436 | 2919  | 1368        | 306      |

Other tracked files (the README excluded):

| Extension | Files | Size      |
| --------- | ----- | --------- |
| `.bin`    | 29    | 133.1 KiB |
| `.f32`    | 6     | 72.0 KiB  |
| `.jinja`  | 2     | 18.4 KiB  |
| `.json`   | 48    | 81.7 MiB  |
| `.md`     | 4     | 77.7 KiB  |
| `.metal`  | 14    | 191.4 KiB |
| `.png`    | 1     | 21.3 KiB  |
| `.py`     | 2     | 19.9 KiB  |
| `.toml`   | 2     | 14.8 KiB  |
| `.txt`    | 2     | 451.8 KiB |
| `.wgsl`   | 11    | 17.1 KiB  |

<!-- generated:stats:end -->

<!-- generated:docs:begin -->

### In-crate documentation

- [Changelog](CHANGELOG.md) (`CHANGELOG.md`)
- [lattice-inference — CONFIG.md](CONFIG.md) (`CONFIG.md`)
- [Design: lattice-inference](DESIGN.md) (`DESIGN.md`)
- [lattice-inference — METAL_TRACE.md](METAL_TRACE.md) (`METAL_TRACE.md`)

<!-- generated:docs:end -->

<!-- generated:features:begin -->

### Features

Default set: `download`, `image`, `std`. Each description is the first sentence of the comment above the feature in `Cargo.toml`.

- `backfill`: Enables `dep:rusqlite`, `f16`.
- `bench-internals`: Enables `lattice-compute/bench-internals`, `lattice-grammar/bench-internals`.
- `bench-ort`: Enables `dep:ort`, `dep:tokenizers`, `f16`.
- `download` (default): Enables `lattice-model-io/download`.
- `f16`: Enables `lattice-weights/f16`.
- `gdn-state-counters`: GDN state-traffic byte counters (issue #422, measure-first for ADR-058). Enables nothing.
- `image` (default): Encoded PNG/JPEG input; raw image tensors remain available without decoding. Enables `dep:image`.
- `metal-gpu`: lattice-fann supplies the shared LoRA descriptor/blend-validation module (issue #615) that `blend_lora_layer_data` and `chat_metal` call, so it must be present whenever this feature is — not just under `mixture`. Enables `dep:metal`, `dep:objc`, `dep:lattice-fann`, `dep:bytemuck`, `dep:lattice-metal-runtime`, `lattice-metal-runtime/metal-gpu`.
- `mixture`: AdapterRouter gate: enables the lattice-fann backed routing module. generate_with_lora_mixture does NOT require this feature — it is always available under `metal-gpu`. blend_lora_layer_data now depends on lattice-fann too (issue #615), but that dep is already pulled in by `metal-gpu` itself, so this feature only adds the routing module. Enables `dep:lattice-fann`.
- `router-learning`: Enables the bounded router trainer for qualification; serving remains opt-in. Enables `mixture`, `lattice-fann/online-router`.
- `serving-support`: Unsupported, version-coupled interface for `lattice-serve` (ADR-103 Amendment 4): the hidden `serving_support` module and the router items it needs. Enables nothing.
- `signpost`: os_signpost instrumentation of the Metal decode path (W1.0), for capturing a Metal System Trace of command-buffer/dispatch/wait structure in Instruments. Enables nothing.
- `std` (default): Enables nothing.
- `test-utils`: Exposes test-only fixture constructors (currently: a tiny all-zero-weight Qwen35Model, ADR-080 C2) as `pub` library API, gated so they never ship in a normal build. Enables nothing.
- `train-backward`: Enables nothing.
- `wgpu-gpu`: Enables `dep:wgpu`, `dep:pollster`, `dep:bytemuck`.
- `x86-gemm`: Stock `gemm` for multi-row `matmul_bt` on Linux x86_64 (ADR-099 F1, step 1). Enables `lattice-compute/x86-gemm`.

<!-- generated:features:end -->

<!-- generated:dependencies:begin -->

### Workspace dependencies

- Depends on: `lattice-arch`, `lattice-compute`, `lattice-fann` (optional, enabled by `metal-gpu`, `mixture`, `router-learning`), `lattice-grammar`, `lattice-inference-core`, `lattice-metal-runtime` (optional, enabled by `metal-gpu`), `lattice-model-io` (no default features), `lattice-quant`, `lattice-tokenizer`, `lattice-weights`
- Dev-depends on: `lattice-model-io` (no default features; features `test-utils`), `lattice-weights` (features `test-utils`)
- Used by: `lattice-embed` (optional), `lattice-serve`, `lattice-tune` (optional)
- Used in tests by: `lattice-embed`, `lattice-serve`

<!-- generated:dependencies:end -->

<!-- generated:adrs:begin -->

### Architecture decisions

ADRs under `docs/adr/` that name `lattice-inference` or `crates/inference/`:

<details>
<summary>79 ADRs</summary>

- [ADR-001: Pure Rust Transformer Engine](../../docs/adr/ADR-001-pure-rust-transformer-engine.md)
- [ADR-002: SIMD Dispatch Strategy](../../docs/adr/ADR-002-simd-dispatch.md)
- [ADR-003: SafeTensors Weight Loading](../../docs/adr/ADR-003-safetensors-loading.md)
- [ADR-004: KV Cache Design](../../docs/adr/ADR-004-kv-cache.md)
- [ADR-005: Tokenizer Architecture](../../docs/adr/ADR-005-tokenizer-architecture.md)
- [ADR-006: Speculative Decoding](../../docs/adr/ADR-006-speculative-decoding.md)
- [ADR-007: Rotary Positional Encoding (RoPE)](../../docs/adr/ADR-007-rope-positional-encoding.md)
- [ADR-008: LoRA Injection via Trait Hook](../../docs/adr/ADR-008-lora-injection.md)
- [ADR-009: Model Architecture Support (BERT and Qwen3)](../../docs/adr/ADR-009-model-architectures.md)
- [ADR-010: Attention Mechanisms](../../docs/adr/ADR-010-attention-mechanisms.md)
- [ADR-011: Sampling Strategies](../../docs/adr/ADR-011-sampling-strategies.md)
- [ADR-013: lattice-embed as SIMD Foundation Layer](../../docs/adr/ADR-013-simd-foundation.md)
- [ADR-017: NativeEmbeddingService Orchestration](../../docs/adr/ADR-017-native-embedding-service.md)
- [ADR-031: LoRA Adapter Management](../../docs/adr/ADR-031-lora-adapter-management.md)
- [ADR-040: Gated Attention (G1 SDPA-Output Gating)](../../docs/adr/ADR-040-gated-attention.md)
- [ADR-041: Differential Attention](../../docs/adr/ADR-041-differential-attention.md)
- [ADR-042: Native Sparse Attention](../../docs/adr/ADR-042-native-sparse-attention.md)
- [ADR-043: LoRA Serving Verification and Qwen3.5-0.8B Support](../../docs/adr/ADR-043-lora-serving-verification.md)
- [ADR-044: QuaRot — Hadamard-Rotated 4-bit Quantization](../../docs/adr/ADR-044-quarot-rotated-quantization.md)
- [ADR-045: QuaRot + LoRA Composition at Inference Time](../../docs/adr/ADR-045-quarot-lora-composition.md)
- [ADR-046: XGrammar Structured Output Engine](../../docs/adr/ADR-046-structured-output.md)
- [ADR-047: Paged KV Cache with Prefix Reuse](../../docs/adr/ADR-047-paged-kv-cache.md)
- [ADR-048: Continuous Batching with Disaggregated Prefill/Decode](../../docs/adr/ADR-048-continuous-batching.md)
- [ADR-049: Vision Encoder Integration (Qwen-VL Path)](../../docs/adr/ADR-049-vision-encoder.md)
- [ADR-050: Rejection Sampling for Speculative Decoding](../../docs/adr/ADR-050-rejection-sampling.md)
- [ADR-051: QuaRot-MTP Rotation Reconciliation](../../docs/adr/ADR-051-quarot-mtp-rotation.md)
- [ADR-052: GDN State Management for Speculative Rollback](../../docs/adr/ADR-052-gdn-speculative-state.md)
- [ADR-053: MoE Metal Dispatch with Expert Coalescing](../../docs/adr/ADR-053-moe-metal-dispatch.md)
- [ADR-054: Rotation-Aware LoRA Training (RoLoRA Integration)](../../docs/adr/ADR-054-rolora-rotation-aware-lora.md)
- [ADR-055: Online Distribution Drift Detection via Sinkhorn Divergence](../../docs/adr/ADR-055-online-drift-detection.md)
- [ADR-056: LoRA Tuning Pipeline](../../docs/adr/ADR-056-lora-tuning-pipeline.md)
- [ADR-057: LoRA Full-Lifecycle Consumer API](../../docs/adr/ADR-057-lora-consumer-api.md)
- [ADR-058: CPU Performance Regression Tracking in CI](../../docs/adr/ADR-058-cpu-perf-regression-ci.md)
- [ADR-059: Composable Layer Architecture](../../docs/adr/ADR-059-composable-layer-architecture.md)
- [ADR-060: Pruning Toolbox — Calibration Infrastructure, Structured Pruning, and Analysis Scorers](../../docs/adr/ADR-060-pruning-toolbox.md)
- [ADR-061: Inference Metrics & Experiment Runner Infrastructure](../../docs/adr/ADR-061-inference-metrics-infrastructure.md)
- [ADR-062: Metal FlashAttention-2 Prefill + KV Cache Quantization Chain](../../docs/adr/ADR-062-metal-fa2-prefill.md)
- [ADR-063: Serving Architecture --- CLI, HTTP Server, API Compatibility](../../docs/adr/ADR-063-serving-architecture.md)
- [ADR-064: CI Gate Taxonomy and Promotion Policy](../../docs/adr/ADR-064-ci-gate-taxonomy.md)
- [ADR-065: Feature Promotion Gates — Merge Measured Primitives, Not Research Ideas](../../docs/adr/ADR-065-feature-promotion-gates.md)
- [ADR-066: Output-Correctness Gate Architecture](../../docs/adr/ADR-066-output-correctness-gate-architecture.md)
- [ADR-068: Pluggable Custom-Grammar Wire Contract for the OpenAI-Compatible Serving Surface](../../docs/adr/ADR-068-grammar-wire-contract.md)
- [ADR-069: Vision Encoder Recalibration for Qwen3.5-0.8B](../../docs/adr/ADR-069-vision-encoder-qwen35-recalibration.md)
- [ADR-071: GDN Recurrent State as a First-Class Serveable Object — Priority and Scope](../../docs/adr/ADR-071-gdn-state-object-priority.md)
- [ADR-072: Weight Quantization Priority (Metal)](../../docs/adr/ADR-072-weight-quantization-priority.md)
- [ADR-073: KV-Cache Quantization Priority (Metal)](../../docs/adr/ADR-073-kv-cache-quantization-priority.md)
- [ADR-074: MTP / Speculative-Decoding Priority (Metal)](../../docs/adr/ADR-074-mtp-speculative-decoding-priority.md)
- [ADR-075: MoE Expert-Offload Priority (Metal)](../../docs/adr/ADR-075-moe-expert-offload-priority.md)
- [ADR-076: Adaptive Reasoning-Budget Priority (Metal + CPU)](../../docs/adr/ADR-076-adaptive-reasoning-priority.md)
- [ADR-077: Pruning + Distillation Priority (Metal + CPU)](../../docs/adr/ADR-077-pruning-distill-priority.md)
- [ADR-078: Multimodal / Vision Serving Priority (Deferral)](../../docs/adr/ADR-078-multimodal-serving-deferral-priority.md)
- [ADR-079: Adaptive Micro-LoRA Brain — the Composed Train → Govern → Compose → Route → Consume Loop](../../docs/adr/ADR-079-adaptive-micro-lora-brain.md)
- [ADR-080: Consolidation of Duplicated Numeric-Contract Helpers (Softmax, HTTP Serving, Decode Policy, GEMM Validation)](../../docs/adr/ADR-080-consolidation-duplicated-contracts.md)
- [ADR-081: Prefill GEMM Optimization Priority (Metal)](../../docs/adr/ADR-081-prefill-gemm-optimization-priority.md)
- [ADR-082: Gemma 4 E2B Support — Staged Multimodal Ladder (Vision + Audio)](../../docs/adr/ADR-082-gemma4-e2b-support.md)
- [ADR-083: Backward SIMD Dispatch Contract](../../docs/adr/ADR-083-backward-simd-dispatch.md)
- [ADR-084: Fallible Validation for Unstable Inference Helpers](../../docs/adr/ADR-084-fallible-inference-validation-helpers.md)
- [ADR-086: Retire the Legacy Generic Qwen Decode API](../../docs/adr/ADR-086-retire-legacy-qwen-decode-api.md)
- [ADR-087 Amendment 1 — open pull request population snapshot](../../docs/adr/ADR-087-amendment-1-population-snapshot.md)
- [ADR-087: bench-compare gate calibration, coverage, and admissible structural proof](../../docs/adr/ADR-087-bench-compare-gate-calibration-and-coverage.md)
- [ADR-088: Sealed Native Embedding Preparation and Attestation](../../docs/adr/ADR-088-sealed-native-embedding-attestation.md)
- [ADR-089: PaddleOCR-VL in lattice: CPU reference forward per slice, each gated by HF goldens, before any accelerated path](../../docs/adr/ADR-089-paddleocr-vl-cpu-reference-gated-by-goldens.md)
- [ADR-090: Shared decoder generation with model-owned execution sessions](../../docs/adr/ADR-090-shared-decoder-generation.md)
- [ADR-091: Learned mixture weights over a resident LoRA pool](../../docs/adr/ADR-091-learned-mixture-weights.md)
- [ADR-092: Model-owned decoder execution and a neutral serving boundary](../../docs/adr/ADR-092-model-owned-decoder-execution.md)
- [ADR-093: Mixture routing is decided per request, and a weight change does not rebuild the adapter](../../docs/adr/ADR-093-per-request-mixture-routing.md)
- [ADR-094: Where the routing code lives, and what feeds it](../../docs/adr/ADR-094-where-the-routing-code-lives.md)
- [ADR-095: The learning surface — feedback endpoint, router state, and what the API reports](../../docs/adr/ADR-095-the-learning-surface.md)
- [ADR-096: Producing a router artifact — schema lineage, bootstrap, the learned action, and admission](../../docs/adr/ADR-096-producing-a-router-artifact.md)
- [ADR-097: Option-letter scoring: one prefill, logits read at the option letters](../../docs/adr/ADR-097-option-letter-scoring.md)
- [ADR-098: A multi-slot decision read-out and a `/v1/systemone` endpoint](../../docs/adr/ADR-098-decision-readout.md)
- [ADR-099: Linux CPU Embedding GEMM, Parallel Execution, and Resource Budgets](../../docs/adr/ADR-099-linux-cpu-embedding-gemm-threading-budget.md)
- [ADR-100: Loading foreign quantized checkpoints without a foreign runtime](../../docs/adr/ADR-100-foreign-quantized-checkpoints.md)
- [ADR-101: Time-series forecasting models in lattice-inference](../../docs/adr/ADR-101-time-series-forecasting.md)
- [ADR-102: Reorganising Metal execution around shared mechanisms and typed model sessions](../../docs/adr/ADR-102-metal-execution-reorganisation.md)
- [ADR-103: Splitting lattice-inference into focused crates behind a facade](../../docs/adr/ADR-103-inference-crate-decomposition.md)
- [ADR-ALIGNMENT-AUDIT: ADR Set Classification as of 2026-06-30](../../docs/adr/ADR-ALIGNMENT-AUDIT.md)
- [Architecture Decision Records](../../docs/adr/INDEX.md)
- [R00 rollout and measurement ledger](../../docs/adr/R00-ROLLOUT.md)

</details>

<!-- generated:adrs:end -->

<!-- generated:benchmarks:begin -->

### Benchmarks

The newest Criterion point estimates `bench-update.yml` stored on the `perf-baselines` branch, from the `elementwise_cpu_bench` target. They come from GitHub-hosted runners and serve as trend data between commits, not as a performance claim; a claim needs a paired A/B (`make bench-compare`).

- `aarch64-linux`: commit `9487ae701021`, recorded 2026-10-26T14:29:30+00:00
- `x86_64-linux`: commit `9487ae701021`, recorded 2026-10-26T14:30:15+00:00

<details>
<summary>17 benchmarks</summary>

| Benchmark                                          | `aarch64-linux` | `x86_64-linux` |
| -------------------------------------------------- | --------------- | -------------- |
| `add_bias_gelu/4096`                               | 6.36 µs         | 2.57 µs        |
| `add_bias_gelu/896`                                | 1.40 µs         | 551.3 ns       |
| `elementwise_mul/4096`                             | 377.1 ns        | 266.6 ns       |
| `gelu/4096`                                        | 6.12 µs         | 2.36 µs        |
| `gelu/896`                                         | 1.34 µs         | 504.0 ns       |
| `layer_norm/4096`                                  | 1.27 µs         | 694.1 ns       |
| `layer_norm/896`                                   | 288.4 ns        | 156.7 ns       |
| `residual_add_layer_norm/fused/hidden384_seq128`   | 21.74 µs        | 11.84 µs       |
| `residual_add_layer_norm/fused/hidden768_seq128`   | 43.50 µs        | 21.05 µs       |
| `residual_add_layer_norm/unfused/hidden384_seq128` | 62.06 µs        | 57.50 µs       |
| `residual_add_layer_norm/unfused/hidden768_seq128` | 123.20 µs       | 105.29 µs      |
| `rms_norm/4096`                                    | 573.7 ns        | 755.1 ns       |
| `rms_norm/896`                                     | 136.3 ns        | 207.4 ns       |
| `silu_inplace/4096`                                | 4.91 µs         | 11.10 µs       |
| `silu_inplace/896`                                 | 1.08 µs         | 2.48 µs        |
| `softmax_attention/128`                            | 20.80 µs        | 9.73 µs        |
| `softmax_attention/512`                            | 326.58 µs       | 144.25 µs      |

</details>

<!-- generated:benchmarks:end -->
