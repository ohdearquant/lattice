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
<summary><code>src/</code>: 127 files</summary>

- `attention/`
  - `decode.rs`: Single-token decode attention, re-exported from `lattice-compute`.
  - `differential.rs`: Differential attention, re-exported from `lattice-compute`.
  - `flash.rs`: Tiled CPU attention over `lattice-compute`'s projection-view kernel.
  - `flash_causal.rs`: Tiled causal flash attention, re-exported from `lattice-compute`.
  - `gated.rs`: Sigmoid output gating for gated attention, re-exported from `lattice-compute`.
  - `gdn.rs`: GatedDeltaNet linear attention, re-exported from `lattice-compute`.
  - `gdn_fused.rs`: Fused GatedDeltaNet decode step, re-exported from `lattice-compute`.
  - `gqa.rs`: Grouped-query attention, re-exported from `lattice-compute`.
  - `mod.rs`: Attention module index for `decode`, `differential`, `flash`, `flash_causal`, `gated`, `gdn`, `gdn_backward`, `gdn_fused`, `gqa`, `native_sparse`, and `standard`; re-exports the standard attention helpers.
  - `native_sparse.rs`: Native sparse attention, re-exported from `lattice-compute`.
  - `softmax_row.rs`: Fail-closed softmax row contract, re-exported from `lattice-compute`.
  - `standard.rs`: Standard multi-head attention over `lattice-compute`'s projection-view kernels.
  - `standard/`
    - `tests.rs`: The facade's BERT-layer attention entry points against `lattice-compute`'s projection-view kernels.
- `backward.rs`: Backward-pass module index for `attention_gqa`, `gradcheck`, `ops`, `dispatch` (`test-utils` builds only), and `tape`.
- `batch.rs`: **Compatibility path.** The items here are re-exported from `lattice-compute` so that code written against the pre-split `lattice-inference` keeps compiling.
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
- `decoder.rs`: Shared driver observation and family measurement compatibility paths.
- `download.rs`: Model-file ensure/download flow, canonical model names, checksum verification, and download helper.
- `encoder.rs`: The text encoder contract of `lattice-arch` (ADR-103 Amendment 21 item 3) and its adapters for the embedding models this crate loads: BERT-family, Qwen3 and EmbeddingGemma 2.
- `error.rs`: Inference error taxonomy.
- `family.rs`: The family contract of `lattice-arch` (ADR-103 Amendment 21 item 3) and the descriptors for the model families this crate loads.
- `family/`
  - `tests.rs`
- `forecast.rs`: Time-series forecasting: shared context preparation and native forecasting models, re-exported from `lattice-forecast`.
- `forward/`
  - `batch_prefill.rs`: Qwen3.5 family compatibility exports.
  - `bitnet_kernel.rs`: **Compatibility path.** Re-exported from `lattice-compute` so that code written against the pre-split `lattice-inference` keeps compiling.
  - `cpu/`
    - `mod.rs`: CPU forward kernels, re-exported from `lattice-compute`.
  - `cpu_f16.rs`: Qwen3.5 family compatibility exports.
  - `cpu_q8.rs`: Qwen3.5 family compatibility exports.
  - `gdn_chunk_ref.rs`: Qwen3.5 family compatibility exports.
  - `gpu/`
    - `mod.rs`: **Compatibility path.** Re-exported from `lattice-model-qwen3-embedding` so that code written against the pre-split `lattice-inference` keeps compiling.
  - `gpu_gemm.rs`: **Compatibility path.** Re-exported from `lattice-compute` so that code written against the pre-split `lattice-inference` keeps compiling.
  - `metal.rs`: Qwen3 Metal backend, re-exported from its family crate.
  - `metal_embeddinggemma2.rs`: EmbeddingGemma 2 Metal backend compatibility exports.
  - `metal_ernie45.rs`: ERNIE4.5 Metal state and cache, owned by the ERNIE4.5 family.
  - `metal_gemm.rs`: Qwen3.5 family compatibility exports.
  - `metal_qwen35.rs`: Qwen3.5 family compatibility exports.
  - `metal_qwen35/`
    - `metal_command_buffer_pool_tests.rs`
    - `public_scheduling_entry_point_tests.rs`
  - `mod.rs`: Forward-kernel module index for batch prefill, BitNet, CPU, f16 CPU, q8 CPU, GPU, Metal, and NEON paths.
  - `moe_expert_cache.rs`: Qwen3.5 family compatibility exports.
  - `neon.rs`: Native Q8_0 NEON dotprod matvec kernel, re-exported from `lattice-model-qwen35`.
  - `neon_forward.rs`: Qwen3.5 family compatibility exports.
- `generation.rs`: Generation facade over `lattice-arch`.
- `grammar.rs`: Grammar facade over `lattice-grammar`.
- `kv_cache/`
  - `cross_turn.rs`: Cache definitions re-exported from `lattice-compute`.
  - `flat.rs`: Flat KV cache storage re-exported from `lattice-compute`.
  - `mod.rs`: KV cache for autoregressive transformer decoding.
  - `paged.rs`: Paged KV cache storage re-exported from `lattice-compute`.
  - `prefix.rs`: Cache definitions re-exported from `lattice-compute`.
- `lib.rs`: **Stability tier**: Experimental
- `lora_file.rs`: Qwen3.5 family compatibility exports.
- `lora_hook.rs`: LoRA adapter hooks and model-specific projection geometry.
- `metrics.rs`: Inference metrics and entropy accumulation, re-exported from `lattice-compute`.
- `model/`
  - `bert.rs`: BERT model and weight-loading paths retained by the inference facade.
  - `bitnet_config.rs`: BitNet configuration facade over `lattice-arch`.
  - `cross_encoder.rs`: BERT cross-encoder paths retained by the inference facade.
  - `embeddinggemma2.rs`: EmbeddingGemma 2 model compatibility exports.
  - `embeddinggemma2_config.rs`: EmbeddingGemma 2 configuration facade over `lattice-arch`.
  - `embeddinggemma2_image.rs`: EmbeddingGemma 2 RGB input compatibility export.
  - `ernie45.rs`: ERNIE4.5 CPU items, owned by the ERNIE4.5 family.
  - `gemma4_cache.rs`: Gemma4 CPU cache, re-exported from its family crate.
  - `gemma4_config.rs`: Gemma 4 configuration facade over `lattice-arch`.
  - `gemma4_model.rs`: Gemma4 CPU model, re-exported from its family crate.
  - `gemma4_ops.rs`: Gemma math primitives and named shared-kernel reexports.
  - `gemma4_preflight.rs`: Gemma4 checkpoint preflight, re-exported from its family crate.
  - `mod.rs`: Model module index for BERT, BitNet config, cross-encoder, Qwen, Qwen3.5, and Qwen3.5 config, with BERT/CrossEncoder/Qwen/Qwen3.5 type re-exports.
  - `paddleocr_vl.rs`: PaddleOCR-VL model and prompt preparation.
  - `qwen.rs`: Qwen3 embedding configuration and optional family implementation.
  - `qwen35/`
    - `mod.rs`: Qwen3.5 family compatibility exports.
  - `qwen35_config.rs`: Qwen3.5 configuration facade over `lattice-arch`.
- `model_format.rs`: Model-directory format detection, re-exported from `lattice-weights`.
- `model_format/`
  - `tests.rs`
- `moe_admission.rs`: Qwen3.5 family compatibility exports.
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
- `rope.rs`: Rotary position embedding tables, re-exported from `lattice-compute`.
- `sampling.rs`: Sampling facade over `lattice-arch`.
- `serving_support.rs`: Unsupported, version-coupled interface for `lattice-serve` (ADR-103 Amendment 4).
- `speculative.rs`: Qwen3.5 family compatibility exports.
- `stop_reason.rs`: Stop-reason facade over `lattice-arch`.
- `test_support.rs`: Test support for the checkpoint-gated unit tests in `crate::model`.
- `test_support/`
  - `tests.rs`
- `tokenizer/`
  - `bpe/`
    - `tests.rs`
  - `mod.rs`: Tokenizer facade over `lattice-tokenizer`.
- `vision/`
  - `checkpoint.rs`: Qwen3.5 vision checkpoint items, owned by the Qwen3.5 family.
  - `mod.rs`: Vision encoder module.
  - `multimodal.rs`: Multimodal input compatibility exports.
  - `paddleocr_preprocess.rs`: PaddleOCR-VL image preprocessing.
  - `paddleocr_vit.rs`: PaddleOCR-VL vision encoder and projector.
  - `pooled_embed.rs`: Qwen3.5 family compatibility exports.
  - `qwen35_merger.rs`: Qwen3.5 visual merger compatibility export.
  - `qwen35_mrope.rs`: Qwen3.5 family compatibility exports.
  - `qwen35_vit.rs`: Qwen3.5 CPU vision items and the shared patch-grid path.
  - `qwen35_vit/`
    - `tests.rs`
  - `qwen35_vit_metal.rs`: Qwen3.5 family compatibility exports.
- `weights.rs`: Weight storage facade over `lattice-weights`.

</details>

Targets declared by the manifest, including the ones Cargo discovers by directory:

- lib (1): `lattice_inference`
- bin (16): `backfill_qwen3` (requires `backfill`, `qwen3`), `bench_decode_ab` (requires `f16`, `metal-gpu`, `qwen35`), `bench_decode_slopefit` (requires `qwen35`), `bench_gdn_prefill_ab` (requires `f16`, `metal-gpu`, `bench-internals`, `qwen35`), `bench_gemma4_driver_pair` (requires `f16`, `gemma4`), `bench_logit_dump` (requires `f16`, `metal-gpu`, `qwen35`), `bench_lora_mixture` (requires `f16`, `metal-gpu`, `qwen35`), `bench_vision_prefill_ab` (requires `f16`, `metal-gpu`, `bench-internals`, `qwen35`), `chat_metal` (requires `f16`, `metal-gpu`, `qwen35`), `dump_quarot_q4_golden` (requires `f16`, `metal-gpu`, `qwen35`), `eval_perplexity` (requires `qwen35`), `gramperf_profile` (requires `qwen35`), `gramtime_profile`, `ppl_metal` (requires `f16`, `metal-gpu`, `qwen35`), `qwen35_debug` (requires `qwen35`), `qwen35_generate` (requires `qwen35`)
- example (35): `bench_concurrent` (requires `f16`, `metal-gpu`, `qwen35`), `bench_decode_streaming` (requires `f16`, `metal-gpu`, `qwen35`), `bench_dispatch2`, `bench_dispatch`, `bench_embed_quality` (requires `bert`, `qwen3`), `bench_embedding` (requires `f16`, `qwen3`), `bench_gdn_decode` (requires `f16`, `metal-gpu`, `qwen35`), `bench_gdn_prefill_ab` (requires `qwen35`), `bench_gdn_state` (requires `f16`, `metal-gpu`, `gdn-state-counters`, `qwen35`), `bench_gdn` (requires `qwen35`), `bench_gemm_shapes` (requires `f16`, `metal-gpu`, `embeddinggemma2`, `qwen35`), `bench_gpu` (requires `f16`, `wgpu-gpu`, `qwen3`), `bench_metal_forward_allocation` (requires `metal-gpu`, `f16`, `qwen35`), `bench_metal` (requires `f16`, `metal-gpu`, `qwen3`), `bench_mps_gemm`, `bench_persistent_state` (requires `f16`, `metal-gpu`, `qwen35`), `bench_profile` (requires `f16`, `qwen3`), `bench_pruning` (requires `qwen35`), `bench_q4_prefill` (requires `f16`, `metal-gpu`, `qwen35`), `bench_q8_prefill` (requires `f16`, `metal-gpu`, `qwen35`), `bench_quality` (requires `qwen35`), `bench_simdgroup`, `bench_stability` (requires `f16`, `metal-gpu`, `qwen35`), `bench_suite` (requires `f16`, `qwen3`, `qwen35`), `decode_profile` (requires `f16`, `metal-gpu`, `qwen35`), `diff_attn_layer23` (requires `train-backward`, `f16`, `qwen35`), `diff_gdn_layer` (requires `train-backward`, `f16`, `qwen35`), `dump_embeddinggemma2_vision` (requires `embeddinggemma2`), `dump_embeddinggemma2` (requires `embeddinggemma2`), `ernie45_trace_dump` (requires `ernie45`), `layer_sweep` (requires `f16`, `qwen3`), `load_cross_encoder` (requires `bert`), `profile_metal_decode` (requires `f16`, `metal-gpu`), `profile_metal` (requires `f16`, `metal-gpu`, `qwen35`), `rss_autorelease_probe` (requires `qwen3`, `qwen35`)
- test (46): `audit_tokenizer_parity`, `batch_compatibility_paths`, `bert_cpu_kernel_policy_api`, `bert_cross_encoder_pooler`, `bert_family_api`, `bert_from_bytes_max_tokens`, `bert_lora_bounds_regression`, `cpu_pre_migration_greedy_golden` (requires `qwen35`), `decoder_allocation_arm` (requires `qwen35`), `dependency_direction_contract`, `differential_attention_test`, `env_var_contract`, `examples_no_preset_fallback`, `facade_api_parity`, `forecast_chronos2_goldens`, `forecast_reverso_goldens`, `forecast_toto2_goldens`, `forward_kernel_compatibility_paths`, `gated_attention_test`, `gemma4_admission_test` (requires `gemma4`), `gemma4_e2e_forward_test` (requires `gemma4`), `gemma4_family_api`, `gemma4_preflight_regression` (requires `gemma4`), `gemma4_tensor_manifest_test`, `gemma4_tokenizer_goldens_test`, `inference_perf_allocation_gate` (requires `bench-internals`), `legacy_generate_removed` (requires `qwen35`), `metal_generation_goldens` (requires `qwen35`), `metal_measurement_lock_contract`, `metal_qwen35_population_contract`, `metal_qwen35_source_contract`, `metal_qwen35_source_population_controls`, `metal_runtime_buffers`, `mmap_trust_boundary_contract`, `native_sparse_attention_test`, `paddleocr_vl_family_api`, `pipeline_boundary_contract`, `quarot_q4_composed_golden` (requires `qwen35`), `qwen3_family_api`, `qwen3_wgpu_compatibility_paths`, `vision_goldens_fixture_test`, `vision_s3_vit_forward_test` (requires `qwen35`), `vision_s3b_vit_metal_gate_test` (requires `qwen35`), `vision_s4_merger_test` (requires `qwen35`), `vision_s5b_e2e_gate_test` (requires `qwen35`), `x86_gemm_allocations`
- bench (25): `attention_dispatch_bench`, `attn_opt_bench`, `backward_stage0` (requires `train-backward`, `test-utils`), `cross_turn_prefix_cache_bench` (requires `metal-gpu`, `f16`, `qwen35`), `decode_attn_bench` (requires `f16`), `differential_attention_bench`, `e2e_bench` (requires `bert`), `elementwise_bench`, `elementwise_cpu_bench`, `f16_convert_bench` (requires `bench-internals`), `gated_attention_bench`, `grammar_mask_bench`, `inference_bench` (requires `bert`), `inference_perf` (requires `qwen35`), `kv_cache_f16_bench`, `kv_cache_layout_bench` (requires `bench-internals`), `lm_head_bench` (requires `qwen35`), `metal_decode_bench` (requires `metal-gpu`, `f16`, `qwen35`), `metrics_bench`, `mtp_decode` (requires `metal-gpu`, `f16`, `qwen35`), `native_sparse_attention_bench`, `pruning_bench`, `quarot_hadamard_bench`, `tokenizer_bench`, `topk_readback`

<!-- generated:layout:end -->

<!-- generated:stats:begin -->

### Statistics

Rust lines by kind (code, comment, blank; doc comments count as comment), test functions (`#[test]` and `#[<runtime>::test]` attributes), items declared `pub` (not `pub(crate)`; reachability through private modules is not resolved) and occurrences of the `unsafe` keyword outside comments and strings.

| Area        | Files | Code  | Comment | Blank | Tests | `pub` items | `unsafe` |
| ----------- | ----- | ----- | ------- | ----- | ----- | ----------- | -------- |
| `src/`      | 127   | 7938  | 2605    | 920   | 82    | 436         | 1        |
| `tests/`    | 95    | 28943 | 3119    | 2099  | 482   | 74          | 18       |
| `benches/`  | 27    | 10765 | 1354    | 1093  | 0     | 1           | 28       |
| `examples/` | 35    | 8507  | 1047    | 972   | 14    | 0           | 26       |
| total       | 284   | 56153 | 8125    | 5084  | 578   | 511         | 73       |

Other tracked files (the README excluded):

| Extension | Files | Size      |
| --------- | ----- | --------- |
| `.bin`    | 29    | 133.1 KiB |
| `.f32`    | 6     | 72.0 KiB  |
| `.jinja`  | 2     | 18.4 KiB  |
| `.json`   | 48    | 81.7 MiB  |
| `.md`     | 4     | 78.3 KiB  |
| `.png`    | 1     | 21.3 KiB  |
| `.py`     | 2     | 19.9 KiB  |
| `.toml`   | 2     | 19.8 KiB  |
| `.txt`    | 2     | 452.6 KiB |

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

Default set: `bert`, `download`, `embeddinggemma2`, `ernie45`, `gemma4`, `image`, `paddleocr-vl`, `qwen3`, `qwen35`, `std`. Each description is the first sentence of the comment above the feature in `Cargo.toml`.

- `backfill`: Enables `dep:rusqlite`, `f16`.
- `bench-internals`: Enables `lattice-compute/bench-internals`, `lattice-grammar/bench-internals`, `lattice-model-qwen35?/bench-internals`.
- `bench-ort`: Enables `dep:ort`, `dep:tokenizers`, `f16`.
- `bert` (default): Enables `dep:lattice-model-bert`.
- `download` (default): Enables `lattice-model-io/download`, `lattice-model-bert?/download`, `lattice-model-qwen3-embedding?/download`.
- `embeddinggemma2` (default): Enables `dep:lattice-model-embeddinggemma2`.
- `ernie45` (default): Enables `dep:lattice-model-ernie45`.
- `f16`: Enables `lattice-weights/f16`, `lattice-router/f16`, `lattice-model-bert?/f16`, `lattice-model-gemma4?/f16`, `lattice-model-ernie45?/f16`, `lattice-model-qwen35?/f16`.
- `gdn-state-counters`: GDN state-traffic byte counters (issue #422, measure-first for ADR-058). Enables `lattice-model-qwen35?/gdn-state-counters`.
- `gemma4` (default): Enables `dep:lattice-model-gemma4`.
- `image` (default): Encoded PNG/JPEG input; raw image tensors remain available without decoding. Enables `lattice-model-io/image`, `lattice-model-embeddinggemma2?/image`, `lattice-model-ernie45?/image`, `lattice-model-qwen35?/image`.
- `metal-gpu`: lattice-fann supplies the shared LoRA descriptor/blend-validation module (issue #615) that `blend_lora_layer_data` and `chat_metal` call, so it must be present whenever this feature is — not just under `mixture`. Enables `dep:metal`, `dep:objc`, `dep:lattice-fann`, `dep:bytemuck`, `dep:lattice-metal-runtime`, `lattice-metal-runtime/metal-gpu`, `lattice-model-embeddinggemma2?/metal-gpu`, `lattice-model-qwen3-embedding?/metal-gpu`, `lattice-model-ernie45?/metal-gpu`, `lattice-model-qwen35?/metal-gpu`, `lattice-router/metal-gpu`.
- `mixture`: AdapterRouter gate: enables the lattice-fann backed routing module. generate_with_lora_mixture does NOT require this feature — it is always available under `metal-gpu`. blend_lora_layer_data now depends on lattice-fann too (issue #615), but that dep is already pulled in by `metal-gpu` itself, so this feature only adds the routing module. Enables `dep:lattice-fann`, `lattice-router/mixture`.
- `paddleocr-vl` (default): Enables `ernie45`, `lattice-model-ernie45?/paddleocr-vl`.
- `qwen3` (default): Enables `dep:lattice-model-qwen3-embedding`.
- `qwen35` (default): Enables `dep:lattice-model-qwen35`, `lattice-router/qwen35`.
- `router-learning`: Enables the bounded router trainer for qualification; serving remains opt-in. Enables `mixture`, `lattice-fann/online-router`, `lattice-model-qwen35?/router-learning`, `lattice-router/router-learning`.
- `serving-support`: Unsupported, version-coupled interface for `lattice-serve` (ADR-103 Amendment 4): the hidden `serving_support` module and the router items it needs. Enables `lattice-model-gemma4?/serving-support`, `lattice-model-qwen35?/serving-support`, `lattice-router/serving-support`.
- `signpost`: os_signpost instrumentation of the Metal decode path (W1.0), for capturing a Metal System Trace of command-buffer/dispatch/wait structure in Instruments. Enables `lattice-model-qwen35?/signpost`.
- `std` (default): Enables `lattice-model-bert?/std`.
- `test-utils`: Exposes test-only fixture constructors (currently: a tiny all-zero-weight Qwen35Model, ADR-080 C2) as `pub` library API, gated so they never ship in a normal build. Enables `lattice-backward?/test-utils`, `lattice-model-bert?/test-utils`, `lattice-model-gemma4?/test-utils`, `lattice-model-qwen35?/test-utils`.
- `train-backward`: Enables `dep:lattice-backward`, `lattice-model-qwen35?/train-backward`.
- `wgpu-gpu`: Enables `dep:bytemuck`, `lattice-compute/wgpu-gpu`, `dep:lattice-model-qwen3-embedding`, `lattice-model-qwen3-embedding/wgpu-gpu`.
- `x86-gemm`: Stock `gemm` for multi-row `matmul_bt` on Linux x86_64 (ADR-099 F1, step 1). Enables `lattice-compute/x86-gemm`, `lattice-model-bert?/x86-gemm`, `lattice-model-qwen3-embedding?/x86-gemm`, `lattice-model-gemma4?/x86-gemm`, `lattice-model-ernie45?/x86-gemm`.

<!-- generated:features:end -->

<!-- generated:dependencies:begin -->

### Workspace dependencies

- Depends on: `lattice-arch`, `lattice-backward` (optional, enabled by `train-backward`), `lattice-compute`, `lattice-fann` (optional, enabled by `metal-gpu`, `mixture`, `router-learning`), `lattice-forecast`, `lattice-grammar`, `lattice-inference-core`, `lattice-metal-runtime` (optional, enabled by `metal-gpu`), `lattice-model-io` (no default features), `lattice-quant`, `lattice-router`, `lattice-tokenizer`, `lattice-weights`
- Dev-depends on: `lattice-model-io` (no default features; features `test-utils`), `lattice-weights` (features `test-utils`)
- Used by: `lattice-serve`, `lattice-tune` (optional)
- Used in tests by: `lattice-embed`, `lattice-serve`

<!-- generated:dependencies:end -->

<!-- generated:adrs:begin -->

### Architecture decisions

ADRs under `docs/adr/` that name `lattice-inference` or `crates/inference/`:

<details>
<summary>80 ADRs</summary>

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
- [ADR-105: lattice-lora, One Home for Adapter Format and Execution Contracts](../../docs/adr/ADR-105-lattice-lora.md)
- [ADR-ALIGNMENT-AUDIT: ADR Set Classification as of 2026-06-30](../../docs/adr/ADR-ALIGNMENT-AUDIT.md)
- [Architecture Decision Records](../../docs/adr/INDEX.md)
- [R00 rollout and measurement ledger](../../docs/adr/R00-ROLLOUT.md)

</details>

<!-- generated:adrs:end -->

<!-- generated:benchmarks:begin -->

### Benchmarks

The newest Criterion point estimates `bench-update.yml` stored on the `perf-baselines` branch, from the `elementwise_cpu_bench` target. They come from GitHub-hosted runners and serve as trend data between commits, not as a performance claim; a claim needs a paired A/B (`make bench-compare`).

- `aarch64-linux`: commit `27102e642d5b`, recorded 2026-10-29T10:59:07+00:00
- `x86_64-linux`: commit `27102e642d5b`, recorded 2026-10-29T11:01:25+00:00

<details>
<summary>17 benchmarks</summary>

| Benchmark                                          | `aarch64-linux` | `x86_64-linux` |
| -------------------------------------------------- | --------------- | -------------- |
| `add_bias_gelu/4096`                               | 6.36 µs         | 2.82 µs        |
| `add_bias_gelu/896`                                | 1.40 µs         | 602.1 ns       |
| `elementwise_mul/4096`                             | 378.6 ns        | 321.0 ns       |
| `gelu/4096`                                        | 6.12 µs         | 2.58 µs        |
| `gelu/896`                                         | 1.34 µs         | 567.5 ns       |
| `layer_norm/4096`                                  | 1.27 µs         | 747.7 ns       |
| `layer_norm/896`                                   | 288.7 ns        | 180.8 ns       |
| `residual_add_layer_norm/fused/hidden384_seq128`   | 21.90 µs        | 13.55 µs       |
| `residual_add_layer_norm/fused/hidden768_seq128`   | 44.38 µs        | 24.23 µs       |
| `residual_add_layer_norm/unfused/hidden384_seq128` | 62.03 µs        | 57.15 µs       |
| `residual_add_layer_norm/unfused/hidden768_seq128` | 122.87 µs       | 113.50 µs      |
| `rms_norm/4096`                                    | 574.2 ns        | 789.5 ns       |
| `rms_norm/896`                                     | 136.9 ns        | 208.6 ns       |
| `silu_inplace/4096`                                | 4.91 µs         | 12.45 µs       |
| `silu_inplace/896`                                 | 1.08 µs         | 2.70 µs        |
| `softmax_attention/128`                            | 20.80 µs        | 10.49 µs       |
| `softmax_attention/512`                            | 328.25 µs       | 159.79 µs      |

</details>

<!-- generated:benchmarks:end -->
