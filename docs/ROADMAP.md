# Lattice Roadmap

Lattice is a pure-Rust transformer inference engine for Apple Silicon and modern CPUs. This
document is the forward-looking view: where the engine is, what is actively being worked, and
the new capability lanes opening next. The issue tracker is the source of truth for status;
this page explains how the issues fit together. Last revised: July 2026.

## Where the engine is

The current generation path is the Qwen3.5/3.6 family: a hybrid of GatedDeltaNet
(linear-attention) and GQA (softmax-attention) layers, running on Metal GPU kernels and
AVX2/NEON SIMD CPU kernels. On top of that core the engine ships:

- An OpenAI-compatible HTTP server (`lattice serve`) and an interactive CLI (`lattice chat`)
- Lattice Studio, a native macOS app over the same engine
- Q4/Q8 quantized inference with committed perplexity baselines
- LoRA fine-tuning (CPU backward pass) plus a governed adapter manifest and runtime mixture
- Grammar-constrained decoding (JSON Schema to PDA)
- Speculative decoding (native multi-token prediction, opt-in)

Benchmarks and the capability matrix live in the [README](../README.md#benchmarks); methodology
in [docs/benchmarks.md](benchmarks.md).

## Active focus

The bulk of the open backlog falls into four streams. These continue regardless of the new
lanes below:

1. **Decode and prefill throughput** on the hybrid path. Decode is weight-bandwidth-bound
   (GEMV/MLP across all layers); prefill work centers on GEMM tiling and occupancy, and on the
   GatedDeltaNet chunked scan. This is the largest open cluster in the tracker.
2. **Serving and Studio hardening.** Fail-closed request validation, cross-turn KV reuse,
   streaming correctness (UTF-8 boundaries, added tokens), and Studio workflows backed by real
   engine state.
3. **Quantization quality and coverage.** Tighter perplexity tracking, kernel coverage for
   quantized paths, and sub-4-bit exploration (a W3 MLP-only weight path exists as a parked
   draft, #515).
4. **Correctness infrastructure.** Cross-framework parity gates in CI (greedy-token agreement
   vs HF transformers), differential tests for every numeric kernel, and mutation-sensitive
   regression tests.

## New capability lanes

Three lanes extend the engine beyond text-in/text-out on one model family. Each has a tracking
issue with the architectural delta and a concrete first milestone.

### Vision tensors ([#564](https://github.com/ohdearquant/lattice/issues/564))

A native image input path: preprocessing, a ViT-class vision encoder, and the projector that
splices visual tokens into the decoder sequence. The Qwen3.6-27B checkpoint already ships its
vision tower weights, and three existing issues (#424, #381, #382) anchor the model-level work.
The lane adds the op set the text engine never needed: patch extraction (Conv2d or
unfold+GEMM), mean-subtracting LayerNorm, GELU variants, bicubic position-embedding
interpolation, and multimodal RoPE.

First milestone: patch embedding plus one encoder block on CPU, with a differential parity
script against the reference implementation (max L2 error below 1e-4).

### Audio tensors ([#565](https://github.com/ohdearquant/lattice/issues/565))

A speech-in path built on the Whisper-style frontend that the Qwen audio models adopt
unchanged: resample to 16 kHz, STFT (window 400, hop 160), 128-channel mel filterbank, log
compression, then a small Conv1D stack into standard transformer encoder blocks. Speech
synthesis (codec tokens plus vocoder) is explicitly a later milestone; it has a different
compute shape and its own tracking will follow once speech-in lands.

First milestone: a pure-Rust log-mel extractor matching torchaudio within 1e-5 L2 distance on
synthetic audio.

### Gemma 4 model family ([#566](https://github.com/ohdearquant/lattice/issues/566))

A second model family beside Qwen, spanning edge (E2B/E4B) to flagship (31B) checkpoints with
an MoE variant. The architecture exercises what the engine does not yet have: interleaved
sliding-window and global attention with dual RoPE bases, QK-normalization, logit softcapping,
GeGLU, scaled and tied embeddings, and per-layer embeddings on the edge models. Sliding-window
layers also change KV-cache geometry (bounded per-layer windows).

First milestone: text-only greedy generation on the smallest checkpoint with first-tokens
parity against HF transformers, the same gate class the CI already runs for Qwen.

## Foundation work the lanes depend on

**Model IR ([#177](https://github.com/ohdearquant/lattice/issues/177), ADR-059).** Today the
forward pass is family-specific. A ModelSpec to IR to KernelPlan layer is the prerequisite for
the Gemma lane and pays for itself again with every family after that: new architectures become
data (a spec) plus a small number of genuinely new kernels, instead of a parallel forward-path
implementation. The Gemma lane begins by sizing the minimal IR slice its first milestone needs.

## Research lanes

Longer-horizon investigations tracked separately: confidence-gated adaptive reasoning (#482),
GatedDeltaNet recurrent state as a serveable object (#481), state-routed sparse attention
(#480), and a structured-pruning/compression cluster (#492, #495, #502, #503). These feed the
main streams as results firm up.

## How work lands

Every performance change carries a before/after benchmark comparison from the same session.
Every cross-framework numeric behavior lands with a differential test against the reference
implementation before integration. PRs touching inference or embed crates run an end-to-end
parity gate against HF transformers in CI. New lanes follow the same discipline from their
first op onward: CPU parity first, Metal kernels second.
