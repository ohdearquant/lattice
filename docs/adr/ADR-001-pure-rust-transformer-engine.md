# ADR-001: Pure Rust Transformer Engine

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-inference

## Context

`lattice-inference` provides local embedding generation (for recall, search, suggest) without
depending on external API calls. The inference engine runs transformer models
(BERT for embeddings, Qwen3 for generation) entirely in Rust — no ONNX, no
Python, no external runtime.

## Decision

### Pure Rust, 57.5K LOC

The engine implements the full transformer forward pass:

- **BERT** (encoder-only) — bidirectional attention, used by mE5/BGE embedding models
- **Qwen3** (decoder-only) — causal GQA + RoPE + SwiGLU, used for generation

### Hardware acceleration

- **Metal GPU** (Apple Silicon) — fused attention kernel, head_dim=128, GQA groups=2
- **NEON** (aarch64 CPU) — fallback when Metal shape constraints don't match
- **AVX2** (x86_64) — 8-wide f32 SIMD
- **Scalar** — universal fallback

Runtime feature detection selects the best path. `LATTICE_NO_GPU=1` forces CPU.

### Model management

Models are downloaded from HuggingFace and cached at `~/.lattice/models/`.
`ensure_model_files()` verifies file hashes before loading. Model format is
safetensors — the engine parses them directly, no framework dependency.

### Speculative decoding (generation only)

N-gram speculative decoding + Multi-Token Prediction verifier for faster
autoregressive generation. Not used for embeddings.

## Consequences

- `lattice-inference` has zero external ML dependencies — pure Rust math.
- ~153 `unsafe` blocks (SIMD intrinsics) — each documented with safety justification.
- Binary size: the inference engine adds ~2MB to the release binary.
- Startup: model loading takes 1–3s depending on model size and disk speed.
