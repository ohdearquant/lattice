# ADR-027: Fine-Tuning Pipeline

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-tune

## Context

lattice-tune supports knowledge distillation from teacher LLMs (Claude, GPT, Gemini).
The tune crate manages the training loop, model registry, and adapter injection.

## Decision

### Three capabilities

1. **LoRA adapter loading** — inject low-rank weight deltas into inference models
   at runtime. No full fine-tuning, just adapter application.

2. **Knowledge distillation** — generate training examples from teacher LLMs,
   train student models (via `lattice-fann`), track versions in a model registry.

3. **Model registry** — version tracking for trained models. Records model hash,
   training config, evaluation metrics, deployment status.

### Not a training framework

This is NOT a general-purpose ML training library. It handles the specific
workflow: teacher generates → student learns → adapter saved → adapter loaded
at inference time. The neural network primitives live in `lattice-fann`.

## Consequences

- `lattice-tune` depends on `lattice-fann` for the training loop.
- Teacher API calls are async — this is the only crate with async.
