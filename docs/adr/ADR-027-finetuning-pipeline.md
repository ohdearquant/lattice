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

### Current implementation status (amended 2026-07-14)

The three capabilities above define the accepted subsystem boundaries; they do
not imply that every stage is implemented end to end. In the shipped code:

- `DistillationPipeline::label_single` fails closed with a `TeacherApi` error
  instead of issuing an HTTP request or returning fabricated labels. A
  deterministic simulated-label path (`label_single_simulated` /
  `label_batch_simulated`) exists only behind the non-default
  `simulated-teacher` feature; its results are marked with `LabelSource::Simulated`
  and `to_training_examples` rejects them rather than attributing them to the
  configured teacher.
- The CPU `TrainingLoop` validates configuration, batches data, invokes
  callbacks, records metrics, and emits in-memory checkpoint records. Its loss
  and accuracy are simulated, and it neither forwards through nor updates a
  `lattice-fann` network.

The provider configurations and training-loop types therefore describe
integration contracts for a future live teacher client and student-training
backend. Their current outputs must not be treated as teacher-generated labels
or trained model metrics.

## Consequences

- `lattice-tune` depends on `lattice-fann`, although the placeholder CPU
  `TrainingLoop` does not currently invoke a network.
- A live teacher integration is expected to require asynchronous I/O. The
  current distillation path is synchronous and contains no HTTP client.
