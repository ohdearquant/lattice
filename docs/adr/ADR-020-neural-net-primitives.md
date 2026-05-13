# ADR-020: Neural Network Primitives

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-fann

## Context

The lattice-tune crate needs a lightweight neural network library for training student
models via knowledge distillation. No external framework dependency (PyTorch,
TensorFlow) — pure Rust, zero-allocation inference.

## Decision

### Dense MLP with pre-allocated buffers

```rust
let net = NetworkBuilder::new()
    .input(384)
    .hidden(256, Activation::ReLU)
    .hidden(128, Activation::ReLU)
    .output(10, Activation::Softmax)
    .build();
```

- **Zero allocation during inference** — all buffers pre-allocated at build time
- **Activations**: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
- **Training**: BackpropTrainer with momentum
- **Optional SIMD** (feature `simd`) and **GPU** (feature `gpu`, wgpu)

### Scope

Small, dense networks only. Not for transformer-scale models. lattice-fann handles:

- Classification heads for knowledge distillation
- Lightweight scoring models trained on distilled data
- Embedding projection layers

## Consequences

- `lattice-fann` is standalone — no external orchestration dependencies.
- `lattice-tune` depends on lattice-fann for the training loop.
- ~7.5K LOC, well-tested, stable API.
