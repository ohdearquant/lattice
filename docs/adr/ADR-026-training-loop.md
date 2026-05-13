# ADR-026: Training Loop

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-fann

## Context

`lattice-fann` needs to train student networks produced by the knowledge distillation pipeline in `lattice-tune`. Training must: (1) compute correct gradients via backpropagation, (2) support mini-batch SGD with momentum and L2 regularization, (3) detect and handle NaN/Inf gradients without crashing the distillation pipeline, (4) converge on standard problems like XOR and linear regression, and (5) produce reproducible results for CI testing.

The training interface must be separable from inference — the `Trainer` trait allows alternative implementations (e.g., GPU-accelerated training in `lattice-tune`'s `GpuTrainer`) without modifying the core backprop logic.

## Decision

A `BackpropTrainer` struct implementing the `Trainer` trait. Training state (momentum velocity buffers) is held inside the trainer, not in the network. The `TrainingConfig` is a value type with a builder API. A `GradientGuardStrategy` enum controls behavior when NaN/Inf gradients are detected.

### Key Design Choices

#### Loss Function and Gradient Computation

MSE (mean squared error) loss: `L = (1/n) * sum((output - target)²)`.

The backward pass computes output layer deltas, then propagates backward through each layer:

- **Softmax output layer**: Full Jacobian applied. `delta[i] = sum_j((output[j] - target[j]) * J[j,i])` where `J[i,j] = output[j] * (delta_{ij} - output[i])`. This is the only case where the full softmax Jacobian is used; `Activation::derivative()` uses the diagonal approximation.
- **All other output layers**: `delta[i] = (output[i] - target[i]) * f'(output[i])`.
- **Hidden layers**: `delta[i] = sum_j(weight[j,i] * delta_next[j]) * f'(activation[i])`. Weights are accessed via `layers[layer_idx+1].weights()` — a `&[f32]` reference with no clone.

Weight gradients accumulate over the batch: `dW[i,j] += delta[i] * input[j]`. Bias gradients: `dB[i] += delta[i]`. After each batch, gradients are averaged by dividing by batch size inside `apply_gradients` (via `lr / batch_size`).

#### Momentum Update (SGD with Momentum)

Classical momentum (not Nesterov):

```
velocity[i] = momentum * velocity[i] - lr * (grad[i] + weight_decay * weight[i])
weight[i] += velocity[i]
```

Velocity buffers (`weight_velocities`, `bias_velocities`) are `Vec<Vec<f32>>` fields on `BackpropTrainer`, one inner `Vec` per layer. They are initialized in `init_velocities()` on the first call to `train()`, zeroed to the correct sizes derived from the network.

Default: `learning_rate = 0.01`, `momentum = 0.9`, `weight_decay = 0.0001`, `batch_size = 32`, `max_epochs = 1000`, `target_error = 0.001`.

#### Gradient Guard

Three strategies, selected per `TrainingConfig::gradient_guard`:

| Strategy          | Behavior                                                             | When to Use                                               |
| ----------------- | -------------------------------------------------------------------- | --------------------------------------------------------- |
| `Error` (default) | Return `FannError::NumericInstability` immediately                   | Fail-fast during development                              |
| `Sanitize`        | Replace NaN/Inf with 0.0, log warning via `tracing::warn!`, continue | Production when occasional instability is expected        |
| `SkipBatch`       | Discard the entire batch, log warning, continue to next batch        | When bad samples are possible but shouldn't stop training |

Gradient validation (`check_gradients_valid`) runs on accumulated batch gradients before `apply_gradients`. The check iterates weight and bias gradient slices, returning the location (layer and index) of the first non-finite value found.

#### Epoch Convergence and Early Stopping

Each epoch computes the average error across all batches. If `avg_error < config.target_error`, training stops early and `TrainingResult.converged = true`. If `avg_error.is_nan()`, a `NumericInstability` error is returned immediately.

`TrainingResult` includes `error_history: Vec<f32>` (one entry per epoch), enabling callers to plot convergence.

#### Reproducible Shuffling

`TrainingConfig::seed: Option<u64>` controls the shuffle RNG. If `Some(seed)`, `SmallRng::seed_from_u64(seed)` is used; if `None`, `SmallRng::from_entropy()`. The RNG is created once before the epoch loop (not per-epoch) to ensure the full shuffle sequence is deterministic across epochs when a seed is provided.

#### No Allocation in Hot Path

Gradient buffers (`weight_grads`, `bias_grads`) and the index vector (`indices`) are allocated once before the epoch loop. Inside each epoch, gradient buffers are zeroed with `fill(0.0)`. The velocity buffers are allocated once in `init_velocities` and reused across all epochs.

The `compute_gradients` method calls `network.forward(input)` for the forward pass (which uses pre-allocated activation buffers), then reads layer weights and activations via `&[f32]` references — no per-sample cloning.

#### `Trainer` Trait

```rust
pub trait Trainer {
    fn train(&mut self, network: &mut Network, inputs: &[Vec<f32>],
             targets: &[Vec<f32>], config: &TrainingConfig) -> FannResult<TrainingResult>;
}
```

`lattice-tune`'s `GpuTrainer` can implement this trait to provide GPU-accelerated training as a drop-in replacement. The separation ensures `lattice-fann`'s core stays sync and allocation-free.

### Alternatives Considered

| Alternative                               | Pros                                                        | Cons                                                                                      | Why Not                                                                                                   |
| ----------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Adam optimizer                            | Faster convergence, less hyperparameter tuning              | More state per parameter (moment estimates), more complex implementation                  | Momentum SGD is sufficient for small classification heads; Adam available in `lattice-tune`'s GPU trainer |
| Cross-entropy loss for classification     | Correct loss for softmax output                             | Requires target as one-hot or probability vector, more complex delta computation          | MSE works for soft labels from teacher models (which sum to ~1); simpler implementation                   |
| Storing pre-activation `z` for derivative | Eliminates diagonal Softmax approximation for hidden layers | Extra buffer per layer, more complex memory management                                    | `lattice-fann` targets final-layer softmax only; storing `z` adds complexity for no current benefit       |
| Training state on `Network`               | Network carries its own optimizer state                     | Couples inference and training; prevents using the same network in multiple training runs | Separation allows using the same trained `Network` with multiple `Trainer` instances                      |

## Consequences

### Positive

- Full softmax Jacobian in the output layer ensures correct gradient flow when the student network has a softmax output (the most common `lattice-tune` configuration)
- `GradientGuardStrategy::Sanitize` keeps the distillation pipeline running through occasional numeric instability without aborting a long teacher labeling session
- Seeded RNG enables bit-exact reproducibility for CI assertions on training convergence
- `error_history` enables post-hoc convergence analysis in `lattice-tune`'s distillation metrics

### Negative

- `BackpropTrainer::compute_gradients` allocates `deltas: Vec<Vec<f32>>` per training sample. This is the most expensive allocation in the training hot path. A pre-allocated deltas buffer (similar to how gradient buffers are pre-allocated) would eliminate it.
- `Trainer` trait is synchronous only — async training requires a different interface, which is why `lattice-tune` implements its own `GpuTrainer`.

### Risks

- The `compute_gradients` method takes `&mut Network` for the forward pass but then holds `&Network` references (layers, activations) for the backward pass. Rust's borrow checker enforces this correctly, but the code pattern (calling `forward` then borrowing the result immutably) is non-obvious.
- `check_gradients_valid` only checks weight and bias gradients, not activations. NaN activations are checked in `Network::forward` via `check_numeric_stability` (always active, including release builds). Both checks are necessary for full coverage.

## References

- `crates/fann/src/training/backprop.rs` — `BackpropTrainer`, `compute_gradients`, `apply_gradients`
- `crates/fann/src/training/mod.rs` — `TrainingConfig`, `TrainingResult`, `Trainer` trait, `GradientGuardStrategy`
- `crates/fann/src/training/gradient.rs` — `check_gradients_valid`, `sanitize_gradients`, `GradientGuardStrategy`
- `crates/fann/src/network/mod.rs` — `check_numeric_stability` (always-on NaN/Inf guard)
