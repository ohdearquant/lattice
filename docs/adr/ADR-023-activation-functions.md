# ADR-023: Activation Functions

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-fann

## Context

The `lattice-fann` neural network library needs activation functions that serve two purposes simultaneously: fast forward-pass inference on small networks and correct gradient computation for backpropagation training. The design must handle the hot path (repeated batch inference) without heap allocation, support optional SIMD acceleration on both Apple Silicon and x86_64, and produce numerically stable outputs to avoid training divergence.

A secondary constraint is that derivative computation must be efficient for backprop. The standard approach — computing `f'(x)` from `x` — requires storing the pre-activation value. This library avoids that storage cost by computing derivatives from the activation _output_ `y = f(x)` where algebraically possible, which holds for sigmoid (`f'(y) = y(1-y)`), tanh (`f'(y) = 1 - y²`), ReLU (`f'(y) = 1 if y > 0`), and LeakyReLU.

## Decision

A single `Activation` enum with six variants. All variants are `Copy` (zero allocation, embedded inline in `Layer`). Two entry points are provided: `forward(x: f32) -> f32` for element-wise scalar use, and `forward_batch(&mut [f32])` for vectorized in-place operation on a layer's output buffer.

### Key Design Choices

#### Variants

| Variant          | Formula                              | Default            |
| ---------------- | ------------------------------------ | ------------------ |
| `Linear`         | `f(x) = x`                           | No                 |
| `Sigmoid`        | `1 / (1 + e^-x)`                     | No                 |
| `Tanh`           | `tanh(x)`                            | No                 |
| `ReLU`           | `max(0, x)`                          | Yes (`#[default]`) |
| `LeakyReLU(f32)` | `x if x > 0, else alpha * x`         | No                 |
| `Softmax`        | `exp(xᵢ - max) / sum(exp(xⱼ - max))` | No                 |

`ReLU` is the default because it is the most common activation for hidden layers in the distillation student networks that `lattice-tune` constructs.

#### SIMD Acceleration (feature `simd`)

Only ReLU and LeakyReLU receive SIMD implementations. These are the only two activations where the operation is a simple pointwise comparison + blend with no transcendental math, making SIMD worthwhile.

- **aarch64 (NEON)**: `vld1q_f32` / `vmaxq_f32` / `vst1q_f32` for ReLU in 4-wide chunks. LeakyReLU uses `vcgeq_f32` + `vbslq_f32` (bitwise select) to avoid branching entirely.
- **x86_64 (AVX2)**: `_mm256_loadu_ps` / `_mm256_max_ps` / `_mm256_storeu_ps` for ReLU in 8-wide chunks. LeakyReLU uses `_mm256_cmp_ps` + `_mm256_blendv_ps`.

Runtime detection (`is_x86_feature_detected!("avx2")`) on x86_64; NEON is mandatory on aarch64 so no runtime check. Both paths process a scalar tail for remainder elements.

Sigmoid and Tanh involve `exp()` / `tanh()` calls that are not worthwhile to SIMD-accelerate at this network scale, and Softmax requires a full-slice reduce that cannot be tiled trivially.

#### Numerically Stable Sigmoid

A branching implementation avoids the `e^(-x)` overflow when `x` is large and negative:

- `x >= 0`: `1 / (1 + exp(-x))`
- `x < 0`: `exp(x) / (1 + exp(x))`

This is the `stable_sigmoid` helper, called by both `forward` and `forward_batch`.

#### Numerically Stable Softmax

The max-subtraction trick: subtract `max(xᵢ)` before exponentiating, preventing overflow when inputs are large. After computing `exp(xᵢ - max)`, divide by the sum. Guard added: if `sum <= 0.0`, division is skipped (all-zero output rather than NaN).

#### Softmax Derivative Approximation

The full Jacobian of softmax is `J[i,j] = s[i](δᵢⱼ - s[j])`, but `derivative()` and `derivative_batch()` use the diagonal approximation `s[i](1 - s[i])`, consistent with sigmoid. This is a known approximation (tracked as `FP-095`) that simplifies the backprop implementation at the cost of gradient accuracy when softmax is used in intermediate layers. In the current use of `lattice-fann` (softmax only on final output layer for classification), the full Jacobian _is_ applied in `backprop.rs`'s `compute_gradients` method, making this approximation only relevant if `derivative()` is called directly.

#### Serde Support

Conditional via `feature = "serde"`. `LeakyReLU(f32)` serializes the alpha value. All variants use `snake_case` names.

### Alternatives Considered

| Alternative                                         | Pros                                    | Cons                                                                                  | Why Not                                                         |
| --------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Trait object per activation (`Box<dyn Activation>`) | Extensible, user-defined activations    | Heap allocation per layer, vtable dispatch, `Clone` requires extra work               | Zero-allocation inference is a first-class constraint           |
| Function pointer stored in layer                    | Low overhead, user-definable            | Cannot derive `Serialize`/`Deserialize`, no derivative association                    | Loses serde and breaks typed interface                          |
| GELU / ELU variants                                 | Modern activations used in transformers | Not needed for small classification student networks in current scope                 | Scope: ADR-001 explicitly limits to dense MLPs; add when needed |
| SIMD for all variants                               | Maximum throughput                      | Transcendental functions (exp, tanh) have no simple SIMD path without external crates | Dependency cost exceeds benefit at current network sizes        |

## Consequences

### Positive

- `Activation` is `Copy` — embedded in `Layer`, no heap allocation
- SIMD paths active by default on Apple Silicon (NEON mandatory) and on AVX2 machines (runtime-detected)
- Backprop implementation can call `derivative(output)` without re-running the forward pass or storing pre-activation state
- Stable sigmoid and softmax prevent NaN/Inf in training at no runtime cost on the non-overflow path

### Negative

- `derivative()` for Softmax uses a diagonal approximation, not the full Jacobian. If the library is extended to softmax in hidden layers, gradient accuracy will degrade
- No user-defined activation functions without forking the enum
- LeakyReLU `alpha` is stored inside the enum variant, so changing alpha requires rebuilding the network (not a concern for current use cases)

### Risks

- The scalar tail in SIMD paths must be correct: off-by-one in the chunk offset computation would silently process elements twice or skip elements. Current implementation uses `n / 4` (or `/ 8`) chunks, then iterates `offset..n` for the tail.
- Softmax `forward()` (single-element) returns `1.0` as a sentinel, which is incorrect if called element-wise on a multi-output layer. The docstring notes this, but calling code must use `forward_batch` for softmax layers.

## References

- `crates/fann/src/activation.rs` — all forward, derivative, and SIMD implementations
- `crates/fann/src/training/backprop.rs` — full softmax Jacobian applied in `compute_gradients`
- `crates/fann/src/network/mod.rs` — `forward_batch` called per layer in `forward_into_buffers`
