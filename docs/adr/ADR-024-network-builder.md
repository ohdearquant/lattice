# ADR-024: Network Builder Pattern

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-fann

## Context

`Network` requires: (1) validated layer dimensions (output of layer N must match input of layer N+1), (2) pre-allocated activation buffers for each layer, (3) optional reproducible weight initialization via seeded RNG. Constructing a `Network` directly from a `Vec<Layer>` is possible but error-prone — callers must get dimensions right and allocate buffers manually.

The library needs a construction interface that is readable, validates invariants eagerly, and handles both random initialization (default) and seeded initialization (for reproducibility in testing and distillation experiments).

## Decision

A fluent `NetworkBuilder` struct with consuming builder methods. The build step validates and constructs; all error reporting is deferred to `build()` / `build_with_seed()`.

### Key Design Choices

#### Fluent API with Consuming Methods

All builder methods take and return `Self` by value. This enables method chaining without holding a mutable reference and makes partial builders inexpressible at runtime (an `input(...)` call must precede any `hidden`/`output` call, enforced by the `input_size: Option<usize>` field).

```
NetworkBuilder::new()
    .input(784)
    .hidden(128, Activation::ReLU)
    .output(10, Activation::Softmax)
    .build()
```

The `output()` method is semantically identical to `hidden()` — both push `(size, Activation)` onto the layers vector. The distinction is purely for readability at the call site.

A `dense(size)` shorthand pushes `(size, Activation::ReLU)`, covering the common case of hidden layers with default activation.

#### Validation in `build()`

Errors returned as `FannError::InvalidBuilder(String)`:

- `input_size` not set
- No layers added
- `input_size == 0`
- Any layer with `size == 0` (reported with layer index)

Layer-to-layer dimension compatibility is validated inside `Network::new()` (not in the builder), because `Network::new` takes a `Vec<Layer>` and must validate regardless of whether the layers came from the builder.

#### Seeded Initialization (`build_with_seed`)

`build_with_seed(seed: u64)` uses `rand::rngs::SmallRng::seed_from_u64(seed)` and passes it to `Layer::new_with_rng`. The RNG is created once and threaded through all layers, so the same seed always produces bitwise-identical weight values regardless of how many hidden layers are added. This is used by `lattice-tune` to ensure reproducible training experiments.

`build()` (without seed) relies on `Layer::new()` which uses its own entropy source.

#### Buffer Pre-Allocation

On `Network::new(layers)`, the activation buffers are allocated:

```
buffers = layers.iter().map(|l| vec![0.0; l.num_outputs()]).collect()
```

The buffers field is skipped during serde serialization (`#[serde(skip)]`) because it is derived from layer dimensions and can be reconstructed on deserialization. `from_bytes` calls `Network::new(layers)` which re-allocates them.

#### Parallel Batch Inference (feature `parallel`)

When the `parallel` feature is enabled, `Network::forward_batch(&[Vec<f32>])` is added. It shares the immutable `&self.layers` across Rayon threads, allocating only fresh activation buffers per input in the parallel iterator. This avoids cloning weight matrices.

### Alternatives Considered

| Alternative                                          | Pros                  | Cons                                                                       | Why Not                                                                     |
| ---------------------------------------------------- | --------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Constructor with explicit `Vec<(usize, Activation)>` | Fewer types           | No guided validation, harder to read                                       | Readability matters for distillation pipeline configuration                 |
| Separate `Layer` type exposed to caller              | Full control          | Callers must get dimensions right before construction                      | Violates fail-fast principle; buffer allocation is an implementation detail |
| Validated builder with `&mut self` methods           | Familiar Rust pattern | Chains require `let mut builder = ...; builder.hidden(...)` — more verbose | Consuming methods enable single-expression construction                     |
| Macro-based network definition                       | Compact syntax        | Compile-time only, no dynamic construction, harder to read errors          | `lattice-tune` constructs networks from runtime config                      |

## Consequences

### Positive

- Invalid configurations caught at `build()` call site with descriptive messages
- `build_with_seed` gives exact reproducibility for CI tests and distillation experiments
- Activation buffers pre-allocated once; zero allocation during inference
- `architecture()` method on `Network` produces a human-readable summary (used in logging)
- `activations(layer_index)` exposes intermediate buffers for training without additional allocation

### Negative

- `output()` is not structurally enforced to be called last — a caller can call `hidden()` after `output()` and nothing prevents it at compile time
- `build()` and `build_with_seed()` are two separate code paths with duplicated validation logic; both must be updated if validation rules change

### Risks

- `serde(skip)` on buffers means deserialized networks have correctly-sized buffers (re-allocated in `Network::new`), but if serde bypass is ever introduced (e.g., unsafe deserialization), the buffers could be absent. The `forward()` method will panic on an empty buffers slice in that case.

## References

- `crates/fann/src/network/builder.rs` — `NetworkBuilder` implementation
- `crates/fann/src/network/mod.rs` — `Network::new`, buffer allocation, `forward_into_buffers`, parallel batch
- `crates/fann/src/network/serialization.rs` — `to_bytes` / `from_bytes` binary format (FANN magic, version 1)
- `crates/fann/src/layer.rs` — `Layer::new`, `Layer::new_with_rng`, `Layer::with_weights`
