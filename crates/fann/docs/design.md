# lattice-fann design

`lattice-fann` is a compact, dense, feedforward neural-network crate. It is
intended for local student models and classifiers where CPU inference must stay
predictable. It is not a general tensor runtime: its core operation is a dense
matrix-vector multiply followed by an activation.

The crate is CPU-first. GPU execution, parallel batches, and persistence are
optional capabilities around the same layer and network representation; a
complete CPU path does not depend on any of them.

## Scope and performance target

`lattice-fann` targets local classifiers of roughly 100,000 parameters, with a
sub-5 ms CPU-inference budget. This is a scope target for small dense models,
not a latency guarantee across architectures, hardware, or feature sets.

## Component map

```text
NetworkBuilder ──creates──> Layer ──owned by──> Network
       │                         │                 │
       │                         │                 ├── CPU forward inference
       │                         │                 ├── optional parallel batches
       │                         │                 └── binary / serde persistence
       │                         │
       │                         └── weights + biases + Activation
       │
BackpropTrainer ──updates──> Network::layers_mut() / Layer parameters

GpuContext + GpuNetwork (feature = "gpu") ──optional accelerated path──> CPU model
```

| Component | Owns | Responsibility |
| --- | --- | --- |
| `NetworkBuilder` | Input width and ordered layer specifications | Validates an architecture and initializes layers. |
| `Layer` | Shape, row-major weights, biases, activation | Writes one dense affine transform and activation into a supplied buffer. |
| `Network` | Layers and preallocated activation buffers | Checks connectivity, sequences CPU inference, and provides inspection APIs. |
| Training types | Optimizer and gradient state | Update a network's existing parameter storage. |
| GPU types | Device-facing context and acceleration state | Provide an optional execution route while retaining the CPU model. |

A `Layer` does not allocate an output vector for `forward`. A `Network` owns the
mutable buffers that carry a sample through its layers. This separates immutable
parameters from per-inference activations and keeps allocation out of the normal
single-sample forward path.

## Building a network

A builder records an input width and a sequence of output widths and
activations. `input(n)` establishes the width consumed by the first layer;
every `hidden` or `output` call appends a layer. `hidden` and `output` are
runtime-equivalent; the latter documents the caller's intent and the last
appended layer is the actual output layer.

```rust
use lattice_fann::{Activation, NetworkBuilder};

let mut network = NetworkBuilder::new()
    .input(4)
    .hidden(8, Activation::ReLU)
    .output(3, Activation::Softmax)
    .build_with_seed(7)?;
# Ok::<(), lattice_fann::FannError>(())
```

`build` initializes weights from entropy. `build_with_seed` instead feeds one
`SmallRng` seeded from the supplied `u64` through every layer construction, so
the same architecture and seed produce the same parameter values. Both paths
enforce the same rules:

1. The input width is present and nonzero.
2. At least one layer is specified, and every layer width is nonzero.
3. Softmax is used only in the final layer.
4. Layer allocation limits are respected during construction.
5. Adjacent layer widths match when `Network::new` constructs the result.

The final connectivity check is repeated by `Network::new`, because callers can
also construct a network directly from `Layer` values.

### Why Softmax is terminal

Softmax couples the values in its output vector. Its derivative is a Jacobian,
not a pointwise function. The trainer computes the output-layer relationship
where it has the required context, while `Activation` derivative helpers expose
only the Jacobian diagonal. An interior Softmax would therefore train with an
incomplete gradient, so the builder rejects it. See ADR-023 for the accepted
limitation.

## CPU inference lifecycle

Construction performs the allocations normal inference needs. `Network::new`
creates one zero-filled activation buffer per layer; buffer *i* has exactly
`layers[i].num_outputs()` elements. The caller's input is borrowed directly and
is not copied into the buffer set.

```text
caller input ──> Layer 0 ──> buffer 0 ──> Layer 1 ──> buffer 1 ──> ... ──> buffer last
```

For each layer, the network writes `activation(Wx + b)` into its buffer and
checks the result for `NaN` or infinity. The next layer reads the previous
buffer and writes its own. Internally, `split_at_mut` yields distinct borrows of
the previous and current buffer without copying an activation vector.

This is a chain of preallocated buffers, not a two-buffer implementation. The
network retains every layer's latest activation so `Network::activations(i)` can
expose it for debugging or training. A later forward pass overwrites all of
them.

`Network::forward` takes `&mut self` because it overwrites these buffers and
returns a borrowed view of the final one. The result remains current only until
the next mutable use of the network. `forward_async` performs the same CPU work
but returns an owned copy for compatibility with the GPU-facing API.

With the `parallel` feature, `forward_batch` shares read-only layer parameters
across Rayon workers and creates a separate activation-buffer set for each
input. It avoids cloning weights but intentionally trades per-input allocations
for independent concurrent mutable state.

For shapes, matrix layout, numerical checks, SIMD dispatch, and serialization,
see [network.md](network.md).

## Training integration

The training module uses the same `Network` representation as deployment.
`Network::layer_mut` and `Network::layers_mut` let training reach parameter
storage; `Layer::weights_mut` and `Layer::biases_mut` expose the slices it
updates. Model format, initialization, activations, and forward math therefore
remain shared between training and inference.

`BackpropTrainer` is configured through `TrainingConfig`.
`GradientGuardStrategy` controls how training reacts to non-finite gradients.
Those are training-time choices: independently, inference scans every layer
output and returns `FannError::NumericInstability` rather than returning a
non-finite prediction. Parameter updates do not invalidate the reusable
activation buffers; the next forward pass overwrites them with new values.

Read [training.md](training.md) for the backpropagation algorithm, optimizer
state, and gradient safety mechanisms.

## Optional GPU integration

The `gpu` feature accelerates rather than replaces the core model. Without it,
the `gpu` module is absent and CPU `Network` is the complete implementation.
With it, `GpuContext` supplies device-facing resources and `GpuNetwork` provides
an accelerated network interface. The CPU representation remains available as
the compatibility and fallback path.

This division keeps model construction, validation, CPU execution, and
serialization free of device requirements. It also keeps GPU resource lifetime
and dispatch policy out of the `Layer` API, so small CPU models do not pay for a
GPU dependency. See [gpu.md](gpu.md) for device policy, resource reuse, and
fallback behavior.

## Features and portability

| Feature | Effect |
| --- | --- |
| `std` | Default standard-library support. |
| `simd` | Enables target-specific CPU matrix-vector and activation kernels with scalar fallback. |
| `parallel` | Adds Rayon-backed batch inference with shared weights and per-input buffers. |
| `serde` | Enables structured persistence that reconstructs transient buffers during load. |
| `gpu` | Adds the WGPU-based acceleration interface without changing CPU availability. |

SIMD is an optimization, not a different numerical model. Unsupported
architectures and unavailable x86 capabilities fall back to scalar code.

## Validation and failure boundaries

Validation occurs at boundaries where invalid data could otherwise create an
oversized allocation, an out-of-bounds parameter access, or a misleading result.

| Boundary | Check | Failure |
| --- | --- | --- |
| Layer construction | Nonzero dimensions, checked product, 100,000,000-element cap | `InvalidLayerDimensions` or `ShapeTooLarge` |
| Explicit parameters | Exact weights and biases for the declared shape | Count-mismatch error |
| Network construction | Nonempty stack and matching adjacent dimensions | `EmptyNetwork` or `InvalidLayerDimensions` |
| CPU forward | Input/output widths and finite layer results | Size mismatch or `NumericInstability` |
| Binary loading | Header, version, bounds before allocation, exact payload length | `InvalidBuilder`, `InvalidLayerDimensions`, `ShapeTooLarge`, or `EmptyNetwork` |
| Serde loading | Constructor validation and buffer reconstruction | Deserialization error rather than unusable state |

The 100,000,000-element guard applies to a single requested tensor allocation.
At four bytes per `f32`, that is approximately 400 MB. It is both a practical
memory limit and a binary-parser hardening boundary: the parser applies it
before it reads and collects a declared weight payload.

## Documentation map

- [network.md](network.md): layer mechanics, builder validation, CPU execution,
  activations, SIMD, errors, and the binary format.
- [training.md](training.md): backpropagation, optimizer state, and gradient
  safeguards.
- [gpu.md](gpu.md): optional device backend, resource lifecycle, and fallback
  policy.
- [INDEX.md](INDEX.md): entry point for this documentation set.
