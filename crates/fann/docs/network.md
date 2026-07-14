# Network core reference

This page describes the CPU network core: dense-layer storage and evaluation,
builder validation, preallocated activation buffers, activation behavior, and
persistence. For training and the optional GPU path, see
[design.md](design.md).

## Mathematical model

`Network` is an ordered, nonempty sequence of `Layer` values. A layer maps an
input vector of width *I* to an output vector of width *O*:

```text
z[o] = bias[o] + sum(input[i] * weight[o, i], i = 0..I)
output[o] = activation(z[o])
```

The output width of layer *k* must equal the input width of layer *k + 1*.
`Network::new` checks this whether layers originate from a builder, binary data,
Serde, or direct construction. There is no input-layer object: the first layer
consumes the slice given to `Network::forward`, and the final layer buffer is the
network result.

## Layer representation

`Layer` owns the data below.

| Field | Shape | Meaning |
| --- | --- | --- |
| `num_inputs` | scalar | Width of a vector accepted by the layer. |
| `num_outputs` | scalar | Width written by the layer. |
| `weights` | `num_outputs * num_inputs` `f32`s | Dense matrix in output-major row order. |
| `biases` | `num_outputs` `f32`s | One additive bias for each output. |
| `activation` | `Activation` | Function applied after the affine calculation. |

The row for output `o` begins at `o * num_inputs`. A two-output, three-input
layer stores its matrix as:

```text
[ w00, w01, w02,  w10, w11, w12 ]
   └── output 0 ──┘  └── output 1 ──┘
```

With input `[x0, x1, x2]`, weights `[1, 2, 3, 4, 5, 6]`, and biases
`[0.1, 0.2]`, the affine values are `1*x0 + 2*x1 + 3*x2 + 0.1` and
`4*x0 + 5*x1 + 6*x2 + 0.2` before activation.

`Layer::forward` requires an input slice of exactly `num_inputs` and an output
slice of exactly `num_outputs`. It writes in place—there is no output allocation
or returned vector. Width mismatches are errors before matrix computation.

### Creation and parameter checks

`Layer::new` creates zero biases and samples weights from a normal
Xavier/Glorot distribution:

```text
mean = 0
standard deviation = sqrt(2 / (num_inputs + num_outputs))
```

`Layer::new_with_rng` uses the same distribution but accepts a caller RNG,
supporting reproducible builder initialization. `Layer::zeros` creates zero
weights and biases. `Layer::with_weights` accepts existing parameter vectors only
after validating all of these invariants:

- both dimensions are nonzero;
- the `num_inputs * num_outputs` product does not overflow;
- the weight tensor has at most 100,000,000 elements;
- the weight vector length exactly equals that product; and
- the bias vector length exactly equals `num_outputs`.

These checks make row indexing safe in `forward` and reject unreasonable
allocation requests. The 100,000,000-element cap is roughly 400 MB for one
`f32` allocation. It is enforced by `validate_layer_dimensions` and applied by
decoding before a weight payload is collected.

`weights`, `biases`, and their mutable forms expose slices for persistence and
training. `Layer::num_params` counts weights plus biases. `get_weight` and
`set_weight` use the output-major index and return `None` or `false` for an
invalid coordinate instead of panicking.

## Builder behavior

`NetworkBuilder` is the usual construction API:

```rust
use lattice_fann::{Activation, NetworkBuilder};

let network = NetworkBuilder::new()
    .input(784)
    .hidden(128, Activation::ReLU)
    .dense(64)
    .output(10, Activation::Softmax)
    .build()?;
# Ok::<(), lattice_fann::FannError>(())
```

The builder stores one input width and ordered `(output_width, activation)`
pairs. Calling `input` again replaces the prior input size. `hidden` and `output`
both append a pair; `output` is a readability marker and does not have distinct
runtime behavior. `dense(size)` is `hidden(size, Activation::ReLU)`.

`build` uses entropy-seeded initialization. `build_with_seed(seed)` creates one
seeded `SmallRng` and passes it through every layer initialization. Given the same
architecture and seed, it produces identical parameter values.

Both build paths reject a missing input, zero input, no layers, or a zero-size
layer. They also reject Softmax before the final layer, then construct each layer
using the preceding width as its input. `Network::new` rechecks the final chain
and allocates reusable activation buffers.

### Softmax is output-only

Softmax normalizes across a vector and has a full Jacobian:

```text
J[i, j] = s[i] * (delta(i, j) - s[j])
```

The general `Activation` derivative helpers provide only its diagonal,
`s[i] * (1 - s[i])`. Output-layer backpropagation has the context to use the
complete relationship, but an interior Softmax would silently train using the
diagonal approximation. The builder rejects that topology; see ADR-023 for the
accepted limitation.

## Preallocated forward execution

When a network is constructed, it allocates exactly one activation `Vec<f32>`
for every layer. Buffer *i* has `layers[i].num_outputs()` values. The user input
is borrowed by the first layer rather than copied.

```text
layers:  L0 (I0 -> O0)       L1 (O0 -> O1)       L2 (O1 -> O2)
buffers:         [O0]                 [O1]                 [O2]

input ───> L0 ───> buffer 0 ───> L1 ───> buffer 1 ───> L2 ───> buffer 2
```

The first layer writes buffer 0. For each later layer, the previous buffer is
read and the current layer buffer is written. `split_at_mut` makes the two
borrows disjoint without cloning or copying activations. Therefore the normal
`Network::forward` hot path makes no activation-buffer allocation.

This is not a two-buffer ping-pong design: the network retains every layer's
latest output. `Network::activations(layer_index)` can expose any current buffer
for debugging or training, and the next forward pass overwrites all of them.

`forward` takes `&mut self` and returns a slice borrowed from the final buffer.
It remains the current output only until the network is mutably used again.
`forward_async` does the same CPU work but copies the final slice into an owned
`Vec<f32>` for an asynchronous interface compatible with the GPU API.

### Finite-output check

After every layer, the network checks each produced value for `NaN` and
infinity. This check is active in release builds: it returns
`FannError::NumericInstability` with the layer and element position rather than
returning a non-finite prediction. The error string is formatted only after a
failure, not on each successful hot-path call.

Shape validation and finite-output validation solve different problems. A valid
shape can still produce `NaN` or infinity from bad input, bad parameters, or
overflow, so both checks are necessary.

### Parallel batches

With the `parallel` feature, `forward_batch` shares immutable layer weights and
biases across Rayon workers. Each input gets its own freshly allocated
activation-buffer set, avoiding mutable sharing without cloning parameter
matrices. This intentionally differs from the single-sample path: it exchanges
per-input allocations for independent parallel execution.

## Activation reference

`Activation` supports single-value evaluation, in-place batch evaluation, and
derivatives calculated from an activation output `y`. `forward` is useful for
pointwise functions. A scalar Softmax evaluates to `1.0`; use `forward_batch` to
normalize an actual output vector.

| Variant | Forward rule | Derivative from output | Range |
| --- | --- | --- | --- |
| `Linear` | `x` | `1` | unbounded |
| `Sigmoid` | `1 / (1 + exp(-x))` | `y * (1 - y)` | `(0, 1)` for finite input |
| `Tanh` | `tanh(x)` | `1 - y²` | `[-1, 1]` |
| `ReLU` | `max(0, x)` | `1` when `y > 0`, else `0` | `[0, infinity)` |
| `LeakyReLU(a)` | `x` if `x > 0`, else `a*x` | `1` when `y > 0`, else `a` | usually unbounded |
| `Softmax` | normalized exponentials over the slice | diagonal only: `y * (1 - y)` | finite results sum to `1` |

`ReLU` is the default. `Activation::DEFAULT_LEAKY_ALPHA` is `0.01`, while a
`LeakyReLU` value stores its chosen alpha. `is_bounded` is true for Sigmoid,
Tanh, and Softmax; `is_softmax` identifies the vector-valued case.

### Stable Sigmoid and Softmax

Sigmoid uses equivalent formulas on either side of zero:

```text
x >= 0:  1 / (1 + exp(-x))
x < 0:   exp(x) / (1 + exp(x))
```

For a large negative input, the latter avoids evaluating `exp(-x)` with a huge
positive exponent. Batch Softmax first subtracts the largest input `m`, then
normalizes `exp(x[i] - m)`. This translation does not change probabilities but
avoids overflow for uniformly large logits.

An empty Softmax slice remains empty. If the exponent sum is zero, no division
is performed. Activations do not repair non-finite input; the network's
layer-output check reports non-finite results after evaluation.

### Derivative scope

Pointwise derivatives can be computed from the output, avoiding a second
pre-activation buffer. Softmax is different: its full derivative couples all
outputs. `derivative` and `derivative_batch` deliberately expose only the
diagonal; output-layer backpropagation handles the full Jacobian. They are not a
general full-Softmax-Jacobian API.

## SIMD paths

The `simd` feature changes implementation strategy, not layer semantics.

| Target | Dot-product path | Vector work | Scalar tail |
| --- | --- | --- | --- |
| AArch64 | NEON | Four 4-lane FMA accumulators: 16 values | 0–3 values |
| x86_64 with AVX-512F | AVX-512 | Four 16-lane FMA accumulators: 64 values | 0–15 values |
| x86_64 with AVX2 + FMA | AVX2/FMA | Four 8-lane FMA accumulators: 32 values | 0–7 values |
| Other or unsupported runtime | Scalar | One value at a time | not applicable |

On x86, runtime feature checks choose AVX-512F first, then AVX2 plus FMA, then
the scalar loop. AArch64 relies on mandatory NEON. The independent accumulators
reduce FMA dependency latency before being combined. x86 matrix code prefetches
the next weight row when one exists; the hint does not change correctness.

Batch ReLU and LeakyReLU also use SIMD: NEON processes four values at a time and
AVX2 processes eight. LeakyReLU uses vector selection instead of a per-lane
branch. Every vector routine handles its remaining tail scalarly, and x86 code
runs only after the required CPU feature is detected.

## Binary format

`Network::to_bytes` emits compact little-endian version-1 data. It has no
padding, footer, checksum, or optional fields; a decoder must consume exactly
the bytes specified below.

### Header

| Bytes | Encoding | Value |
| --- | --- | --- |
| `0..4` | four ASCII bytes | Magic `FANN` |
| `4..8` | little-endian `u32` | Version `1` |
| `8..12` | little-endian `u32` | Layer count `L` |

`L` cannot be zero. The first layer begins at byte 12.

### Layer records

Each layer is written in network order:

| Field | Encoding | Notes |
| --- | --- | --- |
| input width | little-endian `u32` | `I`, nonzero when decoded |
| output width | little-endian `u32` | `O`, nonzero when decoded |
| activation tag | `u8` | Listed below |
| LeakyReLU alpha | little-endian `f32` | Present only for tag `4` |
| weights | `I * O` little-endian `f32`s | Output-major row layout |
| biases | `O` little-endian `f32`s | Output order |

| Tag | Activation | Extra bytes |
| --- | --- | --- |
| `0` | `Linear` | none |
| `1` | `Sigmoid` | none |
| `2` | `Tanh` | none |
| `3` | `ReLU` | none |
| `4` | `LeakyReLU(alpha)` | alpha `f32` |
| `5` | `Softmax` | none |

A non-LeakyReLU record is `9 + 4*I*O + 4*O` bytes. LeakyReLU adds four alpha
bytes. The total serialized size is the 12-byte header plus all layer records.

### Versioning and parsing rules

The reader accepts version `1` only. A changed field order, width, tag, or
semantic must receive a new version and explicit reader support; the current
parser rejects unknown versions rather than guessing an interpretation.

`Network::from_bytes` treats input as untrusted and validates before allocating:

1. Each read checks remaining bytes by subtraction, avoiding an overflow-prone
   `position + count` bounds calculation with hostile counts.
2. Before reserving the layer vector, declared `L` must fit the remaining byte
   budget divided by nine, the smallest layer header.
3. The layer count also receives the general allocation-size guard.
4. Every `I` and `O` is validated before reading weights: zero values,
   multiplication overflow, and more than 100,000,000 elements fail before a
   large payload can be collected.
5. Weight and bias byte counts use checked multiplication before reading.
6. Completed records pass through `Layer::with_weights`; `Network::new` then
   validates connectivity and recreates transient activation buffers.
7. The final reader position must equal the source length. Trailing data is an
   error, never ignored padding.

Bad magic, unsupported version, truncation, unknown tags, impossible counts,
arithmetic overflow, or trailing data produce an error instead of a partial
model. An empty declared network returns `EmptyNetwork`; an oversized layer is
rejected as `ShapeTooLarge` at the early dimension check.

## Serde and errors

The optional `serde` representation is distinct from the compact binary format.
Runtime activation buffers are skipped because they are transient output state,
not model parameters. On deserialization, a `NetworkData` shadow routes layers
through `Network::new` to validate connectivity and rebuild buffers. A `LayerData`
shadow routes vectors through `Layer::with_weights` to validate their lengths.
This prevents a parameter-valid-looking network from reaching a first forward
pass with empty buffers or malformed row storage.

`FannResult<T>` is `Result<T, FannError>`. Core error variants identify the
failed boundary:

| Error | Typical source |
| --- | --- |
| `InputSizeMismatch` / `OutputSizeMismatch` | Forward slice has the wrong width. |
| `WeightCountMismatch` / `BiasCountMismatch` | Parameter vectors disagree with a declared shape. |
| `InvalidLayerDimensions` | A width is zero or adjacent layers do not connect. |
| `ShapeTooLarge` | A tensor exceeds the cap or its product overflows. |
| `EmptyNetwork` | Construction or decoding has no layers. |
| `InvalidBuilder` | Builder setup or binary input is malformed. |
| `NumericInstability` | A layer output contains `NaN` or infinity. |
| `InvalidDistributionParams` | Xavier/Glorot distribution setup failed. |
| `SerializationError` | Serde-specific failure when that feature is enabled. |

For `TrainingError` and `GradientError`, see [training.md](training.md).
