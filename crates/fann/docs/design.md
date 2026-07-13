# lattice-fann design

`lattice-fann` implements lightweight neural network primitives for sub-5ms CPU
inference on small dense networks (up to roughly 100K parameters) — the kind of
model used as a knowledge-distillation student rather than a full transformer.
It provides a fluent `NetworkBuilder`, the common activations (ReLU, Sigmoid,
Tanh, Softmax, LeakyReLU), basic backpropagation training with momentum, and
optional feature-gated batch inference (`parallel`), serialization (`serde`),
and GPU acceleration (`gpu`).

## Architecture

```text
NetworkBuilder --> Network --> [Layer, Layer, ...] --> output
                     |
                     +-- pre-allocated buffers (no alloc during inference)
```

`NetworkBuilder::build()` allocates every intermediate activation buffer once,
up front, sized from the layer dimensions. Each `Layer::forward` then writes
in place into its pre-allocated output buffer, and the top-level `Network::forward`
ping-pongs between two buffers via `split_at_mut()` to satisfy the borrow checker
without copying. This removes heap allocation from the inference hot path
entirely (see ADR-021 for the full rationale and the buffer-ping-pong mechanics).

## Training

`BackpropTrainer` implements gradient descent with momentum, driven by a
`TrainingConfig` (learning rate, max epochs, and related knobs). A typical
training loop:

```rust
use lattice_fann::{NetworkBuilder, Activation, BackpropTrainer, TrainingConfig, Trainer};

let mut network = NetworkBuilder::new()
    .input(2)
    .hidden(4, Activation::Tanh)
    .output(1, Activation::Tanh)
    .build()
    .unwrap();

// XOR training data
let inputs = vec![
    vec![0.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 0.0],
    vec![1.0, 1.0],
];
let targets = vec![
    vec![0.0],
    vec![1.0],
    vec![1.0],
    vec![0.0],
];

let mut trainer = BackpropTrainer::new();
let config = TrainingConfig::new()
    .learning_rate(0.5)
    .max_epochs(1000);

let result = trainer.train(&mut network, &inputs, &targets, &config);
```

`GradientGuardStrategy` controls how the trainer reacts to NaN/Inf gradients
during backprop (see ADR-022 for the guard strategies and their trade-offs).

## GPU backend (`gpu` feature)

```text
GpuContext (device/queue) ─┬─> BufferPool (3-tier, lifecycle tracking)
                           ├─> ShaderManager (compiled pipelines)
                           └─> CircuitBreaker (memory pressure)

GpuNetwork (inference) ──> GpuContext
GpuTrainer (training) ───> GpuContext
```

`GpuNetwork` wraps a CPU `Network` and keeps it as the fallback path. A static
`should_use_gpu(elements, batch_size)` predicate gates every dispatch, since GPU
kernel launch overhead dominates for the small networks this crate targets:

| Operation      | Threshold          | Rationale                                 |
| -------------- | ------------------ | ------------------------------------------ |
| Matrix-vector  | >10,000 elements   | GPU launch overhead dominates below this   |
| Batch matmul   | batch_size > 100   | Amortize kernel launch cost                |
| Activation     | >1,000 elements    | Element-wise is memory-bound               |
| Max dispatch   | 100,000 elements   | Stay under the Metal 2ms watchdog          |

Softmax always runs on CPU after reading the output buffer back from GPU — the
only softmax use in this crate is a final classification layer, so the
CPU post-processing pass is on the cold path, not the hot one.

### Apple Silicon tuning

- Buffers are aligned to 256 bytes (`apple_silicon::BUFFER_ALIGNMENT`).
- Buffers are capped at 128MB (`apple_silicon::MAX_BUFFER_SIZE`).
- Matmul shaders dispatch with a 32-lane workgroup (matches Apple Silicon's SIMD
  group width); other shaders use a 256-lane workgroup for throughput.
- Dispatches are kept under a 1.5ms headroom against the Metal 2ms watchdog
  (`thresholds::MAX_DISPATCH_TIME_MS`) — if a single layer would exceed it, the
  entire forward pass falls back to CPU rather than risking a GPU timeout.

### Buffer pool

`BufferPool` categorizes buffers by size to avoid the ~100-500us cost of a fresh
GPU allocation on every call:

| Tier   | Size range | Max pooled | Max age | Typical use            |
| ------ | ---------- | ---------- | ------- | ----------------------- |
| Small  | < 1MB      | 256        | 5 min   | Biases, small activations |
| Medium | 1-10MB     | 64         | 3 min   | Layer weights            |
| Large  | > 10MB     | 16         | 1 min   | Batch data, large layers |

Each pooled buffer tracks creation time, last-used time, and use count, so idle
or aged-out buffers can be evicted independently per tier.

GPU deallocation is asynchronous even though Rust's `Drop` runs synchronously.
A training loop that repeatedly allocates without releasing can hit an
out-of-memory condition despite buffers appearing freed on the Rust side.
Call `BufferPool::flush` (which releases every cached buffer immediately) and
then poll the device periodically — for example once per epoch:

```ignore
for epoch in 0..epochs {
    for batch in dataset.batches() {
        train_step(&mut network, batch);
    }
    ctx.buffer_pool().flush();
    ctx.poll(); // process pending GPU deallocations
}
```

### Circuit breaker

`CircuitBreaker` classifies memory pressure from a `usage / budget` ratio into
`MemoryPressure::{Low, Medium, High, Critical}` and exposes its own
`CircuitBreakerState::{Closed, Open, HalfOpen}` (the standard circuit-breaker
pattern: closed allows requests, open blocks them, half-open probes recovery).
`GpuContext::set_memory_pressure_handler` registers a callback invoked by
`notify_memory_pressure()` so callers can react — e.g. trigger aggressive
buffer eviction — before the breaker opens and forces CPU fallback.

See ADR-025 for the full alternatives-considered analysis (CUDA, native Metal,
candle/burn) and the known limitations (the buffer pool is not yet wired into
`GpuNetwork::forward_gpu`'s own buffer creation, and softmax in hidden layers
is not GPU-accelerated).
