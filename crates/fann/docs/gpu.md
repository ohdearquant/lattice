# GPU backend

`lattice-fann`'s GPU backend executes dense feed-forward inference through `wgpu` compute
pipelines. It is an acceleration path, not a separate network representation: `GpuNetwork`
keeps an owned CPU `Network` for fallback while holding a GPU upload of each layer's weights
and biases.

This reference describes the implementation in `src/gpu/`, its dispatch policy, and the
operational limits a caller must respect.

## Backend at a glance

```text
GpuContext
  ├── wgpu Device + Queue
  ├── BufferPool ──> CircuitBreaker
  └── ShaderManager ──> cached ComputePipelines

GpuNetwork
  ├── owned CPU Network ──> CPU fallback
  └── GPU layer uploads ──> matmul / activation dispatches ──> staging-buffer readback
```

The backend has four cooperating parts:

| Component | Responsibility |
| --- | --- |
| `GpuContext` | Selects an adapter, creates the `wgpu` device and queue, records device metadata, owns the pool and pipeline cache, and exposes synchronization and pressure APIs. |
| `BufferPool` | Reuses compatible buffers in small, medium, and large tiers, tracks pool-managed bytes, and sheds cached buffers under pressure. |
| `CircuitBreaker` | Stops repeated allocation attempts from cascading after recorded failures, then permits a recovery probe after a cooldown. |
| `GpuNetwork` | Uploads an ordinary `Network`, chooses CPU or GPU inference, submits per-layer work, and reads the final vector back to the CPU. |

The `shaders` module supplies the WGSL source used by `ShaderManager`. `ShaderType` is the
closed set of operations exposed to the current inference path.

## Availability and context creation

`is_gpu_available()` performs a blocking adapter request with the default `wgpu` instance. It
requests a high-performance adapter, has no presentation surface, and does not force a fallback
adapter. It answers only whether an adapter was obtained; it does not create a device or compile
pipelines.

`get_gpu_info()` repeats adapter selection and returns the adapter name, vendor, device type,
backend, maximum buffer size, and maximum compute-workgroup size in the X dimension. Use it for
diagnostics before constructing a context, not as a promise that later device creation will
succeed.

`GpuContext::new()` is asynchronous. It:

1. selects the same high-performance, headless adapter;
2. captures adapter information and reported limits;
3. requests a device with no required optional features, default `wgpu` limits, and performance
   memory hints;
4. creates a `BufferPool` for that device; and
5. creates a `ShaderManager` and warms its common inference pipelines.

`GpuContext::new_blocking()` is the synchronous wrapper around that procedure. Context creation
can return `GpuError::NotAvailable` when no adapter is found or `GpuError::DeviceCreation` when
the device request fails. A caller that wants a CPU-only experience must choose that fallback
when context creation fails; constructing `GpuNetwork` itself requires an existing context.

The context owns `Arc`s for its `Device` and `Queue`, so its pool and manager share the same
device. `info()` exposes `GpuDeviceInfo`, while `device()`, `queue()`, `buffer_pool()`, and
`shader_manager()` expose the corresponding runtime components.

### Polling, waiting, and pool flushing

`poll()` asks `wgpu` to make progress without waiting. `wait()` blocks until pending GPU work
completes. `flush_memory()` first drains the buffer pool and then calls `wait()`, returning the
number of pool bytes it dropped.

This ordering matters. Rust dropping a `wgpu::Buffer` removes the host object, but physical GPU
deallocation can lag while queued work still references resources. A loop can therefore see
buffers disappear at the Rust level while the device retains enough memory to reject the next
allocation. For long-running work, flush the pool and poll or wait at a controlled boundary such
as an epoch boundary; a pool flush alone is not a synchronous release.

`memory_usage()` is pool accounting, not a device-wide memory meter. It includes allocations made
through `BufferPool` until the pool drops them, including checked-out buffers, but it excludes
buffers allocated directly with `wgpu`. In particular, the current `GpuNetwork` creates its
input, output, uniform, and staging buffers directly, so those allocations are neither pooled
nor included in this counter.

## Dispatch policy and platform limits

The module exports the following policy constants. They are deliberately visible so callers can
make the same choices as the built-in wrapper.

| Constant | Value | Current use |
| --- | ---: | --- |
| `MATMUL_MIN_ELEMENTS` | 10,000 | A one-input network uses GPU only at or above this parameter count. |
| `BATCH_MIN_SIZE` | 100 | Part of the batch heuristic: for `batch_size > 1`, `elements * batch_size` must be at least `100 * 100` (10,000). |
| `ACTIVATION_MIN_ELEMENTS` | 1,000 | Published activation crossover policy; the current `GpuNetwork` does not independently gate activation dispatches with it. |
| `MAX_ELEMENTS_PER_DISPATCH` | 100,000 | A layer with more elements than this causes the whole `GpuNetwork` forward pass to restart on CPU. Exactly 100,000 is allowed. |
| `MAX_DISPATCH_TIME_MS` | 1.5 ms | Target headroom for the element-count cap; it is not measured dynamically by the current code. |

`should_use_gpu(elements, batch_size)` uses a simple threshold rather than a runtime benchmark:

```text
batch_size == 1:  elements >= 10,000
batch_size > 1:   elements * batch_size >= 10,000
```

The GPU wrapper calls this once at construction using `network.total_params()` and a batch size
of one. It therefore does not reevaluate the decision per forward pass, and it currently exposes
no batched-forward API. A network below the single-input threshold still uploads its layer data
when constructed, but `forward` takes the CPU route.

### Metal and Apple Silicon

Metal has an approximately 2 ms watchdog limit for a dispatch. The backend reserves a 1.5 ms
target to leave 0.5 ms of headroom for variance. It avoids the watchdog by declining GPU execution
when a layer's `num_inputs * num_outputs` is greater than 100,000; it does not tile oversized
layers. That condition falls back to `forward_cpu(input)` for the entire network, rather than
mixing GPU and CPU layers.

The Apple-Silicon tuning constants are:

| Constant | Value | Meaning |
| --- | ---: | --- |
| `BUFFER_ALIGNMENT` | 256 bytes | Buffer-pool requests are rounded up to this boundary. |
| `MAX_BUFFER_SIZE` | 128 MiB | Published platform limit; the pool and network do not currently enforce it themselves. |
| `WORKGROUP_SIZE` | 32 | Matmul workgroup width, matching Apple Silicon's 32-lane SIMD width. |
| `ACTIVATION_WORKGROUP_SIZE` | 256 | Element-wise workgroup width chosen for throughput. |

`GpuContext::is_apple_silicon()` recognizes a Metal backend whose name includes `Apple`, `M1`,
`M2`, or `M3`. `optimal_workgroup_size("matmul")` reports 32 for those devices and 64 elsewhere;
it reports 256 for other operations on both. This is advisory today: the WGSL shaders embed their
own workgroup sizes, and `GpuNetwork` dispatches using `ShaderType::workgroup_size()`.

## Buffer-pool design

Creating a new GPU allocation can cost roughly 100–500 microseconds, so the pool keeps returned
buffers for a compatible future request. It has three categories based on the **aligned** size:

| Category | Size range | Maximum retained | Maximum age |
| --- | --- | ---: | ---: |
| `Small` | less than 1 MiB | 256 | 300 seconds |
| `Medium` | 1 MiB through less than 10 MiB | 64 | 180 seconds |
| `Large` | 10 MiB or more | 16 | 60 seconds |

### Allocation and reuse

`BufferPool::allocate(size, usage, label)` performs these steps:

1. asks its `CircuitBreaker` whether a request is allowed;
2. rounds the requested size up to a 256-byte boundary;
3. classifies that aligned size into a tier;
4. looks for a returned buffer in that tier with identical `BufferUsages`, a size at least as large
   as requested, and a size no more than twice the request; and
5. either returns the reused buffer or creates a new `wgpu::Buffer` of the aligned size.

The same-usage requirement prevents a buffer from being reused with an incompatible binding or
copy capability. The two-times cap prevents a small request from monopolizing a much larger
allocation. `align_size` uses bit rounding, so the alignment must remain a power of two; the
current 256-byte constant satisfies that requirement.

Each `GpuBuffer` records its size, usage, category, creation time, last-use time, use count, and
an allocation identifier intended for tracing. `mark_used()` updates last use and increments the
counter. A buffer is retainable only while younger than its tier's age limit and when it was used
within the last 60 seconds **or** its reuse rate exceeds one use per hour.

### Returning, evicting, and cleaning

`return_buffer` immediately drops a buffer that is no longer retainable and subtracts its size
from pool accounting. Otherwise it appends the buffer to its category. If that category is at its
configured capacity, it evicts the retained buffer with the lowest use count before inserting the
new one. The implementation's selection is use-count based even though an adjacent code comment
uses the word “oldest”.

`cleanup(pressure)` shrinks every tier to a pressure-dependent target. It sorts the tier by use
count and drops the least-reused buffers first. The target is the floored product
`(1 - cleanup_aggressiveness) * tier_capacity`:

| Pressure | Cleanup aggressiveness | Retained target |
| --- | ---: | ---: |
| `None` | 0.1 | 90% of capacity |
| `Low` | 0.3 | 70% of capacity |
| `Medium` | 0.5 | 50% of capacity |
| `High` | 0.7 | 30% of capacity |
| `Critical` | 1.0 | 0 buffers |

`flush()` drains all categories; `flush_category(category)` drains only one. Both report bytes
dropped and update evictions and current-memory accounting. `pooled_count()` reports the number
of currently returned buffers, while `stats()` supplies allocation, hit, miss, eviction, and
current-byte atomics. `PoolStats::hit_rate()` returns zero when no lookup has occurred.

The pool exposes `record_failure()` and `record_success()` but does not call them automatically
inside `allocate`. Integrations that identify allocation success or failure are responsible for
recording those outcomes if they want circuit-breaker state to reflect them.

## Memory pressure and circuit breaking

`GpuContext` can maintain a caller-provided memory budget. Once set, it maps
`memory_usage() / budget` to `MemoryPressure`:

| Usage ratio | Level | Intended response |
| --- | --- | --- |
| below 60% | `None` | Normal operation; light cleanup still retains 90% of each tier. |
| 60% to below 70% | `Low` | Begin monitoring and reduce the cache target. |
| 70% to below 80% | `Medium` | Reduce allocations and retain half of each tier. |
| 80% to below 90% | `High` | Clean aggressively. |
| 90% or more | `Critical` | Signal that callers should block new allocations and remove all cached pool buffers. |

`check_memory_pressure()` returns `None` until a budget is configured. The context does not
validate the budget or automatically invoke cleanup or allocation blocking. Call
`notify_memory_pressure()` after significant allocations to calculate the current level and
invoke an installed handler with the level and pool-accounted byte count. Notification is not
edge-triggered: each explicit call can invoke the handler even if the level did not change.

An application can use the handler to reduce its own batch size, flush pooled buffers, or change
to CPU work:

```rust,ignore
context.set_memory_budget(512 * 1024 * 1024);
context.set_memory_pressure_handler(|level, bytes| {
    if matches!(level, MemoryPressureLevel::High | MemoryPressureLevel::Critical) {
        eprintln!("GPU pool uses {} MiB at {level:?}", bytes / 1024 / 1024);
    }
});
```

The circuit breaker defaults to five recorded failures and a 30-second recovery timeout in a
new buffer pool. Its state machine is:

```text
Closed -- failures reach threshold --> Open
  ^                                  |
  |                                  | recovery timeout elapses
  |                                  v
  +--------- success ----------- HalfOpen
                                      |
                                      | failure
                                      v
                                    Open
```

- In `Closed`, requests are allowed. The breaker opens when the cumulative failure count reaches
  its configured threshold.
- In `Open`, requests are denied until the recovery timeout has elapsed. The next
  `allow_request()` moves the breaker to `HalfOpen` and allows the recovery probe.
- In `HalfOpen`, a recorded success closes the circuit and resets the failure count; a recorded
  failure immediately reopens it. Although the state represents recovery testing, the current
  implementation allows every `allow_request()` call while it remains half-open; it does not
  enforce a single-probe limit.

Lock poisoning is treated conservatively in the request path: `allow_request()` and `state()`
fail closed, reporting or behaving as `Open`. Statistics and reset paths instead make a
best-effort update. `reset()` returns the breaker to `Closed`, clears both counters and the last
failure time, and refreshes the state-change timestamp.

## Shader inventory and pipeline caching

`ShaderManager` maps a `ShaderType` to WGSL source, a debug label, and a fixed workgroup size.

| Shader type | Operation | Workgroup |
| --- | --- | ---: |
| `MatrixVectorMultiply` | `y = W x + b` | 32 × 1 × 1 |
| `MatrixVectorMultiplyRelu` | `max(0, W x + b)` in one kernel | 32 × 1 × 1 |
| `ReLU` | In-place `max(0, x)` | 256 × 1 × 1 |
| `LeakyReLU` | In-place `x` when positive, otherwise `alpha * x` | 256 × 1 × 1 |
| `Sigmoid` | In-place logistic function | 256 × 1 × 1 |
| `Tanh` | In-place hyperbolic tangent | 256 × 1 × 1 |

Context construction warms `MatrixVectorMultiply`, `MatrixVectorMultiplyRelu`, `ReLU`,
`Sigmoid`, and `Tanh`. `LeakyReLU` remains lazy and is compiled on its first request. A cache hit
returns a cloned `Arc<ComputePipeline>`. A miss compiles the pipeline, inserts it under its
`ShaderType`, and records the elapsed compile time. `CacheStats` provides hit rate, compilation
count, and average compile time; both rate helpers return zero with no observations.

The cache releases the read lock before compiling and takes a write lock only to insert. This
keeps the long compilation step out of the write-locked region, but concurrent first requests for
the same shader can compile it more than once before one insertion replaces the other. The cache
does not currently use a double-check or single-flight mechanism.

`clear_cache()` drops cached pipeline references, while `cached_count()` counts the present
entries. A poisoned cache lock becomes `GpuError::LockPoisoned` for `get_or_compile`; the
best-effort statistics accessor returns default statistics if its read lock is poisoned.

### Kernel behavior

The matmul shader assigns one global invocation to one output row. For row `r`, it calculates:

```text
output[r] = bias[r] + sum(weights[r * cols + c] * input[c], c = 0..cols)
```

It handles the first multiple-of-four columns through four explicit multiply-add statements per
loop iteration, then handles the tail one column at a time. The guard `row >= rows` ensures the
ceil-rounded dispatch does not write beyond the output vector. The fused variant performs the
same computation and writes `max(0, sum)`.

Element-wise shaders guard `index >= size` for the same reason. Sigmoid clamps inputs to
`[-10, 10]` before `exp`; this prevents the `f32` exponential overflow associated with very large
positive values but deliberately makes the extreme tails behave as if they were at the clamp
boundary. Tanh clamps inputs to `[-5, 5]` before evaluating `tanh`, reflecting its faster
saturation. These are part of the GPU numerical behavior.

## `GpuNetwork` execution

`GpuNetwork::new(context, network)` owns the supplied CPU network and immediately uploads every
layer's weight and bias slices into storage buffers. It also computes `use_gpu` once from the
network's total parameter count and the single-input threshold. The CPU network remains available
through `cpu_network()` and is the implementation of all CPU fallbacks.

The upload occurs even when `is_using_gpu()` will be false. Construct a plain `Network` instead
when a context and these unused GPU uploads are not wanted.

### Forward pass

`forward(input)` is asynchronous and first validates that `input.len()` exactly matches the CPU
network's input width. It returns `GpuError::InvalidDimensions` on a mismatch. `forward_sync()`
blocks on the same method.

If the constructor chose CPU, `forward` delegates directly to the owned CPU network. Otherwise it
runs this sequence:

1. Upload the input to a fresh storage/copy-source buffer.
2. For each layer, check `num_inputs * num_outputs`. If it exceeds 100,000, abandon the GPU path
   and recompute the complete network from the original input on CPU.
3. Choose `MatrixVectorMultiplyRelu` for a ReLU layer; all other activations use
   `MatrixVectorMultiply` first.
4. Allocate a fresh output buffer and a 16-byte uniform buffer containing the output-row and
   input-column counts. Bind uniforms, weights, the current vector, biases, and output at the
   shader's five bindings.
5. Submit `ceil(num_outputs / workgroup_size)` workgroups. The matrix shaders use 32 threads per
   workgroup.
6. For Leaky ReLU, sigmoid, or tanh, submit a second in-place element-wise dispatch with its own
   uniform buffer. Linear needs no activation dispatch. Queue submission order keeps this second
   dispatch after its matrix result.
7. Treat the output as the next layer's input, then copy the final vector into a fresh map-readable
   staging buffer and wait for mapping to complete.

The final mapped bytes are copied into a `Vec<f32>` and returned. The staging buffer is temporary;
the final result is CPU-owned.

### Activation support and softmax limitation

ReLU is the only fused activation, saving one compute submission. Leaky ReLU carries its `alpha`
parameter in the activation uniform. Sigmoid and tanh use the clamped numerical behavior described
above. Linear is a no-op.

Softmax is not a WGSL activation in this backend. If the **last** layer declares softmax,
`read_buffer` computes a stable CPU softmax after readback by subtracting the maximum logit,
exponentiating, summing, and normalizing. This costs a device-to-host round trip but avoids a GPU
reduction kernel.

A non-final softmax is unsupported: the GPU activation stage performs no softmax for it, so the
following GPU layer receives the preceding layer's unnormalized output. Do not use an architecture
with hidden softmax layers on this GPU path. Use the CPU network instead until per-layer softmax
fallback or a GPU softmax kernel is implemented.

### Weight synchronization and flushing

`sync_weights()` overwrites each uploaded weight and bias buffer with the current values in the
owned CPU network. Call it after modifying those values through whatever owns or updates the
network; creating `GpuNetwork` does not establish automatic weight synchronization.

`GpuNetwork::flush()` delegates to `GpuContext::flush_memory()`. It drains only the context's
buffer pool and waits for queued work. It does not retroactively pool the network's direct
working buffers, but the wait is still the important synchronization boundary when managing
asynchronous GPU destruction.

## Errors and fallback decisions

`GpuResult<T>` is `Result<T, GpuError>`. The error taxonomy separates unavailable hardware,
device creation, invalid dimensions, pool and allocation pressure, shader or pipeline failures,
execution failures, mapping failures, network failures, watchdog risk, and poisoned internal
locks.

`GpuError::is_recoverable()` returns true for `MemoryPressure`, `PoolExhausted`, and `Execution`.
`GpuError::should_fallback_to_cpu()` returns true for `NotAvailable`, `DeviceCreation`,
`MemoryPressure`, and `WatchdogRisk`. These predicates are guidance for an outer scheduling
policy; the current `GpuNetwork` automatically falls back only for its size decision and an
oversized-layer watchdog guard. It propagates other GPU errors to its caller.

## Operational guidance

Use the GPU path where each dense inference is substantial enough to amortize upload, dispatch,
and readback overhead. The thresholds are deliberately conservative and should not be interpreted
as a universal benchmark result for every adapter or architecture.

For sustained workloads:

1. Construct one context and reuse it, so warmed pipelines and the pool can pay back their setup
   cost.
2. Configure a realistic pool budget if the process needs pressure visibility, and explicitly
   notify the context after significant pool activity.
3. Return eligible `GpuBuffer`s to the pool when using the pool API; otherwise no reuse occurs.
4. At safe workload boundaries, flush pooled buffers and wait or poll so asynchronous releases
   reach the device before the next allocation burst.
5. Keep GPU layers at or below the dispatch cap. Split or redesign larger work outside this
   backend rather than relying on it to tile the layer.
6. Restrict GPU network activations to linear, ReLU, Leaky ReLU, sigmoid, tanh, and final-layer
   softmax. Route hidden-softmax networks through CPU.

For diagnostics, inspect `GpuContext::info()`, `BufferPool::stats()`, pipeline `CacheStats`, and
the circuit-breaker state and counters. A low pool hit rate can mean the workload's allocation
sizes or usage flags are too variable for reuse; it is not evidence that direct allocations made
by `GpuNetwork` are being pooled.
