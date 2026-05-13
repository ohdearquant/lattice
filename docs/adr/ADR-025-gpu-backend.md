# ADR-025: GPU Backend

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-fann

## Context

Training classification student networks via knowledge distillation can benefit from GPU acceleration, particularly for matrix-vector multiplications in the forward pass. However, `lattice-fann` targets small dense networks (hundreds to thousands of parameters) where GPU kernel launch overhead can exceed the computation itself. The design must provide GPU acceleration only when it is actually beneficial, degrade gracefully to CPU when hardware is unavailable or the network is too small, and handle Apple Silicon (Metal via wgpu) without special-casing in calling code.

The GPU backend is gated behind the `gpu` feature flag.

## Decision

A `GpuNetwork` struct wrapping a `Network` (the CPU network is kept as-is for fallback). GPU inference is async via wgpu. A static `should_use_gpu(elements, batch_size)` predicate gates all dispatch. The context (`GpuContext`) holds the wgpu device, queue, a 3-tier buffer pool, and a `ShaderManager` with pipeline caching.

### Key Design Choices

#### CPU/GPU Decision Heuristics

All thresholds are constants in `gpu::thresholds`:

| Operation     | Threshold        | Rationale                                |
| ------------- | ---------------- | ---------------------------------------- |
| Matrix-vector | >10,000 elements | GPU launch overhead dominates below this |
| Batch matmul  | batch_size > 100 | Amortize kernel launch cost              |
| Activation    | >1,000 elements  | Element-wise is memory-bound             |
| Max dispatch  | 100,000 elements | Stay under Metal 2ms watchdog            |

`GpuNetwork::new` evaluates `should_use_gpu(total_params, 1)` at construction time. If the network is too small, `use_gpu = false` and all `forward()` calls delegate to the CPU network immediately, incurring no GPU overhead.

The watchdog check (`elements > MAX_ELEMENTS_PER_DISPATCH`) inside `forward_gpu` provides a second safety net: if a single layer would exceed the Metal 2ms dispatch limit, the entire forward pass falls back to CPU rather than risking a GPU timeout.

#### Shader Types and Compilation

The `ShaderType` enum covers 13 WGSL compute shaders:

- **Inference**: `MatrixVectorMultiply`, `MatrixVectorMultiplyRelu` (fused), `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`
- **Softmax**: `SoftmaxMax`, `SoftmaxExpSum`, `SoftmaxNorm` (3-pass for numerical stability)
- **Optimizers**: `SgdMomentum`, `Adam`, `AdamW`

`ShaderManager` compiles pipelines lazily via `get_or_compile(shader_type)` using a `RwLock<HashMap<ShaderType, Arc<ComputePipeline>>>`. Cache hits take a read lock; misses acquire a write lock and call `wgpu::Device::create_compute_pipeline`. Compile time is tracked in `CacheStats`.

On context creation, `ShaderManager::with_warmup` pre-compiles the five most common inference shaders (`MatrixVectorMultiply`, `MatrixVectorMultiplyRelu`, `ReLU`, `Sigmoid`, `Tanh`) to avoid first-request latency.

#### Fused Matmul+ReLU

When a layer's activation is `Activation::ReLU`, `forward_gpu` selects `ShaderType::MatrixVectorMultiplyRelu` instead of the two-shader path. This saves one GPU dispatch and eliminates the intermediate buffer round-trip for the ReLU pass. Non-ReLU activations use the separate activation shader after matmul.

#### Softmax on CPU

Softmax is applied on CPU after reading the output buffer back from GPU. The comment in `apply_activation` notes the known limitation: `// Softmax is only applied to the final output layer on CPU. Non-final softmax layers produce incorrect GPU results. FIXME(FANN-M5): wire per-layer CPU softmax fallback.`

This is intentional for the current scope: the only use of softmax in `lattice-fann` is the final classification layer of student networks, so the CPU post-processing path is correct.

#### Apple Silicon Specifics

Constants in `gpu::apple_silicon`:

- `BUFFER_ALIGNMENT = 256` bytes
- `MAX_BUFFER_SIZE = 128 MB`
- `WORKGROUP_SIZE = 32` (matches 32-lane SIMD group)
- `ACTIVATION_WORKGROUP_SIZE = 256` (maximize throughput for element-wise ops)

`GpuContext::is_apple_silicon()` checks `backend == Metal && name.contains("Apple"|"M1"|"M2"|"M3")`. `optimal_workgroup_size` returns 32 for matmul and 256 for other ops on Apple Silicon.

Matmul shaders use `[32, 1, 1]` workgroups; all other shaders use `[256, 1, 1]`. This is hard-coded in `ShaderType::workgroup_size()`.

#### Buffer Pool and Circuit Breaker

`BufferPool` (3-tier: Small/Medium/Large) pools GPU buffers by size category to avoid repeated allocation. `flush_memory()` on `GpuContext` calls `buffer_pool.flush()` then `device.poll(Maintain::Wait)` — intended to be called periodically during long training loops to prevent OOM from async VRAM deallocation lag.

`CircuitBreaker` tracks memory pressure as a ratio of `usage / budget`. Four states: `Normal`, `Moderate`, `High`, `Critical`. `GpuContext::set_memory_pressure_handler` registers a callback invoked by `notify_memory_pressure()`.

#### Weight Synchronization

After CPU-side training updates weights, `GpuNetwork::sync_weights()` calls `queue.write_buffer` to push the new values to the GPU weight buffers. This must be called explicitly by the training loop before running GPU inference on updated weights.

### Alternatives Considered

| Alternative                      | Pros                             | Cons                                                                                              | Why Not                                                          |
| -------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| CUDA via cudarc                  | Maximum GPU compute performance  | CUDA-only (no Apple Silicon), large dependency, cross-platform complexity                         | Apple Silicon is primary development target                      |
| Metal-rs (direct Metal)          | Native Apple Silicon performance | macOS-only, no Linux/Windows path for testing                                                     | wgpu provides portability across Metal, Vulkan, DX12             |
| candle / burn (ML framework GPU) | Higher-level, proven ML ops      | Heavy dependency, brings its own tensor type incompatible with `lattice-fann`'s zero-alloc design | ADR-001 mandates standalone crate with no external ML framework  |
| Always-CPU fallback (no GPU)     | Simplest implementation          | Training on large datasets is slow for distillation experiments                                   | GPU provides 10-100x speedup for batch training beyond threshold |

## Consequences

### Positive

- Single async API (`forward(&input).await`) works identically with GPU or CPU backend
- `forward_sync` wraps via `pollster::block_on` for non-async callers
- Weight buffers persist on GPU across forward calls — no per-call re-upload
- `sync_weights()` allows training loop to update CPU weights then mirror to GPU without re-creating the `GpuNetwork`

### Negative

- Softmax-in-hidden-layers is not GPU-accelerated (FIXME(FANN-M5))
- GPU tests are `#[ignore]` by default (require `--features gpu-tests` and physical GPU hardware)
- Per-layer buffer creation in `forward_gpu` allocates a new output buffer on every forward call — the buffer pool is not yet wired into the `GpuNetwork` forward path (the pool is present in `GpuContext` but `GpuNetwork::forward_gpu` uses `device.create_buffer` directly)

### Risks

- The fallback to CPU inside `forward_gpu` (watchdog check) returns from the entire forward pass using the CPU network, losing any GPU work done on previous layers. This is correct but potentially surprising — partial GPU acceleration is not attempted.
- The Metal 2ms watchdog is empirically set; different GPU generations may have different thresholds. `MAX_DISPATCH_TIME_MS = 1.5` is a conservative headroom.

## References

- `crates/fann/src/gpu/mod.rs` — thresholds, apple_silicon constants, `should_use_gpu`, `is_gpu_available`
- `crates/fann/src/gpu/network.rs` — `GpuNetwork`, `forward_gpu`, fused ReLU selection, softmax CPU post-processing
- `crates/fann/src/gpu/context.rs` — `GpuContext`, memory pressure, `flush_memory`
- `crates/fann/src/gpu/shader_manager.rs` — `ShaderManager`, `ShaderType`, pipeline caching, warmup
- `crates/fann/src/gpu/circuit_breaker.rs` — `CircuitBreaker`, `MemoryPressure` levels
- `crates/fann/src/gpu/buffer.rs` — `BufferPool`, 3-tier buffer management
