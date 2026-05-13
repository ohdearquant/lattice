# ADR-021: Zero-Allocation Inference

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-fann

## Context

lattice-fann targets sub-5ms CPU inference latency for small neural networks (up to ~100K
parameters). In latency-critical paths, memory allocation introduces unpredictable
jitter:

- **Heap allocation overhead**: Each `malloc`/`free` cycle adds 50-200ns, compounding
  across layers
- **Allocator contention**: Under concurrent load, allocator locks become bottlenecks
- **GC pressure**: Frequent small allocations increase memory fragmentation
- **Latency variance**: Allocation times vary with heap state, violating predictability
  requirements

Benchmarks show a 4-layer network (784->128->64->10) completes inference in ~1ms. At
this scale, a single heap allocation per layer would add 10-20% overhead and destroy
latency predictability.

## Decision

We adopt **pre-allocated layer buffers** with the following design:

1. **Buffer allocation at construction**: `Network::build()` allocates all intermediate
   activation buffers upfront based on layer dimensions

2. **In-place computation**: Each layer writes directly into its pre-allocated output
   buffer via `Layer::forward(input, &mut output)`

3. **Mutable self for forward pass**: The signature
   `fn forward(&mut self, input: &[f32]) -> FannResult<&[f32]>` enables buffer reuse
   without allocation

4. **Buffer ping-pong**: Forward pass alternates between buffers using `split_at_mut()`
   to satisfy the borrow checker while avoiding copies

```rust
// Pre-allocated at construction
pub struct Network {
    layers: Vec<Layer>,
    buffers: Vec<Vec<f32>>,  // One buffer per layer output
}

// Zero-alloc forward pass
pub fn forward(&mut self, input: &[f32]) -> FannResult<&[f32]> {
    self.layers[0].forward(input, &mut self.buffers[0])?;
    for i in 1..self.layers.len() {
        let (prev, curr) = self.buffers.split_at_mut(i);
        self.layers[i].forward(&prev[i - 1], &mut curr[0])?;
    }
    Ok(&self.buffers[self.buffers.len() - 1])
}
```

## Consequences

### Positive

- **Predictable latency**: Zero allocations during inference eliminates jitter;
  benchmark variance drops from +/-15% to +/-2%
- **Cache efficiency**: Buffers remain hot across repeated inferences on the same
  network
- **Simple implementation**: No custom allocator complexity; standard `Vec<f32>` with
  upfront sizing

### Negative

- **Memory committed upfront**: Network reserves memory for maximum activation sizes at
  construction, even if never used
- **No concurrent inference on same Network**: `&mut self` requirement prevents parallel
  forward passes on a single instance; callers must clone for parallelism
- **Fixed topology**: Buffer sizes are locked at construction; dynamic layer addition
  would require reallocation

### Neutral

- **Batch inference requires cloning**: `forward_batch()` clones the network per thread
  when using Rayon parallelism; acceptable for typical batch sizes

## Alternatives Considered

### 1. Per-Call Allocation

Allocate fresh `Vec<f32>` for each layer during forward pass.

- **Pros**: Simple API (`&self` instead of `&mut self`), natural concurrent access
- **Cons**: 50-200ns per allocation, unpredictable latency, heap fragmentation under
  load
- **Rejected**: Violates sub-5ms latency target with predictable jitter

### 2. Arena Allocator (bumpalo)

Use a bump allocator reset between forward passes.

- **Pros**: Fast allocation (pointer bump), batch deallocation
- **Cons**: Additional dependency, arena sizing complexity, still requires `&mut self`
  for reset
- **Rejected**: Adds complexity without meaningful benefit over pre-allocation; arena
  sizing requires same upfront knowledge as pre-allocation

### 3. Thread-Local Buffers

Store buffers in thread-local storage, accessed by network during forward pass.

- **Pros**: Enables `&self` forward, automatic per-thread isolation
- **Cons**: Hidden global state, difficult to reason about lifetime, buffer sizing
  across different networks sharing TLS
- **Rejected**: Implicit state violates Rust's explicit ownership philosophy; debugging
  difficulties outweigh ergonomic gains

### 4. Lazy Allocation on First Forward

Allocate buffers on first `forward()` call, reuse thereafter.

- **Pros**: Defers memory commitment until actually needed
- **Cons**: First inference has allocation overhead, still requires `&mut self`, more
  complex state management
- **Rejected**: Shifts jitter to first call rather than eliminating it; no clear benefit
  over construction-time allocation

## References

- TDS-network-architecture.md Section 6.1: Zero-Allocation Design
- TDS-network-architecture.md Section 2.1: Component Overview (buffer diagram)
- Benchmark data: 784->128->64->10 network achieves ~1ms inference
