# ADR-003: SafeTensors Weight Loading

**Status**: Accepted\
**Date**: 2026-05-13\
**Crate**: `lattice-inference`

---

## Context

Loading model weights is the dominant startup-latency operation for inference servers. For Qwen3-Embedding-0.6B (28 layers, 1024 hidden size), the checkpoint is distributed across multiple safetensors shards. The design must address:

1. **Parse cost**: The safetensors format uses a JSON header to describe tensor locations within the binary blob. Parsing this header with a full JSON library (serde_json) would pull in a significant dependency chain.
2. **Copy cost**: F32 weights on a little-endian host can be used in-place from the memory-mapped file if alignment permits. Copying them defeats the point of safetensors.
3. **Type conversion**: F16 and BF16 tensors cannot be used in-place by the f32 compute kernels. They must be converted and cached.
4. **Sharding**: Multi-shard checkpoints must be served transparently, opening each shard lazily to avoid holding all file descriptors simultaneously.

Relevant implementation: `src/weights/f32_weights.rs`.

The file contains a hand-written `JsonParser` struct (no serde dependency), `memmap2::Mmap` for the binary data, `TensorMeta` with an `OnceLock<Box<[f32]>>` for lazy conversion cache, and `ShardedSafetensors` for multi-file checkpoints.

Key code paths:

```rust
// Zero-copy F32 path
fn bytes_to_f32_slice(bytes: &[u8]) -> Option<&[f32]> {
    // Returns None if misaligned; caller falls back to copy
}

// F16 conversion (behind `f16` feature flag)
fn convert_f16_bytes_to_f32(bytes: &[u8]) -> Box<[f32]>

// Lazy conversion cache
struct TensorMeta {
    // ...
    cache: OnceLock<Box<[f32]>>,
}

// Sharded access — opens shard file on first access
struct ShardedSafetensors { /* lazy-open per-shard */ }
```

For Qwen3, weights are pre-fused at load time:

- `fused_qkv`: Q, K, V projection rows concatenated into a single `Vec<f32>` → one GEMM replaces three
- `fused_gate_up`: gate and up projection rows concatenated → one GEMM replaces two

The `ShardedQwenBacking` struct holds these owned `Vec<f32>` allocations alongside the `ShardedSafetensors`; the `QwenModel` holds both in a `SafetensorsStorage` enum, and Rust's RFC 1857 struct field drop order guarantees that tensor slices (which borrow from `ShardedQwenBacking`) are dropped before the backing allocation.

---

## Decision

Use **`memmap2::Mmap` with a hand-written JSON header parser** for zero-copy weight access. F32 weights on LE-aligned hardware bypass deserialization entirely. F16/BF16 conversions are deferred to first access and cached in a `OnceLock<Box<[f32]>>` per tensor. Shards are opened lazily on first tensor access. Qwen3 weights are fused (QKV, gate+up) eagerly at load time into owned `Vec<f32>` to collapse multiple GEMM calls.

---

## Key Design Choices

1. **Hand-written JSON parser**: The safetensors header is a constrained subset of JSON (no nested objects beyond the header map, no arrays of arbitrary depth). A custom parser avoids the `serde`, `serde_json`, and `serde_derive` dependency chain while being faster to compile and smaller in binary size.
2. **`OnceLock` conversion cache**: On first access to an F16/BF16 tensor, `get_or_init` performs the conversion and stores it. Subsequent accesses return the cached `Box<[f32]>` with no synchronization overhead (post-init reads are lock-free).
3. **Zero-copy F32**: `bytes_to_f32_slice` returns `None` on misalignment and the caller falls back to a copy. In practice, safetensors aligns tensors to 8 bytes, which satisfies f32's 4-byte alignment requirement on all target platforms.
4. **Eager QKV fusion**: Fusing at load time rather than inference time means the fusion cost is paid once and inference hot paths see a single contiguous weight matrix. The cost is ~30% higher peak memory during the load window (both original and fused forms exist briefly).
5. **`unsafe` 'static lifetime extension**: `ShardedQwenBacking` is heap-allocated via `Box` and co-located with `QwenModel`. The tensor slices that borrow from it are given `'static` lifetime via `mem::transmute`. Safety is maintained by RFC 1857 drop ordering (fields drop in declaration order; the backing box is declared before the tensor slices). This is documented as a known unsafe pattern (57.5K LOC crate note in `src/lib.rs`).

---

## Alternatives Considered

| Alternative                               | Pros                                | Cons                                                                                               | Why Not                                                               |
| ----------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `serde_json` for header parsing           | No custom code; handles edge cases  | Adds ~400KB to binary; slower compile; overkill for this constrained format                        | Dependency budget; safetensors header JSON is well-constrained        |
| `candle` weight loading primitives        | Battle-tested; type-safe tensor API | Pulls in the entire `candle` ecosystem; incompatible with the crate's no-ONNX/no-Python constraint | Defeats the purpose of a pure Rust inference engine                   |
| Copy all weights at load time             | Simpler memory model; no `unsafe`   | 1–4 GB of unnecessary allocations; slower startup; more GC pressure                                | Latency and memory budget                                             |
| Load shards eagerly                       | Simpler code                        | All shard file descriptors held open simultaneously; larger virtual memory footprint               | Resource usage                                                        |
| Lazy QKV fusion (fuse at first GEMM call) | Lower peak memory at load time      | Fusion must be re-done or cached; complicates hot path                                             | Inference is called millions of times; amortize the cost once at load |

---

## Consequences

**Positive**:

- F32 model startup is near-zero-copy: the OS maps pages on demand.
- No serde dependency in the inference crate.
- F16/BF16 conversion cost is paid at most once per tensor per process lifetime.
- Single-GEMM QKV path reduces per-layer arithmetic from 5 matmuls to 3.

**Negative**:

- 153 `unsafe` blocks in the crate (documented in `src/lib.rs`); the 'static extension is one contributor.
- Hand-written JSON parser must be maintained if the safetensors spec evolves.
- Peak load-time memory is ~1.3× final steady-state memory (QKV fusion window).

**Risks**:

- Drop order dependency on RFC 1857 is an invariant that must be preserved. A future refactor that reorders struct fields or moves to `Arc` would invalidate the safety argument. Documented in `src/model/bert.rs`: "WARNING: Do NOT reorder the fields of BertModel."

---

## References

- `src/weights/f32_weights.rs` — `SafetensorsFile`, `JsonParser`, `TensorMeta`, `ShardedSafetensors`
- `src/model/qwen.rs` — `SafetensorsStorage`, `ShardedQwenBacking`, field drop order comment
- `src/lib.rs` — stability note on 153 unsafe blocks
- SafeTensors format spec — https://github.com/huggingface/safetensors
- RFC 1857 — https://rust-lang.github.io/rfcs/1857-stabilize-drop-order.html
