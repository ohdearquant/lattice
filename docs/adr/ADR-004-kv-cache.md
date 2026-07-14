# ADR-004: KV Cache Design

**Status**: Accepted\
**Date**: 2026-05-13\
**Crate**: `lattice-inference`

## Amendment (2026-06-24)

Two facts in this ADR no longer match the implementation:

1. **Storage format changed from f32 to f16.** `flat.rs` stores K and V as
   `Vec<Vec<f16>>`. Writes convert f32 to f16 (`f16::from_f32`); reads dequantize
   via `read_k_into`/`read_v_into`. This halves the memory footprint. For the
   Qwen3-0.6B example (28 layers, 8 KV heads, 128 head_dim, 4096 max_seq_len) the
   footprint is **~448 MB**, not ~896 MB as the original ADR stated. The `total_bytes()`
   method on `FlatKVCacheConfig` reflects this; a test in `flat.rs` asserts
   "expected ~448 MB."

2. **`append_kv` and related mutators now return `Result<_, InferenceError>`.**
   In PRs #290/#298 the previously panicking mutators were hardened to return
   `Result<usize, InferenceError>` (for `append_kv`) and `Result<(), InferenceError>`
   (for `advance`, `advance_by`, `prefill_layer`). The "allocation-free hot path"
   property is preserved — the `Result` path adds only a bounds check, not an
   allocation. Inline corrections are marked with `[CORRECTED]` below.

---

## Context

Autoregressive generation requires storing key and value tensors from all previous positions so that attention over the full prefix is computed without re-encoding tokens. The design must handle two use cases with different constraints:

**Single-sequence inference** (embedding generation, short generation tasks):

- One sequence at a time.
- Predictable maximum length (from model config).
- No eviction needed.
- Lowest possible overhead on the critical path.

**Multi-sequence / multi-model serving** (concurrent embedding generation or batched generation):

- Multiple concurrent sequences, possibly for different models.
- Memory must be shared and reused across sequences.
- Working set may exceed available memory for some sequence lengths.

Relevant implementation: `src/kv_cache/flat.rs` and (paged variant) `src/kv_cache/`.

`FlatKVCacheConfig`:

```rust
pub struct FlatKVCacheConfig {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
}
```

`FlatKVCache`:

```rust
// [CORRECTED: storage is f16, not f32 — see Amendment above]
pub struct FlatKVCache {
    k: Vec<Vec<f16>>,   // [num_layers][max_seq_len * kv_dim]  (f16 elements)
    v: Vec<Vec<f16>>,   // [num_layers][max_seq_len * kv_dim]  (f16 elements)
    seq_len: usize,
}
```

Memory footprint for Qwen3-0.6B (28 layers, 8 KV heads, 128 head_dim, 4096 max_seq_len):

```
[CORRECTED] 28 × 8 × 128 × 4096 × 2 (K+V) × 2 bytes (f16) ≈ 448 MB
(was: 28 × 8 × 128 × 4096 × 2 (K+V) × 4 bytes (f32) = 896 MB)
```

`FlatKVCache` key methods:

- `append_kv()` — O(1) append; [CORRECTED: returns `Result<usize, InferenceError>` (#298), not `()` — see Amendment]
- `truncate_to(n)` — sets `seq_len = n`, no deallocation
- `reset()` — zeros data + `seq_len = 0`
- `reset_fast()` — `seq_len = 0` only (no zero fill; caller owns validity)
- `prefill_layer() + advance_by()` — bulk write for prompt prefill

The paged variant (`PagedKVCache`) allocates in 256-token pages and uses LRU eviction to support multi-model workloads where aggregate sequence memory exceeds RAM.

---

## Decision

Provide **two complementary KV cache implementations**: `FlatKVCache` for single-sequence inference (pre-allocated contiguous layout, no eviction) and `PagedKVCache` for multi-sequence/multi-model serving (page-granular allocation, LRU eviction). The flat cache is the default path for all single-sequence use cases including embedding generation and short decoding loops.

---

## Key Design Choices

1. **Pre-allocation in FlatKVCache**: All memory is allocated at construction time.
   `append_kv()` is a bounds check + slice write — no allocation on the hot path.
   [CORRECTED: `append_kv` returns `Result<usize, InferenceError>` (#298); the
   bounds check now produces an `Err` instead of panicking. The allocation-free
   property is preserved.] For embedding workloads where `forward()` is called
   thousands of times per second, this eliminates GC pressure entirely.
2. **Contiguous layout per layer**: Each layer's K tensor is a single `Vec<f16>` of size `max_seq_len * kv_dim`. The layout `[max_seq_len, kv_dim]` means that reading a full key sequence for attention is a single contiguous memory region; reads dequantize into caller-provided f32 scratch for attention scoring.
3. **`reset_fast()` vs `reset()`**: The fast reset sets only `seq_len = 0` without zeroing. The caller is responsible for not reading stale data. This is correct because every position up to `seq_len` is written before being read. Saves ~7 ms for a 4096-token cache on M3.
4. **256-token page granularity in PagedKVCache**: Balances internal fragmentation (larger pages = more waste for short sequences) against management overhead (smaller pages = more entries in the page table and LRU structure). 256 tokens at 128 head_dim × 8 heads × 2 sides × 4 bytes = 2 MB per page — fits one huge page on Linux.
5. **LRU eviction in PagedKVCache**: Sequences that haven't been accessed recently are evicted. This is correct for multi-user embedding servers where active queries are always in progress and idle sequences can be re-encoded on next request.

---

## Alternatives Considered

| Alternative                          | Pros                                                      | Cons                                                                                             | Why Not                                                                             |
| ------------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| Single paged cache for all use cases | One code path                                             | LRU bookkeeping overhead per append; page indirection adds ~20 ns per attention lookup           | Flat cache latency is critical for single-sequence throughput                       |
| Ring buffer (circular KV cache)      | Fixed memory; supports sliding window attention naturally | Breaks absolute positional encoding (RoPE position index must match the original token position) | RoPE encodes absolute position; sliding window requires different attention masking |
| `Arc<Mutex<Vec>>` per layer          | Enables concurrent writes                                 | Lock contention on every token; unnecessary for single-thread decode                             | Decode is inherently sequential (each token depends on previous)                    |
| Disk-backed KV cache                 | Unlimited sequence length                                 | I/O latency (μs to ms) incompatible with low-latency decode goals                                | Latency budget                                                                      |
| Lazy allocation (grow on demand)     | Lower memory for short sequences                          | Allocation on hot path; heap fragmentation over many sequences                                   | Pre-allocation eliminates the allocation cost entirely                              |

---

## Consequences

**Positive**:

- `FlatKVCache::append_kv()` is allocation-free and branch-minimal on the hot path.
- Memory consumption is predictable and bounded at construction time.
- `prefill_layer()` + `advance_by()` enables bulk writes during prompt prefill, avoiding per-token overhead.
- `truncate_to()` supports speculative decoding rollback without deallocation.

**Negative**:

- `FlatKVCache` reserves full memory for `max_seq_len` at construction regardless of
  actual sequence length. [CORRECTED: for Qwen3-0.6B with 4096 max tokens, this is
  **~448 MB** (f16 storage, 2 bytes/element) — always, even for a 10-token sequence.
  The original ADR stated 896 MB (f32). The f16 change halved the footprint.]
- Two cache types mean two code paths to maintain and test.

**Risks**:

- [CORRECTED: the flat cache footprint for Qwen3-0.6B at 4096 tokens is ~448 MB (f16),
  not 896 MB (f32). The guidance stands: consumers must pass a smaller `max_seq_len`
  for memory-constrained deployments, but the budget is now approximately half what
  the original ADR implied.]
- `reset_fast()` is safe only if the caller guarantees it writes all positions before reading them. A bug that reads a stale position will produce wrong attention — no memory safety violation, but wrong outputs with no error signal.

---

## References

- `src/kv_cache/flat.rs` — `FlatKVCache`, `FlatKVCacheConfig`, all methods
- `src/kv_cache/` — `PagedKVCache` (paged variant with LRU)
- `src/model/qwen.rs` — `QwenModel` constructs `FlatKVCache` from `QwenConfig`
- `src/speculative.rs` — uses `truncate_to()` for cache rollback on rejection
