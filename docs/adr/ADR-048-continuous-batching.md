# ADR-048: Continuous Batching with Disaggregated Prefill/Decode

**Status**: Accepted
**Date**: 2026-05-19
**Crate**: `lattice-inference`
**Depends on**: ADR-004 (KV Cache), ADR-008 (LoRA Injection), ADR-047 (Paged KV Cache)

---

## Context

The current generation API (`generate_greedy`, `generate_sampled`, `generate`) is single-request:
one prompt in, one response out, blocking. The Metal forward path (`MetalQwen35State`,
`forward_step`) is also single-session, as confirmed by `bench_concurrent.rs`:
`forward_step` calls `cmd.commit()` + `wait_until_completed()` per token, serializing all GPU
work through a synchronous barrier with no cross-session pipelining.

Three serving problems emerge at scale:

1. **Head-of-line blocking**: a 4096-token prefill stalls all waiting decode requests.
2. **GPU underutilization during decode**: decode is memory-bandwidth-bound, leaving arithmetic
   units idle. A single decode stream uses ~40% of Apple Silicon GPU compute.
3. **No request lifecycle management**: no queue, no preemption, no SLA.

Two techniques address these independently:

**Continuous (in-flight) batching** (Orca, 2022): requests enter and exit the execution window
at iteration boundaries rather than at request boundaries. The scheduler selects a batch each
iteration; finished sequences are evicted and new ones fill their slots. This is the baseline
for any production LLM serving system (llama.cpp, Ollama, vLLM all use it).

**Disaggregated prefill/decode** (Splitwise, 2024; NVIDIA Dynamo; vLLM V1): separate compute-
bound prefill from bandwidth-bound decode by routing them to distinct workers. On CUDA this
requires KV cache transfer over NVLink or PCIe. On Apple Silicon unified memory, prefill and
decode workers share the same physical memory — the "transfer" is a pointer swap with zero copy.
This is a structural advantage of the Metal/unified-memory architecture.

**Chunked prefill** (TRT-LLM, vLLM V1): long prompts are split into fixed-size chunks (e.g.,
512 tokens) and interleaved with decode steps. Bounds prefill latency spikes without requiring
separate workers.

### Lattice-specific constraints

- `PagedKVCache` (ADR-004/047) provides the per-sequence memory substrate: 256-token pages, LRU
  eviction. Note: current `PagedKVCache` owns a single `PageTable`; multi-sequence support
  requires managing multiple page tables externally (see `paged.rs:247-251`). This ADR defines
  `SequenceManager` as that external owner. ADR-047's prefix cache also needs `AdapterId`-aware
  lookup (defined there) before continuous batching can reuse prefixes across sequences.
- GDN attention (`src/attention/gdn.rs`) carries a per-sequence recurrent state matrix
  (`S ∈ R^{d_model × d_model}`), not just KV pairs. This state must be batched and routed
  alongside KV pages.
- LoRA (ADR-008, ADR-043): different requests may use different adapters loaded via `LoraHook`.
  The scheduler must track adapter identity per request and prevent mixing adapters within a
  single batched matrix multiply.
- Metal command buffers are not thread-safe; a `MTLCommandQueue` is, and multiple encoders
  can be committed from one queue with ordering guarantees.

---

## Decision

Implement continuous batching with chunked prefill. Defer full disaggregation to a second
phase. In unified memory, chunked prefill achieves most of disaggregation's TTFT benefit
without requiring a separate worker process.

**Phase 1 (this ADR)**: Iteration-level scheduler with chunked prefill and a shared
`PagedKVCache`. Single Metal command queue; prefill chunks and decode steps interleaved
within the same iteration.

**Phase 2 (future ADR)**: Disaggregated prefill worker — a separate Tokio task owning a
dedicated Metal command buffer encoder, transferring KV page ownership to the decode pool
via `Arc<PagedKvBlock>` pointer swap (zero physical copy).

---

## Scope

This ADR covers interfaces and data structures only. It does not mandate any implementation
timeline for Phase 2, and does not touch the CPU inference path (`generate.rs`).

**In scope**:
- `InferenceRequest` and `InferenceResponse` types
- `Scheduler` trait and `FifoScheduler` implementation
- Iteration loop structure for the serving task
- GDN state batching protocol
- LoRA adapter grouping constraint

**Out of scope**:
- HTTP/gRPC serving layer
- Speculative decoding integration (ADR-006)
- Quantization during serving (ADR-044)

---

## Architecture

### Request lifecycle

```text
  Client submit
       │
       ▼
  RequestQueue (tokio::sync::mpsc)
       │
       ▼
  Scheduler::select_batch()     ← called each iteration
       │
       ├─ prefill slots: chunk prompt → PagedKvBlock allocation → Metal prefill pass
       │
       └─ decode slots: single token → KV append → Metal decode pass
            │
            ▼
       SamplerPool::sample()    ← per-sequence sampling config
            │
            ▼
       token → stream to caller (tokio::sync::oneshot or watch channel)
            │ (if EOS or max_len)
            ▼
       SequenceManager::release(seq_id)  // drops PageTable + GDN state for this sequence
```

### Core types

```rust
/// Stable identifier for a sequence across its lifetime in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(u64);

/// Submitted by the caller; scheduler takes ownership.
pub struct InferenceRequest {
    pub id: SeqId,
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingConfig,
    /// None = base model. Some = adapter key for LoraHook dispatch.
    pub lora_adapter: Option<AdapterKey>,
    /// Maximum tokens to generate (hard cap enforced by scheduler).
    pub max_new_tokens: usize,
    pub response_tx: tokio::sync::mpsc::Sender<InferenceToken>,
}

/// Streamed back to caller one token at a time.
pub struct InferenceToken {
    pub seq_id: SeqId,
    pub token_id: u32,
    pub finished: bool,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Copy)]
pub enum FinishReason {
    Eos,
    MaxLength,
    Preempted,   // scheduler evicted this sequence under memory pressure
}
```

### Scheduler trait

```rust
pub trait Scheduler: Send {
    /// Called each iteration. Returns:
    /// - `prefill`: sequences whose next chunk should run prefill this iteration.
    /// - `decode`:  sequences ready for a single decode step.
    /// Invariant: all sequences in `decode` have had at least one prefill chunk committed.
    fn select_batch(
        &mut self,
        waiting: &[SeqId],
        running: &[SeqId],
        kv_free_pages: usize,
        gdn_free_slots: usize,
    ) -> SchedulerDecision;

    /// Notify the scheduler that a sequence has been evicted (memory pressure).
    fn on_preempt(&mut self, seq_id: SeqId);
}

pub struct SchedulerDecision {
    pub prefill: Vec<(SeqId, /*chunk_start*/ usize, /*chunk_len*/ usize)>,
    pub decode: Vec<SeqId>,
    /// Sequences to evict before this iteration runs.
    pub evict: Vec<SeqId>,
}
```

`FifoScheduler` (Phase 1): FIFO queue, chunk size 512 tokens, no preemption. Eviction
triggered only when `kv_free_pages < PREFILL_RESERVE_PAGES`.

### GDN state batching

GDN recurrent state `S ∈ R^{d_model × d_model}` per sequence cannot be packed into a
standard KV page. It lives in `GdnStatePool`:

```rust
pub struct GdnStatePool {
    /// One Metal buffer per sequence slot, pre-allocated at pool creation.
    /// Capacity: max_concurrent_seqs × d_model × d_model × sizeof(f32).
    buffers: Vec<metal::Buffer>,
    free_slots: VecDeque<usize>,
    seq_to_slot: HashMap<SeqId, usize>,
}
```

During a batched decode iteration, the Metal kernel receives a list of `(seq_slot, kv_page_ids)`
pairs. The GDN update for each sequence reads/writes its slot independently — no cross-sequence
state sharing occurs. Batch size is bounded by `max_concurrent_seqs` (see Risks).

### LoRA adapter grouping

When multiple active sequences use different LoRA adapters, they cannot share a single batched
matrix multiply because the weight delta is per-adapter. The scheduler groups sequences by
adapter key before constructing each iteration's batch:

```text
Iteration batch = base_model_group ∪ adapter_A_group ∪ adapter_B_group
```

Each group runs as a separate sub-batch. Groups may be processed sequentially within one
command buffer using separate `MTLComputeCommandEncoder` segments. This adds encoder
boundary overhead but maintains correctness.

If a pathological workload has N distinct adapters and N active sequences, throughput
degrades to serial. A future ADR may address this via LoRA merged weights (adapter
specialization at load time).

### Metal command buffer structure (Phase 1)

```text
MTLCommandBuffer
  ├── ComputeCommandEncoder: prefill chunks (all sequences in prefill group)
  │     Layer 0..L: QKV → RoPE → GQA/GDN prefill → FFN
  │     PagedKvCache: block allocation and write
  ├── ComputeCommandEncoder: decode steps (all sequences in decode group)
  │     Layer 0..L: QKV → RoPE → GQA/GDN decode → FFN
  │     PagedKvCache: single-page append
  └── commit() → wait_until_completed()
```

Phase 2 splits this into two separate command buffers submitted concurrently on two
`MTLCommandQueue` instances (one prefill, one decode), using
`addCompletedHandler` for async notification and `Arc<PagedKvBlock>` ownership transfer
between them.

---

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|---|---|---|---|
| **Request-level batching** (static) | Simple scheduler; no mid-batch eviction | Decode waits for slowest sequence; GPU idle during mismatched lengths | Rejected — baseline of 2022, superseded by Orca |
| **Full disaggregation (Phase 2 now)** | Maximum TTFT/TBT separation; cleanest utilization | Two `MTLCommandQueue` instances add synchronization complexity; `Arc<PagedKvBlock>` ownership protocol unproven | Deferred — chunked prefill recovers most of the gain on single-GPU Apple Silicon |
| **Separate process per worker** | True isolation; OS-level memory protection | IPC overhead; no shared Metal heap; defeats unified-memory zero-copy advantage | Rejected — Rust `async` tasks in one process with ownership transfer achieves isolation without IPC |
| **Priority-based scheduler (SLA-aware)** | Better p99 latency for high-priority requests | Priority inversion risk; more complex preemption policy; premature for Phase 1 | Deferred to Phase 2 |
| **Preemption via cache eviction** | Handles memory pressure gracefully | Must serialize evicted sequence's KV pages to CPU or disk; page fault on resume | Accepted for Phase 2; Phase 1 rejects new requests when memory is low instead of evicting |

---

## Risks

**R1: GDN state pool capacity bounds maximum batch size.**
`GdnStatePool` pre-allocates one Metal buffer per concurrent sequence. For Qwen3.5-2B
(d_model=1536): 1536 × 1536 × 4 bytes = 9.4 MB per slot. On M2 Max (96 GB):
available GDN pool ≈ (working_set × 0.2) / 9.4 MB ≈ ~2,000 slots. In practice, KV
pages are the tighter constraint — measure empirically before setting `max_concurrent_seqs`.

**R2: LoRA adapter grouping serializes multi-adapter workloads.**
N adapters across N sequences = N sequential sub-batches. This is a P2 risk (not a
correctness issue), surfaced for tracking. Mitigation: prioritize homogeneous adapter
batches in the scheduler when adapter count exceeds a threshold.

**R3: Metal command buffer granularity.**
A single command buffer containing both prefill and decode encoders cannot be partially
cancelled. If a prefill chunk exceeds the Metal device timeout (default 8 seconds for
long prompts), the entire buffer fails. Mitigation: enforce `chunk_size ≤ 512` tokens,
which bounds prefill time to ~100 ms on M2 Max for Qwen3.5-2B Q4.

**R4: forward_step synchronization.**
The current `forward_step` calls `wait_until_completed()` per token. This must change to
`addCompletedHandler` + `tokio::sync::oneshot` for async notification. The migration
requires restructuring `MetalQwen35State` so it does not hold GPU completion state on the
call stack. This is a non-trivial refactor of the Metal path and must be prototyped before
committing to the full serving loop.

**R5: GDN recurrent state correctness under preemption.**
If a sequence is preempted and its GDN slot is released, the state matrix is lost.
On resume the sequence must restart from the beginning (re-prefill from token 0).
For Phase 1 (no preemption), this is not triggered. For Phase 2, the preemption policy
must serialize the GDN state to a CPU buffer before releasing the Metal slot.

---

## References

- `src/generate.rs` — current single-request generation loop; `forward_with_cache` is the
  target for refactoring into the serving forward path
- `src/kv_cache/` — `PagedKVCache` (256-token pages, LRU eviction) — memory substrate
- `src/attention/gdn.rs` — GDN recurrent state; per-sequence `S` matrix
- `examples/bench_concurrent.rs` — confirms `forward_step` is synchronous (blocking
  `wait_until_completed()`); documents the shared-engine API gap
- Agrawal et al. 2024 — "Taming Throughput-Latency Tradeoff in LLM Inference with
  Disaggregated Prefill-Decode" (Splitwise) — https://arxiv.org/abs/2311.18677
- Yu et al. 2022 — "Orca: A Distributed Serving System for Transformer-Based Generative
  Models" (iteration-level scheduling) — OSDI 2022
- Metal Programming Guide — `MTLCommandQueue`, `addCompletedHandler`,
  `recommendedMaxWorkingSetSize` — developer.apple.com/documentation/metal
- vLLM V1 architecture — chunked prefill and disaggregated prefill design
  — https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html
- ADR-004: KV Cache Design (`docs/adr/ADR-004-kv-cache.md`)
- ADR-008: LoRA Injection via Trait Hook (`docs/adr/ADR-008-lora-injection.md`)
- ADR-043: LoRA Serving Verification (`docs/adr/ADR-043-lora-serving-verification.md`)
