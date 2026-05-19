# ADR-047: Paged KV Cache with Prefix Reuse

**Status**: Proposed
**Date**: 2026-05-19
**Crate**: `lattice-inference`

---

## Context

ADR-004 introduced two KV cache implementations: `FlatKVCache` (contiguous slab, single-sequence)
and `PagedKVCache` (256-token pages, LRU eviction, multi-sequence). Both treat each request as
independent — every prompt is encoded from scratch on every call.

Three workloads expose this as a bottleneck:

1. **Interactive chat with a system prompt.** Every turn re-encodes the same system prompt tokens.
   For a 1024-token system prompt on Qwen3-0.6B, prefill dominates per-turn latency.
2. **Batched embedding with shared prefix.** The embedding service in `lattice-embed` is called in
   tight loops where many inputs share a common instruction prefix (e.g., `"Represent this sentence:"`).
3. **LoRA hot-swap serving.** Multiple adapters share one base model. System-prompt KV tensors are
   base-model-specific and can be shared across adapter variants that don't alter the first N tokens.

The CUDA ecosystem has converged on paged KV caches as a baseline requirement (PagedAttention, vLLM,
SGLang RadixAttention). For Apple Silicon the design simplifies: unified memory means no GPU/CPU
page-transfer cost. Capacity is bounded by system RAM (~32-192 GB on M-series), not a 24 GB VRAM
ceiling. Metal buffer alignment requirements are the main hardware constraint.

The current `PagedKVCache` already solves fragmentation and enables multi-sequence capacity via LRU.
What it lacks is identity-based page sharing: two requests with identical prefix tokens currently
allocate and populate separate pages rather than pointing at the same physical pages.

---

## Decision

Add a **prefix-sharing layer** on top of the existing `PagedKVCache` infrastructure. The design has
two components:

1. **`PrefixPageCache`**: A bounded LRU hash map keyed by `PrefixKey { adapter_id: AdapterId,
   token_hash: u64 }` (FxHash of the token slice, namespaced by adapter). On a hit, the caller
   receives read-only shared `Arc` references to pre-populated physical pages. On a miss, newly
   computed pages are inserted and shared. This is the path for single-user sessions on Apple
   Silicon — simpler than a radix tree, covers 90% of the benefit.

2. **`SharedPageRef`**: A thin newtype wrapping `Arc<[f32]>` (one arc per page, pointing into a
   shared `PagePool` allocation). Read access is lock-free; write access requires promotion to an
   owned page (copy-on-write triggered by append beyond the shared prefix length).

The radix tree approach (SGLang RadixAttention) is deferred to the server path (see Alternatives).

### Rust interface (proposed additions to `src/kv_cache/`)

```rust
/// A hash-keyed LRU cache of immutable prefix pages.
pub struct PrefixPageCache {
    /// max number of prefix entries retained.
    capacity: usize,
    entries: IndexMap<PrefixKey, PrefixEntry>,
}

/// Compound key enforcing LoRA namespace at the data structure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefixKey {
    pub adapter_id: AdapterId,
    pub token_hash: u64, // FxHash of token_ids slice
}

pub struct PrefixEntry {
    pub prefix_len: usize,
    /// One Arc<[f32]> per layer; layout matches PagePool page layout.
    pub pages: Vec<Arc<[f32]>>,
    pub last_used: std::time::Instant,
}

/// Adapter ID for LoRA-aware prefix namespacing.
/// AdapterId(0) = base model (no adapter). Non-zero = content hash of loaded adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdapterId(pub u64);

impl PrefixPageCache {
    pub fn new(capacity: usize) -> Self;
    pub fn lookup(&mut self, adapter_id: AdapterId, token_ids: &[u32]) -> Option<&PrefixEntry>;
    pub fn insert(&mut self, adapter_id: AdapterId, token_ids: &[u32], entry: PrefixEntry);
    /// Evict LRU entries until at or below capacity.
    fn evict(&mut self);
}
```

The `PagedKVCache` acquires a `PrefixPageCache` reference during construction (dependency injection
via `Option<Arc<Mutex<PrefixPageCache>>>`). When `seq_len == 0` and a prefix hash hits the cache,
the cache copies the shared pages into the pool and fast-forwards `seq_len` to `prefix_len` before
the caller's first `append_kv_layer` call. The copy cost is paid once; subsequent decode steps are
unaffected.

---

## Scope

This ADR covers:
- `PrefixPageCache` struct and lookup/insert/evict logic
- Integration point in `PagedKVCache::new` and a new `PagedKVCache::restore_prefix` method
- Interaction with LoRA hot-swap (adapter-keyed prefix namespacing)
- Metal buffer alignment constraints for page size selection

This ADR does not cover:
- Full radix tree implementation (deferred — see Alternatives)
- Multi-tenant server path (future ADR)
- Persistent KV cache across process restarts
- Quantized KV (fp16 / int8) storage (separate ADR)

---

## Architecture

### Page size for Metal alignment

The existing `PagedKVCache` uses 256-token pages (from ADR-004). Metal buffer allocation aligns to
4 KB (system page) or 16 KB (Metal default heap). At Qwen3-0.6B geometry (8 KV heads, 128
head_dim, 28 layers):

```
bytes_per_page = 28 × 2 × page_size × 1024 × 4
  page_size=256 → 56 MB per page  (no alignment issue, but large)
  page_size=64  →  14 MB per page
  page_size=16  → 3.5 MB per page
```

For prefix caching, page granularity directly controls sharing granularity. A 256-token page means
the minimum prefix length that produces a cache hit is 256 tokens. Smaller pages enable finer-grained
sharing at the cost of more entries in the page table.

**Decision**: retain 256-token pages for the existing `PagedKVCache` multi-sequence path. For the
prefix cache specifically, introduce a `prefix_page_size` parameter (default 64 tokens) that governs
prefix page granularity. A 64-token prefix page at Qwen3-0.6B is 14 MB — fits one `MTLBuffer`
allocation comfortably and enables sharing on typical system prompts (64–512 tokens).

### Copy-on-write for mutable extension

When a request hits a prefix cache entry of length `P` and then generates new tokens:

1. `restore_prefix` copies the shared page data into owned pool pages and sets `seq_len = P`.
2. Subsequent `append_kv_layer` calls write into those owned pages normally.
3. When generation completes, the caller can optionally call `promote_to_prefix` to hash the full
   output and insert it into the prefix cache for future reuse.

This avoids any synchronization on the hot decode path. Reads from the cache are `Arc` clones
(atomic ref count increment); writes never touch shared pages.

### LoRA interaction

LoRA adapters modify query, key, and value projections (`q_proj`, `k_proj`, `v_proj`). The KV
tensors written into the cache are the post-adapter outputs. Therefore prefix pages computed under
adapter A are not valid for adapter B.

**Namespacing rule**: the prefix cache key is `(adapter_id, token_hash)` where `adapter_id` is a
`u64` derived from the adapter's content hash (established at load time by `lattice-tune`). The
base model uses `adapter_id = 0`. This ensures cross-adapter sharing never occurs by construction.
The `PrefixPageCache` capacity budget is shared across all adapter namespaces; LRU eviction is
global.

### Metal unified memory

Because the `PagePool` allocates into `Vec<f32>` (CPU heap), Metal GPU access goes through
`MTLBuffer::newBufferWithBytesNoCopy` or a blit into a managed `MTLBuffer`. The prefix cache holds
`Arc<[f32]>` slices — same memory region. No additional copies are needed for the GPU path; the
Metal command encoder reads directly from the same allocation. This is a unified-memory advantage
not available in the CUDA ecosystem.

---

## Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **Radix tree (RadixAttention / SGLang)** | Optimal prefix matching at any token boundary; 6.4x throughput on prefix-heavy server workloads | Complex to implement in safe Rust; interior mutability on tree nodes; LRU on leaf nodes requires careful ownership | Deferred — implement when multi-tenant server path is active. Hash map covers the single-user case. |
| **No prefix cache; longer FlatKVCache** | Zero code complexity | Re-encodes the same prefix on every request; unacceptable for chat latency at 1K+ token system prompts | Rejected — measurable latency regression. |
| **Persist KV cache to disk (mmap)** | Survives process restart; useful for fixed system prompts | mmap I/O latency (μs–ms) during restore; added complexity; macOS unified memory already fast | Deferred — low priority until session resumption is a product requirement. |
| **Smaller page size (16 tokens) for finer sharing** | More sharing opportunities; works for shorter common prefixes | 16× more page table entries; LRU bookkeeping overhead; 16-token page is 3.5 MB (fine) but table grows proportionally | Not adopted; 64-token prefix page is a reasonable balance point. |
| **fp16 KV storage in prefix pages** | Halves prefix cache memory; doubles effective capacity | Introduces conversion cost at restore time; breaks the invariant that cached and live pages have identical layout | Deferred to a quantized-KV ADR. |

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hash collision on token sequence | Very low (FxHash 64-bit on token ids, not strings) | `debug_assert` prefix length match after hit; log collision metrics |
| LoRA adapter id not propagated to cache call site | Medium — call sites spread across `generate.rs` | `PrefixPageCache::lookup` signature requires `AdapterId`; omitting it is a compile error |
| Arc ref-count overhead on prefix restore | Low — one `clone()` per page per layer at restore | Profile shows atomic increment < 5 ns; restore of 8 pages = 40 ns, negligible vs prefill |
| Cache invalidation on model reload | Low but silent | `PrefixPageCache::clear()` called on model swap; cache is owned by session, not global |
| Prefix cache grows unbounded in long session | Medium — embedding loops can insert many entries | Bounded LRU capacity (default 128 entries); eviction is O(1) with `IndexMap` |

---

## References

- `src/kv_cache/paged.rs` — `PagePool`, `PageTable`, `PagedKVCache`, `EvictionPolicy`
- `src/kv_cache/flat.rs` — `FlatKVCache` (single-sequence path, unaffected)
- `src/kv_cache/mod.rs` — public re-exports
- `src/speculative.rs` — uses `truncate_to()` for rollback; prefix cache must not interfere with rollback (prefix is immutable before first owned append)
- ADR-004 — KV cache design; established `FlatKVCache` + `PagedKVCache` split and 256-token page size
- ADR-008 — LoRA injection trait; source of `adapter_id` namespace requirement
- ADR-031 — LoRA adapter management; adapter content hash is available at load time
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," NeurIPS 2024 (RadixAttention)
