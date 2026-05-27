# ADR-062: Metal FlashAttention-2 Prefill + KV Cache Quantization Chain

**Status**: Proposed
**Date**: 2026-05-27
**Crate**: lattice-inference (Metal shaders, KV cache)
**Research**: RQ-4 (`workspaces/20260527/04.md`)
**Issues**: #126 (Metal FA2 prefill), #85 (MLX kernel study), #86 (shader extraction)
**KG entities**: `Metal FA2 Prefill` (0dfbc841), `Metal Fused Attention` (48ee18b2), `FlashAttention-2` (63602a7f), `Chunked Prefill` (018193b3)

---

## Context

### The bottleneck: sequential per-token attention during prefill

Lattice's Metal attention kernel (`forward/metal_qwen35.rs`, 15,546 lines) is decode-only fused. During prefill (`forward_prefill_impl`, line 6238), each token is processed sequentially in a `for t in 0..n` loop (line 6694). For each token `t`, the kernel dispatches `decode_attention` with `cache_len = t + 1` (line 6805-6806):

```rust
// forward/metal_qwen35.rs:6694-6806
for t in 0..n {
    // ... scatter Q + gate, per-head RMS norm, partial RoPE, store K/V ...

    // Causal attention: query Q[t] against cache[0..t+1]
    let cache_len = (t + 1) as u32;
    enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention);
    // ...
    enc.dispatch_thread_groups(
        MTLSize::new(nqh as u64, 1, 1),  // one threadgroup per query head
        MTLSize::new(256, 1, 1),
    );
}
```

This means prefill of `n` tokens dispatches `n` separate Metal compute commands, each processing an attention matrix of size `[1, t+1]`. Total work is O(n^2) dot products across all dispatches, with no Q-dimension parallelism, no tile reuse across query positions, and `n` GPU command buffer synchronization points. At 4K+ context, prefill latency is dominated by this attention loop.

The `decode_attention` kernel itself (line 418-559) is an online-softmax GQA kernel with `Grid: [num_kv_heads, 1, 1], Threads: [256, 1, 1]` -- one threadgroup per KV head, 256 threads tiling over cache tokens in TILE_TOKENS=256 chunks. This design is optimal for M=1 decode but catastrophic for M>1 prefill.

Evidence that this matters: llama.cpp with `--flash-attn` beats MLX by 9 seconds at 8.5K context. The gap is almost entirely prefill throughput.

### Hardware constraint: 32 KiB threadgroup memory

Apple GPU Families 7-10 (M1-M4) share a binding constraint: **32 KiB threadgroup memory per threadgroup** and 1024 max threads per threadgroup ([Apple Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)). Apple's feature table also exposes larger imageblock limits (128-256 KiB), but those are not usable for generic compute kernels. The M1-M4 generation map to Apple GPU Families 7/8/9/10 respectively, but the TGM limit is identical across all four.

This constrains tile sizes. At f16, `BQ=32 BK=32 D=128` uses approximately 18.5 KiB -- safe. At f32, the same tile needs ~34.5 KiB -- exceeds the limit. Runtime selection via `supportsFamily` and `MTLComputePipelineState.maxTotalThreadsPerThreadgroup` is mandatory.

### Existing KV cache: f32-only

Both `FlatKVCache` (flat.rs, 543 lines) and `PagedKVCache` (paged.rs, 1150 lines) store K and V as `Vec<f32>`. The Metal kernel reads f32 from KV buffers at full 32-bit bandwidth. Halving this to f16 is the single cheapest bandwidth win available, and is a prerequisite for the int8/int4 quantization chain.

`PrefixPageCache` (prefix.rs, 439 lines) is built and wired into `PagedKVCache` via `restore_prefix` / `promote_to_prefix`, but operates on `Arc<[f32]>` shared pages. The prefix cache design is format-agnostic -- it copies between page layouts at different page sizes -- but currently assumes f32 throughout.

---

## Decision

### Phase 0: Shader extraction (prerequisite)

**D0: Extract Metal shaders from `metal_qwen35.rs` into standalone `.metal` files.**

The 15,546-line Rust file contains Metal shader source as `include_str!` string literals mixed with Rust dispatch code. This is untestable in isolation, unreviewable in diffs, and unreusable across model architectures. Before any FA2 kernel work, extract to:

```
crates/lattice-metal/
  shaders/
    common/
      types.metal          # shared typedefs, f16-f32 helpers
      simd_utils.metal     # simdgroup reductions
      rope.metal           # partial RoPE kernel
      quant.metal          # int4/int8 nibble unpack, scale apply
      page_table.metal     # logical-to-physical page resolution
    attention/
      decode_f16.metal     # current M=1 decode path, ported to f16 KV
      decode_kv_quant.metal # future: compressed page decode
      fa2_prefill_f16.metal # new: tiled prefill kernel
      fa2_prefill_kv_quant.metal # future: int4/int8 prefill
    kv/
      kv_write_f16.metal   # f32-to-f16 cache store
      kv_quant_i8.metal    # future: int8 quantize-and-store
      kv_quant_i4_wht.metal # future: WHT + int4 pack
      kv_quant_i4_srft.metal # experimental: SRFT rotation
    generated/
      attention_instantiations.metal # template instantiations per tile shape
  build.rs               # xcrun metal -c validation
  src/
    shader_registry.rs   # pipeline cache keyed by {kernel, dtype, D, BQ, BK, GPU family}
    pipeline_cache.rs    # MTLComputePipelineState reuse
```

Three build modes:

1. **Development**: `device.new_library_with_source(include_str!("../shaders/..."), opts)` -- easy debugging, easy specialization.
2. **CI validation**: `xcrun -sdk macosx metal -std=metal3.1 -I shaders/ -c *.metal -o *.air && xcrun metallib *.air -o lattice.metallib` -- catches syntax errors without a Metal GPU.
3. **Release**: precompiled `lattice.metallib` embedded for known tile variants; runtime source compilation as fallback for experimental shapes.

### Phase 1: f16 KV cache

**D1: Store K and V as `half`. Accumulate QK scores, softmax state, and output in `float`.**

Changes to `flat.rs`:
- `FlatKVCache` gets a parallel `k_f16: Vec<Vec<u16>>` / `v_f16: Vec<Vec<u16>>` storage path, gated by a `KvFormat` enum.
- `append_kv` converts f32 input to f16 at write time.
- `get_k` / `get_v` return f16 data; the attention kernel casts inside TGM load.
- An f32 correctness path is preserved for regression testing.

Changes to `paged.rs`:
- `PagePool` data becomes `Vec<u16>` when `KvFormat::F16` is active.
- `gather_k` / `gather_v` return f16 slices; no conversion buffer.

Changes to `prefix.rs`:
- `SharedPageRef` wraps `Arc<[u16]>` for f16 pages. The `copy_token_between_page_layouts` function operates on half-width elements.

Metal kernel changes:
- Tile load casts `half -> float` for QK accumulation and V accumulation.
- Store path writes `float -> half` when populating KV cache from projections.

Expected effect: **halve KV cache traffic**. End-to-end decode speedup less than 2x (weights, projections, softmax, sampling still dominate), but at 18% memory-bandwidth utilization, f16 KV is a high-confidence win.

Note: f16 KV is **not** bit-equivalent to f32 KV. A 2026 paper ([arxiv:2604.15409](https://arxiv.org/abs/2604.15409)) demonstrates deterministic token divergence between FP16-cached and cache-free paths due to FP16 non-associativity. This is the normal production baseline, not a bug.

### Phase 2: FA2 prefill kernel

**D2: Replace the sequential per-token `decode_attention` dispatch during prefill with a query-block x KV-block tiled FlashAttention-2 kernel.**

#### Online softmax with exp2 optimization

The kernel maintains per-row running state in registers: running max `m`, running denominator `l`, and output accumulator `o[BQ, D]`:

```text
for each batch b, query-head hq, query block qb:
    load Q[BQ, D] into threadgroup memory

    m[BQ] = -inf          // running max per query row
    l[BQ] = 0             // running sum per query row
    o[BQ, D] = 0          // output accumulator

    for kb in 0..kv_len step BK:
        load K[BK, D] into TGM, transposed for coalesced access
        scores[BQ, BK] = Q @ K^T * (scale * log2(e))   // pre-multiply for exp2
        apply causal mask inside tile (see D5)

        m_new[BQ] = max(m, rowmax(scores))
        p[BQ, BK] = exp2(scores - m_new)     // base-2 exp, not natural
        alpha[BQ] = exp2(m - m_new)           // rescaling factor

        l = l * alpha + rowsum(p)
        o = o * diag(alpha)                   // rescale previous accumulation

        load V[BK, D] into TGM, row-major
        o += p @ V
        m = m_new

    O[BQ, D] = o / l                         // final normalization
```

Use `exp2` (base-2 exponential) with pre-multiplied `scale * log2(e)` because Metal's `fast::exp2` avoids the slower natural-exponential hardware path. MLX Steel uses the same convention ([mlx/sdpa_vector.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h)).

#### Tile table: threadgroup memory budget

All defaults must fit within the 32 KiB TGM ceiling. TGM holds: Q tile `[BQ, D]`, K tile `[BK, D]` (transposed), V tile `[BK, D]`, softmax accumulators, and padding for bank-conflict avoidance.

Approximate threadgroup memory usage (f16 data tiles, f32 accumulators, 16-byte padding per buffer):

| `D` | `BQ` | `BK` | f16 TGM estimate | f32 TGM estimate | Status |
|----:|-----:|-----:|-----------------:|-----------------:|--------|
|  64 |   32 |   32 |          9.5 KiB |         17.5 KiB | Safe default |
|  80 |   32 |   32 |         11.8 KiB |         21.8 KiB | Safe default |
| 128 |   32 |   16 |         14.5 KiB |         26.5 KiB | Safe default |
| 128 |   32 |   32 |         18.5 KiB |         34.5 KiB | **f16 only** (f32 exceeds 32 KiB) |
|  64 |   32 |   64 |         13.5 KiB |         25.5 KiB | f16/f32 candidate |
|  80 |   32 |   64 |         16.8 KiB |         31.8 KiB | f16 candidate; f32 tight |
| 128 |   32 |   64 |         26.5 KiB |         50.5 KiB | f16 only, likely register-bound |

Starting tile table (mirrors MLX Steel instantiations):

| GPU Family | Head dim | BQ | BK | WM | WN | Threads | Rationale |
|------------|---------|---:|---:|---:|---:|--------:|-----------|
| 7-10 (M1-M4) | 64, 80 | 32 | 32 | 4 | 1 | 128 | MLX Steel default; safe TGM/register footprint |
| 7-10 (M1-M4) | 128 | 32 | 16 | 4 | 1 | 128 | Smaller BK avoids TGM/register pressure at D=128 |
| 8-10 (M2-M4) | 64, 80 | 32 | 64 | 4 | 1 | 128 | Candidate: larger traversal tile if occupancy holds |
| 9-10 (M3-M4) | 128 | 32 | 32 | 4 | 1 | 128 | f16 only; test against BK=16 default |

The 128-thread threadgroup layout (`WM=4, WN=1, 32 threads per simdgroup`) maps to Metal's simdgroup abstraction. MLX's `[[max_total_threads_per_threadgroup(WM * WN * 32)]]` attribute applies directly.

A runtime autotune table keyed by `{gpu_family, head_dim, dtype, causal, gqa_factor}` selects the optimal tile shape. Initial implementation hardcodes the safe defaults; autotune is a follow-up.

#### Simdgroup layout

128 threads organized as WM=4 warps (simdgroups) of 32 threads each. Within the FA2 tile loop:

- **QK scoring**: simdgroup matrix multiply or explicit dot-product accumulation across D dimension. Each simdgroup handles a subset of the BQ rows.
- **Softmax reduction**: per-row max and sum reductions within simdgroups, then cross-simdgroup reduction via threadgroup memory.
- **P @ V accumulation**: distribute D output dimensions across threads; each thread accumulates one or more output dimensions for its assigned query rows.

#### Threadgroup dispatch

```text
grid: (num_q_blocks, num_q_heads, batch)
threadgroup: (WM * WN * 32, 1, 1) = (128, 1, 1)

where num_q_blocks = ceil(seq_len / BQ)
```

Each threadgroup processes one BQ-sized block of queries for one query head. The total grid size is `num_q_blocks * num_q_heads * batch`.

### D3: GQA-aware tiling

For Grouped Query Attention, grid over **query heads** but load K/V by **KV head**:

```text
kv_head_idx = q_head / gqa_factor
```

For Qwen3.5-0.8B (8 Q heads, 2 KV heads, `gqa_factor=4`), each KV head serves 4 Q-head threadgroups. The K/V tiles are loaded from the KV-head's cache region.

Two implementation levels:

1. **Simple path** (ship first): one threadgroup per query-head tile. K/V tiles reloaded for each query head in the GQA group. Simple address computation, no inter-head coordination.

2. **Grouped path** (follow-up): one threadgroup handles `G_sub` query heads sharing the same KV tile. K/V loaded once, reused for multiple Q-head dot products. Reduces KV bandwidth by `G_sub` but increases Q/register pressure. Start with `G_sub=2`, not the full GQA group, unless profiling shows register headroom.

KV quantization metadata must be indexed by `(layer, kv_head, page, group)`, not by query head. This is critical for the int4 path where per-head bit allocation (RateQuant-style) calibrates across all query heads sharing a KV head.

### D4: f16 KV cache (prerequisite, separate PR)

See Phase 1 above. This is a prerequisite for efficient FA2 (halves tile load bandwidth) and for the entire KV quantization chain. Must land before the FA2 kernel work begins.

### D5: Causal masking in tiles

For causal attention, mask positions where key position exceeds query position:

```text
q_pos = q_block_start + qi       // global query position
k_pos = k_block_start + kj       // global key position
masked = causal && (k_pos > q_pos)
```

Set masked scores to `-inf` before `exp2` (they become 0 after exponentiation and do not affect the running sum). MLX Steel applies causal and general mask handling inside the tile loop rather than materializing the full attention matrix.

Three tile categories enable early exit:

1. **Tiles entirely above the diagonal** (`k_block_end <= q_block_start`): all positions are valid. Process normally, no masking overhead.
2. **Tiles entirely below the diagonal** (`k_block_start > q_block_end`): all positions are masked. Skip K/V load entirely (early exit). This saves significant work for long sequences where most tiles below the last Q block are fully masked.
3. **Partial tiles** (diagonal crosses the tile): apply element-wise mask. This is the only case that incurs per-element branching.

For prefix-cache hits where prefill starts at `matched_len`:

```text
q_pos = matched_len + local_suffix_q
k_pos = logical_k
```

This avoids off-by-one bugs when a suffix attends to cached prefix pages plus newly generated suffix KV.

### Phase 3: Paged KV + prefix cache wiring

**D6: Wire `PrefixPageCache` into `forward_prefill_impl`.**

The prefix cache is built (prefix.rs) and integrated into `PagedKVCache` (paged.rs) but not yet called from the forward path. Wiring requires:

1. Before prefill, call `restore_prefix(adapter_id, token_ids)`. On hit, `seq_len` is fast-forwarded to `matched_len`; only the suffix tokens need prefill.
2. The FA2 kernel receives a page table buffer mapping logical token positions to physical pages. Page resolution: `page_id = page_table[logical_token / page_tokens], token_in_page = logical_token % page_tokens, physical_ptr = kv_base + page_offsets[page_id] + token_in_page * stride`.
3. Choose `page_tokens` as a multiple of BK (64 or 128 tokens) so most FA2 KV tiles stay within one page.
4. After successful prefill, call `promote_to_prefix(adapter_id, full_token_ids)` to cache the result for future reuse.

The GPU never traverses the prefix tree. Metal receives only flat buffers: `kv_payload_buffer`, `kv_scale_buffer`, `page_table_buffer`, `batch_descriptor_buffer`.

### Phase 4: Fused int8 KV attention

**D7: int8 KV with fused dequantization inside the attention kernel.**

Architecture (uses the same structure as the final int4 path, but with simpler numerics):

- **K**: per-channel or per-channel-group symmetric quantization. Key channels have stable outlier structure (KIVI analysis).
- **V**: per-token, per-dim-group quantization. Values are mixed by attention weights and lack stable channel patterns.
- **Residual tail**: most recent 128 tokens in f16. Avoids per-token quantization overhead in the hot decode path and reduces quality loss on local attention.
- **Scales**: f16 initially; f32 only for calibration/debug.

Page metadata:

```rust
enum KvFormat {
    F32,
    F16,
    I8 { scheme: QuantScheme },
    I4 { scheme: QuantScheme, rotation: RotationKind },
}

struct KvPageMeta {
    layer: u16,
    kv_head: u16,
    page_id: u32,
    valid_tokens: u16,
    format: KvFormat,
    data_offset: u64,
    scale_offset: u64,
    zero_offset: u64,
    aux_offset: u64,    // rotation seeds, lambda, outlier table
}
```

Physical layout:

```text
K/V data:   [layer][page][kv_head][token_in_page][head_dim]
K scales:   [layer][page_group][kv_head][channel_group]   // per-channel K
V scales:   [layer][page][kv_head][token][dim_group]       // per-token V
```

The fused kernel loads packed int8 tiles, applies `scale * (q_int8 - zero_point)` in registers, and proceeds with the standard QK/softmax/PV flow. No global dequantization buffer in the production path (acceptable as a correctness oracle only).

The attention processes KV in three segments to avoid divergent per-token formats:
1. Compressed old prefix pages (int8)
2. f16 residual tail (last 128-256 tokens)
3. Current prefill suffix / current decode token

### Phase 5: Fused int4 with WHT/block-Hadamard rotation

**D8: int4 KV with Hadamard rotation using existing QuaRot machinery.**

QuaRot's rotation infrastructure (`crates/inference/src/quant/quarot/` -- 11 source files: hadamard.rs, rotation.rs, pipeline.rs, etc.) provides Walsh-Hadamard transforms for power-of-2 dimensions. This extends naturally to KV cache rotation:

- **WHT block size**: 32 or 64 (power-of-two butterfly over head_dim).
- **K rotation**: pair with Q rotation for dot-product invariance. `Q_rot @ K_rot^T = (R @ Q) @ (R @ K)^T = Q @ R^T @ R @ K^T = Q @ K^T` (orthogonality).
- **V rotation**: compensate by inverse rotation after `P @ V_rot`, or fold into `o_proj` weight matrix.
- **Nibble packing**: two int4 values per byte, signed symmetric `[-8, 7]`.
- **Metadata**: `bits_k[layer][kv_head]` and `bits_v[layer][kv_head]` as `u8`, with `format_for_page[layer][kv_head][page] -> KvFormat`. This enables RateQuant-style per-head mixed precision in Phase 6.

Fused kernel: load packed nibbles, unpack via `(byte >> 4)` / `(byte & 0x0f)`, sign-extend, multiply by scale, optionally apply rotation/RoPE, then proceed with standard attention.

**Pre-RoPE K quantization** (optional, int4 quality gate): quantize K before RoPE to preserve per-channel outlier structure (KVQuant insight). Cost: 3/(4G) of attention arithmetic when K RoPE is amortized per KV head group (~10-11% for Qwen's GQA group size 7). Implemented only in kernels that compute a KV-head group together, or that dequantize/RoPE K once and reuse across query heads.

### Phase 6: SRFT and RateQuant (experimental)

**D9: SRFT rotation backend and per-head bit allocation.**

- **SRFT**: sign-randomized FFT rotation as an alternative to WHT. An Apple-Silicon preprint ([arxiv:2605.05699](https://arxiv.org/abs/2605.05699)) reports -3% to -8% ms/token on Gemma-3 1B with 3x persistent memory compression. SRFT and SRHT are statistically indistinguishable for KV quality. Implement only after WHT int4 is stable; drop if it does not beat WHT on Qwen quality/speed.
- **RateQuant**: per-head bit allocation via reverse waterfilling ([arxiv:2605.06675](https://arxiv.org/abs/2605.06675)). For Qwen3-8B at 2.5 average bits: reduces KIVI PPL from 49.3 to 14.9. 1.6 seconds calibration, zero inference overhead. Do not hard-code "K lower than V" or vice versa -- let calibration decide, but expose independent K/V bit budgets per layer per KV head.

---

## Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **MPS/MPSGraph integration** | Private Apple optimizations | AMX is a CPU-side matrix coprocessor; custom Metal compute shaders cannot invoke it. MLX attention is open Metal, not a black-box MPSGraph call. | Rejected -- no actionable private accelerator gap |
| **Separate dequant-to-f16 global buffer** | Simple implementation | Writes and rereads f16 intermediates, throwing away bandwidth advantage. Open-TQ-Metal reports 48x attention speedup at 128K with fused vs dequant-then-attend. | Rejected as primary path; acceptable as correctness oracle |
| **f32 TGM for all tile configurations** | Simplest kernel | `BQ=32 BK=32 D=128` at f32 needs 34.5 KiB, exceeding 32 KiB limit. Forces smaller tiles for the most common head dimension. | Rejected -- f16 TGM with f32 accumulators is the right split |
| **CUDA-faithful FA2 port** | Matches paper exactly | CUDA warp/tensor-core mapping does not apply to Apple GPU simdgroups. | Rejected -- use MLX Steel tile shapes, not CUDA shapes |
| **Full radix tree prefix cache** | Optimal token-boundary prefix matching | Complex in safe Rust; interior mutability, leaf LRU. | Deferred -- hash map covers single-user case per ADR-047 |
| **Post-RoPE K quantization only** | Simpler kernel (no per-tile RoPE) | RoPE disrupts per-channel outlier structure; int4 quality degrades. | Acceptable for int8; re-evaluate at int4 quality gate |

---

## Expected Speedup Envelope

| Change | Attention effect | End-to-end expectation |
|---|---|---|
| f32 KV -> f16 KV | Up to 2x lower KV bandwidth | +15-60% decode throughput, workload-dependent |
| Sequential prefill -> FA2 prefill | Large TTFT improvement at long prompts | 1.3-2x TTFT improvement for long prefill-heavy requests |
| f16 KV -> fused int8 KV | 2x lower KV payload than f16 | Long-context decode improvement; smaller at short context |
| Fused int4 + rotation + f16 tail | ~4x lower payload than f16 plus scale overhead | Can beat f16 at long context if fused; validate on Qwen |
| Prefix cache wiring | Skips matched prefill | Can dominate TTFT for repeated prompts |

Realistic same-hardware target after f16 KV + MLX-style FA2: **~180-230 tok/s** decode (current: ~118 tok/s). With fused int4 at longer contexts, **MLX-level or better** decode throughput is plausible (MLX: ~259 tok/s), but should be treated as a benchmark goal, not a guaranteed sum of independent speedups.

---

## Implementation Order

### Phase 0 -- Shader extraction

**Deliverables**: `.metal` source tree, Rust shader registry, macOS CI `xcrun metal -c` validation, pipeline cache keyed by `{kernel, dtype, D, BQ, BK, GPU family}`.

**Acceptance**: All existing tests pass with extracted shaders. No performance regression.

### Phase 1 -- f16 KV

**Deliverables**: f16 K/V page format, f32 reference path, f16 attention read path, PPL/token-agreement report vs f32.

**Acceptance**: No correctness regressions outside expected fp precision drift (f16 vs f32 PPL delta < 0.05 based on community bounds). Measurable bandwidth reduction in decode.

### Phase 2 -- MLX-style FA2 prefill

**Deliverables**: `fa2_prefill_f16` kernel, tile variants for D=64/80/128, causal and non-causal modes, GQA mapping, page-table-ready addressing (even if initially contiguous).

**Acceptance**: Beats current prefill at 2K+ sequence lengths. Preserves current decode path for M=1. PPL matches current prefill path.

### Phase 3 -- Paged KV + prefix cache

**Deliverables**: Page table buffer in Metal, `PrefixPageCache` lookup in `forward_prefill_impl`, suffix-only prefill on hit, page insertion/eviction/refcounting.

**Acceptance**: Repeated prompt TTFT drops roughly proportional to skipped prefix. No GPU tree traversal.

### Phase 4 -- Fused int8 KV attention

**Deliverables**: int8 page format, K per-channel scales, V per-token scales, f16 residual tail, fused decode attention, f16 oracle comparison.

**Acceptance**: Near-f16 quality (PPL delta < 0.1). Faster than f16 attention at long context. No global dequant buffer in production path.

### Phase 5 -- Fused int4 WHT/block-Hadamard

**Deliverables**: int4 nibble-packed page format, WHT/block-Hadamard rotation, Q/K rotation consistency, V inverse/folded compensation, fused decode attention, fused quant-write kernel.

**Acceptance**: Qwen PPL and LongBench/RULER/NIAH within threshold. Faster than f16 KV at long context. Residual f16 tail enabled.

### Phase 6 -- SRFT and RateQuant

**Deliverables**: SRFT experimental rotation backend, RateQuant-style calibration tool, per-layer/per-KV-head K/V bit budgets, mixed-format page support.

**Acceptance**: Beats WHT on Qwen quality/speed, or is dropped.

---

## Consequences

### Positive

- **TTFT improvement**: FA2 tiling is the single biggest TTFT improvement available. Expect 2-5x prefill speedup at 4K+ context.
- **KV quantization chain**: f16 -> int8 -> int4 gives 2x -> 4x -> 8x KV memory reduction, enabling longer contexts within the same memory budget.
- **Reuses existing infrastructure**: WHT rotation reuses QuaRot's Hadamard machinery (quarot/hadamard.rs, quarot/rotation.rs). Paged KV reuses ADR-047's PagedKVCache and PrefixPageCache.
- **Shader extraction** enables per-kernel profiling, isolated testing, and future model architecture support (Llama, Gemma) without touching the monolithic file.

### Negative

- **Shader extraction** is a large refactor of a 15.5K-line file that must preserve exact numerical behavior.
- **f16 KV** introduces an f32/f16 format split throughout the cache layer, adding complexity to every consumer.
- **Pre-RoPE K quantization** adds per-tile RoPE compute at attention time (~10% overhead for Qwen GQA); only justified at int4 quality gate.
- **Metal shader debugging** is slow (no printf, limited GPU capture). MLX Steel and Draw Things FA2 as reference implementations reduce risk but do not eliminate it.

### Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| TGM pressure prevents larger tiles on some GPU families | Medium | Start with conservative BK=16/32; runtime autotune table |
| f16 KV introduces token divergence vs f32 | Expected | Documented in literature; PPL delta < 0.05; f32 reference path preserved |
| Shader extraction changes numerical behavior | Low | Bit-for-bit comparison tests against pre-extraction outputs |
| int4 quality insufficient without pre-RoPE K | Medium | Gate on PPL/token-agreement; post-RoPE int8 as fallback |
| SRFT complexity not justified by quality/speed gain | Medium | WHT is the production path; SRFT is explicitly experimental and droppable |
| RateQuant calibration is model-specific | Low | Expose per-head bit budgets as metadata; calibration tool is offline |

---

## References

### Papers

- Dao, _FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning_, 2023, [arxiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Liu et al., _KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache_, 2024, [arxiv:2402.02750](https://arxiv.org/html/2402.02750v2)
- Hooper et al., _KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization_, 2024, [openreview:0LXotew9Du](https://openreview.net/forum?id=0LXotew9Du)
- Ashkboos et al., _QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs_, NeurIPS 2024, [openreview:dfqsW38v1X](https://openreview.net/forum?id=dfqsW38v1X)
- Apple SRFT KV Compression preprint, 2026, [arxiv:2605.05699](https://arxiv.org/abs/2605.05699)
- Open-TQ-Metal (fused int4 Metal attention), 2026, [arxiv:2604.19157](https://arxiv.org/abs/2604.19157)
- FP16 KV cache divergence analysis, 2026, [arxiv:2604.15409](https://arxiv.org/abs/2604.15409)
- RateQuant (per-head bit allocation), 2026, [arxiv:2605.06675](https://arxiv.org/abs/2605.06675)
- TurboQuant (online vector quantization for KV), 2025, [arxiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Zheng et al., _SGLang: Efficient Execution of Structured Language Model Programs_ (RadixAttention), NeurIPS 2024, [ar5iv:2312.07104](https://ar5iv.org/html/2312.07104v2)

### Implementation references

- MLX Steel attention template: [steel_attention.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h)
- MLX vector SDPA (exp2 convention): [sdpa_vector.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h)
- MLX kernel build system: [CMakeLists.txt](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/CMakeLists.txt)
- Draw Things Metal FlashAttention: [engineering.drawthings.ai](https://engineering.drawthings.ai/p/integrating-metal-flashattention-accelerating-the-heart-of-image-generation-in-the-apple-ecosystem-16a86142eb18)
- Apple Metal Feature Set Tables: [Metal-Feature-Set-Tables.pdf](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

### Lattice code

- `crates/inference/src/forward/metal_qwen35.rs` -- current 15,546-line Metal shader + dispatch file
- `crates/inference/src/kv_cache/flat.rs` -- FlatKVCache (f32-only)
- `crates/inference/src/kv_cache/paged.rs` -- PagedKVCache with prefix integration
- `crates/inference/src/kv_cache/prefix.rs` -- PrefixPageCache (built, unwired to forward path)
- `crates/inference/src/quant/quarot/` -- QuaRot rotation infrastructure (WHT, absorption, pipeline)

### ADRs

- ADR-044: QuaRot -- Hadamard-Rotated 4-bit Quantization
- ADR-047: Paged KV Cache with Prefix Reuse
- ADR-058: Performance Regression Gate

### KG entities

- `Metal FA2 Prefill` (0dfbc841), `FlashAttention-2` (63602a7f), `Metal Fused Attention` (48ee18b2)
- `Online Softmax` (8e0e5157), `exp2 Online Softmax` (a55e26b9)
- `Apple GPU Threadgroup Memory` (078a8ab4), `Tile Size Family` (7eb265aa), `MLX Steel Attention` (024c7d60)
- `GQA-Aware Tiling` (0a4e7a6b), `GQA` (36e42eb8), `Causal Masking` (86a48852)
- `f16 KV Cache` (5c9c51ca), `int8 KV Cache` (5c4aad3e), `int4 KV Cache` (1258ab2e)
- `Fused Quantized Attention` (c3d84793), `KIVI` (0543a8f8), `KVQuant` (8d870807)
- `Per-Channel KV Quantization` (01d46f27), `Per-Token KV Quantization` (42cf9ef6)
- `Pre-RoPE K Quantization` (7ae64f86), `WHT for KV Rotation` (c9e6564a), `SRFT` (01916441)
- `RateQuant` (e6e8c564), `QuaRot` (e754741e), `Chunked Prefill` (018193b3)
- `Draw Things Metal FA2` (9fb15ed1), `TurboQuant` (f43f50b4)
- `Sarathi-Serve` (cc6c5094), `POD-Attention` (f59c045f)
- `lattice` (1c51f097), `lattice-inference` (6c0a97df), `PagedAttention` (c1d9f859), `RoPE` (e6357762)
