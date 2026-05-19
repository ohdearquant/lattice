# ADR-053: MoE Metal Dispatch with Expert Coalescing

**Status**: Proposed
**Date**: 2026-05-19
**Crate**: lattice-inference

---

## Context

Qwen3.6-35B-A3B (256 routed experts, top-8, 1 shared expert) is the primary MoE target in
lattice-inference. The CPU MoE forward path already exists: `model/qwen35/moe.rs` implements
`moe_ffn_step` which routes tokens through `accumulate_routed_experts` (top-k loop) and
`apply_shared_expert`. Weight layout in `weights.rs` is expert-major: `RoutedExperts::gate_up_proj`
is `[num_experts, 2 * intermediate_size, hidden_size]` contiguous, and `down_proj` is
`[num_experts, hidden_size, intermediate_size]` contiguous.

The Metal forward path (`forward/metal_qwen35.rs`) handles the Qwen3.5-2B hybrid model. The
`metal_qwen35` module encodes one Metal `CommandBuffer` per layer and already ships fused
decode-mode GEMV kernels (`gemv_decode_core` with simdgroup reduction), tiled GEMM for prefill,
and a GQA flash-attention kernel. **MoE layers are not yet handled** — the Metal path falls back
to CPU for any `FeedForwardWeights::Moe` variant.

### The dispatch cost problem

M2 Max measured Metal dispatch overhead: ~0.127 ms per `CommandBuffer` commit+wait cycle
(KG entity `Metal MoE Dispatch Analysis`). Qwen3.6 has 40 layers, each with a MoE FFN.

Naive per-expert dispatch:
- 8 active routed experts × 3 matrix operations each (gate_up, silu_glu, down) = 24 dispatches
- Plus 1 shared expert (3 dispatches)
- Per layer total: 27 dispatches × 0.127 ms = ~3.4 ms
- 40 layers: **136 ms overhead per token** from dispatch alone — this exceeds the entire decode
  budget for a 42–50 tok/s target.

Even a reduced framing of 1 command buffer per expert:
- 9 experts × 40 layers = 360 dispatches × 0.127 ms = **45.7 ms/token** — still 2× the budget.

The only viable path is 1 command buffer per layer encoding all expert work as multiple
compute-command encoders within a single commit. This reduces dispatch overhead to
40 × 0.127 ms = **5.1 ms/token** regardless of expert count.

### Expert weight memory for Qwen3.6-35B-A3B

**Current `Qwen35Config::qwen36_35b_a3b()`** sets `hidden_size: 2048`, `moe_intermediate_size: 512`,
`shared_expert_intermediate_size: 512` with 256 routed experts across 40 layers.

At Q4 with the current config dimensions:
- `gate_up_proj`: 256 experts × 2 × 512 × 2048 × 0.5 bytes = ~256 MiB per layer
- `down_proj`: 256 experts × 2048 × 512 × 0.5 bytes = ~128 MiB per layer
- Total routed expert weights across 40 layers: ~384 MiB × 40 ≈ **15 GiB**
- Shared expert (dense, 512×2048): negligible (~4 MiB)

> **Note**: the local `Qwen35Config::qwen36_35b_a3b()` fixture matches the current upstream
> HF `Qwen/Qwen3.6-35B-A3B` config. Re-verify this section if the target model changes.

M2 Max unified memory is 32–96 GB. Loading all expert weights resident (the MLX approach) is
feasible on 64 GB+ configurations for either dimension set. This ADR targets the resident-all
path as v1; swapping is deferred.

### QuaRot interaction

ADR-044 stores rotated weights per-projection within the `RotationPlan`. For dense layers,
QuaRot absorbs residual-stream rotation into `gate_proj`, `up_proj`, and `down_proj`
individually. For MoE, the same rotation must be applied **per-expert**: each expert's
`gate_up_proj[e]` absorbs input-side rotation and `down_proj[e]` absorbs output-side rotation.

The current `RotationPlan` in `quant/quarot/plan.rs` has no MoE-aware rules — all plan entries
assume non-expert projections addressed by their HF tensor name. Adding expert-indexed rules
requires either parameterized names (e.g., `model.layers.{l}.mlp.experts.{e}.gate_proj`) or a
separate `MoeRotationPlan` that iterates expert indices. The weight layout being expert-major
contiguous (`[num_experts, 2*inter, hidden]`) means the conversion binary can apply rotation
expert-by-expert in sequence without re-layouting.

---

## Decision

Implement Metal MoE dispatch with expert coalescing as a new function `metal_moe_ffn_step` in
`forward/metal_qwen35.rs` (inside the existing `#[cfg(all(target_os = "macos", feature =
"metal-gpu"))]` module). The design has four binding decisions:

**D1: One command buffer per MoE layer, multiple compute encoders within it.**
All routed expert matmuls are encoded into a single `CommandBuffer` before commit. This is
the only approach that stays below the dispatch latency floor. Each compute encoder encodes one
expert's `gate_up` GEMV, silu+glu, and `down` GEMV as separate `dispatchThreadgroups` calls
within the same encoder (no intermediate commits). Encoders execute serially on Apple Silicon
because the GPU has no inter-encoder dependency tracking — parallel expert execution requires
either atomic accumulation or a two-pass reduce, which adds complexity with no latency win at
batch size 1 (decode mode).

**D2: Expert-major weight layout stays; no re-layout at load time.**
`RoutedExperts::gate_up_proj` is already `[num_experts, 2*inter, hidden]` contiguous. Metal
buffer binding uses byte offsets into this buffer: expert `e`'s gate half starts at byte
`e * 2 * inter * hidden * element_size`, and the up half at `(e * 2 * inter + inter) * hidden *
element_size`. This avoids a copy or re-layout step, and the kernel dispatch geometry is
identical to the existing `gemv_decode_core` kernel (M=1, N=inter or hidden, K=hidden or inter).

**D3: Router (top-k softmax) runs on CPU.**
For decode-mode (batch size 1), the router matmul is `[1, hidden] × [num_experts, hidden]^T` →
`[1, num_experts]` — a 2048×256 GEMV. At Q4 this is ~0.25 MB of weight data. CPU NEON handles
this in ~0.05 ms; shipping it to the GPU costs a synchronization boundary before expert
selection can proceed. More importantly, expert selection is a control-flow branch: the CPU
must know which experts are active before encoding any of their compute commands into the
command buffer. Routing on CPU eliminates a GPU→CPU readback that would serialize the entire
dispatch.

**D4: Shared expert weight is pinned to a dedicated Metal buffer.**
The shared expert always activates. Its weights (`gate_proj`, `up_proj`, `down_proj`) are
allocated as a separate `MTLBuffer` at model-load time, distinct from the routed expert
buffer. This allows the shared expert GEMV to be the first encoder in every command buffer,
overlapping encode time with routing on CPU.

---

## Scope

v1 (this ADR):

- `metal_moe_ffn_step`: fused MoE forward for decode mode (M=1), integrated into the Metal
  layer loop alongside the existing `FeedForwardWeights::Dense` branch.
- MSL kernel `moe_expert_gemv`: single-expert GEMV accepting an expert-offset parameter so one
  compiled kernel handles all experts. Reuses the `gemv_decode_core` template already in
  `MSL_SOURCE`.
- CPU router: top-k selection integrated into `metal_moe_ffn_step` before encoding (calling
  existing `compute_router_probs` / `select_top_k` / `renormalize_selected` from `moe.rs` via
  `pub(crate)` promotion).
- Expert coalescing: encode 1 shared + top_k routed expert encoders into 1 command buffer per
  layer.
- Memory layout: no change to `RoutedExperts` or `SharedExpert` structs. New field
  `MoeMetalBuffers` holds pre-allocated `MTLBuffer` handles for the three expert weight tensors.

Deferred to v2:

- Prefill path (M>1): batched GEMM across experts. Not needed until continuous batching targets
  MoE models (ADR-048 covers the scheduler side).
- Expert weight swapping: LRU eviction for 32 GB configurations.
- QuaRot-on-MoE: per-expert rotation absorption in `quantize_quarot` binary. Requires
  parameterized rotation plan rules. ADR-044 v1 deferred this explicitly.
- Pre-gated routing (predict activations before token to prefetch weights).

---

## Architecture

### Component diagram

```
moe_ffn_step (CPU, existing)
    └── accumulate_routed_experts   [CPU fallback, decode only]

metal_moe_ffn_step (new)
    ├── compute_router_probs()      [CPU, reuse from moe.rs]
    ├── select_top_k()              [CPU, reuse from moe.rs]
    ├── renormalize_selected()      [CPU, reuse from moe.rs]
    ├── CommandBuffer::new()        [one per layer]
    │   ├── encode shared expert
    │   │   ├── gemv: gate_up [inter, hidden]
    │   │   ├── silu_glu in-place
    │   │   └── gemv: down [hidden, inter]
    │   ├── encode expert[selected[0]]
    │   │   └── (same pattern, offset into RoutedExperts buffer)
    │   ├── ...
    │   └── encode expert[selected[top_k-1]]
    └── CommandBuffer::commit() + wait_until_completed()
```

### Metal buffer layout

```
MoeMetalBuffers {
    routed_gate_up: MTLBuffer,  // [num_experts * 2 * inter * hidden] f16
    routed_down:    MTLBuffer,  // [num_experts * hidden * inter] f16
    shared_gate_up: MTLBuffer,  // [2 * shared_inter * hidden] f16
    shared_down:    MTLBuffer,  // [hidden * shared_inter] f16
    gate_weight:    MTLBuffer,  // [num_experts * hidden] f32 (router)
    scratch_gate:   MTLBuffer,  // [inter] f32 (reused across experts)
    scratch_up:     MTLBuffer,  // [inter] f32
    scratch_out:    MTLBuffer,  // [hidden] f32 (accumulator)
}
```

### Kernel interface

```msl
kernel void moe_expert_gemv(
    device const float*    x         [[buffer(0)]],  // [hidden] activation
    device const half*     W         [[buffer(1)]],  // full expert weight buffer
    device float*          out       [[buffer(2)]],  // [N] output
    constant GemmParams&   p         [[buffer(3)]],  // M=1, N, K, offsets
    constant uint&         W_offset  [[buffer(4)]],  // element offset into W for this expert
    uint gid [[threadgroup_position_in_grid]],
    ...)
```

`W_offset = expert_id * 2 * inter * hidden` for gate half,
`W_offset = expert_id * 2 * inter * hidden + inter * hidden` for up half.
Reuses `gemv_decode_core<128>` instantiation already in `MSL_SOURCE`.

### Accumulation

Each routed expert result is scaled by its router weight before accumulation. Two options:

- **GPU accumulation**: encode a `scale_add` kernel after each expert's `down` GEMV that reads
  the router weight (a scalar from a small buffer) and accumulates into `scratch_out`.
- **CPU accumulation**: GPU writes each expert's down output to a per-expert scratch slot; CPU
  reads back all top_k results after command buffer completion and accumulates with weights.

For top_k=8 and hidden=2048: CPU readback is 8 × 2048 × 4 bytes = 64 KB. At unified memory
this is a pointer dereference, not a PCIe transfer — readback cost is negligible. GPU
accumulation avoids even this, at the cost of one additional `scale_add` kernel encode per
expert (8 extra dispatches within the same command buffer — negligible vs. the GEMV dispatches).

Decision: GPU accumulation (`scale_add` within command buffer) to keep CPU/GPU data flow
unidirectional and avoid any explicit readback synchronization.

### Integration point

In the existing Metal layer loop (inside `metal_qwen35.rs`), the `FeedForwardWeights::Moe`
arm currently panics or falls through to CPU. After this ADR:

```rust
match &layer.common.ffn {
    FeedForwardWeights::Dense(w) => metal_dense_ffn(w, buffers, cfg),
    FeedForwardWeights::Moe(w)   => metal_moe_ffn_step(w, moe_bufs, scratch, cfg),
}
```

---

## Alternatives Considered

| Alternative | Rationale for rejection |
|---|---|
| Per-expert command buffers (naive) | 360 dispatches × 0.127 ms = 45.7 ms/token. Mathematically excluded by measured dispatch overhead. |
| MLX backend (call Apple's MPS via MLX Rust bindings) | MLX is 3× faster than llama.cpp on MoE on Apple Silicon (measured: Ollama/MLX 58→112 tok/s on Qwen3.5-35B-A3B). But adding MLX as a dependency contradicts the pure-Rust, no-ONNX constraint in AGENTS.md. Not an option in v1. Track as v2 option if native Metal gap persists. |
| GPU router (move top-k softmax to Metal) | Eliminates the CPU→GPU branch for expert selection, but requires a GPU→CPU readback of `[num_experts]` logits before command encoding can start. At batch=1 the GEMV cost (~0.05 ms on NEON) does not justify the synchronization. Reconsider if batch>1 becomes the target. |
| Pre-gated routing (predict expert activations one step ahead) | DeepSeek-V2/V3 uses this to prefetch expert weights. Effective at hiding memory latency for weight-swapping configurations. Unnecessary when all weights are resident in unified memory. Defer to v2 with expert swapping. |
| Expert weight re-layout at load (token-major) | Would allow a single fused `[batch, top_k, inter, hidden]` GEMM. Requires a full re-layout of ~15 GiB of weights at load time and a different buffer structure. No benefit at batch=1 (decode). Defer to v2 prefill path. |
| Shared expert caching with redundancy elimination | DeepSeek caches the shared expert output across consecutive tokens when input hasn't changed. Valid optimization but requires state tracking that doesn't exist in the current decode loop. Defer. |

---

## Risks

**R1: Decode throughput estimate is a model, not a measurement.**
The 42–50 tok/s estimate for M2 Max is derived from dispatch overhead math plus the existing
GEMV kernel throughput numbers (measured for dense layers in `metal_qwen35.rs` benchmarks). Actual
MoE throughput depends on expert weight bandwidth utilization across 8 concurrent GEMV encoders
in the same command buffer. Measure after implementation; do not pin the number until verified.

**R2: Expert memory pressure on 32 GB M2 Pro.**
At ~15 GiB for routed expert weights plus ~2 GB for attention state and activation buffers, the
35B MoE model is borderline on 32 GB. v1 will document the minimum configuration (64 GB
recommended) and reject at load time with a clear error if total buffer allocation would
exceed a configurable threshold (default: 0.85 × device.recommendedMaxWorkingSetSize).

**R3: QuaRot incompatibility.**
ADR-044 explicitly deferred per-expert rotation absorption. A QuaRot-converted Qwen3.6 model
fed into the Metal MoE path will produce wrong outputs because the rotation plan does not cover
expert-indexed projections. Mitigation: the loader must check for the QuaRot conversion marker
in `quantize_index.json` and refuse to load a QuaRot-converted MoE model on the Metal path until
ADR-044 v1 lands. No silent wrong results.

**R4: Router weight dtype mismatch.**
The router gate weight (`MoeRouter::gate`) is `Vec<f32>`. The existing CPU path reads it
directly. For Metal, the gate weight is also small enough (256 × 2048 × 4 bytes = 2.1 MB) to
keep as f32 in its own `MTLBuffer` and run the CPU NEON kernel against the shared-memory pointer,
avoiding a separate GPU router kernel. Confirm the `contents()` pointer is CPU-readable when
using `MTLStorageModeShared` — it is, per the existing weight buffer pattern in `metal_qwen35.rs`.

---

## References

- Existing MoE CPU path: `crates/inference/src/model/qwen35/moe.rs`
- Weight structs: `crates/inference/src/model/qwen35/weights.rs` — `MoeLayerWeights`,
  `RoutedExperts`, `SharedExpert`, `MoeRouter`
- Config: `crates/inference/src/model/qwen35_config.rs` — `Qwen35Config::num_experts`,
  `moe_intermediate_size`, `shared_expert_intermediate_size`
- Metal GEMV kernel template: `crates/inference/src/forward/metal_qwen35.rs` —
  `gemv_decode_core<TG_THREADS>` in `MSL_SOURCE`
- Dispatch pattern: `crates/inference/src/forward/gpu/dispatch.rs` — `dispatch_matmul` for
  command-encoder structure reference
- ADR-044: QuaRot quantization — expert-indexed rotation plan deferred; R3 above
- ADR-048: Continuous batching — prefill-mode MoE GEMM deferred until scheduler targets MoE
- Qwen3.6-35B-A3B HuggingFace config: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- MLX MoE benchmark: Ollama 0.19 (MLX backend), 58→112 tok/s on Qwen3.5-35B-A3B on M3 Max
- DeepSeek-V2 MoE architecture: <https://arxiv.org/abs/2405.04434> (shared expert + pre-gating)
- KG entities: `Metal MoE Dispatch Analysis`, `Expert Coalescing Strategy`
