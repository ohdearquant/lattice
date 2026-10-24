# ADR-053: MoE Metal Dispatch with Expert Coalescing

**Status**: Accepted
**Date**: 2026-05-19
**Crate**: lattice-inference

---

## Amendment (2026-10-24): routing moves to the device on the decode route

**Status of this amendment**: Proposed

### What the shipped decode route does today

The default Metal decode step encodes every layer of a token, MoE layers included, into one
command buffer and commits and waits on it once, after the final head
(`forward/metal_qwen35/inner/step.rs`, the default branch of the forward step). Per-layer
command buffers with a commit and wait each exist only on the profiling path
(`LATTICE_DECODE_PROFILE`). The 40 commit-and-wait cycles priced under "The dispatch cost
problem", and the per-layer `commit() + wait_until_completed()` in the component diagram, do not
describe the shipped default route.

Inside that single buffer, the MoE router runs on the CPU while the layer is being encoded
(`inner/layers/mod.rs`, the router logits loop and the shared-expert sigmoid). It reads
`activations.hidden`, but the post-attention norm that writes that buffer is GPU work encoded
earlier in the same, not yet committed, command buffer, so the CPU sees the previous step's
values. Correct CPU routing therefore needs the buffer committed and waited on after the norm of
every MoE layer: one host round trip per MoE layer per token, added to the one token-end wait.
On an Apple M4, one commit-to-completion round trip carrying a single blit has a median of
155.2 µs (n = 50,000; issue #2211), about 6.2 ms per token across 40 MoE layers at that median.
That is an illustration at the median, not a lower bound and not a measured decode saving.
D3's rationale ("routing on CPU eliminates a GPU→CPU readback") holds only when the router's
input is host-visible before encoding, which it is not on this route.

### D3′ (supersedes D3 on the Eager decode route)

**Router, top-k selection and the shared-expert gate run on the GPU, inside the token's command
buffer.** A router kernel reads the current post-attention norm output and computes the expert
logits, full softmax, ordered top-k and renormalization. It writes K expert ids, K weights and
the K element offsets into the routed-expert buffers to a preallocated per-layer route buffer. A
second kernel computes the shared-expert gate scalar (dot product and sigmoid) from the same
current input. The expert dispatch encodes a fixed K rank iterations whose GEMVs read the
GPU-written offsets and whose accumulation reads the GPU-written weights, in rank order, so the
accumulation order is deterministic. No host read of the hidden state or of route data happens
inside the layer loop; the token-end wait is the only wait.

### Index and offset safety

Device-written ids and offsets are a new memory-safety surface, so these are invariants of the
router kernel, checked by tests:

- Every written expert id is in `[0, num_experts)` for every input, including NaN and infinite
  logits and exact ties. Non-finite logits are ordered by a fixed rule (NaN ranks below every
  finite value; ties go to the lower expert id), so the kernel always writes K valid, distinct
  ids.
- Offsets are computed on the device from those ids, in elements, in the same unit as the
  existing expert GEMV (u32 element offsets, passed as the `moe_expert_gemv` kernel's own
  `W_elem_offset` argument at buffer(4), `shaders/qwen35.metal` around line 2425, separate from
  `GemmParams`, which carries only m, n, k, lda, ldb, ldc; the host computes them in
  `metal_qwen35/inner/layers/mod.rs` around line 613: gate half `e·2·inter·hidden`, up half
  `+inter·hidden`, down `e·hidden·inter`). At route-buffer
  construction the host checks that the largest offset plus one expert's extent fits both u32
  and the bound buffer's length, and refuses to build the device route otherwise. The kernel
  clamps nothing: an id outside the range is a kernel defect caught by the tests below, never a
  silently redirected read.
- Each written weight is finite and the K weights sum to one within f32 rounding.

### Consequences for the other decisions

- **D1's intent holds** (no per-expert or per-encoder submits). The shipped route already goes
  further than D1 as written: one command buffer per token. D3′ keeps that, and the router and
  gate kernels are encoded before the expert work with explicit dependencies (barriers or
  separate encoders), never inferred from host completion.
- **D2 holds** for the first route: the contiguous expert-major F16 buffers ("Eager" storage)
  are addressable by any expert id, so GPU-written offsets into them are valid without host
  involvement. Indirect command buffers are not required.
- **D4 holds in part.** The shared expert keeps its dedicated buffers. Its stated rationale,
  overlapping encode time with CPU routing, no longer applies on the device route; the reason to
  keep it is that the shared expert does not depend on the route. Ordering: the shared-gate
  kernel writes its scalar before the shared-expert output is scaled by it, and both precede the
  accumulation into the layer output.
- **The Cached Q4 storage route is not covered by D3′.** It admits experts into a bounded cache
  on the host when they are selected. Moving selection to the GPU there needs a device-side miss
  path (a correct packed-Q4 expert kernel for experts not in the cache, and per-token frozen
  cache-slot ownership so no in-flight reader's slot is overwritten). That design is a separate
  amendment before its implementation. Until then the Cached route keeps the synchronized CPU
  router, and is labelled as such wherever its decode time is reported.
- **The routing trace** is published from the route buffer after the token-end wait, with the
  original layer index, token position, rank order and normalized weights. Previous-token routes
  may inform cache residency hints; they never substitute for the current token's route.
- **The synchronized CPU router stays as a labelled fallback** for storage or devices the device
  route does not support. A run on the fallback is not a measurement of the device route.

Superseded on the Eager decode route, still in force for Cached Q4 and the CPU fallback: D3, the
Scope v1 "CPU router" bullet, the router lines of the component diagram, R4's "run the CPU NEON
kernel ... avoiding a separate GPU router kernel", and the "GPU router" row under Alternatives
(its rejection assumed a readback before encoding, which D3′ avoids). Each carries a note in
place.

### Acceptance for D3′ (registered before implementation merges)

1. The routing freshness test (the selected experts match the analytic choice from the current
   layer's norm output) passes on the device route, with a marker that fails the test if the
   synchronized CPU router ran instead.
2. A shared-expert witness with nonzero shared-expert output fails when the shared gate reads a
   stale input and passes on the device route. (The existing freshness fixture zeroes the
   shared-expert projections, so it cannot detect a stale shared gate.)
3. Ties, non-finite logits, multi-K selection and pruned layers keep the numerical contract
   above. A forced router row containing NaN and infinite logits, and a forced out-of-range id
   written into the route buffer, each fail the test before any expert GEMV reads an offset
   derived from them.
4. Decode time per token on a real MoE checkpoint, before and after, on the same machine. D3′
   covers Eager F16 only, and the target checkpoint does not fit Eager F16 on the machines
   available now: Qwen3.6-35B-A3B is about 35B parameters, about 70 GB at F16, while the Apple M4
   machine used for Metal measurements has 16 GB of unified memory, so a real checkpoint there runs
   the Cached Q4 route, which D3′ does not cover. Item 4 therefore waits for a machine whose
   unified memory holds the Eager F16 checkpoint with headroom, or for the Cached amendment. The
   device route merges on items 1 to 3, and no real-checkpoint decode figure for it is published
   until item 4 runs. The measurement program is prepared now: a raw MoE decode mode in
   `bench_decode_ab` (token-by-token prefill outside the timer, then N timed single-token forward
   steps with greedy selection, reported as raw forward time rather than serving throughput),
   with the correctness-fixed CPU route as the baseline.

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

> Amendment (2026-10-24): the shipped default route submits one command buffer per token, not
> per layer; per-layer buffers exist only on the profiling path. See the amendment above.

### The dispatch cost problem

M2 Max measured Metal dispatch overhead: ~0.127 ms per `CommandBuffer` commit+wait cycle.
Qwen3.6 has 40 layers, each with a MoE FFN.

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

> Amendment (2026-10-24): the shipped default route submits one command buffer per token, not
> per layer; see the amendment above for what the per-layer round trip costs when the CPU router
> forces one.

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

_Superseded on the Eager decode route by D3′ (2026-10-24 amendment): routing moves to the device
and the experts dispatch from a device-written route buffer, so no readback precedes encoding.
D3 stays in force for Cached Q4 and the CPU fallback._

**D4: Shared expert weight is pinned to a dedicated Metal buffer.**
The shared expert always activates. Its weights (`gate_proj`, `up_proj`, `down_proj`) are
allocated as a separate `MTLBuffer` at model-load time, distinct from the routed expert
buffer. This allows the shared expert GEMV to be the first encoder in every command buffer,
overlapping encode time with routing on CPU.

_Holds in part after the 2026-10-24 amendment: the dedicated buffer stays, and the shared expert
still runs independently of the route because it does not depend on the selected ids. The
overlap with CPU routing no longer applies on the Eager decode route, where routing is on the
device; the shared gate is computed before its scale is applied, as D3′ describes._

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
  `pub(crate)` promotion). _Superseded on the Eager decode route by D3′ (amendment above);
  in force for Cached Q4 and the CPU fallback._
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

> Amendment (2026-10-24): the three CPU router lines are superseded on the Eager decode route by
> the D3′ router and gate kernels, and the shipped route commits and waits once per token rather
> than once per layer.

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

| Alternative                                                                          | Rationale for rejection                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Per-expert command buffers (naive)                                                   | 360 dispatches × 0.127 ms = 45.7 ms/token. Mathematically excluded by measured dispatch overhead.                                                                                                                                                                                |
| MLX backend (call Apple's MPS via MLX Rust bindings)                                 | MLX is 3× faster than llama.cpp on MoE on Apple Silicon (measured: Ollama/MLX 58→112 tok/s on Qwen3.5-35B-A3B). But adding MLX as a dependency contradicts the pure-Rust, no-ONNX constraint in AGENTS.md. Not an option in v1. Track as v2 option if native Metal gap persists. |
| GPU router (move top-k softmax to Metal), adopted as D3′ by the 2026-10-24 amendment | Eliminates the CPU→GPU branch for expert selection, but requires a GPU→CPU readback of `[num_experts]` logits before command encoding can start. At batch=1 the GEMV cost (~0.05 ms on NEON) does not justify the synchronization. Reconsider if batch>1 becomes the target.     |
| Pre-gated routing (predict expert activations one step ahead)                        | DeepSeek-V2/V3 uses this to prefetch expert weights. Effective at hiding memory latency for weight-swapping configurations. Unnecessary when all weights are resident in unified memory. Defer to v2 with expert swapping.                                                       |
| Expert weight re-layout at load (token-major)                                        | Would allow a single fused `[batch, top_k, inter, hidden]` GEMM. Requires a full re-layout of ~15 GiB of weights at load time and a different buffer structure. No benefit at batch=1 (decode). Defer to v2 prefill path.                                                        |
| Shared expert caching with redundancy elimination                                    | DeepSeek caches the shared expert output across consecutive tokens when input hasn't changed. Valid optimization but requires state tracking that doesn't exist in the current decode loop. Defer.                                                                               |

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
_Superseded on the Eager decode route by D3′: the router gate weight stays f32 in its own buffer
but is read by the GPU router kernel. In force for Cached Q4 and the CPU fallback._

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
