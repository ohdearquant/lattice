# ADR-045: QuaRot + LoRA Composition at Inference Time

**Status**: Accepted
**Date**: 2026-05-16
**Author**: Ocean (HaiyangLi)
**Depends on**: ADR-044 (QuaRot rotated quantization, v0 shipped in v0.2.0)

## Context

ADR-044 shipped rotation-aware Q4 quantization: Hadamard rotation absorbed into weight matrices, RMSNorm fusion, forward-equivalence validated. The Metal Q4 forward path (`MetalQwen35State`) runs Qwen3.5-0.8B at 23.96 PPL (vs 25.57 naive Q4) — same memory, better quality.

The next step is **LoRA adapter injection on top of the QuaRot Q4 base**. This combines the best quantized base (QuaRot Q4) with personalization (LoRA). Nobody has shipped this combination — it requires rotation-aware adapter composition that no existing inference engine implements.

### What exists in lattice today

| Component | Location | Status |
|-----------|----------|--------|
| `LoraHook` trait | `crates/inference/src/lora_hook.rs` | Active, CPU-only |
| `LoraAdapter` / `LoraLayer` | `crates/tune/src/lora/mod.rs` | Active, loads PEFT safetensors |
| `LoraConfig` (rank, alpha, targets) | `crates/tune/src/lora/mod.rs` | Active |
| CPU forward LoRA injection | `crates/inference/src/model/qwen35/model.rs:17` | Active — `Qwen35Model.lora` field, called per projection |
| Metal Q4 forward path | `crates/inference/src/forward/metal_qwen35.rs` | Active — **NO LoRA injection** |
| QuaRot seed in artifact | `config.json` in quantize_quarot output | Stored as conversion metadata |
| `RandomizedHadamard` reconstruction | `crates/inference/src/quant/quarot/` | Active — deterministic from seed |

### Gaps

1. **Metal forward has no adapter injection.** The Q4 GEMV runs entirely on GPU; the current `LoraHook` trait operates on CPU `&[f32]` slices. Calling it would require GPU→CPU→GPU round-trips per projection per token — unacceptable latency.
2. **No counter-rotation logic.** Adapters trained on the unrotated model produce deltas in the wrong basis when the base weights are rotated.
3. **No GPU-native adapter kernel.** Need a Metal compute shader that fuses `Q4_GEMV(W_rot, x) + scale * B @ (A @ x)` or at minimum a separate `LoRA_GEMV` dispatch composable with the existing Q4 path.

## Decision

### The Math (counter-rotation derivation)

In a QuaRot-converted model, the residual stream carries `h_rot = R · h` instead of `h`. For a linear projection with weight `W`:

- **Original**: `y = W · h`
- **QuaRot runtime**: input is `R · h`, weight is `W_rot = W · R^T`. Output: `W_rot · (R · h) = W · R^T · R · h = W · h` ✓

With a LoRA adapter `(B, A)` trained on the **unrotated** model:

- **Original**: `y = W · h + s · B · A · h`
- **QuaRot runtime (naive)**: `W_rot · (R · h) + s · B · A · (R · h) = W · h + s · B · A · R · h` ✗
- **Discrepancy**: `B · A · R · h ≠ B · A · h` (the adapter sees the rotated activation)

**Fix — rotate adapter matrices at load time:**

**Input-side projections** (A receives rotated input `R · h`):
Replace `A` with `A_cr = A · R^T`. Then at runtime:

```
s · B · A_cr · (R · h) = s · B · (A · R^T) · (R · h) = s · B · A · R^T · R · h = s · B · A · h  ✓
```

**Output-side projections** (output must be in rotated residual basis):
Replace `B` with `B_rot = R · B`. Then at runtime:

```
s · B_rot · A · x = s · (R · B) · A · x = R · (s · B · A · x)  ✓
```

The delta lands in the rotated residual basis, matching the base projection's output.

Both corrections are **exact** (R is orthogonal: `R^T · R = I`). Zero approximation error. Zero runtime overhead — the rotations are absorbed at adapter load time.

### Cost of adapter rotation

The `RandomizedHadamard` operates in-place via Walsh-Hadamard transforms (`O(d log d)` per vector), NOT dense matrix multiplication. No `d × d` matrix is materialized.

Per LoRA layer:
- **Input-side** (A rotation): `rank` WHT applications of length `d_in`. Cost: `O(rank × d_in × log(d_in))`. For rank=16, d=1024: ~160K FLOPs.
- **Output-side** (B rotation): `rank` WHT applications of length `d_out`. Cost: `O(rank × d_out × log(d_out))`. Same order.

Memory: only a sign vector (`d × 4 bytes` = 4 KB for d=1024) plus the activation buffer being transformed in-place. No dense R matrix.

`R` is reconstructed from the seed via `RandomizedHadamard::new(seed, dim)`. The seed MUST be persisted in the QuaRot artifact metadata (see §Prerequisites below).

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Adapter Load (one-time, CPU)                           │
│                                                         │
│  1. Load PEFT safetensors → LoraAdapter (A, B per layer)│
│  2. Read seed from QuaRot artifact metadata             │
│  3. Reconstruct R = RandomizedHadamard(seed, d)         │
│  4. For each (layer, module):                           │
│     - Input-side:  A_cr = A · R^T  (counter-rotate A)  │
│     - Output-side: B_rot = R · B   (rotate B)          │
│     - Not in plan: ERROR (refuse unknown targets)       │
│  5. Upload corrected matrices to Metal buffers:         │
│     - Input-side:  upload (B, A_cr)                     │
│     - Output-side: upload (B_rot, A)                    │
│  6. Drop sign vector (transient)                        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Metal Forward (per token, GPU)                         │
│                                                         │
│  For each adapted projection in layer:                  │
│    base_out = Q4_GEMV(W_rot, x)        [existing]      │
│    lora_out = scale * B' @ (A' @ x)    [new kernel]    │
│    out = base_out + lora_out                            │
│  (B' and A' are whichever was rotated at load time)     │
└─────────────────────────────────────────────────────────┘
```

### Prerequisites (dependencies on prior work)

The QuaRot converter (`quantize_quarot`) MUST persist rotation metadata in the output artifact. Without it, the adapter loader cannot reconstruct the correct `RandomizedHadamard`. Required fields (to be added in a converter update):

```json
{
  "quarot": {
    "kind": "randomized_hadamard",
    "seed": 12648430,
    "hidden_dim": 1024,
    "plan": "qwen35_residual_stream_linear_layers"
  }
}
```

Until this metadata exists, the caller must supply the seed explicitly. The `load_lora_adapter` API accepts `quarot_seed: Option<u64>` for this reason — `None` means no rotation correction (unrotated base), `Some(seed)` triggers the correction. A future version will auto-detect from artifact metadata and refuse to load an adapter against a QuaRot base without it.

### Which projections get counter-rotated?

The rotation `R` is the **residual-stream** rotation. It's absorbed input-side into projections that consume the residual stream directly. From the `RotationPlan` (ADR-044 step 3a):

| Projection | Absorbs R? | LoRA target? | Rotate A? | Rotate B? |
|-----------|-----------|-------------|-----------|-----------|
| `q_proj` | Yes (input) | Yes | **A ← A·R^T** | No |
| `k_proj` | Yes (input) | Yes | **A ← A·R^T** | No |
| `v_proj` | Yes (input) | Yes | **A ← A·R^T** | No |
| `o_proj` | Yes (output) | Yes | No | **B ← R·B** |
| `gate_proj` | Yes (input) | Yes | **A ← A·R^T** | No |
| `up_proj` | Yes (input) | Yes | **A ← A·R^T** | No |
| `down_proj` | Yes (output) | Yes | No | **B ← R·B** |
| `in_proj_qkv` (GDN) | Yes (input) | Possible | **A ← A·R^T** | No |
| `out_proj` (GDN) | Yes (output) | Possible | No | **B ← R·B** |

Rules:
- **Input-side** (`W ← W·R^T`): counter-rotate A so `B·(A·R^T)·(R·h) = B·A·h` ✓
- **Output-side** (`W ← R·W`): rotate B so `(R·B)·A·x = R·(B·A·x)` — delta lands in rotated residual basis ✓

The plan data structure already encodes which side each projection uses — reuse it at adapter load time.

### Metal LoRA kernel design

Two options:

**(a) Fused kernel**: `out = Q4_GEMV(W, x) + scale * B @ (A @ x)` in a single dispatch. Maximum throughput (one pass over x, one output write). Complex kernel; tightly couples Q4 layout to LoRA.

**(b) Separate dispatch**: existing `Q4_GEMV(W, x)` writes to output buffer; new `LoRA_GEMV(B, A, x, scale)` adds to the same buffer. Two dispatches, slightly more overhead, but decoupled and testable independently.

**Decision: (b) separate dispatch for v1.** Rationale:
- The Q4 GEMV kernel is already complex and battle-tested; don't destabilize it.
- The LoRA GEMV is `d_out × rank + rank × d_in` FLOPs — tiny relative to the Q4 GEMV (which is `d_out × d_in / block_size`). The dispatch overhead is negligible vs the actual compute.
- Testable independently: LoRA kernel can be validated against CPU reference without touching the Q4 path.
- Fusion becomes a v2 optimization if profiling shows dispatch overhead matters.

The LoRA Metal kernel computes:
```
// Phase 1: A_cr @ x → intermediate (rank × 1)
// Phase 2: B @ intermediate → delta (d_out × 1)
// Phase 3: output[i] += scale * delta[i]
```

For rank ≤ 64 and d ≤ 4096, this is a small matmul — a single threadgroup can handle it. No tiling needed at these dimensions.

### API surface

```rust
// In MetalQwen35State (or a new MetalLoraState wrapper):
pub fn load_lora_adapter(
    &mut self,
    adapter: &LoraAdapter,      // from lattice-tune
    quarot_seed: Option<u64>,   // None = no counter-rotation (unrotated base)
    plan: Option<&RotationPlan>, // which projections were input-absorbed
) -> Result<(), String>;

pub fn unload_lora_adapter(&mut self);
```

When `quarot_seed` is `Some(seed)`:
1. Reconstruct `R` from seed + hidden_dim.
2. For each (layer, module) in the adapter where `plan` says input-side absorption: compute `A_cr = A · R^T`.
3. Upload `B` and `A_cr` (or unmodified `A` for output-side projections) to Metal buffers.
4. Set internal flag so the forward path dispatches the LoRA kernel after each adapted Q4 GEMV.

When `quarot_seed` is `None` (unrotated Q4 base): skip counter-rotation, upload A/B directly.

### Testing strategy

| Test | What it validates |
|------|-------------------|
| Counter-rotation unit test | `B · (A · R^T) · (R · h) == B · A · h` for random A, B, h, R |
| Metal LoRA kernel vs CPU reference | `GPU_LoRA_GEMV(B, A, x) == scale * B @ (A @ x)` on synthetic data |
| End-to-end PPL with identity adapter | PPL unchanged when adapter is all-zeros (no-op injection) |
| End-to-end PPL unrotated-base + adapter | Match CPU LoraHook path on `Qwen35Model` |
| End-to-end PPL QuaRot-base + counter-rotated adapter | Match CPU path with manually counter-rotated adapter |
| LoRA load/unload cycle | No memory leak, state resets cleanly |

### Risks

1. **Adapter compatibility**: adapters trained on a different tokenizer / model revision won't match. Gate on config hash at load time.
2. **Multi-adapter / MoLoRA**: this spec covers single-adapter injection. MoLoRA (mixture routing) is a v2 extension — the KG already has the W2-barycenter theory (`W2-Barycenter-Adapter-Blending`, `MoLoRA-W2-Covariance-Gap-Analysis`). The counter-rotation applies identically to each expert's A matrix.
3. **Adapter rank explosion at Q4 precision**: the Q4 GEMV operates in reduced precision; the LoRA addition is f32/f16. This is actually *good* — the adapter delta is the high-precision correction on top of the quantized base. No precision loss on the personalization signal.
4. **GDN layers**: Qwen3.5's GDN (Gated Delta Network) layers have `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`. If users target these with LoRA (uncommon but possible), the same counter-rotation logic applies — the plan already covers them.

## Implementation Steps

| Step | Scope | Blocked by |
|------|-------|-----------|
| 1 | Counter-rotation library: `quarot::lora::counter_rotate_adapter(adapter, seed, plan) -> LoraAdapter` | Nothing (pure math, uses existing `RandomizedHadamard`) |
| 2 | Metal LoRA kernel: `lora_gemv.metal` shader + Rust dispatch wrapper | Nothing (independent of step 1) |
| 3 | `MetalQwen35State::load_lora_adapter` API: orchestrates counter-rotation + buffer upload | Steps 1 + 2 |
| 4 | Forward path integration: dispatch LoRA kernel after each adapted Q4 GEMV | Step 3 |
| 5 | End-to-end validation: PPL with adapter vs CPU reference path | Step 4 |
| 6 | `bin/chat_metal` adapter flag: `--lora-dir <PATH>` for interactive use | Step 4 |

Steps 1 and 2 are independent — parallelize.

## Success Criteria

- PPL with a PEFT adapter on the QuaRot Q4 base matches PPL on the CPU unrotated path with the same adapter (within Q4 quantization noise, < 0.1 PPL).
- Adapter load time < 500ms for rank-16 adapters on Qwen3.5-0.8B (dominated by the counter-rotation matmul + Metal buffer upload, not I/O).
- Zero runtime overhead when no adapter is loaded (existing `NoopLoraHook` pattern: the LoRA kernel dispatch is branch-gated on adapter presence).
- The counter-rotation is invisible to the user: `load_lora_adapter` auto-detects whether the base is QuaRot (from config) and applies the correction internally.

## Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|------------|------|------|---------|
| Train adapters in the rotated basis | No load-time transform needed | Requires rotated model at train time; not backwards-compatible with existing PEFT adapters | Reject for v1 (offer as v2 "native QuaRot LoRA training" mode) |
| Runtime rotation of adapter output (`R · (B · A · x)`) | No adapter modification | Per-token `d×d` matmul overhead on GPU — unacceptable for decode latency | Reject |
| CPU-side LoRA with GPU→CPU→GPU round-trip | Reuses existing `LoraHook` trait | Round-trip latency kills decode throughput (~5× slower) | Reject |
| Fused Q4+LoRA kernel | Maximum throughput | Couples Q4 layout to LoRA; complex; hard to test | Defer to v2 optimization |

## References

- ADR-044: QuaRot rotated quantization (this builds on)
- ADR-043: LoRA serving verification (existing LoRA correctness framework)
- KG: `LoRA-Low-Rank-Adaptation` (e916fb8b), `LoraAdapter` (2d2ee731), `LoraHook` (7668f8d9)
- KG: `QuaRot (Ashkboos 2024)` (86ec6a4f) — now `status: implemented`
- KG: `W2-Barycenter-Adapter-Blending` (e6a40019) — MoLoRA v2 extension
- Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685
- Ashkboos et al. 2024, "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", arXiv:2404.00456

## Knowledge Graph

- New entity on first implementation PR: `QuaRot-LoRA-Composition` (kind: concept, type: technique, status: proposed → implemented)
- Edges: `extends` → `QuaRot (Ashkboos 2024)`, `extends` → `LoRA-Low-Rank-Adaptation`, `implements` from the code entity once shipped
