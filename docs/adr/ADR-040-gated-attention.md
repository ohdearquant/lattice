# ADR-040: Gated Attention (G1 SDPA-Output Gating)

**Status**: Accepted\
**Date**: 2026-05-14\
**Crate**: `lattice-inference`\
**Extends**: ADR-010 (Attention Mechanisms)

---

## Context

Qwen3.5 and Qwen3-Next use an elementwise sigmoid gate applied to the SDPA output before the output projection. The gate is packed into the Q projection weight: `q_proj` has shape `[2*q_dim, hidden]`, producing both Q and gate vectors in a single matmul. After attention computes the context vector, the gate modulates it element-wise: `context[i] *= sigmoid(gate[i])`.

This mechanism was introduced in the "Gated Attention in Large Language Models" paper (NeurIPS 2025 Best Paper, Qwen/Alibaba, arXiv:2505.06708). It eliminates attention sinks (first-token attention mass drops from ~46.7% to ~4.8%) and improves long-context performance (RULER 64K: 37.51→66.60). The latency overhead is under 2%.

Prior to this ADR, the gated attention logic was inlined in `model/qwen35/forward.rs` — functional but not reusable by other model architectures that ship with the same gating pattern (e.g., Qwen3-Next, future Diff Transformer variants with output gating).

---

## Decision

Extract gated attention into a **fifth attention module** at `src/attention/gated.rs` with two public primitives:

1. **`deinterleave_q_gate(q_and_gate, q_buf, gate_buf, num_heads, head_dim)`** — splits the packed `[Q_h0|G_h0|Q_h1|G_h1|...]` layout into separate Q and gate buffers.

2. **`apply_sigmoid_gate(context, gate)`** — applies `context[i] *= sigmoid(gate[i])` with runtime SIMD dispatch (NEON on aarch64, AVX2 on x86_64, scalar fallback).

The Qwen3.5 forward pass calls these functions instead of inlining the logic. Future model architectures that use the same gating pattern (widened `q_proj` with per-head interleaved gate) can reuse the module directly.

---

## Key Design Choices

1. **Free functions, not a struct or trait**: The gate is a post-processing step on an existing attention output, not an alternative attention mechanism. It composes with GQA, MHA, or any other SDPA variant. A struct wrapping the full attention computation would duplicate the GQA/MHA scaffolding; free functions compose cleanly instead.

2. **SIMD sigmoid via Schraudolph fast-exp**: The sigmoid denominator `1 + exp(-x)` uses the same bit-trick approximation as `softmax.rs` (~5–6% relative error on exp, sufficient for gating). This avoids introducing a separate `libm` or polynomial approximation path. The NEON path achieves 6.8–7.6× speedup over scalar; gate overhead is 0.18–0.76 µs per layer.

3. **Deinterleave as a separate function**: The packed `[Q|gate]` layout is a weight-format concern, not an attention-algorithm concern. Keeping deinterleave separate from the gate application means models that store Q and gate in separate tensors (e.g., a future checkpoint format) can skip deinterleave entirely and call `apply_sigmoid_gate` directly.

4. **No headwise variant**: The paper ablates both headwise (one scalar per head) and elementwise (one scalar per head dimension) gating. Qwen3.5 and Qwen3-Next ship with elementwise. The headwise variant is cheaper but cannot load released checkpoints. We implement only elementwise; headwise can be added as a separate function if a checkpoint requires it.

5. **Temporary allocation in deinterleave call site**: The refactored Qwen3.5 forward path copies `q_and_gate` to a temporary `Vec` before calling `deinterleave_q_gate`, because Rust's borrow checker prevents simultaneous mutable access to `scratch.q_and_gate`, `scratch.q_buf`, and `scratch.gate_z`. The allocation is 32 KB at the largest model size (16 heads × 256 dim × 2 × 4 bytes) and is negligible relative to the attention matmul. A future optimization could add a dedicated scratch buffer to eliminate this allocation.

---

## Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|---|---|---|---|
| Keep gate inlined in Qwen3.5 forward pass | Zero abstraction overhead | Not reusable; duplicated when adding Qwen3-Next support | Reuse across model architectures justifies extraction |
| Trait-based `GatedAttention` wrapping full SDPA | Clean interface | Duplicates GQA/MHA compute; gate is a modifier, not a replacement | Free functions compose better with existing attention modules |
| Separate `gate_proj` weight tensor | Cleaner weight loading | Released Qwen3.5/Qwen3-Next checkpoints pack gate into `q_proj`; a separate tensor requires repacking on load | Match the checkpoint format; repacking is a loader concern |
| Standard `exp()` instead of fast-exp | Higher accuracy | 2–3× slower; softmax.rs already establishes the fast-exp precedent | Consistency with existing SIMD strategy; accuracy is sufficient for gating |

---

## Consequences

**Positive**:

- Gated attention is independently testable and benchmarkable (7 unit tests, 3 integration tests, dedicated benchmark).
- Any model architecture using packed `[Q|gate]` projection can reuse the module without code duplication.
- SIMD acceleration is automatic via runtime dispatch — callers do not need to know the target architecture.

**Negative**:

- Fifth attention module adds a fifth test suite to maintain.
- The `to_vec()` copy in the Qwen3.5 call site adds a small per-layer allocation on the decode path.

**Risks**:

- The Schraudolph fast-exp approximation produces ~1% absolute error on sigmoid values near 0.5. For gating this is acceptable (the paper's headwise variant, which is coarser, still shows strong results). However, if a future model uses the gate output for purposes other than multiplicative modulation (e.g., as a routing score), the approximation may need to be revisited.
- **Production path uses approximate sigmoid**: `apply_sigmoid_gate` dispatches to NEON/AVX2 fast-exp in production builds; `apply_sigmoid_gate_scalar` (exact) is used only as fallback and in tests. The integration test covers both paths with tolerances of ≤1e-6 (scalar) and ≤2e-2 (dispatched) respectively.

---

## References

- `src/attention/gated.rs` — `deinterleave_q_gate()`, `apply_sigmoid_gate()`, `apply_sigmoid_gate_scalar()`
- `src/model/qwen35/forward.rs` — call sites in `project_qkv()` and `full_attention_step_from_attn_out()`
- Qwen/Alibaba 2025 — "Gated Attention in Large Language Models" — arXiv:2505.06708, NeurIPS 2025 Best Paper
- ADR-010 — Attention Mechanisms (parent ADR for all attention variants)
