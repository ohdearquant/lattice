# ADR-040: Gated Attention (G1 SDPA-Output Gating)

**Status**: Accepted\
**Date**: 2026-05-14\
**Crate**: `lattice-inference`\
**Extends**: ADR-010 (Attention Mechanisms)

---

## Context

Qwen3.5 and Qwen3-Next use an elementwise sigmoid gate applied to the SDPA output before the output projection. The gate is packed into the Q projection weight: `q_proj` has shape `[2*q_dim, hidden]`, producing both Q and gate vectors in a single matmul. After attention computes the context vector, the gate modulates it element-wise: `context[i] *= sigmoid(gate[i])`.

This mechanism was introduced in "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free" (Qiu et al., Qwen/Alibaba, arXiv:2505.06708, NeurIPS 2025 Best Paper). It eliminates attention sinks (first-token attention mass drops from ~46.7% to ~4.8%) and improves long-context performance (RULER 64K: 37.51→66.60). The latency overhead is under 2%.

Prior to this ADR, the gated attention logic was inlined in `model/qwen35/forward.rs` — functional but not reusable by other model architectures that ship with the same gating pattern (e.g., Qwen3-Next, future Diff Transformer variants with output gating).

---

## Decision

Extract gated attention into a **fifth attention module** at `src/attention/gated.rs` with three public primitives:

1. **`deinterleave_q_gate(q_and_gate, q_buf, gate_buf, num_heads, head_dim)`** — splits the packed `[Q_h0|G_h0|Q_h1|G_h1|...]` layout into separate Q and gate buffers.

2. **`apply_sigmoid_gate(context, gate)`** — applies `context[i] *= sigmoid(gate[i])` using exact `f32::exp`. This is the production default, matching all other Qwen3.5 execution paths.

3. **`apply_sigmoid_gate_fast(context, gate)`** — same operation with runtime SIMD dispatch (NEON on aarch64, AVX2 on x86_64) using the Schraudolph fast-exp approximation. Opt-in only.

The Qwen3.5 forward pass calls `deinterleave_q_gate` and `apply_sigmoid_gate` (exact) instead of inlining the logic. Future model architectures that use the same gating pattern (widened `q_proj` with per-head interleaved gate) can reuse the module directly.

---

## Key Design Choices

1. **Free functions, not a struct or trait**: The gate is a post-processing step on an existing attention output, not an alternative attention mechanism. It composes with GQA, MHA, or any other SDPA variant. A struct wrapping the full attention computation would duplicate the GQA/MHA scaffolding; free functions compose cleanly instead.

2. **Exact sigmoid by default, fast SIMD opt-in**: `apply_sigmoid_gate` uses exact `f32::exp` so the extracted module produces bit-identical output to the pre-refactor inline code and to every other Qwen3.5 backend (prefill, f16, q8, Metal, speculative). The exact gate adds ~1.7–6.9 µs per full-attention layer at Qwen3.5 head shapes. `apply_sigmoid_gate_fast` provides an approximate SIMD path (Schraudolph fast-exp, same bit-trick as `softmax.rs`, ~5–6% relative error) that benchmarks 7.1–7.7× faster than scalar — but adopting it in production requires switching all comparable paths together, which is out of scope for this extraction PR.

3. **Deinterleave as a separate function**: The packed `[Q|gate]` layout is a weight-format concern, not an attention-algorithm concern. Keeping deinterleave separate from the gate application means models that store Q and gate in separate tensors (e.g., a future checkpoint format) can skip deinterleave entirely and call `apply_sigmoid_gate` directly.

4. **No headwise variant**: The paper ablates both headwise (one scalar per head) and elementwise (one scalar per head dimension) gating. Qwen3.5 and Qwen3-Next ship with elementwise. The headwise variant is cheaper but cannot load released checkpoints. We implement only elementwise; headwise can be added as a separate function if a checkpoint requires it.

5. **Zero-allocation deinterleave via struct destructuring**: The Qwen3.5 forward path calls `ForwardScratch::split_q_and_gate()`, which destructures `self` into disjoint field borrows (`q_and_gate`, `q_buf`, `gate_z`) and delegates to `deinterleave_q_gate`. This satisfies the borrow checker without `unsafe` or heap allocation. The production path uses the same extracted function that the benchmarks and unit tests exercise.

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

- Gated attention is independently testable and benchmarkable (7 unit tests, 6 integration tests, dedicated benchmark).
- Any model architecture using packed `[Q|gate]` projection can reuse the module without code duplication.
- SIMD acceleration is automatic via runtime dispatch — callers do not need to know the target architecture.

**Negative**:

- Fifth attention module adds a fifth test suite to maintain.

**Risks**:

- The Schraudolph fast-exp approximation produces ~1% absolute error on sigmoid values near 0.5. For gating this is acceptable (the paper's headwise variant, which is coarser, still shows strong results). However, if a future model uses the gate output for purposes other than multiplicative modulation (e.g., as a routing score), the approximation may need to be revisited.
- **Exact vs approximate sigmoid**: `apply_sigmoid_gate` uses exact `f32::exp` to match all other execution paths (prefill, f16, q8, Metal, speculative). `apply_sigmoid_gate_fast` provides an approximate SIMD variant (Schraudolph fast-exp, ~5–6% relative error, 6–8× speedup) but is opt-in — a future PR that deliberately adopts the approximation across all comparable paths should switch to it.

---

## References

### Papers

- Qiu et al. (2025) — "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free" — arXiv:2505.06708 — <https://arxiv.org/abs/2505.06708> — NeurIPS 2025 Best Paper. Introduces the G1 SDPA-output sigmoid gate; source of the attention-sink and long-context numbers cited in Context.
- Schraudolph, N. N. (1999) — "A Fast, Compact Approximation of the Exponential Function" — Neural Computation 11(4):853–862 — <https://doi.org/10.1162/089976699300016467>. The bit-trick `exp` approximation used by `apply_sigmoid_gate_fast` (and by the existing `softmax.rs` SIMD path).

### Architecture / implementation

- Qwen3-Next model card — <https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct>. The hybrid GatedDeltaNet + Gated Attention layer schedule that motivates a reusable gate module.
- HuggingFace Transformers — `models/qwen3_next/modular_qwen3_next.py` — <https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py>. Reference for the packed `[Q|gate]` `q_proj` layout and per-head interleaving convention that `deinterleave_q_gate` matches.

### Internal

- `src/attention/gated.rs` — `deinterleave_q_gate()`, `apply_sigmoid_gate()`, `apply_sigmoid_gate_fast()`, `apply_sigmoid_gate_scalar()`
- `src/model/qwen35/forward.rs` — call sites in `project_qkv()` and `full_attention_step_from_attn_out()`
- `src/model/qwen35/cache.rs` — `ForwardScratch::split_q_and_gate()`
- ADR-010 — Attention Mechanisms (parent ADR for all attention variants)
