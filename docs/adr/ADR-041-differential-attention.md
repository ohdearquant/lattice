# ADR-041: Differential Attention

**Status**: Accepted\
**Date**: 2026-05-14\
**Crate**: `lattice-inference`\
**Extends**: ADR-010 (Attention Mechanisms)

---

## Context

Differential attention, introduced in the Differential Transformer (Ye et al., Microsoft, ICLR 2025, arXiv:2410.05258), splits Q and K each into two halves along the head dimension, computes two independent causal softmax attention maps, and subtracts the second (scaled by a learnable `lambda_full`) from the first. The subtraction cancels the low-signal "attention-sink" mass that vanilla softmax attention accumulates on irrelevant tokens, producing sparser, more focused attention. The paper reports gains on long-context retrieval, hallucination reduction, and in-context learning.

No model checkpoint currently loaded by Lattice uses differential attention — it is a distinct Transformer variant, not part of Qwen3 or BERT. (An earlier internal research note incorrectly claimed Qwen3 required it; the fact-check corrected this.) This ADR therefore adds differential attention as a **standalone module** with no model wiring, so it is available when a DIFF Transformer checkpoint becomes a target.

An earlier research note also oversimplified the mechanism (scalar lambda, missing sub-layer norm, wrong V shape). The implementation here follows the **exact** reference algorithm from Microsoft's `unilm/Diff-Transformer/multihead_flashdiff_1.py`.

---

## Decision

Implement differential attention as a **sixth attention module** at `src/attention/differential.rs`, following the `gqa.rs` convention: the core function takes Q/K/V already projected and with RoPE already applied; linear projections and RoPE are the caller's responsibility.

Public surface:

- **`DiffAttnConfig`** — `num_heads`, `num_kv_heads`, `head_dim`, `layer_depth`. Helper methods for `lambda_init()`, packed head counts, and buffer dimensions.
- **`DiffLambdaParams`** — the four learnable reparameterization vectors (`lambda_q1/k1/q2/k2`, each `[head_dim]`).
- **`compute_lambda_full()`** — `exp(lq1·lk1) - exp(lq2·lk2) + lambda_init`.
- **`DiffAttnScratch`** — pre-allocated scratch buffers (follows `GqaScratch`).
- **`apply_differential_attention()`** — the causal prefill kernel.

---

## Key Design Choices

1. **Faithful to `multihead_flashdiff_1.py`, not the simplified note**: `lambda_full` is the reparameterized form `exp(lq1·lk1) - exp(lq2·lk2) + lambda_init` (dot products over `head_dim`), not a loaded scalar. `lambda_init = 0.8 - 0.6·exp(-0.3·depth)` is depth-scheduled. The subtracted result is **not** clamped or ReLU'd — negative attention weights are valid. A sub-layer RMSNorm over `2·head_dim` is applied after the value matmul and before the `(1 - lambda_init)` scale; the paper's ablation shows removing it degrades training stability.

2. **V is not split**: Q and K are split into `2·num_heads` / `2·num_kv_heads` packed heads of `head_dim` each. V keeps `num_kv_heads` heads, but each head is `2·head_dim` wide. The split applies only to the query/key similarity computation, not the value projection.

3. **Subtract scores, then one V matmul**: The reference computes `attn1 = softmax1 @ v` and `attn2 = softmax2 @ v` separately, then `attn1 - lambda_full·attn2`. The kernel instead computes `(softmax1 - lambda_full·softmax2) @ v` — a single V matmul. This is exactly equivalent by linearity of matrix multiplication and saves one GEMM per head pair.

4. **GQA via post-split head mapping**: `multihead_flashdiff_1.py` *defines* a `repeat_kv` helper but does **not** call it — it splits Q/K into `q1/q2/k1/k2` first (with `num_heads` / `num_kv_heads` heads respectively), then passes them to `flash_attn_func`, which performs standard GQA grouping internally. The standard convention `kv_head = query_head // n_rep` therefore applies to the *post-split* logical heads. The kernel maps logical pair `h` to KV head `h / n_rep`, packed K heads `2·(h/n_rep)` and `2·(h/n_rep)+1`. This was verified against the Microsoft source — an earlier reading that assumed `repeat_kv` was applied to the pre-split `2·num_kv_heads` axis was wrong.

5. **Additive `-10,000.0` causal mask**: Consistent with ADR-010 design choice #2 — avoids the NaN risk of `-inf` on fully-masked rows. The softmax loop only iterates the causal prefix, so masked positions are also explicitly zeroed.

6. **Standalone, not wired into a model**: There is no checkpoint to load, so wiring differential attention into a forward pass would be speculative. The module is independently testable and benchmarkable; model integration is deferred until a concrete checkpoint target exists.

---

## Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|---|---|---|---|
| Implement the simplified scalar-lambda version from the original research note | Less code | Diverges from the reference; would not load real DIFF Transformer weights; missing the load-bearing sub-layer norm | Correctness requires the faithful algorithm |
| Wire differential attention into a model forward pass now | "Complete" feature | No checkpoint exists to validate against; speculative wiring | Defer until a checkpoint target is identified |
| Two separate V matmuls (mirror the reference exactly) | Trivially matches reference structure | One extra GEMM per head pair for no numerical benefit | `(s1 - λ·s2) @ v` is exactly equivalent and cheaper |
| `multihead_flashdiff_2.py` as the reference | Supports packages without custom qk/v dims | `flashdiff_1` is the "Recommended" reference and is simpler | `flashdiff_1` is canonical |

---

## Consequences

**Positive**:

- Differential attention is independently testable and benchmarkable (6 unit tests, 4 integration tests, dedicated benchmark).
- When a DIFF Transformer checkpoint becomes a target, the math is already implemented and validated.
- Benchmark shows the differential mechanism (second softmax + subtract + sub-RMSNorm) costs only ~4–11% over a same-sized plain GQA kernel.

**Negative**:

- Sixth attention module adds a sixth test suite to maintain.
- The integration test's reference and the optimized kernel share the GQA head-mapping convention. The mapping was manually verified against the Microsoft source, but a fully independent oracle (e.g. tracked vectors from a PyTorch run of the actual reference) would be stronger. MHA cases (`n_rep = 1`) are unambiguous and independently meaningful.

**Risks**:

- No checkpoint validation: the implementation matches the *algorithm* in `multihead_flashdiff_1.py`, but has never been run against real trained DIFF Transformer weights. Weight-key naming, the exact RoPE variant (the reference uses `interleaved=True`), and projection layout conventions will need verification when a checkpoint is integrated.
- The lambda reparameterization vectors must be loaded per-layer from a checkpoint; the current module accepts them as a caller-supplied struct but does not define a weight-loading path.

---

## References

### Papers

- Ye et al. (2025) — "Differential Transformer" — arXiv:2410.05258 — <https://arxiv.org/abs/2410.05258> — ICLR 2025. The mechanism, the depth-scheduled `lambda_init`, and the sub-layer-norm ablation.

### Reference implementation

- Microsoft `unilm` — `Diff-Transformer/multihead_flashdiff_1.py` — <https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_flashdiff_1.py>. The exact algorithm transcribed by this module: head splitting, `lambda_full` reparameterization, `subln`, `(1 - lambda_init)` scale, and the post-split GQA semantics (via `flash_attn_func`, not `repeat_kv`).

### Internal

- `src/attention/differential.rs` — `DiffAttnConfig`, `DiffLambdaParams`, `compute_lambda_full()`, `DiffAttnScratch`, `apply_differential_attention()`
- `crates/inference/tests/differential_attention_test.rs` — naive reference oracle and parity test
- `crates/inference/benches/differential_attention_bench.rs` — throughput and overhead-vs-GQA benchmark
- ADR-010 — Attention Mechanisms (parent ADR for all attention variants)
- ADR-040 — Gated Attention (sibling attention-mechanism ADR)
