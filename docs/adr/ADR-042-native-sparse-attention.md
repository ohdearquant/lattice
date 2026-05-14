# ADR-042: Native Sparse Attention

**Status**: Accepted\
**Date**: 2026-05-14\
**Crate**: `lattice-inference`\
**Extends**: ADR-010 (Attention Mechanisms)

---

## Context

Native Sparse Attention (NSA), introduced by DeepSeek (Yuan et al., ACL 2025, arXiv:2502.11089), is a sparse attention mechanism whose sparse structure participates in the forward pass — it is *natively trainable*, not a post-hoc pruning of a dense model. NSA runs three parallel branches over the KV sequence and merges them with a learned per-head gate:

1. **Compression** — consecutive KV blocks (length `l`, stride `d < l`, overlapping) are mapped to single compressed vectors by a learnable MLP φ; the query attends over the compressed sequence for cheap global context.
2. **Selection** — per-query block importance scores are derived from the compression branch's attention; the top-`n` selection blocks (size `l'`) are gathered and attended at full resolution.
3. **Sliding window** — standard causal local attention over the last `w` tokens.

The paper reports a 2–9× speedup over dense FlashAttention-2 at 64K tokens. **That number is a GPU/FlashAttention-2 result** driven by hardware-aligned block tiling tuned for GPU memory hierarchy; it does not transfer to Lattice's CPU (AVX2/NEON) + Metal targets, and may be negative at the sequence lengths Lattice currently runs.

No public NSA checkpoint exists — DeepSeek released the paper but no weights, and the two public reference implementations (`fla-org/native-sparse-attention`, `lucidrains/native-sparse-attention-pytorch`) **each diverge from the paper**, differently. This ADR therefore adds NSA as a **standalone module** with no model wiring, faithful to the *paper's* equations, available when an NSA checkpoint becomes a target.

This ADR was written **before** the implementation (deviating from the extract-then-ADR pattern of ADR-040/041): NSA has ~13 load-bearing design decisions and several paper-vs-reference divergences that must be pinned before code can be written consistently. It is a living spec — revised if implementation reveals issues.

---

## Decision

Implement NSA as a **seventh attention module** at `src/attention/native_sparse.rs`, following the `gqa.rs` / `differential.rs` convention: the kernel takes already-projected Q/K/V; linear projections are the caller's responsibility. It is a **causal prefill kernel** (multi-token), not the decode/cache path.

Public surface:

- **`NsaConfig`** — `num_heads`, `num_kv_heads`, `head_dim`, and the five NSA hyperparameters `compress_block` (`l`), `compress_stride` (`d`), `select_block` (`l'`), `num_selected` (`n`), `window` (`w`). Helper methods for derived counts and the `d | l`, `d | l'` divisibility invariants.
- **`NsaWeights`** — the learned NSA parameters (caller-supplied, like `differential.rs`'s `DiffLambdaParams`): the K/V compression MLPs `phi_k` / `phi_v`, the intra-block position encodings `k_intrablock_pos` / `v_intrablock_pos`, and the gate projection `g_proj`.
- **`NsaScratch`** — pre-allocated scratch buffers (follows `GqaScratch` / `DiffAttnScratch`).
- **`apply_native_sparse_attention()`** — the three-branch causal prefill kernel.

---

## Key Design Choices

1. **Faithful to the paper's equations, not to either reference implementation.** The paper (Eq. 5, 7–12) is the authority. `lucidrains` is used *only* for the pieces the paper genuinely underspecifies (φ's architecture, the intra-block-PE form, the gate MLP's structure). Where `lucidrains` diverges from the paper, the paper wins:
   - **Importance scores are softmax probabilities** `p_t^cmp` (paper Eq. 8), not the raw pre-softmax logits `lucidrains` uses.
   - **Eq. 9 aggregation is a sum** `Σ Σ p_t^cmp[...]` (paper), not the mean `lucidrains` uses.
   - **No left-padding** of the compression input (`lucidrains` left-pads by `l - d`); compression block `i` covers tokens `[i·d, i·d + l)` exactly, per Eq. 7.
   - **No memory-KV tokens** — `lucidrains` prepends a learnable `compress_mem_kv`; the paper has none.

2. **Compression MLP φ** (paper-underspecified — "a learnable MLP with intra-block position encoding"). Concrete choice from `lucidrains`: a 2-layer MLP `Linear(l·head_dim → l·head_dim) → ReLU → Linear(l·head_dim → head_dim)`, with **independent weights for K and V** (`phi_k`, `phi_v`). A block's `l` tokens are flattened to `l·head_dim` and passed through φ. The φ *weights* are shared across KV heads (one `phi_k` / one `phi_v` for the whole layer), but each KV head produces its **own** compressed K/V sequence — it has its own raw K/V tokens and its own intra-block position encoding, so the compressed outputs `ck` / `cv` are indexed `[kv_head, block, head_dim]`. This is a defensible concrete realization; a real checkpoint may use a different φ — see Risks.

3. **Intra-block position encoding** (paper-underspecified — "intra-block position encoding"). Concrete choice from `lucidrains`: a learned additive tensor of shape `[num_kv_heads, l, head_dim]`, added to each compression block's tokens before φ. Per-KV-head, per-intra-block-position, per-dim.

4. **Compression uses non-RoPE Q/K; selection and sliding-window use RoPE'd Q/K.** φ's intra-block position encoding already encodes within-block position; applying RoPE on top would double-count (the `lucidrains` comment cites arXiv:2501.18795). NSA therefore cannot follow the ADR-010 "caller applies RoPE" convention with a single Q/K pair. The kernel takes **both** the raw and the RoPE'd Q/K (`q`, `k` and `q_rope`, `k_rope`) — the caller already owns a RoPE routine, so this mirrors `differential.rs` taking caller-supplied lambda vectors rather than the kernel owning RoPE. V is RoPE-free in all branches.

5. **Eq. 9 score aggregation requires `d | l` and `d | l'`** — asserted in `NsaConfig`. For query `t`, selection-block `j`'s importance is `p_t^slc[j] = Σ_{m=0}^{l'/d − 1} Σ_{n=0}^{l/d − 1} p_t^cmp[(l'/d)·j + m + n]`, summing the compression-block probabilities that spatially overlap selection-block `j`. **Compression-block causal validity**: a compression block `i` (covering raw tokens `[i·d, i·d + l)`) is causally valid for query `t` iff *all* its tokens are seen — `i·d + l − 1 ≤ t`, equivalently `i·d + l ≤ t + 1`. A compressed block is atomic (no token-level masking), so a block extending past `t` is excluded entirely. Causally invalid compression blocks contribute 0 to both the compression attention (Eq. 8) and the Eq. 9 sum.

6. **GQA group aggregation (Eq. 10)**: importance scores are summed across the `H = num_heads / num_kv_heads` query heads in a group *before* top-`n` selection; all query heads in a group attend the **same** selected blocks. Selection is per-KV-head.

7. **Top-`n` selection with forced blocks.** **Selection-block causal validity**: a selection block `j` (covering tokens `[j·l', (j+1)·l')`) is causally valid for query `t` iff it contains at least one token `≤ t` — `j·l' ≤ t`. A valid block may be **partial**: it can contain tokens `> t` (e.g. the query's own block). Only *full* `l'`-token selection blocks are candidates (`num_select_blocks` counts full blocks); a query in the trailing partial region beyond the last full block cannot select its own partial block, but the sliding window covers it.

   Per the paper's Section 4.1 ("including fixed activating the 1 initial block and 2 local blocks"), the `n` selected blocks always include: the **initial block** (selection-block 0) and the **2 local blocks** (the 2 highest-indexed causally-valid selection blocks for query `t`); the remaining `n − 3` are the top-scored of the rest. If fewer than `n` selection blocks are causally valid, all valid blocks are taken.

   Selection attention is standard scaled-dot-product over the gathered tokens of the selected blocks. Because a selected block may be partial, **token-level causal masking is load-bearing**: within each gathered block, tokens with index `> t` are masked out, so the query attends only `≤ t` (including itself).

8. **Sliding window**: standard causal local attention over `k_rope_{max(0, t−w+1) .. t}` — the last `w` tokens inclusive of `t`.

9. **Gating (Eq. 5)**: three per-head gates `g_cmp, g_slc, g_win = sigmoid(W_g · x_t)` where `W_g` is `[3·num_heads, dim]` and `x_t` is the caller-supplied (already-normed) hidden state. Gates are **independent sigmoids in [0,1], not normalized to sum to 1**. Output: `o_t = g_cmp · o_t^cmp + g_slc · o_t^slc + g_win · o_t^win`. The kernel takes `x` and `g_proj`; computing the gate internally keeps the NSA gating faithful and self-contained.

10. **Empty-branch handling.** Early queries (`t < l + d`, etc.) may have zero causally-valid compression or selection blocks. In that case the branch output is the zero vector — the gate still scales it, and the sliding-window branch (always non-empty for `t ≥ 0`) carries those positions. This is a faithful-ish choice the paper does not spell out; documented and covered by tests.

11. **Uniform `head_dim`.** The paper uses asymmetric dims (`d_k = 192`, `d_v = 128`, an MLA detail). All existing Lattice attention modules use a uniform `head_dim`; NSA follows suit. Asymmetric dims are a deferred extension.

12. **Additive `-10,000.0` causal mask** — consistent with ADR-010 design choice #2 and ADR-041; masked positions are also explicitly zeroed after softmax.

13. **Standalone, not wired into a model.** No checkpoint uses NSA, so wiring it into a forward pass would be speculative. The module is independently testable and benchmarkable; model integration is deferred until a concrete checkpoint target exists.

---

## Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|---|---|---|---|
| Mean-pooling compression (`fla-org` variant) | Parameter-free; no φ weights to supply | Explicitly *not* the paper; discards the learnable MLP φ | Ocean chose the faithful paper algorithm |
| Selection-branch-only ("ship the seed") | Smallest; isolates the novel kernel piece | Not the full mechanism; defers the gated three-branch merge | Ocean chose the full faithful implementation |
| Defer NSA entirely | Avoids building on a nonexistent checkpoint | Leaves the bench-oriented attention series at 2/3 | Ocean chose to build it |
| Follow `lucidrains` exactly | One coherent reference to transcribe | Diverges from the paper (logits-as-scores, mean aggregation, mem-KV, left-pad) | Ocean asked for paper-faithful |
| Single Q/K pair, RoPE'd by caller (ADR-010 convention) | Matches the other modules' signature | NSA's compression branch *must* use non-RoPE Q/K | Correctness requires both raw and RoPE'd Q/K |

---

## Consequences

**Positive**:

- NSA's paper algorithm is implemented and independently testable/benchmarkable; when an NSA checkpoint becomes a target, the math is ready.
- The three branches are separable and individually testable (compression, selection, sliding window, gating).
- Paper-vs-reference divergences are pinned here, so the implementation and its naive reference oracle share one explicit, documented convention.

**Negative**:

- Seventh attention module — the largest one — adds a seventh test suite to maintain. The kernel has ~9 distinct sub-pieces (φ forward ×2, compression attention, Eq. 9 aggregation, Eq. 10 aggregation, top-`n` selection, selection attention, sliding window, gating, merge).
- The kernel signature is heavier than the other attention modules: it takes raw *and* RoPE'd Q/K, the hidden state `x`, and an `NsaWeights` struct.

**Risks**:

- **No checkpoint validation — and, unlike ADR-041, no single canonical reference.** Differential attention could be transcribed from Microsoft's `multihead_flashdiff_1.py`; NSA cannot — the public references diverge from the paper and from each other. Tests therefore validate *self-consistency* (optimized kernel vs. a naive exact-math oracle of the **same** paper-faithful spec) plus causal-masking differential tests and parity invariants. This is genuinely weaker verification than ADR-041 had. An independent oracle would require a trained NSA checkpoint, which does not exist publicly.
- **φ's architecture and the intra-block-PE form are educated guesses** — the paper underspecifies both. A real NSA checkpoint may use a different φ; the weight-loading path will need revision when one exists.
- **Performance is unproven.** The paper's 2–9× is a GPU/FlashAttention-2 number. On CPU/Metal the sparse gather + three-branch + φ overhead may be *slower* than dense attention at Lattice's current sequence lengths. The benchmark measures this honestly rather than assuming the speedup transfers.
- The lambda-style caller-supplied `NsaWeights` has no weight-loading path; the current module accepts them as a struct but does not define checkpoint key naming.

---

## References

### Papers

- Yuan et al. (2025) — "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" — arXiv:2502.11089 — <https://arxiv.org/abs/2502.11089> — ACL 2025. Eq. 5 (gated merge), Eq. 7–8 (compression), Eq. 9–12 (selection), §3.3.3 (sliding window), §4.1 (hyperparameter defaults: `l=32, d=16, l'=64, n=16, w=512`; the forced "1 initial + 2 local" blocks).

### Reference implementations

- `lucidrains/native-sparse-attention-pytorch` — <https://github.com/lucidrains/native-sparse-attention-pytorch> — pure-PyTorch. Source for the paper-underspecified pieces: the φ MLP architecture (`compress_networks.py`), the learned additive intra-block position encoding, the `Linear(dim, 3·heads)` + sigmoid gate, and the non-RoPE-compression / RoPE'd-selection split (`native_sparse_attention.py`). Diverges from the paper on importance-score source, Eq. 9 aggregation, left-padding, and memory-KV — those divergences are *not* followed.
- `fla-org/native-sparse-attention` — <https://github.com/fla-org/native-sparse-attention> — Triton kernels. Uses mean-pooling instead of φ and conflates the compression/selection block sizes; consulted but not transcribed (it is the less paper-faithful variant).

### Internal

- `src/attention/native_sparse.rs` — `NsaConfig`, `NsaWeights`, `NsaScratch`, `apply_native_sparse_attention()`
- `crates/inference/tests/native_sparse_attention_test.rs` — naive paper-faithful reference oracle and parity tests
- `crates/inference/benches/native_sparse_attention_bench.rs` — throughput and sparse-vs-dense benchmark
- ADR-010 — Attention Mechanisms (parent ADR for all attention variants)
- ADR-040 — Gated Attention; ADR-041 — Differential Attention (sibling attention-mechanism ADRs)
