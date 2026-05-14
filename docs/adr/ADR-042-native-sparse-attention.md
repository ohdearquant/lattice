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

DeepSeek released the paper but no official weights. A community checkpoint does exist — `zen-E/NSA-1B` (public, ungated, `LlamaNSAForCausalLM`, `model.safetensors` ~1.38 GB, first published 2026-01-29) — but it is built on `fla-org/native-sparse-attention`, the **mean-pooling** compression variant, not the φ-MLP algorithm this ADR implements; it is therefore *not* a drop-in correctness oracle for this module. The two public reference implementations (`fla-org/native-sparse-attention`, `lucidrains/native-sparse-attention-pytorch`) **each diverge from the paper**, differently. This ADR adds NSA as a **standalone module** with no model wiring, faithful to the *paper's* equations; wiring it to a concrete checkpoint — which would require implementing the fla-org mean-pool variant or training a φ-MLP checkpoint — is deferred.

This ADR was written **before** the implementation (deviating from the extract-then-ADR pattern of ADR-040/041): NSA has ~13 load-bearing design decisions and several paper-vs-reference divergences that must be pinned before code can be written consistently. It is a living spec — revised if implementation reveals issues.

---

## Decision

Implement NSA as a **seventh attention module** at `src/attention/native_sparse.rs`, following the `gqa.rs` / `differential.rs` convention: the kernel takes already-projected Q/K/V; linear projections are the caller's responsibility. It is a **causal prefill kernel** (multi-token), not the decode/cache path.

Public surface:

- **`NsaConfig`** — `num_heads`, `num_kv_heads`, `head_dim`, and the five NSA hyperparameters `compress_block` (`l`), `compress_stride` (`d`), `select_block` (`l'`), `num_selected` (`n`), `window` (`w`). Helper methods for derived counts and a `validate()` enforcing the positivity, `num_heads % num_kv_heads == 0`, `l ≤ l'`, `d | l`, `d | l'`, and `num_selected ≥ 3` invariants.
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

4. **Independent K/V per branch; compression uses non-RoPE Q/K, selection and sliding-window use RoPE'd Q/K.** Two paper requirements together shape the kernel signature:

   - **Paper §3.3.3**: *"To further prevent shortcut learning across attention branches with marginal computational overhead, we provide independent keys and values for three branches."* The compression, selection, and sliding-window branches therefore consume **separately-projected** K and V: `k_cmp`/`v_cmp`, `k_slc`/`v_slc`, `k_win`/`v_win`. A real NSA checkpoint has three K and three V projections per layer; a shared-K/V kernel could not consume it.
   - φ's intra-block position encoding already encodes within-block position; applying RoPE on top of the compression branch would double-count (the `lucidrains` comment cites arXiv:2501.18795). So `k_cmp` and the compression query are **non-RoPE**; `k_slc` and `k_win` are **RoPE'd** by the caller. The paper specifies independent K/V only — *not* independent queries — so the kernel takes one non-RoPE `q` (compression) and one RoPE'd `q_rope` (selection + window). All V is RoPE-free.

   The kernel signature is therefore `(q, q_rope, k_cmp, k_slc, k_win, v_cmp, v_slc, v_win, x, ...)` — 8 caller-supplied activation buffers. This is heavier than the other attention modules, but it is the faithful ABI; the caller already owns the projection and RoPE routines, mirroring `differential.rs` taking caller-supplied lambda vectors rather than the kernel owning RoPE.

5. **Eq. 9 score aggregation requires `l ≤ l'`, `d | l`, `d | l'`** — the paper's stated precondition (*"Given `l ⩽ l′`, `d ∣ l` and `d ∣ l′`, we have:"*), asserted in `NsaConfig::validate`. For query `t`, selection-block `j`'s importance is the paper's Eq. 9 **verbatim**:

   > `p_t^slc[j] = Σ_{m=0}^{l'/d − 1} Σ_{n=0}^{l/d − 1} p_t^cmp[(l'/d)·j − m − n]`

   The compression index is `(l'/d)·j − m − n` — **both `m` and `n` subtracted**, not `+ m + n`. (Round-1 review caught this: an earlier draft of this ADR had `+ m + n`, which selects an upward-shifted, disjoint set of compression blocks; the kernel and the naive oracle had both transcribed the wrong sign from here.) All indices are 0-based: compression block `i` ↔ raw tokens `[i·d, i·d + l)` (Eq. 7); selection block `j` ↔ tokens `[j·l', (j+1)·l')`.

   Worked example for auditability — `l = 4, d = 2, l' = 8` (so `l'/d = 4`, `l/d = 2`), selection block `j = 1`: the index `4 − m − n` over `m ∈ [0,4), n ∈ [0,2)` expands to the multiset `{4, 3, 3, 2, 2, 1, 1, 0}`, so `p_t^slc[1] = p_t^cmp[0] + 2·p_t^cmp[1] + 2·p_t^cmp[2] + 2·p_t^cmp[3] + p_t^cmp[4]`. A compression index that falls outside `[0, valid_cblocks)` — negative (small `j`) or beyond the causal frontier — is not a valid index into the probability vector `p_t^cmp` and contributes 0. `private fn aggregate_selection_importance` isolates this computation so it is unit-testable against this hand-computed expansion, independent of either the kernel's per-token loop or the naive oracle.

   **Compression-block causal validity**: a compression block `i` (covering raw tokens `[i·d, i·d + l)`) is causally valid for query `t` iff *all* its tokens are seen — `i·d + l − 1 ≤ t`, equivalently `i·d + l ≤ t + 1` (matches Eq. 7's `0 ⩽ i ⩽ ⌊(t − l)/d⌋`). A compressed block is atomic (no token-level masking), so a block extending past `t` is excluded entirely. Causally invalid compression blocks contribute 0 to both the compression attention (Eq. 8) and the Eq. 9 sum.

6. **GQA group aggregation (Eq. 10)**: importance scores are summed across the `H = num_heads / num_kv_heads` query heads in a group *before* top-`n` selection; all query heads in a group attend the **same** selected blocks. Selection is per-KV-head.

7. **Top-`n` selection with forced blocks.** **Selection-block causal validity**: a selection block `j` (covering tokens `[j·l', (j+1)·l')`) is causally valid for query `t` iff it contains at least one token `≤ t` — `j·l' ≤ t`. A valid block may be **partial**: it can contain tokens `> t` (e.g. the query's own block). Only *full* `l'`-token selection blocks are candidates (`num_select_blocks` counts full blocks); a query in the trailing partial region beyond the last full block cannot select its own partial block, but the sliding window covers it.

   Per the paper's Section 4.1 ("including fixed activating the 1 initial block and 2 local blocks"), the `n` selected blocks always include: the **initial block** (selection-block 0) and the **2 local blocks** (the 2 highest-indexed causally-valid selection blocks for query `t`); the remaining `n − 3` are the top-scored of the rest. If fewer than `n` selection blocks are causally valid, all valid blocks are taken. The scheme is only defined for `n ≥ 3` (1 initial + 2 local) — `NsaConfig::validate` asserts `num_selected ≥ 3` rather than silently dropping a forced block when `n < 3`.

   Selection attention is standard scaled-dot-product over the gathered tokens of the selected blocks. Because a selected block may be partial, **the gather is hard-causal**: only tokens with index `≤ t` are gathered into the score buffer — future tokens are never scored, never softmaxed, never summed. (An earlier draft soft-masked them with a finite additive sentinel; round-2 review showed that leaks when a real score falls below the sentinel — hard exclusion has no value-dependent failure mode.)

8. **Sliding window**: standard causal local attention over `k_rope_{max(0, t−w+1) .. t}` — the last `w` tokens inclusive of `t`.

9. **Gating (Eq. 5)**: three per-head gates `g_cmp, g_slc, g_win = sigmoid(W_g · x_t)` where `W_g` is `[3·num_heads, dim]` and `x_t` is the caller-supplied (already-normed) hidden state. Gates are **independent sigmoids in [0,1], not normalized to sum to 1**. Output: `o_t = g_cmp · o_t^cmp + g_slc · o_t^slc + g_win · o_t^win`. The kernel takes `x` and `g_proj`; computing the gate internally keeps the NSA gating faithful and self-contained.

10. **Empty-branch handling.** Early queries (`t < l + d`, etc.) may have zero causally-valid compression or selection blocks. In that case the branch output is the zero vector — the gate still scales it, and the sliding-window branch (always non-empty for `t ≥ 0`) carries those positions. This is a faithful-ish choice the paper does not spell out; documented and covered by tests.

11. **Uniform `head_dim`.** The paper uses asymmetric dims (`d_k = 192`, `d_v = 128`, an MLA detail). All existing Lattice attention modules use a uniform `head_dim`; NSA follows suit. Asymmetric dims are a deferred extension.

12. **Hard causal exclusion, no additive mask.** All three branches exclude causally-invalid tokens/blocks *before* softmax — they are never scored: compression skips blocks not fully seen, selection gathers only `≤ t` tokens, the sliding window iterates only `≤ t`. NSA deliberately diverges from the additive `−10,000.0` mask convention of ADR-010/ADR-041: a finite additive sentinel leaks when a real score falls below it (round-2 review caught exactly this in the selection branch), and hard exclusion has no value-dependent failure mode.

13. **Standalone, not wired into a model.** No checkpoint targets this φ-MLP NSA variant — the one public checkpoint, `zen-E/NSA-1B`, is the fla-org mean-pool variant — so wiring it into a forward pass would be speculative. The module is independently testable and benchmarkable; model integration is deferred until a concrete checkpoint target exists.

---

## Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|---|---|---|---|
| Mean-pooling compression (`fla-org` variant) | Parameter-free; no φ weights to supply | Explicitly *not* the paper; discards the learnable MLP φ | Ocean chose the faithful paper algorithm |
| Selection-branch-only ("ship the seed") | Smallest; isolates the novel kernel piece | Not the full mechanism; defers the gated three-branch merge | Ocean chose the full faithful implementation |
| Defer NSA entirely | Avoids building on a nonexistent checkpoint | Leaves the bench-oriented attention series at 2/3 | Ocean chose to build it |
| Follow `lucidrains` exactly | One coherent reference to transcribe | Diverges from the paper (logits-as-scores, mean aggregation, mem-KV, left-pad) | Ocean asked for paper-faithful |
| Single Q/K/V set, RoPE'd by caller (ADR-010 convention) | Matches the other modules' signature | Paper §3.3.3 mandates independent K/V per branch, and compression needs non-RoPE Q/K | Faithfulness requires 8 activation buffers: `q`, `q_rope`, `k_cmp`, `k_slc`, `k_win`, `v_cmp`, `v_slc`, `v_win` |

---

## Consequences

**Positive**:

- NSA's paper algorithm is implemented and independently testable/benchmarkable; when an NSA checkpoint becomes a target, the math is ready.
- The three branches are separable and individually testable (compression, selection, sliding window, gating).
- Paper-vs-reference divergences are pinned here, so the implementation and its naive reference oracle share one explicit, documented convention.

**Negative**:

- Seventh attention module — the largest one — adds a seventh test suite to maintain. The kernel has ~9 distinct sub-pieces (φ forward ×2, compression attention, Eq. 9 aggregation, Eq. 10 aggregation, top-`n` selection, selection attention, sliding window, gating, merge).
- The kernel signature is heavier than the other attention modules: it takes 8 activation buffers (non-RoPE and RoPE'd Q; independent K and V per branch — `k_cmp`/`k_slc`/`k_win`, `v_cmp`/`v_slc`/`v_win`), the hidden state `x`, and an `NsaWeights` struct. This is the faithful ABI per paper §3.3.3, not an ergonomics regression that can be trimmed.

**Risks**:

- **No checkpoint validation for this φ-MLP variant — and, unlike ADR-041, no single canonical reference.** Differential attention could be transcribed from Microsoft's `multihead_flashdiff_1.py`; NSA cannot — the public references diverge from the paper and from each other. A public checkpoint (`zen-E/NSA-1B`) exists, but it is the `fla-org` mean-pool compression variant, not this φ-MLP algorithm, so it is not a drop-in oracle. Tests therefore validate *self-consistency* (optimized kernel vs. a naive exact-math oracle) plus causal-masking differential tests and the window-only-equals-dense-causal invariant. **Round-1 review exposed the structural weakness of self-consistency**: the kernel and the naive oracle had both transcribed a wrong Eq. 9 sign from this ADR, so the parity test could not catch it (implementation-independent, but not *specification*-independent). Mitigation: the corrected Eq. 9 is quoted verbatim from the paper here, and `aggregate_selection_importance` carries a hand-computed unit test grounded directly in the paper — not in either code path. This remains weaker verification than ADR-041 had; implementing the fla-org mean-pool variant to validate against `zen-E/NSA-1B` is the concrete path to checkpoint-level validation.
- **φ's architecture and the intra-block-PE form are educated guesses** — the paper underspecifies both. A real NSA checkpoint may use a different φ; the weight-loading path will need revision when one exists.
- **Performance is unproven.** The paper's 2–9× is a GPU/FlashAttention-2 number. On CPU/Metal the sparse gather + three-branch + φ overhead may be *slower* than dense attention at Lattice's current sequence lengths. The benchmark measures this honestly rather than assuming the speedup transfers.
- The lambda-style caller-supplied `NsaWeights` has no weight-loading path; the current module accepts them as a struct but does not define checkpoint key naming.

---

## References

### Papers

- Yuan et al. (2025) — "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" — arXiv:2502.11089 — <https://arxiv.org/abs/2502.11089> — ACL 2025. Eq. 5 (gated merge), Eq. 7–8 (compression), Eq. 9 (selection-block importance, index `(l'/d)·j − m − n`), Eq. 10–12 (GQA group sum, top-`n` selection), §3.3.3 (sliding window; *independent K/V per branch* — "we provide independent keys and values for three branches"), §4.1 (hyperparameter defaults: `l=32, d=16, l'=64, n=16, w=512`; the forced "1 initial + 2 local" blocks).

### Reference implementations

- `lucidrains/native-sparse-attention-pytorch` — <https://github.com/lucidrains/native-sparse-attention-pytorch> — pure-PyTorch. Source for the paper-underspecified pieces: the φ MLP architecture (`compress_networks.py`), the learned additive intra-block position encoding, the `Linear(dim, 3·heads)` + sigmoid gate, and the non-RoPE-compression / RoPE'd-selection split (`native_sparse_attention.py`). Diverges from the paper on importance-score source, Eq. 9 aggregation, left-padding, and memory-KV — those divergences are *not* followed.
- `fla-org/native-sparse-attention` — <https://github.com/fla-org/native-sparse-attention> — Triton kernels. Uses mean-pooling instead of φ and conflates the compression/selection block sizes; consulted but not transcribed (it is the less paper-faithful variant).

### Internal

- `src/attention/native_sparse.rs` — `NsaConfig`, `NsaWeights`, `NsaScratch`, `apply_native_sparse_attention()`
- `crates/inference/tests/native_sparse_attention_test.rs` — naive paper-faithful reference oracle and parity tests
- `crates/inference/benches/native_sparse_attention_bench.rs` — throughput and sparse-vs-dense benchmark
- ADR-010 — Attention Mechanisms (parent ADR for all attention variants)
- ADR-040 — Gated Attention; ADR-041 — Differential Attention (sibling attention-mechanism ADRs)
