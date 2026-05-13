# ADR-010: Attention Mechanisms

**Status**: Accepted\
**Date**: 2026-05-13\
**Crate**: `lattice-inference`

---

## Context

The crate supports three distinct attention algorithms, each suited to different model architectures and sequence lengths. Selecting the wrong algorithm produces either wrong outputs (bidirectional vs causal masking) or suboptimal performance (full materialization at long sequences).

**Standard Multi-Head Attention (MHA)** — `src/attention/standard.rs`:

- Used by BERT encoder models.
- Bidirectional: no causal mask.
- Equal Q, K, V head counts.
- Pre-allocated `AttentionBuffers` (q/k/v/scores/context/concat/ffn_intermediate + per-head reshape buffers).
- Padding positions masked with `-10,000.0` (not `-inf` to avoid NaN propagation).
- Matmul convention: `matmul_bt` (computes A @ B^T; B is stored row-major, transposed logically).

**Grouped Query Attention (GQA)** — `src/attention/gqa.rs`:

- Used by Qwen3 decoder models.
- Causal (future positions masked to zero after softmax, not with additive mask before softmax).
- Fewer KV heads than Q heads (`groups = num_q_heads / num_kv_heads`).
- `GqaScratch` batches all query groups for a single KV head: `q_batch [groups×seq, head_dim]`, `scores_batch [groups×seq, seq]`.
- macOS: uses Accelerate `sgemm_bt_strided` to read K with stride without copying.
- Non-macOS: copies K to contiguous buffer before GEMM.
- `apply_scaled_causal_softmax_fused()`: single-pass scale + causal mask (zeros future) + softmax.

**Flash Attention (CPU tiled)** — `src/attention/flash.rs`:

- Used for long sequences (>= 96 tokens, `TILED_SEQ_THRESHOLD`).
- Online softmax with tile-based computation to stay within L1 cache.
- Tile size heuristic: maximizes `Br × Bc` (tile area) subject to `2*Br*D + 2*Bc*D + Br*Bc <= 48KB` L1 budget.
- Prefers larger `tile_q` (Br) over `tile_kv` (Bc) when area is equal (output accumulator stays resident).
- Falls back to materialized path for seq_len < 96 (tiling overhead exceeds benefit).

**GatedDeltaNet (linear attention)** — `src/attention/gdn.rs`:

- Used in Qwen3.5 hybrid layers.
- Recurrent O(seq) complexity vs O(seq²) for standard attention.
- State matrix `S [key_dim × value_dim]` updated per token: `S = alpha * (S - beta * k^T @ (S @ k)) + beta * k^T @ v`.
- Causal depthwise conv1d preprocessing of Q/K/V projections.
- `GatedDeltaNetState`: `s_matrices [num_heads × key_dim × value_dim]`, `conv_buffer [conv_dim × (kernel_size-1)]`.

**Metal GPU fused attention** — `src/forward/metal.rs` (macOS only):

- `fused_attention` kernel: online softmax, GQA-aware, causal mask, no global scores buffer.
- `fused_qk_norm_rope`: combines QK normalization + RoPE into 1 kernel dispatch (was 4 separate dispatches).
- Kernel constants `__FA_HEAD_DIM__` and `__FA_GQA_GROUPS__` injected at model init via `msl_source_for()`.

---

## Decision

Implement **four attention variants** (MHA, GQA, CPU flash/tiled, GatedDeltaNet) as separate modules under `src/attention/`. Select the appropriate variant at the call site in the model forward pass based on architecture type. For Metal GPU, use fused kernels that combine multiple attention sub-operations into single dispatches. Dispatch to CPU flash attention when `seq_len >= 96` and the architecture supports it.

---

## Key Design Choices

1. **Separate modules, not a single configurable attention function**: Each variant has distinct internal state (scratch buffers, stride patterns, normalization style). A unified function with a `mode: AttentionMode` parameter would share nothing meaningful between paths while obscuring which path is taken. Separate modules make each path independently testable and optimizable.
2. **`-10,000.0` padding mask, not `-inf`**: Negative infinity can produce NaN in softmax when a row is fully masked (exp(-inf) = 0, sum = 0, 0/0 = NaN). A large negative value (-10,000.0) produces exp(-10,000) ≈ 0 without the NaN risk. This follows the HuggingFace BERT implementation convention.
3. **Zero-out causal mask in GQA (not additive before softmax)**: Standard causal masking adds a large negative value before softmax. The GQA path instead zeros the future-position probabilities after softmax scale but before normalization. This is equivalent for valid sequences but avoids the numerical risk of additive -inf.
4. **`sgemm_bt_strided` on macOS**: Accelerate's BLAS handles strided K access without a copy. K is stored in the KV cache as `[seq_len, kv_dim]`; each KV head's slice is a stride away. On non-macOS, the GQA path copies K to a contiguous buffer before calling the generic BLAS. This is an OS-specific optimization, not a correctness difference.
5. **48 KB L1 budget constant for flash tiling**: A conservative estimate (48 KB = 75% of the minimum 64 KB L1d on Apple Silicon). The constant is documented with its derivation in the source. Choosing too large a budget causes cache thrashing that nullifies the tiling benefit.
6. **MSL kernel constants injected at model init**: The Metal `fused_attention` kernel uses compile-time constants for `HEAD_DIM` and `GQA_GROUPS` to enable static array sizing and loop unrolling in MSL. These are injected via string substitution in `msl_source_for()` at model initialization and compiled once into a `MTLLibrary`. Different model variants produce different compiled kernels.
7. **GatedDeltaNet as a separate module**: Linear attention has a completely different computation graph (recurrent update, no scores matrix, no softmax). It cannot share the flash or GQA scaffolding. The `gated_delta_net_step()` function processes one token at a time (autoregressive); a future CUDA/Metal path would require a different algorithm for parallel prefill.

---

## Alternatives Considered

| Alternative                                      | Pros                          | Cons                                                                                      | Why Not                                                                 |
| ------------------------------------------------ | ----------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Single `attention()` function with flags         | One API surface               | Cannot statically elide irrelevant paths; every addition bloats the function              | Separate modules are independently optimizable and have no shared state |
| FlashAttention-2 (Tri Dao) via Rust FFI          | Optimized CUDA implementation | Requires CUDA; not pure Rust; platform-specific binary                                    | Crate policy: pure Rust, no CUDA                                        |
| xformers attention                               | Memory-efficient; well-tested | Python/CUDA ecosystem; not embeddable                                                     | Same as above                                                           |
| Scaled dot-product attention (PyTorch primitive) | Optimal on its platforms      | Requires LibTorch FFI                                                                     | Dependency constraint                                                   |
| Sparse attention                                 | Sublinear for sparse patterns | Implementation complexity; requires sparse input structure                                | Target workloads are dense sequences                                    |
| Linear attention for all layers                  | O(n) complexity               | Lower quality than softmax attention; not compatible with existing BERT/Qwen3 checkpoints | Qwen3.5 uses GDN selectively in hybrid layers                           |

---

## Consequences

**Positive**:

- Each attention path is independently testable and benchmarkable.
- Flash tiling engages automatically for long sequences without caller intervention.
- Metal fused kernels reduce GPU dispatch overhead from 4 to 1 for QK-norm+RoPE.
- GQA groups batching reduces the number of GEMM calls proportionally to the group count.

**Negative**:

- Four attention modules require four test suites and four sets of correctness checks.
- The `sgemm_bt_strided` macOS optimization creates a platform divergence in GQA (different code paths, same output). A bug that appears only on macOS (or only on non-macOS) is harder to reproduce.
- Flash attention threshold `TILED_SEQ_THRESHOLD = 96` is a heuristic that may need tuning for new hardware targets.

**Risks**:

- GatedDeltaNet state must be reset between unrelated sequences. Failing to call `GatedDeltaNetState::reset()` between users in a multi-tenant server will leak recurrent state across sequences — wrong outputs with no error.
- Metal kernel constants are injected once at model init. A model reload that changes `head_dim` or GQA groups without recompiling the MSL library will produce wrong results silently.

---

## References

- `src/attention/standard.rs` — `multi_head_attention_in_place()`, `AttentionBuffers`, `-10,000.0` mask
- `src/attention/gqa.rs` — `apply_gqa_attention()`, `GqaConfig`, `GqaScratch`, `apply_scaled_causal_softmax_fused()`
- `src/attention/flash.rs` — `TiledAttentionConfig`, `TiledAttentionBuffers`, `optimal_tile_sizes()`, `TILED_SEQ_THRESHOLD`
- `src/attention/gdn.rs` — `gated_delta_net_step()`, `GatedDeltaNetState`, `GatedDeltaNetScratch`
- `src/forward/metal.rs` — `fused_attention`, `fused_qk_norm_rope`, `msl_source_for()`
- Dao et al. 2022 — "FlashAttention" — https://arxiv.org/abs/2205.14135
