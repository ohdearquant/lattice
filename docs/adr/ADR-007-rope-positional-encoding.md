# ADR-007: Rotary Positional Encoding (RoPE)

**Status**: Accepted\
**Date**: 2026-05-13\
**Crate**: `lattice-inference`

## Amendment (2026-06-24)

The original ADR (Decision point 6 and the Consequences section) documented that
`MtpVerifier` uses **interleaved pairs `(2i, 2i+1)`** for partial RoPE. This was
**incorrect as of the v0.2.3 RoPE-convention fix**. The MTP partial RoPE path
(`speculative.rs:mtp_apply_partial_rope`) now uses **split-half `(i, half+i)`**,
matching the main-model convention. The code comment on that function reads:
"Stride-half partial RoPE: rotate pairs (i, half+i) for i in 0..rope_dim/2 —
matches HF rotate_half / MLX traditional=False."

**Do not reintroduce interleaved pairing to `mtp_apply_partial_rope`.** The v0.2.3
bug was the interleaved `(2i, 2i+1)` variant; the fix is split-half `(i, half+i)`.
Cross-reference: the "Differential Test First" section of `CLAUDE.md` in this repo
documents the v0.2.3 RoPE-convention fix and the diagnostic method used to identify
the pairing mismatch.

The original Decision and Context text is preserved below as a historical record.
Inline corrections to the false statements are marked with `[CORRECTED]`.

---

## Context

Transformer attention requires a positional encoding scheme that injects token position into the attention computation. Rotary Position Embedding (RoPE) encodes relative position by rotating query and key vectors by position-dependent angles, achieving relative-position sensitivity without modifying the value vectors or adding learned parameters.

Key design decisions concern:

1. **Precision**: Frequency tables can be computed in f32 or f64.
2. **Layout**: Rotation can be applied to interleaved pairs (dim 0,1 | 2,3 | ...) or split halves (dim 0..half | half..end).
3. **Theta parameter**: BERT-family models use θ=10,000. Qwen3 uses θ=1,000,000 (extended context via longer-wavelength frequencies).
4. **Precomputation**: Tables can be precomputed at model init or computed JIT per position.

Relevant implementation: `src/rope.rs`.

```rust
pub struct RopeTable {
    cos: Vec<f32>,  // [max_seq_len, half_dim]
    sin: Vec<f32>,  // [max_seq_len, half_dim]
    half_dim: usize,
}

impl RopeTable {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64) -> Self {
        // freq[i] = 1.0 / theta^(2*i / head_dim)  -- computed in f64
        // angle[pos, i] = pos * freq[i]             -- computed in f64
        // cos/sin stored as f32
    }

    pub fn apply(&self, x: &mut [f32], position: usize) {
        // Rotates x[i] with x[half_dim + i] for i in 0..half_dim
        // SPLIT-HALF layout, NOT interleaved
    }
}
```

Tests in `src/rope.rs` verify:

- Identity at position 0 (cos=1, sin=0 → no rotation)
- Norm preservation (rotation is an isometry)
- Precision to 1e-6 vs f64 reference up to position 262,143

The MtpVerifier in `src/speculative.rs` uses a separate partial RoPE application
(`mtp_apply_partial_rope()`) that applies rotation only to the first `rope_dim`
dimensions (25% of `head_dim`). [CORRECTED: uses split-half pairs `(i, half+i)`,
not interleaved `(2i, 2i+1)` — see Amendment above.]

---

## Decision

Precompute RoPE frequency tables at model initialization in **f64 precision, stored as f32**. Apply rotation using **split-half layout** (first half of head_dim rotated with second half). Use **θ=1,000,000** for Qwen3 (extended context) and **θ=10,000** for BERT-family models. The `RopeTable` is immutable after construction and can be shared across threads.

---

## Key Design Choices

1. **f64 computation, f32 storage**: The angle `pos * freq[i]` involves multiplying a large integer (position up to 32,768) by a small float (freq[i] can be < 1e-6 for large i). In f32, `pos * freq[i]` for large positions and small frequencies loses significant bits. Computing in f64 and casting to f32 gives sufficient precision without the memory cost of storing the entire table in f64. Tests confirm 1e-6 error bound versus a pure-f64 reference.
2. **Split-half layout instead of interleaved pairs**: The original RoPE paper uses interleaved pairs `(x[0], x[1]), (x[2], x[3])...`. The split-half variant `(x[0], x[head_dim/2]), (x[1], x[head_dim/2+1])...` is algebraically equivalent (same rotation matrix, different indexing convention) but allows vectorized reads: the entire first half and second half are each contiguous in memory, enabling SIMD loads without scatter/gather.
3. **Precomputed table**: The table for Qwen3-0.6B is `32,768 × 64 (half_dim)` f32 entries × 2 (cos+sin) × 4 bytes = 16 MB. This is modest and avoids computing `cos(pos * freq[i])` in the hot path (transcendentals are expensive: ~20–80 cycles vs ~1 cycle for a table lookup).
4. **θ=1,000,000 for Qwen3**: YaRN and similar extended-context RoPE variants use higher theta values to reduce the rotation speed of high-frequency dimensions, extending the effective context window beyond the training length. Qwen3 is trained with θ=1,000,000 and `max_position_embeddings=32,768`.
5. **Immutable after construction**: `RopeTable` is constructed once and shared as `&RopeTable`. No locking. The apply method takes `&mut [f32]` (the head buffer) and `&self`, making parallelism straightforward.
6. **MtpVerifier partial RoPE uses split-half `(i, half+i)`** [CORRECTED: the
   original ADR stated "interleaved pairs `(2i, 2i+1)` — an exception to the
   split-half convention." This was wrong. After the v0.2.3 fix, `mtp_apply_partial_rope`
   uses split-half `(i, half+i)`, identical in pairing style to the main-model
   `RopeTable::apply`. It applies to only 25% of dimensions (`rope_dim = partial_rotary_factor * head_dim`).
   There is no longer a pairing-convention exception for the MTP path.]

---

## Alternatives Considered

| Alternative                                    | Pros                                                                     | Cons                                                                                                                       | Why Not                                                                              |
| ---------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| f32 throughout (table computation and storage) | Less memory; faster table build                                          | Precision loss for large positions × small frequencies; fails 1e-6 precision test                                          | Correctness requirement: embedding quality degrades with position encoding error     |
| f64 storage                                    | Full precision preserved in table                                        | 2× memory (32 MB instead of 16 MB); f64 SIMD is 2× wider                                                                   | f32 with f64 computation meets the 1e-6 error bound; f64 storage unnecessary         |
| JIT computation per token                      | No table memory                                                          | `cos`/`sin` transcendentals per token at every layer × 28 layers × 16 heads; ~10,000 transcendental calls per forward pass | Table lookup is ~10× faster for a 28-layer model                                     |
| Learned positional encoding (BERT style)       | Adds positional context via learned parameters                           | Does not generalize to sequence lengths beyond training; no relative position information                                  | RoPE generalizes to arbitrary lengths; BERT learned PE is used only in the BERT path |
| ALiBi (attention with linear biases)           | No table; relative position via bias; generalizes beyond training length | Changes attention score formula; not compatible with existing GQA and flash attention kernels                              | Architectural incompatibility; RoPE is baked into Qwen3                              |
| Interleaved RoPE (original paper convention)   | Matches HuggingFace transformers default                                 | First half and second half are not contiguous; SIMD requires gather                                                        | Split-half is algebraically equivalent and more SIMD-friendly                        |

---

## Consequences

**Positive**:

- Table lookup in the hot path is a single indexed read: O(1), ~1 cycle after L1 hit.
- Split-half layout enables vectorized application with standard SIMD load instructions.
- Sharing one `RopeTable` instance across all layers saves repeated computation.
- Precision test suite confirms correctness to 1e-6 at position 262,143.

**Negative**:

- 16 MB table per model in steady state. For a system running five concurrent models, this is 80 MB of RoPE tables alone.
- [CORRECTED: as of v0.2.3, there is only one pairing convention in the codebase:
  split-half `(i, half+i)` for both the main-model path and the MTP partial RoPE path.
  The original concern about two conflicting conventions no longer applies.]

**Risks**:

- If a future model uses a non-standard theta schedule (e.g., NTK-aware interpolation with frequency-dependent theta), the current `RopeTable::new()` API accepts only a single scalar theta. The API must be extended.
- [CORRECTED: MTP partial RoPE is now split-half, not interleaved. The residual risk
  is the inverse: a future refactor that "unifies" MTP with the main RoPE path could
  accidentally reintroduce the interleaved bug. The function `mtp_apply_partial_rope`
  is intentionally separate to apply RoPE to only `rope_dim` (25%) of dimensions;
  do not merge it with `RopeTable::apply` without verifying the partial-dimension
  contract is preserved.]

---

## References

- `src/rope.rs` — `RopeTable`, `new()`, `apply()`, test suite
- `src/speculative.rs` — `mtp_apply_partial_rope()`, split-half convention `(i, half+i)`, `partial_rotary_factor=0.25` [CORRECTED from "interleaved-pair convention"]
- Su et al. 2021 — "RoFormer: Enhanced Transformer with Rotary Position Embedding" — https://arxiv.org/abs/2104.09864
- Qwen3 technical report — theta=1,000,000, max_position_embeddings=32,768
