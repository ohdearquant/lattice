# ADR-050: Rejection Sampling for Speculative Decoding

**Status**: Proposed
**Date**: 2026-05-19
**Crate**: `lattice-inference`

---

## Context

ADR-006 introduced two speculative decoding paths: `NgramSpeculator` (prompt-lookup) and
`MtpVerifier` (1-layer MoE transformer). Both verification paths (`mtp_verify_draft` and
`verify_draft`) accept draft tokens via greedy argmax comparison:

```rust
// Current: mtp_verify_draft, line 1156 / 1212
let model_choice = argmax(&target_logits[i - 1]) as u32;
if model_choice != draft[i] { /* reject */ }
```

Greedy argmax acceptance does NOT preserve the target model's output distribution. The output
sequence at temperature > 0 is biased toward whichever tokens the draft model favors, violating
the core guarantee that speculative decoding should produce samples indistinguishable from
auto-regressive sampling from the target. The KG gap entity `ResidualRejectionSamplingVariants`
(status="gap") explicitly flags this deficiency.

**Strict speculative sampling** (Leviathan et al. 2022 / SpecTr, NeurIPS 2023) closes this gap.
For each draft token `d_i` drawn from draft distribution `q`, the target model provides
distribution `p`. The token is accepted with probability:

```
α_i = min(1, p(d_i) / q(d_i))
```

On rejection, a correction token is drawn from:

```
p'(x) = normalize(max(0, p(x) - q(x)))
```

This guarantees the joint output distribution is exactly `p`, regardless of the draft quality.
The per-token acceptance probability is `β(θ) = 1 - TV(q, p)` where TV is total variation
distance. Expected accepted tokens per speculative step is `(1 - β^(K+1)) / (1 - β)` for K
draft tokens (capped geometric, Leviathan et al. ICML 2023, Algorithm 1). Wallclock speedup
depends on the draft/target cost ratio `c`: `speedup = E[accepted] / (1 + K·c)`.

| β | K | E[accepted] | speedup (c=0.05) |
|---|---|-------------|------------------|
| 0.5 | 2 | 1.75 | 1.59 |
| 0.8 | 5 | 3.69 | 2.95 |
| 0.9 | 5 | 4.10 | 2.73 |

**GDN complication.** Qwen3.5-2B's 18 GatedDeltaNet layers maintain a recurrent state matrix
`S[key_dim × value_dim]` per head (see `GatedDeltaNetState::s_matrices`). Unlike KV cache,
which is an append-only sequence of (K, V) pairs truncatable via `FlatKVCache::truncate_to()`,
GDN state is a dense matrix updated in-place at every token via:

```
S = g * S + outer(k, (v - S @ k) * beta)
```

This update is not invertible. Rollback via `truncate_to` handles the KV path but has no
counterpart for GDN state. A speculative rejection must restore the GDN state to the snapshot
taken before the draft was generated. Under the nonexpansive decay bound from the KG entity
`GDN Multi-Round Error Accumulation`, the error from processing N spurious tokens grows as
`O(N * eps)` — silent corruption if rollback is skipped.

---

## Decision

Replace the greedy argmax acceptance criterion in `mtp_verify_draft` with strict speculative
sampling. Add a GDN state snapshot/restore protocol to the `MtpTargetVerifier` trait to enable
correct rollback on rejection.

The implementation is scoped to three changes:

1. **Acceptance criterion**: convert draft and target logits to probability distributions via
   softmax; accept token `d_i` with probability `min(1, p(d_i) / q(d_i))`; on rejection, sample
   the correction token from `normalize(max(0, p - q))`. Temperature is a caller-provided
   parameter; at temperature 0 the sampling path degenerates to the existing argmax behavior with
   no correctness regression.

2. **GDN state snapshot protocol**: extend `MtpTargetVerifier` with `snapshot_gdn_state()` and
   `restore_gdn_state()` methods. Before draft generation begins, the verifier takes a snapshot of
   all GDN layers' `s_matrices` and `conv_buffer` fields. On partial or full rejection, restore
   is called before the correction token forward pass.

3. **MTP KV cache**: `MtpVerifier::rollback_cache_to()` already calls `FlatKVCache::truncate_to()`
   correctly. No change needed there.

Greedy path (temperature = 0.0) is preserved as a fast path that bypasses the softmax and
sampling arithmetic. This maintains backward compatibility with existing benchmarks.

---

## Scope

Files affected:

- `crates/inference/src/speculative.rs`: `MtpTargetVerifier` trait extension, `mtp_verify_draft`
  acceptance loop, new `StrictSamplingConfig` parameter struct.
- `crates/inference/src/attention/gdn.rs`: `GatedDeltaNetState` gains `snapshot()` and
  `restore_from()` methods (clone into/from a caller-owned buffer).

No changes to `NgramSpeculator`, `verify_draft` (n-gram path is greedy by design — it operates
at temperature 0 as a retrieval mechanism, not a sampler), or any Metal kernel.

---

## Architecture

### Sampling math

```
Given:
  q[v] = softmax(draft_logits / T)[v]   -- draft distribution, T = temperature
  p[v] = softmax(target_logits / T)[v]  -- target distribution

For each draft position i:
  u ~ Uniform(0, 1)
  if u < min(1, p[d_i] / q[d_i]):
      accept d_i                         -- no extra forward pass
  else:
      sample correction from p'          -- p'(x) = normalize(max(0, p(x) - q(x)))
      commit correction token, stop draft

At temperature 0: min(1, p[d_i]/q[d_i]) = 1 iff argmax(p) == d_i, else 0.
This recovers the current greedy test exactly.
```

### GDN snapshot protocol

```rust
// Extension to MtpTargetVerifier trait
fn snapshot_gdn_states(&self) -> GdnSnapshot;
fn restore_gdn_states(&mut self, snapshot: &GdnSnapshot);

// GdnSnapshot = Vec<(Vec<f32>, Vec<f32>)>
//   index i = layer i: (s_matrices.clone(), conv_buffer.clone())
// Memory budget: see ADR-052 §Memory Budget for authoritative per-config estimates.
// For Qwen3.5-0.8B: ~19.3 MiB per snapshot (18 GDN layers, 16 heads, 128×128 state + 6144×3 conv)
```

Snapshot frequency: once per speculative step (before draft generation). Not per-token. The
draft tokens are never committed to GDN state during speculative verification — the target model
processes them via a separate batched forward path that does not mutate the recurrent state.
Only accepted tokens are committed; the snapshot is discarded after full acceptance or used for
restore after any rejection.

### Integration with `mtp_verify_draft`

The function signature gains a `StrictSamplingConfig`:

```rust
pub struct StrictSamplingConfig {
    pub temperature: f32,   // 0.0 = greedy (fast path), > 0.0 = strict sampling
    pub rng_seed: u64,      // deterministic seeding for reproducibility
}
```

The acceptance loop is factored into a shared function so both the batched MTP path and the
n-gram path can use it when needed:

```rust
fn sample_acceptance(
    draft_token: u32,
    draft_logits: &[f32],
    target_logits: &[f32],
    cfg: &StrictSamplingConfig,
    rng: &mut SmallRng,
) -> AcceptanceDecision  // Accept | Reject { correction_token }
```

---

## Alternatives Considered

**A. Keep greedy argmax, add temperature via top-p/top-k on the target only.**
Partially improves output diversity but does not satisfy the distribution-preservation theorem.
The output still depends on which draft tokens were proposed. Rejected: correctness gap remains.

**B. SpecTr tree sampling (NeurIPS 2023) with tree-structured drafts.**
SpecTr extends strict sampling to tree-structured draft sets, increasing expected acceptance by
exploring multiple continuations per step. EAGLE-2/3 (arxiv:2406.16858, 2503.01840) show 3-4x
speedup over linear drafts by combining tree drafting with context-aware confidence calibration.
This is the long-term target architecture. Not adopted here because: (a) `MtpVerifier` produces
a linear draft sequence, not a tree; (b) tree verification requires parallel logit comparison
across O(K^depth) paths, needing a Metal kernel rewrite; (c) GDN snapshot cost grows with tree
width. Tree-structured drafting will be addressed in a future ADR once the correctness foundation
(this ADR) is in place.

**C. Speculative rejection sampling only for the MTP path, skip n-gram path.**
The n-gram path is a retrieval mechanism, not a learned sampler, so its "draft distribution"
is degenerate (mass 1 on the proposed token). Strict sampling on this path collapses to greedy.
This is correct and is what this ADR implements: the correction formula handles this automatically
since `q[d_i] = 1.0` makes `p[d_i] / q[d_i] = p[d_i]` which is always `<= 1`. No special-casing
needed.

---

## Risks

**R1: GDN snapshot memory.** See ADR-052 §Memory Budget for authoritative per-config estimates
(~19.3 MiB for Qwen3.5-0.8B). For larger models the snapshot scales with `num_heads * key_dim *
value_dim * num_gdn_layers`. The snapshot is ephemeral (held for one speculative step, then
discarded or restored). Mitigation: snapshot is heap-allocated once and reused across steps via
a `Vec::clear()` + `extend_from_slice` pattern to avoid per-step allocation.

**R2: RNG reproducibility.** Strict sampling introduces stochastic behavior. Tests that assert
exact token sequences must be updated to use seeded RNG or test at temperature=0. The
`StrictSamplingConfig::rng_seed` field enables deterministic reproduction of any generated
sequence.

**R3: Draft logits unavailability.** The current `MtpVerifier::draft_tokens()` returns token IDs
(line 1039), not logits. Strict sampling requires `q[d_i]`, the draft probability. The fix is to
return `Vec<(u32, Vec<f32>)>` from `draft_tokens`, or store logits alongside the draft in a new
`DraftResult` struct. This changes the internal API of `mtp_verify_draft` but not the public
`MtpTargetVerifier` trait.

**R4: Acceptance rate degradation at low temperature.** At very low but nonzero temperature, the
draft and target distributions both sharpen toward their argmax. TV distance approaches 0 and
acceptance rate approaches 1 — the sampling overhead is paid but yields no change from greedy.
This is mathematically correct, not a bug.

---

## References

- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2022),
  arxiv:2211.17192. Original strict speculative sampling proof.
- Sun et al., "SpecTr: Fast Speculative Decoding via Optimal Transport" (NeurIPS 2023).
  Per-token acceptance probability `β(θ) = 1 - TV(p, q)` and batch extension.
- Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" (EMNLP 2024),
  arxiv:2406.16858. Context-aware confidence calibration for tree width.
- Du et al., "EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test"
  (Mar 2025), arxiv:2503.01840. Tri-layer fusion for draft head; P-EAGLE single-pass variant.
- ADR-006: Speculative Decoding (Accepted, 2026-05-13) — baseline implementation this ADR amends.
- KG entities: `Speculative Decoding`, `SpecTr`, `Per-Token Acceptance Probability β(θ)`,
  `ResidualRejectionSamplingVariants` (gap), `GDN Multi-Round Error Accumulation`.
