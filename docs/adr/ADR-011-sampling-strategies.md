# ADR-011: Sampling Strategies

**Status**: Accepted\
**Date**: 2026-05-13\
**Crate**: `lattice-inference`

---

## Context

Token generation requires selecting the next token from a vocabulary distribution. Different use cases have different requirements:

- **Deterministic embedding generation**: Greedy argmax; must be fast; must not allocate.
- **Creative text generation**: Temperature + top-p + top-k sampling; must handle repetition.
- **Speculative decoding verification**: Greedy comparison of draft vs target logits (see ADR-006).

The vocabulary size for Qwen3 is 151,669 tokens. A naive approach to top-k sampling copies the full logit vector (993 KB), sorts it, and samples — suitable for batch sizes of 1 but wasteful when run at every autoregressive step.

Relevant implementation: `src/sampling.rs`.

```rust
pub struct SamplingConfig {
    pub temperature: f32,   // default 0.7 (0 = greedy)
    pub top_k: usize,       // default 50
    pub top_p: f32,         // default 0.9
    pub repetition_penalty: f32, // default 1.1
}
// SamplingConfig::greedy() factory: temperature=0, top_k=1

pub struct CandidateSet {
    // constructors: from_full_logits(), from_candidates()
    // methods: apply_repetition_penalty(), apply_temperature(),
    //          retain_top_k(), sample_top_p()
}

pub struct Sampler {
    rng: Xorshift64,              // 64-bit PRNG, period 2^64-1
    recent_tokens: Vec<u32>,      // max 64 tokens for repetition penalty
    candidate_scratch: Vec<f32>,
    prob_scratch: Vec<f32>,
    logit_scratch: Vec<f32>,      // 3 scratch buffers, pre-allocated
}
```

Key fast paths:

- **Greedy shortcut**: if `temperature <= 0` or `top_k == 1` AND argmax result is not in `recent_tokens`, skips the 993 KB clone entirely.
- **NEON top-k gate**: `select_top_k_neon()` uses `vcgtq_f32` threshold gate; ~95% of vocab tokens are below the k-th threshold and skipped with no memory write.
- **NEON argmax**: `argmax_f32_neon()` uses `vld1q_f32` + `vcgtq_f32` + `vbslq_f32` 4-wide horizontal reduction.

---

## Decision

Implement sampling as a **`Sampler` struct with pre-allocated scratch buffers** and a `CandidateSet` intermediate that separates penalty/temperature/top-k/top-p stages. Use **Xorshift64** as the PRNG. Use a **NEON threshold gate** to skip ~95% of the vocabulary before heap allocation in top-k. Provide a **greedy fast path** that avoids all allocation when argmax is not in the repetition window.

---

## Key Design Choices

1. **Pre-allocated scratch buffers on `Sampler`**: Top-k sampling requires a temporary copy of at least the top-k logits. Three scratch buffers (`candidate_scratch`, `prob_scratch`, `logit_scratch`) are allocated once at `Sampler` construction and reused. Per-call allocation of even a 200-element candidate list would create allocation pressure at 50+ tokens/second.
2. **Xorshift64 PRNG**: Xorshift64 is a 64-bit LFSR with period 2^64-1, passing all BigCrush statistical tests relevant to sampling (uniform distribution, independence). It requires a single 64-bit register and executes in ~3 cycles — orders of magnitude faster than `rand::thread_rng()` (CSPRNG with OS entropy). Sampling does not require cryptographic randomness.
3. **NEON threshold gate for top-k**: The standard top-k implementation copies all logits, partial-sorts to find the k-th largest, then keeps only the top k. The NEON gate first computes a threshold approximation using a streaming min-heap, then does a single pass with `vcgtq_f32` to emit only tokens above the threshold. For vocabulary size 151,669 and k=50, ~99.97% of tokens are below the threshold — ~150,000 of 151,669 are discarded with one 4-wide comparison each.
4. **`from_full_logits()` + `from_candidates()` on `CandidateSet`**: The first constructor builds from the full logit vector (heavy path); the second from a pre-filtered candidate list (used internally after the NEON gate). This separation allows the gate output to feed the second constructor without re-filtering.
5. **Repetition penalty applied to logit space**: `apply_repetition_penalty()` divides positive logits and multiplies negative logits (not a simple additive penalty). This is the standard HuggingFace convention: dividing by 1.1 for a token that appeared recently reduces its probability without creating negative infinity for tokens with positive logits.
6. **Greedy fast path criteria**: The fast path skips when `temperature <= 0` OR `top_k == 1`, AND the argmax token is not in `recent_tokens`. If the argmax is in `recent_tokens`, repetition penalty must be applied, which requires the logit copy. The check is O(64) (linear scan of recent_tokens) — faster than the 993 KB clone.
7. **`recent_tokens` window = 64**: 64 tokens at one per auto-regressive step is ~40–80 words of context. This is sufficient to prevent immediate n-gram repetition in generated text without excessive memory overhead.

---

## Alternatives Considered

| Alternative                                          | Pros                               | Cons                                                                            | Why Not                                                                            |
| ---------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Sort-based top-k (copy, sort, take top-k)            | Simple; correct                    | Full sort O(V log k) for V=151,669; ~993 KB copy + sort; allocates per call     | NEON gate reduces live set to ~100 tokens before sort; ~99.97% savings             |
| Reservoir sampling for top-k                         | O(V) time, O(k) space              | Cannot apply repetition penalty before sampling; requires two passes            | Two-pass structure complicates the pipeline                                        |
| `rand::thread_rng()`                                 | Cryptographically strong; easy API | OS entropy calls; ~100× slower than Xorshift64                                  | Sampling does not require cryptographic randomness                                 |
| Nucleus sampling without top-k pre-filter            | Simpler API                        | Top-p alone requires sorting the full distribution to find the nucleus boundary | Top-k pre-filter reduces the candidate set before top-p                            |
| Beam search                                          | Higher-quality generation          | O(beam_width × vocab_size) memory; incompatible with streaming decode           | Not needed for embedding models; decode quality target is adequacy, not optimality |
| Temperature scaling via softmax (not logit division) | Numerically stable; standard       | Requires computing exp() for all 151,669 logits before sampling                 | Logit-space temperature and top-k let us skip exp() for filtered-out tokens        |

---

## Consequences

**Positive**:

- Zero allocation in the greedy path when argmax is not in `recent_tokens` — the dominant case for embedding generation.
- NEON gate reduces the top-k candidate set to ~100 tokens before any heap write, enabling O(100) sort instead of O(151,669).
- `CandidateSet` API separates penalty, temperature, top-k, and top-p as composable pipeline stages.
- Xorshift64 generates uniform samples in ~3 cycles; negligible overhead vs logit processing.

**Negative**:

- The NEON threshold gate is an approximation: it uses a streaming min-heap to estimate the k-th logit, which may be slightly off due to heap imprecision. In practice, this means the gate may pass slightly more or fewer than exactly `top_k` candidates, with the exact top-k enforced by `retain_top_k()` afterward.
- The greedy fast path only activates when the argmax is not in `recent_tokens`. An adversarial input that forces the model to always predict a recent token (e.g., a degenerate attractor) will always take the slow path.
- `recent_tokens` is a fixed-size Vec<u32> cleared at `Sampler::new()`. Multi-session use requires resetting between sessions or using per-session `Sampler` instances.

**Risks**:

- Xorshift64 has known weaknesses in the low bits of generated numbers (correlations in the last few bits). Top-p sampling maps a uniform float to a token index via cumulative probability comparison, which is not sensitive to low-bit correlations. If a future use case requires sampling with fine-grained probability resolution, a higher-quality PRNG (e.g., PCG64) should be evaluated.
- The repetition penalty divides positive logits and multiplies negative logits. A model where highly-probable tokens have negative logits (unusual but possible after temperature scaling) would have their penalty applied in the opposite direction from intent. This is a known limitation of the HuggingFace convention.

---

## References

- `src/sampling.rs` — `SamplingConfig`, `CandidateSet`, `Sampler`, `select_top_k_neon()`, `argmax_f32_neon()`
- `src/speculative.rs` — greedy argmax used for draft acceptance decision
- Marsaglia 2003 — "Xorshift RNGs" — Journal of Statistical Software
- HuggingFace `transformers` — `RepetitionPenaltyLogitsProcessor` convention
- Holtzman et al. 2019 — "The Curious Case of Neural Text Degeneration" (nucleus/top-p sampling) — https://arxiv.org/abs/1904.09751
