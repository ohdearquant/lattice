# ADR-006: Speculative Decoding

**Status**: Accepted\
**Date**: 2026-05-13\
**Crate**: `lattice-inference`

---

## Context

Autoregressive generation is memory-bandwidth-bound: the dominant cost is loading model weights for each token, not arithmetic. Speculative decoding addresses this by proposing multiple draft tokens in one cheap step, then verifying them in a single target-model forward pass. If the draft is accepted, multiple tokens are committed for the cost of one forward pass.

The crate implements two speculative decoding approaches with different cost/accuracy trade-offs:

**NgramSpeculator** (`src/speculative.rs`):

```rust
pub struct NgramSpeculator {
    prompt_tokens: Vec<u32>,
    max_ngram: usize,   // default 5
    max_draft: usize,   // default 4
}
```

`speculate()` tries the longest matching n-gram first (max_ngram=5), falls back to shorter n-grams, returns up to `max_draft` draft tokens. `find_ngram()` is a linear scan O(prompt_len). No model weights required.

**MtpVerifier** (Multi-Token Prediction) (`src/speculative.rs`):

```rust
pub struct MtpConfig {
    pub draft_length: usize,          // default 4, max 8
    pub partial_rotary_factor: f32,   // 0.25
    pub num_experts: usize,           // 256
    pub num_experts_per_tok: usize,   // 8
}
```

`MtpVerifier` runs a 1-layer transformer with a Mixture-of-Experts FFN (256 experts, top-8 routing), shared expert, gated attention, and partial RoPE (only first 25% of head dimensions). Holds its own `FlatKVCache` and `RopeTable`.

The verification protocol:

- `mtp_verify_draft()`: checks the first draft token against initial target logits without calling `verify_tokens`; batch-verifies the rest; rolls back both KV caches (target model + MTP) on rejection or partial accept.
- `verify_draft()`: model-agnostic closure-based verification loop.
- `generate_with_speculation()`: high-level wrapper combining NgramSpeculator + `verify_draft`.

Key acceptance criterion: greedy argmax on both target and draft logits. The crate does not implement probabilistic rejection sampling (needed for temperature > 0 with statistical correctness guarantees).

---

## Decision

Implement **two-tier speculative decoding**: `NgramSpeculator` (zero training, prompt-lookup, linear scan) as the default speculator, and `MtpVerifier` (learned 1-layer MoE transformer) as the high-quality alternative. Verification uses greedy argmax acceptance. Cache rollback on rejection is handled via `FlatKVCache::truncate_to()`.

---

## Key Design Choices

1. **NgramSpeculator uses greedy argmax, not probabilistic rejection sampling**: Probabilistic rejection sampling (Leviathan et al. 2023) requires sampling from a corrected distribution on rejection, ensuring the output distribution exactly matches the target. Greedy argmax acceptance is simpler, has no statistical correctness guarantee at temperature > 0, but is acceptable for embedding workloads where the sequence is not sampled from a distribution anyway.
2. **Linear scan in `find_ngram()`**: An O(n) scan over prompt tokens is simple and cache-friendly. For typical embedding prompts (256–512 tokens) this is negligible. A hash map of n-grams would use more memory and add initialization cost for a marginal speedup on short prompts.
3. **Partial RoPE in MtpVerifier (factor=0.25)**: The MTP layer only rotates the first 25% of head dimensions. This is a design choice from the original Deepseek-MTP architecture: partial RoPE allows the draft head to use the same RoPE table as the target model but saves compute proportional to the unrotated fraction.
4. **256-expert MoE with top-8 routing**: The MoE FFN gives the MTP layer high capacity without proportionally increasing inference cost (only 8 of 256 experts activate per token). Expert routing is computed once per token; the selected expert weights are loaded and the rest are skipped.
5. **KV cache rollback via `truncate_to()`**: On a partial accept (draft tokens 0..k accepted, token k rejected), both the target model's KV cache and the MTP verifier's KV cache are truncated to the accepted length. `truncate_to()` sets `seq_len` without deallocating, making rollback O(1).
6. **`max_draft=4, max_ngram=5` defaults**: These match the empirical sweet spot where draft acceptance rate is high enough to give >1.5× throughput gain on typical text, and the draft proposal cost is small relative to target model cost.

---

## Alternatives Considered

| Alternative                               | Pros                                     | Cons                                                                                     | Why Not                                                                                    |
| ----------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Probabilistic rejection sampling          | Statistically correct at any temperature | Complex distribution correction on rejection; requires sampling from corrected proposal  | Added complexity; for embedding workloads, output distribution correctness is not required |
| Medusa (multiple independent draft heads) | Multiple speculative paths per step      | Each head is independent; correlation between heads is low; harder to verify in one pass | MTP draft is a sequential chain, higher correlation with target                            |
| Lookup table (static n-gram database)     | No scan overhead                         | Must be built per-corpus; large memory footprint                                         | Prompt-lookup avoids offline corpus requirement                                            |
| Draft model = smaller version of target   | High acceptance rate; same tokenizer     | Requires second model download; doubles model storage                                    | Storage and download cost; MTP verifier is much smaller (1 layer)                          |
| Hash map for n-gram lookup                | O(1) lookup                              | Build cost per prompt; memory overhead; cache-thrashing on long prompts                  | Linear scan is faster for prompt_len < 1024 due to cache line locality                     |

---

## Consequences

**Positive**:

- `NgramSpeculator` requires zero additional model weights and no training.
- `MtpVerifier` draft acceptance rate is significantly higher than n-gram for general text.
- Cache rollback is O(1) regardless of how many tokens are rejected.
- `generate_with_speculation()` API is model-agnostic (takes a verification closure).

**Negative**:

- Greedy acceptance is not statistically correct at temperature > 0. If the crate is extended for probabilistic generation, this must be revisited.
- `MtpVerifier` requires a separate checkpoint (MTP layer weights). The MTP model is not included in the base Qwen3-Embedding checkpoint.
- `find_ngram()` linear scan degrades linearly with prompt length. For prompts > 4096 tokens, a hash-based index would be faster.

**Risks**:

- The MTP layer's partial RoPE implementation (`mtp_apply_partial_rope()`) uses interleaved pairs `(2i, 2i+1)` — distinct from the target model's split-half RoPE. A bug that applies the wrong RoPE pattern to MTP will produce quietly wrong draft tokens with high rejection rates, not a crash.

---

## References

- `src/speculative.rs` — `NgramSpeculator`, `MtpVerifier`, `MtpConfig`, `mtp_verify_draft()`, `generate_with_speculation()`
- `src/kv_cache/flat.rs` — `truncate_to()` used for cache rollback
- Leviathan et al. 2023 — "Fast Inference from Transformers via Speculative Decoding" — https://arxiv.org/abs/2211.17192
- Deepseek-MTP architecture — partial RoPE factor, MoE configuration
