use crate::model::qwen35_config::GenerateConfig;

/// Sample a token from logits using temperature, top-k, top-p, and repetition penalty.
pub(crate) fn sample_token(
    logits: &[f32],
    cfg: &GenerateConfig,
    previous_ids: &[u32],
    rng_state: &mut u64,
) -> u32 {
    let vocab_size = logits.len();
    let mut adjusted = logits.to_vec();

    if cfg.repetition_penalty != 1.0 {
        apply_repetition_penalty(&mut adjusted, previous_ids, cfg.repetition_penalty);
    }

    // A degenerate temperature (non-finite, <= 0, or finite-but-tiny so its
    // reciprocal overflows or scales a logit past f32::MAX) carries no valid
    // scaling and is routed to deterministic greedy argmax before any scaling,
    // matching Sampler::sample, the Metal paths, and the documented "0.0 = greedy"
    // contract. See `crate::sampling::temperature_degenerate`.
    if crate::sampling::temperature_degenerate(cfg.temperature) {
        return greedy_token(&adjusted);
    }

    if cfg.temperature != 1.0 {
        let inv_temp = 1.0 / cfg.temperature;
        for v in &mut adjusted {
            *v *= inv_temp;
        }
    }

    let mut indices: Vec<usize> = (0..vocab_size).collect();
    if cfg.top_k > 0 && cfg.top_k < vocab_size {
        let k = cfg.top_k;
        indices.select_nth_unstable_by(k - 1, |&a, &b| {
            adjusted[b]
                .partial_cmp(&adjusted[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        indices.truncate(k);
    }

    // Sort indices by descending adjusted logit + ascending token-id so that
    // build_softmax_probs iterates in the same order as
    // CandidateSet::sample_top_p_with_scratch (which sorts candidates first).
    // Identical iteration order guarantees bit-identical softmax sums and
    // therefore bit-identical probabilities — required for the cross-path
    // parity test and for exact numerical alignment at the draw boundary.
    indices.sort_unstable_by(|&a, &b| {
        adjusted[b]
            .partial_cmp(&adjusted[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    // `indices` is already in (descending adjusted-logit, ascending token-id)
    // order from the pre-sort above. softmax is monotonic in the logit, so the
    // probabilities build_softmax_probs returns inherit exactly that order
    // (descending probability, ties broken by ascending token-id) without any
    // further sorting. Re-sorting here would double the sort cost — and for the
    // supported `top_k == 0` (no top-k filtering) that means sorting the full
    // vocabulary twice per decode step. apply_top_p and draw_index both rely only
    // on this descending order, which the pre-sort already guarantees.
    let Some(mut probs) = build_softmax_probs(&adjusted, &indices) else {
        return greedy_token(&adjusted);
    };

    if cfg.top_p < 1.0 {
        apply_top_p(&mut probs, cfg.top_p);
    }

    draw_from_distribution(&probs, rng_state)
}

fn apply_repetition_penalty(adjusted: &mut [f32], previous_ids: &[u32], penalty: f32) {
    let vocab_size = adjusted.len();
    // previous_ids is the full token history and routinely repeats ids. Penalize each
    // id exactly once (HF gather-once semantics; matches the Metal CandidateSet path).
    // Applying per occurrence would compound to penalty^N and over-suppress common
    // tokens as the sequence grows. O(n) dedup via a set — negligible beside the
    // full-vocab `adjusted` clone the caller already pays each step.
    let mut seen = std::collections::HashSet::with_capacity(previous_ids.len());
    for &id in previous_ids {
        let idx = id as usize;
        if idx < vocab_size && seen.insert(id) {
            adjusted[idx] = crate::sampling::penalized_logit(adjusted[idx], penalty);
        }
    }
}

/// Greedy argmax with **first-wins** tie-break.
///
/// `Iterator::max_by` keeps the *last* element on `Ordering::Equal`, so tied
/// maximum logits (e.g. `[0.0, 1.0, 1.0]`) returned the higher token-id — the
/// opposite of the engine-wide greedy contract. `speculative::argmax`,
/// `sampling::argmax_f32_scalar`, the Metal greedy path, and `torch.argmax` all
/// return the *first* occurrence (#280). This mirrors them exactly: strict `>`
/// from `NEG_INFINITY` skips `NaN` (a `NaN` comparison is always false) and
/// returns 0 on empty / all-`NaN` input.
fn greedy_token(adjusted: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in adjusted.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

fn build_softmax_probs(adjusted: &[f32], indices: &[usize]) -> Option<Vec<(usize, f32)>> {
    let max_logit = indices
        .iter()
        .map(|&i| adjusted[i])
        .fold(f32::NEG_INFINITY, f32::max);
    // A non-finite max (+INF token, or an empty/all-NaN/-INF index set whose
    // fold stays NEG_INFINITY) makes every `(logit - max).exp()` NaN; the
    // weighted draw then never selects a token and falls through to a wrong
    // one. Signal "degenerate" so the caller returns the argmax instead.
    if !max_logit.is_finite() {
        return None;
    }
    let mut probs: Vec<(usize, f32)> = indices
        .iter()
        .map(|&i| (i, (adjusted[i] - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    // A finite max does not guarantee a finite sum: a NaN logit in a non-max
    // position (NaN never becomes `max_logit`) poisons the sum. Same fallback.
    if !sum.is_finite() || sum <= 0.0 {
        return None;
    }
    for (_, p) in &mut probs {
        *p /= sum;
    }
    Some(probs)
}

fn apply_top_p(probs: &mut Vec<(usize, f32)>, top_p: f32) {
    let mut cumsum = 0.0f32;
    let mut cutoff = probs.len();
    for (i, (_, p)) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff = i + 1;
            break;
        }
    }
    probs.truncate(cutoff);
    let new_sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in probs.iter_mut() {
        *p /= new_sum;
    }
}

fn draw_from_distribution(probs: &[(usize, f32)], rng_state: &mut u64) -> u32 {
    draw_index(probs, xorshift64(rng_state))
}

/// Inverse-CDF draw: return the first token whose cumulative probability exceeds
/// `r`. The boundary is strict `r < cumsum`: the canonical uniform draw gives
/// `r` in `[0, 1)`, so `r < cumsum` is the textbook-correct inverse-CDF and never
/// double-counts a bucket boundary (`r <= cumsum` would, and is also unsafe for an
/// `r` that reached exactly 1.0).
///
/// Falls back to the LAST element when float-summation error leaves the final
/// cumsum just under `r`. Probs are sorted descending and renormalised to
/// sum≈1.0, so cumulative thresholds are c_0<c_1<...<c_{n-1}≈1.0. The last
/// bucket owns the half-open interval `[c_{n-2}, 1.0)`. A draw `r` that floats
/// past c_{n-1} lies in that final interval, so the correct token is the LAST
/// iterated one. This matches `CandidateSet::sample_top_p_with_scratch` (Path A),
/// which also returns `candidates.last()` for the same reason.
fn draw_index(probs: &[(usize, f32)], r: f32) -> u32 {
    let mut cumsum = 0.0f32;
    for &(idx, p) in probs {
        cumsum += p;
        if r < cumsum {
            return idx as u32;
        }
    }
    // Callers guarantee non-empty: build_softmax_probs returns None for empty
    // indices, so draw_index is only reached when probs has at least one element.
    probs[probs.len() - 1].0 as u32
}

/// Canonical uniform f32 in [0, 1) via the shared xorshift64 primitive.
///
/// Delegates to `crate::sampling::xorshift64_next` (shifts 13/7/17, full period
/// over nonzero state) and `crate::sampling::uniform_f32_from_u64` (top-24-bit
/// conversion, provably strictly < 1.0). Both CPU sampling paths share this
/// primitive so a fixed seed yields the same uniform draw stream. After the
/// fallback unification (`draw_index` now returns the LAST element, matching
/// `CandidateSet::sample_top_p_with_scratch`) and the tie-break alignment (the
/// top-k selection and the single pre-softmax index sort fix both the tie order
/// and the softmax summation order to match Path A), a fixed seed produces
/// token-identical streams for the same (logits, config, seed) — including inputs
/// with exact logit ties at the top-k boundary, which the matched ascending
/// token-id tie-break now resolves identically in both paths.
pub(crate) fn xorshift64(state: &mut u64) -> f32 {
    let x = crate::sampling::xorshift64_next(state);
    crate::sampling::uniform_f32_from_u64(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(temperature: f32, top_k: usize) -> GenerateConfig {
        GenerateConfig {
            temperature,
            top_k,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_sample_token_degenerate_temperature_falls_back_to_argmax() {
        // A degenerate temperature (non-finite, <= 0, reciprocal-overflow, or residual
        // post-multiply-overflow band) must route to greedy argmax rather than scaling
        // every logit to inf and collapsing the softmax draw to token 0. The highest
        // logit is index 1, so every degenerate temperature must return 1.
        let logits = [10.0_f32, 11.0, 9.0];
        for bad in [
            f32::NAN,
            f32::INFINITY,
            -1.0,
            0.0,
            1e-45,
            1e-39,
            f32::MIN_POSITIVE,
            1e-37,
        ] {
            let mut rng = 7u64;
            let token = sample_token(&logits, &cfg(bad, 0), &[], &mut rng);
            assert_eq!(
                token, 1,
                "degenerate temperature {bad} must fall back to argmax (token 1)"
            );
        }
    }

    #[test]
    fn test_greedy_tie_break_is_first_wins() {
        // Tied maximum logits must return the FIRST occurrence, matching
        // speculative::argmax, sampling::argmax_f32_scalar, the Metal greedy
        // path, and torch.argmax (#280). The prior `max_by` returned the LAST
        // tied index (token 2 here) — a silent greedy-parity divergence on ties.
        let logits = [0.0_f32, 1.0, 1.0];
        assert_eq!(greedy_token(&logits), 1, "first-wins on tied max");

        // Same contract through the public degenerate-temperature entry point.
        let mut rng = 3u64;
        let token = sample_token(&logits, &cfg(0.0, 0), &[], &mut rng);
        assert_eq!(token, 1, "temperature=0 greedy must be first-wins on ties");

        // NaN is skipped; the first finite max wins.
        let with_nan = [f32::NAN, 2.0_f32, 2.0];
        assert_eq!(
            greedy_token(&with_nan),
            1,
            "NaN skipped, first finite max wins"
        );
    }

    #[test]
    fn test_sample_token_valid_temperature_unchanged() {
        // A normal temperature must still take the scaling + softmax draw path
        // (byte-identical to before the degeneracy guard) and produce a valid token.
        let logits = [10.0_f32, 11.0, 9.0];
        let mut rng = 7u64;
        let token = sample_token(&logits, &cfg(0.7, 0), &[], &mut rng);
        assert!(token < 3, "valid temperature must return an in-range token");
    }

    #[test]
    fn test_inf_logit_routes_to_argmax_not_nan_draw() {
        // +INF logit with the full sampling path (non-degenerate temp, top_k=0)
        // must return the +INF token (index 2), not an arbitrary NaN-softmax draw.
        let logits = [0.0_f32, 1.0, f32::INFINITY];
        let mut rng = 42u64;
        let token = sample_token(&logits, &cfg(1.0, 0), &[], &mut rng);
        assert_eq!(
            token, 2,
            "infinite-logit token must win via argmax fallback"
        );
    }

    #[test]
    fn test_nan_in_nonmax_position_routes_to_argmax() {
        // Finite max but a NaN in a non-max slot poisons the softmax sum; the
        // guard must fall back to the finite argmax (index 0), not the NaN slot.
        let logits = [5.0_f32, f32::NAN, 1.0];
        let mut rng = 7u64;
        let token = sample_token(&logits, &cfg(1.0, 0), &[], &mut rng);
        assert_eq!(
            token, 0,
            "finite argmax must win when a non-max logit is NaN"
        );
    }

    #[test]
    fn test_empty_logits_returns_zero_without_panic() {
        let logits: [f32; 0] = [];
        let mut rng = 1u64;
        let token = sample_token(&logits, &cfg(1.0, 0), &[], &mut rng);
        assert_eq!(token, 0, "empty logits must return 0, never panic");
    }

    #[test]
    fn test_invalid_repetition_penalty_is_noop_not_signflip() {
        // A negative penalty must NOT flip the sign of a seen positive logit.
        // With penalty treated as no-op, the raw argmax (index 0) wins.
        let logits = [5.0_f32, 1.0, 0.5];
        let mut cfg_rp = cfg(0.0, 0); // temp 0 -> greedy, isolates the penalty effect
        cfg_rp.repetition_penalty = -1.0;
        let mut rng = 7u64;
        let token = sample_token(&logits, &cfg_rp, &[0], &mut rng);
        assert_eq!(
            token, 0,
            "invalid penalty must be a no-op, argmax stays index 0"
        );
    }

    #[test]
    fn test_nan_repetition_penalty_is_noop() {
        let logits = [5.0_f32, 1.0, 0.5];
        let mut cfg_rp = cfg(0.0, 0);
        cfg_rp.repetition_penalty = f32::NAN;
        let mut rng = 7u64;
        let token = sample_token(&logits, &cfg_rp, &[0], &mut rng);
        assert_eq!(token, 0, "NaN penalty must be a no-op");
    }

    /// Realistic-input parity regression guard: `Sampler::sample` (Path A) and
    /// `sample_token` (Path B) must produce IDENTICAL token sequences for the same
    /// logits, config, and seed on a typical no-tie logit vector.
    ///
    /// This guards against future drift but does NOT by itself isolate the three
    /// consolidation alignments — its input has no exact ties (so the tie-break
    /// never engages) and never forces the CDF fallback (it fires only on rare
    /// float-rounding events). The mutation-sensitive per-fix proofs live in
    /// `draw_index_fallback_returns_last_bucket_not_first` (fallback) and
    /// `cross_path_parity_with_logit_ties` (tie-break + summation order). What this
    /// test does prove: with the pre-sort in place, the two paths' softmax sums are
    /// bit-identical, so they agree on a full 200-draw realistic stream.
    ///
    /// Seed alignment: `Sampler::with_seed(S)` calls `Rng::new(S)` which sets
    /// `state = S` (for S ≠ 0). `sample_token` takes `rng_state: &mut u64`; when
    /// initialised to `S`, the first `xorshift64(&mut rng_state)` call advances
    /// the same way as `Rng::next_f32()` in Path A. Both paths consume exactly one
    /// RNG step per token draw, so the draw streams are permanently aligned.
    #[test]
    fn cross_path_parity_sampler_vs_sample_token() {
        use crate::model::qwen35_config::GenerateConfig;
        use crate::sampling::{Sampler, SamplingConfig};

        // 64-entry logit vector; hash-derived values in [-10, 10] with no exact
        // ties (all 64 u64 hash values differ in the top 24 bits of f32 with
        // overwhelming probability — expected collisions < 0.001).
        let logits: Vec<f32> = (0..64u64)
            .map(|i| {
                let h = i
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (h as f32 / u64::MAX as f32) * 20.0 - 10.0
            })
            .collect();

        let temperature = 0.8f32;
        let top_k = 40usize;
        let top_p = 0.95f32;
        let seed = 0xdead_beef_cafe_babe_u64;
        let n = 200usize;

        // Path A: Sampler — internal Rng seeded to `seed` via `with_seed`.
        let config_a = SamplingConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: 1.0,
        };
        let mut sampler = Sampler::new(config_a).with_seed(seed);
        let tokens_a: Vec<u32> = (0..n).map(|_| sampler.sample(&logits)).collect();

        // Path B: sample_token — rng_state initialised to the same seed; the first
        // xorshift64_next call advances identically to Rng::next_u64 in Path A.
        let config_b = GenerateConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let mut rng_state = seed;
        let tokens_b: Vec<u32> = (0..n)
            .map(|_| sample_token(&logits, &config_b, &[], &mut rng_state))
            .collect();

        assert_eq!(
            tokens_a, tokens_b,
            "Sampler::sample and sample_token must produce identical token streams \
             for the same logits, config, and seed (consolidation parity proof)"
        );
    }

    #[test]
    fn canonical_rng_streams_match_across_sampling_paths() {
        // Both CPU sampling paths must produce the same seeded float stream once
        // they share xorshift64_next + uniform_f32_from_u64.
        // Note: crate::model::qwen35::sampling is private, so this test lives here
        // where both xorshift64 (local) and the canonical primitives are reachable.
        let seed = 0x1234_5678_9abc_def0u64;
        let n = 32usize;

        // Path 1: this module's xorshift64 (now delegates to the canonical primitives).
        let mut state_q = seed;
        let q35: Vec<f32> = (0..n).map(|_| xorshift64(&mut state_q)).collect();

        // Path 2: direct use of xorshift64_next + uniform_f32_from_u64 from crate::sampling.
        let mut state_c = seed;
        let canonical: Vec<f32> = (0..n)
            .map(|_| {
                let x = crate::sampling::xorshift64_next(&mut state_c);
                crate::sampling::uniform_f32_from_u64(x)
            })
            .collect();

        assert_eq!(
            q35, canonical,
            "xorshift64 and canonical primitives must produce identical streams"
        );
    }

    #[test]
    fn draw_index_uses_strict_less_than_at_exact_boundary() {
        // Lock the draw boundary as `r < cumsum`, NOT `r <= cumsum`. The two
        // operators differ ONLY when `r` exactly equals a partial cumsum. Use
        // values that are exact in f32 (0.5) so there is no rounding ambiguity and
        // no dependence on the RNG reproducing a bit-identical float: r == 0.5,
        // first token prob == 0.5, so cumsum after token 0 is `0.0 + 0.5 == 0.5`.
        //   `r < cumsum`  -> `0.5 < 0.5`  == false -> skip token 0, select token 1
        //   `r <= cumsum` -> `0.5 <= 0.5` == true  -> select token 0
        // A regression back to `<=` flips this and fails the test.
        let probs: Vec<(usize, f32)> = vec![(0, 0.5), (1, 0.5)];
        assert_eq!(
            draw_index(&probs, 0.5),
            1,
            "at the exact boundary (cumsum == r), strict `r < cumsum` must skip \
             token 0 and select token 1 (a `r <= cumsum` regression selects token 0)"
        );

        // Sanity: r below the first bucket selects token 0; r between the two
        // bucket boundaries selects token 1 (these probs sum to exactly 1.0, so
        // the fallback branch is never reached here — see the dedicated test below).
        assert_eq!(
            draw_index(&probs, 0.25),
            0,
            "r below first cumsum picks token 0"
        );
        assert_eq!(
            draw_index(&probs, 0.75),
            1,
            "r between the two bucket boundaries picks token 1"
        );
    }

    #[test]
    fn draw_index_fallback_returns_last_bucket_not_first() {
        // Directly force the float-error fallback branch: probs that sum to LESS
        // than r so `r < cumsum` never fires and the loop runs to completion. The
        // correct fallback is the LAST bucket — on a descending-prob, sum≈1.0 CDF
        // the final bucket owns [c_{n-2}, 1.0), so an r that floats past the last
        // threshold belongs to the LAST token. This matches Path A's
        // `CandidateSet::sample_top_p_with_scratch`, which returns `candidates.last()`.
        //
        // This test is the mutation-sensitive proof of the fallback alignment:
        // reverting `draw_index` to the old `probs[probs.len()-1]` → `probs[0]`
        // returns idx 7 and fails this assertion. The full-stream parity tests do
        // NOT exercise this branch (it fires only on ~1e-7 float-rounding events),
        // which is exactly why this direct unit test exists.
        let probs: Vec<(usize, f32)> = vec![(7, 0.3), (8, 0.3), (9, 0.3)]; // sum 0.9 < r
        assert_eq!(
            draw_index(&probs, 0.95),
            9,
            "when cumsum never reaches r (float error), the draw belongs to the \
             LAST bucket (idx 9); the old `probs[0]` behaviour returns idx 7"
        );
    }

    /// Mutation-sensitive tie-break parity proof: both sampling paths must produce
    /// IDENTICAL token sequences when the logit set contains exact f32 ties, AND
    /// this test must FAIL if Path B breaks ties towards the higher token-id.
    ///
    /// The earlier version of this test was vacuous (codex round-1 finding): it put
    /// the ties at low-probability positions (logit 25 of 64) and used top_p=0.95,
    /// so the nucleus truncated the tied tokens before they could affect a draw —
    /// a reversed tie-break still passed. The fix is to give the tied tokens real
    /// probability mass and disable nucleus truncation so the tie-break is observable:
    ///
    /// - **top_p = 1.0** — no truncation; every kept token participates in draws.
    /// - **Tie at the maximum** (indices 0 & 1 both = 10.0): they carry ~0.38 of the
    ///   mass EACH. A reversed sort-order tie-break swaps which of the two owns the
    ///   first CDF bucket [0, 0.38), so ~77 of 200 draws flip.
    /// - **Tie at the top-k=3 boundary** (indices 2 & 3 both = 9.5): rank 3 is
    ///   contested. The correct selection tie-break retains index 2 (lower id, prob
    ///   ~0.23); a reversed one retains index 3 instead → a different ~0.23-mass token
    ///   appears in ~46 of 200 draws.
    ///
    /// Both ties carry enough probability that a reversed tie-break in EITHER the
    /// `select_nth_unstable_by` selection OR the pre-softmax sort diverges the stream
    /// well within 200 draws. Empirically verified mutation-sensitive: reverting
    /// either Path-B tie-break comparator to descending token-id fails this assertion.
    #[test]
    fn cross_path_parity_with_logit_ties() {
        use crate::model::qwen35_config::GenerateConfig;
        use crate::sampling::{Sampler, SamplingConfig};

        // 8 tokens. Two ties, both at HIGH-probability positions:
        //   idx 0 & 1 tie at the maximum (10.0)        -> sort-order tie-break
        //   idx 2 & 3 tie at the top_k=3 boundary (9.5) -> selection tie-break
        // Remaining tokens descend so the rest of the order is unambiguous.
        let logits: Vec<f32> = vec![10.0, 10.0, 9.5, 9.5, 9.0, 8.0, 7.0, 6.0];

        let temperature = 1.0f32;
        let top_k = 3usize; // rank 3 contested between idx 2 and idx 3
        let top_p = 1.0f32; // NO nucleus truncation — tied tokens stay in play
        let seed = 0xdead_beef_cafe_babe_u64;
        let n = 200usize;

        // Path A: Sampler (CandidateSet::sample_top_p_with_scratch). Keeps {0,1,2}
        // (idx 2 wins the rank-3 tie by lower id), ordered [0,1,2] (lower id first).
        let config_a = SamplingConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: 1.0,
        };
        let mut sampler = Sampler::new(config_a).with_seed(seed);
        let tokens_a: Vec<u32> = (0..n).map(|_| sampler.sample(&logits)).collect();

        // Path B: sample_token (this module's pipeline). With matching tie-breaks it
        // also keeps {0,1,2} ordered [0,1,2]; a reversed tie-break keeps {0,1,3}
        // and/or orders the max tie [1,0], diverging the stream.
        let config_b = GenerateConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let mut rng_state = seed;
        let tokens_b: Vec<u32> = (0..n)
            .map(|_| sample_token(&logits, &config_b, &[], &mut rng_state))
            .collect();

        assert_eq!(
            tokens_a, tokens_b,
            "Sampler::sample and sample_token must produce identical token streams \
             with high-probability ties at the max (idx 0 & 1) and the top_k=3 \
             boundary (idx 2 & 3). A backwards tie-break in either the selection or \
             the pre-softmax sort changes which token wins the contested rank or the \
             order of the two max-probability buckets, flipping dozens of the 200 \
             draws and failing this assertion."
        );
    }
}
