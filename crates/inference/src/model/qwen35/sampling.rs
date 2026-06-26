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

    let Some(mut probs) = build_softmax_probs(&adjusted, &indices) else {
        return greedy_token(&adjusted);
    };

    probs.sort_unstable_by(|(ia, a), (ib, b)| {
        b.partial_cmp(a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ia.cmp(ib))
    });

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

fn greedy_token(adjusted: &[f32]) -> u32 {
    adjusted
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
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
/// `CandidateSet::sample_top_p_with_scratch`) and the tie-break alignment (top-k
/// selection, the probability sort, and the softmax summation order all now match
/// Path A), a fixed seed produces token-identical streams for inputs without exact
/// logit ties at the top-k boundary (a rare event for realistic model outputs).
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

    /// Parity proof: `Sampler::sample` (Path A) and `sample_token` (Path B)
    /// must produce IDENTICAL token sequences for the same logits, config, and seed.
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

        // Sanity: r below the first bucket selects token 0; r at/above the last
        // bucket's cumsum (only reachable via float error) falls back to token 0.
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

    /// Parity proof with exact logit ties: both sampling paths must produce
    /// IDENTICAL token sequences even when the logit set contains exact f32 ties.
    ///
    /// Two tie scenarios are exercised simultaneously:
    ///
    /// 1. **Within-top-40 tie** (indices 5 and 6 share logit 59.0):
    ///    Both are selected by top-k. Correct sort order places index 5 before
    ///    index 6 (ascending token-id tie-break). A reversed tie-break reorders
    ///    the softmax iteration → different accumulated sums → different cumsum
    ///    thresholds → different tokens drawn.
    ///
    /// 2. **Boundary tie** (indices 39 and 40 share logit 25.0):
    ///    Exactly 39 tokens rank above this pair, so rank 40 is contested.
    ///    Correct tie-break retains index 39 (lower id) and evicts index 40.
    ///    A reversed tie-break swaps them → different token pool → divergent
    ///    sequences.
    ///
    /// This test will FAIL if either path breaks ties towards the higher
    /// token-id, because the resulting candidate sets and/or sort orders diverge.
    #[test]
    fn cross_path_parity_with_logit_ties() {
        use crate::model::qwen35_config::GenerateConfig;
        use crate::sampling::{Sampler, SamplingConfig};

        // Base logits: index i gets value (64 − i) so the natural sorted order
        // is strictly descending, all distinct.
        let mut logits: Vec<f32> = (0..64).map(|i| 64.0_f32 - i as f32).collect();

        // Tie 1 (within top-40): indices 5 and 6 both receive logit 59.0.
        // Index 5 originally had 59.0; set index 6 to match.
        // Both survive top-k selection. Correct sort: 5 before 6.
        logits[6] = logits[5]; // 59.0

        // Tie 2 (boundary): indices 39 and 40 both receive logit 25.0.
        // Index 39 originally had 25.0; set index 40 to match.
        // The 39 tokens above this pair fill ranks 1–39, leaving rank 40
        // contested. Correct tie-break: index 39 wins; index 40 is evicted.
        logits[40] = logits[39]; // 25.0

        let temperature = 0.8f32;
        let top_k = 40usize;
        let top_p = 0.95f32;
        let seed = 0xdead_beef_cafe_babe_u64;
        let n = 200usize;

        // Path A: Sampler (CandidateSet::sample_top_p_with_scratch)
        let config_a = SamplingConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: 1.0,
        };
        let mut sampler = Sampler::new(config_a).with_seed(seed);
        let tokens_a: Vec<u32> = (0..n).map(|_| sampler.sample(&logits)).collect();

        // Path B: sample_token (this module's pipeline)
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
             even when logits contain exact f32 ties — within top-k (indices 5 & 6) \
             and at the k=40 boundary (indices 39 & 40). A backwards tie-break \
             diverges the candidate set or softmax iteration order, which changes \
             the probability distribution and fails this assertion."
        );
    }
}
