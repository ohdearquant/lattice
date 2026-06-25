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
        });
        indices.truncate(k);
    }

    let Some(mut probs) = build_softmax_probs(&adjusted, &indices) else {
        return greedy_token(&adjusted);
    };

    probs.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

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
/// `r` that reached exactly 1.0). Falls back to the first (highest-probability)
/// token only if float error leaves the final cumsum just under `r`.
fn draw_index(probs: &[(usize, f32)], r: f32) -> u32 {
    let mut cumsum = 0.0f32;
    for &(idx, p) in probs {
        cumsum += p;
        if r < cumsum {
            return idx as u32;
        }
    }
    probs[0].0 as u32
}

/// Canonical uniform f32 in [0, 1) via the shared xorshift64 primitive.
///
/// Delegates to `crate::sampling::xorshift64_next` (shifts 13/7/17, full period
/// over nonzero state) and `crate::sampling::uniform_f32_from_u64` (top-24-bit
/// conversion, provably strictly < 1.0). Both CPU sampling paths now share this
/// canonical RNG primitive, so a fixed seed yields the same uniform draw stream.
/// This does NOT guarantee identical token streams: the two paths still diverge
/// below the RNG layer (top-k tie-break and the candidate-exhaustion fallback
/// pick in opposite directions), so equal draws can still select different tokens.
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
}
