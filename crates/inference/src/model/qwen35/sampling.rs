//! Qwen3.5 token sampling, repetition penalty, greedy fallback, softmax probability build, top-p filtering, distribution draw, and RNG helper.
use crate::model::qwen35_config::GenerateConfig;

/// Sample a token from logits using temperature, top-k, top-p, and repetition penalty.
///
/// Delegates to `crate::sampling::sample_full_logits`, the shared engine also
/// used by the Metal CPU fallback (`forward/metal_qwen35.rs`'s private
/// `sample_token`). See that function's doc comment for the three
/// optimization levers (context-id-set repetition penalty, partial-select
/// top-k with softmax over the surviving k only, thread-local buffer reuse
/// across decode steps) applied here relative to the old per-call allocating
/// implementation, preserved below as `sample_token_reference` for the
/// mutation-sensitive parity tests.
pub(crate) fn sample_token(
    logits: &[f32],
    cfg: &GenerateConfig,
    previous_ids: &[u32],
    rng_state: &mut u64,
) -> u32 {
    crate::sampling::sample_full_logits(logits, cfg, previous_ids, rng_state)
}

/// Reference oracle: the original allocating implementation of `sample_token`,
/// kept byte-for-byte so a fixed-seed decode stream can be checked against it
/// for regressions introduced by the `sample_full_logits` optimization. Not
/// used on any production call path — test-only, so it (and the helpers below
/// it now exclusively calls) do not trip `-D warnings` dead-code lints in a
/// plain (non-test) `cargo check` / `cargo clippy` pass.
#[cfg(test)]
pub(crate) fn sample_token_reference(
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

#[cfg(test)]
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
/// returns 0 on empty / all-`NaN` / fully-masked (all-`NEG_INFINITY`) input.
///
/// Returning index 0 on such a degenerate distribution is the intentional
/// fail-closed behavior, not an accident of the loop: it is the dense-array
/// form of the engine-wide first-in-set contract, so a fully-masked
/// distribution yields the first in-vocabulary token deterministically and
/// never panics.
#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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

    /// Regression test for a blocker found on PR #651:
    /// `sample_full_logits` always routes through `select_top_k`, whose
    /// scalar/NEON seed phases rewrite a NaN *scaled* logit to
    /// `f32::NEG_INFINITY` so the min-heap root is never NaN. That
    /// sanitization erases the NaN before
    /// `CandidateSet::sample_top_p_with_scratch`'s own poisoned-sum guard
    /// ever sees it, silently turning a fail-closed case into a normal
    /// weighted draw. Proven counterexample: `top_k=0`, logits
    /// `[0.0, NaN, 0.0]`, seed `0x5eed_f00d_1234_5678` returned token 2
    /// pre-fix; the fail-closed contract requires the finite argmax, token 0.
    #[test]
    fn test_full_logit_nan_fails_closed_topk_disabled() {
        let logits = [0.0_f32, f32::NAN, 0.0];
        let mut rng = 0x5eed_f00d_1234_5678_u64;
        let token = sample_token(&logits, &cfg(1.0, 0), &[], &mut rng);
        assert_eq!(
            token, 0,
            "top_k=0 (disabled): a NaN anywhere in the vocab must fail closed \
             to argmax, not a weighted draw over the sanitized candidates"
        );
    }

    /// Same counterexample as `test_full_logit_nan_fails_closed_topk_disabled`,
    /// but with `top_k` larger than the vocab (`select_top_k` clamps `k` to
    /// `logits.len()`, taking the identical full-vocab code path).
    #[test]
    fn test_full_logit_nan_fails_closed_topk_exceeds_vocab() {
        let logits = [0.0_f32, f32::NAN, 0.0];
        let mut rng = 0x5eed_f00d_1234_5678_u64;
        let token = sample_token(&logits, &cfg(1.0, 100), &[], &mut rng);
        assert_eq!(
            token, 0,
            "top_k > vocab: a NaN anywhere in the vocab must fail closed to argmax"
        );
    }

    /// Same counterexample again with `top_k` strictly smaller than the
    /// vocab, so `select_top_k` runs its genuine partial-selection path
    /// (rather than the `k == 0`/`k >= len` full-vocab shortcut). The
    /// fail-closed scan in `sample_full_logits` runs over the *full*
    /// pre-top-k logits, so it must still catch this NaN and return the
    /// global argmax even though top-k would otherwise discard the NaN slot
    /// as the min-heap's worst (sanitized) candidate.
    #[test]
    fn test_full_logit_nan_fails_closed_topk_partial() {
        let logits = [0.0_f32, f32::NAN, 0.0];
        let mut rng = 0x5eed_f00d_1234_5678_u64;
        let token = sample_token(&logits, &cfg(1.0, 2), &[], &mut rng);
        assert_eq!(
            token, 0,
            "top_k < vocab: a NaN anywhere in the vocab must fail closed to \
             argmax, not a partial-selection weighted draw"
        );
    }

    /// All-`-inf` logits (a fully-masked distribution, e.g. every token
    /// repetition-penalized to the floor) must fail closed to the first
    /// token, matching `greedy_token`'s / `argmax_f32`'s documented
    /// first-wins-on-a-degenerate-array contract (#280) rather than
    /// panicking or producing a NaN-poisoned draw.
    #[test]
    fn test_full_logit_all_neg_inf_fails_closed_to_first_token() {
        let logits = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let mut rng = 7u64;
        let token = sample_token(&logits, &cfg(1.0, 0), &[], &mut rng);
        assert_eq!(
            token, 0,
            "all-(-inf) logits must fail closed to the first token"
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

    /// Regression guard for #520: a nonzero repetition_penalty is applied to
    /// logits BEFORE the temperature-degenerate greedy short-circuit, so
    /// `temperature=0.0` ("greedy") does not by itself guarantee the raw
    /// argmax wins when `previous_ids` (the full prompt + generated history)
    /// already contains the argmax token.
    ///
    /// This is exactly the mechanism behind the long-prefill e2e-parity
    /// divergence: `qwen35_generate`'s default `GenerateConfig` carries
    /// `repetition_penalty: 1.1` (production serving default, matching
    /// `chat_metal.rs`), while the HF reference in `e2e_parity_check.py`
    /// applies none. On an ~800+ token prompt that already contains most of
    /// the model's natural high-probability continuation tokens, penalizing
    /// every previously-seen id flips the post-prefill greedy pick away from
    /// the unpenalized argmax — reproduced here with a 3-way logit vector and
    /// a `previous_ids` history containing the argmax token, standing in for
    /// the 816-token prompt. If this test is reverted (repetition_penalty
    /// applied after the greedy short-circuit, or skipped entirely at
    /// temperature=0.0), both assertions below fail because token 0 would win
    /// regardless of `previous_ids`.
    #[test]
    fn test_repetition_penalty_can_flip_greedy_argmax_when_seen_in_history() {
        // Token 0 has the highest raw logit; token 1 is a close second. Chosen
        // so that 5.0 / 1.1 == 4.5454... falls below token 1's unpenalized 4.6,
        // flipping the argmax only when the penalty is actually applied.
        let logits = [5.0_f32, 4.6, 0.1];

        // No penalty (HF reference contract): argmax (token 0) wins regardless
        // of history.
        let mut cfg_no_penalty = cfg(0.0, 0);
        cfg_no_penalty.repetition_penalty = 1.0;
        let mut rng = 7u64;
        let token = sample_token(&logits, &cfg_no_penalty, &[0, 0, 0], &mut rng);
        assert_eq!(
            token, 0,
            "with repetition_penalty=1.0, greedy must return the raw argmax"
        );

        // With repetition_penalty=1.1 (lattice's production default) and token 0
        // already in `previous_ids` (as it would be after a long prefill that
        // happens to contain the argmax token), the penalized logit for token 0
        // drops below token 1's unpenalized logit and the greedy pick flips.
        let mut cfg_penalized = cfg(0.0, 0);
        cfg_penalized.repetition_penalty = 1.1;
        let mut rng = 7u64;
        let token = sample_token(&logits, &cfg_penalized, &[0, 0, 0], &mut rng);
        assert_eq!(
            token, 1,
            "repetition_penalty=1.1 must penalize a previously-seen argmax \
             enough to flip greedy selection to the runner-up — this is the \
             #520 divergence mechanism, not a chunked-prefill bug"
        );
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
    /// The earlier version of this test was vacuous (finding): it put
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

    /// Mutation-sensitive proof that the optimized `sample_token`
    /// (`crate::sampling::sample_full_logits`: context-id-set repetition
    /// penalty, partial-select top-k, thread-local scratch reuse) produces a
    /// BYTE-IDENTICAL token stream to `sample_token_reference` (the original
    /// full-clone / full-sort / fresh-HashSet implementation) under the
    /// issue's default production config — temperature 0.7, top_p 0.9,
    /// top_k 40, repetition_penalty 1.1 — over 256 decode steps with a
    /// realistic, growing history that contains duplicate ids and an
    /// out-of-vocabulary id (mirroring a stop-token id beyond the sampled
    /// vocabulary). If either path's optimization changes the sampled
    /// distribution, this test fails.
    #[test]
    fn optimized_sampler_matches_reference_default_issue_config() {
        let vocab_size = 2048usize;
        let logits: Vec<f32> = (0..vocab_size as u64)
            .map(|i| {
                let h = i
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (h as f32 / u64::MAX as f32) * 20.0 - 10.0
            })
            .collect();

        let cfg = GenerateConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            ..Default::default()
        };
        let seed = 0x5eed_f00d_1234_5678_u64;
        let steps = 256usize;

        // Seed history with a few prompt tokens, including a duplicate and an
        // out-of-vocabulary id (both must be handled identically by both paths).
        let seed_history: Vec<u32> = vec![3, 7, 3, vocab_size as u32 + 5];
        let mut history_opt = seed_history.clone();
        let mut history_ref = seed_history;
        let mut rng_opt = seed;
        let mut rng_ref = seed;

        for step in 0..steps {
            let token_opt = sample_token(&logits, &cfg, &history_opt, &mut rng_opt);
            let token_ref = sample_token_reference(&logits, &cfg, &history_ref, &mut rng_ref);
            assert_eq!(
                token_opt, token_ref,
                "optimized sample_token diverged from sample_token_reference at step {step}"
            );
            history_opt.push(token_opt);
            history_ref.push(token_ref);
        }
    }

    /// Mutation-sensitive proof that repetition penalty is applied BEFORE
    /// top-k selection in the optimized path, not after. Constructs logits
    /// where a previously-seen token sits inside the raw top-k window but
    /// drops below the top-k boundary once its penalty is applied. If an
    /// implementation applied top-k before the penalty, the seen token would
    /// still occupy its raw-rank slot and the optimized stream would diverge
    /// from the reference (which always penalizes first).
    #[test]
    fn optimized_sampler_penalty_before_topk_boundary() {
        // top_k = 4. Raw ranks (descending): 0 (10.0), 1 (9.9), 2 (9.8), 3 (9.7),
        // 4 (9.6), ... Token 3 is seen in history; with repetition_penalty=1.1 its
        // penalized logit (9.7 / 1.1 ≈ 8.818) drops below token 4's unpenalized
        // 9.6? No -- pick values so the drop crosses the boundary explicitly:
        // token 3's raw logit is barely inside the top-4 window; after penalty it
        // must fall below the raw rank-4 logit (token 4), swapping who survives.
        let logits: Vec<f32> = vec![10.0, 9.9, 9.8, 9.7, 9.6, 1.0, 0.5, 0.1];
        let cfg = GenerateConfig {
            temperature: 1.0,
            top_k: 4,
            top_p: 1.0,
            repetition_penalty: 1.1,
            ..Default::default()
        };
        let previous_ids = [3u32];
        let seed = 0xabad_1dea_dead_beefu64;
        let n = 200usize;

        let mut rng_opt = seed;
        let tokens_opt: Vec<u32> = (0..n)
            .map(|_| sample_token(&logits, &cfg, &previous_ids, &mut rng_opt))
            .collect();

        let mut rng_ref = seed;
        let tokens_ref: Vec<u32> = (0..n)
            .map(|_| sample_token_reference(&logits, &cfg, &previous_ids, &mut rng_ref))
            .collect();

        assert_eq!(
            tokens_opt, tokens_ref,
            "optimized sample_token must apply repetition penalty before top-k \
             selection, exactly like sample_token_reference; a top-k-before-penalty \
             regression would keep token 3 in the surviving set instead of token 4 \
             and diverge this stream"
        );

        // Sanity: token 4 (not in `previous_ids`, unpenalized 9.6) must actually be
        // reachable in the surviving set now that token 3's penalized logit
        // (9.7 / 1.1 ≈ 8.818) drops below it -- otherwise this test is vacuous.
        assert!(
            tokens_opt.contains(&4),
            "token 4 must be selectable once the penalty pushes token 3 below it \
             in the top-k ranking (test would be vacuous otherwise)"
        );
    }
}
