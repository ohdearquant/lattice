//! Qwen3.5 token sampling, repetition penalty, greedy fallback, top-n-sigma
//! filtering, softmax probability build, min-p/top-p filtering, distribution
//! draw, and RNG helper.
use crate::generation::GenerateConfig;

/// Sample a token from logits using temperature, top-k, min-p, top-p, and repetition penalty.
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
    // `min_p` is read from `GenerateConfig` (0.0 = disabled, its default, when
    // no caller sets it). `top_n_sigma` has no `GenerateConfig` field yet --
    // see `crate::sampling::Sampler::with_top_n_sigma` -- and no production
    // entry point sets it, so that path stays disabled here.
    crate::sampling::sample_full_logits(logits, cfg, previous_ids, rng_state, cfg.min_p, 0.0)
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
    min_p: f32,
    top_n_sigma: f32,
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

    // Mirrors the optimized engine: top-n-sigma masks the penalized,
    // still-unscaled logits after the degenerate-temperature return (masking
    // strictly below the max cannot change an argmax) and, like it, skips the
    // filter for `top_k == 1`, where the selection is argmax-shaped. This
    // implementation is deliberately independent (two-pass mean/variance vs
    // the engine's single Welford pass) so a statistics bug in either shows
    // up as a parity failure rather than being shared.
    let top_n_sigma_enabled =
        cfg.top_k != 1 && apply_top_n_sigma_reference(&mut adjusted, top_n_sigma);
    if top_n_sigma_enabled {
        let mut has_nan = false;
        let mut max_logit = f32::NEG_INFINITY;
        for &logit in &adjusted {
            has_nan |= logit.is_nan();
            max_logit = max_logit.max(logit);
        }
        if has_nan || !max_logit.is_finite() {
            return greedy_token(&adjusted);
        }
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
    // CandidateSet::sample_min_p_top_p_with_scratch (which sorts candidates first).
    // Identical iteration order guarantees bit-identical softmax sums and
    // therefore bit-identical probabilities — required for the cross-path
    // parity test and for exact numerical alignment at the draw boundary.
    indices.sort_unstable_by(|&a, &b| {
        adjusted[b]
            .partial_cmp(&adjusted[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    if top_n_sigma_enabled {
        indices.retain(|&idx| adjusted[idx] != f32::NEG_INFINITY);
    }

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

    apply_min_p(&mut probs, min_p);

    if cfg.top_p < 1.0 {
        apply_top_p(&mut probs, cfg.top_p);
    }

    draw_from_distribution(&probs, rng_state)
}

/// Test-only oracle for the engine's `apply_top_n_sigma`: masks logits below
/// `max - top_n_sigma * population_stddev`, computing the statistics with an
/// independent two-pass mean/variance instead of the engine's Welford pass.
/// Returns whether the filter was active (so the caller replicates the
/// engine's fail-closed NaN/non-finite-max scan only when it ran).
#[cfg(test)]
fn apply_top_n_sigma_reference(adjusted: &mut [f32], top_n_sigma: f32) -> bool {
    if !top_n_sigma.is_finite() || top_n_sigma <= 0.0 || adjusted.len() <= 1 {
        return false;
    }

    let mut finite_count = 0_u64;
    let mut sum = 0.0_f64;
    let mut max_logit = f32::NEG_INFINITY;
    for &logit in adjusted.iter() {
        if logit == f32::NEG_INFINITY {
            continue;
        }
        if !logit.is_finite() {
            return true;
        }
        finite_count += 1;
        sum += f64::from(logit);
        max_logit = max_logit.max(logit);
    }
    if finite_count == 0 {
        return true;
    }

    let mean = sum / finite_count as f64;
    let variance = adjusted
        .iter()
        .copied()
        .filter(|&logit| logit != f32::NEG_INFINITY)
        .map(|logit| {
            let delta = f64::from(logit) - mean;
            delta * delta
        })
        .sum::<f64>()
        / finite_count as f64;
    let threshold = f64::from(max_logit) - f64::from(top_n_sigma) * variance.sqrt();
    for logit in adjusted {
        if f64::from(*logit) < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
    true
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
    let probs: Vec<(usize, f32)> = indices
        .iter()
        .map(|&i| (i, (adjusted[i] - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    // A finite max does not guarantee a finite sum: a NaN logit in a non-max
    // position (NaN never becomes `max_logit`) poisons the sum. Same fallback.
    if !sum.is_finite() || sum <= 0.0 {
        return None;
    }
    // Deliberately NOT normalized here (issue #1454 / duplicate #1473): `probs`
    // is returned as raw, unnormalized softmax weights `exp(logit - max_logit)`,
    // so the top survivor's weight is always exactly 1.0 (`exp(0.0)`).
    // Normalization happens exactly once, in `apply_min_p`, after any min-p
    // truncation -- mirroring `CandidateSet::sample_min_p_top_p_with_scratch`,
    // which filters on these same unnormalized weights and normalizes a single
    // time. The prior shape normalized here AND again in `apply_min_p` after
    // truncation; both computed the same real-valued quotient, but through two
    // f32 divisions instead of one, and the extra rounding step could move a
    // cumulative sum across the draw threshold at f32 CDF boundaries relative
    // to the engine, which never takes the extra step.
    Some(probs)
}

/// Truncate to survivors whose unnormalized weight is >= `min_p` relative to
/// the top candidate (whose weight from `build_softmax_probs` is always
/// exactly 1.0, so `probs[0].1 * min_p` below evaluates to exactly `min_p`),
/// then normalize exactly once from the (possibly truncated) raw weights.
/// Matches `CandidateSet::sample_min_p_top_p_with_scratch`: filter on
/// unnormalized weights, normalize a single time, whether or not min-p
/// truncated anything.
#[cfg(test)]
fn apply_min_p(probs: &mut Vec<(usize, f32)>, min_p: f32) {
    let min_p = if min_p.is_nan() {
        0.0
    } else {
        min_p.clamp(0.0, 1.0)
    };
    if min_p > 0.0 && !probs.is_empty() {
        let threshold = probs[0].1 * min_p;
        let cutoff = probs
            .iter()
            .position(|&(_, weight)| weight < threshold)
            .unwrap_or(probs.len())
            .max(1);
        probs.truncate(cutoff);
    }

    // `build_softmax_probs` already validated that the full-population sum of
    // these unnormalized weights is finite and positive. Every individual
    // weight is `exp(x)` for `x <= 0` (since `max_logit` is the maximum over
    // this same index set), so each is in `(0, 1]` -- never non-finite -- and
    // the `.max(1)` above guarantees at least the top survivor (weight 1.0)
    // remains. A sum over any non-empty subset of such weights is therefore
    // always finite and >= 1.0; no fallback branch is reachable here.
    let sum: f32 = probs.iter().map(|(_, weight)| weight).sum();
    for (_, weight) in probs.iter_mut() {
        *weight /= sum;
    }
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
/// iterated one. This matches `CandidateSet::sample_min_p_top_p_with_scratch` (Path A),
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
/// `CandidateSet::sample_min_p_top_p_with_scratch`) and the tie-break alignment (the
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
mod tests;
