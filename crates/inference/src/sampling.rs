//! Token sampling strategies for text generation.
//!
//! Supports temperature scaling, top-k filtering, min-p filtering, top-p
//! (nucleus) sampling, and repetition penalty.

use crate::generation::{GenerateConfig, TopLogprob};

/// Greedy argmax over a dense `f32` logit slice, with **first-wins** tie-break
/// (ADR-080 C3, #783).
///
/// `Iterator::max_by` keeps the *last* element on `Ordering::Equal`, so tied
/// maximum logits (e.g. `[0.0, 1.0, 1.0]`) return the higher token id — the
/// opposite of the engine-wide greedy contract. [`CandidateSet::argmax`],
/// `attention` decode paths, and `torch.argmax` all return the *first*
/// occurrence on a tie; this is the canonical shared implementation every
/// dense-logit-slice greedy call site should use instead of hand-rolling its
/// own `max_by` closure (the duplication this ADR-080 cluster closes).
///
/// Strict `>` from `NEG_INFINITY` skips `NaN` (a `NaN` comparison is always
/// false) and returns `0` on empty / all-`NaN` / fully-masked (all
/// `NEG_INFINITY`) input — the dense-array form of the engine-wide
/// fail-closed-to-first-in-set contract, never a panic.
///
/// Its only production callers today (`generate_greedy_mtp`,
/// `generate_greedy_self_spec` in `forward/metal_qwen35.rs`) live behind the
/// `metal-gpu` feature, so a default-feature build never calls it outside
/// this module's own tests.
#[cfg_attr(not(feature = "metal-gpu"), allow(dead_code))]
pub(crate) fn argmax_f32_first_wins(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

#[cfg(test)]
mod argmax_f32_first_wins_tests;

/// **Unstable**: sampling configuration for decoder-only generation; fields
/// and defaults may change as the generation API evolves.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SamplingConfig {
    /// Temperature for logit scaling. 0.0 = greedy, 1.0 = unscaled.
    pub temperature: f32,
    /// Top-k: keep only the k highest-probability tokens. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus): keep tokens whose cumulative probability <= p. 1.0 = disabled.
    pub top_p: f32,
    /// Min-p: keep tokens with probability at least `min_p * max_probability`,
    /// applied before top-p. 0.0 or NaN = disabled; other values clamp to
    /// `[0.0, 1.0]`. Read by [`Sampler::new`], which seeds its own out-of-band
    /// state from this field; [`Sampler::with_min_p`] overrides it afterward.
    pub min_p: f32,
    /// Repetition penalty multiplier. 1.0 = no penalty.
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            min_p: 0.0,
            repetition_penalty: 1.1,
        }
    }
}

impl SamplingConfig {
    /// **Unstable**: greedy decoding config factory.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
        }
    }
}

/// **Unstable**: a single (token_id, logit) candidate pair.
///
/// Used by `CandidateSet` to represent the GPU-compacted top-k results and
/// to share the sampling pipeline between the Metal and CPU generation paths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub token_id: u32,
    pub logit: f32,
}

/// **Unstable**: a set of (token_id, logit) candidates for categorical sampling.
///
/// Provides the sampling pipeline: apply repetition penalty → temperature →
/// top-k → min-p → top-p → sample. Both the Metal compact path and the CPU Qwen
/// path use this type so sampling semantics are shared.
pub struct CandidateSet {
    candidates: Vec<Candidate>,
}

impl CandidateSet {
    /// Build from a full logits slice (CPU Qwen fallback path).
    pub fn from_full_logits(logits: &[f32]) -> Self {
        let candidates = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| Candidate {
                token_id: i as u32,
                logit: l,
            })
            .collect();
        Self { candidates }
    }

    /// Build from an already-compacted candidate array (Metal GPU top-k path).
    pub fn from_candidates(candidates: Vec<Candidate>) -> Self {
        Self { candidates }
    }

    /// Apply repetition penalty in-place over the candidates present.
    pub fn apply_repetition_penalty(&mut self, previous_ids: &[u32], penalty: f32) {
        if penalty == 1.0 {
            return;
        }
        for c in &mut self.candidates {
            if previous_ids.contains(&c.token_id) {
                c.logit = penalized_logit(c.logit, penalty);
            }
        }
    }

    /// Return the token_id of the candidate with the highest logit.
    ///
    /// On an all-masked compact set every logit is `f32::NEG_INFINITY`, so the
    /// comparison loop never updates `best_id`.  The initializer must therefore
    /// be the first candidate's `token_id` rather than the bare sentinel `0`:
    /// after grammar or top-k compaction, token id 0 is not guaranteed to be a
    /// member of the set, and returning it would silently emit an out-of-set
    /// token.  `unwrap_or(0)` is safe only as a last resort for an empty set
    /// (the full-vocab path always contains at least one candidate).
    pub fn argmax(&self) -> u32 {
        let mut best_id = self.candidates.first().map(|c| c.token_id).unwrap_or(0);
        let mut best_val = f32::NEG_INFINITY;
        for c in &self.candidates {
            if c.logit > best_val {
                best_val = c.logit;
                best_id = c.token_id;
            }
        }
        best_id
    }

    /// Scale logits by `1 / temperature` after subtracting the largest finite logit,
    /// so the scaled maximum is 0.0 and no finite logit can overflow to +inf.
    /// No-op when temperature is 1.0,
    /// non-positive, or non-finite (NaN/±inf) — none of those carry a valid
    /// scaling, and `1.0 / NaN` would poison every logit with NaN.
    ///
    /// A finite-but-tiny temperature has a reciprocal that overflows `f32`, so it carries
    /// no valid finite scaling: the `t -> 0+` limit is hard-greedy. Scaling such a
    /// temperature (even with a clamped multiplier) leaves close logits near 50/50, so
    /// `sample_top_p` could still return a non-argmax token. Instead collapse the set to a
    /// one-hot on the argmax — the same hard-argmax route the main `Sampler` takes for a
    /// degenerate temperature (see `temperature_degenerate`).
    pub fn apply_temperature(&mut self, temperature: f32) {
        if !temperature.is_finite() || temperature <= 0.0 || temperature == 1.0 {
            return;
        }
        if temperature_degenerate(temperature) {
            let best = self.argmax();
            for c in &mut self.candidates {
                c.logit = if c.token_id == best {
                    0.0
                } else {
                    f32::NEG_INFINITY
                };
            }
            return;
        }
        // Center only when `inv > 1.0` (see `select_top_k`'s doc comment for the
        // full reasoning): scaling first overflows any logit above
        // `temperature * f32::MAX` to +inf only when `inv > 1`, and two overflowed
        // logits would then tie and break on token id instead of on value.
        // Centering unconditionally trades that failure for the opposite one: with
        // `inv <= 1` a huge negative logit can center-and-overflow to -inf even
        // though plain scaling would have kept it finite.
        let inv = 1.0 / temperature;
        let center = if inv > 1.0 {
            let raw = self
                .candidates
                .iter()
                .map(|c| c.logit)
                .filter(|logit| logit.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);
            if raw.is_finite() { raw } else { 0.0 }
        } else {
            0.0
        };
        for c in &mut self.candidates {
            c.logit = (c.logit - center) * inv;
        }
    }

    /// Keep only the top-`k` candidates by logit (O(n) partial select).
    pub fn retain_top_k(&mut self, k: usize) {
        if k == 0 || k >= self.candidates.len() {
            return;
        }
        self.candidates
            .select_nth_unstable_by(k - 1, candidate_order);
        self.candidates.truncate(k);
    }

    /// Sort descending, apply softmax, optionally truncate with top-p nucleus
    /// filtering, renormalise, and return a weighted-random sample.
    ///
    /// `r` must be a uniform f32 in [0, 1).
    pub fn sample_top_p(&mut self, top_p: f32, r: f32) -> u32 {
        let mut probs = Vec::new();
        self.sample_min_p_top_p_with_scratch(0.0, top_p, r, &mut probs)
    }

    /// Apply min-p and top-p filtering, then sample from the surviving candidates.
    ///
    /// Min-p runs after repetition penalty, temperature scaling, and top-k,
    /// but before top-p. `min_p == 0.0` disables the relative-probability
    /// filter; `min_p == 1.0` retains only candidates tied for maximum
    /// probability. `NaN` is treated as `0.0` so invalid programmatic input
    /// preserves the pre-min-p distribution; all other values clamp to
    /// `[0.0, 1.0]`.
    pub fn sample_min_p_top_p(&mut self, min_p: f32, top_p: f32, r: f32) -> u32 {
        let mut probs = Vec::new();
        self.sample_min_p_top_p_with_scratch(min_p, top_p, r, &mut probs)
    }

    /// Allocation-free top-p path retained for internal regression tests.
    #[cfg(test)]
    fn sample_top_p_with_scratch(&mut self, top_p: f32, r: f32, probs: &mut Vec<f32>) -> u32 {
        self.sample_min_p_top_p_with_scratch(0.0, top_p, r, probs)
    }

    /// Same as [`CandidateSet::sample_min_p_top_p`] but reuses the caller's
    /// probability buffer to avoid allocation.
    fn sample_min_p_top_p_with_scratch(
        &mut self,
        min_p: f32,
        top_p: f32,
        r: f32,
        probs: &mut Vec<f32>,
    ) -> u32 {
        if self.candidates.is_empty() {
            return 0;
        }

        let min_p = if min_p.is_nan() {
            0.0
        } else {
            min_p.clamp(0.0, 1.0)
        };

        // #328: normalise top_p into a defined regime before the `top_p < 1.0`
        // gate below. The raw argument is otherwise interpreted by undefined
        // behaviour: NaN makes `top_p < 1.0` false and silently disables nucleus
        // truncation (full distribution sampled); a negative top_p passes the
        // gate and the `cumsum >= top_p` cutoff matches on the first (largest)
        // token, collapsing to greedy. Mapping NaN/+Inf/>1 → 1.0 (no truncation)
        // and clamping the rest into [0, 1] preserves today's behaviour exactly
        // while making it explicit and refactor-safe. `f32::clamp` returns NaN
        // for a NaN input, so NaN is handled before the clamp.
        let top_p = if top_p.is_nan() {
            1.0
        } else {
            top_p.clamp(0.0, 1.0)
        };

        // With min-p on, the tail is rejected before the sort, so the sort
        // runs over the survivors only. With `top_k == 0` the full sort is
        // the dominant cost of this call, not the `exp` of the tail. See
        // `min_p_prefilter_and_sort` for why the result is identical to
        // sorting everything first.
        let max_logit = if min_p > 0.0 {
            match self.min_p_prefilter_and_sort(min_p) {
                Ok(max_logit) => max_logit,
                Err(token_id) => return token_id,
            }
        } else {
            // Sort descending for deterministic top-p traversal. Untouched:
            // `min_p == 0.0` must stay byte-identical to the pre-#1394
            // algorithm, full sort included.
            self.candidates.sort_by(candidate_order);

            // Softmax — reuse the provided scratch buffer.
            let max_logit = self.candidates[0].logit;
            // A non-finite max (+INF, or all-NaN/-INF) makes (logit - max).exp() produce
            // NaN probabilities, which the weighted-sample loop never selects — it would
            // fall through to the worst candidate. The argmax (candidates[0] after the
            // descending sort) is the correct answer for an infinite-logit token.
            if !max_logit.is_finite() {
                return self.candidates[0].token_id;
            }
            max_logit
        };

        probs.clear();
        probs.extend(self.candidates.iter().map(|c| (c.logit - max_logit).exp()));

        if min_p > 0.0 {
            // Dead in practice after the NaN check above (no remaining candidate
            // can carry a NaN logit, and none but a possible index 0 can be +inf,
            // which `max_logit.is_finite()` already excluded) — kept as the same
            // defense-in-depth fail-closed check the pre-existing algorithm ran,
            // now scoped to the (small) surviving prefix instead of the full set.
            if probs.iter().any(|weight| !weight.is_finite()) {
                return self.candidates[0].token_id;
            }
            let cutoff = probs
                .iter()
                .position(|&weight| weight < min_p)
                .unwrap_or(probs.len())
                .max(1);
            probs.truncate(cutoff);
            self.candidates.truncate(cutoff);
        }

        let sum: f32 = probs.iter().sum();
        // A finite max does not guarantee a finite sum: a NaN logit in a
        // non-max position (NaN sorts last, so it does not become `max_logit`)
        // makes its `exp()` NaN, poisoning `sum` and every normalised prob, and
        // the weighted-sample loop below — `r < cumsum` is always false for a
        // NaN cumsum — would fall through to the worst candidate. Fall back to
        // the argmax (candidates[0] after the descending sort) instead, matching
        // the non-finite-max guard above.
        if !sum.is_finite() || sum <= 0.0 {
            return self.candidates[0].token_id;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top-p nucleus truncation.
        if top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let mut cutoff = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
            self.candidates.truncate(cutoff);
            // Re-normalise.
            let sum: f32 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        // Weighted sample.
        let mut cumsum = 0.0f32;
        for (c, &p) in self.candidates.iter().zip(probs.iter()) {
            cumsum += p;
            if r < cumsum {
                return c.token_id;
            }
        }
        self.candidates.last().map(|c| c.token_id).unwrap_or(0)
    }

    /// #1394 item 2: reject the min-p tail in logit space *before* sorting,
    /// then sort only the survivors. Only called from
    /// [`CandidateSet::sample_min_p_top_p_with_scratch`] when `min_p > 0.0`
    /// (both call sites enforce this; the caller's `min_p == 0.0` path never
    /// reaches here at all).
    ///
    /// A single linear pass over the still-unsorted candidates finds the
    /// `candidate_order`-minimum -- the element `sort_by(candidate_order)`
    /// would place at index 0 -- and whether any NaN logit is present, in
    /// one scan. `candidate_order` is a total order over distinct token ids
    /// (non-NaN ties break on ascending token_id; NaN sorts after every
    /// non-NaN, NaN ties also break on ascending token_id), so the running
    /// minimum under that comparator, after visiting every element, equals
    /// `self.candidates[0]` after a full `sort_by(candidate_order)` -- without
    /// ever materializing that sort.
    ///
    /// Returns `Err(token_id)` for the two fail-closed fallbacks the old,
    /// unconditional post-sort checks used to cover; the caller must return
    /// that token id directly, exactly as the pre-#1394 algorithm did:
    /// - Any NaN logit anywhere in the set. `candidate_order` sorting NaN
    ///   last means the running minimum is still the true numeric argmax
    ///   whenever at least one non-NaN candidate exists (a NaN elsewhere can
    ///   never displace it), and is the smallest-token-id NaN candidate --
    ///   matching what a full sort would have put first -- when every
    ///   candidate is NaN. Both cases are answered correctly by returning
    ///   the running minimum's token id unconditionally.
    /// - A non-finite (but non-NaN, i.e. +-inf) numeric maximum, once no NaN
    ///   is present.
    ///
    /// Returns `Ok(max_logit)` once `self.candidates` has been narrowed to
    /// the min-p survivors and sorted with the identical `candidate_order`
    /// comparator, ready for the unchanged `exp` / p-space-cutoff / top-p /
    /// sample tail shared with the `min_p == 0.0` path.
    fn min_p_prefilter_and_sort(&mut self, min_p: f32) -> Result<f32, u32> {
        let mut order_min = self.candidates[0];
        let mut has_nan = order_min.logit.is_nan();
        for c in &self.candidates[1..] {
            has_nan |= c.logit.is_nan();
            if candidate_order(c, &order_min) == std::cmp::Ordering::Less {
                order_min = *c;
            }
        }

        if has_nan {
            return Err(order_min.token_id);
        }
        if !order_min.logit.is_finite() {
            return Err(order_min.token_id);
        }
        let max_logit = order_min.logit;

        // `ln` and `exp` are not exact float32 inverses, so `logit_threshold`
        // can differ from the true real-valued cutoff by a few ULPs. `margin`
        // widens the threshold *downward* (toward keeping more candidates) by
        // well over an order of magnitude more than that rounding error,
        // scaled by the magnitude of the values that produced it, so the
        // survivors retained below are guaranteed to be a superset of
        // whatever the exact p-space test further down the caller would
        // keep, never a subset — every survivor still runs through that
        // untouched, byte-identical `exp` + `weight < min_p` comparison, so
        // the boundary inclusion decision is unchanged. The logit threshold
        // only decides how many definitely-rejected candidates can skip
        // `exp()` (and now the sort) entirely. `min_p` is clamped by the
        // caller to `(0.0, 1.0]` here, so `log_min_p <= 0.0`, and `margin` is
        // never negative, so `margin >= log_min_p` always holds -- which
        // makes `max_logit >= safe_threshold` algebraically guaranteed:
        // `order_min` always survives its own threshold, so `retain` below
        // can never empty the candidate set.
        let log_min_p = min_p.ln();
        let logit_threshold = max_logit + log_min_p;
        let margin = 256.0 * f32::EPSILON * (max_logit.abs() + log_min_p.abs()).max(1.0);
        let safe_threshold = logit_threshold - margin;
        self.candidates.retain(|c| c.logit >= safe_threshold);

        // Every discarded candidate has `logit < safe_threshold <= max_logit`
        // and every survivor has `logit >= safe_threshold`, and neither side
        // carries a NaN (the `has_nan` check above already returned
        // otherwise), so every survivor's logit is strictly greater than
        // every discarded candidate's logit -- `candidate_order` therefore
        // ranks every survivor strictly before every discarded candidate,
        // regardless of how the survivors sort among themselves. Sorting
        // only the survivors reproduces exactly the prefix (of the same
        // length as the survivor count) that sorting the full set would
        // have produced.
        self.candidates.sort_by(candidate_order);
        Ok(max_logit)
    }
}

/// Canonical xorshift64 state advance shared by every CPU sampling path.
/// Shifts 13/7/17 give a full period over nonzero state.
pub(crate) fn xorshift64_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Canonical uniform f32 in [0, 1) from a 64-bit random word. Uses the top 24
/// bits (the f32 mantissa width) so the result is a uniform over representable
/// f32 values in [0, 1) and is provably strictly < 1.0 — unlike a 53-bit
/// integer cast to f32, which rounds up to exactly 1.0 at the top of the range.
pub(crate) fn uniform_f32_from_u64(x: u64) -> f32 {
    (x >> 40) as f32 / (1u64 << 24) as f32
}

/// Xorshift64 PRNG for sampling (no external dependency).
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        xorshift64_next(&mut self.state)
    }

    /// Returns a uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        uniform_f32_from_u64(self.next_u64())
    }
}

/// **Unstable**: stateful sampler; implementation and state fields may change.
pub struct Sampler {
    config: SamplingConfig,
    /// Min-p: keep tokens with probability at least `min_p * max_probability`.
    /// 0.0 or NaN = disabled; all other values clamp to `[0.0, 1.0]`. Seeded
    /// from `config.min_p` at construction; [`with_min_p`](Self::with_min_p)
    /// overrides it afterward. Kept as its own field (rather than reading
    /// `self.config.min_p` at every `sample` call) so `with_min_p` can diverge
    /// from the config without mutating it.
    min_p: f32,
    /// Top-n-sigma: mask logits below `max_logit - top_n_sigma *
    /// population_stddev` before candidate selection. 0.0, negative, or
    /// non-finite = disabled. Unlike `min_p`, `SamplingConfig`/`GenerateConfig`
    /// have no `top_n_sigma` field yet, so this stays carried out-of-band; set
    /// via [`with_top_n_sigma`](Self::with_top_n_sigma).
    top_n_sigma: f32,
    rng: Rng,
    /// Token IDs seen since the last `reset`: prompt tokens (from `seed_history`)
    /// followed by all generated tokens.  Full history is retained for repetition penalty.
    recent_tokens: Vec<u32>,
    /// Reused dedup set for penalty application; cleared at the start of each use.
    penalty_seen: std::collections::HashSet<u32>,
    /// Reused candidate buffer; avoids 1.9 MB alloc per call at vocab=248,320.
    candidate_scratch: Vec<Candidate>,
    /// Reused probability buffer for min-p/top-p softmax.
    prob_scratch: Vec<f32>,
    /// Reused f32 scratch for adjusted logits; avoids 993 KB alloc per call.
    logit_scratch: Vec<f32>,
}

impl Sampler {
    /// **Unstable**: construct sampler with a non-deterministic seed.
    ///
    /// Seeded from the system clock so independent samplers (e.g. concurrent
    /// requests) do not produce identical token streams. Use
    /// [`with_seed`](Self::with_seed) for reproducible sampling.
    pub fn new(config: SamplingConfig) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x853c_49e6_748f_ea9b);
        let min_p = config.min_p;
        Self {
            config,
            min_p,
            top_n_sigma: 0.0,
            rng: Rng::new(seed),
            recent_tokens: Vec::new(),
            penalty_seen: std::collections::HashSet::new(),
            candidate_scratch: Vec::new(),
            prob_scratch: Vec::new(),
            logit_scratch: Vec::new(),
        }
    }

    /// **Unstable**: set PRNG seed for reproducible sampling.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// **Unstable**: set min-p (relative-probability floor). 0.0 = disabled.
    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    /// **Unstable**: set top-n-sigma (stddev-relative logit floor). 0.0 = disabled.
    pub fn with_top_n_sigma(mut self, top_n_sigma: f32) -> Self {
        self.top_n_sigma = top_n_sigma;
        self
    }

    /// **Unstable**: seed the repetition-penalty history with prompt tokens.
    ///
    /// Call once after construction and before the first `sample` call so
    /// prompt tokens are penalized from the very first generated token,
    /// matching the Qwen3.5 full-history contract.
    pub fn seed_history(&mut self, prompt_ids: &[u32]) {
        self.recent_tokens.extend_from_slice(prompt_ids);
    }

    /// **Unstable**: sample a token ID; sampling strategy details may change.
    ///
    /// `logits` is the raw output from the model's final linear layer,
    /// shape `[vocab_size]`.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        // Greedy fast path: skip the full-vocab clone when argmax is not penalized.
        // A repetition penalty >= 1.0 only lowers penalized tokens, so argmax(raw) ==
        // argmax(penalized) whenever the raw argmax token is not in recent_tokens. In the
        // common case (recent_tokens empty or raw argmax not recently generated) this
        // avoids the 993 KB extend_from_slice entirely. A penalty in (0, 1) BOOSTS recent
        // tokens, so it must fall through to the re-scan even when raw_best is not recent.
        //
        // A degenerate temperature (non-finite, <= 0, or finite-but-tiny so that its
        // reciprocal overflows) carries no valid scaling and is routed here to the same
        // deterministic argmax as the t -> 0+ limit. See `temperature_degenerate`.
        if temperature_degenerate(self.config.temperature) || self.config.top_k == 1 {
            let raw_best = argmax_f32(logits);
            // Conservative sufficient gate (not maximal): take the shortcut when the
            // penalty cannot promote a non-argmax token above raw_best. That holds when
            // the penalty is inactive (a no-op in `penalized_logit`), or it only demotes
            // (> 1.0) AND raw_best is not itself penalized. A penalty in (0, 1) divides a
            // positive logit by < 1, boosting recent tokens, so a non-recent raw_best can
            // be overtaken by a boosted recent token — that case must re-scan.
            //
            // One always-safe case is intentionally left to the re-scan for simplicity:
            // (0, 1) penalty where raw_best is ITSELF recent. `penalized_logit` is monotone,
            // so boosting raw_best keeps it >= every non-recent logit and preserves order
            // against other recent logits, leaving it the argmax. Shortcutting it would
            // need an extra `contains` probe on the hot path to save a rare clone, so we
            // fall through instead — correct, just not maximal.
            let penalty = self.config.repetition_penalty;
            let penalty_inactive = penalty == 1.0 || !penalty.is_finite() || penalty <= 0.0;
            if penalty_inactive || (penalty > 1.0 && !self.recent_tokens.contains(&raw_best)) {
                self.push_token(raw_best);
                return raw_best;
            }
            // raw argmax would change once the penalty is applied — clone + re-scan.
            self.logit_scratch.clear();
            self.logit_scratch.extend_from_slice(logits);
            // Penalize each unique id exactly once (HF gather-once semantics).
            self.apply_penalty_to_logit_scratch(penalty);
            let token = argmax_f32(&self.logit_scratch);
            self.push_token(token);
            return token;
        }

        // Non-greedy path: clone logits for in-place penalty + fused top-k.
        self.logit_scratch.clear();
        self.logit_scratch.extend_from_slice(logits);

        if self.config.repetition_penalty != 1.0 {
            let penalty = self.config.repetition_penalty;
            // Penalize each unique id exactly once (HF gather-once semantics).
            self.apply_penalty_to_logit_scratch(penalty);
        }

        // Top-n-sigma masks logits below `max - n * sigma` ahead of the fused
        // top-k, so a masked token can never enter the candidate set. The
        // greedy fast path above returns before this point: masking strictly
        // below the max cannot change an argmax, so the filter would be a
        // no-op there at the cost of two full-vocabulary passes.
        let top_n_sigma_enabled = top_n_sigma_active(self.top_n_sigma);
        if top_n_sigma_enabled {
            apply_top_n_sigma(&mut self.logit_scratch, self.top_n_sigma);
        }

        let (has_nan, max_logit) = scan_nan_or_nonfinite_max(&self.logit_scratch);
        if has_nan || !max_logit.is_finite() {
            let token = argmax_f32(&self.logit_scratch);
            self.push_token(token);
            return token;
        }

        // Centering is applied only when `inv_temp > 1.0` (see `select_top_k`'s
        // doc comment for the full reasoning), so no logit overflows to +inf in
        // that regime, without introducing a new downward overflow when it isn't.
        let inv_temp = if self.config.temperature != 1.0 {
            1.0 / self.config.temperature
        } else {
            1.0
        };
        let center = if inv_temp > 1.0 { max_logit } else { 0.0 };

        // Streaming min-heap top-k with fused temperature scaling.
        // ~95% of vocab elements are skipped by the NEON threshold gate.
        select_top_k(
            &self.logit_scratch,
            self.config.top_k,
            inv_temp,
            center,
            &mut self.candidate_scratch,
        );
        if top_n_sigma_enabled {
            // `select_top_k` can seed its heap with masked entries when fewer
            // than `top_k` logits survive the filter; drop them so the softmax
            // below only sees survivors.
            self.candidate_scratch
                .retain(|candidate| candidate.logit != f32::NEG_INFINITY);
        }
        let mut cs = CandidateSet {
            candidates: std::mem::take(&mut self.candidate_scratch),
        };
        let r = self.rng.next_f32();
        let token = cs.sample_min_p_top_p_with_scratch(
            self.min_p,
            self.config.top_p,
            r,
            &mut self.prob_scratch,
        );
        self.candidate_scratch = cs.candidates; // restore scratch capacity
        self.push_token(token);
        token
    }

    fn push_token(&mut self, token: u32) {
        self.recent_tokens.push(token);
    }

    /// Apply `penalty` in-place to `logit_scratch` for every unique token id in
    /// `recent_tokens`, penalizing each id exactly once (HF gather-once semantics).
    /// `penalty_seen` is cleared first so capacity is retained across calls.
    fn apply_penalty_to_logit_scratch(&mut self, penalty: f32) {
        self.penalty_seen.clear();
        for &tok in &self.recent_tokens {
            let idx = tok as usize;
            if idx < self.logit_scratch.len() && self.penalty_seen.insert(tok) {
                self.logit_scratch[idx] = penalized_logit(self.logit_scratch[idx], penalty);
            }
        }
    }

    /// **Unstable**: reset sampler state for a new generation sequence.
    pub fn reset(&mut self) {
        self.recent_tokens.clear();
    }
}

thread_local! {
    /// Reused scratch for [`sample_full_logits`], the shared engine behind the
    /// stateless `(logits, cfg, previous_ids, rng_state)` call sites that have
    /// no owning struct to hold per-generation buffers: the Metal CPU fallback
    /// (`forward/metal_qwen35.rs`'s private `sample_token`) and the Qwen CPU
    /// decode loops (`model/qwen35/sampling.rs::sample_token`, shared by the
    /// f16/q8/NEON forward paths). Kept thread-local instead of
    /// per-call so vocab-sized buffers are allocated once per thread and
    /// reused across every decode step, not just within one generation.
    static FULL_LOGIT_SCRATCH: std::cell::RefCell<FullLogitScratch> =
        std::cell::RefCell::new(FullLogitScratch::new());
}

struct FullLogitScratch {
    logit_scratch: Vec<f32>,
    candidate_scratch: Vec<Candidate>,
    prob_scratch: Vec<f32>,
    penalty_seen: std::collections::HashSet<u32>,
}

impl FullLogitScratch {
    fn new() -> Self {
        Self {
            logit_scratch: Vec::new(),
            candidate_scratch: Vec::new(),
            prob_scratch: Vec::new(),
            penalty_seen: std::collections::HashSet::new(),
        }
    }
}

/// Shared full-logit sampling engine for callers that receive the full token
/// history and RNG state as plain arguments each step (rather than owning a
/// [`Sampler`]). Implements the same pipeline as `Sampler::sample`'s non-greedy
/// path: apply repetition penalty once per unique previously-seen id, check
/// the degenerate-temperature greedy short-circuit, then a single fused
/// temperature-scale + partial top-k select feeding a min-p/top-p softmax draw.
///
/// This replaces, at each of its two call sites, a per-call
/// `CandidateSet::from_full_logits` (materializes every vocabulary entry as a
/// `Candidate` before any filtering) plus a per-candidate repetition-penalty
/// scan (`previous_ids.contains(&c.token_id)`, O(vocab * history length)) with:
/// - repetition penalty applied only at the indices named by the unique ids in
///   `previous_ids` (a context-id set, O(unique history length));
/// - `select_top_k`'s streaming min-heap partial selection, so the top-p
///   softmax below only ever runs over the surviving `k` candidates, never the
///   full vocabulary;
/// - thread-local `logit_scratch` / `candidate_scratch` / `prob_scratch` /
///   `penalty_seen` buffers reused across decode steps instead of a fresh
///   vocab-sized allocation per token.
pub(crate) fn sample_full_logits(
    logits: &[f32],
    cfg: &GenerateConfig,
    previous_ids: &[u32],
    rng_state: &mut u64,
    min_p: f32,
    top_n_sigma: f32,
) -> u32 {
    FULL_LOGIT_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let FullLogitScratch {
            logit_scratch,
            candidate_scratch,
            prob_scratch,
            penalty_seen,
        } = &mut *scratch;

        logit_scratch.clear();
        logit_scratch.extend_from_slice(logits);

        if cfg.repetition_penalty != 1.0 {
            // Penalize each unique previously-seen id exactly once (HF
            // gather-once semantics), matching `Sampler::apply_penalty_to_logit_scratch`.
            penalty_seen.clear();
            for &tok in previous_ids {
                let idx = tok as usize;
                if idx < logit_scratch.len() && penalty_seen.insert(tok) {
                    logit_scratch[idx] =
                        penalized_logit(logit_scratch[idx], cfg.repetition_penalty);
                }
            }
        }

        // A degenerate temperature (non-finite, <= 0, or finite-but-tiny so its
        // reciprocal overflows) carries no valid scaling; the t -> 0+ limit is
        // greedy argmax over the already-penalized logits.
        if temperature_degenerate(cfg.temperature) {
            return argmax_f32(logit_scratch);
        }

        // Top-n-sigma runs after the degenerate-temperature return above and
        // is skipped for `top_k == 1`: masking strictly below the max can
        // change neither an argmax nor a k=1 selection, so in both
        // argmax-shaped cases the filter would spend two full-vocabulary
        // passes to produce the token those paths already produce without it.
        let top_n_sigma_enabled = top_n_sigma_active(top_n_sigma) && cfg.top_k != 1;
        if top_n_sigma_enabled {
            apply_top_n_sigma(logit_scratch, top_n_sigma);
        }

        let inv_temp = if cfg.temperature != 1.0 {
            1.0 / cfg.temperature
        } else {
            1.0
        };

        // Fail-closed guard: `select_top_k`'s scalar/NEON seed phases rewrite a
        // NaN scaled logit to NEG_INFINITY so the min-heap root is never NaN
        // (a NaN root would reject every finite candidate in the NEON prefilter).
        // That sanitization is correct for the heap's internal ordering, but it
        // erases the NaN *before* the softmax degeneracy guard in
        // `CandidateSet::sample_min_p_top_p_with_scratch` ever sees it, silently
        // turning a "poisoned distribution" case into "normal sampling over a
        // masked candidate" — exactly the regression this scan closes.
        //
        // Mirror the old full-logit contract (`build_softmax_probs` returning
        // `None` on a non-finite max or a NaN-poisoned sum, both of which fall
        // back to argmax) with a single pass over the pre-scaled logits: any
        // NaN anywhere forces degeneracy (a NaN can never become the max, so it
        // would otherwise poison the softmax sum from a non-max position,
        // #322-class); a non-finite max (every logit `-inf`, or a `+inf` logit)
        // also forces degeneracy, matching `sample_min_p_top_p_with_scratch`'s
        // non-finite-max short-circuit. Scaling alone does NOT preserve
        // finiteness: `logit * inv_temp` overflows to +inf for any finite logit
        // above `temperature * f32::MAX`. The fused top-k therefore subtracts
        // the finite `max_logit` found here before multiplying by the positive
        // finite `inv_temp` (guaranteed by the `temperature_degenerate` check
        // above) -- but only when `inv_temp > 1.0` (see `select_top_k`'s doc
        // comment for the full reasoning): that is the only regime where scaling
        // first can overflow upward, and centering when `inv_temp <= 1.0` would
        // trade that away for a downward overflow instead. That preserves
        // NaN-ness and, when centering applies, maps the maximum to exactly 0.0,
        // so the scaled maximum stays finite and scanning before the fused
        // top-k scale is equivalent to scanning after it.
        let (has_nan, max_logit) = scan_nan_or_nonfinite_max(logit_scratch);
        if has_nan || !max_logit.is_finite() {
            return argmax_f32(logit_scratch);
        }
        let center = if inv_temp > 1.0 { max_logit } else { 0.0 };

        // Streaming min-heap top-k with fused temperature scaling — the softmax
        // draw below runs only over these k survivors, never the full vocabulary.
        select_top_k(
            logit_scratch,
            cfg.top_k,
            inv_temp,
            center,
            candidate_scratch,
        );
        if top_n_sigma_enabled {
            // Same masked-survivor cleanup as `Sampler::sample`.
            candidate_scratch.retain(|candidate| candidate.logit != f32::NEG_INFINITY);
        }

        let mut cs = CandidateSet {
            candidates: std::mem::take(candidate_scratch),
        };
        let r = uniform_f32_from_u64(xorshift64_next(rng_state));
        let token = cs.sample_min_p_top_p_with_scratch(min_p, cfg.top_p, r, prob_scratch);
        *candidate_scratch = cs.candidates; // restore scratch capacity
        token
    })
}

/// A generous upper bound on the magnitude of a transformer lm_head logit. Real
/// logits are O(10) (rarely past ~50); this 1e4 ceiling carries ~100-1000x of
/// headroom. It defines the boundary below which a temperature is treated as
/// greedy: if scaling a logit of this magnitude by `1.0 / t` would overflow f32,
/// the temperature cannot produce a valid ordering for any realistic logit.
const MAX_PLAUSIBLE_ABS_LOGIT: f32 = 1.0e4;

/// True when `temperature` cannot produce a valid finite logit scaling, so sampling
/// must fall back to greedy argmax (the `t -> 0+` limit, `softmax(logits / t) ->
/// argmax(logits)`). Covers four cases:
/// - non-finite temperature (NaN, ±inf): `1.0 / NaN` is NaN and `1.0 / inf` is 0, both
///   of which corrupt the scaled distribution;
/// - `t <= 0`: no valid scaling;
/// - finite but tiny positive `t` where `1.0 / t` itself overflows f32 to +inf
///   (`t < ~2.94e-39`, since `f32::MAX ≈ 3.4e38`): the scaling multiplies every logit
///   by inf and loses their ordering;
/// - finite `t` where `1.0 / t` stays finite but `logit * (1.0 / t)` overflows for a
///   plausible logit (`t` roughly below `MAX_PLAUSIBLE_ABS_LOGIT / f32::MAX ≈ 2.94e-35`).
///   The scaled logits saturate to +inf, the softmax's non-finite-max guard then
///   returns an arbitrary candidate instead of the true argmax. Both overflow bands
///   are well below any valid sampling temperature, and the `t -> 0+` limit is argmax
///   for all of them, so routing them to greedy is exact, not a behavior change.
///
/// Shared by every sampling entry point (CPU `Sampler::sample`, the CPU
/// `model/qwen35::sample_token`, and the Metal `sample_from_candidates` /
/// `sample_token` paths) so the degeneracy check stays identical across backends.
#[inline]
pub(crate) fn temperature_degenerate(temperature: f32) -> bool {
    if !temperature.is_finite() || temperature <= 0.0 {
        return true;
    }
    let inv_temp = 1.0 / temperature;
    !inv_temp.is_finite() || inv_temp > f32::MAX / MAX_PLAUSIBLE_ABS_LOGIT
}

/// Apply repetition penalty to one logit. A penalty of 1.0, or any non-finite
/// or non-positive value, is treated as "no penalty" so an invalid config can
/// never flip a logit's sign or produce an infinity.
#[inline(always)]
pub(crate) fn penalized_logit(logit: f32, penalty: f32) -> f32 {
    if penalty == 1.0 || !penalty.is_finite() || penalty <= 0.0 {
        return logit;
    }
    if logit > 0.0 {
        logit / penalty
    } else {
        logit * penalty
    }
}

/// Whether a top-n-sigma value enables filtering.
#[inline]
pub(crate) fn top_n_sigma_active(top_n_sigma: f32) -> bool {
    top_n_sigma.is_finite() && top_n_sigma > 0.0
}

/// Mask logits below `max_logit - top_n_sigma * population_stddev` to
/// `NEG_INFINITY`, in place, using a single Welford pass for the statistics.
///
/// The statistics exclude `NEG_INFINITY`, which represents an upstream mask.
/// Any other non-finite value leaves the slice untouched so the sampler's
/// existing poisoned-distribution guard can fail closed to argmax.
fn apply_top_n_sigma(logits: &mut [f32], top_n_sigma: f32) {
    if !top_n_sigma_active(top_n_sigma) || logits.len() <= 1 {
        return;
    }

    let mut count = 0_u64;
    let mut mean = 0.0_f64;
    let mut squared_deviation_sum = 0.0_f64;
    let mut max_logit = f32::NEG_INFINITY;

    for &logit in logits.iter() {
        if logit == f32::NEG_INFINITY {
            continue;
        }
        if !logit.is_finite() {
            return;
        }

        count += 1;
        let value = f64::from(logit);
        let delta = value - mean;
        mean += delta / count as f64;
        squared_deviation_sum += delta * (value - mean);
        max_logit = max_logit.max(logit);
    }

    if count == 0 {
        return;
    }

    let stddev = (squared_deviation_sum / count as f64).sqrt();
    let threshold = f64::from(max_logit) - f64::from(top_n_sigma) * stddev;
    for logit in logits {
        if f64::from(*logit) < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// A finite stand-in for degenerate (`-inf`/`NaN`) log-probabilities. Every
/// consumer of [`TokenLogprob`] is eventually JSON-encoded (the
/// OpenAI-compatible HTTP response), and `serde_json` cannot serialize
/// non-finite floats, so any degenerate case is clamped to this sentinel
/// rather than propagating a value that would fail serialization. Chosen far
/// below any real log-probability: natural-log probabilities are always
/// `<= 0`, and for a vocabulary of a few hundred thousand tokens are bounded
/// below by roughly `-13`.
const LOGPROB_NEG_SENTINEL: f32 = -1.0e9;

/// Computes the reporting log-probability of `token_id` plus its top `top_n`
/// alternatives, under a temperature-scaled softmax over one step's raw
/// `logits`.
///
/// Deliberately independent of [`Sampler`]/`sample_token`'s own selection
/// pipeline: only temperature scaling is applied (no repetition penalty, no
/// top-k/top-p truncation), matching what OpenAI's API reports -- the
/// model's own probability estimate, not a post-hoc-filtered one. When
/// `temperature` is degenerate (see [`temperature_degenerate`]), an unscaled
/// (temperature = 1.0) softmax is reported instead of the sampler's `t ->
/// 0+` one-hot fallback: argmax is invariant to positive temperature
/// scaling, so the *selected* token is unaffected, and reporting log(1.0) =
/// 0.0 for the winner with `-inf` for everything else would erase the
/// model's actual confidence rather than describe it.
///
/// Falls back to [`LOGPROB_NEG_SENTINEL`] wherever a value would otherwise be
/// non-finite (e.g. every logit non-finite) or `token_id` is outside the
/// vocabulary covered by `logits`.
///
/// `pub(crate)` (PR #787): this is
/// the pure-computation half of what `record_logprob` used to expose as one
/// combined `pub(crate)` function. `DecodePolicy::record_logprob`
/// (`model::qwen35::generation`) is the only place that pushes the result
/// into a `token_logprobs: &mut Vec<TokenLogprob>` -- a decode call site can
/// read a logprob distribution for its own purposes, but it can no longer
/// independently append to a `token_logprobs` accumulator without going
/// through `DecodePolicy`, since no freestanding "record" function exists to
/// call anymore.
pub(crate) fn compute_step_logprobs(
    logits: &[f32],
    token_id: u32,
    temperature: f32,
    top_n: usize,
) -> (f32, Vec<TopLogprob>) {
    let vocab_size = logits.len();
    let scale = if temperature_degenerate(temperature) {
        1.0
    } else {
        1.0 / temperature
    };

    // Center only when `scale > 1.0` (see `CandidateSet::apply_temperature`):
    // scaling a huge finite logit first would overflow to +inf and send the
    // whole row to the sentinel. Rows with no finite logit keep a zero center,
    // so their non-finite max still short-circuits below.
    let center = if scale > 1.0 {
        let raw = logits
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        if raw.is_finite() { raw } else { 0.0 }
    } else {
        0.0
    };

    let mut max_scaled = f32::NEG_INFINITY;
    for &v in logits {
        let scaled = (v - center) * scale;
        if scaled > max_scaled {
            max_scaled = scaled;
        }
    }
    if !max_scaled.is_finite() {
        let top = if top_n > 0 {
            vec![TopLogprob {
                token_id,
                logprob: LOGPROB_NEG_SENTINEL,
            }]
        } else {
            Vec::new()
        };
        return (LOGPROB_NEG_SENTINEL, top);
    }

    // For a row without NaN, sum >= 1.0: the max term alone contributes
    // exp(0) = 1.0, so log_sum is finite and non-negative (bounded above by
    // ln(vocab_size)) and cannot overflow or introduce a fresh non-finite
    // value here. A NaN entry makes sum (and log_sum) NaN; the per-token
    // finite check in `logprob_of` then maps every result to the sentinel.
    let mut sum = 0.0f32;
    for &v in logits {
        sum += ((v - center) * scale - max_scaled).exp();
    }
    let log_sum = sum.ln();

    let logprob_of = |idx: usize| -> f32 {
        let lp = ((logits[idx] - center) * scale - max_scaled) - log_sum;
        if lp.is_finite() {
            lp
        } else {
            LOGPROB_NEG_SENTINEL
        }
    };

    let token_logprob = if (token_id as usize) < vocab_size {
        logprob_of(token_id as usize)
    } else {
        LOGPROB_NEG_SENTINEL
    };

    let k = top_n.min(vocab_size);
    let top = if k == 0 {
        Vec::new()
    } else {
        // Reuse the same descending-logit, NaN-last, lowest-token-id-wins
        // total order as the Metal-parity top-k path (`candidate_order`)
        // rather than inventing a second comparator.
        //
        // The ordering key is the raw logit when `scale > 1.0`: scaling by a
        // positive factor preserves order, and the raw value cannot overflow.
        // The centered value `logits[i] - center` can: on a row spanning more
        // than `f32::MAX`, distinct finite tail logits all overflow to `-inf`
        // and would then be ordered by token id instead of by logit. When
        // `scale <= 1.0` the center is zero and the key is the scaled logit,
        // exactly as before.
        let mut candidates: Vec<Candidate> = (0..vocab_size)
            .map(|i| Candidate {
                token_id: i as u32,
                logit: if scale > 1.0 {
                    logits[i]
                } else {
                    (logits[i] - center) * scale
                },
            })
            .collect();
        candidates.select_nth_unstable_by(k - 1, candidate_order);
        candidates.truncate(k);
        candidates.sort_unstable_by(candidate_order);
        candidates
            .into_iter()
            .map(|c| TopLogprob {
                token_id: c.token_id,
                logprob: logprob_of(c.token_id as usize),
            })
            .collect()
    };

    (token_logprob, top)
}

/// Fail-closed pre-scan for full-logit sampling paths: detects whether `logits`
/// contains any NaN and computes the NaN-ignoring ("numeric") max in a
/// single pass, so the caller can fall back to `argmax_f32` before
/// `select_top_k`'s scalar/NEON seed phases sanitize a NaN to `NEG_INFINITY`
/// and hide it from the softmax degeneracy guard (PR #651).
/// Runtime-dispatch NEON on aarch64 (same 4-wide-plus-scalar-tail structure
/// as `argmax_f32_neon`) with a scalar fallback.
fn scan_nan_or_nonfinite_max(logits: &[f32]) -> (bool, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON is runtime-detected above; only loads aligned 4-f32 chunks.
            return unsafe { scan_nan_or_nonfinite_max_neon(logits) };
        }
    }
    scan_nan_or_nonfinite_max_scalar(logits)
}

fn scan_nan_or_nonfinite_max_scalar(logits: &[f32]) -> (bool, f32) {
    let mut max_logit = f32::NEG_INFINITY;
    let mut has_nan = false;
    for &v in logits {
        has_nan |= v.is_nan();
        // `f32::max` is IEEE-754 `maxNum`: NaN never wins, matching the old
        // full-logit reference's `.fold(f32::NEG_INFINITY, f32::max)`.
        max_logit = max_logit.max(v);
    }
    (has_nan, max_logit)
}

/// NEON 4-wide NaN-detect + numeric-max scan.  `vmaxnmq_f32` accumulates the
/// NaN-ignoring max per lane (IEEE-754 `maxNum`, matching `f32::max`).
/// `vceqq_f32(v, v)` is all-ones where `v` is non-NaN (self-equal) and
/// all-zero where `v` is NaN (a NaN self-comparison is always false), so
/// OR-accumulating its bitwise complement across every chunk flags any NaN
/// in the array.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scan_nan_or_nonfinite_max_neon(logits: &[f32]) -> (bool, f32) {
    use std::arch::aarch64::*;

    let len = logits.len();
    let mut i = 0usize;
    let mut max_acc = vdupq_n_f32(f32::NEG_INFINITY);
    let mut nan_acc = vdupq_n_u32(0);

    while i + 4 <= len {
        // SAFETY: loop condition guarantees four valid f32 values are in-bounds.
        let v = vld1q_f32(logits.as_ptr().add(i));
        max_acc = vmaxnmq_f32(max_acc, v);
        let is_non_nan_lane = vceqq_f32(v, v);
        nan_acc = vorrq_u32(nan_acc, vmvnq_u32(is_non_nan_lane));
        i += 4;
    }

    // Horizontal reduction: numeric (NaN-ignoring) max across the 4 lanes,
    // and OR-reduce the NaN-flag lanes to a single bool.
    let mut max_logit = vmaxnmvq_f32(max_acc);
    let mut has_nan = vmaxvq_u32(nan_acc) != 0;

    // Scalar tail for the final 0-3 elements.
    while i < len {
        let v = *logits.get_unchecked(i);
        has_nan |= v.is_nan();
        max_logit = max_logit.max(v);
        i += 1;
    }
    (has_nan, max_logit)
}

fn argmax_f32(logits: &[f32]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON is runtime-detected above; only loads aligned 4-f32 chunks.
            return unsafe { argmax_f32_neon(logits) };
        }
    }
    argmax_f32_scalar(logits)
}

fn argmax_f32_scalar(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// NEON 4-wide f32 argmax.  Strict `>` means NaN always loses; ties keep the lowest token id.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn argmax_f32_neon(logits: &[f32]) -> u32 {
    use std::arch::aarch64::*;

    let len = logits.len();
    let mut i = 0usize;
    let mut best_v = vdupq_n_f32(f32::NEG_INFINITY);
    let mut best_i = vdupq_n_u32(0);
    let idx_init = [0u32, 1, 2, 3];
    let mut idx_v = vld1q_u32(idx_init.as_ptr());
    let idx_step = vdupq_n_u32(4);

    while i + 4 <= len {
        // SAFETY: loop condition guarantees four valid f32 values are in-bounds.
        let v = vld1q_f32(logits.as_ptr().add(i));
        let mask = vcgtq_f32(v, best_v); // strict >; NaN produces false (NaN loses).
        best_v = vbslq_f32(mask, v, best_v);
        best_i = vbslq_u32(mask, idx_v, best_i);
        idx_v = vaddq_u32(idx_v, idx_step);
        i += 4;
    }

    // Horizontal reduction across 4 SIMD lanes; equal values keep the lower token id.
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    macro_rules! reduce_lane {
        ($lane:literal) => {{
            let lane_val = vgetq_lane_f32::<$lane>(best_v);
            let lane_idx = vgetq_lane_u32::<$lane>(best_i);
            if lane_val > best_val || (lane_val == best_val && lane_idx < best_idx) {
                best_val = lane_val;
                best_idx = lane_idx;
            }
        }};
    }
    reduce_lane!(0);
    reduce_lane!(1);
    reduce_lane!(2);
    reduce_lane!(3);

    // Scalar tail for the final 0-3 elements.
    while i < len {
        let v = *logits.get_unchecked(i);
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
        i += 1;
    }
    best_idx
}

// --- H1: streaming min-heap top-k -------------------------------------------
//
// Replaces select_nth_unstable_by on 248,320 elements with a k-element min-heap
// scan.  The NEON variant prefilters 4 logits per cycle with vcgtq_f32, skipping
// ~95% of the vocabulary without scalar work.  The heap root always holds the
// current worst candidate; any element strictly better replaces it.

/// True when `a` is the *worse* candidate — it should be evicted first from top-k.
/// This determines which element sits at the min-heap root (the weakest keeper).
#[inline(always)]
fn heap_less(a: &Candidate, b: &Candidate) -> bool {
    match (a.logit.is_nan(), b.logit.is_nan()) {
        (true, _) => true,
        (_, true) => false,
        _ => a.logit < b.logit || (a.logit == b.logit && a.token_id > b.token_id),
    }
}

/// Binary min-heap sift-down; root holds the worst candidate in the current top-k.
fn heap_sift_down(heap: &mut [Candidate], mut pos: usize) {
    let n = heap.len();
    loop {
        let left = 2 * pos + 1;
        let right = 2 * pos + 2;
        let mut smallest = pos;
        if left < n && heap_less(&heap[left], &heap[smallest]) {
            smallest = left;
        }
        if right < n && heap_less(&heap[right], &heap[smallest]) {
            smallest = right;
        }
        if smallest == pos {
            break;
        }
        heap.swap(pos, smallest);
        pos = smallest;
    }
}

/// Floyd O(n) heap construction.
fn heap_build(heap: &mut [Candidate]) {
    if heap.len() <= 1 {
        return;
    }
    let mut i = heap.len() / 2;
    while i > 0 {
        i -= 1;
        heap_sift_down(heap, i);
    }
}

/// Runtime-dispatch top-k: NEON on aarch64 when detected, scalar otherwise.
/// Temperature is fused into the scan — candidates store `(logit - center) * inv_temp`.
///
/// Callers pass `center = max_logit` when `inv_temp > 1.0`, and `center = 0.0`
/// otherwise. When `inv_temp <= 1`, `|logit * inv_temp| <= |logit|`, so scaling the
/// raw logit can never overflow, and centering is skipped so the result is exactly
/// `logit * inv_temp` (bit-identical to no centering at all). When `inv_temp > 1`,
/// scaling the raw logit first can overflow to +inf above `f32::MAX / inv_temp`,
/// tying every overflowed logit together on token id instead of on value; centering
/// first avoids that because every centered value is `<= 0`, so scaling can only
/// overflow *downward*, and a value that reaches `-inf` there already had a true
/// scaled value below `-f32::MAX`, whose probability rounds to zero either way. So
/// centering only when `inv_temp > 1` loses no probability mass in either regime.
fn select_top_k(logits: &[f32], k: usize, inv_temp: f32, center: f32, out: &mut Vec<Candidate>) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON runtime-detected above.
            unsafe { select_top_k_neon(logits, k, inv_temp, center, out) };
            return;
        }
    }
    select_top_k_scalar(logits, k, inv_temp, center, out);
}

fn select_top_k_scalar(
    logits: &[f32],
    k: usize,
    inv_temp: f32,
    center: f32,
    out: &mut Vec<Candidate>,
) {
    out.clear();
    if logits.is_empty() {
        return;
    }
    // k == 0 means "top-k disabled" (SamplingConfig::top_k docs): keep every
    // candidate, matching retain_top_k. Returning an empty set here would force
    // sample_top_p_with_scratch to always emit token 0.
    let k = if k == 0 {
        logits.len()
    } else {
        k.min(logits.len())
    };

    // Phase 1: seed the heap with the first k elements, centered then scaled by inv_temp.
    // NaN seeds are replaced with NEG_INFINITY so the heap root is never NaN.
    // A NaN root would cause vcgtq_f32 to reject all finite candidates in Phase 2.
    out.extend(logits.iter().take(k).enumerate().map(|(i, &raw)| {
        let scaled = (raw - center) * inv_temp;
        Candidate {
            token_id: i as u32,
            logit: if scaled.is_nan() {
                f32::NEG_INFINITY
            } else {
                scaled
            },
        }
    }));
    heap_build(out);

    // Phase 2: scan the rest; evict the root whenever a better candidate arrives.
    for (i, &raw) in logits.iter().enumerate().skip(k) {
        let logit = (raw - center) * inv_temp;
        let cand = Candidate {
            token_id: i as u32,
            logit,
        };
        if heap_less(&out[0], &cand) {
            out[0] = cand;
            heap_sift_down(out, 0);
        }
    }
    // Output is unsorted heap order; caller sorts with candidate_order.
}

/// NEON variant: vcgtq_f32 4-wide threshold gate on scaled logits; only falls to scalar work
/// when at least one element in the 4-wide batch beats the current heap root.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn select_top_k_neon(
    logits: &[f32],
    k: usize,
    inv_temp: f32,
    center: f32,
    out: &mut Vec<Candidate>,
) {
    use std::arch::aarch64::*;

    out.clear();
    if logits.is_empty() {
        return;
    }
    // k == 0 means "top-k disabled": keep every candidate (mirror of the scalar path).
    let k = if k == 0 {
        logits.len()
    } else {
        k.min(logits.len())
    };

    // Phase 1: scalar heap seed (k ≤ 256, no NEON benefit here), centered then scaled.
    // NaN seeds are replaced with NEG_INFINITY so the heap root is never NaN.
    // A NaN root would cause vcgtq_f32 to reject all finite candidates in Phase 2.
    out.extend(logits.iter().take(k).enumerate().map(|(i, &raw)| {
        let scaled = (raw - center) * inv_temp;
        Candidate {
            token_id: i as u32,
            logit: if scaled.is_nan() {
                f32::NEG_INFINITY
            } else {
                scaled
            },
        }
    }));
    heap_build(out);

    let n = logits.len();
    let mut i = k;
    let inv_v = vdupq_n_f32(inv_temp);
    let center_v = vdupq_n_f32(center);

    // Phase 2: 4-wide threshold prefilter on centered, scaled logits.
    while i + 4 <= n {
        let thresh = out[0].logit;
        let thresh_v = vdupq_n_f32(thresh);
        // SAFETY: loop bound guarantees 4 valid elements.
        let raw_v = vld1q_f32(logits.as_ptr().add(i));
        let scaled_v = vmulq_f32(vsubq_f32(raw_v, center_v), inv_v);
        // Strict >: NaN input produces false (NaN never beats threshold).
        let mask = vcgtq_f32(scaled_v, thresh_v);
        // Horizontal OR: non-zero if any lane passed the threshold test.
        let any = vgetq_lane_u32::<0>(mask)
            | vgetq_lane_u32::<1>(mask)
            | vgetq_lane_u32::<2>(mask)
            | vgetq_lane_u32::<3>(mask);
        if any != 0 {
            for j in 0..4usize {
                let logit = (*logits.get_unchecked(i + j) - center) * inv_temp;
                let cand = Candidate {
                    token_id: (i + j) as u32,
                    logit,
                };
                if heap_less(&out[0], &cand) {
                    out[0] = cand;
                    heap_sift_down(out, 0);
                }
            }
        }
        i += 4;
    }

    // Scalar tail for the final 0-3 elements.
    while i < n {
        let logit = (*logits.get_unchecked(i) - center) * inv_temp;
        let cand = Candidate {
            token_id: i as u32,
            logit,
        };
        if heap_less(&out[0], &cand) {
            out[0] = cand;
            heap_sift_down(out, 0);
        }
        i += 1;
    }
}

/// Descending-logit ordering with NaN-last and ascending token_id tie-breaking.
/// Matches Metal `topk_better` semantics: higher logit wins; equal logits → lower token_id wins.
///
/// Two NaN logits compare *equal on the logit* and fall back to the token_id
/// tie-break. The earlier `(true, _) => Greater` form returned `Greater` for
/// both `(a, b)` and `(b, a)` when both were NaN, which is not antisymmetric —
/// `sort_by`/`select_nth_unstable_by` require a total strict-weak order, so a
/// NaN-heavy candidate set could sort nondeterministically. Resolving the
/// NaN/NaN case by token_id restores a total order and keeps the tie-break
/// deterministic (lowest id wins, as for finite ties).
#[inline(always)]
fn candidate_order(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.logit.is_nan(), b.logit.is_nan()) {
        (true, true) => a.token_id.cmp(&b.token_id),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => b
            .logit
            .partial_cmp(&a.logit)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.token_id.cmp(&b.token_id)),
    }
}

#[cfg(test)]
mod tests;
