//! Token sampling strategies for text generation.
//!
//! Supports temperature scaling, top-k filtering, top-p (nucleus) sampling,
//! and repetition penalty.

use crate::model::qwen35_config::{GenerateConfig, TopLogprob};

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
mod argmax_f32_first_wins_tests {
    use super::argmax_f32_first_wins;

    /// Mutation sensitivity: replacing the strict `>` comparison with `>=`
    /// (or switching to `Iterator::max_by`, which keeps the LAST tied
    /// element) flips this to return index 2 instead of 1.
    #[test]
    fn tie_break_is_first_wins() {
        let logits = [0.0_f32, 1.0, 1.0, 0.5];
        assert_eq!(
            argmax_f32_first_wins(&logits),
            1,
            "tied maximum logits must resolve to the FIRST index, not the last"
        );
    }

    #[test]
    fn no_tie_returns_true_max() {
        let logits = [0.1_f32, -2.0, 5.5, -0.001];
        assert_eq!(argmax_f32_first_wins(&logits), 2);
    }

    /// Mutation sensitivity: initializing `best_val` to `0.0` instead of
    /// `NEG_INFINITY` would make this return 0 (all-negative logits never
    /// beat a 0.0 floor) instead of the true argmax at index 1.
    #[test]
    fn all_negative_logits_still_find_true_max() {
        let logits = [-5.0_f32, -1.0, -3.0];
        assert_eq!(argmax_f32_first_wins(&logits), 1);
    }

    /// A NaN in a non-max position must never win: `NaN > best_val` is
    /// always false, so the comparison silently skips it.
    #[test]
    fn nan_in_nonmax_position_is_skipped() {
        let logits = [1.0_f32, f32::NAN, 2.0];
        assert_eq!(argmax_f32_first_wins(&logits), 2);
    }

    /// All-NaN input must fail closed to index 0, never panic.
    #[test]
    fn all_nan_fails_closed_to_zero() {
        let logits = [f32::NAN, f32::NAN, f32::NAN];
        assert_eq!(argmax_f32_first_wins(&logits), 0);
    }

    /// Empty input must fail closed to index 0, never panic.
    #[test]
    fn empty_slice_fails_closed_to_zero() {
        let logits: [f32; 0] = [];
        assert_eq!(argmax_f32_first_wins(&logits), 0);
    }

    /// All-`NEG_INFINITY` (a fully-masked grammar/top-k step) must fail
    /// closed to index 0.
    #[test]
    fn all_neg_infinity_fails_closed_to_zero() {
        let logits = [f32::NEG_INFINITY; 4];
        assert_eq!(argmax_f32_first_wins(&logits), 0);
    }
}

/// **Unstable**: sampling configuration for decoder-only generation; fields
/// and defaults may change as the generation API evolves.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for logit scaling. 0.0 = greedy, 1.0 = unscaled.
    pub temperature: f32,
    /// Top-k: keep only the k highest-probability tokens. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus): keep tokens whose cumulative probability <= p. 1.0 = disabled.
    pub top_p: f32,
    /// Repetition penalty multiplier. 1.0 = no penalty.
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
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
/// top-k → top-p → sample.  Both the Metal compact path and the CPU Qwen
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

    /// Scale logits by `1 / temperature`.  No-op when temperature is 1.0,
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
        let inv = 1.0 / temperature;
        for c in &mut self.candidates {
            c.logit *= inv;
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
        self.sample_top_p_with_scratch(top_p, r, &mut probs)
    }

    /// Same as `sample_top_p` but reuses the caller's probability buffer to avoid allocation.
    fn sample_top_p_with_scratch(&mut self, top_p: f32, r: f32, probs: &mut Vec<f32>) -> u32 {
        if self.candidates.is_empty() {
            return 0;
        }

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

        // Sort descending for deterministic top-p traversal.
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
        probs.clear();
        probs.extend(self.candidates.iter().map(|c| (c.logit - max_logit).exp()));
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
    rng: Rng,
    /// Token IDs seen since the last `reset`: prompt tokens (from `seed_history`)
    /// followed by all generated tokens.  Full history is retained for repetition penalty.
    recent_tokens: Vec<u32>,
    /// Reused dedup set for penalty application; cleared at the start of each use.
    penalty_seen: std::collections::HashSet<u32>,
    /// Reused candidate buffer; avoids 1.9 MB alloc per call at vocab=248,320.
    candidate_scratch: Vec<Candidate>,
    /// Reused probability buffer for top-p softmax.
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
        Self {
            config,
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

        let inv_temp = if self.config.temperature != 1.0 {
            1.0 / self.config.temperature
        } else {
            1.0
        };

        // Streaming min-heap top-k with fused temperature scaling.
        // ~95% of vocab elements are skipped by the NEON threshold gate.
        select_top_k(
            &self.logit_scratch,
            self.config.top_k,
            inv_temp,
            &mut self.candidate_scratch,
        );
        let mut cs = CandidateSet {
            candidates: std::mem::take(&mut self.candidate_scratch),
        };
        let r = self.rng.next_f32();
        let token = cs.sample_top_p_with_scratch(self.config.top_p, r, &mut self.prob_scratch);
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
    /// f16/q8/NEON/batch-prefill forward paths). Kept thread-local instead of
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
/// temperature-scale + partial top-k select feeding a top-p softmax draw.
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
        // `CandidateSet::sample_top_p_with_scratch` ever sees it, silently
        // turning a "poisoned distribution" case into "normal sampling over a
        // masked candidate" — exactly the regression this scan closes.
        //
        // Mirror the old full-logit contract (`build_softmax_probs` returning
        // `None` on a non-finite max or a NaN-poisoned sum, both of which fall
        // back to argmax) with a single pass over the pre-scaled logits: any
        // NaN anywhere forces degeneracy (a NaN can never become the max, so it
        // would otherwise poison the softmax sum from a non-max position,
        // #322-class); a non-finite max (every logit `-inf`, or a `+inf` logit)
        // also forces degeneracy, matching `sample_top_p_with_scratch`'s
        // non-finite-max short-circuit. Scaling by a positive finite `inv_temp`
        // (guaranteed by the `temperature_degenerate` check above) preserves
        // both NaN-ness and finiteness, so scanning before the fused top-k scale
        // is equivalent to scanning after it.
        let (has_nan, max_logit) = scan_nan_or_nonfinite_max(logit_scratch);
        if has_nan || !max_logit.is_finite() {
            return argmax_f32(logit_scratch);
        }

        // Streaming min-heap top-k with fused temperature scaling — the softmax
        // draw below runs only over these k survivors, never the full vocabulary.
        select_top_k(logit_scratch, cfg.top_k, inv_temp, candidate_scratch);

        let mut cs = CandidateSet {
            candidates: std::mem::take(candidate_scratch),
        };
        let r = uniform_f32_from_u64(xorshift64_next(rng_state));
        let token = cs.sample_top_p_with_scratch(cfg.top_p, r, prob_scratch);
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

    let mut max_scaled = f32::NEG_INFINITY;
    for &v in logits {
        let scaled = v * scale;
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

    // sum >= 1.0 always: the max term alone contributes exp(0) = 1.0, so
    // log_sum is finite and non-negative (bounded above by ln(vocab_size)) --
    // it cannot overflow or introduce a fresh non-finite value here.
    let mut sum = 0.0f32;
    for &v in logits {
        sum += (v * scale - max_scaled).exp();
    }
    let log_sum = sum.ln();

    let logprob_of = |idx: usize| -> f32 {
        let lp = (logits[idx] * scale - max_scaled) - log_sum;
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
        let mut candidates: Vec<Candidate> = (0..vocab_size)
            .map(|i| Candidate {
                token_id: i as u32,
                logit: logits[i] * scale,
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

/// Fail-closed pre-scan for `sample_full_logits`: detects whether `logits`
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
/// `inv_temp` is fused into the scan — candidates store scaled logits.
fn select_top_k(logits: &[f32], k: usize, inv_temp: f32, out: &mut Vec<Candidate>) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON runtime-detected above.
            unsafe { select_top_k_neon(logits, k, inv_temp, out) };
            return;
        }
    }
    select_top_k_scalar(logits, k, inv_temp, out);
}

fn select_top_k_scalar(logits: &[f32], k: usize, inv_temp: f32, out: &mut Vec<Candidate>) {
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

    // Phase 1: seed the heap with the first k elements, scaled by inv_temp.
    // NaN seeds are replaced with NEG_INFINITY so the heap root is never NaN.
    // A NaN root would cause vcgtq_f32 to reject all finite candidates in Phase 2.
    out.extend(logits.iter().take(k).enumerate().map(|(i, &raw)| {
        let scaled = raw * inv_temp;
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
        let logit = raw * inv_temp;
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
unsafe fn select_top_k_neon(logits: &[f32], k: usize, inv_temp: f32, out: &mut Vec<Candidate>) {
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

    // Phase 1: scalar heap seed (k ≤ 256, no NEON benefit here), scaled by inv_temp.
    // NaN seeds are replaced with NEG_INFINITY so the heap root is never NaN.
    // A NaN root would cause vcgtq_f32 to reject all finite candidates in Phase 2.
    out.extend(logits.iter().take(k).enumerate().map(|(i, &raw)| {
        let scaled = raw * inv_temp;
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

    // Phase 2: 4-wide threshold prefilter on scaled logits.
    while i + 4 <= n {
        let thresh = out[0].logit;
        let thresh_v = vdupq_n_f32(thresh);
        // SAFETY: loop bound guarantees 4 valid elements.
        let raw_v = vld1q_f32(logits.as_ptr().add(i));
        let scaled_v = vmulq_f32(raw_v, inv_v);
        // Strict >: NaN input produces false (NaN never beats threshold).
        let mask = vcgtq_f32(scaled_v, thresh_v);
        // Horizontal OR: non-zero if any lane passed the threshold test.
        let any = vgetq_lane_u32::<0>(mask)
            | vgetq_lane_u32::<1>(mask)
            | vgetq_lane_u32::<2>(mask)
            | vgetq_lane_u32::<3>(mask);
        if any != 0 {
            for j in 0..4usize {
                let logit = *logits.get_unchecked(i + j) * inv_temp;
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
        let logit = *logits.get_unchecked(i) * inv_temp;
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
mod tests {
    use super::*;

    #[test]
    fn test_greedy_picks_argmax() {
        let config = SamplingConfig::greedy();
        let mut sampler = Sampler::new(config);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(sampler.sample(&logits), 3);
    }

    #[test]
    fn test_temperature_zero_is_greedy() {
        let config = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);
        let logits = vec![1.0, 5.0, 2.0];
        assert_eq!(sampler.sample(&logits), 1);
    }

    #[test]
    fn test_nonfinite_temperature_falls_back_to_argmax() {
        // top_k=2 forces the non-greedy fused-top-k path pre-fix, where a NaN/inf
        // temperature produced `inv_temp = NaN`/`0`, scaling every logit to NaN/0
        // and collapsing the result to token 0. The guard must instead return the
        // true argmax (token 1) for any non-finite temperature.
        let logits = vec![0.0, 100.0, 99.0];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let config = SamplingConfig {
                temperature: bad,
                top_k: 2,
                top_p: 1.0,
                repetition_penalty: 1.0,
            };
            let mut sampler = Sampler::new(config).with_seed(7);
            assert_eq!(
                sampler.sample(&logits),
                1,
                "temperature {bad} must fall back to argmax, not collapse to token 0"
            );
        }
    }

    #[test]
    fn test_tiny_temperature_falls_back_to_argmax() {
        // A finite, positive, but tiny temperature whose reciprocal overflows f32 to
        // +inf (t < ~2.94e-39). Pre-fix `inv_temp = 1.0 / t = inf` scaled every logit
        // to +inf, destroying their ordering, and the fused top-k collapsed to token 0.
        // The t -> 0+ limit of softmax(logits / t) is argmax, so the highest-logit
        // token (1) must win. top_k=2 forces the non-greedy fused-top-k path.
        let logits = vec![10.0, 11.0, 9.0];
        for tiny in [1e-45_f32, 1e-40, 1e-39] {
            let config = SamplingConfig {
                temperature: tiny,
                top_k: 2,
                top_p: 1.0,
                repetition_penalty: 1.0,
            };
            let mut sampler = Sampler::new(config).with_seed(7);
            assert_eq!(
                sampler.sample(&logits),
                1,
                "tiny temperature {tiny} must fall back to argmax, not collapse to token 0"
            );
        }
    }

    #[test]
    fn test_temperature_degenerate_predicate() {
        // Degenerate: non-finite, non-positive, finite-but-tiny where the reciprocal
        // overflows (1e-45..1e-39), and the residual band where the reciprocal is finite
        // but `logit * inv_temp` would overflow for a plausible logit (1e-38..1e-35,
        // including f32::MIN_POSITIVE whose reciprocal 8.5e37 exceeds f32::MAX / 1e4).
        for bad in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -1.0,
            1e-45,
            1e-40,
            1e-39,
            f32::MIN_POSITIVE,
            1e-37,
            1e-35,
        ] {
            assert!(
                temperature_degenerate(bad),
                "temperature {bad} should be degenerate"
            );
        }
        // Valid: normal temperatures, plus small-but-safe values whose reciprocal stays
        // far enough below f32::MAX that scaling a plausible logit cannot overflow
        // (1e-30 -> inv 1e30, well under the f32::MAX / 1e4 ≈ 3.4e34 threshold).
        for ok in [1.0, 0.5, 2.0, 0.1, 1e-3, 1e-30] {
            assert!(
                !temperature_degenerate(ok),
                "temperature {ok} should be valid (no scaling overflow)"
            );
        }
    }

    #[test]
    fn test_residual_band_temperature_falls_back_to_argmax() {
        // The residual band: a finite, positive temperature whose reciprocal stays finite
        // (so the pre-fix `!(1.0 / t).is_finite()` check missed it) yet is large enough
        // that `logit * inv_temp` overflows for ordinary logits. Pre-fix this saturated
        // every scaled logit to +inf and the softmax non-finite-max guard returned an
        // arbitrary candidate. argmax (token 1) is the correct t -> 0+ limit.
        let logits = vec![10.0, 11.0, 9.0];
        for band in [f32::MIN_POSITIVE, 1e-37_f32, 1e-36] {
            let config = SamplingConfig {
                temperature: band,
                top_k: 2,
                top_p: 1.0,
                repetition_penalty: 1.0,
            };
            let mut sampler = Sampler::new(config).with_seed(7);
            assert_eq!(
                sampler.sample(&logits),
                1,
                "residual-band temperature {band} must fall back to argmax, not collapse"
            );
        }
    }

    #[test]
    fn test_apply_temperature_nonfinite_is_noop() {
        // A non-finite (or non-positive) temperature must leave logits untouched —
        // `1.0 / NaN` would poison the whole candidate set, `1.0 / inf` would zero it.
        for bad in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0_f32,
            -2.0_f32,
        ] {
            let mut cs = CandidateSet::from_full_logits(&[1.0, 5.0, 3.0]);
            cs.apply_temperature(bad);
            let logits: Vec<f32> = cs.candidates.iter().map(|c| c.logit).collect();
            assert_eq!(
                logits,
                vec![1.0, 5.0, 3.0],
                "temperature {bad} must be a no-op, leaving logits finite and unscaled"
            );
            assert_eq!(cs.argmax(), 1);
        }
    }

    #[test]
    fn apply_temperature_tiny_positive_temp_stays_greedy() {
        // A finite-but-tiny temperature has a reciprocal that overflows f32. The
        // t -> 0+ limit must stay greedy (argmax). Without the degenerate-branch
        // collapse, `1.0 / 1e-45 == +inf` scales both logits to +inf; the sort then
        // tie-breaks on ascending token_id and `sample_top_p` returns token 0 instead
        // of the argmax (token 1).
        let mut cs = CandidateSet::from_full_logits(&[10.0, 11.0]);
        cs.apply_temperature(1e-45);
        // top_p = 1.0 (no nucleus truncation) + r = 0.0 => pure argmax selection.
        assert_eq!(
            cs.sample_top_p(1.0, 0.0),
            1,
            "tiny positive temperature must resolve to the argmax (greedy), not token 0"
        );
    }

    #[test]
    fn apply_temperature_tiny_temp_is_hard_greedy_for_close_logits() {
        // Even an arbitrarily small logit gap must resolve to hard argmax under a
        // degenerate temperature (the t -> 0+ limit), matching the main Sampler's
        // `temperature_degenerate` route. A finite multiplier (clamped or not) would
        // leave these two near 50/50, so `sample_top_p(1.0, 0.75)` could draw token 0;
        // the one-hot collapse makes the argmax (token 1) the only reachable draw.
        let mut cs = CandidateSet::from_full_logits(&[0.0, f32::from_bits(1)]);
        cs.apply_temperature(1e-45);
        assert_eq!(
            cs.sample_top_p(1.0, 0.75),
            1,
            "degenerate temperature must be hard-greedy even for adjacent logits"
        );
    }

    #[test]
    fn test_top_k_limits_candidates() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let mut sampler = Sampler::new(config).with_seed(123);
        // With top_k=2, only the two highest logits (idx 1 and 3) should be sampled
        let logits = vec![0.0, 10.0, 0.0, 9.0, 0.0];
        let mut counts = [0u32; 5];
        for _ in 0..100 {
            let tok = sampler.sample(&logits);
            counts[tok as usize] += 1;
        }
        // Only tokens 1 and 3 should appear
        assert_eq!(counts[0], 0);
        assert_eq!(counts[2], 0);
        assert_eq!(counts[4], 0);
        assert!(counts[1] > 0);
        assert!(counts[3] > 0);
    }

    #[test]
    fn test_repetition_penalty_reduces_probability() {
        let config = SamplingConfig {
            temperature: 0.0, // greedy
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 100.0, // very strong penalty
        };
        let mut sampler = Sampler::new(config);
        let logits = vec![0.0, 5.0, 4.9]; // token 1 is best

        // First sample: picks 1 (highest)
        let first = sampler.sample(&logits);
        assert_eq!(first, 1);

        // Second sample: token 1 is penalized, should pick 2
        let second = sampler.sample(&logits);
        assert_eq!(second, 2);
    }

    /// A repetition penalty in (0, 1) BOOSTS recent tokens. The greedy fast path
    /// must not shortcut to the raw argmax when a boosted recent token would
    /// overtake it, even though the raw argmax itself is not recent (the
    /// fast-path invariant only holds for penalty >= 1.0).
    #[test]
    fn test_greedy_sub_one_penalty_boosts_recent_token() {
        let config = SamplingConfig {
            temperature: 0.0, // greedy fast path
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 0.5, // < 1.0 boosts recent tokens
        };
        let mut sampler = Sampler::new(config);

        // Step 1: token 1 is the clear argmax; becomes a recent token.
        assert_eq!(sampler.sample(&[0.0, 10.0]), 1);

        // Step 2: raw argmax is token 0 (5.0) and is NOT recent, but recent token
        // 1 is boosted to 4.0 / 0.5 = 8.0 > 5.0, so the penalized argmax is token
        // 1. Before the fix the fast path returned the raw argmax token 0.
        assert_eq!(sampler.sample(&[5.0, 4.0]), 1);
    }

    /// The documented always-safe case the conservative gate leaves to the re-scan:
    /// a (0, 1) penalty where the raw argmax is ITSELF recent. `penalized_logit` is
    /// monotone, so boosting raw_best keeps it the argmax — the fall-through must
    /// still return it. Regression guard for the comment at the gate.
    #[test]
    fn test_greedy_sub_one_penalty_recent_argmax_unchanged() {
        let config = SamplingConfig {
            temperature: 0.0, // greedy fast path
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 0.5, // < 1.0 boosts recent tokens
        };
        let mut sampler = Sampler::new(config);

        // Step 1: token 1 is the clear argmax; becomes a recent token.
        assert_eq!(sampler.sample(&[0.0, 10.0]), 1);

        // Step 2: raw argmax is token 1 (6.0) and IS recent. Boosting it to
        // 6.0 / 0.5 = 12.0 keeps it above token 0's 5.0, so it stays the argmax.
        assert_eq!(sampler.sample(&[5.0, 6.0]), 1);
    }

    #[test]
    fn test_top_p_nucleus_sampling() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            repetition_penalty: 1.0,
        };
        let mut sampler = Sampler::new(config).with_seed(456);
        // Token 0 has much higher logit, should dominate with top_p=0.5
        let logits = vec![10.0, 1.0, 1.0, 1.0, 1.0];
        let mut counts = [0u32; 5];
        for _ in 0..100 {
            counts[sampler.sample(&logits) as usize] += 1;
        }
        // Token 0 should get almost all samples (its probability >> 0.5)
        assert!(counts[0] > 90);
    }

    #[test]
    fn test_sampler_reset() {
        let config = SamplingConfig::greedy();
        let mut sampler = Sampler::new(config);
        sampler.sample(&[1.0, 2.0, 3.0]);
        assert!(!sampler.recent_tokens.is_empty());
        sampler.reset();
        assert!(sampler.recent_tokens.is_empty());
    }

    // ── argmax_f32 NEON parity tests ──────────────────────────────────────────

    fn check_argmax_parity(logits: &[f32]) {
        let scalar = argmax_f32_scalar(logits);
        let dispatch = argmax_f32(logits);
        assert_eq!(
            scalar,
            dispatch,
            "argmax_f32 dispatch differs from scalar for len={}",
            logits.len()
        );
    }

    #[test]
    fn test_argmax_full_vocab() {
        // Deterministic logits covering full Qwen vocab length.
        let n = 248_320usize;
        let logits: Vec<f32> = (0..n)
            .map(|i| {
                let h = (i as u64)
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                (h as f32 / u64::MAX as f32) * 20.0 - 10.0
            })
            .collect();
        check_argmax_parity(&logits);
    }

    #[test]
    fn test_argmax_lower_id_tie_wins() {
        // Both token 2 and 5 have the same max value; lower id (2) must win.
        let mut logits = vec![0.0f32; 10];
        logits[2] = 5.0;
        logits[5] = 5.0;
        assert_eq!(argmax_f32_scalar(&logits), 2, "scalar tie");
        assert_eq!(argmax_f32(&logits), 2, "dispatch tie");
    }

    #[test]
    fn test_argmax_nan_loses_to_real() {
        // NaN at index 0 must not win; index 3 holds the real max.
        let logits = vec![f32::NAN, 1.0, 2.0, 9.0, 0.5];
        assert_eq!(argmax_f32_scalar(&logits), 3, "scalar nan");
        assert_eq!(argmax_f32(&logits), 3, "dispatch nan");
    }

    #[test]
    fn test_argmax_all_nan_returns_0() {
        let logits = vec![f32::NAN, f32::NAN, f32::NAN];
        assert_eq!(argmax_f32_scalar(&logits), 0, "scalar all-nan");
        assert_eq!(argmax_f32(&logits), 0, "dispatch all-nan");
    }

    #[test]
    fn test_argmax_all_neg_inf_returns_0() {
        let logits = vec![f32::NEG_INFINITY; 8];
        assert_eq!(argmax_f32_scalar(&logits), 0, "scalar neg-inf");
        assert_eq!(argmax_f32(&logits), 0, "dispatch neg-inf");
    }

    // ── CandidateSet::argmax correctness ─────────────────────────────────────

    #[test]
    fn test_candidateset_argmax_all_masked_returns_first_in_set_token() {
        // After grammar or top-k masking every logit in a compacted CandidateSet
        // can be NEG_INFINITY (the full vocabulary was masked except for these
        // candidates, whose own scores are also suppressed).  When that happens
        // the comparison loop never fires, so argmax must return the *first
        // candidate's token_id* — an in-set token — not the bare sentinel 0.
        // Token ids 7 and 9 are deliberately chosen to be non-zero; if the
        // initializer regresses to `0u32` this test returns 0 instead of 7 and
        // fails, catching the regression.
        let cs = CandidateSet::from_candidates(vec![
            Candidate {
                token_id: 7,
                logit: f32::NEG_INFINITY,
            },
            Candidate {
                token_id: 9,
                logit: f32::NEG_INFINITY,
            },
        ]);
        assert_eq!(
            cs.argmax(),
            7,
            "all-masked compact set must return first in-set token id"
        );
    }

    #[test]
    fn test_argmax_partial_chunk_lengths() {
        // Test tail lengths 1-7 (0 mod 4 through 3 mod 4, and longer).
        for tail in 1usize..=7 {
            let mut logits = vec![0.0f32; 8 + tail];
            logits[8 + tail - 1] = 99.0; // max is always in the tail
            check_argmax_parity(&logits);
            assert_eq!(
                argmax_f32(&logits),
                (8 + tail - 1) as u32,
                "tail len={tail}"
            );
        }
    }

    // ── H3: candidate_order determinism tests ────────────────────────────────

    #[test]
    fn test_candidate_order_higher_logit_wins() {
        use std::cmp::Ordering;
        let a = Candidate {
            token_id: 10,
            logit: 5.0,
        };
        let b = Candidate {
            token_id: 20,
            logit: 3.0,
        };
        assert_eq!(
            candidate_order(&a, &b),
            Ordering::Less,
            "higher logit must sort first"
        );
    }

    #[test]
    fn test_candidate_order_tie_lower_token_id_wins() {
        use std::cmp::Ordering;
        let a = Candidate {
            token_id: 5,
            logit: 2.0,
        };
        let b = Candidate {
            token_id: 9,
            logit: 2.0,
        };
        assert_eq!(
            candidate_order(&a, &b),
            Ordering::Less,
            "equal logit: lower token_id is first"
        );
        assert_eq!(candidate_order(&b, &a), Ordering::Greater);
    }

    #[test]
    fn test_candidate_order_nan_loses() {
        use std::cmp::Ordering;
        let nan = Candidate {
            token_id: 0,
            logit: f32::NAN,
        };
        let real = Candidate {
            token_id: 99,
            logit: -1000.0,
        };
        assert_eq!(
            candidate_order(&nan, &real),
            Ordering::Greater,
            "NaN must sort last"
        );
        assert_eq!(candidate_order(&real, &nan), Ordering::Less);
    }

    #[test]
    fn test_retain_top_k_tie_breaking() {
        // Three candidates with the same logit; top-2 must keep the two lowest token_ids.
        let mut cs = CandidateSet {
            candidates: vec![
                Candidate {
                    token_id: 7,
                    logit: 1.0,
                },
                Candidate {
                    token_id: 2,
                    logit: 1.0,
                },
                Candidate {
                    token_id: 5,
                    logit: 1.0,
                },
            ],
        };
        cs.retain_top_k(2);
        cs.candidates.sort_by(candidate_order);
        assert_eq!(cs.candidates[0].token_id, 2);
        assert_eq!(cs.candidates[1].token_id, 5);
    }

    #[test]
    fn test_candidate_order_nan_nan_antisymmetric() {
        use std::cmp::Ordering;
        // Two NaN-logit candidates must produce a valid total order: the
        // comparator has to be antisymmetric (a<b ⇒ b>a) and break the tie
        // deterministically by token_id. The old `(true, _) => Greater` form
        // returned Greater for both (a,b) and (b,a), violating the strict-weak
        // order contract of sort_by/select_nth_unstable_by.
        let a = Candidate {
            token_id: 3,
            logit: f32::NAN,
        };
        let b = Candidate {
            token_id: 7,
            logit: f32::NAN,
        };
        assert_eq!(candidate_order(&a, &b), Ordering::Less);
        assert_eq!(candidate_order(&b, &a), Ordering::Greater);
        assert_eq!(candidate_order(&a, &a), Ordering::Equal);

        // A multi-NaN set sorts deterministically by token_id.
        let mut nans = [
            Candidate {
                token_id: 9,
                logit: f32::NAN,
            },
            Candidate {
                token_id: 1,
                logit: f32::NAN,
            },
            Candidate {
                token_id: 4,
                logit: f32::NAN,
            },
        ];
        nans.sort_by(candidate_order);
        assert_eq!(
            nans.iter().map(|c| c.token_id).collect::<Vec<_>>(),
            vec![1, 4, 9],
            "all-NaN set must sort by ascending token_id"
        );
    }

    #[test]
    fn test_sample_top_p_tail_nan_returns_argmax() {
        // A NaN logit in a non-max position sorts last (so the finite max guard
        // does not catch it), but its exp() poisons the softmax sum to NaN. The
        // weighted-sample loop then never matches `r < cumsum` and would fall
        // through to candidates.last() — the NaN-sorted-last token (id 1) — a
        // garbage selection. The sum-finite guard must instead return the argmax
        // (token 0, the highest finite logit). select_top_k sanitises NaN→-inf in
        // the Sampler path, so this state is reachable via the public CandidateSet
        // API; constructing one directly exercises the residual hole.
        let mut cs = CandidateSet::from_candidates(vec![
            Candidate {
                token_id: 0,
                logit: 100.0,
            },
            Candidate {
                token_id: 1,
                logit: f32::NAN,
            },
            Candidate {
                token_id: 2,
                logit: 50.0,
            },
        ]);
        // r in the middle of the distribution; pre-fix this returns last() = 1.
        let token = cs.sample_top_p(1.0, 0.5);
        assert_eq!(
            token, 0,
            "tail-NaN poisons the softmax sum; must fall back to argmax (token 0), not last()"
        );
    }

    #[test]
    fn test_sample_top_p_invalid_top_p_normalized() {
        // #328: an out-of-range top_p must resolve to defined behaviour, not the
        // accidental fall-through of the raw `top_p < 1.0` gate. NaN / +Inf / >1
        // map to 1.0 (no nucleus truncation, identical to top_p == 1.0); a
        // negative top_p collapses to greedy (argmax only). A fresh candidate set
        // is built per call because sample_top_p truncates its buffers in place.
        let make = || {
            CandidateSet::from_candidates(vec![
                Candidate {
                    token_id: 0,
                    logit: 3.0,
                },
                Candidate {
                    token_id: 1,
                    logit: 2.0,
                },
                Candidate {
                    token_id: 2,
                    logit: 1.0,
                },
                Candidate {
                    token_id: 3,
                    logit: 0.0,
                },
            ])
        };

        for &r in &[0.0f32, 0.25, 0.5, 0.75, 0.999] {
            let baseline = make().sample_top_p(1.0, r);
            assert_eq!(
                make().sample_top_p(f32::NAN, r),
                baseline,
                "NaN top_p must behave as top_p == 1.0 at r={r}"
            );
            assert_eq!(
                make().sample_top_p(1.5, r),
                baseline,
                ">1 top_p must behave as top_p == 1.0 at r={r}"
            );
            assert_eq!(
                make().sample_top_p(f32::INFINITY, r),
                baseline,
                "+Inf top_p must behave as top_p == 1.0 at r={r}"
            );
            assert_eq!(
                make().sample_top_p(-0.5, r),
                0,
                "negative top_p must collapse to greedy argmax at r={r}"
            );
        }
    }

    // ── H1: select_top_k correctness tests ──────────────────────────────────

    fn check_top_k_parity(logits: &[f32], k: usize) {
        let mut scalar_out = Vec::new();
        select_top_k_scalar(logits, k, 1.0, &mut scalar_out);
        scalar_out.sort_by(candidate_order);

        let mut dispatch_out = Vec::new();
        select_top_k(logits, k, 1.0, &mut dispatch_out);
        dispatch_out.sort_by(candidate_order);

        assert_eq!(
            scalar_out.len(),
            dispatch_out.len(),
            "k={k}: length mismatch"
        );
        for (i, (s, d)) in scalar_out.iter().zip(dispatch_out.iter()).enumerate() {
            assert_eq!(
                s.token_id, d.token_id,
                "k={k}: position {i} token_id mismatch (scalar={} dispatch={})",
                s.token_id, d.token_id
            );
        }
    }

    #[test]
    fn test_select_top_k_basic() {
        let logits = vec![1.0f32, 5.0, 3.0, 9.0, 2.0, 7.0];
        let mut out = Vec::new();
        select_top_k_scalar(&logits, 3, 1.0, &mut out);
        out.sort_by(candidate_order);
        let ids: Vec<u32> = out.iter().map(|c| c.token_id).collect();
        assert_eq!(ids, vec![3, 5, 1], "top-3 from [1,5,3,9,2,7]");
    }

    #[test]
    fn test_select_top_k_tie_breaking() {
        // Two tokens with logit=5.0; top-1 must select the lower token_id.
        let logits = vec![5.0f32, 5.0, 1.0];
        let mut out = Vec::new();
        select_top_k_scalar(&logits, 1, 1.0, &mut out);
        assert_eq!(out[0].token_id, 0, "tie: lower id must win");
    }

    #[test]
    fn test_select_top_k_nan_excluded() {
        // NaN at position 0 must not appear in top-k.
        let logits = vec![f32::NAN, 2.0, 9.0, 3.0];
        let mut out = Vec::new();
        select_top_k_scalar(&logits, 2, 1.0, &mut out);
        out.sort_by(candidate_order);
        let ids: Vec<u32> = out.iter().map(|c| c.token_id).collect();
        assert_eq!(ids, vec![2, 3], "NaN must not appear in top-2");
    }

    #[test]
    fn test_select_top_k_full_vocab_parity() {
        let n = 248_320usize;
        let logits: Vec<f32> = (0..n)
            .map(|i| {
                let h = (i as u64)
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                (h as f32 / u64::MAX as f32) * 20.0 - 10.0
            })
            .collect();
        check_top_k_parity(&logits, 50);
    }

    #[test]
    fn test_select_top_k_dispatch_tie_breaking() {
        // When two logits are equal, lower token_id must win.
        let logits = vec![5.0f32, 5.0, 5.0, 1.0, 1.0];
        let mut out = Vec::new();
        select_top_k(&logits, 2, 1.0, &mut out);
        out.sort_by(candidate_order);
        let ids: Vec<u32> = out.iter().map(|c| c.token_id).collect();
        assert_eq!(
            ids,
            vec![0, 1],
            "dispatch: tie must break by lower token_id"
        );
    }

    #[test]
    fn test_select_top_k_dispatch_nan_excluded() {
        // NaN logits must never appear in top-k output from dispatch.
        let logits = vec![f32::NAN, 4.0, 9.0, f32::NAN, 7.0];
        let mut out = Vec::new();
        select_top_k(&logits, 2, 1.0, &mut out);
        out.sort_by(candidate_order);
        let ids: Vec<u32> = out.iter().map(|c| c.token_id).collect();
        assert_eq!(ids, vec![2, 4], "dispatch: NaN must not appear in top-2");
    }

    #[test]
    fn test_select_top_k_dispatch_nan_in_seed() {
        // NaN values in the first k elements (seed phase) must not corrupt output.
        // Build logits: [NaN, 2.0, 3.0, NaN, 5.0, 6.0, 7.0, NaN, 9.0, 10.0, 1..=990 as f32]
        let mut logits = vec![
            f32::NAN,
            2.0,
            3.0,
            f32::NAN,
            5.0,
            6.0,
            7.0,
            f32::NAN,
            9.0,
            10.0,
        ];
        logits.extend((1..=990u32).map(|x| x as f32));
        // k=10: both dispatch and scalar must agree after sorting
        let mut scalar_out = Vec::new();
        select_top_k_scalar(&logits, 10, 1.0, &mut scalar_out);
        scalar_out.sort_by(candidate_order);

        let mut dispatch_out = Vec::new();
        select_top_k(&logits, 10, 1.0, &mut dispatch_out);
        dispatch_out.sort_by(candidate_order);

        assert_eq!(scalar_out.len(), dispatch_out.len());
        for (i, (s, d)) in scalar_out.iter().zip(dispatch_out.iter()).enumerate() {
            assert_eq!(
                s.token_id, d.token_id,
                "nan_in_seed: position {i} token_id mismatch (scalar={} dispatch={})",
                s.token_id, d.token_id
            );
        }
    }

    #[test]
    fn test_select_top_k_applies_inv_temp_to_candidate_logits() {
        // inv_temp=0.5 halves all logits; top-2 must be token 3 (2.0) and token 2 (1.5).
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = Vec::new();
        select_top_k(&logits, 2, 0.5, &mut out);
        out.sort_by(candidate_order);
        assert_eq!(out[0].token_id, 3);
        assert_eq!(out[1].token_id, 2);
        assert!(
            (out[0].logit - 2.0).abs() < 1e-6,
            "logit[3] must be 4.0*0.5=2.0"
        );
        assert!(
            (out[1].logit - 1.5).abs() < 1e-6,
            "logit[2] must be 3.0*0.5=1.5"
        );
    }

    #[test]
    fn test_select_top_k_zero_keeps_all() {
        // k == 0 ("disabled") must keep ALL candidates, never an empty set.
        let logits = vec![1.0f32, 5.0, 3.0, 9.0, 2.0];
        let mut scalar_out = Vec::new();
        select_top_k_scalar(&logits, 0, 1.0, &mut scalar_out);
        assert_eq!(scalar_out.len(), 5, "scalar: k=0 must keep all candidates");
        let mut dispatch_out = Vec::new();
        select_top_k(&logits, 0, 1.0, &mut dispatch_out);
        assert_eq!(
            dispatch_out.len(),
            5,
            "dispatch: k=0 must keep all candidates"
        );
    }

    #[test]
    fn test_top_k_zero_is_no_filtering_not_token_zero() {
        // Regression: top_k=0 used to make select_top_k return empty, so the sampler
        // could only ever emit token 0. With it disabled it must sample across the vocab.
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,   // disabled
            top_p: 1.0, // disabled
            repetition_penalty: 1.0,
        };
        let mut sampler = Sampler::new(config).with_seed(42);
        let logits = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut counts = [0u32; 8];
        for _ in 0..400 {
            counts[sampler.sample(&logits) as usize] += 1;
        }
        let nonzero = counts.iter().filter(|&&c| c > 0).count();
        assert!(
            nonzero >= 5,
            "top_k=0 must sample across the vocab, got counts={counts:?}"
        );
    }

    #[test]
    fn test_sample_top_p_handles_pos_inf_logit() {
        // A +INF logit must select that token, not poison the softmax and fall
        // through to the worst candidate.
        let mut cs = CandidateSet {
            candidates: vec![
                Candidate {
                    token_id: 0,
                    logit: 1.0,
                },
                Candidate {
                    token_id: 1,
                    logit: f32::INFINITY,
                },
                Candidate {
                    token_id: 2,
                    logit: 2.0,
                },
            ],
        };
        let mut scratch = Vec::new();
        let tok = cs.sample_top_p_with_scratch(1.0, 0.999, &mut scratch);
        assert_eq!(tok, 1, "+INF logit token must be selected");
    }

    /// #611 defense-in-depth: even if a grammar's `has_finite_logit` guard
    /// were ever bypassed upstream, the sampler itself must not let an
    /// all-masked (all-`NEG_INFINITY`) `CandidateSet` poison softmax into NaN
    /// or panic. The temperature=0 argmax path already has
    /// `test_argmax_all_neg_inf_returns_0` and
    /// `test_candidateset_argmax_all_masked_returns_first_in_set_token`; this
    /// covers the same all-masked input through the temperature>0 / top-p
    /// nucleus path (`sample_top_p_with_scratch`), a distinct code path with
    /// its own pair of non-finite guards.
    ///
    /// Mutation sensitivity (verified empirically, not just asserted): this
    /// input is defended by *two* independent guards — `!max_logit.is_finite()`
    /// and `!sum.is_finite() || sum <= 0.0` a few lines below it. Disabling
    /// either ONE alone still passes (the other one catches it — genuine
    /// defense-in-depth, confirmed by temporarily disabling each in turn).
    /// Disabling BOTH simultaneously computes `(NEG_INFINITY -
    /// NEG_INFINITY).exp()` = `NaN.exp()` = `NaN` for every candidate, and the
    /// `r < cumsum` weighted-sample loop is always false against a NaN cumsum,
    /// so it falls through to the *last* sorted candidate (token_id 3, the
    /// highest id, per the code's own comment) instead of this test's expected
    /// token_id 1 — the assertion below then fails with `left: 3, right: 1`.
    #[test]
    fn test_sample_top_p_all_neg_inf_falls_back_to_first_sorted_candidate() {
        let mut cs = CandidateSet {
            candidates: vec![
                Candidate {
                    token_id: 3,
                    logit: f32::NEG_INFINITY,
                },
                Candidate {
                    token_id: 1,
                    logit: f32::NEG_INFINITY,
                },
                Candidate {
                    token_id: 2,
                    logit: f32::NEG_INFINITY,
                },
            ],
        };
        let mut scratch = Vec::new();
        // top_p < 1.0 so the nucleus-truncation branch is in play too, not just
        // the full-distribution case.
        let tok = cs.sample_top_p_with_scratch(0.9, 0.5, &mut scratch);
        // All logits tie at NEG_INFINITY, so `candidate_order`'s tie-break
        // (ascending token_id) sorts token_id 1 to `candidates[0]` — the
        // non-finite-max guard's defined fallback answer.
        assert_eq!(
            tok, 1,
            "an all-NEG_INFINITY CandidateSet must deterministically fall back \
             to the first sorted candidate's token id, not NaN-derived garbage"
        );
    }

    #[test]
    fn test_penalized_logit_invalid_penalty_is_noop() {
        // Non-positive or non-finite penalties must not flip a logit's sign.
        for &bad in &[0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                penalized_logit(5.0, bad),
                5.0,
                "positive logit unchanged for penalty={bad}"
            );
            assert_eq!(
                penalized_logit(-5.0, bad),
                -5.0,
                "negative logit unchanged for penalty={bad}"
            );
        }
        // Valid penalty still applies as before.
        assert!(
            (penalized_logit(2.0, 2.0) - 1.0).abs() < 1e-6,
            "2.0/2.0 == 1.0"
        );
        assert!(
            (penalized_logit(-2.0, 2.0) - -4.0).abs() < 1e-6,
            "-2.0*2.0 == -4.0"
        );
    }

    #[test]
    fn test_repetition_penalty_applied_once_per_token() {
        // `recent_tokens` accumulates duplicates (`push_token` never dedups), so a
        // token repeated within the window must be penalized exactly ONCE — matching
        // HF's gather-once semantics and the Metal `CandidateSet` path. Applying the
        // penalty per occurrence compounds it (penalty^N) and silently over-suppresses
        // common tokens as a sequence grows.
        let mut s = Sampler::new(SamplingConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 2.0,
        })
        .with_seed(1);
        // Step 1: token 1 is the clear argmax; recent_tokens -> [1].
        assert_eq!(s.sample(&[0.0, 10.0, 0.0]), 1);
        // Step 2: token 1 penalized once (10/2 = 5) still wins; recent_tokens -> [1, 1].
        assert_eq!(s.sample(&[0.0, 10.0, 0.0]), 1);
        // Step 3: token 1 penalized ONCE (10/2 = 5) must beat token 2 (logit 3).
        // The per-occurrence bug penalizes twice (10/2/2 = 2.5) and wrongly picks token 2.
        assert_eq!(
            s.sample(&[0.0, 10.0, 3.0]),
            1,
            "duplicate history id must be penalized once, not per occurrence"
        );
    }

    // ── canonical RNG primitive tests ────────────────────────────────────────

    #[test]
    fn uniform_f32_from_u64_is_always_in_unit_interval() {
        // The canonical 24-bit conversion must be in [0, 1) for all u64 values —
        // in particular u64::MAX must NOT round up to exactly 1.0 (the latent bug
        // in the 53-bit `(s >> 11) as f32 / 2^53` path).
        for x in [
            0u64,
            1,
            u64::MAX,
            0xFFFF_FFFF_FFFF_FFFF,
            1u64 << 40,
            (1u64 << 40) - 1,
            0x8000_0000_0000_0000,
        ] {
            let f = uniform_f32_from_u64(x);
            assert!(
                (0.0..1.0).contains(&f),
                "uniform_f32_from_u64({x:#018x}) = {f} is not in [0, 1)"
            );
        }
    }

    // ── #387 repetition-penalty history contract tests ───────────────────────

    /// First-token prompt penalty: seeded prompt tokens must be penalized on the
    /// very first `sample` call.
    ///
    /// Mutation-sensitive: reverting `seed_history` (or not calling it in
    /// `generate.rs`) leaves `recent_tokens` empty, so token 1 is NOT penalized
    /// and wins with logit 10.0, failing this assertion.
    #[test]
    fn test_seed_history_penalizes_prompt_token_on_first_sample() {
        let config = SamplingConfig {
            temperature: 0.0, // greedy
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 2.0,
        };
        let mut sampler = Sampler::new(config).with_seed(1);
        // Seed token 1 as a prompt token.
        sampler.seed_history(&[1u32]);
        // logits: token 1 = 10.0 wins raw, but 10.0/2.0 = 5.0 < token 0's 6.0 after penalty.
        let token = sampler.sample(&[6.0, 10.0]);
        assert_eq!(
            token, 0,
            "token 1 was seeded in history; penalty 2.0 reduces its adjusted logit \
             below token 0; without seed_history, token 1 wins (mutation: omit seed_history)"
        );
    }

    /// Beyond-64-token history: a token at position 0 in a 65-entry history must
    /// still be penalized after the old 64-entry cap is removed.
    ///
    /// Mutation-sensitive: re-introducing the `max_recent` truncation evicts token 1
    /// from the window, leaving it unpenalized, so token 1 wins with logit 10.0
    /// instead of token 0 with logit 6.0.
    #[test]
    fn test_uncapped_history_penalizes_tokens_beyond_64() {
        let config = SamplingConfig {
            temperature: 0.0, // greedy
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 2.0,
        };
        let mut sampler = Sampler::new(config).with_seed(1);
        // Build a 65-token history THROUGH push_token, the path the removed 64-cap
        // actually lived on. token 1 is the oldest entry (pushed first). Filler tokens
        // use ids >= 2 so they do not compete with tokens 0 or 1.
        sampler.push_token(1);
        for t in 2u32..66 {
            sampler.push_token(t); // 64 filler tokens → total history length = 65
        }
        // Without cap: token 1 penalty applies → 10.0/2.0 = 5.0 < 6.0 → token 0 wins.
        // With old 64-cap restored in push_token: token 1 was evicted from the window →
        // raw_best (token 1) is no longer in history so the greedy shortcut returns it
        // un-penalized → token 1 wins.
        let token = sampler.sample(&[6.0, 10.0]);
        assert_eq!(
            token, 0,
            "token 1 at position 0 in a 65-entry history must still be penalized; \
             the old 64-cap silently dropped it (mutation: restore max_recent truncation)"
        );
    }

    /// Penalize-once / no compounding: a token repeated many times in history is
    /// penalized exactly once, not penalty^N times.
    ///
    /// Mutation-sensitive: reverting the HashSet dedup to per-occurrence penalty
    /// compounds to penalty^4 = 16.0, reducing the adjusted logit from 5.0 to 0.625,
    /// so token 0 wins instead of token 1.
    #[test]
    fn test_penalty_applied_exactly_once_for_repeated_history_token() {
        let config = SamplingConfig {
            temperature: 0.0, // greedy
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 2.0,
        };
        let mut sampler = Sampler::new(config).with_seed(1);
        // Token 1 repeated 4 times in history.
        sampler.seed_history(&[1, 1, 1, 1]);
        // Once penalized: 10.0/2.0 = 5.0 > 4.9 → token 1 wins (correct).
        // Per-occurrence (penalty^4): 10.0/16.0 = 0.625 < 4.9 → token 0 wins (wrong).
        let token = sampler.sample(&[4.9, 10.0]);
        assert_eq!(
            token, 1,
            "token 1 repeated 4× must be penalized once (5.0 > 4.9, token 1 wins); \
             per-occurrence compounding yields 0.625 and wrongly selects token 0 \
             (mutation: replace HashSet dedup with per-occurrence penalty)"
        );
    }

    // --- logprobs (#585) ----------------------------------------------------

    /// Hand-computed softmax reference for `logits = [1.0, 2.0, 3.0]`,
    /// temperature 1.0: `p = softmax(logits)`, `ln(p) = logits - ln(sum(exp(logits)))`.
    /// `sum(exp) = e^1 + e^2 + e^3 = 2.718282 + 7.389056 + 20.085537 = 30.192875`,
    /// `ln(sum) = 3.407606`. So `ln(p[i]) = logits[i] - 3.407606`.
    fn reference_ln_softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
        let log_sum = sum.ln();
        logits.iter().map(|&v| (v - max) - log_sum).collect()
    }

    #[test]
    fn test_compute_step_logprobs_matches_hand_computed_softmax() {
        let logits = [1.0f32, 2.0, 3.0];
        let reference = reference_ln_softmax(&logits);

        let (logprob, top) = compute_step_logprobs(&logits, 2, 1.0, 0);
        assert!(
            (logprob - reference[2]).abs() < 1e-4,
            "token 2 logprob {logprob} should match reference {}",
            reference[2]
        );
        assert!(top.is_empty(), "top_n=0 must return no alternatives");

        for (idx, &want) in reference.iter().enumerate() {
            let (lp, _) = compute_step_logprobs(&logits, idx as u32, 1.0, 0);
            assert!(
                (lp - want).abs() < 1e-4,
                "token {idx} logprob {lp} should match reference {want}"
            );
        }
    }

    #[test]
    fn test_compute_step_logprobs_top_n_sorted_descending_by_probability() {
        let logits = [1.0f32, 2.0, 3.0];
        let reference = reference_ln_softmax(&logits);

        let (_, top) = compute_step_logprobs(&logits, 2, 1.0, 2);
        assert_eq!(top.len(), 2, "top_logprobs=2 must return exactly 2 entries");
        // Descending logit == descending probability at fixed temperature.
        assert_eq!(top[0].token_id, 2, "highest-logit token must be first");
        assert_eq!(
            top[1].token_id, 1,
            "second-highest-logit token must be second"
        );
        assert!((top[0].logprob - reference[2]).abs() < 1e-4);
        assert!((top[1].logprob - reference[1]).abs() < 1e-4);
        assert!(
            top[0].logprob > top[1].logprob,
            "entries must be sorted descending by logprob"
        );
    }

    #[test]
    fn test_compute_step_logprobs_top_n_clamped_to_vocab_size() {
        let logits = [1.0f32, 2.0, 3.0];
        // Requesting more alternatives than the vocabulary must not panic or
        // fabricate entries -- it returns at most `vocab_size` of them.
        let (_, top) = compute_step_logprobs(&logits, 0, 1.0, 20);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_compute_step_logprobs_degenerate_temperature_reports_unscaled_softmax() {
        // temperature <= 0 is degenerate (see `temperature_degenerate`): the
        // *sampler* falls back to greedy argmax, but the reporting distribution
        // here must fall back to an UNSCALED (temperature = 1.0) softmax, not a
        // one-hot `t -> 0+` distribution -- argmax is invariant to positive
        // temperature scaling, so reporting log(1.0) = 0.0 / -inf would erase
        // the model's actual confidence rather than describe it.
        let logits = [1.0f32, 2.0, 3.0];
        let reference = reference_ln_softmax(&logits);
        let (logprob, top) = compute_step_logprobs(&logits, 2, 0.0, 1);
        assert!(
            (logprob - reference[2]).abs() < 1e-4,
            "degenerate temperature must report the T=1.0 softmax logprob \
             ({}), not one-hot 0.0; mutation would collapse this to 0.0",
            reference[2]
        );
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].token_id, 2);
    }

    #[test]
    fn test_compute_step_logprobs_all_nonfinite_logits_falls_back_to_sentinel() {
        let logits = [f32::NAN, f32::NAN, f32::NAN];
        let (logprob, top) = compute_step_logprobs(&logits, 1, 1.0, 3);
        assert_eq!(logprob, LOGPROB_NEG_SENTINEL);
        assert_eq!(
            top,
            vec![TopLogprob {
                token_id: 1,
                logprob: LOGPROB_NEG_SENTINEL
            }],
            "the requested token_id must still be reported (as the sentinel), \
             not dropped or replaced by an arbitrary index"
        );
    }

    #[test]
    fn test_compute_step_logprobs_out_of_vocab_token_id_falls_back_to_sentinel() {
        let logits = [1.0f32, 2.0, 3.0];
        let reference = reference_ln_softmax(&logits);
        // token_id beyond the logits slice: the requested token's logprob is
        // the sentinel, but the top-N alternatives are still computed
        // correctly over the real vocabulary (unaffected by the bad id).
        let (logprob, top) = compute_step_logprobs(&logits, 99, 1.0, 1);
        assert_eq!(logprob, LOGPROB_NEG_SENTINEL);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].token_id, 2);
        assert!((top[0].logprob - reference[2]).abs() < 1e-4);
    }

    // `record_logprob`'s noop/appends behavior moved to
    // `model::qwen35::generation`'s `DecodePolicy::record_logprob` tests
    // (PR #787) -- see
    // `decode_policy_record_logprob_noop_when_not_requested` and
    // `transition_records_one_logprob_per_generated_token` there. The
    // freestanding `pub(crate) record_logprob` this module used to export no
    // longer exists: only `compute_step_logprobs` (pure computation) remains
    // shared, and mutating `token_logprobs` is exclusively `DecodePolicy`'s.

    // -------------------------------------------------------------------
    // CPU-side microbench (issue #650): before/after `sample_full_logits`.
    //
    // `#[ignore]`d so normal `cargo test` runs stay fast; run explicitly with:
    //   cargo test -p lattice-inference --lib --release -- --ignored --nocapture \
    //     sampling::tests::microbench_sample_full_logits_default_issue_config
    // -------------------------------------------------------------------

    /// Literal copy of the pre-optimization `sample_token` algorithm (full
    /// vocab clone, fresh per-call `HashSet`, full `Vec<usize>` index
    /// allocation, allocating softmax-probs `Vec`) — the "before" arm of the
    /// microbench. Mirrors what both `model/qwen35/sampling.rs::sample_token`
    /// and the Metal CPU fallback did before this change; kept local to the
    /// microbench so it has no effect on any production or library code path.
    fn old_full_vocab_sample(
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        previous_ids: &[u32],
        rng_state: &mut u64,
    ) -> u32 {
        let vocab_size = logits.len();
        let mut adjusted = logits.to_vec();

        if repetition_penalty != 1.0 {
            let mut seen = std::collections::HashSet::with_capacity(previous_ids.len());
            for &id in previous_ids {
                let idx = id as usize;
                if idx < vocab_size && seen.insert(id) {
                    adjusted[idx] = penalized_logit(adjusted[idx], repetition_penalty);
                }
            }
        }

        if temperature_degenerate(temperature) {
            return argmax_f32(&adjusted);
        }

        if temperature != 1.0 {
            let inv_temp = 1.0 / temperature;
            for v in &mut adjusted {
                *v *= inv_temp;
            }
        }

        let mut indices: Vec<usize> = (0..vocab_size).collect();
        if top_k > 0 && top_k < vocab_size {
            indices.select_nth_unstable_by(top_k - 1, |&a, &b| {
                adjusted[b]
                    .partial_cmp(&adjusted[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(&b))
            });
            indices.truncate(top_k);
        }
        indices.sort_unstable_by(|&a, &b| {
            adjusted[b]
                .partial_cmp(&adjusted[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });

        let max_logit = indices
            .iter()
            .map(|&i| adjusted[i])
            .fold(f32::NEG_INFINITY, f32::max);
        if !max_logit.is_finite() {
            return argmax_f32(&adjusted);
        }
        let mut probs: Vec<(usize, f32)> = indices
            .iter()
            .map(|&i| (i, (adjusted[i] - max_logit).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        if !sum.is_finite() || sum <= 0.0 {
            return argmax_f32(&adjusted);
        }
        for (_, p) in &mut probs {
            *p /= sum;
        }

        if top_p < 1.0 {
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

        let r = uniform_f32_from_u64(xorshift64_next(rng_state));
        let mut cumsum = 0.0f32;
        for &(idx, p) in &probs {
            cumsum += p;
            if r < cumsum {
                return idx as u32;
            }
        }
        probs.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
    }

    #[test]
    #[ignore = "perf microbench, not a correctness check; run explicitly with --ignored"]
    fn microbench_sample_full_logits_default_issue_config() {
        const VOCAB_SIZE: usize = 248_320; // Qwen3.5 real vocab (issue #650).
        const ITERS: usize = 300;

        let logits: Vec<f32> = (0..VOCAB_SIZE as u64)
            .map(|i| {
                let h = i
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (h as f32 / u64::MAX as f32) * 20.0 - 10.0
            })
            .collect();
        // A 64-token recent-history window, as a real decode loop would pass
        // (previous_ids = prompt + generated-so-far), with duplicates.
        let previous_ids: Vec<u32> = (0..64u32).map(|i| (i * 4099) % VOCAB_SIZE as u32).collect();

        let cfg = SamplingConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
        };
        let gen_cfg = GenerateConfig {
            temperature: cfg.temperature,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
            repetition_penalty: cfg.repetition_penalty,
            ..Default::default()
        };

        let mut rng_before = 0xDEAD_BEEFu64;
        let before_start = std::time::Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(old_full_vocab_sample(
                &logits,
                cfg.temperature,
                cfg.top_k,
                cfg.top_p,
                cfg.repetition_penalty,
                &previous_ids,
                &mut rng_before,
            ));
        }
        let before_elapsed = before_start.elapsed();

        let mut rng_after = 0xDEAD_BEEFu64;
        let after_start = std::time::Instant::now();
        for _ in 0..ITERS {
            std::hint::black_box(sample_full_logits(
                &logits,
                &gen_cfg,
                &previous_ids,
                &mut rng_after,
            ));
        }
        let after_elapsed = after_start.elapsed();

        let before_us_per_call = before_elapsed.as_secs_f64() * 1e6 / ITERS as f64;
        let after_us_per_call = after_elapsed.as_secs_f64() * 1e6 / ITERS as f64;
        eprintln!(
            "microbench sample_full_logits @ vocab={VOCAB_SIZE}, iters={ITERS}, cfg=(temp=0.7,top_k=40,top_p=0.9,rep_penalty=1.1):\n\
             before (old_full_vocab_sample): {before_elapsed:?} total, {before_us_per_call:.1} us/call, {:.1} tok/s\n\
             after  (sample_full_logits):    {after_elapsed:?} total, {after_us_per_call:.1} us/call, {:.1} tok/s",
            1e6 / before_us_per_call,
            1e6 / after_us_per_call,
        );

        assert!(
            after_elapsed < before_elapsed,
            "optimized sample_full_logits ({after_elapsed:?}) must be faster than \
             the old full-vocab-allocating algorithm ({before_elapsed:?}) at the \
             issue's default config and vocab size"
        );
    }
}
