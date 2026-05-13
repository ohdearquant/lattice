//! Token sampling strategies for text generation.
//!
//! Supports temperature scaling, top-k filtering, top-p (nucleus) sampling,
//! and repetition penalty.

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
                if c.logit > 0.0 {
                    c.logit /= penalty;
                } else {
                    c.logit *= penalty;
                }
            }
        }
    }

    /// Return the token_id of the candidate with the highest logit.
    pub fn argmax(&self) -> u32 {
        let mut best_id = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for c in &self.candidates {
            if c.logit > best_val {
                best_val = c.logit;
                best_id = c.token_id;
            }
        }
        best_id
    }

    /// Scale logits by `1 / temperature`.  No-op when temperature == 1.0.
    pub fn apply_temperature(&mut self, temperature: f32) {
        if temperature <= 0.0 || temperature == 1.0 {
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

        // Sort descending for deterministic top-p traversal.
        self.candidates.sort_by(candidate_order);

        // Softmax — reuse the provided scratch buffer.
        let max_logit = self.candidates[0].logit;
        probs.clear();
        probs.extend(self.candidates.iter().map(|c| (c.logit - max_logit).exp()));
        let sum: f32 = probs.iter().sum();
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
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// **Unstable**: stateful sampler; implementation and state fields may change.
pub struct Sampler {
    config: SamplingConfig,
    rng: Rng,
    /// Recently generated token IDs (for repetition penalty).
    recent_tokens: Vec<u32>,
    /// Max tokens to track for repetition penalty.
    max_recent: usize,
    /// Reused candidate buffer; avoids 1.9 MB alloc per call at vocab=248,320.
    candidate_scratch: Vec<Candidate>,
    /// Reused probability buffer for top-p softmax.
    prob_scratch: Vec<f32>,
    /// Reused f32 scratch for adjusted logits; avoids 993 KB alloc per call.
    logit_scratch: Vec<f32>,
}

impl Sampler {
    /// **Unstable**: construct sampler.
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            config,
            rng: Rng::new(42),
            recent_tokens: Vec::new(),
            max_recent: 64,
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

    /// **Unstable**: sample a token ID; sampling strategy details may change.
    ///
    /// `logits` is the raw output from the model's final linear layer,
    /// shape `[vocab_size]`.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        // Greedy fast path: skip the full-vocab clone when argmax is not penalized.
        // Repetition penalty only lowers penalized tokens, so argmax(raw) == argmax(penalized)
        // whenever the raw argmax token is not in recent_tokens. In the common case
        // (recent_tokens empty or raw argmax not recently generated) this avoids the
        // 993 KB extend_from_slice entirely.
        if self.config.temperature <= 0.0 || self.config.top_k == 1 {
            let raw_best = argmax_f32(logits);
            if self.config.repetition_penalty == 1.0 || !self.recent_tokens.contains(&raw_best) {
                self.push_token(raw_best);
                return raw_best;
            }
            // Rare: raw argmax is a recently penalized token — clone + re-scan.
            self.logit_scratch.clear();
            self.logit_scratch.extend_from_slice(logits);
            let penalty = self.config.repetition_penalty;
            for &tok in &self.recent_tokens {
                let idx = tok as usize;
                if idx < self.logit_scratch.len() {
                    if self.logit_scratch[idx] > 0.0 {
                        self.logit_scratch[idx] /= penalty;
                    } else {
                        self.logit_scratch[idx] *= penalty;
                    }
                }
            }
            let token = argmax_f32(&self.logit_scratch);
            self.push_token(token);
            return token;
        }

        // Non-greedy path: clone logits for in-place penalty + fused top-k.
        self.logit_scratch.clear();
        self.logit_scratch.extend_from_slice(logits);
        let adj = &mut self.logit_scratch;

        if self.config.repetition_penalty != 1.0 {
            for &tok in &self.recent_tokens {
                let idx = tok as usize;
                if idx < adj.len() {
                    if adj[idx] > 0.0 {
                        adj[idx] /= self.config.repetition_penalty;
                    } else {
                        adj[idx] *= self.config.repetition_penalty;
                    }
                }
            }
        }

        let inv_temp = if self.config.temperature != 1.0 {
            1.0 / self.config.temperature
        } else {
            1.0
        };

        // Streaming min-heap top-k with fused temperature scaling.
        // ~95% of vocab elements are skipped by the NEON threshold gate.
        select_top_k(
            adj,
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
        if self.recent_tokens.len() > self.max_recent {
            self.recent_tokens.remove(0);
        }
    }

    /// **Unstable**: reset sampler state for a new generation sequence.
    pub fn reset(&mut self) {
        self.recent_tokens.clear();
    }
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
    if k == 0 || logits.is_empty() {
        return;
    }
    let k = k.min(logits.len());

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
    if k == 0 || logits.is_empty() {
        return;
    }
    let k = k.min(logits.len());

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
#[inline(always)]
fn candidate_order(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.logit.is_nan(), b.logit.is_nan()) {
        (true, _) => Ordering::Greater,
        (_, true) => Ordering::Less,
        _ => b
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
}
