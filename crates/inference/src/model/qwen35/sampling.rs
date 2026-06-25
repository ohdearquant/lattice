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

    let mut probs = build_softmax_probs(&adjusted, &indices);

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
            if adjusted[idx] > 0.0 {
                adjusted[idx] /= penalty;
            } else {
                adjusted[idx] *= penalty;
            }
        }
    }
}

fn greedy_token(adjusted: &[f32]) -> u32 {
    adjusted
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn build_softmax_probs(adjusted: &[f32], indices: &[usize]) -> Vec<(usize, f32)> {
    let max_logit = indices
        .iter()
        .map(|&i| adjusted[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = indices
        .iter()
        .map(|&i| (i, (adjusted[i] - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in &mut probs {
        *p /= sum;
    }
    probs
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
    let r = xorshift64(rng_state);
    let mut cumsum = 0.0f32;
    for &(idx, p) in probs {
        cumsum += p;
        if r <= cumsum {
            return idx as u32;
        }
    }
    probs[0].0 as u32
}

/// Xorshift64 PRNG. Returns a value in [0, 1).
pub(crate) fn xorshift64(state: &mut u64) -> f32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    (s >> 11) as f32 / (1u64 << 53) as f32
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
}
