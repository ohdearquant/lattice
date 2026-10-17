//! Query-likelihood reranking over the Qwen3.5 causal language-model scoring path.
//!
//! Scores each candidate by `P(query | candidate)`: condition the model on the
//! candidate text and ask how likely it is to continue with the query text. A
//! candidate the model finds highly predictive of the query (low query-token
//! NLL) ranks above one it finds surprising. This is the classic query-likelihood
//! retrieval model, built entirely from [`Qwen35Model::compute_token_nlls`] — no
//! new kernels. Scoring runs through whatever LoRA hook is currently attached to
//! the model (the default no-op hook, or one set via [`Qwen35Model::set_lora`]),
//! since [`Qwen35Model::compute_token_nlls`] drives the same `forward_step` every
//! other entry point uses.

use super::model::Qwen35Model;
use crate::error::InferenceError;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer;
use std::cmp::Ordering;

impl Qwen35Model {
    /// **Unstable**: rerank `candidates` by query-likelihood `P(query | candidate)`.
    ///
    /// For each candidate, tokenizes `[candidate_tokens, query_tokens]` and asks
    /// [`Self::compute_token_nlls`] to score the concatenation. The mean
    /// negative-log-likelihood over the query-token positions (with the
    /// candidate as conditioning context) becomes the candidate's relevance
    /// score, negated so a higher score means a more relevant candidate.
    ///
    /// Returns `(original_index, relevance_score)` pairs sorted by score
    /// descending; ties keep the original candidate order. Deterministic: no
    /// sampling anywhere on this path, and each candidate gets a fresh KV
    /// cache / GDN state (see [`Self::compute_token_nlls`]).
    ///
    /// Scores are comparable *only within a single call*. Each score is a mean
    /// per-token NLL (negated), so its magnitude depends on the query's length
    /// and token distribution; the same absolute value means different things
    /// for different queries. Rank order within one call is the only stable
    /// signal — callers must not threshold or compare absolute scores across
    /// calls.
    ///
    /// `candidates` empty returns `Ok(vec![])`. If a candidate's tokens would
    /// push `[candidate, query]` past [`Self::max_context`], the candidate is
    /// left-truncated (earliest tokens dropped, keeping the tokens nearest the
    /// query) so the full query always survives intact. Returns a typed
    /// [`InferenceError::InvalidInput`], never a silent zero score, when the
    /// query alone leaves no room for any candidate context, or when the query
    /// or a (possibly truncated) candidate tokenizes to nothing.
    pub fn rerank(
        &self,
        query: &str,
        candidates: &[&str],
    ) -> Result<Vec<(usize, f32)>, InferenceError> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // `Tokenizer::tokenize` pads to the tokenizer's configured `max_seq_len`
        // (4096 by default) — or, for input longer than that, truncates while
        // still reporting the pre-truncation `real_length`, which would then
        // index past the truncated `input_ids`. Bumping `max_seq_len` to cover
        // the longest string this call will tokenize keeps every `real_length`
        // accurate. One bump (one vocab clone) up front, reused below, rather
        // than one per candidate.
        let longest = candidates
            .iter()
            .map(|c| c.len())
            .max()
            .unwrap_or(0)
            .max(query.len());
        let tokenizer = self.tokenizer.with_max_seq_len(longest.saturating_add(64));

        let query_tokens = tokenize_unpadded(&tokenizer, query);
        if query_tokens.is_empty() {
            return Err(InferenceError::InvalidInput(
                "rerank: query tokenizes to zero tokens".into(),
            ));
        }

        let max_context = self.max_context();
        if query_tokens.len() >= max_context {
            return Err(InferenceError::InvalidInput(format!(
                "rerank: query ({} tokens) alone leaves no room for candidate \
                 context within the model context window ({max_context})",
                query_tokens.len()
            )));
        }
        let candidate_budget = max_context - query_tokens.len();

        let mut scored = Vec::with_capacity(candidates.len());
        for (idx, candidate) in candidates.iter().enumerate() {
            let mut candidate_tokens = tokenize_unpadded(&tokenizer, candidate);
            if candidate_tokens.len() > candidate_budget {
                // Left-truncate: drop the earliest candidate tokens, keeping the
                // tail (nearest the query) so the full query is never touched.
                let drop = candidate_tokens.len() - candidate_budget;
                candidate_tokens.drain(..drop);
            }
            if candidate_tokens.is_empty() {
                return Err(InferenceError::InvalidInput(format!(
                    "rerank: candidate {idx} tokenizes to zero tokens; query-likelihood \
                     scoring needs at least one candidate token as conditioning context"
                )));
            }

            let candidate_len = candidate_tokens.len();
            let mut sequence = candidate_tokens;
            sequence.extend_from_slice(&query_tokens);

            let nlls = self.compute_token_nlls(&sequence)?;
            // `nlls[i]` scores `sequence[i + 1]`. The query occupies sequence
            // positions `[candidate_len, candidate_len + query_tokens.len())`, so
            // its NLLs live at `nlls[candidate_len - 1 ..= candidate_len + query_tokens.len() - 2]`.
            let query_nlls = &nlls[candidate_len - 1..candidate_len - 1 + query_tokens.len()];
            let mean_nll: f64 =
                query_nlls.iter().map(|&x| x as f64).sum::<f64>() / query_tokens.len() as f64;
            scored.push((idx, -(mean_nll as f32)));
        }

        scored.sort_by(rerank_order);

        Ok(scored)
    }
}

/// Tokenize `text` and return its unpadded token IDs. `tokenizer` must already
/// be bumped to a `max_seq_len` at or above `text`'s real token count (see
/// [`Qwen35Model::rerank`]), or the returned IDs would be silently truncated.
fn tokenize_unpadded(tokenizer: &BpeTokenizer, text: &str) -> Vec<u32> {
    let input = tokenizer.tokenize(text);
    input.input_ids[..input.real_length].to_vec()
}

/// Sort order for `rerank`'s `(original_index, relevance_score)` pairs:
/// descending by score, ties broken by ascending original index. Mirrors
/// `sampling::candidate_order`'s NaN handling — a NaN score is treated as the
/// worst possible relevance regardless of sign, and the NaN/NaN case falls
/// back to the index tie-break, which keeps this a total order (`sort_by`
/// requires one; comparing two NaNs as merely "equal-ish" is not antisymmetric
/// and can sort nondeterministically with a NaN-heavy input).
fn rerank_order(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => a.0.cmp(&b.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        }
    }
}

#[cfg(test)]
mod tests;
