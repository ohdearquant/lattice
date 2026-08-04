//! Pooled hidden-state embeddings from the loaded generative model.
//!
//! This is the stable, inference-side counterpart to the `train-backward`
//! capture path. It exists so a consumer that embeds text with the *same* model
//! it generates with does not have to enable a feature named for training, nor
//! decide for itself how to collapse a sequence of hidden states into one
//! vector.
//!
//! These are decoder hidden states, not the output of a dedicated embedding
//! model. For general-purpose retrieval a purpose-trained embedding model
//! (`lattice-embed`) will normally score better on the same corpus. Reach for
//! this when the embedding must come from the loaded generative model.

use super::model::Qwen35Model;
use crate::error::InferenceError;

/// How a sequence of per-token hidden states is collapsed into one vector.
///
/// The choice is deliberately part of this API rather than left to each caller,
/// because the two options are not interchangeable and the wrong default is
/// quietly lossy rather than loudly broken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum HiddenPooling {
    /// Take the final position's hidden state. The default, and the right one
    /// for a causal decoder: attention is one-directional, so only the last
    /// position has attended to the whole sequence. This matches the convention
    /// Qwen's own embedding models use.
    #[default]
    LastToken,
    /// Average across all positions. Offered because it is a common request and
    /// occasionally the better choice on short inputs, but note that in a causal
    /// model early positions have seen almost none of the text, so this dilutes
    /// the only fully-informed position with many partly-informed ones.
    Mean,
}

impl Qwen35Model {
    /// Embed `tokens` as a single `[hidden_size]` vector by pooling the model's
    /// final hidden states.
    ///
    /// The returned vector is **not** L2-normalized, matching the convention at
    /// this layer (HuggingFace's `AutoModel` plus a pooling step behaves the
    /// same way). Callers computing cosine similarity should normalize, or use
    /// a cosine routine that normalizes internally.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::Inference`] if `tokens` is empty, longer than
    /// the model's RoPE capacity, or contains an id at or above `vocab_size`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lattice_inference::model::qwen35::{HiddenPooling, Qwen35Model};
    /// # fn demo(model: &Qwen35Model, tokens: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
    /// let v = model.embed_tokens(tokens, HiddenPooling::default())?;
    /// assert_eq!(v.len(), model.hidden_size());
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed_tokens(
        &self,
        tokens: &[u32],
        pooling: HiddenPooling,
    ) -> Result<Vec<f32>, InferenceError> {
        let hiddens = self.final_hidden_states(tokens)?;

        // `final_hidden_states` rejects an empty input, so a caller cannot reach
        // the pooling below with nothing to pool. Checked rather than assumed:
        // an empty mean would silently produce a zero vector, and a zero vector
        // has undefined cosine rather than low cosine, so it would read as an
        // unrelated document instead of as a failure.
        if hiddens.is_empty() {
            return Err(InferenceError::Inference(
                "embed_tokens: no hidden states were produced".to_string(),
            ));
        }

        Ok(pool_hidden(&hiddens, pooling))
    }

    /// The model's hidden size, which is the length of every vector
    /// [`Qwen35Model::embed_tokens`] returns.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}

/// Collapse per-position hidden states into one vector.
///
/// Split out from [`Qwen35Model::embed_tokens`] so it can be tested on input
/// that actually discriminates between the modes. The tiny zero-weight test
/// model produces all-zero hidden states, where last-token and mean pooling
/// agree trivially, so a test driven through the model would pass without
/// exercising either branch.
///
/// Caller must pass a non-empty slice; `embed_tokens` guarantees that.
fn pool_hidden(hiddens: &[Vec<f32>], pooling: HiddenPooling) -> Vec<f32> {
    debug_assert!(
        !hiddens.is_empty(),
        "pool_hidden requires a non-empty slice"
    );
    let dim = hiddens[0].len();
    match pooling {
        HiddenPooling::LastToken => hiddens[hiddens.len() - 1].clone(),
        HiddenPooling::Mean => {
            let n = hiddens.len() as f32;
            let mut acc = vec![0.0_f32; dim];
            for h in hiddens {
                for (a, &x) in acc.iter_mut().zip(h.iter()) {
                    *a += x;
                }
            }
            for a in &mut acc {
                *a /= n;
            }
            acc
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen35::test_support::tiny_zero_model;

    /// Input chosen so the two modes CANNOT agree: last row is [9, 9], mean is
    /// [4, 5]. A pooling bug that returned the wrong branch, or averaged the
    /// wrong axis, moves these numbers.
    fn discriminating_input() -> Vec<Vec<f32>> {
        vec![vec![1.0, 3.0], vec![2.0, 3.0], vec![9.0, 9.0]]
    }

    #[test]
    fn last_token_pooling_takes_the_final_position() {
        let h = discriminating_input();
        assert_eq!(pool_hidden(&h, HiddenPooling::LastToken), vec![9.0, 9.0]);
    }

    #[test]
    fn mean_pooling_averages_across_positions() {
        let h = discriminating_input();
        assert_eq!(pool_hidden(&h, HiddenPooling::Mean), vec![4.0, 5.0]);
    }

    #[test]
    fn the_two_modes_disagree_on_this_input() {
        // Guards the guard: if this ever holds, the two tests above stop
        // distinguishing the branches and would pass with either implementation.
        let h = discriminating_input();
        assert_ne!(
            pool_hidden(&h, HiddenPooling::LastToken),
            pool_hidden(&h, HiddenPooling::Mean)
        );
    }

    #[test]
    fn single_position_makes_both_modes_agree() {
        let h = vec![vec![7.0, -2.0]];
        assert_eq!(
            pool_hidden(&h, HiddenPooling::LastToken),
            pool_hidden(&h, HiddenPooling::Mean)
        );
    }

    #[test]
    fn default_pooling_is_last_token() {
        assert_eq!(HiddenPooling::default(), HiddenPooling::LastToken);
    }

    #[test]
    fn empty_input_is_rejected_rather_than_pooled_to_zeros() {
        let model = tiny_zero_model();
        let err = model
            .embed_tokens(&[], HiddenPooling::default())
            .expect_err("empty token slice must not produce a vector");
        assert!(
            err.to_string().contains("at least 1 token"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn out_of_vocab_token_is_rejected() {
        let model = tiny_zero_model();
        let bad = model.config.vocab_size as u32;
        let err = model
            .embed_tokens(&[bad], HiddenPooling::default())
            .expect_err("out-of-vocab id must be rejected");
        assert!(
            err.to_string().contains("vocab_size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn embedding_length_is_hidden_size() {
        let model = tiny_zero_model();
        let v = model
            .embed_tokens(&[0, 1], HiddenPooling::default())
            .expect("tiny model should embed two in-vocab tokens");
        assert_eq!(v.len(), model.hidden_size());
    }
}
