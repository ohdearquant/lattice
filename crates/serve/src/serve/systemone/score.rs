//! Row scoring: turn a prepared request into the wire response, given a scorer
//! that reads one row's label logits.
//!
//! Each assembled row is scored on its own, as the reference does for its
//! default one-question-per-prompt layout: the scorer runs the row's ids from a
//! fresh state and returns the raw logits at the row's last position over the
//! first `nopts` entries of the label table. This layer divides those logits by
//! the checkpoint's published temperature and takes the softmax over the row's
//! own options, which equals the reference's softmax over the whole label table
//! with every entry past `nopts` masked out. The probabilities then go through
//! [`assemble`] unchanged.
//!
//! The scorer is a parameter so that the same code serves the CPU and Metal
//! backends and can be tested without a checkpoint.

use super::answer::assemble;
use super::{PreparedSystemOne, SystemOneError, SystemOneResponse};

/// A failure while scoring a request: either the request is refused (the
/// reference's 422 class) or the scorer itself failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScoreFailure<E> {
    /// The request or its rows are refused before or after scoring.
    #[error(transparent)]
    Refused(#[from] SystemOneError),
    /// The scorer returned an error for a row.
    #[error("scoring row {row} failed: {error}")]
    Scorer {
        /// The planned row the scorer failed on.
        row: usize,
        /// The scorer's error.
        error: E,
    },
}

/// The softmax of `logits / temperature`, computed in f64.
///
/// # Errors
///
/// Refuses an empty row, a temperature that is not a positive finite number,
/// a non-finite logit, and a logit that overflows when divided by the
/// temperature.
pub fn calibrated_probabilities(
    logits: &[f32],
    temperature: f64,
) -> Result<Vec<f64>, SystemOneError> {
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(SystemOneError::Refused(format!(
            "the checkpoint temperature must be a positive finite number, got {temperature}"
        )));
    }
    if logits.is_empty() {
        return Err(SystemOneError::Refused(
            "a row has no option logits".to_string(),
        ));
    }
    if let Some(index) = logits.iter().position(|logit| !logit.is_finite()) {
        return Err(SystemOneError::Refused(format!(
            "option logit {index} is not finite"
        )));
    }
    let scaled: Vec<f64> = logits
        .iter()
        .map(|&logit| f64::from(logit) / temperature)
        .collect();
    if scaled.iter().any(|value| !value.is_finite()) {
        return Err(SystemOneError::Refused(format!(
            "an option logit divided by the temperature {temperature} is not finite"
        )));
    }
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|value| (value - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    Ok(exps.into_iter().map(|value| value / sum).collect())
}

/// Score every row of a prepared request and build the response.
///
/// `score_row(ids, label_ids)` must return one raw logit per label id, in the
/// order given, read at the last position of `ids` after a prefill from a
/// fresh state. The label ids are the first `nopts` entries of the table the
/// request was prepared against ([`PreparedSystemOne::label_ids`]).
///
/// # Errors
///
/// [`ScoreFailure::Refused`] when a row needs more labels than the table holds,
/// when the scorer returns the wrong number of logits, or when [`assemble`]
/// refuses; [`ScoreFailure::Scorer`] with the row index when the scorer fails.
pub fn score_prepared<E>(
    prepared: &PreparedSystemOne,
    mut score_row: impl FnMut(&[u32], &[u32]) -> Result<Vec<f32>, E>,
) -> Result<SystemOneResponse, ScoreFailure<E>> {
    let mut probs = Vec::with_capacity(prepared.prompt.rows.len());
    for (row_index, row) in prepared.prompt.rows.iter().enumerate() {
        let label_ids = prepared.label_ids.get(..row.nopts).ok_or_else(|| {
            SystemOneError::Refused(format!(
                "row {row_index} has {} options, the label table holds {}",
                row.nopts,
                prepared.label_ids.len()
            ))
        })?;
        let logits = score_row(&row.ids, label_ids).map_err(|error| ScoreFailure::Scorer {
            row: row_index,
            error,
        })?;
        if logits.len() != label_ids.len() {
            return Err(SystemOneError::Refused(format!(
                "the scorer returned {} logits for row {row_index}, which has {} options",
                logits.len(),
                label_ids.len()
            ))
            .into());
        }
        probs.push(calibrated_probabilities(
            &logits,
            prepared.config.temperature,
        )?);
    }
    let unique_tokens = prepared.prompt.unique_tokens();
    let assembled = assemble(&prepared.rendered, &prepared.index, &probs, unique_tokens)?;
    Ok(SystemOneResponse {
        model: SystemOneResponse::keyword_for(&prepared.config),
        answers: assembled.answers,
        input_tokens: assembled.unique_tokens,
        output_tokens: 0,
    })
}

#[cfg(test)]
mod tests;
