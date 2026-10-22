//! Option-letter scoring: one prefill, with the answer read from the next-token logits at the
//! option letters (ADR-097).
//!
//! A classification-style decision lists its options in the prompt with letters (`A`, `B`, ...)
//! and reads the model's distribution over those letters at the last prompt position. This module
//! holds the backend-independent pieces: the [`OptionScores`] result, the pure function that builds
//! it from a full-vocabulary logit row, and the helper that resolves letters to token ids.
//! The scoring entry points live on the models that own a forward pass:
//! [`crate::model::qwen35::Qwen35Model::score_option_letters`] on the CPU and
//! `MetalQwen35State::score_option_letters` on Metal.
//!
//! Scores are uncalibrated: `probs` is the plain softmax over the k letters, and the option order
//! is the caller's to control and to report.
//!
//! [`OptionScores`]: crate::option_scoring::OptionScores

use std::collections::HashSet;

use crate::error::InferenceError;
use crate::tokenizer::Tokenizer;

/// Scores for one option list, in the caller's letter order.
#[derive(Debug, Clone, PartialEq)]
pub struct OptionScores {
    /// The raw next-token logits at the k letter ids, in the order the ids were given.
    pub logits: Vec<f32>,
    /// Softmax over the k letter logits only. Sums to 1.
    pub probs: Vec<f32>,
    /// Probability mass the full-vocabulary softmax puts on the k letters, in `[0, 1]`. A low
    /// value means the model did not answer with a letter. It is reported only and never used to
    /// rescale `probs`.
    pub label_mass: f32,
}

/// **Unstable**: validate a letter-id list against a vocabulary of `vocab_size` tokens.
///
/// Refuses an empty list, an id at or beyond `vocab_size`, and a repeated id.
pub(crate) fn check_letter_ids(
    vocab_size: usize,
    letter_ids: &[u32],
) -> Result<(), InferenceError> {
    if letter_ids.is_empty() {
        return Err(InferenceError::InvalidInput(
            "option scoring: letter_ids must not be empty".into(),
        ));
    }
    let mut seen = HashSet::with_capacity(letter_ids.len());
    for (index, &id) in letter_ids.iter().enumerate() {
        if id as usize >= vocab_size {
            return Err(InferenceError::InvalidInput(format!(
                "option scoring: letter_ids[{index}]={id} out of range: vocab_size is {vocab_size}"
            )));
        }
        if !seen.insert(id) {
            return Err(InferenceError::InvalidInput(format!(
                "option scoring: letter_ids[{index}]={id} is a duplicate letter id"
            )));
        }
    }
    Ok(())
}

/// **Unstable**: build [`OptionScores`] from a full-vocabulary logit row and the letter ids.
///
/// `logits` is the last-position logit row, one entry per vocabulary token. `letter_ids` are the
/// k token ids of the options, in the caller's order.
///
/// # Errors
///
/// [`InferenceError::InvalidInput`] for an empty `letter_ids`, an id at or beyond `logits.len()`,
/// or a repeated id. [`InferenceError::Inference`] when any value in the row is not finite: every
/// entry of the row enters the full-vocabulary softmax behind `label_mass`, so one NaN or infinity
/// anywhere makes the scores meaningless. Nothing is clamped or guessed.
pub fn option_scores_from_logits(
    logits: &[f32],
    letter_ids: &[u32],
) -> Result<OptionScores, InferenceError> {
    check_letter_ids(logits.len(), letter_ids)?;
    if let Some((index, &value)) = logits.iter().enumerate().find(|&(_, v)| !v.is_finite()) {
        return Err(InferenceError::Inference(format!(
            "option scoring: logits[{index}]={value} is not finite"
        )));
    }

    let letter_logits: Vec<f32> = letter_ids.iter().map(|&id| logits[id as usize]).collect();

    // Softmax over the letters, shifted by the letter maximum so the largest term is exactly 1.
    let letter_max = letter_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let letter_terms: Vec<f64> = letter_logits
        .iter()
        .map(|&l| f64::from(l - letter_max).exp())
        .collect();
    let letter_sum: f64 = letter_terms.iter().sum();
    let probs: Vec<f32> = letter_terms
        .iter()
        .map(|&t| (t / letter_sum) as f32)
        .collect();

    // Full-vocabulary mass on the letters. The denominator is built as letter mass plus the mass
    // of every other token, so the ratio cannot exceed 1 through rounding.
    let full_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut is_letter = vec![false; logits.len()];
    for &id in letter_ids {
        is_letter[id as usize] = true;
    }
    let mut letter_mass = 0.0_f64;
    let mut other_mass = 0.0_f64;
    for (index, &l) in logits.iter().enumerate() {
        let term = f64::from(l - full_max).exp();
        if is_letter[index] {
            letter_mass += term;
        } else {
            other_mass += term;
        }
    }
    let label_mass = (letter_mass / (letter_mass + other_mass)) as f32;

    Ok(OptionScores {
        logits: letter_logits,
        probs,
        label_mass,
    })
}

/// **Unstable**: resolve the first `k` option letters (`A`, `B`, ...) to token ids.
///
/// Each letter is encoded on its own with no leading space, which is where a letter lands when it
/// starts a line. Returns the ids in letter order.
///
/// # Errors
///
/// [`InferenceError::InvalidInput`] when `k` is zero or beyond the 26-letter alphabet.
/// [`InferenceError::Tokenizer`] when a letter does not encode to exactly one token (none, several,
/// or a result the tokenizer truncated) or when two letters resolve to the same id. A letter that
/// is not one token cannot be read from a single next-token distribution, so nothing is guessed.
pub fn resolve_option_letters(
    tokenizer: &dyn Tokenizer,
    k: usize,
) -> Result<Vec<u32>, InferenceError> {
    const ALPHABET: usize = 26;
    if k == 0 || k > ALPHABET {
        return Err(InferenceError::InvalidInput(format!(
            "option scoring: k must be in 1..={ALPHABET}, got {k}"
        )));
    }

    let mut ids = Vec::with_capacity(k);
    for offset in 0..k {
        let letter = char::from(b'A' + offset as u8).to_string();
        let encoded = tokenizer.tokenize(&letter);
        if encoded.real_length != 1 || encoded.pre_truncation_len != 1 {
            return Err(InferenceError::Tokenizer(format!(
                "option scoring: letter {letter:?} encodes to {} tokens, expected exactly 1",
                encoded.pre_truncation_len.max(encoded.real_length)
            )));
        }
        let Some(&id) = encoded.input_ids.first() else {
            return Err(InferenceError::Tokenizer(format!(
                "option scoring: letter {letter:?} produced no token ids"
            )));
        };
        if let Some(previous) = ids.iter().position(|&earlier| earlier == id) {
            return Err(InferenceError::Tokenizer(format!(
                "option scoring: letters {:?} and {letter:?} resolve to the same token id {id}",
                char::from(b'A' + previous as u8)
            )));
        }
        ids.push(id);
    }
    Ok(ids)
}

/// **Unstable**: resolve the decider label table: `A`..`Z`, then the two-letter
/// strings `AA`, `AB`, ... that each encode to exactly one token, in that order,
/// until the table holds `max_entries` entries.
///
/// This extends [`resolve_option_letters`]'s rule to a wider table without
/// relaxing it: every entry must encode to exactly one token and no two entries
/// may share an id. The first [`LETTER_LABELS`] entries are required to be `A`..
/// `J` so the narrow layout is byte-identical to the single-letter rendering.
///
/// The `open` token ids (the reference's `tok.encode("\\n(")`, two tokens on the
/// decider tokenizer) are returned alongside, since the wide layout splices them
/// around each label id.
///
/// # Errors
///
/// [`InferenceError::InvalidInput`] when `max_entries` is not at least
/// [`LETTER_LABELS`] or exceeds the number of candidate labels.
/// [`InferenceError::Tokenizer`] when the tokenizer cannot yield `max_entries`
/// distinct single-token labels with `A`..`J` first, or when the open marker
/// encodes to no tokens.
pub fn resolve_decision_labels(
    tokenizer: &dyn Tokenizer,
    max_entries: usize,
) -> Result<DecisionLabelTable, InferenceError> {
    const ALPHABET: usize = 26;
    let candidates = ALPHABET + ALPHABET * ALPHABET;
    if max_entries < LETTER_LABELS || max_entries > candidates {
        return Err(InferenceError::InvalidInput(format!(
            "option scoring: max_entries must be in {LETTER_LABELS}..={candidates}, got {max_entries}"
        )));
    }

    let upper: Vec<char> = (0..ALPHABET)
        .map(|offset| char::from(b'A' + offset as u8))
        .collect();
    let mut names: Vec<String> = upper.iter().map(ToString::to_string).collect();
    for first in &upper {
        for second in &upper {
            names.push(format!("{first}{second}"));
        }
    }

    let mut labels = Vec::with_capacity(max_entries);
    let mut ids = Vec::with_capacity(max_entries);
    let mut seen = HashSet::with_capacity(max_entries);
    for name in &names {
        let mut batch = tokenizer.tokenize_batch(&[name.as_str()]);
        let Some(encoded) = batch.pop() else {
            continue;
        };
        if encoded.real_length != 1 || encoded.pre_truncation_len != 1 {
            continue;
        }
        let Some(&id) = encoded.input_ids.first() else {
            continue;
        };
        if !seen.insert(id) {
            continue;
        }
        labels.push(name.clone());
        ids.push(id);
        if labels.len() == max_entries {
            break;
        }
    }

    if labels.len() != max_entries {
        return Err(InferenceError::Tokenizer(format!(
            "option scoring: the tokenizer yields only {} single-token labels, {max_entries} required",
            labels.len()
        )));
    }
    for (offset, name) in upper.iter().take(LETTER_LABELS).enumerate() {
        let expected = name.to_string();
        if labels.get(offset) != Some(&expected) {
            return Err(InferenceError::Tokenizer(format!(
                "option scoring: label {offset} must be {expected:?}, found {:?}",
                labels.get(offset)
            )));
        }
    }

    let mut open_batch = tokenizer.tokenize_batch(&["\n("]);
    let Some(open_marker) = open_batch.pop() else {
        return Err(InferenceError::Tokenizer(
            "option scoring: the open marker \"\\n(\" produced no token ids".into(),
        ));
    };
    let open_ids: Vec<u32> = open_marker
        .input_ids
        .get(..open_marker.real_length)
        .map(<[u32]>::to_vec)
        .unwrap_or_default();
    if open_ids.is_empty() {
        return Err(InferenceError::Tokenizer(
            "option scoring: the open marker \"\\n(\" produced no token ids".into(),
        ));
    }

    Ok(DecisionLabelTable {
        labels,
        ids,
        open_ids,
    })
}

/// The narrow layout's letter count: at most this many options render as
/// `(A) .. (J)` inside one string.
pub const LETTER_LABELS: usize = 10;

/// **Unstable**: the decider label table resolved against one tokenizer: the
/// label strings, their single token ids (index-aligned), and the token ids of
/// the `"\\n("` marker the wide layout splices around each label id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionLabelTable {
    /// The label spellings, `A`..`J` first.
    pub labels: Vec<String>,
    /// The single token id of each label, index-aligned with `labels`.
    pub ids: Vec<u32>,
    /// The token ids of the `"\\n("` open marker.
    pub open_ids: Vec<u32>,
}

#[cfg(test)]
mod tests;
