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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{BpeTokenizer, TokenizedInput};
    use std::collections::HashMap;

    /// A row with distinct, mixed-sign values so that a wrong index changes the result.
    fn row(vocab: usize) -> Vec<f32> {
        (0..vocab)
            .map(|i| ((i * 37 % 29) as f32) * 0.25 - 3.0)
            .collect()
    }

    /// Reference softmax in f64 over `values`, shifted by the maximum.
    fn reference_softmax(values: &[f32]) -> Vec<f64> {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let terms: Vec<f64> = values.iter().map(|&v| f64::from(v - max).exp()).collect();
        let sum: f64 = terms.iter().sum();
        terms.iter().map(|t| t / sum).collect()
    }

    #[test]
    fn scores_keep_caller_order_and_raw_logit_bits() {
        let logits = row(40);
        let ids = [17_u32, 3, 39, 0];
        let scores = option_scores_from_logits(&logits, &ids).expect("valid row");
        let expected: Vec<u32> = ids.iter().map(|&i| logits[i as usize].to_bits()).collect();
        let actual: Vec<u32> = scores.logits.iter().map(|l| l.to_bits()).collect();
        assert_eq!(actual, expected, "logits follow the caller's id order");

        let reversed: Vec<u32> = ids.iter().rev().copied().collect();
        let flipped = option_scores_from_logits(&logits, &reversed).expect("valid row");
        let mut flipped_logits = flipped.logits.clone();
        flipped_logits.reverse();
        assert_eq!(flipped_logits, scores.logits);
        assert_ne!(
            scores.probs, flipped.probs,
            "probs are positional, so reversing the ids reverses them"
        );
    }

    #[test]
    fn probs_are_the_softmax_of_the_letter_logits_and_sum_to_one() {
        let logits = row(64);
        let ids = [5_u32, 11, 2, 60, 33];
        let scores = option_scores_from_logits(&logits, &ids).expect("valid row");
        let sum: f32 = scores.probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probs sum to {sum}");
        let expected = reference_softmax(&scores.logits);
        for (p, e) in scores.probs.iter().zip(&expected) {
            assert!((f64::from(*p) - e).abs() < 1e-6, "prob {p} vs softmax {e}");
        }
        assert!(scores.probs.iter().all(|p| *p > 0.0));
    }

    #[test]
    fn label_mass_is_the_full_vocabulary_mass_on_the_letters() {
        // Uniform row: k of V equal tokens hold k / V of the mass, and the letters split it evenly.
        let uniform = vec![0.5_f32; 50];
        let scores = option_scores_from_logits(&uniform, &[1, 2, 3, 4]).expect("valid row");
        assert!((scores.label_mass - 4.0 / 50.0).abs() < 1e-6);
        assert!(scores.probs.iter().all(|p| (p - 0.25).abs() < 1e-6));

        // A general row against an independent reference.
        let logits = row(64);
        let ids = [9_u32, 10, 11];
        let scores = option_scores_from_logits(&logits, &ids).expect("valid row");
        let full = reference_softmax(&logits);
        let expected: f64 = ids.iter().map(|&i| full[i as usize]).sum();
        assert!((f64::from(scores.label_mass) - expected).abs() < 1e-6);
        assert!((0.0..=1.0).contains(&scores.label_mass));
    }

    #[test]
    fn label_mass_stays_within_unit_interval_at_both_extremes() {
        // The letters hold essentially all the mass.
        let mut high = vec![-1.0e30_f32; 32];
        high[4] = 10.0;
        high[7] = 10.0;
        let scores = option_scores_from_logits(&high, &[4, 7]).expect("valid row");
        assert!(scores.label_mass <= 1.0 && scores.label_mass > 0.999_999);

        // The letters hold essentially none of it.
        let mut low = vec![0.0_f32; 32];
        low[4] = -1.0e30;
        low[7] = -1.0e30;
        let scores = option_scores_from_logits(&low, &[4, 7]).expect("valid row");
        assert!(scores.label_mass >= 0.0 && scores.label_mass < 1e-6);
        let sum: f32 = scores.probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probs still normalise: {sum}");
    }

    #[test]
    fn softmax_is_stable_for_large_logits() {
        let mut logits = vec![0.0_f32; 16];
        logits[2] = 3.0e4;
        logits[5] = 3.0e4 - 1.0;
        let scores = option_scores_from_logits(&logits, &[2, 5]).expect("valid row");
        assert!(scores.probs.iter().all(|p| p.is_finite()));
        let ratio = scores.probs[1] / scores.probs[0];
        assert!((ratio - (-1.0_f32).exp()).abs() < 1e-5, "ratio {ratio}");
        assert!(scores.label_mass.is_finite());
    }

    #[test]
    fn single_letter_has_probability_one() {
        let scores = option_scores_from_logits(&row(10), &[3]).expect("valid row");
        assert_eq!(scores.probs, vec![1.0]);
    }

    fn assert_invalid_input(result: Result<OptionScores, InferenceError>, needle: &str) {
        match result {
            Err(InferenceError::InvalidInput(message)) => {
                assert!(message.contains(needle), "{message}");
            }
            other => panic!("expected InvalidInput containing {needle:?}, got {other:?}"),
        }
    }

    #[test]
    fn refuses_zero_letters() {
        assert_invalid_input(option_scores_from_logits(&row(8), &[]), "must not be empty");
    }

    #[test]
    fn refuses_a_letter_id_beyond_the_vocabulary() {
        assert_invalid_input(
            option_scores_from_logits(&row(8), &[1, 8]),
            "letter_ids[1]=8 out of range",
        );
        assert_invalid_input(
            option_scores_from_logits(&row(8), &[u32::MAX]),
            "out of range",
        );
        assert_invalid_input(option_scores_from_logits(&[], &[0]), "out of range");
    }

    #[test]
    fn refuses_duplicate_letter_ids() {
        assert_invalid_input(
            option_scores_from_logits(&row(8), &[2, 5, 2]),
            "letter_ids[2]=2 is a duplicate",
        );
    }

    #[test]
    fn refuses_a_non_finite_logit_anywhere_in_the_row() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // At a letter position and at a position that only enters the full softmax.
            for position in [3_usize, 6] {
                let mut logits = row(8);
                logits[position] = bad;
                match option_scores_from_logits(&logits, &[1, 3]) {
                    Err(InferenceError::Inference(message)) => {
                        assert!(message.contains("not finite"), "{message}");
                    }
                    other => panic!("{bad} at {position} must be refused, got {other:?}"),
                }
            }
        }
    }

    #[test]
    fn a_refusal_returns_no_scores() {
        assert!(option_scores_from_logits(&row(8), &[9]).is_err());
        let mut logits = row(8);
        logits[0] = f32::NAN;
        assert!(option_scores_from_logits(&logits, &[1]).is_err());
    }

    /// A byte-level BPE tokenizer whose vocabulary holds the letters `A..=Z` at ids `0..26`
    /// plus the end-of-text token at id 26.
    fn letter_bpe(letters: usize) -> BpeTokenizer {
        let mut vocab = HashMap::new();
        for (id, letter) in (b'A'..=b'Z').take(letters).enumerate() {
            vocab.insert(char::from(letter).to_string(), id as u32);
        }
        vocab.insert("<|endoftext|>".to_string(), 26);
        BpeTokenizer::from_vocab_and_merges(vocab, Vec::new()).expect("letter fixture builds")
    }

    /// A tokenizer that returns a fixed encoding per text, to build cases a real vocabulary
    /// cannot express (a multi-token ASCII letter, a truncated result).
    struct FixedTokenizer {
        table: HashMap<&'static str, (Vec<u32>, usize)>,
    }

    impl Tokenizer for FixedTokenizer {
        fn tokenize(&self, text: &str) -> TokenizedInput {
            let (ids, pre_truncation_len) = self
                .table
                .get(text)
                .cloned()
                .unwrap_or_else(|| (Vec::new(), 0));
            let real_length = ids.len();
            let mut input_ids = ids;
            input_ids.resize(4, 0);
            TokenizedInput {
                input_ids,
                attention_mask: vec![1, 1, 0, 0],
                token_type_ids: vec![0; 4],
                real_length,
                pre_truncation_len,
            }
        }

        fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput> {
            texts.iter().map(|t| self.tokenize(t)).collect()
        }

        fn vocab_size(&self) -> usize {
            64
        }

        fn max_seq_len(&self) -> usize {
            4
        }
    }

    fn fixed(entries: &[(&'static str, &[u32], usize)]) -> FixedTokenizer {
        FixedTokenizer {
            table: entries
                .iter()
                .map(|&(text, ids, pre)| (text, (ids.to_vec(), pre)))
                .collect(),
        }
    }

    #[test]
    fn resolves_letters_in_alphabet_order() {
        let tokenizer = letter_bpe(26);
        assert_eq!(
            resolve_option_letters(&tokenizer, 3).expect("A..C resolve"),
            vec![0, 1, 2]
        );
        let all = resolve_option_letters(&tokenizer, 26).expect("A..Z resolve");
        assert_eq!(all, (0..26).collect::<Vec<u32>>());
    }

    #[test]
    fn refuses_k_zero_and_k_beyond_the_alphabet() {
        let tokenizer = letter_bpe(26);
        for k in [0_usize, 27, 1000] {
            match resolve_option_letters(&tokenizer, k) {
                Err(InferenceError::InvalidInput(message)) => {
                    assert!(message.contains("k must be in 1..=26"), "{message}");
                }
                other => panic!("k={k} must be refused, got {other:?}"),
            }
        }
    }

    #[test]
    fn refuses_a_letter_that_is_missing_from_the_vocabulary() {
        // No `D` in the vocabulary and no unknown token: it encodes to zero tokens.
        let tokenizer = letter_bpe(3);
        assert!(resolve_option_letters(&tokenizer, 3).is_ok());
        match resolve_option_letters(&tokenizer, 4) {
            Err(InferenceError::Tokenizer(message)) => {
                assert!(message.contains("\"D\""), "{message}");
            }
            other => panic!("a missing letter must be refused, got {other:?}"),
        }
    }

    #[test]
    fn refuses_every_letter_when_the_tokenizer_appends_a_token() {
        let tokenizer = letter_bpe(4).with_add_eos();
        match resolve_option_letters(&tokenizer, 2) {
            Err(InferenceError::Tokenizer(message)) => {
                assert!(message.contains("\"A\""), "{message}");
                assert!(message.contains("expected exactly 1"), "{message}");
            }
            other => panic!("an appended end-of-text token must be refused, got {other:?}"),
        }
    }

    #[test]
    fn refuses_when_one_letter_is_multi_token() {
        let tokenizer = fixed(&[("A", &[10], 1), ("B", &[11], 1), ("C", &[12, 13], 2)]);
        assert_eq!(
            resolve_option_letters(&tokenizer, 2).expect("A, B are single tokens"),
            vec![10, 11]
        );
        match resolve_option_letters(&tokenizer, 3) {
            Err(InferenceError::Tokenizer(message)) => {
                assert!(message.contains("\"C\""), "{message}");
                assert!(message.contains("encodes to 2 tokens"), "{message}");
            }
            other => panic!("a two-token letter must be refused, got {other:?}"),
        }
    }

    #[test]
    fn refuses_a_truncated_encoding() {
        // The tokenizer kept one token but reports that two were produced.
        let tokenizer = fixed(&[("A", &[10], 2)]);
        assert!(matches!(
            resolve_option_letters(&tokenizer, 1),
            Err(InferenceError::Tokenizer(_))
        ));
    }

    #[test]
    fn refuses_two_letters_that_share_a_token_id() {
        let tokenizer = fixed(&[("A", &[10], 1), ("B", &[10], 1)]);
        match resolve_option_letters(&tokenizer, 2) {
            Err(InferenceError::Tokenizer(message)) => {
                assert!(message.contains("same token id 10"), "{message}");
            }
            other => panic!("a shared token id must be refused, got {other:?}"),
        }
    }
}
