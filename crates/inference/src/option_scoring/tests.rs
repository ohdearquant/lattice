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

/// A tokenizer with single-token letters `A`..`Z`, all 676 two-letter
/// strings as single tokens, and `"\n("` as two tokens. Enough to exercise
/// the 255-entry table exactly.
fn table_bpe() -> BpeTokenizer {
    let mut vocab = HashMap::new();
    let mut merges = Vec::new();
    for (id, letter) in (b'A'..=b'Z').enumerate() {
        vocab.insert(char::from(letter).to_string(), id as u32);
    }
    for (offset, first) in (b'A'..=b'Z').enumerate() {
        for (inner, second) in (b'A'..=b'Z').enumerate() {
            let left = char::from(first).to_string();
            let right = char::from(second).to_string();
            vocab.insert(format!("{left}{right}"), 26 + (offset * 26 + inner) as u32);
            merges.push((left, right));
        }
    }
    // The newline's byte-level spelling, and the identity-mapped "(".
    vocab.insert("\u{10a}".to_string(), 800);
    vocab.insert("(".to_string(), 801);
    BpeTokenizer::from_vocab_and_merges(vocab, merges).expect("table fixture builds")
}

#[test]
fn resolves_a_full_decision_label_table() {
    let tokenizer = table_bpe();
    let table = resolve_decision_labels(&tokenizer, 255).expect("255 labels resolve");
    assert_eq!(table.labels.len(), 255);
    assert_eq!(table.ids.len(), 255);
    assert_eq!(table.labels[0], "A");
    assert_eq!(table.labels[25], "Z");
    assert_eq!(table.labels[26], "AA");
    assert_eq!(table.ids[0], 0);
    assert_eq!(table.ids[25], 25);
    assert_eq!(table.ids[26], 26);
    let distinct: HashSet<u32> = table.ids.iter().copied().collect();
    assert_eq!(distinct.len(), 255);
    assert_eq!(table.open_ids, vec![800, 801]);
}

#[test]
fn refuses_a_label_table_it_cannot_fill() {
    // Only A..Z are single tokens, so 27 entries cannot be resolved.
    let tokenizer = letter_bpe(26);
    assert!(matches!(
        resolve_decision_labels(&tokenizer, 27),
        Err(InferenceError::Tokenizer(_))
    ));
    // A table shorter than the ten-letter narrow layout is refused outright.
    assert!(matches!(
        resolve_decision_labels(&tokenizer, 3),
        Err(InferenceError::InvalidInput(_))
    ));
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
