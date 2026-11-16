use super::*;
use crate::serve::systemone::{DeciderConfig, prepare_request};
use lattice_inference::option_scoring::DecisionLabelTable;
use lattice_tokenizer::{TokenizedInput, Tokenizer};

/// A byte-level tokenizer, so a prepared request needs no real vocabulary.
struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn tokenize(&self, text: &str) -> TokenizedInput {
        let ids: Vec<u32> = text.bytes().map(u32::from).collect();
        let length = ids.len();
        TokenizedInput {
            input_ids: ids,
            attention_mask: vec![1; length],
            token_type_ids: vec![0; length],
            real_length: length,
            pre_truncation_len: length,
        }
    }

    fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput> {
        texts.iter().map(|text| self.tokenize(text)).collect()
    }

    fn vocab_size(&self) -> usize {
        4096
    }

    fn max_seq_len(&self) -> usize {
        usize::MAX
    }
}

fn config(temperature: f64) -> DeciderConfig {
    DeciderConfig {
        version: "0.8b-v1".to_string(),
        temperature,
        max_state_tokens: 32768,
        max_options: 255,
        isolated_levels: true,
    }
}

fn table(entries: usize) -> DecisionLabelTable {
    DecisionLabelTable {
        labels: (0..entries).map(|index| format!("L{index}")).collect(),
        ids: (0..entries).map(|index| 1000 + index as u32).collect(),
        open_ids: vec![10, 40],
    }
}

const ONE_CHOICE: &str = r#"{"state": "The sky is clear.", "questions": {"colour": {"type": "choice", "instructions": "Pick the colour.", "criteria": ["red", "green", "blue"]}}}"#;

fn prepared(body: &str, temperature: f64) -> PreparedSystemOne {
    prepare_request(&ByteTokenizer, &table(255), &config(temperature), body).expect("prepares")
}

fn reference_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|value| (value - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|value| value / sum).collect()
}

#[test]
fn probabilities_are_the_softmax_of_logits_over_temperature() {
    let logits = [0.5_f32, 2.0, -1.0];
    let got = calibrated_probabilities(&logits, 2.0).expect("scores");
    let expected = reference_softmax(&[0.25, 1.0, -0.5]);
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-15, "{got:?} vs {expected:?}");
    }
    let at_one = calibrated_probabilities(&logits, 1.0).expect("scores");
    assert!(
        (at_one[1] - got[1]).abs() > 1e-3,
        "the temperature must change the distribution"
    );
}

#[test]
fn row_softmax_equals_the_masked_softmax_over_the_whole_table() {
    // The reference reads every label row, sets the entries past the row's option
    // count to -inf, and takes one softmax; the first `nopts` entries must agree.
    let table_logits: Vec<f32> = (0..255)
        .map(|index| ((index * 37) % 11) as f32 * 0.3)
        .collect();
    let nopts = 4;
    let temperature = 1.03;
    let masked: Vec<f64> = table_logits
        .iter()
        .enumerate()
        .map(|(index, &logit)| {
            if index < nopts {
                f64::from(logit) / temperature
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let full = reference_softmax(&masked);
    let row = calibrated_probabilities(&table_logits[..nopts], temperature).expect("scores");
    for index in 0..nopts {
        assert!((row[index] - full[index]).abs() < 1e-15);
    }
    assert!(full[nopts..].iter().all(|&p| p == 0.0));
}

#[test]
fn bad_rows_and_temperatures_are_refused() {
    assert!(calibrated_probabilities(&[], 1.0).is_err());
    assert!(calibrated_probabilities(&[1.0, f32::NAN], 1.0).is_err());
    assert!(calibrated_probabilities(&[1.0, f32::INFINITY], 1.0).is_err());
    for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(calibrated_probabilities(&[1.0, 2.0], temperature).is_err());
    }
}

#[test]
fn a_temperature_that_overflows_the_scaled_logits_is_refused() {
    let tiny: f64 = 1e-310;
    assert!(tiny > 0.0 && tiny.is_finite());
    assert!(calibrated_probabilities(&[1.0, 2.0], tiny).is_err());
    let small = calibrated_probabilities(&[0.0, 0.0], tiny).expect("zero logits stay finite");
    assert_eq!(small, vec![0.5, 0.5]);
}

#[test]
fn prepare_request_records_the_label_ids_it_assembled_against() {
    let request = prepared(ONE_CHOICE, 1.0);
    assert_eq!(request.label_ids, table(255).ids);
}

#[test]
fn each_row_is_scored_with_its_own_ids_and_first_labels() {
    let request = prepared(ONE_CHOICE, 1.0);
    let mut seen = Vec::new();
    let response = score_prepared(&request, |ids, labels| {
        seen.push((ids.to_vec(), labels.to_vec()));
        Ok::<_, String>(vec![0.0, 3.0, 1.0])
    })
    .expect("scores");
    assert_eq!(seen.len(), request.prompt.rows.len());
    assert_eq!(seen[0].0, request.prompt.rows[0].ids);
    assert_eq!(seen[0].1, vec![1000, 1001, 1002]);
    assert_eq!(response.model, "decider-0.8b-v1");
    assert_eq!(response.input_tokens, request.prompt.unique_tokens());
    assert_eq!(response.output_tokens, 0);
    let answer = response
        .answers
        .get("colour")
        .expect("answered")
        .to_serde_json();
    assert_eq!(answer["choice"], serde_json::json!("green"), "{answer}");
}

#[test]
fn a_scorer_failure_names_its_row() {
    let request = prepared(ONE_CHOICE, 1.0);
    let failure =
        score_prepared(&request, |_, _| Err::<Vec<f32>, _>("gpu lost")).expect_err("fails");
    assert_eq!(
        failure,
        ScoreFailure::Scorer {
            row: 0,
            error: "gpu lost"
        }
    );
}

#[test]
fn a_wrong_logit_count_is_refused() {
    let request = prepared(ONE_CHOICE, 1.0);
    let failure =
        score_prepared(&request, |_, _| Ok::<_, String>(vec![0.0, 1.0])).expect_err("refuses");
    assert!(matches!(failure, ScoreFailure::Refused(_)), "{failure:?}");
}

#[test]
fn a_row_wider_than_the_label_table_is_refused() {
    let mut request = prepared(ONE_CHOICE, 1.0);
    request.label_ids.truncate(2);
    let failure = score_prepared(&request, |_, labels| {
        Ok::<_, String>(vec![0.0; labels.len()])
    })
    .expect_err("refuses");
    assert!(matches!(failure, ScoreFailure::Refused(_)), "{failure:?}");
}
