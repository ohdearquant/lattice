//! Answer assembly: turn one probability row per planned row into the
//! reference's `answers` object.
//!
//! A byte-exact port of the reference's `assemble` and `format_answer`, with the
//! Python `round` semantics of [`super::text::python_round`]: each answer
//! renormalizes its first `n` probabilities, takes the first maximum as the
//! choice, and reports `confidence`, `certainty`, and (for a score) the expected
//! level.

use indexmap::IndexMap;

use super::SystemOneError;
use super::request::{IndexEntry, RenderedQuestion, RowKind};
use super::text::{is_zero, python_round};

/// The reported probabilities' rounding, the reference's `nd`.
const ROUND_DIGITS: i32 = 4;

/// A built answer value, in the shape the reference serializes.
#[derive(Debug, Clone, PartialEq)]
pub enum AnswerValue {
    /// A JSON string.
    Str(String),
    /// A JSON number.
    Number(f64),
    /// A JSON boolean (unused today, kept for completeness of the shape).
    Bool(bool),
    /// A JSON object, in insertion order.
    Map(IndexMap<String, AnswerValue>),
    /// A JSON null.
    Null,
}

impl AnswerValue {
    /// A number value.
    pub fn number(value: f64) -> Self {
        AnswerValue::Number(value)
    }

    /// A string value.
    pub fn string(value: impl Into<String>) -> Self {
        AnswerValue::Str(value.into())
    }

    /// Convert to `serde_json::Value`. A non-finite number renders as `null`,
    /// which cannot occur for the renormalized probabilities this layer builds.
    pub fn to_serde_json(&self) -> serde_json::Value {
        match self {
            AnswerValue::Str(text) => serde_json::Value::String(text.clone()),
            AnswerValue::Number(value) => serde_json::Number::from_f64(*value)
                .map_or(serde_json::Value::Null, serde_json::Value::Number),
            AnswerValue::Bool(value) => serde_json::Value::Bool(*value),
            AnswerValue::Null => serde_json::Value::Null,
            AnswerValue::Map(entries) => serde_json::Value::Object(
                entries
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_serde_json()))
                    .collect(),
            ),
        }
    }
}

/// The parsed `assemble` result for one request, plus the inputs it derived.
#[derive(Debug, Clone, PartialEq)]
pub struct AssembledAnswers {
    /// The answers object, keyed by question id, in question order.
    pub answers: IndexMap<String, AnswerValue>,
    /// The rows' longest common prefix token count, for `usage.input_tokens`.
    pub unique_tokens: usize,
}

/// Assemble the answers for one request from one probability row per planned
/// row, in planned order.
pub fn assemble(
    rendered: &IndexMap<String, RenderedQuestion>,
    index: &[IndexEntry],
    probs: &[Vec<f64>],
    unique_tokens: usize,
) -> Result<AssembledAnswers, SystemOneError> {
    let mut answers = IndexMap::new();
    for entry in index {
        let question = rendered.get(&entry.id).ok_or_else(|| {
            SystemOneError::Refused(format!(
                "the answer index names the unknown question {:?}",
                entry.id
            ))
        })?;
        let value = match entry.kind {
            RowKind::Isolated => {
                let mut fit = Vec::with_capacity(entry.count);
                for offset in 0..entry.count {
                    let row = probs.get(entry.first + offset).ok_or_else(|| {
                        SystemOneError::Refused(
                            "the probability rows are fewer than the planned rows".to_string(),
                        )
                    })?;
                    let yes = row.get(1).copied().ok_or_else(|| {
                        SystemOneError::Refused(
                            "an isolated row needs two probability entries".to_string(),
                        )
                    })?;
                    fit.push(yes);
                }
                let (normalized, mass) = combine_isolated(&fit);
                let answer = format_answer(question, &normalized)?;
                let mut level_fit = IndexMap::new();
                for (level, value) in fit.iter().enumerate() {
                    level_fit.insert(
                        level.to_string(),
                        AnswerValue::number(python_round(*value, 4)),
                    );
                }
                let mut fit_mass = IndexMap::new();
                fit_mass.insert("level_fit".to_string(), AnswerValue::Map(level_fit));
                fit_mass.insert(
                    "fit_mass".to_string(),
                    AnswerValue::number(python_round(mass, 4)),
                );
                merge(answer, fit_mass)
            }
            RowKind::List => {
                let row = probs.get(entry.first).ok_or_else(|| {
                    SystemOneError::Refused(
                        "the probability rows are fewer than the planned rows".to_string(),
                    )
                })?;
                format_answer(question, row)?
            }
        };
        answers.insert(entry.id.clone(), value);
    }
    Ok(AssembledAnswers {
        answers,
        unique_tokens,
    })
}

/// Normalize a set of per-level P(fits) values into a distribution, returning
/// the distribution and the unnormalized mass.
pub fn combine_isolated(p_yes: &[f64]) -> (Vec<f64>, f64) {
    let total: f64 = p_yes.iter().sum();
    let total = if is_zero(total) { 1e-9 } else { total };
    (p_yes.iter().map(|value| value / total).collect(), total)
}

/// Build one answer from a rendered question and its probability row.
pub fn format_answer(
    question: &RenderedQuestion,
    probs: &[f64],
) -> Result<AnswerValue, SystemOneError> {
    let count = question.options.len();
    let mut values: Vec<f64> = probs.iter().take(count).copied().collect();
    let sum: f64 = values.iter().sum();
    let sum = if is_zero(sum) { 1.0 } else { sum };
    for value in values.iter_mut() {
        *value /= sum;
    }
    let argmax = first_argmax(&values);

    match question.kind {
        super::request::QuestionKind::Noul => {
            let mut answer = IndexMap::new();
            answer.insert("type".to_string(), AnswerValue::string("noul"));
            answer.insert(
                "noul".to_string(),
                AnswerValue::number(python_round(
                    values.get(1).copied().unwrap_or(0.0),
                    ROUND_DIGITS,
                )),
            );
            Ok(AnswerValue::Map(answer))
        }
        super::request::QuestionKind::Choice => {
            let mut answer = IndexMap::new();
            answer.insert("type".to_string(), AnswerValue::string("choice"));
            answer.insert(
                "choice".to_string(),
                AnswerValue::string(question.names.get(argmax).cloned().unwrap_or_default()),
            );
            answer.insert(
                "confidence".to_string(),
                AnswerValue::number(python_round(
                    values.get(argmax).copied().unwrap_or(0.0),
                    ROUND_DIGITS,
                )),
            );
            answer.insert(
                "certainty".to_string(),
                AnswerValue::number(python_round(certainty(&values), ROUND_DIGITS)),
            );
            let mut probabilities = IndexMap::new();
            for (name, value) in question.names.iter().zip(values.iter()) {
                probabilities.insert(
                    name.clone(),
                    AnswerValue::number(python_round(*value, ROUND_DIGITS)),
                );
            }
            answer.insert("probabilities".to_string(), AnswerValue::Map(probabilities));
            Ok(AnswerValue::Map(answer))
        }
        super::request::QuestionKind::Score => {
            let score: f64 = values
                .iter()
                .enumerate()
                .map(|(level, value)| level as f64 * value)
                .sum();
            let mut answer = IndexMap::new();
            answer.insert("type".to_string(), AnswerValue::string("score"));
            answer.insert(
                "score".to_string(),
                AnswerValue::number(python_round(score, 2)),
            );
            answer.insert(
                "confidence".to_string(),
                AnswerValue::number(python_round(
                    values.get(argmax).copied().unwrap_or(0.0),
                    ROUND_DIGITS,
                )),
            );
            answer.insert(
                "certainty".to_string(),
                AnswerValue::number(python_round(certainty(&values), ROUND_DIGITS)),
            );
            let legend = question.legend.clone().unwrap_or_default();
            let mut legend_map = IndexMap::new();
            for (level, text) in legend.iter().enumerate() {
                legend_map.insert(level.to_string(), AnswerValue::string(text.clone()));
            }
            answer.insert("legend".to_string(), AnswerValue::Map(legend_map));
            let mut probabilities = IndexMap::new();
            for (level, value) in values.iter().enumerate() {
                probabilities.insert(
                    level.to_string(),
                    AnswerValue::number(python_round(*value, ROUND_DIGITS)),
                );
            }
            answer.insert("probabilities".to_string(), AnswerValue::Map(probabilities));
            Ok(AnswerValue::Map(answer))
        }
    }
}

/// The first index of the maximum, matching `max(range(len(p)), key=p.__getitem__)`.
fn first_argmax(values: &[f64]) -> usize {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate().skip(1) {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// `1 - normalised entropy`, in `[0, 1]`.
pub fn certainty(probs: &[f64]) -> f64 {
    if probs.len() <= 1 {
        return 1.0;
    }
    let entropy: f64 = probs
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| -value * value.ln())
        .sum();
    let normalizer = (probs.len() as f64).ln();
    (1.0 - entropy / normalizer).max(0.0)
}

fn merge(left: AnswerValue, extra: IndexMap<String, AnswerValue>) -> AnswerValue {
    match left {
        AnswerValue::Map(mut entries) => {
            for (key, value) in extra {
                entries.insert(key, value);
            }
            AnswerValue::Map(entries)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::super::json;
    use super::super::request::{IndexEntry, RowKind, render_question};
    use super::*;

    fn rendered(text: &str) -> RenderedQuestion {
        render_question(&json::parse(text).expect("spec")).expect("renders")
    }

    #[test]
    fn choice_answer_reports_first_maximum() {
        let question =
            rendered(r#"{"type": "choice", "instructions": "Pick.", "criteria": ["a", "b", "c"]}"#);
        let answer = format_answer(&question, &[0.4, 0.2, 0.4]).expect("answer");
        let map = match answer {
            AnswerValue::Map(entries) => entries,
            other => panic!("{other:?}"),
        };
        assert_eq!(map.get("type"), Some(&AnswerValue::string("choice")));
        assert_eq!(map.get("choice"), Some(&AnswerValue::string("a")));
        assert_eq!(map.get("confidence"), Some(&AnswerValue::number(0.4)));
        assert_eq!(map.get("certainty"), Some(&AnswerValue::number(0.0398)));
    }

    #[test]
    fn noul_answer_reports_the_yes_probability() {
        let question = rendered(r#"{"type": "noul", "instructions": "Is it?"}"#);
        let answer =
            format_answer(&question, &[0.3707666577090162, 0.6292333422909838]).expect("answer");
        let map = match answer {
            AnswerValue::Map(entries) => entries,
            other => panic!("{other:?}"),
        };
        assert_eq!(map.get("type"), Some(&AnswerValue::string("noul")));
        assert_eq!(map.get("noul"), Some(&AnswerValue::number(0.6292)));
    }

    #[test]
    fn isolated_score_reports_level_fit_and_fit_mass() {
        let question = rendered(
            r#"{"type": "score", "instructions": "Rate.", "criteria": ["0: bad", "1: ok", "2: good"]}"#,
        );
        let mut rendered_map = IndexMap::new();
        rendered_map.insert("q".to_string(), question);
        let index = [IndexEntry {
            id: "q".to_string(),
            kind: RowKind::Isolated,
            first: 0,
            count: 3,
        }];
        let probs = vec![
            vec![0.8591427740451351, 0.140857225954865],
            vec![0.9996843971029108, 0.0003156028970891349],
            vec![0.5990360348969275, 0.4009639651030725],
        ];
        let assembled = assemble(&rendered_map, &index, &probs, 0).expect("assembles");
        let answer = match assembled.answers.get("q") {
            Some(AnswerValue::Map(entries)) => entries,
            other => panic!("{other:?}"),
        };
        assert_eq!(answer.get("score"), Some(&AnswerValue::number(1.48)));
        assert_eq!(answer.get("confidence"), Some(&AnswerValue::number(0.7396)));
        assert_eq!(answer.get("fit_mass"), Some(&AnswerValue::number(0.5421)));
        let level_fit = match answer.get("level_fit") {
            Some(AnswerValue::Map(entries)) => entries,
            other => panic!("{other:?}"),
        };
        assert_eq!(level_fit.get("0"), Some(&AnswerValue::number(0.1409)));
    }

    #[test]
    fn combine_isolated_normalizes_and_reports_mass() {
        let (distribution, mass) = combine_isolated(&[0.25, 0.25]);
        assert!((distribution[0] - 0.5).abs() < 1e-12);
        assert!((mass - 0.5).abs() < 1e-12);
    }
}
