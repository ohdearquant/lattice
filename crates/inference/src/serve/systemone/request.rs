//! Request envelope, state rendering, question rendering, and row planning.
//!
//! A byte-exact port of the reference's `S1Req` fields, `render_state`,
//! `annotate_indices`, `render_question`, and `plan_rows`. The reference's
//! deviations from what its own names suggest are reproduced deliberately and
//! flagged where they occur: the separators are Python's default `", "`/`": "`
//! (not compact, despite the reference docstring), `_txt` embeds `null` for a
//! `null` instruction, and a long array gains an `_index` key that takes first
//! position.

use indexmap::IndexMap;

use super::SystemOneError;
use super::json::{self, PyValue};
use super::text::{
    python_repr_of, python_repr_str, python_str_of_list_criterion, strip_level_number,
};

/// The narrow layout's letter count.
pub const LETTER_LABELS: usize = 10;
/// The largest option count a `choice` question accepts.
pub const MAX_CHOICE: usize = 255;
/// The largest level count a `score` question accepts.
pub const MAX_LEVELS: usize = 10;
/// The shortest array length that gains `_index` annotations.
pub const ANNOTATE_MIN: usize = 8;

/// The question kinds this port renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionKind {
    /// Choose one of 2..255 described options.
    Choice,
    /// A score on 2..10 ordered levels.
    Score,
    /// Yes/no.
    Noul,
}

/// The validated envelope of a request.
#[derive(Debug, Clone)]
pub struct SystemOneRequest {
    /// The request's `state`, any JSON value.
    pub state: PyValue,
    /// The `questions` object, one spec per question id, in document order.
    pub questions: IndexMap<String, PyValue>,
    /// The `independent` flag (default true).
    pub independent: bool,
}

/// One rendered question.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedQuestion {
    /// The question text.
    pub question: String,
    /// The option strings, in option order.
    pub options: Vec<String>,
    /// The question kind.
    pub kind: QuestionKind,
    /// The answer key reported per option.
    pub names: Vec<String>,
    /// The score legend, one text per level.
    pub legend: Option<Vec<String>>,
    /// Whether a `score` question's levels are judged one per row.
    pub isolated: bool,
}

/// One planned row.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedRow {
    /// The row's question text.
    pub question: String,
    /// The row's option strings.
    pub options: Vec<String>,
    /// The question id this row belongs to.
    pub id: String,
}

/// Whether a question occupied one row or one row per score level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    /// One row per score level.
    Isolated,
    /// A single row.
    List,
}

/// One entry mapping a question id onto its rows.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexEntry {
    /// The question id.
    pub id: String,
    /// The row kind.
    pub kind: RowKind,
    /// The first row index this question occupies.
    pub first: usize,
    /// The number of rows this question occupies.
    pub count: usize,
}

/// Parse the request envelope: a JSON object with `state` (any value, required),
/// `questions` (object, required), and optional `independent` (bool, default
/// true). `model`, `layout`, and any other field are accepted and ignored.
///
/// `independent: false` (which the reference would use to pack every question
/// into one numbered prompt) is refused here, because this port implements the
/// independent layout only.
pub fn parse_request(body: &str) -> Result<SystemOneRequest, SystemOneError> {
    let root = json::parse(body)?;
    let object = root
        .as_object()
        .ok_or_else(|| SystemOneError::Refused("request body must be a JSON object".to_string()))?;

    let state = object
        .get("state")
        .cloned()
        .ok_or_else(|| SystemOneError::Refused("request is missing the field \"state\"".into()))?;

    let questions = object
        .get("questions")
        .ok_or_else(|| {
            SystemOneError::Refused("request is missing the field \"questions\"".into())
        })?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            SystemOneError::Refused("the field \"questions\" must be a JSON object".to_string())
        })?;

    let independent = match object.get("independent") {
        Some(PyValue::Bool(value)) => *value,
        Some(other) => {
            return Err(SystemOneError::Refused(format!(
                "the field \"independent\" must be a boolean, found {}",
                python_repr_of(other)
            )));
        }
        None => true,
    };
    if !independent {
        return Err(SystemOneError::Refused(
            "the setting \"independent\": false is not supported".to_string(),
        ));
    }

    Ok(SystemOneRequest {
        state,
        questions,
        independent,
    })
}

/// Render the request state: a string passes through; any other JSON value is
/// rendered with Python's default separators and `_index` annotations.
pub fn render_state(state: &PyValue) -> String {
    if let Some(text) = state.as_str() {
        return text.to_string();
    }
    json::render(&annotate_indices(state, ANNOTATE_MIN))
}

/// Write each element's position into arrays of at least `min_len` elements.
///
/// A dict element becomes `{"_index": i, ...}` in that order (an existing
/// `_index` keeps its first position and takes the element's value, so `_index`
/// first and last write wins); any other element becomes
/// `{"_index": i, "value": elem}`. Dicts and short arrays recurse.
pub fn annotate_indices(value: &PyValue, min_len: usize) -> PyValue {
    match value {
        PyValue::Array(elements) if elements.len() >= min_len => PyValue::Array(
            elements
                .iter()
                .enumerate()
                .map(|(index, element)| {
                    let mut entry = IndexMap::new();
                    entry.insert("_index".to_string(), PyValue::Int(index as i128));
                    match element {
                        PyValue::Object(entries) => {
                            for (key, value) in entries {
                                entry.insert(key.clone(), annotate_indices(value, min_len));
                            }
                        }
                        other => {
                            entry.insert("value".to_string(), annotate_indices(other, min_len));
                        }
                    }
                    PyValue::Object(entry)
                })
                .collect(),
        ),
        PyValue::Array(elements) => PyValue::Array(
            elements
                .iter()
                .map(|element| annotate_indices(element, min_len))
                .collect(),
        ),
        PyValue::Object(entries) => PyValue::Object(
            entries
                .iter()
                .map(|(key, value)| (key.clone(), annotate_indices(value, min_len)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Render one question spec.
pub fn render_question(spec: &PyValue) -> Result<RenderedQuestion, SystemOneError> {
    let object = spec.as_object().ok_or_else(|| {
        SystemOneError::Refused("each question must be a JSON object".to_string())
    })?;

    let (kind, kind_repr) = match object.get("type") {
        None => ("choice".to_string(), "'choice'".to_string()),
        Some(PyValue::Str(text)) => (text.clone(), python_repr_str(text)),
        Some(other) => return Err(unknown_type(&python_repr_of(other))),
    };
    let kind = kind.as_str();

    let instruction = match object.get("instructions") {
        Some(value) => txt(value),
        None => match object.get("question") {
            Some(value) => txt(value),
            None => String::new(),
        },
    };
    if instruction.is_empty() {
        return Err(SystemOneError::Refused(
            "question without instructions".to_string(),
        ));
    }

    let criteria = match object.get("criteria") {
        Some(value) => Some(value),
        None => object.get("options"),
    };

    match kind {
        "choice" => {
            let map = choice_criteria(criteria)?;
            if !(2..=MAX_CHOICE).contains(&map.len()) {
                return Err(SystemOneError::Refused(format!(
                    "choice criteria: a map of 2..{MAX_CHOICE} options"
                )));
            }
            let names: Vec<String> = map.keys().cloned().collect();
            let options: Vec<String> = map
                .iter()
                .map(|(name, description)| match description {
                    value if !value.is_none_or_empty_str() => {
                        format!("{name}: {}", txt(value))
                    }
                    _ => name.clone(),
                })
                .collect();
            Ok(RenderedQuestion {
                question: instruction,
                options,
                kind: QuestionKind::Choice,
                names,
                legend: None,
                isolated: isolated_flag(object),
            })
        }
        "score" => {
            let levels = score_criteria(criteria)?;
            if !(2..=MAX_LEVELS).contains(&levels.len()) {
                return Err(SystemOneError::Refused(format!(
                    "score criteria: an ordered list of 2..{MAX_LEVELS} level descriptions"
                )));
            }
            let legend: Vec<String> = levels.iter().map(txt).collect();
            let names: Vec<String> = (0..levels.len()).map(|index| index.to_string()).collect();
            let options: Vec<String> = legend
                .iter()
                .enumerate()
                .map(|(index, text)| format!("{index}: {text}"))
                .collect();
            Ok(RenderedQuestion {
                question: instruction,
                options,
                kind: QuestionKind::Score,
                names,
                legend: Some(legend),
                isolated: isolated_flag(object),
            })
        }
        "noul" | "bool" => {
            let criteria = noul_criteria(criteria)?;
            let false_text = criteria.as_ref().and_then(|entries| entries.get("false"));
            let true_text = criteria.as_ref().and_then(|entries| entries.get("true"));
            Ok(RenderedQuestion {
                question: instruction,
                options: vec![option_line("no", false_text), option_line("yes", true_text)],
                kind: QuestionKind::Noul,
                names: Vec::new(),
                legend: None,
                isolated: isolated_flag(object),
            })
        }
        _ => Err(unknown_type(&kind_repr)),
    }
}

/// Plan the scoring rows for a set of rendered questions, in question order.
pub fn plan_rows(
    rendered: &IndexMap<String, RenderedQuestion>,
    isolated: bool,
) -> (Vec<PlannedRow>, Vec<IndexEntry>) {
    let mut rows = Vec::new();
    let mut index = Vec::new();
    for (id, question) in rendered {
        if isolated && question.kind == QuestionKind::Score && question.isolated {
            let legend = question.legend.as_deref().unwrap_or(&[]);
            index.push(IndexEntry {
                id: id.clone(),
                kind: RowKind::Isolated,
                first: rows.len(),
                count: legend.len(),
            });
            for level in legend {
                rows.push(PlannedRow {
                    question: format!(
                        "{}\nProposed answer: {}\nDoes the proposed answer fit?",
                        question.question,
                        strip_level_number(level)
                    ),
                    options: vec!["no".to_string(), "yes".to_string()],
                    id: id.clone(),
                });
            }
        } else {
            index.push(IndexEntry {
                id: id.clone(),
                kind: RowKind::List,
                first: rows.len(),
                count: 1,
            });
            rows.push(PlannedRow {
                question: question.question.clone(),
                options: question.options.clone(),
                id: id.clone(),
            });
        }
    }
    (rows, index)
}

/// `_txt(v)`: a string passes through; any other value is rendered as JSON.
pub fn txt(value: &PyValue) -> String {
    match value {
        PyValue::Str(text) => text.clone(),
        other => json::render(other),
    }
}

fn unknown_type(type_text: &str) -> SystemOneError {
    SystemOneError::Refused(format!("unknown question type {type_text}"))
}

fn isolated_flag(object: &IndexMap<String, PyValue>) -> bool {
    object
        .get("isolated")
        .map(PyValue::is_truthy)
        .unwrap_or(true)
}

fn option_line(label: &str, description: Option<&PyValue>) -> String {
    match description {
        Some(value) if !value.is_none_or_empty_str() => format!("{label}: {}", txt(value)),
        _ => label.to_string(),
    }
}

/// A choice's criteria as an ordered map from option name to description. A list
/// criteria is mapped to `{str(element): None}`, deduplicating by rendered name.
fn choice_criteria(
    criteria: Option<&PyValue>,
) -> Result<IndexMap<String, PyValue>, SystemOneError> {
    match criteria {
        Some(PyValue::Array(elements)) => {
            let mut map = IndexMap::new();
            for element in elements {
                map.insert(python_str_of_list_criterion(element)?, PyValue::Null);
            }
            Ok(map)
        }
        Some(PyValue::Object(entries)) => Ok(entries.clone()),
        Some(PyValue::Null) | None => Ok(IndexMap::new()),
        Some(_) => Err(SystemOneError::Refused(format!(
            "choice criteria: a map of 2..{MAX_CHOICE} options"
        ))),
    }
}

/// A score's criteria as an ordered list of level descriptions. A dict legend is
/// sorted by `float(key)`; the sort is stable, so keys that parse to the same
/// float keep their document order.
fn score_criteria(criteria: Option<&PyValue>) -> Result<Vec<PyValue>, SystemOneError> {
    match criteria {
        Some(PyValue::Object(entries)) => {
            let mut keyed: Vec<(f64, &PyValue)> = Vec::with_capacity(entries.len());
            for (key, value) in entries {
                let Ok(order) = key.parse::<f64>() else {
                    return Err(SystemOneError::Refused(format!(
                        "could not convert string to float: {}",
                        python_repr_str(key)
                    )));
                };
                keyed.push((order, value));
            }
            keyed.sort_by(|left, right| {
                left.0
                    .partial_cmp(&right.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(keyed.into_iter().map(|(_, value)| value.clone()).collect())
        }
        Some(PyValue::Array(elements)) => Ok(elements.clone()),
        Some(PyValue::Null) | None => Ok(Vec::new()),
        Some(other) => {
            if other.is_truthy() {
                Err(SystemOneError::Refused(format!(
                    "score criteria: an ordered list of 2..{MAX_LEVELS} level descriptions"
                )))
            } else {
                Ok(Vec::new())
            }
        }
    }
}

/// A `noul`'s description map, or `None` for an empty/falsy criteria.
fn noul_criteria(
    criteria: Option<&PyValue>,
) -> Result<Option<IndexMap<String, PyValue>>, SystemOneError> {
    match criteria {
        Some(PyValue::Object(entries)) => Ok(Some(entries.clone())),
        Some(PyValue::Null) | None => Ok(None),
        Some(other) if !other.is_truthy() => Ok(None),
        Some(other) => Err(SystemOneError::Refused(format!(
            "noul criteria must be a map, found {}",
            python_repr_of(other)
        ))),
    }
}

#[cfg(test)]
mod tests;
