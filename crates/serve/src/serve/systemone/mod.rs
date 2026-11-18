//! Request, prompt, and answer layer for the system-one (multi-slot decision)
//! wire format, ported from the public decider checkpoint's own server code.
//!
//! This is the pure, testable half of the decision endpoint: it parses a
//! request, renders its state and questions, plans the scoring rows, assembles
//! each row's prompt ids against the checkpoint's tokenizer, and turns given
//! probability rows back into the wire answers. [`score_prepared`] runs a
//! caller-given row scorer over the assembled rows and applies
//! [`DeciderConfig::temperature`] before the softmax, and [`http`] serves it
//! as `/v1/systemone`, scoring the rows on the serving worker.
//!
//! The layout is state-first and one question per row. `prompt` uses the
//! checkpoint's own tokenizer; the reference encodes state and questions
//! separately (the state once, then each question), which is the length
//! measured for `usage.input_tokens` and the shape the answers were calibrated
//! against.
//!
//! [`DeciderConfig::temperature`]: crate::serve::systemone::DeciderConfig::temperature

pub mod answer;
pub mod config;
pub mod http;
pub mod json;
pub mod prompt;
pub mod request;
pub mod score;
pub mod text;

use std::path::Path;

pub use answer::{AnswerValue, AssembledAnswers, assemble};
pub use config::{DeciderConfig, parse_decider_config};
pub use json::PyValue;
pub use prompt::{AssembledPrompt, RowPrompt, assemble_prompt};
pub use request::{
    IndexEntry, PlannedRow, QuestionKind, RenderedQuestion, RowKind, SystemOneRequest,
    parse_request, plan_rows, render_question, render_state,
};
pub use score::{ScoreFailure, calibrated_probabilities, score_prepared};

use indexmap::IndexMap;

use lattice_inference::option_scoring::{DecisionLabelTable, resolve_decision_labels};
use lattice_tokenizer::Tokenizer;

/// The most scoring rows one request may plan. The reference sets no limit;
/// this one is far above the reference sample's largest request (64
/// questions). The work a request costs is bounded by
/// [`MAX_PROMPT_TOKENS_PER_REQUEST`].
pub const MAX_ROWS_PER_REQUEST: usize = 1024;

/// The most prompt tokens one request may assemble, summed over its rows.
/// Every row holds its own copy of the state followed by one question, so this
/// bounds both the ids the request allocates and the tokens a per-row scorer
/// prefills. It admits 64 rows (the reference sample's largest request) at a
/// 65,536-token row. The check runs before any row copies the state.
pub const MAX_PROMPT_TOKENS_PER_REQUEST: usize = 1 << 22;

/// A refusal by the system-one layer, carrying the message the reference would
/// return as its HTTP 422 body.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SystemOneError {
    /// The request is refused; the message is the reference's 422 text.
    #[error("{0}")]
    Refused(String),
}

/// A fully prepared request: the rendered questions, the planned rows, the
/// assembled prompt, and the answer index. The caller scores the prompt's rows
/// and hands the probability rows to [`assemble`].
#[derive(Debug, Clone)]
pub struct PreparedSystemOne {
    /// The rendered questions, in question order.
    pub rendered: IndexMap<String, RenderedQuestion>,
    /// The row index, in question order.
    pub index: Vec<IndexEntry>,
    /// The assembled prompt.
    pub prompt: AssembledPrompt,
    /// The checkpoint config the request was prepared against.
    pub config: DeciderConfig,
    /// The label table's token ids the prompt was assembled against, in table
    /// order. A row with `nopts` options is scored over the first `nopts`.
    pub label_ids: Vec<u32>,
    /// The input token count the response reports, counted when the prompt was
    /// assembled, so it holds after a caller moves the rows' ids to a scorer.
    pub input_tokens: usize,
}

/// The reference's response shape, with `usage.input_tokens` filled in from the
/// assembled prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemOneResponse {
    /// The response `model` string.
    pub model: String,
    /// The answers object.
    pub answers: IndexMap<String, AnswerValue>,
    /// The input token count.
    pub input_tokens: usize,
    /// The output token count (always zero for this read-out).
    pub output_tokens: usize,
}

impl SystemOneResponse {
    /// The response keyword the reference reports, derived from the config
    /// version (`decider-<version>`).
    pub fn keyword_for(config: &DeciderConfig) -> String {
        format!("decider-{}", config.version)
    }

    /// Render the response as the reference's JSON object.
    pub fn to_serde_json(&self) -> serde_json::Value {
        let mut usage = serde_json::Map::new();
        usage.insert(
            "input_tokens".to_string(),
            serde_json::Value::Number(self.input_tokens.into()),
        );
        usage.insert(
            "output_tokens".to_string(),
            serde_json::Value::Number(self.output_tokens.into()),
        );
        let mut root = serde_json::Map::new();
        root.insert(
            "model".to_string(),
            serde_json::Value::String(self.model.clone()),
        );
        root.insert(
            "answers".to_string(),
            serde_json::Value::Object(
                self.answers
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_serde_json()))
                    .collect(),
            ),
        );
        root.insert("usage".to_string(), serde_json::Value::Object(usage));
        serde_json::Value::Object(root)
    }
}

/// Resolve the checkpoint's label table against its tokenizer.
pub fn resolve_labels(
    tokenizer: &dyn Tokenizer,
    max_options: usize,
) -> Result<DecisionLabelTable, SystemOneError> {
    resolve_decision_labels(tokenizer, max_options)
        .map_err(|error| SystemOneError::Refused(format!("label table: {error}")))
}

/// Prepare a request: parse it, render and plan it, and assemble its prompt.
pub fn prepare_request(
    tokenizer: &dyn Tokenizer,
    labels: &DecisionLabelTable,
    config: &DeciderConfig,
    body: &str,
) -> Result<PreparedSystemOne, SystemOneError> {
    let request = parse_request(body)?;
    let state = render_state(&request.state);
    let mut rendered = IndexMap::with_capacity(request.questions.len());
    for (id, spec) in &request.questions {
        rendered.insert(id.clone(), render_question(spec)?);
    }
    let (rows, index) = plan_rows(&rendered, config.isolated_levels);
    if rows.len() > MAX_ROWS_PER_REQUEST {
        return Err(SystemOneError::Refused(format!(
            "a request may plan at most {MAX_ROWS_PER_REQUEST} scoring rows, this one plans {}",
            rows.len()
        )));
    }
    let prompt = assemble_prompt(tokenizer, &state, &rows, config, labels)?;
    Ok(PreparedSystemOne {
        rendered,
        index,
        input_tokens: prompt.unique_tokens(),
        prompt,
        config: config.clone(),
        label_ids: labels.ids.clone(),
    })
}

/// Load a checkpoint's `decider_config.json` from a directory.
pub fn load_decider_config(checkpoint_dir: &Path) -> Result<DeciderConfig, SystemOneError> {
    let path = checkpoint_dir.join("decider_config.json");
    let text = std::fs::read_to_string(&path).map_err(|error| {
        SystemOneError::Refused(format!("could not read {}: {error}", path.display()))
    })?;
    parse_decider_config(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_keyword_is_derived_from_the_version() {
        let config = DeciderConfig {
            version: "0.8b-v1".to_string(),
            temperature: 1.03,
            max_state_tokens: 32768,
            max_options: 255,
            isolated_levels: true,
        };
        assert_eq!(SystemOneResponse::keyword_for(&config), "decider-0.8b-v1");
    }

    #[test]
    fn config_file_loads_from_a_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("decider_config.json"),
            r#"{"temperature": 1.03, "temperature_schema_first": 1.03, "neutralize_none": false, "version": "0.8b-v1", "base": "b", "max_options": 255, "max_state_tokens": 32768, "schema_first": false, "schema_first_trained": true, "isolated_levels": true, "recipe": "r", "release_date": "2026-09-19"}"#,
        )
        .expect("write config");
        let config = load_decider_config(dir.path()).expect("loads");
        assert_eq!(config.max_options, 255);
    }
}
