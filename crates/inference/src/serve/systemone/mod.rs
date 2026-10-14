//! Request, prompt, and answer layer for the system-one (multi-slot decision)
//! wire format, ported from the public decider checkpoint's own server code.
//!
//! This is the pure, testable half of the decision endpoint: it parses a
//! request, renders its state and questions, plans the scoring rows, assembles
//! each row's prompt ids against the checkpoint's tokenizer, and turns given
//! probability rows back into the wire answers. It holds no HTTP route, calls
//! no model, and applies no temperature; the endpoint that reads probabilities
//! from a model and applies [`DeciderConfig::temperature`] sits above it.
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
pub mod json;
pub mod prompt;
pub mod request;
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

use indexmap::IndexMap;

use crate::option_scoring::{DecisionLabelTable, resolve_decision_labels};
use crate::tokenizer::Tokenizer;

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
    let prompt = assemble_prompt(tokenizer, &state, &rows, config, labels)?;
    Ok(PreparedSystemOne {
        rendered,
        index,
        prompt,
        config: config.clone(),
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
