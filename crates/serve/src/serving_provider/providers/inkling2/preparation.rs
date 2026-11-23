//! The Inkling 2 chat prompt and the request controls its CPU route serves.
//!
//! The prompt is the text-only subset of the checkpoint's published chat template: each message
//! is its role marker, `<|content_text|>`, the content and `<|end_message|>`, and an assistant
//! message is followed by `<|content_model_end_sampling|>`. One `Thinking effort level` system
//! message goes before the first message that is not a system message, or after the last message
//! when every message is a system message. Nothing follows the last message: the model opens its
//! own turn.

use crate::serve::ApiError;
use crate::serve::contract::{
    ChatRequest, GenerationDefaults, MessageContent, NormalizedChatMessage, NormalizedChatRole,
    RequestedChatOptions, ServeProfile, ValidatedChatRequest, normalize_requested_options,
};
use crate::serve::prompt_adapter::PromptAdapter;
use lattice_inference::error::InferenceError;
use lattice_inference::generation::GenerateConfig;
use std::path::Path;

const MESSAGE_SYSTEM: &str = "<|message_system|>";
const MESSAGE_USER: &str = "<|message_user|>";
const MESSAGE_MODEL: &str = "<|message_model|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const END_MESSAGE: &str = "<|end_message|>";
const END_SAMPLING: &str = "<|content_model_end_sampling|>";
const EFFORT_PREFIX: &str = "Thinking effort level: ";
/// The effort the published template writes when a request sets none.
const DEFAULT_EFFORT: &str = "0.9";
/// The effort written for a request that turns thinking off.
const THINKING_OFF_EFFORT: &str = "0";

/// Inkling 2 chat conventions: the prompt and the stop id.
#[derive(Debug, Clone)]
pub(crate) struct Inkling2PromptAdapter {
    stop_token_ids: Vec<u32>,
}

impl Inkling2PromptAdapter {
    /// An adapter whose one stop id is the checkpoint's end-of-sequence id.
    pub(crate) fn new(eos_token_id: u32) -> Self {
        Self {
            stop_token_ids: vec![eos_token_id],
        }
    }

    /// The stop id of the checkpoint in `dir`: the `eos_token_id` of `generation_config.json`,
    /// which must be one integer and equal the `eos_token_id` of `config.json`, the id the
    /// decoder stops on.
    ///
    /// # Errors
    /// Fails when either file is missing, unreadable or malformed, when either `eos_token_id` is
    /// absent, a list or not a token id, or when the two differ.
    pub(crate) fn checkpoint_stop_token_id(dir: &Path) -> Result<u32, InferenceError> {
        let generation = Self::single_eos_token_id(dir, "generation_config.json")?;
        let config = Self::single_eos_token_id(dir, "config.json")?;
        if generation != config {
            return Err(InferenceError::Inference(format!(
                "generation_config.json eos_token_id {generation} differs from the config.json \
                 eos_token_id {config}"
            )));
        }
        Ok(generation)
    }

    fn single_eos_token_id(dir: &Path, file: &str) -> Result<u32, InferenceError> {
        let path = dir.join(file);
        let text = lattice_weights::bounded_config::read_config_json_bounded(&path, file)?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| InferenceError::Inference(format!("invalid {}: {e}", path.display())))?;
        value
            .get("eos_token_id")
            .and_then(serde_json::Value::as_u64)
            .and_then(|id| u32::try_from(id).ok())
            .ok_or_else(|| {
                InferenceError::Inference(format!(
                    "{} has no eos_token_id that is a single token id",
                    path.display()
                ))
            })
    }

    fn render_with_effort(&self, messages: &[NormalizedChatMessage], effort: &str) -> String {
        let mut prompt = String::new();
        let mut effort_written = false;
        for message in messages {
            if !effort_written && message.role != NormalizedChatRole::System {
                push_effort(&mut prompt, effort);
                effort_written = true;
            }
            match message.role {
                NormalizedChatRole::System => {
                    push_message(&mut prompt, MESSAGE_SYSTEM, &message.content);
                }
                NormalizedChatRole::User => {
                    push_message(&mut prompt, MESSAGE_USER, &message.content);
                }
                NormalizedChatRole::Assistant => {
                    push_message(&mut prompt, MESSAGE_MODEL, &message.content);
                    prompt.push_str(END_SAMPLING);
                }
            }
        }
        if !effort_written {
            push_effort(&mut prompt, effort);
        }
        prompt
    }
}

fn push_message(prompt: &mut String, role: &str, content: &str) {
    prompt.push_str(role);
    prompt.push_str(CONTENT_TEXT);
    prompt.push_str(content);
    prompt.push_str(END_MESSAGE);
}

fn push_effort(prompt: &mut String, effort: &str) {
    prompt.push_str(MESSAGE_SYSTEM);
    prompt.push_str(CONTENT_TEXT);
    prompt.push_str(EFFORT_PREFIX);
    prompt.push_str(effort);
    prompt.push_str(END_MESSAGE);
}

fn unsupported(message: &str) -> ApiError {
    ApiError::BadRequest {
        message: message.to_owned(),
        code: "unsupported_feature",
    }
}

impl PromptAdapter for Inkling2PromptAdapter {
    fn render(&self, messages: &[NormalizedChatMessage]) -> String {
        self.render_with_effort(messages, DEFAULT_EFFORT)
    }

    /// Thinking on writes the template's default effort, 0.9; thinking off writes 0.
    fn render_with_thinking(
        &self,
        messages: &[NormalizedChatMessage],
        enable_thinking: bool,
    ) -> String {
        let effort = if enable_thinking {
            DEFAULT_EFFORT
        } else {
            THINKING_OFF_EFFORT
        };
        self.render_with_effort(messages, effort)
    }

    fn stop_token_ids(&self) -> &[u32] {
        &self.stop_token_ids
    }

    fn apply_defaults(
        &self,
        generation: GenerationDefaults,
        options: RequestedChatOptions,
    ) -> Result<ValidatedChatRequest, ApiError> {
        if options
            .messages
            .iter()
            .any(|message| message.image.is_some())
        {
            return Err(ApiError::BadRequest {
                message: "image input requires a vision-capable model".to_owned(),
                code: "vision_unsupported",
            });
        }
        if options.logprobs == Some(true) {
            return Err(unsupported("logprobs are not supported for this model"));
        }
        if !options.stop_strings.is_empty() {
            return Err(unsupported("stop is not supported for this model"));
        }
        if options.reasoning_budget.is_some() {
            return Err(unsupported(
                "reasoning_budget is not supported for this model; \
                 chat_template_kwargs.enable_thinking turns thinking on or off",
            ));
        }
        let max_tokens = crate::serve::contract::apply_max_tokens_policy(
            options.max_tokens.unwrap_or(generation.max_tokens),
            options.max_tokens_policy,
        )?;
        Ok(ValidatedChatRequest {
            messages: options.messages,
            max_tokens,
            total_token_cap: options.total_token_cap_requested.then_some(max_tokens),
            temperature: crate::serve::contract::validate_temperature(
                options.temperature.unwrap_or(generation.temperature),
            )?,
            top_k: options.top_k.unwrap_or(generation.top_k),
            top_p: crate::serve::contract::validate_top_p(
                options.top_p.unwrap_or(generation.top_p),
            )?,
            min_p: options.min_p,
            repetition_penalty: options
                .repetition_penalty
                .unwrap_or(generation.repetition_penalty),
            seed: options.seed,
            stream: options.stream.unwrap_or(false),
            stop_strings: Vec::new(),
            reasoning_budget: None,
            logprobs: None,
        })
    }

    fn generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig {
        let mut c = GenerateConfig::default();
        c.max_new_tokens = req.max_tokens;
        c.total_token_cap = req.total_token_cap;
        c.temperature = req.temperature;
        c.top_k = req.top_k;
        c.top_p = req.top_p;
        if let Some(min_p) = req.min_p {
            c.min_p = min_p;
        }
        c.repetition_penalty = req.repetition_penalty;
        c.seed = req.seed;
        c.stop_token_ids = self.stop_token_ids.clone();
        c.enable_thinking = false;
        c.enable_mtp = None;
        c.grammar = None;
        c.stop_strings = Vec::new();
        c.reasoning_budget = None;
        c.logprobs = None;
        c
    }
}

/// A `lattice serve` request prepared for the Inkling 2 CPU route.
#[derive(Debug)]
pub(crate) struct PreparedInkling2Request {
    pub(crate) prompt: String,
    pub(crate) gen_cfg: GenerateConfig,
    pub(crate) stream: bool,
}

/// Typed content parts are refused rather than flattened: the template writes each text part as
/// its own message, which the flattened message text cannot reproduce.
fn refuse_typed_parts(req: &ChatRequest) -> Result<(), ApiError> {
    if req
        .messages
        .iter()
        .any(|message| matches!(message.content, MessageContent::Parts(_)))
    {
        return Err(unsupported(
            "typed content parts are not supported for this model; send message content as a \
             string",
        ));
    }
    Ok(())
}

/// Validate `req` under the `lattice serve` profile, refuse every control the Inkling 2 route
/// cannot serve, render the prompt with the thinking switch the request sent and check it against
/// the context window.
pub(crate) fn prepare_inkling2_chat_request(
    adapter: &Inkling2PromptAdapter,
    req: &ChatRequest,
    model_id: &str,
    default_max_tokens: usize,
    max_tokens_cap: usize,
    tokenize_len: impl FnOnce(&str) -> usize,
    max_context: impl FnOnce() -> usize,
) -> Result<PreparedInkling2Request, ApiError> {
    let options =
        normalize_requested_options(req, ServeProfile::lattice(model_id, max_tokens_cap))?;
    refuse_typed_parts(req)?;
    let enable_thinking = options.enable_thinking != Some(false);
    let validated =
        adapter.apply_defaults(GenerationDefaults::standard(default_max_tokens), options)?;
    let prompt = adapter.render_with_thinking(&validated.messages, enable_thinking);
    crate::serve::contract::validate_context_window_with_total_cap(
        tokenize_len(&prompt),
        validated.max_tokens,
        validated.reasoning_budget,
        validated.total_token_cap,
        max_context(),
    )?;
    Ok(PreparedInkling2Request {
        gen_cfg: adapter.generate_config(&validated),
        stream: validated.stream,
        prompt,
    })
}

/// Validate `req` under the standalone server's profile and apply the Inkling 2 defaults step.
/// The standalone server refuses Inkling 2 checkpoints at startup, so no request reaches this
/// from it; the preparation contract still answers with the same refusals.
pub(crate) fn normalize_standalone_inkling2(
    adapter: &Inkling2PromptAdapter,
    req: &ChatRequest,
    defaults: GenerationDefaults,
    model_id: &str,
    max_context: usize,
) -> Result<ValidatedChatRequest, ApiError> {
    let options = normalize_requested_options(
        req,
        ServeProfile::lattice_serve(model_id, max_context).with_vision_support(false),
    )?;
    refuse_typed_parts(req)?;
    adapter.apply_defaults(defaults, options)
}
