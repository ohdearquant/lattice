//! The Inkling 2 chat prompt and the request controls its CPU route serves.
//!
//! The prompt is the text-only subset of the checkpoint's published chat template: each message
//! is its role marker, `<|content_text|>`, the content and `<|end_message|>`, and an assistant
//! message is followed by `<|content_model_end_sampling|>`. One `Thinking effort level` system
//! message goes before the first message that is not a system message, or after the last message
//! when every message is a system message. Nothing follows the last message: the model opens its
//! own turn.
//!
//! Message text never becomes a marker, which departs from the published template on purpose.
//! `PromptBuilder` is the one writer of the prompt, and its only ways to add are a marker and
//! text. Each marker is framed by the noncharacters U+FDD0 and U+FDD1, and text holding either is
//! refused, so `prompt_pieces` reads back exactly the markers the builder wrote. The CPU route
//! tokenizes a marker as its id and text with no added-token matching.

use crate::serve::ApiError;
use crate::serve::contract::{
    ChatRequest, GenerationDefaults, MessageContent, NormalizedChatMessage, NormalizedChatRole,
    RequestedChatOptions, ServeProfile, ValidatedChatRequest, normalize_requested_options,
};
use crate::serving_provider::providers::gemma::template_trim;
use lattice_inference::error::InferenceError;
use lattice_inference::generation::GenerateConfig;
use serde_json::value::RawValue;
use std::path::Path;

const EFFORT_PREFIX: &str = "Thinking effort level: ";
/// The effort the published template writes when a request sets none.
const DEFAULT_EFFORT: &str = "0.9";
/// The effort written for a request that turns thinking off.
const THINKING_OFF_EFFORT: &str = "0";
/// Each `reasoning_effort` string the published template accepts, with the number it writes.
const EFFORT_LEVELS: [(&str, &str); 7] = [
    ("none", THINKING_OFF_EFFORT),
    ("minimal", "0.1"),
    ("low", "0.2"),
    ("medium", "0.7"),
    ("high", "0.9"),
    ("xhigh", "0.99"),
    ("max", "0.99"),
];
const EFFORT_NAMES: &str = "none, minimal, low, medium, high, xhigh, max";

/// A marker the prompt template writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Marker {
    MessageSystem,
    MessageUser,
    MessageModel,
    ContentText,
    EndMessage,
    EndSampling,
}

impl Marker {
    /// Every marker, in declaration order.
    pub(crate) const ALL: [Self; 6] = [
        Self::MessageSystem,
        Self::MessageUser,
        Self::MessageModel,
        Self::ContentText,
        Self::EndMessage,
        Self::EndSampling,
    ];

    /// The marker's token as the tokenizer spells it.
    pub(crate) const fn spelling(self) -> &'static str {
        match self {
            Self::MessageSystem => "<|message_system|>",
            Self::MessageUser => "<|message_user|>",
            Self::MessageModel => "<|message_model|>",
            Self::ContentText => "<|content_text|>",
            Self::EndMessage => "<|end_message|>",
            Self::EndSampling => "<|content_model_end_sampling|>",
        }
    }
}

pub(crate) use framing::{Inkling2Prompt, PromptPiece, prompt_pieces};
use framing::{PromptBuilder, codepoint, is_frame};

/// The rendered prompt and its one writer, in a module of their own so that no other code can
/// write to a prompt.
mod framing {
    use super::Marker;
    use crate::serve::ApiError;

    /// Opens a marker in the rendered prompt. U+FDD0 and U+FDD1 are Unicode noncharacters, which the
    /// standard reserves for a program's internal use.
    const MARKER_OPEN: char = '\u{FDD0}';
    /// Closes a marker in the rendered prompt.
    const MARKER_CLOSE: char = '\u{FDD1}';

    /// A rendered prompt, each marker framed by U+FDD0 and U+FDD1.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct Inkling2Prompt {
        framed: String,
    }

    impl Inkling2Prompt {
        /// The framed prompt.
        pub(crate) fn framed(&self) -> &str {
            &self.framed
        }

        /// The prompt as the published template writes it, each marker as its spelling.
        #[cfg(test)]
        pub(crate) fn template_text(&self) -> String {
            self.framed.chars().filter(|&c| !is_frame(c)).collect()
        }

        /// The framed prompt, which [`prompt_pieces`] reads back.
        pub(crate) fn into_framed(self) -> String {
            self.framed
        }
    }

    /// The one writer of an [`Inkling2Prompt`].
    pub(super) struct PromptBuilder {
        framed: String,
    }

    impl PromptBuilder {
        pub(super) fn new() -> Self {
            Self {
                framed: String::new(),
            }
        }

        pub(super) fn marker(&mut self, marker: Marker) {
            self.framed.push(MARKER_OPEN);
            self.framed.push_str(marker.spelling());
            self.framed.push(MARKER_CLOSE);
        }

        /// Appends `text`.
        ///
        /// # Errors
        /// `invalid_messages` when `text` holds U+FDD0 or U+FDD1; nothing is appended.
        pub(super) fn text(&mut self, text: &str) -> Result<(), ApiError> {
            if let Some(c) = text.chars().find(|&c| is_frame(c)) {
                return Err(ApiError::BadRequest {
                    message: format!(
                        "the prompt text contains {}, a noncharacter this model's prompt reserves \
                     for its markers",
                        codepoint(c)
                    ),
                    code: "invalid_messages",
                });
            }
            self.framed.push_str(text);
            Ok(())
        }

        pub(super) fn finish(self) -> Inkling2Prompt {
            Inkling2Prompt {
                framed: self.framed,
            }
        }
    }

    pub(super) fn is_frame(c: char) -> bool {
        c == MARKER_OPEN || c == MARKER_CLOSE
    }

    pub(super) fn codepoint(c: char) -> String {
        format!("U+{:04X}", u32::from(c))
    }

    /// One piece of a framed prompt.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum PromptPiece<'a> {
        Marker(Marker),
        Text(&'a str),
    }

    /// The markers and text of a framed prompt, in order. A frame around anything but a marker's
    /// spelling, and a noncharacter outside a frame, are text; `PromptBuilder` writes neither.
    pub(crate) fn prompt_pieces(framed: &str) -> Vec<PromptPiece<'_>> {
        let mut pieces = Vec::new();
        let mut text_start = 0;
        let mut pos = 0;
        while let Some(offset) = framed[pos..].find(MARKER_OPEN) {
            let open = pos + offset;
            let name_start = open + MARKER_OPEN.len_utf8();
            let marker = framed[name_start..].find(MARKER_CLOSE).and_then(|len| {
                let name = &framed[name_start..name_start + len];
                Marker::ALL
                    .into_iter()
                    .find(|marker| marker.spelling() == name)
                    .map(|marker| (marker, name_start + len + MARKER_CLOSE.len_utf8()))
            });
            match marker {
                Some((marker, end)) => {
                    if text_start < open {
                        pieces.push(PromptPiece::Text(&framed[text_start..open]));
                    }
                    pieces.push(PromptPiece::Marker(marker));
                    text_start = end;
                    pos = end;
                }
                None => pos = name_start,
            }
        }
        if text_start < framed.len() {
            pieces.push(PromptPiece::Text(&framed[text_start..]));
        }
        pieces
    }
}

/// Refuses a message whose content holds U+FDD0 or U+FDD1, naming the message. The prompt
/// builder refuses the same text; this check answers first with the message's position.
fn refuse_reserved_characters(messages: &[NormalizedChatMessage]) -> Result<(), ApiError> {
    for (index, message) in messages.iter().enumerate() {
        if let Some(c) = message.content.chars().find(|&c| is_frame(c)) {
            return Err(ApiError::BadRequest {
                message: format!(
                    "messages[{index}] content contains {}, a noncharacter this model's prompt \
                     reserves for its markers",
                    codepoint(c)
                ),
                code: "invalid_messages",
            });
        }
    }
    Ok(())
}

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

    /// The prompt for `messages`, with `effort` written in the effort message.
    ///
    /// # Errors
    /// `invalid_messages` when a message's content or `effort` holds U+FDD0 or U+FDD1.
    pub(crate) fn render(
        &self,
        messages: &[NormalizedChatMessage],
        effort: &str,
    ) -> Result<Inkling2Prompt, ApiError> {
        let mut prompt = PromptBuilder::new();
        let mut effort_written = false;
        for message in messages {
            if !effort_written && message.role != NormalizedChatRole::System {
                push_effort(&mut prompt, effort)?;
                effort_written = true;
            }
            match message.role {
                NormalizedChatRole::System => {
                    push_message(&mut prompt, Marker::MessageSystem, &message.content)?;
                }
                NormalizedChatRole::User => {
                    push_message(&mut prompt, Marker::MessageUser, &message.content)?;
                }
                NormalizedChatRole::Assistant => {
                    push_message(&mut prompt, Marker::MessageModel, &message.content)?;
                    prompt.marker(Marker::EndSampling);
                }
            }
        }
        if !effort_written {
            push_effort(&mut prompt, effort)?;
        }
        Ok(prompt.finish())
    }

    /// Token ids that end the model's turn.
    pub(crate) fn stop_token_ids(&self) -> &[u32] {
        &self.stop_token_ids
    }
}

fn push_message(prompt: &mut PromptBuilder, role: Marker, content: &str) -> Result<(), ApiError> {
    prompt.marker(role);
    prompt.marker(Marker::ContentText);
    prompt.text(content)?;
    prompt.marker(Marker::EndMessage);
    Ok(())
}

fn push_effort(prompt: &mut PromptBuilder, effort: &str) -> Result<(), ApiError> {
    prompt.marker(Marker::MessageSystem);
    prompt.marker(Marker::ContentText);
    prompt.text(EFFORT_PREFIX)?;
    prompt.text(effort)?;
    prompt.marker(Marker::EndMessage);
    Ok(())
}

/// The effort a request writes. A `reasoning_effort` string is trimmed as the template's `trim`
/// filter trims it and written with that level's number. Without one, thinking off writes 0 and
/// thinking on or no switch writes the template's default, 0.9.
///
/// # Errors
/// `invalid_request_body` when `reasoning_effort` is not a string, `invalid_request` when it is
/// not one of the template's levels or contradicts `enable_thinking`.
fn requested_effort(
    reasoning_effort: Option<&RawValue>,
    enable_thinking: Option<bool>,
) -> Result<&'static str, ApiError> {
    let Some(raw) = reasoning_effort else {
        return Ok(if enable_thinking == Some(false) {
            THINKING_OFF_EFFORT
        } else {
            DEFAULT_EFFORT
        });
    };
    let Ok(sent) = serde_json::from_str::<String>(raw.get()) else {
        return Err(ApiError::BadRequest {
            message: format!("reasoning_effort must be a string, one of {EFFORT_NAMES}"),
            code: "invalid_request_body",
        });
    };
    let key = template_trim(&sent);
    let Some(&(level, number)) = EFFORT_LEVELS.iter().find(|(level, _)| *level == key) else {
        return Err(ApiError::BadRequest {
            message: format!("reasoning_effort is not one of {EFFORT_NAMES}"),
            code: "invalid_request",
        });
    };
    if let Some(thinking) = enable_thinking
        && thinking == (level == "none")
    {
        return Err(ApiError::BadRequest {
            message: format!(
                "reasoning_effort \"{level}\" conflicts with chat_template_kwargs.enable_thinking \
                 {thinking}; send only one of them"
            ),
            code: "invalid_request",
        });
    }
    Ok(number)
}

fn unsupported(message: &str) -> ApiError {
    ApiError::BadRequest {
        message: message.to_owned(),
        code: "unsupported_feature",
    }
}

impl Inkling2PromptAdapter {
    /// Effective request values, with every control the route cannot serve refused.
    pub(crate) fn apply_defaults(
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

    /// The `GenerateConfig` for an effective request.
    pub(crate) fn generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig {
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
    pub(crate) prompt: Inkling2Prompt,
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
/// cannot serve, render the prompt with the effort the request sent and check it against the
/// context window.
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
    refuse_reserved_characters(&options.messages)?;
    let effort = requested_effort(req.reasoning_effort.as_deref(), options.enable_thinking)?;
    let validated =
        adapter.apply_defaults(GenerationDefaults::standard(default_max_tokens), options)?;
    let prompt = adapter.render(&validated.messages, effort)?;
    crate::serve::contract::validate_context_window_with_total_cap(
        tokenize_len(prompt.framed()),
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
    refuse_reserved_characters(&options.messages)?;
    requested_effort(req.reasoning_effort.as_deref(), options.enable_thinking)?;
    adapter.apply_defaults(defaults, options)
}
