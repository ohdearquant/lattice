//! Chat-request preparation shared by the serving binaries.
//!
//! [`prepare_chat_request`] is the `lattice serve` chat handler's
//! pre-generation cascade (validate, render, tokenize, context-window check)
//! and [`build_cfg`] is the `lattice_serve` chat handler's
//! `ValidatedChatRequest`-to-`GenerateConfig` mapping. Both live in the library
//! so that the handlers and the `bench_serve_prepare` measurement example call
//! the same code. They are `#[doc(hidden)]`: they mirror each binary's
//! current handler behaviour and are not a stable API.
//!
//! Family chat conventions (rendering, stop tokens, defaults) come from the
//! model's prompt adapter in `serve::prompt_adapter`; nothing here supplies a
//! family default. [`prepare_gemma_chat_request`] is the Gemma E2B text
//! preparation entry, used by tests and the measurement example until the
//! serving binaries route Gemma.

use crate::forward::metal_qwen35::ChatMessage;
use crate::generation::GenerateConfig;
use crate::serve::ApiError;
use crate::serve::contract::{
    ChatRequest as ChatCompletionRequest, GenerationDefaults, MessageContent, ServeProfile,
    ValidatedChatRequest as ContractValidatedChatRequest,
    normalize_request_with_context_and_budget, normalize_requested_options,
    validate_context_window_with_budget,
};
use crate::serve::into_engine_chat_messages;
use crate::serve::prompt_adapter::{PromptAdapter as _, QwenPromptAdapter};

pub use crate::serve::prompt_adapter::GemmaPromptAdapter;

/// The `lattice_serve` handler's name for the validated request type.
type ValidatedChatRequest = ContractValidatedChatRequest;

/// Output of the full pre-generation validation cascade, ready for
/// `gen_cfg` construction.
#[doc(hidden)]
#[derive(Debug)]
pub struct PreparedChatRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub logprobs: Option<usize>,
    pub prompt: String,
    pub stop_strings: Vec<String>,
    pub reasoning_budget: Option<usize>,
    pub seed: Option<u64>,
    pub stream: bool,
}

/// Production entry point for the shared context-aware normalization
/// cascade: supplies the prompt-aware context-window check (rendering the
/// chat template, tokenizing it, then calling the shared
/// `validate_context_window_with_budget`) as the context check, in the
/// exact order the original inline `chat_completions` cascade used:
/// `stop` is validated *last*, after both the served-model hard
/// requirements and the context-window check that guards against a
/// panic in the blocking generation path. A request that is both
/// over-context and carries a malformed `stop` field must fail with
/// `context_length_exceeded`, not a stop-parsing error — pinned by
/// `cm_serve_context_window_checked_before_stop_parsing`.
///
/// `tokenize_len`/`max_context` are threaded through as thunks (rather
/// than a `&ModelBackend`) so this whole cascade — including the
/// ordering — is testable without constructing a real model: the
/// rendered `prompt` that `tokenize_len` needs only exists once
/// `validate_chat_request` has already run, so the thunk form lets a
/// test control the token count `check_context_window` sees without
/// having to fake a tokenizer.
#[doc(hidden)]
pub fn prepare_chat_request(
    req: &ChatCompletionRequest,
    model_id: &str,
    default_max_tokens: usize,
    max_tokens_cap: usize,
    vision_supported: bool,
    tokenize_len: impl FnOnce(&str) -> usize,
    max_context: impl FnOnce() -> usize,
) -> Result<PreparedChatRequest, ApiError> {
    let (validated, prompt) = normalize_request_with_context_and_budget(
        req,
        GenerationDefaults::standard(default_max_tokens),
        ServeProfile::lattice(model_id, max_tokens_cap).with_vision_support(vision_supported),
        |messages, max_tokens, reasoning_budget| {
            let prompt = QwenPromptAdapter.render(messages);
            let prompt_token_count = tokenize_len(&prompt);
            validate_context_window_with_budget(
                prompt_token_count,
                max_tokens,
                reasoning_budget,
                max_context(),
            )?;
            Ok(prompt)
        },
    )?;
    let ContractValidatedChatRequest {
        messages,
        max_tokens,
        temperature,
        top_p,
        logprobs,
        stop_strings,
        reasoning_budget,
        seed,
        stream,
        ..
    } = validated;
    let messages = into_engine_chat_messages(messages)?;

    Ok(PreparedChatRequest {
        messages,
        max_tokens,
        temperature,
        top_p,
        logprobs,
        prompt,
        stop_strings,
        reasoning_budget,
        seed,
        stream,
    })
}

/// The `lattice serve` chat handler's mapping from a prepared request's
/// sampling fields to its `GenerateConfig`.
#[doc(hidden)]
pub fn lattice_gen_cfg(
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: Option<u64>,
    stop_strings: Vec<String>,
    reasoning_budget: Option<usize>,
    logprobs: Option<usize>,
) -> GenerateConfig {
    QwenPromptAdapter.lattice_generate_config(
        max_tokens,
        temperature,
        top_p,
        seed,
        stop_strings,
        reasoning_budget,
        logprobs,
    )
}

#[doc(hidden)]
pub fn build_cfg(req: &ValidatedChatRequest) -> GenerateConfig {
    QwenPromptAdapter.generate_config(req)
}

/// Output of [`prepare_gemma_chat_request`]: the rendered prompt and the
/// `GenerateConfig` the Gemma adapter builds for it.
#[doc(hidden)]
#[derive(Debug)]
pub struct PreparedGemmaChatRequest {
    pub prompt: String,
    pub gen_cfg: GenerateConfig,
    pub stream: bool,
}

/// Gemma E2B text chat preparation under the `lattice serve` request
/// profile (text only): validate the request, refuse typed content parts
/// and every control the Gemma adapter cannot serve, apply the server's
/// sampling defaults through the adapter, render with the checkpoint's chat
/// template, then check the context window on the rendered prompt's token
/// count.
///
/// Typed content parts are refused rather than flattened: the template
/// trims each text part separately, which the flattened message text cannot
/// reproduce. Not a stable API; see [`GemmaPromptAdapter`].
#[doc(hidden)]
pub fn prepare_gemma_chat_request(
    adapter: &GemmaPromptAdapter,
    req: &ChatCompletionRequest,
    model_id: &str,
    default_max_tokens: usize,
    max_tokens_cap: usize,
    tokenize_len: impl FnOnce(&str) -> usize,
    max_context: impl FnOnce() -> usize,
) -> Result<PreparedGemmaChatRequest, ApiError> {
    let options =
        normalize_requested_options(req, ServeProfile::lattice(model_id, max_tokens_cap))?;
    if req
        .messages
        .iter()
        .any(|message| matches!(message.content, MessageContent::Parts(_)))
    {
        return Err(ApiError::BadRequest {
            message: "typed content parts are not supported for this model; send message \
                      content as a string"
                .to_string(),
            code: "unsupported_feature",
        });
    }
    let validated =
        adapter.apply_defaults(GenerationDefaults::standard(default_max_tokens), options)?;
    let prompt = adapter.render(&validated.messages);
    validate_context_window_with_budget(
        tokenize_len(&prompt),
        validated.max_tokens,
        validated.reasoning_budget,
        max_context(),
    )?;
    Ok(PreparedGemmaChatRequest {
        gen_cfg: adapter.generate_config(&validated),
        stream: validated.stream,
        prompt,
    })
}
