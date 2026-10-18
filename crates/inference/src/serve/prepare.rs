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
//! preparation entry, called by `serving_cpu` for `lattice serve` and by
//! tests and the measurement example.
//!
//! [`prepare_chat_request`]: crate::serve::prepare::prepare_chat_request
//! [`build_cfg`]: crate::serve::prepare::build_cfg
//! [`prepare_gemma_chat_request`]: crate::serve::prepare::prepare_gemma_chat_request

use crate::forward::metal_qwen35::ChatMessage;
use crate::generation::GenerateConfig;
use crate::serve::ApiError;
#[cfg(test)]
use crate::serve::contract::GenerationDefaults;
use crate::serve::contract::{
    ChatRequest as ChatCompletionRequest, ServeProfile,
    ValidatedChatRequest as ContractValidatedChatRequest,
};

pub use crate::serve::prompt_adapter::GemmaPromptAdapter;
pub use crate::serving_preparation::PreparationHandle;
use crate::serving_preparation::PreparedCpuChat;

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
    crate::serving_provider::providers::qwen::prepare_chat_request(
        req,
        model_id,
        default_max_tokens,
        max_tokens_cap,
        vision_supported,
        tokenize_len,
        max_context,
    )
    .map(|(prepared, _)| prepared)
}

/// Retain the effective template mode beside the prepared prompt in its config.
#[doc(hidden)]
pub fn prepare_chat_request_with_config(
    req: &ChatCompletionRequest,
    model_id: &str,
    default_max_tokens: usize,
    max_tokens_cap: usize,
    vision_supported: bool,
    tokenize_len: impl FnOnce(&str) -> usize,
    max_context: impl FnOnce() -> usize,
) -> Result<PreparedCpuChat, ApiError> {
    let (prepared, enable_thinking) =
        crate::serving_provider::providers::qwen::prepare_chat_request(
            req,
            model_id,
            default_max_tokens,
            max_tokens_cap,
            vision_supported,
            tokenize_len,
            max_context,
        )?;
    let config = lattice_gen_cfg_with_thinking(
        prepared.max_tokens,
        prepared.temperature,
        prepared.top_p,
        prepared.seed,
        prepared.stop_strings.clone(),
        prepared.reasoning_budget,
        prepared.logprobs,
        enable_thinking,
    );
    Ok(PreparedCpuChat { prepared, config })
}

/// Keep the standalone Qwen wire options and config in one preparation step.
#[doc(hidden)]
pub fn prepare_standalone_chat_request(
    req: &ChatCompletionRequest,
    defaults: crate::serve::contract::GenerationDefaults,
    profile: ServeProfile<'_>,
) -> Result<(ValidatedChatRequest, GenerateConfig), ApiError> {
    crate::serving_provider::providers::qwen::prepare_standalone_chat_request(
        req, defaults, profile,
    )
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
    lattice_gen_cfg_with_thinking(
        max_tokens,
        temperature,
        top_p,
        seed,
        stop_strings,
        reasoning_budget,
        logprobs,
        true,
    )
}

/// Map prepared CLI options using the request's effective thinking mode.
#[doc(hidden)]
pub fn lattice_gen_cfg_with_thinking(
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: Option<u64>,
    stop_strings: Vec<String>,
    reasoning_budget: Option<usize>,
    logprobs: Option<usize>,
    enable_thinking: bool,
) -> GenerateConfig {
    crate::serving_provider::providers::qwen::lattice_gen_cfg(
        max_tokens,
        temperature,
        top_p,
        seed,
        stop_strings,
        reasoning_budget,
        logprobs,
        enable_thinking,
    )
}

#[doc(hidden)]
pub fn build_cfg(req: &ValidatedChatRequest) -> GenerateConfig {
    crate::serving_provider::providers::qwen::build_cfg(req)
}

/// Whether this request explicitly enabled thinking in its chat template.
#[doc(hidden)]
pub fn explicitly_enabled_thinking(req: &ChatCompletionRequest) -> Result<bool, ApiError> {
    Ok(
        crate::serve::contract::parse_chat_template_kwargs(&req.chat_template_kwargs)?
            == Some(true),
    )
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
    crate::serving_provider::providers::gemma::prepare_gemma_chat_request(
        adapter,
        req,
        model_id,
        default_max_tokens,
        max_tokens_cap,
        tokenize_len,
        max_context,
    )
}

/// The refusal a model without runtime LoRA adapters owes every adapter
/// request. The code is the one `lattice serve` answers with on a backend
/// that cannot take an adapter.
#[doc(hidden)]
pub fn lora_unsupported_backend() -> ApiError {
    crate::serving_provider::providers::gemma::lora_unsupported_backend()
}

#[cfg(test)]
mod tests;
