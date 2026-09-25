//! Chat-request preparation shared by the serving binaries.
//!
//! [`prepare_chat_request`] is the `lattice serve` chat handler's
//! pre-generation cascade (validate, render, tokenize, context-window check)
//! and [`build_cfg`] is the `lattice_serve` chat handler's
//! `ValidatedChatRequest`-to-`GenerateConfig` mapping. Both live in the library
//! so that the handlers and the `bench_serve_prepare` measurement example call
//! the same code. They are `#[doc(hidden)]`: they mirror each binary's
//! current handler behaviour and are not a stable API.

use crate::forward::metal_qwen35::ChatMessage;
use crate::generation::GenerateConfig;
use crate::model::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID;
use crate::serve::ApiError;
use crate::serve::contract::{
    ChatRequest as ChatCompletionRequest, GenerationDefaults, ServeProfile,
    ValidatedChatRequest as ContractValidatedChatRequest,
    normalize_request_with_context_and_budget, validate_context_window_with_budget,
};
use crate::serve::{format_normalized_chat_template, into_engine_chat_messages};

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
            let prompt = format_normalized_chat_template(messages);
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
#[allow(clippy::field_reassign_with_default)]
pub fn lattice_gen_cfg(
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: Option<u64>,
    stop_strings: Vec<String>,
    reasoning_budget: Option<usize>,
    logprobs: Option<usize>,
) -> GenerateConfig {
    let mut gen_cfg = GenerateConfig::default();
    gen_cfg.max_new_tokens = max_tokens;
    gen_cfg.temperature = temperature;
    gen_cfg.top_p = top_p;
    gen_cfg.seed = seed;
    gen_cfg.stop_strings = stop_strings;
    gen_cfg.reasoning_budget = reasoning_budget;
    gen_cfg.logprobs = logprobs;
    gen_cfg
}

#[doc(hidden)]
#[allow(clippy::field_reassign_with_default)]
pub fn build_cfg(req: &ValidatedChatRequest) -> GenerateConfig {
    let mut cfg = GenerateConfig::default();
    cfg.max_new_tokens = req.max_tokens;
    cfg.temperature = req.temperature;
    cfg.top_k = req.top_k;
    cfg.top_p = req.top_p;
    cfg.repetition_penalty = req.repetition_penalty;
    cfg.seed = req.seed;
    cfg.stop_token_ids = vec![QWEN_CHAT_IM_END_TOKEN_ID];
    cfg.enable_thinking = true;
    cfg.enable_mtp = None;
    cfg.grammar = None;
    cfg.stop_strings = req.stop_strings.clone();
    cfg.reasoning_budget = req.reasoning_budget;
    cfg.logprobs = req.logprobs;
    cfg
}
