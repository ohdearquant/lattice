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
//! `QwenChatDefaults` is the one place Qwen's chat defaults are applied:
//! the server's sampling defaults for options a request omitted, the
//! thinking switch and the `<|im_end|>` stop token.

use crate::forward::metal_qwen35::ChatMessage;
use crate::generation::GenerateConfig;
use crate::model::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID;
use crate::serve::ApiError;
use crate::serve::contract::{
    ChatRequest as ChatCompletionRequest, GenerationDefaults, MaxTokensPolicy,
    RequestedChatOptions, ServeProfile, ValidatedChatRequest as ContractValidatedChatRequest,
    apply_max_tokens_policy, normalize_request_with_context_and_budget,
    validate_context_window_with_budget, validate_temperature, validate_top_p,
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
pub fn lattice_gen_cfg(
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: Option<u64>,
    stop_strings: Vec<String>,
    reasoning_budget: Option<usize>,
    logprobs: Option<usize>,
) -> GenerateConfig {
    QwenChatDefaults::lattice_generate_config(
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
    QwenChatDefaults::generate_config(req)
}

/// Qwen's chat defaults step: turns [`RequestedChatOptions`] plus the
/// server's [`GenerationDefaults`] into the effective request values, and
/// the effective values into a Qwen `GenerateConfig`.
///
/// This is the only place a Qwen default is applied. Request validation
/// calls [`Self::max_tokens`], [`Self::temperature`], [`Self::top_p`] and
/// [`Self::reasoning_budget`] at the position each option has always been
/// resolved, so a refusal caused by a default keeps its precedence.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QwenChatDefaults {
    generation: GenerationDefaults,
}

impl QwenChatDefaults {
    pub(crate) const fn new(generation: GenerationDefaults) -> Self {
        Self { generation }
    }

    /// Effective generation-token budget under the profile's policy.
    pub(crate) fn max_tokens(
        &self,
        requested: Option<usize>,
        policy: MaxTokensPolicy,
    ) -> Result<usize, ApiError> {
        apply_max_tokens_policy(requested.unwrap_or(self.generation.max_tokens), policy)
    }

    pub(crate) fn temperature(&self, requested: Option<f32>) -> Result<f32, ApiError> {
        validate_temperature(requested.unwrap_or(self.generation.temperature))
    }

    pub(crate) fn top_p(&self, requested: Option<f32>) -> Result<f32, ApiError> {
        validate_top_p(requested.unwrap_or(self.generation.top_p))
    }

    /// Effective reasoning budget: a requested `0` means "use the default",
    /// and a context-clamped profile leaves room for the generation budget
    /// and the closing token.
    pub(crate) fn reasoning_budget(
        &self,
        requested: Option<usize>,
        supported: bool,
        policy: MaxTokensPolicy,
        max_tokens: usize,
    ) -> Option<usize> {
        let mut reasoning_budget = if supported {
            requested
                .filter(|&value| value > 0)
                .or(self.generation.reasoning_budget)
        } else {
            None
        };
        if let MaxTokensPolicy::ClampToContext { context } = policy {
            let reasoning_room = context.saturating_sub(max_tokens).saturating_sub(1);
            reasoning_budget = reasoning_budget
                .map(|value| value.min(reasoning_room))
                .filter(|&value| value > 0);
        }
        reasoning_budget
    }

    /// Effective request values for validated options.
    pub(crate) fn apply(
        &self,
        options: RequestedChatOptions,
    ) -> Result<ValidatedChatRequest, ApiError> {
        let max_tokens = self.max_tokens(options.max_tokens, options.max_tokens_policy)?;
        let reasoning_budget = self.reasoning_budget(
            options.reasoning_budget,
            options.reasoning_budget_supported,
            options.max_tokens_policy,
            max_tokens,
        );
        let logprobs = if options.logprobs.unwrap_or(false) {
            Some(options.top_logprobs.unwrap_or(0))
        } else {
            None
        };
        Ok(ValidatedChatRequest {
            messages: options.messages,
            max_tokens,
            temperature: self.temperature(options.temperature)?,
            top_k: options.top_k.unwrap_or(self.generation.top_k),
            top_p: self.top_p(options.top_p)?,
            repetition_penalty: options
                .repetition_penalty
                .unwrap_or(self.generation.repetition_penalty),
            seed: options.seed,
            stream: options.stream.unwrap_or(false),
            stop_strings: options.stop_strings,
            reasoning_budget,
            logprobs,
        })
    }

    /// The `lattice_serve` handler's `GenerateConfig` for a validated request.
    #[allow(clippy::field_reassign_with_default)]
    fn generate_config(req: &ValidatedChatRequest) -> GenerateConfig {
        let mut cfg = Self::generate_config_base();
        cfg.max_new_tokens = req.max_tokens;
        cfg.temperature = req.temperature;
        cfg.top_k = req.top_k;
        cfg.top_p = req.top_p;
        cfg.repetition_penalty = req.repetition_penalty;
        cfg.seed = req.seed;
        cfg.enable_mtp = None;
        cfg.grammar = None;
        cfg.stop_strings = req.stop_strings.clone();
        cfg.reasoning_budget = req.reasoning_budget;
        cfg.logprobs = req.logprobs;
        cfg
    }

    /// The `lattice serve` handler's `GenerateConfig` for prepared sampling
    /// fields.
    #[allow(clippy::field_reassign_with_default)]
    fn lattice_generate_config(
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        seed: Option<u64>,
        stop_strings: Vec<String>,
        reasoning_budget: Option<usize>,
        logprobs: Option<usize>,
    ) -> GenerateConfig {
        let mut cfg = Self::generate_config_base();
        cfg.max_new_tokens = max_tokens;
        cfg.temperature = temperature;
        cfg.top_p = top_p;
        cfg.seed = seed;
        cfg.stop_strings = stop_strings;
        cfg.reasoning_budget = reasoning_budget;
        cfg.logprobs = logprobs;
        cfg
    }

    /// Qwen chat generation: thinking on, stopping at `<|im_end|>`.
    #[allow(clippy::field_reassign_with_default)]
    fn generate_config_base() -> GenerateConfig {
        let mut cfg = GenerateConfig::default();
        cfg.stop_token_ids = vec![QWEN_CHAT_IM_END_TOKEN_ID];
        cfg.enable_thinking = true;
        cfg
    }
}
