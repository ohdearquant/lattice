use crate::generation::GenerateConfig;
use crate::model::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID;
use crate::serve::ApiError;
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use crate::serve::contract::normalize_request;
use crate::serve::contract::{
    ChatRequest, GenerationDefaults, MaxTokensPolicy, NormalizedChatMessage, RequestedChatOptions,
    ServeProfile, ValidatedChatRequest, apply_max_tokens_policy,
    normalize_request_with_context_and_budget, validate_context_window_with_budget,
    validate_temperature, validate_top_p,
};
use crate::serve::format_normalized_chat_template;
use crate::serve::into_engine_chat_messages;
use crate::serve::prepare::PreparedChatRequest;
use crate::serve::prompt_adapter::PromptAdapter;
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use crate::serving_preparation::{PreparationHandle, RequestPreparation};
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use crate::tokenizer::Tokenizer as _;
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use crate::tokenizer::bpe::BpeTokenizer;
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub(crate) struct QwenPromptAdapter;

impl QwenPromptAdapter {
    const STOP_TOKEN_IDS: [u32; 1] = [QWEN_CHAT_IM_END_TOKEN_ID];

    #[allow(clippy::field_reassign_with_default)]
    pub(crate) fn lattice_generate_config(
        &self,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        seed: Option<u64>,
        stop_strings: Vec<String>,
        reasoning_budget: Option<usize>,
        logprobs: Option<usize>,
    ) -> GenerateConfig {
        let mut cfg = self.generate_config_base();
        cfg.max_new_tokens = max_tokens;
        cfg.temperature = temperature;
        cfg.top_p = top_p;
        cfg.seed = seed;
        cfg.stop_strings = stop_strings;
        cfg.reasoning_budget = reasoning_budget;
        cfg.logprobs = logprobs;
        cfg
    }

    #[allow(clippy::field_reassign_with_default)]
    fn generate_config_base(&self) -> GenerateConfig {
        let mut cfg = GenerateConfig::default();
        cfg.stop_token_ids = self.stop_token_ids().to_vec();
        cfg.enable_thinking = true;
        cfg
    }
}

impl PromptAdapter for QwenPromptAdapter {
    fn render(&self, messages: &[NormalizedChatMessage]) -> String {
        format_normalized_chat_template(messages)
    }

    fn stop_token_ids(&self) -> &[u32] {
        &Self::STOP_TOKEN_IDS
    }

    fn apply_defaults(
        &self,
        generation: GenerationDefaults,
        options: RequestedChatOptions,
    ) -> Result<ValidatedChatRequest, ApiError> {
        QwenChatDefaults::new(generation).apply(options)
    }

    #[allow(clippy::field_reassign_with_default)]
    fn generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig {
        let mut cfg = self.generate_config_base();
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
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QwenChatDefaults {
    generation: GenerationDefaults,
}

impl QwenChatDefaults {
    pub(crate) const fn new(generation: GenerationDefaults) -> Self {
        Self { generation }
    }

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
}

#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
struct QwenPreparation {
    tokenizer: Arc<BpeTokenizer>,
    model_max_context: usize,
}

#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
impl RequestPreparation for QwenPreparation {
    fn tokenize_len(&self, prompt: &str) -> usize {
        self.tokenizer.tokenize(prompt).pre_truncation_len
    }

    fn prepare_lattice(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
        vision_supported: bool,
    ) -> Result<PreparedChatRequest, ApiError> {
        prepare_chat_request(
            req,
            model_id,
            default_max_tokens,
            max_tokens_cap,
            vision_supported,
            |prompt| self.tokenize_len(prompt),
            || self.model_max_context,
        )
    }

    fn normalize_standalone(
        &self,
        req: &ChatRequest,
        defaults: GenerationDefaults,
        model_id: &str,
        vision_supported: bool,
    ) -> Result<ValidatedChatRequest, ApiError> {
        normalize_request(
            req,
            defaults,
            ServeProfile::lattice_serve(model_id, self.model_max_context)
                .with_vision_support(vision_supported),
        )
    }

    fn standalone_generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig {
        build_cfg(req)
    }

    fn lattice_generate_config(
        &self,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        seed: Option<u64>,
        stop_strings: Vec<String>,
        reasoning_budget: Option<usize>,
        logprobs: Option<usize>,
    ) -> GenerateConfig {
        lattice_gen_cfg(
            max_tokens,
            temperature,
            top_p,
            seed,
            stop_strings,
            reasoning_budget,
            logprobs,
        )
    }

    fn refuse_standalone_unsupported(&self, _req: &ChatRequest) -> Result<(), ApiError> {
        Ok(())
    }

    fn supports_adapters(&self) -> bool {
        true
    }
}

#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
impl PreparationHandle {
    pub(crate) fn qwen(tokenizer: Arc<BpeTokenizer>, model_max_context: usize) -> Self {
        Self::new(Arc::new(QwenPreparation {
            tokenizer,
            model_max_context,
        }))
    }
}

pub(crate) fn prepare_chat_request(
    req: &ChatRequest,
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
    let ValidatedChatRequest {
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

pub(crate) fn lattice_gen_cfg(
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

pub(crate) fn build_cfg(req: &ValidatedChatRequest) -> GenerateConfig {
    QwenPromptAdapter.generate_config(req)
}
