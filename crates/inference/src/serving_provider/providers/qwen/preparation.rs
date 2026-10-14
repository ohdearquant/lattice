use crate::generation::GenerateConfig;
use crate::model::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID;
use crate::serve::ApiError;
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use crate::serve::contract::normalize_request;
use crate::serve::contract::{
    ChatRequest, GenerationDefaults, MaxTokensPolicy, NormalizedChatMessage, RequestedChatOptions,
    ServeProfile, ValidatedChatRequest, apply_max_tokens_policy,
    normalize_request_with_context_and_thinking, validate_context_window_with_budget,
    validate_temperature, validate_top_p,
};
use crate::serve::format_normalized_chat_template;
use crate::serve::into_engine_chat_messages;
use crate::serve::prepare::PreparedChatRequest;
use crate::serve::prompt_adapter::{PromptAdapter, apply_qwen_thinking_mode};
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
        enable_thinking: bool,
    ) -> GenerateConfig {
        let mut cfg = self.generate_config_base();
        cfg.max_new_tokens = max_tokens;
        cfg.temperature = temperature;
        cfg.top_p = top_p;
        cfg.seed = seed;
        cfg.stop_strings = stop_strings;
        cfg.enable_thinking = enable_thinking;
        cfg.reasoning_budget = reasoning_budget.filter(|_| enable_thinking);
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

    fn render_with_thinking(
        &self,
        messages: &[NormalizedChatMessage],
        enable_thinking: bool,
    ) -> String {
        let mut prompt = self.render(messages);
        apply_qwen_thinking_mode(&mut prompt, enable_thinking);
        prompt
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

    pub(crate) fn enable_thinking(&self, requested: Option<bool>) -> bool {
        requested.unwrap_or(true)
    }

    pub(crate) fn reasoning_budget(
        &self,
        requested: Option<usize>,
        supported: bool,
        policy: MaxTokensPolicy,
        max_tokens: usize,
        enable_thinking: bool,
    ) -> Option<usize> {
        let mut reasoning_budget = if supported && enable_thinking {
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
        let enable_thinking = self.enable_thinking(options.enable_thinking);
        let reasoning_budget = self.reasoning_budget(
            options.reasoning_budget,
            options.reasoning_budget_supported,
            options.max_tokens_policy,
            max_tokens,
            enable_thinking,
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

    fn max_context(&self) -> usize {
        self.model_max_context
    }

    fn tokenizer(&self) -> Option<&BpeTokenizer> {
        Some(&self.tokenizer)
    }

    fn prepare_lattice_with_thinking(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
        vision_supported: bool,
    ) -> Result<(PreparedChatRequest, bool), ApiError> {
        prepare_chat_request_with_profile(
            req,
            default_max_tokens,
            ServeProfile::lattice(model_id, max_tokens_cap)
                .with_vision_support(vision_supported)
                .with_logprobs_support(false),
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

    fn prepare_standalone(
        &self,
        req: &ChatRequest,
        defaults: GenerationDefaults,
        model_id: &str,
        vision_supported: bool,
    ) -> Result<(ValidatedChatRequest, GenerateConfig), ApiError> {
        prepare_standalone_chat_request(
            req,
            defaults,
            ServeProfile::lattice_serve(model_id, self.model_max_context)
                .with_vision_support(vision_supported),
        )
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
        enable_thinking: bool,
    ) -> GenerateConfig {
        lattice_gen_cfg(
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
) -> Result<(PreparedChatRequest, bool), ApiError> {
    prepare_chat_request_with_profile(
        req,
        default_max_tokens,
        ServeProfile::lattice(model_id, max_tokens_cap).with_vision_support(vision_supported),
        tokenize_len,
        max_context,
    )
}

fn prepare_chat_request_with_profile(
    req: &ChatRequest,
    default_max_tokens: usize,
    profile: ServeProfile<'_>,
    tokenize_len: impl FnOnce(&str) -> usize,
    max_context: impl FnOnce() -> usize,
) -> Result<(PreparedChatRequest, bool), ApiError> {
    let (validated, prompt, enable_thinking) = normalize_request_with_context_and_thinking(
        req,
        GenerationDefaults::standard(default_max_tokens),
        profile,
        |messages, max_tokens, reasoning_budget, enable_thinking| {
            let prompt = QwenPromptAdapter.render_with_thinking(messages, enable_thinking);
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

    Ok((
        PreparedChatRequest {
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
        },
        enable_thinking,
    ))
}

pub(crate) fn lattice_gen_cfg(
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: Option<u64>,
    stop_strings: Vec<String>,
    reasoning_budget: Option<usize>,
    logprobs: Option<usize>,
    enable_thinking: bool,
) -> GenerateConfig {
    QwenPromptAdapter.lattice_generate_config(
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

pub(crate) fn build_cfg(req: &ValidatedChatRequest) -> GenerateConfig {
    QwenPromptAdapter.generate_config(req)
}

pub(crate) fn prepare_standalone_chat_request(
    req: &ChatRequest,
    defaults: GenerationDefaults,
    profile: ServeProfile<'_>,
) -> Result<(ValidatedChatRequest, GenerateConfig), ApiError> {
    let (validated, _, enable_thinking) =
        normalize_request_with_context_and_thinking(req, defaults, profile, |_, _, _, _| {
            Ok(String::new())
        })?;
    let mut config = build_cfg(&validated);
    config.enable_thinking = enable_thinking;
    config.reasoning_budget = config.reasoning_budget.filter(|_| enable_thinking);
    Ok((validated, config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse as _;
    use serde_json::json;

    fn request(extra: serde_json::Value) -> ChatRequest {
        let mut body = json!({
            "model": "served-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(body).expect("chat request")
    }

    fn metal_preparation() -> PreparationHandle {
        let model = crate::model::qwen35::test_support::tiny_zero_model_with_context(4096);
        PreparationHandle::qwen(Arc::new(model.tokenizer().clone()), model.max_context())
    }

    fn assert_logprobs_refusal(error: ApiError) {
        match &error {
            ApiError::BadRequest { code, message } => {
                assert_eq!(*code, "unsupported_feature");
                assert_eq!(
                    message,
                    "logprobs/top_logprobs are not supported by this server"
                );
            }
            other => panic!("expected logprobs refusal, got {other:?}"),
        }
        assert_eq!(error.into_response().status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn metal_lattice_logprobs_are_refused_before_generation_config() {
        let preparation = metal_preparation();
        for extra in [
            json!({"logprobs": true}),
            json!({"logprobs": true, "top_logprobs": 0}),
            json!({"logprobs": true, "top_logprobs": 2}),
            json!({"logprobs": true, "top_logprobs": 20}),
            json!({"top_logprobs": 0}),
            json!({"logprobs": false, "top_logprobs": 2}),
            json!({"logprobs": true, "stream": true}),
        ] {
            let req = request(extra);
            for vision_supported in [false, true] {
                let error = preparation
                    .prepare_lattice_with_config(&req, "served-model", 64, 4096, vision_supported)
                    .err()
                    .expect("Metal preparation must refuse logprobs");
                assert_logprobs_refusal(error);
                assert_logprobs_refusal(
                    preparation
                        .prepare_lattice(&req, "served-model", 64, 4096, vision_supported)
                        .unwrap_err(),
                );
            }
        }
    }

    #[test]
    fn metal_lattice_disabled_logprobs_preserve_request_preparation() {
        let preparation = metal_preparation();
        for logprobs in [None, Some(false)] {
            for enable_thinking in [true, false] {
                let mut req = request(json!({
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    "stop": ["stop-here"],
                    "seed": 42,
                }));
                req.logprobs = logprobs;
                let prepared = preparation
                    .prepare_lattice_with_config(&req, "served-model", 64, 4096, false)
                    .expect("disabled logprobs must remain supported");
                assert_eq!(prepared.prepared.logprobs, None);
                assert_eq!(prepared.config.logprobs, None);
                assert_eq!(prepared.config.enable_thinking, enable_thinking);
                assert_eq!(prepared.config.seed, Some(42));
                assert_eq!(prepared.config.stop_strings, ["stop-here"]);
            }
        }
    }

    #[test]
    fn metal_logprobs_refusal_precedes_prompt_and_context_work() {
        for model in ["served-model", "wrong-model"] {
            let req = request(json!({"model": model, "logprobs": true}));
            let error = prepare_chat_request_with_profile(
                &req,
                64,
                ServeProfile::lattice("served-model", 4096).with_logprobs_support(false),
                |_| panic!("unsupported logprobs must not tokenize"),
                || panic!("unsupported logprobs must not inspect context"),
            )
            .unwrap_err();
            assert_logprobs_refusal(error);
        }
    }

    #[test]
    fn metal_logprobs_refusal_preserves_other_capability_precedence() {
        let preparation = metal_preparation();
        let req = request(json!({"logprobs": true, "tools": []}));
        let error = preparation
            .prepare_lattice(&req, "served-model", 64, 4096, false)
            .unwrap_err();
        match error {
            ApiError::BadRequest { code, message } => {
                assert_eq!(code, "unsupported_feature");
                assert_eq!(
                    message,
                    "tools and tool_choice are not supported by this server"
                );
            }
            other => panic!("expected tools refusal, got {other:?}"),
        }
    }
}
