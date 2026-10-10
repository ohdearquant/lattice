use crate::error::InferenceError;
use crate::generation::GenerateConfig;
use crate::model::gemma4_config::{Gemma4Config, resolve_stop_token_ids};
use crate::serve::ApiError;
use crate::serve::contract::{
    ChatRequest, GenerationDefaults, MessageContent, NormalizedChatMessage, NormalizedChatRole,
    RequestedChatOptions, ServeProfile, ValidatedChatRequest, normalize_requested_options,
    validate_context_window_with_budget,
};
use crate::serve::prepare::PreparedChatRequest;
use crate::serve::prepare::PreparedGemmaChatRequest;
use crate::serve::prompt_adapter::{GemmaPromptAdapter, PromptAdapter};
use crate::serving_cpu::GemmaCpuServing;
use crate::serving_preparation::{PreparationHandle, PreparedCpuChat, RequestPreparation};
use crate::tokenizer::Tokenizer;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::gemma_bpe::GemmaBpeTokenizer;
use std::path::Path;
use std::sync::Arc;

const GEMMA_TURN_OPEN: &str = "<|turn>";
const GEMMA_TURN_CLOSE: &str = "<turn|>\n";
const GEMMA_THOUGHT_OPEN: &str = "<|channel>";
const GEMMA_THOUGHT_CLOSE: &str = "<channel|>";

impl GemmaPromptAdapter {
    /// Read the adapter from a Gemma 4 checkpoint directory: `config.json`
    /// (admitted as the supported E2B text configuration), the
    /// `generation_config.json` / `config.json` end-of-sequence ids, and the
    /// BOS and end-of-turn spellings in `tokenizer_config.json`.
    ///
    /// # Errors
    /// Fails when a file is missing or malformed, when the BOS or end-of-turn
    /// token is not a single token of `tokenizer`, or when the checkpoint's
    /// stop ids do not include the end-of-turn token the template writes.
    pub fn from_model_dir(
        dir: &Path,
        tokenizer: &GemmaBpeTokenizer,
    ) -> Result<Self, InferenceError> {
        let config = Gemma4Config::from_model_dir(dir)?;
        let stop_token_ids = resolve_stop_token_ids(dir, config.eos_token_id);

        let path = dir.join("tokenizer_config.json");
        let text =
            crate::model::config_file::read_config_json_bounded(&path, "tokenizer_config.json")?;
        let tokenizer_config: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| InferenceError::Inference(format!("invalid {}: {e}", path.display())))?;
        let token_field = |field: &str| -> Result<String, InferenceError> {
            tokenizer_config
                .get(field)
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| {
                    InferenceError::Inference(format!(
                        "{} has no string field '{field}'",
                        path.display()
                    ))
                })
        };
        let bos_token = token_field("bos_token")?;
        let end_of_turn = token_field("eot_token")?;
        single_token_id(tokenizer, &bos_token)?;
        let end_of_turn_id = single_token_id(tokenizer, &end_of_turn)?;
        if GEMMA_TURN_CLOSE.trim_end() != end_of_turn {
            return Err(InferenceError::Inference(format!(
                "tokenizer_config.json end-of-turn token {end_of_turn:?} is not the template's \
                 {:?}",
                GEMMA_TURN_CLOSE.trim_end()
            )));
        }
        if !stop_token_ids.contains(&end_of_turn_id) {
            return Err(InferenceError::Inference(format!(
                "checkpoint stop ids {stop_token_ids:?} omit the end-of-turn token \
                 {end_of_turn:?} (id {end_of_turn_id})"
            )));
        }
        Ok(Self {
            bos_token,
            stop_token_ids,
        })
    }
}

fn single_token_id(tokenizer: &GemmaBpeTokenizer, token: &str) -> Result<u32, InferenceError> {
    let tokenized = tokenizer.tokenize(token);
    match tokenized.input_ids.get(..tokenized.real_length) {
        Some([id]) => Ok(*id),
        _ => Err(InferenceError::Inference(format!(
            "{token:?} is not a single tokenizer token"
        ))),
    }
}

pub(crate) fn template_trim(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

fn strip_thinking(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for part in text.split(GEMMA_THOUGHT_CLOSE) {
        match part.split_once(GEMMA_THOUGHT_OPEN) {
            Some((kept, _)) => out.push_str(kept),
            None => out.push_str(part),
        }
    }
    template_trim(&out).to_owned()
}

fn gemma_role(role: NormalizedChatRole) -> &'static str {
    match role {
        NormalizedChatRole::System => "system",
        NormalizedChatRole::User => "user",
        NormalizedChatRole::Assistant => "model",
    }
}

impl PromptAdapter for GemmaPromptAdapter {
    fn render(&self, messages: &[NormalizedChatMessage]) -> String {
        let mut prompt = self.bos_token.clone();
        let mut turns = messages;
        if let Some((first, rest)) = messages.split_first()
            && first.role == NormalizedChatRole::System
        {
            prompt.push_str(GEMMA_TURN_OPEN);
            prompt.push_str("system\n");
            prompt.push_str(template_trim(&first.content));
            prompt.push_str(GEMMA_TURN_CLOSE);
            turns = rest;
        }
        let mut previous: Option<NormalizedChatRole> = None;
        for message in turns {
            let continues_model_turn = message.role == NormalizedChatRole::Assistant
                && previous == Some(NormalizedChatRole::Assistant);
            if !continues_model_turn {
                prompt.push_str(GEMMA_TURN_OPEN);
                prompt.push_str(gemma_role(message.role));
                prompt.push('\n');
            }
            if message.role == NormalizedChatRole::Assistant {
                prompt.push_str(&strip_thinking(&message.content));
            } else {
                prompt.push_str(template_trim(&message.content));
            }
            prompt.push_str(GEMMA_TURN_CLOSE);
            previous = Some(message.role);
        }
        prompt.push_str(GEMMA_TURN_OPEN);
        prompt.push_str("model\n");
        prompt
    }

    fn stop_token_ids(&self) -> &[u32] {
        &self.stop_token_ids
    }

    fn apply_defaults(
        &self,
        generation: GenerationDefaults,
        options: RequestedChatOptions,
    ) -> Result<ValidatedChatRequest, ApiError> {
        if options.enable_thinking == Some(true) {
            return Err(unsupported(
                "enable_thinking is not supported for this model",
            ));
        }
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
        if options.reasoning_budget.is_some_and(|budget| budget > 0) {
            return Err(unsupported(
                "reasoning_budget is not supported for this model",
            ));
        }
        let max_tokens = crate::serve::contract::apply_max_tokens_policy(
            options.max_tokens.unwrap_or(generation.max_tokens),
            options.max_tokens_policy,
        )?;
        Ok(ValidatedChatRequest {
            messages: options.messages,
            max_tokens,
            temperature: crate::serve::contract::validate_temperature(
                options.temperature.unwrap_or(generation.temperature),
            )?,
            top_k: options.top_k.unwrap_or(generation.top_k),
            top_p: crate::serve::contract::validate_top_p(
                options.top_p.unwrap_or(generation.top_p),
            )?,
            repetition_penalty: options
                .repetition_penalty
                .unwrap_or(generation.repetition_penalty),
            seed: options.seed,
            stream: options.stream.unwrap_or(false),
            stop_strings: options.stop_strings,
            reasoning_budget: None,
            enable_thinking: false,
            logprobs: None,
        })
    }

    fn generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig {
        GenerateConfig {
            max_new_tokens: req.max_tokens,
            temperature: req.temperature,
            top_k: req.top_k,
            top_p: req.top_p,
            min_p: 0.0,
            repetition_penalty: req.repetition_penalty,
            seed: req.seed,
            stop_token_ids: self.stop_token_ids.clone(),
            enable_thinking: false,
            enable_mtp: None,
            grammar: None,
            stop_strings: req.stop_strings.clone(),
            reasoning_budget: req.reasoning_budget,
            logprobs: req.logprobs,
        }
    }
}

fn unsupported(message: &str) -> ApiError {
    ApiError::BadRequest {
        message: message.to_owned(),
        code: "unsupported_feature",
    }
}

struct GemmaPreparation {
    serving: Arc<GemmaCpuServing>,
}

impl RequestPreparation for GemmaPreparation {
    fn tokenize_len(&self, prompt: &str) -> usize {
        self.serving.tokenize_len(prompt)
    }

    fn max_context(&self) -> usize {
        self.serving.max_context()
    }

    fn tokenizer(&self) -> Option<&BpeTokenizer> {
        None
    }

    fn prepare_cpu(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
    ) -> Result<PreparedCpuChat, ApiError> {
        let (prepared, config) =
            self.serving
                .prepare(req, model_id, default_max_tokens, max_tokens_cap)?;
        Ok(PreparedCpuChat { prepared, config })
    }

    fn prepare_lattice(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
        _vision_supported: bool,
    ) -> Result<PreparedChatRequest, ApiError> {
        self.serving
            .prepare(req, model_id, default_max_tokens, max_tokens_cap)
            .map(|(prepared, _)| prepared)
    }

    fn normalize_standalone(
        &self,
        req: &ChatRequest,
        defaults: GenerationDefaults,
        model_id: &str,
        _vision_supported: bool,
    ) -> Result<ValidatedChatRequest, ApiError> {
        self.serving.normalize_standalone(req, defaults, model_id)
    }

    fn standalone_generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig {
        self.serving.standalone_generate_config(req)
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
        _enable_thinking: bool,
    ) -> GenerateConfig {
        self.serving.lattice_generate_config(
            max_tokens,
            temperature,
            top_p,
            seed,
            stop_strings,
            reasoning_budget,
            logprobs,
        )
    }

    fn refuse_standalone_unsupported(&self, req: &ChatRequest) -> Result<(), ApiError> {
        if req
            .lora
            .as_ref()
            .is_some_and(|selection| !selection.is_empty())
        {
            return Err(lora_unsupported_backend());
        }
        if req
            .response_format
            .as_ref()
            .is_some_and(|format| format.r#type == "json_schema")
        {
            return Err(ApiError::BadRequest {
                message: "response_format.type 'json_schema' is not supported for this \
                          model; use 'text'"
                    .to_string(),
                code: "unsupported_feature",
            });
        }
        Ok(())
    }

    fn supports_adapters(&self) -> bool {
        false
    }
}

impl PreparationHandle {
    pub(crate) fn gemma(serving: Arc<GemmaCpuServing>) -> Self {
        Self::new(Arc::new(GemmaPreparation { serving }))
    }
}

pub(crate) fn prepare_gemma_chat_request(
    adapter: &GemmaPromptAdapter,
    req: &ChatRequest,
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

pub(crate) fn lora_unsupported_backend() -> ApiError {
    ApiError::BadRequest {
        message: "runtime LoRA adapters are not supported for this model".to_string(),
        code: "lora_unsupported_backend",
    }
}
