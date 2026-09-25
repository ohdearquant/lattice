//! Per-family chat prompt adapters.
//!
//! A prompt adapter owns one model family's chat conventions: rendering
//! normalized chat messages to the prompt text, the family's stop token ids,
//! and the defaults step that turns [`RequestedChatOptions`] into an
//! effective request. Request validation stays family-neutral; a family
//! default is supplied only here.

use std::path::Path;

use crate::error::InferenceError;
use crate::generation::GenerateConfig;
use crate::model::gemma4_config::{Gemma4Config, resolve_stop_token_ids};
use crate::model::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID;
use crate::serve::ApiError;
use crate::serve::contract::{
    GenerationDefaults, MaxTokensPolicy, NormalizedChatMessage, NormalizedChatRole,
    RequestedChatOptions, ValidatedChatRequest, apply_max_tokens_policy, validate_temperature,
    validate_top_p,
};
use crate::serve::format_normalized_chat_template;
use crate::tokenizer::Tokenizer;
use crate::tokenizer::gemma_bpe::GemmaBpeTokenizer;

/// One model family's chat conventions.
pub(crate) trait PromptAdapter {
    /// The prompt text for `messages`, ending with the open generation turn.
    fn render(&self, messages: &[NormalizedChatMessage]) -> String;

    /// Token ids that end the model's turn.
    fn stop_token_ids(&self) -> &[u32];

    /// Effective request values: `options` as the request sent them, with the
    /// family's defaults (and the server's `generation` defaults) applied to
    /// every omitted option, and every control the family cannot serve
    /// refused.
    fn apply_defaults(
        &self,
        generation: GenerationDefaults,
        options: RequestedChatOptions,
    ) -> Result<ValidatedChatRequest, ApiError>;

    /// The `GenerateConfig` for an effective request.
    fn generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig;
}

/// Qwen chat: ChatML rendering, thinking on, stopping at `<|im_end|>`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QwenPromptAdapter;

impl QwenPromptAdapter {
    const STOP_TOKEN_IDS: [u32; 1] = [QWEN_CHAT_IM_END_TOKEN_ID];

    /// The `lattice serve` handler's `GenerateConfig` for prepared sampling
    /// fields.
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

    /// Qwen chat generation: thinking on, stopping at `<|im_end|>`.
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

    /// The `lattice_serve` handler's `GenerateConfig` for a validated request.
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

/// The Qwen adapter's defaults step: turns [`RequestedChatOptions`] plus the
/// server's [`GenerationDefaults`] into the effective request values.
///
/// Request validation calls [`Self::max_tokens`], [`Self::temperature`],
/// [`Self::top_p`] and [`Self::reasoning_budget`] at the position each option
/// has always been resolved, so a refusal caused by a default keeps its
/// precedence.
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
}

/// Literal turn markers the Gemma 4 chat template writes.
const GEMMA_TURN_OPEN: &str = "<|turn>";
const GEMMA_TURN_CLOSE: &str = "<turn|>\n";
const GEMMA_THOUGHT_OPEN: &str = "<|channel>";
const GEMMA_THOUGHT_CLOSE: &str = "<channel|>";

/// Gemma 4 E2B text chat, rendered exactly as the checkpoint's
/// `chat_template.jinja` renders string-content system, user and assistant
/// turns with `add_generation_prompt=true` and no tools.
///
/// The template has an opt-in thinking mode (`enable_thinking` adds a
/// `<|think|>` system line), but the chat request has no switch for it and
/// the Gemma CPU session cannot enforce a reasoning budget, so this adapter
/// never enables it and refuses a positive `reasoning_budget`. Stop token ids
/// and the BOS spelling are read from the checkpoint, not assumed.
///
/// Not a stable API: constructed by the Gemma preparation entry's callers,
/// and reshaped when the worker-local model factory lands.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GemmaPromptAdapter {
    bos_token: String,
    stop_token_ids: Vec<u32>,
}

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

/// Jinja's `trim` filter is Python's `str.strip()`, whose whitespace set is
/// Rust's plus the four information separators U+001C..U+001F.
fn template_trim(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// The template's `strip_thinking` macro: drops every `<|channel>` span up to
/// its `<channel|>` (or to the end of a segment left open), then trims.
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

fn unsupported(message: &str) -> ApiError {
    ApiError::BadRequest {
        message: message.to_owned(),
        code: "unsupported_feature",
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
        let max_tokens = apply_max_tokens_policy(
            options.max_tokens.unwrap_or(generation.max_tokens),
            options.max_tokens_policy,
        )?;
        Ok(ValidatedChatRequest {
            messages: options.messages,
            max_tokens,
            temperature: validate_temperature(
                options.temperature.unwrap_or(generation.temperature),
            )?,
            top_k: options.top_k.unwrap_or(generation.top_k),
            top_p: validate_top_p(options.top_p.unwrap_or(generation.top_p))?,
            repetition_penalty: options
                .repetition_penalty
                .unwrap_or(generation.repetition_penalty),
            seed: options.seed,
            stream: options.stream.unwrap_or(false),
            stop_strings: options.stop_strings,
            reasoning_budget: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::contract::{ChatRequest, Message, normalize_messages};
    use crate::serve::prepare::{PreparedGemmaChatRequest, prepare_gemma_chat_request};
    use serde_json::{Value, json};
    use std::path::PathBuf;
    use std::sync::OnceLock;

    const MODEL: &str = "gemma-e2b";

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("gemma4")
    }

    fn matrix() -> &'static Value {
        static MATRIX: OnceLock<Value> = OnceLock::new();
        MATRIX.get_or_init(|| {
            let text = std::fs::read_to_string(fixtures().join("chat_template_matrix.json"))
                .expect("committed chat template matrix");
            serde_json::from_str(&text).expect("matrix is JSON")
        })
    }

    fn tokenizer() -> &'static GemmaBpeTokenizer {
        static TOKENIZER: OnceLock<GemmaBpeTokenizer> = OnceLock::new();
        TOKENIZER.get_or_init(|| {
            GemmaBpeTokenizer::from_tokenizer_json(
                &fixtures().join("tokenizer").join("tokenizer.json"),
            )
            .expect("committed Gemma tokenizer")
        })
    }

    /// A checkpoint directory holding the committed E2B `config.json` and
    /// `tokenizer_config.json` plus `generation_config_json`.
    fn checkpoint_dir(generation_config_json: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp checkpoint dir");
        std::fs::copy(
            fixtures().join("e2b_config.json"),
            dir.path().join("config.json"),
        )
        .expect("copy config.json");
        std::fs::copy(
            fixtures().join("tokenizer").join("tokenizer_config.json"),
            dir.path().join("tokenizer_config.json"),
        )
        .expect("copy tokenizer_config.json");
        std::fs::write(
            dir.path().join("generation_config.json"),
            generation_config_json,
        )
        .expect("write generation_config.json");
        dir
    }

    fn adapter() -> &'static GemmaPromptAdapter {
        static ADAPTER: OnceLock<GemmaPromptAdapter> = OnceLock::new();
        ADAPTER.get_or_init(|| {
            let generation = matrix()["generation_config_json"]
                .as_str()
                .expect("recorded generation_config.json");
            let dir = checkpoint_dir(generation);
            GemmaPromptAdapter::from_model_dir(dir.path(), tokenizer()).expect("Gemma adapter")
        })
    }

    fn fixture_ids(key: &str) -> Vec<u32> {
        matrix()[key]
            .as_array()
            .expect("id list")
            .iter()
            .map(|id| u32::try_from(id.as_u64().expect("id")).expect("u32 id"))
            .collect()
    }

    fn prepare(body: Value, max_context: usize) -> Result<PreparedGemmaChatRequest, ApiError> {
        let req: ChatRequest = serde_json::from_value(body).expect("chat request body");
        prepare_gemma_chat_request(adapter(), &req, MODEL, 64, 4096, str::len, || max_context)
    }

    fn refusal_code(result: Result<impl std::fmt::Debug, ApiError>) -> &'static str {
        match result {
            Err(ApiError::BadRequest { code, .. }) => code,
            other => panic!("expected a BadRequest refusal, got {other:?}"),
        }
    }

    fn cases() -> &'static [Value] {
        matrix()["cases"].as_array().expect("cases")
    }

    #[test]
    fn gemma_adapter_renders_the_checkpoint_template_matrix() {
        let rendered_cases: Vec<&Value> = cases()
            .iter()
            .filter(|case| case.get("rendered").is_some())
            .collect();
        assert!(
            rendered_cases.len() >= 8,
            "matrix has {} rendered cases",
            rendered_cases.len()
        );
        for case in rendered_cases {
            let name = case["name"].as_str().expect("case name");
            let expected = case["rendered"].as_str().expect("rendered prompt");
            let messages: Vec<Message> =
                serde_json::from_value(case["messages"].clone()).expect("messages");
            let normalized = normalize_messages(&messages).expect("contract accepts the case");
            let actual = adapter().render(&normalized);
            assert_eq!(
                actual.as_bytes(),
                expected.as_bytes(),
                "{name}: rendered prompt differs"
            );

            if messages.last().map(|message| message.role.as_str()) == Some("user") {
                let prepared = prepare(json!({"model": MODEL, "messages": case["messages"]}), 4096)
                    .unwrap_or_else(|e| panic!("{name}: {e:?}"));
                assert_eq!(prepared.prompt, expected, "{name}: prepared prompt differs");
            }
        }
    }

    #[test]
    fn gemma_matrix_refusals_use_the_contract_codes() {
        let refused: Vec<&Value> = cases()
            .iter()
            .filter(|case| case.get("expect_refusal").is_some())
            .collect();
        assert!(!refused.is_empty());
        for case in refused {
            let name = case["name"].as_str().expect("case name");
            let expected = case["expect_refusal"].as_str().expect("refusal code");
            let code = refusal_code(prepare(
                json!({"model": MODEL, "messages": case["messages"]}),
                4096,
            ));
            assert_eq!(code, expected, "{name}");
        }
    }

    #[test]
    fn gemma_stop_token_ids_come_from_the_checkpoint_configs() {
        let expected = fixture_ids("stop_token_ids");
        assert_eq!(adapter().stop_token_ids(), expected.as_slice());
        let end_of_turn = u32::try_from(
            matrix()["control_token_ids"]["end_of_turn"]
                .as_u64()
                .expect("end-of-turn id"),
        )
        .expect("u32");
        assert!(adapter().stop_token_ids().contains(&end_of_turn));
    }

    #[test]
    fn gemma_generate_config_carries_no_qwen_defaults() {
        let messages = matrix()["thinking_mode"]["messages"].clone();
        let prepared = prepare(json!({"model": MODEL, "messages": messages}), 4096)
            .expect("plain Gemma request");
        let cfg = &prepared.gen_cfg;
        assert!(
            !cfg.enable_thinking,
            "Gemma must not enable thinking by default"
        );
        assert_eq!(cfg.stop_token_ids, fixture_ids("stop_token_ids"));
        assert!(!cfg.stop_token_ids.contains(&QWEN_CHAT_IM_END_TOKEN_ID));
        assert_eq!(cfg.reasoning_budget, None);
        assert_eq!(cfg.logprobs, None);
        assert!(cfg.stop_strings.is_empty());
        assert!(cfg.grammar.is_none());
        let thinking = matrix()["thinking_mode"]["rendered_with_enable_thinking"]
            .as_str()
            .expect("thinking render");
        assert_ne!(prepared.prompt, thinking);
        assert!(!prepared.prompt.contains("<|think|>"));
    }

    #[test]
    fn gemma_refuses_controls_its_session_cannot_honour() {
        let user = json!([{"role": "user", "content": "Hi"}]);
        let refused = [
            (
                json!({"model": MODEL, "messages": user, "logprobs": true}),
                "unsupported_feature",
            ),
            (
                json!({"model": MODEL, "messages": user, "stop": "\n"}),
                "unsupported_feature",
            ),
            (
                json!({"model": MODEL, "messages": user, "reasoning_budget": 64}),
                "unsupported_feature",
            ),
            (
                json!({"model": MODEL, "messages": user, "tools": []}),
                "unsupported_feature",
            ),
            (
                json!({"model": MODEL, "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ]}]}),
                "vision_unsupported",
            ),
            (
                json!({"model": MODEL, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Hi"}
                ]}]}),
                "unsupported_feature",
            ),
        ];
        for (body, expected) in refused {
            assert_eq!(
                refusal_code(prepare(body.clone(), 4096)),
                expected,
                "{body}"
            );
        }

        let zero_budget = prepare(
            json!({"model": MODEL, "messages": user, "reasoning_budget": 0, "logprobs": false}),
            4096,
        )
        .expect("an explicit zero budget asks for no reasoning");
        assert_eq!(zero_budget.gen_cfg.reasoning_budget, None);
        assert!(!zero_budget.gen_cfg.enable_thinking);
    }

    #[test]
    fn gemma_prepare_applies_server_defaults_and_checks_the_context_window() {
        let user = json!([{"role": "user", "content": "Hi"}]);
        let prepared = prepare(json!({"model": MODEL, "messages": user}), 4096).expect("defaults");
        let standard = GenerationDefaults::standard(64);
        assert_eq!(prepared.gen_cfg.max_new_tokens, 64);
        assert_eq!(prepared.gen_cfg.temperature, standard.temperature);
        assert_eq!(prepared.gen_cfg.top_p, standard.top_p);
        assert!(!prepared.stream);

        let explicit = prepare(
            json!({"model": MODEL, "messages": user, "max_tokens": 8, "temperature": 0.0,
                   "top_p": 0.5, "seed": 7, "stream": true}),
            4096,
        )
        .expect("explicit options");
        assert_eq!(explicit.gen_cfg.max_new_tokens, 8);
        assert_eq!(explicit.gen_cfg.temperature, 0.0);
        assert_eq!(explicit.gen_cfg.top_p, 0.5);
        assert_eq!(explicit.gen_cfg.seed, Some(7));
        assert!(explicit.stream);

        let prompt_len = prepared.prompt.len();
        assert_eq!(
            refusal_code(prepare(
                json!({"model": MODEL, "messages": user}),
                prompt_len + 64
            )),
            "context_length_exceeded"
        );
        prepare(json!({"model": MODEL, "messages": user}), prompt_len + 65)
            .expect("prompt + max_tokens + 1 fits exactly");
    }

    #[test]
    fn gemma_template_trim_matches_python_str_strip() {
        let expected: std::collections::BTreeSet<u32> =
            fixture_ids("python_strip_whitespace").into_iter().collect();
        assert!(!expected.is_empty());
        let actual: std::collections::BTreeSet<u32> = (0..=u32::from(char::MAX))
            .filter_map(char::from_u32)
            .filter(|&c| template_trim(c.encode_utf8(&mut [0; 4])).is_empty())
            .map(u32::from)
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn gemma_adapter_refuses_a_checkpoint_without_the_end_of_turn_stop() {
        let dir = checkpoint_dir(r#"{"eos_token_id": [1]}"#);
        let err = GemmaPromptAdapter::from_model_dir(dir.path(), tokenizer())
            .expect_err("stop ids {1} omit <turn|>");
        assert!(
            err.to_string().contains("omit the end-of-turn token"),
            "{err}"
        );
    }
}
