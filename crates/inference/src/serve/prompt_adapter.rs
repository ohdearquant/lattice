//! Per-family chat prompt adapters.
//!
//! A prompt adapter owns one model family's chat conventions: rendering
//! normalized chat messages to the prompt text, the family's stop token ids,
//! and the defaults step that turns [`RequestedChatOptions`] into an
//! effective request. Request validation stays family-neutral; a family
//! default is supplied only here.

use crate::generation::GenerateConfig;
use crate::serve::ApiError;
use crate::serve::contract::{
    GenerationDefaults, NormalizedChatMessage, RequestedChatOptions, ValidatedChatRequest,
};

#[cfg(test)]
pub(crate) use crate::serving_provider::providers::gemma::template_trim;
pub(crate) use crate::serving_provider::providers::qwen::QwenChatDefaults;

pub(crate) fn apply_qwen_thinking_mode(prompt: &mut String, enable_thinking: bool) {
    if !enable_thinking {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
}

/// One model family's chat conventions.
pub(crate) trait PromptAdapter {
    /// The prompt text for `messages`, ending with the open generation turn.
    fn render(&self, messages: &[NormalizedChatMessage]) -> String;

    /// Render with the effective thinking mode for this request.
    fn render_with_thinking(
        &self,
        messages: &[NormalizedChatMessage],
        _enable_thinking: bool,
    ) -> String {
        self.render(messages)
    }

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

/// Gemma 4 E2B text chat adapter loaded from a checkpoint.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GemmaPromptAdapter {
    pub(crate) bos_token: String,
    pub(crate) stop_token_ids: Vec<u32>,
}

#[cfg(test)]
use crate::model::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID;
#[cfg(test)]
use crate::tokenizer::gemma_bpe::GemmaBpeTokenizer;
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
