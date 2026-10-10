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

use crate::forward::metal_qwen35::ChatMessage;
use crate::generation::GenerateConfig;
use crate::serve::ApiError;
#[cfg(test)]
use crate::serve::contract::GenerationDefaults;
use crate::serve::contract::{
    ChatRequest as ChatCompletionRequest, ValidatedChatRequest as ContractValidatedChatRequest,
};

pub use crate::serve::prompt_adapter::GemmaPromptAdapter;
pub use crate::serving_preparation::PreparationHandle;

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
    pub enable_thinking: bool,
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
mod tests {
    use super::{
        ChatCompletionRequest, GenerationDefaults, PreparationHandle, build_cfg, lattice_gen_cfg,
        lattice_gen_cfg_with_thinking, prepare_chat_request,
    };
    use crate::serve::ApiError;
    use crate::tokenizer::bpe::BpeTokenizer;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn handle(model_max_context: usize) -> PreparationHandle {
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            HashMap::from([("a".to_string(), 0), ("b".to_string(), 1)]),
            Vec::new(),
        )
        .expect("tiny tokenizer must construct");
        PreparationHandle::qwen(Arc::new(tokenizer), model_max_context)
    }

    fn request(stop: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "served-model",
            "messages": [{"role": "user", "content": "a".repeat(64)}],
            "max_tokens": 1,
            "stop": stop,
        }))
        .expect("chat request body")
    }

    fn gemma_handle() -> PreparationHandle {
        PreparationHandle::gemma(Arc::new(crate::serving_cpu::tiny_zero_serving()))
    }

    fn gemma_request(extra: serde_json::Value) -> ChatCompletionRequest {
        let mut body = serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
        });
        if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
            body.extend(extra.clone());
        }
        serde_json::from_value(body).expect("chat request body")
    }

    fn refusal_code(error: ApiError) -> &'static str {
        match error {
            ApiError::BadRequest { code, .. } => code,
            other => panic!("expected a BadRequest refusal, got {other:?}"),
        }
    }

    #[test]
    fn qwen_thinking_switch_renders_the_checkpoint_closed_prefix() {
        let base = gemma_request(serde_json::json!({"model": "served-model"}));
        let default =
            prepare_chat_request(&base, "served-model", 64, 4096, false, str::len, || 4096)
                .expect("default thinking request");
        assert_eq!(
            default.prompt,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(default.enable_thinking);
        let cpu = crate::serving_provider::providers::qwen::cpu::from_model(
            crate::model::qwen35::test_support::tiny_zero_model_with_context(4096),
        );
        let cpu_default = cpu
            .prepare(&base, "served-model", 64, 4096)
            .expect("CPU provider default thinking request");
        assert_eq!(cpu_default.prepared.prompt, default.prompt);
        assert!(cpu_default.prepared.enable_thinking);
        assert!(cpu_default.config.enable_thinking);
        for enabled in [true, false] {
            let req = gemma_request(serde_json::json!({
                "model": "served-model",
                "chat_template_kwargs": {"enable_thinking": enabled},
            }));
            let prepared =
                prepare_chat_request(&req, "served-model", 64, 4096, false, str::len, || 4096)
                    .expect("explicit thinking switch");
            let expected = if enabled {
                default.prompt.clone()
            } else {
                "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    .to_string()
            };
            assert_eq!(prepared.prompt, expected);
            assert_eq!(prepared.enable_thinking, enabled);
            let cfg = lattice_gen_cfg_with_thinking(
                prepared.max_tokens,
                prepared.temperature,
                prepared.top_p,
                prepared.seed,
                prepared.stop_strings,
                prepared.reasoning_budget,
                prepared.logprobs,
                prepared.enable_thinking,
            );
            assert_eq!(cfg.enable_thinking, enabled);
            let cpu_prepared = cpu
                .prepare(&req, "served-model", 64, 4096)
                .expect("CPU provider explicit thinking switch");
            assert_eq!(cpu_prepared.prepared.prompt, expected);
            assert_eq!(cpu_prepared.prepared.enable_thinking, enabled);
            assert_eq!(
                crate::serve::GenerateConfigSnapshot::from(&cpu_prepared.config),
                crate::serve::GenerateConfigSnapshot::from(&cfg),
            );
        }
    }

    #[test]
    fn qwen_disabled_thinking_budget_is_inert_before_context_admission() {
        let without_budget = gemma_request(serde_json::json!({
            "model": "served-model",
            "max_tokens": 9,
            "chat_template_kwargs": {"enable_thinking": false},
        }));
        let baseline = prepare_chat_request(
            &without_budget,
            "served-model",
            64,
            4096,
            false,
            str::len,
            || 4096,
        )
        .expect("no-budget request");
        let context = baseline.prompt.len() + 9 + 1;
        let with_budget = gemma_request(serde_json::json!({
            "model": "served-model",
            "max_tokens": 9,
            "reasoning_budget": 4096,
            "chat_template_kwargs": {"enable_thinking": false},
        }));
        let prepared = prepare_chat_request(
            &with_budget,
            "served-model",
            64,
            4096,
            false,
            str::len,
            || context,
        )
        .expect("disabled reasoning budget consumes no context");
        assert_eq!(prepared.prompt, baseline.prompt);
        assert_eq!(prepared.max_tokens, 9);
        assert!(!prepared.enable_thinking);
        assert_eq!(prepared.reasoning_budget, None);
        let cfg =
            lattice_gen_cfg_with_thinking(9, 0.0, 1.0, None, Vec::new(), Some(4096), None, false);
        assert!(!cfg.enable_thinking);
        assert_eq!(cfg.reasoning_budget, None);
        let enabled = gemma_request(serde_json::json!({
            "model": "served-model",
            "max_tokens": 9,
            "reasoning_budget": 4096,
            "chat_template_kwargs": {"enable_thinking": true},
        }));
        let error =
            prepare_chat_request(&enabled, "served-model", 64, 4096, false, str::len, || {
                context
            })
            .expect_err("enabled reasoning budget still consumes context");
        assert_eq!(refusal_code(error), "context_length_exceeded");
    }

    #[test]
    fn qwen_image_preparation_carries_thinking_mode_and_inert_budget() {
        use base64::Engine as _;

        let mut png = Vec::new();
        image::RgbImage::new(1, 1)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("PNG fixture");
        let uri = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&png),
        );
        let base = serde_json::json!({
            "model": "served-model",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": "after"},
            ]}],
            "max_tokens": 9,
        });
        let plain = "<|im_start|>user\nbeforeafter<|im_end|>\n<|im_start|>assistant\n";
        for enabled in [None, Some(true), Some(false)] {
            let mut body = base.clone();
            if let Some(enabled) = enabled {
                body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": enabled});
            }
            if enabled == Some(false) {
                body["reasoning_budget"] = serde_json::json!(4096);
            }
            let req = serde_json::from_value(body).expect("image request");
            let prepared =
                prepare_chat_request(&req, "served-model", 64, 4096, true, str::len, || 4096)
                    .expect("vision-capable preparation");
            let effective = enabled.unwrap_or(true);
            assert_eq!(prepared.enable_thinking, effective);
            assert_eq!(prepared.reasoning_budget, None);
            let expected = if effective {
                plain.to_owned()
            } else {
                "<|im_start|>user\nbeforeafter<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".to_owned()
            };
            assert_eq!(prepared.prompt, expected);
            let image = prepared.messages[0]
                .image
                .as_ref()
                .expect("image survived normalization");
            assert_eq!(image.bytes, png);
            assert_eq!(image.text_offset, "before".len());
        }
    }

    #[test]
    fn qwen_disabled_thinking_discards_the_standalone_default_budget() {
        let handle = handle(4096);
        let defaults = GenerationDefaults {
            reasoning_budget: Some(4096),
            ..GenerationDefaults::standard(64)
        };
        for enabled in [false, true] {
            let validated = handle
                .normalize_standalone(
                    &gemma_request(serde_json::json!({
                        "max_tokens": 9,
                        "chat_template_kwargs": {"enable_thinking": enabled},
                    })),
                    defaults,
                    "served-model",
                    false,
                )
                .expect("standalone request");
            assert_eq!(validated.enable_thinking, enabled);
            let expected_budget = if enabled { Some(4086) } else { None };
            assert_eq!(validated.reasoning_budget, expected_budget);
            let cfg = handle.standalone_generate_config(&validated);
            assert_eq!(cfg.enable_thinking, enabled);
            assert_eq!(cfg.reasoning_budget, expected_budget);
            assert_eq!(
                crate::serve::GenerateConfigSnapshot::from(&cfg),
                crate::serve::GenerateConfigSnapshot::from(&build_cfg(&validated)),
            );
        }
    }

    #[test]
    fn qwen_legacy_config_entries_preserve_the_thinking_default() {
        let free = lattice_gen_cfg(9, 0.3, 0.8, Some(7), vec!["stop".into()], Some(8), Some(2));
        let via_handle = handle(4096).lattice_generate_config(
            9,
            0.3,
            0.8,
            Some(7),
            vec!["stop".into()],
            Some(8),
            Some(2),
        );
        let explicit = lattice_gen_cfg_with_thinking(
            9,
            0.3,
            0.8,
            Some(7),
            vec!["stop".into()],
            Some(8),
            Some(2),
            true,
        );
        assert!(free.enable_thinking);
        assert_eq!(free.reasoning_budget, Some(8));
        assert_eq!(
            crate::serve::GenerateConfigSnapshot::from(&free),
            crate::serve::GenerateConfigSnapshot::from(&via_handle),
        );
        assert_eq!(
            crate::serve::GenerateConfigSnapshot::from(&free),
            crate::serve::GenerateConfigSnapshot::from(&explicit),
        );
    }

    #[test]
    fn gemma_thinking_true_is_refused_and_false_is_a_noop() {
        let handle = gemma_handle();
        let base = handle
            .prepare_lattice(
                &gemma_request(serde_json::json!({"model": "served-model"})),
                "served-model",
                64,
                4096,
                false,
            )
            .expect("plain Gemma request");
        let disabled = handle
            .prepare_lattice(
                &gemma_request(
                    serde_json::json!({"model": "served-model", "chat_template_kwargs": {"enable_thinking": false}}),
                ),
                "served-model",
                64,
                4096,
                false,
            )
            .expect("Gemma false is a no-op");
        assert_eq!(disabled.prompt, base.prompt);
        assert!(!disabled.enable_thinking);
        assert_eq!(disabled.reasoning_budget, base.reasoning_budget);
        let base_validated = handle
            .normalize_standalone(
                &gemma_request(serde_json::json!({})),
                GenerationDefaults::standard(64),
                "served-model",
                false,
            )
            .expect("plain standalone Gemma request");
        let disabled_validated = handle
            .normalize_standalone(
                &gemma_request(
                    serde_json::json!({"chat_template_kwargs": {"enable_thinking": false}}),
                ),
                GenerationDefaults::standard(64),
                "served-model",
                false,
            )
            .expect("standalone Gemma false is a no-op");
        assert!(!disabled_validated.enable_thinking);
        assert_eq!(
            crate::serve::GenerateConfigSnapshot::from(
                &handle.standalone_generate_config(&base_validated)
            ),
            crate::serve::GenerateConfigSnapshot::from(
                &handle.standalone_generate_config(&disabled_validated)
            ),
        );
        let enabled = gemma_request(
            serde_json::json!({"model": "served-model", "chat_template_kwargs": {"enable_thinking": true}}),
        );
        assert_eq!(
            refusal_code(
                handle
                    .prepare_lattice(&enabled, "served-model", 64, 4096, false)
                    .expect_err("Gemma cannot think")
            ),
            "unsupported_feature",
        );
        assert_eq!(
            refusal_code(
                handle
                    .normalize_standalone(
                        &enabled,
                        GenerationDefaults::standard(64),
                        "served-model",
                        false
                    )
                    .expect_err("standalone Gemma cannot think")
            ),
            "unsupported_feature",
        );
    }

    #[test]
    fn the_gemma_handle_refuses_adapters_and_grammar_by_name() {
        let handle = gemma_handle();
        assert!(!handle.supports_adapters());
        for (extra, code) in [
            (
                serde_json::json!({"lora": [{"id": 1, "scale": 1.0}]}),
                "lora_unsupported_backend",
            ),
            (
                serde_json::json!({"response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": {"type": "object"}}}}),
                "unsupported_feature",
            ),
        ] {
            let error = handle
                .refuse_standalone_unsupported(&gemma_request(extra.clone()))
                .expect_err("Gemma cannot serve this");
            assert_eq!(refusal_code(error), code, "{extra}");
        }
        for admitted in [
            serde_json::json!({}),
            serde_json::json!({"lora": []}),
            serde_json::json!({"response_format": {"type": "text"}}),
        ] {
            handle
                .refuse_standalone_unsupported(&gemma_request(admitted.clone()))
                .unwrap_or_else(|error| panic!("{admitted}: {error:?}"));
        }
    }

    #[test]
    fn the_qwen_handle_refuses_nothing_standalone() {
        let handle = handle(64);
        assert!(handle.supports_adapters());
        let request = gemma_request(serde_json::json!({
            "lora": [{"id": 1, "scale": 1.0}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": {"type": "object"}}},
        }));
        handle
            .refuse_standalone_unsupported(&request)
            .expect("Qwen serves adapters and grammar");
    }

    #[test]
    fn the_gemma_handle_normalizes_a_plain_request_and_builds_the_gemma_config() {
        let handle = gemma_handle();
        let validated = handle
            .normalize_standalone(
                &gemma_request(serde_json::json!({})),
                GenerationDefaults::standard(64),
                "served-model",
                false,
            )
            .expect("a plain chat request is admitted");
        assert_eq!(validated.max_tokens, 5);
        let cfg = handle.standalone_generate_config(&validated);
        assert!(
            cfg.stop_token_ids.contains(&106),
            "the checkpoint's end-of-turn id stops the turn: {:?}",
            cfg.stop_token_ids
        );
        assert!(!cfg.enable_thinking);
        assert!(cfg.grammar.is_none());
    }

    #[test]
    fn the_gemma_handle_refuses_the_controls_gemma_cannot_serve_with_stable_codes() {
        let handle = gemma_handle();
        let image = "data:image/png;base64,iVBORw0KGgo=";
        for (extra, code) in [
            (serde_json::json!({"stop": ["x"]}), "unsupported_feature"),
            (
                serde_json::json!({"reasoning_budget": 8}),
                "unsupported_feature",
            ),
            (
                serde_json::json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}),
                "unsupported_feature",
            ),
            (
                serde_json::json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]}]}),
                "vision_unsupported",
            ),
        ] {
            let error = handle
                .normalize_standalone(
                    &gemma_request(extra.clone()),
                    GenerationDefaults::standard(64),
                    "served-model",
                    false,
                )
                .expect_err("Gemma cannot serve this");
            assert_eq!(refusal_code(error), code, "{extra}");
        }
    }

    #[test]
    fn the_gemma_handle_counts_tokens_with_the_gemma_tokenizer() {
        let handle = gemma_handle();
        assert!(handle.tokenize_len("hello world") > 0);
    }

    #[test]
    fn prepare_lattice_checks_context_with_its_tokenizer_before_stop() {
        let err = handle(8)
            .prepare_lattice(
                &request(serde_json::json!([])),
                "served-model",
                1,
                4096,
                false,
            )
            .unwrap_err();
        assert!(
            matches!(
                err,
                ApiError::BadRequest {
                    code: "context_length_exceeded",
                    ..
                }
            ),
            "{err:?}"
        );

        handle(4096)
            .prepare_lattice(
                &request(serde_json::Value::Null),
                "served-model",
                1,
                4096,
                false,
            )
            .expect("a prompt inside the window is admitted");
    }
}
