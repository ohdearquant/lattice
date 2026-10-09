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
    crate::serving_provider::providers::qwen::lattice_gen_cfg(
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
    use super::{ChatCompletionRequest, GenerationDefaults, PreparationHandle};
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
