use super::{
    ChatCompletionRequest, GenerationDefaults, PreparationHandle, lattice_gen_cfg,
    lattice_gen_cfg_with_thinking, prepare_chat_request, prepare_chat_request_with_config,
    prepare_standalone_chat_request,
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
        prepare_chat_request_with_config(&base, "served-model", 64, 4096, false, str::len, || 4096)
            .expect("default thinking request");
    assert_eq!(
        default.prepared.prompt,
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    );
    assert!(default.config.enable_thinking);
    for enabled in [true, false] {
        let req = gemma_request(serde_json::json!({
            "model": "served-model",
            "chat_template_kwargs": {"enable_thinking": enabled},
        }));
        let prepared = prepare_chat_request_with_config(
            &req,
            "served-model",
            64,
            4096,
            false,
            str::len,
            || 4096,
        )
        .expect("explicit thinking switch");
        let expected = if enabled {
            format!("{}<think>\n", default.prepared.prompt)
        } else {
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                .to_string()
        };
        assert_eq!(prepared.prepared.prompt, expected);
        assert_eq!(prepared.config.enable_thinking, enabled);
    }
}

#[test]
fn qwen_explicit_thinking_primes_the_generation_turn() {
    let explicit = prepare_chat_request_with_config(
        &gemma_request(serde_json::json!({
            "model": "served-model",
            "chat_template_kwargs": {"enable_thinking": true},
        })),
        "served-model",
        64,
        4096,
        false,
        str::len,
        || 4096,
    )
    .expect("explicit thinking request");
    assert!(explicit.config.enable_thinking);
    assert_eq!(
        explicit.prepared.prompt,
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
}

#[test]
fn qwen_explicit_thinking_yields_the_closed_block_when_thinking_is_forced_off() {
    let mut prompt = String::from("<|im_start|>assistant\n");
    crate::serve::prompt_adapter::apply_qwen_thinking_mode(&mut prompt, false, true);
    assert_eq!(prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

#[test]
fn qwen_preparation_handle_preserves_the_requested_thinking_mode() {
    let handle = handle(4096);
    let omitted = handle
        .prepare_lattice_with_config(
            &gemma_request(serde_json::json!({"model": "served-model"})),
            "served-model",
            64,
            4096,
            false,
        )
        .expect("default thinking request");
    assert!(omitted.config.enable_thinking);
    for enabled in [true, false] {
        let prepared = handle
            .prepare_lattice_with_config(
                &gemma_request(serde_json::json!({
                    "model": "served-model",
                    "chat_template_kwargs": {"enable_thinking": enabled},
                })),
                "served-model",
                64,
                4096,
                false,
            )
            .expect("explicit thinking switch");
        assert_eq!(prepared.config.enable_thinking, enabled);
        let expected = if enabled {
            format!("{}<think>\n", omitted.prepared.prompt)
        } else {
            format!("{}<think>\n\n</think>\n\n", omitted.prepared.prompt)
        };
        assert_eq!(prepared.prepared.prompt, expected);
    }
}

#[test]
fn qwen_disabled_thinking_budget_is_inert_before_context_admission() {
    let without_budget = gemma_request(serde_json::json!({
        "model": "served-model",
        "max_tokens": 9,
        "chat_template_kwargs": {"enable_thinking": false},
    }));
    let baseline = prepare_chat_request_with_config(
        &without_budget,
        "served-model",
        64,
        4096,
        false,
        str::len,
        || 4096,
    )
    .expect("no-budget request");
    let context = baseline.prepared.prompt.len() + 9 + 1;
    let with_budget = gemma_request(serde_json::json!({
        "model": "served-model",
        "max_tokens": 9,
        "reasoning_budget": 4096,
        "chat_template_kwargs": {"enable_thinking": false},
    }));
    let prepared = prepare_chat_request_with_config(
        &with_budget,
        "served-model",
        64,
        4096,
        false,
        str::len,
        || context,
    )
    .expect("disabled reasoning budget consumes no context");
    assert_eq!(prepared.prepared.prompt, baseline.prepared.prompt);
    assert_eq!(prepared.prepared.max_tokens, 9);
    assert!(!prepared.config.enable_thinking);
    assert_eq!(prepared.prepared.reasoning_budget, None);
    let cfg = lattice_gen_cfg_with_thinking(9, 0.0, 1.0, None, Vec::new(), Some(4096), None, false);
    assert!(!cfg.enable_thinking);
    assert_eq!(cfg.reasoning_budget, None);
    let enabled = gemma_request(serde_json::json!({
        "model": "served-model",
        "max_tokens": 9,
        "reasoning_budget": 4096,
        "chat_template_kwargs": {"enable_thinking": true},
    }));
    let error = prepare_chat_request(&enabled, "served-model", 64, 4096, false, str::len, || {
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
        let prepared = prepare_chat_request_with_config(
            &req,
            "served-model",
            64,
            4096,
            true,
            str::len,
            || 4096,
        )
        .expect("vision-capable preparation");
        let effective = enabled.unwrap_or(true);
        assert_eq!(prepared.config.enable_thinking, effective);
        assert_eq!(prepared.prepared.reasoning_budget, None);
        let expected = if enabled == Some(true) {
            format!("{plain}<think>\n")
        } else if effective {
            plain.to_owned()
        } else {
            "<|im_start|>user\nbeforeafter<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".to_owned()
        };
        assert_eq!(prepared.prepared.prompt, expected);
        let image = prepared.prepared.messages[0]
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
        let request = gemma_request(serde_json::json!({
            "max_tokens": 9,
            "chat_template_kwargs": {"enable_thinking": enabled},
        }));
        let (validated, cfg) = handle
            .prepare_standalone(&request, defaults, "served-model", false)
            .expect("standalone request");
        let expected_budget = if enabled { Some(4086) } else { None };
        assert_eq!(validated.reasoning_budget, expected_budget);
        assert_eq!(cfg.enable_thinking, enabled);
        assert_eq!(cfg.reasoning_budget, expected_budget);
        let (_, free_cfg) = prepare_standalone_chat_request(
            &request,
            defaults,
            crate::serve::contract::ServeProfile::lattice_serve("served-model", 4096),
        )
        .expect("standalone free preparation");
        assert_eq!(
            crate::serve::GenerateConfigSnapshot::from(&cfg),
            crate::serve::GenerateConfigSnapshot::from(&free_cfg),
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
        .prepare_lattice_with_config(
            &gemma_request(serde_json::json!({"model": "served-model"})),
            "served-model",
            64,
            4096,
            false,
        )
        .expect("plain Gemma request");
    let disabled = handle
            .prepare_lattice_with_config(
                &gemma_request(
                    serde_json::json!({"model": "served-model", "chat_template_kwargs": {"enable_thinking": false}}),
                ),
                "served-model",
                64,
                4096,
                false,
            )
            .expect("Gemma false is a no-op");
    assert_eq!(disabled.prepared.prompt, base.prepared.prompt);
    assert!(!disabled.config.enable_thinking);
    assert_eq!(
        disabled.prepared.reasoning_budget,
        base.prepared.reasoning_budget
    );
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
            &gemma_request(serde_json::json!({"chat_template_kwargs": {"enable_thinking": false}})),
            GenerationDefaults::standard(64),
            "served-model",
            false,
        )
        .expect("standalone Gemma false is a no-op");
    assert!(
        !handle
            .standalone_generate_config(&disabled_validated)
            .enable_thinking
    );
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
