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
