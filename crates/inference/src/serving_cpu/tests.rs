use super::*;
use crate::serve::route::ServedRoute;
use serde_json::json;

fn serving() -> GemmaCpuServing {
    tiny_zero_serving()
}

fn request(extra: serde_json::Value) -> ChatRequest {
    let mut body = json!({
        "model": "served-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 5,
    });
    if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        body.extend(extra.clone());
    }
    serde_json::from_value(body).expect("chat request body")
}

#[test]
fn prepare_renders_the_gemma_template_and_builds_the_gemma_config() {
    let serving = serving();
    let (prepared, gen_cfg) = serving
        .prepare(&request(json!({"stream": true})), "served-model", 64, 4096)
        .expect("a plain chat request is admitted");
    assert!(
        prepared
            .prompt
            .starts_with("<bos><|turn>user\nhello<turn|>\n")
    );
    assert!(prepared.prompt.ends_with("<|turn>model\n"));
    assert!(prepared.stream);
    assert!(prepared.messages.is_empty());
    assert_eq!(prepared.max_tokens, 5);
    assert_eq!(gen_cfg.max_new_tokens, 5);
    assert!(
        gen_cfg.stop_token_ids.contains(&106),
        "the checkpoint's end-of-turn id stops the turn: {:?}",
        gen_cfg.stop_token_ids
    );
    assert!(!gen_cfg.enable_thinking);
}

#[test]
fn prepare_refuses_the_controls_gemma_cannot_serve_with_the_contract_codes() {
    let serving = serving();
    for (extra, code) in [
        (json!({"stop": ["x"]}), "unsupported_feature"),
        (json!({"logprobs": true}), "unsupported_feature"),
        (json!({"reasoning_budget": 8}), "unsupported_feature"),
    ] {
        match serving.prepare(&request(extra.clone()), "served-model", 64, 4096) {
            Err(ApiError::BadRequest { code: got, .. }) => {
                assert_eq!(got, code, "{extra}");
            }
            other => panic!("{extra}: expected a BadRequest refusal, got {other:?}"),
        }
    }
}

#[test]
fn prepare_checks_the_prompt_against_the_checkpoint_context_window() {
    let serving = serving();
    let long = "word ".repeat(serving.max_context() + 8);
    let req: ChatRequest = serde_json::from_value(json!({
        "model": "served-model",
        "messages": [{"role": "user", "content": long}],
        "max_tokens": 5,
    }))
    .expect("chat request body");
    match serving.prepare(&req, "served-model", 64, 4096) {
        Err(ApiError::BadRequest { code, .. }) => assert_eq!(code, "context_length_exceeded"),
        other => panic!("expected context_length_exceeded, got {other:?}"),
    }
}

#[test]
fn generation_runs_under_the_shared_driver_and_says_so() {
    let serving = serving();
    let cfg = GenerateConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        repetition_penalty: 1.0,
        stop_token_ids: vec![],
        ..Default::default()
    };
    let (output, evidence) = serving
        .generate_ids(&[2, 3], &cfg)
        .expect("generation over the tiny model must succeed");
    assert_eq!(output.token_ids.len(), 3);
    assert_eq!(
        evidence,
        DriverEvidence {
            opened: 3,
            consumed: 2
        }
    );
    assert!(evidence.went_through_driver());
}

fn greedy_cfg() -> GenerateConfig {
    GenerateConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        repetition_penalty: 1.0,
        stop_token_ids: vec![],
        ..Default::default()
    }
}

#[test]
fn qwen_traced_generation_reports_the_drivers_own_counters() {
    let model = crate::model::qwen35::test_support::tiny_zero_model();
    let cfg = greedy_cfg();
    let (reference, trace) = model
        .generate_with_trace("a", &cfg)
        .expect("tiny model generates");
    let (output, evidence) =
        qwen_generate_traced(&model, "a", &cfg).expect("traced generation succeeds");
    assert_eq!(output.token_ids, reference.token_ids);
    assert_eq!(evidence, DriverEvidence::from_trace(trace));
    assert!(evidence.went_through_driver(), "{evidence:?}");
    assert_eq!(evidence.consumed + 1, evidence.opened, "{evidence:?}");
}

#[test]
fn qwen_traced_streaming_reports_the_drivers_own_counters() {
    let model = crate::model::qwen35::test_support::tiny_zero_model();
    let cfg = greedy_cfg();
    let reference = model
        .generate("a", &cfg)
        .expect("tiny model generates")
        .token_ids;
    let mut deltas = 0usize;
    let (output, evidence) = qwen_generate_streaming_traced(
        &model,
        "a",
        &cfg,
        |_| {
            deltas += 1;
            true
        },
        || false,
    )
    .expect("traced streaming succeeds");
    assert_eq!(output.token_ids, reference);
    assert!(evidence.went_through_driver(), "{evidence:?}");
    assert_eq!(evidence.consumed + 1, evidence.opened, "{evidence:?}");
    assert!(deltas <= output.generated_tokens);
}

fn shared_route_assertion(marker: &str) -> Result<(), &'static str> {
    if marker.contains(" driver=shared ") {
        Ok(())
    } else if marker.contains(" driver=bypassed ") {
        Err("driver=bypassed")
    } else {
        Err("missing driver disposition")
    }
}

#[test]
fn supported_route_marker_rejects_a_test_bypassing_adapter() {
    let model = crate::model::qwen35::test_support::tiny_zero_model();
    let cfg = crate::generation::GenerateConfig {
        max_new_tokens: 2,
        temperature: 0.0,
        ..Default::default()
    };
    let (output, evidence) =
        qwen_generate_traced(&model, "a", &cfg).expect("the supported CPU route generates");
    assert!(output.generated_tokens > 0);
    let supported = ServedRoute::QWEN35_CPU.request_marker(false, evidence);
    assert_eq!(shared_route_assertion(&supported), Ok(()), "{supported}");
    eprintln!("R00-CONTROL supported marker accepted: {supported}");

    struct BypassingAdapter;

    impl BypassingAdapter {
        fn request_marker(&self) -> String {
            ServedRoute::QWEN35_CPU.request_marker(false, DriverEvidence::default())
        }
    }

    let bypassed = BypassingAdapter.request_marker();
    eprintln!("R00-CONTROL bypass adapter rejected as driver=bypassed: {bypassed}");
    assert_eq!(
        shared_route_assertion(&bypassed),
        Err("driver=bypassed"),
        "{bypassed}"
    );
}
