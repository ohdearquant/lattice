use super::*;
use lattice_serve::serve::{
    ExpectedObservation, GenerateConfigSnapshot, OBSERVATION_GOLDEN_USER_HI_THERE_CHATML,
    ProductionAdapterObservation, assert_observation_matches,
};
use std::sync::Mutex;
use tower::ServiceExt as _;

/// Builds the fixture state + fires the fixed `{"messages":[{"role":
/// "user","content":"hi there"}],"temperature":1.3,"top_p":0.55,
/// "seed":7,"max_tokens":9}` request against a real router, with the
/// injected CPU host closure recording a
/// `ProductionAdapterObservation` -- strictly below the real
/// request-parse/`format_chat_template`/`GenerateConfig`-construction path
/// (issue #828). `stopped` is threaded through a single local
/// variable into both the recorded observation and the returned
/// `GenerateOutput`, so a caller of this helper can vary it and prove
/// the observation genuinely mirrors what the seam returned rather
/// than an independent hardcoded literal.
const OBSERVATION_GOLDEN_REQUEST_BODY: &str = r#"{"model":"test-model","messages":[{"role":"user","content":"hi there"}],"temperature":1.3,"top_p":0.55,"seed":7,"max_tokens":9}"#;
const OBSERVATION_GOLDEN_NO_THINK_CHATML: &str =
    "<|im_start|>user\nhi there<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// `plain` as the Qwen route hands it to the CPU adapter: each template marker framed by
/// U+FDD0 before it and U+FDD1 after it. Written here rather than taken from the renderer, so
/// a framing change in the renderer shows up against this test.
fn framed(plain: &str) -> String {
    let open = char::from_u32(0xFDD0).expect("U+FDD0 is a char");
    let close = char::from_u32(0xFDD1).expect("U+FDD1 is a char");
    ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
        .iter()
        .fold(plain.to_owned(), |text, marker| {
            text.replace(marker, &format!("{open}{marker}{close}"))
        })
}

/// The prompt length the adapter's generation and admission use for a framed prompt.
fn framed_prompt_tokens(prompt: &str) -> usize {
    lattice_serve::serving_cpu::qwen_framed_prompt_len(
        lattice_inference::model::qwen35::test_support::tiny_zero_model().tokenizer(),
        prompt,
    )
}

fn thinking_request_body(enable_thinking: bool, reasoning_budget: Option<usize>) -> String {
    let mut body: serde_json::Value =
        serde_json::from_str(OBSERVATION_GOLDEN_REQUEST_BODY).expect("fixture request JSON");
    body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": enable_thinking});
    if let Some(budget) = reasoning_budget {
        body["reasoning_budget"] = serde_json::json!(budget);
    }
    body.to_string()
}

async fn run_observed(stopped: bool, body: &str) -> ProductionAdapterObservation {
    let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
    let observed: Arc<Mutex<Option<ProductionAdapterObservation>>> = Arc::new(Mutex::new(None));
    let observed_for_closure = Arc::clone(&observed);
    #[allow(clippy::type_complexity)]
    let generate: Arc<
        dyn Fn(
                &str,
                &lattice_inference::GenerateConfig,
                &mut dyn FnMut(&str) -> bool,
                &mut dyn FnMut() -> bool,
            ) -> Result<GenerateOutput, lattice_inference::error::InferenceError>
            + Send
            + Sync,
    > = Arc::new(move |prompt, cfg, _on_token, _should_cancel| {
        let prompt_tokens = framed_prompt_tokens(prompt);
        *observed_for_closure
            .lock()
            .expect("observation mutex poisoned") = Some(ProductionAdapterObservation {
            rendered_prompt: Some(prompt.to_string()),
            messages: None,
            gen_cfg: GenerateConfigSnapshot::from(cfg),
            prompt_tokens,
            stopped,
        });
        Ok(GenerateOutput {
            text: "ok".to_string(),
            token_ids: vec![1],
            prompt_tokens,
            generated_tokens: 1,
            stopped,
            stop_reason: if stopped {
                Some(lattice_inference::StopReason::Eos)
            } else {
                None
            },
            token_logprobs: vec![],
        })
    });
    let state = AppState {
        model: ModelBackend::Cpu(
            lattice_serve::serving_cpu_host::SharedCpuHandle::with_qwen_generator_for_test(
                Arc::new(model),
                move |prompt, config, on_token, should_cancel| {
                    generate(prompt, config, on_token, should_cancel)
                },
            ),
        ),
        default_max_tokens: 64,
        max_tokens_cap: 64,
        model_id: "test-model".to_string(),
        request_counter: Arc::new(AtomicU64::new(0)),
        embedding_model: None,
        embedding_admission: Arc::new(Semaphore::new(EMBEDDING_MAX_CONCURRENT_JOBS)),
        cpu_admission: Arc::new(Semaphore::new(CPU_MAX_CONCURRENT_REQUESTS)),
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        router_state: None,
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        episodes: episode_store(),
    };
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .expect("fixture request must build");
    let response = router(state)
        .oneshot(request)
        .await
        .expect("router must produce a response, not a transport error");
    assert_eq!(response.status(), StatusCode::OK);

    observed
        .lock()
        .expect("observation mutex poisoned")
        .clone()
        .expect("the injected generate closure must have recorded an observation")
}

/// The `GenerateConfig` `lattice.rs`'s real `chat_completions` ->
/// `prepare_chat_request`/`build_cfg`-equivalent construction (see
/// its `let gen_cfg = ...` literal in this file) must produce for the
/// fixed request body `run_observed` sends: every explicitly-set
/// field mirrors the request, every other field is
/// `GenerateConfig::default()` -- exactly like production's own
/// `..Default::default()` tail.
#[allow(clippy::field_reassign_with_default)]
fn expected_gen_cfg() -> GenerateConfigSnapshot {
    let mut cfg = lattice_inference::GenerateConfig::default();
    cfg.max_new_tokens = 9;
    cfg.temperature = 1.3;
    cfg.top_p = 0.55;
    cfg.seed = Some(7);
    cfg.stop_strings = vec![];
    cfg.logprobs = None;
    GenerateConfigSnapshot::from(&cfg)
}

#[tokio::test]
async fn chat_completions_non_streaming_observation_captures_real_config_and_prompt() {
    let obs = run_observed(true, OBSERVATION_GOLDEN_REQUEST_BODY).await;
    let mut expected = expected_gen_cfg();
    expected.enable_thinking = false;
    let rendered = framed(OBSERVATION_GOLDEN_NO_THINK_CHATML);
    let expected_prompt_tokens = framed_prompt_tokens(&rendered);
    assert_observation_matches(
        &obs,
        &ExpectedObservation {
            gen_cfg: expected,
            rendered_prompt: Some(&rendered),
            messages: None,
            prompt_tokens: expected_prompt_tokens,
            stopped: true,
        },
    );
}

/// Proves `ProductionAdapterObservation::stopped` is genuinely
/// derived from what the generation seam returned, not an
/// independent hardcoded literal (the
/// pre-fix `lattice_serve.rs` observation stored `stopped: true`
/// unconditionally). Running the exact same request through
/// `run_observed(false)` must observe `stopped == false`.
#[tokio::test]
async fn chat_completions_non_streaming_observation_captures_real_stopped_false() {
    let obs = run_observed(false, OBSERVATION_GOLDEN_REQUEST_BODY).await;
    assert!(
        !obs.stopped,
        "observation must report the seam's actual stopped=false, not a hardcoded true"
    );
}

/// #831 config-capture regression guard: a request that sets
/// `reasoning_budget` must reach the real `GenerateConfig` this
/// binary hands its generation adapter, not get silently dropped
/// between `prepare_chat_request` and `gen_cfg` construction.
/// Mutation-sensitive: dropping the `reasoning_budget` field
/// assignment in `chat_completions_with_request`'s `gen_cfg`
/// literal, or zeroing the parsed value, turns this `Some(5)`
/// into `None`.
#[tokio::test]
async fn chat_completions_non_streaming_observation_captures_real_reasoning_budget() {
    let body = r#"{"model":"test-model","messages":[{"role":"user","content":"hi there"}],"temperature":1.3,"top_p":0.55,"seed":7,"max_tokens":9,"reasoning_budget":5}"#;
    let obs = run_observed(true, body).await;
    assert!(obs.gen_cfg.enable_thinking);
    assert_eq!(
        obs.gen_cfg.reasoning_budget,
        Some(5),
        "reasoning_budget from the request must reach the real GenerateConfig"
    );
    let primed = framed(&format!(
        "{OBSERVATION_GOLDEN_USER_HI_THERE_CHATML}<think>\n"
    ));
    assert_eq!(obs.rendered_prompt.as_deref(), Some(primed.as_str()));
}

#[tokio::test]
async fn chat_completions_non_streaming_observation_captures_real_enable_thinking() {
    let body = thinking_request_body(false, None);
    let obs = run_observed(true, &body).await;
    let mut expected = expected_gen_cfg();
    expected.enable_thinking = false;
    let rendered = framed(OBSERVATION_GOLDEN_NO_THINK_CHATML);
    let prompt_tokens = framed_prompt_tokens(&rendered);
    assert_observation_matches(
        &obs,
        &ExpectedObservation {
            gen_cfg: expected,
            rendered_prompt: Some(&rendered),
            messages: None,
            prompt_tokens,
            stopped: true,
        },
    );
}

#[tokio::test]
async fn chat_completions_non_streaming_observation_resolves_thinking_mode() {
    let explicit = thinking_request_body(true, None);
    let primed = framed(&format!(
        "{OBSERVATION_GOLDEN_USER_HI_THERE_CHATML}<think>\n"
    ));
    let no_think = framed(OBSERVATION_GOLDEN_NO_THINK_CHATML);
    for (body, rendered, enable_thinking) in [
        (OBSERVATION_GOLDEN_REQUEST_BODY, no_think.as_str(), false),
        (explicit.as_str(), primed.as_str(), true),
    ] {
        let obs = run_observed(true, body).await;
        let mut expected = expected_gen_cfg();
        expected.enable_thinking = enable_thinking;
        assert_eq!(obs.gen_cfg, expected);
        assert_eq!(obs.rendered_prompt.as_deref(), Some(rendered));
    }
}

#[tokio::test]
async fn chat_completions_non_streaming_observation_keeps_no_think_budget_inert() {
    // Exceeding the fixture's context makes a mistakenly active budget refuse admission.
    let body = thinking_request_body(false, Some(4096));
    let obs = run_observed(true, &body).await;
    assert!(!obs.gen_cfg.enable_thinking);
    assert_eq!(obs.gen_cfg.reasoning_budget, None);
    assert_eq!(obs.gen_cfg.max_new_tokens, 9);
    assert_eq!(
        obs.rendered_prompt.as_deref(),
        Some(framed(OBSERVATION_GOLDEN_NO_THINK_CHATML).as_str())
    );
}
