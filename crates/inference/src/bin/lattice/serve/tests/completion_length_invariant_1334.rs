use super::*;
use http_body_util::BodyExt as _;
use tower::ServiceExt as _;

/// `AppState` whose CPU generation is a canned `GenerateOutput`
/// reporting exactly `generated_tokens`, regardless of the request.
fn state_returning(max_tokens_cap: usize, generated_tokens: usize) -> AppState {
    let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
    AppState {
        model: ModelBackend::Cpu(
            lattice_inference::serving_cpu_host::SharedCpuHandle::with_qwen_generator_for_test(
                Arc::new(model),
                move |_prompt, _cfg, _on_token, _should_cancel| {
                    Ok(GenerateOutput {
                        text: "x".repeat(generated_tokens),
                        token_ids: vec![0; generated_tokens],
                        prompt_tokens: 1,
                        generated_tokens,
                        stopped: true,
                        stop_reason: Some(lattice_inference::StopReason::Length),
                        token_logprobs: vec![],
                    })
                },
            ),
        ),
        default_max_tokens: max_tokens_cap,
        max_tokens_cap,
        model_id: "test-model".to_string(),
        request_counter: Arc::new(AtomicU64::new(0)),
        embedding_model: None,
        embedding_admission: Arc::new(Semaphore::new(EMBEDDING_MAX_CONCURRENT_JOBS)),
        cpu_generation_admission: Arc::new(Semaphore::new(CPU_GENERATION_MAX_CONCURRENT_JOBS)),
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        router_state: None,
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        episodes: episode_store(),
    }
}

/// Builds the fixed `{"model":...,"messages":[...],"max_tokens":M[,
/// "reasoning_budget":R][,"stream":true]}` request body. Omitting
/// `reasoning_budget` (the "unset" half of the zero/unset cases)
/// exercises a different code path through `prepare_chat_request`
/// than sending an explicit `0` does, so callers choose which one a
/// given case needs.
fn request_body(max_tokens: usize, reasoning_budget: Option<usize>, stream: bool) -> String {
    let mut body = format!(
        r#"{{"model":"test-model","messages":[{{"role":"user","content":"hi"}}],"max_tokens":{max_tokens}"#
    );
    if let Some(budget) = reasoning_budget {
        body.push_str(&format!(r#","reasoning_budget":{budget}"#));
    }
    if stream {
        body.push_str(r#","stream":true"#);
    }
    body.push('}');
    body
}

async fn post(state: AppState, body: String) -> axum::http::Response<axum::body::Body> {
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body))
        .expect("fixture request must build");
    router(state)
        .oneshot(request)
        .await
        .expect("router must produce a response, not a transport error")
}

const MAX_TOKENS: usize = 10;
const REASONING_BUDGET: usize = 4;

// --- Non-streaming ---

async fn assert_non_streaming(
    reasoning_budget: Option<usize>,
    generated_tokens: usize,
    expect_success: bool,
    case: &str,
) {
    let state = state_returning(64, generated_tokens);
    let body = request_body(MAX_TOKENS, reasoning_budget, false);
    let response = post(state, body).await;
    if expect_success {
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "{case}: generated_tokens={generated_tokens} at/under the decode budget \
                     must be accepted, not reported as an inference failure"
        );
    } else {
        assert_eq!(
            response.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "{case}: generated_tokens={generated_tokens}, one past the decode budget, \
                     must still be rejected by the invariant"
        );
    }
}

#[tokio::test]
async fn non_streaming_exact_acceptance_positive_budget() {
    // The +1 is the forced </think> delimiter (#1372): a positive
    // reasoning_budget allows rb + max_tokens + 1 generated tokens,
    // not rb + max_tokens.
    assert_non_streaming(
        Some(REASONING_BUDGET),
        MAX_TOKENS + REASONING_BUDGET + 1,
        true,
        "non-streaming, positive reasoning_budget, exact budget",
    )
    .await;
}

#[tokio::test]
async fn non_streaming_one_past_rejection_positive_budget() {
    assert_non_streaming(
        Some(REASONING_BUDGET),
        MAX_TOKENS + REASONING_BUDGET + 2,
        false,
        "non-streaming, positive reasoning_budget, one past budget",
    )
    .await;
}

#[tokio::test]
async fn non_streaming_exact_acceptance_unset_budget() {
    assert_non_streaming(
        None,
        MAX_TOKENS,
        true,
        "non-streaming, unset reasoning_budget, exact budget",
    )
    .await;
}

#[tokio::test]
async fn non_streaming_one_past_rejection_unset_budget() {
    assert_non_streaming(
        None,
        MAX_TOKENS + 1,
        false,
        "non-streaming, unset reasoning_budget, one past budget",
    )
    .await;
}

// --- Streaming ---

async fn assert_streaming(
    reasoning_budget: Option<usize>,
    generated_tokens: usize,
    expect_success: bool,
    case: &str,
) {
    let state = state_returning(64, generated_tokens);
    let body = request_body(MAX_TOKENS, reasoning_budget, true);
    let response = post(state, body).await;
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "{case}: an SSE response commits with 200 regardless of how generation \
                 concludes -- failure surfaces as an in-stream error event"
    );
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("SSE response body must be readable")
        .to_bytes();
    let text = String::from_utf8(bytes.to_vec()).expect("SSE body must be valid UTF-8");
    let has_error_event = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .any(|payload| payload.contains("\"error\""));
    // The fixture's `GenerateOutput.stopped` is `true` (`finish_reason_for`
    // -> `lattice_inference::serve::finish_reason` maps `stopped: true` to
    // `"stop"`, not `"length"` -- `stop_reason: Length` is set alongside it
    // but `finish_reason_for` derives only from `stopped`).
    let has_finish_reason = text.contains("\"finish_reason\":\"stop\"");
    if expect_success {
        assert!(
            !has_error_event,
            "{case}: generated_tokens={generated_tokens} at/under the decode budget \
                     must not emit an SSE error event; got: {text}"
        );
        assert!(
            has_finish_reason,
            "{case}: a successful completion must emit finish_reason \"stop\"; \
                     got: {text}"
        );
    } else {
        assert!(
            has_error_event,
            "{case}: generated_tokens={generated_tokens}, one past the decode budget, \
                     must still emit an SSE error event; got: {text}"
        );
        assert!(
            !has_finish_reason,
            "{case}: an invariant violation must not also emit a clean finish_reason; \
                     got: {text}"
        );
    }
}

#[tokio::test]
async fn streaming_exact_acceptance_positive_budget() {
    // The +1 is the forced </think> delimiter (#1372): a positive
    // reasoning_budget allows rb + max_tokens + 1 generated tokens,
    // not rb + max_tokens.
    assert_streaming(
        Some(REASONING_BUDGET),
        MAX_TOKENS + REASONING_BUDGET + 1,
        true,
        "streaming, positive reasoning_budget, exact budget",
    )
    .await;
}

#[tokio::test]
async fn streaming_one_past_rejection_positive_budget() {
    assert_streaming(
        Some(REASONING_BUDGET),
        MAX_TOKENS + REASONING_BUDGET + 2,
        false,
        "streaming, positive reasoning_budget, one past budget",
    )
    .await;
}

#[tokio::test]
async fn streaming_exact_acceptance_unset_budget() {
    assert_streaming(
        None,
        MAX_TOKENS,
        true,
        "streaming, unset reasoning_budget, exact budget",
    )
    .await;
}

#[tokio::test]
async fn streaming_one_past_rejection_unset_budget() {
    assert_streaming(
        None,
        MAX_TOKENS + 1,
        false,
        "streaming, unset reasoning_budget, one past budget",
    )
    .await;
}
