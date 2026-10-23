use super::*;
use axum::body::Body;
use axum::http::StatusCode;
use lattice_inference::serve::metal_worker::{ContextWindowPolicy, spawn_fake_with_cap};
use std::sync::mpsc as std_mpsc;
use tower::ServiceExt as _;

/// A real (if tiny) tokenizer -- NOT a hand-rolled minimal
/// vocab. `lattice_serve.rs`'s equivalent
/// `real_router_admission_cap::single_slot_blocking_state` uses
/// the same `test_support::tiny_zero_model` tokenizer for
/// exactly this reason: a hand-rolled single-entry vocab (e.g.
/// just `{"hi": 0}`) lacks the byte-level fallback tokens a
/// real BPE tokenizer's byte-encoder table relies on, so
/// tokenizing the full rendered ChatML prompt against it can
/// silently misbehave in ways a real tokenizer never does.
fn tiny_tokenizer() -> lattice_inference::tokenizer::bpe::BpeTokenizer {
    lattice_inference::model::qwen35::test_support::tiny_zero_model()
        .tokenizer()
        .clone()
}

/// A real worker thread (cap=1) whose `generate` blocks on
/// `unblock_rx.recv()` until the test releases it, so exactly
/// one request can be "in flight" for as long as the test
/// needs -- long enough to prove a second concurrent request is
/// rejected by admission rather than racing to observe an
/// already-freed slot. Mirrors `lattice_serve.rs`'s
/// `real_router_admission_cap::single_slot_blocking_state`.
fn single_slot_blocking_state() -> (AppState, std_mpsc::Sender<()>, std_mpsc::Receiver<()>) {
    let (unblock_tx, unblock_rx) = std_mpsc::channel::<()>();
    let unblock_rx = std::sync::Mutex::new(unblock_rx);
    let (started_tx, started_rx) = std_mpsc::channel::<()>();
    let client = spawn_fake_with_cap(
        1,
        ContextWindowPolicy::PromptAndMaxTokens,
        4096,
        tiny_tokenizer(),
        move |_messages, _cfg, prompt_tokens, _on_token, _should_cancel| {
            let _ = started_tx.send(());
            // Blocks the dedicated fake-worker OS thread (not
            // the tokio runtime) until the test explicitly lets
            // it proceed.
            let _ = unblock_rx.lock().unwrap().recv();
            Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens,
                generated_tokens: 0,
                stopped: true,
                stop_reason: None,
                token_logprobs: vec![],
            })
        },
    );
    let state = AppState {
        model: ModelBackend::Metal {
            handle: MetalHandle { client },
            tokenizer: Arc::new(tiny_tokenizer()),
            max_context: 4096,
        },
        default_max_tokens: 16,
        max_tokens_cap: 4096,
        model_id: "test-model".to_string(),
        request_counter: Arc::new(AtomicU64::new(0)),
        embedding_model: None,
        embedding_admission: Arc::new(Semaphore::new(EMBEDDING_MAX_CONCURRENT_JOBS)),
        cpu_generation_admission: Arc::new(Semaphore::new(CPU_GENERATION_MAX_CONCURRENT_JOBS)),
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        router_state: None,
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        episodes: episode_store(),
    };
    (state, unblock_tx, started_rx)
}

fn chat_request(stream: bool) -> axum::http::Request<Body> {
    let body = if stream {
        r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#
    } else {
        r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#
    };
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("fixture request must build")
}

/// Asserts the shared 503 `server_busy` JSON envelope, AND that
/// no SSE response was ever committed (issue #939's exact
/// regression for the streaming arm: a 200 SSE response whose
/// body happened to end immediately would previously slip past
/// a status-code-only assertion).
async fn assert_503_server_busy_no_sse(response: axum::http::Response<Body>, case: &str) {
    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "{case}: a request submitted while the admission cap is full must be \
                     rejected with HTTP 503, fail-fast, before any response -- streaming \
                     or not -- is committed"
    );
    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        !content_type.contains("text/event-stream"),
        "{case}: a 503 rejection must never carry an SSE content-type -- that \
                     would mean a streaming response had already committed before \
                     admission was checked: {content_type}"
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("503 response body must be readable");
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("{case}: 503 response must be JSON, not an SSE body: {e}"));
    assert_eq!(value["error"]["code"], "server_busy", "{case}: {value}");
    assert_eq!(value["error"]["type"], "server_error", "{case}: {value}");
    assert!(value["error"]["param"].is_null(), "{case}: {value}");
    assert!(
        !value["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .is_empty(),
        "{case}: 503 envelope must carry a human-readable message: {value}"
    );
}

#[tokio::test]
async fn chat_completions_streaming_returns_503_before_sse_commit_at_cap() {
    let (state, unblock_tx, started_rx) = single_slot_blocking_state();
    let app = router(state);

    let app1 = app.clone();
    let handle1 = tokio::spawn(async move { app1.oneshot(chat_request(false)).await });
    tokio::task::spawn_blocking(move || started_rx.recv())
        .await
        .expect("blocking wait must not panic")
        .expect("request 1's worker-thread generate() must signal it started");

    let response2 = app
        .clone()
        .oneshot(chat_request(true))
        .await
        .expect("router must produce a response, not a transport error");
    assert_503_server_busy_no_sse(response2, "stream:true").await;

    unblock_tx.send(()).expect("unblock send must succeed");
    let response1 = handle1
        .await
        .expect("request 1's task must not panic")
        .expect("router must produce a response, not a transport error");
    assert_eq!(
        response1.status(),
        StatusCode::OK,
        "request 1 itself was never over any cap and must succeed normally"
    );
}

#[tokio::test]
async fn chat_completions_non_streaming_returns_503_server_busy_not_500_at_cap() {
    let (state, unblock_tx, started_rx) = single_slot_blocking_state();
    let app = router(state);

    let app1 = app.clone();
    let handle1 = tokio::spawn(async move { app1.oneshot(chat_request(false)).await });
    tokio::task::spawn_blocking(move || started_rx.recv())
        .await
        .expect("blocking wait must not panic")
        .expect("request 1's worker-thread generate() must signal it started");

    let response2 = app
        .clone()
        .oneshot(chat_request(false))
        .await
        .expect("router must produce a response, not a transport error");
    assert_503_server_busy_no_sse(response2, "non-streaming").await;

    unblock_tx.send(()).expect("unblock send must succeed");
    let response1 = handle1
        .await
        .expect("request 1's task must not panic")
        .expect("router must produce a response, not a transport error");
    assert_eq!(
        response1.status(),
        StatusCode::OK,
        "request 1 itself was never over any cap and must succeed normally"
    );
}
