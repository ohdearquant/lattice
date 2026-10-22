use super::*;
use lattice_inference::serve::metal_worker::spawn_fake_with_cap;
use std::sync::mpsc as std_mpsc;
use tower::ServiceExt as _;

/// A real worker thread (cap=1) whose `generate` blocks on
/// `unblock_rx.recv()` until the test releases it, so exactly
/// one request can be "in flight" for as long as the test needs
/// -- long enough to prove a second concurrent request is
/// rejected by admission rather than racing to observe an
/// already-freed slot.
fn single_slot_blocking_state() -> (
    AppState,
    std_mpsc::Sender<()>,
    std_mpsc::Receiver<()>, // "generate() started" signal
) {
    let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
        .tokenizer()
        .clone();
    let (unblock_tx, unblock_rx) = std_mpsc::channel::<()>();
    let unblock_rx = std::sync::Mutex::new(unblock_rx);
    let (started_tx, started_rx) = std_mpsc::channel::<()>();
    let jobs = spawn_fake_with_cap(
        1,
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        4096,
        tokenizer,
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
        jobs,
        model_id: Arc::from("test-model"),
        defaults: GenerationDefaults::standard(100),
        model_max_context: 4096,
        max_pending: 1_000_000,
        metrics: Arc::new(ServeMetrics::default()),
        vocab_bytes: Arc::new(vec![]),
        grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
        embedding: None,
    };
    (state, unblock_tx, started_rx)
}

fn simple_chat_request() -> axum::http::Request<Body> {
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#.to_string(),
        ))
        .expect("fixture request must build")
}

#[tokio::test]
async fn chat_completions_returns_503_json_envelope_when_admission_cap_reached() {
    let (state, unblock_tx, started_rx) = single_slot_blocking_state();
    let app = router(state);

    // Request 1: admitted (cap=1, 0 outstanding); its worker-thread
    // `generate` call blocks until we release it below. Driven in
    // a background task since it will not resolve until then.
    let app1 = app.clone();
    let request1 = simple_chat_request();
    let handle1 = tokio::spawn(async move { app1.oneshot(request1).await });

    // Deterministic readiness: wait for `generate` to actually
    // start (i.e. request 1's job has been dequeued and its
    // admission permit is held) before firing request 2, rather
    // than a fixed sleep.
    tokio::task::spawn_blocking(move || started_rx.recv())
        .await
        .expect("blocking wait must not panic")
        .expect("request 1's worker-thread generate() must signal it started");

    // Request 2: cap is now full (1/1) -- must be rejected with
    // a 503 JSON envelope, fail-fast, before it ever reaches the
    // worker thread's tokenizer/generate call.
    let request2 = simple_chat_request();
    let response2 = app
        .clone()
        .oneshot(request2)
        .await
        .expect("router must produce a response, not a transport error");
    assert_eq!(
        response2.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "a request submitted while the admission cap is full must be \
                     rejected with HTTP 503, fail-fast, before any SSE stream is \
                     committed and before any tokenization/generation work runs"
    );
    let bytes2 = axum::body::to_bytes(response2.into_body(), usize::MAX)
        .await
        .expect("503 response body must be readable");
    let value2: serde_json::Value =
        serde_json::from_slice(&bytes2).expect("503 response must be JSON");
    assert_eq!(value2["error"]["code"], "server_busy");
    assert_eq!(value2["error"]["type"], "server_error");
    assert!(
        !value2["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .is_empty(),
        "503 envelope must carry a human-readable message: {value2}"
    );

    // Release request 1 so it completes normally and the
    // background task/worker thread wind down cleanly.
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
