use super::*;
use axum::body::Body;
use axum::http::StatusCode;
use http_body_util::BodyExt as _;
use std::sync::Mutex;
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};
use tower::ServiceExt as _;

/// `AppState` whose CPU generation is cap=1 and blocks, on its first
/// call, until the test releases it -- long enough to prove a second
/// request is rejected by admission. `started_rx` signals that the
/// blocking generation has actually begun, so the test never races the
/// permit being taken.
fn single_slot_blocking_state() -> (
    AppState,
    std_mpsc::Sender<()>,
    std_mpsc::Receiver<()>,
    Arc<Semaphore>,
) {
    let (unblock_tx, unblock_rx) = std_mpsc::channel::<()>();
    let unblock_rx = Arc::new(Mutex::new(unblock_rx));
    let (started_tx, started_rx) = std_mpsc::channel::<()>();
    // `std::sync::mpsc::Sender` is `Send` but not `Sync`, and the
    // injected generator must be `Sync`; wrapping it in a mutex gives
    // the closure a `Sync` handle to signal through.
    let started_tx = Arc::new(Mutex::new(started_tx));
    // Only the FIRST call blocks. A mutation that removes admission
    // lets a second request reach generation; returning immediately
    // there turns the resulting failure into a fast assertion instead
    // of a hung test, so the mutation control is bounded in time.
    let first_call = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let model = Arc::new(lattice_inference::model::qwen35::test_support::tiny_zero_model());
    let mut state = tiny_state(64);
    let admission = Arc::new(Semaphore::new(1));
    state.cpu_generation_admission = Arc::clone(&admission);
    state.model = ModelBackend::Cpu(
        lattice_inference::serving_cpu_host::SharedCpuHandle::with_qwen_generator_for_test(
            model,
            move |_prompt, _cfg, _on_token, _should_cancel| {
                if first_call.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    let _ = started_tx.lock().unwrap().send(());
                    let _ = unblock_rx.lock().unwrap().recv();
                }
                Ok(GenerateOutput {
                    text: "x".to_string(),
                    token_ids: vec![0],
                    prompt_tokens: 1,
                    generated_tokens: 1,
                    stopped: true,
                    stop_reason: Some(lattice_inference::StopReason::Eos),
                    token_logprobs: vec![],
                })
            },
        ),
    );
    (state, unblock_tx, started_rx, admission)
}

fn chat_request(stream: bool) -> axum::http::Request<Body> {
    let body = if stream {
        r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream":true}"#
    } else {
        r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16}"#
    };
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("fixture request must build")
}

async fn assert_503_server_busy_no_sse(response: axum::http::Response<Body>, case: &str) {
    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "{case}: a CPU request over the generation cap must be rejected with HTTP 503 \
                 before any generation work starts"
    );
    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        !content_type.contains("text/event-stream"),
        "{case}: a 503 rejection must never carry an SSE content-type: {content_type}"
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("503 response body must be readable");
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("{case}: 503 response must be JSON, not an SSE body: {e}"));
    assert_eq!(value["error"]["code"], "server_busy", "{case}: {value}");
    assert_eq!(value["error"]["type"], "server_error", "{case}: {value}");
    assert!(value["error"]["param"].is_null(), "{case}: {value}");
}

/// Waits (off the async runtime) for request 1's blocking generation
/// to signal it began, so the cap is provably held before request 2.
async fn wait_started(started_rx: std_mpsc::Receiver<()>) {
    tokio::task::spawn_blocking(move || started_rx.recv())
        .await
        .expect("blocking wait must not panic")
        .expect("the in-flight generation must signal it started");
}

/// Waits until the live admission semaphore shows a free slot. The
/// 10 s bound and the poll interval only bound how long a FAILING case
/// takes; no passing result depends on a sleep length.
async fn wait_slot_free(admission: &Semaphore) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while admission.available_permits() == 0 {
        assert!(
            Instant::now() < deadline,
            "the admission slot was never released after the in-flight generation ended"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Poll budget for the cancellable generator below. 1000 polls at 5 ms
/// is a 5 s give-up bound: it only caps how long a broken cancel path
/// takes to be *reported*, and it sits under `wait_slot_free`'s own 10 s
/// bound so a broken cancel path fails on the explicit assertion below
/// rather than on the generic slot timeout. Test-only: the production
/// cancel path polls without a budget because a real cancelled request
/// has already stopped its decode loop.
const CANCEL_POLL_LIMIT: usize = 1000;
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Waits until a flag an in-flight generation sets becomes true. Same
/// contract as [`wait_slot_free`]: the bound and the poll interval only
/// bound how long a FAILING case takes, and never decide a passing one.
async fn wait_flag(flag: &std::sync::atomic::AtomicBool, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !flag.load(std::sync::atomic::Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Cap 1 and a generator that FAILS on its first call and succeeds
/// afterwards, for the "a failed generation releases its slot" pair.
/// The error is an existing `InferenceError` variant,
/// `InferenceError::Inference(String)` (`crates/inference/src/error.rs:27`).
/// The serve arms surface it as HTTP 500 `internal_error` on the
/// non-streaming path (`serve.rs:1467-1472`) and as an SSE error event
/// on the streaming path (`StreamMsg::Failed`), so request 1 differs per
/// arm while request 2 (which succeeds) proves the slot was admitted.
fn failing_state() -> (AppState, Arc<Semaphore>) {
    let model = Arc::new(lattice_inference::model::qwen35::test_support::tiny_zero_model());
    let mut state = tiny_state(64);
    let admission = Arc::new(Semaphore::new(1));
    state.cpu_generation_admission = Arc::clone(&admission);
    let first_call = Arc::new(std::sync::atomic::AtomicBool::new(true));
    state.model = ModelBackend::Cpu(
        lattice_inference::serving_cpu_host::SharedCpuHandle::with_qwen_generator_for_test(
            model,
            move |_prompt, _cfg, _on_token, _should_cancel| {
                if first_call.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    return Err(lattice_inference::error::InferenceError::Inference(
                        "generator failed".to_string(),
                    ));
                }
                Ok(GenerateOutput {
                    text: "ok".to_string(),
                    token_ids: vec![0],
                    prompt_tokens: 1,
                    generated_tokens: 1,
                    stopped: true,
                    stop_reason: Some(lattice_inference::StopReason::Eos),
                    token_logprobs: vec![],
                })
            },
        ),
    );
    (state, admission)
}

/// Cap 1 and a generator that emits one delta and then waits for the
/// client to go away: it polls `should_cancel` until it returns true (or
/// `CANCEL_POLL_LIMIT` polls expire) and records whether cancellation was
/// actually observed, then returns `Ok` with an `Interrupt` stop reason.
/// Recording the observation is what makes a broken cancel path fail the
/// test instead of silently passing through the give-up branch.
fn cancellable_streaming_state() -> (AppState, Arc<Semaphore>, Arc<std::sync::atomic::AtomicBool>) {
    let model = Arc::new(lattice_inference::model::qwen35::test_support::tiny_zero_model());
    let mut state = tiny_state(64);
    let admission = Arc::new(Semaphore::new(1));
    state.cpu_generation_admission = Arc::clone(&admission);
    let observed_cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed = Arc::clone(&observed_cancel);
    state.model = ModelBackend::Cpu(
        lattice_inference::serving_cpu_host::SharedCpuHandle::with_qwen_generator_for_test(
            model,
            move |_prompt, _cfg, on_token, should_cancel| {
                let _ = on_token("delta");
                let mut polls = 0usize;
                while !should_cancel() {
                    if polls >= CANCEL_POLL_LIMIT {
                        break;
                    }
                    polls += 1;
                    std::thread::sleep(CANCEL_POLL_INTERVAL);
                }
                observed.store(should_cancel(), std::sync::atomic::Ordering::SeqCst);
                Ok(GenerateOutput {
                    text: "delta".to_string(),
                    token_ids: vec![0],
                    prompt_tokens: 1,
                    generated_tokens: 1,
                    stopped: false,
                    stop_reason: Some(lattice_inference::StopReason::Interrupt),
                    token_logprobs: vec![],
                })
            },
        ),
    );
    (state, admission, observed_cancel)
}

#[tokio::test]
async fn cpu_generation_non_streaming_returns_503_server_busy_at_cap() {
    let (state, unblock_tx, started_rx, _admission) = single_slot_blocking_state();
    let app = router(state);

    let app1 = app.clone();
    let handle1 = tokio::spawn(async move { app1.oneshot(chat_request(false)).await });
    wait_started(started_rx).await;

    let response2 = app
        .clone()
        .oneshot(chat_request(false))
        .await
        .expect("router must produce a response, not a transport error");
    assert_503_server_busy_no_sse(response2, "cpu non-streaming").await;

    unblock_tx.send(()).expect("unblock send must succeed");
    let response1 = handle1
        .await
        .expect("request 1's task must not panic")
        .expect("router must produce a response, not a transport error");
    assert_eq!(response1.status(), StatusCode::OK);
}

#[tokio::test]
async fn cpu_generation_streaming_returns_503_before_sse_commit_at_cap() {
    let (state, unblock_tx, started_rx, _admission) = single_slot_blocking_state();
    let app = router(state);

    let app1 = app.clone();
    let handle1 = tokio::spawn(async move { app1.oneshot(chat_request(false)).await });
    wait_started(started_rx).await;

    let response2 = app
        .clone()
        .oneshot(chat_request(true))
        .await
        .expect("router must produce a response, not a transport error");
    assert_503_server_busy_no_sse(response2, "cpu stream:true").await;

    unblock_tx.send(()).expect("unblock send must succeed");
    let _response1 = handle1
        .await
        .expect("request 1's task must not panic")
        .expect("router must produce a response, not a transport error");
}

#[tokio::test]
async fn cpu_generation_slot_reopens_after_the_in_flight_generation_completes() {
    let (state, unblock_tx, started_rx, _admission) = single_slot_blocking_state();
    let app = router(state);

    let app1 = app.clone();
    let handle1 = tokio::spawn(async move { app1.oneshot(chat_request(false)).await });
    wait_started(started_rx).await;

    unblock_tx.send(()).expect("unblock send must succeed");
    let response1 = handle1
        .await
        .expect("request 1's task must not panic")
        .expect("router must produce a response, not a transport error");
    assert_eq!(response1.status(), StatusCode::OK);

    let response2 = app
        .oneshot(chat_request(false))
        .await
        .expect("router must produce a response, not a transport error");
    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "the slot released by the finished generation must admit the next request"
    );
}

/// Request 1 is STREAMING and holds the only slot while it blocks in the
/// generator, so a second non-streaming request is rejected; after the
/// stream is released the SSE body drains to completion.
///
/// Request 1's RESPONSE HEADERS are awaited before request 2 is issued.
/// That is what separates "the slot is held by the running generation"
/// from "the slot is held by the HTTP future": once the headers arrive
/// the handler has already returned, so a permit held in the handler's
/// own scope would already be released here.
#[tokio::test]
async fn cpu_generation_streaming_holds_the_slot_against_a_second_request() {
    let (state, unblock_tx, started_rx, _admission) = single_slot_blocking_state();
    let app = router(state);

    let app1 = app.clone();
    let handle1 = tokio::spawn(async move { app1.oneshot(chat_request(true)).await });
    wait_started(started_rx).await;
    let response1 = handle1
        .await
        .expect("request 1's task must not panic")
        .expect("router must produce a response, not a transport error");
    assert_eq!(
        response1.status(),
        StatusCode::OK,
        "the admitted streaming request must not be disturbed by the rejected one"
    );

    let response2 = app
        .clone()
        .oneshot(chat_request(false))
        .await
        .expect("router must produce a response, not a transport error");
    assert_503_server_busy_no_sse(response2, "streaming request 1 held the slot").await;

    unblock_tx.send(()).expect("unblock send must succeed");
    let body = response1
        .into_body()
        .collect()
        .await
        .expect("the SSE body must drain to completion")
        .to_bytes();
    assert!(
        String::from_utf8_lossy(&body).contains("[DONE]"),
        "the streaming response must finish normally"
    );
}

/// A client disconnect must not free the slot: request 1's HTTP future is
/// aborted (dropping it) while its blocking generation still runs, request
/// 2 is still rejected, and only after the blocking job ends does the slot
/// reopen for request 3.
#[tokio::test]
async fn cpu_generation_client_disconnect_does_not_free_the_slot() {
    let (state, unblock_tx, started_rx, admission) = single_slot_blocking_state();
    let app = router(state);

    let app1 = app.clone();
    let handle1 = tokio::spawn(async move { app1.oneshot(chat_request(false)).await });
    wait_started(started_rx).await;

    // Drops the HTTP future. The permit lives in the handler's detached
    // task, which a dropped HTTP future does not cancel.
    handle1.abort();
    let _ = handle1.await;

    let response2 = app
        .clone()
        .oneshot(chat_request(false))
        .await
        .expect("router must produce a response, not a transport error");
    assert_503_server_busy_no_sse(response2, "after client disconnect").await;

    unblock_tx.send(()).expect("unblock send must succeed");
    wait_slot_free(&admission).await;

    let response3 = app
        .oneshot(chat_request(false))
        .await
        .expect("router must produce a response, not a transport error");
    assert_eq!(
        response3.status(),
        StatusCode::OK,
        "the slot must reopen once the disconnected request's generation ends"
    );
}

/// (g) FAILURE releases the slot on the non-streaming arm: request 1
/// returns an error (HTTP 500, the arm's existing mapping of a failed
/// generation), and request 2 -- handled by the same cap-1 semaphore --
/// is admitted once the failed generation's permit is released.
#[tokio::test]
async fn cpu_generation_non_streaming_failure_releases_the_slot() {
    let (state, admission) = failing_state();
    let app = router(state);

    let response1 = app
        .clone()
        .oneshot(chat_request(false))
        .await
        .expect("router must produce a response, not a transport error");
    assert_eq!(
        response1.status(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "a failed generation must surface as the arm's existing 500 error status"
    );

    wait_slot_free(&admission).await;

    let response2 = app
        .oneshot(chat_request(false))
        .await
        .expect("router must produce a response, not a transport error");
    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "the slot released by the failed generation must admit the next request"
    );
}

/// (g) FAILURE releases the slot on the streaming arm: request 1 is
/// `stream:true`, its generator fails, and the SSE body carries the
/// failure envelope and terminates; request 2 is then admitted.
#[tokio::test]
async fn cpu_generation_streaming_failure_releases_the_slot() {
    let (state, admission) = failing_state();
    let app = router(state);

    let response1 = app
        .clone()
        .oneshot(chat_request(true))
        .await
        .expect("router must produce a response, not a transport error");
    assert_eq!(
        response1.status(),
        StatusCode::OK,
        "the streaming arm commits its SSE response before generation settles"
    );
    let body = response1
        .into_body()
        .collect()
        .await
        .expect("the failed SSE body must drain to completion")
        .to_bytes();
    let text = String::from_utf8_lossy(&body);
    assert!(
        text.contains("internal_error"),
        "a failed streaming generation must emit an error payload; got: {text}"
    );

    wait_slot_free(&admission).await;

    let response2 = app
        .oneshot(chat_request(false))
        .await
        .expect("router must produce a response, not a transport error");
    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "the slot released by the failed streaming generation must admit the next request"
    );
}

/// (h) STREAMING CANCELLATION releases the slot: request 1 streams a
/// delta and then polls `should_cancel` until the client goes away.
/// The test reads the frame carrying that delta and drops the body,
/// which is the client-disconnect path (`serve.rs:1360-1365`: the
/// `flat_map` closure owns `cancel_guard`, so dropping the response
/// drops the guard), then confirms the generator actually observed the
/// cancel, that the slot came back, and that request 2 is admitted.
#[tokio::test]
async fn cpu_generation_streaming_cancellation_releases_the_slot() {
    let (state, admission, observed_cancel) = cancellable_streaming_state();
    let app = router(state);

    let response1 = app
        .clone()
        .oneshot(chat_request(true))
        .await
        .expect("router must produce a response, not a transport error");
    assert_eq!(response1.status(), StatusCode::OK);
    let mut body = response1.into_body();

    // Frame 1: role chunk. Frame 2: the generator's one delta. The
    // delta frame proves generation reached `on_token` and is now
    // parked on the cancel poll.
    for case in ["role chunk", "delta chunk"] {
        let frame = tokio::time::timeout(Duration::from_secs(10), body.frame())
            .await
            .unwrap_or_else(|_| panic!("{case} frame must arrive quickly"))
            .unwrap_or_else(|| panic!("{case} frame must exist"))
            .unwrap_or_else(|error| panic!("{case} frame must be Ok: {error}"));
        drop(frame);
    }

    // Client goes away mid-stream.
    drop(body);

    wait_slot_free(&admission).await;
    wait_flag(
        &observed_cancel,
        "the generator must observe the cancel signal raised by dropping the response body",
    )
    .await;

    let response2 = app
        .oneshot(chat_request(false))
        .await
        .expect("router must produce a response, not a transport error");
    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "the slot released by the cancelled streaming generation must admit the next request"
    );
}
