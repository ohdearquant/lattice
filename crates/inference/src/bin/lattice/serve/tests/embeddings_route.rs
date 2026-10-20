use super::*;
use axum::body::Body;
use lattice_inference::serve::embeddings::test_support::{tiny_embedding_model, tiny_png_data_uri};
use tower::ServiceExt as _;

pub(super) fn post_embeddings(body: serde_json::Value) -> axum::http::Request<Body> {
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request fixture must build")
}

pub(super) fn state_with_embedder() -> AppState {
    let mut state = tiny_state(64);
    state.embedding_model = Some(Arc::new(tiny_embedding_model()));
    state
}

pub(super) async fn json_body(response: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body must be readable");
    serde_json::from_slice(&bytes).expect("response body must be JSON")
}

// ── `/v1/lora/*` routes (runtime adapter load, Ocean 2026-09-16) ──
//
// These arms all return before a worker is ever reached, which is what a
// CPU-backed `tiny_state` can prove: that the routes exist, that the request
// contract is enforced, and that a server which cannot take an adapter says
// so instead of accepting one. The success path needs a live Metal engine and
// a real adapter file and is covered by the engine tests.

fn post_lora_load(body: &str) -> axum::http::Request<Body> {
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/lora/load")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request fixture must build")
}

fn post_lora_unload(body: &str) -> axum::http::Request<Body> {
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/lora/unload")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request fixture must build")
}

/// The route has to be registered on the router, not merely written: a
/// handler nobody routed to is a 404 that reads exactly like an unsupported
/// build. Both arms assert a non-404 status.
#[tokio::test]
async fn both_lora_routes_are_registered() {
    let load = router(tiny_state(64))
        .oneshot(post_lora_load(r#"{"path":"/tmp/a.safetensors"}"#))
        .await
        .expect("router must return a response");
    assert_ne!(load.status(), StatusCode::NOT_FOUND);
    let unload = router(tiny_state(64))
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/lora/unload")
                .body(Body::empty())
                .expect("request fixture must build"),
        )
        .await
        .expect("router must return a response");
    assert_ne!(unload.status(), StatusCode::NOT_FOUND);
}

/// A CPU-backed server cannot load an adapter: the single-slot adapter API
/// is a Metal-engine capability. Refusing with a named code is the whole
/// point -- accepting the request and doing nothing would report a loaded
/// adapter that is not loaded.
#[tokio::test]
async fn a_cpu_backend_refuses_an_adapter_by_name() {
    let response = router(tiny_state(64))
        .oneshot(post_lora_load(r#"{"path":"/tmp/a.safetensors"}"#))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(response).await["error"]["code"],
        "lora_unsupported_backend"
    );
}

#[tokio::test]
async fn lora_unload_on_a_cpu_backend_refuses_by_name() {
    let response = router(tiny_state(64))
        .oneshot(post_lora_unload(r#"{"id":0}"#))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(response).await["error"]["code"],
        "lora_unsupported_backend"
    );
}

/// Gated to the supported build on purpose. On a build with no Metal
/// compiled in, the refusal is true of every request, so it answers before
/// anything is read and no request ever reaches the content-type check --
/// which is the arm `the_build_refusal_precedes_the_request_contract` pins.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[tokio::test]
async fn lora_load_without_json_content_type_is_415() {
    let response = router(tiny_state(64))
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/lora/load")
                .body(Body::from(r#"{"path":"/tmp/a.safetensors"}"#))
                .expect("request fixture must build"),
        )
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

/// Request-contract failures are answered before the backend question, so
/// this arm distinguishes itself from `lora_unsupported_backend` on the same
/// CPU state: a malformed body is the caller's error whatever the *runtime*
/// state is. It says nothing about a build that cannot serve the route at
/// all, hence the gate.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[tokio::test]
async fn lora_load_rejects_an_unknown_field_before_asking_the_backend() {
    let response = router(tiny_state(64))
        .oneshot(post_lora_load(
            r#"{"path":"/tmp/a.safetensors","scale":2.0}"#,
        ))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(response).await["error"]["code"],
        "invalid_request"
    );
}

/// Unload's half of the same rule. `adapter_client` used to answer first
/// here, so a malformed unload body got the backend's answer while the
/// identical mistake on load got the caller's. Same order, both routes.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[tokio::test]
async fn lora_unload_rejects_an_unknown_field_before_asking_the_backend() {
    let response = router(tiny_state(64))
        .oneshot(post_lora_unload(r#"{"id":0,"path":"/tmp/a"}"#))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(response).await["error"]["code"],
        "invalid_request"
    );
}

/// The inverse of the two gated arms above. A refusal that is a property
/// of the build is true of every request, so it answers before the request
/// is read: this body is malformed twice over -- no JSON content type and an
/// unknown field -- and still gets the build's answer, not the caller's.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
#[tokio::test]
async fn the_build_refusal_precedes_the_request_contract() {
    let response = router(tiny_state(64))
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/lora/load")
                .body(Body::from(r#"{"path":"/tmp/a.safetensors","scale":2.0}"#))
                .expect("request fixture must build"),
        )
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(response).await["error"]["code"],
        "lora_unsupported_backend"
    );
}

#[tokio::test]
async fn no_embedding_model_loaded_fails_closed() {
    let response = router(tiny_state(64))
        .oneshot(post_embeddings(serde_json::json!({"input": "a"})))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value = json_body(response).await;
    assert_eq!(value["error"]["code"], "vision_unsupported");
}

#[tokio::test]
async fn happy_path_text_only() {
    let response = router(state_with_embedder())
        .oneshot(post_embeddings(serde_json::json!({"input": "a"})))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::OK);
    let value = json_body(response).await;
    assert_eq!(value["object"], "list");
    assert_eq!(value["data"][0]["object"], "embedding");
    assert_eq!(value["data"][0]["index"], 0);
    assert_eq!(value["data"][0]["embedding"].as_array().unwrap().len(), 8);
}

#[tokio::test]
async fn happy_path_image() {
    let response = router(state_with_embedder())
        .oneshot(post_embeddings(serde_json::json!({
            "input": {"type": "image_url", "image_url": {"url": tiny_png_data_uri(0)}},
        })))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::OK);
    let value = json_body(response).await;
    let embedding = value["data"][0]["embedding"].as_array().unwrap();
    let norm: f64 = embedding
        .iter()
        .map(|x| x.as_f64().unwrap().powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "expected unit norm, got {norm}");
}

#[tokio::test]
async fn mixed_batch_preserves_input_order() {
    let response = router(state_with_embedder())
        .oneshot(post_embeddings(serde_json::json!({
            "input": [
                "a",
                {"type": "image_url", "image_url": {"url": tiny_png_data_uri(1)}},
                "b",
            ],
        })))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::OK);
    let value = json_body(response).await;
    let data = value["data"].as_array().unwrap();
    assert_eq!(data.len(), 3);
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);
    assert_eq!(data[2]["index"], 2);
}

#[tokio::test]
async fn remote_url_rejected() {
    let response = router(state_with_embedder())
        .oneshot(post_embeddings(serde_json::json!({
            "input": {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        })))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value = json_body(response).await;
    assert_eq!(value["error"]["code"], "unsupported_image_url_scheme");
}

#[tokio::test]
async fn malformed_data_uri_rejected() {
    let response = router(state_with_embedder())
                .oneshot(post_embeddings(serde_json::json!({
                    "input": {"type": "image_url", "image_url": {"url": "data:image/png,not-base64-marked"}},
                })))
                .await
                .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn invalid_pooling_rejected() {
    let response = router(state_with_embedder())
        .oneshot(post_embeddings(
            serde_json::json!({"input": "a", "pooling": "max"}),
        ))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value = json_body(response).await;
    assert_eq!(value["error"]["code"], "invalid_pooling");
}

#[tokio::test]
async fn empty_input_rejected() {
    let response = router(state_with_embedder())
        .oneshot(post_embeddings(serde_json::json!({"input": []})))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value = json_body(response).await;
    assert_eq!(value["error"]["code"], "invalid_input");
}

#[tokio::test]
async fn embeddings_route_advertised_at_root() {
    let response = router(tiny_state(64))
        .oneshot(
            axum::http::Request::builder()
                .method("GET")
                .uri("/")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("router must return a response");
    let value = json_body(response).await;
    assert!(
        value["endpoints"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e == "/v1/embeddings")
    );
}

// ── admission cap + token budget (issue #1397) ──

/// A request that fills the last free admission slot (holding
/// `EMBEDDING_MAX_CONCURRENT_JOBS - 1` permits externally leaves
/// exactly one) must be admitted; one more held permit (the cap
/// itself now fully outstanding) must get the documented 503
/// `server_busy` envelope, before any `spawn_blocking` work runs.
/// Both boundaries in one test so the admitted case is proven on
/// the exact state the refused case then tightens by one permit.
#[tokio::test]
async fn admission_cap_admits_at_cap_and_refuses_past_it() {
    let state = state_with_embedder();
    let admission = state.embedding_admission.clone();
    let mut held: Vec<_> = (0..EMBEDDING_MAX_CONCURRENT_JOBS - 1)
        .map(|_| {
            admission
                .clone()
                .try_acquire_owned()
                .expect("slot must be available below the cap")
        })
        .collect();

    let response = router(state.clone())
        .oneshot(post_embeddings(serde_json::json!({"input": "a"})))
        .await
        .expect("router must return a response");
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "a request that fills the last free slot (at the cap) must be admitted"
    );

    held.push(
        admission
            .clone()
            .try_acquire_owned()
            .expect("the cap-th slot must still be available"),
    );
    let response = router(state)
        .oneshot(post_embeddings(serde_json::json!({"input": "a"})))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let value = json_body(response).await;
    assert_eq!(value["error"]["code"], "server_busy");
    drop(held);
}

/// Regression coverage for the admission-control hole issue #1397
/// describes: a client disconnecting mid-request must not release
/// the admission permit early while the already-started
/// `spawn_blocking` job keeps running -- repeated disconnects could
/// otherwise accumulate unbounded concurrent jobs past
/// `EMBEDDING_MAX_CONCURRENT_JOBS`. Exercises
/// [`run_pooled_embedding_job`] directly (the same helper
/// `embeddings()` calls) with a deterministic channel-gated
/// closure, mirroring `lattice_serve.rs`'s
/// `embedding_permit_survives_dropped_request_future`.
///
/// `tokio::spawn` + `JoinHandle::abort()` simulates axum dropping a
/// disconnected client's handler future mid-`.await`: at the point
/// of `abort()` the task is suspended awaiting the inner
/// `spawn_blocking` `JoinHandle` (not actively running), so tokio
/// drops the task's future there -- the same event a dropped
/// request future produces.
#[tokio::test]
async fn pooled_embedding_permit_survives_dropped_request_future() {
    let admission = Arc::new(Semaphore::new(1));
    let permit = admission
        .clone()
        .try_acquire_owned()
        .expect("single slot must be admitted");

    let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let mut started_tx = Some(started_tx);

    let task = tokio::spawn(run_pooled_embedding_job(permit, move || {
        started_tx
            .take()
            .expect("closure runs exactly once")
            .send(())
            .expect("test task must still be waiting on started_rx");
        // Deterministic handshake, not a sleep: blocks until the
        // test explicitly lets the "embed_items" job finish.
        release_rx
            .recv()
            .expect("release_tx must not be dropped before sending");
        Ok::<(), ApiError>(())
    }));

    started_rx
        .await
        .expect("job must signal that it started running");

    // Simulate the client disconnecting: cancel the outstanding
    // request future while the blocking job is still in flight.
    task.abort();
    let join_result = task.await;
    assert!(
        join_result.is_err() && join_result.unwrap_err().is_cancelled(),
        "aborted task must resolve as cancelled"
    );

    assert_eq!(
        admission.available_permits(),
        0,
        "admission slot must stay held while the embed_items job is still running, \
                 not release just because the request future was dropped"
    );

    // Let the blocking job finish and confirm the slot frees up.
    release_tx
        .send(())
        .expect("blocking closure must still be waiting on release_rx");
    for _ in 0..200 {
        if admission.available_permits() == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(
        admission.available_permits(),
        1,
        "admission slot must free once the embed_items job actually completes"
    );
}

/// Aggregate scaffold-token budget (`MAX_EMBEDDING_TOTAL_TOKENS`):
/// every item individually fits the tiny model's 512-token context
/// window (each item is well under that), but the sum across items
/// exceeds the request-wide budget, so the aggregate gate -- not
/// the per-item window check -- must be what rejects this request.
#[tokio::test]
async fn aggregate_token_budget_exceeded_400() {
    use lattice_inference::serve::embeddings::MAX_EMBEDDING_TOTAL_TOKENS;
    // `tiny_embedding_model`'s vocab has no merges, so each "a"
    // character tokenizes to its own token (see that fixture's doc
    // comment): 480 tokens per item, comfortably under the tiny
    // model's 512-token context window, times 80 items comfortably
    // clears the 32,768-token aggregate budget.
    const {
        assert!(
            80 * 480 > MAX_EMBEDDING_TOTAL_TOKENS,
            "fixture must actually exceed the budget it tests"
        );
    }
    let item = "a".repeat(480);
    let items: Vec<serde_json::Value> =
        std::iter::repeat_n(serde_json::Value::String(item), 80).collect();
    let response = router(state_with_embedder())
        .oneshot(post_embeddings(serde_json::json!({ "input": items })))
        .await
        .expect("router must return a response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value = json_body(response).await;
    assert_eq!(value["error"]["code"], "batch_token_budget_exceeded");
}

/// Priority ordering: with the admission cap fully
/// held, a request that would ALSO fail the aggregate token budget
/// must still be refused by admission (503 `server_busy`), not by
/// the budget check (400 `batch_token_budget_exceeded`) --
/// `check_pooled_embedding_total_tokens` runs inside the admitted
/// `spawn_blocking` job, after `try_acquire_embedding_slot`, so a
/// request that never gets a permit never reaches it. This is
/// busy-before-budget, deliberately not the sibling route's own
/// within-request order: see `check_pooled_embedding_total_tokens`'s
/// doc comment. Reverting the fix (running the budget check before
/// admission, on the handler thread) turns this 503 into a 400 --
/// the two error codes are asserted separately so that mutation is
/// visible in the failure message, not just a boolean.
#[tokio::test]
async fn admission_refusal_takes_priority_over_token_budget() {
    use lattice_inference::serve::embeddings::MAX_EMBEDDING_TOTAL_TOKENS;
    const {
        assert!(
            80 * 480 > MAX_EMBEDDING_TOTAL_TOKENS,
            "fixture must actually exceed the budget it tests"
        );
    }
    let state = state_with_embedder();
    let admission = state.embedding_admission.clone();
    let held: Vec<_> = (0..EMBEDDING_MAX_CONCURRENT_JOBS)
        .map(|_| {
            admission
                .clone()
                .try_acquire_owned()
                .expect("slot must be available below the cap")
        })
        .collect();

    let item = "a".repeat(480);
    let items: Vec<serde_json::Value> =
        std::iter::repeat_n(serde_json::Value::String(item), 80).collect();
    let response = router(state)
        .oneshot(post_embeddings(serde_json::json!({ "input": items })))
        .await
        .expect("router must return a response");
    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "a fully-held admission cap must refuse before the budget check ever \
                 runs, even for a request that would also fail that check"
    );
    let value = json_body(response).await;
    assert_eq!(value["error"]["code"], "server_busy");
    drop(held);
}
