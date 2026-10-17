use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use tower::ServiceExt as _;

const LOAD: &str = "/v1/lora/load";
const UNLOAD: &str = "/v1/lora/unload";

async fn assert_response_and_metrics(
    response: Response,
    state: &AppState,
    route: &str,
    status: StatusCode,
    code: &str,
    message: &str,
) {
    assert_eq!(response.status(), status, "{route}");
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let error: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        error,
        json!({"error": {
            "message": message,
            "type": if status == StatusCode::SERVICE_UNAVAILABLE {
                "server_error"
            } else {
                "invalid_request_error"
            },
            "code": code,
            "param": null,
        }}),
        "{route}"
    );
    let metrics = state.metrics.render(&state.model_id, 0);
    let requests: Vec<_> = metrics
        .lines()
        .filter(|line| line.starts_with("lattice_http_requests_total{"))
        .collect();
    let expected = format!(
        "lattice_http_requests_total{{method=\"POST\",route=\"{route}\",status=\"{}\",model=\"test-model\"}} 1",
        status.as_u16(),
    );
    assert_eq!(requests, [expected.as_str()], "{metrics}");
    let durations: Vec<_> = metrics
        .lines()
        .filter(|line| line.starts_with("lattice_http_request_duration_seconds_count{"))
        .collect();
    let expected = format!(
        "lattice_http_request_duration_seconds_count{{route=\"{route}\",model=\"test-model\"}} 1"
    );
    assert_eq!(durations, [expected.as_str()], "{metrics}");
    assert!(
        !metrics
            .lines()
            .any(|line| line.starts_with("lattice_errors_total{")),
        "{metrics}"
    );
}

async fn assert_request_admission_order(route: &'static str) {
    for case in [
        "missing_content_type",
        "wrong_content_type",
        "oversize",
        "stream_failure",
        "malformed",
        "unknown_field",
        "missing_required",
        "valid",
    ] {
        let (state, mut jobs) = test_app_state_with_jobs();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let app = Router::new()
            .route(
                route,
                post(
                    move |state: State<AppState>, headers: HeaderMap, body: Body| {
                        let calls = Arc::clone(&observed);
                        async move {
                            let support = move |_: &AppState| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                false
                            };
                            if route == LOAD {
                                lora_load_with_adapter_support(state, headers, body, support).await
                            } else {
                                lora_unload_with_adapter_support(state, headers, body, support)
                                    .await
                            }
                        }
                    },
                ),
            )
            .with_state(state.clone());
        let valid = if route == LOAD {
            r#"{"path":"never-read.safetensors","name":"test"}"#
        } else {
            r#"{"id":7}"#
        };
        let text = match case {
            "malformed" => "{",
            "missing_required" => "{}",
            "unknown_field" if route == LOAD => r#"{"path":"never-read.safetensors","scale":1}"#,
            "unknown_field" => r#"{"id":7,"path":"never-read.safetensors"}"#,
            _ => valid,
        };
        let (status, code, message) = match case {
            "missing_content_type" | "wrong_content_type" => (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/json".to_string(),
            ),
            "oversize" => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "request_body_too_large",
                "request body exceeds 1 MiB limit".into(),
            ),
            "stream_failure" => (
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "invalid request body".into(),
            ),
            "valid" => (
                StatusCode::BAD_REQUEST,
                "lora_unsupported_backend",
                "runtime LoRA adapters are not supported for this model".into(),
            ),
            _ => {
                let error = if route == LOAD {
                    lattice_inference::serve::lora::parse_lora_load(text.as_bytes())
                        .map(|_| ())
                        .unwrap_err()
                } else {
                    lattice_inference::serve::lora::parse_lora_unload(text.as_bytes())
                        .map(|_| ())
                        .unwrap_err()
                };
                (
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    error.message().to_owned(),
                )
            }
        };
        let body = match case {
            "oversize" => Body::from(vec![b' '; REQUEST_BODY_LIMIT_BYTES + 1]),
            "stream_failure" => Body::from_stream(futures::stream::once(async {
                Err::<String, _>(std::io::Error::other("synthetic body read failure"))
            })),
            _ => Body::from(text),
        };
        let mut request = axum::http::Request::builder().method("POST").uri(route);
        if case != "missing_content_type" {
            request = request.header(
                "content-type",
                if case == "wrong_content_type" {
                    "text/plain"
                } else {
                    "application/json"
                },
            );
        }
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            app.oneshot(request.body(body).unwrap()),
        )
        .await
        .expect("refusal must not await an adapter reply")
        .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            usize::from(case == "valid"),
            "{route} {case}"
        );
        assert!(
            matches!(
                jobs.try_recv(),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty)
            ),
            "{route} {case}"
        );
        assert_response_and_metrics(response, &state, route, status, code, &message).await;
    }
}

#[tokio::test]
async fn lora_load_validates_request_before_adapter_support() {
    assert_request_admission_order(LOAD).await;
}

#[tokio::test]
async fn lora_unload_validates_request_before_adapter_support() {
    assert_request_admission_order(UNLOAD).await;
}

#[tokio::test]
async fn supported_unload_wrapper_submits_before_closed_reply() {
    let (state, mut jobs) = test_app_state_with_jobs();
    assert!(adapters_supported(&state));
    let app = Router::new()
        .route(UNLOAD, post(lora_unload))
        .with_state(state.clone());
    let request = axum::http::Request::builder()
        .method("POST")
        .uri(UNLOAD)
        .header("content-type", "application/json")
        .body(Body::from(r#"{"id":7}"#))
        .unwrap();
    let (response, ()) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        tokio::join!(app.oneshot(request), async {
            let message = jobs.recv().await.expect("supported unload must submit");
            assert!(matches!(&message, WorkerMessage::Control(_)));
            drop(message);
        })
    })
    .await
    .expect("closing the control reply must finish the request");
    assert!(matches!(
        jobs.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
    assert_response_and_metrics(
        response.unwrap(),
        &state,
        UNLOAD,
        StatusCode::SERVICE_UNAVAILABLE,
        "server_busy",
        "the inference worker is not running; no adapter was unloaded",
    )
    .await;
}
