use super::*;
use lattice_inference::serve::lora::{
    AdapterControlError, AdapterControlReceiver, AdapterControlResult, AdapterIndex,
    AdapterMetadata, LoraSelection,
};
use std::cell::Cell;

fn confirmed_adapter() -> AdapterMetadata {
    AdapterMetadata {
        id: 41,
        name: "confirmed-name".into(),
        path: "confirmed-path".into(),
        rank: 3,
        layers: 5,
    }
}

fn reply_receiver(
    reply: Option<Result<AdapterControlResult, AdapterControlError>>,
) -> AdapterControlReceiver {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    if let Some(reply) = reply {
        sender.send(reply).unwrap();
    }
    receiver
}

fn error_body(status: StatusCode, code: &str, message: &str) -> Value {
    json!({"error": {
        "message": message,
        "type": if status == StatusCode::SERVICE_UNAVAILABLE {
            "server_error"
        } else {
            "invalid_request_error"
        },
        "code": code,
        "param": null,
    }})
}

async fn assert_post(
    response: Response,
    metrics: &ServeMetrics,
    route: &str,
    status: StatusCode,
    expected: Value,
) {
    assert_eq!(response.status(), status);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), expected);
    let rendered = metrics.render("test-model", 0);
    let requests: Vec<_> = rendered
        .lines()
        .filter(|line| line.starts_with("lattice_http_requests_total{"))
        .collect();
    let request = format!(
        "lattice_http_requests_total{{method=\"POST\",route=\"{route}\",status=\"{}\",model=\"test-model\"}} 1",
        status.as_u16(),
    );
    assert_eq!(requests, [request.as_str()], "{rendered}");
    let durations: Vec<_> = rendered
        .lines()
        .filter(|line| line.starts_with("lattice_http_request_duration_seconds_count{"))
        .collect();
    let duration = format!(
        "lattice_http_request_duration_seconds_count{{route=\"{route}\",model=\"test-model\"}} 1"
    );
    assert_eq!(durations, [duration.as_str()], "{rendered}");
    assert!(
        !rendered
            .lines()
            .any(|line| line.starts_with("lattice_errors_total{")),
        "{rendered}"
    );
}

async fn run_load(
    body: Body,
    failure_stage: &str,
    reply: Option<Result<AdapterControlResult, AdapterControlError>>,
) -> (Response, ServeMetrics, [usize; 3]) {
    let metrics = ServeMetrics::default();
    let calls = [Cell::new(0), Cell::new(0), Cell::new(0)];
    let response = lora_load_with_backend(
        &metrics,
        test_json_headers(),
        body,
        || {
            calls[0].set(calls[0].get() + 1);
            if failure_stage == "runtime" {
                Err(ApiError::BadRequest {
                    code: "lora_unsupported_backend",
                    message: "runtime refused".into(),
                })
            } else {
                Ok(7)
            }
        },
        |path, name| {
            calls[1].set(calls[1].get() + 1);
            assert_eq!((path, name), ("requested-path", "requested-name"));
            if failure_stage == "prepare" {
                Err(ApiError::BadRequest {
                    code: "lora_load_failed",
                    message: "preparation refused".into(),
                })
            } else {
                Ok(("prepared-path", "prepared-name"))
            }
        },
        |backend, prepared| {
            calls[2].set(calls[2].get() + 1);
            assert_eq!(backend, 7);
            assert_eq!(prepared, ("prepared-path", "prepared-name"));
            if failure_stage == "submit" {
                Err(ApiError::ServiceUnavailable {
                    message: "control slot occupied".into(),
                })
            } else {
                Ok(reply_receiver(reply))
            }
        },
    )
    .await;
    (response, metrics, calls.map(|calls| calls.get()))
}

async fn run_unload(
    failure_stage: &str,
    reply: Option<Result<AdapterControlResult, AdapterControlError>>,
) -> (Response, ServeMetrics, [usize; 2]) {
    let metrics = ServeMetrics::default();
    let calls = [Cell::new(0), Cell::new(0)];
    let response = lora_unload_with_backend(
        &metrics,
        test_json_headers(),
        Body::from(r#"{"id":7}"#),
        || {
            calls[0].set(calls[0].get() + 1);
            if failure_stage == "runtime" {
                Err(ApiError::BadRequest {
                    code: "lora_unsupported_backend",
                    message: "runtime refused".into(),
                })
            } else {
                Ok(3)
            }
        },
        |backend, id| {
            calls[1].set(calls[1].get() + 1);
            assert_eq!((backend, id), (3, 7));
            if failure_stage == "submit" {
                Err(ApiError::ServiceUnavailable {
                    message: "control slot occupied".into(),
                })
            } else {
                Ok(reply_receiver(reply))
            }
        },
    )
    .await;
    (response, metrics, calls.map(|calls| calls.get()))
}

fn load_body() -> Body {
    Body::from(r#"{"path":"requested-path","name":"requested-name"}"#)
}

#[tokio::test]
async fn lora_load_wrapper_preserves_success_and_failures() {
    const ROUTE: &str = "/v1/lora/load";
    let (response, metrics, calls) = run_load(
        load_body(),
        "",
        Some(Ok(AdapterControlResult::Loaded(confirmed_adapter()))),
    )
    .await;
    assert_eq!(calls, [1, 1, 1]);
    assert_post(
        response,
        &metrics,
        ROUTE,
        StatusCode::OK,
        json!({"object":"lora.adapter","status":"loaded","id":41,
                        "name":"confirmed-name","path":"confirmed-path","rank":3,"layers":5}),
    )
    .await;
    let (response, metrics, calls) =
        run_load(load_body(), "", Some(Err(AdapterControlError::NotFound(7)))).await;
    assert_eq!(calls, [1, 1, 1]);
    assert_post(
        response,
        &metrics,
        ROUTE,
        StatusCode::BAD_REQUEST,
        error_body(
            StatusCode::BAD_REQUEST,
            "lora_adapter_not_found",
            "unknown LoRA adapter id 7",
        ),
    )
    .await;
    for (stage, reply, message) in [
        (
            "",
            Some(Ok(AdapterControlResult::Unloaded(99))),
            "unexpected adapter command reply",
        ),
        (
            "",
            None,
            "the inference worker is not running; the adapter was not loaded",
        ),
        ("submit", None, "control slot occupied"),
    ] {
        let (response, metrics, calls) = run_load(load_body(), stage, reply).await;
        assert_eq!(calls, [1, 1, 1]);
        assert_post(
            response,
            &metrics,
            ROUTE,
            StatusCode::SERVICE_UNAVAILABLE,
            error_body(StatusCode::SERVICE_UNAVAILABLE, "server_busy", message),
        )
        .await;
    }
    for (stage, calls, code, message) in [
        (
            "runtime",
            [1, 0, 0],
            "lora_unsupported_backend",
            "runtime refused",
        ),
        (
            "prepare",
            [1, 1, 0],
            "lora_load_failed",
            "preparation refused",
        ),
    ] {
        let (response, metrics, observed) = run_load(load_body(), stage, None).await;
        assert_eq!(observed, calls);
        assert_post(
            response,
            &metrics,
            ROUTE,
            StatusCode::BAD_REQUEST,
            error_body(StatusCode::BAD_REQUEST, code, message),
        )
        .await;
    }
}

#[tokio::test]
async fn lora_unload_wrapper_preserves_success_and_failures() {
    const ROUTE: &str = "/v1/lora/unload";
    let (response, metrics, calls) =
        run_unload("", Some(Ok(AdapterControlResult::Unloaded(41)))).await;
    assert_eq!(calls, [1, 1]);
    assert_post(
        response,
        &metrics,
        ROUTE,
        StatusCode::OK,
        json!({"object":"lora.adapter","status":"unloaded","id":41}),
    )
    .await;
    let (response, metrics, calls) =
        run_unload("", Some(Err(AdapterControlError::NotFound(7)))).await;
    assert_eq!(calls, [1, 1]);
    assert_post(
        response,
        &metrics,
        ROUTE,
        StatusCode::BAD_REQUEST,
        error_body(
            StatusCode::BAD_REQUEST,
            "lora_adapter_not_found",
            "unknown LoRA adapter id 7",
        ),
    )
    .await;
    for (stage, reply, message) in [
        (
            "",
            Some(Ok(AdapterControlResult::Loaded(confirmed_adapter()))),
            "unexpected adapter command reply",
        ),
        (
            "",
            None,
            "the inference worker is not running; no adapter was unloaded",
        ),
        ("submit", None, "control slot occupied"),
    ] {
        let (response, metrics, calls) = run_unload(stage, reply).await;
        assert_eq!(calls, [1, 1]);
        assert_post(
            response,
            &metrics,
            ROUTE,
            StatusCode::SERVICE_UNAVAILABLE,
            error_body(StatusCode::SERVICE_UNAVAILABLE, "server_busy", message),
        )
        .await;
    }
    let (response, metrics, calls) = run_unload("runtime", None).await;
    assert_eq!(calls, [1, 0]);
    assert_post(
        response,
        &metrics,
        ROUTE,
        StatusCode::BAD_REQUEST,
        error_body(
            StatusCode::BAD_REQUEST,
            "lora_unsupported_backend",
            "runtime refused",
        ),
    )
    .await;
}

#[tokio::test]
async fn lora_list_wrapper_preserves_confirmed_snapshot() {
    let calls = [Cell::new(0), Cell::new(0)];
    let response = lora_list_with_backend(
        || {
            calls[0].set(calls[0].get() + 1);
            Ok(7)
        },
        |backend| {
            calls[1].set(calls[1].get() + 1);
            assert_eq!(backend, 7);
            AdapterIndex {
                adapters: vec![confirmed_adapter()],
                applied: vec![
                    serde_json::from_value::<LoraSelection>(json!({
                        "id": 41,
                        "scale": 0.25,
                    }))
                    .unwrap(),
                ],
                blend_refusal: None,
            }
        },
    );
    assert_eq!(calls.map(|calls| calls.get()), [1, 1]);
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&bytes).unwrap(),
        json!({"adapters":[{"id":41,"name":"confirmed-name","path":"confirmed-path",
                        "rank":3,"layers":5}],"applied":[{"id":41,"scale":0.25}],
                        "router":{"enabled":false}}),
    );
    let snapshot_calls = Cell::new(0);
    let response = lora_list_with_backend(
        || {
            Err::<(), _>(ApiError::BadRequest {
                code: "lora_unsupported_backend",
                message: "runtime refused".into(),
            })
        },
        |_| {
            snapshot_calls.set(snapshot_calls.get() + 1);
            AdapterIndex::default()
        },
    );
    assert_eq!(snapshot_calls.get(), 0);
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&bytes).unwrap(),
        error_body(
            StatusCode::BAD_REQUEST,
            "lora_unsupported_backend",
            "runtime refused"
        )
    );
    let state = test_app_state();
    assert_eq!(
        lora_list(State(state.clone())).await.status(),
        StatusCode::OK
    );
    let metrics = state.metrics.render(&state.model_id, 0);
    assert!(
        !metrics
            .lines()
            .any(|line| line.starts_with("lattice_http_requests_total{")
                || line.starts_with("lattice_http_request_duration_seconds_count{")),
        "{metrics}"
    );
}
