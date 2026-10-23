use super::*;
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse as _, Response};
use std::cell::RefCell;

fn json_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("content-type", "application/json".parse().unwrap());
    headers
}

fn refusal(message: &str) -> ApiError {
    ApiError::BadRequest {
        code: "lora_unsupported_backend",
        message: message.into(),
    }
}

fn unread_body() -> Body {
    Body::from_stream(futures::stream::poll_fn(
        |_| -> std::task::Poll<Option<Result<Bytes, std::io::Error>>> {
            panic!("body must not be polled")
        },
    ))
}

async fn value(response: Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn assert_error(error: ApiError, status: StatusCode, code: &str, message: &str) {
    let response = error.into_response();
    assert_eq!(response.status(), status);
    assert_eq!(
        value(response).await,
        serde_json::json!({"error": {
            "message": message,
            "type": if status == StatusCode::SERVICE_UNAVAILABLE {
                "server_error"
            } else {
                "invalid_request_error"
            },
            "code": code,
            "param": null,
        }})
    );
}

async fn before_backend(
    load: bool,
    headers: HeaderMap,
    body: Body,
    build: Result<(), ApiError>,
) -> ApiError {
    if load {
        load_request::<(), ()>(
            headers,
            body,
            build,
            || panic!("runtime must not be resolved"),
            |_, _| panic!("load must not be prepared"),
            |_, _| panic!("load must not be submitted"),
        )
        .await
        .unwrap_err()
    } else {
        unload_request::<()>(
            headers,
            body,
            build,
            || panic!("runtime must not be resolved"),
            |_, _| panic!("unload must not be submitted"),
        )
        .await
        .unwrap_err()
    }
}

fn confirmed() -> AdapterMetadata {
    AdapterMetadata {
        id: 71,
        name: "confirmed name".into(),
        path: "/confirmed/path".into(),
        rank: 16,
        layers: 27,
    }
}

fn receiver(
    reply: Option<Result<AdapterControlResult, AdapterControlError>>,
) -> AdapterControlReceiver {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    if let Some(reply) = reply {
        sender.send(reply).unwrap();
    }
    receiver
}

async fn with_reply(
    load: bool,
    reply: Option<Result<AdapterControlResult, AdapterControlError>>,
) -> Result<Response, ApiError> {
    if load {
        load_request(
            json_headers(),
            Body::from(r#"{"path":"requested","name":"request name"}"#),
            Ok(()),
            || Ok(()),
            |_, _| Ok(()),
            |(), ()| Ok(receiver(reply)),
        )
        .await
    } else {
        unload_request(
            json_headers(),
            Body::from(r#"{"id":4}"#),
            Ok(()),
            || Ok(()),
            |(), _| Ok(receiver(reply)),
        )
        .await
    }
}

#[tokio::test]
async fn build_refusal_precedes_headers_body_and_backend() {
    for load in [true, false] {
        let error =
            before_backend(load, HeaderMap::new(), unread_body(), Err(refusal("build"))).await;
        assert_error(
            error,
            StatusCode::BAD_REQUEST,
            "lora_unsupported_backend",
            "build",
        )
        .await;
    }
}

#[tokio::test]
async fn content_type_refusal_precedes_body_and_backend() {
    for load in [true, false] {
        for wrong_type in [false, true] {
            let mut headers = HeaderMap::new();
            if wrong_type {
                headers.insert("content-type", "text/plain".parse().unwrap());
            }
            let error = before_backend(load, headers, unread_body(), Ok(())).await;
            assert_error(
                error,
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/json",
            )
            .await;
        }
    }
}

#[tokio::test]
async fn an_oversized_body_precedes_backend_and_is_refused_as_too_large() {
    for load in [true, false] {
        let oversized = Body::from(vec![b' '; super::super::REQUEST_BODY_LIMIT_BYTES + 1]);
        let error = before_backend(load, json_headers(), oversized, Ok(())).await;
        assert_error(
            error,
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_body_too_large",
            "request body exceeds 1 MiB limit",
        )
        .await;
    }
}

#[tokio::test]
async fn a_broken_body_precedes_backend_and_keeps_the_shared_error() {
    for load in [true, false] {
        let broken = Body::from_stream(futures::stream::once(async {
            Err::<Bytes, _>(std::io::Error::other("broken body"))
        }));
        let error = before_backend(load, json_headers(), broken, Ok(())).await;
        assert_error(
            error,
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "invalid request body",
        )
        .await;
    }
}

#[tokio::test]
async fn strict_parsing_precedes_runtime_admission() {
    for (load, inputs) in [
        (
            true,
            &[
                "{",
                "{}",
                r#"{"path":"x","extra":1}"#,
                r#"{"path":" "}"#,
                r#"{"path":"x","name":" "}"#,
                r#"{"path":3}"#,
            ][..],
        ),
        (
            false,
            &[
                "{",
                "{}",
                r#"{"id":1,"extra":1}"#,
                r#"{"id":-1}"#,
                r#"{"id":4294967296}"#,
                r#"{"id":"1"}"#,
            ][..],
        ),
    ] {
        for input in inputs {
            let expected = if load {
                parse_lora_load(input.as_bytes()).unwrap_err()
            } else {
                parse_lora_unload(input.as_bytes()).unwrap_err()
            };
            let error = before_backend(load, json_headers(), Body::from(*input), Ok(())).await;
            assert_error(
                error,
                StatusCode::BAD_REQUEST,
                "invalid_request",
                expected.message(),
            )
            .await;
        }
    }
}

#[tokio::test]
async fn runtime_refusal_precedes_preparation_and_submission() {
    let load = load_request::<(), ()>(
        json_headers(),
        Body::from(r#"{"path":"requested"}"#),
        Ok(()),
        || Err(refusal("runtime")),
        |_, _| panic!("runtime refusal must precede preparation"),
        |_, _| panic!("runtime refusal must precede submission"),
    )
    .await
    .unwrap_err();
    let unload = unload_request::<()>(
        json_headers(),
        Body::from(r#"{"id":4}"#),
        Ok(()),
        || Err(refusal("runtime")),
        |_, _| panic!("runtime refusal must precede submission"),
    )
    .await
    .unwrap_err();
    for error in [load, unload] {
        assert_error(
            error,
            StatusCode::BAD_REQUEST,
            "lora_unsupported_backend",
            "runtime",
        )
        .await;
    }
}

#[tokio::test]
async fn preparation_errors_precede_submission() {
    for code in ["lora_adapter_not_found", "lora_load_failed"] {
        let calls = RefCell::new(Vec::new());
        let error = load_request::<(), ()>(
            json_headers(),
            Body::from(r#"{"path":"requested"}"#),
            Ok(()),
            || {
                calls.borrow_mut().push("resolve");
                Ok(())
            },
            |path, name| {
                calls.borrow_mut().push("prepare");
                assert_eq!((path, name), ("requested", "requested"));
                Err(ApiError::BadRequest {
                    code,
                    message: "preparation diagnosis".into(),
                })
            },
            |_, _| panic!("failed preparation must not submit"),
        )
        .await
        .unwrap_err();
        assert_eq!(*calls.borrow(), ["resolve", "prepare"]);
        assert_error(
            error,
            StatusCode::BAD_REQUEST,
            code,
            "preparation diagnosis",
        )
        .await;
    }
}

#[tokio::test]
async fn submit_errors_preserve_status_and_diagnosis() {
    let load = load_request(
        json_headers(),
        Body::from(r#"{"path":"requested"}"#),
        Ok(()),
        || Ok(()),
        |_, _| Ok(()),
        |(), ()| Err(worker_unavailable("control slot busy")),
    )
    .await
    .unwrap_err();
    let unload = unload_request(
        json_headers(),
        Body::from(r#"{"id":4}"#),
        Ok(()),
        || Ok(()),
        |(), id| {
            assert_eq!(id, 4);
            Err(worker_unavailable("control slot busy"))
        },
    )
    .await
    .unwrap_err();
    for error in [load, unload] {
        assert_error(
            error,
            StatusCode::SERVICE_UNAVAILABLE,
            "server_busy",
            "control slot busy",
        )
        .await;
    }
}

#[tokio::test]
async fn load_preserves_literal_input_order_and_confirmed_metadata() {
    let calls = RefCell::new(Vec::new());
    let response = load_request(
        json_headers(),
        Body::from(r#"{"path":" requested ","name":" request name "}"#),
        Ok(()),
        || {
            calls.borrow_mut().push("resolve");
            Ok(17)
        },
        |path, name| {
            calls.borrow_mut().push("prepare");
            assert_eq!((path, name), (" requested ", " request name "));
            Ok((path.to_owned(), name.to_owned(), 2, 3))
        },
        |backend, prepared| {
            calls.borrow_mut().push("submit");
            assert_eq!(backend, 17);
            assert_eq!(
                prepared,
                (" requested ".into(), " request name ".into(), 2, 3)
            );
            Ok(receiver(Some(Ok(
                AdapterControlResult::Loaded(confirmed()),
            ))))
        },
    )
    .await
    .unwrap();
    assert_eq!(*calls.borrow(), ["resolve", "prepare", "submit"]);
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        value(response).await,
        serde_json::json!({
            "object":"lora.adapter", "status":"loaded", "id":71,
            "name":"confirmed name", "path":"/confirmed/path", "rank":16, "layers":27,
        })
    );
}

#[tokio::test]
async fn unload_submits_requested_id_and_returns_confirmed_id() {
    let calls = RefCell::new(Vec::new());
    let response = unload_request(
        json_headers(),
        Body::from(r#"{"id":4}"#),
        Ok(()),
        || {
            calls.borrow_mut().push("resolve");
            Ok(17)
        },
        |backend, id| {
            calls.borrow_mut().push("submit");
            assert_eq!((backend, id), (17, 4));
            Ok(receiver(Some(Ok(AdapterControlResult::Unloaded(71)))))
        },
    )
    .await
    .unwrap();
    assert_eq!(*calls.borrow(), ["resolve", "submit"]);
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        value(response).await,
        serde_json::json!({"object":"lora.adapter", "status":"unloaded", "id":71})
    );
}

#[tokio::test]
async fn every_worker_error_keeps_its_code_and_message() {
    for load in [true, false] {
        for (error, code, message) in [
            (
                AdapterControlError::InvalidAdapter("bad tensor".into()),
                "lora_load_failed",
                "bad tensor",
            ),
            (
                AdapterControlError::NotFound(8),
                "lora_adapter_not_found",
                "unknown LoRA adapter id 8",
            ),
            (
                AdapterControlError::CountLimit { limit: 2 },
                "lora_residency_limit_exceeded",
                "resident LoRA adapter count limit reached (2)",
            ),
            (
                AdapterControlError::ByteLimit { limit: 32 },
                "lora_residency_limit_exceeded",
                "resident LoRA adapter tensor payload byte limit exceeded (32)",
            ),
            (
                AdapterControlError::IdExhausted,
                "lora_load_failed",
                "LoRA adapter id space exhausted",
            ),
        ] {
            let error = with_reply(load, Some(Err(error))).await.unwrap_err();
            assert_error(error, StatusCode::BAD_REQUEST, code, message).await;
        }
    }
}

#[tokio::test]
async fn wrong_reply_variants_refuse_in_both_operations() {
    for (load, reply) in [
        (true, AdapterControlResult::Unloaded(71)),
        (false, AdapterControlResult::Loaded(confirmed())),
    ] {
        let error = with_reply(load, Some(Ok(reply))).await.unwrap_err();
        assert_error(
            error,
            StatusCode::SERVICE_UNAVAILABLE,
            "server_busy",
            "unexpected adapter command reply",
        )
        .await;
    }
}

#[tokio::test]
async fn dropped_replies_keep_operation_specific_diagnoses() {
    for (load, message) in [
        (
            true,
            "the inference worker is not running; the adapter was not loaded",
        ),
        (
            false,
            "the inference worker is not running; no adapter was unloaded",
        ),
    ] {
        let error = with_reply(load, None).await.unwrap_err();
        assert_error(
            error,
            StatusCode::SERVICE_UNAVAILABLE,
            "server_busy",
            message,
        )
        .await;
    }
}

#[tokio::test]
async fn list_refusals_precede_snapshot_access() {
    let build = list_request::<()>(
        Err(refusal("build")),
        || panic!("build refusal must precede runtime"),
        |()| panic!("build refusal must precede snapshot"),
        None,
    )
    .unwrap_err();
    let runtime = list_request::<()>(
        Ok(()),
        || Err(refusal("runtime")),
        |()| panic!("runtime refusal must precede snapshot"),
        None,
    )
    .unwrap_err();
    assert_error(
        build,
        StatusCode::BAD_REQUEST,
        "lora_unsupported_backend",
        "build",
    )
    .await;
    assert_error(
        runtime,
        StatusCode::BAD_REQUEST,
        "lora_unsupported_backend",
        "runtime",
    )
    .await;
}

#[tokio::test]
async fn list_reads_one_confirmed_snapshot_without_a_router() {
    let calls = RefCell::new(Vec::new());
    let response = list_request(
        Ok(()),
        || {
            calls.borrow_mut().push("resolve");
            Ok(17)
        },
        |backend| {
            calls.borrow_mut().push("snapshot");
            assert_eq!(backend, 17);
            AdapterIndex {
                adapters: vec![confirmed()],
                applied: vec![LoraSelection { id: 71, scale: 0.5 }],
                ..Default::default()
            }
        },
        None,
    )
    .unwrap();
    assert_eq!(*calls.borrow(), ["resolve", "snapshot"]);
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        value(response).await,
        serde_json::json!({
            "adapters":[{"id":71,"name":"confirmed name","path":"/confirmed/path","rank":16,"layers":27}],
            "applied":[{"id":71,"scale":0.5}], "router":{"enabled":false},
        })
    );
}

#[cfg(feature = "mixture")]
#[tokio::test]
async fn list_reports_the_served_router_beside_residency() {
    use crate::router_state::{
        ResolvedRouter, RouterArtifact, RouterProvenance, TrainedRepresentation,
    };
    use crate::serve::routing::{EmbedderIdentity, ServedRouter};
    use lattice_fann::{Activation, NetworkBuilder};

    let artifact = RouterArtifact {
        version: 7,
        adapter_names: vec!["confirmed name".into()],
        representation: TrainedRepresentation {
            embedding_model: "gme-qwen35".into(),
            pooling: "mean_visual".into(),
            prompt_source: "last_user_message".into(),
            loader_format: crate::serve::embeddings::QWEN35_F16_DECODER_LOADER.into(),
            input_width: 4,
        },
        gate_bytes: NetworkBuilder::new()
            .input(4)
            .output(1, Activation::Linear)
            .build()
            .unwrap()
            .to_bytes(),
        provenance: RouterProvenance::Legacy,
    };
    let expected_version = artifact.version_label();
    let identity = EmbedderIdentity::resolve(std::path::Path::new("/models"), Some("gme-qwen35"));
    let served = ServedRouter::new(
        ResolvedRouter {
            artifact,
            pinned: true,
        },
        &identity,
        4,
        false,
    )
    .unwrap();
    let response = list_request(
        Ok(()),
        || Ok(()),
        |()| AdapterIndex {
            adapters: vec![confirmed()],
            ..Default::default()
        },
        Some(&served),
    )
    .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = value(response).await;
    assert_eq!(body["adapters"][0]["id"], 71);
    assert_eq!(body["applied"], serde_json::json!([]));
    assert_eq!(
        body["router"],
        serde_json::json!({
            "enabled":true, "version":expected_version, "pinned":true, "embedder":"gme-qwen35",
            "adapter_names":["confirmed name"], "routable":true, "missing":[], "unexpected":[],
            "duplicate_trained":null, "duplicate_resident":null, "blend_refusal":null,
            "training_state":"unrecorded",
            "policy":{
                "kind":"uniform", "tau":null, "epsilon":0.0,
                "exploration_scale":null, "learning_enabled":false,
            },
        })
    );
}

#[test]
fn native_compatible_callbacks_produce_send_futures() {
    fn assert_send<T: Send>(_: T) {}
    assert_send(load_request::<(), ()>(
        json_headers(),
        Body::empty(),
        Ok(()),
        || Ok(()),
        |_, _| Ok(()),
        |(), ()| Ok(receiver(None)),
    ));
    assert_send(unload_request::<()>(
        json_headers(),
        Body::empty(),
        Ok(()),
        || Ok(()),
        |(), _| Ok(receiver(None)),
    ));
}
