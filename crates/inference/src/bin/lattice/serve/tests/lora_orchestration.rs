use super::*;
use axum::body::Body;
use lattice_inference::serve::lora::{
    AdapterControlError, AdapterControlReceiver, AdapterControlResult, AdapterIndex,
    AdapterMetadata,
};
use std::cell::RefCell;

fn json_headers() -> axum::http::HeaderMap {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    headers
}

fn confirmed() -> AdapterMetadata {
    AdapterMetadata {
        id: 81,
        name: "confirmed-name".into(),
        path: "/resident.safetensors".into(),
        rank: 8,
        layers: 4,
    }
}

fn reply(
    outcome: Option<Result<AdapterControlResult, AdapterControlError>>,
) -> AdapterControlReceiver {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    if let Some(outcome) = outcome {
        sender.send(outcome).unwrap();
    }
    receiver
}

async fn response(result: Result<Response, ApiError>) -> (StatusCode, Value) {
    let response = result.unwrap_or_else(IntoResponse::into_response);
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn assert_error(
    result: Result<Response, ApiError>,
    status: StatusCode,
    code: &str,
    message: &str,
) {
    assert_eq!(
        response(result).await,
        (
            status,
            serde_json::json!({
                "error": {
                    "message": message,
                    "type": if status.is_server_error() { "server_error" } else { "invalid_request_error" },
                    "code": code,
                    "param": null,
                }
            })
        )
    );
}

async fn load_reply(
    outcome: Option<Result<AdapterControlResult, AdapterControlError>>,
) -> Result<Response, ApiError> {
    let calls = RefCell::new(Vec::new());
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        lora_load_with_backend(
            json_headers(),
            Body::from(r#"{"path":"/request.safetensors","name":"request-name"}"#),
            Ok(()),
            || {
                calls.borrow_mut().push("resolve");
                Ok(42)
            },
            |path, name| {
                calls.borrow_mut().push("prepare");
                assert_eq!((path, name), ("/request.safetensors", "request-name"));
                Ok("prepared-command")
            },
            |backend, prepared| {
                calls.borrow_mut().push("submit");
                assert_eq!((backend, prepared), (42, "prepared-command"));
                Ok(reply(outcome))
            },
        ),
    )
    .await
    .expect("fake control must complete without a worker");
    assert_eq!(*calls.borrow(), ["resolve", "prepare", "submit"]);
    result
}

async fn unload_reply(
    outcome: Option<Result<AdapterControlResult, AdapterControlError>>,
) -> Result<Response, ApiError> {
    let calls = RefCell::new(Vec::new());
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        lora_unload_with_backend(
            json_headers(),
            Body::from(r#"{"id":7}"#),
            Ok(()),
            || {
                calls.borrow_mut().push("resolve");
                Ok(42)
            },
            |backend, id| {
                calls.borrow_mut().push("submit");
                assert_eq!((backend, id), (42, 7));
                Ok(reply(outcome))
            },
        ),
    )
    .await
    .expect("fake control must complete without a worker");
    assert_eq!(*calls.borrow(), ["resolve", "submit"]);
    result
}

#[tokio::test]
async fn lora_load_wrapper_preserves_success_and_failures() {
    assert_eq!(
        response(load_reply(Some(Ok(AdapterControlResult::Loaded(confirmed())))).await).await,
        (
            StatusCode::OK,
            serde_json::json!({
                "object":"lora.adapter", "status":"loaded", "id":81,
                "name":"confirmed-name", "path":"/resident.safetensors", "rank":8, "layers":4,
            })
        )
    );
    assert_error(
        load_reply(Some(Err(AdapterControlError::NotFound(99)))).await,
        StatusCode::BAD_REQUEST,
        "lora_adapter_not_found",
        "unknown LoRA adapter id 99",
    )
    .await;
}

#[tokio::test]
async fn lora_unload_wrapper_preserves_success_and_failures() {
    assert_eq!(
        response(unload_reply(Some(Ok(AdapterControlResult::Unloaded(81)))).await).await,
        (
            StatusCode::OK,
            serde_json::json!({"object":"lora.adapter", "status":"unloaded", "id":81})
        )
    );
    assert_error(
        unload_reply(Some(Err(AdapterControlError::NotFound(99)))).await,
        StatusCode::BAD_REQUEST,
        "lora_adapter_not_found",
        "unknown LoRA adapter id 99",
    )
    .await;
}

#[tokio::test]
async fn lora_list_wrapper_preserves_confirmed_snapshot() {
    let calls = RefCell::new(Vec::new());
    let result = lora_list_with_backend(
        Ok(()),
        || {
            calls.borrow_mut().push("resolve");
            Ok(42)
        },
        |backend| {
            assert_eq!(backend, 42);
            calls.borrow_mut().push("snapshot");
            AdapterIndex {
                adapters: vec![confirmed()],
                applied: vec![serde_json::from_str(r#"{"id":81,"scale":0.5}"#).unwrap()],
                ..Default::default()
            }
        },
        None,
    );
    assert_eq!(*calls.borrow(), ["resolve", "snapshot"]);
    assert_eq!(
        response(result).await,
        (
            StatusCode::OK,
            serde_json::json!({
                "adapters":[{"id":81,"name":"confirmed-name","path":"/resident.safetensors","rank":8,"layers":4}],
                "applied":[{"id":81,"scale":0.5}], "router":{"enabled":false},
            })
        )
    );
    assert_error(
        lora_list_with_backend::<()>(
            Err(lora_unsupported_backend("fixture build refusal")),
            || panic!("build refusal must not resolve"),
            |()| panic!("build refusal must not snapshot"),
            None,
        ),
        StatusCode::BAD_REQUEST,
        "lora_unsupported_backend",
        "fixture build refusal",
    )
    .await;
    assert_error(
        lora_list_with_backend::<()>(
            Ok(()),
            || Err(lora_unsupported_backend("fixture runtime refusal")),
            |()| panic!("runtime refusal must not snapshot"),
            None,
        ),
        StatusCode::BAD_REQUEST,
        "lora_unsupported_backend",
        "fixture runtime refusal",
    )
    .await;
}

#[cfg(feature = "mixture")]
#[tokio::test]
async fn lora_list_wrapper_preserves_served_router_report() {
    use lattice_fann::{Activation, NetworkBuilder};
    use lattice_inference::router_state::{
        ResolvedRouter, RouterArtifact, RouterProvenance, TrainedRepresentation,
    };
    use lattice_inference::serve::routing::{EmbedderIdentity, ServedRouter};

    let artifact = RouterArtifact {
        version: 7,
        adapter_names: vec!["confirmed-name".into()],
        representation: TrainedRepresentation {
            embedding_model: "gme-qwen35".into(),
            pooling: "mean_visual".into(),
            prompt_source: "last_user_message".into(),
            loader_format: lattice_inference::serve::embeddings::QWEN35_F16_DECODER_LOADER.into(),
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
    let result = lora_list_with_backend(
        Ok(()),
        || Ok(()),
        |()| AdapterIndex {
            adapters: vec![confirmed()],
            applied: vec![serde_json::from_str(r#"{"id":81,"scale":0.5}"#).unwrap()],
            ..Default::default()
        },
        Some(&served),
    );
    assert_eq!(
        response(result).await,
        (
            StatusCode::OK,
            serde_json::json!({
                "adapters":[{"id":81,"name":"confirmed-name","path":"/resident.safetensors","rank":8,"layers":4}],
                "applied":[{"id":81,"scale":0.5}],
                "router": {
                    "enabled":true, "version":expected_version, "pinned":true, "embedder":"gme-qwen35",
                    "adapter_names":["confirmed-name"], "routable":true, "missing":[], "unexpected":[],
                    "duplicate_trained":null, "duplicate_resident":null, "blend_refusal":null,
                    "training_state":"unrecorded",
                    "policy": {
                        "kind":"uniform", "tau":null, "epsilon":0.0,
                        "exploration_scale":null, "learning_enabled":false,
                    },
                },
            })
        )
    );
}
