use super::*;

#[derive(Clone)]
struct RouterDropProbe {
    cohort: std::sync::Arc<()>,
    dropped: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Drop for RouterDropProbe {
    fn drop(&mut self) {
        if std::sync::Arc::strong_count(&self.cohort) == 1 {
            self.dropped
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

#[derive(Clone)]
struct StuckDrainState {
    started: tokio::sync::mpsc::UnboundedSender<()>,
    finished: tokio::sync::mpsc::UnboundedSender<()>,
    release: std::sync::Arc<tokio::sync::Notify>,
}

async fn stuck_drain_handler(
    axum::extract::State(state): axum::extract::State<StuckDrainState>,
) -> &'static str {
    let _ = state.started.send(());
    state.release.notified().await;
    let _ = state.finished.send(());
    "released"
}

#[tokio::test]
async fn shared_server_runner_drops_router_after_injected_shutdown() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback listener must bind");
    let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let probe = RouterDropProbe {
        cohort: std::sync::Arc::new(()),
        dropped: dropped.clone(),
    };
    let app = axum::Router::new().layer(axum::Extension(probe));
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    let server = tokio::spawn(serve_with_shutdown(
        listener,
        app,
        async move {
            let _ = shutdown_rx.await;
        },
        Duration::from_secs(1),
    ));
    shutdown_tx
        .send(())
        .expect("server must still own the shutdown receiver");
    tokio::time::timeout(std::time::Duration::from_secs(1), server)
        .await
        .expect("shared runner must honor its shutdown future")
        .expect("shared runner task must not panic")
        .expect("shared runner must stop cleanly");

    assert!(
        dropped.load(std::sync::atomic::Ordering::SeqCst),
        "router state must be dropped before the shared runner returns"
    );
}

#[tokio::test]
async fn shared_server_runner_bounds_a_stuck_connection_drain() {
    use tokio::io::AsyncWriteExt as _;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback listener must bind");
    let address = listener
        .local_addr()
        .expect("bound listener must expose its local address");
    let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
    let (finished_tx, mut finished_rx) = tokio::sync::mpsc::unbounded_channel();
    let release = std::sync::Arc::new(tokio::sync::Notify::new());
    let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let probe = RouterDropProbe {
        cohort: std::sync::Arc::new(()),
        dropped: dropped.clone(),
    };
    let app = axum::Router::new()
        .route("/stuck", axum::routing::get(stuck_drain_handler))
        .with_state(StuckDrainState {
            started: started_tx,
            finished: finished_tx,
            release: release.clone(),
        })
        .layer(axum::Extension(probe));
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let mut server = tokio::spawn(serve_with_shutdown(
        listener,
        app,
        async move {
            let _ = shutdown_rx.await;
        },
        Duration::from_millis(20),
    ));

    let mut client = tokio::net::TcpStream::connect(address)
        .await
        .expect("test client must connect");
    client
        .write_all(b"GET /stuck HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .expect("test client must write its request");
    tokio::time::timeout(Duration::from_secs(1), started_rx.recv())
        .await
        .expect("stuck handler must start before shutdown")
        .expect("stuck handler start channel must remain open");
    shutdown_tx
        .send(())
        .expect("server must still own the shutdown receiver");

    let bounded = tokio::time::timeout(Duration::from_millis(500), &mut server).await;
    if bounded.is_err() {
        release.notify_waiters();
        drop(client);
        let _ = tokio::time::timeout(Duration::from_secs(1), &mut server).await;
        panic!("stuck connection drain must not bypass the configured deadline");
    }
    let error = bounded
        .expect("checked above")
        .expect("shared runner task must not panic")
        .expect_err("stuck connection drain must return a timeout error");
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    let message = error.to_string();
    for required_warning in [
        "hard process exit",
        "in-flight responses",
        "partially written files",
        "unflushed telemetry",
    ] {
        assert!(
            message.contains(required_warning),
            "timeout error must warn operators about {required_warning}: {message}"
        );
    }
    assert!(
        dropped.load(std::sync::atomic::Ordering::SeqCst),
        "forced connection cancellation must drop router state before returning"
    );

    release.notify_waiters();
    drop(client);
    assert!(
        tokio::time::timeout(Duration::from_millis(50), finished_rx.recv())
            .await
            .expect("aborted handler finish channel must close promptly")
            .is_none(),
        "timed-out handler must be cancelled rather than detached"
    );
}

#[test]
fn both_server_binaries_use_shared_graceful_runner() {
    // `lattice` is now a module tree (#834), not a single file; concatenate
    // its split source files so the substring/window checks below still
    // see the same source text as when it was one file.
    let lattice = concat!(
        include_str!("../bin/lattice/main.rs"),
        include_str!("../bin/lattice/chat.rs"),
        include_str!("../bin/lattice/doctor.rs"),
        include_str!("../bin/lattice/serve.rs"),
    );
    let lattice_serve = include_str!("../bin/lattice_serve/main.rs");
    for (name, source) in [("lattice", lattice), ("lattice_serve", lattice_serve)] {
        assert!(
            source.contains("serve::serve_until_shutdown(listener, app)"),
            "{name} must route process signals through the shared graceful runner"
        );
        assert!(
            !source.contains("axum::serve(listener, app)"),
            "{name} must not bypass the shared graceful runner"
        );
        let call = source
            .find("serve::serve_until_shutdown(listener, app)")
            .expect("shared graceful runner call must exist");
        let hard_exit_boundary = &source[call..source.len().min(call + 512)];
        assert!(
            hard_exit_boundary.contains("std::process::exit(1);"),
            "{name} must hard-exit if bounded connection draining fails"
        );
    }
    assert!(
        lattice.contains("drop(app);"),
        "lattice bind failure must drop router state before process::exit"
    );
}

#[test]
fn contract_to_engine_message_adapter_preserves_roles_and_content() {
    let normalized = vec![
        contract::NormalizedChatMessage {
            role: contract::NormalizedChatRole::System,
            content: "system-content".to_string(),
            image: None,
        },
        contract::NormalizedChatMessage {
            role: contract::NormalizedChatRole::User,
            content: "user-content".to_string(),
            image: None,
        },
        contract::NormalizedChatMessage {
            role: contract::NormalizedChatRole::Assistant,
            content: "assistant-content".to_string(),
            image: None,
        },
    ];
    let rendered = format_normalized_chat_template(&normalized);
    let owned =
        into_engine_chat_messages(normalized).expect("valid messages must reach the engine");
    assert_eq!(
        rendered,
        crate::forward::metal_qwen35::format_chat_template(&owned)
    );

    let expected = [
        (
            crate::forward::metal_qwen35::ChatRole::System,
            "system-content",
        ),
        (crate::forward::metal_qwen35::ChatRole::User, "user-content"),
        (
            crate::forward::metal_qwen35::ChatRole::Assistant,
            "assistant-content",
        ),
    ];
    for (owned, (role, content)) in owned.iter().zip(expected) {
        assert_eq!(owned.role, role);
        assert_eq!(owned.content, content);
    }
}

#[test]
fn contract_to_engine_message_adapter_preserves_image_and_position() {
    let owned = into_engine_chat_messages(vec![contract::NormalizedChatMessage {
        role: contract::NormalizedChatRole::User,
        content: "beforeafter".to_string(),
        image: Some(contract::NormalizedChatImage {
            bytes: vec![0x89, b'P', b'N', b'G'],
            text_offset: "before".len(),
        }),
    }])
    .expect("valid user image must reach the engine");
    let image = owned[0]
        .image
        .as_ref()
        .expect("normalized image must reach the engine message");
    assert_eq!(image.bytes, [0x89, b'P', b'N', b'G']);
    assert_eq!(image.text_offset, "before".len());
}

#[test]
fn contract_to_engine_message_adapter_rejects_non_user_image_without_panicking() {
    let error = into_engine_chat_messages(vec![contract::NormalizedChatMessage {
        role: contract::NormalizedChatRole::System,
        content: "policy".to_string(),
        image: Some(contract::NormalizedChatImage {
            bytes: vec![0x89, b'P', b'N', b'G'],
            text_offset: 0,
        }),
    }])
    .expect_err("caller-constructed invalid normalized state must fail closed");
    assert!(matches!(
        error,
        ApiError::BadRequest {
            code: "invalid_image_role",
            ..
        }
    ));
}
#[test]
fn finish_reason_stopped_true_is_stop() {
    assert_eq!(finish_reason(true), "stop");
}

#[test]
fn finish_reason_stopped_false_is_length() {
    assert_eq!(finish_reason(false), "length");
}

#[test]
fn reject_zero_max_tokens_rejects_zero() {
    let err = reject_zero_max_tokens(0).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_max_tokens",
            ..
        }
    ));
}

#[test]
fn reject_zero_max_tokens_accepts_positive() {
    assert!(reject_zero_max_tokens(1).is_ok());
    assert!(reject_zero_max_tokens(4096).is_ok());
}

#[test]
fn root_body_shape() {
    let body = root_body();
    assert_eq!(body["name"], "lattice");
    assert_eq!(body["object"], "engine");
    assert_eq!(
        body["endpoints"],
        serde_json::json!(["/v1/chat/completions", "/v1/models", "/health"])
    );
}

#[test]
fn models_list_body_shape() {
    let body = models_list_body("my-model", 1_700_000_000);
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "my-model");
    assert_eq!(body["data"][0]["object"], "model");
    assert_eq!(body["data"][0]["created"], 1_700_000_000);
    assert_eq!(body["data"][0]["owned_by"], "lattice");
}

fn headers_with_content_type(value: &str) -> axum::http::HeaderMap {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_str(value).unwrap(),
    );
    headers
}

#[test]
fn require_json_content_type_accepts_application_json() {
    require_json_content_type(&headers_with_content_type("application/json")).unwrap();
}

#[test]
fn require_json_content_type_accepts_json_with_charset_param() {
    // axum's own rule strips parameters before comparing type/subtype.
    require_json_content_type(&headers_with_content_type(
        "application/json; charset=utf-8",
    ))
    .unwrap();
}

#[test]
fn require_json_content_type_accepts_structured_suffix() {
    require_json_content_type(&headers_with_content_type("application/vnd.api+json")).unwrap();
}

#[test]
fn require_json_content_type_rejects_text_plain() {
    let err = require_json_content_type(&headers_with_content_type("text/plain")).unwrap_err();
    assert!(matches!(err, ApiError::UnsupportedMediaType { .. }));
    assert_eq!(err.code(), "unsupported_media_type");
}

#[test]
fn require_json_content_type_rejects_missing_header() {
    let err = require_json_content_type(&axum::http::HeaderMap::new()).unwrap_err();
    assert!(matches!(err, ApiError::UnsupportedMediaType { .. }));
}

#[test]
fn require_json_content_type_rejects_unparsable_header() {
    // A raw byte sequence that fails `to_str()` -- the header value
    // isn't a valid mime token at all.
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_bytes(&[0xff, 0xfe]).unwrap(),
    );
    let err = require_json_content_type(&headers).unwrap_err();
    assert!(matches!(err, ApiError::UnsupportedMediaType { .. }));
}

#[test]
fn unsupported_media_type_into_response_is_415() {
    let response = (ApiError::UnsupportedMediaType {
        message: "Content-Type must be application/json".to_string(),
    })
    .into_response();
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[test]
fn api_error_bad_request_envelope_shape() {
    let err = ApiError::BadRequest {
        message: "bad".to_string(),
        code: "some_code",
    };
    assert_eq!(err.message(), "bad");
    // IntoResponse is exercised at the HTTP layer in each binary's own
    // tests (axum::response::Response has no public body-introspection
    // API worth duplicating here); this pins the pure data this module
    // owns instead.
}

#[test]
fn api_error_internal_message() {
    let err = ApiError::Internal {
        message: "oops".to_string(),
    };
    assert_eq!(err.message(), "oops");
}

#[test]
fn api_error_payload_too_large_message() {
    let err = ApiError::PayloadTooLarge {
        message: "too big".to_string(),
    };
    assert_eq!(err.message(), "too big");
}

#[test]
fn cancel_pair_receiver_starts_false() {
    let (_guard, rx) = cancel_pair();
    assert!(!*rx.borrow());
}

#[test]
fn cancel_on_drop_flips_receiver_true_on_drop() {
    let (guard, rx) = cancel_pair();
    assert!(!*rx.borrow());
    drop(guard);
    assert!(*rx.borrow());
}

#[test]
fn cancel_on_drop_leaves_receiver_false_while_alive() {
    let (guard, rx) = cancel_pair();
    assert!(!*rx.borrow());
    // Guard still in scope -- receiver must not have flipped yet.
    assert!(!*rx.borrow());
    drop(guard);
}

// -------------------------------------------------------------------
// FieldExpectation / ExpectedResponse / check_sse_events (issue #828)
// -------------------------------------------------------------------

#[test]
fn field_expectation_eq_matches_and_reports_mismatch() {
    let body = serde_json::json!({"object": "chat.completion", "usage": {"total_tokens": 9}});
    assert!(
        FieldExpectation::Eq {
            json_pointer: "/object",
            scalar: Scalar::Str("chat.completion"),
        }
        .check(&body)
        .is_ok()
    );
    assert!(
        FieldExpectation::Eq {
            json_pointer: "/usage/total_tokens",
            scalar: Scalar::U64(9),
        }
        .check(&body)
        .is_ok()
    );
    let err = FieldExpectation::Eq {
        json_pointer: "/object",
        scalar: Scalar::Str("chat.completion.chunk"),
    }
    .check(&body)
    .unwrap_err();
    assert!(err.contains("/object"), "error must name the field: {err}");
}

#[test]
fn field_expectation_eq_missing_field_is_an_error() {
    let body = serde_json::json!({});
    let err = FieldExpectation::Eq {
        json_pointer: "/model",
        scalar: Scalar::Str("x"),
    }
    .check(&body)
    .unwrap_err();
    assert!(err.contains("absent"));
}

#[test]
fn field_expectation_absent_passes_when_missing_fails_when_present() {
    let body = serde_json::json!({"logprobs": null});
    assert!(
        FieldExpectation::Absent {
            json_pointer: "/choices"
        }
        .check(&body)
        .is_ok()
    );
    // `Value::pointer` finds `null` -- present-but-null is still
    // "present" for this check (matches `#[serde(skip_serializing_if =
    // "Option::is_none")]`'s OMITTED contract, not a `null` literal).
    let err = FieldExpectation::Absent {
        json_pointer: "/logprobs",
    }
    .check(&body)
    .unwrap_err();
    assert!(err.contains("expected absent"));
}

#[test]
fn field_expectation_array_len_checks_exact_length() {
    let body = serde_json::json!({"choices": [{"index": 0}]});
    assert!(
        FieldExpectation::ArrayLen {
            json_pointer: "/choices",
            len: 1,
        }
        .check(&body)
        .is_ok()
    );
    assert!(
        FieldExpectation::ArrayLen {
            json_pointer: "/choices",
            len: 2,
        }
        .check(&body)
        .is_err()
    );
}

fn sse_body(lines: &[&str]) -> String {
    lines.iter().map(|l| format!("data: {l}\n\n")).collect()
}

#[test]
fn check_sse_events_accepts_well_formed_baseline_stream() {
    let body = sse_body(&[
        r#"{"id":"chatcmpl-1","created":1,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
        r#"{"choices":[{"index":0,"delta":{"content":"hel"},"finish_reason":null}]}"#,
        r#"{"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}"#,
        r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ]);
    check_sse_events(&body, BASELINE_SSE_EVENTS).expect("well-formed stream must pass");
}

#[test]
fn check_sse_events_requires_at_least_one_content_delta() {
    let body = sse_body(&[
        r#"{"id":"chatcmpl-1","created":1,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
        r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ]);
    let err = check_sse_events(&body, BASELINE_SSE_EVENTS).unwrap_err();
    assert!(err.contains("ContentDelta"), "error: {err}");
}

#[test]
fn check_sse_events_rejects_wrong_finish_reason() {
    let body = sse_body(&[
        r#"{"id":"chatcmpl-1","created":1,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
        r#"{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#,
        r#"{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#,
        "[DONE]",
    ]);
    assert!(check_sse_events(&body, BASELINE_SSE_EVENTS).is_err());
}

#[test]
fn check_sse_events_rejects_missing_done_sentinel() {
    let body = sse_body(&[
        r#"{"id":"chatcmpl-1","created":1,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
        r#"{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#,
        r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
    ]);
    let err = check_sse_events(&body, BASELINE_SSE_EVENTS).unwrap_err();
    assert!(
        err.contains("Done") || err.contains("stream ended"),
        "error: {err}"
    );
}

#[test]
fn check_sse_events_rejects_role_opener_missing_id_or_created() {
    let missing_id = sse_body(&[
        r#"{"created":1,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
    ]);
    let err = check_sse_events(&missing_id, &[EventExpectation::RoleOpener]).unwrap_err();
    assert!(err.contains("/id"), "error: {err}");

    let missing_created = sse_body(&[
        r#"{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
    ]);
    let err = check_sse_events(&missing_created, &[EventExpectation::RoleOpener]).unwrap_err();
    assert!(err.contains("/created"), "error: {err}");
}

#[test]
fn generate_config_snapshot_captures_every_field() {
    let cfg = GenerateConfig {
        max_new_tokens: 42,
        temperature: 1.3,
        top_k: 7,
        top_p: 0.55,
        min_p: 0.15,
        repetition_penalty: 1.05,
        seed: Some(9),
        stop_token_ids: vec![100],
        enable_thinking: false,
        enable_mtp: Some(true),
        grammar: None,
        stop_strings: vec!["STOP".to_string()],
        reasoning_budget: Some(3),
        logprobs: Some(2),
    };
    let snapshot = GenerateConfigSnapshot::from(&cfg);
    assert_eq!(snapshot.max_new_tokens, 42);
    assert_eq!(snapshot.temperature, 1.3);
    assert_eq!(snapshot.top_k, 7);
    assert_eq!(snapshot.top_p, 0.55);
    assert_eq!(snapshot.min_p, 0.15);
    assert_eq!(snapshot.repetition_penalty, 1.05);
    assert_eq!(snapshot.seed, Some(9));
    assert_eq!(snapshot.stop_token_ids, vec![100]);
    assert!(!snapshot.enable_thinking);
    assert_eq!(snapshot.enable_mtp, Some(true));
    assert!(!snapshot.has_grammar);
    assert_eq!(snapshot.stop_strings, vec!["STOP".to_string()]);
    assert_eq!(snapshot.reasoning_budget, Some(3));
    assert_eq!(snapshot.logprobs, Some(2));
}

#[test]
fn chat_completions_parity_cases_expected_status_matches_variant() {
    // Every case's declared status must agree with its own variant --
    // a cheap sanity check that catches a copy-paste status/variant
    // mismatch in the const table itself, independent of any HTTP call.
    for case in CHAT_COMPLETIONS_PARITY_CASES {
        for binary in [Binary::Lattice, Binary::LatticeServe] {
            let expected = case.expected(binary);
            match expected {
                ExpectedResponse::Error { status, .. } => assert!(
                    !(200..300).contains(&status),
                    "case '{}': Error variant must not carry a 2xx status",
                    case.name
                ),
                ExpectedResponse::Json { status, .. } | ExpectedResponse::Sse { status, .. } => {
                    assert_eq!(
                        expected.status(),
                        status,
                        "case '{}': status() must match the variant's own status",
                        case.name
                    );
                }
            }
        }
    }
}
