use super::*;
use axum::body::to_bytes;
use axum::http::header::CONTENT_TYPE;

fn app_state_with_jobs(jobs: MetalWorkerClient, vocab_bytes: Arc<Vec<Vec<u8>>>) -> AppState {
    AppState {
        jobs,
        model_id: Arc::from("test-model"),
        defaults: GenerationDefaults::standard(100),
        model_max_context: 4096,
        max_pending: 1_000_000,
        metrics: Arc::new(ServeMetrics::default()),
        vocab_bytes,
        grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
        embedding: None,
    }
}

fn completion(text: &str, stopped: bool) -> GenerateOutput {
    GenerateOutput {
        text: text.to_owned(),
        token_ids: vec![1, 2],
        prompt_tokens: 3,
        generated_tokens: 2,
        stopped,
        stop_reason: Some(if stopped {
            lattice_inference::StopReason::Eos
        } else {
            lattice_inference::StopReason::Length
        }),
        token_logprobs: vec![],
    }
}

async fn response_json(response: Response) -> (StatusCode, serde_json::Value) {
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body must be readable");
    (
        status,
        serde_json::from_slice(&bytes).expect("response body must be JSON"),
    )
}

async fn response_text(response: Response) -> (StatusCode, String) {
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body must be readable");
    (
        status,
        String::from_utf8(bytes.to_vec()).expect("response must be UTF-8"),
    )
}

#[tokio::test]
async fn unsupported_content_type_precedes_invalid_body_with_full_error_envelope() {
    let response = chat_completions(
        State(test_app_state()),
        HeaderMap::new(),
        Body::from("not json"),
    )
    .await;
    let (status, value) = response_json(response).await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert_eq!(
        value,
        json!({
            "error": {
                "message": "Content-Type must be application/json",
                "type": "invalid_request_error",
                "code": "unsupported_media_type",
                "param": null
            }
        })
    );
}

#[tokio::test]
async fn malformed_body_has_full_error_envelope_after_json_content_type() {
    let response = chat_completions(
        State(test_app_state()),
        test_json_headers(),
        Body::from("not json"),
    )
    .await;
    let (status, value) = response_json(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        value,
        json!({
            "error": {
                "message": "invalid JSON request body",
                "type": "invalid_request_error",
                "code": "invalid_request_body",
                "param": null
            }
        })
    );
}

#[tokio::test]
async fn metal_logprobs_refusal_has_full_error_envelope() {
    let response = chat_completions(
        State(test_app_state()),
        test_json_headers(),
        Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"logprobs":true}"#),
    )
    .await;
    let (status, value) = response_json(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        value,
        json!({
            "error": {
                "message": "logprobs/top_logprobs are not supported by this server",
                "type": "invalid_request_error",
                "code": "unsupported_feature",
                "param": null
            }
        })
    );
}

#[test]
fn tokenizer_vocab_bytes_preserve_multibyte_raw_and_unresolved_ids() {
    let tokenizer = lattice_inference::tokenizer::bpe::BpeTokenizer::from_vocab_and_merges(
        std::collections::HashMap::from([
            ("å¥½".to_owned(), 0),
            ("å".to_owned(), 1),
            ("¥".to_owned(), 2),
        ]),
        vec![],
    )
    .expect("synthetic byte-level BPE vocabulary must load");
    let vocab_bytes = tokenizer.vocab_bytes(4).expect("full table must decode");
    assert_eq!(vocab_bytes[0], [0xe5, 0xa5, 0xbd]);
    assert_eq!(vocab_bytes[1], [0xe5]);
    assert!(std::str::from_utf8(&vocab_bytes[1]).is_err());
    assert_eq!(vocab_bytes[2], [0xa5]);
    assert!(vocab_bytes[3].is_empty());
    assert_eq!(tokenizer.token_for_id(0), Some("å¥½"));
    assert_eq!(tokenizer.token_for_id(3), None);
}

#[test]
fn worker_panic_before_readiness_returns_thread_exited() {
    let result = MetalWorker::spawn(
        || -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String> {
            panic!("injected worker startup panic");
        },
        lattice_inference::serve::metal_worker::DEFAULT_MAX_PENDING_JOBS,
        lattice_inference::serve::lora::ResidencyLimits::default(),
    );
    let Err(error) = result else {
        panic!("worker panic must close the readiness channel");
    };
    assert!(matches!(error, StartupError::ThreadExited));
}

#[tokio::test]
async fn admission_refusal_is_json_without_stream_frames() {
    let (jobs, mut jobs_rx) =
        lattice_inference::serve::metal_worker::test_client_and_jobs_with_cap(1);
    let state = app_state_with_jobs(jobs, Arc::new(vec![]));
    let first_state = state.clone();
    let first = tokio::spawn(async move {
        chat_completions(
            State(first_state),
            test_json_headers(),
            Body::from(r#"{"messages":[{"role":"user","content":"first"}]}"#),
        )
        .await
    });
    let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await else {
        panic!("the first request must occupy the worker admission slot");
    };

    let mut headers = test_json_headers();
    headers.insert(
        CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );
    let response = chat_completions(
        State(state),
        headers,
        Body::from(r#"{"messages":[{"role":"user","content":"second"}],"stream":true}"#),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let (status, body) = response_text(response).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(!body.contains("data:"), "503 response was not SSE: {body}");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&body).expect("503 body must be JSON"),
        json!({
            "error": {
                "message": "too many outstanding requests; the inference worker's pending-job queue is full, retry shortly",
                "type": "server_error",
                "code": "server_busy",
                "param": null
            }
        })
    );

    let _ = job.reply(WorkerEvent::Complete(completion("unused", true)));
    // The worker closes a job's event channel after its terminal event.
    drop(job);
    let first_response = first.await.expect("first request task must finish");
    assert_eq!(first_response.status(), StatusCode::OK);
}

#[tokio::test]
async fn streaming_wire_order_has_role_delta_finish_and_done_without_usage() {
    let (jobs, mut jobs_rx) = test_client_and_jobs();
    let state = app_state_with_jobs(jobs, Arc::new(vec![]));
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
            let _ = job.reply(WorkerEvent::Delta("é".to_owned()));
            let _ = job.reply(WorkerEvent::Complete(completion("not displayed", true)));
        }
    });

    let response = chat_completions(
        State(state),
        test_json_headers(),
        Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let (_, body) = response_text(response).await;
    let data = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect::<Vec<_>>();
    assert_eq!(
        data.len(),
        4,
        "expected role, delta, finish, and done: {body}"
    );
    let role: serde_json::Value = serde_json::from_str(data[0]).expect("role event JSON");
    let delta: serde_json::Value = serde_json::from_str(data[1]).expect("delta event JSON");
    let finish: serde_json::Value = serde_json::from_str(data[2]).expect("finish event JSON");
    assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(delta["choices"][0]["delta"]["content"], "é");
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    assert!(data[..3].iter().all(|event| !event.contains("usage")));
    assert_eq!(data[3], "[DONE]");
    assert!(!body.contains("not displayed"));
}

#[tokio::test]
async fn chat_responses_wait_for_worker_events_before_committing() {
    for streaming in [false, true] {
        let (jobs, mut jobs_rx) = test_client_and_jobs();
        let state = app_state_with_jobs(jobs, Arc::new(vec![]));
        let body = Body::from(
            json!({
                "messages": [{"role": "user", "content": "hi"}],
                "stream": streaming
            })
            .to_string(),
        );
        let response_task =
            tokio::spawn(
                async move { chat_completions(State(state), test_json_headers(), body).await },
            );
        let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await else {
            panic!("request must submit a worker job");
        };
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        assert!(
            !response_task.is_finished(),
            "the handler must await the first worker event before responding"
        );
        let _ = job.reply(WorkerEvent::Delta("first".to_owned()));
        if streaming {
            for _ in 0..1000 {
                if response_task.is_finished() {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert!(
                response_task.is_finished(),
                "streaming status commits after the first accepted worker event"
            );
        } else {
            for _ in 0..50 {
                tokio::task::yield_now().await;
            }
            assert!(
                !response_task.is_finished(),
                "non-streaming waits for worker completion after its first delta"
            );
        }
        let _ = job.reply(WorkerEvent::Complete(completion("first", true)));
        // The worker closes a job's event channel after its terminal event.
        drop(job);
        let response = response_task.await.expect("request task must finish");
        assert_eq!(response.status(), StatusCode::OK);
        let _ = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body must be readable");
    }
}

#[tokio::test]
async fn non_streaming_content_uses_delta_text_instead_of_complete_text() {
    let (jobs, mut jobs_rx) = test_client_and_jobs();
    let state = app_state_with_jobs(jobs, Arc::new(vec![]));
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
            let _ = job.reply(WorkerEvent::Delta("delta sentinel".to_owned()));
            let _ = job.reply(WorkerEvent::Complete(completion("complete sentinel", true)));
        }
    });
    let response = chat_completions(
        State(state),
        test_json_headers(),
        Body::from(r#"{"messages":[{"role":"user","content":"hi"}]}"#),
    )
    .await;
    let (status, value) = response_json(response).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(value["choices"][0]["message"]["content"], "delta sentinel");
    assert!(value.to_string().contains("delta sentinel"));
    assert!(!value.to_string().contains("complete sentinel"));
}

#[tokio::test]
async fn first_worker_event_failures_and_cancellation_keep_their_http_shapes() {
    for (event, expected_status, expected_body) in [
        (
            WorkerEvent::Failed("private failure".to_owned()),
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"error":{"message":"inference failed","type":"server_error","code":"internal_error","param":null}}),
        ),
        (
            WorkerEvent::ConstraintBlocked("private block".to_owned()),
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"error":{"message":"inference failed","type":"server_error","code":"internal_error","param":null}}),
        ),
        (
            WorkerEvent::Rejected(ApiError::BadRequest {
                message: "prompt rejected".to_owned(),
                code: "context_length_exceeded",
            }),
            StatusCode::BAD_REQUEST,
            json!({"error":{"message":"prompt rejected","type":"invalid_request_error","code":"context_length_exceeded","param":null}}),
        ),
    ] {
        let (jobs, mut jobs_rx) = test_client_and_jobs();
        let state = app_state_with_jobs(jobs, Arc::new(vec![]));
        tokio::spawn(async move {
            if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
                let _ = job.reply(event);
            }
        });
        let response = chat_completions(
            State(state),
            test_json_headers(),
            Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#),
        )
        .await;
        let (status, body) = response_json(response).await;
        assert_eq!(status, expected_status);
        assert_eq!(body, expected_body);
    }

    let (jobs, mut jobs_rx) = test_client_and_jobs();
    let state = app_state_with_jobs(jobs, Arc::new(vec![]));
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
            let _ = job.reply(WorkerEvent::Cancelled);
        }
    });
    let response = chat_completions(
        State(state),
        test_json_headers(),
        Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let (_, body) = response_text(response).await;
    let data = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect::<Vec<_>>();
    assert_eq!(
        data.len(),
        3,
        "cancelled first event becomes empty completion: {body}"
    );
    let role: serde_json::Value = serde_json::from_str(data[0]).expect("role event JSON");
    let finish: serde_json::Value = serde_json::from_str(data[1]).expect("finish event JSON");
    assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert_eq!(data[2], "[DONE]");
}

#[tokio::test]
async fn grammar_failure_envelopes_keep_specific_codes_and_messages() {
    let cases = [
        (
            WorkerEvent::ConstraintBlocked("grammar exhausted".to_owned()),
            None,
            json!({"error":{"message":"structured output generation was blocked: no legal token continues the schema from this state","type":"server_error","code":"blocked_constraint","param":null}}),
        ),
        (
            WorkerEvent::Complete(completion("", false)),
            None,
            json!({"error":{"message":"structured output did not complete within max_tokens (stop_reason=Some(Length)); refusing to return partial JSON","type":"server_error","code":"length_limit","param":null}}),
        ),
        (
            WorkerEvent::Complete(completion("", true)),
            Some(r#"{"unexpected":1}"#),
            json!({"error":{"message":"generated text failed independent JSON Schema validation","type":"server_error","code":"validation_failed","param":null}}),
        ),
    ];

    for (terminal, delta, expected_body) in cases {
        let (jobs, mut jobs_rx) = test_client_and_jobs();
        let state = app_state_with_jobs(jobs, route_test_vocab());
        tokio::spawn(async move {
            if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
                if let Some(delta) = delta {
                    let _ = job.reply(WorkerEvent::Delta(delta.to_owned()));
                }
                let _ = job.reply(terminal);
            }
        });
        let response = chat_completions(
            State(state),
            test_json_headers(),
            Body::from(structured_body(V0_ROUTE_SCHEMA, Some("true"), None)),
        )
        .await;
        let (status, body) = response_json(response).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body, expected_body);
    }
}

#[tokio::test]
async fn repeated_structured_requests_reuse_the_loaded_vocabulary_cache_entry() {
    let vocab = route_test_vocab();
    let (jobs, mut jobs_rx) = test_client_and_jobs();
    let state = app_state_with_jobs(jobs, Arc::clone(&vocab));
    let cache = Arc::clone(&state.grammar_cache);
    tokio::spawn(async move {
        for _ in 0..2 {
            if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
                let _ = job.reply(WorkerEvent::Delta(r#"{"ok":true}"#.to_owned()));
                let _ = job.reply(WorkerEvent::Complete(completion(r#"{"ok":true}"#, true)));
            }
        }
    });

    for _ in 0..2 {
        let response = chat_completions(
            State(state.clone()),
            test_json_headers(),
            Body::from(structured_body(V0_ROUTE_SCHEMA, Some("true"), None)),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let _ = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body must be readable");
    }
    assert_eq!(cache.compile_count(), 1);
    assert!(Arc::strong_count(&vocab) > 1);
}

#[tokio::test]
async fn a_midstream_failure_keeps_the_committed_sse_status_and_error_event_order() {
    let (jobs, mut jobs_rx) = test_client_and_jobs();
    let state = app_state_with_jobs(jobs, Arc::new(vec![]));
    tokio::spawn(async move {
        if let Some(WorkerMessage::Generate(job)) = jobs_rx.recv().await {
            let _ = job.reply(WorkerEvent::Delta("partial".to_owned()));
            let _ = job.reply(WorkerEvent::Failed("private cause".to_owned()));
        }
    });
    let response = chat_completions(
        State(state),
        test_json_headers(),
        Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let (_, body) = response_text(response).await;
    let data = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect::<Vec<_>>();
    assert_eq!(
        data.len(),
        4,
        "expected role, delta, failure, and done: {body}"
    );
    let role: serde_json::Value = serde_json::from_str(data[0]).expect("role event JSON");
    let delta: serde_json::Value = serde_json::from_str(data[1]).expect("delta event JSON");
    let failure: serde_json::Value = serde_json::from_str(data[2]).expect("failure event JSON");
    assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(delta["choices"][0]["delta"]["content"], "partial");
    assert_eq!(
        failure,
        json!({"error":{"message":"inference failed","type":"server_error","code":"internal_error","param":null}})
    );
    assert_eq!(data[3], "[DONE]");
}
