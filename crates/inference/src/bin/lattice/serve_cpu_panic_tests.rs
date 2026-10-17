use super::tests::tiny_state_with_test_generator;
use super::*;
use axum::http::StatusCode;
use http_body_util::BodyExt;
use std::sync::Mutex;
use std::time::Duration;

mod cpu_panic_responses {
    use super::*;
    use std::sync::mpsc;
    use tokio::sync::oneshot;
    use tower::ServiceExt as _;

    const PANIC_PAYLOAD: &str = "private CPU generator panic payload";
    const RESPONSE_TIMEOUT: Duration = Duration::from_secs(10);
    const CLEANUP_TIMEOUT: Duration = Duration::from_secs(30);

    fn request(stream: bool) -> axum::http::Request<axum::body::Body> {
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                serde_json::json!({
                    "model": "test-model",
                    "messages": [{ "role": "user", "content": "hi" }],
                    "max_tokens": 64,
                    "stream": stream,
                })
                .to_string(),
            ))
            .expect("fixture request must build")
    }

    fn output(text: &str) -> GenerateOutput {
        GenerateOutput {
            text: text.to_owned(),
            token_ids: vec![0],
            prompt_tokens: 1,
            generated_tokens: 1,
            stopped: true,
            stop_reason: Some(lattice_inference::StopReason::Eos),
            token_logprobs: vec![],
        }
    }

    async fn send(app: axum::Router, stream: bool) -> Response {
        tokio::time::timeout(RESPONSE_TIMEOUT, app.oneshot(request(stream)))
            .await
            .expect("the router must respond within the test bound")
            .expect("the router must return an HTTP response")
    }

    async fn response_text(response: Response) -> String {
        let bytes = tokio::time::timeout(RESPONSE_TIMEOUT, response.into_body().collect())
            .await
            .expect("generation must not leave the response body open")
            .expect("the response body must remain readable")
            .to_bytes();
        String::from_utf8(bytes.to_vec()).expect("response body must be valid UTF-8")
    }

    fn generic_error() -> serde_json::Value {
        serde_json::json!({
            "error": {
                "message": "inference failed",
                "type": "server_error",
                "code": "internal_error",
                "param": null,
            }
        })
    }

    fn events(text: &str, expected: usize) -> Vec<serde_json::Value> {
        assert!(
            !text.contains(PANIC_PAYLOAD),
            "panic details must stay private"
        );
        let payloads: Vec<_> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect();
        assert_eq!(
            payloads.len(),
            expected,
            "unexpected SSE population: {text}"
        );
        assert_eq!(payloads.last(), Some(&"[DONE]"));
        assert_eq!(
            payloads
                .iter()
                .filter(|payload| **payload == "[DONE]")
                .count(),
            1
        );
        let events: Vec<serde_json::Value> = payloads[..payloads.len() - 1]
            .iter()
            .map(|payload| serde_json::from_str(payload).expect("SSE event must be JSON"))
            .collect();
        assert_eq!(
            events[0]["choices"][0]["delta"],
            serde_json::json!({ "role": "assistant" })
        );
        events
    }

    async fn panic_response(stream: bool, partial: Option<&'static str>) -> Response {
        let state = tiny_state_with_test_generator(64, move |_prompt, _cfg, on_token, _cancel| {
            if let Some(partial) = partial {
                assert!(on_token(partial), "the client must still accept deltas");
            }
            panic!("{PANIC_PAYLOAD}");
        });
        send(router(state), stream).await
    }

    async fn assert_streaming_panic_response(partial: Option<&'static str>) {
        let response = panic_response(true, partial).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let text = response_text(response).await;
        let events = events(&text, 3 + usize::from(partial.is_some()));
        if let Some(partial) = partial {
            assert_eq!(
                events[1]["choices"][0]["delta"],
                serde_json::json!({ "content": partial })
            );
        }
        assert_eq!(events.last(), Some(&generic_error()));
        assert_eq!(
            events
                .iter()
                .filter(|event| event.get("error").is_some())
                .count(),
            1
        );
        for event in &events {
            if let Some(choices) = event["choices"].as_array() {
                for choice in choices {
                    assert!(
                        choice["finish_reason"].is_null(),
                        "a CPU panic must not emit a successful finish: {text}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn streaming_panic_after_partial_emits_one_terminal_error() {
        assert_streaming_panic_response(Some("partial")).await;
    }

    #[tokio::test]
    async fn streaming_panic_before_delta_emits_one_terminal_error() {
        assert_streaming_panic_response(None).await;
    }

    #[tokio::test]
    async fn non_streaming_panic_returns_generic_500() {
        let response = panic_response(false, None).await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.headers()["content-type"], "application/json");
        let text = response_text(response).await;
        assert!(
            !text.contains(PANIC_PAYLOAD),
            "panic details must stay private"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&text).unwrap(),
            generic_error()
        );
    }

    #[tokio::test]
    async fn streaming_completion_preserves_delta_finish_and_done() {
        let state = tiny_state_with_test_generator(64, |_prompt, _cfg, on_token, _cancel| {
            assert!(on_token("complete"));
            Ok(output("complete"))
        });
        let response = send(router(state), true).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let text = response_text(response).await;
        let events = events(&text, 4);
        assert!(events.iter().all(|event| event.get("error").is_none()));
        assert_eq!(
            events[1]["choices"][0]["delta"],
            serde_json::json!({"content": "complete"})
        );
        assert!(events[0]["choices"][0]["finish_reason"].is_null());
        assert!(events[1]["choices"][0]["finish_reason"].is_null());
        assert_eq!(events[2]["choices"][0]["delta"], serde_json::json!({}));
        assert_eq!(events[2]["choices"][0]["finish_reason"], "stop");
        for event in &events {
            assert_eq!(event["object"], "chat.completion.chunk");
            assert_eq!(event["model"], "test-model");
            assert_eq!(event["id"], events[0]["id"]);
            assert_eq!(event["choices"].as_array().unwrap().len(), 1);
        }
    }

    struct BlockingCleanup {
        started: Option<oneshot::Sender<()>>,
        release: mpsc::Receiver<()>,
        outcome: Option<oneshot::Sender<bool>>,
    }

    impl Drop for BlockingCleanup {
        fn drop(&mut self) {
            if let Some(started) = self.started.take() {
                let _ = started.send(());
            }
            let released = self.release.recv_timeout(CLEANUP_TIMEOUT).is_ok();
            if let Some(outcome) = self.outcome.take() {
                let _ = outcome.send(released);
            }
        }
    }

    #[tokio::test]
    async fn streaming_disconnect_holds_admission_until_panic_cleanup_finishes() {
        let (panic_tx, panic_rx) = mpsc::channel();
        let panic_rx = Mutex::new(panic_rx);
        let (cleanup_started_tx, cleanup_started_rx) = oneshot::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (outcome_tx, outcome_rx) = oneshot::channel();
        let cleanup = Mutex::new(Some(BlockingCleanup {
            started: Some(cleanup_started_tx),
            release: release_rx,
            outcome: Some(outcome_tx),
        }));
        let mut state =
            tiny_state_with_test_generator(64, move |_prompt, _cfg, on_token, _cancel| {
                let first_cleanup = cleanup.lock().unwrap().take();
                if let Some(_cleanup) = first_cleanup {
                    assert!(on_token("partial"));
                    let checkpoint = panic_rx.lock().unwrap().recv_timeout(CLEANUP_TIMEOUT);
                    if checkpoint.is_err() {
                        return Err(lattice_inference::error::InferenceError::Inference(
                            "test panic checkpoint was not released".to_owned(),
                        ));
                    }
                    panic!("{PANIC_PAYLOAD}");
                }
                assert!(on_token("ok"));
                Ok(output("ok"))
            });
        let admission = Arc::new(Semaphore::new(1));
        state.cpu_generation_admission = Arc::clone(&admission);
        let app = router(state);
        let response = send(app.clone(), true).await;
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body();
        let mut initial = Vec::new();
        for _ in 0..2 {
            let frame = tokio::time::timeout(RESPONSE_TIMEOUT, body.frame())
                .await
                .expect("role and partial delta must arrive before disconnect")
                .expect("the body must contain the expected frame")
                .expect("the frame must be readable");
            initial.extend_from_slice(frame.data_ref().expect("the SSE frame must contain data"));
        }
        let initial = String::from_utf8(initial).unwrap();
        assert!(initial.contains("\"role\":\"assistant\""));
        assert!(initial.contains("\"content\":\"partial\""));
        drop(body);
        panic_tx
            .send(())
            .expect("the first generator must be waiting");
        tokio::time::timeout(RESPONSE_TIMEOUT, cleanup_started_rx)
            .await
            .expect("panic unwinding must reach cleanup")
            .expect("the cleanup checkpoint must be reported");

        let available_during_cleanup = admission.available_permits();
        let rejected = send(app.clone(), false).await;
        let rejected_status = rejected.status();
        let rejected_type = rejected.headers()["content-type"]
            .to_str()
            .unwrap()
            .to_owned();
        let rejected_body = response_text(rejected).await;
        release_tx
            .send(())
            .expect("cleanup must still be waiting for release");
        let released = tokio::time::timeout(RESPONSE_TIMEOUT, outcome_rx)
            .await
            .expect("cleanup must complete within the test bound")
            .expect("cleanup must report its release outcome");
        let permit = tokio::time::timeout(RESPONSE_TIMEOUT, Arc::clone(&admission).acquire_owned())
            .await
            .expect("the slot must reopen after actual blocking cleanup")
            .expect("the admission semaphore must stay open");
        drop(permit);
        let recovered = send(app, false).await;
        let recovered_status = recovered.status();
        let recovered_body = response_text(recovered).await;

        assert!(
            released,
            "cleanup must be explicitly released, not time out"
        );
        assert_eq!(available_during_cleanup, 0);
        assert_eq!(rejected_status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(rejected_type, "application/json");
        let rejected: serde_json::Value = serde_json::from_str(&rejected_body).unwrap();
        assert_eq!(rejected["error"]["code"], "server_busy");
        assert_eq!(rejected["error"]["type"], "server_error");
        assert!(rejected["error"]["param"].is_null());
        assert!(!rejected_body.contains(PANIC_PAYLOAD));
        assert_eq!(recovered_status, StatusCode::OK);
        let recovered: serde_json::Value = serde_json::from_str(&recovered_body).unwrap();
        assert_eq!(recovered["choices"][0]["message"]["content"], "ok");
    }
}
