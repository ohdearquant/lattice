use super::*;
use tower::ServiceExt as _;

#[tokio::test]
async fn total_completion_cap_reaches_worker_and_controls_response_in_both_modes() {
    for streaming in [false, true] {
        for (fields, reasoning, returned, expected_cap, expected_error) in [
            (
                serde_json::json!({"max_completion_tokens":5}),
                Some(3),
                5,
                Some(5),
                false,
            ),
            (
                serde_json::json!({"max_completion_tokens":5,"max_tokens":5}),
                Some(3),
                5,
                Some(5),
                false,
            ),
            (
                serde_json::json!({"max_completion_tokens":5}),
                Some(3),
                6,
                Some(5),
                true,
            ),
            (serde_json::json!({"max_tokens":5}), Some(3), 6, None, false),
            (
                serde_json::json!({"max_completion_tokens":5,"chat_template_kwargs":{"enable_thinking":false}}),
                Some(3),
                5,
                Some(5),
                false,
            ),
        ] {
            let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
            let observed = Arc::clone(&seen);
            let state = tiny_state_with_test_generator(64, move |_, cfg, on_token, _| {
                observed.lock().unwrap().push((
                    cfg.total_token_cap,
                    cfg.reasoning_budget,
                    cfg.enable_thinking,
                ));
                let _ = on_token("x");
                Ok(GenerateOutput {
                    text: "x".into(),
                    token_ids: vec![1; returned],
                    prompt_tokens: 1,
                    generated_tokens: returned,
                    stopped: false,
                    stop_reason: Some(lattice_inference::StopReason::Length),
                    token_logprobs: vec![],
                })
            });
            let mut body = serde_json::json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"reasoning_budget":reasoning,"stream":streaming});
            body.as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            if streaming {
                body["stream_options"] = serde_json::json!({"include_usage":true});
            }
            let req = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap();
            let response = router(state).oneshot(req).await.unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let payload = String::from_utf8(bytes.to_vec()).unwrap();
            let observed = seen.lock().unwrap();
            assert_eq!(observed.len(), 1);
            assert_eq!(observed[0].0, expected_cap);
            if body.get("chat_template_kwargs").is_some() {
                assert_eq!((observed[0].1, observed[0].2), (None, false));
            }
            if expected_error {
                if streaming {
                    assert!(payload.contains("internal_error"), "{payload}");
                } else {
                    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
                }
            } else {
                assert_eq!(status, StatusCode::OK, "{payload}");
                if streaming {
                    assert!(
                        payload.contains(&format!("\"completion_tokens\":{returned}")),
                        "{payload}"
                    );
                } else {
                    let json: serde_json::Value = serde_json::from_str(&payload).unwrap();
                    assert_eq!(json["usage"]["completion_tokens"], returned);
                }
            }
        }
    }
}

fn parity_result(payload: &str) -> serde_json::Value {
    let mut last_delta = None;
    let mut finish_reason = None;
    let mut completion_tokens = None;
    let mut error = false;
    for line in payload.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            continue;
        }
        let event: serde_json::Value = serde_json::from_str(data).unwrap();
        error |= event.get("error").is_some();
        if let Some(choices) = event["choices"].as_array() {
            for choice in choices {
                if let Some(content) = choice["delta"]["content"].as_str() {
                    last_delta = Some(content.to_string());
                }
                if let Some(reason) = choice["finish_reason"].as_str() {
                    finish_reason = Some(reason.to_string());
                }
            }
        }
        if let Some(tokens) = event["usage"]["completion_tokens"].as_u64() {
            completion_tokens = Some(tokens);
        }
    }
    serde_json::json!({"last_delta":last_delta, "finish_reason":finish_reason, "completion_tokens":completion_tokens,"error":error})
}

#[tokio::test]
async fn cross_binary_total_cap_streaming_fixture() {
    for (case, returned, stopped) in [
        ("below", 1, false),
        ("at", 2, false),
        ("stop_at", 2, true),
        ("over_after_delta", 3, false),
    ] {
        let state = tiny_state_with_test_generator(64, move |_, cfg, on_token, _| {
            assert_eq!(cfg.total_token_cap, Some(2));
            let _ = on_token("A");
            if returned >= 2 {
                let _ = on_token("B");
            }
            if returned == 3 {
                let _ = on_token("C");
            }
            Ok(GenerateOutput {
                text: match returned {
                    1 => "A",
                    2 => "AB",
                    _ => "ABC",
                }
                .into(),
                token_ids: vec![1; returned],
                prompt_tokens: 1,
                generated_tokens: returned,
                stopped,
                stop_reason: Some(if stopped {
                    lattice_inference::StopReason::Eos
                } else {
                    lattice_inference::StopReason::Length
                }),
                token_logprobs: vec![],
            })
        });

        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(serde_json::json!({
                "model":"test-model", "messages":[{"role":"user","content":"hi"}],
                "max_completion_tokens":2, "stream":true, "stream_options":{"include_usage":true}
            }).to_string()))
            .unwrap();
        let response = router(state).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload = String::from_utf8(bytes.to_vec()).unwrap();
        let result = parity_result(&payload);
        let expected = if returned == 3 {
            serde_json::json!({"last_delta":"C","finish_reason":null,"completion_tokens":null,"error":true})
        } else {
            serde_json::json!({"last_delta":if returned == 1 { "A" } else { "B" },
                "finish_reason":if stopped { "stop" } else { "length" },
                "completion_tokens":returned,"error":false})
        };
        assert_eq!(result, expected, "{case}: {payload}");
    }
}
