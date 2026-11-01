use super::*;
use tower::ServiceExt as _;

#[tokio::test]
async fn tool_history_is_rejected_before_submission_and_empty_is_accepted() {
    for streaming in [false, true] {
        for (field, rejected) in [
            (None, None),
            (Some(serde_json::Value::Null), None),
            (Some(serde_json::json!([])), None),
            (Some(serde_json::json!([{"id":"call"}])), Some("tool_calls")),
            (Some(serde_json::json!({})), Some("tool_calls")),
            (Some(serde_json::json!("bad")), Some("tool_calls")),
        ] {
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let observed = Arc::clone(&calls);
            let state =
                tiny_state_with_test_generator(64, move |_prompt, _cfg, on_token, _cancel| {
                    observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let _ = on_token("ok");
                    Ok(GenerateOutput {
                        text: "ok".into(),
                        token_ids: vec![1],
                        prompt_tokens: 1,
                        generated_tokens: 1,
                        stopped: true,
                        stop_reason: Some(lattice_inference::StopReason::Eos),
                        token_logprobs: vec![],
                    })
                });
            let mut message = serde_json::json!({"role":"assistant","content":"hello"});
            if let Some(field) = field {
                message["tool_calls"] = field;
            }
            let body = serde_json::json!({"model":"test-model","messages":[message,{"role":"user","content":"hi"}],"max_tokens":8,"stream":streaming});
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap();
            let response = router(state).oneshot(request).await.unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            if let Some(field) = rejected {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(error["error"]["code"], "unsupported_feature");
                assert_eq!(field, "tool_calls");
                assert_eq!(
                    error["error"]["message"],
                    "messages[].tool_calls is not supported by this server"
                );
                assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
            } else {
                assert_eq!(status, StatusCode::OK);
                assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
            }
        }
        for message in [
            serde_json::json!({"role":"tool","content":"x"}),
            serde_json::json!({"role":"tool","content":null}),
            serde_json::json!({"role":"assistant","content":null,"tool_calls":[{}]}),
        ] {
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let captured = Arc::clone(&calls);
            let state = tiny_state_with_test_generator(64, move |_, _, _, _| {
                captured.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                panic!("unexpected submission")
            });
            let field = if message["role"] == "tool" {
                "role"
            } else {
                "tool_calls"
            };
            let body = serde_json::json!({"model":"test-model","messages":[message,{"role":"user","content":"hi"}],"stream":streaming});
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap();
            let response = router(state).oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(error["error"]["code"], "unsupported_feature");
            let expected = if field == "role" {
                "role 'tool' is not supported by this server"
            } else {
                "messages[].tool_calls is not supported by this server"
            };
            assert_eq!(error["error"]["message"], expected);
            assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        }
    }
}
