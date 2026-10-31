use super::*;
use tower::ServiceExt as _;

#[tokio::test]
async fn lattice_profile_ignores_min_p_in_streaming_and_non_streaming() {
    for streaming in [false, true] {
        for value in [
            None,
            Some(serde_json::Value::Null),
            Some(serde_json::json!(0.0)),
            Some(serde_json::json!(0.5)),
            Some(serde_json::json!(1.0)),
            Some(serde_json::json!(-1)),
            Some(serde_json::json!({"bad":true})),
        ] {
            let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
            let captured = Arc::clone(&observed);
            let state =
                tiny_state_with_test_generator(64, move |_prompt, cfg, on_token, _cancel| {
                    captured.lock().unwrap().push(cfg.min_p);
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
            let mut body = serde_json::json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":streaming});
            if let Some(value) = value {
                body["min_p"] = value;
            }
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap();
            let response = router(state).oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(*observed.lock().unwrap(), vec![0.0]);
        }
    }
}
