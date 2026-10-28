use super::*;
use tower::ServiceExt as _;

fn fixture_state(fail: bool, interrupt: bool) -> AppState {
    tiny_state_with_test_generator(64, move |_prompt, _cfg, on_token, _cancel| {
        let _ = on_token("é");
        if fail {
            return Err(lattice_inference::error::InferenceError::Inference(
                "fixture failure".into(),
            ));
        }
        Ok(GenerateOutput {
            text: "é".into(),
            token_ids: vec![1, 2],
            prompt_tokens: 3,
            generated_tokens: 2,
            stopped: !interrupt,
            stop_reason: Some(if interrupt {
                lattice_inference::StopReason::Interrupt
            } else {
                lattice_inference::StopReason::Eos
            }),
            token_logprobs: vec![],
        })
    })
}

async fn request(body: serde_json::Value, fail: bool, interrupt: bool) -> (StatusCode, String) {
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();
    let response = router(fixture_state(fail, interrupt))
        .oneshot(req)
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn body() -> serde_json::Value {
    serde_json::json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true})
}

#[tokio::test]
async fn stream_usage_frames_use_authoritative_counts_and_preserve_default() {
    for option in [None, Some(false), Some(true)] {
        let mut input = body();
        if let Some(enabled) = option {
            input["stream_options"] = serde_json::json!({"include_usage": enabled});
        }
        let (status, text) = request(input, false, false).await;
        assert_eq!(status, StatusCode::OK);
        let frames = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect::<Vec<_>>();
        let expected_len = if option == Some(true) { 5 } else { 4 };
        assert_eq!(frames.len(), expected_len, "{text}");
        let chunks = frames[..frames.len() - 1]
            .iter()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "é");
        assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
        assert_eq!(frames.last(), Some(&"[DONE]"));
        if option == Some(true) {
            assert!(
                chunks[..3]
                    .iter()
                    .all(|chunk| chunk.get("usage") == Some(&serde_json::Value::Null))
            );
            assert_eq!(chunks[3]["choices"], serde_json::json!([]));
            assert_eq!(
                chunks[3]["usage"],
                serde_json::json!({"prompt_tokens":3,"completion_tokens":2,"total_tokens":5})
            );
            assert_eq!(chunks[3]["id"], chunks[0]["id"]);
            assert_eq!(chunks[3]["model"], chunks[0]["model"]);
        } else {
            assert!(chunks.iter().all(|chunk| chunk.get("usage").is_none()));
        }
    }
}

#[tokio::test]
async fn failed_or_interrupted_stream_does_not_claim_final_usage() {
    for (fail, interrupt) in [(true, false), (false, true)] {
        let mut input = body();
        input["stream_options"] = serde_json::json!({"include_usage":true});
        let (status, text) = request(input, fail, interrupt).await;
        assert_eq!(status, StatusCode::OK);
        let frames = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect::<Vec<_>>();
        assert_eq!(frames.last(), Some(&"[DONE]"));
        let chunks = frames[..frames.len() - 1]
            .iter()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert!(
            chunks
                .iter()
                .all(|chunk| !chunk["choices"].as_array().is_some_and(Vec::is_empty))
        );
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.get("usage").is_none_or(|usage| usage.is_null()))
        );
        if fail {
            assert!(chunks.iter().any(|chunk| chunk.get("error").is_some()));
        }
    }
}

#[tokio::test]
async fn stream_options_reject_nonstream_and_unknown_fields() {
    let mut input = body();
    input["stream"] = serde_json::json!(false);
    input["stream_options"] = serde_json::json!({"include_usage":true});
    let (status, text) = request(input, false, false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&text).unwrap()["error"]["code"],
        "unsupported_feature"
    );
    let mut input = body();
    input["stream_options"] = serde_json::json!({"include_obfuscation":false});
    let (status, _) = request(input, false, false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
