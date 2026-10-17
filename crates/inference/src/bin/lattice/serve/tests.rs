fn input_message(role: String, content: MessageContent) -> Message {
    let mut message: Message = serde_json::from_str(r#"{"role":"","content":""}"#).unwrap();
    message.role = role;
    message.content = content;
    message
}

fn input_image_url(
    url: String,
    detail: Option<String>,
) -> lattice_inference::serve::contract::ImageUrl {
    let mut image: lattice_inference::serve::contract::ImageUrl =
        serde_json::from_str(r#"{"url":""}"#).unwrap();
    image.url = url;
    image.detail = detail;
    image
}

use super::*;
use axum::http::StatusCode;
use lattice_inference::forward::metal_qwen35::ChatRole;

/// #832 checklist: "Add a Metal-feature test asserting an explicit
/// common-worker marker, so a silently-reintroduced per-binary
/// fallback cannot pass green." `MetalHandle::client`'s field type
/// is private, so this test can only be written from inside
/// `mod serve` (same module, so private-field access is allowed) —
/// it constructs a `MetalHandle` directly from a
/// `lattice_inference::serve::metal_worker::MetalWorkerClient`,
/// which only type-checks if `MetalHandle` still wraps that exact
/// shared type. A private per-binary fallback worker (any type
/// other than the shared `MetalWorkerClient`) would fail to compile
/// here, not just fail some behavioral assertion at runtime.
/// `test-utils`-gated: `test_client_and_jobs` (the only way to
/// obtain a real `MetalWorkerClient` without a live GPU worker
/// thread) lives behind that feature — see its own doc comment.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[test]
fn metal_handle_is_backed_by_the_shared_metal_worker_client() {
    fn build_from_shared_client(
        client: lattice_inference::serve::metal_worker::MetalWorkerClient,
    ) -> MetalHandle {
        MetalHandle { client }
    }
    let (client, _jobs_rx) = lattice_inference::serve::metal_worker::test_client_and_jobs();
    let _handle: MetalHandle = build_from_shared_client(client);
}

/// #939 regression coverage: the unified `lattice serve` HTTP
/// adapter's admission ordering. Before this fix, the Metal
/// streaming arm called `MetalWorkerClient::submit` (the ONE way
/// admission (#932) can reject a request) INSIDE a detached
/// `tokio::spawn` task, after `chat_completions` had already
/// returned `Sse::new(...).into_response()` -- so an admission
/// rejection surfaced as a normal-looking stream that immediately
/// ended, never as HTTP 503. The non-streaming arm separately
/// collapsed `ApiError::ServiceUnavailable` into a generic 500. Both
/// arms are driven here through the REAL `router()` + a real
/// worker thread (`spawn_fake_with_cap`, cap=1, `generate` blocked
/// on a channel so exactly one request is provably "in flight"),
/// mirroring `lattice_serve.rs`'s own
/// `real_router_admission_cap::chat_completions_returns_503_json_envelope_when_admission_cap_reached`
/// -- that test alone did not catch this bug because it only
/// covers the daemon's separate, already-correct adapter.
#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
mod admission_cap_939;

#[test]
fn validate_max_tokens_rejects_zero() {
    let err = validate_max_tokens(Some(0), None, 256, 4096).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_max_tokens",
            ..
        }
    ));
}

#[test]
fn validate_max_tokens_rejects_above_cap() {
    let err = validate_max_tokens(Some(9999), None, 256, 4096).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "max_tokens_exceeds_limit",
            ..
        }
    ));
}

#[test]
fn validate_max_tokens_uses_default_when_absent() {
    assert_eq!(validate_max_tokens(None, None, 128, 4096).unwrap(), 128);
}

#[test]
fn validate_max_tokens_alias_agrees() {
    assert_eq!(
        validate_max_tokens(Some(512), Some(512), 256, 4096).unwrap(),
        512
    );
}

#[test]
fn validate_max_tokens_alias_conflict_rejected() {
    let err = validate_max_tokens(Some(100), Some(200), 256, 4096).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_request",
            ..
        }
    ));
}

#[test]
fn validate_temperature_rejects_negative() {
    let err = validate_temperature(Some(-0.1)).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_temperature",
            ..
        }
    ));
}

#[test]
fn validate_temperature_rejects_above_two() {
    let err = validate_temperature(Some(2.1)).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_temperature",
            ..
        }
    ));
}

#[test]
fn validate_temperature_accepts_boundary() {
    assert_eq!(validate_temperature(Some(0.0)).unwrap(), 0.0);
    assert_eq!(validate_temperature(Some(2.0)).unwrap(), 2.0);
}

#[test]
fn validate_top_p_rejects_zero() {
    let err = validate_top_p(Some(0.0)).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_top_p",
            ..
        }
    ));
}

#[test]
fn validate_top_p_rejects_above_one() {
    let err = validate_top_p(Some(1.1)).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_top_p",
            ..
        }
    ));
}

#[test]
fn validate_top_p_accepts_one() {
    assert_eq!(validate_top_p(Some(1.0)).unwrap(), 1.0);
}

#[test]
fn chat_template_multi_message_chatml() {
    let messages = vec![
        input_message(
            "system".to_string(),
            MessageContent::Text("Be helpful.".to_string()),
        ),
        input_message(
            "user".to_string(),
            MessageContent::Text("Hello".to_string()),
        ),
    ];
    let prompt = format_chat_template(&to_chat_messages(&messages).unwrap());
    assert!(prompt.contains("<|im_start|>system\nBe helpful.<|im_end|>"));
    assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
}

// Role/content-part rejection cases are covered by the
// `to_chat_messages_rejects_*` tests below -- `to_chat_messages` is
// the sole validation entry point the ChatML renderer sits behind
// (#668), so there is no second renderer left to test separately.

// Exercises finish_reason_for via the real helper function used by the handler.
// A cap-reached output has stopped=false → "length".
// A stop-condition output has stopped=true → "stop".
#[test]
fn finish_reason_length_only_at_cap() {
    use lattice_inference::GenerateOutput;
    let cap = GenerateOutput {
        text: String::new(),
        token_ids: vec![],
        prompt_tokens: 10,
        generated_tokens: 64,
        stopped: false,
        stop_reason: Some(lattice_inference::StopReason::Length),
        token_logprobs: vec![],
    };
    assert_eq!(super::finish_reason_for(&cap), "length");

    let natural = GenerateOutput {
        text: "hello".into(),
        token_ids: vec![1, 2, 3],
        prompt_tokens: 10,
        generated_tokens: 3,
        stopped: true,
        stop_reason: Some(lattice_inference::StopReason::Eos),
        token_logprobs: vec![],
    };
    assert_eq!(super::finish_reason_for(&natural), "stop");
}

// M1 regression: a stop-string hit at exactly max_new_tokens must yield "stop",
// not "length". The old token-count formula (generated == cap → "length") would
// mislabel this case because the stop-completing token is included in generated_ids
// before the stop is detected.
//
// This test calls the real finish_reason_for helper. It is RED when
// finish_reason_for reverts to the old `generated_tokens == max_tokens` formula.
#[test]
fn finish_reason_stop_string_at_cap_is_stop_not_length() {
    use lattice_inference::GenerateOutput;
    let max_tokens: usize = 4;
    // stop-string hit at exactly the token budget:
    // stopped=true because a stop string matched; generated_tokens==max_tokens
    // because the matching token is included in generated_ids before truncation.
    let output = GenerateOutput {
        text: "hi".into(),
        token_ids: vec![1, 2, 3, 4],
        prompt_tokens: 5,
        generated_tokens: max_tokens,
        stopped: true,
        stop_reason: Some(lattice_inference::StopReason::Eos),
        token_logprobs: vec![],
    };
    assert_eq!(
        super::finish_reason_for(&output),
        "stop",
        "stop-string hit at cap must yield finish_reason=stop, not length"
    );
}

// Natural length cap (no stop condition) must still yield "length".
#[test]
fn finish_reason_natural_length_cap_is_length() {
    use lattice_inference::GenerateOutput;
    let output = GenerateOutput {
        text: "hi".into(),
        token_ids: vec![1, 2, 3, 4],
        prompt_tokens: 5,
        generated_tokens: 4,
        stopped: false,
        stop_reason: Some(lattice_inference::StopReason::Length),
        token_logprobs: vec![],
    };
    assert_eq!(super::finish_reason_for(&output), "length");
}

#[test]
fn reject_unsupported_stream_true_ok() {
    // stream=true is now handled by the streaming path and must NOT be
    // rejected by reject_unsupported.
    let mut req = serde_json::from_str::<ChatCompletionRequest>("{}").unwrap();
    req.model = Some("m".to_string());
    req.stream = Some(true);
    assert!(reject_unsupported(&req).is_ok());
}

#[test]
fn reject_unsupported_stream_and_logprobs_rejected() {
    // #585: logprobs is implemented on the non-streaming path only;
    // combined with stream: true it must be rejected, not silently
    // ignored.
    let mut req = bare_req();
    req.stream = Some(true);
    req.logprobs = Some(true);
    let err = reject_unsupported(&req).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "unsupported_feature",
            ..
        }
    ));
}

// -----------------------------------------------------------------------
// ChatCompletionChunk serialization
// -----------------------------------------------------------------------

#[test]
fn chunk_content_delta_serializes_correctly() {
    let chunk = ChatCompletionChunk {
        id: "chatcmpl-1-0".to_string(),
        object: "chat.completion.chunk",
        created: 1_000_000,
        model: "test-model".to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: Some("Hello".to_string()),
            },
            finish_reason: None,
        }],
    };
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(
        json.contains("\"object\":\"chat.completion.chunk\""),
        "must contain object field"
    );
    assert!(
        json.contains("\"delta\":{\"content\":\"Hello\"}"),
        "delta must contain only content when role is None"
    );
    // finish_reason must be absent (not null) when None
    assert!(
        !json.contains("finish_reason"),
        "finish_reason must be omitted when None"
    );
}

#[test]
fn chunk_finish_delta_serializes_correctly() {
    let chunk = ChatCompletionChunk {
        id: "chatcmpl-1-0".to_string(),
        object: "chat.completion.chunk",
        created: 1_000_000,
        model: "test-model".to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: None,
            },
            finish_reason: Some("stop"),
        }],
    };
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(
        json.contains("\"finish_reason\":\"stop\""),
        "finish chunk must include finish_reason"
    );
    // delta should be empty object since both role and content are None
    assert!(
        json.contains("\"delta\":{}"),
        "finish chunk delta must be empty object"
    );
}

#[test]
fn reject_unsupported_n_gt_1() {
    let mut req = serde_json::from_str::<ChatCompletionRequest>("{}").unwrap();
    req.model = Some("m".to_string());
    req.n = Some(3);
    let err = reject_unsupported(&req).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "unsupported_feature",
            ..
        }
    ));
}

#[test]
fn reject_unsupported_response_format_json() {
    let mut req = serde_json::from_str::<ChatCompletionRequest>("{}").unwrap();
    req.model = Some("m".to_string());
    req.response_format = Some(
        serde_json::from_value::<ResponseFormat>(
            serde_json::json!({"type": "json_object".to_string()}),
        )
        .unwrap(),
    );
    let err = reject_unsupported(&req).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "unsupported_feature",
            ..
        }
    ));
}

// -----------------------------------------------------------------------
// reject_unsupported — remaining fields
// -----------------------------------------------------------------------

#[cfg(feature = "test-utils")]
#[tokio::test]
async fn lora_list_on_cpu_refuses_by_name() {
    let error = lora_list(State(tiny_state(64))).await.unwrap_err();
    assert_eq!(error.code(), "lora_unsupported_backend");
    assert!(error.message().contains("Metal"));
}

#[cfg(feature = "test-utils")]
#[tokio::test]
async fn lora_selection_on_cpu_refuses_by_name() {
    for stream in [false, true] {
        let request = serde_json::from_value(serde_json::json!({"messages":[{"role":"user","content":"hi"}],"lora":[{"id":7,"scale":1.0}],"stream":stream})).unwrap();
        let error = chat_completions_with_request(State(tiny_state(64)), request)
            .await
            .unwrap_err();
        assert_eq!(error.code(), "lora_unsupported_backend");
        assert!(error.message().contains("Metal"));
    }
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn lora_duplicate_id_is_400_before_unified_admission() {
    use lattice_inference::serve::lora::AdapterMetadata;
    use lattice_inference::serve::metal_worker::test_client_and_jobs_with_adapters;
    for stream in [false, true] {
        let (client, mut jobs) = test_client_and_jobs_with_adapters(vec![AdapterMetadata {
            id: 7,
            name: "test".into(),
            path: "test.safetensors".into(),
            rank: 1,
            layers: 1,
        }]);
        let mut state = tiny_state(64);
        let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
            .tokenizer()
            .clone();
        state.model = ModelBackend::Metal {
            handle: MetalHandle { client },
            tokenizer: Arc::new(tokenizer),
            max_context: 4096,
        };
        let request = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "lora": [{"id": 7, "scale": 1.0}, {"id": 7, "scale": -0.25}],
            "stream": stream,
        }))
        .unwrap();
        let error = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            chat_completions_with_request(State(state.clone()), request),
        )
        .await
        .expect("duplicate selection must return before a worker reply")
        .unwrap_err();
        assert!(matches!(error, ApiError::BadRequest { .. }));
        assert_eq!(error.code(), "lora_duplicate_adapter_id");
        assert_eq!(error.message(), "duplicate LoRA adapter id 7");
        assert!(matches!(
            jobs.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn the_router_key_is_added_beside_residency_and_never_wraps_it() {
    // The regression this pins. Adding the router as
    // `json!({"adapters": index, "router": ...})` reads like an additive
    // change and silently rewrites two existing fields: top-level
    // `adapters` stops being an array and `applied` moves a level down.
    // Asserting only that `router` is present would pass against that bug,
    // so the pre-existing keys are asserted in their ORIGINAL positions.
    use lattice_inference::router_state::{ResolvedRouter, RouterArtifact};
    use lattice_inference::serve::lora::{AdapterIndex, AdapterMetadata, LoraSelection};

    let index = AdapterIndex {
        adapters: vec![AdapterMetadata {
            id: 0,
            name: "technical".into(),
            path: "/p/a.safetensors".into(),
            rank: 8,
            layers: 24,
        }],
        applied: vec![
            serde_json::from_str::<LoraSelection>(r#"{"id":0,"scale":1.0}"#).expect("selection"),
        ],
        ..Default::default()
    };

    let body = lattice_inference::serve::lora::lora_list_body(&index, None);
    assert!(
        body["adapters"].is_array(),
        "top-level adapters must stay an array, got {}",
        body["adapters"]
    );
    assert_eq!(body["adapters"][0]["name"], "technical");
    assert!(
        body["applied"].is_array(),
        "applied must stay at the top level, got {body}"
    );
    assert_eq!(body["applied"][0]["id"], 0);
    assert_eq!(body["router"]["enabled"], false);

    // A gate that is serving reports its version AND whether a pin put it
    // there; the version alone cannot distinguish the two.
    let resolved = ResolvedRouter {
        artifact: RouterArtifact {
            version: 7,
            adapter_names: vec!["technical".into()],
            representation: lattice_inference::router_state::TrainedRepresentation {
                embedding_model: "gme-qwen35".into(),
                pooling: "mean_visual".into(),
                prompt_source: "last_user_message".into(),
                loader_format: lattice_inference::serve::embeddings::QWEN35_F16_DECODER_LOADER
                    .into(),
                input_width: 4,
            },
            gate_bytes: vec![1, 2, 3],
            provenance: lattice_inference::router_state::RouterProvenance::Legacy,
        },
        pinned: true,
    };
    let applied = || lattice_inference::router_state::AppliedPolicy {
        kind: "uniform".to_string(),
        tau: None,
        epsilon: 0.0,
        exploration_scale: None,
        learning_enabled: false,
    };
    let body = lattice_inference::serve::lora::lora_list_body(
        &index,
        Some(resolved.report("gme-qwen35", applied())),
    );
    assert!(
        body["adapters"].is_array(),
        "residency shape must not depend on the router"
    );
    assert!(body["applied"].is_array());
    assert_eq!(body["router"]["enabled"], true);
    assert_eq!(body["router"]["pinned"], true);
    assert_eq!(body["router"]["adapter_names"][0], "technical");
    assert_eq!(
        body["router"]["embedder"], "gme-qwen35",
        "the endpoint reports what THIS server embeds with, so an operator              can compare it against the checkpoint they think they passed"
    );
    assert!(
        body["router"]["version"]
            .as_str()
            .is_some_and(|v| v.starts_with("7:")),
        "version label carries the counter, got {}",
        body["router"]["version"]
    );

    assert_eq!(
        body["router"]["routable"], true,
        "this fixture's trained name IS resident, so routing can run"
    );
    assert_eq!(
        body["router"]["missing"],
        serde_json::json!([]),
        "nothing is missing when the trained set is resident"
    );
    assert_eq!(
        body["router"]["blend_refusal"],
        serde_json::Value::Null,
        "no blend refusal is published for this fixture, so the router block reports none"
    );

    // Issue #1735: a resident set matching the trained names exactly can
    // still refuse to route because the full set cannot be BLENDED, not
    // because any name comparison fails. `routable` and the reason must
    // both reflect it, additively -- `adapters`/`applied` untouched.
    let mut over_budget = index.clone();
    over_budget.blend_refusal = Some("summed rank exceeds MAX_BLEND_RANK_TOTAL".to_string());
    let body = lattice_inference::serve::lora::lora_list_body(
        &over_budget,
        Some(resolved.report("gme-qwen35", applied())),
    );
    assert!(
        body["adapters"].is_array() && body["applied"].is_array(),
        "a blend refusal must not change the residency snapshot's shape"
    );
    assert_eq!(
        body["router"]["routable"], false,
        "the names match exactly, but the blend refuses"
    );
    assert_eq!(body["router"]["missing"], serde_json::json!([]));
    assert_eq!(body["router"]["unexpected"], serde_json::json!([]));
    assert_eq!(
        body["router"]["blend_refusal"],
        "summed rank exceeds MAX_BLEND_RANK_TOTAL"
    );

    // The other direction, against a residency that genuinely lacks the
    // trained name. Both arms are here because a routable-only assertion
    // passes against a `routable` hard-coded true, and a missing-only
    // assertion passes against one hard-coded false.
    let empty = AdapterIndex::default();
    let body = lattice_inference::serve::lora::lora_list_body(
        &empty,
        Some(resolved.report("gme-qwen35", applied())),
    );
    assert_eq!(
        body["router"]["routable"], false,
        "a trained name that is not resident cannot be routed to"
    );
    assert_eq!(
        body["router"]["missing"][0], "technical",
        "an operator must be able to see the 400s coming, and by name"
    );

    let unpinned = ResolvedRouter {
        pinned: false,
        ..resolved
    };
    assert_eq!(
        lattice_inference::serve::lora::lora_list_body(
            &index,
            Some(unpinned.report("gme-qwen35", applied()))
        )["router"]["pinned"],
        false,
        "an unpinned gate must not report itself pinned"
    );
}

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
#[tokio::test]
async fn lora_selection_reaches_unified_worker_and_unknown_id_is_400() {
    use lattice_inference::serve::lora::{AdapterMetadata, LoraSelection};
    use lattice_inference::serve::metal_worker::{
        WorkerEvent, WorkerMessage, test_client_and_jobs_with_adapters,
    };
    for (stream, reject) in [(false, false), (true, false), (false, true), (true, true)] {
        let (client, mut jobs) = test_client_and_jobs_with_adapters(vec![AdapterMetadata {
            id: 7,
            name: "test".into(),
            path: "test.safetensors".into(),
            rank: 1,
            layers: 1,
        }]);
        let mut state = tiny_state(64);
        let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
            .tokenizer()
            .clone();
        state.model = ModelBackend::Metal {
            handle: MetalHandle { client },
            tokenizer: Arc::new(tokenizer),
            max_context: 4096,
        };
        let bad = serde_json::from_value(serde_json::json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"lora":[{"id":99,"scale":1.0}],"stream":stream})).unwrap();
        let error = chat_completions_with_request(State(state.clone()), bad)
            .await
            .unwrap_err();
        assert!(matches!(error, ApiError::BadRequest { .. }));
        assert!(error.message().contains("99"));
        let worker = tokio::spawn(async move {
            let Some(WorkerMessage::Generate(job)) = jobs.recv().await else {
                panic!("missing generation")
            };
            assert_eq!(
                job.lora_selection(),
                &[serde_json::from_str::<LoraSelection>(r#"{"id":7,"scale":0.25}"#).unwrap()]
            );
            if reject {
                job.reply(WorkerEvent::Rejected(
                    lattice_inference::serve::lora::unknown_adapter(7),
                ));
                return;
            }
            job.reply(WorkerEvent::Complete(GenerateOutput {
                text: "ok".into(),
                token_ids: vec![0],
                prompt_tokens: 1,
                generated_tokens: 1,
                stopped: true,
                stop_reason: None,
                token_logprobs: vec![],
            }));
        });
        let request = serde_json::from_value(serde_json::json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"lora":[{"id":7,"scale":0.25}],"stream":stream})).unwrap();
        let result = chat_completions_with_request(State(state), request).await;
        if reject {
            let error = result.unwrap_err();
            assert!(matches!(error, ApiError::BadRequest { .. }));
            assert!(error.message().contains("7"));
            worker.await.unwrap();
            continue;
        }
        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        worker.await.unwrap();
    }
}

fn bare_req() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"model":"m"}"#).unwrap()
}

#[test]
fn reject_unsupported_tools_rejected() {
    let mut req = bare_req();
    req.tools = Some(serde_json::json!([]));
    let err = reject_unsupported(&req).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "unsupported_feature",
            ..
        }
    ));
}

#[test]
fn reject_unsupported_tool_choice_rejected() {
    let mut req = bare_req();
    req.tool_choice = Some(serde_json::json!("auto"));
    let err = reject_unsupported(&req).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "unsupported_feature",
            ..
        }
    ));
}

#[test]
fn reject_unsupported_logprobs_true_ok() {
    // #585: logprobs is now implemented on the non-streaming path, so a
    // standalone `logprobs: true` (no `stream: true`) must be accepted
    // here — validation of the value itself is `validate_logprobs`'s job.
    let mut req = bare_req();
    req.logprobs = Some(true);
    assert!(reject_unsupported(&req).is_ok());
}

#[test]
fn reject_unsupported_stop_now_accepted() {
    // stop is no longer rejected by reject_unsupported; it is parsed separately.
    let mut req = bare_req();
    req.stop = Some(serde_json::json!("</s>"));
    assert!(reject_unsupported(&req).is_ok());
}

// -----------------------------------------------------------------------
// parse_stop_strings
// -----------------------------------------------------------------------

#[test]
fn parse_stop_strings_null_gives_empty() {
    assert_eq!(parse_stop_strings(&None).unwrap(), Vec::<String>::new());
    assert_eq!(
        parse_stop_strings(&Some(serde_json::Value::Null)).unwrap(),
        Vec::<String>::new()
    );
}

#[test]
fn parse_stop_strings_single_string_gives_vec_of_one() {
    let v = parse_stop_strings(&Some(serde_json::json!("</s>"))).unwrap();
    assert_eq!(v, vec!["</s>".to_string()]);
}

#[test]
fn parse_stop_strings_array_of_two_accepted() {
    let v = parse_stop_strings(&Some(serde_json::json!(["</s>", "\nUser:"]))).unwrap();
    assert_eq!(v, vec!["</s>".to_string(), "\nUser:".to_string()]);
}

#[test]
fn parse_stop_strings_empty_array_rejected() {
    let err = parse_stop_strings(&Some(serde_json::json!([]))).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_stop",
            ..
        }
    ));
}

#[test]
fn parse_stop_strings_array_over_four_rejected() {
    let err = parse_stop_strings(&Some(serde_json::json!(["a", "b", "c", "d", "e"]))).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_stop",
            ..
        }
    ));
}

#[test]
fn parse_stop_strings_array_with_number_rejected() {
    let err = parse_stop_strings(&Some(serde_json::json!(["ok", 42]))).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_stop",
            ..
        }
    ));
}

#[test]
fn parse_stop_strings_empty_string_element_rejected() {
    let err = parse_stop_strings(&Some(serde_json::json!(["ok", ""]))).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_stop",
            ..
        }
    ));
}

#[test]
fn parse_stop_strings_empty_string_scalar_rejected() {
    let err = parse_stop_strings(&Some(serde_json::json!(""))).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_stop",
            ..
        }
    ));
}

#[test]
fn parse_stop_strings_array_exactly_four_accepted() {
    let v = parse_stop_strings(&Some(serde_json::json!(["a", "b", "c", "d"]))).unwrap();
    assert_eq!(v.len(), 4);
}

#[test]
fn reject_unsupported_stream_false_ok() {
    // stream=false must not trigger a rejection.
    let mut req = bare_req();
    req.stream = Some(false);
    assert!(reject_unsupported(&req).is_ok());
}

#[test]
fn reject_unsupported_n_1_ok() {
    let mut req = bare_req();
    req.n = Some(1);
    assert!(reject_unsupported(&req).is_ok());
}

#[test]
fn reject_unsupported_response_format_text_ok() {
    let mut req = bare_req();
    req.response_format = Some(
        serde_json::from_value::<ResponseFormat>(serde_json::json!({"type": "text".to_string()}))
            .unwrap(),
    );
    assert!(reject_unsupported(&req).is_ok());
}

#[test]
fn reject_unsupported_logprobs_false_ok() {
    let mut req = bare_req();
    req.logprobs = Some(false);
    assert!(reject_unsupported(&req).is_ok());
}

// -----------------------------------------------------------------------
// validate_max_tokens — additional edge cases
// -----------------------------------------------------------------------

#[test]
fn validate_max_tokens_at_exactly_cap_ok() {
    assert_eq!(
        validate_max_tokens(Some(4096), None, 256, 4096).unwrap(),
        4096
    );
}

#[test]
fn validate_max_tokens_max_completion_only_ok() {
    assert_eq!(
        validate_max_tokens(None, Some(512), 256, 4096).unwrap(),
        512
    );
}

// -----------------------------------------------------------------------
// validate_temperature — default path
// -----------------------------------------------------------------------

#[test]
fn validate_temperature_none_uses_default() {
    assert_eq!(
        validate_temperature(None).unwrap(),
        GenerationDefaults::standard(1).temperature
    );
}

// -----------------------------------------------------------------------
// validate_top_p — default path
// -----------------------------------------------------------------------

#[test]
fn validate_top_p_none_uses_default() {
    assert_eq!(
        validate_top_p(None).unwrap(),
        GenerationDefaults::standard(1).top_p
    );
}

// -----------------------------------------------------------------------
// format_chat_template (via to_chat_messages) — additional rendering cases
// -----------------------------------------------------------------------

#[test]
fn chat_template_user_only() {
    let msgs = vec![input_message(
        "user".to_string(),
        MessageContent::Text("hi".to_string()),
    )];
    let prompt = format_chat_template(&to_chat_messages(&msgs).unwrap());
    assert_eq!(
        prompt,
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn chat_template_multi_turn_assistant() {
    let msgs = vec![
        input_message("user".to_string(), MessageContent::Text("q1".to_string())),
        input_message(
            "assistant".to_string(),
            MessageContent::Text("a1".to_string()),
        ),
        input_message("user".to_string(), MessageContent::Text("q2".to_string())),
    ];
    let prompt = format_chat_template(&to_chat_messages(&msgs).unwrap());
    assert!(prompt.contains("<|im_start|>user\nq1<|im_end|>"));
    assert!(prompt.contains("<|im_start|>assistant\na1<|im_end|>"));
    assert!(prompt.contains("<|im_start|>user\nq2<|im_end|>"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn chat_template_content_parts_text_ok() {
    let msgs = vec![input_message(
        "user".to_string(),
        MessageContent::Parts(vec![
            ContentPart::Text {
                text: "hello".to_string(),
            },
            ContentPart::Text {
                text: " world".to_string(),
            },
        ]),
    )];
    let prompt = format_chat_template(&to_chat_messages(&msgs).unwrap());
    assert!(prompt.contains("<|im_start|>user\nhello world<|im_end|>"));
}

// -----------------------------------------------------------------------
// `to_chat_messages` validation (#661, CPU-available since #668) — the
// sole validation entry point the ChatML renderer sits behind.
// -----------------------------------------------------------------------

#[test]
fn to_chat_messages_rejects_invalid_role() {
    let messages = vec![input_message(
        "function".to_string(),
        MessageContent::Text("data".to_string()),
    )];
    let err = to_chat_messages(&messages).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_role",
            ..
        }
    ));
}

#[test]
fn to_chat_messages_rejects_tool_role() {
    let messages = vec![input_message(
        "tool".to_string(),
        MessageContent::Text("result".to_string()),
    )];
    let err = to_chat_messages(&messages).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "unsupported_feature",
            ..
        }
    ));
}

#[test]
fn to_chat_messages_rejects_developer_role() {
    let messages = vec![input_message(
        "developer".to_string(),
        MessageContent::Text("system prompt".to_string()),
    )];
    let err = to_chat_messages(&messages).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "unsupported_feature",
            ..
        }
    ));
}

#[test]
fn to_chat_messages_rejects_non_text_content_part() {
    let messages = vec![input_message(
        "user".to_string(),
        MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: input_image_url("https://example.com/image.png".to_string(), None),
        }]),
    )];
    let err = to_chat_messages(&messages).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "vision_unsupported",
            ..
        }
    ));
}

#[test]
fn to_chat_messages_accepts_valid_roles() {
    let messages = vec![
        input_message(
            "system".to_string(),
            MessageContent::Text("Be helpful.".to_string()),
        ),
        input_message("user".to_string(), MessageContent::Text("q1".to_string())),
        input_message(
            "assistant".to_string(),
            MessageContent::Text("a1".to_string()),
        ),
    ];
    let chat_messages = to_chat_messages(&messages).unwrap();
    assert_eq!(chat_messages.len(), 3);
    assert_eq!(chat_messages[0].role, ChatRole::System);
    assert_eq!(chat_messages[0].content, "Be helpful.");
    assert_eq!(chat_messages[1].role, ChatRole::User);
    assert_eq!(chat_messages[2].role, ChatRole::Assistant);
}

// -----------------------------------------------------------------------
// Error envelope JSON shape
// -----------------------------------------------------------------------

/// Drains an `axum::response::Response` body into a parsed `Value`,
/// for asserting on the shared `lattice_inference::serve::ApiError`
/// envelope shape (ADR-080 C2, #782) from this binary's own tests.
async fn response_json(response: axum::response::Response) -> serde_json::Value {
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body reads");
    serde_json::from_slice(&body).expect("response body is valid JSON")
}

#[tokio::test]
async fn error_envelope_bad_request_shape() {
    let err = ApiError::BadRequest {
        message: "test error".to_string(),
        code: "invalid_request",
    };
    // Variant check kept separate so we know err itself was constructed correctly.
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_request",
            ..
        }
    ));
    // Verify the shared ApiError serialises to the OpenAI envelope shape:
    // {"error":{"message":"...","type":"invalid_request_error","code":"...","param":null}}
    let response = ApiError::BadRequest {
        message: "test error".to_string(),
        code: "invalid_request",
    }
    .into_response();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let v = response_json(response).await;
    assert!(v["error"].is_object(), "top-level key must be 'error'");
    assert_eq!(v["error"]["message"], "test error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["code"], "invalid_request");
    assert!(v["error"]["param"].is_null());
}

#[tokio::test]
async fn error_envelope_payload_too_large_shape() {
    let response = ApiError::PayloadTooLarge {
        message: "request body exceeds 1 MiB limit".to_string(),
    }
    .into_response();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let v = response_json(response).await;
    assert_eq!(v["error"]["code"], "request_body_too_large");
}

#[tokio::test]
async fn error_envelope_internal_shape() {
    let response = ApiError::Internal {
        message: "inference failed".to_string(),
    }
    .into_response();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let v = response_json(response).await;
    assert_eq!(v["error"]["type"], "server_error");
    assert_eq!(v["error"]["code"], "internal_error");
}

// -----------------------------------------------------------------------
// message content normalization (via the shared normalize_messages,
// exercised here through the `to_chat_messages` alias)
// -----------------------------------------------------------------------

#[test]
fn message_text_plain_string() {
    let messages = [input_message(
        "user".to_string(),
        MessageContent::Text("hello".to_string()),
    )];
    assert_eq!(to_chat_messages(&messages).unwrap()[0].content, "hello");
}

#[test]
fn message_text_parts_concatenates() {
    let messages = [input_message(
        "user".to_string(),
        MessageContent::Parts(vec![
            ContentPart::Text {
                text: "foo".to_string(),
            },
            ContentPart::Text {
                text: "bar".to_string(),
            },
        ]),
    )];
    assert_eq!(to_chat_messages(&messages).unwrap()[0].content, "foobar");
}

#[test]
fn message_text_parts_rejects_image() {
    let messages = [input_message(
        "user".to_string(),
        MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: input_image_url("https://example.com/image.png".to_string(), None),
        }]),
    )];
    let err = to_chat_messages(&messages).unwrap_err();
    match err {
        ApiError::BadRequest { message, code } => {
            assert_eq!(code, "vision_unsupported");
            assert_eq!(message, "image input requires a vision-capable model");
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn message_text_parts_rejects_unknown_part_type() {
    let messages = [input_message(
        "user".to_string(),
        MessageContent::Parts(vec![ContentPart::Unsupported {
            kind: "file".to_string(),
        }]),
    )];
    let err = to_chat_messages(&messages).unwrap_err();
    match err {
        ApiError::BadRequest { message, code } => {
            assert_eq!(code, "unsupported_feature");
            assert_eq!(
                message,
                "content part type 'file' is not supported; only 'text' and 'image_url' \
                         parts are accepted"
            );
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

// -----------------------------------------------------------------------
// validate_logprobs (#585)
// -----------------------------------------------------------------------

#[test]
fn validate_logprobs_absent_disables_capture() {
    assert_eq!(validate_logprobs(None, None).unwrap(), None);
}

#[test]
fn validate_logprobs_false_disables_capture() {
    assert_eq!(validate_logprobs(Some(false), None).unwrap(), None);
}

#[test]
fn validate_logprobs_true_no_top_logprobs_defaults_to_zero() {
    // logprobs: true with no top_logprobs still captures the sampled
    // token's own logprob, just with no alternatives.
    assert_eq!(validate_logprobs(Some(true), None).unwrap(), Some(0));
}

#[test]
fn validate_logprobs_true_with_top_logprobs_ok() {
    assert_eq!(validate_logprobs(Some(true), Some(5)).unwrap(), Some(5));
}

#[test]
fn validate_logprobs_top_logprobs_at_boundary_twenty_ok() {
    assert_eq!(validate_logprobs(Some(true), Some(20)).unwrap(), Some(20));
}

#[test]
fn validate_logprobs_top_logprobs_over_twenty_rejected() {
    let err = validate_logprobs(Some(true), Some(21)).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_top_logprobs",
            ..
        }
    ));
}

#[test]
fn validate_logprobs_top_logprobs_without_logprobs_true_rejected() {
    let err = validate_logprobs(None, Some(5)).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_request",
            ..
        }
    ));
}

#[test]
fn validate_logprobs_top_logprobs_with_logprobs_false_rejected() {
    let err = validate_logprobs(Some(false), Some(5)).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_request",
            ..
        }
    ));
}

// -----------------------------------------------------------------------
// render_token_logprob / build_choice_logprobs (#585)
// -----------------------------------------------------------------------

/// Tiny in-memory tokenizer for logprob-rendering tests — no merges,
/// just a fixed id -> token vocabulary large enough to exercise both a
/// known and an unresolved token id. "Hello"/"world" are plain ASCII in
/// the printable range the GPT-2 byte table maps to itself, so they
/// round-trip byte-for-byte through `byte_decode_token[_bytes]`.
fn logprob_test_tokenizer() -> lattice_inference::tokenizer::bpe::BpeTokenizer {
    let vocab: std::collections::HashMap<String, u32> =
        [("Hello".to_string(), 0u32), ("world".to_string(), 1u32)]
            .into_iter()
            .collect();
    lattice_inference::tokenizer::bpe::BpeTokenizer::from_vocab_and_merges(vocab, vec![])
        .expect("in-memory test vocab must construct")
}

#[test]
fn render_token_logprob_resolves_known_token() {
    let tokenizer = logprob_test_tokenizer();
    let (token, bytes) = render_token_logprob(&tokenizer, 0);
    assert_eq!(token, "Hello");
    assert_eq!(bytes, Some(b"Hello".to_vec()));
}

#[test]
fn render_token_logprob_unresolved_id_fails_closed() {
    // Token id 999 does not exist in the 2-entry test vocab: this must
    // fail closed with a visibly synthetic token and no bytes, never panic.
    let tokenizer = logprob_test_tokenizer();
    let (token, bytes) = render_token_logprob(&tokenizer, 999);
    assert_eq!(token, "<|unresolved_token_999|>");
    assert_eq!(bytes, None);
}

#[test]
fn build_choice_logprobs_shapes_content_and_alternatives() {
    let tokenizer = logprob_test_tokenizer();
    let token_logprobs = vec![
        TokenLogprob {
            token_id: 0,
            logprob: -0.1,
            top: vec![
                lattice_inference::TopLogprob {
                    token_id: 0,
                    logprob: -0.1,
                },
                lattice_inference::TopLogprob {
                    token_id: 1,
                    logprob: -2.3,
                },
            ],
        },
        TokenLogprob {
            token_id: 1,
            logprob: -0.05,
            top: vec![],
        },
    ];
    let choice_logprobs = build_choice_logprobs(&tokenizer, &token_logprobs);
    assert_eq!(choice_logprobs.content.len(), 2);

    assert_eq!(choice_logprobs.content[0].token, "Hello");
    assert_eq!(choice_logprobs.content[0].logprob, -0.1);
    assert_eq!(choice_logprobs.content[0].top_logprobs.len(), 2);
    assert_eq!(choice_logprobs.content[0].top_logprobs[0].token, "Hello");
    assert_eq!(choice_logprobs.content[0].top_logprobs[1].token, "world");

    assert_eq!(choice_logprobs.content[1].token, "world");
    assert_eq!(choice_logprobs.content[1].logprob, -0.05);
    assert!(choice_logprobs.content[1].top_logprobs.is_empty());
}

// -----------------------------------------------------------------------
// Choice.logprobs — JSON shape (#585)
// -----------------------------------------------------------------------

#[test]
fn choice_logprobs_omitted_from_json_when_none() {
    // The no-logprobs-requested response must be byte-identical to
    // before this feature existed: the key is absent, not `null`.
    let choice = Choice {
        index: 0,
        message: ResponseMessage {
            role: "assistant".to_string(),
            content: "hi".to_string(),
        },
        finish_reason: "stop".to_string(),
        logprobs: None,
    };
    let json = serde_json::to_string(&choice).unwrap();
    assert!(
        !json.contains("logprobs"),
        "logprobs key must be entirely absent when None, got: {json}"
    );
}

#[test]
fn choice_logprobs_present_when_requested() {
    let choice = Choice {
        index: 0,
        message: ResponseMessage {
            role: "assistant".to_string(),
            content: "hi".to_string(),
        },
        finish_reason: "stop".to_string(),
        logprobs: Some(ChoiceLogprobs {
            content: vec![TokenLogprobEntry {
                token: "hi".to_string(),
                logprob: -0.2,
                bytes: Some(b"hi".to_vec()),
                top_logprobs: vec![],
            }],
        }),
    };
    let json = serde_json::to_string(&choice).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["logprobs"]["content"][0]["token"], "hi");
    assert_eq!(v["logprobs"]["content"][0]["logprob"], -0.2);
    assert_eq!(
        v["logprobs"]["content"][0]["bytes"],
        serde_json::json!([104, 105])
    );
    assert_eq!(
        v["logprobs"]["content"][0]["top_logprobs"],
        serde_json::json!([])
    );
}

// -----------------------------------------------------------------------
// Capability-matrix fixtures (#654) — `validate_chat_request` cascade.
//
// Each `#[test]` fn name below is a fixture ID cited from
// `docs/capability-matrix.md`'s Fixture column; `scripts/check-capability-
// matrix.sh` greps this file for `fn <fixture_id>` and fails the build if
// a matrix row cites an ID that no longer exists here. These three checks
// (model-id match, empty messages, last-role-must-be-user) previously ran
// only inline in `chat_completions` with no dedicated test at all.
// -----------------------------------------------------------------------

fn user_msg(text: &str) -> Message {
    input_message("user".to_string(), MessageContent::Text(text.to_string()))
}

#[test]
fn cm_serve_model_mismatch_rejected() {
    let mut req = bare_req();
    req.model = Some("some-other-model".to_string());
    req.messages = vec![user_msg("hi")];
    let err = validate_chat_request(&req, "served-model", 256, 4096).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "model_not_found",
            ..
        }
    ));
}

#[test]
fn cm_serve_model_match_passes_model_check() {
    let mut req = bare_req();
    req.model = Some("served-model".to_string());
    req.messages = vec![user_msg("hi")];
    assert!(validate_chat_request(&req, "served-model", 256, 4096).is_ok());
}

#[test]
fn cm_serve_empty_messages_rejected() {
    let mut req = bare_req();
    req.model = Some("served-model".to_string());
    req.messages = vec![];
    let err = validate_chat_request(&req, "served-model", 256, 4096).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_messages",
            ..
        }
    ));
}

#[test]
fn cm_serve_last_message_not_user_rejected() {
    let mut req = bare_req();
    req.model = Some("served-model".to_string());
    req.messages = vec![
        user_msg("hi"),
        input_message(
            "assistant".to_string(),
            MessageContent::Text("hello".to_string()),
        ),
    ];
    let err = validate_chat_request(&req, "served-model", 256, 4096).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_messages",
            ..
        }
    ));
}

#[test]
fn cm_serve_unsupported_feature_rejected_before_model_check() {
    // `reject_unsupported` (tools/n/response_format/stream+logprobs) runs
    // first in the cascade: a request that both targets the wrong model
    // AND asks for `tools` must fail on the tools rejection, not the
    // model-mismatch check, so callers get the more specific error.
    let mut req = bare_req();
    req.model = Some("some-other-model".to_string());
    req.messages = vec![user_msg("hi")];
    req.tools = Some(serde_json::json!([{"type": "function"}]));
    let err = validate_chat_request(&req, "served-model", 256, 4096).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "unsupported_feature",
            ..
        }
    ));
}

#[test]
fn cm_serve_stop_sequences_accepted_end_to_end() {
    // Full-cascade check that a well-formed request carrying `stop`
    // resolves through to `PreparedChatRequest.stop_strings` — the
    // capability matrix's "supported" claim for `stop` on this surface.
    let mut req = bare_req();
    req.model = Some("served-model".to_string());
    req.messages = vec![user_msg("hi")];
    req.stop = Some(serde_json::json!(["\n\n"]));
    let prepared =
        prepare_chat_request(&req, "served-model", 256, 4096, false, |_| 1, || 4096).unwrap();
    assert_eq!(prepared.stop_strings, vec!["\n\n".to_string()]);
}

#[test]
fn cm_serve_context_window_checked_before_stop_parsing() {
    // Regression fixture for a refactor bug: extracting stop-sequence
    // parsing into the pre-model validation cascade moved it ahead of
    // the context-window check. The pre-refactor inline sequence
    // checked the context window BEFORE parsing `stop`. A request that is both
    // over-context and carries a malformed `stop` field must
    // therefore fail with `context_length_exceeded`, not a
    // stop-parsing error.
    //
    // This drives `prepare_chat_request` itself (not just its
    // sub-functions in isolation), with a `tokenize_len` thunk that
    // reports the whole context window as already consumed by the
    // prompt — so it is sensitive to a future reordering of the
    // `check_context_window` / `parse_stop_strings` calls inside
    // `prepare_chat_request`, not just to whether each sub-function
    // works in isolation.
    let mut req = bare_req();
    req.model = Some("served-model".to_string());
    req.messages = vec![user_msg("hi")];
    req.stop = Some(serde_json::json!([]));
    let err = prepare_chat_request(&req, "served-model", 256, 4096, false, |_| 4096, || 4096)
        .unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "context_length_exceeded",
            ..
        }
    ));
}

#[test]
fn cm_serve_logprobs_resolved_end_to_end() {
    // Full-cascade check backing the matrix's "supported, non-streaming
    // only" `logprobs`/`top_logprobs` claim for `lattice serve`.
    let mut req = bare_req();
    req.model = Some("served-model".to_string());
    req.messages = vec![user_msg("hi")];
    req.logprobs = Some(true);
    req.top_logprobs = Some(3);
    let validated = validate_chat_request(&req, "served-model", 256, 4096).unwrap();
    assert_eq!(validated.logprobs, Some(3));
}

// -----------------------------------------------------------------------
// Shared `AppState` builder for the router-level test modules below --
// both need a real (tiny, deterministic) CPU model, gated behind
// `test-utils` (see `lattice_inference::model::qwen35::test_support`)
// for the same reason: bin targets can't see this crate's own
// `#[cfg(test)]`-only fixtures across the bin/lib compilation
// boundary, only a real Cargo feature crosses it.
// -----------------------------------------------------------------------
/// The feedback handler's three successes and three refusals, against a
/// store this test populates directly.
///
/// Direct population rather than a routed completion: this asks what the
/// handler does with an episode, and going through generation to get one
/// would make a handler test fail for model reasons.
#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "mixture",
    feature = "test-utils"
))]
#[tokio::test]
async fn the_feedback_handler_separates_its_three_outcomes_and_two_refusals() {
    use lattice_inference::router_episode::{PolicyIdentity, RouterEpisode};
    use tower::ServiceExt;

    let state = tiny_state_with_router(64);
    let epoch = state
        .router_state
        .as_ref()
        .expect("the fixture installs a router")
        .policy_epoch();
    state
        .episodes
        .lock()
        .expect("fresh store")
        .record(RouterEpisode {
            completion_id: "chatcmpl-1-0".into(),
            policy: PolicyIdentity {
                artifact_version: "1:abcdef0123456789".into(),
                schema_id: "schema-1".into(),
                epoch,
            },
            adapter_names: vec!["technical".into()],
            context_vector: vec![0.5, 0.5],
            selected_scores: vec![("technical".into(), 0.9)],
            sampled_scores: vec![("technical".into(), 0.9)],
            pre_floor_weights: vec![("technical".into(), 1.0)],
            applied_weights: vec![("technical".into(), 1.0)],
            dropped: Vec::new(),
            tau: Some(0.7),
            exploration: None,
        });

    let post = |state: AppState, body: &'static str| async move {
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/router/feedback")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body))
            .expect("fixture request must build");
        let response = router(state).oneshot(request).await.expect("a response");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("a body");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("the body is JSON");
        (status, value)
    };

    let (status, body) = post(
        state.clone(),
        r#"{"completion_id":"chatcmpl-1-0","signal":"helpful"}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert_eq!(body["outcome"], "recorded");

    let (status, body) = post(
        state.clone(),
        r#"{"completion_id":"chatcmpl-1-0","signal":"helpful"}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert_eq!(
        body["outcome"], "idempotent",
        "a retry of the same signal must not refuse"
    );

    let (status, body) = post(
        state.clone(),
        r#"{"completion_id":"chatcmpl-1-0","signal":"unhelpful"}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "conflicting_judgement");

    let (status, body) = post(
        state.clone(),
        r#"{"completion_id":"never-routed","signal":"helpful"}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "unknown_completion");

    let (status, body) = post(
        state.clone(),
        r#"{"completion_id":"chatcmpl-1-0","signal":"maybe"}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_feedback_signal");

    // A caller supplying provenance is refused rather than having it
    // quietly dropped: accepting this body would let a caller believe it
    // had told the server which adapter to credit.
    let (status, body) = post(
        state.clone(),
        r#"{"completion_id":"chatcmpl-1-0","signal":"helpful","context_vector":[0.1]}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_feedback_body");
}

/// The census publishes the three counts a rate would hide, and the
/// eviction count that says it is a window.
#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "mixture",
    feature = "test-utils"
))]
#[tokio::test]
async fn the_episode_census_publishes_unjudged_separately_from_judged() {
    use lattice_inference::router_episode::{PolicyIdentity, RouterEpisode};
    use tower::ServiceExt;

    let state = tiny_state_with_router(64);
    let epoch = state
        .router_state
        .as_ref()
        .expect("the fixture installs a router")
        .policy_epoch();
    {
        let mut store = state.episodes.lock().expect("fresh store");
        for id in ["a", "b", "c"] {
            store.record(RouterEpisode {
                completion_id: id.into(),
                policy: PolicyIdentity {
                    artifact_version: "1:abcdef0123456789".into(),
                    schema_id: "schema-1".into(),
                    epoch,
                },
                adapter_names: vec!["technical".into()],
                context_vector: vec![0.5],
                selected_scores: vec![("technical".into(), 0.9)],
                sampled_scores: vec![("technical".into(), 0.9)],
                pre_floor_weights: vec![("technical".into(), 1.0)],
                applied_weights: vec![("technical".into(), 1.0)],
                dropped: Vec::new(),
                tau: Some(0.7),
                exploration: None,
            });
        }
        store
            .judge(
                "a",
                lattice_inference::router_episode::EpisodeSignal::Helpful,
                Some(epoch),
            )
            .expect("judge");
    }

    let request = axum::http::Request::builder()
        .method("GET")
        .uri("/v1/router/episodes")
        .body(axum::body::Body::empty())
        .expect("fixture request must build");
    let response = router(state).oneshot(request).await.expect("a response");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("a body");
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
    assert_eq!(body["recorded"], 3);
    assert_eq!(body["judged"], 1);
    assert_eq!(
        body["unjudged"], 2,
        "unjudged must be published, not folded into a rate"
    );
    assert_eq!(body["stale"], 0);
    assert_eq!(body["evicted"], 0);
}

/// Every declared router-feedback route is registered here, with the same
/// must-not-match control: `ROUTER_FEEDBACK_ROUTES` is a list, and a list
/// is not a registration.
///
/// These are `lattice`-only by design. The other serving binary takes no
/// `--router-state`, records no episodes, and would answer every judgement
/// with `unknown_completion`; a 404 there is the honest answer. That
/// asymmetry is why they are a separate const from `LORA_ROUTES` rather
/// than three more rows in it.
#[cfg(feature = "test-utils")]
#[tokio::test]
async fn every_router_feedback_route_is_registered_in_this_binary() {
    use lattice_inference::serve::lora::ROUTER_FEEDBACK_ROUTES;
    use tower::ServiceExt;

    assert!(
        !ROUTER_FEEDBACK_ROUTES.is_empty(),
        "the route list is empty, so this test would pass while checking nothing"
    );

    for (path, methods) in ROUTER_FEEDBACK_ROUTES {
        for method in *methods {
            let request = axum::http::Request::builder()
                .method(*method)
                .uri(*path)
                .header("content-type", "application/json")
                .body(axum::body::Body::from("{}"))
                .expect("fixture request must build");
            let response = router(tiny_state(64))
                .oneshot(request)
                .await
                .expect("router must produce a response, not a transport error");
            assert_ne!(
                response.status(),
                axum::http::StatusCode::NOT_FOUND,
                "{method} {path} is in ROUTER_FEEDBACK_ROUTES but not registered here"
            );
        }
    }

    // The must-not-match control: a neighbouring path nobody registered
    // still 404s, so the loop above is not passing because the router
    // answers everything.
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/router/definitely-not-a-route")
        .header("content-type", "application/json")
        .body(axum::body::Body::from("{}"))
        .expect("fixture request must build");
    let response = router(tiny_state(64))
        .oneshot(request)
        .await
        .expect("router must produce a response");
    assert_eq!(
        response.status(),
        axum::http::StatusCode::NOT_FOUND,
        "an unregistered path answered, so the loop above proves nothing"
    );
}

/// ADR-095 decision 4: every entry in the shared route list is registered
/// in THIS binary. A route added to the list and forgotten here reds this
/// test.
///
/// The assertion is only that the response is not 404. A 405 or a 4xx from
/// validation both count as registered -- the test asks whether the route
/// exists and nothing else, because anything more would need the state the
/// two binaries do not share. Asserting a specific status would make this
/// a behaviour test that passes or fails for reasons unrelated to
/// registration.
#[cfg(feature = "test-utils")]
#[tokio::test]
async fn every_shared_lora_route_is_registered_in_this_binary() {
    use lattice_inference::serve::lora::LORA_ROUTES;
    use tower::ServiceExt;

    assert!(
        !LORA_ROUTES.is_empty(),
        "the route list is empty, so this test would pass while checking nothing"
    );

    for (path, methods) in LORA_ROUTES {
        for method in *methods {
            let request = axum::http::Request::builder()
                .method(*method)
                .uri(*path)
                .header("content-type", "application/json")
                .body(axum::body::Body::from("{}"))
                .expect("fixture request must build");
            let response = router(tiny_state(64))
                .oneshot(request)
                .await
                .expect("router must produce a response, not a transport error");
            assert_ne!(
                response.status(),
                axum::http::StatusCode::NOT_FOUND,
                "{method} {path} is in LORA_ROUTES but not registered in this binary"
            );
            // A 405 means the path exists under a DIFFERENT method, so the listed
            // method is not registered either; checking 404 alone would pass it.
            assert_ne!(
                response.status(),
                axum::http::StatusCode::METHOD_NOT_ALLOWED,
                "{method} {path} is in LORA_ROUTES but this binary registers {path} under another method"
            );
        }
    }

    // The must-not-match control: the same machinery reports 404 for a
    // path nobody registered. Without it, a router that answered
    // everything -- a catch-all fallback, say -- would pass the loop above
    // while proving nothing about any individual route.
    let request = axum::http::Request::builder()
        .method("GET")
        .uri("/v1/lora/definitely-not-a-route")
        .body(axum::body::Body::empty())
        .expect("control request must build");
    let response = router(tiny_state(64))
        .oneshot(request)
        .await
        .expect("router must produce a response");
    assert_eq!(
        response.status(),
        axum::http::StatusCode::NOT_FOUND,
        "an unregistered path did not 404, so the loop above cannot distinguish \
             a registered route from a router that answers everything"
    );
}

#[cfg(feature = "test-utils")]
fn tiny_state(max_tokens_cap: usize) -> AppState {
    let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
    AppState {
        model: ModelBackend::Cpu(
            lattice_inference::serving_cpu_host::SharedCpuHandle::from_qwen_model_for_test(model),
        ),
        default_max_tokens: max_tokens_cap,
        max_tokens_cap,
        model_id: "test-model".to_string(),
        request_counter: Arc::new(AtomicU64::new(0)),
        embedding_model: None,
        embedding_admission: Arc::new(Semaphore::new(EMBEDDING_MAX_CONCURRENT_JOBS)),
        cpu_generation_admission: Arc::new(Semaphore::new(CPU_GENERATION_MAX_CONCURRENT_JOBS)),
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        router_state: None,
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        episodes: episode_store(),
    }
}

/// A state whose router is real, for the handlers that compare a
/// judgement against the epoch a gate is serving under.
///
/// The `None` router the plain fixture carries is a state the feedback
/// path can no longer be exercised from: with no gate there is no epoch,
/// and every judgement is stale by construction. That is production's
/// answer too -- a server with no router records no episode -- so the
/// fixture installs the gate rather than the handler inventing an epoch
/// for a router that is not there.
#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "mixture",
    feature = "test-utils"
))]
fn tiny_state_with_router(max_tokens_cap: usize) -> AppState {
    use lattice_inference::router_init::{BootstrapSpec, bootstrap_artifact};
    use lattice_inference::router_state::{ResolvedRouter, TrainedRepresentation};
    use lattice_inference::serve::embeddings::QWEN35_F16_DECODER_LOADER;
    use lattice_inference::serve::routing::{PromptSource, ServedRouter};

    const WIDTH: usize = 4;
    let embedder = lattice_inference::serve::routing::EmbedderIdentity::resolve(
        std::path::Path::new("/models"),
        Some("gme-qwen35"),
    );
    let artifact = bootstrap_artifact(&BootstrapSpec {
        adapter_names: vec!["technical".to_string()],
        representation: TrainedRepresentation {
            embedding_model: embedder.as_str().to_string(),
            pooling: "mean_visual".to_string(),
            prompt_source: PromptSource::SERVED.as_str().to_string(),
            loader_format: QWEN35_F16_DECODER_LOADER.to_string(),
            input_width: WIDTH as u64,
        },
        schema_id: "schema-1".to_string(),
        base_model: "test-model".to_string(),
        // The uniform bootstrap: this fixture builds a server with learning
        // off, which is what these tests are about.
        tau: None,
        exploration_scale: None,
    })
    .expect("the bootstrap specification must be valid");
    let served = ServedRouter::new(
        ResolvedRouter {
            artifact,
            pinned: false,
        },
        &embedder,
        WIDTH,
        false,
    )
    .expect("a bootstrap artifact must be servable by the build that wrote it");

    AppState {
        router_state: Some(std::sync::Arc::new(served)),
        ..tiny_state(max_tokens_cap)
    }
}

// -----------------------------------------------------------------------
// CPU chat-generation admission (`--max-cpu-generations`). Mirrors
// `admission_cap_939`'s Metal-worker tests: a real `router()` over a CPU
// backend whose injected generator blocks on a channel, so exactly one
// generation is provably in flight and a second concurrent request is
// rejected by admission rather than racing to observe a freed slot.
// -----------------------------------------------------------------------
#[cfg(feature = "test-utils")]
mod cpu_generation_admission;

#[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
mod vision_content_parts_1135;

// -----------------------------------------------------------------------
// POST /v1/embeddings router-level contract tests.
// -----------------------------------------------------------------------
#[cfg(feature = "test-utils")]
mod embeddings_route;

// -----------------------------------------------------------------------
// HTTP-level client-disconnect cancellation (ADR-080 C2) -- gated behind `test-utils` (see
// `lattice_inference::model::qwen35::test_support`) because it needs a
// real, tiny, deterministic CPU model to exercise the actual
// `chat_completions` -> `body_stream`'s `cancel_guard` capture ->
// `generate_streaming_with_cancel` composition end to end, not just the
// primitive (already unit/mutation-tested in
// `model/qwen35/generation.rs`) or the guard type (already unit-tested
// in `serve/mod.rs`) in isolation. "The disclosed
// HTTP-level disconnect test gap ... does not waive that gate."
// -----------------------------------------------------------------------
#[cfg(feature = "test-utils")]
mod disconnect_cancellation;

/// Builds an `AppState` whose CPU streaming generation is the
/// injected `generate` closure instead of the real tiny model's
/// decode loop, via the host's test generator. Tokenizer/
/// context-window behavior still comes from a real tiny model
/// (`tiny_zero_model()`), so request validation ahead of the
/// streaming branch is unchanged from the other HTTP-level
/// tests in this file.
#[cfg(feature = "test-utils")]
pub(super) fn tiny_state_with_test_generator(
    max_tokens_cap: usize,
    generate: impl Fn(
        &str,
        &lattice_inference::GenerateConfig,
        &mut dyn FnMut(&str) -> bool,
        &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, lattice_inference::error::InferenceError>
    + Send
    + Sync
    + 'static,
) -> AppState {
    let model = Arc::new(lattice_inference::model::qwen35::test_support::tiny_zero_model());
    AppState {
        model: ModelBackend::Cpu(
            lattice_inference::serving_cpu_host::SharedCpuHandle::with_qwen_generator_for_test(
                model, generate,
            ),
        ),
        default_max_tokens: max_tokens_cap,
        max_tokens_cap,
        model_id: "test-model".to_string(),
        request_counter: Arc::new(AtomicU64::new(0)),
        embedding_model: None,
        embedding_admission: Arc::new(Semaphore::new(EMBEDDING_MAX_CONCURRENT_JOBS)),
        cpu_generation_admission: Arc::new(Semaphore::new(CPU_GENERATION_MAX_CONCURRENT_JOBS)),
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        router_state: None,
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        episodes: episode_store(),
    }
}

// -----------------------------------------------------------------------
// Post-drop generator-side cancellation probe (ADR-080 C2):
// `chat_completions_streaming_
// disconnect_stops_generation` above proves guard retention BEFORE
// the response is returned (frame 2 is a real content delta), but
// does NOT prove `should_cancel` reaching the generator AFTER the
// drop, independently of `on_token`'s own failed-send stop
// condition -- the exact reverse mutation (`cancel_rx`
// predicate replaced by `move || false`, `on_token`'s failed-send
// path left intact) left that test green in 0.02s, because the
// failed send alone stops a real decode loop just as fast as a
// correctly wired cancellation would.
//
// This test isolates the two signals: the host's test generator
// (test-only, see its doc comment) sends exactly ONE delta -- so
// `on_token` is never called again after that point -- then enters a
// phase that polls ONLY `should_cancel`, in a loop that never touches
// `on_token`/the reply channel at all, and reports which of two
// outcomes ended that loop over a side channel `chat_completions`
// itself never sees. Because the probe stops polling `on_token`
// entirely after the first delta, `on_token`'s failed-send masking
// effect (the exact thing that hid this bug from the original
// test) cannot fire here -- only `should_cancel`'s own return value
// can end the loop.
// -----------------------------------------------------------------------
#[cfg(feature = "test-utils")]
mod post_drop_cancellation_probe;

// -----------------------------------------------------------------------
// Streaming context-overflow status parity (ADR-080 C2): `lattice.rs` already gets this right
// structurally -- `prepare_chat_request`'s context-window preflight
// (`check_context_window`) runs unconditionally, before
// `chat_completions` ever branches on `req.stream` -- so a `stream:
// true` request that overflows the model's context window returns
// HTTP 400 `context_length_exceeded` before any SSE stream is ever
// built. `cm_serve_context_window_checked_before_stop_parsing` above
// already pins the underlying cascade ordering as a pure function;
// this drives the SAME contract through the real `Router`.
//
// ADR-080 C2: this now
// builds its request from `lattice_inference::serve`'s shared
// `OVERFLOW_PARITY_*` constants -- the SAME body/limits
// `lattice_serve.rs`'s `real_router_overflow_parity` module drives
// through its own real router and real worker -- so the two really
// are the identical request the old doc comment here merely
// claimed. The tiny test model's context window
// (`test_support::tiny_zero_model`'s `max_position_embeddings`) is
// fixed at `OVERFLOW_PARITY_CONTEXT_WINDOW` (1024) precisely so this
// side's "effective context limit" matches the daemon side's
// explicitly-configured `AppState.model_max_context` of the same
// value.
// -----------------------------------------------------------------------
#[cfg(feature = "test-utils")]
mod streaming_context_overflow {
    use super::*;
    use lattice_inference::serve::{OVERFLOW_PARITY_MAX_TOKENS_CAP, OVERFLOW_PARITY_REQUEST_BODY};
    use tower::ServiceExt as _;

    #[tokio::test]
    async fn chat_completions_streaming_context_overflow_returns_400_before_committing_sse() {
        let body = axum::body::Body::from(OVERFLOW_PARITY_REQUEST_BODY.to_string());
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(body)
            .expect("fixture request must build");
        let response = router(tiny_state(OVERFLOW_PARITY_MAX_TOKENS_CAP))
            .oneshot(request)
            .await
            .expect("router must produce a response, not a transport error");
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "an over-context stream:true request must be rejected with HTTP \
                     400 before any SSE stream is committed, matching \
                     lattice_serve.rs's equivalent preflight for the identical \
                     shared-fixture request body"
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("error response body must be readable");
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("error response must be JSON");
        assert_eq!(value["error"]["code"], "context_length_exceeded");
    }
}

#[cfg(feature = "test-utils")]
mod prompt_above_tokenizer_cap {
    use super::*;
    use tower::ServiceExt as _;

    #[tokio::test]
    async fn over_window_prompt_is_refused_with_its_untruncated_token_count() {
        let cap = lattice_inference::model::qwen35::test_support::tiny_zero_model()
            .tokenizer()
            .max_seq_len();
        let content = "ab".repeat(cap);
        let body = format!(
            r#"{{"model":"test-model","messages":[{{"role":"user","content":"{content}"}}],"max_tokens":16}}"#
        );
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body))
            .expect("fixture request must build");
        let response = router(tiny_state(16))
            .oneshot(request)
            .await
            .expect("router must produce a response, not a transport error");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("error response body must be readable");
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("error response must be JSON");
        assert_eq!(value["error"]["code"], "context_length_exceeded");
        let message = value["error"]["message"]
            .as_str()
            .expect("error message must be a string");
        let prompt_tokens: usize = message
            .strip_prefix("prompt (")
            .and_then(|rest| rest.split_once(" tokens)"))
            .and_then(|(count, _)| count.parse().ok())
            .unwrap_or_else(|| panic!("unexpected refusal message: {message}"));
        assert!(
            prompt_tokens >= 2 * cap,
            "admission must count every prompt token, not the tokenizer's {cap}-token \
                 truncation: {message}"
        );
    }
}

// -----------------------------------------------------------------------
// Message-flood bound: proves the fix at the real HTTP layer, not just
// at `serve::contract`'s own unit-test level -- the router's
// `chat_completions` entry point must reject a body with more than
// `MAX_MESSAGE_COUNT` tiny messages.
// -----------------------------------------------------------------------
#[cfg(feature = "test-utils")]
mod message_flood {
    use super::*;
    use lattice_inference::serve::contract::MAX_MESSAGE_COUNT;
    use tower::ServiceExt as _;

    #[tokio::test]
    async fn chat_completions_rejects_message_flood() {
        // One more message than the bound, each as small as the wire
        // format allows: comfortably under the 1 MiB body cap, but
        // tens of thousands of entries. The message-count bound is
        // enforced inline while `ChatCompletionRequest::messages`
        // deserializes, so this never allocates a `Vec<Message>`
        // entry per message before rejecting -- see this test's
        // sibling unit tests in `serve::contract` for direct coverage
        // of that deserializer.
        let messages: Vec<String> = (0..MAX_MESSAGE_COUNT + 1)
            .map(|_| r#"{"role":"user","content":""}"#.to_string())
            .collect();
        let body = format!(
            r#"{{"model":"test-model","messages":[{}]}}"#,
            messages.join(",")
        );
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body))
            .expect("fixture request must build");
        let response = router(tiny_state(64))
            .oneshot(request)
            .await
            .expect("router must produce a response, not a transport error");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("error response body must be readable");
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("error response must be JSON");
        assert_eq!(value["error"]["code"], "invalid_request_body");
    }
}

// -----------------------------------------------------------------------
// Content-Type precedence: an invalid-MIME request must be rejected by
// `require_json_content_type` before the body is ever read as JSON.
// -----------------------------------------------------------------------
#[cfg(feature = "test-utils")]
mod content_type_precedence {
    use super::*;
    use tower::ServiceExt as _;

    /// A body that is not valid JSON at all. Combined with a
    /// non-JSON `Content-Type`, this distinguishes ordering: if the
    /// Content-Type guard runs first, the response is 415
    /// `unsupported_media_type`; if it were reordered to run after
    /// the body is parsed, this body would instead fail JSON
    /// parsing first and surface as 400 `invalid_request_body`.
    #[tokio::test]
    async fn invalid_content_type_rejected_before_body_is_parsed() {
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "text/plain")
            .body(axum::body::Body::from("this is not json"))
            .expect("fixture request must build");
        let response = router(tiny_state(64))
            .oneshot(request)
            .await
            .expect("router must produce a response, not a transport error");
        assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("error response body must be readable");
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("error response must be JSON");
        assert_eq!(value["error"]["code"], "unsupported_media_type");
    }
}

// -----------------------------------------------------------------------
// Cross-binary `/v1/chat/completions` parity table (ADR-080 C2):
// drives every fixture body in
// `lattice_inference::serve::CHAT_COMPLETIONS_PARITY_CASES` through
// THIS binary's real `Router` via `tower::ServiceExt::oneshot`, and
// compares the resulting status + error code against the case's
// `lattice`-side expectation. `lattice_serve.rs`'s own test module
// runs the SAME table against its own router, asserting the
// `lattice_serve`-side expectation -- together the two prove
// same-input parity (or a documented, intentional divergence) at the
// real HTTP layer, not just in each binary's private validation
// helpers. Gated behind `test-utils` for the same reason as
// `disconnect_cancellation`: `router()` needs a real `AppState`,
// which needs a real (tiny) CPU model.
// -----------------------------------------------------------------------
#[cfg(feature = "test-utils")]
mod parity_table;

// -----------------------------------------------------------------------
// Production-adapter observation (issue #828): proves the shared
// `ProductionAdapterObservation`/`GenerateConfigSnapshot` types
// actually capture what THIS binary's real `chat_completions` ->
// `prepare_chat_request` -> `GenerateConfig` construction produces,
// not a value the test independently reconstructs. The injected
// injected generator closure below runs strictly below that real
// path -- it records the `&GenerateConfig`/`&str` prompt it was
// actually called with, then returns a canned result; it never
// recomputes `build_cfg`/`validate_temperature`/etc. itself.
//
// DISPUTED (issue #828):
// this observation captures `rendered_prompt`, not `messages`, and
// that is the real shape of this seam, not an omission. `chat_completions`
// computes `to_chat_messages(&req.messages)` (the normalized message
// list) unconditionally whenever `feature = "metal-gpu"` is compiled
// in, but that value is consumed ONLY by the `ModelBackend::Metal`
// match arm (`handle.generate_streaming[_with_cancel](chat_messages,
// ...)`); the `ModelBackend::Cpu` test host this test
// seam exercises never receive it -- their real `generate`/
// `generate_streaming_with_cancel` calls take only `(&prompt,
// &gen_cfg, ...)`. This mirrors `ProductionAdapterObservation`'s own
// documented contract in `serve/mod.rs` ("exactly one of
// `rendered_prompt`/`messages` is `Some` per capture, reflecting
// which shape that binary's real adapter actually receives, not a
// missing capture").
//
// Observing `messages` at the CPU seam authentically (not by
// re-deriving `to_chat_messages` independently in the test, which
// would be tautological -- exactly the bug this
// module was written to fix) would require a `MetalFakeGenerate`
// test double for `ModelBackend::Metal`. `MetalHandle::spawn`
// hard-requires loading a real Q4 model directory onto a real Metal
// GPU worker thread (`MetalQwen35State::from_q4_dir`) -- there is no
// model-agnostic seam there the way the test generator mirrors
// `ModelBackend::Cpu`. Building one would mean adding a new
// production `ModelBackend` variant and mocking the async engine
// handle's job-channel protocol: real production-code surface
// expansion, not a test-only capture. That is out of scope for this
// fix round; tracked as a follow-up if a Metal-path observation is
// wanted (would need its own issue -- #828's fixture data and CI
// environment target the CPU/tiny-tokenizer seam only).
// -----------------------------------------------------------------------
#[cfg(feature = "test-utils")]
mod production_adapter_observation;

// -----------------------------------------------------------------------
// Completion-length invariant accounts for reasoning_budget (#1334)
//
// Admission (`serve::contract::validate_context_window_with_budget`)
// accepts `prompt + max_tokens + reasoning_budget + 1 <= max_context`,
// but the post-generation invariant checked only `generated_tokens >
// max_tokens` -- so a completion that legitimately spent its reasoning
// allowance passed preflight, generated correctly, and was then reported
// as an inference failure. These tests drive the real
// `chat_completions_with_request` cascade (through the real `router()`,
// for streaming's SSE framing) with an injected CPU host generator
// closure that reports a caller-chosen `generated_tokens`, so the
// engine's *reported* completion length can be placed exactly at or one
// token past the decode budget independent of what the tiny model would
// actually decode.
// -----------------------------------------------------------------------
#[cfg(feature = "test-utils")]
mod completion_length_invariant_1334;

mod lora_orchestration;
