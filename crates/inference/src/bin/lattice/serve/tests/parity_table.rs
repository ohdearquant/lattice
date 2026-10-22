use super::*;
use lattice_inference::StopReason;
use lattice_inference::serve::{
    BASELINE_CANNED_COMPLETION_TOKENS, BASELINE_CANNED_PROMPT_TOKENS, BASELINE_CANNED_TEXT, Binary,
    CHAT_COMPLETIONS_PARITY_CASES, ExpectedResponse, check_sse_events,
};
use tower::ServiceExt as _;

/// Small enough that `max_tokens_over_cap_reject_vs_clamp`'s
/// 999999 genuinely exceeds both `default_max_tokens` and
/// `max_tokens_cap`.
const CAP: usize = 64;

/// Deterministic CPU generation seam for every `Json`/`Sse`
/// row (issue #828): the real request-parse/normalize/
/// `GenerateConfig`-build/handler/serialization path all still
/// runs unmodified -- only the actual model forward pass is
/// replaced through the CPU host's test generator
/// injection seam the disconnect-cancellation probe uses.
/// Content deltas are pushed through `on_token` (what the
/// streaming arm reads) AND the returned `GenerateOutput.text`
/// carries the identical concatenated text (what the
/// non-streaming arm reads) so one closure serves both shapes.
fn baseline_fake_state(max_tokens_cap: usize) -> AppState {
    let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
    #[allow(clippy::type_complexity)]
    let generate: Arc<
        dyn Fn(
                &str,
                &lattice_inference::GenerateConfig,
                &mut dyn FnMut(&str) -> bool,
                &mut dyn FnMut() -> bool,
            ) -> Result<GenerateOutput, lattice_inference::error::InferenceError>
            + Send
            + Sync,
    > = Arc::new(|_prompt, _cfg, on_token, _should_cancel| {
        for chunk in ["hello", " world"] {
            if !on_token(chunk) {
                break;
            }
        }
        Ok(GenerateOutput {
            text: BASELINE_CANNED_TEXT.to_string(),
            token_ids: vec![1, 2],
            prompt_tokens: BASELINE_CANNED_PROMPT_TOKENS as usize,
            generated_tokens: BASELINE_CANNED_COMPLETION_TOKENS as usize,
            stopped: true,
            stop_reason: Some(StopReason::Eos),
            token_logprobs: vec![],
        })
    });
    AppState {
        model: ModelBackend::Cpu(
            lattice_inference::serving_cpu_host::SharedCpuHandle::with_qwen_generator_for_test(
                Arc::new(model),
                move |prompt, config, on_token, should_cancel| {
                    generate(prompt, config, on_token, should_cancel)
                },
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

#[tokio::test]
async fn chat_completions_matches_shared_parity_table() {
    for case in CHAT_COMPLETIONS_PARITY_CASES {
        let expected = case.expected(Binary::Lattice);
        // Error-shaped rows never reach generation (rejected at
        // validation), so they keep using the plain real tiny
        // model exactly as before #828; only the new `Json`/
        // `Sse` rows need the deterministic generation seam.
        let app = match expected {
            ExpectedResponse::Error { .. } => router(tiny_state(CAP)),
            ExpectedResponse::Json { .. } | ExpectedResponse::Sse { .. } => {
                router(baseline_fake_state(CAP))
            }
        };
        let request = axum::http::Request::builder()
            .method(case.method)
            .uri(case.path)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(case.body.build()))
            .expect("fixture request must build");
        let response = app
            .oneshot(request)
            .await
            .expect("router must produce a response, not a transport error");

        let status = response.status().as_u16();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body reads");
        let text = String::from_utf8_lossy(&body);

        assert_eq!(
            status,
            expected.status(),
            "case '{}': expected status {}, got {status} (body: {text})",
            case.name,
            expected.status(),
        );

        match expected {
            ExpectedResponse::Error { code, .. } => {
                let value: serde_json::Value = serde_json::from_slice(&body).unwrap_or_else(|e| {
                    panic!(
                        "case '{}': non-2xx response body must be the shared \
                                         error envelope JSON: {e} (body: {text})",
                        case.name,
                    )
                });
                assert_eq!(
                    value["error"]["code"], code,
                    "case '{}': expected error code '{code}', got {} \
                                 (full body: {value})",
                    case.name, value["error"]["code"]
                );
            }
            ExpectedResponse::Json { fields, .. } => {
                let value: serde_json::Value = serde_json::from_slice(&body).unwrap_or_else(|e| {
                    panic!(
                        "case '{}': 2xx response body must be JSON: {e} \
                                         (body: {text})",
                        case.name,
                    )
                });
                for field in fields {
                    field.check(&value).unwrap_or_else(|e| {
                        panic!("case '{}': field check failed: {e}", case.name)
                    });
                }
            }
            ExpectedResponse::Sse { events, .. } => {
                check_sse_events(&text, events)
                    .unwrap_or_else(|e| panic!("case '{}': SSE check failed: {e}", case.name));
            }
        }
    }
}
