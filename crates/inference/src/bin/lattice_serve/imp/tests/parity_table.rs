use super::*;
use lattice_inference::serve::{
    BASELINE_CANNED_COMPLETION_TOKENS, BASELINE_CANNED_PROMPT_TOKENS, Binary,
    CHAT_COMPLETIONS_PARITY_CASES, ExpectedResponse, check_sse_events,
};
use tower::ServiceExt as _;

/// Deterministic worker-thread generation seam for every
/// `Json`/`Sse` row (issue #828): a REAL background thread runs
/// the actual `run_worker_loop` + `check_prompt_fits_window`
/// production code (via `metal_worker::spawn_fake`, issue
/// #832's cross-binary test seam; real tiny tokenizer, no Metal
/// engine) -- only the terminal generation call itself is
/// replaced with a canned deterministic completion. Content
/// deltas are pushed through `on_token` (what both the
/// streaming and non-streaming arms of `chat_completions` read
/// content from on this binary -- see its `WorkerEvent::Delta`
/// accumulation), so one seam serves both response shapes.
fn baseline_fake_worker_state(model_max_context: usize) -> AppState {
    let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
        .tokenizer()
        .clone();
    let jobs = spawn_fake(
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        model_max_context,
        tokenizer,
        |_messages, _cfg, _prompt_tokens, on_token, _should_cancel| {
            // The real `check_prompt_fits_window` (mutation-sensitive
            // exactly like `real_router_overflow_parity` above) has
            // already run inside `spawn_fake` by the time this
            // closure is called; its measured `_prompt_tokens` is
            // intentionally NOT what gets reported below -- the
            // returned usage counts are the fixed canned figures
            // both binaries' field checks assert against
            // (`BASELINE_CANNED_*`).
            for (chunk, id) in [("hello", 1u32), (" world", 2u32)] {
                on_token(chunk, id);
            }
            Ok(GenerateOutput {
                text: "hello world".to_string(),
                token_ids: vec![1, 2],
                prompt_tokens: BASELINE_CANNED_PROMPT_TOKENS as usize,
                generated_tokens: BASELINE_CANNED_COMPLETION_TOKENS as usize,
                stopped: true,
                stop_reason: None,
                token_logprobs: vec![],
            })
        },
    );
    AppState {
        jobs,
        model_id: Arc::from("test-model"),
        defaults: GenerationDefaults::standard(100),
        model_max_context,
        max_pending: 1_000_000,
        metrics: Arc::new(ServeMetrics::default()),
        vocab_bytes: Arc::new(vec![]),
        grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
        embedding: None,
    }
}

#[tokio::test]
async fn chat_completions_matches_shared_parity_table() {
    for case in CHAT_COMPLETIONS_PARITY_CASES {
        let expected = case.expected(Binary::LatticeServe);
        // Error-shaped rows never reach the job queue (rejected
        // at validation) or, for `max_tokens_over_cap_reject_vs_clamp`,
        // rely on the no-worker harness artifact its own
        // doc comment describes -- both keep using the plain
        // no-worker state exactly as before #828.
        let app = match expected {
            ExpectedResponse::Error { .. } => router(test_app_state()),
            ExpectedResponse::Json { .. } | ExpectedResponse::Sse { .. } => {
                router(baseline_fake_worker_state(4096))
            }
        };
        let request = axum::http::Request::builder()
            .method(case.method)
            .uri(case.path)
            .header("content-type", "application/json")
            .body(Body::from(case.body.build()))
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
