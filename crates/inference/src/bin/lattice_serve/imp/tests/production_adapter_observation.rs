use super::*;
use lattice_inference::Tokenizer;
use lattice_inference::serve::{
    ExpectedObservation, GenerateConfigSnapshot, OBSERVATION_GOLDEN_USER_HI_THERE_CHATML,
    ProductionAdapterObservation, assert_observation_matches,
};
use std::sync::Mutex;

/// Mirrors `lattice.rs`'s equivalent helper (issue #828):
/// fires the fixed `{"messages":[{"role":"user","content":"hi
/// there"}],"temperature":1.3,"top_p":0.55,"seed":7,"max_tokens":9}`
/// request through a REAL background thread running the actual
/// `run_worker_loop` + `check_prompt_fits_window` production code
/// (via `metal_worker::spawn_fake`, issue #832's cross-binary test
/// seam; real tiny tokenizer, no Metal engine) -- only the terminal
/// generation call is replaced with a canned completion. `stopped`
/// is threaded through a single local variable into both the
/// recorded observation and the returned `GenerateOutput`, and the
/// real `check_prompt_fits_window` measured length -- surfaced by
/// `spawn_fake` as the closure's `prompt_tokens` argument -- is the
/// ONLY source for `prompt_tokens` here.
async fn run_observed(model_max_context: usize, stopped: bool) -> ProductionAdapterObservation {
    let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
        .tokenizer()
        .clone();
    let observed: Arc<Mutex<Option<ProductionAdapterObservation>>> = Arc::new(Mutex::new(None));
    let observed_for_worker = Arc::clone(&observed);
    let jobs = spawn_fake(
        ContextWindowPolicy::PromptAndDecodeWithDelimiter,
        model_max_context,
        tokenizer,
        move |messages, cfg, prompt_tokens, on_token, _should_cancel| {
            let normalized: Vec<(String, String)> = messages
                .iter()
                .map(|m| {
                    let role = match m.role {
                        lattice_inference::forward::metal_qwen35::ChatRole::System => "system",
                        lattice_inference::forward::metal_qwen35::ChatRole::User => "user",
                        lattice_inference::forward::metal_qwen35::ChatRole::Assistant => {
                            "assistant"
                        }
                        // This fixture only ever constructs the
                        // three known roles; a future role isn't
                        // yet exercised by this test.
                        _ => "unknown",
                    };
                    (role.to_string(), m.content.clone())
                })
                .collect();
            // Real production check (mutation-sensitive to
            // `check_prompt_fits_window` exactly like
            // `real_router_overflow_parity`/`baseline_fake_worker_state`
            // above) has already run inside `spawn_fake` by the
            // time this closure is called; its measured
            // `prompt_tokens` IS what gets reported below.
            *observed_for_worker
                .lock()
                .expect("observation mutex poisoned") = Some(ProductionAdapterObservation {
                rendered_prompt: None,
                messages: Some(normalized),
                gen_cfg: GenerateConfigSnapshot::from(cfg),
                prompt_tokens,
                stopped,
            });
            on_token("ok", 1);
            Ok(GenerateOutput {
                text: "ok".to_string(),
                token_ids: vec![1],
                prompt_tokens,
                generated_tokens: 1,
                stopped,
                stop_reason: None,
                token_logprobs: vec![],
            })
        },
    );
    let state = AppState {
        jobs,
        model_id: Arc::from("test-model"),
        defaults: GenerationDefaults::standard(100),
        model_max_context,
        max_pending: 1_000_000,
        metrics: Arc::new(ServeMetrics::default()),
        vocab_bytes: Arc::new(vec![]),
        grammar_cache: Arc::new(GrammarCache::new(GRAMMAR_CACHE_CAPACITY)),
        embedding: None,
    };
    let body = Body::from(
                    r#"{"messages":[{"role":"user","content":"hi there"}],"temperature":1.3,"top_p":0.55,"seed":7,"max_tokens":9}"#
                        .to_string(),
                );
    let response = chat_completions(State(state), test_json_headers(), body).await;
    assert_eq!(response.status(), StatusCode::OK);

    observed
        .lock()
        .expect("observation mutex poisoned")
        .clone()
        .expect("the injected worker-loop generate closure must have recorded an observation")
}

/// The `GenerateConfig` `lattice_serve.rs`'s real `build_cfg` must
/// produce for the fixed request `run_observed` sends, given
/// `run_observed`'s `GenerationDefaults` above: every explicitly-set field
/// mirrors the request; `build_cfg` always sets the remaining
/// fields (`stop_token_ids`, `enable_thinking`, `enable_mtp`,
/// `grammar`, `reasoning_budget`, `logprobs`, `stop_strings`) to
/// the exact same values `GenerateConfig::default()` carries.
fn expected_gen_cfg() -> GenerateConfigSnapshot {
    let mut cfg = lattice_inference::GenerateConfig::default();
    cfg.max_new_tokens = 9;
    cfg.temperature = 1.3;
    cfg.top_p = 0.55;
    cfg.seed = Some(7);
    GenerateConfigSnapshot::from(&cfg)
}

#[tokio::test]
async fn chat_completions_non_streaming_observation_captures_real_config_and_messages() {
    let obs = run_observed(4096, true).await;
    let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
        .tokenizer()
        .clone();
    let rendered_prompt =
        format!("{OBSERVATION_GOLDEN_USER_HI_THERE_CHATML}<think>\n\n</think>\n\n");
    let expected_prompt_tokens = tokenizer.tokenize(&rendered_prompt).real_length;
    let mut expected_gen_cfg = expected_gen_cfg();
    expected_gen_cfg.enable_thinking = false;
    assert_observation_matches(
        &obs,
        &ExpectedObservation {
            gen_cfg: expected_gen_cfg,
            rendered_prompt: None,
            messages: Some(&[("user", "hi there")]),
            prompt_tokens: expected_prompt_tokens,
            stopped: true,
        },
    );
}

/// Proves `stopped` is genuinely derived from what the worker's
/// generation closure returned, not an independent hardcoded
/// literal (this was previously
/// `stopped: true` regardless of the closure's actual return).
#[tokio::test]
async fn chat_completions_non_streaming_observation_captures_real_stopped_false() {
    let obs = run_observed(4096, false).await;
    assert!(
        !obs.stopped,
        "observation must report the worker's actual stopped=false, not a hardcoded true"
    );
}
